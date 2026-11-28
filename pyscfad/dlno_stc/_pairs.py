"""Boys multipole screening and directed weak-pair completion.

Concrete weak algebra uses explicit NumPy/SciPy forward and reverse kernels;
JAX is used only for shared molecular producers and pytree bookkeeping.
Fixed primary-screen reconstruction uses each singleton Boys target.
Weak pairs accumulate compact endpoint
cotangents before replaying each endpoint response once; no native DLNO
module is imported by this implementation.
"""
import gc
from collections import Counter
from types import SimpleNamespace
from functools import partial
from typing import NamedTuple

import jax
import numpy as np
import scipy.linalg as scipy_linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pyscfad.lno.tools import fake_mol_by_atom
from pyscfad.tools import resource_profile
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.lib._threading import blas_threads
from pyscfad.lib._multipole import endpoint_matches_reference, crop_endpoint_cotangent
from pyscfad.lib import _multipole_numpy
from . import _weak_numpy
from pyscfad.lib._cartesian_moments import cartesian_components

_MOMENT_SLAB_BYTES = 32 * 1024**2


def pad_endpoint(endpoint):
    """Concrete endpoints need no padding or shape-specific compilation."""
    _weak_numpy.require_supported(endpoint)
    return jax.tree.map(np.asarray, endpoint)


def _multipole_pair_energy(left, right, order):
    _weak_numpy.require_supported(left, right)
    return _multipole_numpy.pair(left, right, order)


def pair_energy_and_grad(left, right, order):
    _weak_numpy.require_supported(left, right)
    return _multipole_numpy.pair(left, right, order, with_grad=True)


class WeakScreen(NamedTuple):
    """One normalized Boys target and its primary-domain virtual space."""

    occupied_energy: object
    occupied_coeff: object
    virtual_energy: object
    virtual_coeff: object


class _SharedPAOPullback(NamedTuple):
    """One target response and the shared live preprojection reverse rule."""

    target_pullback: object
    preprojection_pullback: object
    products_pullback: object = None
    occupied_index: int | None = None


class _WeakScreenProducts(NamedTuple):
    pao: object
    occupied: tuple


class _WeakProductsPullback(NamedTuple):
    """Shared reverse rule and allocation metadata, without cached primals."""

    pullback: object
    shapes: object


def _weak_screen_products(common):
    """Compute global PAO and independent target products once per invocation."""
    return _weak_numpy.products(common)[0]


def _cached_weak_screen_products(common, products):
    """Attach the exact ambient response without recomputing cached primals."""
    return _weak_numpy.products(common, cached=products)


def _weak_pao_preprojection_key(fragment):
    """Only identical ordered fixed labels identify the same live PAO map."""
    from ._selection import FixedPAOSubspaceSelection
    selection = fragment.weak_virtual
    if not isinstance(selection, FixedPAOSubspaceSelection):
        return None
    return tuple(tuple(int(index) for index in labels) for labels in (
        selection.parent_columns, selection.support_ao_indices,
        fragment.primary_ao_indices, selection.canonical_keep,
        selection.overlap_keep, selection.completeness_keep))


def _apply_weak_screen_pullback(pullback, screen_bar, shared_pao_bars, shared_product_bars):
    """Apply a shared reverse rule once, at its last active target."""
    if isinstance(pullback, _SharedPAOPullback):
        from .adjoint import add_cotangent
        products = pullback.products_pullback
        if products is None:
            common_bar, pao_bar = pullback.target_pullback(screen_bar)
        else:
            common_bar, pao_bar, occupied_bar = pullback.target_pullback(screen_bar)
        shared = pullback.preprojection_pullback
        remaining, previous = shared_pao_bars.pop(shared)
        pao_bar = pao_bar if previous is None else previous + pao_bar
        product_pao_bar = None
        if remaining == 1:
            if products is None:
                shared_bar, = shared(pao_bar)
            else:
                shared_bar, product_pao_bar = shared(pao_bar)
            common_bar = jax.tree_util.tree_map(add_cotangent, common_bar, shared_bar)
        else:
            shared_pao_bars[shared] = (remaining - 1, pao_bar)
        if products is not None:
            remaining, product_bar = shared_product_bars.pop(products.pullback)
            if product_bar is None:
                product_bar = jax.tree_util.tree_map(
                    lambda shape: np.zeros(shape.shape, shape.dtype), products.shapes)
            occupied_bars = list(product_bar.occupied)
            occupied_bars[pullback.occupied_index] += occupied_bar
            pao_bars = product_bar.pao
            if product_pao_bar is not None:
                pao_bars = jax.tree_util.tree_map(add_cotangent, pao_bars, product_pao_bar)
            product_bar = _WeakScreenProducts(pao_bars, tuple(occupied_bars))
            if remaining == 1:
                shared_bar, = products.pullback(product_bar)
                common_bar = jax.tree_util.tree_map(add_cotangent, common_bar, shared_bar)
            else:
                shared_product_bars[products.pullback] = (remaining - 1, product_bar)
    else:
        common_bar, = pullback(screen_bar)
    return common_bar


def _fake_mol_with_traced_centers(mol, atmlst):
    """Build an atom submolecule while retaining its coordinate response."""
    fake_mol = fake_mol_by_atom(mol, atmlst)
    coords = getattr(mol, 'coords', None)
    if atmlst is not None and coords is not None:
        # ``fake_mol_by_atom`` rebuilds the concrete integral environment but
        # its shallow copy otherwise retains the full parent coordinate leaf.
        # Slice that leaf so JAX scatters a local-moment cotangent back to the
        # corresponding parent atoms rather than to atoms 0..len(atmlst)-1.
        _weak_numpy.require_supported(coords)
        fake_mol.coords = np.asarray(coords)[np.asarray(atmlst, dtype=int)]
    return fake_mol


def _packed_moment_integral(fake_mol, rank, *, bra_derivative=False):
    """Pack shell-row slabs before retaining AO operators or their response.

    Libcint still evaluates its standard component set within each slab.
    The full redundant AO tensor is never allocated: higher-rank slabs are
    limited to 32 MiB (or one shell), and only unique components leave this function.
    The explicit coordinate reverse contracts shell derivatives separately.
    """
    nao = fake_mol.nao
    intor = {0: 'int1e_ovlp', 1: 'int1e_r', 2: 'int1e_rr', 3: 'int1e_rrr'}[rank]
    indices, _ = cartesian_components(rank)
    ncart = 3**rank
    if bra_derivative:
        intor += '_dr10'
        indices = (np.arange(3)[:, None] * ncart + indices).ravel()
        ncart *= 3
    # Overlap and dipoles have no redundant Cartesian components. Avoid
    # packing copies and repeated slab calls, especially for order two.
    if rank <= 1:
        with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
            value = fake_mol.intor(intor)
            return value.reshape(ncart, nao, nao)
    ao_loc = fake_mol.ao_loc_nr()
    rows = max(1, _MOMENT_SLAB_BYTES // max(8 * ncart * nao, 1))
    packed = None
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        start = 0
        while start < fake_mol.nbas:
            end = max(start + 1, int(np.searchsorted(ao_loc, ao_loc[start] + rows, side='right') - 1))
            end = min(end, fake_mol.nbas)
            p0, p1 = ao_loc[start], ao_loc[end]
            block = fake_mol.intor(intor, shls_slice=(start, end, 0, fake_mol.nbas))
            block = block.reshape(ncart, p1 - p0, nao)
            _weak_numpy.require_supported(block)
            if packed is None:
                packed = np.empty((len(indices), nao, nao), dtype=block.dtype)
            packed[:, p0:p1] = np.asarray(block)[indices]
            del block
            start = end
    if packed is None:
        packed = np.empty((len(indices), nao, nao))
    return packed


def _origin_zero_moments(fake_mol, order):
    """Packed host AO moments; coordinate response uses the native VJP."""
    _weak_numpy.require_supported(fake_mol)
    return tuple(_packed_moment_integral(fake_mol, rank).reshape(
        fake_mol.nao, fake_mol.nao) if rank == 0 else
        _packed_moment_integral(fake_mol, rank) if rank <= order else None
        for rank in range(4))


def _native_weak_moments_supported(mf):
    """The shell contraction supplies concrete, real coordinate response only."""
    from pyscfad.gto import Mole
    mol = getattr(mf, 'mol', None)
    if (not isinstance(mol, Mole) or mol.coords is None
            or any(getattr(mol, name, None) is not None
                   for name in ('exp', 'ctr_coeff', 'r0'))
            or any(isinstance(leaf, jax.core.Tracer)
                   for leaf in jax.tree_util.tree_leaves(mf))
            or mol.coords.dtype != np.float64):
        return False
    # GTOint2c_r0_vjp reserves CACHESIZE=144000 doubles for shell output.
    # Bound the largest supported moment (octupole), including contractions;
    # Reject unusually large shells before entering the native contraction.
    largest_shell = int(np.max(np.diff(mol.ao_loc_nr()), initial=0))
    return 0 < largest_shell and 3 * 27 * largest_shell ** 2 <= 144000


def _origin_zero_moments_vjp(mf, atoms, order):
    """Replay raw values, deferring coordinate derivatives until contraction.

    The submolecule coordinate map is reversed by a NumPy scatter. Incoming
    weights combine all endpoints on this support before
    the C driver generates each bra shell block and contracts W + W.T. No
    full bra/ket derivative arrays are constructed or retained.
    """
    if not _native_weak_moments_supported(mf):
        raise TypeError('Weak-pair pullback requires concrete float64 coordinate-only molecules '
                        'within the native moment shell-size limit')

    fake_mol = _fake_mol_with_traced_centers(mf.mol, atoms)
    raw = _origin_zero_moments(fake_mol, order)

    def pullback(raw_bar):
        _weak_numpy.require_supported(raw_bar)

        from pyscfad.gto._moleintor_vjp import _int1e_r0_deriv
        coords_bar = np.zeros((fake_mol.natm, 3), dtype=np.float64)
        names = ('int1e_ovlp', 'int1e_r', 'int1e_rr', 'int1e_rrr')
        with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
            for rank, (name, weight) in enumerate(zip(names, raw_bar)):
                if weight is not None:
                    if rank >= 2:
                        # The forward keeps one representative per symmetric
                        # Cartesian component. Scatter its adjoint once into
                        # libcint's full layout; replicating permutations would
                        # overcount mixed components. This buffer is temporary.
                        packed = np.asarray(weight)
                        weight = np.zeros((3 ** rank, fake_mol.nao, fake_mol.nao),
                                          dtype=packed.dtype)
                        weight[cartesian_components(rank)[0]] = packed
                    # hermi=1 folds both AO legs itself; do not symmetrize
                    # the weights again. The native driver supplies the
                    # nuclear-coordinate minus sign and reduces to atoms.
                    coords_bar += _int1e_r0_deriv(
                        fake_mol._add_suffix(name + '_dr10'), None,
                        3 ** rank, 1, fake_mol, weight)
        mf_bar = _weak_numpy.zeros(mf)
        indices = np.arange(mf.mol.natm) if atoms is None else np.asarray(atoms, dtype=int)
        np.add.at(mf_bar.mol.coords, indices, coords_bar)
        return (mf_bar,)

    return raw, pullback


def _multipole_orbital_data(mol, screen, atoms, order, *, ao_moments=None):
    """Build the one Boys endpoint using translated raw transition moments."""
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")
    occupied = screen.occupied_coeff
    virtual = screen.virtual_coeff
    if occupied.ndim != 2 or occupied.shape[1] != 1:
        raise ValueError("a Boys weak screen requires one occupied target")
    if virtual.ndim != 2 or virtual.shape[0] != occupied.shape[0]:
        raise ValueError("weak-screen occupied and virtual AO dimensions differ")
    if ao_moments is None:
        fake_mol = _fake_mol_with_traced_centers(mol, atoms)
        ao_moments = _origin_zero_moments(fake_mol, max(1, order - 1))
    return _multipole_numpy.endpoint(screen, ao_moments, order)


def build_weak_multipole_screen(common, static, fragment_index, *, _pao_preprojection=None,
                                _occupied_projection_rhs=None):
    """Replay saved primary supports/ranks with a normalized current target."""
    fragment = static.fragments[int(fragment_index)]
    if fragment.weak_virtual is None:
        return None
    return _weak_numpy.fixed_screen(common, static, int(fragment_index),
        candidate=_pao_preprojection, rhs=_occupied_projection_rhs)[0]


@blas_threads(1)
def _finish_weak_screen_reference(occupied, virtual, fock, fragment_index):
    """Finish concrete primary selection without compiling its Fock operations.

    Live response uses the explicit NumPy primary-space pullback. This helper
    serves forward-only reference selection.
    """
    occupied, virtual, fock = map(np.asarray, (occupied, virtual, fock))
    occupied_energy = np.real(np.einsum('ui,uv,vi->i', occupied.conj(), fock, occupied))
    if virtual.shape[1] == 0:
        virtual_energy = np.zeros((0,), dtype=np.real(virtual).dtype)
    else:
        projected = virtual.T.conj() @ fock @ virtual
        projected = .5 * (projected + projected.T.conj())
        if virtual.shape[1] == 1:
            virtual_energy = np.real(projected).reshape(-1)
        else:
            virtual_energy, rotation = scipy_linalg.eigh(projected)
            virtual = virtual @ rotation
    gaps = virtual_energy[:, None] - occupied_energy
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise RuntimeError(f"Boys target {fragment_index} has invalid screening denominators")
    return WeakScreen(occupied_energy, occupied, virtual_energy, virtual)


def _validate_weak_pair(static, left_index, right_index):
    left, right = int(left_index), int(right_index)
    if left == right:
        raise ValueError("a weak pair must contain two distinct fragments")
    left, right = sorted((left, right))
    if not (0 <= left < len(static.fragments) and 0 <= right < len(static.fragments)):
        raise IndexError("weak-pair fragment index is out of range")
    if weak_pair_weight(static, left, right) == 0:
        raise ValueError(f"fragment pair ({left}, {right}) is classified strong")
    return left, right


def weak_screen_pair_energy(mf, left, right, static, left_index, right_index):
    """Evaluate the missing directed shares of one unordered -4 pair."""
    # Preserve endpoint identity when accepting a reversed pair.
    if int(left_index) > int(right_index):
        left, right = right, left
    left_index, right_index = _validate_weak_pair(static, left_index, right_index)
    if left is None or right is None:
        raise RuntimeError("a fixed weak pair has no multipole screen")
    order = static.thresholds.multipole_order
    lhs = _multipole_orbital_data(mf.mol, left, static.fragments[left_index].primary_atoms, order)
    rhs = _multipole_orbital_data(mf.mol, right, static.fragments[right_index].primary_atoms, order)
    return weak_pair_weight(static, left_index, right_index) * np.real(
        _multipole_pair_energy(pad_endpoint(lhs), pad_endpoint(rhs), order))


def build_weak_endpoint(mf, common, static, fragment_index):
    """Build a forward-only endpoint without retaining orbital pullbacks."""
    configure_compilation_cache()
    screen = build_weak_multipole_screen(common, static, fragment_index)
    return _weak_endpoint_from_screen(mf, screen, static, fragment_index)


def _weak_endpoint_from_screen(mf, screen, static, fragment_index, *, ao_moments=None):
    """Contract AO moments using the supplied, differentiable orbital screen."""
    if screen is None:
        raise RuntimeError(f"a fixed weak endpoint {fragment_index} has no multipole screen")
    return _multipole_orbital_data(
        mf.mol, screen, static.fragments[fragment_index].primary_atoms,
        static.thresholds.multipole_order, ao_moments=ao_moments,
    )


def _prepare_weak_endpoint(mf, common, static, fragment_index, *, ao_moments=None,
                           pao_preparation=None, products_preparation=None):
    """Retain the orbital reverse rule from the actual forward evaluation.

    Optional raw AO moments serve only this primal contraction. They are
    never retained by the orbital pullback; replay rebuilds their complete
    molecular dependence when applying the endpoint cotangent.
    """
    moments = partial(_weak_endpoint_from_screen, static=static, fragment_index=fragment_index)
    if ao_moments is not None:
        moments = partial(moments, ao_moments=ao_moments)
    _weak_numpy.require_supported(common)
    if pao_preparation is None:
        screen, screen_pullback = _weak_numpy.fixed_screen(common, static, fragment_index)
    else:
        preprojection, preprojection_pullback = pao_preparation
        fragment = static.fragments[fragment_index]
        if products_preparation is None:
            occupied_index = fragment.fragment_index
            target = np.asarray(common.fragment_occupied_data[occupied_index].iao_coeff)
            overlap = np.asarray(common.s1e)
            screen, target_pb = _weak_numpy.fixed_target(common, preprojection,
                overlap @ target, fragment, static.thresholds, fragment_index)
            def target_pullback(bar):
                common_bar, candidate_bar, rhs_bar = target_pb(bar)
                common_bar.s1e[:] += rhs_bar @ target.T
                common_bar.fragment_occupied_data[occupied_index].iao_coeff[:] += overlap.T @ rhs_bar
                return common_bar, candidate_bar
            screen_pullback = _SharedPAOPullback(target_pullback, preprojection_pullback)
            return moments(mf, screen), (screen, screen_pullback)
        products, products_pullback = products_preparation
        occupied_index = fragment.fragment_index
        screen, target_pullback = _weak_numpy.fixed_target(common, preprojection,
            products.occupied[occupied_index], fragment, static.thresholds, fragment_index)
        screen_pullback = _SharedPAOPullback(target_pullback, preprojection_pullback,
                                             products_pullback, occupied_index)
    return moments(mf, screen), (screen, screen_pullback)


def _replay_weak_endpoint(mf, linearization, static, fragment_index, *,
                          shared_pao_bars=None, shared_product_bars=None):
    """Replay AO moments in the saved screen and compose its orbital response."""
    screen, screen_pullback = linearization
    if isinstance(screen_pullback, _SharedPAOPullback):
        linearization = (screen, lambda bar: (_apply_weak_screen_pullback(
            screen_pullback, bar, shared_pao_bars, shared_product_bars),))
    atoms = static.fragments[fragment_index].primary_atoms
    raw, raw_pullback = _origin_zero_moments_vjp(
        mf, atoms, max(1, static.thresholds.multipole_order - 1))
    endpoint, moment_pullback = _weak_moment_vjp(mf, screen, raw, static, fragment_index)

    def pullback(endpoint_bar):
        screen_bar, raw_bar = moment_pullback(endpoint_bar)
        common_bar, = linearization[1](screen_bar)
        mf_bar, = raw_pullback(raw_bar)
        return mf_bar, common_bar

    return endpoint, pullback


def _weak_moment_vjp(mf, screen, raw, static, index):
    return _multipole_numpy.endpoint(screen, raw, static.thresholds.multipole_order,
                                      with_grad=True)


def weak_pair_energy(mf, common, static, left_index, right_index):
    """Evaluate one pair using the same endpoint boundary as the full sum."""
    left_index, right_index = _validate_weak_pair(static, left_index, right_index)
    return weak_pair_weight(static, left_index, right_index) * np.real(_multipole_pair_energy(
        pad_endpoint(build_weak_endpoint(mf, common, static, left_index)),
        pad_endpoint(build_weak_endpoint(mf, common, static, right_index)),
        static.thresholds.multipole_order,
    ))


def weak_pair_weight(static, left, right):
    """Missing endpoint shares: zero, half, or one unordered pair energy.

    A grouped target may include a partner without the reciprocal domain
    including that target. Each included direction is already in domain MP2.
    """
    return 0.5 * (int(not static.strong_mask[left, right])
                  + int(not static.strong_mask[right, left]))


def weak_pair_indices(static):
    """List pairs with at least one missing domain contribution, once each."""
    return tuple((left, right) for left in range(len(static.fragments))
                 for right in range(left + 1, len(static.fragments))
                 if weak_pair_weight(static, left, right) > 0)


def _weak_endpoints(mf, common, static, pairs):
    indices = sorted({index for pair in pairs for index in pair})
    if len(indices) < len(static.fragments):
        # Partial queries should pay only for their selected PAO parents.
        return {index: build_weak_endpoint(mf, common, static, index) for index in indices}
    return {index: (build_weak_endpoint(mf, common, static, index) if screen is None
                    else _weak_endpoint_from_screen(mf, screen, static, index))
            for index, screen in _forward_weak_screens(common, static, indices)}


def _forward_weak_screens(common, static, indices, *, products=None):
    """Yield current screens sharing primal products, without reverse tapes.

    A None entry denotes an empty or unavailable screen. Each yielded
    screen may be contracted and released before constructing the next.
    """
    preprojections = {}
    for index in indices:
        fragment = static.fragments[index]
        key = (_weak_pao_preprojection_key(fragment)
               if getattr(fragment, 'weak_virtual', None) is not None else None)
        if key is None:
            yield index, None
            continue
        if products is None:
            configure_compilation_cache()
            products = _weak_screen_products(common)
        if key not in preprojections:
            preprojections[key] = _weak_numpy.fixed_preprojection(common, products.pao, fragment)[0]
        yield index, build_weak_multipole_screen(
            common, static, index, _pao_preprojection=preprojections[key],
            _occupied_projection_rhs=products.occupied[fragment.fragment_index])


def _check_endpoint_replay(rebuilt, original, index):
    if not endpoint_matches_reference(rebuilt, original):
        raise RuntimeError(f"weak AO-moment values changed on replay: endpoint {index}")


def weak_pair_energy_sum(mf, common, static, *, pairs=None):
    """Evaluate all weak edges while constructing each endpoint only once."""
    pairs = weak_pair_indices(static) if pairs is None else tuple(pairs)
    with resource_profile.section('stc.domain.weak_prepare'):
        endpoints = {index: pad_endpoint(value) for index, value in
                     _weak_endpoints(mf, common, static, pairs).items()}
        if resource_profile.enabled():
            jax.block_until_ready(endpoints)
    with resource_profile.section('stc.cc.weak_forward', pairs=len(pairs)):
        xp = np
        energy = xp.zeros((), dtype=common.s1e.dtype)
        for left, right in pairs:
            energy += weak_pair_weight(static, left, right) * _multipole_pair_energy(
                endpoints[left], endpoints[right], static.thresholds.multipole_order)
        if resource_profile.enabled():
            jax.block_until_ready(energy)
    return xp.real(energy)


def _grouped_weak_pullback(mf, static, endpoints, originals, linearizations,
                          endpoint_bars, bars, shared_pao_bars, shared_product_bars):
    """Replay one raw AO tape per ordered support and release it promptly."""
    from .adjoint import add_cotangent
    supports = {}
    for index in endpoint_bars:
        atoms = tuple(int(atom) for atom in static.fragments[index].primary_atoms)
        supports.setdefault(atoms, []).append(index)
    order = static.thresholds.multipole_order
    for atoms, indices in supports.items():
        raw, raw_pullback = _origin_zero_moments_vjp(mf, atoms, max(1, order - 1))
        raw_bar = _weak_numpy.zeros(raw)
        for index in indices:
            endpoint_bar = endpoint_bars.pop(index)
            del endpoints[index]
            original = originals.pop(index)
            screen, screen_pullback = linearizations.pop(index)
            rebuilt, moment_pullback = _weak_moment_vjp(mf, screen, raw, static, index)
            _check_endpoint_replay(rebuilt, original, index)
            screen_bar, operators_bar = moment_pullback(
                crop_endpoint_cotangent(endpoint_bar, original))
            common_bar = _apply_weak_screen_pullback(
                screen_pullback, screen_bar, shared_pao_bars, shared_product_bars)
            bars = (bars[0], jax.tree_util.tree_map(add_cotangent, bars[1], common_bar))
            raw_bar = jax.tree_util.tree_map(add_cotangent, raw_bar, operators_bar)
            jax.block_until_ready((bars[1], raw_bar))
            del (rebuilt, moment_pullback, original, endpoint_bar, screen,
                 screen_pullback, screen_bar, operators_bar, common_bar)
        mf_bar, = raw_pullback(raw_bar)
        bars = (jax.tree_util.tree_map(add_cotangent, bars[0], mf_bar), bars[1])
        jax.block_until_ready(bars)
        del raw, raw_pullback, raw_bar, mf_bar
    return bars


def weak_pair_energy_and_grad(mf, common, static, *, pairs=None,
                             prepared_endpoints=None, _grouped_ao_response=False,
                             _prepared_pairs=None):
    """Reuse orbital pullbacks and replay one AO-moment tape at a time.

    Each orbital screen and its reverse rule come from one forward evaluation.
    Pair contractions share compact transition moments. Their accumulated
    bars pass through AO-moment replay using the exact saved screen, then
    through its retained orbital pullback. No eigensolver is replayed.

    ``prepared_endpoints`` transfers ownership of screening preparations
    made at these exact ``mf`` and ``common`` inputs. Entries are consumed
    once; missing endpoints are prepared here and unused entries released.
    The caller must retain the screening weak selections and only refine
    the strong mask before consuming these preparations.

    The selection-owned path may share the raw AO response across endpoints
    with identical ordered supports, retaining only one support's AO tape.
    Its ``_prepared_pairs`` transfers the screening energy and accumulated
    endpoint bars, so pair contractions are not evaluated a second time.
    """
    from .adjoint import add_cotangent
    configure_compilation_cache()
    pairs = weak_pair_indices(static) if pairs is None else tuple(pairs)
    _weak_numpy.require_supported(mf, common)
    xp = np
    bars = _weak_numpy.zeros((mf, common))
    energy = xp.zeros((), dtype=common.s1e.dtype)
    originals, linearizations = {}, {}
    with resource_profile.section('stc.domain.weak_prepare'):
        for index in sorted({index for pair in pairs for index in pair}):
            if prepared_endpoints is not None and index in prepared_endpoints:
                originals[index], linearizations[index] = prepared_endpoints.pop(index)
            else:
                originals[index], linearizations[index] = _prepare_weak_endpoint(
                    mf, common, static, index)
        if prepared_endpoints is not None:
            # Strong-only endpoints have no response. Drop their orbital tapes
            # before accumulating bars for the final weak graph.
            prepared_endpoints.clear()
        # Count only endpoints in the final weak graph. Clearing unused
        # preparations above releases groups without any weak contribution.
        shared_pao_bars = {pullback: (count, None) for pullback, count in Counter(
            value[1].preprojection_pullback for value in linearizations.values()
            if isinstance(value[1], _SharedPAOPullback)).items()}
        shared_product_bars = {pullback: (count, None) for pullback, count in Counter(
            value[1].products_pullback.pullback for value in linearizations.values()
            if isinstance(value[1], _SharedPAOPullback)
            and value[1].products_pullback is not None).items()}
        endpoints = {index: pad_endpoint(value) for index, value in originals.items()}
        zero = _weak_numpy.zeros
        endpoint_bars = {index: zero(value) for index, value in endpoints.items()}
        if resource_profile.enabled():
            jax.block_until_ready((endpoints, endpoint_bars))
    if _prepared_pairs is not None:
        if _prepared_pairs['pairs'] != tuple(sorted(pairs)):
            raise ValueError('screening pair response does not match the final weak graph')
        if set(_prepared_pairs['endpoint_bars']) != set(endpoints):
            raise ValueError('screening pair response has incompatible endpoints')
        with resource_profile.section('stc.cc.weak_pair_reuse', pairs=len(pairs)):
            energy = _prepared_pairs.pop('energy')
            endpoint_bars = _prepared_pairs.pop('endpoint_bars')
            _prepared_pairs.clear()
    else:
        with resource_profile.section('stc.cc.weak_pair_value_and_grad', pairs=len(pairs)):
            for left, right in pairs:
                value, (left_bar, right_bar) = pair_energy_and_grad(
                    endpoints[left], endpoints[right], static.thresholds.multipole_order)
                weight = weak_pair_weight(static, left, right)
                value = weight * value
                left_bar, right_bar = jax.tree_util.tree_map(
                    lambda x: weight * x, (left_bar, right_bar))
                endpoint_bars[left] = jax.tree_util.tree_map(add_cotangent, endpoint_bars[left], left_bar)
                endpoint_bars[right] = jax.tree_util.tree_map(add_cotangent, endpoint_bars[right], right_bar)
                energy += value
                del left_bar, right_bar
            if resource_profile.enabled():
                jax.block_until_ready((energy, endpoint_bars))
    with resource_profile.section('stc.domain.weak_pullback', includes_replay=True):
        if _grouped_ao_response:
            bars = _grouped_weak_pullback(
                mf, static, endpoints, originals, linearizations, endpoint_bars,
                bars, shared_pao_bars, shared_product_bars)
        else:
            for index in tuple(endpoint_bars):
                endpoint_bar = endpoint_bars.pop(index)
                del endpoints[index]
                original = originals.pop(index)
                linearization = linearizations.pop(index)
                if isinstance(linearization[1], _SharedPAOPullback):
                    rebuilt, pullback = _replay_weak_endpoint(
                        mf, linearization, static, index, shared_pao_bars=shared_pao_bars,
                        shared_product_bars=shared_product_bars)
                else:
                    rebuilt, pullback = _replay_weak_endpoint(mf, linearization, static, index)
                _check_endpoint_replay(rebuilt, original, index)
                contribution = pullback(crop_endpoint_cotangent(endpoint_bar, original))
                bars = jax.tree_util.tree_map(add_cotangent, bars, contribution)
                jax.block_until_ready(bars)
                del rebuilt, original, pullback, contribution, endpoint_bar, linearization
    if pairs:
        # Release each tape promptly, but scan the Python heap only after
        # the complete weak response. Automatic collection remains enabled.
        gc.collect()
    return xp.real(energy), bars


def _exact_boys_pair_scores(mf, common, static, *, correlation_auxbasis=None):
    """Concrete exact classification through two labelled occupied contractions.

    Project amplitudes only after canonical denominators have been applied.
    This is the native exact screening convention (twice directed OS,
    then combine orientations), without constructing occupied weight matrices.
    One virtual pair is held at a time.
    """
    from pyscfad.mp import dfmp2
    pt = dfmp2.MP2(mf, frozen=static.frozen)
    eris = pt.ao2mo()
    nocc, nvir = int(pt.nocc), int(pt.nmo - pt.nocc)
    from ._df_basis import can_reuse_scf_df
    if correlation_auxbasis is None or can_reuse_scf_df(mf, correlation_auxbasis):
        lov = pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)
    else:
        from pyscfad.lno.df import get_local_Lov
        lov = get_local_Lov(mf, eris.mo_coeff, nocc, np.arange(mf.mol.natm),
                            integral_direct=True, correlation_auxbasis=correlation_auxbasis)
    lov = np.asarray(lov).reshape(-1, nocc, nvir)
    rotation = np.asarray(common.occupied_coeff).T @ np.asarray(common.s1e) @ np.asarray(common.iao_coeff)
    rotation = rotation[:, [int(labels[0]) for labels in static.frag_lolist]]
    energies = np.asarray(eris.mo_energy)
    gaps = energies[:nocc, None] - energies[None, nocc:]
    directed = np.zeros((rotation.shape[1],) * 2)
    for a in range(nvir):
        for b in range(nvir):
            integrals = lov[:, :, a].T @ lov[:, :, b]
            denominator = gaps[:, a, None] + gaps[None, :, b]
            if np.any(np.abs(denominator) < 1e-14):
                raise ValueError("an MP2 energy denominator is zero")
            target_integrals = rotation.T @ integrals @ rotation
            target_amplitudes = rotation.T @ (integrals / denominator) @ rotation
            directed += target_integrals * target_amplitudes
    directed *= 2
    physical = directed + directed.T
    np.fill_diagonal(physical, np.diag(directed))
    return physical


def _link_compact_domains(compact, strength, forced, strong, threshold):
    """Promote internal pairs in place and map tagged targets to shared virtual support."""
    # Each BP list contains distinct atom indices. The product counts shared atoms.
    sizes = np.array([len(atoms) for atoms in compact])
    atoms = np.concatenate(compact).astype(np.intp, copy=False)
    incidence = csr_matrix((np.ones(atoms.size, dtype=np.int32), atoms,
                            np.r_[0, sizes.cumsum()]),
                           shape=(len(compact), int(atoms.max(initial=-1)) + 1))
    _, labels = connected_components((incidence @ incidence.T) >= 2, directed=False)
    linked_members = {}
    for label in np.flatnonzero(np.bincount(labels) > 1):
        members = np.flatnonzero(labels == label)
        block = np.ix_(members, members)
        internal = strength[block]
        # Keep component eligibility fixed at the normal cutoff throughout a sweep.
        if np.any((internal <= 1e-5) & ~forced[block]):
            strong[block] |= internal > threshold / 100
            for member in members:
                linked_members[member] = members
    return linked_members


class _PairScreen(NamedTuple):
    pair_energy: np.ndarray
    weak_energy: np.ndarray
    forced: np.ndarray
    strong_pairs: np.ndarray  # symmetric, before target grouping
    domain_mask: np.ndarray   # directed, after target grouping
    linked_members: dict


def _screen_pair_response(endpoints, static, mask, response):
    """Differentiate only missing shares in the final occupied domains."""
    from .adjoint import add_cotangent
    order = static.thresholds.multipole_order
    selected = SimpleNamespace(strong_mask=mask, fragments=static.fragments)
    pairs = weak_pair_indices(selected)
    accumulated = {}
    xp = np
    energy = xp.zeros(())
    with resource_profile.section('stc.domain.screen_pair_value_and_grad'):
        for left, right in pairs:
            weight = weak_pair_weight(selected, left, right)
            value, bars = pair_energy_and_grad(endpoints[left], endpoints[right], order)
            energy += weight * value
            for index, bar in zip((left, right), bars):
                bar = jax.tree_util.tree_map(lambda x: weight * x, bar)
                accumulated[index] = (bar if index not in accumulated else
                    jax.tree_util.tree_map(add_cotangent, accumulated[index], bar))
            jax.block_until_ready((energy, accumulated))
    response.update(pairs=pairs, energy=xp.real(energy), endpoint_bars=accumulated)


def screen_pairs(mf, common, static, model='multipole', full_domain=False, **kwargs):
    """Return the pair scores, weak estimates, forced pairs and raw strong pairs."""
    return screen_pair_domains(mf, common, static, model, full_domain, **kwargs)[:4]


def screen_pair_domains(mf, common, static, model='multipole', full_domain=False,
                        *, prepared_endpoints=None, _products_preparation=None, _products=None,
                        _pair_response=None, correlation_auxbasis=None, _reference_endpoints=None,
                        _primary_preparations=None):
    """Classify pairs and form the membership masks used by both energies and forces.

    When supplied, ``prepared_endpoints`` is cleared and filled with each
    available endpoint and its orbital pullback from this evaluation. The
    mapping belongs to this exact reference and is consumed by
    :func:`weak_pair_energy_and_grad` after the final strong mask is known.
    Private product inputs belong to the same reference: a preparation carries
    its ambient pullback; plain products serve forward-only selection.
    ``_reference_endpoints`` reuses compact endpoints built from that same
    forward selection's primary spaces. It must not replace response replay.
    ``_primary_preparations`` transfers the live primary orbital reverse rules
    constructed during selection, avoiding a second numerical space build.
    ``_pair_response`` receives the final weak energy and compact endpoint bars
    from the same pair evaluations used for classification.
    """
    model = str(model).lower().replace('_', '-')
    if _reference_endpoints is not None and prepared_endpoints is not None:
        raise ValueError('reference endpoints cannot replace response replay')
    if _primary_preparations is not None and prepared_endpoints is None:
        raise ValueError('primary response preparations require prepared endpoints')
    if model not in ('multipole', 'multipole-os', 'os-multipole', 'exact', 'all', 'all-strong'):
        raise ValueError(f"unsupported Boys pair energy model {model!r}")
    if prepared_endpoints is not None:
        prepared_endpoints.clear()
    if _pair_response is not None:
        if prepared_endpoints is None:
            raise ValueError('pair response requires prepared screening endpoints')
        _pair_response.clear()
        xp = np
        _pair_response.update(pairs=(), energy=xp.zeros(()), endpoint_bars={})
    nfrag = len(static.fragments)
    if full_domain or static.thresholds.pair_energy <= 0 or model in ('all', 'all-strong'):
        strong = np.ones((nfrag, nfrag), dtype=bool)
        return _PairScreen(np.zeros((nfrag, nfrag)), np.zeros((nfrag, nfrag)),
                           strong.copy(), strong, strong, {})
    order = static.thresholds.multipole_order
    if _reference_endpoints is not None:
        if set(_reference_endpoints) != set(range(nfrag)):
            raise ValueError('reference endpoints must cover every fragment')
        endpoints = [_reference_endpoints[index] for index in range(nfrag)]
    elif prepared_endpoints is None:
        endpoints = [None] * nfrag
        products = _products if _products_preparation is None else _products_preparation[0]
        for index, screen in _forward_weak_screens(common, static, range(nfrag), products=products):
            if screen is None:
                screen = build_weak_multipole_screen(common, static, index)
            if screen is not None:
                endpoints[index] = pad_endpoint(_multipole_orbital_data(
                    mf.mol, screen, static.fragments[index].primary_atoms, order))
    else:
        endpoints = [None] * nfrag
        supports, pao_preparations = {}, {}
        products_preparation = _products_preparation
        if products_preparation is None and _products is not None:
            products_preparation = _cached_weak_screen_products(common, _products)
        for index, fragment in enumerate(static.fragments):
            if fragment.weak_virtual is not None:
                atoms = tuple(int(atom) for atom in fragment.primary_atoms)
                supports.setdefault(atoms, []).append(index)
        for atoms, indices in supports.items():
            # Raw AO operators depend on the ordered support, not the target.
            # Keep only this group's arrays; no AO reverse tape is made here.
            fake_mol = _fake_mol_with_traced_centers(mf.mol, atoms)
            ao_moments = _origin_zero_moments(fake_mol, max(1, order - 1))
            for index in indices:
                if _primary_preparations is not None and index in _primary_preparations:
                    linearization = _primary_preparations.pop(index)
                    endpoint = _weak_endpoint_from_screen(
                        mf, linearization[0], static, index, ao_moments=ao_moments)
                    prepared_endpoints[index] = endpoint, linearization
                    endpoints[index] = pad_endpoint(endpoint)
                    del endpoint, linearization
                    continue
                fragment = static.fragments[index]
                key = _weak_pao_preprojection_key(fragment)
                preparation = {}
                if key is not None:
                    if products_preparation is None:
                        products_preparation = _weak_numpy.products(common)
                    if key not in pao_preparations:
                        pao_preparations[key] = _weak_numpy.fixed_preprojection(
                            common, products_preparation[0].pao, fragment)
                    preparation['pao_preparation'] = pao_preparations[key]
                    preparation['products_preparation'] = products_preparation
                prepared_endpoints[index] = _prepare_weak_endpoint(
                    mf, common, static, index, ao_moments=ao_moments, **preparation)
                endpoints[index] = pad_endpoint(prepared_endpoints[index][0])
            del ao_moments, fake_mol
        pao_preparations.clear()
    weak = np.zeros((nfrag, nfrag))
    forced = np.eye(nfrag, dtype=bool)
    coordinates = np.asarray(mf.mol.atom_coords())
    for left in range(nfrag):
        for right in range(left):
            atoms_left = static.frag_atmlist[left]
            atoms_right = static.frag_atmlist[right]
            if atoms_left is None or len(atoms_left) == 0:
                atoms_left = static.compact_bp_domain[left]
            if atoms_right is None or len(atoms_right) == 0:
                atoms_right = static.compact_bp_domain[right]
            separation = np.linalg.norm(coordinates[np.asarray(atoms_left), None, :]
                                        - coordinates[None, np.asarray(atoms_right), :], axis=-1).min()
            if (endpoints[left] is None or endpoints[right] is None
                    or separation < static.thresholds.near_pair_distance):
                forced[left, right] = forced[right, left] = True
                continue
            value = float(_multipole_pair_energy(endpoints[left], endpoints[right], order))
            if not np.isfinite(value):
                raise RuntimeError(f"nonfinite Boys screening pair estimate for targets {left}, {right}")
            weak[left, right] = weak[right, left] = value
    from ._df_basis import basis_kwargs
    exact = (_exact_boys_pair_scores(mf, common, static, **basis_kwargs(correlation_auxbasis))
             if model == 'exact' else None)
    pair = exact if exact is not None else weak
    strength = np.abs(pair)
    threshold = static.thresholds.pair_energy
    strong = (strength > threshold) | forced
    linked_members = {}
    if getattr(static, 'domain_link', False):
        linked_members = _link_compact_domains(static.compact_bp_domain, strength, forced, strong, threshold)
    strong = strong | strong.T
    np.fill_diagonal(strong, True)
    mask = strong
    if getattr(static, 'boys_grouping', 'singleton') in ('atom', 'atom_singleton'):
        from ._selection import _group_domain_mask
        mask = _group_domain_mask(strong, static.physical_groups)
    if _pair_response is not None:
        _screen_pair_response(endpoints, static, mask, _pair_response)
    return _PairScreen(pair, weak, forced, strong, mask, linked_members)
