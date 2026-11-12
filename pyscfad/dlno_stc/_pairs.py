"""Boys multipole screening and directed weak-pair completion.

Moment translation, Cartesian response contractions, and pair kernels are
shared through ``pyscfad.lib._multipole``. Fixed primary-screen reconstruction
uses each singleton Boys target. Weak pairs accumulate compact endpoint
cotangents before replaying each endpoint response once; no native DLNO
module is imported by this implementation.
"""
import gc
from collections import Counter
from types import SimpleNamespace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad.lno.tools import fake_mol_by_atom, project_mo
from pyscfad.tools import resource_profile
from pyscfad.lib._compilation_cache import configure_compilation_cache
from pyscfad.lib._multipole import (
    coordinate_moment_jvp, endpoint_matches_reference,
    pad_endpoint, crop_endpoint_cotangent,
    pair_energy as _multipole_pair_energy,
    pair_energy_and_grad, transition_moments,
)
from pyscfad.lib._weak_endpoint import prepare_endpoint, replay_endpoint


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
    from .domain import _pao_products
    targets = tuple(target.iao_coeff for target in common.fragment_occupied_data)
    if targets:
        combined = common.s1e @ jnp.concatenate(targets, axis=1)
        offsets = np.cumsum([target.shape[1] for target in targets[:-1]])
        occupied = tuple(jnp.split(combined, offsets, axis=1))
    else:
        occupied = ()
    return _WeakScreenProducts(_pao_products(common.pao_coeff, common.s1e), occupied)


def _cached_weak_screen_products(common, products):
    """Attach the exact ambient response without recomputing cached primals."""
    overlap, pao = common.s1e, common.pao_coeff
    overlap_pao = products.pao[1]
    targets = tuple(target.iao_coeff for target in common.fragment_occupied_data)
    combined = jnp.concatenate(targets, axis=1) if targets else None
    offsets = np.cumsum([target.shape[1] for target in targets[:-1]])

    def differential(tangent):
        overlap_t, pao_t = tangent.s1e, tangent.pao_coeff
        overlap_pao_t = overlap_t @ pao + overlap @ pao_t
        gram_t = pao_t.T.conj() @ overlap_pao + pao.T.conj() @ overlap_pao_t
        occupied_t = ()
        if targets:
            combined_t = jnp.concatenate(
                tuple(target.iao_coeff for target in tangent.fragment_occupied_data), axis=1)
            product_t = overlap_t @ combined + overlap @ combined_t
            occupied_t = tuple(jnp.split(product_t, offsets, axis=1))
        return _WeakScreenProducts((gram_t, overlap_pao_t), occupied_t)

    metadata = lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype)
    pullback = jax.linear_transpose(differential, jax.tree_util.tree_map(metadata, common))
    return products, _WeakProductsPullback(pullback, jax.tree_util.tree_map(metadata, products))


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
                    lambda shape: jnp.zeros(shape.shape, shape.dtype), products.shapes)
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
        fake_mol.coords = coords[jnp.asarray(atmlst, dtype=int)]
    return fake_mol


def _raw_cartesian_moment_impl(fake_mol, rank):
    """Evaluate an AO Cartesian moment about the fixed global origin."""
    nao = fake_mol.nao
    intor = {2: 'int1e_rr', 3: 'int1e_rrr'}[rank]
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        moment = jnp.asarray(fake_mol.intor(intor))
    return moment.reshape((3,) * rank + (nao, nao))


def _raw_cartesian_moment_coordinate_jvp(fake_mol, coords_t, rank):
    """Coordinate JVP with the Cartesian operator axes kept in order.

    For ``int1e_rr[_r]_dr01`` libcint places the ket derivative axis after
    the two or three Cartesian operator axes.  The generic two-centre
    integral JVP treats it as the leading axis, which is only correct for a
    scalar operator.  Keep the reordering local to the multipole operators
    until the general integral rule supports arbitrary Cartesian rank.
    """
    nao = fake_mol.nao
    intor = {2: 'int1e_rr', 3: 'int1e_rrr'}[rank]
    ncart = 3 ** rank
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        bra = -jnp.asarray(fake_mol.intor(f'{intor}_dr10'))
        ket = -jnp.asarray(fake_mol.intor(f'{intor}_dr01'))
    bra = bra.reshape(3, ncart, nao, nao)
    ket = ket.reshape(ncart, 3, nao, nao).transpose(1, 0, 2, 3)

    ao_owner = np.empty(nao, dtype=np.int32)
    for atom, (_, _, p0, p1) in enumerate(fake_mol.aoslice_by_atom()):
        ao_owner[p0:p1] = atom
    tangent = coordinate_moment_jvp(bra, ket, coords_t, ao_owner)
    return tangent.reshape((3,) * rank + (nao, nao))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _raw_cartesian_moment(fake_mol, rank):
    """Cartesian moment with a rank-aware nuclear-coordinate JVP."""
    return _raw_cartesian_moment_impl(fake_mol, rank)


@_raw_cartesian_moment.defjvp
def _raw_cartesian_moment_jvp(rank, primals, tangents):
    fake_mol, = primals
    fake_mol_t, = tangents
    primal = _raw_cartesian_moment(fake_mol, rank)

    # Preserve the pre-existing exponent/contraction-coefficient response by
    # asking the general integral rule for a JVP with its coordinate tangent
    # zeroed.  The coordinate part is supplied below with the correct
    # Cartesian component ordering.
    other_tangent = jnp.zeros_like(primal)
    if any(
        getattr(fake_mol, name, None) is not None
        for name in ('exp', 'ctr_coeff', 'r0')
    ):
        basis_tangent = fake_mol_t.copy(deep=False)
        if basis_tangent.coords is not None:
            basis_tangent.coords = jnp.zeros_like(basis_tangent.coords)
        _, other_tangent = jax.jvp(
            partial(_raw_cartesian_moment_impl, rank=rank),
            (fake_mol,),
            (basis_tangent,),
        )

    coordinate_tangent = jnp.zeros_like(primal)
    if fake_mol.coords is not None:
        coordinate_tangent = _raw_cartesian_moment_coordinate_jvp(
            fake_mol, fake_mol_t.coords, rank
        )
    return primal, other_tangent + coordinate_tangent


def _origin_zero_moments(fake_mol, order):
    """Evaluate raw Cartesian moments about the fixed global origin.

    PySCF stores the common origin in a NumPy ``_env`` buffer, so its
    ``with_common_origin`` context cannot consume a traced, geometry-dependent
    origin.  A concrete zero origin is safe inside a JAX trace; the moments
    can then be translated with ordinary JAX algebra.
    """
    nao = fake_mol.nao
    with fake_mol.with_common_origin((0.0, 0.0, 0.0)):
        overlap = jnp.asarray(fake_mol.intor('int1e_ovlp')).reshape(nao, nao)
        r = jnp.asarray(fake_mol.intor('int1e_r')).reshape(3, nao, nao)
        rr = None
        rrr = None
        # A plain PySCF Mole is not a registered JAX pytree, so it cannot be
        # passed through the custom-JVP wrapper merely because some unrelated
        # input (for example an orbital energy) is traced.  Use the identical
        # primal integral call when this molecule has no differentiable basis
        # or coordinate leaves.
        moment = _raw_cartesian_moment
        if not any(
            getattr(fake_mol, name, None) is not None
            for name in ('coords', 'exp', 'ctr_coeff', 'r0')
        ):
            moment = _raw_cartesian_moment_impl
        if order >= 2:
            rr = moment(fake_mol, 2)
        if order >= 3:
            rrr = moment(fake_mol, 3)
    return overlap, r, rr, rrr


def _multipole_orbital_data(mol, screen, atoms, order, *, ao_moments=None):
    """Build the one Boys endpoint using translated raw transition moments."""
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")
    occupied = jnp.asarray(screen.occupied_coeff)
    virtual = jnp.asarray(screen.virtual_coeff)
    if occupied.ndim != 2 or occupied.shape[1] != 1:
        raise ValueError("a Boys weak screen requires one occupied target")
    if virtual.ndim != 2 or virtual.shape[0] != occupied.shape[0]:
        raise ValueError("weak-screen occupied and virtual AO dimensions differ")
    if ao_moments is None:
        fake_mol = _fake_mol_with_traced_centers(mol, atoms)
        ao_moments = _origin_zero_moments(fake_mol, max(1, order - 1))
    overlap, dipole, second, third = ao_moments
    center, mu, theta, omega = transition_moments(
        occupied, virtual, overlap, dipole, second, third, order
    )
    excitation = jnp.asarray(screen.virtual_energy) - jnp.asarray(screen.occupied_energy)[0]
    return (center[0], mu[0], excitation,
            None if theta is None else theta[0],
            None if omega is None else omega[0])


def build_weak_multipole_screen(common, static, fragment_index, *, _pao_preprojection=None,
                                _occupied_projection_rhs=None):
    """Replay saved primary supports/ranks with a normalized current target."""
    from .domain import (_fixed_metric_orthonormalize, _fixed_semicanonicalize,
                         _rebuild_selected_pao_space)
    fragment = static.fragments[int(fragment_index)]
    if fragment.weak_virtual is None:
        return None
    ao = fragment.primary_ao_indices
    overlap = common.s1e[np.ix_(ao, ao)]
    fock = common.fock[np.ix_(ao, ao)]
    if _occupied_projection_rhs is None:
        occupied = project_mo(common.fragment_occupied_data[fragment.fragment_index].iao_coeff,
                              common.s1e[ao], overlap)
    else:
        occupied = jnp.linalg.solve(overlap, _occupied_projection_rhs[ao])
    norm2 = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), overlap, occupied))
    current = np.asarray(jax.device_get(jax.lax.stop_gradient(norm2)))
    keep = fragment.weak_occ_norm_keep
    if (not np.isfinite(current).all()
            or not np.array_equal(np.flatnonzero(current > static.thresholds.metric_rank), keep)
            or len(keep) != 1):
        raise RuntimeError("Boys screening target rank changed; rebuild domains")
    occupied = occupied[:, keep] / jnp.sqrt(norm2[keep])[None, :]
    occupied_energy = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), fock, occupied))
    span = _fixed_metric_orthonormalize(occupied, overlap, fragment.weak_occ_span_metric_keep)
    virtual = _rebuild_selected_pao_space(
        common, fragment.weak_virtual, ao, span,
        boys_metric_threshold=static.thresholds.metric_rank,
        preprojection=_pao_preprojection,
    )
    virtual_energy, virtual = _fixed_semicanonicalize(virtual, fock)
    gaps = np.asarray(jax.device_get(jax.lax.stop_gradient(virtual_energy[:, None] - occupied_energy)))
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
    return weak_pair_weight(static, left_index, right_index) * jnp.real(
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
    if pao_preparation is not None:
        preprojection, preprojection_pullback = pao_preparation
        if products_preparation is not None:
            products, products_pullback = products_preparation
            occupied_index = static.fragments[fragment_index].fragment_index
            screen, target_pullback = jax.vjp(
                lambda current, candidate, rhs: build_weak_multipole_screen(
                    current, static, fragment_index, _pao_preprojection=candidate,
                    _occupied_projection_rhs=rhs),
                common, preprojection, products.occupied[occupied_index])
            return moments(mf, screen), (screen, _SharedPAOPullback(
                target_pullback, preprojection_pullback, products_pullback, occupied_index))
        screen, target_pullback = jax.vjp(
            lambda current, candidate: build_weak_multipole_screen(
                current, static, fragment_index, _pao_preprojection=candidate),
            common, preprojection)
        return moments(mf, screen), (screen, _SharedPAOPullback(
            target_pullback, preprojection_pullback))
    return prepare_endpoint(
        partial(build_weak_multipole_screen, static=static, fragment_index=fragment_index),
        moments, mf, common)


def _replay_weak_endpoint(mf, linearization, static, fragment_index, *,
                          shared_pao_bars=None, shared_product_bars=None):
    """Replay AO moments in the saved screen and compose its orbital response."""
    screen, screen_pullback = linearization
    if isinstance(screen_pullback, _SharedPAOPullback):
        linearization = (screen, lambda bar: (_apply_weak_screen_pullback(
            screen_pullback, bar, shared_pao_bars, shared_product_bars),))
    return replay_endpoint(
        partial(_weak_endpoint_from_screen, static=static, fragment_index=fragment_index),
        mf, linearization)


def weak_pair_energy(mf, common, static, left_index, right_index):
    """Evaluate one pair using the same endpoint boundary as the full sum."""
    left_index, right_index = _validate_weak_pair(static, left_index, right_index)
    return weak_pair_weight(static, left_index, right_index) * jnp.real(_multipole_pair_energy(
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

    A None entry requests the caller's legacy endpoint path. Each yielded
    screen may be contracted and released before constructing the next.
    """
    from .domain import _rebuild_selected_pao_preprojection
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
            preprojections[key] = _rebuild_selected_pao_preprojection(
                common, products.pao, selection=fragment.weak_virtual,
                representation_ao_indices=fragment.primary_ao_indices)
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
        energy = jnp.zeros((), dtype=common.s1e.dtype)
        for left, right in pairs:
            energy += weak_pair_weight(static, left, right) * _multipole_pair_energy(
                endpoints[left], endpoints[right], static.thresholds.multipole_order)
        if resource_profile.enabled():
            jax.block_until_ready(energy)
    return jnp.real(energy)


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
        raw, raw_pullback = jax.vjp(
            lambda current: _origin_zero_moments(
                _fake_mol_with_traced_centers(current.mol, atoms), max(1, order - 1)),
            mf)
        raw_bar = jax.tree_util.tree_map(jnp.zeros_like, raw)
        for index in indices:
            endpoint_bar = endpoint_bars.pop(index)
            del endpoints[index]
            original = originals.pop(index)
            screen, screen_pullback = linearizations.pop(index)
            rebuilt, moment_pullback = jax.vjp(
                lambda saved, operators: _weak_endpoint_from_screen(
                    mf, saved, static, index, ao_moments=operators),
                screen, raw)
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

    Each orbital screen and its reverse rule come from one forward VJP.
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
    from .adjoint import add_cotangent, zero_term_cotangents
    configure_compilation_cache()
    pairs = weak_pair_indices(static) if pairs is None else tuple(pairs)
    bars = zero_term_cotangents(mf, common)
    energy = jnp.zeros((), dtype=common.s1e.dtype)
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
        endpoint_bars = {index: jax.tree_util.tree_map(jnp.zeros_like, value)
                         for index, value in endpoints.items()}
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
    return jnp.real(energy), bars


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


def _screen_pair_response(endpoints, static, forced, exact_scores, response):
    """Classify orbital pairs, then differentiate only missing domain shares."""
    from .adjoint import add_cotangent
    from ._selection import _group_domain_mask
    nfrag = len(endpoints)
    weak = np.zeros((nfrag, nfrag))
    order = static.thresholds.multipole_order
    for left in range(nfrag):
        for right in range(left):
            if forced[left, right]:
                continue
            value = float(_multipole_pair_energy(endpoints[left], endpoints[right], order))
            if not np.isfinite(value):
                raise RuntimeError(f'nonfinite Boys screening pair estimate for targets {left}, {right}')
            weak[left, right] = weak[right, left] = value
    scores = weak if exact_scores is None else exact_scores
    mask = (np.abs(scores) > static.thresholds.pair_energy) | forced
    if getattr(static, 'boys_grouping', 'singleton') == 'atom':
        mask = _group_domain_mask(mask, static.physical_groups)
    selected = SimpleNamespace(strong_mask=mask, fragments=static.fragments)
    pairs = weak_pair_indices(selected)
    accumulated = {}
    energy = jnp.zeros(())
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
    response.update(pairs=pairs, energy=jnp.real(energy), endpoint_bars=accumulated)
    return weak


def screen_pairs(mf, common, static, model='multipole', force_full_domains=False,
                 *, prepared_endpoints=None, _products_preparation=None, _products=None,
                 _pair_response=None, correlation_auxbasis=None):
    """Concrete Boys pair classification on provisional primary selections.

    When supplied, ``prepared_endpoints`` is cleared and filled with each
    available endpoint and its orbital pullback from this evaluation. The
    mapping belongs to this exact reference and is consumed by
    :func:`weak_pair_energy_and_grad` after the final strong mask is known.
    Private product inputs belong to the same reference: a preparation carries
    its ambient pullback; plain products serve forward-only selection.
    ``_pair_response`` receives the final weak energy and compact endpoint bars
    from the same pair evaluations used for classification.
    """
    model = str(model).lower().replace('_', '-')
    if model not in ('multipole', 'multipole-os', 'os-multipole', 'exact', 'all', 'all-strong'):
        raise ValueError(f"unsupported Boys pair energy model {model!r}")
    if prepared_endpoints is not None:
        prepared_endpoints.clear()
    if _pair_response is not None:
        if prepared_endpoints is None:
            raise ValueError('pair response requires prepared screening endpoints')
        _pair_response.clear()
        _pair_response.update(pairs=(), energy=jnp.zeros(()), endpoint_bars={})
    nfrag = len(static.fragments)
    if force_full_domains or static.thresholds.pair_energy <= 0 or model in ('all', 'all-strong'):
        return (np.zeros((nfrag, nfrag)), np.zeros((nfrag, nfrag)),
                np.ones((nfrag, nfrag), dtype=bool), np.ones((nfrag, nfrag), dtype=bool))
    order = static.thresholds.multipole_order
    if prepared_endpoints is None:
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
                fragment = static.fragments[index]
                key = _weak_pao_preprojection_key(fragment)
                preparation = {}
                if key is not None:
                    if products_preparation is None:
                        products, products_pullback = jax.vjp(_weak_screen_products, common)
                        shapes = jax.tree_util.tree_map(
                            lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), products)
                        products_preparation = products, _WeakProductsPullback(products_pullback, shapes)
                    if key not in pao_preparations:
                        from .domain import _rebuild_selected_pao_preprojection
                        pao_preparations[key] = jax.vjp(partial(
                            _rebuild_selected_pao_preprojection,
                            selection=fragment.weak_virtual,
                            representation_ao_indices=fragment.primary_ao_indices),
                            common, products_preparation[0].pao)
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
            if _pair_response is None:
                value = float(_multipole_pair_energy(endpoints[left], endpoints[right], order))
                if not np.isfinite(value):
                    raise RuntimeError(f"nonfinite Boys screening pair estimate for targets {left}, {right}")
                weak[left, right] = weak[right, left] = value
    from ._df_basis import basis_kwargs
    exact = (_exact_boys_pair_scores(mf, common, static, **basis_kwargs(correlation_auxbasis))
             if model == 'exact' else None)
    if _pair_response is not None:
        weak = _screen_pair_response(endpoints, static, forced, exact, _pair_response)
    pair = exact if exact is not None else weak
    strong = (np.abs(pair) > static.thresholds.pair_energy) | forced
    strong = strong | strong.T
    np.fill_diagonal(strong, True)
    return pair, weak, forced, strong
