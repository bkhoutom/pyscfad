"""Weightless Boys multipole screening and weak-pair response.

Moment translation, Cartesian response contractions, and pair kernels are
shared through ``pyscfad.lib._multipole``. Fixed primary-screen reconstruction
uses each singleton Boys target. Weak pairs accumulate compact endpoint
cotangents before replaying each endpoint response once; no native DLNO
module is imported by this implementation.
"""
import gc
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad.lno.tools import fake_mol_by_atom, project_mo
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


def _multipole_orbital_data(mol, screen, atoms, order):
    """Build the one Boys endpoint using translated raw transition moments."""
    if order not in (2, 3, 4):
        raise ValueError("multipole order must be 2, 3, or 4")
    occupied = jnp.asarray(screen.occupied_coeff)
    virtual = jnp.asarray(screen.virtual_coeff)
    if occupied.ndim != 2 or occupied.shape[1] != 1:
        raise ValueError("a Boys weak screen requires one occupied target")
    if virtual.ndim != 2 or virtual.shape[0] != occupied.shape[0]:
        raise ValueError("weak-screen occupied and virtual AO dimensions differ")
    fake_mol = _fake_mol_with_traced_centers(mol, atoms)
    overlap, dipole, second, third = _origin_zero_moments(
        fake_mol, max(1, order - 1)
    )
    center, mu, theta, omega = transition_moments(
        occupied, virtual, overlap, dipole, second, third, order
    )
    excitation = jnp.asarray(screen.virtual_energy) - jnp.asarray(screen.occupied_energy)[0]
    return (center[0], mu[0], excitation,
            None if theta is None else theta[0],
            None if omega is None else omega[0])


def build_weak_multipole_screen(common, static, fragment_index):
    """Replay saved primary supports/ranks with a normalized current target."""
    from .domain import (_fixed_metric_orthonormalize, _fixed_semicanonicalize,
                         _rebuild_selected_pao_space)
    fragment = static.fragments[int(fragment_index)]
    if fragment.weak_virtual is None:
        return None
    ao = fragment.primary_ao_indices
    overlap = common.s1e[np.ix_(ao, ao)]
    fock = common.fock[np.ix_(ao, ao)]
    occupied = project_mo(common.fragment_occupied_data[fragment.fragment_index].iao_coeff,
                          common.s1e[ao], overlap)
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
    if bool(static.strong_mask[left, right]):
        raise ValueError(f"fragment pair ({left}, {right}) is classified strong")
    return left, right


def weak_screen_pair_energy(mf, left, right, static, left_index, right_index):
    """Evaluate one unordered weak pair, with the native -4 prefactor."""
    # Preserve endpoint identity when accepting a reversed pair.
    if int(left_index) > int(right_index):
        left, right = right, left
    left_index, right_index = _validate_weak_pair(static, left_index, right_index)
    if left is None or right is None:
        raise RuntimeError("a fixed weak pair has no multipole screen")
    order = static.thresholds.multipole_order
    lhs = _multipole_orbital_data(mf.mol, left, static.fragments[left_index].primary_atoms, order)
    rhs = _multipole_orbital_data(mf.mol, right, static.fragments[right_index].primary_atoms, order)
    return jnp.real(_multipole_pair_energy(pad_endpoint(lhs), pad_endpoint(rhs), order))


def build_weak_endpoint(mf, common, static, fragment_index):
    """Build a forward-only endpoint without retaining orbital pullbacks."""
    configure_compilation_cache()
    screen = build_weak_multipole_screen(common, static, fragment_index)
    return _weak_endpoint_from_screen(mf, screen, static, fragment_index)


def _weak_endpoint_from_screen(mf, screen, static, fragment_index):
    """Contract AO moments using the supplied, differentiable orbital screen."""
    if screen is None:
        raise RuntimeError(f"a fixed weak endpoint {fragment_index} has no multipole screen")
    return _multipole_orbital_data(
        mf.mol, screen, static.fragments[fragment_index].primary_atoms,
        static.thresholds.multipole_order,
    )


def _prepare_weak_endpoint(mf, common, static, fragment_index):
    """Retain the orbital reverse rule from the actual forward evaluation."""
    return prepare_endpoint(
        partial(build_weak_multipole_screen, static=static, fragment_index=fragment_index),
        partial(_weak_endpoint_from_screen, static=static, fragment_index=fragment_index),
        mf, common)


def _replay_weak_endpoint(mf, linearization, static, fragment_index):
    """Replay AO moments in the saved screen and compose its orbital response."""
    return replay_endpoint(
        partial(_weak_endpoint_from_screen, static=static, fragment_index=fragment_index),
        mf, linearization)


def weak_pair_energy(mf, common, static, left_index, right_index):
    """Evaluate one pair using the same endpoint boundary as the full sum."""
    left_index, right_index = _validate_weak_pair(static, left_index, right_index)
    return jnp.real(_multipole_pair_energy(
        pad_endpoint(build_weak_endpoint(mf, common, static, left_index)),
        pad_endpoint(build_weak_endpoint(mf, common, static, right_index)),
        static.thresholds.multipole_order,
    ))


def weak_pair_indices(static):
    """List each weak fragment pair exactly once in fragment order."""
    return tuple((left, right) for left in range(len(static.fragments))
                 for right in range(left + 1, len(static.fragments))
                 if not bool(static.strong_mask[left, right]))


def _weak_endpoints(mf, common, static, pairs):
    indices = sorted({index for pair in pairs for index in pair})
    return {index: build_weak_endpoint(mf, common, static, index) for index in indices}


def _check_endpoint_replay(rebuilt, original, index):
    if not endpoint_matches_reference(rebuilt, original):
        raise RuntimeError(f"weak AO-moment values changed on replay: endpoint {index}")


def weak_pair_energy_sum(mf, common, static, *, pairs=None):
    """Evaluate all weak edges while constructing each endpoint only once."""
    pairs = weak_pair_indices(static) if pairs is None else tuple(pairs)
    endpoints = {index: pad_endpoint(value) for index, value in
                 _weak_endpoints(mf, common, static, pairs).items()}
    energy = jnp.zeros((), dtype=common.s1e.dtype)
    for left, right in pairs:
        energy += _multipole_pair_energy(
            endpoints[left], endpoints[right], static.thresholds.multipole_order)
    return jnp.real(energy)


def weak_pair_energy_and_grad(mf, common, static, *, pairs=None):
    """Reuse orbital pullbacks and replay one AO-moment tape at a time.

    Each orbital screen and its reverse rule come from one forward VJP.
    Pair contractions share compact transition moments. Their accumulated
    bars pass through AO-moment replay using the exact saved screen, then
    through its retained orbital pullback. No eigensolver is replayed.
    """
    from .adjoint import add_cotangent, zero_term_cotangents
    configure_compilation_cache()
    pairs = weak_pair_indices(static) if pairs is None else tuple(pairs)
    bars = zero_term_cotangents(mf, common)
    energy = jnp.zeros((), dtype=common.s1e.dtype)
    originals, linearizations = {}, {}
    for index in sorted({index for pair in pairs for index in pair}):
        originals[index], linearizations[index] = _prepare_weak_endpoint(
            mf, common, static, index)
    endpoints = {index: pad_endpoint(value) for index, value in originals.items()}
    endpoint_bars = {index: jax.tree_util.tree_map(jnp.zeros_like, value)
                     for index, value in endpoints.items()}
    for left, right in pairs:
        value, (left_bar, right_bar) = pair_energy_and_grad(
            endpoints[left], endpoints[right], static.thresholds.multipole_order)
        endpoint_bars[left] = jax.tree_util.tree_map(add_cotangent, endpoint_bars[left], left_bar)
        endpoint_bars[right] = jax.tree_util.tree_map(add_cotangent, endpoint_bars[right], right_bar)
        energy += value
        del left_bar, right_bar
    for index in tuple(endpoint_bars):
        endpoint_bar = endpoint_bars.pop(index)
        del endpoints[index]
        original = originals.pop(index)
        linearization = linearizations.pop(index)
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


def _exact_boys_pair_scores(mf, common, static):
    """Concrete exact classification through two labelled occupied contractions.

    Project amplitudes only after canonical denominators have been applied.
    This is the native exact screening convention (four times directed OS,
    then combine orientations), without constructing occupied weight matrices.
    One virtual pair is held at a time.
    """
    from pyscfad.mp import dfmp2
    pt = dfmp2.MP2(mf, frozen=static.frozen)
    eris = pt.ao2mo()
    nocc, nvir = int(pt.nocc), int(pt.nmo - pt.nocc)
    lov = np.asarray(pt.loop_ao2mo(eris.mo_coeff, nocc, with_t2=False)).reshape(-1, nocc, nvir)
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
    directed *= 4
    physical = directed + directed.T
    np.fill_diagonal(physical, np.diag(directed))
    return physical


def screen_pairs(mf, common, static, model='multipole', force_full_domains=False):
    """Concrete Boys pair classification on provisional primary selections."""
    model = str(model).lower().replace('_', '-')
    if model not in ('multipole', 'multipole-os', 'os-multipole', 'exact', 'all', 'all-strong'):
        raise ValueError(f"unsupported Boys pair energy model {model!r}")
    nfrag = len(static.fragments)
    if force_full_domains or static.thresholds.pair_energy <= 0 or model in ('all', 'all-strong'):
        return (np.zeros((nfrag, nfrag)), np.zeros((nfrag, nfrag)),
                np.ones((nfrag, nfrag), dtype=bool), np.ones((nfrag, nfrag), dtype=bool))
    screens = [build_weak_multipole_screen(common, static, index) for index in range(nfrag)]
    order = static.thresholds.multipole_order
    endpoints = [None if screen is None else pad_endpoint(_multipole_orbital_data(
        mf.mol, screen, static.fragments[index].primary_atoms, order))
        for index, screen in enumerate(screens)]
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
    pair = _exact_boys_pair_scores(mf, common, static) if model == 'exact' else weak
    strong = (np.abs(pair) > static.thresholds.pair_energy) | forced
    strong = strong | strong.T
    np.fill_diagonal(strong, True)
    return pair, weak, forced, strong
