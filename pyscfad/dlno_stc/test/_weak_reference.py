"""Small JAX mathematical oracles for the explicit weak-pair VJP tests.

Production weak-pair APIs never import this module or dispatch to JAX AD.
"""
import jax
import jax.numpy as jnp
import numpy as np
from pyscfad import scipy
from pyscfad.lno.tools import project_mo
from pyscfad.lib._multipole import transition_moments, pair_energy
from pyscfad.dlno_stc import domain, _pairs
from pyscfad.dlno_stc._pairs import WeakScreen, _WeakScreenProducts
from pyscfad.dlno_stc._primary_response import _eigh, _metric_space, _keep
from pyscfad.dlno_stc._selection import _PAO_ORTH_THRESHOLD, _orthogonalize
from pyscfad.dlno_stc.domain import _cholesky_metric_orthonormalize

def _weak_screen_products(common):
    """Compute global PAO and independent target products once per invocation."""
    from pyscfad.dlno_stc.domain import _pao_products
    targets = tuple(target.iao_coeff for target in common.fragment_occupied_data)
    if targets:
        combined = common.s1e @ jnp.concatenate(targets, axis=1)
        offsets = np.cumsum([target.shape[1] for target in targets[:-1]])
        occupied = tuple(jnp.split(combined, offsets, axis=1))
    else:
        occupied = ()
    return _WeakScreenProducts(_pao_products(common.pao_coeff, common.s1e), occupied)


def build_weak_multipole_screen(common, static, fragment_index, *, _pao_preprojection=None,
                                _occupied_projection_rhs=None):
    """Replay saved primary supports/ranks with a normalized current target."""
    from pyscfad.dlno_stc.domain import (_fixed_metric_orthonormalize,
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
    span = _fixed_metric_orthonormalize(occupied, overlap, fragment.weak_occ_span_metric_keep)
    virtual = _rebuild_selected_pao_space(
        common, fragment.weak_virtual, ao, span,
        boys_metric_threshold=static.thresholds.metric_rank,
        preprojection=_pao_preprojection,
    )
    return _finish_weak_screen(occupied, virtual, fock, fragment_index)


def _finish_weak_screen(occupied, virtual, fock, fragment_index):
    """Semicanonicalize an already selected or replayed primary space."""
    from pyscfad.dlno_stc.domain import _fixed_semicanonicalize
    occupied = jnp.asarray(occupied)
    occupied_energy = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), fock, occupied))
    virtual_energy, virtual = _fixed_semicanonicalize(virtual, fock)
    gaps = np.asarray(jax.device_get(jax.lax.stop_gradient(virtual_energy[:, None] - occupied_energy)))
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise RuntimeError(f"Boys target {fragment_index} has invalid screening denominators")
    return WeakScreen(occupied_energy, occupied, virtual_energy, virtual)


def _select_preprojection(common, products, *, columns, support, ao, thresholds,
                          completeness_threshold=None):
    gram_all, overlap_pao = products
    gram = gram_all[np.ix_(columns, columns)]
    vectors, canonical = _eigh(gram, _PAO_ORTH_THRESHOLD, deg_thresh=_PAO_ORTH_THRESHOLD)
    empty = np.empty(0, dtype=np.int32)
    if not canonical.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, empty, empty)
    transform = _cholesky_metric_orthonormalize(vectors[:, canonical], gram)
    tmp = overlap_pao[np.ix_(support, columns)] @ transform
    projected = jnp.linalg.solve(common.s1e[np.ix_(support, support)], tmp)
    vectors, overlap_keep = _eigh(tmp.T.conj() @ projected, thresholds.domain_pao,
                                 deg_thresh=_PAO_ORTH_THRESHOLD)
    if not overlap_keep.size:
        return jnp.zeros((len(ao), 0), dtype=gram.dtype), (canonical, overlap_keep, empty)
    rotation = vectors[:, overlap_keep]
    if np.array_equal(support, ao):
        candidate, rhs = projected @ rotation, tmp @ rotation
    else:
        rhs = overlap_pao[np.ix_(ao, columns)] @ (transform @ rotation)
        candidate = jnp.linalg.solve(common.s1e[np.ix_(ao, ao)], rhs)
    completeness = jnp.real(jnp.sum(candidate.conj() * rhs, axis=0))
    complete = _keep(completeness, thresholds.bp_pao if completeness_threshold is None
                     else completeness_threshold)
    return candidate[:, complete], (canonical, overlap_keep, complete)


def _select_target(common, candidate, occupied_rhs, *, ao, thresholds, index):
    overlap, fock = common.s1e[np.ix_(ao, ao)], common.fock[np.ix_(ao, ao)]
    occupied = jnp.linalg.solve(overlap, occupied_rhs[ao])
    norm2 = jnp.real(jnp.einsum('ui,uv,vi->i', occupied.conj(), overlap, occupied))
    norm_keep = _keep(norm2, thresholds.metric_rank)
    empty = np.empty(0, dtype=np.int32)
    if not norm_keep.size:
        return None, (norm_keep, empty, empty)
    occupied = occupied[:, norm_keep] / jnp.sqrt(norm2[norm_keep])[None, :]
    span, span_keep = _metric_space(occupied, overlap, thresholds.metric_rank)
    virtual, metric_keep = _metric_space(
        _orthogonalize(span, candidate, overlap), overlap, thresholds.metric_rank)
    if not virtual.shape[1]:
        return None, (norm_keep, span_keep, metric_keep)
    return _finish_weak_screen(occupied, virtual, fock, index), (norm_keep, span_keep, metric_keep)


def endpoint(mf, common, static, index):
    screen = build_weak_multipole_screen(common, static, index)
    mol = _pairs._fake_mol_with_traced_centers(mf.mol, static.fragments[index].primary_atoms)
    raw = _pairs._origin_zero_moments(mol, max(1, static.thresholds.multipole_order-1))
    center, mu, theta, omega = transition_moments(screen.occupied_coeff,
        screen.virtual_coeff, *raw, static.thresholds.multipole_order)
    return (center[0], mu[0], screen.virtual_energy-screen.occupied_energy[0],
            None if theta is None else theta[0], None if omega is None else omega[0])


def energy(mf, common, static, pairs=None):
    return jnp.array(0.) + sum(_pairs.weak_pair_weight(static, i, j)*pair_energy(
        endpoint(mf, common, static, i), endpoint(mf, common, static, j),
        static.thresholds.multipole_order) for i,j in (_pairs.weak_pair_indices(static) if pairs is None else pairs))
