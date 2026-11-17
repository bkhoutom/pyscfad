"""Explicit real host linear algebra for weak primary spaces and their VJPs.

Only fixed selection labels and small matrix factorizations are retained.
NumPy/SciPy perform both passes; JAX is used only for pytree bookkeeping.
"""
import jax
import numpy as np
from scipy import linalg

from ._selection import _PAO_ORTH_THRESHOLD
from pyscfad.lno.lno_base import SEMICANONICAL_DEG_THRESH


def require_supported(*trees):
    """Weak-pair APIs accept concrete real float64 inputs and explicit VJPs."""
    for tree in trees:
        for value in jax.tree.leaves(tree):
            if isinstance(value, jax.core.Tracer):
                raise TypeError('Weak pairs use an explicit NumPy/SciPy pullback; '
                                'call weak_pair_energy_and_grad instead of JAX AD')
            dtype = getattr(value, 'dtype', None)
            if dtype is not None and np.dtype(dtype).kind in 'fc' and dtype != np.float64:
                raise TypeError('Weak pairs require real float64 inputs')


def zeros(tree):
    return jax.tree.map(np.zeros_like, tree)


def sym(x):
    return (x + x.T) * .5


def _keep(values, threshold):
    if not np.isfinite(values).all():
        raise RuntimeError('nonfinite primary-space selection spectrum')
    return np.flatnonzero(values > threshold).astype(np.int32)


def eigh(matrix, deg_thresh=1e-9):
    """Match the existing symmetric generalized-eigh response at B=I."""
    if not len(matrix):
        return (np.zeros(0, dtype=matrix.dtype), np.zeros_like(matrix)), lambda eb, ub: np.zeros_like(matrix)
    values, vectors = linalg.eigh(sym(matrix), np.eye(len(matrix)), driver='gvd',
                                  check_finite=False)
    def pullback(values_bar, vectors_bar):
        gaps = values[None, :] - values[:, None]
        factor = np.zeros_like(gaps)
        np.divide(1., gaps, out=factor, where=np.abs(gaps) > deg_thresh)
        inner = factor * (vectors.T @ vectors_bar)
        inner[np.diag_indices_from(inner)] += values_bar
        return sym(vectors @ inner @ vectors.T)
    return (values, vectors), pullback


def normalize(coeff, overlap):
    """C (chol(sym(C.T S C)))**-T, with its complete metric response."""
    metric = sym(coeff.T @ overlap @ coeff)
    lower = linalg.cholesky(metric, lower=True, check_finite=False)
    value = linalg.solve_triangular(lower, coeff.T, lower=True, check_finite=False).T
    def pullback(bar):
        cbar = linalg.solve_triangular(lower.T, bar.T, lower=False, check_finite=False).T
        lbar = -cbar.T @ value
        phi = np.tril(lower.T @ lbar)
        phi[np.diag_indices_from(phi)] *= .5
        gbar = linalg.solve_triangular(lower.T, sym(phi), lower=False, check_finite=False)
        gbar = linalg.solve_triangular(lower.T, gbar.T, lower=False, check_finite=False).T
        cbar += overlap @ coeff @ gbar.T + overlap.T @ coeff @ gbar
        return cbar, coeff @ gbar @ coeff.T
    return value, pullback


def metric_space(coeff, overlap, threshold, keep=None):
    """Select ranks concretely, then retain only the selected-span reverse."""
    if coeff.shape[1] == 0:
        return coeff, lambda bar: (np.zeros_like(coeff), np.zeros_like(overlap)), np.empty(0, int)
    if keep is None or not np.array_equal(keep, np.arange(coeff.shape[1])):
        (values, vectors), eig_pb = eigh(coeff.T @ overlap @ coeff)
        labels = _keep(values, threshold) if keep is None else np.asarray(keep)
    else:
        labels = np.asarray(keep)
    if not len(labels):
        return coeff[:, :0], lambda bar: (np.zeros_like(coeff), np.zeros_like(overlap)), labels
    if np.array_equal(labels, np.arange(coeff.shape[1])):
        value, pb = normalize(coeff, overlap)
        return value, pb, labels
    rotation = vectors[:, labels]
    value, norm_pb = normalize(coeff @ rotation, overlap)
    def pullback(bar):
        retained_bar, sbar = norm_pb(bar)
        vbar = np.zeros_like(vectors)
        vbar[:, labels] = coeff.T @ retained_bar
        gbar = eig_pb(np.zeros_like(values), vbar)
        cbar = retained_bar @ rotation.T + overlap @ coeff @ gbar.T + overlap.T @ coeff @ gbar
        return cbar, sbar + coeff @ gbar @ coeff.T
    return value, pullback, labels


def project_out(occupied, candidate, overlap):
    projection = occupied.T @ overlap @ candidate
    result = candidate - occupied @ projection
    def pullback(bar):
        pbar = -occupied.T @ bar
        return (-bar @ projection.T + overlap @ candidate @ pbar.T,
                bar + overlap.T @ occupied @ pbar,
                occupied @ pbar @ candidate.T)
    return result, pullback


def products(common, cached=None):
    """Shared S@PAO, PAO Gram and S@target products, with one host reverse."""
    require_supported(common)
    from ._pairs import _WeakScreenProducts, _WeakProductsPullback
    s, p = np.asarray(common.s1e), np.asarray(common.pao_coeff)
    targets = tuple(np.asarray(x.iao_coeff) for x in common.fragment_occupied_data)
    combined = np.concatenate(targets, axis=1) if targets else None
    offsets = np.cumsum([x.shape[1] for x in targets[:-1]])
    if cached is None:
        sp = s @ p
        occupied = tuple(np.split(s @ combined, offsets, axis=1)) if targets else ()
        value = _WeakScreenProducts((p.T @ sp, sp), occupied)
    else:
        value = jax.tree.map(np.asarray, cached)
        sp = value.pao[1]
    def pullback(bar):
        (gbar, spbar), target_bars = bar
        zbar = p @ gbar + spbar
        pbar = sp @ gbar.T + s.T @ zbar
        sbar = zbar @ p.T
        target_result = ()
        if targets:
            combined_bar = np.concatenate(target_bars, axis=1)
            sbar += combined_bar @ combined.T
            target_result = tuple(np.split(s.T @ combined_bar, offsets, axis=1))
        common_bar = zeros(common)
        target_tree = tuple(x._replace(iao_coeff=b) for x, b in
                            zip(common_bar.fragment_occupied_data, target_result))
        return (common_bar._replace(s1e=sbar, pao_coeff=pbar, fragment_occupied_data=target_tree),)
    shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), value)
    return value, _WeakProductsPullback(pullback, shapes)


def preprojection(common, product_values, *, columns, support, ao,
                  thresholds=None, selection=None, completeness_threshold=None):
    """Build or replay fixed PAO labels using shared products and local blocks."""
    require_supported(common)
    s, p = np.asarray(common.s1e), np.asarray(common.pao_coeff)
    gram_all, sp = map(np.asarray, product_values)
    columns, support, ao = map(np.asarray, (columns, support, ao))
    gram = sym(gram_all[np.ix_(columns, columns)])
    (e, u), gram_pb = eigh(gram, _PAO_ORTH_THRESHOLD)
    canonical = _keep(e, _PAO_ORTH_THRESHOLD) if selection is None else selection.canonical_keep
    empty = np.empty(0, dtype=np.int32)
    def empty_pb(bar):
        return zeros(common), (np.zeros_like(gram_all), np.zeros_like(sp))
    if not len(canonical):
        return np.zeros((len(ao), 0)), empty_pb, (canonical, empty, empty)
    transform, norm_pb = normalize(u[:, canonical], gram)
    a = sp[np.ix_(support, columns)]
    rhs = a @ transform
    ss = s[np.ix_(support, support)]
    lu = linalg.lu_factor(ss, check_finite=False)
    projected = linalg.lu_solve(lu, rhs, check_finite=False)
    (e2, rotation_all), overlap_pb = eigh(rhs.T @ projected, _PAO_ORTH_THRESHOLD)
    overlap_keep = _keep(e2, thresholds.domain_pao) if selection is None else selection.overlap_keep
    if not len(overlap_keep):
        return np.zeros((len(ao), 0)), empty_pb, (canonical, overlap_keep, empty)
    rotation = rotation_all[:, overlap_keep]
    same = np.array_equal(support, ao)
    full = selection is not None and np.array_equal(ao, np.arange(len(s)))
    rep = sp[np.ix_(ao, columns)]
    rep_lu = None
    if full:
        candidate = p[:, columns] @ (transform @ rotation)
    elif same:
        candidate = projected @ rotation
        candidate_rhs = rhs @ rotation
    else:
        candidate_rhs = rep @ (transform @ rotation)
        rep_lu = linalg.lu_factor(s[np.ix_(ao, ao)], check_finite=False)
        candidate = linalg.lu_solve(rep_lu, candidate_rhs, check_finite=False)
    if selection is None:
        threshold = thresholds.bp_pao if completeness_threshold is None else completeness_threshold
        complete = _keep(np.sum(candidate * candidate_rhs, axis=0), threshold)
    else:
        complete = selection.completeness_keep
    value = candidate[:, complete]
    def pullback(bar):
        candidate_bar = np.zeros_like(candidate)
        candidate_bar[:, complete] = bar
        common_bar = zeros(common)
        gbar_all, spbar = np.zeros_like(gram_all), np.zeros_like(sp)
        sbar, pbar = common_bar.s1e, common_bar.pao_coeff
        projected_bar = np.zeros_like(projected)
        if full:
            pbar[:, columns] += candidate_bar @ (transform @ rotation).T
            tbar = p[:, columns].T @ candidate_bar @ rotation.T
            rbar = transform.T @ p[:, columns].T @ candidate_bar
        elif same:
            projected_bar += candidate_bar @ rotation.T
            rbar = projected.T @ candidate_bar
            tbar = np.zeros_like(transform)
        else:
            rb = linalg.lu_solve(rep_lu, candidate_bar, trans=1, check_finite=False)
            sbar[np.ix_(ao, ao)] -= rb @ candidate.T
            spbar[np.ix_(ao, columns)] += rb @ (transform @ rotation).T
            tbar = rep.T @ rb @ rotation.T
            rbar = transform.T @ rep.T @ rb
        rbar_all = np.zeros_like(rotation_all)
        rbar_all[:, overlap_keep] = rbar
        hbar = overlap_pb(np.zeros_like(e2), rbar_all)
        rhs_bar = projected @ hbar.T
        projected_bar += rhs @ hbar
        pb = linalg.lu_solve(lu, projected_bar, trans=1, check_finite=False)
        rhs_bar += pb
        sbar[np.ix_(support, support)] -= pb @ projected.T
        spbar[np.ix_(support, columns)] += rhs_bar @ transform.T
        tbar += a.T @ rhs_bar
        uk_bar, gbar = norm_pb(tbar)
        ubar = np.zeros_like(u)
        ubar[:, canonical] = uk_bar
        gbar += gram_pb(np.zeros_like(e), ubar)
        gbar_all[np.ix_(columns, columns)] = sym(gbar)
        return common_bar, (gbar_all, spbar)
    return value, pullback, (canonical, overlap_keep, complete)


def target(common, candidate, occupied_rhs, *, ao, thresholds, index,
           norm_keep=None, span_keep=None, metric_keep=None):
    """Projected target, PAO complement and full semicanonical screen VJP."""
    require_supported(common)
    from ._pairs import WeakScreen
    s = np.asarray(common.s1e)[np.ix_(ao, ao)]
    f = np.asarray(common.fock)[np.ix_(ao, ao)]
    candidate, occupied_rhs = map(np.asarray, (candidate, occupied_rhs))
    lu = linalg.lu_factor(s, check_finite=False)
    projected = linalg.lu_solve(lu, occupied_rhs[ao], check_finite=False)
    norm2 = np.sum(projected * (s @ projected), axis=0)
    keep = _keep(norm2, thresholds.metric_rank)
    if norm_keep is not None and (not np.array_equal(keep, norm_keep) or len(keep) != 1):
        raise RuntimeError('Boys screening target rank changed; rebuild domains')
    if not len(keep):
        return None, None, (keep, np.empty(0, int), np.empty(0, int))
    occupied = projected[:, keep] / np.sqrt(norm2[keep])
    span, span_pb, span_labels = metric_space(occupied, s, thresholds.metric_rank, span_keep)
    complement, complement_pb = project_out(span, candidate, s)
    if metric_keep is not None:
        current = _keep(linalg.eigvalsh(sym(complement.T @ s @ complement)), thresholds.metric_rank)
        if not np.array_equal(current, metric_keep):
            raise RuntimeError('Boys virtual ED metric rank changed; rebuild domains')
    virtual, metric_pb, metric_labels = metric_space(complement, s, thresholds.metric_rank, metric_keep)
    if not virtual.shape[1] and metric_keep is None:
        return None, None, (keep, span_labels, metric_labels)
    eo = np.sum(occupied * (f @ occupied), axis=0)
    if virtual.shape[1] == 1:
        ev, rotation = np.diag(virtual.T @ f @ virtual), np.eye(1)
        def fock_pb(ebar, ubar):
            return np.diag(ebar)
    else:
        (ev, rotation), fock_pb = eigh(virtual.T @ f @ virtual, SEMICANONICAL_DEG_THRESH)
    value = WeakScreen(eo, occupied, ev, virtual @ rotation)
    gaps = ev[:, None] - eo
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise RuntimeError(f'Boys target {index} has invalid screening denominators')
    def pullback(bar):
        eo_bar, co_bar, ev_bar, cv_bar = map(np.asarray, bar)
        hbar = fock_pb(ev_bar, virtual.T @ cv_bar)
        vbar = cv_bar @ rotation.T + f @ virtual @ hbar.T + f.T @ virtual @ hbar
        fbar = virtual @ hbar @ virtual.T + (occupied * eo_bar) @ occupied.T
        cbar = co_bar + ((f + f.T) @ occupied) * eo_bar
        complement_bar, sbar = metric_pb(vbar)
        span_bar, candidate_bar, ds = complement_pb(complement_bar)
        sbar += ds
        dc, ds = span_pb(span_bar)
        cbar += dc; sbar += ds
        original = projected[:, keep]
        nbar = -.5 * np.sum(cbar * original, axis=0) / norm2[keep]**1.5
        projected_bar = np.zeros_like(projected)
        projected_bar[:, keep] = cbar / np.sqrt(norm2[keep]) + ((s+s.T) @ original) * nbar
        sbar += (original * nbar) @ original.T
        rhs_bar = np.zeros_like(occupied_rhs)
        rhs_bar[ao] = linalg.lu_solve(lu, projected_bar, trans=1, check_finite=False)
        sbar -= rhs_bar[ao] @ projected.T
        common_bar = zeros(common)
        common_bar.s1e[np.ix_(ao, ao)] += sbar
        common_bar.fock[np.ix_(ao, ao)] += fbar
        return common_bar, candidate_bar, rhs_bar
    return value, pullback, (keep, span_labels, metric_labels)


def fixed_preprojection(common, product_values, fragment):
    selection = fragment.weak_virtual
    return preprojection(common, product_values, columns=selection.parent_columns,
        support=selection.support_ao_indices, ao=fragment.primary_ao_indices, selection=selection)[:2]


def fixed_target(common, candidate, rhs, fragment, thresholds, index):
    return target(common, candidate, rhs, ao=fragment.primary_ao_indices,
        thresholds=thresholds, index=index, norm_keep=fragment.weak_occ_norm_keep,
        span_keep=fragment.weak_occ_span_metric_keep,
        metric_keep=fragment.weak_virtual.metric_keep)[:2]


def fixed_screen(common, static, index, *, candidate=None, rhs=None):
    """Standalone endpoint response; only selected PAO parents are multiplied."""
    require_supported(common)
    from dataclasses import replace
    fragment = static.fragments[index]
    s = np.asarray(common.s1e)
    target_index = fragment.fragment_index
    occupied = np.asarray(common.fragment_occupied_data[target_index].iao_coeff)
    if rhs is None:
        rhs = s @ occupied
    if candidate is not None:
        return fixed_target(common, np.asarray(candidate), np.asarray(rhs), fragment, static.thresholds, index)
    columns = fragment.weak_virtual.parent_columns
    p = np.asarray(common.pao_coeff)[:, columns]
    sp = s @ p
    restricted = common._replace(pao_coeff=p)
    selection = replace(fragment.weak_virtual, parent_columns=np.arange(len(columns)))
    candidate, pre_pb, _ = preprojection(restricted, (p.T @ sp, sp),
        columns=selection.parent_columns, support=selection.support_ao_indices,
        ao=fragment.primary_ao_indices, selection=selection)
    value, target_pb = fixed_target(common, candidate, np.asarray(rhs), fragment, static.thresholds, index)
    def pullback(bar):
        result, candidate_bar, rhs_bar = target_pb(bar)
        pre_bar, (gram_bar, sp_bar) = pre_pb(candidate_bar)
        zbar = p @ gram_bar + sp_bar
        result.s1e[:] += pre_bar.s1e + zbar @ p.T + rhs_bar @ occupied.T
        result.pao_coeff[:, columns] += pre_bar.pao_coeff + sp @ gram_bar.T + s.T @ zbar
        result.fragment_occupied_data[target_index].iao_coeff[:] += s.T @ rhs_bar
        return (result,)
    return value, pullback
