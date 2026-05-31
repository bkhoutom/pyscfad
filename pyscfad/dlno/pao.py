import numpy as np
import scipy.linalg as _sla_numpy
import jax
import jax.numpy as jnp
from pyscfad import scipy
from . import util


def _is_traced(*xs):
    for x in xs:
        if isinstance(x, jax.core.Tracer):
            return True
    return False


def _canonical_orth(s, thr=1e-6):
    """Build the canonical-orthonormalization matrix from the overlap ``s``.

    When ``s`` is a concrete (non-tracer) array, take a plain numpy path: no
    JAX tracing, no XLA compilation, allocator returns large mmaps on free.
    When ``s`` is a tracer (called from inside ``jax.value_and_grad``), use
    the JAX path so the gradient propagates.
    """
    if _is_traced(s):
        e, v = scipy.linalg.eigh(jnp.asarray(s), deg_thresh=thr)
        idx = e > thr
        return v[:, idx] / jnp.sqrt(e[idx])[None, :]
    s_np = np.asarray(s)
    e, v = _sla_numpy.eigh(s_np)
    idx = e > thr
    return v[:, idx] / np.sqrt(e[idx])[None, :]


# Back-compat alias for any external callers.
_canonical_orth_jax = _canonical_orth

def pao(mol, mos, s1e=None, norm_thr=1e-4):
    """Compute PAOs.

    Parameters
    ----------
    norm_thr : float, default=1e-4
        PAOs with norms smaller than ``norm_thr`` are discarded.

    Returns
    -------
    paos : Normalized PAOs.
    inv_idx : Indices mapping AOs to PAOs.

    Notes
    -----
    ``mos`` need to be orthonormal.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    if _is_traced(mos, s1e):
        mos = jnp.asarray(mos)
        if mos.ndim == 1:
            mos = mos.reshape(-1,1)
        assert mos.ndim == 2
        nao, nmo = mos.shape
        s1e = jnp.asarray(s1e)
        paos = jnp.eye(nao) - mos @ (mos.T.conj() @ s1e)
        norm = jnp.sqrt(util.einsum('ui,uv,vi->i', paos.conj(), s1e, paos))
        pao_idx = jnp.where(norm > norm_thr)[0]
        inv_idx = np.full(nao, -1, dtype=np.int32)
        inv_idx[np.asarray(pao_idx)] = np.arange(len(pao_idx))
        paos = paos[:,pao_idx] / norm[pao_idx]
        return paos, inv_idx

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1, 1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    s1e = np.asarray(s1e)
    paos = np.eye(nao) - mos @ (mos.T.conj() @ s1e)
    norm = np.sqrt(np.einsum('ui,uv,vi->i', paos.conj(), s1e, paos))
    pao_idx = np.where(norm > norm_thr)[0]
    inv_idx = np.full(nao, -1, dtype=np.int32)
    inv_idx[pao_idx] = np.arange(len(pao_idx))
    paos = paos[:, pao_idx] / norm[pao_idx]
    return paos, inv_idx


def pao_by_atom(mol, paos, atmlst, ao2pao_map=None):
    if ao2pao_map is None:
        ao2pao_map = np.arange(mol.nao)

    aoslices = mol.aoslice_by_atom()[:,2:]
    pao_idx = np.empty((0,), dtype=np.int32)
    for i0, i1 in aoslices[atmlst].reshape(-1,2):
        idx = ao2pao_map[i0:i1]
        pao_idx = np.append(pao_idx, idx[idx >= 0])
    return paos[:,pao_idx].reshape(-1, pao_idx.size)


def pao_overlap_with_domain(
        mol, paos, bp_domain, p_domain=None,
        ao2pao_map=None, s1e=None, ovlp_thr=1e-4, orth_thr=1e-6
    ):
    """PAOs in the larger domain that overlap with the smaller domain.

    Dispatches on whether the inputs are JAX tracers: under
    ``jax.value_and_grad`` we keep the JAX path so the gradient propagates;
    outside (eager builds) we use plain numpy/scipy to avoid XLA compile
    cache growth from per-LMO/per-fragment shape variation.
    """
    if p_domain is None:
        p_domain = bp_domain
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    pao_pd = pao_by_atom(mol, paos, p_domain, ao2pao_map)
    ao_idx_bp = util.ao_index_by_atom(mol, bp_domain)

    if _is_traced(paos, s1e, pao_pd):
        x = _canonical_orth(pao_pd.T.conj() @ s1e @ pao_pd, thr=orth_thr)
        pao_pd_orth = pao_pd @ x
        s21 = s1e[ao_idx_bp]
        s22 = s1e[np.ix_(ao_idx_bp, ao_idx_bp)]
        tmp = s21 @ pao_pd_orth
        ovlp = tmp.T.conj() @ jnp.linalg.solve(jnp.asarray(s22), tmp)
        w, v = scipy.linalg.eigh(jnp.asarray(ovlp), deg_thresh=max(orth_thr, 1e-6))
        return pao_pd_orth @ v[:, w > ovlp_thr]

    pao_pd_np = np.asarray(pao_pd)
    s1e_np = np.asarray(s1e)
    x = _canonical_orth(pao_pd_np.T.conj() @ s1e_np @ pao_pd_np, thr=orth_thr)
    pao_pd_orth = pao_pd_np @ x
    s21 = s1e_np[ao_idx_bp]
    s22 = s1e_np[np.ix_(ao_idx_bp, ao_idx_bp)]
    tmp = s21 @ pao_pd_orth
    ovlp = tmp.T.conj() @ np.linalg.solve(s22, tmp)
    w, v = _sla_numpy.eigh(ovlp)
    return pao_pd_orth @ v[:, w > ovlp_thr]
