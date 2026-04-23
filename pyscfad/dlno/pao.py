import numpy as np
import jax.numpy as jnp
from pyscfad import scipy
from . import util


def _canonical_orth_jax(s, thr=1e-6):
    e, v = scipy.linalg.eigh(jnp.asarray(s), deg_thresh=thr)
    idx = e > thr
    return v[:, idx] / jnp.sqrt(e[idx])[None, :]

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
    """
    if p_domain is None:
        p_domain = bp_domain
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')

    pao_pd = pao_by_atom(mol, paos, p_domain, ao2pao_map)

    x = _canonical_orth_jax(pao_pd.T.conj() @ s1e @ pao_pd, thr=orth_thr)
    pao_pd_orth = pao_pd @ x

    ao_idx_bp = util.ao_index_by_atom(mol, bp_domain)
    s21 = s1e[ao_idx_bp]
    s22 = s1e[np.ix_(ao_idx_bp, ao_idx_bp)]
    s22_inv = jnp.linalg.inv(jnp.asarray(s22))

    tmp = s21 @ pao_pd_orth
    ovlp = tmp.T.conj() @ s22_inv @ tmp
    w, v = scipy.linalg.eigh(jnp.asarray(ovlp), deg_thresh=max(orth_thr, 1e-6))
    return pao_pd_orth @ v[:, w>ovlp_thr]
