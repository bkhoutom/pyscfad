from functools import reduce
import numpy as np
import jax
import jax.numpy as jnp
from pyscf.gto.mole import inter_distance
from . import util

def get_bp_domain(mol, mos, s1e=None, bp_thr=0.999,
                  q_thr=None, atmlst=None):
    """BP domains based on partial Mulliken charges.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5*(1-bp_thr))
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mos = np.asarray(mos)
    if mos.ndim == 1:
        mos = mos.reshape(-1,1)
    assert mos.ndim == 2
    nao, nmo = mos.shape
    # TODO project MOs onto smaller basis
    assert nao == mol.nao

    rr = atom_distance(mol, atmlst)
    aoslices = mol.aoslice_by_atom()[:,2:]
    bp_atmlst = []

    s1e = np.asarray(s1e)
    # Gross orbital populations for all orbitals at once. This matches the
    # per-orbital expression sum_v C_ui S_uv C_vi used below, but avoids
    # rebuilding an AO x AO outer product for every orbital.
    gop = mos * (s1e @ mos)
    q_by_atom = np.abs(np.asarray([
        np.sum(gop[slice(*aoslices[a])], axis=0) for a in atmlst
    ]))
    sorted_atom_idx = np.argsort(rr, axis=1)

    for i in range(nmo):
        orbi = mos[:, i]
        q = q_by_atom[:, i]

        _atms = atmlst[q > q_thr]
        av = _compute_av_numpy(mol, orbi, s1e, _atms)

        if av < bp_thr:
            center_id = int(np.argsort(-q)[0])
            _sorted_atm_idx = sorted_atom_idx[center_id]

            for iatm in _sorted_atm_idx[1:]:
                a = atmlst[iatm]
                if a not in _atms:
                    _atms = np.append(_atms, a)
                    av = _compute_av_numpy(mol, orbi, s1e, _atms)
                    if av >= bp_thr:
                        break
        bp_atmlst.append(_atms)

    return util.list_to_array(bp_atmlst)


def _fragment_trace_completeness_numpy(mol, occupied_block, s1e=None,
                                       atmlst=None):
    """Fraction of a fragment occupied block recovered on an AO domain.

    For an AO-coefficient block ``X``, the recovered population is

    ``Tr[X^H S[:,D] S[D,D]^-1 S[D,:] X] / Tr[X^H S X]``.

    Unlike applying the scalar BP criterion column by column, this quantity is
    invariant to a unitary rotation among the columns of ``X``.  The columns do
    not need to be normalized or mutually orthogonal; consequently weak
    occupied components of an IAO fragment retain their physical weights.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    occupied_block = np.asarray(occupied_block)
    if occupied_block.ndim == 1:
        occupied_block = occupied_block.reshape(-1, 1)
    if occupied_block.ndim != 2 or occupied_block.shape[0] != mol.nao:
        raise ValueError('occupied_block must have shape (mol.nao, nocc_block).')
    if occupied_block.shape[1] == 0:
        raise ValueError('occupied_block must contain at least one column.')

    s1e = np.asarray(s1e)
    sx = s1e @ occupied_block
    total = np.real(np.sum(occupied_block.conj() * sx))
    scale = max(1.0, float(np.linalg.norm(occupied_block))**2)
    if not np.isfinite(total) or total <= np.finfo(float).eps * scale:
        raise ValueError('occupied_block has a vanishing or invalid S norm.')

    atmlst = np.asarray(atmlst, dtype=np.int32).ravel()
    if atmlst.size == 0:
        return 0.0
    ao_idx = util.ao_index_by_atom(mol, atmlst)
    v = sx[ao_idx]
    recovered_coeff = np.linalg.solve(s1e[np.ix_(ao_idx, ao_idx)], v)
    recovered = np.real(np.sum(v.conj() * recovered_coeff))
    return float(recovered / total)


def get_fragment_bp_domain(mol, occupied_blocks, s1e=None, bp_thr=0.999,
                           q_thr=None, atmlsts=None):
    """BP atom domains for occupied components of IAO fragments.

    Parameters
    ----------
    occupied_blocks : sequence of arrays
        AO-coefficient blocks ``X_F = P_occ A_F``, one for each fragment.
        The blocks need not be normalized or mutually orthogonal.
    atmlsts : sequence of atom-index arrays, optional
        Fixed seed atoms for each fragment.  When omitted, atoms whose
        rotation-invariant aggregate Mulliken population exceeds ``q_thr`` are
        used as seeds.

    Notes
    -----
    Starting from the seed set, the domain is enlarged one atom at a time.  The
    next atom minimizes its distance to the current domain; aggregate fragment
    population and atom index provide deterministic tie breaking.  Selection
    stops when the trace-recovered fragment population reaches ``bp_thr``.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if q_thr is None:
        q_thr = min(0.05, 5 * (1 - bp_thr))
    if not 0 <= bp_thr <= 1:
        raise ValueError('bp_thr must lie between zero and one.')

    # A single dense rank-2 block is a useful shorthand for one fragment.  A
    # rank-3 array and an object/list sequence continue to mean many blocks.
    if isinstance(occupied_blocks, np.ndarray) and occupied_blocks.ndim == 2:
        occupied_blocks = [occupied_blocks]
    else:
        occupied_blocks = list(occupied_blocks)

    nfrag = len(occupied_blocks)
    if atmlsts is None:
        atmlsts = [None] * nfrag
    else:
        atmlsts = list(atmlsts)
        if len(atmlsts) != nfrag:
            raise ValueError('atmlsts must have one seed atom list per block.')

    s1e = np.asarray(s1e)
    rr = atom_distance(mol)
    aoslices = mol.aoslice_by_atom()[:, 2:]
    all_atoms = np.arange(mol.natm, dtype=np.int32)
    fragment_domains = []

    for occupied_block, seed_atoms in zip(occupied_blocks, atmlsts):
        occupied_block = np.asarray(occupied_block)
        if occupied_block.ndim == 1:
            occupied_block = occupied_block.reshape(-1, 1)
        if occupied_block.ndim != 2 or occupied_block.shape[0] != mol.nao:
            raise ValueError(
                'Each occupied block must have shape (mol.nao, nocc_block).'
            )
        if occupied_block.shape[1] == 0:
            raise ValueError('Occupied fragment blocks may not be empty.')

        sx = s1e @ occupied_block
        # Summing over the complete column block before taking the magnitude
        # makes this population invariant under X_F -> X_F U_F.
        q = np.abs(np.asarray([
            np.sum(
                occupied_block[slice(*aoslices[a])].conj()
                * sx[slice(*aoslices[a])]
            )
            for a in all_atoms
        ]))

        if seed_atoms is None:
            selected = all_atoms[q > q_thr]
        else:
            selected = np.unique(np.asarray(seed_atoms, dtype=np.int32).ravel())
            if np.any((selected < 0) | (selected >= mol.natm)):
                raise ValueError('Fragment seed atom index is out of range.')
        if selected.size == 0:
            selected = np.asarray([int(np.argmax(q))], dtype=np.int32)

        completeness = _fragment_trace_completeness_numpy(
            mol, occupied_block, s1e=s1e, atmlst=selected
        )
        while completeness < bp_thr and selected.size < mol.natm:
            remaining = np.setdiff1d(all_atoms, selected, assume_unique=True)
            distance_to_domain = np.min(rr[np.ix_(remaining, selected)], axis=1)
            # np.lexsort uses the last key as primary: distance first, then
            # larger population, and finally atom index.
            order = np.lexsort((remaining, -q[remaining], distance_to_domain))
            selected = np.sort(np.append(selected, remaining[order[0]]))
            completeness = _fragment_trace_completeness_numpy(
                mol, occupied_block, s1e=s1e, atmlst=selected
            )

        fragment_domains.append(selected)

    return util.list_to_array(fragment_domains)


def _compute_av_numpy(mol, mo, s1e=None, atmlst=None):
    """NumPy BP value for non-differentiable domain topology selection."""
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    mo = np.asarray(mo)
    ao_idx = util.ao_index_by_atom(mol, atmlst)
    s1e = np.asarray(s1e)
    v = s1e[ao_idx] @ mo
    a = np.linalg.solve(s1e[np.ix_(ao_idx, ao_idx)], v)
    return np.sum(a * v, axis=0)


def get_primary_domain(mol, lmo_bp_domain, pao_bp_domain, ao2pao_map=None):
    """Extend LMO BP domain by PAO BP domains.
    """
    if ao2pao_map is None:
        ao2pao_map = np.arange(mol.nao)

    nocc = len(lmo_bp_domain)
    aoslices = mol.aoslice_by_atom()[:,2:]
    pd_atmlst = []

    for i in range(nocc):
        _atms = np.empty((0,), dtype=np.int32)
        for a in lmo_bp_domain[i]:
            _tmp = ao2pao_map[slice(*aoslices[a])]
            pao_idx = _tmp[_tmp >= 0]
            _atms = np.union1d(_atms, reduce(np.union1d, pao_bp_domain[pao_idx]))
        pd_atmlst.append(np.union1d(lmo_bp_domain[i], _atms))

    return util.list_to_array(pd_atmlst)


def _compute_av(mol, mo, s1e=None, atmlst=None):
    """Compute BP value.

    Dispatches to a pure numpy path on concrete inputs to avoid XLA
    compile-cache growth in per-LMO eager prescreen builds. Keeps the JAX
    path when called under ``jax.value_and_grad`` so the gradient propagates.
    """
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if atmlst is None:
        atmlst = np.arange(mol.natm)

    if isinstance(mo, jax.core.Tracer) or (
        s1e is not None and isinstance(s1e, jax.core.Tracer)
    ):
        mo = jnp.asarray(mo)
        ao_idx = util.ao_index_by_atom(mol, atmlst)
        s1e = jnp.asarray(s1e)
        v = s1e[ao_idx] @ mo
        a = jnp.linalg.solve(
            s1e[jnp.ix_(jnp.asarray(ao_idx), jnp.asarray(ao_idx))], v
        )
        return jnp.sum(a * v, axis=0)
    return _compute_av_numpy(mol, mo, s1e=s1e, atmlst=atmlst)


def atom_distance(mol, atmlst=None):
    """Atomic distance array
    """
    if atmlst is None:
        atmlst = np.arange(mol.natm)
    coords = np.asarray(mol.atom_coords())[np.asarray(atmlst)].reshape(-1,3)
    diff = coords[:,None,:] - coords[None,:,:]
    return np.linalg.norm(diff, axis=-1)
