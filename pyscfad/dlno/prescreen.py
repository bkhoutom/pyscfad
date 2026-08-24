"""DLNO prescreen compatibility shim for pyscfad.

This module provides `build_dlno_prescreen_data` and helper I/O wrappers
that try to use the local `pyscfad.dlno` implementation when available,
falling back to an externally installed `dlno` package.
"""
from contextlib import contextmanager
from functools import reduce
import time
import numpy as onp
import numpy as np
import scipy.linalg as onp_scipy_linalg
import jax
import jax.numpy as jnp
from pyscf import gto, lo, scf, mp, lib
from pyscf.data.elements import chemcore

from pyscfad.ao2mo import _ao2mo
from pyscfad.mp import dfmp2
from pyscfad.lo.orth import vec_lowdin
from pyscfad import scipy
from pyscfad.lno import lno_base
from pyscfad.ops import stop_grad
from pyscfad.tools import resource_profile
from pyscfad.dlno import multipole_numpy as static_multipole

try:
    from pyscfad.dlno import dlno as dlno_mod
    from pyscfad.dlno import pao as dlno_pao
    from pyscfad.dlno import util as dlno_util
except Exception:
    # Fall back to system-installed dlno if local copy not complete
    from dlno import dlno as dlno_mod
    from dlno import pao as dlno_pao
    from dlno import util as dlno_util


def _is_traced(*xs):
    for x in xs:
        if isinstance(x, jax.core.Tracer):
            return True
    return False


def _diag_subspace_energies(vir_prescreen, fock_block):
    """Compute diag(V^H F V) eagerly with numpy when possible.

    When ``vir_prescreen`` or ``fock_block`` are JAX tracers we stay on the
    JAX path so gradients propagate; otherwise we drop straight into numpy
    to skip XLA compilation and avoid pinning the intermediates in the
    JAX allocator.
    """
    if _is_traced(vir_prescreen, fock_block):
        return jnp.real(jnp.diag(vir_prescreen.T.conj() @ fock_block @ vir_prescreen))
    v = onp.asarray(vir_prescreen)
    f = onp.asarray(fock_block)
    return onp.real(onp.einsum('ji,jk,ki->i', v.conj(), f, v))


def _maybe_asarray(x):
    """Return ``jnp.asarray(x)`` only if ``x`` already lives in JAX.

    Used at the end of the build/rebuild loops where converting a freshly
    computed numpy array to a JAX array does no useful work — the caller
    will consume it as numpy anyway when not under tracing.
    """
    if _is_traced(x):
        return jnp.asarray(x)
    return onp.asarray(x)


@contextmanager
def _topology_profile_section(rows, name):
    start = time.perf_counter()
    profile_start = resource_profile.start()
    try:
        yield
    finally:
        wall_s = time.perf_counter() - start
        rows.append({'section': name, 'wall_s': wall_s})
        resource_profile.finish(
            f'prescreen.{name}',
            profile_start,
        )


def union_objects(obj_list):
    obj_list = [np.asarray(x, dtype=np.int32) for x in obj_list if len(x) > 0]
    if not obj_list:
        return np.empty((0,), dtype=np.int32)
    return reduce(np.union1d, obj_list)


def strong_pair_lists(pair_energy, pair_energy_thr):
    pair_energy = onp.asarray(stop_grad(pair_energy))
    pair_energy_with_self = pair_energy + onp.eye(pair_energy.shape[0])
    return [
        onp.where(onp.abs(pair_energy_with_self[i]) > pair_energy_thr)[0]
        for i in range(pair_energy.shape[0])
    ]


def _int_frozen(frozen):
    if frozen is None:
        return 0
    if isinstance(frozen, (int, np.integer)):
        return int(frozen)
    raise NotImplementedError('DLNO prescreen currently supports integer frozen cores only.')


def _active_nocc(mf, frozen=None):
    return int(np.count_nonzero(mf.mo_occ > 0)) - _int_frozen(frozen)


def _pao_space(mf, frozen=None, s1e=None, norm_thr=1e-4):
    if s1e is None:
        s1e = mf.mol.intor_symmetric('int1e_ovlp')
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    mos = mf.mo_coeff[:, :nocc]
    return dlno_pao.pao(mf.mol, mos, s1e, norm_thr)


def _semicanonicalize_local(mol, mos, fock, atmlst):
    """Eigh of the local Fock block in the (small) ``mos`` basis.

    Dispatches on whether ``mos`` / ``fock`` are JAX tracers. Concrete inputs
    take a numpy/scipy path; tracer inputs keep the JAX path so the gradient
    propagates inside ``jax.value_and_grad``.
    """
    ao_idx = dlno_util.ao_index_by_atom(mol, atmlst)
    if isinstance(mos, jax.core.Tracer) or isinstance(fock, jax.core.Tracer):
        fock22 = fock[np.ix_(ao_idx, ao_idx)]
        f = mos.T.conj() @ fock22 @ mos
        if mos.shape[1] == 1:
            return jnp.asarray(f).reshape(-1), mos
        w, v = scipy.linalg.eigh(f)
        return w, mos @ v
    mos_np = onp.asarray(mos)
    fock_np = onp.asarray(fock)
    fock22 = fock_np[onp.ix_(ao_idx, ao_idx)]
    f = mos_np.T.conj() @ fock22 @ mos_np
    if mos_np.shape[1] == 1:
        return f.reshape(-1), mos_np
    w, v = onp_scipy_linalg.eigh(f)
    return w, mos_np @ v


def _multipole_pair_data_numpy(mol, lmo, pao, e_occ, e_vir, atmlst, order):
    return static_multipole.multipole_orbital_data(
        mol, e_occ, lmo, e_vir, pao, atmlst=atmlst, order=order
    )


def _precompute_multipole_pair_data_numpy(mol, occ, vir, e_occ, e_vir,
                                          atmlst, order):
    return [
        _multipole_pair_data_numpy(mol, lmo_i, pao_i, eo_i, ev_i, atmlst_i, order)
        for lmo_i, pao_i, eo_i, ev_i, atmlst_i
        in zip(occ, vir, e_occ, e_vir, atmlst)
    ]


def _pair_energy_from_multipole_data_numpy(pair_data, order):
    return static_multipole.multipole_pair_energy_matrix(
        pair_data, order=order
    )


def _pair_energy_multipole_numpy(mol, e_occ, mo_occ, e_vir, mo_vir,
                                 atmlst=None, order=4):
    """Static NumPy version of the DLNO multipole pair screen."""
    nocc = len(e_occ)
    if atmlst is None:
        atmlst = [None] * nocc

    pair_data = _precompute_multipole_pair_data_numpy(
        mol, mo_occ, mo_vir, e_occ, e_vir, atmlst, order
    )
    return _pair_energy_from_multipole_data_numpy(pair_data, order)


def _loop_ov_blocks(mymp, mo_coeff, nocc):
    ijslice = (0, nocc, nocc, mo_coeff.shape[1])
    for eri1 in mymp.with_df.loop():
        qov = _ao2mo.nr_e2(eri1, mo_coeff, ijslice, aosym='s2')
        yield onp.asarray(qov).reshape(-1, nocc, mo_coeff.shape[1] - nocc)


def _occupied_pair_energy_matrix_blockwise(mymp, mo_coeff, mo_energy):
    nocc = mymp.nocc
    nvir = mymp.nmo - nocc
    eia = onp.asarray(mo_energy[:nocc, None] - mo_energy[None, nocc:])
    pair_occ = onp.zeros((nocc, nocc))

    for i in range(nocc):
        gi = onp.zeros((nocc, nvir, nvir))
        for qov in _loop_ov_blocks(mymp, mo_coeff, nocc):
            la = qov[:, i, :]
            buf = la.T @ qov.reshape(qov.shape[0], -1)
            gi += buf.reshape(nvir, nocc, nvir).transpose(1, 0, 2)

        denom = eia[:, None, :] + eia[i, :, None]
        t2i = gi / denom
        pair_occ[i] = 2.0 * onp.einsum('jab,jab->j', t2i, gi)
        pair_occ[i] -= onp.einsum('jab,jba->j', t2i, gi)

    pair_occ = 0.5 * (pair_occ + pair_occ.T)
    return np.asarray(pair_occ)


def _occupied_pair_energy_matrix(mf, frozen=None):
    mymp = dfmp2.MP2(mf, frozen=frozen)
    eris = mymp.ao2mo()
    mo_coeff = eris.mo_coeff
    mo_energy = eris.mo_energy

    nocc = mymp.nocc
    nvir = mymp.nmo - nocc

    if getattr(mymp.with_df, 'incore', True):
        try:
            Lov = mymp.loop_ao2mo(mo_coeff, nocc, with_t2=False)
            ovov = (Lov.T @ Lov).reshape(nocc, nvir, nocc, nvir)
            oovv = ovov.transpose(0, 2, 1, 3)

            eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
            denom = eia[:, None, :, None] + eia[None, :, None, :]
            t2 = oovv / denom
            pair_occ = 2 * np.einsum('ijab,ijab->ij', t2, oovv)
            pair_occ -= np.einsum('ijab,ijba->ij', t2, oovv)
        except RuntimeError:
            pair_occ = _occupied_pair_energy_matrix_blockwise(mymp, mo_coeff, mo_energy)
    else:
        pair_occ = _occupied_pair_energy_matrix_blockwise(mymp, mo_coeff, mo_energy)
    return pair_occ, mo_coeff[:, :nocc]


def _lo_pair_energy_matrix(mf, lo_coeff, frozen=None, s1e=None):
    if s1e is None:
        s1e = mf.mol.intor_symmetric('int1e_ovlp')

    pair_occ, orbocc = _occupied_pair_energy_matrix(mf, frozen=frozen)
    u_lo_occ = lo_coeff.T.conj() @ s1e @ orbocc
    pair_lo = u_lo_occ @ pair_occ @ u_lo_occ.T.conj()
    pair_lo = 0.5 * (pair_lo + pair_lo.T.conj())
    return pair_lo.real


def _canonicalize_single_lo_domain(
    mol,
    lo_vec,
    pao_i,
    primary_domain,
    fock,
    s1e,
    pao_bp_domain_thr,
):
    ao_idx = dlno_util.ao_index_by_atom(mol, primary_domain)
    s21 = s1e[ao_idx]
    s22 = s1e[np.ix_(ao_idx, ao_idx)]

    lo_i = dlno_util.project_mo(lo_vec, s21, s22)
    lo_i = vec_lowdin(lo_i, s=s22)
    e_occ_i, lo_i_canon = _semicanonicalize_local(mol, lo_i, fock, primary_domain)

    if pao_i.shape[1] > 0:
        av = dlno_mod._compute_av(mol, pao_i, s1e=s1e, atmlst=primary_domain)
        pao_i = pao_i[:, av > pao_bp_domain_thr]
    if pao_i.shape[1] > 0:
        pao_i = dlno_util.project_mo(pao_i, s21, s22)
        pao_i = vec_lowdin(pao_i, s=s22)
        pao_i = dlno_util.orthogonalize(lo_i_canon, pao_i, s22)
        pao_i = vec_lowdin(pao_i, s=s22)
        e_vir_i, pao_i_canon = _semicanonicalize_local(mol, pao_i, fock, primary_domain)
    else:
        e_vir_i = onp.zeros((0,))
        pao_i_canon = onp.zeros((len(ao_idx), 0))
    return _maybe_asarray(e_occ_i).reshape(-1), lo_i_canon, _maybe_asarray(e_vir_i), pao_i_canon


def _lo_pair_energy_matrix_multipole(
    mf,
    lo_coeff,
    lmo_bp_domain,
    lmo_primary_domain,
    pao,
    ao2pao_map,
    *,
    s1e=None,
    fock=None,
    domain_pao_thr=1e-4,
    pao_bp_domain_thr=0.98,
    multipole_order=4,
):
    mol = mf.mol
    if s1e is None:
        s1e = mol.intor_symmetric('int1e_ovlp')
    if fock is None:
        fock = mf.get_fock()

    pair_data = []
    for i in range(lo_coeff.shape[1]):
        lo_i = lo_coeff[:, i:i+1]
        pao_i = dlno_pao.pao_overlap_with_domain(
            mol,
            pao,
            lmo_bp_domain[i],
            ao2pao_map=ao2pao_map,
            s1e=s1e,
            ovlp_thr=domain_pao_thr,
        )
        eo_i, vo_i, ev_i, vv_i = _canonicalize_single_lo_domain(
            mol,
            lo_i,
            pao_i,
            lmo_primary_domain[i],
            fock,
            s1e,
            pao_bp_domain_thr,
        )
        pair_data.append(
            _multipole_pair_data_numpy(
                mol,
                vo_i,
                vv_i,
                eo_i[0],
                ev_i,
                lmo_primary_domain[i],
                multipole_order,
            )
        )

    pair_lo = _pair_energy_from_multipole_data_numpy(pair_data, multipole_order)
    pair_lo = 0.5 * (pair_lo + pair_lo.T.conj())
    return onp.asarray(pair_lo.real)


def _build_lo_indexed_prescreen_data(
    mf,
    lo_coeff,
    frag_lolist,
    frozen=None,
    lmo_bp_domain_thr=0.999,
    pao_bp_domain_thr=0.98,
    domain_pao_thr=1e-4,
    pair_energy_thr=1e-4,
    pair_energy_model='exact',
    pao_norm_thr=1e-4,
    multipole_order=4,
):
    profile_rows = []
    mol = mf.mol
    with _topology_profile_section(profile_rows, 'overlap/fock'):
        s1e = mol.intor_symmetric('int1e_ovlp')
        fock = mf.get_fock()

    lmo_bp_domain = get_bp_domain = None
    # Treat domain topology as fixed metadata. The selected prescreen spaces
    # inside those domains remain the only objects consumed downstream.
    with _topology_profile_section(profile_rows, 'LMO BP domains'):
        lmo_bp_domain = dlno_mod.get_bp_domain(mol, lo_coeff, s1e, lmo_bp_domain_thr) if hasattr(dlno_mod, 'get_bp_domain') else None
        if lmo_bp_domain is None:
            from pyscfad.dlno.domain import get_bp_domain as _get_bp_domain
            lmo_bp_domain = _get_bp_domain(mol, lo_coeff, s1e, lmo_bp_domain_thr)
    with _topology_profile_section(profile_rows, 'PAO build'):
        pao, ao2pao_map = _pao_space(mf, frozen=frozen, s1e=s1e, norm_thr=pao_norm_thr)
    from pyscfad.dlno.domain import get_bp_domain as _get_bp_domain, get_primary_domain as _get_primary_domain
    with _topology_profile_section(profile_rows, 'PAO BP domains'):
        pao_bp_domain = _get_bp_domain(mol, pao, s1e, pao_bp_domain_thr)
    with _topology_profile_section(profile_rows, 'primary domains'):
        lmo_primary_domain = _get_primary_domain(mol, lmo_bp_domain, pao_bp_domain, ao2pao_map)

    with _topology_profile_section(profile_rows, f'{pair_energy_model} pair energies'):
        if pair_energy_model == 'exact':
            pair_energy = _lo_pair_energy_matrix(mf, lo_coeff, frozen=frozen, s1e=s1e)
        elif pair_energy_model == 'multipole':
            pair_energy = _lo_pair_energy_matrix_multipole(
                mf,
                lo_coeff,
                lmo_bp_domain,
                lmo_primary_domain,
                pao,
                ao2pao_map,
                s1e=s1e,
                fock=fock,
                domain_pao_thr=domain_pao_thr,
                pao_bp_domain_thr=pao_bp_domain_thr,
                multipole_order=multipole_order,
            )
        else:
            raise ValueError(f'Unknown pair_energy_model: {pair_energy_model}')

    with _topology_profile_section(profile_rows, 'strong-pair domains'):
        strong_pairs = strong_pair_lists(pair_energy, pair_energy_thr)
        extended_bp_domain = [
            union_objects([lmo_bp_domain[j] for j in idx]) for idx in strong_pairs
        ]
        extended_primary_domain = [
            union_objects([lmo_primary_domain[j] for j in idx]) for idx in strong_pairs
        ]

    fragment_data = []
    with _topology_profile_section(profile_rows, 'fragment prescreen spaces'):
        for ifrag, loidx in enumerate(frag_lolist):
            loidx = np.asarray(loidx, dtype=np.int32)
            frag_strong = union_objects([strong_pairs[i] for i in loidx])
            frag_ext_bp = union_objects([extended_bp_domain[i] for i in loidx])
            frag_ext_primary = union_objects([extended_primary_domain[i] for i in loidx])

            ao_idx = dlno_util.ao_index_by_atom(mol, frag_ext_primary)
            s21 = s1e[ao_idx]
            s22 = s1e[np.ix_(ao_idx, ao_idx)]

            lmo_block = lo_coeff[:, frag_strong]
            lmo_block_prj = dlno_util.project_mo(lmo_block, s21, s22)
            lmo_block_prj = vec_lowdin(lmo_block_prj, s=s22)
            e_occ_prescreen, occ_prescreen = _semicanonicalize_local(
                mol, lmo_block_prj, fock, frag_ext_primary
            )

            frag_pao = dlno_pao.pao_overlap_with_domain(
                mol,
                pao,
                list(frag_ext_bp),
                ao2pao_map=ao2pao_map,
                s1e=s1e,
                ovlp_thr=domain_pao_thr,
            )
            if frag_pao.shape[1] > 0:
                av = dlno_mod._compute_av(mol, frag_pao, s1e=s1e, atmlst=frag_ext_primary)
                frag_pao = frag_pao[:, av > pao_bp_domain_thr]
            if frag_pao.shape[1] > 0:
                frag_pao_prj = dlno_util.project_mo(frag_pao, s21, s22)
                frag_pao_prj = dlno_util.orthogonalize(occ_prescreen, frag_pao_prj, s22)
                frag_pao_prj = lno_base.orthonormalize_metric_colspace_smooth(
                    frag_pao_prj, s22, thresh=1e-10
                )
                # The downstream DLNO path only consumes the virtual prescreen span.
                # Keeping these coefficients in a fixed-gauge orthonormal basis avoids
                # introducing an extra semicanonical rotation whose derivative can
                # become disproportionately large in near-degenerate local spaces.
                vir_prescreen = frag_pao_prj
                if vir_prescreen.shape[1] > 0:
                    fock22 = fock[np.ix_(ao_idx, ao_idx)]
                    e_vir_prescreen = _diag_subspace_energies(vir_prescreen, fock22)
                else:
                    e_vir_prescreen = onp.zeros((0,))
            else:
                e_vir_prescreen = onp.zeros((0,))
                vir_prescreen = onp.zeros((len(ao_idx), 0))

            fragment_data.append(
                {
                    'fragment_index': ifrag,
                    'lo_indices': loidx,
                    'strong_lmo_indices': frag_strong,
                    'extended_bp_domain': frag_ext_bp,
                    'extended_primary_domain': frag_ext_primary,
                    'occ_prescreen_energies': e_occ_prescreen,
                    'occ_prescreen_coeff': occ_prescreen,
                    'vir_prescreen_energies': e_vir_prescreen,
                    'vir_prescreen_coeff': vir_prescreen,
                }
            )

    return {
        'frozen': frozen,
        'lo_coeff': lo_coeff,
        'frag_lolist': frag_lolist,
        's1e': s1e,
        'fock': fock,
        'pao_norm_thr': pao_norm_thr,
        'domain_pao_thr': domain_pao_thr,
        'pao_bp_domain_thr': pao_bp_domain_thr,
        'lmo_bp_domain': lmo_bp_domain,
        'pao_bp_domain': pao_bp_domain,
        'lmo_primary_domain': lmo_primary_domain,
        'pao': pao,
        'ao2pao_map': ao2pao_map,
        'pair_energy': pair_energy,
        'pair_energy_model': pair_energy_model,
        'strong_pairs': strong_pairs,
        'extended_bp_domain': extended_bp_domain,
        'extended_primary_domain': extended_primary_domain,
        'fragment_data': fragment_data,
        'topology_profile': profile_rows,
    }


def build_dlno_prescreen_data(
    mf,
    lo_coeff,
    frag_lolist,
    frozen=None,
    lmo_bp_domain_thr=0.999,
    pao_bp_domain_thr=0.98,
    domain_pao_thr=1e-4,
    pair_energy_thr=1e-4,
    pair_energy_model='multipole',
    pao_norm_thr=1e-4,
    multipole_order=4,
):
    """Precompute DLNO metadata for use with LNO in pyscfad.

    This is a compatibility wrapper that mirrors the behavior of the
    pyscf-forge prescreen helper. If a full local `pyscfad.dlno` is
    available it will be preferred; otherwise it falls back to an
    external `dlno` package if present.
    """
    if lo_coeff.shape[1] != _active_nocc(mf, frozen):
        return _build_lo_indexed_prescreen_data(
            mf,
            lo_coeff,
            frag_lolist,
            frozen=frozen,
            lmo_bp_domain_thr=lmo_bp_domain_thr,
            pao_bp_domain_thr=pao_bp_domain_thr,
            domain_pao_thr=domain_pao_thr,
            pair_energy_thr=pair_energy_thr,
            pair_energy_model=pair_energy_model,
            pao_norm_thr=pao_norm_thr,
            multipole_order=multipole_order,
        )

    profile_rows = []
    mol = mf.mol
    with _topology_profile_section(profile_rows, 'DLNO object'):
        dlno = dlno_mod.DLNO(mf, frozen=frozen)
        dlno.lmo = lo_coeff
        dlno.lmo_bp_domain_thr = lmo_bp_domain_thr
        dlno.pao_bp_domain_thr = pao_bp_domain_thr
        dlno.domain_pao_thr = domain_pao_thr
        dlno.pair_energy_thr = pair_energy_thr
        dlno.multipole_order = multipole_order

    with _topology_profile_section(profile_rows, 'overlap/fock'):
        s1e = dlno.s1e
        fock = dlno.fock
    with _topology_profile_section(profile_rows, 'LMO BP domains'):
        lmo_bp_domain = dlno.lmo_bp_domain
    with _topology_profile_section(profile_rows, 'PAO build'):
        pao = dlno.pao
        ao2pao_map = dlno.ao2pao_map
    with _topology_profile_section(profile_rows, 'PAO BP domains'):
        pao_bp_domain = dlno.pao_bp_domain
    with _topology_profile_section(profile_rows, 'primary domains'):
        lmo_primary_domain = dlno.build_lmo_primary_domain()
        dlno.lmo_primary_domain = lmo_primary_domain

    with _topology_profile_section(profile_rows, 'domain PAOs'):
        domain_pao = dlno.build_domain_pao()
    with _topology_profile_section(profile_rows, 'domain canonicalization'):
        (eo, vo), (ev, vv) = dlno.canonicalize(domain_pao)
    with _topology_profile_section(profile_rows, 'multipole pair energies'):
        pair_energy = _pair_energy_multipole_numpy(
            mol,
            eo,
            vo,
            ev,
            vv,
            lmo_primary_domain,
            multipole_order,
        )
    with _topology_profile_section(profile_rows, 'strong-pair domains'):
        strong_pairs = strong_pair_lists(pair_energy, pair_energy_thr)

        extended_bp_domain = [
            union_objects([lmo_bp_domain[j] for j in idx]) for idx in strong_pairs
        ]
        extended_primary_domain = [
            union_objects([lmo_primary_domain[j] for j in idx]) for idx in strong_pairs
        ]
    fragment_data = []
    with _topology_profile_section(profile_rows, 'fragment prescreen spaces'):
        for ifrag, loidx in enumerate(frag_lolist):
            loidx = np.asarray(loidx, dtype=np.int32)
            frag_strong = union_objects([strong_pairs[i] for i in loidx])
            frag_ext_bp = union_objects([extended_bp_domain[i] for i in loidx])
            frag_ext_primary = union_objects([extended_primary_domain[i] for i in loidx])

            ao_idx = dlno_util.ao_index_by_atom(mol, frag_ext_primary)
            s21 = s1e[ao_idx]
            s22 = s1e[np.ix_(ao_idx, ao_idx)]

            lmo_block = lo_coeff[:, frag_strong]
            lmo_block_prj = dlno_util.project_mo(lmo_block, s21, s22)
            lmo_block_prj = vec_lowdin(lmo_block_prj, s=s22)
            e_occ_prescreen, occ_prescreen = dlno_mod.semicanonicalize(
                mol, lmo_block_prj, fock, frag_ext_primary
            )

            frag_pao = dlno_pao.pao_overlap_with_domain(
                mol,
                pao,
                list(frag_ext_bp),
                ao2pao_map=ao2pao_map,
                s1e=s1e,
                ovlp_thr=domain_pao_thr,
            )
            if frag_pao.shape[1] > 0:
                av = dlno_mod._compute_av(mol, frag_pao, s1e=s1e, atmlst=frag_ext_primary)
                frag_pao = frag_pao[:, av > pao_bp_domain_thr]
            if frag_pao.shape[1] > 0:
                frag_pao_prj = dlno_util.project_mo(frag_pao, s21, s22)
                frag_pao_prj = dlno_util.orthogonalize(occ_prescreen, frag_pao_prj, s22)
                frag_pao_prj = lno_base.orthonormalize_metric_colspace_smooth(
                    frag_pao_prj, s22, thresh=1e-10
                )
                vir_prescreen = frag_pao_prj
                if vir_prescreen.shape[1] > 0:
                    fock22 = fock[np.ix_(ao_idx, ao_idx)]
                    e_vir_prescreen = _diag_subspace_energies(vir_prescreen, fock22)
                else:
                    e_vir_prescreen = onp.zeros((0,))
            else:
                e_vir_prescreen = onp.zeros((0,))
                vir_prescreen = onp.zeros((len(ao_idx), 0))

            fragment_data.append(
                {
                    "fragment_index": ifrag,
                    "lo_indices": loidx,
                    "strong_lmo_indices": frag_strong,
                    "extended_bp_domain": frag_ext_bp,
                    "extended_primary_domain": frag_ext_primary,
                    "occ_prescreen_energies": _maybe_asarray(e_occ_prescreen),
                    "occ_prescreen_coeff": occ_prescreen,
                    "vir_prescreen_energies": _maybe_asarray(e_vir_prescreen),
                    "vir_prescreen_coeff": vir_prescreen,
                }
            )

    return {
        "frozen": frozen,
        "lo_coeff": lo_coeff,
        "frag_lolist": frag_lolist,
        "s1e": s1e,
        "fock": fock,
        "pao_norm_thr": pao_norm_thr,
        "domain_pao_thr": domain_pao_thr,
        "pao_bp_domain_thr": pao_bp_domain_thr,
        "lmo_bp_domain": lmo_bp_domain,
        "pao_bp_domain": pao_bp_domain,
        "lmo_primary_domain": lmo_primary_domain,
        "pao": pao,
        "ao2pao_map": ao2pao_map,
        "domain_pao": domain_pao,
        "local_occ_energies": eo,
        "local_occ_orbitals": vo,
        "local_vir_energies": ev,
        "local_vir_orbitals": vv,
        "pair_energy": pair_energy,
        "strong_pairs": strong_pairs,
        "extended_bp_domain": extended_bp_domain,
        "extended_primary_domain": extended_primary_domain,
        "fragment_data": fragment_data,
        "topology_profile": profile_rows,
    }


def rebuild_dlno_prescreen_data(mf, lo_coeff, topology_data, *, frozen=None):
    """Rebuild prescreen orbital spaces from the current SCF state.

    The combinatorial DLNO metadata such as strong pairs and atom domains are
    taken from ``topology_data`` and treated as fixed. The fragment-local
    occupied/virtual prescreen spaces are reconstructed from the current
    ``mf``/``lo_coeff`` so their first-order response remains differentiable.
    """
    profile_total = resource_profile.start()
    mol = mf.mol
    if frozen is None:
        frozen = topology_data.get("frozen", None)

    profile_setup = resource_profile.start()
    s1e = mol.intor_symmetric("int1e_ovlp")
    fock = mf.get_fock()
    pao_norm_thr = topology_data.get("pao_norm_thr", 1e-4)
    domain_pao_thr = topology_data.get("domain_pao_thr", 1e-4)
    pao_bp_domain_thr = topology_data.get("pao_bp_domain_thr", 0.98)

    pao, ao2pao_map = _pao_space(
        mf, frozen=frozen, s1e=s1e, norm_thr=pao_norm_thr
    )
    resource_profile.finish(
        'prescreen.rebuild_overlap_fock_pao',
        profile_setup,
        pao_shape=tuple(pao.shape),
        pao_mib=resource_profile.estimated_array_mib(pao),
    )

    fragment_data = []
    for frag in topology_data["fragment_data"]:
        profile_fragment = resource_profile.start()
        loidx = np.asarray(frag["lo_indices"], dtype=np.int32)
        frag_strong = np.asarray(frag["strong_lmo_indices"], dtype=np.int32)
        frag_ext_bp = np.asarray(frag["extended_bp_domain"], dtype=np.int32)
        frag_ext_primary = np.asarray(frag["extended_primary_domain"], dtype=np.int32)

        ao_idx = dlno_util.ao_index_by_atom(mol, frag_ext_primary)
        s21 = s1e[ao_idx]
        s22 = s1e[np.ix_(ao_idx, ao_idx)]

        lmo_block = lo_coeff[:, frag_strong]
        lmo_block_prj = dlno_util.project_mo(lmo_block, s21, s22)
        lmo_block_prj = vec_lowdin(lmo_block_prj, s=s22)
        e_occ_prescreen, occ_prescreen = _semicanonicalize_local(
            mol, lmo_block_prj, fock, frag_ext_primary
        )

        frag_pao = dlno_pao.pao_overlap_with_domain(
            mol,
            pao,
            list(frag_ext_bp),
            ao2pao_map=ao2pao_map,
            s1e=s1e,
            ovlp_thr=domain_pao_thr,
        )
        if frag_pao.shape[1] > 0:
            av = dlno_mod._compute_av(mol, frag_pao, s1e=s1e, atmlst=frag_ext_primary)
            frag_pao = frag_pao[:, av > pao_bp_domain_thr]
        if frag_pao.shape[1] > 0:
            frag_pao_prj = dlno_util.project_mo(frag_pao, s21, s22)
            frag_pao_prj = dlno_util.orthogonalize(occ_prescreen, frag_pao_prj, s22)
            frag_pao_prj = lno_base.orthonormalize_metric_colspace_smooth(
                frag_pao_prj, s22, thresh=1e-10
            )
            vir_prescreen = frag_pao_prj
            if vir_prescreen.shape[1] > 0:
                fock22 = fock[np.ix_(ao_idx, ao_idx)]
                e_vir_prescreen = _diag_subspace_energies(vir_prescreen, fock22)
            else:
                e_vir_prescreen = onp.zeros((0,))
        else:
            e_vir_prescreen = onp.zeros((0,))
            vir_prescreen = onp.zeros((len(ao_idx), 0))

        fragment_data.append(
            {
                "fragment_index": frag["fragment_index"],
                "lo_indices": loidx,
                "strong_lmo_indices": frag_strong,
                "extended_bp_domain": frag_ext_bp,
                "extended_primary_domain": frag_ext_primary,
                "occ_prescreen_energies": _maybe_asarray(e_occ_prescreen),
                "occ_prescreen_coeff": occ_prescreen,
                "vir_prescreen_energies": _maybe_asarray(e_vir_prescreen),
                "vir_prescreen_coeff": vir_prescreen,
            }
        )
        resource_profile.finish(
            'prescreen.rebuild_fragment_spaces',
            profile_fragment,
            frag=int(frag["fragment_index"]) + 1,
            domain_atoms=int(frag_ext_primary.size),
            domain_aos=int(ao_idx.size),
            occ=int(occ_prescreen.shape[1]),
            vir=int(vir_prescreen.shape[1]),
            coeff_mib=resource_profile.estimated_array_mib(
                occ_prescreen, vir_prescreen
            ),
        )

    data = dict(topology_data)
    data.update(
        {
            "frozen": frozen,
            "lo_coeff": lo_coeff,
            "s1e": s1e,
            "fock": fock,
            "pao": pao,
            "ao2pao_map": ao2pao_map,
            "fragment_data": fragment_data,
        }
    )
    resource_profile.finish(
        'prescreen.rebuild_total',
        profile_total,
        fragments=len(fragment_data),
    )
    return data


def print_summary(data):
    pair_energy = data["pair_energy"]
    print()
    print("DLNO precomputed metadata for pyscf.lno prescreening")
    print("====================================================")
    print(f"Number of LMOs                  : {data['lo_coeff'].shape[1]}")
    print(f"Pair-energy matrix shape        : {pair_energy.shape}")
    print(f"Max |pair energy|               : {np.max(np.abs(pair_energy)):.6e}")
    print(f"Mean strong partners per LMO    : {np.mean([len(x) for x in data['strong_pairs']]):.2f}")
    print()

    for frag in data["fragment_data"]:
        print(f"Fragment {frag['fragment_index']}")
        print(f"  LO indices                    : {frag['lo_indices'].tolist()}")
        print(f"  Strong LMOs                   : {frag['strong_lmo_indices'].tolist()}")
        print(f"  Extended BP domain atoms      : {frag['extended_bp_domain'].tolist()}")
        print(f"  Extended primary domain atoms : {frag['extended_primary_domain'].tolist()}")
        print(f"  Occ prescreen size            : {frag['occ_prescreen_coeff'].shape[1]}")
        print(f"  Vir prescreen size            : {frag['vir_prescreen_coeff'].shape[1]}")
        print()


def load_or_run_scf(mf, chkfile, cderi_file=None):
    mf.chkfile = str(chkfile)
    if cderi_file is not None and getattr(mf, "with_df", None) is not None:
        mf.with_df._cderi_to_save = str(cderi_file)
        if cderi_file.exists():
            mf.with_df._cderi = str(cderi_file)
    if chkfile.exists():
        print(f"Loading SCF checkpoint from {chkfile}")
        mf.__dict__.update(lib.chkfile.load(str(chkfile), "scf"))
        mf.converged = True
    else:
        print(f"Running SCF and saving checkpoint to {chkfile}")
        mf.kernel()
    return mf


def load_or_localize_pm(mol, orbocc, lo_coeff_file, localize=None):
    from pyscfad.lo.boys import boys as boys_localize

    if lo_coeff_file.exists():
        print(f"Loading localized orbitals from {lo_coeff_file}")
        return np.load(lo_coeff_file, allow_pickle=False)

    print(f"Running Pipek-Mezey localization and saving to {lo_coeff_file}")
    orbocc_np = np.asarray(orbocc)
    if localize is None:
        localizers = [
            ("pyscf_pipek", lambda m, c: lo.PipekMezey(m, c).kernel()),
            ("pyscfad_boys", lambda m, c: boys_localize(m, c)),
            ("identity", lambda _m, c: c),
        ]
    else:
        localizers = [
            (
                getattr(localize, "__name__", type(localize).__name__),
                lambda m, c: localize(m, c),
            ),
            ("pyscf_pipek", lambda m, c: lo.PipekMezey(m, c).kernel()),
            ("pyscfad_boys", lambda m, c: boys_localize(m, c)),
            ("identity", lambda _m, c: c),
        ]

    last_err = None
    for name, fn in localizers:
        try:
            coeff_in = orbocc if name == getattr(localize, "__name__", None) else orbocc_np
            lo_coeff = fn(mol, coeff_in)
            if name != getattr(localize, "__name__", None):
                print(f"Localized orbitals generated with fallback localizer: {name}")
            break
        except Exception as err:
            last_err = err
            print(f"Localizer {name} failed: {err}")
    else:
        raise RuntimeError("All orbital-localization fallbacks failed.") from last_err

    np.save(lo_coeff_file, lo_coeff)
    return lo_coeff


def load_or_run_mp2(mf, frozen, mp2_ecorr_file, mp2_factory=None, kernel_kwargs=None):
    if mp2_ecorr_file.exists():
        print(f"Loading MP2 correlation energy from {mp2_ecorr_file}")
        return float(np.load(mp2_ecorr_file, allow_pickle=False))

    print(f"Running MP2 and saving correlation energy to {mp2_ecorr_file}")
    if mp2_factory is None:
        mmp = mp.MP2(mf, frozen=frozen)
    else:
        mmp = mp2_factory(mf, frozen)
    if kernel_kwargs is None:
        kernel_kwargs = {}
    else:
        kernel_kwargs = dict(kernel_kwargs)
    if mp2_factory is None and "with_t2" not in kernel_kwargs:
        kernel_kwargs["with_t2"] = False
    mmp.kernel(**kernel_kwargs)
    e_corr = float(mmp.e_corr)
    np.save(mp2_ecorr_file, np.asarray(e_corr))
    return e_corr
