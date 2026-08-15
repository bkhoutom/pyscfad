import warnings

import jax
import jax.numpy as jnp
import numpy

from pyscfad import config, df, gto, scf
from pyscfad.dlno import ccsd as dlno_ccsd
from pyscfad.lno import ccsd as lnoccsd, lno_base
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)
config.update('pyscfad_ccsd_implicit_diff', True)
config.update('pyscfad_dfccsd_custom_response', True)


def _build_water():
    mol = gto.Mole(
        atom='''
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        ''',
        basis='sto3g',
        verbose=0,
        max_memory=4000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _prepare_outcore_cderi(mol, cderi_file):
    with_df = df.DF(mol, incore=False)
    with_df._cderi_to_save = str(cderi_file)
    with_df.max_memory = mol.max_memory
    with_df.build()


def _density_fit_from_cderi(mol, cderi_file):
    mf = scf.RHF(mol)
    with_df = df.DF(mol, incore=False)
    with_df.max_memory = mol.max_memory
    with_df.auxmol = df.addons.make_auxmol(mol, with_df.auxbasis)
    with_df._cderi_to_save = str(cderi_file)
    with_df._cderi = numpy.zeros((0, 0))
    with_df._prefer_cderi_to_save = True
    return mf.density_fit(with_df=with_df)


def _build_local_orbitals_and_fragments(mf, thresh):
    lo_coeff = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=0).get_lo(lo_type='iao')
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def _make_solver(mf, thresh, dlno_data=None):
    if dlno_data is None:
        mycc = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=0)
    else:
        mycc = DLNOCCSD(mf, thresh=thresh, frozen=0,
                        dlno_prescreen_data=dlno_data)
    mycc.thresh_occ = thresh
    mycc.thresh_vir = thresh
    mycc.lo_type = 'iao'
    mycc.no_type = 'ie'
    mycc.ccsd_t = False
    return mycc


class _FakeEris:
    def __init__(self, ovov):
        self.ovov = ovov


class _ReportMol:
    nao = 7

    @staticmethod
    def aoslice_by_atom():
        return numpy.asarray([
            [0, 0, 0, 2],
            [0, 0, 2, 5],
            [0, 0, 5, 7],
        ])


def test_dlno_space_report_contains_domain_and_fragment_dimensions():
    prescreen_data = {
        'lmo_primary_domain': (
            numpy.asarray([0, 1]),
            numpy.asarray([2]),
        ),
        'fragment_data': (
            {
                'lo_indices': numpy.asarray([0]),
                'extended_primary_domain': numpy.asarray([0, 1]),
                'occ_prescreen_coeff': numpy.zeros((5, 2)),
                'vir_prescreen_coeff': numpy.zeros((5, 4)),
            },
            {
                'lo_indices': numpy.asarray([1]),
                'extended_primary_domain': numpy.asarray([2]),
                'occ_prescreen_coeff': numpy.zeros((2, 1)),
                'vir_prescreen_coeff': numpy.zeros((2, 3)),
            },
        ),
    }

    report = dlno_ccsd._format_dlno_space_report(
        _ReportMol(), prescreen_data
    )
    rows = [line.split() for line in report.splitlines()]

    assert 'Total atomic orbitals : 7' in report
    assert 'Orbital domains       : 2' in report
    assert 'Fragments             : 2' in report
    assert 'Occ vectors' in report
    assert 'Vir vectors' in report
    assert ['1', '2', '5'] in rows
    assert ['2', '1', '2'] in rows
    assert ['1', '2', '5', '1', '2', '4'] in rows
    assert ['2', '1', '2', '1', '1', '3'] in rows


def test_active_space_screening_report(capsys):
    fragment = {
        'extended_primary_domain': numpy.asarray([0, 1]),
        'occ_prescreen_coeff': numpy.zeros((5, 3)),
    }

    lno_base._print_active_space_screening(
        'Fragment 1/2', fragment, 2,
        prescreen_nocc=8, prescreen_nvir=20,
        screened_nocc=6, screened_nvir=14,
    )
    output = capsys.readouterr().out

    assert 'Fragment 1/2 active-space screening' in output
    assert 'Domain       : 2 atoms / 5 AOs; fragment LOs = 2' in output
    assert 'Prescreened  : 8 occ / 20 vir (28 MOs)' in output
    assert 'PNO-screened : 6 occ / 14 vir (20 MOs)' in output


def test_projected_sos_fragment_energy_matches_direct_os_term():
    ovov = jnp.asarray(
        numpy.arange(3 * 2 * 3 * 2, dtype=float).reshape(3, 2, 3, 2) / 23.0
        - 0.4
    )
    t2 = jnp.asarray(
        numpy.arange(3 * 3 * 2 * 2, dtype=float).reshape(3, 3, 2, 2) / 19.0
        - 0.7
    )
    prj = jnp.asarray([
        [0.8, 0.3, -0.2],
        [0.1, 0.5, 0.4],
    ])
    c_os = 1.3

    m = prj.T @ prj
    direct = jnp.einsum('pq,pjab,qajb->', m, t2, ovov)
    exchange = jnp.einsum('pq,pjab,qbja->', m, t2, ovov)

    e_sos = lnoccsd._sos_mp2_fragment_energy_jax(ovov, t2, prj, c_os)
    e_full = lnoccsd.mp2_fragment_energy(_FakeEris(ovov), t2, prj)

    assert abs(e_sos - c_os * direct) < 1e-12
    assert abs(e_full - (2.0 * direct - exchange)) < 1e-12
    assert abs(e_full - e_sos / c_os) > 1e-6

    def energy_t2(t2_):
        return lnoccsd._sos_mp2_fragment_energy_jax(ovov, t2_, prj, c_os)

    grad_t2 = jax.grad(energy_t2)(t2)
    idx = (1, 2, 0, 1)
    eps = 1e-5
    fd = (
        energy_t2(t2.at[idx].add(eps))
        - energy_t2(t2.at[idx].add(-eps))
    ) / (2.0 * eps)
    assert abs(grad_t2[idx] - fd) < 1e-7


def _build_static_topology(mol, thresh, cderi_file):
    mf = _density_fit_from_cderi(mol, cderi_file)
    mf.kernel()
    lo_coeff, frag_lolist = _build_local_orbitals_and_fragments(mf, thresh)
    topology = build_dlno_prescreen_data(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.0,
        pao_bp_domain_thr=0.0,
        domain_pao_thr=0.0,
        pair_energy_thr=0.0,
        multipole_order=2,
    )
    return frag_lolist, topology


def test_full_scope_mp2_correction_is_method_consistent_for_full_domains(tmp_path):
    mol = _build_water()
    thresh = 0.0
    cderi_file = tmp_path / 'water-cderi-full-scope.h5'
    _prepare_outcore_cderi(mol, cderi_file)
    frag_lolist, _ = _build_static_topology(mol, thresh, cderi_file)

    def build_mf(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        mf.kernel()
        return mf

    common_kwargs = dict(
        build_mf=build_mf,
        frag_lolist=frag_lolist,
        mp2_correction_scope='full',
        frozen=0,
        thresh_occ=thresh,
        thresh_vir=thresh,
        lo_type='iao',
        no_type='ie',
        ccsd_t=False,
        lmo_bp_domain_thr=0.0,
        pao_bp_domain_thr=0.0,
        domain_pao_thr=0.0,
        pair_energy_thr=0.0,
        multipole_order=2,
    )
    e_uncorrected, _ = DLNOCCSD.value_and_grad(
        mol, include_mp2_correction=False, **common_kwargs
    )

    for method in ('mp2', 'sos'):
        e_corrected, _ = DLNOCCSD.value_and_grad(
            mol,
            include_mp2_correction=True,
            mp2_correction_method=method,
            **common_kwargs,
        )
        assert abs(e_corrected - e_uncorrected) < 1e-6


def test_dlno_ccsd_gradient_matches_parent_lno_for_full_domains(tmp_path):
    mol = _build_water()
    thresh = 0.0
    cderi_file = tmp_path / 'water-cderi.h5'
    _prepare_outcore_cderi(mol, cderi_file)
    static_frag_lolist, static_topology = _build_static_topology(
        mol, thresh, cderi_file
    )

    def lno_energy(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        ehf = mf.kernel()
        lo_coeff, _ = _build_local_orbitals_and_fragments(mf, thresh)
        mycc = _make_solver(mf, thresh)
        mycc.kernel(orbloc=lo_coeff)
        return ehf + mycc.e_corr

    def dlno_energy(mol_):
        mf = _density_fit_from_cderi(mol_, cderi_file)
        ehf = mf.kernel()
        lo_coeff, _ = _build_local_orbitals_and_fragments(mf, thresh)
        dlno_data = rebuild_dlno_prescreen_data(
            mf, lo_coeff, static_topology, frozen=0
        )
        mycc = _make_solver(mf, thresh, dlno_data=dlno_data)
        mycc.kernel(frag_lolist=static_frag_lolist, orbloc=lo_coeff)
        return ehf + mycc.e_corr

    e_lno, g_lno = jax.value_and_grad(lno_energy)(mol)
    e_dlno, g_dlno = jax.value_and_grad(dlno_energy)(mol)

    assert abs(e_dlno - e_lno) < 1e-6
    assert numpy.max(numpy.abs(g_dlno.coords - g_lno.coords)) < 1e-6
