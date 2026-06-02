import warnings

import jax
import numpy

from pyscfad import config, df, gto, scf
from pyscfad.lno import ccsd as lnoccsd
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


def _make_solver(mf, thresh):
    mycc = lnoccsd.LNOCCSD(mf, thresh=thresh, frozen=0)
    mycc.thresh_occ = thresh
    mycc.thresh_vir = thresh
    mycc.lo_type = 'iao'
    mycc.no_type = 'ie'
    mycc.ccsd_t = False
    return mycc


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
        mycc = _make_solver(mf, thresh)
        mycc.use_dlno_prescreen = True
        mycc.dlno_prescreen_data = dlno_data
        mycc.kernel(frag_lolist=static_frag_lolist, orbloc=lo_coeff)
        return ehf + mycc.e_corr

    e_lno, g_lno = jax.value_and_grad(lno_energy)(mol)
    e_dlno, g_dlno = jax.value_and_grad(dlno_energy)(mol)

    assert abs(e_dlno - e_lno) < 1e-6
    assert numpy.max(numpy.abs(g_dlno.coords - g_lno.coords)) < 1e-6
