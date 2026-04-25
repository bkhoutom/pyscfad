"""Compare LNO- and DLNO-CCSD(T) gradients with local MP2 corrections."""

from pathlib import Path
import tempfile
import warnings

import jax
import numpy as np

from pyscfad import config, gto, mp, scf
from pyscfad.df import addons as df_addons
from pyscfad.df import incore as df_incore
from pyscfad.lno import LNOCCSD, LNOMP2
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

for key, value in {
    "pyscfad_moleintor_opt": True,
    "pyscfad_scf_implicit_diff": True,
    "pyscfad_scf_first_order_custom": True,
    "pyscfad_ccsd_implicit_diff": True,
}.items():
    config.update(key, value)


BASIS = "def2-svp"
THRESH = 1e-5
LO_TYPE = "iao"
DISK_DF_MAX_MEMORY = 1
atom = """
O 0.000000000000 0.000000000000 0.000000000000
H 0.756950327264 0.000000000000 0.585882276618
H -0.756950327264 0.000000000000 0.585882276618
O 4.000000000000 0.000000000000 0.000000000000
H 4.756950327264 0.000000000000 0.585882276618
H 3.243049672736 0.000000000000 0.585882276618
O 8.000000000000 0.000000000000 0.000000000000
H 8.756950327264 0.000000000000 0.585882276618
H 7.243049672736 0.000000000000 0.585882276618
"""


def build_mol(atom_spec=atom):
    mol = gto.Mole(atom=atom_spec, basis=BASIS, verbose=2)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_rhf(
    mol,
    *,
    df_max_memory=None,
    cderi_file=None,
    read_cderi=False,
):
    mf = scf.RHF(mol).density_fit()
    default_df_max_memory = mf.with_df.max_memory
    if df_max_memory is not None:
        mf.with_df.max_memory = df_max_memory
    if cderi_file is not None:
        mf.with_df.incore = False
        mf.with_df._cderi_to_save = str(cderi_file)
        if read_cderi:
            if not Path(cderi_file).exists():
                raise FileNotFoundError(cderi_file)
            mf.with_df.auxmol = df_addons.make_auxmol(mol, mf.with_df.auxbasis)
            mf.with_df._cderi = df_incore.cholesky_eri(
                mol,
                auxmol=mf.with_df.auxmol,
                int3c=mol._add_suffix("int3c2e"),
                int2c=mol._add_suffix("int2c2e"),
                max_memory=max(mf.with_df.max_memory, 4096),
                verbose=0,
            )
            mf.with_df._prefer_cderi_to_save = True
    mf.kernel()
    if read_cderi:
        mf.with_df.max_memory = default_df_max_memory
        mf.with_df._prefer_cderi_to_save = False
    return mf


def build_scf_cderi_file(mol, cderi_file, *, df_max_memory=DISK_DF_MAX_MEMORY):
    mf = scf.RHF(mol).density_fit()
    mf.with_df.incore = False
    mf.with_df.max_memory = df_max_memory
    mf.with_df._cderi_to_save = str(cderi_file)
    mf.with_df.build()
    with df_addons.load(str(cderi_file), "j3c") as feri:
        return tuple(feri.shape)


def build_local_orbitals_and_fragments(mf, *, thresh=THRESH, lo_type=LO_TYPE):
    lo = LNOCCSD(mf, thresh=thresh, frozen=0).get_lo(lo_type=lo_type)
    frag_atoms = stop_trace(autofrag)(mf.mol)
    frag_los = stop_trace(map_lo_to_frag)(mf.mol, lo, frag_atoms, verbose=mf.mol.verbose)
    return lo, frag_los


def make_canonical_mp2_solver(mf):
    pt = mp.dfmp2.MP2(mf, frozen=0)
    pt.kernel(with_t2=False)
    return pt


def make_local_mp2_solver(mf, *, thresh=THRESH, lo_type=LO_TYPE):
    pt = LNOMP2(mf, thresh=thresh, frozen=0)
    pt.thresh_occ = pt.thresh_vir = thresh
    pt.lo_type, pt.no_type = lo_type, "ie"
    return pt


def make_cc_solver(mf, *, thresh=THRESH, lo_type=LO_TYPE, ccsd_t=True):
    cc = LNOCCSD(mf, thresh=thresh, frozen=0)
    cc.thresh_occ = cc.thresh_vir = thresh
    cc.lo_type, cc.no_type, cc.ccsd_t = lo_type, "ie", ccsd_t
    return cc


def build_dlno_topology(
    mf,
    lo,
    frag_los,
    *,
    domain_pao_thr=THRESH,
    pair_energy_thr=THRESH,
    lmo_bp_domain_thr=0.9,
    pao_bp_domain_thr=0.9,
    multipole_order=2,
):
    return stop_trace(build_dlno_prescreen_data)(
        mf,
        lo,
        frag_los,
        frozen=0,
        lmo_bp_domain_thr=lmo_bp_domain_thr,
        pao_bp_domain_thr=pao_bp_domain_thr,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
        multipole_order=multipole_order,
    )


def build_dlno_data(
    mf,
    lo,
    frag_los,
    *,
    topology=None,
    domain_pao_thr=THRESH,
    pair_energy_thr=THRESH,
    lmo_bp_domain_thr=0.9,
    pao_bp_domain_thr=0.9,
    multipole_order=2,
):
    if topology is None:
        topology = build_dlno_topology(
            mf,
            lo,
            frag_los,
            domain_pao_thr=domain_pao_thr,
            pair_energy_thr=pair_energy_thr,
            lmo_bp_domain_thr=lmo_bp_domain_thr,
            pao_bp_domain_thr=pao_bp_domain_thr,
            multipole_order=multipole_order,
        )
    return rebuild_dlno_prescreen_data(mf, lo, topology, frozen=0)


def enable_dlno_prescreen(solver, dlno_data):
    solver.use_dlno_prescreen = True
    solver.dlno_prescreen_data = dlno_data
    return solver


def lno_total_energy(mol, *, thresh=THRESH, ccsd_t=True):
    mf = run_rhf(mol)
    lo, _ = build_local_orbitals_and_fragments(mf, thresh=thresh)
    pt = make_canonical_mp2_solver(mf)
    cc = make_cc_solver(mf, thresh=thresh, ccsd_t=ccsd_t)
    cc.kernel(orbloc=lo)
    total = mf.e_tot + cc.e_corr_pt2corrected(pt.e_corr)
    return total, {
        "mp2_corr": pt.e_corr,
        "cc_pt2": cc.e_corr_pt2,
    }


def dlno_total_energy(
    mol,
    *,
    thresh=THRESH,
    ccsd_t=True,
    domain_pao_thr=THRESH,
    pair_energy_thr=THRESH,
    topology=None,
    df_max_memory=None,
    cderi_file=None,
):
    mf = run_rhf(
        mol,
        df_max_memory=df_max_memory,
        cderi_file=cderi_file,
        read_cderi=cderi_file is not None,
    )
    lo, frag_los = build_local_orbitals_and_fragments(mf, thresh=thresh)
    if topology is None:
        topology = build_dlno_topology(
            mf,
            lo,
            frag_los,
            domain_pao_thr=domain_pao_thr,
            pair_energy_thr=pair_energy_thr,
        )
    dlno_data = build_dlno_data(mf, lo, frag_los, topology=topology)

    pt = enable_dlno_prescreen(make_local_mp2_solver(mf, thresh=thresh), dlno_data)
    pt.kernel(frag_lolist=frag_los, orbloc=lo)
    cc = enable_dlno_prescreen(make_cc_solver(mf, thresh=thresh, ccsd_t=ccsd_t), dlno_data)
    cc.kernel(frag_lolist=frag_los, orbloc=lo)
    total = mf.e_tot + cc.e_corr_pt2corrected(pt.e_corr)
    return total, {
        "mp2_corr": pt.e_corr,
        "cc_pt2": cc.e_corr_pt2,
    }


if __name__ == "__main__":
    mol = build_mol()
    mf_ref = run_rhf(mol)
    lo_ref, frag_los_ref = build_local_orbitals_and_fragments(mf_ref)
    dlno_topology_ref = build_dlno_topology(mf_ref, lo_ref, frag_los_ref)

    def dlno_total_energy_fixed_topology(mm):
        return dlno_total_energy(mm, topology=dlno_topology_ref)

    (e_lno, info_lno), g_lno = jax.value_and_grad(lno_total_energy, has_aux=True)(mol)
    (e_dlno, info_dlno), g_dlno = jax.value_and_grad(
        dlno_total_energy_fixed_topology,
        has_aux=True,
    )(mol)

    with tempfile.TemporaryDirectory(prefix="pyscfad-disk-df-") as tmpdir:
        cderi_file = Path(tmpdir) / "scf_cderi.h5"
        cderi_shape = build_scf_cderi_file(mol, cderi_file)

        def dlno_total_energy_disk_df(mm):
            return dlno_total_energy(
                mm,
                topology=dlno_topology_ref,
                df_max_memory=DISK_DF_MAX_MEMORY,
                cderi_file=cderi_file,
            )

        (e_dlno_disk, info_dlno_disk), g_dlno_disk = jax.value_and_grad(
            dlno_total_energy_disk_df,
            has_aux=True,
        )(mol)

    g_lno_arr = np.asarray(g_lno.coords)
    g_dlno_arr = np.asarray(g_dlno.coords)
    g_dlno_disk_arr = np.asarray(g_dlno_disk.coords)
    g_diff = g_dlno_arr - g_lno_arr
    g_disk_diff = g_dlno_disk_arr - g_dlno_arr

    print("Canonical MP2 correction:", float(info_lno["mp2_corr"]))
    print("DLNO MP2 correction:", float(info_dlno["mp2_corr"]))
    print("Disk-backed DLNO MP2 correction:", float(info_dlno_disk["mp2_corr"]))
    print("MP2 correction difference (DLNO - canonical):", float(info_dlno["mp2_corr"] - info_lno["mp2_corr"]))
    print(
        "MP2 correction difference (disk DLNO - in-core DLNO):",
        float(info_dlno_disk["mp2_corr"] - info_dlno["mp2_corr"]),
    )
    print("LNO CCSD(T) internal PT2 term:", float(info_lno["cc_pt2"]))
    print("DLNO CCSD(T) internal PT2 term:", float(info_dlno["cc_pt2"]))
    print("Disk-backed DLNO CCSD(T) internal PT2 term:", float(info_dlno_disk["cc_pt2"]))
    print("LNO-CCSD(T) (canonical-MP2-corrected) energy:", float(e_lno))
    print("LNO-CCSD(T) (canonical-MP2-corrected) gradient:")
    print(g_lno.coords)
    print("DLNO-CCSD(T) (DLNO-MP2-corrected) energy:", float(e_dlno))
    print("DLNO-CCSD(T) (DLNO-MP2-corrected) gradient:")
    print(g_dlno.coords)
    print(
        f"Disk-backed SCF cderi shape with max_memory={DISK_DF_MAX_MEMORY} MB:",
        cderi_shape,
    )
    print("DLNO-CCSD(T) disk-SCF-DF (DLNO-MP2-corrected) energy:", float(e_dlno_disk))
    print("DLNO-CCSD(T) disk-SCF-DF (DLNO-MP2-corrected) gradient:")
    print(g_dlno_disk.coords)
    print("Energy difference (DLNO - LNO):", float(e_dlno - e_lno))
    print("Max |gradient difference|:", float(np.max(np.abs(g_diff))))
    print("RMS |gradient difference|:", float(np.sqrt(np.mean(g_diff**2))))
    print("Energy difference (disk DLNO - in-core DLNO):", float(e_dlno_disk - e_dlno))
    print("Max |disk gradient difference|:", float(np.max(np.abs(g_disk_diff))))
    print("RMS |disk gradient difference|:", float(np.sqrt(np.mean(g_disk_diff**2))))
