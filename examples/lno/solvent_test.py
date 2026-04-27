"""Time DLNO-CCSD(T) gradients for growing cubic water clusters."""

from contextlib import contextmanager
import csv
import os
from pathlib import Path
import tempfile
import time
import warnings

import jax
import numpy as np

from pyscfad import config, df, gto, mp, scf
from pyscfad.lno import LNOCCSD
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace
from pyscfad.lno import ccsd as lnoccsd


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)

for key, value in {
    "pyscfad_moleintor_opt": True,
    "pyscfad_scf_implicit_diff": True,
    "pyscfad_scf_first_order_custom": True,
    "pyscfad_ccsd_implicit_diff": True,
    "pyscfad_dfccsd_custom_response": True,
}.items():
    config.update(key, value)


BASIS = "def2-svp"
LO_TYPE = "iao"  # Change to "pm" or "boys" to try other localization types.
BUILD_THR = 1e-4
FINAL_THR = 1e-5
DOMAIN_THR = 1e-4
DF_INCORE_SAFETY = 0.8
# Increase this value, or set WATER_GRID_NMAX in the environment, to sweep
# 1x1x1, 2x2x2, ..., NMAX x NMAX x NMAX water clusters.
NMAX = 4
WATER_SPACING_ANG = 3.5
CSV_PATH = Path(
    os.environ.get(
        "SOLVENT_TIMING_CSV",
        Path(__file__).with_name("solvent_test_timings.csv"),
    )
)

CSV_COLUMNS = [
    "N",
    "n_waters",
    "n_atoms",
    "energy",
    "scf_s",
    "local_orbitals_fragments_s",
    "dlno_prescreen_s",
    "mp2_s",
    "dlno_ccsd_t_s",
    "forward_subtotal_s",
    "gradient_evaluation_total_s",
    "ad_backward_residual_s",
]


class TimingBreakdown:
    def __init__(self):
        self.rows = []

    @contextmanager
    def section(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.rows.append((name, time.perf_counter() - start))

    def total(self):
        return sum(elapsed for _, elapsed in self.rows)

    def elapsed(self, name):
        return sum(elapsed for section, elapsed in self.rows if section == name)

    def print_summary(self):
        print()
        print("Timing breakdown")
        width = max((len(name) for name, _ in self.rows), default=0)
        for name, elapsed in self.rows:
            print(f"  {name:<{width}} : {elapsed:8.3f} s")
        print(f"  {'Forward subtotal':<{width}} : {self.total():8.3f} s")


@contextmanager
def timed_section(timings, name):
    if timings is None:
        yield
    else:
        with timings.section(name):
            yield


def read_water_monomer():
    xyz_path = Path(__file__).with_name("water_dimer.xyz")
    atom_lines = xyz_path.read_text().splitlines()[2:5]
    monomer = []
    for line in atom_lines:
        symbol, *xyz = line.split()
        monomer.append((symbol, np.asarray(xyz, dtype=float)))

    origin = monomer[0][1]
    return [(symbol, coords - origin) for symbol, coords in monomer]


def build_water_grid_atom(grid_n):
    monomer = read_water_monomer()
    atom = []
    for ix in range(grid_n):
        for iy in range(grid_n):
            for iz in range(grid_n):
                shift = WATER_SPACING_ANG * np.asarray((ix, iy, iz), dtype=float)
                atom.extend(
                    (symbol, tuple(coords + shift))
                    for symbol, coords in monomer
                )
    return atom


def build_mol(grid_n):
    atom = build_water_grid_atom(grid_n)
    mol = gto.Mole(
        atom=atom,
        basis=BASIS,
        unit="Angstrom",
        verbose=2,
        max_memory=4000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def estimate_packed_df_size_mb(mol, auxmol):
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    nao_pair = nao * (nao + 1) // 2
    return naux * nao_pair * 8 / 1e6


def make_density_fitted_rhf(mol, cderi_file=None, cderi_ready=False):
    mf = scf.RHF(mol)
    with_df = df.DF(mol)
    with_df.max_memory = mol.max_memory
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose
    with_df.auxmol = df.addons.make_auxmol(mol, with_df.auxbasis)

    packed_df_size_mb = estimate_packed_df_size_mb(mol, with_df.auxmol)
    with_df.incore = packed_df_size_mb < DF_INCORE_SAFETY * mol.max_memory
    if not with_df.incore:
        print(
            "Using out-of-core DF for SCF: "
            f"packed cderi estimate = {packed_df_size_mb / 1024:.2f} GiB, "
            f"max_memory = {mol.max_memory / 1024:.2f} GiB",
            flush=True,
        )
    if not with_df.incore and cderi_file is not None:
        with_df._cderi_to_save = str(cderi_file)
        if cderi_ready:
            with_df._cderi = np.zeros((0, 0))
            with_df._prefer_cderi_to_save = True
    return mf.density_fit(with_df=with_df)


def prepare_outcore_cderi(mol, timings=None):
    mf = make_density_fitted_rhf(mol)
    if mf.with_df.incore:
        return None

    fd, filename = tempfile.mkstemp(prefix="pyscfad-solvent-cderi-", suffix=".h5")
    os.close(fd)
    cderi_file = Path(filename)
    try:
        with timed_section(timings, "SCF DF build"):
            mf.with_df._cderi_to_save = str(cderi_file)
            mf.with_df.build()
    except Exception:
        cderi_file.unlink(missing_ok=True)
        raise
    return cderi_file


def elapsed_scf(timings):
    return timings.elapsed("SCF DF build") + timings.elapsed("SCF")


def make_cc_solver(mf, *, thresh, lo_type=LO_TYPE, ccsd_t=False):
    cc = LNOCCSD(mf, thresh=thresh, frozen=0)
    cc.thresh_occ = cc.thresh_vir = thresh
    cc.lo_type, cc.no_type, cc.ccsd_t = lo_type, "ie", ccsd_t
    return cc


def build_local_orbitals_and_fragments(mf):
    lo_coeff = lnoccsd.LNOCCSD(mf, thresh=BUILD_THR, frozen=0).get_lo(lo_type=LO_TYPE)
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def build_dlno_data(mf, lo_coeff, frag_lolist):
    topology = stop_trace(build_dlno_prescreen_data)(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=0,
        lmo_bp_domain_thr=0.9,
        pao_bp_domain_thr=0.9,
        domain_pao_thr=DOMAIN_THR,
        pair_energy_thr=DOMAIN_THR,
        multipole_order=2,
    )
    return rebuild_dlno_prescreen_data(mf, lo_coeff, topology, frozen=0)

def make_canonical_mp2_solver(mf):
    pt = mp.dfmp2.MP2(mf, frozen=0)
    pt.kernel(with_t2=False)
    return pt

def dlno_total_energy(mol, timings=None, cderi_file=None):
    with timed_section(timings, "SCF"):
        mf = make_density_fitted_rhf(
            mol,
            cderi_file=cderi_file,
            cderi_ready=cderi_file is not None,
        )
        ehf = mf.kernel()

    with timed_section(timings, "Local orbitals/fragments"):
        lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf)

    with timed_section(timings, "DLNO prescreen"):
        dlno_data = build_dlno_data(mf, lo_coeff, frag_lolist)

    with timed_section(timings, "MP2"):
        mp = make_canonical_mp2_solver(mf)

    with timed_section(timings, "DLNO-CCSD(T)"):
        mycc = make_cc_solver(mf, thresh=FINAL_THR, lo_type=LO_TYPE, ccsd_t=True)
        mycc.use_dlno_prescreen = True
        mycc.dlno_prescreen_data = dlno_data
        mycc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
    return ehf + mycc.e_corr_pt2corrected(mp.e_corr)


def append_csv_row(row):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def run_grid_point(grid_n):
    print()
    print(
        f"Running water grid {grid_n} x {grid_n} x {grid_n} "
        f"with spacing {WATER_SPACING_ANG:.2f} Angstrom",
        flush=True,
    )

    mol = build_mol(grid_n)
    timings = TimingBreakdown()
    grad_start = time.perf_counter()
    cderi_file = prepare_outcore_cderi(mol, timings=timings)
    try:
        e_dlno, g_dlno = jax.value_and_grad(
            lambda m: dlno_total_energy(
                m,
                timings=timings,
                cderi_file=cderi_file,
            )
        )(mol)
        grad_elapsed = time.perf_counter() - grad_start
    finally:
        if cderi_file is not None:
            cderi_file.unlink(missing_ok=True)

    g_dlno_arr = np.asarray(g_dlno.coords)
    forward_subtotal = timings.total()
    ad_backward_residual = grad_elapsed - forward_subtotal

    print()
    print("Testing DLNO-prescreened CCSD with the custom first-order CPHF SCF backend,")
    print("and comparing both LNO and DLNO against canonical DF-CCSD on the same system.")
    print()
    print("SCF backend: cphf")
    print(f"LO type: {LO_TYPE}")
    print(
        f"Water grid: {grid_n} x {grid_n} x {grid_n}, "
        f"spacing = {WATER_SPACING_ANG:.2f} Angstrom, atoms = {mol.natm}"
    )
    print(f"DLNO-prescreen total energy:{float(e_dlno): .12f}")
    print("DLNO-prescreen gradient:\n", g_dlno_arr)
    timings.print_summary()
    print(f"  {'Gradient evaluation total':<24} : {grad_elapsed:8.3f} s")
    print(
        f"  {'AD/backward residual':<24} : "
        f"{ad_backward_residual:8.3f} s"
    )

    row = {
        "N": grid_n,
        "n_waters": grid_n**3,
        "n_atoms": mol.natm,
        "energy": float(e_dlno),
        "scf_s": elapsed_scf(timings),
        "local_orbitals_fragments_s": timings.elapsed("Local orbitals/fragments"),
        "dlno_prescreen_s": timings.elapsed("DLNO prescreen"),
        "mp2_s": timings.elapsed("MP2"),
        "dlno_ccsd_t_s": timings.elapsed("DLNO-CCSD(T)"),
        "forward_subtotal_s": forward_subtotal,
        "gradient_evaluation_total_s": grad_elapsed,
        "ad_backward_residual_s": ad_backward_residual,
    }
    append_csv_row(row)
    print(f"Appended timing row to {CSV_PATH}")
    return row



if __name__ == "__main__":
    print(f"Writing timing sweep rows to {CSV_PATH}")
    print(f"Sweeping N = 1 ... {NMAX}")
    for grid_n in range(1, NMAX + 1):
        run_grid_point(grid_n)
