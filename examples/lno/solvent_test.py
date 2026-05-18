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

from pyscfad import config, df, gto, scf
from pyscfad.lno import LNOCCSD, LNOMP2
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.lno import lno_base


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
CCSD_LNO_OCC_THR = 1e-4
CCSD_LNO_VIR_THR = CCSD_LNO_OCC_THR / 10.0
MP2_LNO_OCC_THR = 1e-5
MP2_LNO_VIR_THR = MP2_LNO_OCC_THR / 10.0
DOMAIN_THR = 1e-4
DF_INCORE_SAFETY = 0.8
# Increase this value, or set WATER_GRID_NMAX in the environment, to sweep
# 1x1x1, 2x2x2, ..., NMAX x NMAX x NMAX water clusters.
NMAX = int(os.environ.get("WATER_GRID_NMAX", "4"))
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
        print(f"  {'Static+forward subtotal':<{width}} : {self.total():8.3f} s")


@contextmanager
def timed_section(timings, name):
    if timings is None:
        yield
    else:
        with timings.section(name):
            yield


def verbose_enabled(obj, level=2):
    try:
        return int(getattr(obj, "verbose", 0)) >= level
    except (TypeError, ValueError):
        return False


def print_phase(mol, title):
    if verbose_enabled(mol, 2):
        print()
        print(title)
        print("-" * len(title))


def print_scalar(mol, label, value):
    if not verbose_enabled(mol, 2):
        return
    try:
        print(f"  {label}: {float(value): .12f}")
    except Exception:
        jax.debug.print(f"  {label}: {{value: .12f}}", value=value)


def atom_count(atoms):
    return int(np.asarray(atoms, dtype=np.int32).size)


def ao_count_for_atoms(mol, atoms):
    atoms = np.asarray(atoms, dtype=np.int32).ravel()
    if atoms.size == 0:
        return 0
    aoslices = mol.aoslice_by_atom()[:, 2:]
    return int(sum(stop - start for start, stop in aoslices[atoms]))


def print_topology_profile(mol, data):
    rows = data.get("topology_profile") or ()
    if not rows:
        return
    print("  topology build timings:")
    for row in rows:
        print(f"    {row['section']:<28} {row['wall_s']:8.3f} s")


def print_threshold_summary(mol):
    if not verbose_enabled(mol, 2):
        return
    print()
    print("DLNO Thresholds")
    print("---------------")
    print(
        "  "
        f"CCSD LNO occ/vir = {CCSD_LNO_OCC_THR:.1e}/"
        f"{CCSD_LNO_VIR_THR:.1e}"
    )
    print(
        "  "
        f"MP2  LNO occ/vir = {MP2_LNO_OCC_THR:.1e}/"
        f"{MP2_LNO_VIR_THR:.1e}"
    )
    print(f"  domain/pair threshold = {DOMAIN_THR:.1e}")


def print_dlno_domain_summary(mol, data, title, include_timing=False):
    if not verbose_enabled(mol, 2):
        return

    fragments = data.get("fragment_data", ())
    print_phase(mol, title)
    print(f"  number of domains/fragments: {len(fragments)}")
    if fragments:
        print(
            "  "
            f"{'frag':>4} {'LOs':>5} {'strong':>7} "
            f"{'atoms':>7} {'AOs':>6} {'occ':>6} {'vir':>6}"
        )
    for frag in fragments:
        ifrag = int(frag.get("fragment_index", 0))
        loidx = frag.get("lo_indices", ())
        strong = frag.get("strong_lmo_indices", ())
        domain = frag.get("extended_primary_domain", ())
        occ = frag.get("occ_prescreen_coeff")
        vir = frag.get("vir_prescreen_coeff")
        nocc = 0 if occ is None else int(occ.shape[1])
        nvir = 0 if vir is None else int(vir.shape[1])
        print(
            "  "
            f"{ifrag:4d} {len(loidx):5d} {len(strong):7d} "
            f"{atom_count(domain):7d} {ao_count_for_atoms(mol, domain):6d} "
            f"{nocc:6d} {nvir:6d}"
        )
    if include_timing:
        print_topology_profile(mol, data)


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


def configure_lno_solver(solver, *, occ_thr, vir_thr, lo_type=LO_TYPE):
    solver.thresh_occ = occ_thr
    solver.thresh_vir = vir_thr
    solver.lo_type, solver.no_type = lo_type, "ie"
    return solver


def make_mp2_solver(mf, *, occ_thr=MP2_LNO_OCC_THR,
                    vir_thr=MP2_LNO_VIR_THR, lo_type=LO_TYPE):
    return configure_lno_solver(
        LNOMP2(mf, thresh=occ_thr, frozen=0),
        occ_thr=occ_thr,
        vir_thr=vir_thr,
        lo_type=lo_type,
    )


def make_cc_solver(mf, *, occ_thr=CCSD_LNO_OCC_THR,
                   vir_thr=CCSD_LNO_VIR_THR, lo_type=LO_TYPE,
                   ccsd_t=False):
    cc = configure_lno_solver(
        LNOCCSD(mf, thresh=occ_thr, frozen=0),
        occ_thr=occ_thr,
        vir_thr=vir_thr,
        lo_type=lo_type,
    )
    cc.ccsd_t = ccsd_t
    return cc


def build_local_orbitals_and_fragments(mf):
    # The LNO threshold is not used by get_lo; this object just provides the
    # localization helper used by the LNO code path.
    lo_coeff = lnoccsd.LNOCCSD(
        mf, thresh=CCSD_LNO_OCC_THR, frozen=0
    ).get_lo(lo_type=LO_TYPE)
    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def build_local_orbitals(mf):
    return lnoccsd.LNOCCSD(
        mf, thresh=CCSD_LNO_OCC_THR, frozen=0
    ).get_lo(lo_type=LO_TYPE)


def build_dlno_topology(mf, lo_coeff, frag_lolist):
    return build_dlno_prescreen_data(
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


def build_dlno_data(mf, lo_coeff, frag_lolist=None, topology=None):
    if topology is None:
        if frag_lolist is None:
            raise ValueError("frag_lolist is required when topology is not supplied")
        topology = build_dlno_topology(mf, lo_coeff, frag_lolist)
    return rebuild_dlno_prescreen_data(mf, lo_coeff, topology, frozen=0)


def precompute_static_dlno_inputs(mol, cderi_file=None, timings=None):
    with timed_section(timings, "Static DLNO topology"):
        print_threshold_summary(mol)
        print_phase(mol, "SCF")
        mf = make_density_fitted_rhf(
            mol,
            cderi_file=cderi_file,
            cderi_ready=cderi_file is not None,
        )
        ehf = mf.kernel()
        print_scalar(mol, "SCF energy", ehf)
        lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf)
        topology = build_dlno_topology(mf, lo_coeff, frag_lolist)
        print_dlno_domain_summary(
            mol, topology, "DLNO Domains", include_timing=True
        )
    return frag_lolist, topology

def dlno_total_energy(mol, timings=None, cderi_file=None, static_inputs=None):
    if static_inputs is None:
        static_frag_lolist = None
        static_topology = None
    else:
        static_frag_lolist, static_topology = static_inputs
    report_setup = static_inputs is None

    with timed_section(timings, "SCF"):
        if report_setup:
            print_phase(mol, "SCF")
        mf = make_density_fitted_rhf(
            mol,
            cderi_file=cderi_file,
            cderi_ready=cderi_file is not None,
        )
        ehf = mf.kernel()
        if report_setup:
            print_scalar(mol, "SCF energy", ehf)

    with timed_section(timings, "Local orbitals/fragments"):
        if report_setup:
            print_phase(mol, "Local Orbitals")
        if static_frag_lolist is None:
            lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf)
        else:
            lo_coeff = build_local_orbitals(mf)
            frag_lolist = static_frag_lolist
        if report_setup and verbose_enabled(mol, 2):
            print(f"  localized occupied orbitals: {lo_coeff.shape[1]}")
            print(f"  fragments: {len(frag_lolist)}")

    with timed_section(timings, "DLNO prescreen"):
        dlno_data = build_dlno_data(
            mf,
            lo_coeff,
            frag_lolist=frag_lolist,
            topology=static_topology,
        )
        if report_setup:
            print_dlno_domain_summary(mol, dlno_data, "DLNO Domains")

    with timed_section(timings, "DLNO-MP2"):
        print_phase(mol, "Forward DLNO-MP2 Fragments")
        pt = make_mp2_solver(mf, lo_type=LO_TYPE)
        pt.use_dlno_prescreen = True
        pt.dlno_prescreen_data = dlno_data
        pt.profile_fragments = True
        pt.profile_label = "DLNO-MP2"
        pt.profile_pass = "forward"
        pt.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
        print_scalar(mol, "DLNO-MP2 correlation energy", pt.e_corr)

    with timed_section(timings, "DLNO-CCSD(T)"):
        print_phase(mol, "Forward DLNO-CCSD(T) Fragments")
        mycc = make_cc_solver(mf, lo_type=LO_TYPE, ccsd_t=True)
        mycc.use_dlno_prescreen = True
        mycc.dlno_prescreen_data = dlno_data
        mycc.profile_fragments = True
        mycc.profile_label = "DLNO-CCSD(T)"
        mycc.profile_pass = "forward"
        mycc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
        print_scalar(mol, "DLNO fragment MP2 correlation", mycc.e_corr_pt2)
        print_scalar(mol, "DLNO fragment CCSD correlation", mycc.e_corr_ccsd)
        print_scalar(mol, "DLNO fragment (T) correction", mycc.e_corr_ccsd_t)
    return ehf + mycc.e_corr_pt2corrected(pt.e_corr)


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
        static_inputs = precompute_static_dlno_inputs(
            mol,
            cderi_file=cderi_file,
            timings=timings,
        )
        lno_base.clear_fragment_profile()
        e_dlno, g_dlno = jax.value_and_grad(
            lambda m: dlno_total_energy(
                m,
                timings=timings,
                cderi_file=cderi_file,
                static_inputs=static_inputs,
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
    print("DLNO-CCSD(T) gradient summary")
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
        "mp2_s": timings.elapsed("DLNO-MP2"),
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
