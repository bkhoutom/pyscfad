"""Reusable workflow helpers for DLNO-prescreened LNO calculations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import threading
import time
import warnings

import jax
from mpi4py import MPI
import numpy as np

# lno_base reads this switch at import time. Force the low-memory custom VJP
# that replays one fragment at a time in the backward pass.
os.environ["PYSCFAD_LNO_FRAGMENT_REPLAY_VJP"] = "1"

from pyscfad import config, df, gto, mp, scf
from pyscfad.lno import ccsd as lnoccsd
from pyscfad.lno.ccsd_mpi import LNOCCSD as MPILNOCCSD
from pyscfad.lno.mp2_mpi import LNOMP2 as MPILNOMP2
from pyscfad.lno.prescreen import build_dlno_prescreen_data, rebuild_dlno_prescreen_data
from pyscfad.lno.tools import autofrag, map_lo_to_frag
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not JSON-serializable",
)


DEFAULT_CONFIG_FLAGS = {
    "pyscfad_moleintor_opt": True,
    "pyscfad_scf_implicit_diff": True,
    "pyscfad_scf_first_order_custom": True,
    "pyscfad_ccsd_implicit_diff": True,
    "pyscfad_dfccsd_custom_response": True,
}

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


@dataclass(frozen=True)
class CalculationSettings:
    basis: str = "def2-svp"
    lo_type: str = "iao"
    lno_occ_thr: float = 1.0e-5
    lno_vir_thr: float = 1.0e-5
    mp2_lno_occ_thr: float = 1.0e-7
    mp2_lno_vir_thr: float = 1.0e-7
    dlno_ccsd_domain_pao_thr: float = 1.0e-4
    dlno_ccsd_pair_energy_thr: float = 1.0e-4
    lmo_bp_domain_thr: float = 0.9
    pao_bp_domain_thr: float = 0.9
    multipole_order: int = 4
    max_memory_mb: int = 2000
    frozen_core: int = 0
    verbose: int = 4
    frag_lolist_mode: str = "auto"
    mp2_correction: str = "dlno"
    ccsd_t: bool = True
    low_memory_gradient: bool = True
    scf_init_guess: str = "minao"
    scf_chkfile: str | None = None

    def with_parameter(self, name: str, value):
        if not hasattr(self, name):
            valid = ", ".join(self.__dataclass_fields__)
            raise ValueError(f"Unknown setting {name!r}. Valid settings: {valid}")
        return replace(self, **{name: value})


def current_rss_mb():
    """Return current process RSS in MiB."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


class MemorySampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_mb = current_rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, current_rss_mb())

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, current_rss_mb())


class Timer:
    def __init__(self):
        self.rows = []

    @contextmanager
    def section(self, name, *, subtotal=True):
        start = time.perf_counter()
        start_mem = current_rss_mb()
        try:
            with MemorySampler() as sampler:
                yield
        finally:
            end_mem = current_rss_mb()
            elapsed = time.perf_counter() - start
            peak_mem = max(sampler.peak_mb, start_mem, end_mem)
            self.rows.append((name, elapsed, start_mem, end_mem, peak_mem, subtotal))

    def elapsed(self, name):
        return sum(elapsed for section, elapsed, *_ in self.rows if section == name)

    def subtotal(self):
        return sum(elapsed for _, elapsed, *rest in self.rows if rest[-1])

    def print_summary(self):
        print()
        print("Timing and memory breakdown")
        width = max(
            max((len(name) for name, *_ in self.rows), default=0),
            len("Forward subtotal"),
        )
        print(
            f"  {'Section':<{width}}   {'Time (s)':>8}   {'Before MB':>10}   "
            f"{'After MB':>10}   {'Delta MB':>10}   {'Peak MB':>10}"
        )

        total_rows = []
        for name, elapsed, start_mem, end_mem, peak_mem, subtotal in self.rows:
            if subtotal:
                self._print_row(width, name, elapsed, start_mem, end_mem, peak_mem)
            else:
                total_rows.append((name, elapsed, start_mem, end_mem, peak_mem))

        self._print_row(width, "Forward subtotal", self.subtotal(), None, None, None)
        for row in total_rows:
            self._print_row(width, *row)

    @staticmethod
    def _format_mem(value):
        if value is None:
            return ""
        return f"{value:10.1f}"

    @classmethod
    def _print_row(cls, width, name, elapsed, start_mem, end_mem, peak_mem):
        delta_mem = None if start_mem is None or end_mem is None else end_mem - start_mem
        print(
            f"  {name:<{width}}   {elapsed:8.3f}   "
            f"{cls._format_mem(start_mem):>10}   "
            f"{cls._format_mem(end_mem):>10}   "
            f"{cls._format_mem(delta_mem):>10}   "
            f"{cls._format_mem(peak_mem):>10}"
        )


@contextmanager
def timed_section(timer, name):
    if timer is None:
        yield
    else:
        with timer.section(name):
            yield


def configure_pyscfad(flags=None):
    for key, value in (flags or DEFAULT_CONFIG_FLAGS).items():
        config.update(key, value)


def build_mol(xyz_path, settings: CalculationSettings):
    xyz_path = Path(xyz_path).expanduser().resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"Expected molecule file at {xyz_path}")

    mol = gto.Mole(
        atom=str(xyz_path),
        basis=settings.basis,
        unit="Angstrom",
        verbose=settings.verbose if RANK == 0 else 0,
        max_memory=settings.max_memory_mb,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def system_metadata(mol, settings: CalculationSettings, frozen):
    nelec = tuple(int(x) for x in mol.nelec)
    nelectron = int(sum(nelec))
    nao = int(mol.nao_nr())
    nmo = nao
    nocc_total = nelectron // 2
    nvir_total = nmo - nocc_total

    return {
        "natm": int(mol.natm),
        "nelectron": nelectron,
        "nelec": nelec,
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "nao": nao,
        "nmo": nmo,
        "nocc_total": nocc_total,
        "nvir_total": nvir_total,
        "frozen_core": frozen,
        "basis": settings.basis,
        "lo_type": settings.lo_type,
    }


def lno_metadata(lo_coeff, frag_lolist=None, dlno_data=None):
    metadata = {"nlo": int(lo_coeff.shape[1])}
    if frag_lolist is not None:
        metadata["nfrag_total"] = len(frag_lolist)
    if dlno_data is not None:
        metadata["nfrag_rank"] = len(dlno_data.get("fragment_data", ()))
    return metadata


def _host_scalar(value):
    if value is None:
        return None
    arr = np.asarray(jax.device_get(value))
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr.tolist()


def finalize_fragment_diagnostics(rows, *, rank=RANK):
    out = []
    for row in rows or ():
        item = {}
        for key, value in row.items():
            if key.startswith("e_"):
                item[key] = _host_scalar(value)
            elif value is None:
                item[key] = None
            else:
                item[key] = int(value)
        item["rank"] = int(rank)
        out.append(item)
    return out


def gather_fragment_diagnostics(local_by_section):
    local = {
        section: finalize_fragment_diagnostics(rows)
        for section, rows in (local_by_section or {}).items()
    }
    gathered = COMM.gather(local, root=0)
    if RANK != 0:
        return None

    merged = {}
    for rank_data in gathered:
        for section, rows in rank_data.items():
            merged.setdefault(section, []).extend(rows)
    for rows in merged.values():
        rows.sort(key=lambda row: (row.get("fragment_index", -1), row.get("rank", -1)))
    return merged


def domain_fragment_diagnostics(dlno_data):
    rows = []
    for local_index, frag in enumerate(dlno_data.get("fragment_data", ())):
        rows.append(
            {
                "fragment_index": int(frag.get("fragment_index", local_index)),
                "local_fragment_index": int(local_index),
                "nlo_fragment": int(np.asarray(frag.get("lo_indices", ())).size),
                "dlno_strong_lmo_count": int(
                    np.asarray(frag.get("strong_lmo_indices", ())).size
                ),
                "dlno_extended_bp_atom_count": int(
                    np.asarray(frag.get("extended_bp_domain", ())).size
                ),
                "dlno_extended_primary_atom_count": int(
                    np.asarray(frag.get("extended_primary_domain", ())).size
                ),
                "dlno_occ_prescreen_size": int(
                    np.asarray(frag.get("occ_prescreen_coeff", np.zeros((0, 0)))).shape[1]
                ),
                "dlno_vir_prescreen_size": int(
                    np.asarray(frag.get("vir_prescreen_coeff", np.zeros((0, 0)))).shape[1]
                ),
            }
        )
    return rows


def print_system_metadata(metadata):
    print(
        "System: "
        f"natm={metadata['natm']}, "
        f"nelectron={metadata['nelectron']} "
        f"(alpha={metadata['nelec'][0]}, beta={metadata['nelec'][1]}), "
        f"charge={metadata['charge']}, spin={metadata['spin']}"
    )
    print(
        "Orbital space: "
        f"nao={metadata['nao']}, nmo={metadata['nmo']}, "
        f"nocc={metadata['nocc_total']}, nvir={metadata['nvir_total']}"
    )
    print(f"Frozen orbitals from input: {metadata['frozen_core']}")


def estimate_packed_df_size_mb(mol, auxmol):
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    nao_pair = nao * (nao + 1) // 2
    return naux * nao_pair * 8 / 1e6


def make_density_fitted_rhf(
    mol,
    *,
    cderi_file=None,
    cderi_ready=False,
    scf_init_guess=None,
    scf_chkfile=None,
):
    mf = scf.RHF(mol)
    with_df = df.DF(mol)
    with_df.max_memory = mol.max_memory
    with_df.stdout = mf.stdout
    with_df.verbose = mf.verbose
    with_df.auxmol = df.addons.make_auxmol(mol, with_df.auxbasis)
    with_df.incore = False

    packed_df_size_mb = estimate_packed_df_size_mb(mol, with_df.auxmol)
    if RANK == 0:
        print(
            "Using out-of-core DF for SCF: "
            f"packed cderi estimate = {packed_df_size_mb / 1024:.2f} GiB, "
            f"max_memory = {mol.max_memory / 1024:.2f} GiB",
            flush=True,
        )

    if cderi_file is not None:
        with_df._cderi_to_save = str(cderi_file)
        if cderi_ready:
            with_df._cderi = np.zeros((0, 0))
            with_df._prefer_cderi_to_save = True
    mf = mf.density_fit(with_df=with_df)
    if scf_init_guess is not None:
        mf.init_guess = scf_init_guess
    if scf_chkfile is not None:
        mf.chkfile = str(Path(scf_chkfile).expanduser())
    return mf

def prepare_outcore_cderi(mol, timer=None):
    fd, filename = tempfile.mkstemp(prefix="pyscfad-solvent-mpi-cderi-", suffix=".h5")
    os.close(fd)
    cderi_file = Path(filename)
    try:
        mf = make_density_fitted_rhf(mol, cderi_file=cderi_file)
        with timed_section(timer, "DF build"):
            mf.with_df.build()
    except Exception:
        cderi_file.unlink(missing_ok=True)
        raise
    return cderi_file


def prepare_shared_outcore_cderi(mol, timer=None):
    if RANK == 0:
        cderi_file = prepare_outcore_cderi(mol, timer=timer)
        cderi_name = str(cderi_file)
    else:
        cderi_name = None

    cderi_name = COMM.bcast(cderi_name, root=0)
    COMM.Barrier()
    return Path(cderi_name)


def build_local_orbitals_and_fragments(mf, frozen, settings: CalculationSettings):
    lo_coeff = lnoccsd.LNOCCSD(
        mf, thresh=min(settings.lno_occ_thr, settings.lno_vir_thr), frozen=frozen
    ).get_lo(lo_type=settings.lo_type)

    if settings.frag_lolist_mode == "1o":
        frag_lolist = stop_trace(lambda nlo: [[i] for i in range(nlo)])(
            lo_coeff.shape[1]
        )
        return lo_coeff, frag_lolist
    if settings.frag_lolist_mode != "auto":
        raise ValueError(
            "frag_lolist_mode must be 'auto' or '1o', "
            f"got {settings.frag_lolist_mode!r}"
        )

    frag_atmlist = stop_trace(autofrag)(mf.mol)
    frag_lolist = stop_trace(map_lo_to_frag)(
        mf.mol, lo_coeff, frag_atmlist, verbose=mf.mol.verbose
    )
    return lo_coeff, frag_lolist


def scf_reference_matrices(mf):
    s1e = mf.get_ovlp()
    dm = mf.make_rdm1()
    h1e = mf.get_hcore(mf.mol, s1e=s1e)
    vhf = mf.get_veff(mf.mol, dm, s1e=s1e)
    fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm)
    return s1e, fock


def build_dlno_data(
    mf,
    lo_coeff,
    frag_lolist,
    frozen,
    settings: CalculationSettings,
    *,
    domain_pao_thr,
    pair_energy_thr,
    s1e=None,
    fock=None,
):
    topology = stop_trace(build_dlno_prescreen_data)(
        mf,
        lo_coeff,
        frag_lolist,
        frozen=frozen,
        lmo_bp_domain_thr=settings.lmo_bp_domain_thr,
        pao_bp_domain_thr=settings.pao_bp_domain_thr,
        domain_pao_thr=domain_pao_thr,
        pair_energy_thr=pair_energy_thr,
        multipole_order=settings.multipole_order,
        s1e=s1e,
        fock=fock,
    )
    return rebuild_dlno_prescreen_data(
        mf, lo_coeff, topology, frozen=frozen, s1e=s1e, fock=fock)


def enable_dlno_prescreen(solver, dlno_data):
    solver.use_dlno_prescreen = True
    solver.dlno_prescreen_data = dlno_data
    return solver


def rank_fragment_indices(nfrag):
    return [ifrag for ifrag in range(nfrag) if ifrag % SIZE == RANK]


def filter_dlno_data_for_rank(dlno_data, frag_lolist):
    local_indices = rank_fragment_indices(len(frag_lolist))
    return {
        "fragment_data": [dlno_data["fragment_data"][i] for i in local_indices],
    }


def make_cc_solver(mf, frozen, settings: CalculationSettings, *, s1e=None, fock=None):
    cc = MPILNOCCSD(
        mf,
        thresh=min(settings.lno_occ_thr, settings.lno_vir_thr),
        frozen=frozen,
        fock=fock,
        s1e=s1e,
    )
    cc.thresh_occ = settings.lno_occ_thr
    cc.thresh_vir = settings.lno_vir_thr
    cc.lo_type = settings.lo_type
    cc.no_type = "ie"
    cc.ccsd_t = settings.ccsd_t
    return cc


def make_mp2_solver(mf, frozen, settings: CalculationSettings, *, s1e=None, fock=None):
    pt = MPILNOMP2(
        mf,
        thresh=min(settings.mp2_lno_occ_thr, settings.mp2_lno_vir_thr),
        frozen=frozen,
        fock=fock,
        s1e=s1e,
    )
    pt.thresh_occ = settings.mp2_lno_occ_thr
    pt.thresh_vir = settings.mp2_lno_vir_thr
    pt.lo_type = settings.lo_type
    pt.no_type = "ie"
    return pt


def make_canonical_mp2_solver(mf, frozen):
    pt = mp.dfmp2.MP2(mf, frozen=frozen)
    pt.kernel(with_t2=False)
    return pt


def run_mp2_correction(
    mf, frag_lolist, lo_coeff, frozen, settings, dlno_data, timer=None,
    *, s1e=None, fock=None
):
    mode = settings.mp2_correction.lower()
    if mode in ("dlno", "local"):
        with timed_section(timer, "DLNO-MP2 correction"):
            pt = enable_dlno_prescreen(
                make_mp2_solver(mf, frozen, settings, s1e=s1e, fock=fock),
                dlno_data,
            )
            pt.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)
            return pt.e_corr, getattr(pt, "fragment_diagnostics", [])

    if mode == "canonical":
        if RANK == 0:
            with timed_section(timer, "Canonical MP2 correction"):
                pt = make_canonical_mp2_solver(mf, frozen)
                return pt.e_corr, []
        return 0.0, []

    raise ValueError(
        "mp2_correction must be 'dlno' or 'canonical', "
        f"got {settings.mp2_correction!r}"
    )


def dlno_ccsd_t_with_mp2_correction(
    mol,
    settings: CalculationSettings,
    *,
    cderi_file=None,
    timer=None,
    return_metadata=False,
):
    ncore = settings.frozen_core
    if RANK == 0:
        print(f"Frozen core orbitals: ncore = {ncore}", flush=True)

    with timed_section(timer, "SCF"):
        mf = make_density_fitted_rhf(
            mol,
            cderi_file=cderi_file,
            cderi_ready=cderi_file is not None,
            scf_init_guess=settings.scf_init_guess,
            scf_chkfile=settings.scf_chkfile,
        )
        ehf = mf.kernel()
        s1e, fock = scf_reference_matrices(mf)

    with timed_section(timer, "Local orbitals/fragments"):
        lo_coeff, frag_lolist = build_local_orbitals_and_fragments(mf, ncore, settings)

    with timed_section(timer, "DLNO-CCSD domains"):
        ccsd_dlno_data = build_dlno_data(
            mf,
            lo_coeff,
            frag_lolist,
            ncore,
            settings,
            domain_pao_thr=settings.dlno_ccsd_domain_pao_thr,
            pair_energy_thr=settings.dlno_ccsd_pair_energy_thr,
            s1e=s1e,
            fock=fock,
        )
        ccsd_dlno_data = filter_dlno_data_for_rank(ccsd_dlno_data, frag_lolist)

    mp2_dlno_data = ccsd_dlno_data

    emp2, mp2_fragment_diagnostics = run_mp2_correction(
        mf, frag_lolist, lo_coeff, ncore, settings, mp2_dlno_data, timer=timer,
        s1e=s1e, fock=fock,
    )

    method_name = "DLNO-CCSD(T)" if settings.ccsd_t else "DLNO-CCSD"
    with timed_section(timer, method_name):
        cc = enable_dlno_prescreen(
            make_cc_solver(mf, ncore, settings, s1e=s1e, fock=fock),
            ccsd_dlno_data,
        )
        cc.kernel(frag_lolist=frag_lolist, orbloc=lo_coeff)

    e_corr = cc.e_corr_pt2corrected(emp2)
    metadata = system_metadata(mol, settings, ncore)
    metadata.update(lno_metadata(lo_coeff, frag_lolist, ccsd_dlno_data))
    metadata["fragment_diagnostics"] = {
        "DLNO-MP2 correction": mp2_fragment_diagnostics,
        method_name: getattr(cc, "fragment_diagnostics", []),
    }
    if RANK == 0:
        energy = ehf + e_corr
    else:
        energy = e_corr

    if return_metadata:
        return energy, metadata
    return energy


def run_gradient(xyz_path, settings: CalculationSettings):
    configure_pyscfad()
    mol = build_mol(xyz_path, settings)
    timer = Timer()
    with timer.section("Gradient evaluation total", subtotal=False):
        cderi_file = prepare_shared_outcore_cderi(mol, timer=timer)
        if RANK == 0:
            print(
                "Low-memory gradient: fragment replay VJP enabled; "
                "global CDERI is read from disk.",
                flush=True,
            )
        try:
            (energy, aux_metadata), grad = jax.value_and_grad(
                lambda m: dlno_ccsd_t_with_mp2_correction(
                    m,
                    settings,
                    cderi_file=cderi_file,
                    timer=timer,
                    return_metadata=True,
                ),
                has_aux=True,
            )(mol)
        finally:
            COMM.Barrier()
            if RANK == 0:
                cderi_file.unlink(missing_ok=True)

        energy_local = np.asarray([float(energy)], dtype=float)
        grad_local = np.asarray(grad.coords, dtype=float)
        if RANK == 0:
            energy_total = np.zeros_like(energy_local)
            grad_total = np.zeros_like(grad_local)
        else:
            energy_total = None
            grad_total = None

        COMM.Reduce(
            [energy_local, MPI.DOUBLE], [energy_total, MPI.DOUBLE], op=MPI.SUM, root=0
        )
        COMM.Reduce(
            [grad_local, MPI.DOUBLE], [grad_total, MPI.DOUBLE], op=MPI.SUM, root=0
        )
        fragment_diagnostics = gather_fragment_diagnostics(
            aux_metadata.get("fragment_diagnostics", {})
        )

    if RANK != 0:
        return None

    return {
        "energy": float(energy_total[0]),
        "gradient": grad_total,
        "gradient_norm": float(np.linalg.norm(grad_total)),
        "elapsed_s": timer.elapsed("Gradient evaluation total"),
        "timer": timer,
        "settings": settings,
        "xyz_path": Path(xyz_path).expanduser().resolve(),
        "metadata": aux_metadata,
        "fragment_diagnostics": fragment_diagnostics,
    }


def run_dlno_domain_build(xyz_path, settings: CalculationSettings):
    """Run only through SCF, local orbitals/fragments, and DLNO domains."""
    configure_pyscfad()
    mol = build_mol(xyz_path, settings)
    timer = Timer()
    with timer.section("DLNO domain build total", subtotal=False):
        cderi_file = prepare_shared_outcore_cderi(mol, timer=timer)
        try:
            ncore = settings.frozen_core
            if RANK == 0:
                print(f"Frozen core orbitals: ncore = {ncore}", flush=True)

            with timed_section(timer, "SCF"):
                mf = make_density_fitted_rhf(
                    mol,
                    cderi_file=cderi_file,
                    cderi_ready=cderi_file is not None,
                    scf_init_guess=settings.scf_init_guess,
                    scf_chkfile=settings.scf_chkfile,
                )
                mf.kernel()
                s1e, fock = scf_reference_matrices(mf)

            with timed_section(timer, "Local orbitals/fragments"):
                lo_coeff, frag_lolist = build_local_orbitals_and_fragments(
                    mf, ncore, settings
                )

            with timed_section(timer, "DLNO-CCSD domains"):
                dlno_data = build_dlno_data(
                    mf,
                    lo_coeff,
                    frag_lolist,
                    ncore,
                    settings,
                    domain_pao_thr=settings.dlno_ccsd_domain_pao_thr,
                    pair_energy_thr=settings.dlno_ccsd_pair_energy_thr,
                    s1e=s1e,
                    fock=fock,
                )
                dlno_data = filter_dlno_data_for_rank(dlno_data, frag_lolist)
                local_fragment_diagnostics = {
                    "DLNO domains": domain_fragment_diagnostics(dlno_data)
                }
        finally:
            COMM.Barrier()
            if RANK == 0:
                cderi_file.unlink(missing_ok=True)

    fragment_diagnostics = gather_fragment_diagnostics(local_fragment_diagnostics)

    if RANK != 0:
        return None

    metadata = system_metadata(mol, settings, settings.frozen_core)
    metadata.update(lno_metadata(lo_coeff, frag_lolist, dlno_data))
    return {
        "elapsed_s": timer.elapsed("DLNO domain build total"),
        "timer": timer,
        "settings": settings,
        "xyz_path": Path(xyz_path).expanduser().resolve(),
        "metadata": metadata,
        "nlo": metadata["nlo"],
        "nfrag_total": metadata["nfrag_total"],
        "nfrag_rank": metadata["nfrag_rank"],
        "fragment_diagnostics": fragment_diagnostics,
    }


def print_domain_build_result(result):
    settings = result["settings"]
    print()
    print("DLNO domain build only")
    print(f"Molecule: {result['xyz_path']}")
    print(f"Basis: {settings.basis}")
    print(f"LO type: {settings.lo_type}")
    print_system_metadata(result["metadata"])
    print(f"Number of LOs: {result['metadata']['nlo']}")
    print(f"Fragments total: {result['metadata']['nfrag_total']}")
    print(f"Fragments on rank 0: {result['metadata']['nfrag_rank']}")
    print_fragment_diagnostics(result.get("fragment_diagnostics"))
    result["timer"].print_summary()


def _fmt_count(value):
    return "" if value is None else str(value)


def _fmt_energy(value):
    return "" if value is None else f"{value: .8e}"


def print_fragment_diagnostics(fragment_diagnostics):
    if not fragment_diagnostics:
        return

    for section, rows in fragment_diagnostics.items():
        if not rows:
            continue
        print()
        print(f"{section} fragment diagnostics")
        has_energy = any(any(key.startswith("e_") for key in row) for row in rows)
        print(
            "Legend: frag=global fragment index, rank=MPI rank, loc=rank-local "
            "fragment index, LO=fragment localized orbitals, DLNO str=strong "
            "LMOs, BP=extended Boughton-Pulay domain atoms, prim=extended "
            "primary domain atoms, D-occ/D-vir=DLNO prescreen occ/vir sizes, "
            "LNO occ/vir=active LNO occ/vir sizes, LNO/parent=active LNO "
            "orbitals out of parent active orbitals, LMP2/CCSD/(T)=fragment "
            "energy contributions."
        )
        if has_energy:
            header = (
                f"{'frag':>5} {'rank':>4} {'loc':>4} {'LO':>4} "
                f"{'DLNO str':>8} {'BP':>4} {'prim':>5} {'D-occ':>5} {'D-vir':>5} "
                f"{'LNO occ':>7} {'LNO vir':>7} {'LNO/parent':>12} "
                f"{'LMP2':>14} {'CCSD':>14} {'(T)':>14}"
            )
        else:
            header = (
                f"{'frag':>5} {'rank':>4} {'loc':>4} {'LO':>4} "
                f"{'DLNO str':>8} {'BP':>4} {'prim':>5} {'D-occ':>5} {'D-vir':>5}"
            )
        print(header)
        print("-" * len(header))
        for row in rows:
            base = (
                f"{row.get('fragment_index', ''):>5} "
                f"{row.get('rank', ''):>4} "
                f"{row.get('local_fragment_index', ''):>4} "
                f"{row.get('nlo_fragment', ''):>4} "
                f"{_fmt_count(row.get('dlno_strong_lmo_count')):>8} "
                f"{_fmt_count(row.get('dlno_extended_bp_atom_count')):>4} "
                f"{_fmt_count(row.get('dlno_extended_primary_atom_count')):>5} "
                f"{_fmt_count(row.get('dlno_occ_prescreen_size')):>5} "
                f"{_fmt_count(row.get('dlno_vir_prescreen_size')):>5}"
            )
            if has_energy:
                lno_size = ""
                if row.get("lno_active_nmo") is not None:
                    lno_size = f"{row.get('lno_active_nmo')}/{row.get('parent_active_nmo')}"
                base = (
                    f"{base} "
                    f"{_fmt_count(row.get('lno_active_nocc')):>7} "
                    f"{_fmt_count(row.get('lno_active_nvir')):>7} "
                    f"{lno_size:>12} "
                    f"{_fmt_energy(row.get('e_lmp2')):>14} "
                    f"{_fmt_energy(row.get('e_ccsd')):>14} "
                    f"{_fmt_energy(row.get('e_ccsd_t')):>14}"
                )
            print(base)


def print_result(result):
    settings = result["settings"]
    method_name = "DLNO-CCSD(T)" if settings.ccsd_t else "DLNO-CCSD"
    print()
    print(f"{method_name} with {settings.mp2_correction.upper()}-MP2 correction")
    print(f"Molecule: {result['xyz_path']}")
    print(f"Basis: {settings.basis}")
    print(f"LO type: {settings.lo_type}")
    print_system_metadata(result["metadata"])
    if "nlo" in result["metadata"]:
        print(f"Number of LOs: {result['metadata']['nlo']}")
    if "nfrag_total" in result["metadata"]:
        print(f"Fragments total: {result['metadata']['nfrag_total']}")
    if "nfrag_rank" in result["metadata"]:
        print(f"Fragments on rank 0: {result['metadata']['nfrag_rank']}")
    print(f"LNO thresholds: occ={settings.lno_occ_thr}, vir={settings.lno_vir_thr}")
    print(f"Low-memory gradient: {settings.low_memory_gradient}")
    print(f"Perturbative triples: {settings.ccsd_t}")
    print(f"MP2 correction: {settings.mp2_correction}")
    if settings.mp2_correction.lower() in ("dlno", "local"):
        print(
            "MP2 LNO thresholds: "
            f"occ={settings.mp2_lno_occ_thr}, vir={settings.mp2_lno_vir_thr}"
        )
    print(f"Energy: {result['energy']: .15f}")
    print(f"Gradient norm: {result['gradient_norm']:.15e}")
    print("Gradient:")
    print(result["gradient"])
    print_fragment_diagnostics(result.get("fragment_diagnostics"))
    result["timer"].print_summary()


def make_water_box(n, out="water.xyz"):
    mw, NA, rho = 18.01528, 6.02214076e23, 1.0
    L = ((n * mw / NA) / rho * 1e24) ** (1/3)

    out_path = Path(out).expanduser().resolve()
    workdir = out_path.parent
    workdir.mkdir(parents=True, exist_ok=True)
    h2o_path = workdir / "h2o.xyz"
    packmol_path = workdir / "packmol.inp"

    h2o_path.write_text("""3
w
O 0 0 0
H 0.9572 0 0
H -0.23999 0.92730 0
""")

    packmol_path.write_text(f"""tolerance 2.0
filetype xyz
output {out_path.name}

structure {h2o_path.name}
number {n}
inside box 0 0 0 {L:.4f} {L:.4f} {L:.4f}
end structure
""")

    with packmol_path.open("rb") as packmol_input:
        subprocess.run(
            ["packmol"],
            stdin=packmol_input,
            check=True,
            stdout=subprocess.DEVNULL,
            cwd=str(workdir),
        )
    return L
