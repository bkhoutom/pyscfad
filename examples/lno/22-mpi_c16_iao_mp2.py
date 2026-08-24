"""MPI IAO-local-MP2 energy and gradient for C16H34/cc-pVTZ.

Rank 0 defines the common orbital gauge; workers adopt its MO state.  Run as

    mpirun -np 2 .venv/bin/python examples/lno/22-mpi_c16_iao_mp2.py
"""

import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB", "128")
os.environ.setdefault("PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB", "128")

import numpy
from mpi4py import MPI

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds
from pyscfad.dlno.iao_mp2_mpi import IAOFragmentMP2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

config.update("pyscfad_moleintor_opt", True)

BASIS = "cc-pvtz"
AUXBASIS = "cc-pvtz-ri"
FROZEN = 16  # carbon 1s
PAIR_THRESHOLD = 1e-4

scratch = tempfile.TemporaryDirectory(prefix="pyscfad-c16-mp2-") \
    if rank == 0 else None
CDERI_PATH = comm.bcast(
    str(Path(scratch.name) / "cderi.h5") if rank == 0 else None, root=0
)

mol = gto.Mole(
    atom=str(Path(__file__).with_name("c16h34.xyz")),
    basis=BASIS,
    max_memory=2000,
    verbose=4 if rank == 0 else 0,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)

wall_start = time.perf_counter()
timing = {}
if rank == 0:
    print(
        f"[C16] {mol.natm} atoms, {mol.nao_nr()} AOs, "
        f"cc-pVTZ/cc-pVTZ-RI, frozen={FROZEN}, "
        f"pair threshold={PAIR_THRESHOLD:.1e}, MPI ranks={comm.Get_size()}",
        flush=True,
    )
    print("[C16] out-of-core CDERI build: starting", flush=True)
    start = time.perf_counter()
    df_builder = scf.RHF(mol).density_fit(auxbasis=AUXBASIS).with_df
    df_builder.max_memory = mol.max_memory
    df_builder._cderi_to_save = CDERI_PATH
    df_builder.build()
    timing["cderi"] = time.perf_counter() - start
    print(
        f"[C16] out-of-core CDERI build: done in "
        f"{timing['cderi']:.1f} s; naux={df_builder.auxmol.nao_nr()}",
        flush=True,
    )
    del df_builder
comm.Barrier()


def build_mf(mol_, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    mf = scf.RHF(mol_).density_fit(auxbasis=AUXBASIS)
    mf.with_df.max_memory = mol_.max_memory
    mf.with_df.attach_outcore_cderi(CDERI_PATH)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-5
    mf.max_cycle = 100
    if mo_coeff_init is None:
        start = time.perf_counter()
        mf.kernel()
        timing["scf"] = time.perf_counter() - start
        if not mf.converged:
            raise RuntimeError("DF-RHF did not converge")
        # Retain the useful SCF cycles above, then suppress PySCF's opaque
        # per-domain ``mem_avail/mem_blk/auxlen`` diagnostics.  The explicit
        # IAO-MP2 progress stream below reports those stages instead.
        mf.verbose = 0
        mf.mol.verbose = 0
    else:
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
    return mf


thresholds = IAOFragmentMP2Thresholds(
    pair_energy=PAIR_THRESHOLD,
    mp2_block_memory_mb=128.0,
)
energy, mol_bar, details = IAOFragmentMP2.value_and_grad(
    mol, build_mf=build_mf, frozen=FROZEN, thresholds=thresholds,
    pair_energy_model="multipole", include_hf=True, comm=comm,
    return_details=True, progress=True,
)

if rank == 0:
    total_seconds = time.perf_counter() - wall_start
    gradient = numpy.asarray(mol_bar.coords)
    strong_rows = {
        term.left_fragment: term.energy
        for term in details.terms if term.kind == "strong"
    }

    print("\nIAO-local-MP2 C16H34/cc-pVTZ result")
    print(f"MPI ranks                  {comm.Get_size()}")
    print(f"Total energy               {float(energy):+.12f} Eh")
    print(f"Correlation energy         {details.e_corr:+.12f} Eh")
    print(f"  weighted ED-row sum      {details.e_strong:+.12f} Eh")
    print(f"  weak multipole energy    {details.e_weak:+.12f} Eh")
    print(f"Fragments (N_F)            {details.n_fragments}")
    print(f"Strong/weak pairs          {details.n_strong_pairs}/"
          f"{details.n_weak_pairs}")

    print("\nED dimensions")
    print("  F  atoms   AO  occ  vir   E_strong(F) / Eh   partners (incl. self)")
    for fragment in details.fragments:
        partners = ",".join(str(i) for i in fragment.strong_fragments)
        print(f" {fragment.fragment_index:2d}  "
              f"{fragment.n_domain_atoms:5d} "
              f"{fragment.n_domain_ao:4d} "
              f"{fragment.n_domain_occ:4d} "
              f"{fragment.n_domain_vir:4d}  "
              f"{strong_rows[fragment.fragment_index]:+.10f}   "
              f"[{partners}]")

    t = details.timing
    accounted = timing["cderi"] + timing["scf"] + t.total_seconds
    print("\nTiming (seconds; strong/weak totals are rank-seconds)")
    print(f"  CDERI build               {timing['cderi']:.2f}")
    print(f"  SCF forward               {timing['scf']:.2f}")
    print(f"  common forward/reverse    {t.common_forward_seconds:10.2f} / "
          f"{t.common_reverse_seconds:.2f}")
    print(f"  strong forward/reverse    {t.strong_forward_seconds:10.2f} / "
          f"{t.strong_reverse_seconds:.2f}")
    print(f"  weak forward/reverse      {t.weak_forward_seconds:10.2f} / "
          f"{t.weak_reverse_seconds:.2f}")
    print(f"  frame build/replay        {t.frame_build_seconds:10.2f} / "
          f"{t.frame_replay_seconds:.2f}")
    print(f"  correlation gradient      {t.total_seconds:.2f}")
    print(f"  remaining setup/response  {total_seconds - accounted:.2f}")
    print(f"  total wall                {total_seconds:.2f}")

    print("\nGradient (Hartree/Bohr):")
    print(gradient)

comm.Barrier()
if rank == 0:
    scratch.cleanup()
