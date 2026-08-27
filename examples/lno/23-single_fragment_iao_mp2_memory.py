"""One-fragment IAO-local-MP2 energy and nuclear-gradient memory probe.

This is the single-fragment counterpart of ``22-mpi_c16_iao_mp2.py``.  It
builds the same reference and fixed topology, but evaluates only one strong
ED-row contribution.  That contribution is pulled back through its ED frame,
the common IAO/PAO construction, and the implicit SCF response.  Hartree--Fock
and weak-pair energies are deliberately not seeded.

Run directly (no MPI is needed):

    .venv/bin/python -u examples/lno/23-single_fragment_iao_mp2_memory.py

The ``[DLNO-RESOURCE]`` lines report current and sampled peak RSS for every
forward and reverse phase.  The default run evaluates only the fragment with
the smallest estimated local ``Lov`` tensor.  Set ``FRAGMENT_INDEX`` below to
an integer to select a specific fragment instead.  Set
``PYSCF_NUM_THREADS`` in the environment to override the default of eight
PySCF/OpenMP/Accelerate threads.
"""

import gc
import os
from pathlib import Path
import tempfile
import time

THREADS = max(int(os.environ.get("PYSCF_NUM_THREADS", "8")), 1)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)
os.environ.setdefault("PYSCFAD_LNO_LOCAL_DIRECT_INT3C_BLOCK_MB", "128")
os.environ.setdefault("PYSCFAD_DF_CDERI_BAR_AUX_BLOCK_MB", "128")
os.environ.setdefault("PYSCFAD_DLNO_RESOURCE_PROFILE", "1")
os.environ.setdefault("PYSCFAD_DLNO_RESOURCE_SAMPLE_MS", "10")

import jax
import jax.numpy as jnp
import numpy
from pyscf import lib as pyscf_lib
from pyscf.df import addons as df_addons
from pyscf.scf import chkfile as pyscf_scf_chkfile

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    build_strong_ed_domain,
    rebuild_iao_mp2_common,
    strong_domain_energy,
)
from pyscfad.lno import lno_base
from pyscfad.ops import stop_trace
from pyscfad.tools import resource_profile


GEOMETRY = "water_24.xyz"
BASIS = "cc-pvtz"
AUXBASIS = "cc-pvtz-ri"
FROZEN = 24
PAIR_THRESHOLD = 1e-4
FRAGMENT_INDEX = None
SCF_CHECKPOINT = "water_24_ccpvtz_df_rhf.chk"


def _add_cotangent(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if hasattr(left, "dtype") and left.dtype == jax.dtypes.float0:
        return right
    if hasattr(right, "dtype") and right.dtype == jax.dtypes.float0:
        return left
    return left + right


def main():
    wall_start = time.perf_counter()
    pyscf_lib.num_threads(THREADS)
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_scf_first_order_custom", False)

    mol = gto.Mole(
        atom=str(Path(__file__).with_name(GEOMETRY)),
        basis=BASIS,
        max_memory=2000,
        verbose=3,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    print(
        f"[single-fragment] geometry={GEOMETRY}; atoms={mol.natm}; "
        f"AOs={mol.nao_nr()}; basis={BASIS}; frozen={FROZEN}; "
        f"requested F="
        f"{FRAGMENT_INDEX if FRAGMENT_INDEX is not None else 'min Lov'}; "
        f"threads={pyscf_lib.num_threads()}",
        flush=True,
    )

    global_auxmol = df_addons.make_auxmol(mol, AUXBASIS)
    naux_global = int(global_auxmol.nao_nr())
    cderi_gib = (
        naux_global
        * mol.nao_nr()
        * (mol.nao_nr() + 1)
        // 2
        * numpy.dtype(numpy.float64).itemsize
        / 1024.0**3
    )
    del global_auxmol
    print(
        f"[single-fragment] naux={naux_global}; estimated packed "
        f"CDERI scratch={cderi_gib:.2f} GiB",
        flush=True,
    )
    checkpoint_path = Path(__file__).with_name(SCF_CHECKPOINT)
    restart_from_checkpoint = checkpoint_path.is_file()
    checkpoint_dm = None
    if restart_from_checkpoint:
        _, checkpoint_data = pyscf_scf_chkfile.load_scf(
            str(checkpoint_path)
        )
        checkpoint_coeff = numpy.asarray(checkpoint_data["mo_coeff"])
        checkpoint_occ = numpy.asarray(checkpoint_data["mo_occ"])
        checkpoint_dm = numpy.einsum(
            "pi,i,qi->pq",
            checkpoint_coeff,
            checkpoint_occ,
            checkpoint_coeff.conj(),
            optimize=True,
        )
        del checkpoint_coeff, checkpoint_occ, checkpoint_data
    print(
        f"[single-fragment] SCF checkpoint={checkpoint_path}; "
        f"restart={'yes' if restart_from_checkpoint else 'no'}; "
        f"PySCF OpenMP threads={pyscf_lib.num_threads()}; "
        f"BLAS thread limit={THREADS}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="pyscfad-single-fragment-mp2-"
    ) as scratch:
        cderi_path = str(Path(scratch) / "cderi.h5")
        with resource_profile.section("single_fragment_cderi_build"):
            df_builder = scf.RHF(mol).density_fit(
                auxbasis=AUXBASIS
            ).with_df
            df_builder.max_memory = mol.max_memory
            df_builder._cderi_to_save = cderi_path
            df_builder.build()
            if int(df_builder.auxmol.nao_nr()) != naux_global:
                raise RuntimeError("preview and built auxiliary sizes differ")
            del df_builder

        def build_mf(mol_):
            mf_ = scf.RHF(mol_).density_fit(auxbasis=AUXBASIS)
            mf_.with_df.max_memory = mol_.max_memory
            mf_.with_df.attach_outcore_cderi(cderi_path)
            mf_.chkfile = None
            mf_.verbose = 4
            mf_.max_cycle = 100
            mf_.kernel(dm0=checkpoint_dm)
            if not mf_.converged:
                raise RuntimeError("DF-RHF did not converge")
            mf_.verbose = 0
            mf_.mol.verbose = 0
            return mf_

        with resource_profile.section("single_fragment_scf_forward"):
            mf, scf_pullback = jax.vjp(build_mf, mol)
            jax.block_until_ready(mf.e_tot)
        e_hf = float(jax.device_get(mf.e_tot))

        thresholds = IAOFragmentMP2Thresholds(
            pair_energy=PAIR_THRESHOLD,
            mp2_block_memory_mb=128.0,
        )

        def build_static(mf_):
            topology = build_iao_fragment_topology(
                mf_,
                frozen=FROZEN,
                thresholds=thresholds,
                pair_energy_model="multipole",
            )
            return build_iao_mp2_static_selections(mf_, topology)

        with resource_profile.section("single_fragment_fixed_topology"):
            static = stop_trace(build_static)(mf)

        def local_naux(fragment_):
            local_mol_ = lno_base.make_local_mol(
                mol, fragment_.extended_atoms
            )
            local_auxmol_ = df_addons.make_auxmol(local_mol_, AUXBASIS)
            value = int(local_auxmol_.nao_nr())
            del local_auxmol_, local_mol_
            return value

        with resource_profile.section("single_fragment_selection"):
            if FRAGMENT_INDEX is None:
                local_naux_by_fragment = tuple(
                    local_naux(item) for item in static.fragments
                )
                fragment_index = min(
                    range(len(static.fragments)),
                    key=lambda index: (
                        local_naux_by_fragment[index]
                        * len(static.fragments[index].strong_occ_metric_keep)
                        * len(
                            static.fragments[index].strong_virtual.metric_keep
                        )
                    ),
                )
                naux_local = local_naux_by_fragment[fragment_index]
            else:
                fragment_index = int(FRAGMENT_INDEX)
                if not 0 <= fragment_index < len(static.fragments):
                    raise ValueError(
                        f"FRAGMENT_INDEX={fragment_index} is outside "
                        f"[0, {len(static.fragments)})"
                    )
                naux_local = local_naux(static.fragments[fragment_index])
        fragment = static.fragments[fragment_index]
        strong_mask = numpy.asarray(static.strong_mask, dtype=bool)
        nfragment = len(static.fragments)
        nstrong_pair = int(numpy.count_nonzero(
            numpy.triu(strong_mask, k=1)
        ))
        ntotal_pair = nfragment * (nfragment - 1) // 2

        mp2_forward_profile = resource_profile.start()
        with resource_profile.section("single_fragment_common_forward"):
            common, common_pullback = jax.vjp(
                lambda mf_: rebuild_iao_mp2_common(mf_, static), mf
            )
            jax.block_until_ready(common)

        with resource_profile.section("single_fragment_domain_forward"):
            domain, domain_pullback = jax.vjp(
                lambda common_: build_strong_ed_domain(
                    common_, static, fragment_index
                ),
                common,
            )
            jax.block_until_ready(domain)

        lov_mib = (
            naux_local
            * domain.occupied_coeff.shape[1]
            * domain.virtual_coeff.shape[1]
            * numpy.dtype(numpy.float64).itemsize
            / 1024.0**2
        )

        print("\nSelected strong fragment", flush=True)
        print(f"  F                         {fragment_index}", flush=True)
        print(
            f"  strong partners           "
            f"{list(map(int, fragment.strong_fragments))}",
            flush=True,
        )
        print(f"  estimated Lov size        {lov_mib:.1f} MiB", flush=True)
        print(
            f"  ED atoms/AO/aux/occ/vir   "
            f"{len(fragment.extended_atoms)}/"
            f"{len(fragment.extended_ao_indices)}/"
            f"{naux_local}/"
            f"{domain.occupied_coeff.shape[1]}/"
            f"{domain.virtual_coeff.shape[1]}",
            flush=True,
        )
        print(
            f"  all strong/weak pairs     "
            f"{nstrong_pair}/{ntotal_pair - nstrong_pair}",
            flush=True,
        )
        print(f"  global auxiliary size     {naux_global}", flush=True)

        def fragment_energy(mf_, domain_):
            return strong_domain_energy(
                mf_, domain_, static, fragment_index
            ).total

        with resource_profile.section("single_fragment_mp2_forward"):
            energy, fragment_pullback = jax.vjp(
                fragment_energy, mf, domain
            )
            jax.block_until_ready(energy)
        resource_profile.finish(
            "single_fragment_mp2_total_forward",
            mp2_forward_profile,
            fragment=fragment_index,
            lov_mib=float(lov_mib),
        )

        mp2_reverse_profile = resource_profile.start()
        with resource_profile.section("single_fragment_mp2_reverse"):
            direct_mf_bar, domain_bar = fragment_pullback(
                jnp.ones((), dtype=jnp.asarray(energy).dtype)
            )
            jax.block_until_ready((direct_mf_bar, domain_bar))
        del fragment_pullback, domain
        gc.collect()

        with resource_profile.section("single_fragment_domain_reverse"):
            common_bar, = domain_pullback(domain_bar)
            jax.block_until_ready(common_bar)
        del domain_pullback, domain_bar
        gc.collect()

        with resource_profile.section("single_fragment_common_reverse"):
            indirect_mf_bar, = common_pullback(common_bar)
            mf_bar = jax.tree_util.tree_map(
                _add_cotangent, direct_mf_bar, indirect_mf_bar
            )
            jax.block_until_ready(mf_bar)
        del common_pullback, common_bar, direct_mf_bar, indirect_mf_bar
        gc.collect()
        resource_profile.finish(
            "single_fragment_mp2_total_reverse",
            mp2_reverse_profile,
            fragment=fragment_index,
            lov_mib=float(lov_mib),
        )

        with resource_profile.section("single_fragment_scf_reverse"):
            mol_bar, = scf_pullback(mf_bar)
            jax.block_until_ready(mol_bar)

        gradient = numpy.asarray(mol_bar.coords)
        print("\nSingle-fragment result")
        print(f"  Reference HF energy       {e_hf:+.12f} Eh")
        print(
            f"  Strong contribution E(F)  "
            f"{float(jax.device_get(energy)):+.12f} Eh"
        )
        print(
            f"  |dE(F)/dR|                "
            f"{numpy.linalg.norm(gradient):.8e} Eh/bohr"
        )
        print(
            "  Evaluated terms            1 strong fragment; "
            "0 weak pairs; no HF energy seed"
        )
        total_wall_seconds = time.perf_counter() - wall_start
        print(f"  Total wall time            {total_wall_seconds:.1f} s")
        print("\nGradient of this fragment contribution (Hartree/Bohr):")
        print(gradient)
        resource_profile.checkpoint(
            "single_fragment_complete",
            fragment=fragment_index,
            energy=float(jax.device_get(energy)),
            gradient_norm=float(numpy.linalg.norm(gradient)),
            total_wall_seconds=float(total_wall_seconds),
        )


if __name__ == "__main__":
    main()
