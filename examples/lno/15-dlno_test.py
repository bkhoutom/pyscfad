"""Minimal MPI low-memory DLNO threshold sweep script.

Edit the values below, then run:

    mpirun -n 2 python run_test.py
"""

from pathlib import Path

from pyscfad.lno.dlno_workflow import (
    COMM,
    RANK,
    SIZE,
    CalculationSettings,
    make_water_box,
    print_domain_build_result,
    print_result,
    run_dlno_domain_build,
    run_gradient,
)


NWATER = None  # Set to None when using XYZ_FILE.
XYZ_FILE = "h2_dimer.xyz"  # Set to an XYZ filename to bypass generated water_N.xyz.
FROZEN_CORE = 0

SWEEP_PARAMETER = "dlno_ccsd_pair_energy_thr"
SWEEP_VALUES = [1.0e-5]
MP2_CORRECTION = "dlno"  # "dlno" or "canonical"
CCSD_T = True
LOW_MEMORY_GRADIENT = True
DOMAIN_ONLY = False


BASE_SETTINGS = CalculationSettings(
    basis="631g",
    lo_type="iao",
    lno_occ_thr=1.0e-5,
    lno_vir_thr=1.0e-6,
    mp2_lno_occ_thr=1.0e-7,
    mp2_lno_vir_thr=1.0e-6,
    dlno_ccsd_domain_pao_thr=1.0e-5,
    dlno_ccsd_pair_energy_thr=1.0e-5,
    lmo_bp_domain_thr=0.9999,
    pao_bp_domain_thr=0.98,
    multipole_order=4,
    max_memory_mb=2000,
    frozen_core=FROZEN_CORE,
    verbose=4,
    frag_lolist_mode="auto",
    mp2_correction=MP2_CORRECTION,
    ccsd_t=CCSD_T,
    low_memory_gradient=LOW_MEMORY_GRADIENT,
)


def resolve_xyz_path():
    if XYZ_FILE is not None:
        return Path(__file__).with_name(XYZ_FILE), f"XYZ_FILE={XYZ_FILE}"

    if NWATER is None:
        raise ValueError("Either XYZ_FILE or NWATER must be set.")

    xyz_name = f"water_{NWATER}.xyz"
    xyz_path = Path(__file__).with_name(xyz_name)
    if RANK == 0:
        make_water_box(NWATER, out=str(xyz_path))
    COMM.Barrier()
    return xyz_path, f"NWATER={NWATER}"


def format_metadata_summary(result):
    metadata = result.get("metadata", {})
    parts = []
    for key in ("nelectron", "nao", "frozen_core", "nlo", "nfrag_total"):
        if key in metadata:
            if key == "nfrag_total":
                label = "fragments"
            elif key == "frozen_core":
                label = "frozen"
            else:
                label = key
            parts.append(f"{label}={metadata[key]}")
    return ", ".join(parts)


def main():
    xyz_path, molecule_source = resolve_xyz_path()
    results = []

    for value in SWEEP_VALUES:
        settings = BASE_SETTINGS.with_parameter(SWEEP_PARAMETER, value)

        if RANK == 0:
            print()
            print("=" * 80)
            print(f"MPI sweep with nproc={SIZE}: {SWEEP_PARAMETER} = {value}")
            print(f"Molecule source: {molecule_source}")
            print(f"Molecule: {xyz_path.name}")
            print(f"Low-memory gradient: {settings.low_memory_gradient}")
            print(f"Domain only: {DOMAIN_ONLY}")
            print(f"Perturbative triples: {settings.ccsd_t}")
            print(f"MP2 correction: {settings.mp2_correction}")
            print("=" * 80)

        if DOMAIN_ONLY:
            result = run_dlno_domain_build(xyz_path, settings)
        else:
            result = run_gradient(xyz_path, settings)
        if RANK == 0:
            if DOMAIN_ONLY:
                print_domain_build_result(result)
            else:
                print_result(result)
            results.append(result)

    if RANK == 0:
        print()
        print("Sweep summary")
        for value, result in zip(SWEEP_VALUES, results):
            metadata_summary = format_metadata_summary(result)
            metadata_text = f", {metadata_summary}" if metadata_summary else ""
            if DOMAIN_ONLY:
                print(
                    f"{SWEEP_PARAMETER}={value}: "
                    f"elapsed={result['elapsed_s']:.3f} s"
                    f"{metadata_text}"
                )
            else:
                print(
                    f"{SWEEP_PARAMETER}={value}: "
                    f"energy={result['energy']: .15f}, "
                    f"grad_norm={result['gradient_norm']:.15e}"
                    f"{metadata_text}"
                )


if __name__ == "__main__":
    main()
