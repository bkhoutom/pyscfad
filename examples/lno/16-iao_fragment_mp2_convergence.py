"""IAO-fragment extended-domain MP2 convergence for a linear alkane.

This example is deliberately an energy-only benchmark.  It builds an
all-trans-like ``C_n H_(2n+2)`` geometry without an external geometry package,
runs density-fitted RHF, and compares the IAO-fragment ED-MP2 correlation
energy with canonical DF-MP2 as the fragment-pair threshold is tightened.

The IAO fragment weights, rather than unit-weight projectors onto overlapping
fragment occupied spans, partition the MP2 energy.  Consequently the
all-strong/full-domain endpoint is the canonical DF-MP2 energy (up to numerical
precision).  Use small systems first; for example::

    python examples/lno/16-iao_fragment_mp2_convergence.py \
        --ncarbon 6 --basis sto-3g --pair-thresholds 1e-3 1e-4 0

The ``1e-3`` point is intentionally loose and is useful only for displaying a
convergence trend; it is not a recommended production threshold for the
multipole pair estimate.

Larger basis sets and long chains can be substantially more expensive because
the optional canonical reference is not local.  Pass ``--skip-canonical`` when
only domain sizes and local-energy convergence are needed.  For large
canonical references, ``--canonical-backend native`` uses PySCF's disk-backed
native DF-MP2 implementation and does not store the MP2 amplitudes.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
    evaluate_iao_fragment_mp2,
)
from pyscfad.mp import dfmp2


CC_BOND = 1.54
CH_BOND = 1.09
CCC_ANGLE_DEG = 112.0


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def _perpendicular_frame(axis):
    """Return two deterministic unit vectors perpendicular to ``axis``."""
    axis = _unit(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(axis, reference))
    second = _unit(np.cross(axis, first))
    return first, second


def _terminal_hydrogen_directions(carbon_bond):
    """Three tetrahedral C--H directions opposite one C--C bond."""
    axis = _unit(carbon_bond)
    first, second = _perpendicular_frame(axis)
    axial = -1.0 / 3.0
    radial = np.sqrt(1.0 - axial * axial)
    directions = []
    for azimuth in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        directions.append(
            axial * axis
            + radial * (
                np.cos(azimuth) * first + np.sin(azimuth) * second
            )
        )
    return directions


def _internal_hydrogen_directions(left_bond, right_bond):
    """Two approximately tetrahedral C--H directions for a CH2 center."""
    left = _unit(left_bond)
    right = _unit(right_bond)
    cosine = float(np.dot(left, right))
    target_cosine = -1.0 / 3.0
    in_plane = target_cosine / (1.0 + cosine) * (left + right)
    normal = _unit(np.cross(left, right))
    normal_scale2 = 1.0 - float(np.dot(in_plane, in_plane))
    if normal_scale2 <= 0.0:
        raise ValueError("C--C--C angle is incompatible with this CH2 builder")
    normal_scale = np.sqrt(normal_scale2)
    return [in_plane + normal_scale * normal,
            in_plane - normal_scale * normal]


def make_n_alkane(ncarbon, cc_bond=CC_BOND, ch_bond=CH_BOND,
                  ccc_angle_deg=CCC_ANGLE_DEG):
    """Return ``[(element, (x, y, z)), ...]`` for a linear alkane.

    The carbon backbone is a planar zig-zag.  The terminal methyl hydrogens
    have exact tetrahedral directions relative to their C--C bond; the two
    hydrogens on each internal carbon are placed symmetrically above and below
    the carbon plane.
    """
    if ncarbon < 1:
        raise ValueError("ncarbon must be at least one")

    if ncarbon == 1:
        carbon = np.zeros((1, 3))
        tetrahedron = np.asarray([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]) / np.sqrt(3.0)
        atoms = [("C", carbon[0])]
        atoms.extend(("H", ch_bond * direction) for direction in tetrahedron)
        validate_alkane(atoms, ncarbon)
        return [(element, tuple(coordinate)) for element, coordinate in atoms]

    # The angle between consecutive forward bond vectors is 180 - C-C-C.
    half_turn = np.deg2rad(180.0 - ccc_angle_deg) / 2.0
    carbon = [np.zeros(3)]
    for bond_index in range(ncarbon - 1):
        angle = half_turn if bond_index % 2 == 0 else -half_turn
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        carbon.append(carbon[-1] + cc_bond * direction)
    carbon = np.asarray(carbon)

    atoms = [("C", coordinate) for coordinate in carbon]
    for index, coordinate in enumerate(carbon):
        if index == 0:
            directions = _terminal_hydrogen_directions(carbon[1] - coordinate)
        elif index == ncarbon - 1:
            directions = _terminal_hydrogen_directions(
                carbon[-2] - coordinate
            )
        else:
            directions = _internal_hydrogen_directions(
                carbon[index - 1] - coordinate,
                carbon[index + 1] - coordinate,
            )
        atoms.extend(
            ("H", coordinate + ch_bond * direction)
            for direction in directions
        )

    validate_alkane(atoms, ncarbon)
    return [(element, tuple(coordinate)) for element, coordinate in atoms]


def validate_alkane(atoms, ncarbon):
    """Validate formula, bond counts, and single-bond valences."""
    elements = [element for element, _ in atoms]
    coordinates = np.asarray([coordinate for _, coordinate in atoms])
    expected_hydrogen = 2 * ncarbon + 2
    if elements.count("C") != ncarbon:
        raise ValueError("incorrect number of carbon atoms")
    if elements.count("H") != expected_hydrogen:
        raise ValueError("incorrect number of hydrogen atoms")

    adjacency = np.zeros((len(atoms), len(atoms)), dtype=bool)
    for left in range(len(atoms)):
        for right in range(left):
            pair = frozenset((elements[left], elements[right]))
            distance = np.linalg.norm(coordinates[left] - coordinates[right])
            if pair == {"C"}:
                bonded = distance < 1.75
            elif pair == {"C", "H"}:
                bonded = distance < 1.25
            else:
                bonded = False
            adjacency[left, right] = adjacency[right, left] = bonded

    valence = adjacency.sum(axis=1)
    expected_valence = np.asarray([4 if element == "C" else 1
                                   for element in elements])
    if not np.array_equal(valence, expected_valence):
        bad = [
            f"{elements[index]}{index + 1}:{int(value)}"
            for index, value in enumerate(valence)
            if value != expected_valence[index]
        ]
        raise ValueError("invalid generated valences: " + ", ".join(bad))

    carbon_indices = [
        i for i, element in enumerate(elements) if element == "C"
    ]
    hydrogen_indices = [
        i for i, element in enumerate(elements) if element == "H"
    ]
    n_cc = int(adjacency[np.ix_(carbon_indices, carbon_indices)].sum() // 2)
    n_ch = int(adjacency[np.ix_(carbon_indices, hydrogen_indices)].sum())
    if n_cc != max(ncarbon - 1, 0) or n_ch != expected_hydrogen:
        raise ValueError("generated connectivity is not a linear alkane")


def build_molecule(atoms, basis, verbose=0, max_memory=4000.0):
    mol = gto.Mole()
    mol.atom = atoms
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def run_df_rhf(mol, auxbasis=None):
    """Run DF-RHF with an optional explicit fitting basis."""
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.with_df.incore = False
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("density-fitted RHF did not converge")
    return mf


def _scf_auxbasis(mf):
    """Return the exact auxiliary basis realized by the SCF DF object.

    ``with_df.auxbasis`` is often ``None`` when PySCF selected the default
    JK-fitting basis.  Passing ``None`` to ``dfmp2_native.DFRMP2`` would instead
    ask it for its default *MP2*-fitting basis.  In that case use the basis of
    the already-built SCF auxiliary molecule so the two reference backends use
    the same density-fitting approximation.
    """
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise ValueError("canonical DF-MP2 requires a density-fitted SCF object")
    auxbasis = getattr(with_df, "auxbasis", None)
    if auxbasis is not None:
        return auxbasis
    auxmol = getattr(with_df, "auxmol", None)
    if auxmol is None:
        with_df.build()
        auxmol = getattr(with_df, "auxmol", None)
    if auxmol is None:
        raise RuntimeError("the SCF auxiliary basis could not be resolved")
    return auxmol.basis


def canonical_dfmp2_energy(mf, *, frozen=0, backend="pyscfad"):
    """Return canonical DF-MP2 with the SCF DF basis and frozen convention.

    The ``native`` backend is PySCF's ``dfmp2_native.DFRMP2``.  It streams the
    occupied--virtual three-index integrals through temporary HDF5 files and
    its energy-only kernel never constructs or stores a T2 tensor.
    """
    if backend == "pyscfad":
        solver = dfmp2.MP2(mf, frozen=frozen)
        solver.kernel(with_t2=False)
        return float(np.asarray(solver.e_corr))
    if backend == "native":
        # Import lazily: small/default examples do not need this optional
        # non-differentiable reference implementation.
        from pyscf.mp import dfmp2_native

        auxbasis = _scf_auxbasis(mf)
        with dfmp2_native.DFRMP2(
            mf, frozen=frozen, auxbasis=auxbasis
        ) as solver:
            # The native NumPy/BLAS implementation accesses ``.flags`` on its
            # coefficient arrays; a converged PySCFAD reference may hold JAX
            # arrays even though this reference calculation is intentionally
            # non-differentiable.
            solver.mo_coeff = np.asarray(mf.mo_coeff)
            solver.mo_energy = np.asarray(mf.mo_energy)
            return float(solver.kernel())
    raise ValueError(f"unknown canonical DF-MP2 backend {backend!r}")


def _mean_max(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.1f}/{int(values.max())}"


def summarize_result(
    pair_threshold,
    result,
    canonical_energy,
    topology_elapsed,
    evaluation_elapsed,
):
    fragments = result.fragments
    topology = result.topology
    timing = result.timing
    upper = np.triu(topology.strong_mask, k=1)
    n_strong = int(np.count_nonzero(upper))
    nfrag = len(fragments)
    n_pairs = nfrag * (nfrag - 1) // 2
    if canonical_energy is None:
        error = "--"
    else:
        error = f"{result.e_corr - canonical_energy:+.3e}"
    return {
        "threshold": f"{pair_threshold:.1e}",
        "energy": f"{result.e_corr:+.10f}",
        "error": error,
        "pairs": f"{n_strong}/{n_pairs}",
        "atoms": _mean_max([fragment.extended_atoms.size
                            for fragment in fragments]),
        "aos": _mean_max([fragment.n_domain_ao for fragment in fragments]),
        "occ": _mean_max([fragment.n_domain_occ for fragment in fragments]),
        "vir": _mean_max([fragment.n_domain_vir for fragment in fragments]),
        "topology_seconds": f"{topology_elapsed:.1f}",
        "evaluation_seconds": f"{evaluation_elapsed:.1f}",
        "seconds": f"{topology_elapsed + evaluation_elapsed:.1f}",
        "ed_orbital_seconds": f"{timing.ed_orbital_seconds:.1f}",
        "local_ri_lov_seconds": f"{timing.local_ri_lov_seconds:.1f}",
        "weighted_ed_mp2_seconds": f"{timing.weighted_ed_mp2_seconds:.1f}",
        "weak_bookkeeping_seconds": f"{timing.weak_bookkeeping_seconds:.1f}",
        "profiled_seconds": f"{timing.total_seconds:.1f}",
    }


def print_table(rows):
    columns = [
        ("threshold", "pair tau"),
        ("energy", "E_corr"),
        ("error", "error"),
        ("pairs", "strong"),
        ("atoms", "ED atoms"),
        ("aos", "ED AOs"),
        ("occ", "ED occ"),
        ("vir", "ED vir"),
        ("topology_seconds", "topo/s"),
        ("evaluation_seconds", "MP2/s"),
        ("seconds", "total/s"),
    ]
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows))
        for key, label in columns
    }
    print("  ".join(label.rjust(widths[key]) for key, label in columns))
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  ".join(row[key].rjust(widths[key]) for key, _ in columns))
    print("\nDomain entries are mean/max; strong pairs exclude self-pairs.")


def print_timing_table(rows):
    columns = [
        ("threshold", "pair tau"),
        ("ed_orbital_seconds", "ED orbitals/s"),
        ("local_ri_lov_seconds", "local RI+Lov/s"),
        ("weighted_ed_mp2_seconds", "weighted ED-MP2/s"),
        ("weak_bookkeeping_seconds", "weak+book/s"),
        ("profiled_seconds", "profiled/s"),
        ("evaluation_seconds", "outer MP2/s"),
    ]
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows))
        for key, label in columns
    }
    print("\nPost-topology ED wall-time split (MRCC-comparable categories):")
    print("  ".join(label.rjust(widths[key]) for key, label in columns))
    print("  ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print("  ".join(row[key].rjust(widths[key]) for key, _ in columns))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark weighted IAO-fragment ED-MP2 on a generated linear "
            "alkane. Pair tau = 0 makes every fragment pair strong; "
            "--full-domain-check additionally forces every AO into every ED."
        )
    )
    parser.add_argument("--ncarbon", type=int, default=6,
                        help="number of carbon atoms (default: 6)")
    parser.add_argument("--basis", default="sto-3g",
                        help="orbital basis (default: sto-3g)")
    parser.add_argument(
        "--auxbasis", default=None,
        help=(
            "density-fitting basis used consistently by SCF, local, and "
            "canonical calculations (default: PySCF JK-fit choice)"
        ),
    )
    parser.add_argument(
        "--pair-thresholds", type=float, nargs="+",
        default=[1e-3, 1e-4, 1.5e-5, 0.0], metavar="TAU",
        help=(
            "strong-pair thresholds, listed loose to tight by default; "
            "1e-3 is a deliberately loose diagnostic point"
        ),
    )
    parser.add_argument(
        "--pair-model", choices=("multipole", "exact"), default="multipole",
        help=(
            "pair screening model; exact uses the Nagy-scaled OS pair "
            "increment and the weak correction remains multipole based; "
            "exact forms global DF factors and is a small-system diagnostic"
        ),
    )
    parser.add_argument("--bp-occ", type=float, default=0.985,
                        help="compact occupied BP recovery (default: 0.985)")
    parser.add_argument("--bp-primary", type=float, default=0.999,
                        help="primary-domain occupied BP recovery (default: 0.999)")
    parser.add_argument("--bp-ed", type=float, default=0.9998,
                        help="actual ED occupied BP recovery (default: 0.9998)")
    parser.add_argument("--bp-pao", type=float, default=0.98,
                        help="PAO BP recovery threshold (default: 0.98)")
    parser.add_argument("--domain-pao", type=float, default=1e-4,
                        help="PAO/domain overlap threshold (default: 1e-4)")
    parser.add_argument("--ed-pao", type=float, default=0.995,
                        help="PAO completeness required in the ED (default: 0.995)")
    parser.add_argument("--occupied-weight", type=float, default=1e-4,
                        help="normalized occupied-span cutoff (default: 1e-4)")
    parser.add_argument("--near-pair-distance", type=float, default=3.5,
                        help="distance in bohr below which pairs are forced strong")
    parser.add_argument("--mp2-block-memory", type=float, default=256.0,
                        help="MB budget for each blocked ED-MP2 contraction")
    parser.add_argument("--skip-canonical", action="store_true",
                        help="skip the nonlocal canonical DF-MP2 reference")
    parser.add_argument(
        "--canonical-backend", choices=("pyscfad", "native"),
        default="pyscfad",
        help=(
            "canonical reference implementation (default: pyscfad); native "
            "uses PySCF dfmp2_native.DFRMP2, writes transformed OV fitting "
            "integrals to temporary HDF5 storage, and stores no T2 amplitudes"
        ),
    )
    parser.add_argument(
        "--frozen-core", type=int, default=0, metavar="N",
        help=(
            "number of lowest occupied orbitals frozen in both IAO-fragment "
            "and canonical MP2 calculations (default: 0)"
        ),
    )
    parser.add_argument(
        "--full-domain-check", action="store_true",
        help="append an exact weighted all-AO/all-strong validation row",
    )
    parser.add_argument("--print-geometry", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--max-memory", type=float, default=4000.0,
                        help="PySCF memory limit in MB (default: 4000)")
    args = parser.parse_args(argv)
    if args.ncarbon < 1:
        parser.error("--ncarbon must be at least one")
    if args.frozen_core < 0:
        parser.error("--frozen-core must be non-negative")
    if any(threshold < 0.0 for threshold in args.pair_thresholds):
        parser.error("pair thresholds must be non-negative")
    for name in ("bp_occ", "bp_primary", "bp_ed", "bp_pao", "ed_pao"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.domain_pao < 0.0 or args.occupied_weight < 0.0:
        parser.error("PAO and occupied-weight cutoffs must be non-negative")
    if args.near_pair_distance < 0.0:
        parser.error("--near-pair-distance must be non-negative")
    if args.mp2_block_memory <= 0.0:
        parser.error("--mp2-block-memory must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    config.update("pyscfad_moleintor_opt", True)

    atoms = make_n_alkane(args.ncarbon)
    formula = f"C{args.ncarbon}H{2 * args.ncarbon + 2}"
    print(f"Generated and valence-validated {formula}: {len(atoms)} atoms")
    if args.print_geometry:
        for element, coordinate in atoms:
            print(f"{element:2s} {coordinate[0]:16.10f} "
                  f"{coordinate[1]:16.10f} {coordinate[2]:16.10f}")

    mol = build_molecule(
        atoms, args.basis, verbose=args.verbose, max_memory=args.max_memory
    )
    start = time.perf_counter()
    mf = run_df_rhf(mol, auxbasis=args.auxbasis)
    scf_elapsed = time.perf_counter() - start
    print(
        f"DF-RHF: E = {float(np.asarray(mf.e_tot)):.12f}, "
        f"basis = {args.basis}, auxbasis = {args.auxbasis or 'auto'}, "
        f"AOs = {mol.nao}, time = {scf_elapsed:.1f} s"
    )
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
    if args.frozen_core >= nocc:
        raise ValueError(
            f"--frozen-core={args.frozen_core} leaves no active occupied "
            f"orbitals (RHF has {nocc})"
        )

    reference = None
    if not args.skip_canonical:
        start = time.perf_counter()
        reference = canonical_dfmp2_energy(
            mf, frozen=args.frozen_core, backend=args.canonical_backend
        )
        print(
            f"Canonical DF-MP2 ({args.canonical_backend}, "
            f"frozen core = {args.frozen_core}): "
            f"E_corr = {reference:.12f}, "
            f"time = {time.perf_counter() - start:.1f} s"
        )

    print("Local ED DF transform: integral-direct raw 3c -> Lov")

    rows = []
    for pair_threshold in args.pair_thresholds:
        thresholds = IAOFragmentMP2Thresholds(
            bp_occ=args.bp_occ,
            bp_primary=args.bp_primary,
            bp_ed=args.bp_ed,
            bp_pao=args.bp_pao,
            domain_pao=args.domain_pao,
            ed_pao=args.ed_pao,
            occupied_weight=args.occupied_weight,
            pair_energy=pair_threshold,
            near_pair_distance=args.near_pair_distance,
            mp2_block_memory_mb=args.mp2_block_memory,
        )
        start = time.perf_counter()
        topology = build_iao_fragment_topology(
            mf,
            frozen=args.frozen_core,
            thresholds=thresholds,
            pair_energy_model=args.pair_model,
        )
        topology_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        result = evaluate_iao_fragment_mp2(mf, topology)
        rows.append(summarize_result(
            pair_threshold,
            result,
            reference,
            topology_elapsed,
            time.perf_counter() - start,
        ))

    if args.full_domain_check:
        thresholds = IAOFragmentMP2Thresholds(
            pair_energy=0.0,
            pao_norm=1e-10,
            domain_pao=0.0,
            ed_pao=0.0,
            occupied_weight=1e-12,
        )
        start = time.perf_counter()
        topology = build_iao_fragment_topology(
            mf,
            frozen=args.frozen_core,
            thresholds=thresholds,
            pair_energy_model="all",
            force_full_domains=True,
        )
        topology_elapsed = time.perf_counter() - start
        start = time.perf_counter()
        result = evaluate_iao_fragment_mp2(mf, topology)
        row = summarize_result(
            0.0,
            result,
            reference,
            topology_elapsed,
            time.perf_counter() - start,
        )
        row["threshold"] = "full"
        rows.append(row)

    print()
    print_table(rows)
    print_timing_table(rows)


if __name__ == "__main__":
    main()
