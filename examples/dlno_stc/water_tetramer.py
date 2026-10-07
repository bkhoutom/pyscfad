"""Validate the in-memory weighted STC-MP2 workflow against a water-4 run.

The reference directory contains water_4.xyz, results/ccsdt_water4_boys_mpi4x2
(.json, .npz, .domains.csv, .scf.chk), and scratch/local_mpi4x2/scf_cderi.h5.
The SCF checkpoint and DF integrals are read only. New outputs go to output/.
Run serially or under mpiexec; only rank zero prepares molecular AD.
"""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--selections-dir", type=Path,
                        help="optional existing finite run containing static.h5")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/dlno_stc_water_tetramer"))
    parser.add_argument("--energy-only", action="store_true")
    parser.add_argument("--native-reference", type=Path,
                        help="reuse a current-code native_reference.npz from an earlier run")
    parser.add_argument("--mode", choices=("deterministic", "stochastic"),
                        default="deterministic")
    parser.add_argument("--laplace-points", type=int, default=40)
    parser.add_argument("--laplace-scale", type=float, default=5.5)
    parser.add_argument("--production-samples", type=int, default=4096)
    parser.add_argument("--keep-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--max-memory", type=float, default=3000)
    parser.add_argument("--energy-atol", type=float, default=1e-7)
    parser.add_argument("--gradient-atol", type=float, default=1e-6)
    return parser.parse_args()


def laplace_grid(points, scale):
    """Explicit Gaussian quadrature; its error is measured independently below."""
    if points < 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Laplace points must be >=2 and scale must be positive")
    nodes, weights = np.polynomial.laguerre.laggauss(points)
    return nodes / scale, weights * np.exp(nodes) / scale


def prepare_reference(args):
    from pyscf.scf import chkfile
    from pyscfad import config, gto, scf
    from pyscfad.dlno import _selection, domain
    from pyscfad.dlno.mp2 import _fix_restart_mo_phases
    from pyscfad.dlno_stc.finite import load_finite_static_from_run

    source = args.reference_dir.resolve()
    prefix = source / "results/ccsdt_water4_boys_mpi4x2"
    reference = json.loads(prefix.with_suffix(".json").read_text())
    settings = reference["settings"]
    if reference["lo_type"] != "boys" or settings["force_full_domains"]:
        raise ValueError("reference must contain finite singleton Boys domains")
    config.update("pyscfad_moleintor_opt", True)
    mol = gto.Mole(atom=str(source / settings["xyz_file"]), unit="Angstrom",
                   basis=settings["basis"], max_memory=args.max_memory, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    _, saved_scf = chkfile.load_scf(str(prefix.with_suffix(".scf.chk")))
    co, occ = np.asarray(saved_scf["mo_coeff"]), np.asarray(saved_scf["mo_occ"])
    dm0 = (co * occ) @ co.T
    cderi = source / "scratch/local_mpi4x2/scf_cderi.h5"

    def build_mf(current_mol):
        mf = scf.RHF(current_mol).density_fit(auxbasis=settings["auxbasis"])
        mf.with_df.max_memory = args.max_memory
        mf.with_df.attach_outcore_cderi(str(cderi))
        mf.conv_tol = settings["mf_conv_tol"]
        mf.conv_tol_grad = settings["mf_conv_tol_grad"]
        mf.chkfile = None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("reference water-tetramer DF-RHF did not converge")
        return _fix_restart_mo_phases(mf)

    mf = build_mf(mol)
    np.testing.assert_allclose(float(mf.e_tot), reference["reference_hf_energy"],
                               rtol=0, atol=1e-7)
    if args.selections_dir is not None:
        static = load_finite_static_from_run(
            mf, args.selections_dir, frozen=settings["frozen"],
        )
    else:
        topology = domain.build_domain_topology(
            mf, frozen=settings["frozen"], lo_type="boys",
            lo_kwargs=reference["lo_kwargs"], force_full_domains=False,
            pair_energy_model=settings["pair_energy_model"],
            thresholds=domain.DLNOThresholds(**reference["thresholds"]),
        )
        static = _selection.build_domain_selections(mf, topology)
        del topology
    with prefix.with_suffix(".domains.csv").open(newline="") as handle:
        expected = list(csv.DictReader(handle))
    dimensions = []
    for fragment, row in zip(static.fragments, expected):
        actual = [fragment.fragment_index, len(fragment.extended_ao_indices),
                  len(fragment.strong_occ_metric_keep),
                  len(fragment.strong_virtual.metric_keep)]
        wanted = [int(row[key]) for key in
                  ("fragment_index", "n_domain_ao", "n_domain_occ", "n_domain_vir")]
        if actual != wanted:
            raise AssertionError(f"domain dimensions differ: {actual} != {wanted}")
        if (fragment.extended_atoms.tolist() != json.loads(row["extended_atoms"])
                or fragment.strong_fragments.tolist()
                != json.loads(row["strong_fragments"])):
            raise AssertionError(f"domain {fragment.fragment_index} selections differ")
        dimensions.append(actual)
    if len(dimensions) != len(expected) or len(dimensions) != 16:
        raise AssertionError("reference must contain all 16 Boys target domains")
    if not any(row[1] < mol.nao for row in dimensions):
        raise AssertionError("fixture does not contain genuinely finite AO domains")
    nweak = int(np.count_nonzero(np.triu(~static.strong_mask, k=1)))
    if nweak != reference["details"]["n_weak_pair_terms"] or nweak == 0:
        raise AssertionError("weak-pair coverage differs from the reference")
    with np.load(prefix.with_suffix(".npz")) as data:
        gradient = data["gradient"].copy()
    return mol, mf, build_mf, static, reference, gradient, dimensions, nweak


def current_native_reference(args, mol, mf, build_mf, static, historical):
    """Generate the native baseline using this checkout's corrected weak terms.

The supplied saved run predates the Oct 2 multipole prefactor/sign correction.
Its geometry, SCF, and selections remain useful; its weak energy and gradient
must not be treated as a current-code reference.
"""
    import jax
    from pyscfad import config_update
    from pyscfad.dlno import multipole
    from pyscfad.dlno._restart import scientific_digest
    from pyscfad.dlno.mp2 import (
        _add_cotangent, correlation_energy, correlation_value_and_grad,
    )
    from pyscfad.dlno_stc._workflow import revision_status

    # Include local native dependencies, including uncommitted changes. A git
    # revision alone does not identify the code used in an editable checkout.
    source_root = Path(multipole.__file__).parents[1]
    native_sources = hashlib.sha256()
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        if "test" in relative.parts or relative.parts[0] == "dlno_stc":
            continue
        native_sources.update(str(relative).encode() + b"\0")
        native_sources.update(source.read_bytes())
    identity = {"static_digest": scientific_digest(static),
                "coords_bohr": np.asarray(mol.atom_coords()).tolist(),
                "native_sources_sha256": native_sources.hexdigest(),
                "native_multipole_sha256": hashlib.sha256(
                    Path(multipole.__file__).read_bytes()).hexdigest()}
    started = time.perf_counter()
    if args.native_reference is not None:
        info = json.loads(args.native_reference.with_suffix(".json").read_text())
        if any(info.get(key) != value for key, value in identity.items()):
            raise ValueError("native reference does not match the fixed selections/current multipole code")
        with np.load(args.native_reference) as data:
            corr, hf = float(data["correlation_energy"]), float(data["hf_energy"])
            gradient = data["gradient"].copy() if "gradient" in data else None
        if not args.energy_only and gradient is None:
            raise ValueError("native reference contains no molecular gradient")
        np.testing.assert_allclose(hf, float(mf.e_tot), rtol=0, atol=1e-8)
    else:
        print("Computing current-code native MP2 reference", flush=True)
        hf = float(mf.e_tot)
        gradient = None
        if args.energy_only:
            corr = float(correlation_energy(mf, static))
            info = {}
        else:
            with (config_update("pyscfad_scf_implicit_diff", True),
                  config_update("pyscfad_scf_first_order_custom", False)):
                native_mf, scf_pullback = jax.vjp(build_mf, mol)
            corr_value, mf_bar, details = correlation_value_and_grad(
                native_mf, static, return_details=True,
            )
            hf_value, hf_pullback = jax.vjp(lambda current: current.e_tot, native_mf)
            hf_bar, = hf_pullback(jax.numpy.ones_like(hf_value))
            mf_bar = jax.tree_util.tree_map(_add_cotangent, mf_bar, hf_bar)
            mol_bar, = scf_pullback(mf_bar)
            jax.block_until_ready((corr_value, hf_value, mol_bar))
            corr, hf = float(corr_value), float(hf_value)
            gradient = np.asarray(mol_bar.coords).copy()
            info = {"strong_energy": details.e_strong, "weak_energy": details.e_weak}
            del native_mf, scf_pullback, hf_pullback, mf_bar, hf_bar, mol_bar
        revision, dirty = revision_status()
        info.update(identity, code_revision=revision, dirty_worktree=dirty,
                    correlation_energy=corr, hf_energy=hf, total_energy=corr + hf,
                    historical_correlation_energy=historical["details"]["e_corr"],
                    historical_correlation_difference=corr - historical["details"]["e_corr"],
                    elapsed_seconds=time.perf_counter() - started)
        arrays = {"correlation_energy": corr, "hf_energy": hf}
        if gradient is not None:
            arrays["gradient"] = gradient
        np.savez(args.output_dir / "native_reference.npz", **arrays)
        (args.output_dir / "native_reference.json").write_text(
            json.dumps(info, indent=2) + "\n")
        print("Native reference:", json.dumps(info), flush=True)
    current = dict(historical)
    current["details"] = dict(historical["details"], e_corr=corr)
    current["energy"] = corr + hf
    current["reference_hf_energy"] = hf
    gc.collect()
    return current, gradient, info


def quadrature_check(mf, static, roots, weights):
    """Check 1/d and 1/d² across actual local spectral denominator bounds."""
    from pyscfad.dlno.dlno_base import rebuild_domain_data
    from pyscfad.dlno_stc.domain import build_local_strong_ed_domain

    common = rebuild_domain_data(mf, static)
    lower, upper = np.inf, 0.0
    for index, fragment in enumerate(static.fragments):
        domain = build_local_strong_ed_domain(common, static, index)
        ao = fragment.extended_ao_indices
        fock = np.asarray(common.fock)[np.ix_(ao, ao)]
        co, cv = np.asarray(domain.occupied_coeff), np.asarray(domain.virtual_coeff)
        if cv.shape[1] == 0:
            continue
        eo = np.linalg.eigvalsh(co.T @ fock @ co)
        ev = np.linalg.eigvalsh(cv.T @ fock @ cv)
        lower = min(lower, 2 * (ev.min() - eo.max()))
        upper = max(upper, 2 * (ev.max() - eo.min()))
        del domain
    if not np.isfinite(lower) or lower <= 0:
        raise ValueError("tetramer does not have positive local MP2 denominators")
    denominators = np.geomspace(lower, upper, 1000)
    values = np.exp(-denominators[:, None] * roots)
    inverse_error = float(np.max(np.abs(denominators * (values @ weights) - 1)))
    derivative_error = float(np.max(np.abs(
        denominators**2 * (values @ (weights * roots)) - 1)))
    return {"denominator_min": float(lower), "denominator_max": float(upper),
            "inverse_max_relative_error": inverse_error,
            "inverse_derivative_max_relative_error": derivative_error}


def run(args, comm):
    from pyscfad.dlno_stc.driver import kernel, value_and_grad

    rank = 0 if comm is None else comm.Get_rank()
    roots, weights = laplace_grid(args.laplace_points, args.laplace_scale)
    controls = {"mode": args.mode, "laplace_roots": roots,
                "laplace_weights": weights, "virtual_block_size": 64,
                "auxiliary_group_size": 32, "global_seed": args.seed}
    if args.mode == "stochastic":
        controls.update(production_samples=args.production_samples,
                        virtual_keep_fraction=args.keep_fraction)
    error = None
    if rank == 0:
        try:
            prepared = prepare_reference(args)
            mol, mf, build_mf, static, reference, reference_gradient, dims, nweak = prepared
            quad = quadrature_check(mf, static, roots, weights)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print("Quadrature check:", json.dumps(quad), flush=True)
            historical_reference = reference
            reference, current_gradient, native_info = current_native_reference(
                args, mol, mf, build_mf, static, historical_reference,
            )
            if current_gradient is not None:
                reference_gradient = current_gradient
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if comm is not None:
        error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"reference preparation failed: {error}")
    if rank != 0:
        mol = mf = build_mf = static = None
    started = time.perf_counter()
    corr = kernel(mf, static, controls=controls, comm=comm)
    gradient_result = None
    if not args.energy_only:
        if rank == 0:
            # The gradient entry point creates its own SCF and shared AD tapes.
            del prepared, mf
            gc.collect()
        gradient_result = value_and_grad(
            mol, build_mf, static, controls=controls, comm=comm, include_hf=True,
        )
    if rank != 0:
        return
    corr = float(corr)
    report = {"mode": args.mode, "mpi_size": 1 if comm is None else comm.Get_size(),
              "basis": reference["settings"]["basis"],
              "frozen": reference["settings"]["frozen"], "n_domains": len(dims),
              "n_weak_pairs": nweak, "domain_sizes": dims,
              "laplace_points": args.laplace_points, "laplace_scale": args.laplace_scale,
              "quadrature": quad, "energy_unit": "Eh", "gradient_unit": "Eh/bohr",
              "correlation_energy": corr,
              "reference_correlation_energy": reference["details"]["e_corr"],
              "correlation_energy_error": corr - reference["details"]["e_corr"],
              "historical_correlation_energy": historical_reference["details"]["e_corr"],
              "historical_correlation_difference": corr - historical_reference["details"]["e_corr"],
              "native_reference": native_info,
              "elapsed_seconds": time.perf_counter() - started,
              "energy_tolerance": args.energy_atol, "gradient_tolerance": args.gradient_atol}
    arrays = {"correlation_energy": corr}
    if gradient_result is not None:
        energy, bar = gradient_result
        gradient = np.asarray(bar.coords)
        difference = gradient - reference_gradient
        report.update(total_energy=float(energy), reference_total_energy=reference["energy"],
                      total_energy_error=float(energy) - reference["energy"],
                      max_gradient_error=float(np.max(np.abs(difference))),
                      rms_gradient_error=float(np.sqrt(np.mean(difference**2))),
                      gradient_sum=gradient.sum(axis=0).tolist())
        arrays.update(energy=float(energy), gradient=gradient,
                      reference_gradient=reference_gradient, gradient_error=difference)
    np.savez(args.output_dir / "results.npz", **arrays)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    # Sampling uncertainty is reported by the driver; deterministic tolerances
    # assess the interface and quadrature without conflating stochastic noise.
    if args.mode == "deterministic":
        np.testing.assert_allclose(corr, reference["details"]["e_corr"],
                                   atol=args.energy_atol, rtol=0)
        if gradient_result is not None:
            np.testing.assert_allclose(energy, reference["energy"],
                                       atol=args.energy_atol, rtol=0)
            np.testing.assert_allclose(gradient, reference_gradient,
                                       atol=args.gradient_atol, rtol=0)
        print("Water-tetramer deterministic workflow validation PASS", flush=True)


if __name__ == "__main__":
    args = arguments()
    try:
        from mpi4py import MPI
        communicator = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
    except ImportError:
        communicator = None
    try:
        run(args, communicator)
    except Exception:
        if communicator is not None:
            communicator.Abort(1)
        raise
