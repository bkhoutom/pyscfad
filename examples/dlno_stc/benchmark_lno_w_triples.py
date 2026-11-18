"""Resumable smooth LNO(W)-CCSD(T) energy/force benchmark.

The serial queue runs one realization per process with deterministic domain/LIS
selection. Fresh calculations share one differentiable SCF between selection
and production, after an independent reference SCF. Each result is written
immediately after completion. Checkpoint restarts restore the converged SCF
and rebuild implicit response without repeating either SCF solve.
"""

import argparse
from dataclasses import asdict
import faulthandler
import gc
import json
import math
import os
from pathlib import Path
import time

from benchmark_affinity import (configure_cpu_workload, check_worker_affinity,
                                ensure_worker_stack)

os.environ["PYSCFAD_DLNO_RESOURCE_PROFILE"] = "1"
faulthandler.enable()


_SGX_DEFAULTS = dict(integral_backend='auto', screening='per_pair',
                     mixed_precision=False, energy_tol='auto',
                     potential_tol='auto', fp32_multiplier=1e5)


def _sgx_tolerance(value):
    if value == 'auto':
        return value
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('SGX tolerance must be auto or finite and positive')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scf-jk', choices=('df', 'dfj-sgx'), default='df',
                        help='SCF and response J/K backend; dfj-sgx avoids global DF factors')
    parser.add_argument('--sgx-grid-level', type=int, choices=range(8), default=2)
    parser.add_argument('--sgx-fit-overlap', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sgx-integral-backend', choices=('auto', 'generated', 'libcint'),
                        default=_SGX_DEFAULTS['integral_backend'])
    parser.add_argument('--sgx-screening', choices=('static', 'adaptive', 'per_pair'),
                        default=_SGX_DEFAULTS['screening'])
    parser.add_argument('--sgx-mixed-precision', action=argparse.BooleanOptionalAction,
                        default=_SGX_DEFAULTS['mixed_precision'],
                        help='use mixed precision for forward, response and geometry SGX kernels')
    parser.add_argument('--sgx-energy-tol', type=_sgx_tolerance,
                        default=_SGX_DEFAULTS['energy_tol'])
    parser.add_argument('--sgx-potential-tol', type=_sgx_tolerance,
                        default=_SGX_DEFAULTS['potential_tol'])
    parser.add_argument('--sgx-fp32-multiplier', type=float,
                        default=_SGX_DEFAULTS['fp32_multiplier'])
    parser.add_argument("--scf-df-precond", action="store_true",
                        help="Enable host SCF response preconditioning with separate total/workspace budgets")
    parser.add_argument("--threads", type=int, required=True,
                        help="CPU affinity budget and numerical worker count")
    parser.add_argument("--xyz-file", type=Path, required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--frozen", type=int, required=True)
    parser.add_argument("--boys-grouping", choices=("system", "atom", "molecule"), required=True)
    parser.add_argument("--domain-thresholds", type=Path,
                        help="JSON object of BoysDomainThresholds overrides; omitted fields use library defaults")
    parser.add_argument("--targets-meh", type=float, nargs="+", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--job-indices", type=int, nargs="+",
                        help="run these zero-based stochastic jobs from the full target/seed grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="save serial CC progress; use a separate directory per realization")
    parser.add_argument("--resume", action="store_true",
                        help="resume compatible progress from --checkpoint-dir")
    parser.add_argument("--pilot-samples", type=int, default=20000)
    parser.add_argument("--max-memory-mb", type=float, default=800000)
    parser.add_argument("--df-storage", choices=("incore", "outcore"), default="incore",
                        help="store DF factors in RAM or stream them and their response from disk")
    parser.add_argument("--df-builder", choices=("standard", "direct"), default="standard",
                        help="outcore builder: standard repacks a scratch tensor; direct writes final-layout panels")
    parser.add_argument("--df-memory-mb", type=float, default=4000,
                        help="working-memory budget for streamed DF operations")
    parser.add_argument("--fragment-df-mode", choices=("direct", "half"),
                        default=os.environ.get('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'half'),
                        help="regenerate local integrals or retain H_ED for one-pass reuse (default)")
    parser.add_argument("--cc-conv-tol", type=float, default=1e-7)
    parser.add_argument("--cc-conv-tol-normt", type=float, default=1e-5)
    parser.add_argument("--lno-thresh-occ", type=float, default=1e-4,
                        help="occupied W-eigenvalue cutoff for LIS selection")
    parser.add_argument("--lno-thresh-vir", type=float, default=1e-5,
                        help="virtual W-eigenvalue cutoff for LIS selection")
    parser.add_argument("--triples-method", choices=("smooth",),
                        default="smooth")
    parser.add_argument("--triples-integral-transform", choices=("df", "rotate"),
                        default="df", help="dress DF factors or rotate cached CCSD integrals")
    parser.add_argument("--triples-laplace-points", type=int, default=4)
    parser.add_argument("--smooth-trace-exponent", type=float, default=5.)
    parser.add_argument("--triples-variance-allocation", choices=("occupied_count", "mp2_energy"),
                        default="occupied_count")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--deterministic-only", action="store_true")
    mode.add_argument("--stochastic-only", action="store_true")
    mode.add_argument("--compare-solvers", action="store_true",
                      help="run the deterministic reference and requested stochastic jobs")
    args = parser.parse_args()
    if args.resume and args.checkpoint_dir is None:
        parser.error('--resume requires --checkpoint-dir')
    if args.checkpoint_dir is not None and (args.compare_solvers or args.stochastic_only):
        parser.error('this checkpoint runner currently supports --deterministic-only')
    if args.boys_grouping == "molecule":
        args.boys_grouping = "system"
    sgx_options = {key: getattr(args, 'sgx_' + key) for key in _SGX_DEFAULTS}
    sgx_settings = {'sgx_' + key: value for key, value in sgx_options.items()}
    if not math.isfinite(args.sgx_fp32_multiplier) or args.sgx_fp32_multiplier <= 0:
        parser.error('--sgx-fp32-multiplier must be finite and positive')
    if (args.sgx_integral_backend == 'libcint'
            and (args.sgx_screening != 'static' or args.sgx_mixed_precision)):
        parser.error('adaptive screening and mixed precision require the generated SGX backend')
    if args.scf_jk == 'dfj-sgx' and args.scf_df_precond:
        parser.error('--scf-df-precond requires --scf-jk df')
    os.environ['PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE'] = args.fragment_df_mode
    if args.triples_variance_allocation != 'occupied_count':
        parser.error('W pipeline requires occupied_count variance allocation')
    if args.threads < 1:
        parser.error("--threads must be positive")
    benchmark_cpus = configure_cpu_workload(args.threads)
    if not (0 < args.cc_conv_tol < float("inf")
            and 0 < args.cc_conv_tol_normt < float("inf")):
        parser.error("CCSD convergence tolerances must be finite and positive")
    if not all(math.isfinite(x) and x >= 0
               for x in (args.lno_thresh_occ, args.lno_thresh_vir)):
        parser.error("LIS cutoffs must be finite and nonnegative")
    args.deterministic_only = not (args.stochastic_only or args.compare_solvers)
    if not args.deterministic_only and (not args.targets_meh or not args.seeds):
        parser.error("stochastic jobs require --targets-meh and --seeds")
    frame_mode = "full"
    if not math.isfinite(args.df_memory_mb) or args.df_memory_mb <= 0:
        parser.error("DF memory budget must be finite and positive")
    if args.df_builder == 'direct' and args.df_storage != 'outcore':
        parser.error('--df-builder direct requires --df-storage outcore')
    if any(x <= 0 for x in args.targets_meh) or len(set(args.targets_meh)) != len(args.targets_meh):
        parser.error("targets must be positive and distinct")
    if any(x < 0 for x in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be nonnegative and distinct")
    total_jobs = len(args.targets_meh) * len(args.seeds)
    if args.job_indices is not None and (
            len(set(args.job_indices)) != len(args.job_indices)
            or any(i < 0 or i >= total_jobs for i in args.job_indices)):
        parser.error(f"job indices must be distinct integers from 0 to {total_jobs-1}")

    import jax
    import numpy as np
    import pyscfad
    from pyscf import gto as pgto, scf as pscf
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.lno.ccsd import RCCSD
    from pyscfad.tools.linear_solver import eager_gmres

    pyscfad.config.update('pyscfad_scf_df_precond', args.scf_df_precond)
    # Large coordinate-gradient runs need the native packed-integral VJP.
    # The generic JAX AO2MO pullback expands entire auxiliary blocks to s1.
    pyscfad.config.update('pyscfad_moleintor_opt', True)
    if args.scf_df_precond:
        # Public CLI totals are decimal MB; this legacy cache cap uses MiB.
        # The preconditioner further limits storage to half actual headroom.
        os.environ.setdefault('PYSCFAD_PRECOND_CACHE_MEMORY_MB',
                              str(args.max_memory_mb * 1e6 / 1024**2))

    # Molecule grouping defines one full domain; atom grouping uses finite domains.
    full_domain = args.boys_grouping == "system"
    threshold_options = (json.loads(args.domain_thresholds.read_text())
                         if args.domain_thresholds is not None else {})
    domain_thresholds = dlno_stc.BoysDomainThresholds(**threshold_options)
    snapshot_root = os.environ.get("LNO_BENCHMARK_SOURCE_ROOT")
    if snapshot_root and not Path(pyscfad.__file__).resolve().is_relative_to(Path(snapshot_root).resolve()):
        raise RuntimeError("benchmark did not import its pinned source snapshot")

    RCCSD.conv_tol = args.cc_conv_tol
    RCCSD.conv_tol_normt = args.cc_conv_tol_normt
    RCCSD.max_cycle = 200
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "settings.json").exists():
        previous = json.loads((output / "settings.json").read_text())
        if previous.get('domain_thresholds') != asdict(domain_thresholds):
            raise ValueError('domain thresholds differ; use a new output directory')
        for key, value, old_default in (
                ('scf_jk', args.scf_jk, 'df'),
                ('sgx_grid_level', args.sgx_grid_level, 2),
                ('sgx_fit_overlap', args.sgx_fit_overlap, True),
                ('lis_eigenspace', 'domain', 'global'),
                ('lis_space', 'domain', 'global'),
                ('triples_variance_allocation', args.triples_variance_allocation, 'mp2_energy')):
            if previous.get(key, old_default) != value:
                raise ValueError(f'{key} differs; use a new output directory')
        for key, value in sgx_options.items():
            if previous.get('sgx_' + key, _SGX_DEFAULTS[key]) != value:
                raise ValueError(f'SGX {key} differs; use a new output directory')
        if previous.get('scf_df_precond', False) != args.scf_df_precond:
            raise ValueError('SCF preconditioning differs; use a new output directory')
        if previous.get('triples_integral_transform', 'df') != args.triples_integral_transform:
            raise ValueError('triples integral transform differs; use a new output directory')
        requested = dict(cc_conv_tol=args.cc_conv_tol,
                         cc_conv_tol_normt=args.cc_conv_tol_normt,
                         scf_jk_workspace_memory_mb=args.df_memory_mb,
                         lno_thresh_occ=args.lno_thresh_occ,
                         lno_thresh_vir=args.lno_thresh_vir,
                         triples_mode=args.triples_method,
                         full_domain=full_domain,
                         cpu_affinity=benchmark_cpus, threads=args.threads)
        if previous.get("smooth_frame_mode", "clustered") != frame_mode:
            raise ValueError("smooth frame mode differs from existing results; use a new output directory")
        cache_enabled = os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "true").lower() != "false"
        if ("jax_persistent_cache_enabled" in previous
                and previous["jax_persistent_cache_enabled"] != cache_enabled):
            raise ValueError("compilation-cache policy differs from existing results")
        for key, value in requested.items():
            if previous.get(key) != value:
                raise ValueError(f"{key} differs from existing results; use a new output directory")
        if (previous.get("df_builder", "standard") != args.df_builder or
                previous.get("df_storage", "incore") != args.df_storage or
                (args.df_storage == "outcore" and
                 previous.get("df_memory_mb") != args.df_memory_mb)):
            raise ValueError("DF storage settings differ; use a new output directory")
    xyz = args.xyz_file.resolve()
    started = time.perf_counter()

    def report(message):
        print(f"[BENCHMARK {time.perf_counter()-started:.1f}s] {message}", flush=True)

    def save(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    basis = args.basis
    auxbasis = None
    if args.basis.lower().replace(" ", "") == "def2-sv(p)":
        # Explicit H/C definition: def2-SVP on carbon and its two s shells
        # on hydrogen. Do not let name normalization substitute def2-SVP.
        basis = {"C": pgto.basis.load("def2-svp", "C"),
                 "H": [shell for shell in pgto.basis.load("def2-svp", "H")
                       if shell[0] == 0]}
        auxbasis = "def2-svp-jkfit"
    mol = gto.Mole(atom=str(xyz), basis=basis, unit="Angstrom",
                   max_memory=args.max_memory_mb, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    if args.boys_grouping == "system":
        assert elements == ["C"] * 5 + ["H"] * 6
        expected_groups = 1
    else:
        carbons = elements.count("C")
        hydrogens = elements.count("H")
        oxygens = elements.count("O")
        if oxygens > 0 and hydrogens == 2 * oxygens and oxygens + hydrogens == len(elements):
            if args.frozen != oxygens:
                raise ValueError("water benchmark requires one frozen oxygen core per water")
            expected_groups = oxygens
        elif carbons >= 2 and hydrogens == 2 * carbons + 2 and carbons + hydrogens == len(elements):
            expected_groups = carbons
        else:
            raise ValueError("atom-grouped benchmark requires (H2O)_n or C_n H_(2n+2)")
    saved_scf = saved_selection = None
    boys_options = dict(init_guess="atomic", conv_tol=1e-10,
                        gmres_options=dict(restart=120, maxiter=100))
    if args.resume:
        from pyscfad.dlno_stc._ccsd_restart_reference import prepare_resume, load_scf_reference
        args.resume = prepare_resume(args.checkpoint_dir, True)
        if args.resume or (args.checkpoint_dir / '.scf-reference.h5').is_file():
            saved_scf = load_scf_reference(args.checkpoint_dir, mol)
            report('Restoring converged SCF; no cold or warm SCF iterations')
        if args.resume:
            saved_selection = ccsd.load_static_selections(
                args.checkpoint_dir, mol,
                domain_options=dict(frozen=args.frozen, thresholds=domain_thresholds,
                    boys_grouping=args.boys_grouping, full_domain=full_domain,
                    lo_kwargs=boys_options, pair_energy_model='multipole'),
                selection_options=dict(thresh_occ=args.lno_thresh_occ,
                    thresh_vir=args.lno_thresh_vir, internal_rank_threshold=1e-6,
                    triples_laplace_points=args.triples_laplace_points))
            report('Loaded saved Boys/domain/LIS selections; no selection searches')
    reference_scf_mode = 'restored' if saved_scf is not None else 'solved'
    refmol = pgto.M(atom=str(xyz), basis=basis, unit="Angstrom",
                     max_memory=args.max_memory_mb, verbose=0)
    ref = pscf.RHF(refmol).density_fit(auxbasis=auxbasis)
    if args.scf_jk == 'dfj-sgx':
        from pyscfad.sgx import sgx_fit
        ref = sgx_fit(pscf.RHF(refmol), auxbasis=auxbasis,
                      grid_level=args.sgx_grid_level, fit_ovlp=args.sgx_fit_overlap,
                      df_memory_mb=args.df_memory_mb, **sgx_options)
        report(f'Integral-direct DF-J / SGX-K (grid {args.sgx_grid_level}); no global SCF DF factors')
    cderi_path = (output / "scf_cderi.h5"
                  if args.scf_jk == 'df' and args.df_storage == "outcore" else None)
    df_setup_seconds = None
    if cderi_path is not None:
        # The response primitives require a single packed j3c dataset.
        # Build it once at this geometry and attach it to each traced SCF;
        # their custom VJPs still differentiate the integrals and metric.
        ref.with_df._compatible_format = True
        ref.with_df._cderi_to_save = str(cderi_path)
        ref.with_df.max_memory = args.df_memory_mb
        report(f"Building disk-backed DF factors ({args.df_builder}); working-memory budget {args.df_memory_mb:g} MB")
        stage_start = time.perf_counter()
        if args.df_builder == 'direct':
            from pyscf.df.addons import make_auxmol
            from pyscfad.df import direct_outcore
            ref.with_df.auxmol = make_auxmol(refmol, ref.with_df.auxbasis)
            direct_outcore.cholesky_eri(refmol, cderi_path, auxmol=ref.with_df.auxmol,
                                        max_memory=args.df_memory_mb, verbose=4)
            ref.with_df._cderi = str(cderi_path)
        else:
            ref.with_df.build()
        df_setup_seconds = time.perf_counter() - stage_start
        report(f"DF factors ready in {df_setup_seconds:.1f} s")
    ref.conv_tol, ref.conv_tol_grad = 1e-9, 1e-6
    stage_start = time.perf_counter()
    if saved_scf is None:
        ref.kernel()
        reference_scf_seconds = time.perf_counter() - stage_start
    else:
        for name, value in saved_scf['canonical'].items():
            setattr(ref, name, value)
        ref.converged = True
        reference_scf_seconds = 0.0
    if not ref.converged:
        raise RuntimeError("reference RHF failed")
    dm0 = np.asarray(ref.make_rdm1())
    if args.scf_jk == 'dfj-sgx':
        # The remaining reference checks use only e_tot. Release its separate
        # metric/grid caches before allocating the differentiable SCF backend.
        ref.with_df.reset()
    report(f"Reference SCF complete in {reference_scf_seconds:.1f} s")

    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=auxbasis)
        # Correlation DF reads this field independently of the SGX/J-K budget.
        mf.with_df.max_memory = args.df_memory_mb
        if args.scf_jk == 'dfj-sgx':
            mf = sgx_fit(mf, auxbasis=auxbasis, grid_level=args.sgx_grid_level,
                         fit_ovlp=args.sgx_fit_overlap, df_memory_mb=args.df_memory_mb,
                         **sgx_options)
        # This is scratch space, not a ceiling on the whole process RSS.
        # Keep J/K panels useful when retained correlation state exceeds it.
        mf.with_df.jk_workspace_memory = args.df_memory_mb
        if args.scf_df_precond:
            mf.with_df.precond_total_memory = args.max_memory_mb
            mf.with_df.precond_workspace_memory = args.df_memory_mb
        if cderi_path is not None:
            mf.with_df.incore = False
            mf.with_df.max_memory = args.df_memory_mb
            mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-9, 1e-6, None
        if saved_scf is not None:
            from pyscfad.dlno_stc._ccsd_restart_reference import restore_scf_reference
            if cderi_path is not None:
                with eager_gmres():
                    restore_scf_reference(mf, saved_scf)
            else:
                restore_scf_reference(mf, saved_scf)
        elif cderi_path is not None:
            # The SCF response operator reads disk blocks. Construct its
            # GMRES solver outside JIT, preserving the existing tolerances.
            with eager_gmres():
                mf.kernel(dm0=dm0)
        else:
            mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError("differentiable RHF failed")
        if saved_scf is None and args.checkpoint_dir is not None:
            from pyscfad.ops import stop_trace
            from pyscfad.scf._response_data import lookup
            from pyscfad.dlno_stc._ccsd_restart_reference import save_early_scf_reference
            data = lookup(mf)
            stop_trace(lambda host, fock: save_early_scf_reference(
                args.checkpoint_dir, host, reference_energy=float(ref.e_tot),
                fock=fock))(mf, None if data is None else data[1])
        return mf

    baseline = dict(mode="deterministic", threads=args.threads, global_seed=666)
    matrix = dict(mode="deterministic", threads=args.threads,
                  memory_limit_mb=args.max_memory_mb)
    selection, settings = {}, {}
    initial_ad_scf_seconds = 0.0
    production_started = None

    def build_static(mf):
        """Share production SCF; later realizations reuse the same fixed labels."""
        if abs(float(ref.e_tot) - float(mf.e_tot)) > 1e-7:
            raise RuntimeError("reference and differentiable SCF disagree")
        if selection:
            return selection['static']
        if saved_selection is None:
            stage_start = time.perf_counter()
            domains = dlno_stc.build_domain_selections(
                mf, frozen=args.frozen, full_domain=full_domain,
                thresholds=domain_thresholds, boys_grouping=args.boys_grouping,
                lo_kwargs=boys_options)
            domain_selection_seconds = time.perf_counter() - stage_start
            report(f"Domain selection complete in {domain_selection_seconds:.1f} s")
        else:
            domains = saved_selection.mp2_static
            domain_selection_seconds = 0.0
        if len(domains.physical_groups) != expected_groups:
            raise RuntimeError(f"expected {expected_groups} physical fragments")
        mp2_domains = []
        for index, members in enumerate(domains.physical_groups):
            fragment = domains.fragments[members[0]]
            mp2_domains.append(dict(fragment=index, target_orbitals=list(members),
                primary_atoms=fragment.primary_atoms.tolist(),
                extended_atoms=fragment.extended_atoms.tolist(),
                pao_center_atoms=fragment.pao_center_atoms.tolist(),
                nao=len(fragment.extended_ao_indices),
                nocc=len(fragment.strong_occ_metric_keep),
                nvir=(None if fragment.virtual_anchor_columns is None else
                      len(fragment.virtual_anchor_columns))))
        domain_report = dict(full_domain=full_domain,
            ed_selection_pending=domains.ed_selection_pending,
            thresholds=asdict(domains.thresholds), pair_energy_model=domains.pair_energy_model,
            domain_convention=domains.domain_convention,
            physical_fragments=expected_groups, molecular_nao=mol.nao,
            total_active_occ=len(domains.active_occ_indices),
            total_active_vir=len(domains.active_vir_indices),
            full_ao_domains=sum(d['nao'] == mol.nao for d in mp2_domains),
            weak_orbital_pairs=int(np.count_nonzero(np.triu(~(domains.strong_mask & domains.strong_mask.T), 1))),
            fragments=mp2_domains)
        save(output / "domain_selection.json", domain_report)
        report(f"MP2 domain selection: full_domain={full_domain}; "
               f"{domain_report['full_ao_domains']}/{expected_groups} domains span all AOs; "
               f"(AO, occ, vir)={[(d['nao'], d['nocc'], d['nvir']) for d in mp2_domains]}")
        roots, weights = domains.mp2_grid
        diagnostics = domains.mp2_grid_diagnostics
        selection_start = time.perf_counter()
        report(f"Configuring W-LIS selection: occupied cutoff={args.lno_thresh_occ:g}, "
               f"virtual cutoff={args.lno_thresh_vir:g}")
        selection_options = dict( thresh_occ=args.lno_thresh_occ,

                                 thresh_vir=args.lno_thresh_vir,
                                 baseline_controls=baseline, matrix_controls=matrix,
                                 merge_identical=True)
        if saved_selection is None:
            selected = ccsd.build_static_selections(
                mf, domains,
                triples_laplace_points=args.triples_laplace_points,
                **selection_options)
            selection_seconds = time.perf_counter() - selection_start
        else:
            selected = saved_selection
            selection_seconds = 0.0
        dimensions = [ccsd._fragment_dimensions(selected, i) for i in range(expected_groups)]
        frame_blocks = []
        affinity = check_worker_affinity(benchmark_cpus)
        settings["lis_space"] = "domain"  # Fixed scientific identity for saved references.
        settings.update(geometry=str(xyz), elements=elements, basis=args.basis,
                        auxbasis=auxbasis or "auto", frozen=args.frozen, boys_grouping=args.boys_grouping,
                        full_domain=full_domain, physical_fragments=expected_groups,
                        domain_thresholds=asdict(domains.thresholds), mp2_domains=mp2_domains,
                        mp2_energy_source="w_matrix_trace", mp2_response="joint_energy_w_pullback",
                        pair_energy_model=domains.pair_energy_model,
                        lno_thresh_occ=args.lno_thresh_occ, lno_thresh_vir=args.lno_thresh_vir,


                        lis_eigenspace='domain',
                        triples_variance_allocation=args.triples_variance_allocation,
                        selection_timer_scope='plan_only',
                        fragment_occupied_counts=[len(members) for members in domains.physical_groups],
                        total_active_occupied=len(domains.active_occ_indices),
                        nao=mol.nao, lis_dimensions=dimensions, threads=args.threads,
                        cpu_affinity=benchmark_cpus, affinity_check=affinity,
                        max_memory_mb=args.max_memory_mb, cc_conv_tol=args.cc_conv_tol,
                        df_storage=args.df_storage, df_memory_mb=args.df_memory_mb,
                        correlation_df_memory_mb=mf.with_df.max_memory,
                        moleintor_opt=bool(pyscfad.config.moleintor_opt),
                        df_builder=args.df_builder,
                        omp_dynamic=os.environ.get('OMP_DYNAMIC'),
                        mkl_dynamic=os.environ.get('MKL_DYNAMIC'),
                        shared_df_cache_mode=args.fragment_df_mode,
                        scf_jk=args.scf_jk, sgx_grid_level=args.sgx_grid_level,
                        sgx_fit_overlap=args.sgx_fit_overlap,
                        global_scf_df_factors=args.scf_jk == 'df',
                        scf_response_solver=("eager_gmres" if cderi_path is not None
                                             or args.scf_jk == 'dfj-sgx' else "default"),
                        scf_jk_workspace_memory_mb=mf.with_df.jk_workspace_memory,
                        scf_df_precond=args.scf_df_precond,
                        scf_precond_total_memory_mb=args.max_memory_mb if args.scf_df_precond else None,
                        scf_precond_workspace_memory_mb=args.df_memory_mb if args.scf_df_precond else None,
                        scf_precond_cache_cap_mib=os.environ.get('PYSCFAD_PRECOND_CACHE_MEMORY_MB'),
                        df_setup_seconds=df_setup_seconds,
                        reference_scf_seconds=reference_scf_seconds,
                        reference_scf_mode=reference_scf_mode,
                        initial_ad_scf_seconds=initial_ad_scf_seconds,
                        selection_in_production=production_started is not None,
                        selection_source=('checkpoint' if saved_selection is not None else 'constructed'),
                        domain_selection_seconds=domain_selection_seconds,
                        cc_conv_tol_normt=args.cc_conv_tol_normt, cc_max_cycle=200,
                        targets_meh=args.targets_meh, seeds=args.seeds,
                        pilot_samples=args.pilot_samples, triples_mode=args.triples_method,
                        triples_integral_transform=args.triples_integral_transform,
                        smooth_frame_mode=frame_mode,
                        smooth_frame_blocks=frame_blocks,
                        jax_persistent_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
                        jax_compilation_cache_dir=jax.config.jax_compilation_cache_dir,
                        code_snapshot_sha256=os.environ.get("LNO_BENCHMARK_SNAPSHOT_SHA256"),
                        source_package_path=str(Path(pyscfad.__file__).resolve()),
                        triples_laplace_points=args.triples_laplace_points,
                        smooth_trace_exponent=args.smooth_trace_exponent,
                        selection_seconds=selection_seconds,
                        preselection_setup_seconds=selection_start-started,
                        mp2_laplace_grid=dict(roots=roots.tolist(), weights=weights.tolist(),
                                              diagnostics=diagnostics),
                        reference_hf_energy_eh=float(ref.e_tot),
                        setup_seconds=((production_started if production_started is not None
                                        else time.perf_counter())-started))
        settings.update(sgx_settings)
        save(output / "settings.json", settings)
        report(f"CPU affinity {benchmark_cpus}; all {affinity['runtime_threads']} runtime threads verified")
        report(f"Planned {expected_groups} fragments: one W build and shared input pullback; "
               "LIS ranks/anchors will be selected within each production fragment")
        report(f"Smooth frame blocks {frame_blocks}; persistent JAX cache enabled: "
               f"{bool(jax.config.jax_enable_compilation_cache)}")
        selection.update(static=selected, hf_energy=float(mf.e_tot))
        return selected

    options = dict(baseline_controls=baseline, matrix_controls=matrix,
                    ccsd_t=True, include_hf=True,
                   return_components=True, merge_identical=True)

    deterministic_controls = dict(mode="deterministic")
    if args.triples_integral_transform == "rotate":
        deterministic_controls['integral_transform'] = 'rotate'
    jobs = []
    if not args.stochastic_only:
        jobs.append(("deterministic", deterministic_controls))
    for target_index, target in enumerate(args.targets_meh):
        for seed_index, seed in enumerate(args.seeds):
            if args.deterministic_only:
                continue
            if (args.job_indices is not None
                    and target_index * len(args.seeds) + seed_index not in args.job_indices):
                continue
            label = f"stochastic_t{target:g}_seed{seed}"
            controls = dict(mode="stochastic", nsamples=args.pilot_samples,
                            pilot_samples=args.pilot_samples,
                            global_seed=seed, energy_tolerance=target / 1000,
                            variance_allocation=args.triples_variance_allocation)
            controls.update(trace_exponent=args.smooth_trace_exponent)
            if args.triples_integral_transform == "rotate":
                controls['integral_transform'] = 'rotate'
            jobs.append((label, controls))
    for label, controls in jobs:
        destination = output / f"{label}.json"
        if destination.exists():
            report(f"Skipping completed {label}")
            continue
        report(f"BEGIN {label}")
        phase = time.perf_counter()
        selection_in_production = saved_selection is None and not selection
        if selection_in_production:
            production_started = phase
        report("Using saved domain selections and rebuilding response" if saved_selection is not None
               else "Sharing production SCF with domain selection and weak-pair response")
        energy, molecule_bar, components = ccsd.value_and_grad(
            mol, build_mf, build_static=build_static,
            triples_controls=controls, checkpoint_dir=args.checkpoint_dir,
            resume=args.resume, **options)
        jax.block_until_ready((energy, molecule_bar))
        force = -np.asarray(molecule_bar.coords)
        if not np.isfinite(force).all() or not np.isfinite(float(energy)):
            raise RuntimeError(f"{label} returned nonfinite energy or force")
        fragments = asdict(components)["fragments"]
        if len(fragments) != expected_groups:
            raise RuntimeError(f"{label} returned {len(fragments)} fragments")
        dimensions = [(f['nocc_lis'], f['nvir_lis']) for f in fragments]
        frame_blocks = [dict(occupied=[no], virtual=[nv]) for no, nv in dimensions]
        settings.update(lis_dimensions=dimensions, smooth_frame_blocks=frame_blocks)
        save(output / 'settings.json', settings)
        result = dict(label=label, controls=controls, total_energy_eh=float(energy),
                      force_eh_per_bohr=force.tolist(), components=asdict(components),
                      geometry=str(xyz), basis=args.basis, frozen=args.frozen,
                      hf_energy=selection["hf_energy"], boys_grouping=args.boys_grouping,
                      cc_conv_tol=args.cc_conv_tol,
                      cc_conv_tol_normt=args.cc_conv_tol_normt,
                      threads=args.threads,
                      scf_df_precond=args.scf_df_precond,
                      scf_jk=args.scf_jk, sgx_grid_level=args.sgx_grid_level,
                      sgx_fit_overlap=args.sgx_fit_overlap,
                      df_storage=args.df_storage, df_memory_mb=args.df_memory_mb,
                      df_builder=args.df_builder,
                      omp_dynamic=os.environ.get('OMP_DYNAMIC'),
                      mkl_dynamic=os.environ.get('MKL_DYNAMIC'),
                      shared_df_cache_mode=args.fragment_df_mode,
                      full_domain=full_domain,
                      domain_thresholds=settings["domain_thresholds"],
                      mp2_domains=settings["mp2_domains"],
                      mp2_energy_source="w_matrix_trace", mp2_response="joint_energy_w_pullback",
                      triples_mode=args.triples_method,
                      triples_integral_transform=args.triples_integral_transform,


                      lis_eigenspace='domain',
                      triples_variance_allocation=args.triples_variance_allocation,
                      smooth_frame_mode=frame_mode,
                      smooth_frame_blocks=settings["smooth_frame_blocks"],
                      jax_persistent_cache_enabled=bool(jax.config.jax_enable_compilation_cache),
                      code_snapshot_sha256=os.environ.get("LNO_BENCHMARK_SNAPSHOT_SHA256"),
                      selection_in_production=selection_in_production,
                      setup_seconds=settings["setup_seconds"],
                      selection_seconds=settings["selection_seconds"],
                      preselection_setup_seconds=settings["preselection_setup_seconds"],
                      lno_thresh_occ=args.lno_thresh_occ, lno_thresh_vir=args.lno_thresh_vir,
                      elapsed_seconds=time.perf_counter()-phase,
                      completed_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        result.update(sgx_settings)
        result.update(check_worker_affinity(benchmark_cpus))
        save(destination, result)
        report(f"END {label}: {result['total_energy_eh']:.12f} Eh; "
               f"{result['elapsed_seconds']:.1f} s")
        del energy, molecule_bar, components, force, fragments
        gc.collect()
    if cderi_path is not None:
        cderi_path.unlink()
        report("Removed completed run's temporary DF factors")
    report("All requested calculations complete")


if __name__ == "__main__":
    ensure_worker_stack()
    main()
