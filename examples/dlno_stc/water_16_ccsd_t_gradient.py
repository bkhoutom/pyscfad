"""Water-cluster domain-MP2/LNO-CCSD with stochastic (T) energy and gradient.

Use run_water_16_ccsd_t.sh to set numerical thread limits before Python starts.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import inspect
import math
import os
import sys
from pathlib import Path
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz', type=Path, default=Path(__file__).with_name('water_16.xyz'))
    parser.add_argument('--waters', type=int, default=16,
                        help='number of water molecules; freezes one oxygen core per water')
    parser.add_argument('--basis', default='cc-pvtz')
    parser.add_argument('--auxbasis', default='auto', help='DF auxiliary basis; auto uses PySCF default')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-memory-mb', type=float, default=10000,
                        help='PySCF and W workspace budget in MB, not a hard process limit')
    parser.add_argument('--df-storage', choices=('outcore', 'incore'), default='outcore',
                        help='store global SCF density-fitting factors on disk (default) or in RAM')
    parser.add_argument('--df-memory-mb', type=float, default=1000,
                        help='working-memory budget for disk-backed density fitting in MB')
    parser.add_argument('--matrix-target-batch-size', type=int, default=4,
                        help='W targets sharing MP2 propagation per batch (default: 4; 1 reduces amplitude storage)')
    parser.add_argument('--pair-energy', type=float, default=1.5e-5, help='strong-pair threshold in Eh')
    parser.add_argument('--lno-occ', type=float, default=1e-4, help='occupied W cutoff')
    parser.add_argument('--lno-vir', type=float, default=1e-5, help='virtual W cutoff')
    parser.add_argument('--triples-tolerance-meh', type=float, default=0.1,
                        help='target molecular (T) energy standard error in mEh')
    parser.add_argument('--pilot-samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--output-dir', type=Path, default=Path('output/water_16_ccpvtz_stc'))
    parser.add_argument('--resume', action='store_true', help='resume a compatible serial checkpoint')
    parser.add_argument('--vvvv-direct', action=argparse.BooleanOptionalAction, default=True,
                        help='use lower-memory CCSD virtual ladder (default: enabled)')
    args = parser.parse_args()
    if args.waters < 1:
        parser.error('waters must be positive')
    if args.threads < 1 or args.pilot_samples < 2 or not 0 <= args.seed < 2**64:
        parser.error('threads must be positive, pilot-samples >= 2, and seed fit uint64')
    if not all(math.isfinite(v) and v > 0 for v in
               (args.max_memory_mb, args.df_memory_mb, args.triples_tolerance_meh)):
        parser.error('memory and triples tolerance must be finite and positive')
    if args.df_memory_mb > args.max_memory_mb:
        parser.error('df-memory-mb must not exceed max-memory-mb')
    if args.matrix_target_batch_size < 1:
        parser.error('matrix-target-batch-size must be positive')
    if not all(math.isfinite(v) and v >= 0 for v in (args.pair_energy, args.lno_occ, args.lno_vir)):
        parser.error('pair and LNO thresholds must be finite and nonnegative')
    return args


def main():
    args = arguments()
    started = time.perf_counter()
    # Keep direct invocation and the launcher consistent, before numerical imports.
    for name in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'PYSCFAD_DENSE_BLAS_THREADS',
                 'PJRT_NPROC'):
        os.environ[name] = str(args.threads)
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'
    for name, value in (
            ('JAX_CPU_ENABLE_ASYNC_DISPATCH', 'false'),
            ('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false'),
            ('PYSCFAD_STC_CLEAR_JAX_CACHES', '1'),
            ('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '1')):
        os.environ.setdefault(name, value)
    output = args.output_dir.resolve()
    if (output / 'result.json').exists():
        raise FileExistsError(f'{output / "result.json"} already exists; choose a new output directory')
    from pyscfad.dlno_stc.timings import TimingReport
    with TimingReport(output, started=started):
        calculate(args)


def calculate(args):
    import jax
    import numpy as np
    import pyscf
    from mpi4py import MPI
    from pyscf import lib, gto as pgto, df as pdf
    from pyscfad import config, gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.controls import default_laplace_grid
    from pyscfad.tools.linear_solver import eager_gmres
    from pyscfad.tools import resource_profile
    from pyscf.lo.boys import Boys as PySCFBoys
    print(f'Python: {sys.executable}\nPySCF: {pyscf.__version__} ({pyscf.__file__})', flush=True)
    if 'dip' not in inspect.signature(PySCFBoys.get_grad).parameters:
        raise RuntimeError('This branch requires the PySCF 2.14 Boys interface. '
                           'Use run_water_16_ccsd_t.sh with the repository .venv; '
                           'an older PySCF checkout may be overriding it through PYTHONPATH.')
    try:
        from pyscfad.dlno_stc import _stc_mp2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('The STC native extension is not installed; build it using '
                           'pyscfad/dlno_stc/README.md before running this calculation') from exc

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 1:
        raise RuntimeError('This input requires exactly one MPI process')
    lib.num_threads(args.threads)
    # Avoid full coordinate-JVP integral tensors; use the native first-order path.
    config.update('pyscfad_moleintor_opt', True)
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise FileExistsError(f'{output / "result.json"} already exists; choose a new output directory')
    xyz = args.xyz.resolve()
    auxbasis = None if args.auxbasis.lower() in ('auto', 'none') else args.auxbasis
    mol = gto.Mole(atom=str(xyz), basis=args.basis, unit='Angstrom',
                   max_memory=args.max_memory_mb, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    if (len(symbols) != 3 * args.waters or symbols.count('O') != args.waters
            or symbols.count('H') != 2 * args.waters):
        raise ValueError(f'geometry must contain {args.waters} oxygen and '
                         f'{2 * args.waters} hydrogen atoms')

    frozen = args.waters  # One oxygen 1s orbital per water molecule.
    roots, weights = default_laplace_grid()
    baseline = dict(mode='deterministic', threads=args.threads,
                    laplace_roots=roots, laplace_weights=weights)
    matrix = dict(mode='deterministic', threads=args.threads,
                  target_batch_size=args.matrix_target_batch_size,
                  memory_limit_mb=args.max_memory_mb,
                  laplace_roots=roots, laplace_weights=weights)
    triples = dict(mode='stochastic', nsamples=args.pilot_samples,
                   pilot_samples=args.pilot_samples, global_seed=args.seed,
                   energy_tolerance=args.triples_tolerance_meh / 1000,
                   variance_allocation='occupied_count')
    thresholds = dlno_stc.BoysDomainThresholds(pair_energy=args.pair_energy)
    settings = dict(geometry=str(xyz), basis=args.basis, auxbasis=args.auxbasis,
                    python=sys.executable, pyscf_version=pyscf.__version__, pyscf_path=pyscf.__file__,
                    waters=args.waters, frozen=frozen, mpi_processes=1, threads=args.threads,
                    max_memory_mb=args.max_memory_mb, boys_grouping='atom',
                    df_storage=args.df_storage, df_memory_mb=args.df_memory_mb,
                    df_jk_workspace_mb=args.df_memory_mb if args.df_storage == 'outcore' else None,
                    moleintor_opt=True, matrix_target_batch_size=args.matrix_target_batch_size,
                    runtime_environment={key: os.environ.get(key) for key in
                        ('PYSCFAD_STC_CLEAR_JAX_CACHES', 'PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE',
                         'JAX_CPU_ENABLE_ASYNC_DISPATCH', 'XLA_FLAGS', 'PJRT_NPROC',
                         'PYSCFAD_DENSE_BLAS_THREADS', 'PYSCFAD_STC_DENSE_BLAS_THREADS',
                         'PYSCFAD_SCF_OCCUPIED_CACHE', 'PYSCFAD_SCF_RESPONSE_FACTOR_CACHE_MB')},
                    domain_thresholds=asdict(thresholds), lno_occ=args.lno_occ,
                    lno_vir=args.lno_vir, matrix_kind='W', mp2_mode='deterministic',
                    triples=triples, vvvv_direct=args.vvvv_direct,
                    laplace_grid=dict(kind='default', roots=roots.tolist(), weights=weights.tolist()),
                    gradient_unit='Eh/bohr')
    print('Calculation settings:\n' + json.dumps(settings, indent=2), flush=True)
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')

    cderi_path = output / 'scf_cderi.h5' if args.df_storage == 'outcore' else None
    if cderi_path is not None:
        # Build packed factors at this geometry outside JAX tracing. The attached
        # DF object's custom VJPs still differentiate the integrals and metric.
        reference_mol = pgto.M(atom=str(xyz), basis=args.basis, unit='Angstrom',
                              max_memory=args.df_memory_mb, verbose=4)
        reference_df = pdf.DF(reference_mol, auxbasis=auxbasis)
        reference_df._compatible_format = True
        reference_df._cderi_to_save = str(cderi_path)
        reference_df.max_memory = args.df_memory_mb
        # Reuse only factors created by this input with matching scientific
        # metadata and a completed packed dataset. This also speeds restarts.
        import h5py
        from pyscf.df.addons import make_auxmol
        naux = make_auxmol(reference_mol, auxbasis).nao
        fingerprint = dict(geometry_sha256=hashlib.sha256(xyz.read_bytes()).hexdigest(),
                           basis=args.basis, auxbasis=auxbasis, pyscf_version=pyscf.__version__,
                           shape=[naux, reference_mol.nao * (reference_mol.nao + 1) // 2])
        manifest = output / 'scf_cderi.json'
        reusable = False
        if cderi_path.exists() and manifest.exists():
            try:
                with h5py.File(cderi_path, 'r') as factors:
                    reusable = (json.loads(manifest.read_text()) == fingerprint
                                and isinstance(factors.get('j3c'), h5py.Dataset)
                                and list(factors['j3c'].shape) == fingerprint['shape'])
            except (OSError, ValueError):
                pass
        if reusable:
            print(f'Reusing validated disk-backed DF factors at {cderi_path}', flush=True)
        else:
            manifest.unlink(missing_ok=True)
            print(f'Building disk-backed DF factors at {cderi_path}', flush=True)
            with resource_profile.section('water.df_factors'):
                reference_df.build()
            manifest.write_text(json.dumps(fingerprint, indent=2) + '\n')
        del reference_df, reference_mol
        print('Disk-backed DF factors ready; starting differentiable SCF', flush=True)

    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=auxbasis)
        if cderi_path is not None:
            mf.with_df.incore = False
            mf.with_df.max_memory = args.df_memory_mb
            mf.with_df.jk_workspace_memory = args.df_memory_mb
            mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-9, 1e-6, None
        if cderi_path is None:
            mf.kernel()
        else:
            # Disk-reading response operators must execute outside JIT.
            with eager_gmres():
                mf.kernel()
        if not mf.converged:
            raise RuntimeError('DF-RHF did not converge')
        return mf

    def build_static(mf):
        domains = dlno_stc.build_domain_selections(
            mf, frozen=frozen, thresholds=thresholds, boys_grouping='atom',
            lo_kwargs=dict(init_guess='atomic', conv_tol=1e-10,
                           gmres_options=dict(restart=120, maxiter=100)))
        print(f'Selected {len(domains.physical_groups)} physical fragments, '
              f'{len(domains.active_occ_indices)} active occupied orbitals', flush=True)
        domain_sizes = []
        for index, members in enumerate(domains.physical_groups):
            fragment = domains.fragments[members[0]]
            domain_sizes.append(dict(fragment=index, targets=len(members),
                nao=len(fragment.extended_ao_indices),
                nocc=len(fragment.strong_occ_metric_keep),
                nvir=len(fragment.virtual_anchor_columns)))
        (output / 'domain_sizes.json').write_text(json.dumps(domain_sizes, indent=2) + '\n')
        print('MP2 domain sizes: ' + json.dumps(domain_sizes), flush=True)
        return ccsd.build_static_selections(
            mf, domains, matrix_kind='W', thresh_occ=args.lno_occ, thresh_vir=args.lno_vir,
            baseline_controls=baseline, matrix_controls=matrix, smooth=True,
            smooth_laplace_points=10, single_pass=True, merge_identical=True)

    energy, mol_bar, components = ccsd.value_and_grad(
        mol, build_mf, build_static=build_static, baseline_controls=baseline,
        matrix_controls=matrix, matrix_kind='W', ccsd_t=True,
        triples_controls=triples, include_hf=True, return_components=True,
        checkpoint_dir=str(output / 'checkpoint'), resume=args.resume,
        vvvv_direct=args.vvvv_direct)
    jax.block_until_ready((energy, mol_bar))
    gradient = np.asarray(mol_bar.coords)
    if not np.isfinite(float(energy)) or not np.isfinite(gradient).all():
        raise RuntimeError('calculation returned nonfinite energy or gradient')
    result = dict(settings=settings, total_energy_eh=float(energy),
                  gradient_eh_per_bohr=gradient.tolist(), components=asdict(components),
                  triples_energy_standard_error_eh=float(components.triples_standard_error),
                  elapsed_seconds=time.perf_counter() - started)
    result_path = output / 'result.json'
    temporary = result_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(result_path)
    print(f'Total energy: {float(energy):.12f} Eh\nGradient (Eh/bohr):\n{gradient}', flush=True)
    print(f'Results written to {result_path}', flush=True)


if __name__ == '__main__':
    main()
