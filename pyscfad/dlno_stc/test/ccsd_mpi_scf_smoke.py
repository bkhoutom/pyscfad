"""Real CC MPI SCF acceptance and paired prepared-water4 timing entry point."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from unittest.mock import patch


def water4_fixture(directory, threads):
    """Use the frozen benchmark inputs with a live SCF solve from saved dm0."""
    import numpy as np
    from pyscfad import gto, scf
    from pyscfad.lno.ccsd import RCCSD
    from pyscfad.scf import hf
    from pyscfad.tools import linear_solver

    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in ('canonical_mf.npz', 'scf_cderi.h5', 'deterministic_static.pkl'):
        digest = hashlib.sha256()
        with (directory / name).open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b''):
                digest.update(block)
        assert digest.hexdigest() == manifest['files'][name], name
    RCCSD.conv_tol = 1e-11
    RCCSD.conv_tol_normt = 1e-9
    RCCSD.max_cycle = 150
    def tight(**kwargs):
        kwargs.update(manifest['scf_response_gmres'])
        return linear_solver.gen_gmres(**kwargs)
    hf.gen_gmres = tight
    settings = manifest['settings']
    mol = gto.Mole(atom=settings['xyz_file'], basis=manifest['basis'], unit='Angstrom',
                   verbose=0, max_memory=settings['max_memory'])
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    with np.load(directory / 'canonical_mf.npz') as saved:
        occupied = saved['mo_coeff'][:, saved['mo_occ'] > 0]
        dm0 = 2 * occupied @ occupied.T
    def build_mf(current, **initial):
        mf = scf.RHF(current).density_fit(auxbasis=manifest['auxbasis'])
        mf.with_df.max_memory = settings['max_memory']
        mf.with_df.attach_outcore_cderi(str(directory / 'scf_cderi.h5'))
        mf.conv_tol = settings['mf_conv_tol']
        mf.conv_tol_grad = settings['mf_conv_tol_grad']
        mf.max_cycle = 150
        mf.chkfile = None
        if initial.get('mo_coeff_init') is not None:
            for name in ('mo_coeff', 'mo_energy', 'mo_occ', 'e_tot'):
                setattr(mf, name, initial[name + '_init'])
            mf.converged = True
        elif not initial.get('skip_scf', False):
            mf.kernel(dm0=dm0)
            assert mf.converged
        return mf
    with (directory / 'deterministic_static.pkl').open('rb') as handle:
        static = pickle.load(handle)
    baseline = dict(manifest['baseline_controls'], threads=threads)
    matrix = dict(manifest['matrix_controls']['deterministic'], threads=threads)
    return mol, build_mf, static, baseline, matrix


def run(mode, output, *, fixture='small', fixture_dir=None, threads=1):
    import jax
    import numpy as np
    from mpi4py import MPI
    import pyscfad
    from pyscfad import config_update
    from pyscfad.df import mpi_df_jk
    from pyscfad.dlno_stc import ccsd, driver, _mpi_ccsd, _mpi_scf
    from pyscfad.dlno_stc.test import _boys_acceptance as accept

    comm = MPI.COMM_WORLD
    rank = comm.rank
    with_grad = mode != 'kernel'
    enabled = mode not in ('disabled', 'serial') and comm.size > 1
    counts = dict(forward=0, density_vjp=0, coordinate_vjp=0)
    norms = dict.fromkeys(counts, 0.)
    phases = {name: dict.fromkeys(counts, 0) for name in ('initial', 'final')}
    shared = dict(common_forward=0, common_reverse=0, scf_reverse=0)
    times = dict(setup=0., initial=0., final=0., total=0.)
    builders = dict(scf=0, skeleton=0)
    state = dict(phase='initial', serves=0, service=None, correlation_df=None,
                 source_survived=False, source_fingerprint=None, source_path=None,
                 source_verified=False, declared_source_path=None)
    before_tokens = set(mpi_df_jk._EXECUTORS)
    before_active = mpi_df_jk.get_active_executor()
    cls = mpi_df_jk.MPIDFJKExecutor

    if fixture == 'water4':
        if fixture_dir is None:
            raise ValueError('--fixture-dir is required for water4')
        mol, base_builder, static, baseline, matrix = water4_fixture(fixture_dir, threads)
        build_static = None
    else:
        mol, base_builder, static = accept.molecule(), accept.build_mf, None
        baseline = dict(accept.controls(), threads=threads)
        matrix = dict(baseline)
        def build_static(mf):
            state['selecting'] = True
            try:
                return ccsd.build_static_selections(mf, accept.selection(mf), smooth=False, matrix_kind='W',
                    thresh_occ=1e-5, thresh_vir=1e-5,
                    baseline_controls=baseline, matrix_controls=matrix)
            finally:
                state['selecting'] = False

    def build_mf(current, **initial):
        skeleton = bool(initial.get('skip_scf')) or initial.get('mo_coeff_init') is not None
        builders['skeleton' if skeleton else 'scf'] += 1
        if fixture == 'small' and initial.get('skip_scf'):
            from pyscfad import scf
            mf = scf.RHF(current).density_fit()
            mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
            return mf
        mf = base_builder(current, **initial)
        source = _mpi_scf._source(mf.with_df)
        if isinstance(source, (str, Path)):
            state['declared_source_path'] = str(source)
        return mf

    def timed(name, function, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = function(*args, **kwargs)
            jax.block_until_ready(result)
            return result
        finally:
            times[name] += time.perf_counter() - start

    def execution(original, name):
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            jax.block_until_ready(result)
            counts[name] += 1
            phases[state['phase']][name] += 1
            if name == 'coordinate_vjp':
                norms[name] += sum(float(np.linalg.norm(np.asarray(bar.coords))) for bar in result)
            else:
                norms[name] += float(np.linalg.norm(np.asarray(result)))
            return result
        return wrapped

    service_init = _mpi_scf.DomainSCFService.__init__
    def setup(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            service_init(self, *args, **kwargs)
            state['service'] = self
        finally:
            times['setup'] += time.perf_counter() - start

    run_initial = _mpi_scf.DomainSCFService.run_initial
    def initial(self, function):
        return timed('initial', run_initial, self, function)

    run_final = _mpi_scf.DomainSCFService.run_final
    def final(self, function):
        state['phase'] = 'final'
        state['source_verified'] = self.executor.source_verified
        if rank:
            correlation_df = state['correlation_df']
            assert correlation_df is not self.worker_df
            source = _mpi_scf._source(self.worker_df)
            auxmol = self.worker_df.auxmol
            # Correlation legitimately clears/replaces its DF cache. Prove the
            # dedicated service source survives even on owners with no CC group.
            correlation_df._cderi = correlation_df._cderi_to_save = None
            correlation_df.auxmol = None
            correlation_df._get_cderi_source = lambda: None
            assert _mpi_scf._source(self.worker_df) is source
            assert self.worker_df.auxmol is auxmol and auxmol is not None
            state['source_survived'] = True
        return timed('final', run_final, self, function)

    serve = cls.serve
    def serving(self, *args, **kwargs):
        state['phase'] = 'initial' if state['serves'] == 0 else 'final'
        state['serves'] += 1
        return serve(self, *args, **kwargs)

    fingerprint = _mpi_scf._source_fingerprint
    def fingerprinted(dfobj):
        result = fingerprint(dfobj)
        state['source_fingerprint'] = result
        source = _mpi_scf._source(dfobj)
        state['source_path'] = str(source) if isinstance(source, (str, Path)) else None
        return result

    initialize = driver._initialize
    def initialized(*args, **kwargs):
        result = initialize(*args, **kwargs)
        if not state.get('selecting', False):
            shared['common_forward'] += 1
        if kwargs.get('with_grad'):
            original = result['common_pullback']
            def reverse(bar):
                shared['common_reverse'] += 1
                return original(bar)
            result['common_pullback'] = reverse
        return result

    finish_response = driver._finish_response
    def response(*args, **kwargs):
        shared['scf_reverse'] += 1
        if mode == 'response-failure' and rank == 0:
            raise RuntimeError('deliberate CC root final-response failure')
        return finish_response(*args, **kwargs)

    build_worker = _mpi_ccsd._build_cc_worker
    def worker(*args, **kwargs):
        mf = build_worker(*args, **kwargs)
        state['correlation_df'] = mf.with_df
        return mf

    run_group = ccsd._run_group
    def group(*args, **kwargs):
        assert mpi_df_jk.get_active_executor() is None, 'SCF hook active during CC group'
        if mode == 'group-failure' and rank == 1:
            raise RuntimeError('deliberate CC paused worker-group failure')
        return run_group(*args, **kwargs)

    run_cc = ccsd._run
    def cc_run(initializer, *args, **kwargs):
        if not enabled:
            original = initializer
            initializer = lambda: timed('initial', original)
        return run_cc(initializer, *args, **kwargs)

    finish_cc = ccsd._finish
    def finishing(*args, **kwargs):
        state['phase'] = 'final'
        if enabled:
            return finish_cc(*args, **kwargs)
        return timed('final', finish_cc, *args, **kwargs)

    expected_failure = {'group-failure': 'paused worker-group failure',
                        'response-failure': 'root final-response failure',
                        'invalid-option': 'parallel_scf_jk must be a bool'}.get(mode)
    failed = False
    options = dict(baseline_controls=baseline, matrix_controls=matrix, ccsd_t=True,
                   include_hf=True, parallel_scf_jk=mode != 'disabled',
                   comm=None if mode == 'serial' else comm)
    if mode == 'invalid-option':
        options['parallel_scf_jk'] = 'invalid' if rank == 0 else False
    if build_static is not None:
        options['build_static'] = build_static
    with ExitStack() as stack:
        stack.enter_context(config_update('pyscfad_moleintor_opt', True))
        stack.enter_context(config_update('pyscfad_scf_df_precond', False))
        for obj, name, replacement in (
            (cls, '_execute_forward', execution(cls._execute_forward, 'forward')),
            (cls, '_execute_density_vjp', execution(cls._execute_density_vjp, 'density_vjp')),
            (cls, '_execute_coordinate_vjp_block', execution(cls._execute_coordinate_vjp_block, 'coordinate_vjp')),
            (cls, 'serve', serving),
            (_mpi_scf.DomainSCFService, '__init__', setup),
            (_mpi_scf.DomainSCFService, 'run_initial', initial),
            (_mpi_scf.DomainSCFService, 'run_final', final),
            (_mpi_scf, '_source_fingerprint', fingerprinted),
            (driver, '_initialize', initialized), (driver, '_finish_response', response),
            (_mpi_ccsd, '_build_cc_worker', worker), (ccsd, '_run_group', group),
            (ccsd, '_run', cc_run), (ccsd, '_finish', finishing)):
            stack.enter_context(patch.object(obj, name, replacement))
        # Fixture preparation and its checks precede the production interval.
        # Every rank starts together; the receipt includes service setup/close.
        comm.Barrier()
        started = time.perf_counter()
        try:
            if with_grad:
                result = ccsd.value_and_grad(mol, build_mf, static, **options)
            else:
                mf = build_mf(mol) if rank == 0 else build_mf(mol, skip_scf=True)
                if rank == 0 and static is None:
                    static = build_static(mf)
                options.pop('build_static', None)
                result = ccsd.kernel(mf, static, **options)
            jax.block_until_ready(result)
        except (RuntimeError, ValueError) as exc:
            if expected_failure is None:
                raise
            assert expected_failure in str(exc), str(exc)
            failed, result = True, None
        finally:
            times['total'] = time.perf_counter() - started
    assert set(mpi_df_jk._EXECUTORS) == before_tokens, 'executor registry token leaked'
    assert mpi_df_jk.get_active_executor() is before_active, 'active executor leaked'
    observations = comm.allgather(dict(counts=counts, norms=norms, phases=phases,
        shared_phases=shared, serves=state['serves'], builder_calls=builders,
        source_survived=state['source_survived'], source_verified=state['source_verified'],
        source_fingerprint=state['source_fingerprint'],
        source_path=state['source_path'] or state['declared_source_path'],
        failed=failed, timings=times, affinity=sorted(os.sched_getaffinity(0)),
        registry_clean=True, active_executor_clean=True))
    maxima = {key: max(row['timings'][key] for row in observations) for key in times}
    from pyscfad.dlno_stc import _stc_mp2
    fixture_hashes = {}
    if fixture_dir is not None:
        for name in ('manifest.json', 'deterministic_static.pkl'):
            fixture_hashes[name] = hashlib.sha256((fixture_dir / name).read_bytes()).hexdigest()
    values = dict(mode=mode, fixture=fixture, ranks=comm.size, threads=threads,
                  package_path=pyscfad.__file__, native_path=_stc_mp2.__file__,
                  affinity=sorted(os.sched_getaffinity(0)), fixture_hashes=fixture_hashes,
                  preconditioner=False, observations=observations, timings_max_seconds=maxima,
                  environment={key: os.environ.get(key) for key in
                               ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_PROC_BIND', 'OMP_PLACES',
                                'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                                'PYSCFAD_DENSE_BLAS_THREADS', 'JAX_ENABLE_X64', 'JAX_PLATFORMS',
                                'XLA_FLAGS', 'JAX_ENABLE_COMPILATION_CACHE')})
    if expected_failure:
        assert all(row['failed'] for row in observations), observations
        values['failure'] = expected_failure
    else:
        assert observations[0]['shared_phases'] == dict(
            common_forward=1, common_reverse=int(with_grad), scf_reverse=int(with_grad)), observations
        assert all(not any(row['shared_phases'].values()) for row in observations[1:]), observations
        if enabled:
            for row in observations:
                assert row['counts']['forward'] > 0 and row['norms']['forward'] > 0, observations
                if with_grad:
                    assert all(value > 0 for value in row['counts'].values()), observations
                    assert all(value > 0 for value in row['norms'].values()), observations
                else:
                    assert row['counts']['density_vjp'] == row['counts']['coordinate_vjp'] == 0, observations
                assert row['phases']['initial']['forward'] > 0, observations
                if with_grad:
                    assert row['phases']['final']['density_vjp'] > 0, observations
                    assert row['phases']['final']['coordinate_vjp'] > 0, observations
                assert row['source_verified'], observations
            for row in observations[1:]:
                assert row['serves'] == 2 and row['source_survived'], observations
                assert row['builder_calls'] == dict(scf=0, skeleton=2 if with_grad else 1), observations
            assert all(row['source_fingerprint'] == observations[0]['source_fingerprint'] for row in observations), observations
        else:
            assert all(not any(row['counts'].values()) and row['serves'] == 0 for row in observations), observations
        if fixture == 'water4':
            assert all(row['source_path'] == str(fixture_dir / 'scf_cderi.h5') for row in observations), observations
        if rank == 0:
            energy, bar = result if with_grad else (result, None)
            assert np.isfinite(float(energy))
            values['energy'] = float(energy)
            if with_grad:
                assert np.isfinite(np.asarray(bar.coords)).all()
                values['gradient'] = np.asarray(bar.coords).tolist()
        else:
            assert result is None
    if rank == 0:
        if output:
            output.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
        print('CC MPI SCF smoke PASS: ' + mode, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', default='enabled', choices=('enabled', 'disabled', 'serial',
                        'kernel', 'group-failure', 'response-failure', 'invalid-option'))
    parser.add_argument('--fixture', default='small', choices=('small', 'water4'))
    parser.add_argument('--fixture-dir', type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.mode, args.output, fixture=args.fixture, fixture_dir=args.fixture_dir, threads=args.threads)


if __name__ == '__main__':
    main()
