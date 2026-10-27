"""Real MPI acceptance for STC's root-owned distributed SCF/J/K service."""
import argparse
from contextlib import ExitStack
import importlib.abc
import json
from pathlib import Path
import sys
from unittest.mock import patch


def run(mode, output):
    # The standalone STC path must retain its native-DLNO independence.
    class BlockNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'pyscfad.dlno' or fullname.startswith('pyscfad.dlno.'):
                raise ImportError('native DLNO blocked by STC SCF acceptance')

    sys.meta_path.insert(0, BlockNative())
    import jax
    import numpy as np
    from mpi4py import MPI
    from pyscfad import config_update, dlno_stc
    from pyscfad.df import mpi_df_jk, df
    from pyscfad.scf import hf
    from scipy.sparse.linalg import LinearOperator
    from pyscfad.dlno_stc import driver
    from pyscfad.dlno_stc.test import _boys_acceptance as accept

    comm = MPI.COMM_WORLD
    rank = comm.rank
    expect_service = mode not in ('disabled', 'serial', 'system') and comm.size > 1
    shared_phases = dict(common_forward=0, common_reverse=0, scf_reverse=0)
    counts = dict(forward=0, density_vjp=0, coordinate_vjp=0)
    preconditioner_counts = dict(setup=0, apply=0)
    norms = dict(forward=0., density_vjp=0., coordinate_vjp=0.)
    phase_counts = {name: dict.fromkeys(counts, 0) for name in ('initialize', 'response')}
    phase = 'initialize'
    serves = 0
    source_builds = 0
    builder_calls = dict(scf=0, skeleton=0)
    before_tokens = set(mpi_df_jk._EXECUTORS)
    before_executor = mpi_df_jk.get_active_executor()
    cls = mpi_df_jk.MPIDFJKExecutor
    forward = cls._execute_forward
    density = cls._execute_density_vjp
    coordinate = cls._execute_coordinate_vjp_block
    serve = cls.serve
    build_df = df.DF.build
    close_response = driver._finish
    prepare_group = driver._prepare
    rebuild_common = driver.rebuild_domain_data
    gen_precond = hf._gen_precond

    def counted_precond(*args, **kwargs):
        precond = gen_precond(*args, **kwargs)
        if precond is None:
            return None
        preconditioner_counts['setup'] += 1
        def apply(vector):
            preconditioner_counts['apply'] += 1
            return precond @ vector
        return LinearOperator(precond.shape, matvec=apply, dtype=precond.dtype)

    def counted_forward(self, dfobj, payload):
        result = forward(self, dfobj, payload)
        counts['forward'] += 1
        phase_counts[phase]['forward'] += 1
        norms['forward'] += float(np.linalg.norm(np.stack(result)))
        return result

    def counted_density(self, dfobj, payload):
        result = density(self, dfobj, payload)
        counts['density_vjp'] += 1
        phase_counts[phase]['density_vjp'] += 1
        norms['density_vjp'] += float(np.linalg.norm(result))
        return result

    def counted_coordinate(self, dfobj, payload, shls_slice, ints_bar):
        result = coordinate(self, dfobj, payload, shls_slice, ints_bar)
        counts['coordinate_vjp'] += 1
        phase_counts[phase]['coordinate_vjp'] += 1
        norms['coordinate_vjp'] += sum(float(np.linalg.norm(np.asarray(bar.coords)))
                                       for bar in result)
        return result

    def counted_serve(self, dfobj):
        nonlocal phase, serves
        phase = 'initialize' if serves == 0 else 'response'
        serves += 1
        return serve(self, dfobj)

    def counted_build_df(self, *args, **kwargs):
        nonlocal source_builds
        source_builds += 1
        if mode == 'worker-df-failure' and rank == 1:
            raise RuntimeError('deliberate STC SCF worker DF build failure')
        result = build_df(self, *args, **kwargs)
        if mode == 'source-corruption' and rank == 1:
            factors = np.array(self._cderi, copy=True)
            factors.reshape(-1)[0] += 1e-3
            self._cderi = jax.numpy.asarray(factors)
        return result

    def counted_rebuild_common(*args, **kwargs):
        assert (mpi_df_jk.get_active_executor() is not None) == expect_service
        shared_phases['common_forward'] += 1
        return rebuild_common(*args, **kwargs)

    def counted_close_response(*args, **kwargs):
        nonlocal phase
        assert (mpi_df_jk.get_active_executor() is not None) == expect_service
        shared_phases['common_reverse'] += 1
        phase = 'response'
        return close_response(*args, **kwargs)

    def counted_prepare_group(*args, **kwargs):
        if mode == 'group-failure' and rank == 1:
            raise RuntimeError('deliberate STC paused group failure')
        return prepare_group(*args, **kwargs)

    def build_mf(mol, **initial):
        builder_calls['skeleton' if initial.get('mo_coeff_init') is not None else 'scf'] += 1
        if mode == 'worker-failure' and rank == 1:
            raise RuntimeError('deliberate STC SCF worker skeleton failure')
        if mode == 'root-failure' and rank == 0:
            raise RuntimeError('deliberate STC SCF root initialization failure')
        if rank == 0:
            assert (mpi_df_jk.get_active_executor() is not None) == expect_service
        return accept.build_mf(mol, **initial)

    original_finish = driver._finish_response

    def finish(*args, **kwargs):
        assert (mpi_df_jk.get_active_executor() is not None) == expect_service
        shared_phases['scf_reverse'] += 1
        if mode == 'response-failure' and rank == 0:
            raise RuntimeError('deliberate STC SCF root response failure')
        return original_finish(*args, **kwargs)

    options = {}
    if mode == 'disabled':
        options['parallel_scf_jk'] = False
    if mode == 'system':
        options.update(scope='system', frozen=2)
    else:
        options['build_static'] = accept.selection
    active_comm = None if mode == 'serial' else comm
    expected_failure = {'worker-failure': 'worker skeleton failure',
                        'root-failure': 'root initialization failure',
                        'response-failure': 'root response failure',
                        'worker-df-failure': 'worker DF build failure',
                        'source-corruption': 'integral contents differ',
                        'group-failure': 'paused group failure'}.get(mode)
    failed = False
    with ExitStack() as stack:
        # Packed factors make this acceptance exercise the host response
        # preconditioner in both the serial reference and root-owned MPI solve.
        stack.enter_context(config_update('pyscfad_moleintor_opt', True))
        stack.enter_context(config_update('pyscfad_scf_df_precond', True))
        stack.enter_context(patch.object(hf, '_gen_precond', counted_precond))
        stack.enter_context(patch.object(cls, '_execute_forward', counted_forward))
        stack.enter_context(patch.object(cls, '_execute_density_vjp', counted_density))
        stack.enter_context(patch.object(cls, '_execute_coordinate_vjp_block', counted_coordinate))
        stack.enter_context(patch.object(driver, '_finish_response', finish))
        stack.enter_context(patch.object(driver, '_finish', counted_close_response))
        stack.enter_context(patch.object(driver, '_prepare', counted_prepare_group))
        stack.enter_context(patch.object(driver, 'rebuild_domain_data', counted_rebuild_common))
        stack.enter_context(patch.object(cls, 'serve', counted_serve))
        stack.enter_context(patch.object(df.DF, 'build', counted_build_df))
        try:
            result = dlno_stc.value_and_grad(
                accept.molecule(), build_mf, controls=accept.controls() if rank == 0 else None,
                comm=active_comm, include_hf=True, **options)
        except RuntimeError as exc:
            if expected_failure is None:
                raise
            assert expected_failure in str(exc), str(exc)
            failed = True
            result = None
    assert set(mpi_df_jk._EXECUTORS) == before_tokens, 'executor registry token leaked'
    assert mpi_df_jk.get_active_executor() is before_executor, 'active executor leaked'
    observations = comm.allgather(dict(counts=counts, norms=norms, failed=failed,
        preconditioner=preconditioner_counts,
        phases=phase_counts, shared_phases=shared_phases, source_builds=source_builds,
        builder_calls=builder_calls, serves=serves))
    if expected_failure:
        assert all(row['failed'] for row in observations), observations
        values = {'failure': expected_failure, 'observations': observations}
    else:
        enabled = expect_service
        assert all(value > 0 for value in observations[0]['preconditioner'].values()), observations
        assert all(not any(row['preconditioner'].values()) for row in observations[1:]), observations
        if mode != 'system':
            assert observations[0]['shared_phases'] == dict(
                common_forward=1, common_reverse=1, scf_reverse=1), observations
        assert all(not any(row['shared_phases'].values()) for row in observations[1:]), observations
        if enabled:
            for worker in observations:
                assert all(value > 0 for value in worker['counts'].values()), observations
                assert all(value > 0 for value in worker['norms'].values()), observations
                assert worker['phases']['initialize']['forward'] > 0, observations
                assert worker['phases']['response']['density_vjp'] > 0, observations
                assert worker['phases']['response']['coordinate_vjp'] > 0, observations
            for worker in observations[1:]:
                assert worker['source_builds'] == 1, observations
                assert worker['builder_calls'] == dict(scf=0, skeleton=2), observations
                assert worker['serves'] == 2, observations
        else:
            assert all(not any(row['counts'].values()) for row in observations), observations
            assert all(row['source_builds'] == 0 for row in observations[1:]), observations
        if rank == 0:
            energy, bar = result
            assert np.isfinite(energy) and np.isfinite(np.asarray(bar.coords)).all()
            values = dict(energy=float(energy), gradient=np.asarray(bar.coords).tolist(),
                          observations=observations)
        else:
            assert result is None
            values = None
    assert not any(name == 'pyscfad.dlno' or name.startswith('pyscfad.dlno.')
                   for name in sys.modules)
    if rank == 0:
        if output:
            output.write_text(json.dumps(values, indent=2) + '\n')
        print('STC MPI SCF smoke PASS: ' + mode, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', default='default', choices=('default', 'serial', 'disabled',
                        'system', 'worker-failure', 'worker-df-failure', 'root-failure', 'response-failure',
                        'source-corruption', 'group-failure'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.mode, args.output)


if __name__ == '__main__':
    main()
