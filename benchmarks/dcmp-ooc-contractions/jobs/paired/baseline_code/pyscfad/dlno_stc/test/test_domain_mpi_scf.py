"""Timeout-protected public tests for MPI-assisted domain SCF and response."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_domain_mpi import SOURCE_ROOT


def launch(tmp_path, mode='default', ranks=2):
    pytest.importorskip('mpi4py')
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    output = tmp_path / f'scf-{mode}-{ranks}.json'
    command = [sys.executable, '-m', 'pyscfad.dlno_stc.test.domain_mpi_scf_smoke',
               '--mode', mode, '--output', str(output)]
    if mode != 'serial':
        command = [launcher, '-n', str(ranks), *command]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_DYNAMIC='FALSE',
               BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', PYSCFAD_DENSE_BLAS_THREADS='1',
               JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for name in ('OMP_PLACES', 'GOMP_CPU_AFFINITY', 'KMP_AFFINITY', 'KMP_HW_SUBSET'):
        env.pop(name, None)
    timeout = 90 if mode.endswith('failure') or mode == 'source-corruption' else 240
    try:
        result = subprocess.run(command, cwd=SOURCE_ROOT, env=env, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'STC MPI SCF {mode} hung beyond {timeout}s: {exc.stdout!r} {exc.stderr!r}')
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert 'STC MPI SCF smoke PASS' in result.stdout
    return json.loads(output.read_text())


def test_domain_default_parallel_scf_matches_serial_and_opt_out(tmp_path):
    reference = launch(tmp_path, 'serial', 1)
    for mode, ranks in (('default', 1), ('disabled', 2), ('default', 2), ('default', 4)):
        result = launch(tmp_path, mode, ranks)
        np.testing.assert_allclose(result['energy'], reference['energy'], atol=2e-10, rtol=0)
        np.testing.assert_allclose(result['gradient'], reference['gradient'], atol=2e-8, rtol=0)


def test_system_scope_keeps_root_owned_scf(tmp_path):
    launch(tmp_path, 'system', 2)


@pytest.mark.parametrize('mode', ['worker-failure', 'worker-df-failure', 'root-failure',
                                  'response-failure', 'source-corruption', 'group-failure'])
def test_parallel_scf_failure_is_synchronized_and_cleans_registry(tmp_path, mode):
    launch(tmp_path, mode, 2)


def test_serial_import_does_not_initialize_mpi():
    code = "import sys; import pyscfad.dlno_stc; assert 'mpi4py.MPI' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], cwd=SOURCE_ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_eager_gmres_context_preserves_default_and_unwinds_on_error():
    from pyscfad import config_update
    from pyscfad.tools import linear_solver

    with config_update('pyscfad_moleintor_opt', False):
        assert linear_solver.gen_gmres().func is linear_solver.jax_gmres
        with pytest.raises(RuntimeError, match='leave eager solver scope'):
            with linear_solver.eager_gmres():
                assert linear_solver.gen_gmres().func is linear_solver.pyscfad_gmres
                with linear_solver.eager_gmres():
                    assert linear_solver.gen_gmres().func is linear_solver.pyscfad_gmres
                assert linear_solver.gen_gmres().func is linear_solver.pyscfad_gmres
                raise RuntimeError('leave eager solver scope')
        assert linear_solver.gen_gmres().func is linear_solver.jax_gmres


def test_source_view_preserves_auxiliary_coordinate_subtree():
    import jax
    from pyscfad import scf
    from pyscfad.df.addons import make_auxmol
    from pyscfad.dlno_stc import _mpi_scf
    from pyscfad.dlno_stc.test import _boys_acceptance as accept

    reference = scf.RHF(accept.molecule()).density_fit().with_df
    reference.auxmol = make_auxmol(reference.mol, reference.auxbasis)
    factors = np.ones((2, reference.mol.nao ** 2))
    reference._cderi = factors
    view = _mpi_scf._source_view(reference, lambda: factors)
    assert view.auxmol is reference.auxmol
    assert view.mol is reference.mol
    assert reference._cderi is factors
    assert view._get_cderi_source() is factors
    paths = {jax.tree_util.keystr(path) for path, _ in
             jax.tree_util.tree_flatten_with_path(view)[0]}
    assert '[<flat index 0>].coords' in paths
    assert '[<flat index 1>].coords' in paths
