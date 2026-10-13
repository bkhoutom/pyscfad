"""Policy and fresh-process reuse checks for the local compilation cache."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import pytest


@pytest.fixture
def cache_config(tmp_path, monkeypatch):
    keys = ('jax_enable_compilation_cache', 'jax_compilation_cache_dir',
            'jax_persistent_cache_min_compile_time_secs',
            'jax_persistent_cache_min_entry_size_bytes')
    saved = {key: getattr(jax.config, key) for key in keys}
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path/'cache'))
    for name in ('JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE',
                 'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS',
                 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES',
                 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK',
                 'SLURM_PROCID'):
        monkeypatch.delenv(name, raising=False)
    for key, value in zip(keys, (True, None, 1.0, 0)):
        jax.config.update(key, value)
    yield tmp_path/'cache'
    for key, value in saved.items():
        jax.config.update(key, value)


def runtime():
    return importlib.import_module('pyscfad.lib._compilation_cache')


def test_cache_initialization_is_lazy_and_keeps_small_kernels(cache_config):
    module = importlib.reload(runtime())
    assert jax.config.jax_compilation_cache_dir is None
    assert not cache_config.exists()
    module.configure_compilation_cache()
    directory = cache_config/'pyscfad/jax'
    assert Path(jax.config.jax_compilation_cache_dir) == directory
    assert directory.is_dir()
    assert directory.stat().st_mode & 0o077 == 0
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == -1
    module.configure_compilation_cache()
    assert Path(jax.config.jax_compilation_cache_dir) == directory


def test_explicit_disable_can_be_changed_before_a_later_call(cache_config):
    jax.config.update('jax_enable_compilation_cache', False)
    runtime().configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir is None
    assert not cache_config.exists()
    jax.config.update('jax_enable_compilation_cache', True)
    runtime().configure_compilation_cache()
    assert (cache_config/'pyscfad/jax').is_dir()


@pytest.mark.parametrize('variable', [
    'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK', 'SLURM_PROCID',
])
def test_automatic_cache_is_partitioned_by_launcher_rank(cache_config, monkeypatch, variable):
    monkeypatch.setenv(variable, '2')
    runtime().configure_compilation_cache()
    directory = cache_config/'pyscfad/jax/mpi-rank-2'
    assert Path(jax.config.jax_compilation_cache_dir) == directory
    assert directory.is_dir()
    assert directory.stat().st_mode & 0o077 == 0


def test_launcher_rank_precedence_and_invalid_values(cache_config, monkeypatch):
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', 'invalid')
    monkeypatch.setenv('PMI_RANK', '-1')
    monkeypatch.setenv('PMIX_RANK', '3')
    monkeypatch.setenv('SLURM_PROCID', '4')
    runtime().configure_compilation_cache()
    assert Path(jax.config.jax_compilation_cache_dir) == cache_config/'pyscfad/jax/mpi-rank-3'


@pytest.mark.parametrize('directory', ['', '/explicit/user/cache'])
def test_existing_user_cache_policy_is_preserved(cache_config, monkeypatch, directory):
    monkeypatch.setenv('OMPI_COMM_WORLD_RANK', '7')
    jax.config.update('jax_compilation_cache_dir', directory)
    runtime().configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir == directory
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0
    assert not cache_config.exists()


def test_user_can_replace_an_automatic_cache_path(cache_config, tmp_path):
    runtime().configure_compilation_cache()
    replacement = str(tmp_path/'replacement')
    jax.config.update('jax_compilation_cache_dir', replacement)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 4.0)
    runtime().configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir == replacement
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 4.0


@pytest.mark.parametrize('environment', [False, True])
def test_explicit_thresholds_are_preserved(cache_config, monkeypatch, environment):
    if environment:
        # Explicit environment settings can deliberately equal JAX defaults.
        monkeypatch.setenv('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '1')
        monkeypatch.setenv('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
        expected_time, expected_size = 1.0, 0
    else:
        expected_time, expected_size = 5.0, 4096
        jax.config.update('jax_persistent_cache_min_compile_time_secs', expected_time)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', expected_size)
    runtime().configure_compilation_cache()
    assert jax.config.jax_persistent_cache_min_compile_time_secs == expected_time
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == expected_size


def test_unwritable_automatic_cache_does_not_stop_calculation(cache_config, monkeypatch):
    module = runtime()
    with monkeypatch.context() as scope:
        def denied(*args, **kwargs):
            raise PermissionError('cache directory is read-only')
        scope.setattr(Path, 'mkdir', denied)
        with pytest.warns(RuntimeWarning, match='compilation cache'):
            module.configure_compilation_cache()
    assert jax.config.jax_compilation_cache_dir is None
    # A temporary filesystem failure must not disable later initialization.
    module.configure_compilation_cache()
    assert (cache_config/'pyscfad/jax').is_dir()


@pytest.mark.parametrize('rank', [None, '1', '2'])
def test_compiled_kernel_is_reused_in_a_fresh_process(cache_config, rank):
    program = '''
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax._src import compiler
from pyscfad.lib._compilation_cache import configure_compilation_cache
configure_compilation_cache()
count = [0]
original = compiler.backend_compile_and_load
def measured(*args, **kwargs):
    count[0] += 1
    return original(*args, **kwargs)
compiler.backend_compile_and_load = measured
kernel = jax.jit(lambda x: jnp.sum(jnp.sin(x)**2))
value = float(kernel(np.arange(7, dtype=np.float64)))
print(json.dumps({'compiles': count[0], 'value': value,
                  'directory': jax.config.jax_compilation_cache_dir}))
'''
    env = dict(os.environ, JAX_ENABLE_COMPILATION_CACHE='true')
    if rank is not None:
        env['OMPI_COMM_WORLD_RANK'] = rank
    first, second = [json.loads(subprocess.check_output(
        [sys.executable, '-c', program], env=env, text=True)) for _ in range(2)]
    assert first['compiles'] == 1
    assert second['compiles'] == 0
    assert first['value'] == second['value']
    directory = cache_config/'pyscfad/jax'
    if rank is not None:
        directory /= f'mpi-rank-{rank}'
    assert first['directory'] == second['directory'] == str(directory)
