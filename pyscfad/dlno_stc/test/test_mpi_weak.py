"""Small numeric MPI tests; run with mpiexec -n 4 python -m pytest this_file."""
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import _pairs


def test_weak_mpi_subprocess():
    """Bound the real collective failure checks so a regression cannot hang CI."""
    MPI = pytest.importorskip('mpi4py.MPI')
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip('the subprocess wrapper runs outside mpiexec')
    launcher = shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('mpiexec is unavailable')
    source = Path(__file__).resolve().parents[3]
    env = dict(os.environ, OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYSCFAD_DENSE_BLAS_THREADS='1', JAX_PLATFORMS='cpu',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    env['PYTHONPATH'] = str(source) + os.pathsep + env.get('PYTHONPATH', '')
    command = [launcher, '-n', '4', sys.executable, '-m', 'pytest', __file__,
               '-q', '-k', 'not test_weak_mpi_subprocess']
    try:
        completed = subprocess.run(command, cwd=source, env=env, capture_output=True,
                                   text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'weak MPI checks hung past 120 s: {exc.stdout!r} {exc.stderr!r}')
    assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
    assert '11 passed' in completed.stdout


class Inputs(NamedTuple):
    coordinates: object
    scale: object


class Common(NamedTuple):
    s1e: object


def _fixture(order):
    mf = Inputs(jnp.array([[0., .1, .2], [.2, .1, -.1], [-.1, .3, .2]]), jnp.array(.7))
    common = Common(jnp.array([.2, .4, .6]))
    static = SimpleNamespace(thresholds=SimpleNamespace(multipole_order=order))
    pairs = ((2, 5), (2, 9), (5, 9))

    def endpoint(m, c, s, index):
        # Different compact sizes exercise variable-size exchange buffers.
        index = (2, 5, 9).index(index)
        nvir = index + 1
        center = m.coordinates[index] + jnp.array([5. * index, .2 * index, 0.])
        mu = jnp.arange(1., 3*nvir + 1).reshape(3, nvir) * (.03 + .01*c.s1e[index])
        gap = jnp.arange(nvir)*.3 + .6 + .01*m.scale + .02*c.s1e[index]
        theta = jnp.arange(9*nvir).reshape(3, 3, nvir)*(.001 + .0001*m.scale) if order > 2 else None
        omega = jnp.arange(27*nvir).reshape(3, 3, 3, nvir)*(.0001 + .00001*c.s1e[index]) if order > 3 else None
        return center, mu, gap, theta, omega
    return mf, common, static, pairs, endpoint


def _comm():
    MPI = pytest.importorskip('mpi4py.MPI')
    if MPI.COMM_WORLD.Get_size() < 2:
        pytest.skip('requires mpiexec -n 2 or more')
    return MPI.COMM_WORLD


@pytest.mark.parametrize('order,with_grad', [(2, True), (4, True), (3, False)])
def test_distributed_endpoint_reuse_sign_alignment_and_idle_rank(monkeypatch, order, with_grad):
    comm = _comm()
    from pyscfad.dlno_stc import _mpi_weak
    mf, common, static, pairs, endpoint = _fixture(order)
    calls = Counter()

    def counted(m, c, s, index):
        calls[index] += 1
        result = endpoint(m, c, s, index)
        if calls[index] > 1:
            center, mu, gap, theta, omega = result
            signs = jnp.where(jnp.arange((2, 5, 9).index(index) + 1) % 2, 1., -1.)
            result = (center, mu*signs, gap,
                      None if theta is None else theta*signs,
                      None if omega is None else omega*signs)
        return result

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', counted)
    value, bars = _mpi_weak.weak_pair_contribution(
        mf, common, static, pairs, comm, with_grad=with_grad)
    values = comm.allgather(float(value))
    host_bars = comm.allgather(jax.tree_util.tree_map(np.asarray, bars)) if with_grad else None
    if not with_grad:
        assert bars is None
    all_calls = comm.allgather(dict(calls))
    owned = [i for slot, i in enumerate((2, 5, 9))
             if slot % comm.Get_size() == comm.Get_rank()]
    assert calls == Counter({i: 2 if with_grad else 1 for i in owned})
    assert sum(sum(row.values()) for row in all_calls) == 3*(2 if with_grad else 1)

    def direct(m, c):
        return sum(_pairs._multipole_pair_energy(endpoint(m, c, static, i),
                   endpoint(m, c, static, j), order) for i, j in pairs)

    expected, expected_bars = jax.value_and_grad(direct, argnums=(0, 1))(mf, common)
    np.testing.assert_allclose(sum(values), expected, rtol=2e-12, atol=1e-14)
    if with_grad:
        total_bars = jax.tree_util.tree_map(lambda *xs: sum(xs), *host_bars)
        for got, want in zip(jax.tree_util.tree_leaves(total_bars),
                             jax.tree_util.tree_leaves(expected_bars)):
            np.testing.assert_allclose(got, want, rtol=2e-11, atol=1e-13)
    assert _mpi_weak.LAST_DIAGNOSTICS['n_endpoint_builds'] == len(owned)
    assert _mpi_weak.LAST_DIAGNOSTICS['n_endpoint_replays'] == len(owned)*with_grad
    if comm.Get_rank() >= 3:
        for leaf in jax.tree_util.tree_leaves(bars):
            np.testing.assert_array_equal(leaf, np.zeros_like(leaf))


def test_empty_graph_returns_live_zero_trees(monkeypatch):
    comm = _comm()
    from pyscfad.dlno_stc import _mpi_weak
    mf, common, static, _, _ = _fixture(2)
    value, bars = _mpi_weak.weak_pair_contribution(mf, common, static, (), comm, with_grad=True)
    assert float(value) == 0.
    assert jax.tree_util.tree_structure(bars) == jax.tree_util.tree_structure((mf, common))
    for leaf in jax.tree_util.tree_leaves(bars):
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))


@pytest.mark.parametrize('phase', ['build', 'pair', 'replay-gap', 'replay-moment'])
def test_worker_errors_are_collective(monkeypatch, phase):
    comm = _comm()
    from pyscfad.dlno_stc import _mpi_weak
    mf, common, static, pairs, endpoint = _fixture(2)
    calls = Counter()

    def fail(m, c, s, index):
        calls[index] += 1
        if comm.Get_rank() == 1 and index == 5:
            if phase == 'build':
                raise ValueError('injected weak build failure')
            if phase.startswith('replay-') and calls[index] > 1:
                center, mu, gap, theta, omega = endpoint(m, c, s, index)
                if phase == 'replay-gap':
                    gap = gap + .1
                else:
                    mu = mu.at[0, 0].add(.1)
                return center, mu, gap, theta, omega
        return endpoint(m, c, s, index)

    monkeypatch.setattr(_pairs, 'build_weak_endpoint', fail)
    real_pair = _pairs.pair_energy_and_grad
    def fail_pair(*args):
        if comm.Get_rank() == 1:
            raise ValueError('injected weak pair failure')
        return real_pair(*args)
    if phase == 'pair':
        monkeypatch.setattr(_pairs, 'pair_energy_and_grad', fail_pair)
    with pytest.raises(RuntimeError, match='weak|endpoint'):
        _mpi_weak.weak_pair_contribution(mf, common, static, pairs, comm, with_grad=True)


def test_endpoint_receive_allocation_failure_is_collective(monkeypatch):
    comm = _comm()
    from pyscfad.dlno_stc import _mpi_weak
    from pyscfad.dlno_stc._mpi import _raise_if_any_rank_failed
    mf, common, static, _, endpoint = _fixture(2)
    indices, owners = [2], {2: 0}
    prepared = {2: endpoint(mf, common, static, 2)} if comm.Get_rank() == 0 else {}
    real_empty = np.empty

    def fail_receive(shape, *args, **kwargs):
        if comm.Get_rank() == 1 and shape and kwargs.get('dtype') is np.uint8:
            raise MemoryError('injected weak receive allocation')
        return real_empty(shape, *args, **kwargs)

    monkeypatch.setattr(_mpi_weak.np, 'empty', fail_receive)
    with pytest.raises(RuntimeError, match='weak.*allocation'):
        _mpi_weak._exchange_endpoints(comm, prepared, indices, owners,
                                      _raise_if_any_rank_failed)


@pytest.mark.parametrize('empty', [False, True])
def test_energy_only_avoids_all_cotangent_preparation(monkeypatch, empty):
    comm = _comm()
    from pyscfad.dlno_stc import _mpi_weak
    mf, common, static, pairs, endpoint = _fixture(2)
    monkeypatch.setattr(_pairs, 'build_weak_endpoint', endpoint)
    def forbidden(*args):
        raise AssertionError('energy-only weak path prepared cotangents')
    monkeypatch.setattr(_mpi_weak, 'zero_term_cotangents', forbidden)
    monkeypatch.setattr(_pairs, 'pair_energy_and_grad', forbidden)
    value, bars = _mpi_weak.weak_pair_contribution(
        mf, common, static, () if empty else pairs, comm, with_grad=False)
    assert bars is None
    assert np.isfinite(float(value))
