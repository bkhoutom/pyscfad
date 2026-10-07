"""Native parallel ownership preserves fixed-sample energies and every adjoint."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


WORKER = r"""
import sys
import numpy as np
from pyscfad.dlno_stc import _stc_mp2 as native
rng = np.random.default_rng(329)
foo = np.diag([-1.2, -.9, -.6])
foo[0, 1] = foo[1, 0] = .07
fvv = np.diag(np.linspace(.4, 1.4, 7))
fvv[1, 2] = fvv[2, 1] = .05
B = rng.normal(size=(5, 3, 7)) * .08
B *= np.array([.3, 1., 2.])[None, :, None]
M = np.array([[.2, -.6, .7]])
W = np.array([[.9, .1, -.2], [.1, .7, .05], [-.2, .05, .8]])
offsets = np.array([0, 1, 3, 5], dtype=np.int64)
saved = {}
for scope in ('full', 'domain'):
    for mode in ('deterministic', 'stochastic'):
        controls = dict(mode=mode, laplace_roots=[0., .3, 1.1],
                        laplace_weights=[.07, .4, .6], virtual_block_size=3,
                        global_seed=713, production_samples=10007,
                        min_production_samples=2, virtual_keep_fraction=.34)
        before = B.copy()
        if scope == 'full':
            out = native.solve_full(foo, fvv, B, controls, offsets, True)
        else:
            out = native.solve(foo, fvv, B, M, W, controls, offsets, True)
        np.testing.assert_array_equal(B, before)
        again = (native.solve_full(foo, fvv, B, controls, offsets, True)
                 if scope == 'full' else
                 native.solve(foo, fvv, B, M, W, controls, offsets, True))
        prefix = scope + '_' + mode + '_'
        saved[prefix + 'energy'] = out['energy']
        saved[prefix + 'error'] = out['energy_standard_error']
        for key, value in out['cotangents'].items():
            np.testing.assert_array_equal(value, again['cotangents'][key])
            saved[prefix + key] = value
np.savez(sys.argv[1], **saved)
"""


def test_native_fixed_draws_and_all_bars_across_thread_counts(tmp_path):
    pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    outputs = []
    for threads in (1, 2, 4):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                   OMP_PROC_BIND='FALSE', OMP_DYNAMIC='FALSE',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        output = tmp_path / f'omp{threads}.npz'
        subprocess.run([sys.executable, '-c', WORKER, str(output)],
                       env=env, check=True, capture_output=True, text=True)
        outputs.append(dict(np.load(output)))
    for result in outputs[1:]:
        for key, expected in outputs[0].items():
            np.testing.assert_allclose(result[key], expected,
                                       rtol=3e-12, atol=3e-12, err_msg=key)


def test_zero_auxiliary_padding_preserves_tiled_energy_and_gradients():
    native = pytest.importorskip('pyscfad.dlno_stc._stc_mp2')
    rng = np.random.default_rng(229)
    foo, fvv = -np.eye(2), np.eye(31)
    B = rng.normal(size=(3, 2, 31)) * .04
    padded = np.zeros((100000, 2, 31))
    padded[:3] = B
    M = np.array([[.3, -.8]])
    W = np.array([[.9, .1], [.1, .6]])
    controls = dict(mode='deterministic', laplace_roots=[0.],
                    laplace_weights=[1.], virtual_block_size=13)
    # Zero auxiliary rows cannot change the physical contraction. Their size
    # exercises the bounded-workspace path with rectangular final panels.
    for scope in ('full', 'domain'):
        def solve(tensor):
            if scope == 'full':
                return native.solve_full(foo, fvv, tensor, controls, None, True)
            return native.solve(foo, fvv, tensor, M, W, controls, None, True)
        reference, result = solve(B), solve(padded)
        np.testing.assert_allclose(result['energy'], reference['energy'],
                                   rtol=2e-12, atol=2e-12)
        for key, expected in reference['cotangents'].items():
            actual = result['cotangents'][key]
            if key == 'B':
                np.testing.assert_allclose(actual[:3], expected,
                                           rtol=2e-12, atol=2e-12)
                np.testing.assert_array_equal(actual[3:], 0.)
            else:
                np.testing.assert_allclose(actual, expected,
                                           rtol=2e-12, atol=2e-12)
