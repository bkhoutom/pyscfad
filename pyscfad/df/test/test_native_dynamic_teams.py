"""Local adaptive-team policies must not leak into the caller's runtime."""
import ctypes

import numpy as np
import pytest
from pyscf import lib

from pyscfad.df import _cderi_vjp, _int3c_cross_opt
from pyscfad.df.test.test_reverse_native_slices import _molecules


@pytest.mark.parametrize('dynamic', [0, 1])
@pytest.mark.parametrize('kernel', ['integral', 'packing'])
def test_native_team_policy_restores_caller(dynamic, kernel, monkeypatch):
    runtime = (_int3c_cross_opt.libcgto if kernel == 'integral'
               else _cderi_vjp.libao2mo_vjp)
    runtime.omp_get_dynamic.restype = ctypes.c_int
    runtime.omp_set_dynamic.argtypes = [ctypes.c_int]
    runtime.omp_get_max_threads.restype = ctypes.c_int
    previous = runtime.omp_get_dynamic()
    try:
        with lib.with_omp_threads(4):
            runtime.omp_set_dynamic(dynamic)
            maximum = runtime.omp_get_max_threads()
            if kernel == 'integral':
                mol, aux = _molecules()
                seed = np.ones((mol.nao*(mol.nao+1)//2, aux.nao), order='F')
                bars = _int3c_cross_opt.int3c_cross_bwd(
                    'int3c2e', 1, 's2ij', None, None, (mol, aux), seed)
                assert all(np.isfinite(b.coords).all() for b in bars)
            else:
                monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', 'tiled')
                rng = np.random.default_rng(71)
                c = rng.normal(size=(129, 7))
                seed = rng.normal(size=(2, 7, 7))
                result = _cderi_vjp._int3c_mo_packed_density(c, c, seed)
                assert np.isfinite(result).all()
            assert runtime.omp_get_dynamic() == dynamic
            assert runtime.omp_get_max_threads() == maximum
    finally:
        runtime.omp_set_dynamic(previous)
