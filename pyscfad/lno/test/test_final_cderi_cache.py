from types import SimpleNamespace

import h5py
import numpy
import pytest

from pyscfad.df import addons
from pyscfad.dlno import ccsd_mpi


def test_final_cderi_cache_context_selects_disk_or_memory(tmp_path):
    path = tmp_path / 'cderi.h5'
    values = numpy.arange(20, dtype=numpy.float64).reshape(4, 5)
    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset('j3c', data=values)

    with_df = SimpleNamespace(_get_cderi_source=lambda: str(path))
    mf = SimpleNamespace(with_df=with_df)

    with ccsd_mpi._final_cderi_cache_context(mf, 'disk') as resident:
        assert resident is None

    with ccsd_mpi._final_cderi_cache_context(mf, 'memory') as resident:
        assert numpy.array_equal(resident, values)
        with addons.load(str(path), 'j3c') as loaded:
            assert loaded is resident


def test_final_cderi_cache_context_rejects_invalid_mode():
    with pytest.raises(ValueError, match="either 'disk' or 'memory'"):
        ccsd_mpi._final_cderi_cache_context(SimpleNamespace(), 'auto')


def test_final_cderi_cache_context_requires_density_fitting():
    with pytest.raises(RuntimeError, match='density-fitted SCF object'):
        ccsd_mpi._final_cderi_cache_context(SimpleNamespace(), 'memory')
