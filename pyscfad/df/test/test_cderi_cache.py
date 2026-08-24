from pathlib import Path

import h5py
import numpy
import pytest

from pyscfad.df import addons


def _write_cderi(path, values, dataname='j3c'):
    with h5py.File(path, 'w') as h5f:
        h5f.create_dataset(dataname, data=values)


def test_cderi_memory_cache_reuses_one_resident_array(tmp_path, monkeypatch):
    path = tmp_path / 'cderi.h5'
    values = numpy.arange(30, dtype=numpy.float64).reshape(5, 6)
    _write_cderi(path, values)

    read_direct = h5py.Dataset.read_direct
    preload_calls = []

    def count_preload(dataset, destination, *args, **kwargs):
        preload_calls.append(dataset.name)
        return read_direct(dataset, destination, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, 'read_direct', count_preload)

    with addons.load(str(path), 'j3c') as on_disk:
        assert isinstance(on_disk, h5py.Dataset)

    with addons.cderi_memory_cache(str(path)) as resident:
        assert isinstance(resident, numpy.ndarray)
        assert resident.flags.c_contiguous
        assert numpy.array_equal(resident, values)

        with addons.load(str(path), 'j3c') as first:
            with addons.load(Path(path), 'j3c') as second:
                assert first is resident
                assert second is resident

        assert preload_calls == ['/j3c']

    with addons.load(str(path), 'j3c') as on_disk:
        assert isinstance(on_disk, h5py.Dataset)


def test_cderi_memory_cache_is_cleared_after_exception(tmp_path):
    path = tmp_path / 'cderi.h5'
    _write_cderi(path, numpy.ones((3, 4)))

    with pytest.raises(RuntimeError, match='stop inside cache'):
        with addons.cderi_memory_cache(str(path)):
            raise RuntimeError('stop inside cache')

    with addons.load(str(path), 'j3c') as on_disk:
        assert isinstance(on_disk, h5py.Dataset)


def test_cderi_memory_cache_requires_file_source():
    with pytest.raises(TypeError, match='file-backed source'):
        with addons.cderi_memory_cache(numpy.zeros((2, 3))):
            pass
