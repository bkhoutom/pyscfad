from types import SimpleNamespace

import numpy
import pytest

from pyscfad.dlno.ccsd_mpi import _validate_fragment_mo_width


def _fake_mf(nmo):
    return SimpleNamespace(mo_coeff=numpy.empty((2 * nmo, nmo)))


def test_fragment_mo_width_accepts_full_layout():
    _validate_fragment_mo_width(
        _fake_mf(6), numpy.empty((12, 6)), rank=1, fragment=3,
    )


def test_fragment_mo_width_accepts_empty_fragment():
    _validate_fragment_mo_width(
        _fake_mf(6), None, rank=1, fragment=3,
    )


def test_fragment_mo_width_rejects_missing_columns():
    with pytest.raises(
        RuntimeError,
        match=(
            r'Invalid fragment MO layout on rank 2, fragment 4: '
            r'expected 6 MO columns, got 5'
        ),
    ):
        _validate_fragment_mo_width(
            _fake_mf(6), numpy.empty((12, 5)), rank=2, fragment=4,
        )


def test_fragment_mo_width_rejects_nonmatrix_layout():
    with pytest.raises(RuntimeError, match=r'expected a rank-2 array'):
        _validate_fragment_mo_width(
            _fake_mf(6), numpy.empty(6), rank=0, fragment=1,
        )
