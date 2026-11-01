"""Regressions for retaining canonical MP2's forward MO factors."""

import h5py
import jax
import numpy
import pytest

from pyscfad import config_update, gto
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons, incore
from pyscfad.mp import dfmp2


@pytest.fixture
def transform_case(tmp_path):
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g", verbose=0, max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    auxmol = addons.make_auxmol(mol, "weigend")
    source = tmp_path / "cderi.h5"
    with config_update("pyscfad_moleintor_opt", True):
        cderi = incore.cholesky_eri(
            mol, auxmol=auxmol,
            int3c=mol._add_suffix("int3c2e"),
            int2c=mol._add_suffix("int2c2e"),
            aosym="s2ij", verbose=0,
        )
    with h5py.File(source, "w") as handle:
        handle.create_dataset("j3c", data=numpy.asarray(cderi))
    rng = numpy.random.default_rng(63)
    coeff = np.asarray(rng.normal(size=(mol.nao, 5)))
    # An unused MO column also checks the full coefficient cotangent.
    orbs_slice = (1, 3, 3, 5)
    ybars = tuple(np.asarray(rng.normal(size=(auxmol.nao, 4)))
                  for _ in range(2))
    return mol, auxmol, coeff, str(source), orbs_slice, ybars


def _dense_transform(mol, auxmol, coeff, orbs_slice):
    cderi = incore.cholesky_eri(
        mol, auxmol=auxmol,
        int3c=mol._add_suffix("int3c2e"),
        int2c=mol._add_suffix("int2c2e"),
        aosym="s2ij", verbose=0,
    )
    return _ao2mo.nr_e2(cderi, coeff, orbs_slice, aosym="s2")


def _assert_bars_close(actual, expected):
    for index in (0, 1):
        numpy.testing.assert_allclose(
            actual[index].coords, expected[index].coords,
            atol=2e-8, rtol=2e-8,
        )
    numpy.testing.assert_allclose(
        actual[2], expected[2], atol=1e-9, rtol=1e-9,
    )


def test_saved_lov_repeated_pullbacks_match_dense(transform_case):
    """Saving Lov must preserve arbitrary cotangents and repeated pullbacks."""
    mol, auxmol, coeff, source, orbs_slice, ybars = transform_case
    with config_update("pyscfad_moleintor_opt", True):
        expected, reference_pullback = jax.vjp(
            lambda m, a, c: _dense_transform(m, a, c, orbs_slice),
            mol, auxmol, coeff,
        )
        actual, pullback = jax.vjp(
            lambda m, a, c: dfmp2._outcore_nr_e2(
                m, a, c, source, 1000, orbs_slice, "s2"
            ),
            mol, auxmol, coeff,
        )
        original_output = numpy.array(actual, copy=True)
        numpy.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        # Reuse the same residual with different cotangents, then repeat the
        # first. A stale cotangent or destructive update of Lov changes bars.
        for ybar in (ybars[0], ybars[1], ybars[0]):
            _assert_bars_close(pullback(ybar), reference_pullback(ybar))
            numpy.testing.assert_array_equal(actual, original_output)


def test_backward_reuses_lov_without_streamed_ao2mo(
        transform_case, monkeypatch):
    """The MO adjoint still reads CDERI, but no reverse AO2MO is needed."""
    mol, auxmol, coeff, source, orbs_slice, ybars = transform_case
    calls = {"ao2mo": 0, "mo_adjoint": 0}
    original_transform = _cderi_vjp._stream_nr_e2_from_cderi_source
    original_mo_adjoint = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source

    def count_transform(*args, **kwargs):
        calls["ao2mo"] += 1
        return original_transform(*args, **kwargs)

    def count_mo_adjoint(*args, **kwargs):
        calls["mo_adjoint"] += 1
        return original_mo_adjoint(*args, **kwargs)

    monkeypatch.setattr(
        _cderi_vjp, "_stream_nr_e2_from_cderi_source", count_transform
    )
    monkeypatch.setattr(
        _cderi_vjp, "nr_e2_mo_coeff_vjp_from_cderi_source", count_mo_adjoint
    )
    with config_update("pyscfad_moleintor_opt", True):
        _, pullback = jax.vjp(
            lambda m, a, c: dfmp2._outcore_nr_e2(
                m, a, c, source, 1000, orbs_slice, "s2"
            ),
            mol, auxmol, coeff,
        )
        assert calls["ao2mo"] == 1
        for index, ybar in enumerate(ybars, start=1):
            pullback(ybar)
            assert calls["mo_adjoint"] == index
            assert calls["ao2mo"] == 1, (
                "backward reconstructed forward Lov from disk-backed CDERI"
            )
