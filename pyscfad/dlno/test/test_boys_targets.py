import importlib

import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.lno.lno_base import get_iao


def _targets():
    return importlib.import_module("pyscfad.dlno.targets")


def test_boys_requires_complete_singleton_partition():
    validate = _targets().validate_boys_target_groups
    assert [x.tolist() for x in validate(None, 3)] == [[0], [1], [2]]
    assert [x.tolist() for x in validate([[2], [0], [1]], 3)] == [[2], [0], [1]]
    for bad in ([[0, 1], [2]], [[0], [0], [2]], [[0], [1]],
                [[], [1], [2]], [[0], [1], [3]], [[0.5], [1], [2]]):
        with pytest.raises(ValueError, match="singleton"):
            validate(bad, 3)


def test_target_options_and_one_orbital():
    targets = _targets()
    assert targets.resolve_target_options("BOYS", None) == ("boys", {})
    for mode, kwargs in (("pm", None), ("iao", {"conv_tol": 1e-9}),
                         ("boys", {"minao": "minao"})):
        with pytest.raises(ValueError):
            targets.resolve_target_options(mode, kwargs)
    coeff = np.array([[0.6], [0.8]])
    np.testing.assert_array_equal(targets.build_targets(None, coeff, lo_type="boys"), coeff)
    with pytest.raises(ValueError, match="occupied"):
        targets.build_targets(None, np.zeros((2, 0)), lo_type="boys")


@pytest.fixture(scope="module")
def water_mf():
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0.05 0.76 0.59; H -0.03 -0.70 0.63"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def test_boys_spans_active_occupied_space_and_iao_default(water_mf):
    mf = water_mf
    occ = np.asarray(mf.mo_coeff)[:, 1:5]
    vir = np.asarray(mf.mo_coeff)[:, 5:]
    overlap = np.asarray(mf.get_ovlp())
    targets = _targets()
    localized = np.asarray(targets.build_targets(mf.mol, occ, lo_type="boys"))
    np.testing.assert_allclose(localized.T @ overlap @ localized, np.eye(4), atol=1e-10)
    transform = localized.T @ overlap @ occ
    np.testing.assert_allclose(transform.T @ transform, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(localized.T @ overlap @ vir, 0, atol=1e-10)
    np.testing.assert_allclose(targets.build_targets(mf.mol, occ), get_iao(mf.mol, occ), atol=1e-12)
