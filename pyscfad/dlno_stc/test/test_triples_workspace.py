"""User-selected triples workspace reaches native allocation and survives restart."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import controls, mrcc_triples, _native_triples as native
from pyscfad.dlno_stc.test.test_triples_stable import fixture
from pyscfad.dlno_stc.smooth_triples import make_solver
from pyscfad import lib


@pytest.mark.parametrize('value', [0.5, 16, 64])
def test_deterministic_workspace_control(value):
    supplied = {'workspace_gib': value}
    actual = controls.normalize_triples_controls(supplied, ccsd_t=True)
    assert actual == {'mode': 'deterministic', 'partition': 'mrcc',
                      'workspace_gib': float(value)}
    assert supplied == {'workspace_gib': value}


@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf, True, None, 'bad'])
def test_invalid_workspace_is_rejected(value):
    with pytest.raises(ValueError, match='workspace_gib.*finite and positive'):
        controls.normalize_triples_controls({'workspace_gib': value}, ccsd_t=True)
    with pytest.raises(ValueError, match='workspace_gib.*finite and positive'):
        mrcc_triples.run_mrcc(fixture(selected=2), np.array([.37]), np.array([1.]),
                             with_grad=False, max_memory=2000., workspace_gib=value)


def test_stochastic_workspace_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match='workspace_gib.*deterministic'):
        controls.normalize_triples_controls(
            {'mode': 'stochastic', 'nsamples': 16, 'workspace_gib': 64}, ccsd_t=True)


@pytest.mark.parametrize('with_grad', [False, True])
def test_native_workspace_above_16_gib_preserves_energy_and_bars(with_grad):
    engine = native.StableTriplesDressing(fixture(selected=2), full_external=True,
                                        mrcc_partition=True)
    prepared = engine.prepare(.37)
    default = native.triples_mrcc_point(prepared, with_grad=with_grad)
    larger = native.triples_mrcc_point(prepared, with_grad=with_grad,
                                      workspace_limit_bytes=68719476736)
    assert default['workspace_budget_bytes'] == 17179869184
    assert larger['workspace_budget_bytes'] == 68719476736
    np.testing.assert_allclose(larger['energy'], default['energy'], rtol=3e-12, atol=3e-13)
    if with_grad:
        for actual, expected in zip(larger['cotangents'], default['cotangents']):
            np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-12)


@pytest.mark.parametrize('with_grad', [False, True])
def test_workspace_cap_and_headroom_reach_native_without_changing_result(monkeypatch, with_grad):
    monkeypatch.setattr(mrcc_triples, 'current_memory', lambda: (0., 0.))
    inputs = fixture(selected=2)
    common = dict(with_grad=with_grad, max_memory=137438.953472)
    baseline, expected_bars = mrcc_triples.run_mrcc(inputs, np.array([.37]),
                                                 np.array([1.]), **common)
    assert baseline[1]['workspace_budget_bytes'] == 17179869184
    for cap, total_mb, expected_budget in [(64., 137438.953472, 68719476736),
                                           (64., 10., 10000000),
                                           (1e-6, 2000., 1073)]:
        (actual, diagnostics), bars = mrcc_triples.run_mrcc(
            inputs, np.array([.37]), np.array([1.]), with_grad=with_grad,
            max_memory=total_mb, workspace_gib=cap)
        assert diagnostics['workspace_budget_bytes'] == expected_budget
        np.testing.assert_allclose(actual, baseline[0], rtol=3e-12, atol=3e-13)
        if with_grad:
            for got, want in zip(bars, expected_bars):
                np.testing.assert_allclose(got, want, rtol=3e-10, atol=3e-12)
        else:
            assert bars is None


def test_old_native_module_requires_rebuild(monkeypatch):
    monkeypatch.delattr(native.cpp, 'TRIPLES_CONFIGURABLE_WORKSPACE', raising=False)
    with pytest.raises(RuntimeError, match='Rebuild.*workspace'):
        mrcc_triples.run_mrcc(fixture(selected=2), np.array([.37]), np.array([1.]),
                             with_grad=False, max_memory=2000., workspace_gib=64.)


def test_solver_forwards_workspace_control(monkeypatch):
    monkeypatch.setattr(mrcc_triples, 'current_memory', lambda: (0., 0.))
    t1, t2, gv, go, gg, fov, foo, fvv, u = fixture(selected=2)
    options = controls.normalize_triples_controls({'workspace_gib': 64.}, ccsd_t=True)
    solver = make_solver(options, (np.array([.37]), np.array([1.]), {}), with_grad=False)
    eris = SimpleNamespace(ovvv=lib.pack_tril(gv.reshape(12, 4, 4)).reshape(3, 4, -1),
                           ovoo=go, ovov=gg, fock=np.block([[foo, fov], [fov.T, fvv]]))
    energy, diagnostics = solver(SimpleNamespace(max_memory=137438.953472), eris, u.T,
                                 t1=t1, t2=t2)
    assert np.isfinite(energy)
    assert diagnostics['workspace_budget_bytes'] == 68719476736
