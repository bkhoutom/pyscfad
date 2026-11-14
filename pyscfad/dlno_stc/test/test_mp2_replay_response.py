"""Saved MP2 packets retain the production SCF and DF response lifecycle."""
from functools import partial

import numpy as np
import pytest

from pyscfad import config_update, df, dlno_stc, scf
from pyscfad.df import _ooc_cotangent
from pyscfad.dlno_stc import backend, domain, driver, mp2
from pyscfad.dlno_stc.test.test_correlation_auxbasis import molecule, _controls, RIFIT
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.tools import linear_solver


@pytest.mark.parametrize('outcore', [False, True])
def test_replay_reuses_final_scf_data_and_closes_shared_response(
        molecule, tmp_path, monkeypatch, outcore):
    monkeypatch.setattr(scf.hf, 'gen_gmres', partial(
        linear_solver.gen_gmres_with_default_kwargs, tol=1e-11))
    path = str(tmp_path / 'scf.h5')
    if outcore:
        source = df.DF(molecule, auxbasis='weigend', incore=False)
        source._cderi_to_save = path
        source.build()
    builds, rebuilds, receipts = [], [], []

    def build(current):
        builds.append(1)
        source = df.DF(current, auxbasis='weigend', incore=not outcore)
        if outcore:
            source.attach_outcore_cderi(path)
        mf = scf.RHF(current).density_fit(with_df=source)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        with linear_solver.eager_gmres():
            mf.kernel()
        return mf

    controls = _controls()
    with config_update('pyscfad_moleintor_opt', True):
        mf = build(molecule)
        static = dlno_stc.build_domain_selections(mf, frozen=1,
            force_full_domains=True, boys_grouping='molecule', correlation_auxbasis=RIFIT)
        assert static.ed_selection_pending
        packet_dir = tmp_path / 'packets'
        rows = mp2.export_run(mf, static, packet_dir, controls=controls,
                              correlation_auxbasis=RIFIT)
        assert static.ed_selection_pending  # Export preserves the caller's plan.
        for row in rows:
            folder = packet_dir / f"fragment_{row['fragment_id']:04d}"
            mp2.solve_request(folder / 'input.h5', folder / 'result.h5', backend.solve)
        expected = dlno_stc.value_and_grad(molecule, build, static, controls=controls,
                                           include_hf=True, correlation_auxbasis=RIFIT)

        original_data = domain.build_scf_data
        def counted_data(*args, **kwargs):
            rebuilds.append(1)
            return original_data(*args, **kwargs)
        monkeypatch.setattr(domain, 'build_scf_data', counted_data)
        monkeypatch.setattr(driver, 'build_scf_data', counted_data)
        original_finish = _ooc_cotangent.GeometryScope.finish
        def finished(scope, bar):
            result = original_finish(scope, bar)
            receipts.append(dict(scope.stats))
            return result
        monkeypatch.setattr(_ooc_cotangent.GeometryScope, 'finish', finished)
        builds.clear()
        actual = mp2.imported_value_and_grad(molecule, build, static,
            lambda index: mp2.load_disk_packet(packet_dir, index),
            include_hf=True, correlation_auxbasis=RIFIT)

    np.testing.assert_allclose(actual[0], expected[0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(actual[1].coords, expected[1].coords, atol=2e-8, rtol=0)
    assert len(builds) == 1
    assert rebuilds == []
    assert len(receipts) == int(outcore)
    if outcore:
        assert receipts[0]['geometry_passes'] == 1
        assert receipts[0]['metric_builds'] == receipts[0]['cholesky_builds'] == 1
    assert _ooc_cotangent._scope.get() is None


def test_partial_replays_sum_to_complete_strong_response(boys_water_dimer, tmp_path, monkeypatch):
    from pyscfad.dlno_stc.test.test_finite_workflow import build_mf
    mf, _, static = boys_water_dimer
    controls = dict(_controls(), energy_tolerance=3e-4)
    rows = mp2.export_run(mf, static, tmp_path, controls=controls)
    assert len(rows) > 1
    for row in rows:
        folder = tmp_path / f"fragment_{row['fragment_id']:04d}"
        mp2.solve_request(folder / 'input.h5', folder / 'result.h5', backend.solve)
    reads = []
    def load(index):
        reads.append(index)
        return mp2.load_disk_packet(tmp_path, index)

    options = dict(include_hf=False, include_weak=False)
    original_scope = driver._mp2_df_scope
    scopes = []
    def checked_scope(state, atoms, aux_atoms=None):
        assert aux_atoms is not None
        assert any(np.array_equal(atoms, fragment.extended_atoms)
                   and np.array_equal(aux_atoms, fragment.pao_center_atoms)
                   for fragment in static.fragments)
        scopes.append(1)
        return original_scope(state, atoms, aux_atoms)
    monkeypatch.setattr(driver, '_mp2_df_scope', checked_scope)
    complete = mp2.imported_value_and_grad(mf.mol, build_mf, static, load, **options)
    assert reads == list(range(len(rows)))
    partitions = ((0,), tuple(range(1, len(rows))))
    partials = []
    for indices in partitions:
        reads.clear()
        partials.append(mp2.imported_value_and_grad(
            mf.mol, build_mf, static, load, fragment_ids=indices, **options))
        assert reads == list(indices)
    np.testing.assert_allclose(sum(value for value, _ in partials), complete[0], atol=1e-11, rtol=0)
    np.testing.assert_allclose(sum(np.asarray(bar.coords) for _, bar in partials),
                               complete[1].coords, atol=2e-8, rtol=0)
    assert scopes
