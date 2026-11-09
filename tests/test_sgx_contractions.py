"""Compact contractions versus full dense algebra, including signed seeds."""
import numpy as np
import pytest
from pyscf.sgx import sgx_jk
from pyscfad.sgx import _native, _sgx_grad


def panels():
    rng = np.random.default_rng(145)
    loc = np.array([0, 1, 4, 9, 15], dtype=np.int32)
    ngrids = 2*_sgx_grad._CONTRACTION_BLOCK+13
    mask = rng.integers(0, 3, size=((ngrids+55)//56, len(loc)-1), dtype=np.uint8)
    mask[:16] = 0  # An empty subpanel must still retain the F*dG terms.
    mask[16:24, 1:] = 0
    ao = rng.normal(size=(ngrids, loc[-1]))
    dao = rng.normal(size=(3, ngrids, loc[-1]))
    for p in range(0, ngrids, sgx_jk.BLKSIZE):
        active = np.repeat(mask[p//sgx_jk.BLKSIZE] != 0, np.diff(loc))
        ao[p:p+sgx_jk.BLKSIZE, ~active] = 0.
        dao[:, p:p+sgx_jk.BLKSIZE, ~active] = 0.
    return rng, loc, mask, ao, dao


def test_compact_front_all_outputs_and_tail_reuse():
    rng, loc, mask, ao, _ = panels()
    dm = rng.normal(size=(3, loc[-1], loc[-1]))  # Deliberately nonsymmetric.
    expected = np.einsum('gu,xuv->xvg', ao, dm)
    actual = _sgx_grad._front_contract(ao, dm, mask, loc)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=2e-14)
    tail = _sgx_grad._front_contract(ao[:73], dm, mask, loc, out=actual)
    np.testing.assert_allclose(tail, expected[:, :, :73], atol=1e-13, rtol=2e-14)
    dense = _sgx_grad._front_contract(ao, dm, None, loc)
    np.testing.assert_allclose(dense, expected, atol=1e-13, rtol=2e-14)


@pytest.mark.parametrize('native', [False, True])
def test_compact_geometry_reductions_against_dense(native, monkeypatch):
    dll = _native.library()
    if native and (dll is None or not hasattr(dll, 'SGX_contract_geometry')):
        pytest.skip('Rebuilt libsgx_ad required for fused reductions')
    if not native:
        monkeypatch.setattr(_native, 'reduce_geometry', lambda *a: False)
    rng, loc, mask, ao, dao = panels()
    nao, ngrids = loc[-1], len(ao)
    fg = rng.normal(size=(4, nao, ngrids))
    gv = rng.normal(size=fg.shape)
    dgv = rng.normal(size=(4, 3, nao, ngrids))
    projected = rng.normal(size=(4, nao, nao))
    weights = rng.normal(size=ngrids)
    atom = rng.normal(size=(3, nao))
    energy = rng.normal(size=ngrids)
    # Non-contiguous output view as used for a panel of the full-grid dxed.
    motion = rng.normal(size=(3, 2*ngrids))[:, :ngrids]
    ea, ee, em = atom.copy(), energy.copy(), motion.copy()
    for i in range(len(fg)):
        j = i ^ 1
        tmp = np.einsum('uv,vg->ug', projected[j], gv[i])*(.5*weights)
        wfg = fg[j]*(.5*weights)
        ea -= np.einsum('xgu,ug->xu', dao, tmp)
        ea -= np.einsum('xug,ug->xu', dgv[i], wfg)
        ee += np.einsum('ug,ug->g', fg[j], gv[i])
        em -= np.einsum('ug,xug->xg', wfg, dgv[i])
        em -= np.einsum('ug,xgu->xg', tmp, dao)
    _sgx_grad._geometry_contract(dao, fg, gv, dgv, projected, weights,
                                mask, loc, atom, energy, motion)
    np.testing.assert_allclose(atom, ea, atol=8e-12, rtol=3e-13)
    np.testing.assert_allclose(energy, ee, atol=5e-13, rtol=3e-13)
    np.testing.assert_allclose(motion, em, atol=1e-12, rtol=3e-13)


def test_panel_alignment_preserves_forward_response_and_geometry(monkeypatch):
    """Changing outer panels must preserve screened tiles on every K path."""
    from pyscf import gto, scf
    from pyscfad.sgx import sgx_fit
    dll = _native.library()
    if dll is None or dll.SGX_generated_version() < 4:
        pytest.skip('Per-pair generated SGX required')
    mol = gto.M(atom='O 0 0 0; H 0 -.7 .5; H 0 .7 .5',
                basis='cc-pvdz', verbose=0, max_memory=2000)
    mf = sgx_fit(scf.RHF(mol), grid_level=0, max_memory=500,
                 integral_backend='generated', screening='per_pair',
                 mixed_precision=True, energy_tol=1e-10, potential_tol=1e-8,
                 fp32_multiplier=1e5)
    mf.build()
    dm = scf.RHF(mol).get_init_guess()
    mf.get_k(dm=dm)  # Build the shared overlap map before varying panels.
    obj = mf.with_df
    plan = _native.get_plan(obj._pjs_data)
    rng = np.random.default_rng(462)
    bar = rng.normal(size=dm.shape)/mol.nao
    bar = .5*(bar+bar.T)
    original = _sgx_grad._grid_plan
    results, counts, observed = [], [], []
    for points in (896, 1792):
        def planner(*args, **kwargs):
            result = original(*args, **kwargs)
            assert result['native_batch_points'] == 128
            assert result['block_alignment_points'] == 896
            result.update(block_points=points, grid_bytes=points*result['bytes_per_grid'],
                          selected_grid_jobs=points//128)
            observed.append(bool(args[4]))
            return result
        monkeypatch.setattr(_sgx_grad, '_grid_plan', planner)
        plan.reset_stats()
        forward = mf.get_k(dm=dm)
        forward_counts = plan.screening_stats().copy()
        geometry = _sgx_grad.geometry(obj, dm, bar, mf.direct_scf_tol)
        all_counts = plan.screening_stats().copy()
        accuracy = plan.accuracy
        response = _sgx_grad.density_response(obj, bar, mf.direct_scf_tol)
        assert plan.accuracy is accuracy
        results.append((forward, geometry, response))
        counts.append((forward_counts, all_counts))
    for before, after in zip(*results):
        np.testing.assert_allclose(after, before, atol=1e-12, rtol=0)
    assert counts[0] == counts[1]
    assert True in observed and False in observed
