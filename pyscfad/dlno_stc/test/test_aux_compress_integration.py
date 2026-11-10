"""Auxiliary compression at the LIS solver and public driver boundaries."""
import jax
import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno_stc import ccsd
from pyscfad.lno import ccsd as lno_ccsd
from pyscfad.dlno_stc.test.test_fragment_df_inputs import molecular_case


@pytest.fixture(scope='module')
def hydrogen():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 0.8', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
@pytest.mark.parametrize('tol', [-1., float('nan'), float('inf')])
def test_invalid_tolerance_rejected_before_preparation(api, tol):
    options = dict(baseline_controls=None, matrix_controls=None,
                   lis_aux_svd_tol=tol)
    with pytest.raises(ValueError, match='lis_aux_svd_tol'):
        if api == 'kernel':
            ccsd.kernel(None, None, **options)
        else:
            ccsd.value_and_grad(None, lambda mol: None, **options)


def test_compressor_precedes_every_integral_block(hydrogen):
    cc = lno_ccsd.RCCSD(hydrogen)
    original = cc.ao2mo()
    cc.df_compressor = lambda factors: factors * 0.5
    compressed = cc.ao2mo()
    for name in ('Loo', 'Lov', 'Lvv'):
        np.testing.assert_allclose(getattr(compressed, name),
                                   0.5 * getattr(original, name), atol=1e-13)
    for name in ('oooo', 'ovoo', 'ovov', 'oovv'):
        np.testing.assert_allclose(getattr(compressed, name),
                                   0.25 * getattr(original, name), atol=1e-13)
    np.testing.assert_array_equal(compressed.fock, original.fock)


def test_actual_impurity_vjp_factors_once(hydrogen, monkeypatch):
    from pyscfad.dlno_stc import _aux_compress

    mf = hydrogen
    compressor = _aux_compress.make_compressor(1e-4)
    calls = []
    original_svd = np.linalg.svd

    def counted(*args, **kwargs):
        calls.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, 'svd', counted)
    occupied = mf.mo_coeff[:, :1]
    fock, overlap = mf.get_fock(), mf.get_ovlp()

    def correction(coeff):
        emp2, ecc, et = lno_ccsd._impurity_solve_core(
            mf, coeff, occupied, fock, overlap, df_compressor=compressor)
        return ecc + et - emp2

    value, pullback = jax.vjp(correction, mf.mo_coeff)
    assert len(calls) == 1
    gradient, = pullback(jax.numpy.ones_like(value))
    assert len(calls) == 1
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert compressor.diagnostics['retained_naux'] < compressor.diagnostics['original_naux']


def test_domain_correction_reports_completed_compression(molecular_case, monkeypatch):
    """The W/LIS correction consumes each SVD once and reports its reverse work."""
    case = molecular_case
    state = ccsd.driver._initialize(case.mf, case.domains, case.controls,
        with_grad=True, scf_pullback=case.pullback)
    calls = []
    original_svd = np.linalg.svd

    def counted(*args, **kwargs):
        calls.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(np.linalg, 'svd', counted)
    records = ccsd._run_group(state, case.static, case.group,
        case.controls, case.controls, with_grad=True, verbose_imp=0,
        ccsd_t=False, include_baseline=False, fragment_indices=case.indices,
        lis_aux_svd_tol=1e-4)
    assert len(calls) == len(records) == len(case.indices)
    value, gradient = ccsd.driver._finish(state, with_grad=True, include_hf=False)
    assert np.isfinite(value) and np.isfinite(gradient.coords).all()
    for record in records:
        info = record.auxiliary_compression
        assert info['retained_naux'] < info['original_naux']
        assert info['forward_s'] > 0 and info['pullback_s'] > 0
        assert record.triples_diagnostics is None


def test_restart_rejects_different_compression(molecular_case, tmp_path):
    """An existing CC energy cannot be resumed with another integral model."""
    from pyscfad.dlno_stc._restart import RestartMismatchError
    case = molecular_case
    options = dict(baseline_controls=case.controls, matrix_controls=case.controls,
        include_baseline=False, fragment_indices=case.indices,
        checkpoint_dir=tmp_path, lis_aux_svd_tol=1e-4)
    reference = ccsd.kernel(case.mf, case.static, **options)
    resumed = ccsd.kernel(case.mf, case.static, resume=True, **options)
    np.testing.assert_allclose(resumed, reference, atol=1e-13, rtol=0)
    options['lis_aux_svd_tol'] = 1e-6
    with pytest.raises(RestartMismatchError):
        ccsd.kernel(case.mf, case.static, resume=True, **options)


def test_replica_compression_diagnostics_are_separate_from_triples():
    from pyscfad.dlno_stc import _stc_precision

    controls = dict(energy_tolerance=1e-3, precision_pilot_replicas=2,
                    precision_min_replicas=2, precision_max_replicas=2)

    def evaluate(config, with_grad):
        diagnostic = dict(ccsd={}, auxiliary_compression=dict(
            original_naux=12, retained_naux=4, forward_s=.1,
            pullback_s=.2 if with_grad else 0.))
        return -.1, ((-.2, -.3, 0.), diagnostic), np.ones(2) if with_grad else None

    _, (_, diagnostic), bars = _stc_precision.run_replicas(
        evaluate, controls, 0, 1, with_grad=True)
    compression = diagnostic['auxiliary_compression']
    assert len(compression['pilot']) == len(compression['production']) == 2
    assert all(row['retained_naux'] == 4 for row in compression['production'])
    assert all(row['pullback_s'] == .2 for row in compression['production'])
    assert 'replica_triples_diagnostics' not in diagnostic
    np.testing.assert_array_equal(bars, np.ones(2))
