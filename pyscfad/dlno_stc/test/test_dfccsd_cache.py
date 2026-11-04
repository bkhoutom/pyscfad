"""DF ladder cache integration through the physical LNO solver and CC APIs."""
import copy
import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscfad import config_update, gto, lib, scf
from pyscfad.cc import dfccsd
from pyscfad.cc import ccsd_lambda
from pyscfad.dlno_stc import ccsd
from pyscfad.lno import ccsd as lno_ccsd


def test_response_integral_adjoint_reuses_the_single_linearization(monkeypatch):
    with config_update('pyscfad_moleintor_opt', True):
        mol = gto.Mole(atom='H 0 0 0; H 0 0 1.4; H 0 0 3.4; H 0 0 4.8',
                       basis='sto-3g', unit='Bohr', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        solver = dfccsd.RCCSD(scf.RHF(mol).density_fit().run())
        eris = solver.ao2mo()
        solver.kernel(eris=eris)
        t1, t2 = solver.t1, solver.t2
        _, expected_lambda = ccsd_lambda.solve_response_lambda(
            solver, eris, t1, t2, 1., t1 * .01, t2 * .02, max_cycle=3)
        expected = dfccsd.lagrangian_grad(solver, eris, t1, t2, 1., expected_lambda)
        calls = []
        original = solver.update_amps
        def counted(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        monkeypatch.setattr(solver, 'update_amps', counted)
        _, actual_lambda, saved = ccsd_lambda.solve_response_lambda(
            solver, eris, t1, t2, 1., t1 * .01, t2 * .02,
            max_cycle=3, return_intermediates=True)
        actual = dfccsd.lagrangian_grad(
            solver, eris, t1, t2, 1., actual_lambda, intermediates=saved)
        assert len(calls) == 1
        numpy.testing.assert_allclose(actual_lambda, expected_lambda, atol=1e-12, rtol=0)
        for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            numpy.testing.assert_allclose(got, want, atol=1e-11, rtol=0)


def test_ccsd_reads_cached_ovvv_for_updates_and_amplitude_response():
    """Once cached, changing Lov cannot change an amplitude contraction."""
    with config_update('pyscfad_moleintor_opt', True):
        mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                       basis='sto-3g', verbose=0, max_memory=4000)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit().run()
        solver = lno_ccsd.RCCSD(mf)
        eris = solver.ao2mo()
        t1, t2 = solver.init_amps(eris=eris)[1:]
        poisoned = copy.copy(eris)
        poisoned.Lov = jnp.full_like(eris.Lov, jnp.nan)
        expected, pull_expected = jax.vjp(
            lambda a, b: solver.update_amps(a, b, eris), t1, t2)
        actual, pull_actual = jax.vjp(
            lambda a, b: solver.update_amps(a, b, poisoned), t1, t2)
        seeds = tuple(jnp.ones_like(x) for x in expected)
        for got, want in zip((*actual, *pull_actual(seeds)),
                             (*expected, *pull_expected(seeds))):
            numpy.testing.assert_allclose(got, want, atol=1e-12, rtol=1e-12)


def test_smooth_triples_fragment_response_with_cached_ovvv(monkeypatch):
    """The LNO ERI VJP composes with CCSD and the native smooth (T) VJP."""
    from pyscfad.dlno_stc.smooth_triples import make_solver

    cached_mode = [True]
    original = lno_ccsd._make_df_eris_incore

    def build(*args, **kwargs):
        eris = original(*args, **kwargs)
        assert eris.ovvv is not None
        if not cached_mode[0]:
            eris.ovvv = None
        return eris

    monkeypatch.setattr(lno_ccsd, '_make_df_eris_incore', build)
    with config_update('pyscfad_moleintor_opt', True), \
         config_update('pyscfad_ccsd_implicit_diff', True):
        mol = gto.Mole(atom='H 0 0 0; H 0 0 1.4; H 0 0 3.4; H 0 0 4.8',
                       basis='sto-3g', unit='Bohr', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit().run()
        fock, overlap = mf.get_fock(), mf.get_ovlp()
        fragment = mf.mo_coeff[:, :1]
        triples = make_solver({'mode': 'deterministic'},
            (numpy.array([.3]), numpy.array([1.]), {}), with_grad=True)

        def objective(coeff):
            (emp2, ecc, et), _ = lno_ccsd._impurity_solve_core(
                mf, coeff, fragment, fock, overlap,
                ccsd_t=True, triples_solver=triples)
            return ecc + et - emp2

        value, bar = jax.value_and_grad(objective)(mf.mo_coeff)
        triples = make_solver({'mode': 'deterministic', 'integral_transform': 'rotate'},
            (numpy.array([.3]), numpy.array([1.]), {}), with_grad=True)
        rotated, rotated_bar = jax.value_and_grad(objective)(mf.mo_coeff)
        numpy.testing.assert_allclose(rotated, value, atol=1e-11, rtol=0)
        numpy.testing.assert_allclose(rotated_bar, bar, atol=2e-9, rtol=0)
        triples = make_solver({'mode': 'deterministic'},
            (numpy.array([.3]), numpy.array([1.]), {}), with_grad=True)
        cached_mode[0] = False
        reference, refbar = jax.value_and_grad(objective)(mf.mo_coeff)
        numpy.testing.assert_allclose(value, reference, atol=1e-11, rtol=0)
        numpy.testing.assert_allclose(bar, refbar, atol=2e-9, rtol=0)


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('entry', ['kernel', 'value_and_grad'])
def test_public_cc_api_forwards_ladder_mode(monkeypatch, direct, entry):
    """A requested memory/performance mode must reach the serial/MPI runner."""
    captured = {}
    def run(*args, **options):
        captured.update(options)
        return 'prepared'
    monkeypatch.setattr(ccsd, '_run', run)
    options = dict(baseline_controls=None, matrix_controls=None, vvvv_direct=direct)
    if entry == 'kernel':
        result = ccsd.kernel(None, None, **options)
    else:
        result = ccsd.value_and_grad(None, None, **options)
    assert result == 'prepared'
    assert captured['vvvv_direct'] is direct


@pytest.mark.parametrize('direct', [False, True])
def test_lno_integral_builder_obeys_ladder_mode(direct, monkeypatch):
    """LNO overrides ao2mo; it must populate the cache just like canonical CC."""
    # Independent of JAX executables retained by preceding gradient tests.
    monkeypatch.setattr(dfccsd, 'current_memory', lambda: (0, 0))
    with config_update('pyscfad_moleintor_opt', True):
        mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                       basis='sto-3g', verbose=0, max_memory=4000)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit()
        mf.kernel()
        solver = lno_ccsd.RCCSD(mf)
        solver.vvvv_direct = direct
        eris = solver.ao2mo()
        cached = getattr(eris, '_vvvv_pair', None)
        assert (cached is None) is direct
        assert eris.Lov is not None and eris.Lvv is not None
        assert eris.ovvv is not None
        expected_ovvv = numpy.einsum('Pia,Pq->iaq', eris.Lov, eris.Lvv)
        numpy.testing.assert_allclose(eris.ovvv, expected_ovvv, atol=1e-12)
        tau = numpy.random.default_rng(10).normal(size=(3, 2, 2))
        L = numpy.asarray(lib.unpack_tril(eris.Lvv))
        expected = numpy.einsum('pcd,Pac,Pbd->pab', tau, L, L)
        numpy.testing.assert_allclose(eris._contract_vvvv_t2(solver, tau), expected,
                                      atol=1e-12, rtol=1e-12)


def test_impurity_solver_sets_ladder_mode_before_integral_build(monkeypatch):
    """Inner fragment RCCSD must obey the option before ao2mo creates its cache."""
    monkeypatch.setattr(dfccsd, 'current_memory', lambda: (0, 0))
    with config_update('pyscfad_moleintor_opt', True):
        mol = gto.Mole(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol).density_fit().run()
        observed = []
        ao2mo = lno_ccsd.RCCSD.ao2mo
        def build(solver, *args, **kwargs):
            eris = ao2mo(solver, *args, **kwargs)
            observed.append((solver.vvvv_direct, eris._vvvv_pair is not None))
            return eris
        monkeypatch.setattr(lno_ccsd.RCCSD, 'ao2mo', build)
        args = (mf, mf.mo_coeff, mf.mo_coeff[:, :1], mf.get_fock(), mf.get_ovlp())
        cached = lno_ccsd._impurity_solve_core(*args, vvvv_direct=False)
        direct = lno_ccsd._impurity_solve_core(*args, vvvv_direct=True)
        assert observed == [(False, True), (True, False)]
        numpy.testing.assert_allclose(cached, direct, atol=1e-12, rtol=0)
