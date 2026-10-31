"""DF ladder cache integration through the physical LNO solver and CC APIs."""
import numpy
import pytest

from pyscfad import config_update, gto, lib, scf
from pyscfad.dlno_stc import ccsd
from pyscfad.lno import ccsd as lno_ccsd


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
def test_lno_integral_builder_obeys_ladder_mode(direct):
    """LNO overrides ao2mo; it must populate the cache just like canonical CC."""
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
        assert eris.ovvv is None
        tau = numpy.random.default_rng(10).normal(size=(3, 2, 2))
        L = numpy.asarray(lib.unpack_tril(eris.Lvv))
        expected = numpy.einsum('pcd,Pac,Pbd->pab', tau, L, L)
        numpy.testing.assert_allclose(eris._contract_vvvv_t2(solver, tau), expected,
                                      atol=1e-12, rtol=1e-12)


def test_impurity_solver_sets_ladder_mode_before_integral_build(monkeypatch):
    """Inner fragment RCCSD must obey the option before ao2mo creates its cache."""
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
