import jax
import numpy
import pytest

from pyscfad import config, gto, scf
from pyscfad.cc import dfccsd


def _make_mol():
    mol = gto.Mole()
    mol.atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def _configure():
    config.update("pyscfad_moleintor_opt", True)
    config.update("pyscfad_scf_implicit_diff", True)
    config.update("pyscfad_ccsd_implicit_diff", True)


def test_dfccsd_grad_matches_implicit_when_custom_enabled():
    mol = _make_mol()

    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        mycc = dfccsd.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot

    _configure()

    config.update("pyscfad_scf_first_order_custom", False)
    grad_ref = jax.grad(energy)(mol)

    config.update("pyscfad_scf_first_order_custom", True)
    grad_test = jax.grad(energy)(mol)

    diff = numpy.asarray(grad_test.coords) - numpy.asarray(grad_ref.coords)
    assert numpy.max(numpy.abs(diff)) < 1e-7


def test_dfccsd_grad_with_explicit_dm0_matches_implicit_when_custom_enabled():
    mol = _make_mol()

    mf_ref = scf.RHF(mol).density_fit()
    mf_ref.conv_tol = 1e-10
    mf_ref.kernel()
    dm_ref = mf_ref.make_rdm1()

    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.kernel(dm_ref)
        mycc = dfccsd.RCCSD(mf)
        mycc.kernel()
        return mycc.e_tot

    _configure()

    config.update("pyscfad_scf_first_order_custom", False)
    grad_ref = jax.grad(energy)(mol)

    config.update("pyscfad_scf_first_order_custom", True)
    grad_test = jax.grad(energy)(mol)

    diff = numpy.asarray(grad_test.coords) - numpy.asarray(grad_ref.coords)
    assert numpy.max(numpy.abs(diff)) < 1e-10
