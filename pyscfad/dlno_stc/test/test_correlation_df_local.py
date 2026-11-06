"""Independent local-fit checks for the optional correlation auxiliary basis."""
from types import SimpleNamespace

import h5py
import jax
import numpy
import pytest

from pyscfad import config_update, gto, scf
from pyscfad import numpy as np
from pyscfad.df import addons, incore
from pyscfad.lno import df as lno_df
from pyscfad.lno import ccsd as lno_ccsd
from pyscfad.lno import _df_shared
from pyscfad.lno import tools


ATOMS = numpy.asarray([0, 2], dtype=numpy.int32)
CORRELATION_BASIS = 'cc-pvdz-ri'


def _problem():
    mol = gto.Mole(atom='O 0 0 0; H .1 0 1; H 0 1 .1',
                   basis='sto-3g', verbose=0, max_memory=2000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    nao = len(tools.ao_index_by_atom(mol, ATOMS))
    coeff = np.asarray(numpy.random.default_rng(181).normal(size=(nao, 4)) * .2)
    return mol, coeff


def _mf(mol):
    # A sentinel global fit makes accidental JK-source use fail immediately.
    return SimpleNamespace(mol=mol, get_ovlp=lambda: mol.intor('int1e_ovlp'),
        with_df=SimpleNamespace(auxbasis='weigend', max_memory=64,
                                _cderi=object(), auxmol=object()))


def _oracle(mol, coeff, atoms, basis, slices):
    local = lno_df.make_local_mol(mol, atoms)
    aux = addons.make_auxmol(local, basis)
    factors = incore.cholesky_eri(local, auxmol=aux, aosym='s1')
    k0, k1, l0, l1 = slices
    return np.einsum('Puv,ui,vj->Pij', factors.reshape(aux.nao, local.nao, local.nao),
                     coeff[:, k0:k1], coeff[:, l0:l1]).reshape(aux.nao, -1)


def _forbid_storage(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Explicit correlation fit must use local direct integrals')
    monkeypatch.setattr(h5py, 'File', forbidden)
    monkeypatch.setattr(lno_df, 'get_local_df', forbidden)


@pytest.mark.parametrize('basis', [CORRELATION_BASIS,
    {'O': CORRELATION_BASIS, 'H': CORRELATION_BASIS}])
def test_unequal_local_basis_matches_integral_oracle_and_preserves_jk(basis, monkeypatch):
    mol, coeff = _problem()
    mf = _mf(mol)
    before = vars(mf.with_df).copy()
    local = lno_df.make_local_mol(mol, ATOMS)
    jk_aux = addons.make_auxmol(local, mf.with_df.auxbasis)
    corr_aux = addons.make_auxmol(local, basis)
    assert jk_aux.nao != corr_aux.nao
    expected = _oracle(mol, coeff, ATOMS, basis, (0, 2, 2, 4))
    _forbid_storage(monkeypatch)
    # Even an explicit false cannot request packed AO storage for this new fit.
    actual = lno_df.get_local_Lov(mf, coeff, 2, ATOMS, integral_direct=False,
                                 correlation_auxbasis=basis)
    assert actual.shape == (corr_aux.nao, 2, 2)
    numpy.testing.assert_allclose(actual.reshape(corr_aux.nao, -1), expected,
                                  atol=2e-11, rtol=2e-11)
    all_pairs = lno_df.transform_df_to_mo(mf, coeff, (0, 4, 0, 4),
        atmlst=ATOMS, coeff_is_local=True, correlation_auxbasis=basis)
    numpy.testing.assert_allclose(all_pairs.reshape(-1, 4, 4)[:, :2, 2:], actual,
                                  atol=2e-11, rtol=2e-11)
    assert vars(mf.with_df) == before


def test_explicit_system_transform_uses_all_atoms_direct(monkeypatch):
    mol, _ = _problem()
    coeff = np.asarray(numpy.random.default_rng(11).normal(size=(mol.nao, 4)) * .2)
    atoms = numpy.arange(mol.natm)
    expected = _oracle(mol, coeff, atoms, CORRELATION_BASIS, (0, 2, 2, 4))
    _forbid_storage(monkeypatch)
    actual = lno_df.transform_df_to_mo(_mf(mol), coeff, (0, 2, 2, 4),
                                     correlation_auxbasis=CORRELATION_BASIS)
    numpy.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)


def test_correlation_coordinate_coefficient_and_shared_vjp_match_oracle(monkeypatch):
    mol, coeff = _problem()
    def objective(m, c):
        y = lno_df.get_local_Lov(_mf(m), c, 2, ATOMS,
                                 correlation_auxbasis=CORRELATION_BASIS)
        return np.sum(y**2)
    def reference(m, c):
        y = _oracle(m, c, ATOMS, CORRELATION_BASIS, (0, 2, 2, 4))
        return np.sum(y**2)
    expected, (mol_ref, coeff_ref) = jax.value_and_grad(reference, argnums=(0, 1))(mol, coeff)
    _forbid_storage(monkeypatch)
    actual, (mol_bar, coeff_bar) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
    numpy.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
    numpy.testing.assert_allclose(mol_bar.coords, mol_ref.coords, atol=2e-9, rtol=2e-8)
    numpy.testing.assert_allclose(coeff_bar, coeff_ref, atol=2e-10, rtol=2e-9)
    with _df_shared.reverse_scope(_mf(mol), ATOMS,
            correlation_auxbasis=CORRELATION_BASIS) as scope:
        _, (mol_shared, coeff_shared) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
        assert scope.receipt['consumers'] == 1
        geometry = scope.finish()
        numpy.testing.assert_allclose(mol_shared.coords + geometry, mol_ref.coords,
                                      atol=2e-9, rtol=2e-8)
        numpy.testing.assert_allclose(coeff_shared, coeff_ref, atol=2e-10, rtol=2e-9)
        assert scope.receipt['metric_builds'] == scope.receipt['geometry_passes'] == 1


def test_none_preserves_existing_local_direct_values():
    mol, coeff = _problem()
    mf = _mf(mol)
    implicit = lno_df.get_local_Lov(mf, coeff, 2, ATOMS, integral_direct=True)
    explicit = lno_df.get_local_Lov(mf, coeff, 2, ATOMS, integral_direct=True,
                                   correlation_auxbasis=None)
    numpy.testing.assert_array_equal(explicit, implicit)
    expected = _oracle(mol, coeff, ATOMS, mf.with_df.auxbasis, (0, 2, 2, 4))
    numpy.testing.assert_allclose(explicit.reshape(expected.shape), expected,
                                  atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('atoms', [ATOMS, None])
def test_impurity_core_and_cc_integrals_use_separate_fit(atoms, monkeypatch):
    with config_update('pyscfad_moleintor_opt', True):
        mol, _ = _problem()
        mf = scf.RHF(mol).density_fit(auxbasis='weigend').run()
        parent = mf.with_df
        jk_source, jk_auxmol = parent._cderi, parent.auxmol
        coeff = mf.mo_coeff
        overlap = mf.get_ovlp()
        domain_atoms = numpy.arange(mol.natm) if atoms is None else atoms
        if atoms is None:
            local_coeff = coeff
        else:
            ao = tools.ao_index_by_atom(mol, atoms)
            local_coeff = jax.scipy.linalg.solve(overlap[np.ix_(ao, ao)],
                                                overlap[ao] @ coeff, assume_a='pos')
        expected = _oracle(mol, local_coeff, domain_atoms, CORRELATION_BASIS,
                           (0, coeff.shape[1], 0, coeff.shape[1]))
        expected = expected.reshape(-1, coeff.shape[1], coeff.shape[1])
        seen = []
        def amplitudes(mcc, eris, t1, t2, *, prjlo):
            assert mcc.correlation_auxbasis == CORRELATION_BASIS
            numpy.testing.assert_allclose(eris.Lov, expected[:, :mcc.nocc, mcc.nocc:],
                                          atol=2e-11, rtol=2e-11)
            seen.append(True)
            return t1, t2, {}
        _forbid_storage(monkeypatch)
        lno_ccsd._impurity_solve_core(mf, coeff, coeff[:, :1], mf.get_fock(), overlap,
            frag_prescreen=None if atoms is None else {'extended_primary_domain': atoms},
            amplitude_solver=amplitudes, correlation_auxbasis=CORRELATION_BASIS)
        assert seen == [True]
        assert mf.with_df is parent
        assert parent._cderi is jk_source and parent.auxmol is jk_auxmol
        assert parent.auxbasis == 'weigend'
