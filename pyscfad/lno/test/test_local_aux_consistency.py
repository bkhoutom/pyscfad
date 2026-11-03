"""Fragment CC integrals and ED MP2 must share their local fitting metric."""

from types import SimpleNamespace

import jax
import numpy
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad import numpy as np
from pyscfad.df import addons, incore
from pyscfad.lno import df as lno_df
from pyscfad.lno import ccsd as lno_ccsd
from pyscfad.lno import tools


def _problem():
    mol = gto.Mole(atom='O 0 0 0; H 0 0 1; H 0 1 0',
                   basis='sto-3g', verbose=0, max_memory=2000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    atoms = numpy.asarray([0, 2], dtype=numpy.int32)
    coeff = np.asarray(numpy.random.default_rng(81).normal(size=(mol.nao, 4)))
    return mol, atoms, coeff


def _holder(mol, incore_storage):
    with_df = df.DF(mol, auxbasis='weigend', incore=incore_storage)
    with_df.max_memory = 200
    # The parent global DF may already exist; its presence must not change
    # the auxiliary basis chosen for a fragment.
    with_df.build()
    return SimpleNamespace(mol=mol, with_df=with_df,
                           get_ovlp=lambda: mol.intor('int1e_ovlp'))


def _project(mol, coeff, atoms):
    ao = tools.ao_index_by_atom(mol, atoms)
    overlap = mol.intor('int1e_ovlp')
    return jax.scipy.linalg.solve(overlap[np.ix_(ao, ao)],
                                  overlap[ao] @ coeff, assume_a='pos')


@pytest.mark.parametrize('incore_storage', [False, True])
@pytest.mark.parametrize('nmo', [4, 7])
def test_fragment_all_mo_local_aux_energy_and_force(incore_storage, nmo, monkeypatch):
    """All-pair CC tensors and coordinate/coefficient responses match dense local DF."""
    mol, atoms, coeff = _problem()
    coeff = np.asarray(numpy.random.default_rng(81).normal(size=(mol.nao, nmo)))
    local = lno_df.make_local_mol(mol, atoms)
    aux = addons.make_auxmol(local, 'weigend')
    global_aux = addons.make_auxmol(mol, 'weigend')
    assert aux.nao < global_aux.nao
    weight = np.asarray(numpy.random.default_rng(7).normal(size=(aux.nao, nmo*nmo)))
    parent_df = _holder(mol, incore_storage).with_df

    def reference(m, c):
        lm = lno_df.make_local_mol(m, atoms)
        la = addons.make_auxmol(lm, 'weigend')
        b = incore.cholesky_eri(lm, auxmol=la, aosym='s1')
        projected = _project(m, c, atoms)
        y = np.einsum('Puv,ui,vj->Pij', b.reshape(la.nao, lm.nao, lm.nao),
                      projected, projected).reshape(la.nao, -1)
        return np.sum(y * weight)

    def actual(m, c):
        # Production builds the parent disk cache before tracing the force.
        # Rebuild only the differentiable domain data inside this objective.
        mf = SimpleNamespace(mol=m, with_df=parent_df,
                             get_ovlp=lambda: m.intor('int1e_ovlp'))
        y = lno_df.transform_df_to_mo(mf, c, (0, nmo, 0, nmo), atmlst=atoms)
        assert y.shape == (aux.nao, nmo*nmo)
        return np.sum(y * weight)

    # Allocating cached per-domain AO-pair DF defeats the memory bound.
    def reject_cached_local_df(*args, **kwargs):
        raise AssertionError('Fragment CC should transform shell blocks directly')
    monkeypatch.setattr(lno_df, 'get_local_df', reject_cached_local_df)
    expected, expected_bar = jax.value_and_grad(reference, argnums=(0, 1))(mol, coeff)
    value, bar = jax.value_and_grad(actual, argnums=(0, 1))(mol, coeff)
    numpy.testing.assert_allclose(value, expected, atol=1e-9, rtol=1e-9)
    for result, ref in zip(jax.tree_util.tree_leaves(bar),
                           jax.tree_util.tree_leaves(expected_bar)):
        numpy.testing.assert_allclose(result, ref, atol=2e-8, rtol=2e-8)


def test_full_lis_mp2_matches_extended_domain_mp2():
    """With identical orbital spaces, the MP2 add/subtract terms must cancel."""
    mol, atoms, coeff = _problem()
    mf = _holder(mol, False)
    projected = _project(mol, coeff, atoms)
    # The W preparation calls get_local_Lov with integral_direct=True.
    ed_lov = lno_df.get_local_Lov(mf, projected, 2, atoms, integral_direct=True)
    cc_lpq = lno_df.transform_df_to_mo(mf, coeff, (0, 4, 0, 4), atmlst=atoms)
    cc_lov = cc_lpq.reshape(-1, 4, 4)[:, :2, 2:]
    numpy.testing.assert_allclose(cc_lov, ed_lov, atol=1e-11, rtol=1e-11)
    eps_o, eps_v = np.asarray([-1., -.7]), np.asarray([.3, .8])
    denom = (eps_o[:, None, None, None] + eps_o[None, :, None, None]
             - eps_v[None, None, :, None] - eps_v[None, None, None, :])
    def mp2(lov):
        g = np.einsum('Pia,Pjb->ijab', lov, lov)
        return np.sum(g * (2*g - g.transpose(0, 1, 3, 2)) / denom)
    numpy.testing.assert_allclose(mp2(cc_lov), mp2(ed_lov), atol=1e-10, rtol=1e-11)


def test_fragment_direct_budget_respects_user_total_and_workspace(monkeypatch):
    mol, atoms, coeff = _problem()
    mf = _holder(mol, False)
    seen = []
    def capture(m, a, c, limit, slices):
        seen.append(limit)
        return np.zeros((a.nao, 16))
    monkeypatch.setattr(lno_df, '_local_direct_nr_e2', capture)
    monkeypatch.setattr(lno_df.pyscf_lib, 'current_memory', lambda: (1000., 0.))
    lno_df.transform_df_to_mo(mf, coeff, (0, 4, 0, 4), atmlst=atoms)
    mol.max_memory = 1100
    lno_df.transform_df_to_mo(mf, coeff, (0, 4, 0, 4), atmlst=atoms)
    assert seen == [1200, 1100]


def test_fragment_ccsd_integral_builder_uses_domain_auxiliary_basis():
    """The actual CC builder must pass its domain through to local fitting."""
    with config_update('pyscfad_moleintor_opt', True):
        mol, atoms, _ = _problem()
        mf = scf.RHF(mol).density_fit(auxbasis='weigend').run()
        solver = lno_ccsd.RCCSD(mf)
        solver._domain_atmlst = atoms
        eris = solver.ao2mo()
        local = lno_df.make_local_mol(mol, atoms)
        aux = addons.make_auxmol(local, mf.with_df.auxbasis)
        assert aux.nao < mf.with_df.auxmol.nao
        assert eris.Lov.shape[0] == aux.nao
        projected = _project(mol, mf.mo_coeff, atoms)
        ed_lov = lno_df.get_local_Lov(mf, projected, solver.nocc, atoms,
                                     integral_direct=True)
        numpy.testing.assert_allclose(eris.Lov, ed_lov, atol=1e-11, rtol=1e-11)
