"""Shared local integral response, including different frames and W feedback."""
from types import SimpleNamespace
import sys

import h5py
import jax
import numpy
import pytest

from pyscfad import gto
from pyscfad import numpy as np
from pyscfad.lno import df as lno_df
from pyscfad.lno import _df_shared
from pyscfad.dlno_stc import ccsd


ATOMS = numpy.asarray([0, 2], dtype=numpy.int32)


def _mf(mol):
    return SimpleNamespace(mol=mol, get_ovlp=lambda: mol.intor('int1e_ovlp'),
        with_df=SimpleNamespace(auxbasis='weigend', max_memory=64),
        verbose=0, stdout=sys.stdout)


def _problem(nmo):
    mol = gto.Mole(atom='O 0 0 0; H .1 0 1; H 0 1 .1', basis='sto-3g',
                   verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    coeff = np.asarray(numpy.random.default_rng(79).normal(size=(mol.nao, nmo)) * .2)
    return mol, coeff


def _energy(mol, coeff):
    # Different ED OV and LIS all-pair spaces, with a downstream dependence
    # on ED factors standing in for the W -> LIS frame response.
    ed = lno_df.transform_df_to_mo(_mf(mol), coeff,
                                  (0, 2, 2, coeff.shape[1]), atmlst=ATOMS)
    lis_coeff = coeff[:, :3] * (1 + .03 * np.sum(ed**2))
    lis = lno_df.transform_df_to_mo(_mf(mol), lis_coeff, (0, 3, 0, 3),
                                   atmlst=ATOMS).reshape(-1, 3, 3)
    weight = np.asarray([[.2, -.1, .3], [.5, .7, -.2], [-.4, .2, .1]])
    return .1*np.sum(ed**2) + np.sum(lis**2*weight) - .2*np.sum(lis[:, :1, 1:]**2)


@pytest.mark.parametrize('nmo', [4, 7])
@pytest.mark.parametrize('seed', [.7, -.3])
def test_shared_coordinate_and_orbital_bars_match_separate_reverses(nmo, seed, monkeypatch):
    mol, coeff = _problem(nmo)
    objective = lambda m, c: seed * _energy(m, c)
    value_ref, (mol_ref, coeff_ref) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
    original_rows = _df_shared.LocalDFReverse._rows
    monkeypatch.setattr(_df_shared.LocalDFReverse, '_rows',
        lambda self, *args: min(2, original_rows(self, *args)))
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        value, (mol_bar, coeff_bar) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
        assert len(scope.terms) == 2
        assert all(isinstance(term[0], h5py.Dataset) for term in scope.terms)
        geometry = scope.finish()
        numpy.testing.assert_allclose(mol_bar.coords + geometry, mol_ref.coords,
                                      atol=2e-9, rtol=2e-8)
        numpy.testing.assert_allclose(coeff_bar, coeff_ref, atol=2e-10, rtol=2e-9)
        numpy.testing.assert_allclose(value, value_ref, atol=1e-13)
        assert scope.receipt['consumers'] == 2
        assert scope.receipt['geometry_passes'] == scope.receipt['metric_passes'] == 1
        assert scope.receipt['max_density_rows'] <= 2
        assert scope.receipt['spooled_bytes'] > 0
        assert scope.terms == []
    assert scope._file is None and scope._handle is None
    # Finite differences include the overlap projection and ED-to-LIS chain.
    step = 1e-5
    # Update libcint's geometry as well as the traced coordinate attribute.
    # Assigning only .coords leaves ordinary forward integral values unchanged.
    plus = mol.set_geom_(numpy.asarray(mol.coords.at[2, 1].add(step)),
                         unit='Bohr', inplace=False)
    minus = mol.set_geom_(numpy.asarray(mol.coords.at[2, 1].add(-step)),
                          unit='Bohr', inplace=False)
    fd = (objective(plus, coeff) - objective(minus, coeff)) / (2*step)
    numpy.testing.assert_allclose(mol_bar.coords[2, 1] + geometry[2, 1], fd,
                                  atol=2e-8, rtol=3e-6)


def test_owner_adds_shared_geometry_once_and_preserves_other_bars(monkeypatch):
    mol, coeff = _problem(4)
    expected, reference = jax.value_and_grad(lambda m: _energy(m, coeff))(mol)
    static = SimpleNamespace(mp2_static=SimpleNamespace(
        fragments=[SimpleNamespace(extended_atoms=ATOMS)]))
    group = SimpleNamespace(representative_index=0)
    initial = .123
    state = dict(mf=_mf(mol))
    def responses(state, *args, **kwargs):
        value, bar = jax.value_and_grad(lambda m: _energy(m, coeff))(mol)
        bar.coords = bar.coords + initial
        state['mf_bar'] = SimpleNamespace(mol=bar)
        return [float(value)]
    monkeypatch.setattr(ccsd, '_run_group_responses', responses)
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '1')
    records = ccsd._run_group(state, static, group, {}, {}, with_grad=True,
                              verbose_imp=0, ccsd_t=True)
    numpy.testing.assert_allclose(records, [expected], atol=1e-13)
    numpy.testing.assert_allclose(state['mf_bar'].mol.coords, reference.coords+initial,
                                  atol=2e-9, rtol=2e-8)


def test_budget_exhaustion_and_scope_failure_do_not_leak(monkeypatch):
    mol, coeff = _problem(4)
    with pytest.raises(MemoryError, match='user memory'):
        with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
            monkeypatch.setattr(_df_shared.pyscf_lib, 'current_memory',
                                lambda: (mol.max_memory+1., 0.))
            jax.grad(lambda m: _energy(m, coeff))(mol)
    assert scope._file is None and scope._handle is None
    assert _df_shared._ACTIVE.get() is None
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='finished and added'):
        with _df_shared.reverse_scope(_mf(mol), ATOMS) as unfinished:
            jax.grad(lambda m: _energy(m, coeff))(mol)
    assert unfinished._file is None and unfinished._handle is None
    assert _df_shared._ACTIVE.get() is None
    # A subsequent ordinary gradient must still include its integral response.
    bar = jax.grad(lambda m: _energy(m, coeff))(mol)
    assert numpy.isfinite(bar.coords).all()


def test_nested_scope_and_double_finish_rejected():
    mol, _ = _problem(4)
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        with pytest.raises(RuntimeError, match='Nested'):
            with _df_shared.reverse_scope(_mf(mol), ATOMS):
                pass
        numpy.testing.assert_array_equal(scope.finish(), numpy.zeros((mol.natm, 3)))
        with pytest.raises(RuntimeError, match='already finished'):
            scope.finish()
