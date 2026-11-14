"""Packed real AO moments preserve screening energies and nuclear response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto
from pyscfad.dlno_stc import _pairs
from pyscfad.lib._cartesian_moments import cartesian_components
from pyscfad.lib import _multipole
from pyscfad.lib.test.test_multipole import _moment_inputs, _reference_transition


@pytest.fixture(scope='module', params=[False, True], ids=['spherical', 'cartesian'])
def mol(request):
    molecule = gto.Mole(atom='O .2 -.4 .1; H .1 .8 1.3; H 1.2 .3 -.7',
                        unit='Bohr', basis='cc-pvdz', cart=request.param, verbose=0)
    molecule.build(trace_exp=False, trace_ctr_coeff=False)
    return molecule


@pytest.mark.parametrize('rank', [0, 1, 2, 3])
def test_packed_slabs_and_coordinate_response(mol, rank, monkeypatch):
    # Exercise multiple shell slabs even on a small molecule.
    monkeypatch.setattr(_pairs, '_MOMENT_SLAB_BYTES', 1024)
    indices, inverse = cartesian_components(rank)
    intor = ('int1e_ovlp', 'int1e_r', 'int1e_rr', 'int1e_rrr')[rank]
    with mol.with_common_origin((0., 0., 0.)):
        full = np.asarray(mol.intor(intor)).reshape(3**rank, mol.nao, mol.nao)
        bra = -np.asarray(mol.intor(intor + '_dr10')).reshape(3, 3**rank, mol.nao, mol.nao)
        ket = -np.asarray(mol.intor(intor + '_dr01')).reshape(3**rank, 3, mol.nao, mol.nao).transpose(1, 0, 2, 3)
    np.testing.assert_allclose(bra.swapaxes(-1, -2), ket, atol=3e-13, rtol=3e-13)
    actual, pullback = jax.vjp(lambda m: _pairs._raw_cartesian_moment(m, rank), mol)
    assert actual.shape == (len(indices), mol.nao, mol.nao)
    np.testing.assert_allclose(np.asarray(actual)[inverse], full, atol=3e-13, rtol=3e-13)
    rng = np.random.default_rng(83 + rank)
    seed = rng.normal(size=actual.shape)
    gradient, = pullback(seed)
    direction = rng.normal(size=mol.coords.shape)
    owner = np.repeat(np.arange(mol.natm), np.diff(mol.aoslice_by_atom()[:, 2:], axis=1).ravel())
    expected_t = _multipole.coordinate_moment_jvp(bra, ket, direction, owner)[indices]
    np.testing.assert_allclose(np.sum(gradient.coords * direction), np.sum(expected_t * seed),
                               rtol=2e-11, atol=2e-10)
    step = 2e-5
    def reference(sign):
        shifted = mol.set_geom_(np.asarray(mol.coords) + sign * step * direction,
                                unit='Bohr', inplace=False)
        with shifted.with_common_origin((0., 0., 0.)):
            value = np.asarray(shifted.intor(intor)).reshape(3**rank, mol.nao, mol.nao)
        return np.sum(value[indices] * seed)
    finite_difference = (reference(1) - reference(-1)) / (2 * step)
    np.testing.assert_allclose(np.sum(gradient.coords * direction), finite_difference,
                               rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize('order', [2, 3, 4])
def test_all_orders_use_only_bra_derivative_and_unique_components(mol, order, monkeypatch):
    calls = []
    original = type(mol).intor
    def record(self, name, *args, **kwargs):
        calls.append(name)
        return original(self, name, *args, **kwargs)
    monkeypatch.setattr(type(mol), 'intor', record)
    raw, pullback = jax.vjp(lambda m: _pairs._origin_zero_moments(m, order - 1), mol)
    gradient, = pullback(jax.tree.map(jnp.ones_like, raw))
    jax.block_until_ready(gradient)
    assert sum(value.size // mol.nao**2 for value in raw if value is not None) == {2:4, 3:10, 4:20}[order]
    assert not any('_dr01' in name for name in calls)
    assert sum('_dr10' in name for name in calls) == order


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('shared', [False, True])
def test_packed_transition_values_and_response(order, shared):
    dense = _moment_inputs(shared)
    packed = dense[:4] + tuple(value.reshape(3**rank, *value.shape[-2:])[cartesian_components(rank)[0]]
                              for rank, value in ((2, dense[4]), (3, dense[5])))
    def expand(values):
        return values[:4] + tuple(value[cartesian_components(rank)[1]].reshape((3,)*rank + value.shape[-2:])
                                 for rank, value in ((2, values[4]), (3, values[5])))
    got, pb = jax.vjp(lambda *values: _multipole.transition_moments(*values, order), *packed)
    expected, ref_pb = jax.vjp(lambda *values: _multipole.transition_moments(*expand(values), order), *packed)
    independent = _reference_transition(dense, order)
    rng = np.random.default_rng(736)
    seed = jax.tree.map(lambda x: rng.normal(size=x.shape), got)
    for a, b, c in zip(jax.tree.leaves(got), jax.tree.leaves(expected), jax.tree.leaves(independent)):
        np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(a, c, atol=2e-11, rtol=2e-12)
    for a, b in zip(jax.tree.leaves(pb(seed)), jax.tree.leaves(ref_pb(seed))):
        np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-11)
