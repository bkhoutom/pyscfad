"""Native weak AO-moment response preserves nuclear gradients without AO tapes."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno_stc import _pairs
from pyscfad.gto import _moleintor_vjp


def _molecule(cart=False, *, trace_exp=False, trace_ctr_coeff=False, basis='cc-pvdz'):
    mol = gto.Mole()
    mol.atom = 'O 0.11 -0.23 0.07; H 0.84 0.15 0.62; H -0.47 0.71 -0.31'
    mol.basis = basis
    mol.cart = cart
    mol.verbose = 0
    mol.build(trace_exp=trace_exp, trace_ctr_coeff=trace_ctr_coeff)
    return mol


def _seeds(raw, seed=52):
    rng = np.random.default_rng(seed)
    return tuple(None if value is None else jnp.asarray(
        rng.normal(size=value.shape)) for value in raw)


def _dense_coordinate_reference(mol, atoms, seeds):
    """Independent explicit bra-plus-ket integral contraction, without AD."""
    local = _pairs._fake_mol_with_traced_centers(mol, atoms)
    result = np.zeros((mol.natm, 3))
    names = ('int1e_ovlp', 'int1e_r', 'int1e_rr', 'int1e_rrr')
    with local.with_common_origin((0., 0., 0.)):
        for rank, (name, seed) in enumerate(zip(names, seeds)):
            if seed is None:
                continue
            ncomp, nao = 3**rank, local.nao
            # Select independent operator components directly in the dense
            # reference, retaining both AO derivative legs.
            from itertools import combinations_with_replacement
            indices = [sum(axis * 3**(rank-k-1) for k, axis in enumerate(axes))
                       for axes in combinations_with_replacement(range(3), rank)]
            weight = np.asarray(seed).reshape(len(indices), nao, nao)
            bra = -np.asarray(local.intor(name + '_dr10')).reshape(3, ncomp, nao, nao)
            ket = -np.asarray(local.intor(name + '_dr01')).reshape(
                ncomp, 3, nao, nao).transpose(1, 0, 2, 3)
            bra, ket = bra[:, indices], ket[:, indices]
            for atom, (_, _, p0, p1) in zip(atoms, local.aoslice_by_atom()):
                result[atom] += np.einsum('puv,xpuv->x', weight[:, p0:p1], bra[:, :, p0:p1])
                result[atom] += np.einsum('puv,xpuv->x', weight[:, :, p0:p1], ket[:, :, :, p0:p1])
    return result


def test_zero_weights_give_zero_moment_coordinate_response():
    response = getattr(_pairs, '_origin_zero_moments_vjp', None)
    assert callable(response), 'weak AO moments need a direct native reverse contraction'
    mf = scf.RHF(_molecule())
    raw, pullback = response(mf, (0, 1, 2), 3)
    mf_bar, = pullback(jax.tree_util.tree_map(jnp.zeros_like, raw))
    np.testing.assert_array_equal(mf_bar.mol.coords, np.zeros((3, 3)))


@pytest.mark.parametrize('cart', [False, True])
@pytest.mark.parametrize('atoms', [(0, 1, 2), (2, 0)])
@pytest.mark.parametrize('order', [1, 2, 3])
def test_nonsymmetric_weights_match_both_ao_legs_and_scatter_to_parent(cart, atoms, order):
    mol = _molecule(cart)
    mf = scf.RHF(mol)
    # A common origin from another calculation must not leak into the reverse rule.
    mol.set_common_origin((1.7, -.9, 2.3))
    raw, pullback = _pairs._origin_zero_moments_vjp(mf, atoms, order)
    seeds = _seeds(raw)
    expected = _dense_coordinate_reference(mol, atoms, seeds)
    actual, = pullback(seeds)
    np.testing.assert_allclose(actual.mol.coords, expected, rtol=2e-12, atol=1e-10)
    if atoms == (2, 0):
        np.testing.assert_array_equal(actual.mol.coords[1], np.zeros(3))


def test_native_derivatives_are_deferred_and_never_materialized_through_mole_intor(monkeypatch):
    mf = scf.RHF(_molecule())
    original_intor = gto.Mole.intor
    original_native = _moleintor_vjp._int1e_r0_deriv
    native_calls = []

    def forbid_full_derivatives(self, name, *args, **kwargs):
        assert '_dr' not in name and not name.startswith('int1e_ip'), (
            'coordinate response materialized a full AO derivative array: ' + name)
        return original_intor(self, name, *args, **kwargs)

    def count_native(name, shls_slice, comp, hermi, *args, **kwargs):
        native_calls.append((name, comp, hermi))
        return original_native(name, shls_slice, comp, hermi, *args, **kwargs)

    monkeypatch.setattr(gto.Mole, 'intor', forbid_full_derivatives)
    monkeypatch.setattr(_moleintor_vjp, '_int1e_r0_deriv', count_native)
    raw, pullback = _pairs._origin_zero_moments_vjp(mf, (0, 1, 2), 3)
    assert native_calls == [], 'derivative integrals must wait until the weights are available'
    actual, = pullback(_seeds(raw))
    assert np.linalg.norm(actual.mol.coords) > 1.
    assert sorted((comp, hermi) for _, comp, hermi in native_calls) == [
        (1, 1), (3, 1), (9, 1), (27, 1)]
    assert all('_dr01' not in name for name, _, _ in native_calls)


def test_native_moment_coordinate_gradient_matches_finite_difference():
    mol = _molecule()
    atoms = (2, 0)
    raw, pullback = _pairs._origin_zero_moments_vjp(scf.RHF(mol), atoms, 3)
    seeds = _seeds(raw)
    gradient, = pullback(seeds)
    direction = np.array([[.2, -.3, .1], [.4, .6, -.7], [-.1, .2, .5]])

    def value(displacement):
        changed = mol.set_geom_(np.asarray(mol.coords) + displacement * direction,
                                unit='Bohr', inplace=False)
        local = _pairs._fake_mol_with_traced_centers(changed, atoms)
        moments = _pairs._origin_zero_moments(local, 3)
        return sum(float(np.vdot(w, m)) for w, m in zip(seeds, moments))

    step = 2e-4
    finite_difference = (value(-2*step) - 8*value(-step)
                         + 8*value(step) - value(2*step)) / (12*step)
    analytic = float(np.vdot(gradient.mol.coords, direction))
    np.testing.assert_allclose(analytic, finite_difference, rtol=2e-9, atol=2e-8)


def test_existing_forward_jvp_agrees_with_native_reverse():
    mol = _molecule()
    raw, pullback = _pairs._origin_zero_moments_vjp(scf.RHF(mol), (0, 1, 2), 3)
    seeds = _seeds(raw)
    tangent_mol = jax.tree_util.tree_map(jnp.zeros_like, mol)
    tangent_mol.coords = jnp.array([[.2, -.3, .1], [.4, .6, -.7], [-.1, .2, .5]])
    _, tangent = jax.jvp(lambda value: _pairs._origin_zero_moments(value, 3),
                         (mol,), (tangent_mol,))
    reverse, = pullback(seeds)
    expected = sum(float(jnp.vdot(w, t)) for w, t in zip(seeds, tangent))
    np.testing.assert_allclose(np.vdot(reverse.mol.coords, tangent_mol.coords),
                               expected, rtol=2e-12, atol=1e-10)


def test_traced_pullback_weights_preserve_linear_response():
    mf = scf.RHF(_molecule())
    raw, pullback = _pairs._origin_zero_moments_vjp(mf, (2, 0), 3)
    seeds, tangent = _seeds(raw), _seeds(raw, seed=71)
    actual, actual_tangent = jax.jvp(lambda weights: pullback(weights)[0].mol.coords,
                                    (seeds,), (tangent,))
    expected, = pullback(seeds)
    expected_tangent, = pullback(tangent)
    np.testing.assert_allclose(actual, expected.mol.coords, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(actual_tangent, expected_tangent.mol.coords,
                               rtol=2e-12, atol=1e-10)


def test_large_contracted_shell_keeps_response_within_native_scratch(monkeypatch):
    # 43 contracted s AOs in one shell exceed the native octupole output
    # scratch: 3 * 27 * 43**2 > 144000 doubles. Never enter that C routine.
    coefficients = np.eye(43)
    basis = {'He': [[0, *[[float(i + 1), *row]
                         for i, row in enumerate(coefficients)]]]}
    mol = gto.Mole(atom='He 0.1 0.2 0.3', basis=basis, verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol)
    assert not _pairs._native_weak_moments_supported(mf)

    def forbid_native(*args, **kwargs):
        raise AssertionError('shell output exceeds the native derivative scratch buffer')

    monkeypatch.setattr(_moleintor_vjp, '_int1e_r0_deriv', forbid_native)
    raw, pullback = _pairs._origin_zero_moments_vjp(mf, (0,), 3)
    actual, = pullback(_seeds(raw))
    assert np.isfinite(actual.mol.coords).all()


@pytest.mark.parametrize('traced', ['exp', 'ctr_coeff'])
def test_traced_basis_parameters_keep_the_existing_reverse_rule(traced, monkeypatch):
    mol = _molecule(basis='sto-3g', trace_exp=traced == 'exp',
                    trace_ctr_coeff=traced == 'ctr_coeff')
    mf = scf.RHF(mol)
    assert not _pairs._native_weak_moments_supported(mf)

    def forbid_native(*args, **kwargs):
        raise AssertionError('coordinate-only native rule used for active basis parameters')

    monkeypatch.setattr(_moleintor_vjp, '_int1e_r0_deriv', forbid_native)
    # Use every atom: the existing submolecule adapter does not slice basis leaves.
    evaluate = lambda current: _pairs._origin_zero_moments(
        _pairs._fake_mol_with_traced_centers(current.mol, (0, 1, 2)), 1)
    expected_raw, expected_pullback = jax.vjp(evaluate, mf)
    raw, pullback = _pairs._origin_zero_moments_vjp(mf, (0, 1, 2), 1)
    seeds = _seeds(raw)
    actual, = pullback(seeds)
    expected, = expected_pullback(seeds)
    for got, want in zip(jax.tree_util.tree_leaves(raw), jax.tree_util.tree_leaves(expected_raw)):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=1e-10)
    assert np.linalg.norm(getattr(actual.mol, traced)) > 1e-3


def test_nongrouped_endpoint_replay_preserves_molecule_and_orbital_response(monkeypatch):
    mf = scf.RHF(_molecule())
    rng = np.random.default_rng(81)
    screen = _pairs.WeakScreen(
        jnp.array([-.5]),
        jnp.asarray(rng.normal(size=(mf.mol.nao, 1)) * .2),
        jnp.array([.4, .7, .9]),
        jnp.asarray(rng.normal(size=(mf.mol.nao, 3)) * .1),
    )
    static = SimpleNamespace(
        fragments=(SimpleNamespace(primary_atoms=np.array([0, 1, 2])),),
        thresholds=SimpleNamespace(multipole_order=4),
    )
    reference, reference_pullback = jax.vjp(
        lambda m, s: _pairs._weak_endpoint_from_screen(m, s, static, 0), mf, screen)
    seed = _seeds(reference)
    expected_mf, expected_screen = reference_pullback(seed)
    original_intor = gto.Mole.intor

    def forbid_full_derivatives(self, name, *args, **kwargs):
        assert '_dr' not in name and not name.startswith('int1e_ip'), (
            'nongrouped endpoint replay retained a full derivative array: ' + name)
        return original_intor(self, name, *args, **kwargs)

    monkeypatch.setattr(gto.Mole, 'intor', forbid_full_derivatives)
    actual, pullback = _pairs._replay_weak_endpoint(
        mf, (screen, lambda bar: (bar,)), static, 0)
    actual_mf, actual_screen = pullback(seed)
    for got, want in zip(actual, reference):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=1e-10)
    for got, want in zip(jax.tree_util.tree_leaves((actual_mf, actual_screen)),
                         jax.tree_util.tree_leaves((expected_mf, expected_screen))):
        np.testing.assert_allclose(got, want, rtol=2e-12, atol=1e-10)
    assert np.linalg.norm(actual_screen.occupied_coeff) > 1e-6
    assert np.linalg.norm(actual_screen.virtual_coeff) > 1e-6
