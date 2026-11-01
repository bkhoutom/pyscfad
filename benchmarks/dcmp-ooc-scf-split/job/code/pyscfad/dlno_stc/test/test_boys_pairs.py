"""Boys weak pairs keep native moments, response, and unordered bookkeeping."""
from importlib import import_module, util
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto


def _pairs():
    assert util.find_spec('pyscfad.dlno_stc._pairs') is not None, 'STC Boys pair utilities are missing'
    return import_module('pyscfad.dlno_stc._pairs')


@pytest.fixture(scope='module')
def screens():
    mol = gto.Mole(atom='H 0 0 0; H 0.1 0 1.3; H 9 0.3 0.2; H 9.2 0.4 1.6',
                   unit='Bohr', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    spaces = []
    for atoms in ([0, 1], [2, 3]):
        idx = np.asarray(atoms)
        s = overlap[np.ix_(idx, idx)]
        occ = np.asarray([1., 1.])
        occ /= np.sqrt(occ @ s @ occ)
        vir = np.asarray([1., -1.])
        vir -= occ * (occ @ s @ vir)
        vir /= np.sqrt(vir @ s @ vir)
        spaces.append((jnp.asarray(occ[:, None]), jnp.asarray(vir[:, None])))
    return mol, spaces


@pytest.mark.parametrize('order', [2, 3, 4])
def test_weak_energy_and_cartesian_response_match_native(screens, order):
    pytest.importorskip('pyscfad.dlno.mp2')
    pairs = _pairs()
    from pyscfad.dlno.mp2 import weak_screen_pair_energy as native_energy
    from pyscfad.dlno.dlno_base import WeakScreen as NativeScreen
    mol, spaces = screens
    left = pairs.WeakScreen(jnp.asarray([-.55]), spaces[0][0], jnp.asarray([.3]), spaces[0][1])
    right = pairs.WeakScreen(jnp.asarray([-.49]), spaces[1][0], jnp.asarray([.4]), spaces[1][1])
    static = SimpleNamespace(fragments=(SimpleNamespace(primary_atoms=np.array([0, 1])),
                                       SimpleNamespace(primary_atoms=np.array([2, 3]))),
                             strong_mask=np.eye(2, dtype=bool),
                             thresholds=SimpleNamespace(multipole_order=order))
    native_left = NativeScreen(jnp.ones(1), *left)
    native_right = NativeScreen(jnp.ones(1), *right)
    def evaluate(molecule):
        return pairs.weak_screen_pair_energy(SimpleNamespace(mol=molecule), left, right, static, 0, 1)
    def reference(molecule):
        return native_energy(SimpleNamespace(mol=molecule), native_left, native_right, static, 0, 1)
    energy, grad = jax.value_and_grad(evaluate)(mol)
    expected, expected_grad = jax.value_and_grad(reference)(mol)
    assert energy < 0
    np.testing.assert_allclose(energy, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(grad.coords, expected_grad.coords, atol=2e-7, rtol=2e-5)
    np.testing.assert_allclose(pairs.weak_screen_pair_energy(SimpleNamespace(mol=mol), right, left,
                                                           static, 1, 0), energy, atol=1e-12, rtol=0)


def test_unordered_weak_pairs_are_enumerated_once():
    pairs = _pairs()
    static = SimpleNamespace(fragments=(None,) * 4,
                             strong_mask=np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                                   [1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool))
    assert pairs.weak_pair_indices(static) == ((0, 1), (0, 3), (1, 2), (2, 3))
    with pytest.raises(ValueError, match='distinct'):
        pairs._validate_weak_pair(static, 0, 0)
    with pytest.raises(ValueError, match='strong'):
        pairs._validate_weak_pair(static, 0, 2)
    with pytest.raises(IndexError):
        pairs._validate_weak_pair(static, -1, 3)


def test_pair_prefactor_and_order_three_sign():
    pairs = _pairs()
    # Separation x=2, dipoles along x: dipole term -2/8, mixed term
    # +(5-2)/16, denominator 2, and unordered pair prefactor -4.
    left = (jnp.zeros(3), jnp.array([[1.], [0.], [0.]]), jnp.ones(1),
            jnp.zeros((3, 3, 1)), None)
    theta = jnp.zeros((3, 3, 1)).at[0, 0, 0].set(1.)
    right = (jnp.array([2., 0., 0.]), jnp.array([[1.], [0.], [0.]]),
             jnp.ones(1), theta, None)
    np.testing.assert_allclose(pairs._multipole_pair_energy(left, right, 2), -.125, atol=1e-14)
    # The transverse contraction subtracts 2/16 as well: net third-order +3/16.
    np.testing.assert_allclose(pairs._multipole_pair_energy(left, right, 3), -.0078125, atol=1e-14)


def test_rebuilt_screen_normalizes_current_boys_target():
    pairs = _pairs()
    selection = SimpleNamespace(parent_columns=np.array([0]), support_ao_indices=np.array([0, 1]),
                                canonical_keep=np.array([0]), overlap_keep=np.array([0]),
                                completeness_keep=np.array([0]), metric_keep=np.array([0]))
    fragment = SimpleNamespace(fragment_index=0, weak_virtual=selection, primary_ao_indices=np.array([0, 1]),
                               weak_occ_norm_keep=np.array([0]), weak_occ_span_metric_keep=np.array([0]))
    static = SimpleNamespace(fragments=(fragment,), thresholds=SimpleNamespace(metric_rank=1e-10))
    def rebuild(scale):
        common = SimpleNamespace(s1e=jnp.diag(jnp.array([2., 3.])), fock=jnp.diag(jnp.array([-1., .9])),
                                 pao_coeff=jnp.array([[0.], [1.]]),
                                 fragment_occupied_data=(SimpleNamespace(iao_coeff=jnp.array([[scale], [0.]])),))
        return pairs.build_weak_multipole_screen(common, static, 0)
    screen = rebuild(2.)
    assert not hasattr(screen, 'weights')
    np.testing.assert_allclose(screen.occupied_coeff, np.array([[1 / np.sqrt(2)], [0.]]), atol=1e-12)
    np.testing.assert_allclose(screen.occupied_energy, [-.5], atol=1e-12)
    np.testing.assert_allclose(screen.virtual_energy, [.3], atol=1e-12)
    for left, right in zip(screen, rebuild(.25)):
        np.testing.assert_allclose(left, right, atol=1e-12)
    with pytest.raises(RuntimeError, match='rank'):
        rebuild(0.)


def test_all_strong_classification_bypasses_screen_rebuilding():
    pairs = _pairs()
    static = SimpleNamespace(fragments=(None, None, None), thresholds=SimpleNamespace(pair_energy=1e-5))
    pair, weak, forced, strong = pairs.screen_pairs(None, None, static, model='all_strong')
    np.testing.assert_array_equal(pair, np.zeros((3, 3)))
    np.testing.assert_array_equal(weak, pair)
    assert np.all(forced) and np.all(strong)
    with pytest.raises(ValueError, match='unsupported'):
        pairs.screen_pairs(None, None, static, model='unknown')


@pytest.fixture(scope='module')
def water_reference():
    pytest.importorskip('pyscfad.dlno.domain')
    from dataclasses import replace
    from pyscfad.dlno_stc.test._boys_acceptance import molecule, build_mf, selection, LO_OPTIONS
    from pyscfad.dlno.domain import DLNOThresholds, build_domain_topology
    from pyscfad.dlno._selection import build_domain_selections
    mf = build_mf(molecule())
    static = selection(mf)
    topology = build_domain_topology(mf, frozen=2, lo_type='boys', lo_kwargs=LO_OPTIONS,
                                     frag_lolist=static.frag_lolist,
                                     thresholds=DLNOThresholds(pair_energy=1e-4))
    native = build_domain_selections(mf, topology)
    # Compare continuous response from identical saved localization references.
    native = replace(native, target_reference_coeff=static.target_reference_coeff,
                     target_reference_coords=static.target_reference_coords)
    return mf, static, native


@pytest.mark.parametrize('order', [2, 3, 4])
def test_full_weak_cartesian_response_matches_native(water_reference, order):
    from dataclasses import replace
    from pyscfad import config_update
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf, record_result
    from pyscfad.dlno_stc import domain
    from pyscfad.dlno_stc.adjoint import add_cotangent, zero_term_cotangents
    from pyscfad.dlno import dlno_base
    from pyscfad.dlno.mp2 import _weak_pair_energy as native_energy
    pairs = _pairs()
    reference, static, native = water_reference
    static = replace(static, thresholds=replace(static.thresholds, multipole_order=order))
    native = replace(native, thresholds=replace(native.thresholds, multipole_order=order))
    np.testing.assert_array_equal(static.strong_mask, native.strong_mask)
    indices = pairs.weak_pair_indices(static)
    assert len(indices) > 0
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        mf, scf_pullback = jax.vjp(build_mf, reference.mol)
    common, common_pullback = jax.vjp(lambda x: domain.rebuild_domain_data(x, static), mf)
    value, (mf_bar, common_bar) = pairs.weak_pair_energy_and_grad(mf, common, static)
    (common_mf_bar,) = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, common_mf_bar)
    (gradient,) = scf_pullback(mf_bar)
    native_common, native_common_pullback = jax.vjp(lambda x: dlno_base.rebuild_domain_data(x, native), mf)
    native_mf_bar, native_common_bar = zero_term_cotangents(mf, native_common)
    expected = 0.
    for left, right in indices:
        energy, pullback = jax.vjp(lambda m, c: native_energy(m, c, native, left, right), mf, native_common)
        left_bar, right_bar = pullback(jnp.ones_like(energy))
        native_mf_bar = jax.tree_util.tree_map(add_cotangent, native_mf_bar, left_bar)
        native_common_bar = jax.tree_util.tree_map(add_cotangent, native_common_bar, right_bar)
        expected += energy
        del pullback
    (common_mf_bar,) = native_common_pullback(native_common_bar)
    native_mf_bar = jax.tree_util.tree_map(add_cotangent, native_mf_bar, common_mf_bar)
    (expected_gradient,) = scf_pullback(native_mf_bar)
    energy_error = abs(float(value - expected))
    gradient_error = float(np.max(np.abs(np.asarray(gradient.coords - expected_gradient.coords))))
    record_result(f'weak_pair_order_{order}', {'weak_pairs': len(indices), 'energy': float(value),
                  'native_energy': float(expected), 'energy_abs_error': energy_error,
                  'gradient_max_abs_error': gradient_error,
                  'gradient_max_abs': float(np.max(np.abs(np.asarray(gradient.coords))))})
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(gradient.coords, expected_gradient.coords, atol=2e-7, rtol=2e-5)


def test_exact_classification_matches_native_two_sided_diagnostic(water_reference):
    from pyscfad.mp import dfmp2
    from pyscfad.dlno.domain import _exact_global_fragment_pairs
    from pyscfad.dlno_stc.domain import rebuild_domain_data
    from pyscfad.dlno_stc.test._boys_acceptance import record_result
    pairs = _pairs()
    mf, static, _ = water_reference
    common = rebuild_domain_data(mf, static)
    rotation = np.asarray(common.occupied_coeff).T @ np.asarray(common.s1e) @ np.asarray(common.iao_coeff)
    # Native diagnostic is an independent reference; weights exist only here.
    occupied_data = tuple(SimpleNamespace(occupied_weight=np.outer(rotation[:, int(indices[0])],
                                                                   rotation[:, int(indices[0])]))
                          for indices in static.frag_lolist)
    _, expected = _exact_global_fragment_pairs(dfmp2.MP2(mf, frozen=static.frozen), occupied_data)
    actual = pairs._exact_boys_pair_scores(mf, common, static)
    record_result('exact_boys_pair_scores', {'max_abs_error': float(np.max(np.abs(actual - expected)))})
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    classified, weak, forced, strong = pairs.screen_pairs(mf, common, static, model='exact')
    np.testing.assert_allclose(classified, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(strong, (np.abs(expected) > static.thresholds.pair_energy) | forced)
    for alias in ('multipole_os', 'os_multipole'):
        score, estimate, _, mask = pairs.screen_pairs(mf, common, static, model=alias)
        np.testing.assert_allclose(score, weak, atol=1e-12, rtol=0)
        np.testing.assert_allclose(estimate, weak, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(mask, static.strong_mask)
