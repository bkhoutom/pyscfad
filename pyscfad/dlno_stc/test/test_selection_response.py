"""Selection receipts keep one live common frame and consume weak work once."""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import driver


class Common(NamedTuple):
    s1e: object


def test_receipt_reuses_exact_common_and_preserves_full_chain_rule(monkeypatch):
    from pyscfad.dlno_stc._selection_response import SelectionResponse
    mf = jnp.array([.4, .7])
    data, data_pb = jax.vjp(lambda x: (x*x,), mf)
    calls = []
    def build(current, static, *, scf_data):
        calls.append(static)
        return Common(current + 3*scf_data[0])
    monkeypatch.setattr(driver, 'rebuild_domain_data', build)
    receipt = SelectionResponse(mf, data, data_pb)
    provisional, final = object(), object()
    common = receipt.prepare(provisional)
    receipt.bind(final)
    adopted = receipt.take(final)
    assert adopted is not None and adopted[0] is common
    assert calls == [provisional]
    actual, = adopted[1](Common(jnp.ones(2)))
    np.testing.assert_allclose(actual, 1 + 6*mf, rtol=1e-14)
    assert receipt.take(final) is None


def test_receipt_rejects_replaced_static_and_releases_values(monkeypatch):
    from pyscfad.dlno_stc._selection_response import SelectionResponse
    monkeypatch.setattr(driver, 'rebuild_domain_data',
                        lambda m, s, *, scf_data: Common(m))
    mf = jnp.array([.4])
    data, pb = jax.vjp(lambda x: (x,), mf)
    receipt = SelectionResponse(mf, data, pb)
    receipt.prepare(object())
    receipt.bind(object())
    assert receipt.take(object()) is None
    assert receipt.common is None
    assert receipt.pending is None


@pytest.mark.parametrize('requested', [((0, 1),), ((0, 2),)])
def test_prepared_weak_is_consumed_once_and_subset_mismatch_falls_back(monkeypatch, requested):
    stored = ((0, 1),)
    cached, fallback = (jnp.array(-.3), ('direct', 'common')), (jnp.array(-.2), ('d2', 'c2'))
    state = {'mf': object(), 'common': object(), '_prepared_weak': (stored, cached)}
    calls = []
    def evaluate(*args, **kwargs):
        calls.append(kwargs['pairs'])
        return fallback
    monkeypatch.setattr(driver, 'weak_pair_energy_and_grad', evaluate)
    actual = driver._weak_energy_and_grad(state, object(), requested)
    assert actual is (cached if requested == stored else fallback)
    assert '_prepared_weak' not in state
    assert driver._weak_energy_and_grad(state, object(), requested) is fallback
    assert len(calls) == (1 if requested == stored else 2)


@jax.tree_util.register_pytree_node_class
class _SelectionMF:
    """Minimal mutable MF boundary, with real JAX detachment and pullbacks."""
    def __init__(self, x):
        self.x = x

    def tree_flatten(self):
        return (self.x,), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(children[0])


def test_selection_boundary_keeps_live_response_off_detached_reference(monkeypatch):
    monkeypatch.setattr(driver, 'build_scf_data', lambda mf: (2*mf.x,))
    monkeypatch.setattr(driver, 'rebuild_domain_data',
                        lambda mf, static, *, scf_data: Common(mf.x + scf_data[0]))
    mf = _SelectionMF(jnp.array([.2, .5]))
    selected = object()
    references = []

    def choose(reference):
        references.append(reference)
        assert reference is not mf
        receipt = reference._stc_selection_response
        assert receipt.mf is mf
        receipt.prepare(object())
        receipt.bind(selected)
        # Custom selection callbacks can discard/replace their own copy.
        reference.x = None
        return selected

    actual, _, _, receipt = driver._select_domain_reference(mf, choose)
    assert actual is selected
    common, pullback, pending = receipt.take(selected)
    np.testing.assert_allclose(common.s1e, 3*mf.x)
    bar, = pullback(Common(jnp.ones_like(mf.x)))
    np.testing.assert_allclose(bar.x, 3.)
    assert pending is None
    for reference in (mf, *references):
        assert not hasattr(reference, '_stc_scf_data')
        assert not hasattr(reference, '_stc_selection_response')


@pytest.mark.parametrize('reuse_weak', [False, True])
def test_selection_boundary_cleans_transient_metadata_on_error(monkeypatch, reuse_weak):
    monkeypatch.setattr(driver, 'build_scf_data', lambda mf: (2*mf.x,))
    mf = _SelectionMF(jnp.array([.2]))
    references = []

    def fail(reference):
        references.append(reference)
        assert (reference._stc_selection_response is not None) == reuse_weak
        raise ValueError('selection intentionally failed')

    with pytest.raises(ValueError, match='selection intentionally failed'):
        driver._select_domain_reference(mf, fail, reuse_weak=reuse_weak)
    for reference in (mf, *references):
        assert not hasattr(reference, '_stc_scf_data')
        assert not hasattr(reference, '_stc_selection_response')


@pytest.mark.parametrize('return_first', [False, True])
def test_multiple_selections_only_latest_exact_result_can_be_reused(monkeypatch, return_first):
    from pyscfad.dlno_stc._selection_response import SelectionResponse
    mf = jnp.array([.4])
    data, pb = jax.vjp(lambda x: (x*x,), mf)
    monkeypatch.setattr(driver, 'rebuild_domain_data',
                        lambda m, s, *, scf_data: Common(m*s.scale + scf_data[0]))
    receipt = SelectionResponse(mf, data, pb)
    first, second = SimpleNamespace(scale=2.), SimpleNamespace(scale=5.)
    receipt.prepare(first)
    receipt.pending = ('obsolete',)
    receipt.bind(first)
    latest = receipt.prepare(second)
    assert receipt.pending is None
    receipt.bind(second)
    adopted = receipt.take(first if return_first else second)
    if return_first:
        assert adopted is None
    else:
        assert adopted[0] is latest
        actual, = adopted[1](Common(jnp.ones_like(mf)))
        np.testing.assert_allclose(actual, 5 + 2*mf)
    assert receipt.common is None and receipt.pending is None


@pytest.mark.parametrize('changed', ['mo_coeff', 'mo_energy', 'mo_occ', 'coords', '_atm', '_bas', '_env'])
def test_changed_selection_reference_disables_reuse(changed):
    from copy import deepcopy
    from pyscfad.dlno_stc._selection_response import SelectionResponse

    class Molecule:
        def __init__(self):
            self.coords = np.array([[0., 0., 0.], [0., 0., 1.]])
            self._atm = np.array([[1, 0], [1, 3]])
            self._bas = np.array([[0, 1], [1, 1]])
            self._env = np.array([.2, .3])

        def atom_coords(self):
            return self.coords

    mf = SimpleNamespace(mo_coeff=np.eye(2), mo_energy=np.array([-.3, .2]),
                         mo_occ=np.array([2., 0.]), mol=Molecule())
    receipt = SelectionResponse(mf, None, None)
    reference = deepcopy(mf)
    assert receipt.matches_reference(reference)
    owner = reference if changed.startswith('mo_') else reference.mol
    getattr(owner, changed).flat[0] += 1
    assert not receipt.matches_reference(reference)


@pytest.mark.parametrize('include_baseline, resume, expected_reuse', [
    (True, False, True),
    (True, True, False),
    (False, False, False),
    (False, True, False),
])
def test_public_ccsd_selection_reuses_weak_only_for_fresh_baseline(
        monkeypatch, tmp_path, include_baseline, resume, expected_reuse):
    """A resumed checkpoint must decide whether weak work is needed first."""
    from pyscfad.dlno_stc import ccsd

    class SelectionReached(Exception):
        pass

    mol = SimpleNamespace(exp=None, ctr_coeff=None, r0=None)
    mf = _SelectionMF(jnp.array([.2]))
    calls = []

    def build_mf(current):
        raise AssertionError('this control-flow test must not run SCF')

    def initialize_scf(current, builder, *, combine_df):
        assert current is mol and builder is build_mf and combine_df
        return mf, lambda bar: (bar,)

    def select_reference(current, selector, *, reuse_weak):
        assert current is mf
        assert callable(selector)
        assert reuse_weak is expected_reuse
        calls.append(reuse_weak)
        raise SelectionReached

    def run(initialize, *args, **options):
        assert options['include_baseline'] is include_baseline
        assert options['resume'] is resume
        return initialize()

    monkeypatch.setattr(driver, '_initialize_scf', initialize_scf)
    monkeypatch.setattr(driver, '_select_domain_reference', select_reference)
    monkeypatch.setattr(ccsd, '_run', run)
    with pytest.raises(SelectionReached):
        ccsd.value_and_grad(
            mol, build_mf, build_static=lambda reference: None,
            baseline_controls={}, matrix_controls={}, include_baseline=include_baseline,
            resume=resume, checkpoint_dir=tmp_path / 'checkpoint')
    assert calls == [expected_reuse]


def test_ccsd_weak_completion_consumes_prepared_energy_and_all_bars(monkeypatch):
    from pyscfad.dlno_stc import ccsd
    pairs = ((0, 1),)
    state = dict(
        mf=object(), common=object(), weak_pairs=pairs,
        energy=jnp.array(2.), baseline_energy=-1., weak_energy=.05,
        mf_bar=jnp.array([1., 2.]), common_bar=Common(jnp.array([3., 4.])),
        _prepared_weak=(pairs, (jnp.array(-.5),
                               (jnp.array([.1, .2]), Common(jnp.array([.3, .4]))))))

    def forbidden(*args, **kwargs):
        raise AssertionError('completed selection-time weak response was recomputed')

    monkeypatch.setattr(driver, 'weak_pair_energy_and_grad', forbidden)
    ccsd._run_weak(state, SimpleNamespace(mp2_static=object()), with_grad=True)
    assert '_prepared_weak' not in state
    np.testing.assert_allclose(state['energy'], 1.5)
    np.testing.assert_allclose(state['baseline_energy'], -1.5)
    np.testing.assert_allclose(state['weak_energy'], -.45)
    np.testing.assert_allclose(state['mf_bar'], [1.1, 2.2])
    np.testing.assert_allclose(state['common_bar'].s1e, [3.3, 4.4])
