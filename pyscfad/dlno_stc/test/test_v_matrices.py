"""Factorized direct V against explicit four-index Coulomb oracles."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import backend, ccsd, _restart
from pyscfad.dlno_stc.controls import normalize_matrix_controls
from pyscfad.dlno_stc.lis import LISSelections
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


def explicit(b, targets, xp=np):
    g = xp.einsum('Pia,Pjb->ijab', b[:, targets, :], b)
    return (xp.einsum('ijab,ikab->jk', g, g),
            .5 * (xp.einsum('ijab,ijcb->ac', g, g)
                  + xp.einsum('ijba,ijbc->ac', g, g)))


@pytest.mark.parametrize('batch', [None, 1, 2])
@pytest.mark.parametrize('targets', [(1,), (2, 0), (0, 1, 2)])
def test_factorized_v_and_response_match_explicit_coulomb(targets, batch):
    rng = np.random.default_rng(63)
    b = rng.normal(size=(7, 3, 4)) * .2
    inputs = dict(B=b, foo=-np.eye(3), fvv=np.eye(4))
    result = backend.solve_matrices(inputs, dict(target_indices=targets),
                                   dict(target_batch_size=batch), matrix_kind='V')
    assert 'energy' not in result
    assert 'laplace_roots' not in result['controls']
    expected = explicit(b, list(targets))
    for name, reference in zip(('occupied', 'virtual'), expected):
        np.testing.assert_allclose(result[name], reference, atol=2e-15)
        assert np.linalg.eigvalsh(result[name]).min() >= -1e-14
    np.testing.assert_allclose(np.trace(result['occupied']), np.trace(result['virtual']))
    so, sv = rng.normal(size=(3, 3)), rng.normal(size=(4, 4))
    bars = backend.pullback_matrices(inputs, result, energy_bar=0.,
                                    occupied_bar=so, virtual_bar=sv)

    def loss(factors):
        vo, vv = explicit(factors, jnp.array(targets), jnp)
        return jnp.sum(vo * so) + jnp.sum(vv * sv)

    np.testing.assert_allclose(bars['B'], jax.grad(loss)(b), atol=3e-14)
    assert not np.any(bars['foo']) and not np.any(bars['fvv'])
    direction = rng.normal(size=b.shape)
    h = 1e-5
    fd = (loss(b + h * direction) - loss(b - h * direction)) / (2 * h)
    np.testing.assert_allclose(np.sum(bars['B'] * direction), fd, atol=2e-8)


def test_v_controls_replay_and_empty_virtual_space():
    with pytest.raises(NotImplementedError, match='deterministic'):
        normalize_matrix_controls({'mode': 'stochastic'}, 'V')
    with pytest.raises(ValueError, match='Laplace'):
        normalize_matrix_controls({'laplace_roots': [1.], 'laplace_weights': [1.]}, 'V')
    inputs = dict(B=np.ones((2, 2, 1)), foo=-np.eye(2), fvv=np.eye(1))
    result = backend.solve_matrices(inputs, dict(target_indices=(0,)), {}, matrix_kind='V')
    seeds = dict(occupied_bar=np.eye(2), virtual_bar=np.eye(1))
    with pytest.raises(ValueError, match='energy_bar=0'):
        backend.pullback_matrices(inputs, result, energy_bar=1., **seeds)
    with pytest.raises(ValueError, match='fingerprint'):
        backend.pullback_matrices(dict(inputs, B=inputs['B'] * 2), result, energy_bar=0., **seeds)
    inputs = dict(B=np.zeros((2, 2, 0)), foo=-np.eye(2), fvv=np.zeros((0, 0)))
    result = backend.solve_matrices(inputs, dict(target_indices=(0,)), {}, matrix_kind='V')
    assert not np.any(result['occupied'])
    bars = backend.pullback_matrices(inputs, result, energy_bar=0.,
                                    occupied_bar=np.eye(2), virtual_bar=np.zeros((0, 0)))
    assert bars['B'].shape == (2, 2, 0)


def test_baseline_validation_and_checkpoint_identity():
    with pytest.raises(ValueError, match='requires baseline_kind'):
        ccsd.build_static_selections(None, None, thresh_occ=0., thresh_vir=0., matrix_kind='V')
    ccsd._validate_baseline('global', 'V', {'mode': 'stochastic'})
    with pytest.raises(ValueError, match='baseline_kind'):
        ccsd.build_static_selections(None, None, thresh_occ=0., thresh_vir=0., baseline_kind='typo')
    old = ccsd.CCSDSelections(LISSelections(None, 0., 0., 1e-6, ()), ((), (), ()))
    encoded = _restart._encode_value(old)
    assert 'baseline_kind' not in dict(encoded['fields'])
    assert _restart._decode_value(encoded, None).baseline_kind == 'domain'
    new = replace(old, baseline_kind='global', lis_static=replace(old.lis_static, matrix_kind='V'))
    restored = _restart._decode_value(_restart._encode_value(new), None)
    assert restored.baseline_kind == 'global' and restored.lis_static.matrix_kind == 'V'
    assert _restart.scientific_digest(new) != _restart.scientific_digest(old)
    from pyscfad.dlno_stc._ccsd_restart_reference import _selection_check
    with pytest.raises(_restart.RestartMismatchError, match='baseline_kind'):
        _selection_check(replace(old, baseline_kind='global'), old)


@pytest.mark.parametrize('kind', ['D', 'W', 'V'])
def test_global_baseline_local_groups_only_reverse_lis_corrections(group_case, kind):
    case = group_case
    options = dict(baseline_kind='global', matrix_kind=kind, matrix_controls={})
    state, _ = case.run(**options)
    assert state.get('baseline_energy', 0.) == 0.
    assert case.replay_energy_bars == [0., 0.]
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17), ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(**options, with_grad=False, mf=.2 + dm, common=-.1 + dc)
        minus, _ = case.run(**options, with_grad=False, mf=.2 - dm, common=-.1 - dc)
        np.testing.assert_allclose(state[name] - initial,
            (plus['energy'] - minus['energy']) / (2 * step), atol=2e-9, rtol=2e-7)


def test_orbital_w_reverse_can_omit_domain_energy():
    from pyscfad.dlno_stc.test.test_backend import fixture
    inputs = fixture()
    batch = backend.solve_orbital_matrices(inputs, ((0, (0,)), (1, (1,))), {})
    seeds = [dict(occupied=np.eye(2), virtual=np.eye(3)) for _ in batch.forwards]
    actual = backend.pullback_orbital_matrices(inputs, batch, seeds, energy_bar=0.)
    forwards = [backend.solve_matrices(inputs, dict(fragment_id=i, target_indices=(i,)), {})
                for i in range(2)]
    packets = [backend.pullback_matrices(inputs, forward, energy_bar=0.,
                occupied_bar=seed['occupied'], virtual_bar=seed['virtual'])
               for forward, seed in zip(forwards, seeds)]
    for name in inputs:
        np.testing.assert_allclose(actual[name], sum(p[name] for p in packets), atol=2e-13)
