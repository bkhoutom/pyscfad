"""Selected input-frame contraction, independent reverse, and boundary checks."""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture, grid, KEYS


def solve(x, targets=(0,), controls=None, **kwargs):
    return native().solve_selected(*(x[k] for k in KEYS), targets,
                                   grid() if controls is None else controls, **kwargs)


def reference(x, controls, targets):
    foo, fvv, B = (jnp.asarray(x[k]) for k in KEYS)
    bare = jnp.einsum('Pia,Pjb->ijab', B, B)
    value = 0.
    for beta, weight in zip(controls['laplace_roots'], controls['laplace_weights']):
        dressed = jnp.einsum('ij,Pjb,ba->Pia', jsp_linalg.expm(beta*foo), B,
                             jsp_linalg.expm(-beta*fvv))
        pair = jnp.einsum('Pia,Pjb->ijab', dressed, dressed)
        value += weight*jnp.sum((bare*(pair.transpose(0, 1, 3, 2)-2*pair))[jnp.array(targets)])
    return value


def test_selected_binding_exists():
    assert hasattr(native(), 'solve_selected')


@pytest.mark.parametrize('repeated', [False, True])
@pytest.mark.parametrize('targets', [(0,), (1,), (0, 1)])
def test_selected_formula_input_frame_and_three_bars(repeated, targets):
    x, controls = full_fixture(repeated), grid()
    saved = {k: v.copy() for k, v in x.items()}
    out = solve(x, targets, controls)
    value, bars = jax.value_and_grad(reference)(x, controls, targets)
    np.testing.assert_allclose(out['energy'], value, rtol=2e-11, atol=2e-13)
    assert set(out['cotangents']) == set(KEYS)
    assert out['energy_standard_error'] == 0
    assert out['diagnostics']['energy_identity'] == 'boys_target_laplace_mp2_v1'
    assert tuple(out['diagnostics']['target_indices']) == targets
    for key in KEYS:
        expected = np.asarray(bars[key])
        if key != 'B':
            expected = (expected+expected.T)/2
        np.testing.assert_allclose(out['cotangents'][key], expected, rtol=2e-10, atol=2e-12)
        np.testing.assert_array_equal(x[key], saved[key])
    assert 'cotangents' not in solve(x, targets, controls, with_grad=False)


def test_group_additivity_all_targets_system_and_directional_reverse():
    x = full_fixture()
    group = solve(x, (0, 1))
    rows = [solve(x, (i,)) for i in range(2)]
    full = native().solve_full(*(x[k] for k in KEYS), grid(), with_grad=True)
    np.testing.assert_allclose(group['energy'], full['energy'], atol=2e-13)
    np.testing.assert_allclose(group['energy'], sum(row['energy'] for row in rows), atol=2e-13)
    rng = np.random.default_rng(613)
    for key in KEYS:
        np.testing.assert_allclose(group['cotangents'][key], full['cotangents'][key], atol=2e-12)
        np.testing.assert_allclose(group['cotangents'][key], sum(row['cotangents'][key] for row in rows), atol=2e-12)
        direction = rng.normal(size=x[key].shape)
        if key != 'B':
            direction = (direction+direction.T)/2
        plus, minus = dict(x), dict(x)
        plus[key], minus[key] = x[key]+1e-5*direction, x[key]-1e-5*direction
        finite = (solve(plus, (1,))['energy']-solve(minus, (1,))['energy'])/2e-5
        np.testing.assert_allclose(np.sum(rows[1]['cotangents'][key]*direction), finite, rtol=2e-7, atol=2e-10)


@pytest.mark.parametrize('targets', [(), (0, 0), (-1,), (2,), (True,), (np.bool_(True),), (10**100,), (1.,), [0], None])
def test_invalid_target_labels(targets):
    with pytest.raises((ValueError, TypeError)):
        solve(full_fixture(), targets)


def test_missing_targets_and_malformed_buffers():
    x = full_fixture()
    with pytest.raises(TypeError):
        native().solve_selected(*(x[k] for k in KEYS), controls=grid())
    for key, value in [('B', np.asfortranarray(x['B'])), ('foo', x['foo'].astype('f4')),
                       ('fvv', x['fvv'][:2, :2].copy()), ('B', x['B'][0].copy()),
                       ('foo', np.array([[0., 1.], [0., 0.]]))]:
        with pytest.raises((ValueError, TypeError)):
            solve(dict(x, **{key: value}))


@pytest.mark.parametrize('empty', ['virtual', 'auxiliary'])
def test_selected_empty_spaces(empty):
    x = full_fixture()
    if empty == 'virtual':
        x['fvv'], x['B'] = np.empty((0, 0)), np.empty((4, 2, 0))
    else:
        x['B'] = np.empty((0, 2, 3))
    out = solve(x)
    assert out['energy'] == out['energy_standard_error'] == 0
    for key in KEYS:
        np.testing.assert_array_equal(out['cotangents'][key], np.zeros_like(x[key]))


def test_selected_source_proposals_seeded_reverse_and_all_kept():
    x = full_fixture()
    controls = dict(grid(), mode='stochastic', system_workload_cutoff=.2,
                    production_samples=24000, pilot_samples=32,
                    min_production_samples=2, global_seed=17)
    exact = solve(x, (1,), grid())
    out = solve(x, (1,), controls, aux_offsets=np.array([0, 1, 4]))
    again = solve(x, (1,), controls, aux_offsets=np.array([0, 1, 4]))
    assert out['energy'] == again['energy']
    assert abs(out['energy']-exact['energy']) < 7*out['energy_standard_error']+1e-13
    assert out['diagnostics']['actual_sample_count'] > 0
    for key in KEYS:
        np.testing.assert_array_equal(out['cotangents'][key], again['cotangents'][key])
    assert solve(x, (1,), controls, aux_offsets=np.array([0, 1, 4]), with_grad=False)['energy'] == out['energy']
    kept = solve(x, (1,), dict(controls, system_workload_cutoff=1e-100))
    assert kept['energy_standard_error'] == 0
    np.testing.assert_allclose(kept['energy'], exact['energy'], atol=2e-13)
    for key in KEYS:
        np.testing.assert_allclose(kept['cotangents'][key], exact['cotangents'][key], atol=2e-12)


def test_selected_scalar_coincident_roles_and_zero_proposal_support():
    x = dict(foo=-np.eye(1), fvv=np.eye(1), B=np.ones((1, 1, 1)))
    controls = dict(mode='stochastic', laplace_roots=[0.], laplace_weights=[1.],
                    system_workload_cutoff=2., production_samples=2,
                    pilot_samples=2, min_production_samples=2, global_seed=7)
    out = solve(x, (0,), controls)
    assert out['energy'] == -1
    np.testing.assert_allclose(out['cotangents']['B'], -4.)
    x['B'][:] = 0
    zero = solve(x, (0,), controls)
    assert zero['energy'] == zero['energy_standard_error'] == 0
    # Preserve source underflow: products of the absolute 1e-100 floors vanish.
    assert zero['diagnostics']['actual_sample_count'] == 0
    for key in KEYS:
        np.testing.assert_array_equal(zero['cotangents'][key], np.zeros_like(x[key]))
