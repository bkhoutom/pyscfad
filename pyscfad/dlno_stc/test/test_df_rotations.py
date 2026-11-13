"""Joint LIS rotations preserve physical tensors and their input responses."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import stc_ccsd as adapter, smooth_triples
from pyscfad.dlno_stc.test.test_shared_stc_frame import (
    _inputs, _physical, _eris, _rotations, _reference_energies,
    _assert_energy_and_response, GRID,
)
from pyscfad.lno.ccsd import ccsd_fragment_energy


def test_df_rotations_default_and_validation():
    controls = adapter.normalize_controls(None)
    assert controls['df_rotations'] is True
    assert controls['df_rotation_tol'] == 5e-3
    assert adapter.normalize_controls(controls) == controls
    assert adapter.normalize_controls({'df_rotations': False})['df_rotations'] is False
    for value in (None, 1, 'yes'):
        with pytest.raises(TypeError, match='df_rotations'):
            adapter.normalize_controls({'df_rotations': value})
    for value in (0., -1., float('inf'), float('nan'), True, None):
        with pytest.raises((TypeError, ValueError), match='df_rotation_tol'):
            adapter.normalize_controls({'df_rotation_tol': value})


def _backend():
    # Only initialization bookkeeping is replaced: transformations, Fock
    # diagonalization, projected energies and triples are production routines.
    return SimpleNamespace(_LocalDFERIs=SimpleNamespace,
        _eia=lambda energy, no: energy[:no, None] - energy[None, no:],
        _mp2_amplitudes=lambda lov, uo, uv, denom: jnp.zeros(
            (lov.shape[1], lov.shape[1], lov.shape[2], lov.shape[2])))


@pytest.mark.parametrize('integral_transform', ['df', 'rotate'])
def test_joint_rotations_preserve_projected_energy_and_all_physical_responses(integral_transform):
    x = _inputs()
    qo, qv = _rotations()
    na = x['Bvo'].shape[0]
    qa = jnp.asarray(np.linalg.qr(np.random.default_rng(83).normal(size=(na, na)))[0])
    triples = smooth_triples.make_solver(
        dict(mode='deterministic', integral_transform=integral_transform), GRID, with_grad=True)

    def evaluate(inputs):
        physical = _physical(inputs)
        eris = _eris(physical)
        eris.mo_coeff = jnp.eye(sum(inputs['t1'].shape))
        local = adapter._make_local_eris(SimpleNamespace(mol=None), eris, qo, qv,
                                         _backend(), qa=qa)
        a, b = adapter._rotate_amplitudes(physical['t1'], physical['t2'], qo, qv)
        view = adapter._SharedFrameERIs(local)
        projector = physical['prjlo'] @ qo
        cc = ccsd_fragment_energy(view, a, b, projector)
        et, _ = triples(None, view, projector, t1=a, t2=b)
        return jnp.stack((cc, et))

    _assert_energy_and_response(evaluate, lambda inputs: _reference_energies(inputs)[1:], x)


def test_sampling_frame_uses_compressed_factors_once_and_composes_rotations(monkeypatch):
    external = pytest.importorskip('stc_cc_jax.lo')
    qo, qv = _rotations()
    lov = jnp.asarray(np.random.default_rng(91).normal(size=(5, 3, 3)))
    rng = np.random.default_rng(781)
    uo, uv, ua = tuple(np.linalg.qr(rng.normal(size=(n, n)))[0] for n in (3, 3, 5))
    calls = []
    monkeypatch.setattr(adapter, '_working_rotations', lambda *args: (qo, qv))
    def optimize(factors, tol):
        calls.append((factors.copy(), tol))
        return uo, uv, ua
    monkeypatch.setattr(external, '_df_rotations', optimize)
    # No AO DF tensor exists here: the optimizer must consume the LIS factors.
    eris = SimpleNamespace(Lov=lov)
    frame = adapter.SamplingFrame()
    config = adapter.normalize_controls(None)
    def transformed(factors):
        ro, rv, ra = frame.rotations(None, SimpleNamespace(Lov=factors), config)
        return jnp.einsum('xY,xia,iI,aA->YIA', ra, factors, ro, rv)
    answer, pullback = jax.vjp(transformed, lov)
    got_bar, = pullback(jnp.ones_like(answer))
    transformed(lov)  # second replica in the same physical LIS
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], np.einsum('xia,iI,aA->IAx', lov, qo, qv))
    assert calls[0][1] == config['df_rotation_tol']
    ro, rv, ra = frame.rotations(None, eris, config)
    np.testing.assert_allclose(ro, qo @ uo)
    np.testing.assert_allclose(rv, qv @ uv)
    np.testing.assert_allclose(ra, ua)
    expected = jax.grad(lambda factors: jnp.einsum(
        'xY,xia,iI,aA->YIA', ua, factors, qo @ uo, qv @ uv).sum())(lov)
    np.testing.assert_allclose(got_bar, expected)
    # Another physical evaluation owns a fresh preparation, even at equal rank.
    adapter.SamplingFrame().rotations(None, SimpleNamespace(Lov=lov * 1.1), config)
    assert len(calls) == 2


def test_disabled_and_zero_factor_frames_avoid_df_optimizer(monkeypatch):
    external = pytest.importorskip('stc_cc_jax.lo')
    def forbidden(*args):
        pytest.fail('DF optimization must be skipped')
    monkeypatch.setattr(external, '_df_rotations', forbidden)
    eris = SimpleNamespace(Lov=jnp.zeros((5, 2, 3)))
    rotations = adapter.SamplingFrame().rotations(None, eris,
        adapter.normalize_controls({'localize': False}))
    for rotation, n in zip(rotations, (2, 3, 5)):
        np.testing.assert_array_equal(rotation, np.eye(n))
    monkeypatch.setattr(adapter, '_working_rotations', lambda *args: (jnp.eye(2), jnp.eye(3)))
    for overrides in ({}, {'df_rotations': False}):
        adapter.SamplingFrame().rotations(None, eris, adapter.normalize_controls(overrides))
