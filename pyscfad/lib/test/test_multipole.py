"""Numerical and derivative checks for the compiled multipole kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno.multipole_numpy import multipole_pair_energy


@pytest.mark.parametrize('rank', [2, 3])
def test_coordinate_moment_jvp_matches_atom_slice_response(rank):
    from pyscfad.lib import _multipole

    rng = np.random.default_rng(240 + rank)
    slices = ((0, 2), (2, 5), (5, 6))
    bra = rng.normal(size=(3, 3 ** rank, 6, 6))
    ket = rng.normal(size=bra.shape)
    coords_t = rng.normal(size=(3, 3))
    ao_owner = np.repeat(np.arange(3), [2, 3, 1])

    def reference(tangent):
        result = np.zeros((3 ** rank, 6, 6))
        for atom, (p0, p1) in enumerate(slices):
            result[:, p0:p1, :] += np.einsum(
                'x,xpuv->puv', tangent[atom], bra[:, :, p0:p1, :]
            )
            result[:, :, p0:p1] += np.einsum(
                'x,xpuv->puv', tangent[atom], ket[:, :, :, p0:p1]
            )
        return result

    actual = _multipole.coordinate_moment_jvp(bra, ket, coords_t, ao_owner)
    np.testing.assert_allclose(actual, reference(coords_t), rtol=1e-13, atol=1e-13)
    direction = rng.normal(size=coords_t.shape)
    _, derivative = jax.jvp(
        lambda value: _multipole.coordinate_moment_jvp(bra, ket, value, ao_owner),
        (jnp.asarray(coords_t),), (jnp.asarray(direction),),
    )
    np.testing.assert_allclose(
        derivative, reference(direction), rtol=1e-13, atol=1e-13
    )


def _moment_inputs(shared_virtual):
    rng = np.random.default_rng(302)
    nao, nmode, nvir = 4, 2, 3
    occupied = rng.normal(size=(nao, nmode))
    occupied /= np.linalg.norm(occupied, axis=0)
    virtual = rng.normal(size=(nao, nvir))
    if not shared_virtual:
        virtual = np.stack((virtual, rng.normal(size=(nao, nvir))))
    # A quadrature representation gives symmetric Cartesian AO moments.
    positions = rng.normal(size=(9, 3))
    ao = rng.normal(size=(9, nao))
    overlap = np.einsum('pu,pv->uv', ao, ao) / 9
    dipole = np.einsum('px,pu,pv->xuv', positions, ao, ao) / 9
    second = np.einsum('px,py,pu,pv->xyuv', positions, positions, ao, ao) / 9
    third = np.einsum(
        'px,py,pz,pu,pv->xyzuv', positions, positions, positions, ao, ao
    ) / 9
    return occupied, virtual, overlap, dipole, second, third


def _reference_transition(inputs, order):
    """Translate the AO operators before independent scalar contractions."""
    occupied, virtual, overlap, dipole, second, third = inputs
    records = []
    for mode, orbital in enumerate(occupied.T):
        space = virtual if virtual.ndim == 2 else virtual[mode]
        center = np.einsum('u,xuv,v->x', orbital.conj(), dipole, orbital)

        def transition(operator):
            return np.einsum('...uv,v,ua->...a', operator, orbital, space.conj())

        theta, omega = None, None
        if order > 2:
            translated = np.empty_like(second)
            for x in range(3):
                for y in range(3):
                    translated[x, y] = (
                        second[x, y] - center[x] * dipole[y]
                        - center[y] * dipole[x]
                        + center[x] * center[y] * overlap
                    )
            quadrupole = 3 * translated
            trace = np.trace(translated, axis1=0, axis2=1)
            for x in range(3):
                quadrupole[x, x] -= trace
            theta = transition(quadrupole / 2)
        if order > 3:
            translated = np.empty_like(third)
            for x in range(3):
                for y in range(3):
                    for z in range(3):
                        translated[x, y, z] = (
                            third[x, y, z] - center[x] * second[y, z]
                            - center[y] * second[x, z] - center[z] * second[x, y]
                            + center[x] * center[y] * dipole[z]
                            + center[x] * center[z] * dipole[y]
                            + center[y] * center[z] * dipole[x]
                            - center[x] * center[y] * center[z] * overlap
                        )
            octupole = 5 * translated
            for x in range(3):
                octupole[:, x, x] -= np.trace(translated, axis1=1, axis2=2)
                octupole[x, :, x] -= np.trace(translated, axis1=2, axis2=0)
                octupole[x, x, :] -= np.trace(translated, axis1=0, axis2=1)
            omega = transition(octupole / 2)
        records.append((center, transition(dipole), theta, omega))
    return tuple(
        None if records[0][index] is None else np.stack([r[index] for r in records])
        for index in range(4)
    )


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('shared_virtual', [True, False])
def test_transition_moments_match_ao_origin_translation(order, shared_virtual):
    from pyscfad.lib import _multipole

    inputs = _moment_inputs(shared_virtual)
    used_inputs = inputs[:4] + (
        inputs[4] if order > 2 else None,
        inputs[5] if order > 3 else None,
    )
    actual = _multipole.transition_moments(*used_inputs, order)
    expected = _reference_transition(inputs, order)
    for value, reference in zip(actual, expected):
        if reference is None:
            assert value is None
        else:
            np.testing.assert_allclose(value, reference, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('shared_virtual', [True, False])
def test_transition_moments_jvp_matches_independent_finite_difference(
        order, shared_virtual):
    from pyscfad.lib import _multipole

    inputs = _moment_inputs(shared_virtual)
    rng = np.random.default_rng(909)
    directions = tuple(rng.normal(size=value.shape) * 0.03 for value in inputs)
    _, tangent = jax.jvp(
        lambda *values: _multipole.transition_moments(*values, order),
        tuple(jnp.asarray(value) for value in inputs),
        tuple(jnp.asarray(value) for value in directions),
    )
    step = 1e-4
    shifted = {
        multiple: _reference_transition(
            tuple(value + multiple * step * direction
                  for value, direction in zip(inputs, directions)), order
        )
        for multiple in (-2, -1, 1, 2)
    }
    for index, value in enumerate(tangent):
        if value is None:
            continue
        reference = (
            shifted[-2][index] - 8 * shifted[-1][index]
            + 8 * shifted[1][index] - shifted[2][index]
        ) / (12 * step)
        np.testing.assert_allclose(value, reference, rtol=2e-7, atol=3e-10)


def _endpoint_records(order, widths=(2, 3)):
    rng = np.random.default_rng(811)
    records = []
    for nvir, center in zip(widths, ([0.2, -0.1, 0.4], [1.3, 2.7, 4.1])):
        records.append((
            np.asarray(center),
            rng.normal(size=(3, nvir)),
            rng.uniform(0.3, 1.2, size=nvir),
            rng.normal(size=(3, 3, nvir)) if order > 2 else None,
            rng.normal(size=(3, 3, 3, nvir)) if order > 3 else None,
        ))
    return tuple(records)


@pytest.mark.parametrize('order', [2, 3, 4])
def test_complex_endpoint_gradients_differentiate_real_energy(order):
    from pyscfad.lib import _multipole

    rng = np.random.default_rng(813)
    records = tuple(
        tuple(
            item + 0.1j * rng.normal(size=item.shape)
            if index in (1, 3, 4) and item is not None else item
            for index, item in enumerate(record)
        )
        for record in _endpoint_records(order)
    )
    expected = jax.value_and_grad(
        lambda left, right: jnp.real(_multipole.pair_energy(left, right, order)),
        argnums=(0, 1),
    )(*records)
    actual = _multipole.pair_energy_and_grad(*records, order)
    for value, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(value, reference, rtol=2e-12, atol=2e-12)
    assert not jnp.iscomplexobj(actual[0])


@pytest.mark.parametrize('order', [2, 3, 4])
def test_pair_energy_and_endpoint_gradients_match_numpy_finite_difference(order):
    from pyscfad.lib import _multipole

    left, right = _endpoint_records(order)
    value, gradients = _multipole.pair_energy_and_grad(left, right, order)
    np.testing.assert_allclose(
        value, multipole_pair_energy(left, right, order), rtol=2e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        _multipole.pair_energy(left, right, order), value, rtol=2e-12, atol=1e-14
    )
    rng = np.random.default_rng(81)
    for endpoint in (0, 1):
        record = (left, right)[endpoint]
        direction = tuple(
            None if item is None else rng.normal(size=item.shape) * 0.03
            for item in record
        )
        ad_directional = sum(
            np.vdot(grad, delta) for grad, delta in zip(gradients[endpoint], direction)
            if delta is not None
        )
        step = 1e-4
        shifted = {}
        for multiple in (-2, -1, 1, 2):
            records = [left, right]
            records[endpoint] = tuple(
                None if item is None else item + multiple * step * delta
                for item, delta in zip(record, direction)
            )
            shifted[multiple] = multipole_pair_energy(*records, order)
        fd_directional = (
            shifted[-2] - 8 * shifted[-1] + 8 * shifted[1] - shifted[2]
        ) / (12 * step)
        np.testing.assert_allclose(
            ad_directional, fd_directional, rtol=2e-7, atol=2e-11
        )


def test_pair_padding_default_and_runtime_override(monkeypatch):
    from pyscfad.lib import _multipole

    monkeypatch.delenv('PYSCFAD_WEAK_PAIR_BUCKET_SIZE', raising=False)
    left, right = _endpoint_records(4, widths=(81, 129))
    assert _multipole.pad_endpoint(left)[2].shape == (128,)
    assert _multipole.pad_endpoint(right)[2].shape == (256,)

    monkeypatch.setenv('PYSCFAD_WEAK_PAIR_BUCKET_SIZE', '64')
    assert _multipole.pad_endpoint(left)[2].shape == (128,)
    assert _multipole.pad_endpoint(right)[2].shape == (192,)
    assert _multipole.pad_endpoint(left, bucket_size=32)[2].shape == (96,)
    assert _multipole.pad_endpoint(left, bucket_size=1) is left


@pytest.mark.parametrize('value', ['0', '-1', '1.5', '', 'invalid'])
def test_pair_padding_rejects_invalid_environment(monkeypatch, value):
    from pyscfad.lib import _multipole

    monkeypatch.setenv('PYSCFAD_WEAK_PAIR_BUCKET_SIZE', value)
    left, _ = _endpoint_records(2)
    with pytest.raises(ValueError, match='positive integer'):
        _multipole.pad_endpoint(left)
    # An explicit valid argument takes precedence over the environment.
    assert _multipole.pad_endpoint(left, bucket_size=8)[2].shape == (8,)


@pytest.mark.parametrize('value', [0, -1, 1.5, True, np.bool_(True), '64'])
def test_pair_padding_rejects_invalid_explicit_size(value):
    from pyscfad.lib import _multipole

    left, _ = _endpoint_records(2)
    with pytest.raises(ValueError, match='positive integer'):
        _multipole.pad_endpoint(left, bucket_size=value)


@pytest.mark.parametrize('order', [2, 3, 4])
@pytest.mark.parametrize('bucket_size', [32, 128])
def test_pair_padding_preserves_energy_and_endpoint_gradients(monkeypatch, order, bucket_size):
    from pyscfad.lib import _multipole

    monkeypatch.setenv('PYSCFAD_WEAK_PAIR_BUCKET_SIZE', str(bucket_size))
    records = _endpoint_records(order, widths=(31, 33))
    padded = tuple(_multipole.pad_endpoint(value) for value in records)
    energy, bars = _multipole.pair_energy_and_grad(*padded, order)
    reference, reference_bars = _multipole.pair_energy_and_grad(*records, order)
    np.testing.assert_allclose(energy, reference, rtol=2e-12, atol=2e-12)
    for original, extended, bar, reference_bar in zip(records, padded, bars, reference_bars):
        width = original[2].shape[0]
        assert extended[0] is original[0]
        for slot in range(1, 5):
            if original[slot] is None:
                assert extended[slot] is None
            else:
                np.testing.assert_array_equal(extended[slot][..., :width], original[slot])
                np.testing.assert_array_equal(extended[slot][..., width:], 1.0 if slot == 2 else 0.0)
                np.testing.assert_array_equal(bar[slot][..., width:], 0.0)
        cropped = _multipole.crop_endpoint_cotangent(bar, original)
        for actual, expected in zip(jax.tree.leaves(cropped), jax.tree.leaves(reference_bar)):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_pair_padding_reuses_compiled_kernels_within_runtime_bucket(monkeypatch):
    from pyscfad.lib import _multipole

    monkeypatch.setenv('PYSCFAD_WEAK_PAIR_BUCKET_SIZE', '128')
    cache_sizes = []
    for widths in ((81, 105), (95, 120), (127, 128)):
        # Production endpoint builders return JAX arrays, including at exact
        # bucket boundaries where pad_endpoint returns its input unchanged.
        records = jax.tree.map(jnp.asarray, _endpoint_records(4, widths))
        endpoints = tuple(_multipole.pad_endpoint(value) for value in records)
        for kernel in (_multipole.pair_energy, _multipole.pair_energy_and_grad):
            jax.block_until_ready(kernel(*endpoints, 4))
        cache_sizes.append((_multipole.pair_energy._cache_size(),
                            _multipole.pair_energy_and_grad._cache_size()))
    assert cache_sizes[0] == cache_sizes[1] == cache_sizes[2]
