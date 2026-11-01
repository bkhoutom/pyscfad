"""Prepared symmetric Fock blocks retain the complete projection response."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from pyscfad import numpy as adnp
from pyscfad.dlno_stc import prepare
from pyscfad.dlno_stc.protocol import validate_full_inputs


@pytest.fixture(params=(False, True), ids=('direct', 'cached'))
def projection(request, monkeypatch):
    rng = np.random.default_rng(1207)
    fock = rng.normal(size=(7, 7))
    fock = (fock + fock.T) / 2
    # Large AO coefficients and cancellation in nearly diagonal MO blocks
    # expose projection roundoff without running molecular integrals or SCF.
    _, coeff = np.linalg.eigh(fock)
    coeff *= 100
    occupied, virtual = np.arange(3), np.arange(3, 7)
    occupied_rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    virtual_rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    factors = rng.normal(size=(2, 3, 4))

    # The synthetic coefficients deliberately have no molecular metric.
    # Isolate localization with a differentiable rotated frame so these
    # tests retain independent AO-Fock and raw-coefficient variations.
    boys_reference = SimpleNamespace(
        target_reference_coeff=coeff[:, occupied] @ occupied_rotation,
        target_reference_coords=np.zeros((1, 3)), lo_kwargs={},
    )
    virtual_anchor_columns = np.arange(len(virtual), dtype=np.int32)

    def local_frame(mf, occupied_, virtual_, *, boys_reference,
                    virtual_anchor_columns):
        return SimpleNamespace(
            occupied_coeff=mf.mo_coeff[:, occupied_] @ occupied_rotation,
            virtual_coeff=mf.mo_coeff[:, virtual_] @ virtual_rotation,
            boys_reference=boys_reference,
            virtual_anchor_columns=virtual_anchor_columns,
        )

    monkeypatch.setattr(prepare, 'build_system_local_frame', local_frame)

    # Isolate the Fock response; the two real preparation branches still run.
    monkeypatch.setattr(prepare.lno_df, 'get_local_Lov',
                        lambda *args, **kwargs: adnp.asarray(factors))
    monkeypatch.setattr(prepare.lno_df, 'transform_df_to_mo',
                        lambda *args, **kwargs: adnp.asarray(factors))

    def inputs(fock_, coeff_):
        mf = SimpleNamespace(
            mol=SimpleNamespace(natm=1), mo_coeff=coeff_,
            with_df=SimpleNamespace(_cderi=object() if request.param else None),
            get_fock=lambda: fock_,
        )
        return prepare.prepare_system_inputs(
            mf, occupied, virtual, boys_reference=boys_reference,
            virtual_anchor_columns=virtual_anchor_columns,
        )

    return (inputs, adnp.asarray(fock), adnp.asarray(coeff), factors,
            occupied_rotation, virtual_rotation)


def test_projected_fock_blocks_are_exactly_symmetric(projection):
    inputs, fock, coeff, factors, occupied_rotation, virtual_rotation = projection
    result = inputs(fock, coeff)
    for name, columns, rotation in (
            ('foo', slice(0, 3), occupied_rotation),
            ('fvv', slice(3, 7), virtual_rotation)):
        block = np.asarray(result[name])
        np.testing.assert_array_equal(block, block.T)
        c = np.asarray(coeff[:, columns]) @ rotation
        expected = c.T @ np.asarray(fock) @ c
        expected = (expected + expected.T) / 2
        np.testing.assert_allclose(block, expected, atol=1e-10, rtol=1e-12)
        assert np.max(np.abs(block - np.diag(np.diag(block)))) > 1.0
    validate_full_inputs(result)
    np.testing.assert_array_equal(result['B'], factors)


def test_projection_pullback_symmetrizes_general_seed(projection):
    inputs, fock, coeff, _, occupied_rotation, virtual_rotation = projection
    result, pullback = jax.vjp(inputs, fock, coeff)
    rng = np.random.default_rng(917)
    seeds = {name: adnp.asarray(rng.normal(size=value.shape))
             for name, value in result.items()}
    seeds['B'] = adnp.zeros_like(result['B'])
    fock_bar, coeff_bar = pullback(seeds)

    # For Y=sym(C.T F C), the adjoint entering the projection is sym(G).
    # These matrix differentials provide an oracle independent of JAX's VJP.
    expected_fock = np.zeros_like(fock)
    expected_coeff = np.zeros_like(coeff)
    for name, columns, rotation in (
            ('foo', slice(0, 3), occupied_rotation),
            ('fvv', slice(3, 7), virtual_rotation)):
        c = np.asarray(coeff[:, columns]) @ rotation
        g = np.asarray(seeds[name])
        g = (g + g.T) / 2
        expected_fock += c @ g @ c.T
        local_coeff_bar = fock @ c @ g.T + fock.T @ c @ g
        expected_coeff[:, columns] = local_coeff_bar @ rotation.T
    np.testing.assert_allclose(fock_bar, expected_fock, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(coeff_bar, expected_coeff, atol=1e-10, rtol=1e-12)
