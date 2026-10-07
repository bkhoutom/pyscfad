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
    factors = rng.normal(size=(2, 3, 4))

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
        return prepare.prepare_system_inputs(mf, occupied, virtual)

    return inputs, adnp.asarray(fock), adnp.asarray(coeff), factors


def test_projected_fock_blocks_are_exactly_symmetric(projection):
    inputs, fock, coeff, factors = projection
    result = inputs(fock, coeff)
    for name in ('foo', 'fvv'):
        block = np.asarray(result[name])
        np.testing.assert_array_equal(block, block.T)
    validate_full_inputs(result)
    np.testing.assert_array_equal(result['B'], factors)


def test_projection_pullback_symmetrizes_general_seed(projection):
    inputs, fock, coeff, _ = projection
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
    for name, columns in (('foo', slice(0, 3)), ('fvv', slice(3, 7))):
        c = np.asarray(coeff[:, columns])
        g = np.asarray(seeds[name])
        g = (g + g.T) / 2
        expected_fock += c @ g @ c.T
        expected_coeff[:, columns] = fock @ c @ g.T + fock.T @ c @ g
    np.testing.assert_allclose(fock_bar, expected_fock, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(coeff_bar, expected_coeff, atol=1e-10, rtol=1e-12)
