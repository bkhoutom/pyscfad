import jax
import numpy
import pytest

from pyscfad import numpy as np
from pyscfad.lno.lno_base import _dlno_outside_space


@pytest.mark.parametrize('complex_case', (False, True))
def test_dlno_outside_space_preserves_full_projector(complex_case):
    rng = numpy.random.default_rng(20260816)
    full_raw = rng.normal(size=(19, 12))
    selected_raw = rng.normal(size=(12, 12))
    if complex_case:
        full_raw = full_raw + 1j * rng.normal(size=full_raw.shape)
        selected_raw = (
            selected_raw + 1j * rng.normal(size=selected_raw.shape)
        )

    full, _ = numpy.linalg.qr(full_raw, mode='reduced')
    rotation, _ = numpy.linalg.qr(selected_raw)
    selected = full @ rotation[:, :5]
    outside = numpy.asarray(_dlno_outside_space(
        np.asarray(full), np.asarray(selected), 1e-8,
    ))

    assert outside.shape == (19, 7)
    assert numpy.isfinite(outside).all()
    assert numpy.linalg.norm(
        outside.T.conj() @ outside - numpy.eye(7), 2,
    ) < 1e-10
    assert numpy.linalg.norm(selected.T.conj() @ outside, 2) < 1e-10
    combined = numpy.column_stack((selected, outside))
    assert numpy.linalg.norm(
        combined @ combined.T.conj() - full @ full.T.conj(), 2,
    ) < 1e-10


def test_dlno_outside_space_fixed_width_and_edges():
    full = np.eye(12)
    assert _dlno_outside_space(full, None, 1e-8).shape == (12, 12)
    assert _dlno_outside_space(
        full, full[:, :0], 1e-8,
    ).shape == (12, 12)
    assert _dlno_outside_space(full, full, 1e-8).shape == (12, 0)

    water16_virtuals = np.eye(848)
    outside = _dlno_outside_space(
        water16_virtuals, water16_virtuals[:, :353], 1e-8,
    )
    assert outside.shape == (848, 495)
    assert numpy.isfinite(numpy.asarray(outside)).all()


def test_dlno_outside_space_is_frozen_and_jittable():
    full = np.eye(12)
    selected = full[:, :5]

    def objective(selected_space):
        return np.sum(_dlno_outside_space(
            full, selected_space, 1e-8,
        ))

    value, pullback = jax.vjp(objective, selected)
    gradient = pullback(np.ones_like(value))[0]
    assert float(np.linalg.norm(gradient)) == 0.0

    outside = jax.jit(
        lambda selected_space: _dlno_outside_space(
            full, selected_space, 1e-8,
        )
    )(selected)
    assert outside.shape == (12, 7)


def test_dlno_outside_space_rejects_invalid_inputs():
    full = numpy.eye(12)
    selected = full[:, :5]

    nonfinite = full.copy()
    nonfinite[0, 0] = numpy.nan
    with pytest.raises(FloatingPointError, match='Non-finite'):
        _dlno_outside_space(
            np.asarray(nonfinite), np.asarray(selected), 1e-8,
        )

    rank_deficient_full = full.copy()
    rank_deficient_full[:, 1] = rank_deficient_full[:, 0]
    with pytest.raises(RuntimeError, match='full space.*rank deficient'):
        _dlno_outside_space(
            np.asarray(rank_deficient_full), np.asarray(selected), 1e-8,
        )

    rank_deficient_selected = selected.copy()
    rank_deficient_selected[:, 1] = rank_deficient_selected[:, 0]
    with pytest.raises(RuntimeError, match='selected space.*rank deficient'):
        _dlno_outside_space(
            np.asarray(full), np.asarray(rank_deficient_selected), 1e-8,
        )

    full_rectangular = numpy.eye(19)[:, :12]
    noncontained = full_rectangular[:, :5].copy()
    noncontained[:, 4] = numpy.eye(19)[:, 18]
    with pytest.raises(RuntimeError, match='not contained'):
        _dlno_outside_space(
            np.asarray(full_rectangular), np.asarray(noncontained), 1e-8,
        )

    with pytest.raises(ValueError, match='incompatible shape'):
        _dlno_outside_space(
            np.eye(12), np.eye(11)[:, :5], 1e-8,
        )
    with pytest.raises(ValueError, match='wider than full_space'):
        _dlno_outside_space(
            np.eye(12)[:, :5], np.eye(12)[:, :6], 1e-8,
        )
