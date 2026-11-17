"""Concrete CPU endpoint contractions with size-independent JAX translations.

The driver keeps the general JAX path for tracing and unsupported inputs.
Native BLAS handles physical AO dimensions; only 128-column moment tiles reach
the compiled translation and its exact transpose. No eigensystem is rebuilt.
"""
import jax
import numpy as np

from pyscfad.lib._cartesian_moments import cartesian_components
from pyscfad.lib._multipole import _translate_moments
from pyscfad.lib._threading import dense_blas_threads


_TILE = 128


def is_concrete_cpu(tree):
    """Whether leaves can be read on the host without crossing an AD boundary."""
    for value in jax.tree_util.tree_leaves(tree):
        if isinstance(value, jax.core.Tracer):
            return False
        if isinstance(value, jax.Array):
            if (not value.is_fully_addressable
                    or any(device.platform != 'cpu' for device in value.devices())):
                return False
        elif not isinstance(value, np.ndarray):
            return False
    return True


def supports(screen, raw, order):
    """Restrict the native rule to one double-precision packed order-4 endpoint."""
    if (order != 4 or not jax.config.x64_enabled
            or not hasattr(screen, 'occupied_coeff') or len(raw) != 4
            or any(value is None for value in raw)
            or not is_concrete_cpu((screen, raw))):
        return False
    eo, occupied, ev, virtual = screen
    if (occupied.ndim != 2 or occupied.shape[1] != 1 or occupied.shape[0] == 0
            or virtual.ndim != 2 or virtual.shape[0] != occupied.shape[0]
            or eo.shape != (1,) or ev.shape != (virtual.shape[1],)
            or eo.dtype != np.float64 or ev.dtype != np.float64):
        return False
    rows = occupied.shape[0]
    shapes = ((rows, rows), (3, rows, rows), (6, rows, rows), (10, rows, rows))
    dtype = occupied.dtype
    return (dtype in (np.dtype(np.float64), np.dtype(np.complex128))
            and virtual.dtype == dtype
            and all(value.shape == shape and value.dtype == dtype
                    for value, shape in zip(raw, shapes)))


def _tile_columns(value, start):
    result = np.zeros(value.shape[:-1] + (_TILE,), dtype=value.dtype)
    width = min(_TILE, value.shape[-1] - start)
    result[..., :width] = value[..., start:start + width]
    return result


@jax.jit
def _translation_tile(center, transition):
    second = transition[4:10][cartesian_components(2)[1]].reshape(1, 3, 3, _TILE)
    third = transition[10:20][cartesian_components(3)[1]].reshape(1, 3, 3, 3, _TILE)
    values = _translate_moments(
        center[None, :], transition[None, 0], transition[None, 1:4], second, third, 4)
    return tuple(value[0] for value in values)


@jax.jit
def _translation_transpose(center, transition, seed):
    _, pullback = jax.vjp(_translation_tile, center, transition)
    return pullback(seed)


def endpoint(screen, raw, order):
    """Evaluate one supported endpoint without retaining its moment tape."""
    return linearize(screen, raw, order)[0]


def linearize(screen, raw, order):
    """Return an endpoint and its full first-order input pullback.

    All arguments and eventual cotangents must be concrete CPU arrays. The
    caller supplies the general JAX fallback when differentiating the returned
    linear map. Complex cotangents follow JAX's convention, without assuming
    Hermitian AO operators or imposing symmetry on their ambient derivative.
    """
    if not supports(screen, raw, order):
        raise ValueError('native weak moments require concrete packed order-4 CPU inputs')
    eo, occupied, ev, virtual = map(np.asarray, screen)
    occupied = occupied[:, 0]
    rows, columns = virtual.shape
    operators = np.concatenate((np.asarray(raw[0])[None], *map(np.asarray, raw[1:])), axis=0)
    with dense_blas_threads():
        projected = (operators.reshape(-1, rows) @ occupied).reshape(20, rows)
        center = projected[1:4] @ occupied.conj()
        transition = projected @ virtual.conj()
    moments = tuple(np.empty(shape + (columns,), dtype=occupied.dtype)
                    for shape in ((3,), (3, 3), (3, 3, 3)))
    for start in range(0, columns, _TILE):
        width = min(_TILE, columns - start)
        values = _translation_tile(center, _tile_columns(transition, start))
        for target, value in zip(moments, values):
            target[..., start:start + width] = np.asarray(value)[..., :width]
    value = (center, moments[0], ev - eo[0], moments[1], moments[2])

    def pullback(seed):
        center_bar = np.array(seed[0], copy=True)
        transition_bar = np.zeros_like(transition)
        for start in range(0, columns, _TILE):
            width = min(_TILE, columns - start)
            weights = tuple(_tile_columns(np.asarray(seed[k]), start) for k in (1, 3, 4))
            cb, tb = _translation_transpose(center, _tile_columns(transition, start), weights)
            center_bar += np.asarray(cb)
            transition_bar[:, start:start + width] = np.asarray(tb)[:, :width]
        with dense_blas_threads():
            virtual_bar = (projected.T @ transition_bar).conj()
            projected_bar = transition_bar @ virtual.conj().T
            projected_bar[1:4] += center_bar[:, None] * occupied.conj()[None, :]
            occupied_bar = (center_bar @ projected[1:4]).conj()
            occupied_bar += operators.reshape(-1, rows).T @ projected_bar.ravel()
            operators_bar = np.einsum('kr,c->krc', projected_bar, occupied)
        gap_bar = np.asarray(seed[2])
        screen_bar = screen._replace(
            occupied_energy=np.asarray([-gap_bar.sum()], dtype=eo.dtype),
            occupied_coeff=occupied_bar[:, None], virtual_energy=gap_bar.copy(),
            virtual_coeff=virtual_bar)
        return screen_bar, (operators_bar[0], operators_bar[1:4],
                            operators_bar[4:10], operators_bar[10:20])

    return value, pullback
