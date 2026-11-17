"""Real multipole forward contractions and explicit NumPy pullbacks.

No tracing or shape-specialized compilation is used. Bars use the ordinary
real Frobenius inner product. The JAX implementation remains the AD oracle.
"""
from functools import lru_cache
from itertools import product

import numpy as np

from ._cartesian_moments import cartesian_components


def _polynomial(n, rank, terms, derivative):
    indices = np.indices((3,) * rank)
    value = np.zeros((3,) * rank)
    tangent = np.zeros((3,) + value.shape) if derivative else None
    for coefficient, axes, equalities in terms:
        base = np.full(value.shape, coefficient, dtype=float)
        for a, b in equalities:
            base *= indices[a] == indices[b]
        term = base.copy()
        for a in axes:
            term *= n[indices[a]]
        value += term
        if derivative:
            for position, a in enumerate(axes):
                term = base.copy()
                for k, b in enumerate(axes):
                    if k != position:
                        term *= n[indices[b]]
                for xyz in range(3):
                    tangent[xyz] += term * (indices[a] == xyz)
    return value, tangent


def _interaction(displacement, order, derivative):
    distance = np.linalg.norm(displacement)
    if not distance > 0:
        raise ValueError('weak multipole endpoints must have distinct centers')
    n = displacement / distance
    size = 3 if order == 2 else 12 if order == 3 else 39
    interaction = np.zeros((size, size))
    gradient = np.zeros((3, size, size)) if derivative else None

    def block(left, right, rank, power, terms, divisor=1., transpose_sign=None):
        value, dn = _polynomial(n, rank, terms, derivative)
        shape = (left.stop-left.start, right.stop-right.start)
        value = value.reshape(shape) / (divisor * distance**power)
        interaction[left, right] = value
        if transpose_sign is not None:
            interaction[right, left] = transpose_sign * value.T
        if derivative:
            dn = dn.reshape((3,) + shape) / (divisor * distance**power)
            dr = (np.einsum('kx,kij->xij', np.eye(3)-np.outer(n, n), dn)
                  - power * n[:, None, None] * value) / distance
            gradient[:, left, right] = dr
            if transpose_sign is not None:
                gradient[:, right, left] = transpose_sign * dr.swapaxes(1, 2)

    mu, theta, omega = slice(0, 3), slice(3, 12), slice(12, 39)
    block(mu, mu, 2, 3, [(1, (), ((0, 1),)), (-3, (0, 1), ())])
    if order > 2:
        block(mu, theta, 3, 4, [(5, (0, 1, 2), ()), (-2, (2,), ((0, 1),))],
              transpose_sign=-1)
    if order > 3:
        block(mu, omega, 4, 5,
              [(9, (2, 3), ((0, 1),)), (-21, (0, 1, 2, 3), ())],
              divisor=3)
        block(omega, mu, 4, 5,
              [(9, (0, 1), ((2, 3),)), (-21, (0, 1, 2, 3), ())], divisor=3)
        block(theta, theta, 4, 5,
              [(35, (0, 1, 2, 3), ()), (-20, (1, 3), ((0, 2),)),
               (2, (), ((0, 2), (1, 3)))], divisor=3)
    return interaction, gradient


def pair(left, right, order, *, with_grad=False):
    """Evaluate -4 sum(K**2/D), optionally returning both endpoint bars."""
    if order not in (2, 3, 4):
        raise ValueError('multipole order must be 2, 3 or 4')
    left = tuple(None if x is None else np.asarray(x) for x in left)
    right = tuple(None if x is None else np.asarray(x) for x in right)
    slots = (1,) + ((3,) if order > 2 else ()) + ((4,) if order > 3 else ())
    a = np.concatenate([left[k].reshape(3**(k-1 if k > 1 else 1), -1)
                        for k in slots], axis=0)
    b = np.concatenate([right[k].reshape(3**(k-1 if k > 1 else 1), -1)
                        for k in slots], axis=0)
    h, dh = _interaction(right[0]-left[0], order, with_grad)
    k = a.T @ h @ b
    gap = left[2][:, None] + right[2][None, :]
    ratio = k / gap
    energy = np.float64(-4 * np.sum(k * ratio))
    if not with_grad:
        return energy
    k_bar = -8 * ratio
    gap_bar = 4 * ratio**2
    a_bar, b_bar = h @ b @ k_bar.T, h.T @ a @ k_bar
    h_bar = a @ k_bar @ b.T
    displacement_bar = np.einsum('xij,ij->x', dh, h_bar)
    bars = []
    for endpoint, moments_bar, center_bar, excitation_bar in (
            (left, a_bar, -displacement_bar, gap_bar.sum(axis=1)),
            (right, b_bar, displacement_bar, gap_bar.sum(axis=0))):
        bar = [None if x is None else np.zeros_like(x) for x in endpoint]
        bar[0], bar[2] = center_bar, excitation_bar
        offset = 0
        for slot in slots:
            count = int(np.prod(endpoint[slot].shape[:-1]))
            bar[slot] = moments_bar[offset:offset+count].reshape(endpoint[slot].shape)
            offset += count
        bars.append(tuple(bar))
    return energy, tuple(bars)


@lru_cache(None)
def _traceless(rank):
    """Matrix for the existing Cartesian quadrupole/octupole convention."""
    size = 3**rank
    raw = np.eye(size).reshape((3,)*rank + (size,))
    eye = np.eye(3)
    if rank == 2:
        result = .5 * (3*raw - np.einsum('xy,a->xya', eye, np.einsum('xxa->a', raw)))
    else:
        result = .5 * (5*raw
            - np.einsum('yz,xa->xyza', eye, np.einsum('xyya->xa', raw))
            - np.einsum('xz,ya->xyza', eye, np.einsum('xyxa->ya', raw))
            - np.einsum('xy,za->xyza', eye, np.einsum('xxza->za', raw)))
    return result.reshape(size, size)


def _translation(center, rank, derivative):
    matrices = [np.zeros((3**rank, 3**s)) for s in range(rank+1)]
    gradients = [np.zeros((3, 3**rank, 3**s)) for s in range(rank+1)] if derivative else None
    for row, axes in enumerate(product(range(3), repeat=rank)):
        for mask in range(1 << rank):
            retained = [axes[i] for i in range(rank) if mask & (1 << i)]
            shifted = [axes[i] for i in range(rank) if not mask & (1 << i)]
            column = sum(a * 3**(len(retained)-i-1) for i, a in enumerate(retained))
            sign = (-1)**len(shifted)
            matrices[len(retained)][row, column] += sign * np.prod(center[shifted])
            if derivative:
                for i, a in enumerate(shifted):
                    gradients[len(retained)][a, row, column] += (
                        sign * np.prod(center[shifted[:i]+shifted[i+1:]]))
    return matrices, gradients


def endpoint(screen, operators, order, *, with_grad=False):
    """One occupied endpoint, with bars for its screen and raw AO operators."""
    eo, occupied, ev, virtual = map(np.asarray, screen)
    if occupied.shape[1] != 1:
        raise ValueError('weak endpoints require one occupied target')
    c = occupied[:, 0]
    nvir = virtual.shape[1]
    ops, maps, raw = [], [], []
    for rank in range(order):
        op = np.asarray(operators[rank]).reshape((-1, c.size, c.size))
        mapping = (cartesian_components(rank)[1] if op.shape[0] != 3**rank
                   else np.arange(3**rank))
        projected = np.stack([virtual.T @ (item @ c) for item in op])
        ops.append(op); maps.append(mapping); raw.append(projected[mapping])
    center = np.array([c @ item @ c for item in ops[1]])
    moments = [None, raw[1], None, None]
    translations = {}
    for rank in range(2, order):
        matrices, gradients = _translation(center, rank, with_grad)
        shifted = sum(matrix @ value for matrix, value in zip(matrices, raw))
        moments[rank] = (_traceless(rank) @ shifted).reshape((3,)*rank + (nvir,))
        translations[rank] = matrices, gradients
    result = (center, moments[1], ev-eo[0], moments[2], moments[3])
    if not with_grad:
        return result

    def pullback(bar):
        center_bar, mu_bar, gap_bar, theta_bar, omega_bar = bar
        center_bar = np.array(center_bar, copy=True)
        raw_bar = [np.zeros_like(value) for value in raw]
        raw_bar[1] += mu_bar
        for rank, moment_bar in ((2, theta_bar), (3, omega_bar)):
            if rank >= order:
                continue
            shifted_bar = _traceless(rank).T @ np.asarray(moment_bar).reshape(3**rank, nvir)
            matrices, gradients = translations[rank]
            for s, matrix in enumerate(matrices):
                raw_bar[s] += matrix.T @ shifted_bar
                center_bar += np.einsum('xra,ra->x', gradients[s] @ raw[s], shifted_bar)
        c_bar, v_bar = np.zeros_like(c), np.zeros_like(virtual)
        operator_bars = [None] * len(operators)
        for rank, op in enumerate(ops):
            packed_bar = np.zeros((len(op), nvir))
            np.add.at(packed_bar, maps[rank], raw_bar[rank])
            op_bar = np.empty_like(op)
            for k, (item, weights) in enumerate(zip(op, packed_bar)):
                v_bar += np.outer(item @ c, weights)
                projected_bar = virtual @ weights
                c_bar += item.T @ projected_bar
                op_bar[k] = np.outer(projected_bar, c)
                if rank == 1:
                    c_bar += center_bar[k] * ((item + item.T) @ c)
                    op_bar[k] += center_bar[k] * np.outer(c, c)
            operator_bars[rank] = op_bar.reshape(operators[rank].shape)
        screen_bar = type(screen)(-np.asarray(gap_bar).sum(keepdims=True), c_bar[:, None],
                                   np.asarray(gap_bar), v_bar)
        return screen_bar, tuple(operator_bars)

    return result, pullback
