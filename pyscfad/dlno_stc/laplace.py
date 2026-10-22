"""Fixed and molecule-fitted positive Laplace quadrature.

Spectra are active canonical orbital energies in hartree, with frozen orbitals
already excluded. Fit once on the host after SCF and keep the arrays fixed for
selection, differentiation, replay and finite differences. Kernel diagnostics
measure quadrature alone; they do not bound molecular energy or force errors.

The histogram and analytic residual/Jacobian follow the reference
``fit_kat_laplace_grid.py`` distributed with the fitted-LT tests.
"""
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_laguerre

from .controls import default_laplace_grid


def molecular_denominator_bounds(occupied: np.ndarray,
                                 virtual: np.ndarray) -> tuple[float, float]:
    """Exact active-orbital bounds, before histogram binning."""
    if occupied.size == 0 or virtual.size == 0 or not np.all(np.isfinite(occupied)) \
            or not np.all(np.isfinite(virtual)):
        raise ValueError("Orbital-energy arrays must be nonempty and finite")
    lo = 2 * float(np.min(virtual) - np.max(occupied))
    hi = 2 * float(np.max(virtual) - np.min(occupied))
    if lo <= 0:
        raise ValueError("All active virtual-minus-occupied gaps must be positive")
    return lo, hi


def denominator_histogram(occupied: np.ndarray, virtual: np.ndarray,
                          bins: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return denominator-bin centers and normalized counts within [lo, hi]."""
    actual_lo, actual_hi = molecular_denominator_bounds(occupied, virtual)
    gap_min, gap_max = actual_lo / 2, actual_hi / 2
    if gap_max == gap_min:
        centers = np.array([2 * gap_min])
        counts = np.array([1.0])
    else:
        spacing = (gap_max - gap_min) / (bins - 1)
        singles = np.zeros(bins)
        # O(nocc*nvirt) work but only O(nvirt+bins) peak memory.
        for e_i in occupied:
            position = np.clip((virtual - e_i - gap_min) / spacing, 0, bins - 1)
            left = np.floor(position).astype(np.intp)
            fraction = position - left
            right = np.minimum(left + 1, bins - 1)
            singles += np.bincount(left, weights=1 - fraction, minlength=bins)
            singles += np.bincount(right, weights=fraction, minlength=bins)
        counts = np.convolve(singles, singles)
        centers = actual_lo + np.arange(counts.size) * spacing
        centers[-1] = actual_hi
    mask = (centers >= lo) & (centers <= hi) & (counts > 0)
    if not np.any(mask):
        raise ValueError("No denominators fall inside the fitting interval")
    counts = counts[mask]
    return centers[mask], counts / np.sum(counts)


def residual_and_jacobian(z: np.ndarray, delta: np.ndarray, probability: np.ndarray,
                          lam: float) -> tuple[np.ndarray, np.ndarray]:
    n = z.size // 2
    t, w = np.exp(z[:n]), np.exp(z[n:])
    expval = np.exp(-delta[:, None] * t)
    weighted = expval * w
    q = np.sum(weighted, axis=1)
    g = weighted @ t
    root_p = np.sqrt(probability)
    energy = root_p * (q - 1 / delta)
    jac_e = np.concatenate((-delta[:, None] * t * weighted, weighted), axis=1)
    jac_e *= root_p[:, None]
    if lam == 0:
        return energy, jac_e
    root_pg = np.sqrt(lam) * root_p * delta
    derivative = root_pg * (g - 1 / delta**2)
    jac_g = np.concatenate((t * weighted * (1 - delta[:, None] * t),
                            t * weighted), axis=1)
    jac_g *= root_pg[:, None]
    return np.r_[energy, derivative], np.vstack((jac_e, jac_g))


def _spectra(occupied, virtual):
    arrays = []
    for name, value in (("occupied", occupied), ("virtual", virtual)):
        if value is None or np.iscomplexobj(value):
            raise ValueError(f"{name} must be a finite nonempty real 1-D spectrum")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite nonempty real 1-D spectrum") from exc
        if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite nonempty real 1-D spectrum")
        arrays.append(array)
    molecular_denominator_bounds(*arrays)
    return arrays


def fit_grid(delta, probability, points, lam):
    """Choose the best successful positive fit from the three reference starts."""
    laguerre_t, laguerre_w = roots_laguerre(points)
    best, best_norm = None, np.inf
    for scale_factor in (0.7, 1.0, 1.4):
        scale = np.sqrt(delta[0] * delta[-1]) * scale_factor
        cache_z, cache_result = None, None

        def evaluate(z):
            nonlocal cache_z, cache_result
            if cache_z is None or not np.array_equal(z, cache_z):
                cache_z = z.copy()
                cache_result = residual_and_jacobian(z, delta, probability, lam)
            return cache_result

        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                z0 = np.log(np.r_[laguerre_t / scale,
                                  laguerre_w * np.exp(laguerre_t) / scale])
                fit = least_squares(lambda z: evaluate(z)[0], z0,
                    jac=lambda z: evaluate(z)[1], max_nfev=5000,
                    ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if not fit.success or not np.all(np.isfinite(fit.x)):
                    continue
                roots, weights = np.exp(fit.x[:points]), np.exp(fit.x[points:])
                norm = np.linalg.norm(fit.fun)
                if (not np.isfinite(norm) or not np.all(np.isfinite(roots))
                        or not np.all(np.isfinite(weights))
                        or not np.all(roots > 0) or not np.all(weights > 0)):
                    continue
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            continue
        if norm < best_norm:
            best, best_norm = (roots, weights), norm
    if best is None:
        raise RuntimeError("All least-squares starts failed to produce a successful positive grid")
    order = np.argsort(best[0])
    return (np.array(best[0][order], dtype=np.float64, copy=True),
            np.array(best[1][order], dtype=np.float64, copy=True))


def laplace_grid_diagnostics(occupied, virtual, roots, weights):
    """Evaluate a resolved grid on exact bounds and the 600-point gap histogram.

    Maximum relative errors are sampled at 10001 equally spaced denominators,
    including both endpoints. Weighted RMS values use the normalized histogram;
    the derivative residual is multiplied by Delta, without the lambda factor.
    """
    occupied, virtual = _spectra(occupied, virtual)
    lo, hi = molecular_denominator_bounds(occupied, virtual)
    delta, probability = denominator_histogram(occupied, virtual, 600, lo, hi)
    dense = np.linspace(lo, hi, 10001)
    roots, weights = np.asarray(roots), np.asarray(weights)
    kernel = np.exp(-dense[:, None] * roots)
    fit_kernel = np.exp(-delta[:, None] * roots)
    fit_energy = fit_kernel @ weights
    fit_derivative = fit_kernel @ (roots * weights)
    return dict(denominator_bounds=[lo, hi],
        weighted_energy_rms=float(np.sqrt(np.sum(probability * (fit_energy - 1/delta)**2))),
        weighted_derivative_rms=float(np.sqrt(np.sum(probability *
            (delta * (fit_derivative - 1/delta**2))**2))),
        max_relative_energy_error=float(np.max(np.abs(dense * (kernel @ weights) - 1))),
        max_relative_derivative_error=float(np.max(np.abs(dense**2 *
            (kernel @ (roots * weights)) - 1))))


def make_laplace_grid(occupied=None, virtual=None, *, kind="default", points=6,
                      lambda_gradient=0.1):
    """Return independent float64 ``(roots, weights, diagnostics)``.

    ``default`` preserves the original six points, including their factor 2.6;
    fitting parameters are ignored. Without spectra its numerical diagnostics
    are None. ``fitted`` requires both active spectra, a positive integer point
    count, and a finite nonnegative derivative weight. There is no fallback.
    """
    if kind not in ("default", "fitted"):
        raise ValueError("kind must be 'default' or 'fitted'")
    diagnostics = dict(kind=kind, fitting_seconds=0., denominator_bounds=None,
        weighted_energy_rms=None, weighted_derivative_rms=None,
        max_relative_energy_error=None, max_relative_derivative_error=None)
    if kind == "default":
        roots, weights = default_laplace_grid()
        if occupied is not None or virtual is not None:
            diagnostics.update(laplace_grid_diagnostics(occupied, virtual, roots, weights))
    else:
        occupied, virtual = _spectra(occupied, virtual)
        if (not isinstance(points, (int, np.integer)) or isinstance(points, (bool, np.bool_))
                or points < 1):
            raise ValueError("points must be an integer >= 1")
        try:
            lam = float(lambda_gradient)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("lambda_gradient must be finite and >= 0") from exc
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("lambda_gradient must be finite and >= 0")
        started = perf_counter()
        lo, hi = molecular_denominator_bounds(occupied, virtual)
        delta, probability = denominator_histogram(occupied, virtual, 600, lo, hi)
        roots, weights = fit_grid(delta, probability, points, lam)
        diagnostics['fitting_seconds'] = perf_counter() - started
        diagnostics['lambda_gradient'] = lam
        diagnostics.update(laplace_grid_diagnostics(occupied, virtual, roots, weights))
    diagnostics['points'] = int(roots.size)
    return roots, weights, diagnostics
