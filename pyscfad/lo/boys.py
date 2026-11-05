# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from contextlib import contextmanager
from functools import partial
import numpy
import jax
from pyscf.lib import logger
from pyscf.lo import boys as pyscf_boys
from pyscfad import numpy as np
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.soscf.ciah import (
    extract_rotation,
    pack_uniq_var,
)
from pyscfad.tools.linear_solver import gen_gmres, precond_by_hdiag, eager_gmres
from pyscfad.scipy.linalg import logm

# modified from pyscf v2.6
def kernel(localizer, mo_coeff=None, callback=None, verbose=None,
           return_u=False):
    from pyscf.tools import mo_mapping
    from pyscf.soscf import ciah
    if mo_coeff is not None:
        localizer.mo_coeff = numpy.asarray(mo_coeff, order='C')
    # Newer PySCF stores the rotation dimension on the optimizer itself.
    if hasattr(localizer, 'norb'):
        localizer.norb = localizer.mo_coeff.shape[1]
    if localizer.mo_coeff.shape[1] <= 1:
        return localizer.mo_coeff

    if localizer.verbose >= logger.WARN:
        localizer.check_sanity()
    localizer.dump_flags()

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(localizer, verbose=verbose)

    if localizer.conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(localizer.conv_tol*.1)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = localizer.conv_tol_grad

    if mo_coeff is None:
        if getattr(localizer, 'mol', None) and localizer.mol.natm == 0:
            # For customized Hamiltonian
            u0 = localizer.get_init_guess('random')
        else:
            u0 = localizer.get_init_guess(localizer.init_guess)
    else:
        u0 = localizer.get_init_guess(None)

    rotaiter = ciah.rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=log)
    u, g_orb, stat = next(rotaiter)
    cput1 = log.timer('initializing CIAH', *cput0)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = 0
    for imacro in range(localizer.max_cycle):
        norm_gorb = numpy.linalg.norm(g_orb)
        u0 = numpy.dot(u0, u)
        e = localizer.cost_function(u0)
        e_last, de = e, e-e_last

        log.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        cput1 = log.timer(f'cycle= {imacro+1}', *cput1)

        if (norm_gorb < conv_tol_grad and abs(de) < localizer.conv_tol
                and stat.tot_hop < localizer.ah_max_cycle):
            conv = True

        if callable(callback):
            callback(locals())

        if conv:
            break

        u, g_orb, stat = rotaiter.send(u0)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d KF %d Hx',
             imacro+1, e, norm_gorb,
             tot_kf+imacro+1, tot_hop)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    sorted_idx = mo_mapping.mo_1to1map(u0)
    if return_u:
        return u0, sorted_idx
    localizer.mo_coeff = numpy.dot(localizer.mo_coeff, u0[:,sorted_idx])
    return localizer.mo_coeff


class Boys(pyscf_boys.Boys):
    """CIAH Boys optimization with dipoles projected once per forward solve."""

    def get_init_guess(self, key='atomic'):
        if hasattr(key, 'shape'):
            u0 = numpy.asarray(key)
        else:
            u0 = pyscf_boys.Boys.get_init_guess(self, key=key)
        return u0

    @contextmanager
    def _cache_dipoles(self, mo_coeff=None):
        """Scope numerical caches to one fixed molecule and orbital subspace.

        Only projected dipoles and the scalar r2 trace are cached during CIAH.
        The differentiable optimality condition builds its own live dipoles.
        """
        coeff = numpy.asarray(self.mo_coeff if mo_coeff is None else mo_coeff)
        center = (numpy.einsum('z,zx->x', self.mol.atom_charges(), self.mol.atom_coords())
                  / self.mol.atom_charges().sum())
        dipoles = pyscf_boys.dipole_integral(self.mol, coeff, center)
        with self.mol.with_common_origin(center):
            r2 = numpy.asarray(self.mol.intor_symmetric('int1e_r2'))
        spread_trace = numpy.einsum('pi,pi->', coeff.conj(), r2 @ coeff).real
        del r2
        previous = getattr(self, '_boys_dipole_cache', None)
        self._boys_dipole_cache = (dipoles, spread_trace)
        try:
            yield
        finally:
            if previous is None:
                del self._boys_dipole_cache
            else:
                self._boys_dipole_cache = previous

    def _rotated_cached_dipoles(self, u):
        dipoles = self._boys_dipole_cache[0]
        return dipoles if u is None else u.conj().T @ (dipoles @ u)

    def get_grad(self, u=None, dip=None):
        if dip is None and getattr(self, '_boys_dipole_cache', None) is not None:
            dip = self._rotated_cached_dipoles(u)
        return super().get_grad(u, dip)

    def gen_g_hop(self, u=None):
        if getattr(self, '_boys_dipole_cache', None) is None:
            return super().gen_g_hop(u)
        # Same analytic gradient/Hessian action as PySCF Boys.gen_g_hop,
        # evaluated entirely in the fixed localization subspace.
        dip = self._rotated_cached_dipoles(u)
        g0 = numpy.einsum('xii,xip->pi', dip, dip)
        g = self.get_grad(u, dip)
        h_diag = numpy.einsum('xii,xpp->pi', dip, dip) * 2
        h_diag -= g0.diagonal() + g0.diagonal().reshape(-1, 1)
        h_diag += numpy.einsum('xip,xip->pi', dip, dip) * 2
        h_diag += numpy.einsum('xip,xpi->pi', dip, dip) * 2
        h_diag = -self.pack_uniq_var(h_diag) * 4
        g0 = g0 + g0.conj().T

        def h_op(x):
            x = self.unpack_uniq_var(x)
            norb = x.shape[0]
            hx = numpy.dot(x.T, g0.T).conj()
            hx += numpy.einsum('xip,xi->pi', dip, numpy.einsum('qi,xiq->xi', x, dip)) * 2
            hx -= numpy.einsum('xpp,xip->pi', dip,
                              numpy.dot(dip.reshape(-1, norb), x).reshape(3, norb, norb)) * 2
            hx -= numpy.einsum('xip,xp->pi', dip, numpy.einsum('qp,xpq->xp', x, dip)) * 2
            return -self.pack_uniq_var(hx-hx.conj().T) * 2

        return g, h_op, h_diag

    def cost_function(self, u=None):
        if getattr(self, '_boys_dipole_cache', None) is None:
            return super().cost_function(u)
        dip = self._rotated_cached_dipoles(u)
        return self._boys_dipole_cache[1] - numpy.einsum('xii,xii->', dip, dip).real

    def kernel(self, mo_coeff=None, callback=None, verbose=None, return_u=False):
        coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        if coeff.shape[1] <= 1:
            return kernel(self, mo_coeff, callback, verbose, return_u)
        with self._cache_dipoles(coeff):
            return kernel(self, mo_coeff, callback, verbose, return_u)


def dipole_integral(mol, mo_coeff):
    # FIXME do we need charge center response?
    #charge_center = numpy.einsum('z,zx->x', mol.atom_charges(),
    #                             stop_grad(mol.atom_coords()))
    # NOTE set origin as the charge center,
    # otherwise, will need to compute charge center response
    charge_center = numpy.zeros(3)
    with mol.with_common_origin(charge_center):
        r = mol.intor('int1e_r')
        dip = np.einsum('ui,xuv,vj->xij', mo_coeff.conj(), r, mo_coeff)
    return dip

def cost_function(x, mol, mo_coeff):
    u = extract_rotation(x)
    mo_coeff = np.dot(mo_coeff, u)
    dip = dipole_integral(mol, mo_coeff)
    r2 = mol.intor_symmetric('int1e_r2')
    r2 = np.einsum('pi,pi->', mo_coeff, np.dot(r2, mo_coeff))
    val = r2 - np.einsum('xii,xii->', dip, dip)
    return val

def _opt_cond(x, mol, mo_coeff):
    g = jax.grad(cost_function, 0)(x, mol, mo_coeff)
    return g


def _projected_rotated_dipoles(x, dipoles):
    """Rotate three occupied-space dipoles in the original exp(x) chart."""
    u = extract_rotation(x)
    return np.einsum('pi,apq,qj->aij', u.conj(), dipoles, u)


def _projected_cost(x, dipoles):
    dip = _projected_rotated_dipoles(x, dipoles)
    # Tr(U.T @ C.T @ r2 @ C @ U) is invariant under orthogonal U. Its
    # x-gradient and the external derivatives of that gradient are zero.
    return -np.einsum('aii,aii->', dip, dip)


# Arrays are explicit arguments, so both the gradient and its compiled VJPs
# can be reused for different geometries/seeds with the same occupied size.
# Host integrals remain outside this compiled, purely numerical boundary.
_projected_gradient = jax.jit(jax.grad(_projected_cost, 0))


def _projected_opt_cond(x, mol, mo_coeff):
    return _projected_gradient(x, dipole_integral(mol, mo_coeff))


def _projected_opt_cond_aux(x, mol, mo_coeff):
    dipoles = dipole_integral(mol, mo_coeff)
    return _projected_gradient(x, dipoles), (x, dipoles)


@jax.jit
def _projected_tangent_diagonal(x, dipoles):
    dip = _projected_rotated_dipoles(x, dipoles)
    centers = np.diagonal(dip, axis1=-2, axis2=-1)
    differences = centers[:, :, None] - centers[:, None, :]
    diagonal = 4 * np.sum(differences ** 2, axis=0) - 16 * np.sum(dip ** 2, axis=0)
    return pack_uniq_var(diagonal)


def _projected_preconditioner(aux, mol, mo_coeff, *, solver=None):
    """Positive regularized tangent diagonal, only an approximate inverse.

    The true action still uses the exact original exponential coordinates.
    In particular, this diagonal never shifts or replaces the response Hessian.
    """
    from pyscfad.tools import linear_solver

    del mol, mo_coeff
    if solver is None:
        solver = linear_solver.gen_gmres()
    # Concrete eager inputs can still select a JAX solver. Its preconditioner
    # must be a JAX-compatible callable, not a scipy LinearOperator. Preserve
    # this distinction through both partials and profiling decorators.
    while isinstance(solver, partial) or hasattr(solver, '__wrapped__'):
        solver = solver.func if isinstance(solver, partial) else solver.__wrapped__
    if solver not in (linear_solver.pyscfad_gmres, linear_solver.gmres_safe):
        return None
    diagonal = numpy.asarray(_projected_tangent_diagonal(*aux), dtype=float)
    magnitude = numpy.abs(diagonal)
    floor = max(1e-10, float(numpy.max(magnitude, initial=0.0)) * 1e-8)
    return precond_by_hdiag(numpy.maximum(magnitude, floor))


def _extract_x0(loc, u):
    if numpy.linalg.det(u) < 0:
        u[:,0] *= -1
    assert numpy.isclose(numpy.linalg.det(u), 1, atol=1e-8)

    h_diag = loc.gen_g_hop(u)[2]
    if numpy.any(h_diag < -1e-6):
        logger.warn(loc,
                    'Saddle point reached in orbital localization.'
                    f'\n{h_diag}')

    mat = logm(u, real=True)
    if not numpy.isreal(mat).all():
        raise RuntimeError('Complex solutions are not supported for '
                           'differentiating the Boys localiztion.')
    x = pack_uniq_var(numpy.real(mat))
    return x


class BoysCurvatureError(RuntimeError):
    """A stationary frame may survive an inconclusive curvature check."""


def _lowest_curvature(loc, u):
    """Lowest Hessian eigenpair, normally using only Hessian actions."""
    from scipy.linalg import eigh
    from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

    gradient, action, diagonal = loc.gen_g_hop(u)
    size = gradient.size

    def dense_minimum():
        matrix = numpy.empty((size, size))
        unit = numpy.zeros(size)
        for column in range(size):
            unit[column] = 1.
            matrix[:, column] = action(unit)
            unit[column] = 0.
        return eigh(.5 * (matrix + matrix.T), subset_by_index=(0, 0))

    if size <= 2:
        values, vectors = dense_minimum()
        return float(values[0]), vectors[:, 0]
    # ARPACK's default 20-vector subspace stalls on the tightly clustered
    # negative modes of long chains. A larger subspace resolves that cluster.
    # A uniform shift also makes its relative convergence test meaningful for
    # near-zero modes. This changes neither eigenvectors nor their ordering;
    # the returned Rayleigh quotient and residual use the original Hessian.
    shift = 2 * max(1., float(numpy.max(numpy.abs(diagonal))))
    operator = LinearOperator((size, size),
                              matvec=lambda x: action(x) - shift * x, dtype=float)
    # A nonsymmetric starting vector can see symmetry-breaking directions.
    # A local RNG keeps the check repeatable without affecting user sampling.
    initial = numpy.random.default_rng(7301).normal(size=size)
    try:
        _, vectors = eigsh(operator, k=1, which='SA', tol=1e-8 / shift,
                           ncv=min(size, 64), v0=initial, maxiter=max(200, 10 * size))
    except ArpackNoConvergence:
        # Bound dense fallback storage (one 4096-square array is 128 MiB).
        # Never accept a missing or unconverged Ritz pair as a stable minimum.
        if size > 4096:
            raise
        _, vectors = dense_minimum()
    direction = vectors[:, 0]
    applied = action(direction)
    value = float(direction @ applied)
    residual = numpy.linalg.norm(applied - value * direction)
    if not numpy.isfinite(value) or not numpy.isfinite(residual) or residual > 1e-7:
        raise BoysCurvatureError(f'Boys lowest-curvature eigenpair residual is {residual:g}')
    return value, direction


def _stabilize_boys(loc, u, sorted_idx, *, max_saddle_escapes=3):
    """Try bounded saddle escapes, retaining the best stationary candidate."""
    from scipy.linalg import expm
    from scipy.sparse.linalg import ArpackError

    tolerance = loc.conv_tol_grad
    if tolerance is None:
        tolerance = numpy.sqrt(loc.conv_tol * .1)
    best = None
    reason = ('saddle escapes disabled for branch replay' if max_saddle_escapes == 0
              else 'three saddle escapes exhausted' if max_saddle_escapes == 3
              else f'{max_saddle_escapes} saddle escapes exhausted')
    for attempt in range(max_saddle_escapes + 1):
        value = loc.cost_function(u)
        gradient = numpy.linalg.norm(loc.get_grad(u))
        if not numpy.isfinite(value) or not numpy.isfinite(gradient):
            raise RuntimeError('Boys localization has nonfinite objective or gradient')
        if gradient <= tolerance and (best is None or value < best[0]):
            best = (value, u.copy(), numpy.array(sorted_idx, copy=True), gradient)
        try:
            curvature, direction = _lowest_curvature(loc, u)
        except (ArpackError, BoysCurvatureError) as error:
            reason = f'curvature check unavailable ({error})'
            break
        if curvature >= -1e-6:
            if gradient <= tolerance:
                return u, sorted_idx
            reason = 'optimizer stopped before satisfying the stationarity tolerance'
            break
        if attempt == max_saddle_escapes:
            break
        for step in (.15, .075, .0375, .01875):
            trials = [u @ expm(loc.unpack_uniq_var(sign * step * direction))
                      for sign in (1., -1.)]
            trial = min(trials, key=loc.cost_function)
            if loc.cost_function(trial) < value - 1e-12:
                break
        else:
            reason = 'negative-curvature trial steps did not lower the objective'
            break
        logger.info(loc, 'Restarting Boys localization along negative curvature %.6g', curvature)
        loc.init_guess = trial
        u, sorted_idx = kernel(loc, return_u=True)
    if best is None:
        raise RuntimeError(f'No sufficiently stationary Boys solution: {reason}; '
                           f'last gradient {gradient:.6g}, tolerance {tolerance:.6g}')
    warnings.warn(f'Boys localization: {reason}; retaining the best stationary solution '
                  f'(gradient {best[3]:.6g}, tolerance {tolerance:.6g}). '
                  'Negative curvature is allowed; the original response equation will be checked.',
                  RuntimeWarning, stacklevel=2)
    return best[1], best[2]


def _boys(x, mol, mo_coeff, *,
          init_guess=None,
          conv_tol=None, conv_tol_grad=None, max_cycle=None,
          max_saddle_escapes=3):
    if (not isinstance(max_saddle_escapes, (int, numpy.integer))
            or max_saddle_escapes < 0):
        raise ValueError('max_saddle_escapes must be a nonnegative integer')
    mo_coeff = numpy.array(mo_coeff, dtype=mo_coeff.dtype, order='C')
    loc = Boys(mol, mo_coeff=mo_coeff)
    if init_guess is not None:
        loc.init_guess = init_guess
    if conv_tol is not None:
        loc.conv_tol = conv_tol
    if conv_tol_grad is not None:
        loc.conv_tol_grad = conv_tol_grad
    if max_cycle is not None:
        loc.max_cycle = max_cycle

    # CIAH tests squared trial-vector norms and augmented-Hessian changes
    # against absolute cutoffs. Tight outer tolerances require smaller inner
    # cutoffs, or near-converged replay can discard every remaining step.
    grad_tol = (loc.conv_tol_grad if loc.conv_tol_grad is not None
                else numpy.sqrt(loc.conv_tol * .1))
    # Include the final Hessian-diagonal check in the same cache lifetime.
    with loc._cache_dipoles():
        # CIAH's discarded trial-vector norm is a rotation norm, not a
        # gradient norm. Long molecules can have Hessian entries near 1e4,
        # requiring correspondingly smaller rotations to finish convergence.
        hessian_scale = max(1., float(numpy.max(numpy.abs(loc.gen_g_hop()[2]))))
        inner_tol = .01 * (grad_tol / hessian_scale) ** 2
        loc.ah_conv_tol = min(loc.ah_conv_tol, inner_tol)
        loc.ah_lindep = min(loc.ah_lindep, inner_tol)
        u, sorted_idx = kernel(loc,
            mo_coeff=mo_coeff if init_guess is None else None, return_u=True)
        u, sorted_idx = _stabilize_boys(loc, u, sorted_idx,
                                      max_saddle_escapes=max_saddle_escapes)
        x = _extract_x0(loc, u)
    return x, sorted_idx

def boys(mol, mo_coeff, *,
         init_guess=None,
         conv_tol=None, conv_tol_grad=None, max_cycle=None,
         symmetry=False, gmres_options=None, max_saddle_escapes=3):
    '''
    Boys localization. See also `pyscf.lo.boys.Boys`.

    Arguments:
        max_saddle_escapes : int
            Maximum negative-curvature restarts after the initial solve.
            Default is three. Use zero when replaying a previously selected
            stationary branch: additional escapes could invalidate its saved
            orbital labels. Stationarity and response checks still apply.
        symmetry : bool
            For certain symmetric molecules, orbital localizations
            may have degenerate solutions. The orbital hessians are
            singular in such cases. Setting `symmetry=True` will
            declare these as gauge null modes and solve in the Hessian range.
            The original response equation must remain compatible; a right-
            hand side with a null component is rejected, never projected away
            silently. Default value is `False`.
        gmres_options : dict
            Options of the GMRES solver for computing the derivatives.
            Defaults are ``tol=1e-7`` and ``atol=1e-7`` for the original
            response residual; explicit caller options take precedence. See
            `pyscfad.tools.linear_solver.gen_gmres` for other options.

    Notes:
        The forward CIAH solve caches three projected dipole matrices and the
        rotation-invariant r2 trace. Every solve rebuilds these from its current
        molecule and input orbitals; only small orbital-space rotations are
        performed during optimization.

        By default at most three saddle escapes are attempted. If a minimum
        cannot be reached, the best sufficiently stationary candidate is
        retained with a warning. Negative curvature is allowed in the response.
        Failure to find any stationary candidate still raises.

        Every response checks the original unpreconditioned residual. Small
        concrete systems report a spectral condition number; large or traced
        systems report response amplification and a condition estimate or
        lower bound, which cannot certify the absence of other ill-conditioned
        modes.

        The response uses occupied-space dipoles and a compiled rotation
        gradient. The default explicitly selects the host GMRES adapter and
        its diagonal preconditioner, independently of the integral backend.
        Traced solves fall back to JAX without a host preconditioner.

        Set ``BOYS_RESPONSE_VARIANT=baseline`` to recover the original
        optimality condition, or ``projected`` for the unpreconditioned
        projected response. The default is ``projected_precond``.
    '''
    if mo_coeff.shape[-1] == 1:
        return mo_coeff
    gmres_options = {} if gmres_options is None else dict(gmres_options)
    gmres_options.setdefault('tol', 1e-7)
    gmres_options.setdefault('atol', 1e-7)

    # Retain the original response as an explicit diagnostic fallback.
    variant = os.environ.get('BOYS_RESPONSE_VARIANT', 'projected_precond')
    response_options = {}
    if variant == 'baseline':
        optimality_cond = _opt_cond
    elif variant == 'projected':
        optimality_cond = _projected_opt_cond
    elif variant == 'projected_precond':
        optimality_cond = _projected_opt_cond_aux
        response_options = dict(optimality_fun_has_aux=True,
                                gen_precond=_projected_preconditioner,
                                gen_precond_host_only=True)
    else:
        raise ValueError(f'Unknown BOYS_RESPONSE_VARIANT: {variant!r}')
    from pyscfad.lo._boys_response import checked_solver
    # The Boys wrapper checks gauge compatibility; the legacy generic safe
    # solver can silently discard incompatible right-hand-side components.
    if variant == 'projected_precond':
        # Boys uses compiled orbital-space actions, irrespective of whether
        # AO integrals use the optimized backend. Without this override the
        # usual STC driver selects unpreconditioned JAX GMRES; its success
        # status does not certify the requested true-residual tolerance.
        # The host adapter still falls back to JAX when its inputs are traced.
        with eager_gmres():
            solver = gen_gmres(safe=False, **gmres_options)
    else:
        solver = gen_gmres(safe=False, **gmres_options)
    solver = checked_solver(solver, symmetry=symmetry,
                            tol=gmres_options['tol'],
                            atol=gmres_options['atol'] or 0.,
                            maxiter=gmres_options.get('maxiter', 30) or 30,
                            restart=gmres_options.get('restart', 20) or 20,
                            log=logger.new_logger(mol))
    _boys_iter = make_implicit_diff(_boys,
                                    implicit_diff=True,
                                    fixed_point=False,
                                    optimality_cond=optimality_cond,
                                    solver=solver,
                                    has_aux=True,
                                    **response_options)

    x, sorted_idx = _boys_iter(None, mol, mo_coeff,
                               init_guess=init_guess,
                               conv_tol=conv_tol,
                               conv_tol_grad=conv_tol_grad,
                               max_cycle=max_cycle,
                               max_saddle_escapes=max_saddle_escapes)

    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])
