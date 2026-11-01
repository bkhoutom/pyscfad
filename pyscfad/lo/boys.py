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
from pyscfad.tools.linear_solver import gen_gmres, precond_by_hdiag
from pyscfad.scipy.linalg import logm

# modified from pyscf v2.6
def kernel(localizer, mo_coeff=None, callback=None, verbose=None,
           return_u=False):
    from pyscf.tools import mo_mapping
    from pyscf.soscf import ciah
    if mo_coeff is not None:
        localizer.mo_coeff = numpy.asarray(mo_coeff, order='C')
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
    log.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
# Sort the localized orbitals, to make each localized orbitals as close as
# possible to the corresponding input orbitals
    sorted_idx = mo_mapping.mo_1to1map(u0)
    if return_u:
        return u0, sorted_idx
    localizer.mo_coeff = numpy.dot(localizer.mo_coeff, u0[:,sorted_idx])
    return localizer.mo_coeff


class Boys(pyscf_boys.Boys):
    def get_init_guess(self, key='atomic'):
        if hasattr(key, 'shape'):
            u0 = numpy.asarray(key)
        else:
            u0 = pyscf_boys.Boys.get_init_guess(self, key=key)
        return u0

    kernel = kernel


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

def _boys(x, mol, mo_coeff, *,
          init_guess=None,
          conv_tol=None, conv_tol_grad=None, max_cycle=None):
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
    inner_tol = .01 * grad_tol ** 2
    loc.ah_conv_tol = min(loc.ah_conv_tol, inner_tol)
    loc.ah_lindep = min(loc.ah_lindep, inner_tol)

    if init_guess is None:
        u, sorted_idx = loc.kernel(mo_coeff=mo_coeff, return_u=True)
    else:
        u, sorted_idx = loc.kernel(mo_coeff=None, return_u=True)

    x = _extract_x0(loc, u)
    return x, sorted_idx

def boys(mol, mo_coeff, *,
         init_guess=None,
         conv_tol=None, conv_tol_grad=None, max_cycle=None,
         symmetry=False, gmres_options=None):
    '''
    Boys localization. See also `pyscf.lo.boys.Boys`.

    Arguments:
        symmetry : bool
            For certain symmetric molecules, orbital localizations
            may have degenerate solutions. The orbital hessians are
            singular in such cases. Setting `symmetry=True` will
            solve the linear equations in the range space when computing
            the derivatives. Default valule is `False`.
        gmres_options : dict
            Options of the GMRES solver for computing the derivatives.
            The default projected, preconditioned response uses ``tol=1e-10``;
            explicit caller options take precedence. See
            `pyscfad.tools.linear_solver.gen_gmres` for other options.

    Notes:
        The response uses occupied-space dipoles and a compiled rotation
        gradient. A diagonal preconditioner is applied to compatible host
        solvers; traced and JAX solves retain the exact projected response
        without a host preconditioner. The forward localization is unchanged.

        Set ``BOYS_RESPONSE_VARIANT=baseline`` to recover the original response
        and solver defaults, or ``projected`` for the unpreconditioned projected
        response. The default is ``projected_precond``.
    '''
    if mo_coeff.shape[-1] == 1:
        return mo_coeff
    gmres_options = {} if gmres_options is None else dict(gmres_options)

    # Retain the original response as an explicit diagnostic fallback.
    variant = os.environ.get('BOYS_RESPONSE_VARIANT', 'projected_precond')
    response_options = {}
    if variant == 'baseline':
        optimality_cond = _opt_cond
    elif variant == 'projected':
        optimality_cond = _projected_opt_cond
    elif variant == 'projected_precond':
        # The tighter default keeps adjoints insensitive to the different
        # preconditioned stopping history. Never override caller tolerances.
        gmres_options.setdefault('tol', 1e-10)
        optimality_cond = _projected_opt_cond_aux
        response_options = dict(optimality_fun_has_aux=True,
                                gen_precond=_projected_preconditioner,
                                gen_precond_host_only=True)
    else:
        raise ValueError(f'Unknown BOYS_RESPONSE_VARIANT: {variant!r}')
    solver = gen_gmres(safe=symmetry, **gmres_options)
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
                               max_cycle=max_cycle)

    u = extract_rotation(x)
    # pylint: disable=invalid-sequence-index
    return np.dot(mo_coeff, u[:,sorted_idx])
