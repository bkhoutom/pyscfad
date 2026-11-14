"""Boys spreads plus a squared off-diagonal Fock penalty.

All ordered off-diagonal entries (i != j) are included. In atomic units,
``lambda_=100`` multiplies Eh**2 to give a spread contribution in bohr**2.
The AO Fock is an explicit differentiable argument, including for response.
"""
from contextlib import contextmanager
from functools import partial
import numpy as onp
import jax
from pyscf.lib import logger
from pyscfad import numpy as np
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.tools.linear_solver import gen_gmres, eager_gmres
from pyscfad.soscf.ciah import extract_rotation
from . import boys as base


def _weight(value):
    value = float(value)
    if not onp.isfinite(value) or value < 0:
        raise ValueError('lambda_ must be finite and nonnegative')
    return value


def _operators(mol, coeff, fock, lambda_):
    dipoles = base.dipole_integral(mol, coeff)
    projected = coeff.T.conj() @ fock @ coeff
    return np.concatenate((dipoles, np.sqrt(lambda_) * projected[None]), axis=0)


class LambdaBoys(base.Boys):
    """CIAH optimizer for sum(spreads) + lambda_ * sum(i != j, Fij**2)."""
    _keys = base.Boys._keys | {'lambda_', 'fock_ao'}

    def __init__(self, mol, mo_coeff, fock, lambda_=100.):
        super().__init__(mol, mo_coeff)
        self.lambda_ = _weight(lambda_)
        self.fock_ao = onp.asarray(fock)
        if (self.fock_ao.shape != (mo_coeff.shape[0], mo_coeff.shape[0])
                or not onp.isfinite(self.fock_ao).all()
                or onp.iscomplexobj(self.fock_ao) or onp.iscomplexobj(mo_coeff)
                or not onp.allclose(self.fock_ao, self.fock_ao.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('lambda-Boys requires real orbitals and a finite symmetric AO Fock')

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'lambda-Boys: lambda = %g; squared Fock entries summed over i != j', self.lambda_)
        return self

    @contextmanager
    def _cache_dipoles(self, mo_coeff=None):
        coeff = onp.asarray(self.mo_coeff if mo_coeff is None else mo_coeff)
        with super()._cache_dipoles(coeff):
            dipoles, spread_trace = self._boys_dipole_cache
            fock = coeff.T @ self.fock_ao @ coeff
            # Tr(F^2) is invariant under occupied rotations, exactly like
            # the sum of <r^2>. Only squared diagonal entries need gradients.
            self._boys_dipole_cache = (
                onp.concatenate((dipoles, onp.sqrt(self.lambda_) * fock[None])),
                spread_trace + self.lambda_ * onp.sum(fock*fock))
            yield

    def get_grad(self, u=None, dip=None):
        if dip is not None or getattr(self, '_boys_dipole_cache', None) is not None:
            return super().get_grad(u, dip)
        with self._cache_dipoles():
            return super().get_grad(u)

    def gen_g_hop(self, u=None):
        if getattr(self, '_boys_dipole_cache', None) is not None:
            return super().gen_g_hop(u)
        with self._cache_dipoles():
            return super().gen_g_hop(u)

    def cost_function(self, u=None):
        if getattr(self, '_boys_dipole_cache', None) is not None:
            return super().cost_function(u)
        with self._cache_dipoles():
            return super().cost_function(u)


def _solve(x, mol, coeff, fock, *, lambda_=100., init_guess=None,
           conv_tol=None, conv_tol_grad=None, max_cycle=None, max_saddle_escapes=3):
    return base._boys(x, mol, coeff, init_guess=init_guess, conv_tol=conv_tol,
                     conv_tol_grad=conv_tol_grad, max_cycle=max_cycle,
                     max_saddle_escapes=max_saddle_escapes,
                     _localizer_factory=partial(LambdaBoys, fock=fock, lambda_=lambda_))


def _condition(x, mol, coeff, fock, *, lambda_):
    operators = _operators(mol, coeff, fock, lambda_)
    return base._projected_gradient(x, operators), (x, operators)


def _preconditioner(aux, mol, coeff, fock, *, solver=None):
    return base._projected_preconditioner(aux, mol, coeff, solver=solver)


def _implicit(mol, numerical, lambda_, symmetry, gmres_options):
    from ._boys_response import checked_solver
    options = dict(gmres_options or {})
    options.setdefault('tol', 1e-7); options.setdefault('atol', 1e-7)
    with eager_gmres():
        solver = gen_gmres(safe=False, **options)
    solver = checked_solver(solver, symmetry=symmetry, tol=options['tol'],
                            atol=options['atol'] or 0.,
                            maxiter=options.get('maxiter', 30) or 30,
                            restart=options.get('restart', 20) or 20,
                            log=logger.new_logger(mol))
    return make_implicit_diff(partial(numerical, lambda_=lambda_), implicit_diff=True,
                              fixed_point=False, optimality_cond=partial(_condition, lambda_=lambda_),
                              solver=solver, has_aux=True, optimality_fun_has_aux=True,
                              gen_precond=_preconditioner, gen_precond_host_only=True)


def lambda_boys(mol, mo_coeff, fock, *, lambda_=100., init_guess=None,
                conv_tol=None, conv_tol_grad=None, max_cycle=None,
                symmetry=False, gmres_options=None, max_saddle_escapes=3):
    """Localize with a squared Fock penalty; AO Fock response is retained.

    ``lambda_=0`` recovers ordinary Boys. The default 100 uses bohr and Eh,
    and counts both Fij and Fji. The input Fock belongs to the current SCF
    reference and is held fixed during orbital rotations, not during AD.
    """
    weight = _weight(lambda_)
    if mo_coeff.shape[1] <= 1:
        return mo_coeff
    solve = _implicit(mol, _solve, weight, symmetry, gmres_options)
    x, order = solve(None, mol, mo_coeff, fock, init_guess=init_guess,
                     conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
                     max_cycle=max_cycle, max_saddle_escapes=max_saddle_escapes)
    return np.dot(mo_coeff, extract_rotation(x)[:, order])


def _stationary_root(x, mol, coeff, fock, *, lambda_=100., conv_tol=None, conv_tol_grad=None):
    size = coeff.shape[1]
    zero = np.zeros(size*(size-1)//2, dtype=coeff.dtype)
    gradient = float(np.linalg.norm(_condition(zero, mol, coeff, fock, lambda_=lambda_)[0]))
    tolerance = conv_tol_grad if conv_tol_grad is not None else onp.sqrt((1e-6 if conv_tol is None else conv_tol)*.1)
    if not onp.isfinite(gradient) or gradient > tolerance:
        raise base.BoysStationarityError(
            f'lambda-Boys frame is not stationary: gradient {gradient:g}, tolerance {tolerance:g}')
    return zero, onp.arange(size)


def at_stationary_frame(mol, mo_coeff, fock, *, lambda_=100., conv_tol=None,
                        conv_tol_grad=None, symmetry=False, gmres_options=None):
    """Attach the full lambda-Boys response without optimizing again."""
    weight = _weight(lambda_)
    if mo_coeff.shape[1] <= 1:
        return mo_coeff
    attach = _implicit(mol, _stationary_root, weight, symmetry, gmres_options)
    x, order = attach(None, mol, mo_coeff, fock, conv_tol=conv_tol, conv_tol_grad=conv_tol_grad)
    return np.dot(mo_coeff, extract_rotation(x)[:, order])
