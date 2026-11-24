"""Prepare coarse SCF/lambda guesses before solving on the final grid."""
from contextvars import ContextVar

import numpy as np
from pyscf import lib

_response_backend = ContextVar('sgx_response_grid_backend', default=None)


def response_backend(backend):
    active = _response_backend.get()
    return active[1] if active is not None and active[0] is backend else backend


def coarse_backend(backend):
    from .sgx import _Backend
    # Direct construction disables sequencing: exactly two stages, not a
    # recursive sequence through every smaller grid.
    return _Backend(backend.auxbasis, backend.grid_level - 1,
                    backend.fit_ovlp, backend.df_memory_mb,
                    backend.integral_backend, backend.accuracy)


def fresh_diis(mf, dm, s1e, h1e, vhf):
    if not mf.diis:
        return None
    diis = mf.DIIS(mf)
    diis.space = mf.diis_space
    diis.rollback = mf.diis_space_rollback
    diis.damp = mf.diis_damp
    _, diis.Corth = mf.eig(mf.get_fock(h1e, s1e, vhf, dm), s1e)
    return diis


def converge_coarse_scf(dm, mf, s1e, h1e, *, conv_tol, conv_tol_grad,
                        callback, log, cput1):
    from pyscfad.scf.hf import _scf
    backend = getattr(mf, '_sgx_backend', None)
    if (backend is None or not backend.grid_sequence or
            backend.grid_level == 0 or mf.max_cycle <= 0):
        return dm, False
    coarse = mf.copy()
    coarse._sgx_backend = coarse_backend(backend)
    vhf = coarse.get_veff(coarse.mol, dm, s1e=s1e)
    energy = coarse.energy_tot(dm, h1e, vhf)
    diis = fresh_diis(coarse, dm, s1e, h1e, vhf)
    factor = backend.coarse_conv_factor
    log.info('SGX SCF converging on coarse grid %d (tolerance factor %g)',
             backend.grid_level - 1, factor)
    result = _scf(dm, coarse, s1e, h1e, conv_tol=conv_tol * factor,
                  conv_tol_grad=conv_tol_grad * factor, diis=diis, callback=callback,
                  dump_chk=False, log=log, e_tot=energy, vhf=vhf, cput1=cput1)
    if not result[1]:
        raise RuntimeError(f'SGX coarse-grid SCF did not converge on grid {backend.grid_level - 1}')
    log.info('SGX coarse SCF converged; restarting on final grid %d with fresh DIIS',
             backend.grid_level)
    return result[0], True


def converge_coarse_lambda(backend, matvec, b, *, tol, atol, M, restart, maxiter):
    """Prepare a coarse warm start without a final true-residual check."""
    from pyscfad.sgx._coarse_lambda import coarse_lambda_guess
    if not backend.grid_sequence or backend.grid_level == 0 or not np.any(b):
        return None
    coarse = coarse_backend(backend)
    tol = tol * backend.coarse_conv_factor
    atol = None if atol is None else atol * backend.coarse_conv_factor
    token = _response_backend.set((backend, coarse))
    try:
        if backend._native is not None:
            lib.logger.info(backend._native.mol,
                            'SGX lambda warm start on coarse grid %d (rtol %g, atol %g)',
                            coarse.grid_level, tol, float(atol or 0.))
        # This call has no warm_start hook and no final-stage checkpoint
        # callback. The final-grid solver alone certifies convergence.
        value = coarse_lambda_guess(matvec, b, tol=tol, atol=atol, M=M,
                         restart=restart, maxiter=maxiter,
                         log=(lib.logger.new_logger(backend._native.mol)
                              if backend._native is not None else None))
        if coarse._native is not None:
            lib.logger.info(coarse._native.mol,
                            'SGX coarse lambda warm start ready; '
                            'restarting on final grid %d with fresh GMRES',
                            backend.grid_level)
        return value
    finally:
        _response_backend.reset(token)


class NativeGridSequence:
    """Apply the same converged stages to the native PySCF reference."""
    _keys = {'_sgx_grid_sequence', '_sgx_coarse_conv_factor'}

    def scf(self, dm0=None, **kwargs):
        from .sgx import _configure
        obj = self.with_df
        if not self._sgx_grid_sequence or obj.grids_level_f == 0 or self.max_cycle <= 0:
            return super().scf(dm0=dm0, **kwargs)
        coarse = self.copy()
        coarse._sgx_grid_sequence = False
        factor = self._sgx_coarse_conv_factor
        coarse.conv_tol = self.conv_tol * factor
        grad_tol = self.conv_tol_grad
        if grad_tol is None:
            grad_tol = np.sqrt(self.conv_tol)
        coarse.conv_tol_grad = grad_tol * factor
        coarse.with_df = _configure(self.mol, obj.auxbasis, obj.grids_level_f - 1,
                                   obj.fit_ovlp, obj._response_workspace_bytes / 1e6,
                                   obj._integral_backend, obj._sgx_accuracy)
        coarse.chkfile = None
        coarse.diis = bool(self.diis)
        lib.logger.info(self, 'SGX SCF converging on coarse grid %d', obj.grids_level_f - 1)
        coarse.kernel(dm0=dm0, **kwargs)
        if not coarse.converged:
            raise RuntimeError(f'SGX coarse-grid SCF did not converge on grid {obj.grids_level_f - 1}')
        lib.logger.info(self, 'SGX coarse SCF converged; restarting on final grid %d with fresh DIIS',
                        obj.grids_level_f)
        diis = self.diis
        try:
            # Even a caller-supplied DIIS instance must not carry history
            # into the restarted final stage.
            self.diis = bool(diis)
            return super().scf(dm0=coarse.make_rdm1(), **kwargs)
        finally:
            self.diis = diis
