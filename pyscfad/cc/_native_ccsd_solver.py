# Copyright 2026 The PySCFAD Authors (Apache-2.0).
"""Dense RCCSD convergence, DIIS and fused energy; no amplitude packing.

Iteration ordering and DIIS coefficients follow PySCF (Apache-2.0,
Copyright 2014-2021 The PySCF Developers). Only real resident solves opt in.
"""
import numpy as np
import scipy.linalg
from contextlib import contextmanager
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from ._native_kernels import _ccsd_kernels as cpp


def energy(mycc, t1=None, t2=None, eris=None):
    t1 = mycc.t1 if t1 is None else t1
    t2 = mycc.t2 if t2 is None else t2
    if (cpp is not None and hasattr(cpp, 'ccsd_energy')
            and getattr(mycc, 'cpp_kernels', False) and eris is not None
            and all(isinstance(x, np.ndarray) and x.dtype == np.float64
                    for x in (t1, t2, eris.fock, eris.ovvo))):
        return cpp.ccsd_energy(t1, t2, eris.fock, eris.ovvo)
    return ccsd.energy(mycc, t1, t2, eris)


class DenseDIIS:
    """PySCF's implicit-error DIIS with dense amplitude/error histories.

    The error is the trial minus the previous extrapolated output. Weighted
    dot products preserve the independent-element metric of packed RCCSD.
    Public updates own snapshots; the solver can transfer fresh trial arrays.
    """
    def __init__(self, space):
        if space < 2:
            raise ValueError('dense DIIS requires a history space of at least two')
        self.space = int(space)
        self.head = self.count = 0
        self.previous = None
        self.vectors = [None]*self.space
        self.errors = [None]*self.space
        self.h = np.zeros((self.space+1, self.space+1))
        self.h[0, 1:] = self.h[1:, 0] = 1.
        self.coefficients = None

    @staticmethod
    def combine(coefficients, vectors):
        return cpp.dense_combine(np.asarray(coefficients, dtype=np.float64),
                                 [x[0] for x in vectors], [x[1] for x in vectors])

    def update(self, t1, t2, *, take_ownership=False):
        trial = (t1, t2)
        if not take_ownership:
            trial = self.combine([1.], [trial])
        if self.previous is None:
            self.previous = trial
            return (t1, t2)
        slot = self.head % self.space
        # Drop overwritten history before allocating its replacement.
        self.vectors[slot] = self.errors[slot] = None
        error = self.combine([1., -1.], [trial, self.previous])
        self.vectors[slot], self.errors[slot] = trial, error
        self.count = min(self.count+1, self.space)
        for j in range(self.count):
            value = cpp.weighted_dot(*error, *self.errors[j])
            self.h[slot+1, j+1] = self.h[j+1, slot+1] = value
        self.head = slot+1
        h = self.h[:self.count+1, :self.count+1]
        g = np.zeros(self.count+1)
        g[0] = 1.
        w, v = scipy.linalg.eigh(h)
        if np.any(abs(w) < 1e-14):
            keep = abs(w) > 1e-14
            c = (v[:, keep]/w[keep]) @ (v[:, keep].T @ g)
        else:
            c = np.linalg.solve(h, g)
        self.coefficients = c[1:]
        self.previous = None
        result = self.combine(c[1:], self.vectors[:self.count])
        self.previous = result
        return result


def _supported(mycc, t1, t2, eris):
    if (cpp is None or not hasattr(cpp, 'difference_norm')
            or not getattr(mycc, 'cpp_kernels', False)
            or mycc.cc2 or mycc.direct or getattr(mycc, 'dcsd', False)
            or not mycc.incore_complete or mycc.diis_file is not None
            or type(mycc.diis) is not bool
            or (mycc.diis and mycc.diis_space < 2)
            or t1 is None or t2 is None or eris is None
            or any(not isinstance(x, np.ndarray) or x.dtype != np.float64 for x in (t1, t2))):
        return False
    # The caller's initial amplitudes can remain live after current amplitudes
    # change. Reserve that growth and future DIIS slots, not another full set
    # of residual buffers in addition to the forward planner's phase peaks.
    from ._native_ccsd_forward import _native_plan
    history = 2*mycc.diis_space if mycc.diis else 0
    return _native_plan(mycc, t1, t2, eris,
        reserve_bytes=(history+1)*(t1.nbytes+t2.nbytes)) is not None


@contextmanager
def _memory_reservation(mycc, extra):
    sentinel = object()
    previous = getattr(mycc, '_native_ccsd_future_bytes', sentinel)
    mycc._native_ccsd_future_bytes = extra
    try:
        yield
    finally:
        if previous is sentinel:
            del mycc._native_ccsd_future_bytes
        else:
            mycc._native_ccsd_future_bytes = previous


def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None, callback=None):
    """Preserve PySCF convergence/DIIS ordering with direct tensor operations."""
    if not _supported(mycc, t1, t2, eris):
        return ccsd.kernel(mycc, eris, t1, t2, max_cycle, tol, tolnormt, verbose, callback)
    log = logger.new_logger(mycc, verbose)
    name = mycc.__class__.__name__
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)
    adiis = DenseDIIS(mycc.diis_space) if mycc.diis else None
    converged = False
    mycc.cycles = 0
    for istep in range(max_cycle):
        unfilled = 2*(adiis.space-adiis.count) if adiis is not None else 0
        future = (unfilled+int(istep == 0))*(t1.nbytes+t2.nbytes)
        with _memory_reservation(mycc, future):
            t1new, t2new = mycc.update_amps(t1, t2, eris)
        if callback is not None:
            callback(locals())
        normt = cpp.difference_norm(t1new, t2new, t1, t2)
        if mycc.iterative_damping < 1.:
            alpha = float(mycc.iterative_damping)
            t1new, t2new = DenseDIIS.combine([1-alpha, alpha], [(t1,t2), (t1new,t2new)])
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if (adiis is not None and istep >= mycc.diis_start_cycle
                and abs(eccsd-eold) < mycc.diis_start_energy_diff):
            t1, t2 = adiis.update(t1, t2, take_ownership=True)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        mycc.cycles = istep+1
        log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, name, eccsd, eccsd-eold, normt)
        cput1 = log.timer(f'{name} iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            converged = True
            break
    log.timer(name, *cput0)
    return converged, eccsd, t1, t2
