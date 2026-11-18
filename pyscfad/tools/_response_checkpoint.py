"""Crash-safe SCF adjoints and completed force contractions.

An explicitly owned CC checkpoint scope enables these records. Other roots
(Boys, CC amplitudes, etc.) and ordinary non-checkpointed differentiation are
unaffected. Only numerical arrays are saved, never VJP closures or Krylov bases.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import numpy as np

_active = ContextVar('pyscfad_response_checkpoint', default=None)


def tag_solver(solver):
    tagged = partial(solver)
    tagged._scf_response_checkpoint = True
    return tagged


@contextmanager
def scope(manager):
    token = _active.set(manager if manager is not None and manager.enabled else None)
    try:
        yield
    finally:
        _active.reset(token)


def session(solver, *, traced=False):
    manager = _active.get()
    if manager is None or traced or not getattr(solver, '_scf_response_checkpoint', False):
        return None
    return LambdaCheckpoint(manager, solver)


def enabled():
    return _active.get() is not None


class LambdaCheckpoint:
    """Checkpoint host GMRES at restart-cycle boundaries and on convergence."""
    def __init__(self, manager, solver):
        from . import linear_solver
        self.manager = manager
        options = {}
        while isinstance(solver, partial):
            options = {**(solver.keywords or {}), **options}
            if solver.args:
                raise TypeError('checkpointed SCF solver cannot bind positional arguments')
            solver = solver.func
        if solver not in (linear_solver.jax_gmres, linear_solver.pyscfad_gmres):
            raise NotImplementedError('SCF lambda checkpoints require standard GMRES')
        options.pop('solve_method', None)
        if options.get('callback') is not None and options.get('callback_type') != 'x':
            raise ValueError('checkpointed SCF GMRES callbacks must accept iterates (callback_type=x)')
        self.options = options
        self.cycles = 0

    def _load(self, stage, b):
        from pyscfad.dlno_stc._restart import RestartCorruptionError
        record = self.manager.load_record(stage)
        if record is None:
            return None
        value = np.asarray(record.scalars['lambda'])
        if value.shape != b.shape or value.dtype != b.dtype or not np.isfinite(value).all():
            raise RestartCorruptionError('invalid saved SCF lambda shape, dtype or values')
        if record.scalars.get('schema') != 1:
            raise RestartCorruptionError('unsupported saved SCF lambda schema')
        self.cycles = int(record.scalars['cycles'])
        return value

    def threshold(self, b, kwargs):
        options = {**self.options, **kwargs}
        return max(float(options.get('atol') or 0.),
                   float(options.get('tol', 1e-5)) * np.linalg.norm(b))

    def prepare(self, matvec, b, kwargs):
        """Validate a completed solution; otherwise return the latest x0.

        Residual validation also handles harmless replay roundoff, changed
        RHS seeds, and tighter tolerances without trusting an old success flag.
        """
        self.b = np.asarray(b)
        candidate = self._load('scf-lambda-converged', self.b)
        if candidate is not None:
            residual = np.linalg.norm(np.asarray(matvec(candidate)) - self.b)
            if np.isfinite(residual) and residual <= self.threshold(self.b, kwargs):
                return candidate, True
        latest = self._load('scf-lambda-iterate', self.b)
        return (latest if latest is not None else candidate), False

    def _save(self, stage, value, **extra):
        value = np.asarray(value).reshape(self.b.shape).copy()
        if not np.isfinite(value).all():
            raise RuntimeError('refusing to checkpoint nonfinite SCF lambda')
        self.manager.save_record(stage, scalars=dict(
            schema=1, **{'lambda': value}, rhs=self.b, cycles=self.cycles, **extra))

    def solve(self, matvec, b, x0, **kwargs):
        from pyscfad._src.scipy.sparse.linalg import gmres
        options = {**self.options, **kwargs}
        original_callback = options.pop('callback', None)
        options.pop('callback_type', None)
        if x0 is not None:
            options['x0'] = x0

        def save_iterate(value):
            self.cycles += 1
            self._save('scf-lambda-iterate', value)
            if original_callback is not None:
                original_callback(value)

        value, info = gmres(matvec, b, callback=save_iterate, callback_type='x', **options)
        residual = np.linalg.norm(np.asarray(matvec(value)) - self.b)
        threshold = self.threshold(self.b, kwargs)
        if info != 0 or not np.isfinite(residual) or residual > threshold:
            self._save('scf-lambda-iterate', value)
            raise RuntimeError(f'SCF lambda residual {residual} exceeds tolerance {threshold}')
        # Commit before any external-argument VJP/force assembly starts.
        self._save('scf-lambda-converged', value,
                   residual_norm=float(residual), tolerance=float(threshold))
        return value


def contraction(name, inputs, compute):
    """Reuse an atomically committed numerical contraction with identical seeds.

    Reconstructed derivative seeds must match exactly. Changes cause a cache
    miss; scientific run identity and checksums are validated by the manager.
    """
    manager = _active.get()
    if manager is None:
        return compute(), False
    from pyscfad.dlno_stc._restart import scientific_digest
    identity = scientific_digest(inputs)
    stage = 'scf-force-' + name
    saved = manager.load_record(stage, key=identity)
    if saved is not None:
        return saved.scalars['value'], True
    value = compute()
    manager.save_record(stage, key=identity, scalars=dict(value=value))
    return value, False
