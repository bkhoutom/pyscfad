"""Expose live final SCF matrices to an explicitly owned outer VJP.

The capture lives only during SCF construction. Tracers are never installed
as auxiliary metadata on an SCF pytree or in a process-wide matrix cache.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_capture = ContextVar('pyscfad_final_scf_data', default=None)


@contextmanager
def capture():
    records = {}
    token = _capture.set(records)
    try:
        yield records
    finally:
        records.clear()
        _capture.reset(token)


def record(mf, overlap, fock):
    records = _capture.get()
    if records is not None:
        records[id(mf)] = (mf, (overlap, fock))
