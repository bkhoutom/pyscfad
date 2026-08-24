"""MPI boundary for IAO-DLNO-CCSD(T).

MPI CC must root-build and replay every gauge-bearing IAO-MP2-selected LIS
frame, just as
``iao_mp2_mpi`` does for MP2 ED/screen frames.  Until that bounded-memory
frame protocol is implemented, multi-rank CC is rejected rather than
silently adding cotangents expressed in inconsistent gauges.
"""

from mpi4py import MPI

from .ccsd import DLNOCCSD as _SerialDLNOCCSD


__all__ = ["DLNOCCSD"]


class DLNOCCSD(_SerialDLNOCCSD):
    """COMM_SELF-compatible wrapper; multi-rank IAO-LIS CC is forthcoming."""

    @classmethod
    def value_and_grad(cls, *args, comm=None, root=0, **kwargs):
        if comm is None:
            comm = MPI.COMM_WORLD
        if comm.Get_size() != 1:
            raise NotImplementedError(
                "MPI IAO-DLNO-CCSD(T) is not yet enabled: its LIS frames "
                "must be built on one root gauge and streamed to workers. "
                "The gauge-safe MPI IAO-DLNO-MP2 energy and gradient are "
                "available in pyscfad.dlno.iao_mp2_mpi."
            )
        if int(root) != comm.Get_rank():
            raise ValueError("root must identify the sole COMM_SELF rank")
        return _SerialDLNOCCSD.value_and_grad(*args, **kwargs)

    def kernel(self, *args, comm=None, root=0, **kwargs):
        if comm is None:
            comm = MPI.COMM_WORLD
        if comm.Get_size() != 1:
            raise NotImplementedError(
                "MPI IAO-DLNO-CCSD(T) requires the pending root-gauge LIS "
                "streaming implementation"
            )
        if int(root) != comm.Get_rank():
            raise ValueError("root must identify the sole COMM_SELF rank")
        return super().kernel(*args, **kwargs)
