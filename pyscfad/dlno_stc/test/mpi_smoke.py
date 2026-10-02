"""Real-MPI smoke test: mpiexec -n 2 python -m pyscfad.dlno_stc.test.mpi_smoke."""

import numpy as np
import tempfile
from pathlib import Path
from mpi4py import MPI

from pyscfad.dlno_stc.parallel import run_backend
from pyscfad.dlno_stc.exchange import write_input, read_result
from pyscfad.dlno_stc.mp2 import solve_request
from pyscfad.dlno_stc.test.reference import sampled_backend


def packet():
    return {"foo": np.array([[-0.8, 0.07], [0.07, -0.5]]),
            "fvv": np.array([[0.3, -0.02], [-0.02, 0.7]]),
            "B": np.arange(16, dtype=np.float64).reshape(4, 2, 2) / 30}


def metadata():
    return {"schema_version": 1, "method": "mp2",
            "energy_kind": "quadratic_test_v1", "fragment_id": 0,
            "target_index": 0, "basis_frame": "orthonormal_local",
            "B_axes": "Pia", "request_fingerprint": "b" * 64,
            "units": {"energy": "Eh", "length": "bohr"}, "frozen": [0],
            "orbital_order": {"occupied": [0, 1], "virtual": [0, 1]},
            "auxiliary_order": "test-aux", "checkpoint_id": "test-checkpoint",
            "domain_options": {"full_support": True}, "code_revision": "test",
            "dirty_worktree": False}


def check_team(comm):
    rank, size = comm.Get_rank(), comm.Get_size()
    x = packet() if rank == 0 else None
    md = metadata() if rank == 0 else None
    for nblocks in (1, size + 1, 7):
        controls = {"global_seed": 17, "sample_blocks": nblocks} if rank == 0 else None
        result = run_backend(x, md, controls, sampled_backend, comm=comm)
        if rank == 0:
            baseline = run_backend(packet(), metadata(),
                                   {"global_seed": 17, "sample_blocks": nblocks},
                                   sampled_backend)
            assert result["metadata"]["actual_sample_count"] == nblocks
            np.testing.assert_allclose(result["energy"], baseline["energy"],
                                       rtol=1e-12, atol=1e-12)
            for key in ("foo", "fvv", "B"):
                np.testing.assert_allclose(result["cotangents"][key],
                                           baseline["cotangents"][key],
                                           rtol=1e-12, atol=1e-12)
        else:
            assert result is None
    # A worker-specific invalid backend must fail on every rank before
    # root enters the sampled backend's reductions.
    if size > 1:
        try:
            run_backend(x, md, {"global_seed": 17, "sample_blocks": 3},
                        sampled_backend if rank == 0 else None, comm=comm)
        except ValueError as exc:
            assert "backend" in str(exc)
        else:
            raise AssertionError("worker-only invalid backend did not reach root")

    for bad_inputs, bad_controls in ((None, {"sample_blocks": 3}),
                                     (x, {"sample_blocks": 0})):
        try:
            run_backend(bad_inputs if rank == 0 else None, md if rank == 0 else None,
                        bad_controls if rank == 0 else None,
                        sampled_backend, comm=comm)
        except ValueError as exc:
            assert "collective preflight failed" in str(exc)
        else:
            raise AssertionError("preflight failure did not reach every rank")


def main():
    if MPI.Query_thread() < MPI.THREAD_FUNNELED:
        raise RuntimeError("MPI_THREAD_FUNNELED support is required")
    world = MPI.COMM_WORLD
    check_team(world)
    if world.Get_size() >= 4:
        subcomm = world.Split(world.Get_rank() % 2, world.Get_rank())
        try:
            check_team(subcomm)
        finally:
            subcomm.Free()
    # Only the team root reads and writes the request/result files.
    temporary = tempfile.TemporaryDirectory() if world.Get_rank() == 0 else None
    directory = world.bcast(temporary.name if temporary is not None else None, root=0)
    request = Path(directory) / "input.h5"
    result_path = Path(directory) / "result.h5"
    controls = {"global_seed": 17, "sample_blocks": 3}
    if world.Get_rank() == 0:
        write_input(request, packet(), metadata(), controls)
    solved = solve_request(request, result_path, sampled_backend, comm=world)
    if world.Get_rank() == 0:
        assert solved is not None
        saved = read_result(result_path, packet(), metadata() | {
            "request_fingerprint": solved["metadata"]["request_fingerprint"]}, controls=controls)
        np.testing.assert_allclose(saved["energy"], solved["energy"])
    else:
        assert solved is None
    world.Barrier()
    if world.Get_rank() == 0:
        temporary.cleanup()

    # Root read errors must reach workers before any backend collective.
    try:
        solve_request(Path(directory) / "missing.h5", result_path, sampled_backend,
                      comm=world)
    except Exception:
        pass
    else:
        raise AssertionError("missing root input did not fail on every rank")
    world.Barrier()
    if world.Get_rank() == 0:
        print(f"MPI smoke PASS ({world.Get_size()} ranks)", flush=True)


if __name__ == "__main__":
    main()
