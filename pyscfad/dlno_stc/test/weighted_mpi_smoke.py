"""Domain/system replica smoke: mpiexec -n 2 python -m ...weighted_mpi_smoke."""

from functools import partial

import numpy as np
from mpi4py import MPI

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.parallel import run_backend
from pyscfad.dlno_stc.protocol import ARRAY_NAMES, WEIGHTED_ARRAY_NAMES


def packet():
    return {
        "foo": np.array([[-0.8, 0.07], [0.07, -0.5]]),
        "fvv": np.array([[0.3, -0.02], [-0.02, 0.7]]),
        "B": np.arange(16, dtype=np.float64).reshape(4, 2, 2) / 30,
        "target_projection": np.array([[0.8, 0.3]]),
        "partner_weight": np.array([[0.9, 0.1], [0.1, 0.7]]),
    }


def check_team(comm, scope="domain"):
    rank, size = comm.Get_rank(), comm.Get_size()
    names = WEIGHTED_ARRAY_NAMES if scope == "domain" else ARRAY_NAMES
    inputs = {key: packet()[key] for key in names} if rank == 0 else None
    solve = backend.solve if scope == "domain" else partial(backend.solve, scope="system")
    metadata = {"fragment_id": 3} if rank == 0 else None
    controls = {"mode": "deterministic", "global_seed": 17,
                "laplace_roots": [0.1, 0.8], "laplace_weights": [0.2, 0.7]}
    result = run_backend(inputs, metadata, controls if rank == 0 else None,
                         solve, comm=comm, array_names=names)
    if rank == 0:
        serial = solve(inputs, {"fragment_id": 3}, controls)
        np.testing.assert_allclose(result["energy"], serial["energy"], atol=1e-14)
        for key in names:
            np.testing.assert_allclose(result["cotangents"][key],
                                       serial["cotangents"][key], atol=1e-14)
        assert result["energy_standard_error"] == 0
        assert result["metadata"]["replica_count"] == size
        assert len(set(result["diagnostics"]["replica_seeds"])) == size
    else:
        assert result is None

    # Capture each genuine stochastic solve before reduction mutates its
    # buffers, and compare with the arithmetic mean of those same replicas.
    original = backend._solve_local
    captured = None

    def capture_replica(*args, **kwargs):
        nonlocal captured
        local = original(*args, **kwargs)
        captured = {"energy": float(local["energy"]),
                    "sigma": float(local["energy_standard_error"]),
                    "bars": {key: value.copy() for key, value in local["cotangents"].items()}}
        return local

    sampled = dict(controls, mode="stochastic", virtual_keep_fraction=0.5,
                   production_samples=97, auxiliary_group_size=2)
    backend._solve_local = capture_replica
    try:
        result = run_backend(inputs, metadata, sampled if rank == 0 else None,
                             solve, comm=comm, array_names=names)
        replicas = comm.gather(captured, root=0)
        if rank == 0:
            np.testing.assert_allclose(result["energy"],
                                       np.mean([local["energy"] for local in replicas]),
                                       atol=1e-14, rtol=1e-14)
            sigma = np.sqrt(sum(local["sigma"] ** 2 for local in replicas)) / size
            np.testing.assert_allclose(result["energy_standard_error"], sigma)
            for key in names:
                expected = np.mean([local["bars"][key] for local in replicas], axis=0)
                np.testing.assert_allclose(result["cotangents"][key], expected,
                                           atol=1e-14, rtol=1e-14)
    finally:
        backend._solve_local = original

    # Different sample counts test complete-replica means rather than pooling.

    def known_replica(x, md, settings, *, with_grad, aux_offsets, **kwargs):
        result = original(x, md, settings, with_grad=with_grad, aux_offsets=aux_offsets, **kwargs)
        result["energy"] = np.float64(2 + 4 * rank)
        result["energy_standard_error"] = np.float64(3 + rank)
        result["cotangents"] = {key: np.full_like(x[key], 1 + 2 * rank)
                                for key in names}
        result["diagnostics"]["actual_sample_count"] = 5 + 9 * rank
        return result

    backend._solve_local = known_replica
    try:
        result = run_backend(inputs, metadata, controls if rank == 0 else None,
                             solve, comm=comm, array_names=names)
        if rank == 0:
            np.testing.assert_allclose(result["energy"], 2 + 2 * (size - 1))
            sigma = np.sqrt(sum((3 + r) ** 2 for r in range(size))) / size
            np.testing.assert_allclose(result["energy_standard_error"], sigma)
            for bar in result["cotangents"].values():
                np.testing.assert_allclose(bar, size)
    finally:
        backend._solve_local = original

    def fail_worker(*args, **kwargs):
        if rank == size - 1:
            raise RuntimeError("deliberate worker numerical failure")
        return original(*args, **kwargs)

    backend._solve_local = fail_worker
    try:
        try:
            run_backend(inputs, metadata, controls if rank == 0 else None,
                        solve, comm=comm, array_names=names)
        except RuntimeError as exc:
            assert "worker numerical failure" in str(exc)
        else:
            raise AssertionError("worker error did not reach every rank")
    finally:
        backend._solve_local = original
    comm.Barrier()

    from pyscfad.dlno_stc.driver import _root_call

    for phase in ("initial preparation", "domain pullback", "final response"):
        def fail_root():
            raise RuntimeError(f"deliberate {phase} failure")

        try:
            _root_call(comm, rank, fail_root, phase)
        except RuntimeError as exc:
            assert phase in str(exc)
        else:
            raise AssertionError("root error did not reach every rank")
    comm.Barrier()


def check_scope_dispatch(comm):
    from pyscfad import dlno_stc, gto, scf
    from pyscfad.dlno_stc.driver import _root_call

    rank = comm.Get_rank()

    def prepare():
        mol = gto.Mole(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g", verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        return scf.RHF(mol).density_fit().run()

    mf = _root_call(comm, rank, prepare, "scope fixture")
    result = dlno_stc.kernel(
        mf, scope="system" if rank == 0 else "domain", frozen=1 if rank == 0 else None,
        controls={"laplace_roots": [0.1], "laplace_weights": [1.0]} if rank == 0 else None,
        comm=comm,
    )
    assert (result == 0.0) if rank == 0 else (result is None)
    try:
        dlno_stc.kernel(None, scope="invalid" if rank == 0 else "system",
                        controls={} if rank == 0 else None, comm=comm)
    except (ValueError, RuntimeError) as exc:
        assert "scope" in str(exc)
    else:
        raise AssertionError("invalid root scope did not reach every rank")


def main():
    check_team(MPI.COMM_WORLD)
    check_team(MPI.COMM_WORLD, scope="system")
    check_scope_dispatch(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Domain/system MPI smoke PASS", flush=True)


if __name__ == "__main__":
    main()
