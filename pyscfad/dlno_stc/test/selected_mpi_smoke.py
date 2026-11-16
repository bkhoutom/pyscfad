"""Domain/system replica smoke: mpiexec -n 2 python -m ...selected_mpi_smoke."""

from functools import partial

import numpy as np
from mpi4py import MPI

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.protocol import ARRAY_NAMES


def check_scope_dispatch(comm):
    from pyscfad import dlno_stc, gto, scf
    from pyscfad.dlno_stc.driver import _root_call

    rank = comm.Get_rank()

    mol = gto.Mole(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g", verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    from pyscfad.dlno_stc.test._boys_acceptance import build_mf
    mf = (build_mf(mol) if rank == 0 else scf.RHF(mol).density_fit())
    controls = {"laplace_roots": [0.1], "laplace_weights": [1.0]}
    for parallel_scf_jk in (False, True):
        for frozen in (0, 1):
            options = dict(scope="system" if rank == 0 else "domain",
                           frozen=frozen if rank == 0 else None,
                           controls=controls if rank == 0 else None,
                           comm=comm, parallel_scf_jk=parallel_scf_jk)
            result = dlno_stc.kernel(mf, **options)
            response = dlno_stc.value_and_grad(mol, build_mf, **options)
            if rank == 0:
                expected = dlno_stc.value_and_grad(
                    mol, build_mf, scope="system", frozen=frozen, controls=controls)
                np.testing.assert_allclose(result, expected[0], atol=1e-10, rtol=0)
                np.testing.assert_allclose(response[0], expected[0], atol=1e-10, rtol=0)
                np.testing.assert_allclose(response[1].coords, expected[1].coords,
                                           atol=1e-8, rtol=0)
                if frozen:
                    assert result == 0.0
            else:
                assert result is response is None
    try:
        dlno_stc.kernel(None, scope="invalid" if rank == 0 else "system",
                        controls={} if rank == 0 else None, comm=comm)
    except (ValueError, RuntimeError) as exc:
        assert "scope" in str(exc)
    else:
        raise AssertionError("invalid root scope did not reach every rank")


def main():
    check_scope_dispatch(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Domain/system MPI smoke PASS", flush=True)


if __name__ == "__main__":
    main()
