"""Run with mpiexec -n 3 to check one global baseline and an idle CC owner."""
import json
import argparse
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from pyscfad import gto, scf
from pyscfad.dlno_stc import build_domain_selections, ccsd, _global_mp2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic', action='store_true')
    args = parser.parse_args()
    baseline = (dict(mode='stochastic', global_seed=19, system_workload_cutoff=.2,
                     auxiliary_group_size=4, pilot_samples=32, min_production_samples=32,
                     production_samples=2000) if args.stochastic else None)
    comm = MPI.COMM_WORLD
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .75; H 0 0 6; H 0 0 6.75',
                   basis='sto-3g', verbose=0, max_memory=6000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    selected = None
    if comm.rank == 0:
        mf.conv_tol = 1e-12
        mf.kernel()
        domains = build_domain_selections(mf, frozen=0, mp2_laplace_points=12)
        assert not np.all(domains.strong_mask)
        selected = ccsd.build_static_selections(mf, domains,
            thresh_occ=1e-4, thresh_vir=1e-5, matrix_kind='V', baseline_kind='global',
            baseline_controls=baseline)
    calls = []
    original = _global_mp2.accumulate

    def accumulate(*args, **kwargs):
        calls.append(comm.rank)
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('MPI global baseline evaluated weak completion')

    with patch.object(_global_mp2, 'accumulate', accumulate), patch.object(ccsd, '_run_weak', forbidden):
        result = ccsd.kernel(mf, selected, comm=comm, parallel_scf_jk=False,
                             return_components=True, mp2_aux_svd_tol=None,
                             baseline_controls=baseline)
    all_calls = comm.gather(calls, root=0)
    if comm.rank == 0:
        assert all_calls == [[0]] + [[] for _ in range(comm.size - 1)]
        energy, parts = result
        serial, serial_parts = ccsd.kernel(mf, selected, return_components=True, mp2_aux_svd_tol=None,
                                           baseline_controls=baseline)
        np.testing.assert_allclose(energy, serial, atol=2e-11, rtol=0)
        np.testing.assert_allclose(parts.e_mp2_base, serial_parts.e_mp2_base, atol=2e-13)
        assert parts.baseline_standard_error == serial_parts.baseline_standard_error
        if args.stochastic:
            assert parts.baseline_standard_error > 0.
        assert parts.e_mp2_weak == 0. and len(parts.fragments) == 2
        print(json.dumps(dict(energy=float(energy), baseline=parts.e_mp2_base,
                              weak=parts.e_mp2_weak, baseline_calls=all_calls,
                              baseline_standard_error=parts.baseline_standard_error)))


if __name__ == '__main__':
    main()
