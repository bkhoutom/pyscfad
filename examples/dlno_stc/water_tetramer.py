"""Finite selected-target STC MP2 for a water tetramer geometry."""
import argparse
from pathlib import Path
import json
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz-file', type=Path, required=True)
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--auxbasis', default=None)
    parser.add_argument('--frozen', type=int, default=4)
    parser.add_argument('--energy-only', action='store_true')
    parser.add_argument('--mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--energy-tolerance', type=float, default=.0003)
    parser.add_argument('--production-samples', type=int)
    parser.add_argument('--seed', type=int, default=713)
    parser.add_argument('--pair-energy', type=float, default=1e-4)
    parser.add_argument('--unmerged', action='store_true')
    parser.add_argument('--mpi', action='store_true', help='distribute domains over MPI ranks')
    parser.add_argument('--parallel-scf-jk', action=argparse.BooleanOptionalAction,
                        default=True, help='distribute DF SCF/common work for multi-rank domain MPI')
    parser.add_argument('--output', type=Path, help='write root JSON results to this file')
    args = parser.parse_args()
    # Set OpenMP/BLAS/JAX thread limits in the launcher before these imports.
    import jax
    import numpy as np
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import backend
    from pyscfad.dlno_stc.controls import default_laplace_grid
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    else:
        comm = None
    rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    mol = gto.Mole(atom=str(args.xyz_file), basis=args.basis, verbose=3 if rank == 0 else 0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    def build_mf(current, *, mo_coeff_init=None, mo_energy_init=None,
                 mo_occ_init=None, e_tot_init=None):
        mf = scf.RHF(current).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-11
        mf.chkfile = None
        if mo_coeff_init is not None:
            mf.mo_coeff, mf.mo_energy = mo_coeff_init, mo_energy_init
            mf.mo_occ, mf.e_tot = mo_occ_init, e_tot_init
            mf.converged = True
            return mf
        mf.kernel()
        return mf

    static = None

    def build_static(mf):
        nonlocal static
        # The gradient path selects domains from the detached single-SCF
        # reference supplied by value_and_grad, outside differentiation.
        static = dlno_stc.build_domain_selections(mf, frozen=args.frozen,
            thresholds=dlno_stc.BoysDomainThresholds(pair_energy=args.pair_energy))
        return static

    started = time.perf_counter()
    roots, weights = default_laplace_grid()
    controls = dict(mode=args.mode, laplace_roots=roots, laplace_weights=weights,
                    energy_tolerance=args.energy_tolerance, global_seed=args.seed)
    if args.production_samples is not None:
        controls['production_samples'] = args.production_samples
    from pyscfad.dlno_stc.driver import kernel, value_and_grad
    calls = []
    solve = backend.solve

    def observed_solve(inputs, metadata, settings, **kwargs):
        before = time.perf_counter()
        result = solve(inputs, metadata, settings, **kwargs)
        calls.append(dict(rank=rank, fragment_id=metadata['fragment_id'],
            shapes={key: list(value.shape) for key, value in inputs.items()},
            elapsed_seconds=time.perf_counter()-before,
            energy=float(result['energy']), diagnostics=result['diagnostics']))
        return result

    with patch.object(backend, 'solve', observed_solve):
        if args.energy_only:
            error = None
            mf = scf.RHF(mol).density_fit(auxbasis=args.auxbasis)
            if rank == 0:
                try:
                    mf = build_mf(mol)
                    static = build_static(mf)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            if comm is not None:
                error = comm.bcast(error, root=0)
            if error is not None:
                raise RuntimeError(error)
            result = kernel(mf, static, scope='domain', controls=controls if rank == 0 else None,
                            comm=comm, merge_identical=not args.unmerged,
                            parallel_scf_jk=args.parallel_scf_jk)
        else:
            result = value_and_grad(mol, build_mf,
                build_static=build_static if rank == 0 else None, scope='domain',
                controls=controls if rank == 0 else None, comm=comm, include_hf=False,
                merge_identical=not args.unmerged, parallel_scf_jk=args.parallel_scf_jk)
        jax.block_until_ready(result)
    elapsed = time.perf_counter()-started
    observations = [calls] if comm is None else comm.gather(calls, root=0)
    if rank != 0:
        return
    energy, gradient = (result, None) if args.energy_only else result
    groups = dlno_stc.build_domain_groups(static, merge_identical=not args.unmerged)
    summary = dict(scope='domain', correlation_energy=float(energy), mpi_size=size,
        parallel_scf_jk=args.parallel_scf_jk and size > 1,
        elapsed_seconds=elapsed, original_targets=len(static.fragments),
        group_sizes=[len(g.target_indices) for g in groups],
        calls=sorted((call for rows in observations for call in rows),
                     key=lambda call: call['fragment_id']),
        gradient=None if gradient is None else np.asarray(gradient.coords).tolist())
    serialized = json.dumps(summary, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import sys
        if '--mpi' in sys.argv:
            import traceback
            from mpi4py import MPI
            traceback.print_exc()
            MPI.COMM_WORLD.Abort(1)
        raise
