"""Finite selected-target STC MP2 for a water tetramer geometry."""
import argparse
from pathlib import Path
import json
import time
import numpy as np
from pyscfad import gto, scf, dlno_stc
from pyscfad.dlno_stc.controls import default_laplace_grid


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
    args = parser.parse_args()
    mol = gto.Mole(atom=str(args.xyz_file), basis=args.basis, verbose=3)
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-11
        mf.kernel()
        return mf

    started = time.perf_counter()
    mf = build_mf(mol)
    static = dlno_stc.build_domain_selections(mf, frozen=args.frozen,
        thresholds=dlno_stc.BoysDomainThresholds(pair_energy=args.pair_energy))
    roots, weights = default_laplace_grid()
    controls = dict(mode=args.mode, laplace_roots=roots, laplace_weights=weights,
                    energy_tolerance=args.energy_tolerance, global_seed=args.seed)
    if args.production_samples is not None:
        controls['production_samples'] = args.production_samples
    from pyscfad.dlno_stc.driver import kernel, value_and_grad
    if args.energy_only:
        energy = kernel(mf, static, scope='domain', controls=controls,
                        merge_identical=not args.unmerged)
        gradient = None
    else:
        energy, gradient = value_and_grad(mol, build_mf, static, scope='domain',
            controls=controls, include_hf=False, merge_identical=not args.unmerged)
    groups = dlno_stc.build_domain_groups(static, merge_identical=not args.unmerged)
    print(json.dumps(dict(scope='domain', correlation_energy=float(energy),
        elapsed_seconds=time.perf_counter()-started, original_targets=len(static.fragments),
        group_sizes=[len(g.target_indices) for g in groups],
        gradient=None if gradient is None else np.asarray(gradient.coords).tolist()), indent=2))


if __name__ == '__main__':
    main()
