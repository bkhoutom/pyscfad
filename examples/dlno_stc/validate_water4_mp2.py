"""Compare domain MP2 direct/half/replay gradients and a fixed-selection FD.

Set OpenMP/BLAS thread limits before launching Python. Example::

    python validate_water4_mp2.py --xyz-file water4.xyz --output results/water4

The output directory must be new. Packets and numerical results are retained
there; SCF scratch files are temporary. Timings include compilation and are
diagnostic, not a controlled performance benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xyz-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--auxbasis', default='cc-pvdz-jkfit')
    parser.add_argument('--correlation-auxbasis', default='cc-pvdz-ri')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-memory', type=float, default=10000)
    parser.add_argument('--pair-energy', type=float, default=5e-5)
    parser.add_argument('--boys-grad-tol', type=float, default=1e-9)
    parser.add_argument('--fd-step', type=float, default=5e-5, help='Bohr')
    args = parser.parse_args()
    if args.threads < 1 or args.fd_step <= 0:
        parser.error('threads and fd-step must be positive')

    import jax
    import numpy as np
    from pyscfad import config, df, dlno_stc, gto, scf
    from pyscfad.dlno_stc import backend, mp2
    from pyscfad.lno import _df_shared
    from pyscfad.tools import linear_solver

    config.update('pyscfad_moleintor_opt', True)
    args.output.mkdir(parents=True, exist_ok=False)
    mol = gto.Mole(atom=str(args.xyz_file.resolve()), basis=args.basis,
                   verbose=3, max_memory=args.max_memory)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    if mol.natm != 12 or sum(mol.atom_charge(i) == 8 for i in range(12)) != 4:
        raise ValueError('expected a water4 geometry')
    nodes, weights = np.polynomial.legendre.leggauss(64)
    controls = dict(mode='deterministic', threads=args.threads,
                    laplace_roots=20 * (nodes + 1), laplace_weights=20 * weights)
    correlation = dict(correlation_auxbasis=args.correlation_auxbasis)
    report = dict(settings={name: str(value) if isinstance(value, Path) else value
                            for name, value in vars(args).items()}, runs={})
    arrays, values = {}, {}
    with tempfile.TemporaryDirectory(prefix='water4-mp2-') as scratch:
        scf_path = str(Path(scratch) / 'scf.h5')
        def stage(current):
            # Stage the numerical file outside tracing; attaching it inside
            # build still registers its full molecular/auxiliary response.
            obj = df.DF(current, auxbasis=args.auxbasis, incore=False)
            obj._cderi_to_save = scf_path
            obj.build()

        stage(mol)
        def build(current):
            obj = df.DF(current, auxbasis=args.auxbasis, incore=False)
            obj.attach_outcore_cderi(scf_path)
            mf = scf.RHF(current).density_fit(with_df=obj)
            mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
            mf.chkfile = None
            with linear_solver.eager_gmres():
                mf.kernel()
            if not mf.converged:
                raise RuntimeError('SCF did not converge')
            return mf

        mf = build(mol)
        static = dlno_stc.build_domain_selections(mf, frozen=4,
            thresholds=dlno_stc.BoysDomainThresholds(pair_energy=args.pair_energy),
            lo_kwargs=dict(conv_tol=1e-12, conv_tol_grad=args.boys_grad_tol, max_cycle=200,
                           gmres_options={'tol': 1e-10}), **correlation)
        groups = dlno_stc.build_domain_groups(static)
        report['groups'] = [dict(targets=list(g.target_indices),
            nao=len(static.fragments[g.representative_index].extended_ao_indices)) for g in groups]
        report['weak_pairs'] = int(np.count_nonzero(np.triu(~(static.strong_mask & static.strong_mask.T), 1)))
        receipts = []
        finish = _df_shared.LocalDFReverse.finish
        def observed(scope):
            coords = finish(scope)
            receipts.append(dict(scope.receipt))
            return coords

        def measure(name, function):
            receipts.clear()
            started = time.perf_counter()
            energy, gradient = function()
            jax.block_until_ready((energy, gradient))
            elapsed = time.perf_counter() - started
            values[name] = (float(energy), np.asarray(gradient.coords))
            arrays[name + '_gradient'] = values[name][1]
            report['runs'][name] = dict(energy=float(energy), seconds=elapsed,
                                       df_receipts=list(receipts))
            print('COMPLETED', name, float(energy), elapsed, flush=True)

        with patch.object(_df_shared.LocalDFReverse, 'finish', observed):
            for mode in ('direct', 'half'):
                with patch.dict(os.environ, {'PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE': mode}):
                    measure(mode, lambda: dlno_stc.value_and_grad(
                        mol, build, static, controls=controls, include_hf=True, **correlation))
            packets = args.output / 'packets'
            rows = mp2.export_run(mf, static, packets, controls=controls, **correlation)
            for row in rows:
                folder = packets / f"fragment_{row['fragment_id']:04d}"
                mp2.solve_request(folder / 'input.h5', folder / 'result.h5', backend.solve)
            with patch.dict(os.environ, {'PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE': 'half'}):
                measure('replay', lambda: mp2.imported_value_and_grad(
                    mol, build, static, lambda index: mp2.load_disk_packet(packets, index),
                    include_hf=True, **correlation))

        report['comparisons'] = {}
        for name in ('half', 'replay'):
            de = abs(values[name][0] - values['direct'][0])
            dg = float(np.max(np.abs(values[name][1] - values['direct'][1])))
            report['comparisons'][name] = dict(energy_error=de, gradient_max_error=dg)
        direction = np.zeros((mol.natm, 3))
        direction[1, 2], direction[4, 0] = 1., -.4
        def displaced(sign):
            moved = mol.set_geom_(np.asarray(mol.atom_coords()) + sign * args.fd_step * direction,
                                  unit='Bohr', inplace=False)
            stage(moved)
            current = build(moved)
            return float(current.e_tot + dlno_stc.kernel(
                current, static, controls=controls, **correlation))
        with patch.dict(os.environ, {'PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE': 'direct'}):
            finite = (displaced(1) - displaced(-1)) / (2 * args.fd_step)
        analytic = float(np.sum(values['half'][1] * direction))
        report['finite_difference'] = dict(analytic=analytic, finite=finite,
                                           error=abs(analytic-finite), step_bohr=args.fd_step)
        arrays['direction'] = direction
        np.savez(args.output / 'gradients.npz', **arrays)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

        for name in ('half', 'replay'):
            np.testing.assert_allclose(values[name][0], values['direct'][0], atol=1e-9, rtol=0)
            np.testing.assert_allclose(values[name][1], values['direct'][1], atol=2e-7, rtol=0)
        np.testing.assert_allclose(analytic, finite, atol=2e-6, rtol=0)
        for name in ('direct', 'half', 'replay'):
            for receipt in report['runs'][name]['df_receipts']:
                assert receipt['raw_source_passes'] == (2 if name == 'direct' else 1)
                assert receipt['metric_builds'] == receipt['cholesky_builds'] == 1
                assert receipt['geometry_passes'] == receipt['metric_passes'] == 1
                assert receipt['half_resident_bytes'] == 0
        print(json.dumps(dict(comparisons=report['comparisons'],
                              finite_difference=report['finite_difference']), indent=2))


if __name__ == '__main__':
    main()
