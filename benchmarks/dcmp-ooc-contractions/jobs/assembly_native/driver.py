#!/usr/bin/env python3
"""Measure density assembly followed immediately by the real int3c VJP.

Run in a Slurm compute step, with local input.xyz and the candidate package
on PYTHONPATH. Timings test a possible thread interaction; they do not assume
that interaction explains the earlier full-job timing variation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='assembly_native_results.json')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--rank', type=int, default=80)
    parser.add_argument('--dense-threads', type=int, default=16)
    parser.add_argument('--omp-threads', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run this numerical experiment in a Slurm compute-node job.')
    if args.samples < 5 or min(args.warmups, args.rows, args.rank,
                               args.dense_threads, args.omp_threads) < 1:
        parser.error('Use at least five samples and positive dimensions/threads/warmups.')
    input_path = Path('input.xyz')
    if not input_path.is_file():
        parser.error('Local input.xyz is required; submit from this condition directory.')

    import jax
    import numpy as np
    from scipy.linalg.blas import dgemm
    from pyscf import lib
    from pyscf.ao2mo.outcore import balance_partition
    from pyscfad import gto
    from pyscfad.df import addons, _cderi_vjp, _int3c_cross_opt
    from pyscfad.lib._threading import dense_blas_threads
    from jaxlib.cpu import _lapack
    from threadpoolctl import threadpool_info, threadpool_limits

    jax.config.update('jax_enable_x64', True)
    _lapack.initialize()
    os.environ['PYSCFAD_DENSE_BLAS_THREADS'] = str(args.dense_threads)
    if not hasattr(_cderi_vjp, '_pack_tril_symmetric_sum'):
        raise RuntimeError('Use the candidate package containing the packed-density helper.')
    lib.num_threads(args.omp_threads)
    rng = np.random.default_rng(117305)
    variants = ('original', 'packed_j_16', 'packed_j_1', 'native_full_j')
    records = {name: [] for name in variants}
    validation = []

    with (threadpool_limits(limits=args.omp_threads, user_api='openmp'),
          threadpool_limits(limits=1, user_api='blas')):
        pools_outside = threadpool_info()
        with dense_blas_threads():
            pools_dense = threadpool_info()
        mol = gto.Mole(atom=str(input_path), basis='cc-pvtz', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        auxmol = addons.make_auxmol(mol, 'cc-pvtz-jkfit')
        n, rank = mol.nao, args.rank
        if n != 800 or auxmol.nao != 2042:
            raise AssertionError(f'Expected dCMP dimensions (800,2042), got {(n, auxmol.nao)}')
        s = rng.standard_normal((n, rank))/np.sqrt(n)
        g = rng.standard_normal((n, n))/np.sqrt(n)
        g = (g+g.T)*.5
        with dense_blas_threads():
            d = s@s.T
        st = np.ascontiguousarray(2*s.T)
        full_columns = np.asfortranarray(np.stack((g.ravel(), d.ravel()), axis=1))
        packed_columns = _cderi_vjp._pack_tril_symmetric_sum(np.stack((g, d))).T
        triangle_i, triangle_j = np.tril_indices(n)
        diagonal = np.arange(n)
        diagonal = diagonal*(diagonal+3)//2
        ranges = balance_partition(auxmol.ao_loc, args.rows)
        chosen = (len(ranges)//8, len(ranges)//2, 7*len(ranges)//8)
        blocks = []
        for index in chosen:
            sh0, sh1, _ = ranges[index]
            p0, p1 = map(int, auxmol.ao_loc[[sh0, sh1]])
            blocks.append({
                'range_index': index, 'shell_range': [int(sh0), int(sh1)],
                'auxiliary_range': [p0, p1],
                'shell_slice': (0, mol.nbas, 0, mol.nbas,
                                mol.nbas+int(sh0), mol.nbas+int(sh1)),
                'panel': rng.standard_normal(((p1-p0)*n, rank))/np.sqrt(n),
                'coefficients': rng.standard_normal((2, p1-p0)),
            })

        def assemble(name, block):
            p = block['auxiliary_range'][1]-block['auxiliary_range'][0]
            coefficients = np.array(block['coefficients'], order='F')
            if name in ('original', 'native_full_j'):
                with dense_blas_threads():
                    density = (block['panel']@st).reshape(p, n, n)
                    updated = dgemm(1., full_columns, coefficients, beta=1.,
                                    c=density.reshape(p, -1).T, overwrite_c=1)
                    density = updated.T.reshape(p, n, n)
                if name == 'native_full_j':
                    return _cderi_vjp._pack_tril_symmetric_sum(density)
                density = lib.hermi_sum(density, axes=(0, 2, 1), inplace=True)
                if not np.array_equal(density, density.swapaxes(1, 2)):
                    raise AssertionError('Original native path requires symmetric density.')
                packed = np.zeros((p, n*(n+1)//2))
                packed[:] = lib.pack_tril(density)
                packed[:, diagonal] *= .5
                return packed
            with dense_blas_threads():
                density = (block['panel']@st).reshape(p, n, n)
            packed = _cderi_vjp._pack_tril_symmetric_sum(density)
            del density
            if name == 'packed_j_16':
                with dense_blas_threads():
                    updated = dgemm(1., packed_columns, coefficients, beta=1.,
                                    c=packed.T, overwrite_c=1)
            else:
                # Baseline BLAS=1. K expansion alone uses the dense scope.
                updated = dgemm(1., packed_columns, coefficients, beta=1.,
                                c=packed.T, overwrite_c=1)
            if not np.shares_memory(updated, packed):
                packed[:] = updated.T
            return packed

        def run(name, block):
            started = time.perf_counter()
            packed = assemble(name, block)
            assembled = time.perf_counter()
            bars = _int3c_cross_opt.int3c_cross_bwd(
                mol._add_suffix('int3c2e'), 1, 's2ij', block['shell_slice'],
                None, (mol, auxmol), packed.T)
            coordinates = tuple(np.asarray(bar.coords) for bar in bars)
            finished = time.perf_counter()
            return packed, coordinates, {
                'assembly_seconds': assembled-started,
                'native_seconds': finished-assembled,
                'total_seconds': finished-started,
            }

        # Validate against explicit physical densities and derivative integrals,
        # outside timed regions. All molecular and auxiliary coordinates count.
        for block in blocks:
            p0, p1 = block['auxiliary_range']
            p = p1-p0
            with dense_blas_threads():
                raw = (2*block['panel']@s.T).reshape(p, n, n)
            raw += block['coefficients'][0, :, None, None]*g
            raw += block['coefficients'][1, :, None, None]*d
            physical = raw+raw.swapaxes(1, 2)
            expected_packed = physical[:, triangle_i, triangle_j].copy()
            expected_packed[:, diagonal] *= .5
            del raw
            ints = np.asarray(_int3c_cross_opt.int3c_cross(
                mol, auxmol, intor=mol._add_suffix('int3c2e_ip1'), comp=3,
                aosym='s1', shls_slice=block['shell_slice']))
            per_aux_ao = np.einsum('pij,xijp->pix', physical, ints)
            del ints, physical
            expected_mol = np.zeros((mol.natm, 3))
            expected_aux = np.zeros((auxmol.natm, 3))
            for atom, (_, _, a0, a1) in enumerate(mol.aoslice_by_atom()):
                expected_mol[atom] = -per_aux_ao[:, a0:a1].sum(axis=(0, 1))
            for atom, (_, _, a0, a1) in enumerate(auxmol.aoslice_by_atom()):
                begin, end = max(int(a0), p0), min(int(a1), p1)
                if begin < end:
                    expected_aux[atom] = per_aux_ao[begin-p0:end-p0].sum(axis=(0, 1))
            del per_aux_ao
            expected_coordinates = expected_mol, expected_aux
            block['expected_coordinates'] = expected_coordinates
            for name in variants:
                packed, coords, _ = run(name, block)
                np.testing.assert_allclose(packed, expected_packed, rtol=2e-11, atol=2e-11)
                for actual, expected in zip(coords, expected_coordinates):
                    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8)
                validation.append({
                    'variant': name, 'auxiliary_range': [p0, p1],
                    'packing_max_absolute_error': float(np.max(np.abs(packed-expected_packed))),
                    'coordinate_max_absolute_error': max(float(np.max(np.abs(a-e)))
                                                         for a, e in zip(coords, expected_coordinates)),
                })
                del packed, coords
            del expected_packed
        print(json.dumps({'validation': validation}), flush=True)

        for warm in range(args.warmups):
            for name in variants if warm % 2 == 0 else variants[::-1]:
                for block in blocks:
                    packed, coords, _ = run(name, block)
                    del packed, coords
        orders = []
        for sample in range(args.samples):
            offset = (sample//2) % len(variants)
            order = variants[offset:]+variants[:offset]
            if sample % 2:
                order = order[::-1]
            orders.append(order)
            block_order = blocks[sample % 3:]+blocks[:sample % 3]
            for name in order:
                totals = {'assembly_seconds': 0., 'native_seconds': 0., 'total_seconds': 0.}
                detail = []
                for block in block_order:
                    packed, coords, seconds = run(name, block)
                    for actual, expected in zip(coords, block['expected_coordinates']):
                        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-8)
                    for key in totals:
                        totals[key] += seconds[key]
                    detail.append({'auxiliary_range': block['auxiliary_range'], **seconds})
                    del packed, coords
                records[name].append({**totals, 'blocks': detail})
                print(json.dumps({'sample': sample, 'variant': name, **totals}), flush=True)
        pools_final = threadpool_info()

    medians = {name: {key: statistics.median(record[key] for record in samples)
                      for key in ('assembly_seconds', 'native_seconds', 'total_seconds')}
               for name, samples in records.items()}
    result = {
        'settings': vars(args), 'samples': records, 'median_seconds': medians,
        'median_paired_total_speedup': {
            name: statistics.median(a['total_seconds']/b['total_seconds']
                                    for a, b in zip(records['original'], records[name]))
            for name in variants[1:]},
        'sample_orders': orders, 'validation': validation,
        'blocks': [{key: value for key, value in block.items()
                    if key not in ('panel', 'coefficients', 'expected_coordinates')}
                   for block in blocks],
        'dimensions': {'nao': n, 'naux': auxmol.nao, 'rank': rank, 'natm': mol.natm},
        'affinity': sorted(os.sched_getaffinity(0)), 'hostname': platform.node(),
        'python': sys.executable, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'thread_pools_outside_dense_scope': pools_outside,
        'thread_pools_inside_dense_scope': pools_dense,
        'thread_pools_after_samples': pools_final,
        'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in (input_path, Path(__file__), Path(_cderi_vjp.__file__),
                                Path(_int3c_cross_opt.__file__))},
        'notes': [
            'Three real dCMP auxiliary shell blocks, full 800-AO coordinate contractions.',
            'Random rank-80 factors and symmetric G/D replace expensive SCF replay.',
            'All timings include assembly immediately followed by the unchanged native derivative call.',
            'All BLAS pools are one thread outside explicit dense scopes; OpenMP remains sixteen.',
            'Packed J sequential variant scopes only K expansion to sixteen BLAS threads.',
            'Five or more alternating samples assess net overhead and timing variability.',
            'The experiment tests thread interaction; it does not establish a cause for full-job variance.',
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'median_seconds': medians, 'output': args.output}), flush=True)


if __name__ == '__main__':
    main()
