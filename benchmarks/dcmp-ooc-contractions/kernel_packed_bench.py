#!/usr/bin/env python3
"""Focused occupied-reverse benchmark; launch with srun on a compute node.

The original and prior flattened variants are compared with the existing
packed AO2MO half-transform. Includes packing, layout copies, panel writes,
factor responses and dense thread-scope overhead; excludes input file reads.
"""

import argparse
import ctypes
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
    parser.add_argument('--output', default='kernel_packed_results.json')
    parser.add_argument('--nao', type=int, default=800)
    parser.add_argument('--rank', type=int, default=80)
    parser.add_argument('--rows', type=int, default=26)
    parser.add_argument('--tail-rows', type=int, default=14)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--dense-threads', type=int, default=16)
    parser.add_argument('--omp-threads', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run this numerical benchmark in a Slurm compute-node job.')
    if args.samples < 7 or args.warmups < 1:
        parser.error('Use at least seven samples and one warmup.')
    if min(args.nao, args.rank, args.rows, args.tail_rows,
           args.dense_threads, args.omp_threads) < 1:
        parser.error('Dimensions and thread counts must be positive.')

    import numpy as np
    from pyscf import lib
    from threadpoolctl import threadpool_info, threadpool_limits

    ao2mo = lib.load_library('libao2mo')
    driver = ao2mo.AO2MOnr_e2_drv
    transform = ao2mo.AO2MOtranse2_nr_s2
    multiply = ao2mo.AO2MOmmm_bra_nr_s2
    driver.restype = None

    def dense():
        return threadpool_limits(limits=args.dense_threads, user_api='blas')

    def prepare(s, g):
        """One-time per-VJP setup; D is also required later by the reverse."""
        if s.dtype != np.float64 or g.dtype != np.float64:
            return None
        started = time.perf_counter()
        with dense():
            d = s@s.T
        i, j = np.tril_indices(len(s))
        weights = np.asfortranarray(2*np.stack((d[i, j], g[i, j]), axis=1))
        weights[i == j] *= .5
        return np.asfortranarray(s), weights, time.perf_counter()-started

    def half_transform(packed, sf):
        """Result is C-order (auxiliary, occupied, AO), like the existing caller."""
        p, n, rank = len(packed), *sf.shape
        out = np.empty((p, rank, n))
        driver(transform, multiply,
               out.ctypes.data_as(ctypes.c_void_p),
               packed.ctypes.data_as(ctypes.c_void_p),
               sf.ctypes.data_as(ctypes.c_void_p),
               ctypes.c_int(p), ctypes.c_int(n),
               (ctypes.c_int*4)(0, rank, 0, n),
               lib.c_null_ptr(), ctypes.c_int(0))
        return out

    def original(packed, s, gseed, hseed, with_j, with_k, response, setup=None,
                 flattened=False):
        del setup
        p, n, rank = len(packed), *s.shape
        b = lib.unpack_tril(packed)
        with dense():
            t = ((b.reshape(-1, n)@s).reshape(p, n, rank)
                 if flattened else b@s)
            q = np.einsum('ar,par->p', s, t) if with_j else None
            g = np.einsum('ab,pab->p', gseed, b) if with_j else None
            sbar = np.zeros_like(s) if response else None
            if with_j and response:
                sbar += 2*np.einsum('p,par->ar', g, t)
            tp = rp = None
            if with_k:
                if flattened:
                    tr = np.ascontiguousarray(t.transpose(0, 2, 1))
                    rr = tr.reshape(-1, n)@hseed.T
                    r = np.ascontiguousarray(rr.reshape(p, rank, n).transpose(0, 2, 1))
                else:
                    r = hseed@t
                if response:
                    if flattened and not np.iscomplexobj(b):
                        sbar += 2*(b.reshape(-1, n).T@r.reshape(-1, rank))
                    else:
                        sbar += 2*np.sum(b@r, axis=0)
                tp = np.empty((p, n*rank), dtype=t.dtype)
                rp = np.empty((p, n*rank), dtype=r.dtype)
                tp[:] = t.reshape(p, -1)
                rp[:] = r.reshape(p, -1)
        return q, g, tp, rp, sbar

    def flattened(*values):
        return original(*values, flattened=True)

    def packed_native(packed, s, gseed, hseed, with_j, with_k, response, setup,
                      thin_q=False, batched_reduction=False):
        if (setup is None or any(x.dtype != np.float64 for x in
                                 (packed, s, gseed, hseed))):
            return original(packed, s, gseed, hseed, with_j, with_k, response)
        sf, weights, _ = setup
        p, n, rank = len(packed), *s.shape
        # Both OpenMP operations are outside the dense BLAS scope. The native
        # driver uses per-worker unpack+dsymm with baseline BLAS=1.
        tr = half_transform(packed, sf) if (with_k or response or thin_q) else None
        b = lib.unpack_tril(packed) if with_k and response else None
        with dense():
            q = g = None
            if with_j:
                if thin_q:
                    q = np.einsum('ra,pra->p', s.T, tr)
                    g = packed@weights[:, 1]
                else:
                    qg = packed@weights
                    q, g = qg[:, 0], qg[:, 1]
            sbar = np.zeros_like(s) if response else None
            if with_j and response:
                sbar += 2*np.einsum('p,pra->ar', g, tr)
            tp = rp = None
            if with_k:
                rr = (tr.reshape(-1, n)@hseed.T).reshape(p, rank, n)
                # Copy straight into the existing panel layout, without an
                # intermediate reshape copy of the transposed thin factors.
                tp = np.empty((p, n*rank))
                rp = np.empty_like(tp)
                tp.reshape(p, n, rank)[:] = tr.transpose(0, 2, 1)
                rp.reshape(p, n, rank)[:] = rr.transpose(0, 2, 1)
                if response:
                    if batched_reduction:
                        sbar += 2*np.sum(b@rp.reshape(p, n, rank), axis=0)
                    else:
                        sbar += 2*(b.reshape(-1, n).T@rp.reshape(-1, rank))
        return q, g, tp, rp, sbar

    def packed_thin_q(*values):
        return packed_native(*values, thin_q=True)

    def packed_batched_reduce(*values):
        return packed_native(*values, batched_reduction=True)

    def check(actual, expected):
        if len(actual) != len(expected):
            raise AssertionError('Output tuple lengths differ.')
        errors = []
        for a, e in zip(actual, expected):
            if e is None:
                if a is not None:
                    raise AssertionError('Disabled output was materialized.')
                continue
            np.testing.assert_allclose(a, e, rtol=2e-11, atol=2e-10)
            errors.append(float(np.max(np.abs(a-e))))
        return max(errors, default=0.)

    rng = np.random.default_rng(917283)

    def data(n, rank, p):
        packed = rng.standard_normal((p, n*(n+1)//2))/np.sqrt(n)
        s = rng.standard_normal((n, rank))/np.sqrt(n)
        g, h = (rng.standard_normal((n, n))/np.sqrt(n) for _ in range(2))
        return packed, s, (g+g.T)*.5, (h+h.T)*.5

    variants = {'original': original, 'flattened': flattened,
                'packed_native': packed_native,
                'packed_native_thin_q': packed_thin_q,
                'packed_native_batched_reduction': packed_batched_reduce}

    def validate_small():
        packed, s, g, h = data(7, 3, 4)
        i, j = np.tril_indices(7)
        b = np.zeros((4, 7, 7))
        b[:, i, j] = packed
        b[:, j, i] = packed
        t = np.einsum('pij,jr->pir', b, s)
        r = np.einsum('ij,pjr->pir', h, t)
        q = np.einsum('ar,par->p', s, t)
        gv = np.einsum('ij,pij->p', g, b)
        count = 0
        expanded = np.empty((7, 6))
        expanded[:, ::2] = s
        for ss in (s, np.asfortranarray(s), expanded[:, ::2]):
            setup = prepare(ss, g)
            native_t = half_transform(packed, setup[0])
            np.testing.assert_allclose(native_t, t.transpose(0, 2, 1), atol=2e-12)
            for with_j, with_k in ((True, True), (True, False), (False, True)):
                for response in (False, True):
                    sb = np.zeros_like(s) if response else None
                    if response and with_j:
                        sb += 2*np.einsum('p,pir->ir', gv, t)
                    if response and with_k:
                        sb += 2*np.einsum('pij,pjr->ir', b, r)
                    expected = (q if with_j else None, gv if with_j else None,
                                t.reshape(4, -1) if with_k else None,
                                r.reshape(4, -1) if with_k else None, sb)
                    for fn in variants.values():
                        check(fn(packed, ss, g, h, with_j, with_k, response, setup), expected)
                        count += 1
        # Unsupported complex inputs must return the unmodified original rule.
        for complex_index in range(4):
            inputs = [packed, s, g, h]
            inputs[complex_index] = inputs[complex_index].astype(np.complex128)
            inputs[complex_index] += .01j*rng.standard_normal(inputs[complex_index].shape)
            if complex_index == 0:
                inputs[0][:, i == j] = inputs[0][:, i == j].real
            expected = original(*inputs, True, True, False)
            for fn in (packed_native, packed_thin_q, packed_batched_reduce):
                check(fn(*inputs, True, True, False, prepare(inputs[1], inputs[2])), expected)
                count += 1
        return {'passed': True, 'checked_variant_outputs': count,
                'independent_reference': 'explicit small einsums and direct triangle expansion',
                'layouts': ['C', 'Fortran', 'strided'],
                'flags': ['J+K', 'J-only', 'K-only', 'factor-response on/off'],
                'complex_fallback': 'packed/S/G/H individually complex'}

    timings = {}

    def compare(name, functions):
        names = list(functions)
        reference = functions['original']()
        errors = {key: check(fn(), reference) for key, fn in functions.items()}
        del reference
        for warm in range(args.warmups):
            for key in names if warm % 2 == 0 else names[::-1]:
                value = functions[key]()
                del value
        values = {key: [] for key in names}
        orders = []
        for sample in range(args.samples):
            shift = (sample//2) % len(names)
            order = names[shift:]+names[:shift]
            if sample % 2:
                order = order[::-1]
            orders.append(order)
            for key in order:
                started = time.perf_counter()
                value = functions[key]()
                values[key].append(time.perf_counter()-started)
                del value
        entry = {'seconds': values,
                 'median_seconds': {key: statistics.median(v) for key, v in values.items()},
                 'minimum_seconds': {key: min(v) for key, v in values.items()},
                 'maximum_seconds': {key: max(v) for key, v in values.items()},
                 'maximum_absolute_error': errors, 'sample_order': orders,
                 'median_paired_speedup': {}, 'samples_faster_than_original': {}}
        for key in names[1:]:
            pairs = list(zip(values['original'], values[key]))
            entry['median_paired_speedup'][key] = statistics.median(a/b for a, b in pairs)
            entry['samples_faster_than_original'][key] = sum(b < a for a, b in pairs)
        timings[name] = entry
        print(json.dumps({'case': name, **entry}), flush=True)

    lib.num_threads(args.omp_threads)
    with (threadpool_limits(limits=args.omp_threads, user_api='openmp'),
          threadpool_limits(limits=1, user_api='blas')):
        outside_pools = threadpool_info()
        with dense():
            inside_pools = threadpool_info()
        validation = validate_small()
        packed, s, g, h = data(args.nao, args.rank, max(args.rows, args.tail_rows))
        setup = prepare(s, g)
        # Setup is paid once per reverse, not per streamed block. Report it
        # explicitly so any full-pass estimate can add it back.
        setup_seconds = [setup[2]]
        for _ in range(args.samples-1):
            alternative_setup = prepare(s, g)
            setup_seconds.append(alternative_setup[2])
        del alternative_setup
        for rows in dict.fromkeys((args.rows, args.tail_rows)):
            for response in (False, True):
                current = dict(variants)
                if not response:
                    current.pop('packed_native_batched_reduction')
                functions = {
                    key: (lambda fn=fn, rows=rows, response=response:
                          fn(packed[:rows], s, g, h, True, True, response, setup))
                    for key, fn in current.items()
                }
                compare(f'J_K_rows_{rows}_factor_response_{response}', functions)

    result = {
        'settings': vars(args), 'timings': timings, 'validation': validation,
        'one_time_setup_seconds': setup_seconds,
        'one_time_setup_median_seconds': statistics.median(setup_seconds),
        'thread_pools_outside_dense_scope': outside_pools,
        'thread_pools_inside_dense_scope': inside_pools,
        'affinity': sorted(os.sched_getaffinity(0)), 'hostname': platform.node(),
        'python': sys.executable, 'python_version': sys.version,
        'numpy_version': np.__version__, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native_driver_scratch_bytes_estimate': args.omp_threads*8*(args.nao+args.rank)*(2*args.nao),
        'environment': {key: val for key, val in os.environ.items()
                        if key.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'SLURM_',
                                           'PYSCFAD_', 'SCIPY_DF_'))},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes': [
            'Full output validation precedes each timed comparison.',
            'All variants include dense thread scope, required layout copies and thin panel writes.',
            'Original/flattened include full unpack; native unpacks full B only with factor response.',
            'Native OpenMP half-transform runs with baseline BLAS=1 outside dense scopes.',
            'Packed q/g weights and Fortran S are one-time per-VJP setup, separately timed.',
            'Thin-q variant keeps q from native t and contracts only g with packed factors.',
            'Repeated block timings exclude file reads and later metric/coordinate stages.',
            'No production source has been modified by this benchmark.',
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
