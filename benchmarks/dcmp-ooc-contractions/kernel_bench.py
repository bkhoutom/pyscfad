#!/usr/bin/env python3
"""Compare bounded DF reverse kernels; run only inside a Slurm compute job.

Example: srun python -u kernel_bench.py --output kernel_results.json
No package source is modified. NumPy/SciPy/PySCF imports occur only after
checking the Slurm environment. Timing includes required layout copies.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='kernel_results.json')
    parser.add_argument('--nao', type=int, default=800)
    parser.add_argument('--rank', type=int, default=80)
    parser.add_argument('--rows', type=int, default=26)
    parser.add_argument('--packing-rows', type=int, default=16)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--dense-threads', type=int, default=16)
    parser.add_argument('--omp-threads', type=int, default=16)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Run this numerical benchmark in a Slurm compute-node job.')
    if min(args.nao, args.rank, args.rows, args.packing_rows, args.samples,
           args.dense_threads, args.omp_threads) < 1 or args.warmups < 1:
        parser.error('Dimensions, samples, threads and warmups must be positive.')

    import numpy as np
    from scipy.linalg.blas import dgemm
    from pyscf import lib
    from pyscfadlib import libnp_helper_vjp
    from threadpoolctl import threadpool_info, threadpool_limits

    def dense():
        return threadpool_limits(limits=args.dense_threads, user_api='blas')

    native = libnp_helper_vjp.NPdunpack_tril_2d_vjp
    native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_int]
    native.restype = None

    def direct_pack(a):
        a = np.ascontiguousarray(a, dtype=np.float64)
        p, n, _ = a.shape
        out = np.empty((p, n*(n+1)//2))
        native(p, n, out.ctypes.data, a.ctypes.data, 1)
        return out

    def old_pack(a):
        # A newly computed density is C contiguous and may be overwritten.
        density = lib.hermi_sum(a.copy(), axes=(0, 2, 1), inplace=True)
        if not np.array_equal(density, density.swapaxes(1, 2)):
            raise AssertionError('Original symmetric native path was not selected.')
        n = a.shape[-1]
        diag = np.arange(n)
        diag = diag*(diag+1)//2+diag
        out = np.zeros((a.shape[0], n*(n+1)//2))
        out[:] = lib.pack_tril(density)
        out[:, diag] *= .5
        return out

    def old_pack_owned(a):
        # Full assembly already owns its input; avoid an artificial copy.
        density = lib.hermi_sum(a, axes=(0, 2, 1), inplace=True)
        if not np.array_equal(density, density.swapaxes(1, 2)):
            raise AssertionError('Original symmetric native path was not selected.')
        n = a.shape[-1]
        diag = np.arange(n)
        diag = diag*(diag+1)//2+diag
        out = np.zeros((a.shape[0], n*(n+1)//2))
        out[:] = lib.pack_tril(density)
        out[:, diag] *= .5
        return out

    def flatten_t(b, s):
        p, n, _ = b.shape
        return (b.reshape(-1, n) @ s).reshape(p, n, s.shape[1])

    def flatten_r(h, t):
        p, n, rank = t.shape
        transposed = np.ascontiguousarray(t.transpose(0, 2, 1))
        result = transposed.reshape(-1, n) @ h.T
        return np.ascontiguousarray(result.reshape(p, rank, n).transpose(0, 2, 1))

    def flatten_reduce(b, r):
        if np.iscomplexobj(b):
            return np.sum(b @ r, axis=0)
        # lib.unpack_tril makes real B symmetric; this uses no conjugation.
        return b.reshape(-1, b.shape[-1]).T @ r.reshape(-1, r.shape[-1])

    def block(packed, s, gseed, hseed, response, flattened):
        b = lib.unpack_tril(packed)
        with dense():
            t = flatten_t(b, s) if flattened else b @ s
            q = np.einsum('ar,par->p', s, t)
            g = np.einsum('ab,pab->p', gseed, b)
            r = flatten_r(hseed, t) if flattened else hseed @ t
            sbar = np.zeros_like(s)
            if response:
                sbar += 2*np.einsum('p,par->ar', g, t)
                sbar += 2*(flatten_reduce(b, r) if flattened else
                           np.sum(b @ r, axis=0))
            # Include the writes into the existing C-order thin panels.
            t_panel = np.empty((len(packed), s.size))
            r_panel = np.empty_like(t_panel)
            t_panel[:] = t.reshape(len(packed), -1)
            r_panel[:] = r.reshape(len(packed), -1)
        return q, g, t_panel, r_panel, sbar

    def check(actual, expected):
        if not isinstance(actual, tuple):
            actual, expected = (actual,), (expected,)
        errors = []
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a, e, rtol=2e-11, atol=2e-10)
            errors.append(float(np.max(np.abs(a-e))) if a.size else 0.)
        return max(errors, default=0.)

    timings = {}

    def compare(name, variants):
        names = list(variants)
        reference = variants[names[0]]()
        errors = {key: check(fn(), reference) for key, fn in variants.items()}
        del reference
        for _ in range(args.warmups):
            for fn in variants.values():
                result = fn()
                del result
        seconds = {key: [] for key in variants}
        orders = []
        for sample in range(args.samples):
            order = names if sample % 2 == 0 else names[::-1]
            orders.append(order[:])
            for key in order:
                started = time.perf_counter()
                result = variants[key]()
                seconds[key].append(time.perf_counter()-started)
                del result
        entry = {
            'seconds': seconds,
            'median_seconds': {key: statistics.median(val) for key, val in seconds.items()},
            'maximum_absolute_error': errors,
            'sample_order': orders,
        }
        timings[name] = entry
        print(json.dumps({'case': name, **entry}), flush=True)

    def with_dense(fn):
        def run():
            with dense():
                return fn()
        return run

    rng = np.random.default_rng(824709)

    def initial_data(n, rank, p):
        packed = rng.standard_normal((p, n*(n+1)//2))/np.sqrt(n)
        s = rng.standard_normal((n, rank))/np.sqrt(n)
        g, h = (rng.standard_normal((n, n))/np.sqrt(n) for _ in range(2))
        return packed, s, (g+g.T)*.5, (h+h.T)*.5

    def small_validation():
        packed, s, g, h = initial_data(7, 3, 4)
        i, j = np.tril_indices(7)
        b = np.zeros((4, 7, 7))
        b[:, i, j] = packed
        b[:, j, i] = packed
        t = np.einsum('pij,jr->pir', b, s)
        r = np.einsum('ij,pjr->pir', h, t)
        q = np.einsum('ar,par->p', s, t)
        gv = np.einsum('ij,pij->p', g, b)
        sb = 2*np.einsum('p,pir->ir', gv, t)+2*np.einsum('pij,pjr->ir', b, r)
        expected = q, gv, t.reshape(4, -1), r.reshape(4, -1), sb
        check(block(packed, s, g, h, True, False), expected)
        check(block(packed, s, g, h, True, True), expected)
        # Exercise both contiguous layouts and a noncontiguous factor view.
        expanded = np.empty((7, 6))
        expanded[:, ::2] = s
        for ss in (np.asfortranarray(s), expanded[:, ::2]):
            check(block(packed, ss, g, h, True, True), expected)
        raw = rng.standard_normal((4, 7, 7))
        independent = raw[:, i, j]+raw[:, j, i]
        independent[:, i == j] *= .5
        check(direct_pack(raw), independent)
        check(old_pack(raw), independent)
        # The complex fallback must preserve the original B@r semantics.
        complex_b = b+1j*rng.standard_normal(b.shape)
        check(flatten_reduce(complex_b, r), np.einsum('pij,pjr->ir', complex_b, r))
        return {'passed': True, 'initial_shape': [4, 7, 3],
                'checks': ['explicit_einsums', 'C_F_strided_factors',
                           'independent_triangle_sum', 'complex_reduction_fallback']}

    def density_variants(n, rank, p, with_k=True):
        panel = rng.standard_normal((p*n, rank))/np.sqrt(n)
        s = rng.standard_normal((n, rank))/np.sqrt(n)
        g = rng.standard_normal((n, n))/np.sqrt(n)
        g = (g+g.T)*.5
        with dense():
            d = s@s.T
        coefficients = np.asfortranarray(rng.standard_normal((2, p)))
        st = np.ascontiguousarray(2*s.T)
        columns = np.asfortranarray(np.stack((g.ravel(), d.ravel()), axis=1))
        i, j = np.tril_indices(n)
        packed_columns = np.asfortranarray(2*np.stack((g[i, j], d[i, j]), axis=1))
        packed_columns[i == j] *= .5

        def raw_density(include_j):
            with dense():
                if with_k:
                    a = (panel@st).reshape(p, n, n)
                else:
                    a = np.zeros((p, n, n))
                if include_j:
                    a = dgemm(1., columns, coefficients, beta=1.,
                              c=a.reshape(p, -1).T, overwrite_c=1).T.reshape(p, n, n)
            return a

        def original():
            return old_pack_owned(raw_density(True))

        def native_full_j():
            return direct_pack(raw_density(True))

        def native_packed_j():
            if with_k:
                out = direct_pack(raw_density(False))
            else:
                out = np.zeros((p, n*(n+1)//2))
            with dense():
                out = dgemm(1., packed_columns, coefficients, beta=1.,
                            c=out.T, overwrite_c=1).T
            return out

        if n <= 7:
            # This deliberately avoids both dgemm and the native pack routine.
            raw = (2*np.einsum('pir,jr->pij', panel.reshape(p, n, rank), s)
                   if with_k else np.zeros((p, n, n)))
            raw += np.einsum('p,ij->pij', coefficients[0], g)
            raw += np.einsum('p,ij->pij', coefficients[1], d)
            expected = raw[:, i, j]+raw[:, j, i]
            expected[:, i == j] *= .5
            for fn in (original, native_full_j, native_packed_j):
                check(fn(), expected)
        return {'original': original, 'native_full_j': native_full_j,
                'native_packed_j': native_packed_j}

    # BLAS is sequential except for explicit dense scopes. Native packing and
    # unpacking run outside those scopes with the allocated OpenMP threads.
    lib.num_threads(args.omp_threads)
    with (threadpool_limits(limits=args.omp_threads, user_api='openmp'),
          threadpool_limits(limits=1, user_api='blas')):
        outside_pools = threadpool_info()
        with dense():
            inside_pools = threadpool_info()
        validation = small_validation()
        density_variants(7, 3, 4, True)
        density_variants(7, 3, 4, False)
        packed, s, g, h = initial_data(args.nao, args.rank, args.rows)
        b = lib.unpack_tril(packed)
        with dense():
            t = b@s
            r = h@t
        compare('t_BS', {
            'original': with_dense(lambda: b@s),
            'flattened': with_dense(lambda: flatten_t(b, s)),
        })
        compare('r_Ht_including_thin_copies', {
            'original': with_dense(lambda: h@t),
            'flattened': with_dense(lambda: flatten_r(h, t)),
        })
        compare('factor_exchange_reduction', {
            'original': with_dense(lambda: np.sum(b@r, axis=0)),
            'flattened': with_dense(lambda: flatten_reduce(b, r)),
        })
        for response in (False, True):
            compare(f'complete_initial_block_factor_response_{response}', {
                'original': lambda response=response: block(packed, s, g, h, response, False),
                'flattened': lambda response=response: block(packed, s, g, h, response, True),
            })
        # A shortened block exercises the last 14 rows of naux=2042 at rows=26.
        tail = 2042 % args.rows
        if tail:
            compare('complete_initial_tail_block', {
                'original': lambda: block(packed[:tail], s, g, h, True, False),
                'flattened': lambda: block(packed[:tail], s, g, h, True, True),
            })
        del packed, s, g, h, b, t, r
        raw = rng.standard_normal((args.packing_rows, args.nao, args.nao))
        compare('packing_only_original_includes_owned_input_copy', {
            'original': lambda: old_pack(raw),
            'native_direct': lambda: direct_pack(raw),
        })
        del raw
        for with_k in (True, False):
            compare('complete_density_JK' if with_k else 'complete_density_J_only',
                    density_variants(args.nao, args.rank, args.packing_rows, with_k))

    record = {
        'settings': vars(args), 'timings': timings, 'validation': validation,
        'hostname': platform.node(), 'python': sys.executable,
        'python_version': sys.version, 'numpy_version': np.__version__,
        'affinity': sorted(os.sched_getaffinity(0)),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'thread_pools_outside_dense_scope': outside_pools,
        'thread_pools_inside_dense_scope': inside_pools,
        'environment': {key: val for key, val in os.environ.items()
                        if key.startswith(('OMP_', 'MKL_', 'OPENBLAS_', 'SLURM_',
                                           'PYSCFAD_', 'SCIPY_DF_'))},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes': [
            'All times are wall seconds, with one or more warmups and alternating order.',
            'Dense scopes include threadpoolctl overhead; native OpenMP stays outside.',
            'Packing-only original copies its input to preserve repeats; complete assembly does not.',
            'Complete initial blocks include unpack and thin panel writes, but not file reads.',
            'Density setup columns are precomputed once, as in the reverse routine.',
        ],
    }
    Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
