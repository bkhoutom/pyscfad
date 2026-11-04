"""Compare local DF density packing on representative water-32 tensor shapes.

Run with the same BLAS libraries and CPU binding as the molecular benchmark:
  python benchmark_df_density_packing.py --threads 8 16 32 --output packing.json

Inputs are reproducible synthetic arrays. Timings cover MO-to-AO contraction
and packing, not derivative integrals or a complete molecular gradient.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from pyscf import lib
from threadpoolctl import ThreadpoolController
from pyscfad.df import _cderi_vjp as vjp
from pyscfad.lib._threading import blas_threads, dense_blas_scope, dense_blas_threads


CASES = dict(median_ed=(1176, 52, 686), large_ed=(1772, 104, 1403),
             large_lis=(1772, 113, 113), dense_ed=(1772, 1507, 1507))


def legacy_density(ck, cl, z, out):
    """Original dense contraction, symmetric sum, packing and diagonal scaling."""
    nq, nk, nl = z.shape
    nao = ck.shape[0]
    with dense_blas_threads():
        if nk <= nl:
            tmp = (z.reshape(-1, nl) @ cl.T).reshape(nq, nk, nao)
            dense = (tmp.transpose(0, 2, 1).reshape(-1, nk) @ ck.T).reshape(
                nq, nao, nao).swapaxes(1, 2)
        else:
            tmp = (z.transpose(0, 2, 1).reshape(-1, nk) @ ck.T).reshape(
                nq, nl, nao).transpose(0, 2, 1)
            dense = (tmp.reshape(-1, nl) @ cl.T).reshape(nq, nao, nao)
    del tmp
    dense = dense + dense.swapaxes(1, 2)
    out[:] = lib.pack_tril(dense)
    diagonal = np.arange(nao)
    out[:, diagonal*(diagonal+3)//2] *= .5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--aux-rows', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(*args.threads, args.aux_rows, args.repeats) <= 0:
        parser.error('thread counts, auxiliary rows and repeats must be positive')
    rng = np.random.default_rng(58)
    controller = ThreadpoolController()
    records = []
    with dense_blas_scope(), blas_threads(1):
        for name in args.cases:
            nao, nk, nl = CASES[name]
            ck = rng.normal(size=(nao, nk))*.1
            cl = rng.normal(size=(nao, nl))*.1
            z = rng.normal(size=(args.aux_rows, nk, nl))*.1
            out = np.empty((args.aux_rows, nao*(nao+1)//2))
            for threads in args.threads:
                os.environ['PYSCFAD_DENSE_BLAS_THREADS'] = str(threads)
                elapsed = {key: [] for key in ('legacy', 'gemm', 'auto')}
                errors = {}
                workspace = {}
                def run(backend):
                    if backend == 'legacy':
                        legacy_density(ck, cl, z, out)
                    else:
                        os.environ['PYSCFAD_DF_MO_DENSITY_BACKEND'] = backend
                        vjp._int3c_mo_packed_density(ck, cl, z, out, workspace=workspace)
                with controller.limit(limits=threads, user_api='openmp'):
                    run('legacy')
                    expected = out.copy()
                    for backend in elapsed:
                        run(backend)
                        errors[backend] = float(np.max(np.abs(out-expected)))
                        np.testing.assert_allclose(out, expected, atol=1e-10, rtol=1e-11)
                    # Alternate ordering to reduce warmup/cache-order bias.
                    for repeat in range(args.repeats):
                        for backend in list(elapsed)[::1 if repeat % 2 == 0 else -1]:
                            started = perf_counter()
                            run(backend)
                            elapsed[backend].append(perf_counter()-started)
                row = dict(case=name, nao=nao, kc=nk, lc=nl, naux=args.aux_rows,
                    threads=threads, median_seconds={k: median(x) for k,x in elapsed.items()},
                    samples_seconds=elapsed, max_difference=errors)
                records.append(row)
                print(json.dumps(row), flush=True)
            del ck, cl, z, out, expected, workspace
    report = dict(native_tiles_available=vjp._half_to_packed_tiles is not None,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  threadpools=controller.info(), results=records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
