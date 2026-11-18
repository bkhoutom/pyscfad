"""Standalone synthetic-factor RSS probe; no SCF or AO-integral workspace.

Run in separate processes with the old/new extension paths and legacy/native
layouts. The legacy setup retains its two preparation factors; native shares
one. This measures the factor handoff and scalar solver, not molecular RSS.
"""
import argparse
import importlib.util
import json
import resource
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--layout', choices=('legacy', 'native'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_stc_mp2', args.extension)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    q, no, nv = 2048, 32, 256  # One factor is 128 MiB.
    if args.layout == 'legacy':
        b = np.empty((q, no, nv))
    else:
        b = np.empty((q, no * nv), order='F').reshape(q, no, nv)
    rng = np.random.default_rng(5127)
    for start in range(0, q, 16):
        b[start:start+16] = rng.normal(size=(16, no, nv)) * .003
    residual = b.reshape(q, no * nv).copy() if args.layout == 'legacy' else b
    assert np.shares_memory(residual, b) == (args.layout == 'native')
    foo, fvv = -np.eye(no), np.eye(nv)
    controls = dict(mode='deterministic', threads=2, target_batch_size=1,
                    laplace_roots=[.2, .7], laplace_weights=[.4, .6])
    solve = native.solve_selected if args.layout == 'legacy' else native._solve_selected_borrowed
    start = time.perf_counter()
    result = solve(foo, fvv, b, (0, 3), controls, with_grad=True)
    seconds = time.perf_counter() - start
    report = dict(layout=args.layout, shape=[q, no, nv], factor_mib=b.nbytes / 2**20,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  solve_seconds=seconds, energy=result['energy'],
                  bar_norms={k: float(np.linalg.norm(v))
                             for k, v in result['cotangents'].items()},
                  diagnostics=result['diagnostics'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
