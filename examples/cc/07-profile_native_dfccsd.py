"""Profile resident CCSD from saved fragment tensors, with independent OMP/BLAS limits.

Inputs use the --inputs format of 06-native_dfccsd_benchmark.py. No SCF, AO2MO,
or disk operations are included in the measured solver phases. Use --no-profile
for uninstrumented thread scaling. Keep --cpus fixed when changing thread counts.
"""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--omp', type=int, default=32)
p.add_argument('--backend', choices=['native', 'jax'], default='native')
p.add_argument('--cpp', action=argparse.BooleanOptionalAction, default=True,
               help='Use optional C++ kernels when the extension is built')
p.add_argument('--blas', type=int, default=32)
p.add_argument('--cpus', default='0-31')
p.add_argument('--phase', choices=['both', 'forward', 'lambda'], default='both')
p.add_argument('--amplitudes', type=Path)
p.add_argument('--no-profile', action='store_true')
p.add_argument('--ladder', choices=['df', 'packed', 'auto'], default='df',
               help='DF tiles, diagnostic packed override, or production cache policy')
p.add_argument('--max-memory-mb', type=float, default=900000)
p.add_argument('--conv-tol', type=float, default=1e-12)
p.add_argument('--conv-tol-normt', type=float, default=1e-10)
a = p.parse_args()
cores = []
for part in a.cpus.split(','):
    ends = list(map(int, part.split('-')))
    cores.extend(range(ends[0], ends[-1]+1))
os.sched_setaffinity(0, cores)
os.environ['OMP_NUM_THREADS'] = str(a.omp)
os.environ['OPENBLAS_NUM_THREADS'] = str(a.blas)
os.environ['MKL_NUM_THREADS'] = str(a.blas)
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')

import numpy as np
import jax
from pyscf import lib
from pyscf.cc import ccsd
from threadpoolctl import threadpool_limits, threadpool_info
from pyscfad.cc import _dfccsd_native as native

a.output.parent.mkdir(parents=True, exist_ok=True)
with np.load(a.inputs) as data:
    arrays = [data[n] for n in native.ERI_NAMES]
    t1, t2 = data['t1'], data['t2']
options = dict(max_memory=a.max_memory_mb, max_cycle=200, conv_tol=a.conv_tol,
               conv_tol_normt=a.conv_tol_normt, iterative_damping=1., diis_space=3,
               diis_start_cycle=0, diis_start_energy_diff=1e9, level_shift=0.,
               verbose=4, stdout=sys.stdout, diis=True, threads=a.omp,
               vvvv_direct=True, cpp_kernels=a.cpp)  # cache setup is timed separately below
cc, eris = native.host_objects(options, arrays, t1.shape[0])
if a.backend == 'jax':
    from pyscfad.cc import dfccsd as ad_dfccsd, ccsd_lambda as ad_lambda
    host = cc
    cc = ad_dfccsd.RCCSD(host._scf)
    for key, value in options.items():
        if key != 'threads':
            setattr(cc, key, value)
    cc._scf = cc.with_df = None  # no SCF/DF builder is needed for saved ERIs
    cc._nocc, cc._nmo = t1.shape[0], sum(t1.shape)
    cc.incore_complete, cc.vvvv_direct = True, a.ladder == 'df'
    eris = ad_dfccsd._ChemistsERIs(host.mol)
    eris.nocc, eris.naux = t1.shape[0], arrays[-1].shape[0]
    for name, value in zip(native.ERI_NAMES, arrays):
        setattr(eris, name, jax.numpy.asarray(value))
    t1, t2 = jax.numpy.asarray(t1), jax.numpy.asarray(t2)
result = dict(backend=a.backend, cpp=a.cpp, omp=a.omp, blas=a.blas, affinity=cores, nocc=t1.shape[0],
              nvir=t1.shape[1], naux=eris.naux, max_memory_mb=a.max_memory_mb,
              profiled=not a.no_profile, ladder=a.ladder, phases={},
              conv_tol=a.conv_tol, conv_tol_normt=a.conv_tol_normt,
              input_sha256={n: hashlib.sha256(x.tobytes()).hexdigest()
                            for n, x in zip(native.ERI_NAMES, arrays)},
              environment={n: os.getenv(n) for n in ('OMP_WAIT_POLICY',
                  'GOMP_SPINCOUNT', 'OMP_DYNAMIC', 'OMP_THREAD_LIMIT')})

def measure(name, fn):
    profiler = cProfile.Profile()
    start, cpu = time.perf_counter(), time.process_time()
    if not a.no_profile:
        profiler.enable()
    value = fn()
    jax.block_until_ready(value)
    if not a.no_profile:
        profiler.disable()
    wall, used = time.perf_counter()-start, time.process_time()-cpu
    entry = dict(wall_s=wall, cpu_s=used, average_cpus=used/wall)
    if not a.no_profile:
        profiler.dump_stats(str(a.output.with_suffix('.'+name+'.pstats')))
        with a.output.with_suffix('.'+name+'.txt').open('w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative').print_stats(45)
            stats.sort_stats('tottime').print_stats(35)
        entry['functions'] = sorted([
            dict(file=k[0], line=k[1], name=k[2], calls=v[1], self_s=v[2], total_s=v[3])
            for k, v in stats.stats.items()], key=lambda x: -x['total_s'])
    result['phases'][name] = entry
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('PROFILE', name, wall, used, flush=True)
    return value

with lib.with_omp_threads(a.omp), threadpool_limits(a.omp, user_api='openmp'), \
        threadpool_limits(a.blas, user_api='blas'):
    result['pools'] = threadpool_info()
    if a.ladder in ('packed', 'auto') and a.backend == 'jax':
        measure('ladder_setup', lambda: ad_dfccsd._maybe_cache_vvvv(cc, eris))
        result['ladder_cache_mb'] = eris._vvvv_pair.size*8/1e6
    elif a.ladder == 'auto':
        cc.vvvv_direct = False
        measure('ladder_setup', lambda: native._maybe_cache_vvvv(cc, eris))
        result['ladder_cache_mb'] = 0 if eris.vvvv is None else eris.vvvv.nbytes/1e6
    elif a.ladder == 'packed':
        required_mb = 8*eris.vvL.shape[0]**2/1e6
        if required_mb > .25*max(0, cc.max_memory-lib.current_memory()[0]):
            raise MemoryError('Diagnostic virtual-pair cache exceeds memory allowance')
        packed = measure('ladder_setup', lambda: eris.vvL @ eris.vvL.T)
        result['ladder_cache_mb'] = required_mb
        def contract(mycc, tensor, direct=False, out=None, verbose=None):
            assert not direct
            return ccsd._contract_vvvv_t2(mycc, eris.mol, packed, tensor, out, verbose)
        eris._contract_vvvv_t2 = contract
    if a.phase in ('both', 'forward'):
        conv, energy, t1, t2 = measure('forward', lambda: ad_dfccsd._dfccsd_kernel_plain(
            cc, eris, t1, t2) if a.backend == 'jax' else ccsd.kernel(
            cc, eris, t1, t2, max_cycle=cc.max_cycle, tol=cc.conv_tol,
            tolnormt=cc.conv_tol_normt, verbose=cc.verbose))
        assert conv
        result.update(energy=float(energy), forward_iterations=getattr(cc, 'cycles', None))
        np.savez(a.output.with_suffix('.amplitudes.npz'), t1=t1, t2=t2)
    elif a.amplitudes:
        with np.load(a.amplitudes) as data:
            t1, t2 = data['t1'], data['t2']
    else:
        raise ValueError('--phase lambda requires --amplitudes')
    if a.phase in ('both', 'lambda'):
        rng = np.random.default_rng(515)
        bars = (.7, rng.normal(size=t1.shape)*.001, rng.normal(size=t2.shape)*.001)
        conv, vector = measure('lambda', lambda: ad_lambda.solve_response_lambda(
            cc, eris, t1, t2, *bars, max_cycle=cc.max_cycle, tol=cc.conv_tol_normt)
            if a.backend == 'jax' else native.solve_response(cc, eris, t1, t2, *bars))
        assert conv
        np.save(a.output.with_suffix('.response.npy'), vector)
a.output.write_text(json.dumps(result, indent=2)+'\n')
