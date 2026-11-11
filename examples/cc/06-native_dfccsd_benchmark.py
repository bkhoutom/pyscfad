"""Compare resident fragment CCSD forward/response at a fixed CPU allocation.

Run each backend in a fresh process, for example:
  OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  python examples/cc/06-native_dfccsd_benchmark.py --backend native --threads 32

The fragment model is a water cluster in a frozen-core DF active space.
Reported response derivatives include an arbitrary amplitude source, as in
projected energies/(T). SCF/integral setup is excluded from kernel timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=('native', 'jax'), required=True)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--waters', type=int, default=4)
parser.add_argument('--basis', default='cc-pvdz')
parser.add_argument('--vvvv-direct', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--inputs', type=Path,
                    help='Reuse identical ERIs/initial amplitudes; create if absent')
args = parser.parse_args()
# Select physical cores before JAX creates its runtime workers.
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key = ((topology/'physical_package_id').read_text(),
           (topology/'core_id').read_text())
    cores.setdefault(key, cpu)
if len(cores) < args.threads:
    raise RuntimeError('Requested more physical cores than available')
os.sched_setaffinity(0, list(cores.values())[:args.threads])
os.environ['OMP_NUM_THREADS'] = str(args.threads)
os.environ['OMP_THREAD_LIMIT'] = str(args.threads)
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')

import jax
import numpy as np
from pyscf import gto, scf, lib
from threadpoolctl import threadpool_info
from pyscfad.cc import dfccsd, ccsd_lambda
from pyscfad.cc import _dfccsd_native as native

lib.num_threads(args.threads)
atom = []
for k in range(args.waters):
    x, z = 2.9*(k % 2), 2.9*(k // 2)
    atom.extend([('O', (x, 0., z)), ('H', (x+.1, -.75, z+.57)),
                 ('H', (x, .8, z+.61))])
mol = gto.M(atom=atom, basis=args.basis, verbose=0, max_memory=24000)
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-12
mf.kernel()
assert mf.converged
cc = dfccsd.RCCSD(mf, frozen=args.waters)
cc.native_cpu = args.backend == 'native'
cc.incore_complete = True
cc.vvvv_direct = args.vvvv_direct
cc.diis_space = 3
cc.max_cycle = 100
cc.conv_tol = 1e-12
cc.conv_tol_normt = 1e-10
eris = cc.ao2mo()
t1, t2 = cc.init_amps(eris)[1:]
if args.inputs is not None:
    if args.inputs.exists():
        with np.load(args.inputs) as data:
            for name in native.ERI_NAMES:
                setattr(eris, name, jax.numpy.asarray(data[name]))
            t1, t2 = jax.numpy.asarray(data['t1']), jax.numpy.asarray(data['t2'])
    else:
        args.inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.inputs, t1=np.asarray(t1), t2=np.asarray(t2),
                 **{name: np.asarray(getattr(eris, name)) for name in native.ERI_NAMES})
jax.block_until_ready((eris, t1, t2))
result = dict(backend=args.backend, threads=args.threads, waters=args.waters,
              basis=args.basis, nocc=t1.shape[0], nvir=t1.shape[1],
              naux=eris.Lvv.shape[0], affinity=sorted(os.sched_getaffinity(0)),
              jax_version=jax.__version__, pools=threadpool_info(), timings={})
result['convergence'] = dict(energy=cc.conv_tol, amplitudes=cc.conv_tol_normt,
                             max_cycle=cc.max_cycle, diis_space=cc.diis_space)
result['vvvv_direct'] = cc.vvvv_direct
result['input_sha256'] = {name: hashlib.sha256(
    np.asarray(getattr(eris, name)).tobytes()).hexdigest() for name in native.ERI_NAMES}
result['environment'] = {k: os.environ.get(k) for k in (
    'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_WAIT_POLICY',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'XLA_FLAGS')}
result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path(dfccsd.__file__).parent.glob('*native*.py')}
def measure(name, fn):
    start, cpu = time.perf_counter(), time.process_time()
    value = fn()
    jax.block_until_ready(value)
    wall, used = time.perf_counter()-start, time.process_time()-cpu
    result['timings'][name] = dict(wall_s=wall, cpu_s=used, average_cpus=used/wall)
    print(name, result['timings'][name], flush=True)
    return value
conv, energy, t1, t2 = measure('forward', lambda: (
    native.forward(cc, eris, t1, t2) if cc.native_cpu
    else dfccsd._dfccsd_kernel_plain(cc, eris, t1, t2)))
assert conv
result['energy'] = float(energy)
rng = np.random.default_rng(515)
bars = (.7, rng.normal(size=t1.shape)*.001, rng.normal(size=t2.shape)*.001)
if cc.native_cpu:
    from threadpoolctl import threadpool_limits
    with threadpool_limits(args.threads):
        result['active_pools'] = threadpool_info()
        host, he = native.host_objects(native.settings(cc),
            [getattr(eris, n) for n in native.ERI_NAMES], t1.shape[0])
        a, b = np.asarray(t1), np.asarray(t2)
        conv, vector = measure('lambda', lambda: native.solve_response(host, he, a, b, *bars))
        gradients = measure('integral_adjoint', lambda: native.integral_cotangent(host, he, a, b, bars[0], vector))
else:
    conv, vector, saved = measure('lambda', lambda: ccsd_lambda.solve_response_lambda(
        cc, eris, t1, t2, *bars, max_cycle=cc.max_cycle, tol=cc.conv_tol_normt,
        return_intermediates=True))
    gradient_eris = measure('integral_adjoint', lambda: dfccsd.lagrangian_grad(
        cc, eris, t1, t2, bars[0], vector, intermediates=saved))
    gradients = tuple(getattr(gradient_eris, name) for name in native.BAR_NAMES)
assert conv
result['max_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output.with_suffix('.npz'), t1=np.asarray(t1), t2=np.asarray(t2),
         response=np.asarray(vector), **{name: np.asarray(g) for name, g in zip(native.BAR_NAMES, gradients)})
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
