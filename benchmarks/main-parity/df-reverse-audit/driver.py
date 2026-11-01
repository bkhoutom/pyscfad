"""Compare native static/dynamic schedules with identical dCMP integral work."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
spec = json.loads((root / 'input.json').read_text())
assert os.environ.get('SLURM_JOB_ID'), 'Run numerical work in the supplied batch job'
sys.path.insert(0, spec['pyscf_parent'])
import numpy as np
from pyscf import df, gto, lib
from pyscf.gto.moleintor import ascint3, make_loc, make_cintopt
from threadpoolctl import threadpool_info

assert Path(gto.__file__).resolve().parents[1] == Path(spec['pyscf_parent']) / 'pyscf'
assert lib.num_threads() == spec['threads'] == len(os.sched_getaffinity(0))
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, expected in spec['native_sha256'].items():
    assert digest(path) == expected, path
native = Path(spec['native'])
handles = {name: ctypes.CDLL(str(root / f'libschedule_{name}.so'))
           for name in ('static', 'dynamic')}
integral = ctypes.CDLL(str(native / 'libcgto_vjp.so'))
mol = gto.M(atom=spec['atom'], basis=spec['basis'], verbose=0)
auxmol = df.addons.make_auxmol(mol, spec['auxbasis'])
pmol = mol + auxmol
atm = np.asarray(pmol._atm, dtype=np.int32, order='C')
bas = np.asarray(pmol._bas, dtype=np.int32, order='C')
env = np.asarray(pmol._env, dtype=np.float64, order='C')
intor = ascint3(mol._add_suffix('int3c2e_ip1'))
ao_loc = np.asarray(make_loc(bas, intor), dtype=np.int32, order='C')
cintopt = make_cintopt(atm, bas, env, intor)
ptr = lambda x: x.ctypes.data_as(ctypes.c_void_p)
npair = mol.nao * (mol.nao + 1) // 2
rng = np.random.default_rng(spec['seed'])
blocks = []
for atom, (b0, b1, p0, p1) in enumerate(auxmol.aoslice_by_atom()):
    if b0 == b1:
        continue
    # C(aux,pair).T is exactly F(pair,aux), matching both native interfaces.
    ybar = (rng.standard_normal((p1-p0, npair)) *
            spec['synthetic_cotangent_scale']).T
    assert ybar.flags.f_contiguous
    assert np.shares_memory(ybar, np.asarray(ybar, order='F'))
    blocks.append((atom, int(b0), int(b1), ybar))

metadata = dict(job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
    affinity=sorted(os.sched_getaffinity(0)), python=sys.executable,
    nao=mol.nao, naux=auxmol.nao, natm=mol.natm, nbas=mol.nbas,
    aux_nbas=auxmol.nbas, calls_per_pass=len(blocks),
    cotangent_bytes=sum(x.nbytes for _, _, _, x in blocks),
    threadpools=threadpool_info(),
    native_sha256=spec['native_sha256'],
    diagnostic_libraries={name: {'sha256': digest(root / f'libschedule_{name}.so'),
        'ldd': subprocess.check_output(['ldd', str(root / f'libschedule_{name}.so')], text=True)}
        for name in handles},
    runs=[])
metadata['cpu_topology'] = [dict(cpu=c,
    socket=int(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text()),
    core=int(Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text()))
    for c in metadata['affinity']]
print(json.dumps({k:v for k,v in metadata.items() if k != 'diagnostic_libraries'}, indent=2), flush=True)
reference = None
for name in spec['order']:
    handle = handles[name]
    driver = handle.GTOnr3c_ij_r0_vjp
    driver.restype = None
    fill = handle.GTOnr3c_ij_r0_vjp_s2ij
    primary = np.zeros((mol.natm, 3))
    auxiliary = np.zeros((auxmol.natm, 3))
    atom_times = []
    start, cpu_start = time.perf_counter(), time.process_time()
    for atom, b0, b1, ybar in blocks:
        result = np.zeros_like(primary)
        shell_slice = (ctypes.c_int * 6)(0, mol.nbas, 0, mol.nbas,
                                        mol.nbas + b0, mol.nbas + b1)
        atom_start = time.perf_counter()
        driver(getattr(integral, intor), fill, ptr(result), ptr(ybar),
               ctypes.c_int(3), shell_slice, ptr(ao_loc), cintopt,
               ptr(atm), ctypes.c_int(len(atm)), ctypes.c_int(mol.natm),
               ptr(bas), ctypes.c_int(len(bas)), ptr(env))
        atom_times.append(time.perf_counter() - atom_start)
        primary += result
        auxiliary[atom] = -result.sum(axis=0)
    elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu_start
    gradient = np.concatenate((primary.ravel(), auxiliary.ravel()))
    assert np.isfinite(gradient).all()
    if reference is None:
        reference = gradient.copy()
    error = float(np.max(np.abs(gradient-reference)))
    assert np.allclose(gradient, reference, rtol=2e-11, atol=2e-11), error
    result = dict(schedule=name, seconds=elapsed, cpu_seconds=cpu_elapsed,
                  max_error=error, atom_seconds=atom_times,
                  gradient=gradient.tolist())
    metadata['runs'].append(result)
    print(json.dumps({k:v for k,v in result.items() if k != 'gradient'}), flush=True)
    (root / 'results.json').write_text(json.dumps(metadata, indent=2)+'\n')
means = {name: float(np.mean([r['seconds'] for r in metadata['runs']
                             if r['schedule'] == name])) for name in handles}
metadata.update(status='ok', mean_seconds=means,
                static_over_dynamic=means['static']/means['dynamic'])
(root / 'results.json').write_text(json.dumps(metadata, indent=2)+'\n')
print('All scheduling comparisons passed:', means, flush=True)
