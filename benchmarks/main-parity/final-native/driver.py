"""Validate the consolidated native package and the changed 3c work schedule."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

condition = Path(__file__).resolve().parent
spec = json.loads((condition / 'input.json').read_text())
assert os.environ.get('SLURM_JOB_ID'), 'Run numerical work through submit.sh'
source = Path(spec['source'])
native = condition / 'pyscfadlib'
sys.meta_path[:] = [finder for finder in sys.meta_path
                   if not finder.__module__.startswith(('_editable_skbc_pyscfadlib',
                                                        '__editable___pyscfad_'))]
sys.path[:0] = [str(condition), str(source), spec['pyscf_parent']]
import jax
import numpy as np
from pyscf import lib
import pyscf
import pyscfad
import pyscfadlib
from pyscfad import config_update, gto, scf
from pyscfad.df import addons, _int3c_cross_opt
from pyscfad.mp import dfmp2
from threadpoolctl import threadpool_info

assert Path(pyscfad.__file__).is_relative_to(source)
assert Path(pyscfadlib.__file__).is_relative_to(native)
assert Path(pyscf.__file__).is_relative_to(Path(spec['pyscf_parent']))
assert len(os.sched_getaffinity(0)) == 16
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
native_hashes = {str(p.relative_to(native)): digest(p)
                 for p in sorted(native.rglob('*.so*')) if p.is_file()}
baseline = json.loads((condition / 'baseline-native-sha256.json').read_text())
for path, sha in baseline.items():
    if path.startswith('thirdparty/'):
        assert native_hashes[path] == sha, f'Pinned dependency changed: {path}'
libraries = sorted(native.glob('*.so'))
assert len(libraries) == 10, libraries
linkage = {p.name: subprocess.check_output(['ldd', str(p)], text=True)
           for p in libraries}
assert all('not found' not in text for text in linkage.values())
result = dict(job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
              affinity=sorted(os.sched_getaffinity(0)), python=sys.executable,
              imports=dict(pyscf=pyscf.__file__, pyscfad=pyscfad.__file__,
                           pyscfadlib=pyscfadlib.__file__),
              native_sha256=native_hashes, linkage=linkage,
              threadpools=threadpool_info(),
              schedule_source_sha256=digest(native / 'vjp/gto/fill_nr_3c_vjp.c'))
(condition / 'native-sha256.json').write_text(json.dumps(native_hashes, indent=2)+'\n')
print('Validated native imports, ten libraries, linkage, and pinned dependencies.', flush=True)

def molecule(basis):
    mol = gto.Mole(atom=spec['atom'], basis=basis, verbose=0, max_memory=12000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol

def displaced(mol, direction, step):
    moved = mol.copy()
    moved.set_geom_(np.asarray(mol.coords) + step*direction, unit='Bohr')
    return moved

mol = molecule(spec['integral_basis'])
auxmol = addons.make_auxmol(mol, spec['integral_auxbasis'])
rng = np.random.default_rng(spec['seed'])
cotangent = jax.numpy.asarray(rng.normal(size=(mol.nao*(mol.nao+1)//2, auxmol.nao)))

def integral_objective(mol_, auxmol_):
    return jax.numpy.sum(_int3c_cross_opt.int3c_cross(mol_, auxmol_)*cotangent)

gradients = []
schedules = [(None, 1), (None, 16), (None, 16), ('dynamic', 1),
             ('dynamic', 16), ('dynamic', 16), ('unknown', 16), ('static', 16)]
for schedule, threads in schedules:
    if schedule is None:
        os.environ.pop('PYSCFAD_INTEGRAL_VJP_SCHEDULE', None)
    else:
        os.environ['PYSCFAD_INTEGRAL_VJP_SCHEDULE'] = schedule
    lib.num_threads(threads)
    assert lib.num_threads() == threads
    _, bars = jax.value_and_grad(integral_objective, argnums=(0, 1))(mol, auxmol)
    gradients.append(tuple(np.asarray(bar.coords) for bar in bars))
    print(f'Completed 3c VJP with {threads} OpenMP threads, schedule={schedule}.', flush=True)
for gradient in gradients[1:]:
    for actual, expected in zip(gradient, gradients[0]):
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)
# The unchanged default and unknown-value fallback must retain bitwise
# repeatability for a fixed thread count; dynamic scheduling promises tolerance.
for repeat in [2, 6, 7]:
    for actual, expected in zip(gradients[repeat], gradients[1]):
        np.testing.assert_array_equal(actual, expected)
os.environ['PYSCFAD_INTEGRAL_VJP_SCHEDULE'] = 'dynamic'
directions = [rng.normal(size=x.shape) for x in gradients[0]]
directions = [x/np.linalg.norm(x) for x in directions]
step = spec['finite_difference_step']
analytic = sum(float(np.sum(g*d)) for g, d in zip(gradients[0], directions))
plus = integral_objective(displaced(mol, directions[0], step),
                          displaced(auxmol, directions[1], step))
minus = integral_objective(displaced(mol, directions[0], -step),
                           displaced(auxmol, directions[1], -step))
finite_difference = float((plus-minus)/(2*step))
assert abs(analytic-finite_difference) < 3e-6, (analytic, finite_difference)
result['int3c'] = dict(gradient_1=[x.tolist() for x in gradients[0]],
                      gradient_16_dynamic=[x.tolist() for x in gradients[4]],
                      default_static_repeat_bitwise_equal=True,
                      unknown_schedule_falls_back_to_static=True,
                      max_thread_error=max(float(np.max(abs(a-b)))
                                           for a,b in zip(gradients[0], gradients[1])),
                      max_dynamic_static_error=max(float(np.max(abs(a-b)))
                                           for a,b in zip(gradients[1], gradients[4])),
                      max_dynamic_repeat_error=max(float(np.max(abs(a-b)))
                                           for a,b in zip(gradients[4], gradients[5])),
                      directional_ad=analytic, directional_fd=finite_difference)
print('Three-centre thread/repeat/finite-difference checks passed:', result['int3c'], flush=True)

mol = molecule(spec['mp2_basis'])
def energy(mol_):
    mf = scf.RHF(mol_).density_fit(auxbasis=spec['mp2_auxbasis'])
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged
    return mf.e_tot + dfmp2.MP2(mf, frozen=1).kernel(with_t2=False)[0]

with (config_update('pyscfad_moleintor_opt', True),
      config_update('pyscfad_scf_implicit_diff', True),
      config_update('pyscfad_scf_df_precond', True)):
    value, gradient = jax.value_and_grad(energy)(mol)
    direction = rng.normal(size=(mol.natm, 3))
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(np.asarray(gradient.coords)*direction))
    finite_difference = float((energy(displaced(mol, direction, step))
                               - energy(displaced(mol, direction, -step)))/(2*step))
assert abs(analytic-finite_difference) < 2e-7, (analytic, finite_difference)
result['water_mp2'] = dict(energy=float(value), gradient=np.asarray(gradient.coords).tolist(),
                          directional_ad=analytic, directional_fd=finite_difference,
                          absolute_error=abs(analytic-finite_difference))
result['status'] = 'passed'
(condition / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
print('Water DF-MP2 finite-difference check passed:', result['water_mp2'], flush=True)
