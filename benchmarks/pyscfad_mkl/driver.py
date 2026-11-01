"""Validate isolated oldenv builds on water, including a frozen-core MP2 derivative."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / 'input.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(variant):
    # Editable finders otherwise take precedence over the isolated PYTHONPATH.
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if not getattr(finder, '__module__', type(finder).__module__).startswith(
                           ('_editable_skbc_pyscfadlib', '__editable___pyscfad_'))]
    import numpy as np
    import jax
    import pyscf
    import pyscfad
    import pyscfadlib
    from pyscf import gto, scf
    from pyscf.mp import dfmp2
    from pyscfad import config, gto as ad_gto, scf as ad_scf
    from pyscfad.mp import dfmp2 as ad_dfmp2
    from threadpoolctl import threadpool_info

    expected = ROOT / variant
    assert Path(pyscf.__file__).resolve() == expected / 'pyscf/__init__.py'
    assert Path(pyscfadlib.__file__).resolve() == expected / 'pyscfadlib/__init__.py'
    assert Path(pyscfad.__file__).resolve() == ROOT / 'python/pyscfad/__init__.py'
    jax.config.update('jax_enable_x64', True)
    config.update('pyscfad_moleintor_opt', True)
    config.update('pyscfad_scf_implicit_diff', True)
    settings = SPEC['water']

    def configure(mf):
        mf.conv_tol = settings['conv_tol']
        mf.conv_tol_grad = settings['conv_tol_grad']
        mf.max_cycle = settings['max_cycle']
        mf.chkfile = None
        return mf

    def conventional(molecule):
        mf = configure(scf.RHF(molecule).density_fit(auxbasis=settings['auxbasis']))
        mf.kernel()
        assert mf.converged
        mp = dfmp2.MP2(mf, frozen=settings['frozen'])
        mp.kernel(with_t2=False)
        return float(mp.e_tot), mf

    mol = gto.M(atom=settings['atom'], unit=settings['unit'], basis=settings['basis'],
                verbose=0, max_memory=3000)
    ref_energy, ref_mf = conventional(mol)
    grad_method = ref_mf.nuc_grad_method()
    grad_method.auxbasis_response = True
    analytic_rhf_grad = grad_method.kernel()
    ad_mol = ad_gto.Mole(atom=settings['atom'], unit=settings['unit'],
                          basis=settings['basis'], verbose=0, max_memory=3000)
    ad_mol.build(trace_coords=True, trace_exp=False, trace_ctr_coeff=False)

    def rhf_energy(current):
        mf = configure(ad_scf.RHF(current).density_fit(auxbasis=settings['auxbasis']))
        value = mf.kernel()
        return value, mf.converged

    (ad_rhf_e, converged), ad_rhf_grad = jax.value_and_grad(rhf_energy, has_aux=True)(ad_mol)
    jax.block_until_ready((ad_rhf_e, ad_rhf_grad))
    assert bool(converged)
    rhf_error = float(np.max(np.abs(np.asarray(ad_rhf_grad.coords) - analytic_rhf_grad)))
    assert abs(float(ad_rhf_e) - ref_mf.e_tot) < settings['energy_tolerance']
    assert rhf_error < settings['gradient_tolerance'], rhf_error

    def scf_state(current):
        mf = configure(ad_scf.RHF(current).density_fit(auxbasis=settings['auxbasis']))
        mf.kernel()
        return (mf.mol, mf.with_df, mf.e_tot, mf.mo_coeff, mf.mo_energy), mf.converged

    def mp2_energy(state):
        molecule, with_df, ehf, mo_coeff, mo_energy = state
        mf = ad_scf.RHF(molecule).density_fit(with_df=with_df)
        mf.e_tot, mf.mo_coeff, mf.mo_energy = ehf, mo_coeff, mo_energy
        mf.mo_occ = mf.get_occ()
        mf.converged = True
        mp = ad_dfmp2.MP2(mf, frozen=settings['frozen'])
        mp.kernel(with_t2=False)
        return mp.e_tot

    timings = []
    for phase in ('first', 'warm'):
        start = time.perf_counter()
        state, scf_pullback, converged = jax.vjp(scf_state, ad_mol, has_aux=True)
        assert bool(converged)
        energy, mp2_pullback = jax.vjp(mp2_energy, state)
        state_bar, = mp2_pullback(jax.numpy.ones_like(energy))
        gradient, = scf_pullback(state_bar)
        jax.block_until_ready((energy, gradient))
        timings.append({'phase': phase, 'seconds': time.perf_counter() - start})
        assert abs(float(energy) - ref_energy) < settings['energy_tolerance']
        assert np.isfinite(np.asarray(gradient.coords)).all()

    coordinates = mol.atom_coords()
    atom, axis, step = settings['fd_atom'], settings['fd_axis'], settings['fd_step_bohr']
    displaced_energies = []
    for direction in (1, -1):
        displaced = coordinates.copy()
        displaced[atom, axis] += direction * step
        new_mol = mol.set_geom_(displaced, unit='Bohr', inplace=False)
        displaced_energies.append(conventional(new_mol)[0])
    fd = (displaced_energies[0] - displaced_energies[1]) / (2 * step)
    fd_error = abs(float(gradient.coords[atom, axis]) - fd)
    assert fd_error < settings['gradient_tolerance'], (gradient.coords[atom, axis], fd, fd_error)

    dependencies = {}
    for library in sorted((expected / 'pyscfadlib').glob('*.so')):
        output = subprocess.check_output(['ldd', str(library)], text=True)
        assert 'not found' not in output
        if variant == 'candidate':
            assert 'flexiblas' not in output.lower()
            if library.name.startswith(('libcgto_ad', 'libcc_vjp', 'libcvhf_vjp', 'pyscfad_lapack')):
                assert 'libmkl_sequential.so' in output, output
                assert 'libiomp' not in output, output
        dependencies[library.name] = output
    if variant == 'candidate':
        for name in ('libnp_helper.so', 'libmp.so', 'libcvhf.so'):
            output = subprocess.check_output(['ldd', str(expected / 'pyscf/lib' / name)], text=True)
            assert 'libmkl_sequential.so' in output and 'not found' not in output
            assert 'libiomp' not in output
            dependencies['pyscf/' + name] = output
    for rel, key in [('pyscfadlib/thirdparty/lib/libcintad.so', 'retained_libcintad_sha256'),
                     ('pyscf/lib/deps/lib/libcint.so', 'retained_pyscf_libcint_sha256'),
                     ('pyscf/lib/libsolvent.so', 'retained_solvent_sha256')]:
        assert digest(expected / rel) == SPEC[key], rel

    return {'status': 'ok', 'variant': variant, 'python': sys.executable,
            'pyscf_path': pyscf.__file__, 'pyscfad_path': pyscfad.__file__,
            'pyscfadlib_path': pyscfadlib.__file__, 'hostname': socket.gethostname(),
            'job': os.environ['SLURM_JOB_ID'], 'energy_hartree': float(energy),
            'gradient_hartree_per_bohr': np.asarray(gradient.coords).tolist(),
            'rhf_analytic_gradient_error': rhf_error,
            'mp2_reference_energy_error': abs(float(energy) - ref_energy),
            'mp2_fd_gradient': fd, 'mp2_fd_gradient_error': fd_error,
            'timings': timings, 'threadpools': threadpool_info(), 'dependencies': dependencies}


if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith('login'):
    raise SystemExit('Run scientific validation inside submit.sh on a compute node.')
if len(sys.argv) == 2:
    result = run(sys.argv[1])
    (ROOT / 'results' / (sys.argv[1] + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
else:
    for variant in ('baseline', 'candidate'):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / variant) + os.pathsep + str(ROOT / 'python')
        subprocess.run([sys.executable, '-u', str(ROOT / 'driver.py'), variant],
                       env=env, check=True, cwd=ROOT)
    rows = {name: json.loads((ROOT / 'results' / (name + '.json')).read_text())
            for name in ('baseline', 'candidate')}
    energy_error = abs(rows['candidate']['energy_hartree'] - rows['baseline']['energy_hartree'])
    gradient_error = max(abs(a - b) for x, y in zip(rows['candidate']['gradient_hartree_per_bohr'],
                                                   rows['baseline']['gradient_hartree_per_bohr'])
                         for a, b in zip(x, y))
    assert energy_error < SPEC['water']['energy_tolerance']
    assert gradient_error < SPEC['water']['gradient_tolerance']
    result = {'status': 'ok', 'job': os.environ['SLURM_JOB_ID'],
              'energy_difference': energy_error, 'gradient_difference': gradient_error,
              'variants': rows,
              'scope': 'Water correctness/ABI check; not a representative dCMP performance benchmark.'}
    (ROOT / 'results/summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Isolated baseline and MKL water validations passed.', flush=True)
