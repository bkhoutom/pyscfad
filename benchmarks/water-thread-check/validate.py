"""Compare the job's threaded DF path with the original STC gradient path."""
import json
from pathlib import Path

import jax
import numpy as np
from pyscfad import dlno_stc, gto, scf
import compare_mp2_LT as candidate

mol = gto.Mole(atom='water_2.xyz', basis='sto-3g', verbose=0, max_memory=8000)
mol.build(trace_exp=False, trace_ctr_coeff=False)
roots, weights = np.polynomial.legendre.leggauss(64)
controls = {'mode': 'deterministic', 'laplace_roots': 20 * (roots + 1),
            'laplace_weights': 20 * weights}
values = []
for label, build in [('original', lambda m: scf.RHF(m).density_fit().run()),
                     ('threaded', candidate.build_mf)]:
    candidate.check_pools(1)
    energy, gradient = dlno_stc.value_and_grad(
        mol, build, scope='system', frozen=None, controls=controls, include_hf=True)
    jax.block_until_ready((energy, gradient))
    values.append((float(energy), np.asarray(gradient.coords)))
    print(label, values[-1][0], values[-1][1], flush=True)
    candidate.check_pools(1)
energy_error = abs(values[1][0] - values[0][0])
gradient_error = float(np.max(np.abs(values[1][1] - values[0][1])))
np.testing.assert_allclose(values[1][0], values[0][0], atol=2e-10, rtol=0)
np.testing.assert_allclose(values[1][1], values[0][1], atol=2e-8, rtol=0)
assert set(candidate.phase_seconds) == {'df_forward', 'df_reverse'}
result = {'energy_error': energy_error, 'gradient_max_error': gradient_error,
          'threads': candidate.NTHREADS, 'scipy_df_threads': candidate.DF_THREADS,
          'df_phase_seconds': candidate.phase_seconds}
Path('validation.json').write_text(json.dumps(result, indent=2) + '\n')
print('PASS', result, flush=True)
