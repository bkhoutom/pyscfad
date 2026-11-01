"""Exercise the benchmark drivers on frozen-core water and check the gradient."""
import json
import runpy
from pathlib import Path

import numpy as np

ad = runpy.run_path('reproduce_dcmp_ad.py')
ref = runpy.run_path('pyscf_scf.py')
energy_error = abs(float(ad['e']) - float(ref['e']))
gradient = np.asarray(ad['grad'])
assert np.isfinite(gradient).all()
assert energy_error < 1e-8, energy_error

h = 1e-3  # bohr; PySCFAD coordinate gradients are in Hartree / bohr.
coordinates = ref['mol'].atom_coords()
displaced_energies = []
for direction in (1, -1):
    displaced = coordinates.copy()
    displaced[1, 2] += direction * h
    mol = ref['mol'].set_geom_(displaced, unit='Bohr', inplace=False)
    mol.verbose = 0
    displaced_energies.append(float(ref['energy'](mol)))
finite_difference = (displaced_energies[0] - displaced_energies[1]) / (2 * h)
gradient_error = abs(gradient[1, 2] - finite_difference)
assert gradient_error < 2e-6, (gradient[1, 2], finite_difference, gradient_error)
result = {
    'energy_error_hartree': energy_error,
    'gradient_ad_hartree_per_bohr': float(gradient[1, 2]),
    'gradient_fd_hartree_per_bohr': finite_difference,
    'gradient_error_hartree_per_bohr': float(gradient_error),
}
Path('results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
print('Both benchmark drivers and the frozen-core gradient check passed.')
