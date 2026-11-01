"""Compare energy and every coordinate-gradient component from the three runs."""
import json
from pathlib import Path
import numpy as np

reference = json.loads(Path('incore.json').read_text())
outcore = json.loads(Path('outcore.json').read_text())
plain = json.loads(Path('pyscf.json').read_text())
grad_ref = np.asarray(reference['gradient'])
grad_ooc = np.asarray(outcore['gradient'])
assert grad_ref.shape == grad_ooc.shape == (6, 3)
assert np.isfinite(grad_ref).all() and np.isfinite(grad_ooc).all()
assert all(np.isfinite(x['energy']) for x in (reference, outcore, plain))
errors = {
    'outcore_vs_incore_energy_hartree': abs(outcore['energy'] - reference['energy']),
    'max_gradient_error_hartree_per_bohr': float(np.max(np.abs(grad_ooc - grad_ref))),
    'plain_vs_ad_energy_hartree': abs(plain['energy'] - outcore['energy']),
}
print(json.dumps(errors, indent=2), flush=True)
assert errors['outcore_vs_incore_energy_hartree'] < 1e-9
assert errors['max_gradient_error_hartree_per_bohr'] < 2e-7
assert errors['plain_vs_ad_energy_hartree'] < 1e-9
Path('verification.json').write_text(json.dumps({'passed': True, **errors}, indent=2) + '\n')
print('PASS: disk-backed and in-core energies/full gradients agree.', flush=True)
