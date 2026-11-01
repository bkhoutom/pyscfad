"""Compare complete gradients and timed phases for the two native builds."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
results = {name: json.loads((root / f'{name}-benchmark.json').read_text())
           for name in ('original', 'candidate')}
old, new = results['original'], results['candidate']
assert old['hostname'] == new['hostname']
assert old['affinity'] == new['affinity']
assert len(old['gradient']) == len(new['gradient']) == 34
energy_error = abs(old['energy'] - new['energy'])
gradient_error = max(abs(x-y) for a,b in zip(old['gradient'], new['gradient'])
                     for x,y in zip(a,b))
assert energy_error < 1e-8, energy_error
assert gradient_error < 1e-7, gradient_error
for name, result in results.items():
    text = (root / f'{name}.out').read_text()
    result['phase_seconds'] = {label: float(seconds) for label, seconds in
        re.findall(r'^(SCF forward[^:]*|MP2 forward[^:]*|MP2 pullback|SCF pullback[^:]*): ([\d.]+) s$', text, re.M)}
summary = dict(results=results, energy_error_hartree=energy_error,
               gradient_max_error_hartree_per_bohr=gradient_error,
               speedup=old['total_seconds']/new['total_seconds'],
               note='One run each on the same allocation; compiler, BLAS backend and effective BLAS threading differ.')
(root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({name: {'seconds': r['total_seconds'], 'phases': r['phase_seconds']}
                  for name, r in results.items()}, indent=2))
print('Energy difference:', energy_error)
print('Maximum gradient difference:', gradient_error)
print('Speedup:', summary['speedup'])
