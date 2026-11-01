"""Compare saved dCMP results; this does not run a calculation."""
import json
import math
import sys
from pathlib import Path

job_id = sys.argv[1]
cases = ('baseline_ooc', 'optimized_ooc', 'optimized_incore')
records = {case: json.loads(Path(f'results.{job_id}.{case}.json').read_text())
           for case in cases}
baseline = records['baseline_ooc']
differences = {}
for case, result in records.items():
    for key in ('node', 'cpu_affinity', 'input_sha256', 'driver_sha256',
                'max_memory_mb', 'settings'):
        assert result[key] == baseline[key], (case, key)
    gradient = result['gradient_hartree_per_bohr']
    assert len(gradient) == 34 and all(len(row) == 3 for row in gradient)
    assert all(math.isfinite(x) for row in gradient for x in row)
    energy_delta = abs(result['energy_hartree'] - baseline['energy_hartree'])
    gradient_delta = max(abs(x - y)
                         for row, reference in zip(gradient, baseline['gradient_hartree_per_bohr'])
                         for x, y in zip(row, reference))
    assert energy_delta <= 1e-9, (case, energy_delta)
    assert gradient_delta <= 2e-7, (case, gradient_delta)
    differences[case] = {'energy_hartree': energy_delta,
                         'max_gradient_hartree_per_bohr': gradient_delta}
    print(f'{case}: total={result["total_seconds"]:.6f} s; '
          f'dE={energy_delta:.3e} Eh; max|dgrad|={gradient_delta:.3e} Eh/bohr')

for phase in baseline['phase_seconds']:
    print(phase, ' / '.join(f'{records[case]["phase_seconds"][phase]:.6f}'
                            for case in cases))
summary = {'job_id': job_id, 'node': baseline['node'],
           'case_order': cases, 'differences_from_baseline': differences,
           'total_seconds': {case: value['total_seconds'] for case, value in records.items()},
           'phase_seconds': {case: value['phase_seconds'] for case, value in records.items()},
           'peak_rss_gib': {case: value['peak_rss_kib'] / 2**20 for case, value in records.items()},
           'all_checks_passed': True}
Path(f'comparison.{job_id}.json').write_text(json.dumps(summary, indent=2) + '\n')
