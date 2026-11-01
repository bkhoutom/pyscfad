"""Validate repeated paired timings, socket placement and complete gradients."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
labels = ['main-a', 'fork-a', 'fork-b', 'main-b']
data = {label:json.loads((root/f'{label}-benchmark.json').read_text()) for label in labels}
first = data['main-a']
for label, result in data.items():
    assert result['hostname'] == first['hostname']
    assert result['job_id'] == first['job_id']
    assert result['affinity'] == first['affinity'] and len(result['affinity']) == 16
    assert result['cpu_topology'] == first['cpu_topology']
    assert len(result['gradient']) == 34 and all(len(row)==3 for row in result['gradient'])
    assert abs(result['energy']-first['energy']) < 1e-8
    error = max(abs(a-b) for row,ref in zip(result['gradient'],first['gradient'])
                for a,b in zip(row,ref))
    assert error < 1e-7 and result['gradient_max_error'] < 1e-7
    result['paired_gradient_max_error'] = error
    phases = result['phase_seconds']
    result['scf_forward_with_df'] = phases['df_forward']+phases['scf_forward']
    result['mp2_forward_and_reverse'] = phases['mp2_forward']+phases['mp2_pullback']
    result['scf_reverse_with_df'] = phases['scf_pullback']+phases['df_pullback']
    print(label, 'total', result['total_seconds'], 'SCF', result['scf_forward_with_df'],
          'MP2 both', result['mp2_forward_and_reverse'], 'SCF reverse', result['scf_reverse_with_df'])
for a,b in [('main-a','main-b'),('fork-a','fork-b')]:
    assert data[a]['source_hashes'] == data[b]['source_hashes']
    assert data[a]['native_hashes'] == data[b]['native_hashes']
averages = {branch:mean(data[f'{branch}-{run}']['total_seconds'] for run in ('a','b'))
            for branch in ('main','fork')}
report = dict(results=data, mean_seconds=averages,
              fork_over_main=averages['fork']/averages['main'],
              timing_note='Two cold fresh-process runs per branch, ABBA order, identical 16 physical cores and recorded socket/NUMA placement on one shared node; not a statistical confidence interval.')
(root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print('Mean total seconds:', averages, 'fork/main:', report['fork_over_main'])
