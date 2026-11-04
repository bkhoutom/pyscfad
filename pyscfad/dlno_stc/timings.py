"""Additive wall-time reports for the serial domain MP2/LNO-CCSD(T) workflow.

Raw timers are inclusive. The report partitions their actual intervals, assigning
overlap to the innermost named stage. Native sub-timers divide fused operations
without repeating any scientific work.
"""
from collections import defaultdict
from contextlib import AbstractContextManager
import csv
import json
from pathlib import Path
import time

from pyscfad.tools import resource_profile


STAGES = ('SCF', 'Domain', 'MP2', 'Density matrix (W)', 'LIS', 'CCSD', 'CCSD(T)')
PHASES = {
    'water.df_factors': ('Forward', 'SCF'),
    'water16.df_factors': ('Forward', 'SCF'),
    'stc.domain.scf': ('Forward', 'SCF'),
    'stc.domain.scf_data_prepare': ('Forward', 'Domain'),
    'stc.domain.selection': ('Forward', 'Domain'),
    'stc.domain.common_prepare': ('Forward', 'Domain'),
    'stc.cc.prepare': ('Forward', 'Domain'),
    'stc.cc.baseline': ('Forward', 'MP2'),
    'stc.cc.matrix_forward': ('Forward', 'Density matrix (W)'),
    'stc.cc.lis_selection': ('Forward', 'LIS'),
    'stc.lis.build': ('Forward', 'LIS'),
    'stc.cc.correction_forward': ('Forward', 'Domain'),
    'solver.df_ao2mo_and_eris': ('Forward', 'Domain'),
    'solver.mp2_reference': ('Forward', 'MP2'),
    'solver.ccsd_iterations_and_energy': ('Forward', 'CCSD'),
    'solver.perturbative_triples': ('Forward', 'CCSD(T)'),
    'stc.triples.pullback_scale': ('Reverse', 'CCSD(T)'),
    'ccsd_bwd.total': ('Reverse', 'CCSD'),
    'stc.lis.pullback': ('Reverse', 'LIS'),
    'solver.mp2_reference_pullback': ('Reverse', 'MP2'),
    'stc.cc.correction_pullback': ('Reverse', 'Domain'),
    'stc.cc.matrix_pullback': ('Reverse', 'Density matrix (W)'),
    'stc.cc.input_pullback': ('Reverse', 'Domain'),
    'stc.cc.shared_df_integral_pullback': ('Reverse', 'Domain'),
    'stc.domain.common_pullback': ('Reverse', 'Domain'),
    'stc.domain.scf_response': ('Reverse', 'SCF'),
    'backward.local_DF_MO-coeff_backward': ('Reverse', 'Domain'),
    'stc.cc.weak_forward': ('Forward', 'MP2'),
    'stc.cc.weak_pullback': ('Reverse', 'MP2'),
    'stc.domain.weak_prepare': ('Forward', 'Domain'),
    'stc.domain.weak_pullback': ('Reverse', 'Domain'),
    'stc.cc.weak_pair_value_and_grad': ('Shared', 'MP2 weak pairs (forward + reverse)'),
}
for _scope in ('local DF', 'fragment DF', 'global DF'):
    for _operation in ('MO-coeff backward', 'AO2MO MO-coeff backward',
                       'integral derivative backward'):
        PHASES[f'backward.{_scope} {_operation}'.replace(' ', '_')] = ('Reverse', 'Domain')


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Cannot serialize timing detail {type(value).__name__}')


def summarize(events, *, started=None, ended=None, complete=False):
    """Partition completed timer intervals; preserve partial-run uncertainty."""
    events = [dict(e, phase=e['phase'].replace(' ', '_')) for e in events]
    measured = [e for e in events if e.get('start_wall_s') is not None]
    intervals = [e for e in measured if e['phase'] in PHASES]
    if started is None:
        started = min((e['start_wall_s'] for e in intervals), default=0.)
    if ended is None:
        ended = max((e['end_wall_s'] for e in events), default=started)
    elapsed = max(0., ended-started)
    totals, fragments = defaultdict(float), defaultdict(lambda: defaultdict(float))
    phase_totals = defaultdict(lambda: dict(calls=0, inclusive_wall_seconds=0.,
                                           exclusive_wall_seconds=0., inclusive_cpu_seconds=0.))
    for event in measured:
        row = phase_totals[event['phase']]
        row['calls'] += 1
        row['inclusive_wall_seconds'] += max(0., min(ended, event['end_wall_s'])
                                            - max(started, event['start_wall_s']))
        row['inclusive_cpu_seconds'] += event.get('cpu_s') or 0.
    edges = sorted({started, ended, *(max(started, min(ended, e[k]))
                   for e in measured for k in ('start_wall_s', 'end_wall_s'))})
    for left, right in zip(edges, edges[1:]):
        mid = (left+right)/2
        active = [e for e in measured if e['start_wall_s'] <= mid < e['end_wall_s']]
        if active:
            leaf = min(active, key=lambda e: (e['end_wall_s']-e['start_wall_s'],
                                             -e['start_wall_s']))
            phase_totals[leaf['phase']]['exclusive_wall_seconds'] += right-left
        active = [e for e in active if e['phase'] in PHASES]
        if not active:
            continue
        event = min(active, key=lambda e: (e['end_wall_s']-e['start_wall_s'],
                                          -e['start_wall_s']))
        key = PHASES[event['phase']]
        dt = right-left
        totals[key] += dt
        owners = [e for e in active if e.get('details', {}).get('frag') is not None
                  and e['phase'] != 'stc.cc.shared_df_integral_pullback']
        if owners:
            owner = min(owners, key=lambda e: e['end_wall_s']-e['start_wall_s'])
            fragments[int(owner['details']['frag'])][key] += dt

    # Native phases are disjoint measurements inside the Python wrapper timers.
    # Reclassifying them preserves the total; no nested time is counted twice.
    decompositions = []
    native_totals = defaultdict(lambda: defaultdict(float))
    for event in events:
        phase = event['phase']
        details = event.get('details', {})
        timing = details.get('timings', {})
        if phase in ('stc.matrix.native_forward_timings',
                     'stc.matrix.native_reverse_timings', 'stc.triples.native_timings'):
            for name, value in timing.items():
                if name.endswith('_seconds'):
                    native_totals[phase][name] += float(value)
        if phase == 'stc.matrix.native_forward_timings':
            if not {'propagation_seconds', 'amplitude_seconds'} <= timing.keys():
                continue
            seconds = sum(float(timing.get(k, 0.)) for k in
                          ('propagation_seconds', 'amplitude_seconds'))
            source, dest = ('Forward', 'Density matrix (W)'), ('Forward', 'MP2')
        elif phase == 'stc.matrix.native_reverse_timings':
            if not {'propagation_seconds', 'amplitude_seconds', 'reverse_seconds'} <= timing.keys():
                continue
            seconds = sum(float(timing.get(k, 0.)) for k in
                          ('propagation_seconds', 'amplitude_seconds',
                           'reverse_seconds'))
            source, dest = ('Reverse', 'Density matrix (W)'), ('Reverse', 'MP2')
        elif phase == 'stc.triples.native_timings':
            seconds = sum(float(timing.get(k, 0.)) for k in
                          ('vjp_reverse_seconds', 'laplace_pullback_seconds',
                           'pilot_reverse_seconds'))
            source, dest = ('Forward', 'CCSD(T)'), ('Reverse', 'CCSD(T)')
        else:
            continue
        # A native checkpoint can precede completion of its containing timer
        # in a partial run. Defer its attribution until that timer is present.
        parents = [e for e in intervals if PHASES[e['phase']] == source
                   and e['start_wall_s'] <= event['end_wall_s'] <= e['end_wall_s']]
        if not parents:
            continue
        owner = details.get('frag')
        if owner is None:
            owners = [e for e in parents if e.get('details', {}).get('frag') is not None]
            owner = owners[0]['details']['frag'] if owners else None
        if seconds < 0. or seconds > totals[source] + 1e-6:
            raise ValueError(f'Invalid native timing decomposition for {phase}: {seconds}')
        totals[source] -= seconds
        totals[dest] += seconds
        if owner is not None:
            fragments[int(owner)][source] -= seconds
            fragments[int(owner)][dest] += seconds
        decompositions.append(dict(phase=phase, seconds=seconds, fragment=owner))

    order = [('Forward', s) for s in STAGES] + [('Reverse', s) for s in reversed(STAGES)]
    order.append(('Shared', 'MP2 weak pairs (forward + reverse)'))
    rows = [dict(direction=d, stage=s, wall_seconds=totals[d, s]) for d, s in order]
    known = sum(row['wall_seconds'] for row in rows)
    residual = max(0., elapsed-known)
    missing = []
    for parent, detail in (('stc.cc.matrix_forward', 'stc.matrix.native_forward_timings'),
                           ('stc.cc.matrix_pullback', 'stc.matrix.native_reverse_timings')):
        required = {'propagation_seconds', 'amplitude_seconds'}
        if parent.endswith('pullback'):
            required.add('reverse_seconds')
        for interval in (e for e in intervals if e['phase'] == parent):
            receipts = [e for e in events if e['phase'] == detail
                        and interval['start_wall_s'] <= e['end_wall_s'] <= interval['end_wall_s']]
            if not receipts or any(not required <= e['details'].get('timings', {}).keys()
                                   for e in receipts):
                missing.append(parent)
                break
    return dict(accounting_version=2, complete=bool(complete), elapsed_seconds=elapsed,
        start_wall_s=started, end_wall_s=ended, rows=rows,
        overhead_or_unfinished_seconds=residual,
        accounted_seconds=known+residual,
        per_fragment={str(f): [dict(direction=d, stage=s, wall_seconds=values[d, s])
                             for d, s in order] for f, values in sorted(fragments.items())},
        native_decompositions=decompositions,
        phase_timings=[dict(phase=phase, **values)
                       for phase, values in sorted(phase_totals.items())],
        native_timings={phase: dict(values) for phase, values in native_totals.items()},
        native_split_unavailable=missing,
        notes=[
            'Wall seconds; nested timers are partitioned, not summed.',
            'MP2 and W share one pass: MP2 includes propagation/amplitudes; W includes density contractions and wrapper overhead.',
            'CCSD(T) rows contain the incremental stochastic (T) work, excluding CCSD.',
            'Most reverse (T) work executes during sampling and is reassigned to reverse (T) here.',
            'Weak-pair MP2 energy and derivatives are fused in one JIT executable; their shared row is measured once and cannot be split without changing execution.',
            'LIS contains fixed rank/frame/grid selection and differentiable LIS construction/response. The fixed selections have no reverse step.',
            'Domain includes domain/frame preparation, local integral transformations and their derivatives, and remaining correction/ERI propagation in both directions.',
            'phase_timings exposes inclusive and exclusive measurements. Only exclusive wall times can be added; native_timings are nested inside the Python measurements, not additional work.',
            'MP2 reverse includes amplitude/propagator reconstruction as well as their derivatives. SCF reverse is an implicit response solve, not a replay of SCF iterations; no fixed forward/reverse time ratio is expected.',
            'Fragments execute forward/reverse in turn; table order is logical, not chronological.',
            'Per-fragment rows include attributable work; shared/global domain and SCF work appears only in the overall totals.',
            'If native_split_unavailable is nonempty, the corresponding W row still includes shared MP2 work.',
            'Overhead includes imports, I/O, cache cleanup, checkpoints, and uninstrumented work; in partial reports it also includes unfinished stages.',
        ])


class TimingReport(AbstractContextManager):
    """Flush raw events as they happen and write an additive JSON/CSV report."""
    def __init__(self, directory, *, started=None):
        self.directory = Path(directory)
        self.started = time.perf_counter() if started is None else started
        self.events = []
        self.last_write = self.started

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ('timing_events.jsonl', 'timings.json', 'timings.csv'):
            path = self.directory/name
            if path.exists():
                path.rename(path.with_name(path.name + '.' + str(time.time_ns())))
        self.stream = (self.directory/'timing_events.jsonl').open('w')
        self.context = resource_profile.capture_events(self.record)
        self.context.__enter__()
        return self

    def record(self, event):
        self.events.append(event)
        self.stream.write(json.dumps(event, default=_json_default) + '\n')
        self.stream.flush()
        if event['end_wall_s'] - self.last_write >= 15:
            self.write(ended=event['end_wall_s'], complete=False)

    def write(self, *, ended, complete):
        report = summarize(self.events, started=self.started, ended=ended, complete=complete)
        path = self.directory/'timings.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(path)
        path = self.directory/'timings.csv'
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('direction', 'stage', 'wall_seconds'))
            writer.writeheader()
            writer.writerows(report['rows'])
            writer.writerow(dict(direction='Other', stage='Overhead / unfinished',
                                 wall_seconds=report['overhead_or_unfinished_seconds']))
        temp.replace(path)
        self.last_write = ended
        return report

    def __exit__(self, exc_type, exc_value, traceback):
        self.context.__exit__(exc_type, exc_value, traceback)
        self.stream.close()
        report = self.write(ended=time.perf_counter(), complete=exc_type is None)
        print('\nStage timings (wall seconds; additive):', flush=True)
        for row in report['rows']:
            print(f"  {row['direction']:7s} {row['stage']:20s} {row['wall_seconds']:12.3f}", flush=True)
        print(f"  {'Other':7s} {'Overhead / unfinished':20s} {report['overhead_or_unfinished_seconds']:12.3f}", flush=True)
        print(f"  {'Total':7s} {'':20s} {report['elapsed_seconds']:12.3f}", flush=True)
        return False
