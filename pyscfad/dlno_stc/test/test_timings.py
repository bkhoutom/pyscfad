"""Wall-time accounting must not double count nested or fused operations."""
import json

import pytest

from pyscfad.dlno_stc.timings import TimingReport, summarize
from pyscfad.tools import resource_profile


def event(phase, start, end, **details):
    return dict(phase=phase, start_wall_s=start, end_wall_s=end, details=details)


def test_nested_and_native_times_add_to_elapsed():
    events = [
        event('stc.domain.scf', 0., 2.),
        event('stc.cc.matrix_forward', 2., 12., frag=1),
        event('stc.matrix.native_forward_timings', None, 11.9, frag=1,
              timings=dict(propagation_seconds=1., amplitude_seconds=2.,
                           contraction_seconds=6., native_total_seconds=9.5)),
        event('stc.cc.correction_forward', 12., 20., frag=1),
        event('solver.ccsd_iterations_and_energy', 13., 17., frag=1),
        event('solver.perturbative_triples', 17., 20., frag=1),
        event('stc.triples.native_timings', None, 19.9,
              timings=dict(vjp_reverse_seconds=.6, laplace_pullback_seconds=.4)),
        event('stc.cc.correction_pullback', 20., 25., frag=1),
        event('ccsd_bwd.total', 21., 24.),
        event('stc.cc.matrix_pullback', 25., 33., frag=1),
        event('stc.matrix.native_reverse_timings', None, 32.9, frag=1,
              timings=dict(propagation_seconds=1., amplitude_seconds=1.,
                           reverse_seconds=2., contraction_seconds=2., seed_seconds=1.)),
    ]
    report = summarize(events, started=0., ended=35., complete=True)
    times = {(row['direction'], row['stage']): row['wall_seconds'] for row in report['rows']}
    assert times['Forward', 'MP2'] == 3.
    assert times['Forward', 'Density matrix (W)'] == 7.
    assert times['Forward', 'CCSD(T)'] == 2.
    assert times['Reverse', 'CCSD(T)'] == 1.
    assert times['Reverse', 'MP2'] == 4.
    assert times['Reverse', 'Density matrix (W)'] == 4.
    assert times['Reverse', 'CCSD'] == 3.
    assert report['overhead_or_unfinished_seconds'] == 2.
    assert sum(times.values()) + report['overhead_or_unfinished_seconds'] == 35.
    assert sum(row['wall_seconds'] for row in report['per_fragment']['1']) == 31.


def test_partial_native_event_waits_for_parent_timer():
    events = [event('stc.triples.native_timings', None, 9.,
                    timings=dict(vjp_reverse_seconds=2., laplace_pullback_seconds=1.))]
    report = summarize(events, started=0., ended=10.)
    assert report['native_decompositions'] == []
    assert report['overhead_or_unfinished_seconds'] == 10.
    assert not report['complete']


def test_native_total_alone_is_explicitly_unsplit():
    events = [event('stc.cc.matrix_forward', 0., 3., frag=1),
              event('stc.matrix.native_forward_timings', None, 2.9, frag=1,
                    timings=dict(native_total_seconds=2.5))]
    report = summarize(events, started=0., ended=3., complete=True)
    assert report['native_split_unavailable'] == ['stc.cc.matrix_forward']
    assert report['native_decompositions'] == []


def test_raw_reverse_names_and_shared_domain_ownership():
    events = [event('stc.cc.correction_pullback', 0., 3., frag=2),
              event('stc.lis.pullback', 0., .5, frag=2),
              event('backward.local DF MO-coeff backward', 1., 2.),
              event('stc.cc.shared_df_integral_pullback', 3., 5., frag=57)]
    report = summarize(events, started=0., ended=5., complete=True)
    times = {(r['direction'], r['stage']): r['wall_seconds'] for r in report['rows']}
    assert times['Reverse', 'LIS'] == .5
    assert times['Reverse', 'Domain'] == 4.5
    assert set(report['per_fragment']) == {'2'}


def test_lis_excludes_integral_transform_and_reports_fixed_selection():
    events = [event('stc.cc.lis_selection', 0., 3.),
              event('stc.lis.fit_grid', 1., 3.),
              event('stc.cc.correction_forward', 3., 10.),
              event('stc.lis.build', 3., 5.),
              event('solver.df_ao2mo_and_eris', 5., 9.),
              event('stc.cc.correction_pullback', 10., 15.),
              event('stc.lis.pullback', 10., 11.)]
    report = summarize(events, started=0., ended=16., complete=True)
    times = {(r['direction'], r['stage']): r['wall_seconds'] for r in report['rows']}
    assert times['Forward', 'LIS'] == 5.
    assert times['Reverse', 'LIS'] == 1.
    assert times['Forward', 'Domain'] == 5.
    assert times['Reverse', 'Domain'] == 4.
    phases = {r['phase']: r for r in report['phase_timings']}
    assert phases['stc.cc.lis_selection']['inclusive_wall_seconds'] == 3.
    assert phases['stc.cc.lis_selection']['exclusive_wall_seconds'] == 1.
    assert phases['stc.cc.correction_forward']['exclusive_wall_seconds'] == 1.
    assert sum(r['exclusive_wall_seconds'] for r in phases.values()) == 15.
    assert report['overhead_or_unfinished_seconds'] == 1.


def test_capture_restores_context_and_writes_failure_report(tmp_path, monkeypatch):
    monkeypatch.delenv('PYSCFAD_DLNO_RESOURCE_PROFILE', raising=False)
    assert not resource_profile.enabled()
    with pytest.raises(ValueError, match='calculation failed'):
        with TimingReport(tmp_path):
            assert resource_profile.enabled()
            with resource_profile.section('stc.domain.scf'):
                pass
            raise ValueError('calculation failed')
    assert not resource_profile.enabled()
    report = json.loads((tmp_path/'timings.json').read_text())
    assert not report['complete']
    assert report['accounted_seconds'] == pytest.approx(report['elapsed_seconds'])
    records = [json.loads(line) for line in (tmp_path/'timing_events.jsonl').read_text().splitlines()]
    assert records[0]['phase'] == 'stc.domain.scf'
    assert records[0]['end_wall_s'] >= records[0]['start_wall_s']
    assert (tmp_path/'timings.csv').exists()
