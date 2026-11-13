"""Sampling rotations are shared only within one physical LIS evaluation."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, stc_ccsd
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('with_grad', [False, True])
def test_replica_sampling_frame_is_shared_and_expires(group_case, monkeypatch, with_grad):
    """Pilots/production share a gauge; other fragments/geometries do not."""
    created, seen = [], []

    class Frame:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(stc_ccsd, 'SamplingFrame', Frame, raising=False)
    original = ccsd._solve_fragment

    def fragment(*args, **options):
        seen.append((args[3], options.get('sampling_frame'), options['with_grad']))
        values = original(*args, **options)
        return values, {'ccsd': {'seed': options['stc_cc_controls']['seed']}}

    monkeypatch.setattr(ccsd, '_solve_fragment', fragment)
    controls = dict(seed=71, energy_tolerance=1e3, precision_pilot_replicas=2,
                    precision_min_replicas=2, precision_max_replicas=2)
    for mf in (.2, .3):
        group_case.run(matrix_kind='W', with_grad=with_grad, mf=mf,
                       cc_solver='stc', stc_cc_controls=controls)
    assert len(created) == 4
    assert len(seen) == 16
    for frame_index, frame in enumerate(created):
        batch = seen[4*frame_index:4*frame_index+4]
        assert [entry[0] for entry in batch] == [frame_index % 2] * 4
        assert all(entry[1] is frame for entry in batch)
        assert [entry[2] for entry in batch] == [False, False, with_grad, with_grad]
    assert len({id(frame) for frame in created}) == 4


def test_fragment_passes_scoped_frame_to_stc_adapter(monkeypatch):
    """The frame reaches the adapter after the LIS has been constructed."""
    frame = object()
    sampling_frame = object()
    controls = {'seed': 71}
    prepared = SimpleNamespace(active_occupied_coeff=np.eye(3)[:, :1],
        active_virtual_coeff=np.eye(3)[:, 1:], mo_coeff=np.eye(3),
        fragment_iao_coeff=np.eye(3)[:, :1], frozen=[])
    common = SimpleNamespace(fock=np.eye(3), s1e=np.eye(3))
    fragment = SimpleNamespace(extended_atoms=np.array([0]), pao_center_atoms=np.array([0]))
    static = SimpleNamespace(mp2_static=SimpleNamespace(fragments=[fragment]),
                             lis_static=SimpleNamespace(smooth_frames=None))
    group = SimpleNamespace(representative_index=0)
    solver = object()
    observed = []

    monkeypatch.setattr(ccsd, '_build_solver_lis', lambda *args: prepared)

    def make_solver(configuration, fragment_index, *, sampling_frame):
        assert configuration is controls and fragment_index == 0
        observed.append(sampling_frame)
        return solver

    def solve(*args, **options):
        assert options['amplitude_solver'] is solver
        assert options['active_nocc'] == 1
        return (0., 0., 0.), {'ccsd': {}}

    monkeypatch.setattr(stc_ccsd, 'make_solver', make_solver)
    monkeypatch.setattr(ccsd, '_impurity_solve_core', solve)
    ccsd._solve_fragment(None, common, static, 0, {}, group,
        verbose_imp=0, ccsd_t=False, cc_solver='stc', stc_cc_controls=controls,
        frame=frame, sampling_frame=sampling_frame)
    assert observed == [sampling_frame]
