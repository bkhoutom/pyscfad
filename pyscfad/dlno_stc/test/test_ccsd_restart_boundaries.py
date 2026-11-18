"""Checkpoint transaction and cumulative-progress tests at interruption points."""
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc import _restart as storage
from pyscfad.dlno_stc._ccsd_restart_reference import prepare_resume
from pyscfad.dlno_stc._selection import BoysDomainThresholds, DomainGroup
from pyscfad.dlno_stc.test.test_triples import checkpoint_reference


class Interrupted(RuntimeError):
    pass


@pytest.mark.parametrize('boundary', ['weak', 'inside_group', 'before_replace',
                                     'after_commit', 'final_response'])
def test_committed_groups_survive_interruption(tmp_path, monkeypatch, checkpoint_reference, boundary):
    """Use real checkpoint I/O; deterministic mock work isolates bookkeeping."""
    mf, static = checkpoint_reference
    groups = tuple(DomainGroup((i,), i, (i,)) for i in range(3))
    monkeypatch.setattr(ccsd, 'build_domain_groups', lambda *a, **k: groups)
    monkeypatch.setattr(ccsd, '_clear_jax_caches', lambda *a, **k: None)
    calls, weak_calls = [], []
    armed = True
    saves = 0

    def weak(state, *args, **kwargs):
        weak_calls.append(1)
        state['energy'] += .25
        state['mf_bar'] += .5
        state['common_bar'] += .75
        if armed and boundary == 'weak':
            raise Interrupted

    def group(state, selected, current, *args, **kwargs):
        index = current.representative_index
        calls.append(index)
        value = index + 1.
        state['energy'] += value
        state['mf_bar'] += 2 * value
        state['common_bar'] += 3 * value
        if armed and boundary == 'inside_group' and index == 1:
            raise Interrupted
        return [ccsd.FragmentComponents(index, 0., value, 0., 0.)]

    def committed(stage, key, path):
        nonlocal saves
        if stage == 'cc-progress':
            saves += 1
            if armed and boundary == 'after_commit' and saves == 3:
                raise Interrupted

    def before_replace(temporary, path):
        if armed and boundary == 'before_replace' and path.parent.name == 'cc-progress' and saves == 2:
            raise Interrupted

    def finish(state, *args, **kwargs):
        if armed and boundary == 'final_response':
            raise Interrupted
        return tuple(float(state[k]) for k in ('energy', 'mf_bar', 'common_bar'))

    monkeypatch.setattr(ccsd, '_run_weak', weak)
    monkeypatch.setattr(ccsd, '_run_group', group)
    monkeypatch.setattr(ccsd, '_finish', finish)
    monkeypatch.setattr(storage, '_CHECKPOINT_EVENT_HOOK', committed)
    monkeypatch.setattr(storage, '_POST_SAVE_TEST_HOOK', before_replace)

    def execute(resume):
        state = dict(mf=mf, energy=0., variance=0., baseline_energy=0., weak_energy=0.,
                     mf_bar=jnp.asarray(0.), common_bar=jnp.asarray(0.))
        return ccsd._run_serial(state, static, {}, {}, with_grad=True, include_hf=True,
            merge_identical=True, verbose_imp=0, ccsd_t=False, return_components=False,
            checkpoint_dir=tmp_path, resume=resume)

    with pytest.raises(Interrupted):
        execute(False)
    first_calls, first_weak = calls[:], len(weak_calls)
    calls.clear()
    armed = False
    np.testing.assert_allclose(execute(True), (6.25, 12.5, 18.75), atol=0., rtol=0.)
    expected = {'weak': [0, 1, 2], 'inside_group': [1, 2],
                'before_replace': [1, 2], 'after_commit': [2], 'final_response': []}
    assert calls == expected[boundary]
    assert len(weak_calls) - first_weak == int(boundary == 'weak')
    if boundary in ('inside_group', 'before_replace'):
        assert first_calls == [0, 1]


@pytest.mark.parametrize('stage', ['no_directory', 'manifest', 'static', 'reference'])
def test_resume_after_interrupted_initialization(tmp_path, stage):
    path = tmp_path / 'checkpoint'
    if stage == 'no_directory':
        assert prepare_resume(path, True) is False
        return
    manager = storage.RestartManager(path, method='stc-ccsd', scientific_payload={'test': 1})
    if stage in ('static', 'reference'):
        manager.save_static(BoysDomainThresholds())
    if stage == 'reference':
        manager.save_record('cc-reference', scalars={'test': 1})
    manager.close()
    if stage == 'reference':
        assert prepare_resume(path, True) is True
    else:
        with pytest.warns(RuntimeWarning, match='no fragment progress'):
            assert prepare_resume(path, True) is False
        assert not (path / 'run.json').exists()
        assert list((path / '.initialization-history').glob('*/run.json'))
        # A second crash/retry before the new manifest is also recoverable.
        assert prepare_resume(path, True) is False


def test_incomplete_initialization_never_resets_existing_progress(tmp_path):
    manager = storage.RestartManager(tmp_path, method='stc-ccsd', scientific_payload={'test': 1})
    manager.save_record('cc-progress', scalars={'next_group': 2})
    path = manager.record_path('cc-progress')
    manager.close()
    original = path.read_bytes()
    assert prepare_resume(tmp_path, True) is True
    assert path.read_bytes() == original
    path.write_bytes(b'corrupt but must not be silently discarded')
    assert prepare_resume(tmp_path, True) is True
    assert path.read_bytes() == b'corrupt but must not be silently discarded'


def test_initialization_recovery_respects_live_writer(tmp_path):
    if storage.fcntl is None:
        pytest.skip('POSIX writer leases unavailable')
    manager = storage.RestartManager(tmp_path, method='stc-ccsd', scientific_payload={'test': 1})
    original = (tmp_path / 'run.json').read_bytes()
    try:
        with pytest.raises(storage.RestartManifestError, match='another writer'):
            prepare_resume(tmp_path, True)
        assert (tmp_path / 'run.json').read_bytes() == original
    finally:
        manager.close()
