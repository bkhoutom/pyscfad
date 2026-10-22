"""Total triples energy budgets are normalized and allocated before execution."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, controls
from pyscfad.dlno_stc._selection import DomainGroup
from pyscfad.dlno_stc.lis import LISSelections
from pyscfad.dlno_stc.test.test_triples import checkpoint_reference


@pytest.mark.parametrize('value', [1e-4, 0, np.float64(5e-5)])
def test_energy_tolerance_accepts_finite_nonnegative_values_without_mutating_input(value):
    supplied = dict(mode='stochastic', nsamples=20, energy_tolerance=value)
    normalized = controls.normalize_triples_controls(supplied, ccsd_t=True)
    assert normalized.get('energy_tolerance', 0.) == value
    assert normalized['nsamples'] == 20
    assert supplied == dict(mode='stochastic', nsamples=20, energy_tolerance=value)


@pytest.mark.parametrize('value', [-1e-4, np.nan, np.inf, -np.inf, True, np.bool_(False), None,
                                  'bad', 1j, [], {}])
def test_energy_tolerance_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='energy_tolerance'):
        controls.normalize_triples_controls(
            dict(mode='stochastic', nsamples=20, energy_tolerance=value), ccsd_t=True)


def test_zero_and_omitted_tolerance_have_identical_normalized_fixed_count_controls():
    supplied = dict(mode='stochastic', nsamples=20, global_seed=11)
    omitted = controls.normalize_triples_controls(supplied, ccsd_t=True)
    zero = controls.normalize_triples_controls({**supplied, 'energy_tolerance': 0.}, ccsd_t=True)
    assert omitted == zero == supplied


@pytest.mark.parametrize('nsamples', [0, 1, True, 2.0])
def test_tolerance_mode_still_requires_integer_pilot_count_of_at_least_two(nsamples):
    with pytest.raises(ValueError, match='nsamples'):
        controls.normalize_triples_controls(
            dict(mode='stochastic', nsamples=nsamples, energy_tolerance=1e-4), ccsd_t=True)


def test_derived_standard_error_is_not_a_public_sampling_control():
    with pytest.raises(ValueError, match='target_standard_error'):
        controls.normalize_triples_controls(dict(mode='stochastic', nsamples=20,
            target_standard_error=1e-4), ccsd_t=True)


@pytest.fixture
def public_reference(monkeypatch):
    """Keep real normalization/selection validation, replace SCF and CC solves."""
    settings = dict(mode='deterministic', threads=1)
    baseline, matrix = ccsd._normalize_controls(settings, settings, 'D')
    domain = SimpleNamespace(fragments=(object(),) * 16)
    lis = LISSelections(domain, 'D', 0., 0., 1e-6, (object(),) * 16)
    static = ccsd.CCSDSelections(lis, ccsd._scientific_identity(baseline, matrix))
    monkeypatch.setattr(ccsd.driver, '_coordinate_only', lambda mol: None)
    monkeypatch.setattr(ccsd.driver, '_initialize_scf', lambda mol, build: (None, None))
    monkeypatch.setattr(ccsd.driver, '_initialize', lambda *a, **k: dict(
        energy=np.float64(0.), variance=0., baseline_energy=0., weak_energy=0.))
    monkeypatch.setattr(ccsd, '_finish', lambda state, static, records, **k: tuple(records))
    return static, settings


def _public_call(api, static, settings, **options):
    args = (None, static) if api == 'kernel' else (None, None, static)
    return getattr(ccsd, api)(*args, baseline_controls=settings, matrix_controls=settings,
                             ccsd_t=True, include_baseline=False, **options)


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
@pytest.mark.parametrize('tolerance,indices,expected', [
    (1e-4, None, 2.5e-5), (5e-5, None, 1.25e-5),
    (1e-4, (0, 3, 7, 12), 5e-5), (1e-4, (12,), 1e-4),
])
@pytest.mark.parametrize('merged', [False, True])
def test_public_initializers_allocate_by_global_requested_singletons(
        monkeypatch, public_reference, api, tolerance, indices, expected, merged):
    static, settings = public_reference
    groups = ((DomainGroup(tuple(range(16)), 0, tuple(range(16))),) if merged else
              tuple(DomainGroup((i,), i, (i,)) for i in range(16)))
    monkeypatch.setattr(ccsd, 'build_domain_groups', lambda *a, **k: groups)

    def solve(state, static, group, baseline, matrix, **options):
        target = options['triples_controls']['target_standard_error']
        return [ccsd.FragmentComponents(i, 0., 0., 0., 0.,
                    triples_diagnostics={'target_standard_error': target})
                for i in group.fragment_indices if i in options['fragment_indices']]

    monkeypatch.setattr(ccsd, '_run_group', solve)
    supplied = dict(mode='stochastic', nsamples=20, energy_tolerance=tolerance)
    result = _public_call(api, static, settings, triples_controls=supplied,
                          fragment_indices=indices, merge_identical=merged)
    assert tuple(row.fragment_index for row in result) == (tuple(range(16)) if indices is None else indices)
    assert all(row.triples_diagnostics['target_standard_error'] == expected for row in result)
    assert 'target_standard_error' not in supplied


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
def test_empty_requested_subset_makes_no_triples_calls(monkeypatch, public_reference, api):
    static, settings = public_reference
    monkeypatch.setattr(ccsd, 'build_domain_groups', lambda *a, **k:
                        (DomainGroup(tuple(range(16)), 0, tuple(range(16))),))
    def forbidden(*a, **k):
        raise AssertionError('empty subset attempted a triples group')
    monkeypatch.setattr(ccsd, '_run_group', forbidden)
    result = _public_call(api, static, settings, fragment_indices=(),
                          triples_controls=dict(mode='stochastic', nsamples=20, energy_tolerance=1e-4))
    assert result == ()


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
def test_allocation_precedes_mpi_distribution(monkeypatch, public_reference, api):
    from pyscfad.dlno_stc import _mpi_ccsd
    static, settings = public_reference
    comm = SimpleNamespace(Get_size=lambda: 4, Get_rank=lambda: 0)
    monkeypatch.setattr(ccsd.driver, '_root_call', lambda comm, rank, call, context: call())
    def distribute(initializer, worker_initializer, comm, **options):
        state, selected, baseline, matrix = initializer()
        return state['cc_triples_controls'], state['cc_fragment_indices']
    monkeypatch.setattr(_mpi_ccsd, '_run_mpi', distribute)
    allocated, indices = _public_call(api, static, settings, comm=comm,
        triples_controls=dict(mode='stochastic', nsamples=20, energy_tolerance=1e-4))
    assert allocated['target_standard_error'] == 2.5e-5
    assert indices == tuple(range(16))


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
def test_allocation_precedes_checkpoint_restore_and_keeps_requested_count(
        monkeypatch, public_reference, api):
    from pyscfad.dlno_stc import _ccsd_checkpoint
    static, settings = public_reference
    groups = tuple(DomainGroup((i,), i, (i,)) for i in range(16))
    monkeypatch.setattr(ccsd, 'build_domain_groups', lambda *a, **k: groups)
    def checkpoint(state, static, groups, **kwargs):
        # A resumed run has only one unfinished group, but retains N=16.
        target = kwargs['options']['triples_controls']['target_standard_error']
        assert target == 2.5e-5
        return None
    monkeypatch.setattr(_ccsd_checkpoint, 'open_checkpoint', checkpoint)
    monkeypatch.setattr(_ccsd_checkpoint, 'restore_progress', lambda *a, **k: (15, True, []))
    def solve(state, static, group, baseline, matrix, **options):
        return [ccsd.FragmentComponents(group.representative_index, 0., 0., 0., 0.,
            triples_diagnostics=options['triples_controls'])]
    monkeypatch.setattr(ccsd, '_run_group', solve)
    result = _public_call(api, static, settings,
        triples_controls=dict(mode='stochastic', nsamples=20, energy_tolerance=1e-4))
    assert len(result) == 1
    assert result[0].fragment_index == 15
    assert result[0].triples_diagnostics['target_standard_error'] == 2.5e-5


def test_checkpoint_rejects_changed_tolerance_and_accepts_omitted_zero(
        tmp_path, checkpoint_reference):
    from pyscfad.dlno_stc._ccsd_checkpoint import open_checkpoint
    from pyscfad.dlno_stc._restart import RestartMismatchError
    mf, static = checkpoint_reference
    state = dict(mf=mf)
    groups = ccsd.build_domain_groups(static.mp2_static)
    base = dict(mode='stochastic', nsamples=20, global_seed=11)
    def options(tolerance=None):
        supplied = base if tolerance is None else {**base, 'energy_tolerance': tolerance}
        return dict(ccsd_t=True, triples_controls=controls.normalize_triples_controls(supplied, ccsd_t=True))
    for name, initial, resumed in [('changed', 1e-4, 5e-5), ('fixed', None, 0.)]:
        directory = tmp_path / name
        manager = open_checkpoint(state, static, groups, checkpoint_dir=directory,
                                  resume=False, options=options(initial))
        manager.close()
        if name == 'changed':
            with pytest.raises(RestartMismatchError):
                open_checkpoint(state, static, groups, checkpoint_dir=directory,
                                resume=True, options=options(resumed))
        else:
            manager = open_checkpoint(state, static, groups, checkpoint_dir=directory,
                                      resume=True, options=options(resumed))
            manager.close()


@pytest.mark.parametrize('api', ['kernel', 'value_and_grad'])
def test_positive_tolerance_that_underflows_per_fragment_target_fails_early(
        monkeypatch, public_reference, api):
    static, settings = public_reference
    def forbidden(*a, **k):
        raise AssertionError('unusable tolerance reached common-data preparation')
    monkeypatch.setattr(ccsd.driver, '_initialize', forbidden)
    with pytest.raises(ValueError, match='energy_tolerance.*too small'):
        _public_call(api, static, settings, triples_controls=dict(
            mode='stochastic', nsamples=20, energy_tolerance=np.nextafter(0., 1.)))


@pytest.mark.parametrize('value', [
    complex(1e-4, 0.), np.complex128(1e-4 + 0j),
    np.complex128(1e-4 + 1j), np.complex128(complex(1e-4, np.nan)),
    np.clongdouble('0.0001+0j'), '1e-4',
])
def test_tolerance_requires_real_values_representable_without_underflow(value):
    with pytest.raises(ValueError, match='energy_tolerance'):
        controls.normalize_triples_controls(
            dict(mode='stochastic', nsamples=20, energy_tolerance=value), ccsd_t=True)


@pytest.mark.parametrize('literal', ['1e-400', '-1e-400'])
def test_extended_precision_tolerance_cannot_narrow_to_zero(literal):
    if np.finfo(np.longdouble).minexp >= np.finfo(np.float64).minexp:
        pytest.skip('longdouble does not extend the float64 exponent range')
    value = np.longdouble(literal)
    with pytest.raises(ValueError, match='energy_tolerance'):
        controls.normalize_triples_controls(
            dict(mode='stochastic', nsamples=20, energy_tolerance=value), ccsd_t=True)


@pytest.mark.parametrize('value', [np.longdouble('0.0001'), np.longdouble('0'),
                                  np.longdouble('-0'), 1])
def test_representable_real_scalars_keep_tolerance_semantics(value):
    normalized = controls.normalize_triples_controls(
        dict(mode='stochastic', nsamples=20, energy_tolerance=value), ccsd_t=True)
    assert normalized.get('energy_tolerance', 0.) == float(value)
