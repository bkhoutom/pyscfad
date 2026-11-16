"""Forward selection hands numerical preparation to the identical reference."""
from copy import copy, deepcopy
from dataclasses import replace
import gc
import pickle
import weakref


import numpy as np
import pytest

from pyscfad import gto, scf
from pyscfad.dlno_stc import _selection, domain, driver


@pytest.fixture(scope='module')
def reference():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .75; H 0 0 7; H 0 0 7.75',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-11
    mf.kernel()
    return mf


def select(mf, **options):
    return _selection.build_domain_selections(
        mf, frozen=0, thresholds=_selection.BoysDomainThresholds(
            pair_energy=1., near_pair_distance=1.), **options)


def forbidden(*args, **kwargs):
    raise AssertionError('same-state energy rebuilt selection preparation')


def test_repeated_kernel_reuses_scf_boys_and_weak_preparation(reference, monkeypatch):
    static = select(reference)
    assert driver.weak_pair_indices(static), 'fixture must exercise weak completion'
    before = pickle.dumps(static)
    expected = driver.kernel(reference, replace(static), controls={'threads': 1})
    monkeypatch.setattr(driver, 'rebuild_domain_data', forbidden)
    monkeypatch.setattr(domain, 'build_scf_data', forbidden)
    monkeypatch.setattr(domain, '_rebuild_boys_targets', forbidden)
    monkeypatch.setattr(driver, 'weak_pair_energy_sum', forbidden)
    for _ in range(2):
        actual = driver.kernel(reference, static, controls={'threads': 1})
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)
    assert pickle.dumps(static) == before


def test_supplied_stationary_target_frame_reuses_preparation(reference, monkeypatch):
    original = select(reference)
    supplied = original.target_reference_coeff.copy()
    static = select(reference, target_coeff=supplied)
    monkeypatch.setattr(driver, 'rebuild_domain_data', forbidden)
    state = driver._initialize(reference, static, driver._driver_controls({}), with_grad=False)
    np.testing.assert_array_equal(state['common'].iao_coeff, supplied)
    # Mutating the caller's target array must not alter the saved numerical frame.
    supplied[:] = 0.
    np.testing.assert_array_equal(state['common'].iao_coeff, static.target_reference_coeff)


@pytest.mark.parametrize('angle,options', [(.27, {}), (1e-7, {}),
                                            (1e-3, {'conv_tol_grad': 1.})])
def test_nonstationary_supplied_target_keeps_existing_refinement(reference, monkeypatch, angle, options):
    original = select(reference)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    static = select(reference, target_coeff=original.target_reference_coeff @ rotation, lo_kwargs=options)
    expected = domain.rebuild_domain_data(reference, static)
    assert np.max(abs(expected.iao_coeff - static.target_reference_coeff)) > .1 * angle
    calls = []
    rebuild = driver.rebuild_domain_data
    def count(*args, **kwargs):
        calls.append(1)
        return rebuild(*args, **kwargs)
    monkeypatch.setattr(driver, 'rebuild_domain_data', count)
    state = driver._initialize(reference, static, driver._driver_controls({}), with_grad=False)
    assert calls == [1]
    assert state['_prepared_weak_energy'] is None
    np.testing.assert_allclose(state['common'].iao_coeff, expected.iao_coeff, atol=1e-12)


@pytest.mark.parametrize('changed', ['orbitals', 'energies', 'coords', 'basis',
                                     'df_source', 'df_basis', 'static'])
def test_mutated_then_restored_state_falls_back_and_recovers(reference, monkeypatch, changed):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    common, _ = reuse.lookup(reference, static)
    fitting_state = vars(reference.with_df).copy()
    if changed == 'orbitals':
        owner, name = reference, 'mo_coeff'
        saved = getattr(owner, name)
        value = np.array(saved)
        value[0, 0] += .001
    elif changed == 'energies':
        owner, name = reference, 'mo_energy'
        saved = getattr(owner, name)
        value = np.array(saved) + .001
    elif changed == 'coords':
        owner, name = reference.mol, 'coords'
        saved = getattr(owner, name)
        value = np.array(saved)
        value[0, 0] += .001
    elif changed == 'basis':
        owner, name = reference.mol, '_basis'
        saved = getattr(owner, name)
        value = deepcopy(saved)
        value['H'][0][1][0] += .001
    elif changed == 'df_source':
        owner, name = reference.with_df, '_cderi'
        saved = getattr(owner, name)
        value = saved.at[0, 0].add(.001)
    elif changed == 'df_basis':
        owner, name = reference.with_df, 'auxbasis'
        saved = getattr(owner, name)
        value = 'weigend'
    else:
        owner, name = static, 'weak_pair_energy'
        saved = getattr(owner, name)
        value = saved.copy()
        value[0, 1] += .001
    # In-place edits to ordinary arrays must be detected too.
    if changed in ('orbitals', 'energies', 'coords'):
        object.__setattr__(owner, name, np.array(saved))
        assert reuse.lookup(reference, static) is not None
        getattr(owner, name)[...] = value
    elif changed == 'static':
        saved = saved.copy()
        getattr(owner, name)[...] = value
    else:
        object.__setattr__(owner, name, value)
    calls = []
    monkeypatch.setattr(driver, 'rebuild_domain_data',
                        lambda *args, **kwargs: calls.append(1) or common)
    try:
        assert reuse.lookup(reference, static) is None
        state = driver._initialize(reference, static, driver._driver_controls({}), with_grad=False)
        assert calls == [1]
        assert state['_prepared_weak_energy'] is None
    finally:
        if changed == 'static':
            getattr(owner, name)[...] = saved
        elif changed == 'df_basis':
            vars(owner).clear()
            vars(owner).update(fitting_state)
        else:
            object.__setattr__(owner, name, saved)
    assert reuse.lookup(reference, static) is not None


def test_copied_mf_replaced_and_serialized_selections_rebuild(reference, monkeypatch):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    common, _ = reuse.lookup(reference, static)
    calls = []
    monkeypatch.setattr(driver, 'rebuild_domain_data',
                        lambda *args, **kwargs: calls.append(1) or common)
    for mf, labels in ((copy(reference), static), (reference, replace(static)),
                       (reference, pickle.loads(pickle.dumps(static)))):
        assert reuse.lookup(mf, labels) is None
        state = driver._initialize(mf, labels, driver._driver_controls({}), with_grad=False)
        assert state['_prepared_weak_energy'] is None
    assert len(calls) == 3


def test_gradient_initialization_never_consults_forward_receipt(reference, monkeypatch):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    common, _ = reuse.lookup(reference, static)
    calls = []
    pullback = lambda value: value
    monkeypatch.setattr(reuse, 'lookup', forbidden)
    monkeypatch.setattr(driver, '_build_common_with_pullback',
                        lambda *args: calls.append(1) or (common, pullback))
    monkeypatch.setattr(driver, '_zero_term_cotangents', lambda *args: (None, None))
    state = driver._initialize(
        reference, static, driver._driver_controls({}), with_grad=True,
        scf_data=(common.s1e, common.fock), scf_data_pullback=pullback)
    assert calls == [1]
    assert state['common_pullback'] is pullback
    assert state['_prepared_weak_energy'] is None


def test_weak_reuse_keeps_directed_half_weights_and_subset_fallback(reference, monkeypatch):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    common, original = reuse.lookup(reference, static)
    strong = static.strong_mask.copy()
    strong[0, 1] = True
    directed = replace(static, strong_mask=strong)
    reuse.remember(reference, directed, common)
    state = {'_prepared_weak_energy': reuse.lookup(reference, directed)[1],
             'mf': reference, 'common': common}
    np.testing.assert_allclose(driver._weak_energy(state, directed, original[0]), original[1] / 2)
    calls = []
    monkeypatch.setattr(driver, 'weak_pair_energy_sum',
                        lambda *args, **kwargs: calls.append(kwargs['pairs']) or 0.)
    assert driver._weak_energy(state, directed, ()) == 0.
    assert calls == [()]


def test_mutable_integral_buffer_uses_conservative_fallback(reference, monkeypatch):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    common, _ = reuse.lookup(reference, static)
    with monkeypatch.context() as patch:
        patch.setattr(reference.with_df, '_cderi', np.array(reference.with_df._cderi))
        assert reuse.lookup(reference, static) is None
        reuse.remember(reference, static, common)
        assert reuse.lookup(reference, static) is None


def test_receipt_is_bounded_and_releases_dead_owners(reference):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    first = select(reference)
    common, _ = reuse.lookup(reference, first)
    common_ref = weakref.ref(common.s1e)
    second = select(reference)
    assert reuse.lookup(reference, first) is None
    del common
    gc.collect()
    assert common_ref() is None
    latest_ref = weakref.ref(reuse.lookup(reference, second)[0].s1e)
    del second
    gc.collect()
    assert latest_ref() is None
    assert reuse._latest is None


@pytest.mark.parametrize('method', ['get_fock', 'get_ovlp', 'get_hcore', 'get_veff'])
def test_changed_scf_operator_rejects_preparation(reference, monkeypatch, method):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    original = getattr(reference, method)
    with monkeypatch.context() as patch:
        patch.setattr(reference, method, lambda *args, **kwargs: original(*args, **kwargs))
        assert reuse.lookup(reference, static) is None
    vars(reference).pop(method, None)
    assert reuse.lookup(reference, static) is not None


def test_replaced_fitting_object_rejects_preparation(reference, monkeypatch):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    with monkeypatch.context() as patch:
        patch.setattr(reference, 'with_df', copy(reference.with_df))
        assert reuse.lookup(reference, static) is None
    assert reuse.lookup(reference, static) is not None


class _BroadcastComm:
    def __init__(self, rank, values):
        self.rank = rank
        self.values = values
        self.calls = 0

    def Get_rank(self):
        return self.rank

    def bcast(self, value, root):
        assert root == 0
        self.calls += 1
        if self.rank == 0:
            self.values.append(value)
        return self.values[self.calls - 1]


def test_mpi_prepared_weak_scalar_is_added_only_on_root(monkeypatch):
    from types import SimpleNamespace
    from pyscfad.dlno_stc import _mpi, _mpi_weak
    monkeypatch.setattr(_mpi, 'weak_pair_contribution', forbidden)
    common = SimpleNamespace(s1e=np.eye(1))
    pairs = ((0, 1), (0, 2))
    broadcasts, energies = [], []
    for rank in range(3):
        state = {'mf': object(), 'common': common}
        if rank == 0:
            state['_prepared_weak_energy'] = (pairs, -.125)
        _mpi_weak.LAST_DIAGNOSTICS.update(stale=True, n_endpoint_builds=99)
        comm = _BroadcastComm(rank, broadcasts)
        energy, bars = _mpi._weak_contribution(state, object(), pairs, comm, with_grad=False)
        energies.append(float(energy))
        assert bars is None
        assert 'stale' not in _mpi_weak.LAST_DIAGNOSTICS
        assert _mpi_weak.LAST_DIAGNOSTICS['n_endpoint_builds'] == 0
        assert _mpi_weak.LAST_DIAGNOSTICS['n_endpoint_replays'] == 0
        assert comm.calls == 1
    assert energies == [-.125, 0., 0.]
    assert sum(energies) == -.125
    assert broadcasts == [-.125]


@pytest.mark.parametrize('with_grad,stored', [(True, ((0, 1),)),
                                             (False, ((0, 2),)), (False, None)])
def test_mpi_weak_gradient_and_unprepared_pairs_keep_owner_path(monkeypatch, with_grad, stored):
    from types import SimpleNamespace
    from pyscfad.dlno_stc import _mpi
    pairs, calls, broadcasts = ((0, 1),), [], []
    def fallback(*args, **kwargs):
        calls.append((args[4].Get_rank(), kwargs['with_grad']))
        return -.05, 'live bars'
    monkeypatch.setattr(_mpi, 'weak_pair_contribution', fallback)
    for rank in range(3):
        state = {'mf': object(), 'common': SimpleNamespace(s1e=np.eye(1))}
        if rank == 0 and stored is not None:
            state['_prepared_weak_energy'] = (stored, -.125)
        comm = _BroadcastComm(rank, broadcasts)
        result = _mpi._weak_contribution(state, object(), pairs, comm, with_grad=with_grad)
        assert result == (-.05, 'live bars')
        assert comm.calls == (0 if with_grad else 1)
    assert calls == [(rank, with_grad) for rank in range(3)]


def test_custom_class_operator_with_mutable_field_is_not_reused(reference):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    class ShiftedFock(type(reference)):
        def get_fock(self, *args, **kwargs):
            return super().get_fock(*args, **kwargs) + self.extra_field * np.eye(self.mol.nao)
    current = copy(reference)
    current.__class__ = ShiftedFock
    current.extra_field = 0.
    static = select(current)
    assert reuse.lookup(current, static) is None
    current.extra_field = .01
    assert reuse.lookup(current, static) is None


@pytest.mark.parametrize('method', ['intor', 'intor_symmetric', 'atom_coords'])
def test_changed_molecular_operator_rejects_preparation(reference, monkeypatch, method):
    from pyscfad.dlno_stc import _energy_preparation as reuse
    static = select(reference)
    original = getattr(reference.mol, method)
    with monkeypatch.context() as patch:
        patch.setattr(reference.mol, method, lambda *args, **kwargs: original(*args, **kwargs))
        assert reuse.lookup(reference, static) is None
    # monkeypatch restores a bound method as an instance attribute; remove it
    # so later fake molecules do not inherit a method bound to this full mol.
    vars(reference.mol).pop(method, None)
    assert reuse.lookup(reference, static) is not None
