"""Storage ownership and numerical checks for the four-factor scalar path."""
import gc
import weakref
from contextlib import nullcontext
from dataclasses import replace

import jax
import numpy as np
import pytest

from pyscfad.dlno_stc.test.test_backend import native
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture, grid, KEYS
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def native_layout(b):
    return np.asfortranarray(b.reshape(b.shape[0], -1)).reshape(b.shape)


@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('targets', [(0,), (1, 0)])
def test_borrowed_factor_matches_public_solver_and_owns_returned_bar(with_grad, targets):
    x = full_fixture()
    controls = dict(grid(), threads=2)
    expected = native().solve_selected(*(x[k] for k in KEYS), targets, controls,
                                       with_grad=with_grad)
    b = native_layout(x['B'])
    b.flags.writeable = False
    result = native()._solve_selected_borrowed(
        x['foo'], x['fvv'], b, targets, controls, with_grad=with_grad)
    np.testing.assert_allclose(result['energy'], expected['energy'], atol=2e-13)
    assert result['diagnostics']['borrowed_input_B']
    assert result['diagnostics']['peak_factor_buffers'] == (4 if with_grad else 2)
    np.testing.assert_array_equal(b, x['B'])
    reference = weakref.ref(b)
    del b
    gc.collect()
    assert reference() is None
    if with_grad:
        for k in KEYS:
            np.testing.assert_allclose(result['cotangents'][k],
                                       expected['cotangents'][k], atol=2e-12)


def test_borrowed_binding_rejects_non_native_layout_and_stochastic_mode():
    x = full_fixture()
    with pytest.raises(ValueError, match='layout'):
        native()._solve_selected_borrowed(*(x[k] for k in KEYS), (0,), grid())
    x['B'] = native_layout(x['B'])
    with pytest.raises(ValueError, match='deterministic'):
        native()._solve_selected_borrowed(*(x[k] for k in KEYS), (0,),
            dict(grid(), mode='stochastic', production_samples=32))


@pytest.mark.parametrize('shape', [(4, 2, 0), (0, 2, 3), (4, 2, 1), (1, 2, 3)])
def test_borrowed_factor_empty_and_singleton_dimensions(shape):
    q, no, nv = shape
    b = np.empty((q, no * nv), order='F').reshape(shape)
    b.fill(.07)
    foo, fvv = -np.eye(no), np.eye(nv)
    actual = native()._solve_selected_borrowed(foo, fvv, b, (0,), grid())
    expected = native().solve_selected(foo, fvv, np.ascontiguousarray(b), (0,), grid())
    np.testing.assert_allclose(actual['energy'], expected['energy'], atol=2e-13)
    for key in KEYS:
        np.testing.assert_allclose(actual['cotangents'][key], expected['cotangents'][key], atol=2e-12)


@pytest.mark.parametrize('bad', ['nonfinite', 'wrong_dtype', 'unaligned'])
def test_borrowed_binding_validates_host_storage(bad):
    x = full_fixture()
    b = native_layout(x['B'])
    if bad == 'nonfinite':
        b[1, 0, 0] = np.nan
    elif bad == 'wrong_dtype':
        b = b.astype(np.float32)
    else:
        b = np.ndarray(b.shape, dtype='float64', buffer=bytearray(b.nbytes + 1),
                       offset=1, strides=b.strides)
    with pytest.raises((ValueError, TypeError)):
        native()._solve_selected_borrowed(x['foo'], x['fvv'], b, (0,), grid())


@pytest.mark.parametrize('with_grad', [False, True])
def test_empty_auxiliary_backend_handles_automatic_offsets(with_grad):
    from pyscfad.dlno_stc import backend
    x = full_fixture()
    x['B'] = x['B'][:0]
    actual = backend.solve(x, dict(target_indices=(0,)), grid(), with_grad=with_grad)
    assert actual['energy'] == 0.
    if with_grad:
        for k in KEYS:
            assert actual['cotangents'][k].shape == x[k].shape
            np.testing.assert_array_equal(actual['cotangents'][k], 0.)


def test_memory_forecast_uses_the_prepared_auxiliary_support(boys_water_dimer, monkeypatch):
    from pyscfad.dlno_stc import driver, build_domain_groups, _workflow
    mf, common, static = boys_water_dimer
    group = build_domain_groups(static)[0]
    index = group.representative_index
    fragments = list(static.fragments)
    fragments[index] = replace(fragments[index], pao_center_atoms=fragments[index].extended_atoms[:1])
    static = replace(static, fragments=tuple(fragments))
    measured = []
    monkeypatch.setattr(_workflow, 'assess_memory',
                        lambda *args, **kwargs: measured.append(kwargs['unavoidable_bytes']))
    inputs, _, _ = driver._prepare(dict(mf=mf, common=common), static, group,
                                  dict(mode='deterministic'), False)
    assert measured == [inputs['B'].nbytes]


def test_unshared_forward_retains_dependent_auxiliary_metric_fallback(boys_water_dimer):
    from pyscfad.dlno_stc import domain, prepare
    from pyscfad.dlno_stc._mp2_preparation import project_inputs
    mf, common, static = boys_water_dimer
    frame = domain.build_boys_ed_frame(common, static, 0)
    fock = common.fock[np.ix_(frame.extended_ao_indices, frame.extended_ao_indices)]
    # Two identical s shells make the fitting metric exactly dependent.
    basis = {'O': [[0, [1., 1.]], [0, [1., 1.]], [1, [.8, 1.]]],
             'H': [[0, [.6, 1.]]]}
    args = (mf, fock, frame.occupied_coeff, frame.virtual_coeff, frame.extended_atoms)
    kwargs = dict(aux_atoms=static.fragments[0].pao_center_atoms, correlation_auxbasis=basis)
    expected = prepare._project_inputs(*args, **kwargs)
    actual, pullback = project_inputs(*args, **kwargs, with_grad=False)
    assert pullback is None
    for key in KEYS:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('cache_mode', ['direct', 'half', None])
@pytest.mark.parametrize('deferred', [False, True])
def test_host_preparation_shares_factor_and_matches_full_vjp(boys_water_dimer,
                                                           monkeypatch, cache_mode, deferred):
    from pyscfad.dlno_stc import driver, build_domain_groups, build_domain_selections, backend
    mf, common, static = boys_water_dimer
    if deferred:
        static = build_domain_selections(mf, frozen=2)
        assert static.ed_selection_pending
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', cache_mode or 'direct')
    group = build_domain_groups(static)[0]
    fragment = static.fragments[group.representative_index]
    state = dict(mf=mf, common=common)
    def scope():
        return (nullcontext(None) if cache_mode is None else driver._mp2_df_scope(
            state, fragment.extended_atoms, fragment.pao_center_atoms))
    with scope() as shared:
        expected, _, old_pullback, frame = driver._prepare(
            state, static, group, dict(mode='deterministic'), True, return_frame=True)
        bars = {k: np.random.default_rng(i).normal(size=v.shape)
                for i, (k, v) in enumerate(expected.items())}
        for k in ('foo', 'fvv'):
            bars[k] = (bars[k] + bars[k].T) / 2
        expected_bars = old_pullback((bars, (np.zeros_like(frame.occupied_coeff),
                                            np.zeros_like(frame.virtual_coeff))))
        expected_geometry = shared.finish() if shared else 0.
    with scope() as shared:
        inputs, host, pullback = driver._prepare(state, static, group,
                                               dict(mode='deterministic'), True)
        assert isinstance(inputs['B'], np.ndarray)
        assert np.shares_memory(inputs['B'], host['B'])
        assert np.shares_memory(host['B'], backend.host_inputs(host, native_factors=True)['B'])
        if cache_mode == 'half':
            assert np.shares_memory(host['B'], shared.half.entries[0].fitted)
        for k in KEYS:
            np.testing.assert_allclose(inputs[k], expected[k], atol=2e-12)
        from pyscfad.dlno_stc.adjoint import apply_selected_pullback
        result = dict(energy=np.float64(0), energy_standard_error=np.float64(0),
                      diagnostics={}, cotangents={k: -.5 * bar for k, bar in bars.items()})
        actual_bars = apply_selected_pullback(pullback, inputs, result, energy_bar=-2.)
        actual_geometry = shared.finish() if shared else 0.
    for actual, reference in zip(jax.tree.leaves(actual_bars), jax.tree.leaves(expected_bars)):
        if getattr(actual, 'dtype', None) != jax.dtypes.float0:
            np.testing.assert_allclose(actual, reference, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(actual_geometry, expected_geometry, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize('cache_mode', ['direct', 'half'])
def test_host_project_empty_virtual_space_has_zero_response(boys_water_dimer, monkeypatch, cache_mode):
    from pyscfad.dlno_stc import driver, domain
    from pyscfad.dlno_stc._mp2_preparation import project_inputs
    mf, common, static = boys_water_dimer
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', cache_mode)
    frame = domain.build_boys_ed_frame(common, static, 0)
    fock = common.fock[np.ix_(frame.extended_ao_indices, frame.extended_ao_indices)]
    with driver._mp2_df_scope(dict(mf=mf), frame.extended_atoms,
                             static.fragments[0].pao_center_atoms) as shared:
        def no_metric():
            raise AssertionError('an empty orbital-pair space must not factor the metric')
        monkeypatch.setattr(shared, 'metric_factor', no_metric)
        inputs, pullback = project_inputs(
            mf, fock, frame.occupied_coeff, frame.virtual_coeff[:, :0], frame.extended_atoms,
            aux_atoms=static.fragments[0].pao_center_atoms, with_grad=True)
        assert inputs['B'].shape[2] == 0
        bars = pullback({k: np.zeros_like(v) for k, v in inputs.items()})
        for bar in jax.tree.leaves(bars):
            if getattr(bar, 'dtype', None) != jax.dtypes.float0:
                np.testing.assert_array_equal(bar, 0.)
        np.testing.assert_array_equal(shared.finish(), 0.)


def test_host_forward_does_not_build_reverse_tapes(boys_water_dimer, monkeypatch):
    from pyscfad.dlno_stc import driver, build_domain_groups
    mf, common, static = boys_water_dimer
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'direct')
    group = build_domain_groups(static)[0]
    fragment = static.fragments[group.representative_index]
    state = dict(mf=mf, common=common)
    def forbidden(*args, **kwargs):
        raise AssertionError('forward-only MP2 must not construct a VJP')
    monkeypatch.setattr(jax, 'vjp', forbidden)
    with driver._mp2_df_scope(state, fragment.extended_atoms,
                             fragment.pao_center_atoms, with_grad=False) as shared:
        inputs, host, pullback = driver._prepare(state, static, group,
                                               dict(mode='deterministic'), False)
        shared.finish()
    assert pullback is None
    assert np.shares_memory(inputs['B'], host['B'])
