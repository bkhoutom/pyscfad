"""Atom domains, shared MP2 work, and independently seeded orbital corrections."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad.dlno_stc import backend, ccsd, _selection
from pyscfad.dlno_stc.test.test_backend import native, fixture
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer
from pyscfad.dlno_stc.test.test_atom_grouping import _assert_saved_selection_equal
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('size', [None, 1, 2])
@pytest.mark.parametrize('repeated', [False, True])
def test_shared_orbital_w_and_reverse_match_individual_targets(size, repeated, threads):
    native()
    x = fixture(repeated)
    fragments = ((17, (1,)), (3, (0,)))  # labels and target order need not agree
    controls = dict(mode='deterministic', threads=threads, target_batch_size=size)
    batch = backend.solve_orbital_matrices(x, fragments, controls)
    rng = np.random.default_rng(16)
    seeds = [dict(occupied=rng.normal(size=x['foo'].shape),
                  virtual=rng.normal(size=x['fvv'].shape)) for _ in fragments]
    expected = {name: np.zeros_like(value) for name, value in x.items()}
    for (index, targets), actual, seed in zip(fragments, batch.forwards, seeds):
        reference = backend.solve_matrices(x, dict(fragment_id=index, target_indices=targets), controls)
        for key in ('energy', 'occupied', 'virtual'):
            np.testing.assert_allclose(actual[key], reference[key], atol=2e-13, rtol=2e-12)
        bars = backend.pullback_matrices(x, reference, energy_bar=1.,
            occupied_bar=seed['occupied'], virtual_bar=seed['virtual'])
        for key in x:
            expected[key] += bars[key]
    actual = backend.pullback_orbital_matrices(x, batch, seeds)
    for key in x:
        np.testing.assert_allclose(actual[key], expected[key], atol=3e-12, rtol=3e-11)
    directions = {key: rng.normal(size=value.shape) for key, value in x.items()}
    for key in ('foo', 'fvv'):
        directions[key] = .5 * (directions[key] + directions[key].T)
    def loss(sign):
        shifted = {key: value + sign*1e-5*directions[key] for key, value in x.items()}
        result = backend.solve_orbital_matrices(shifted, fragments, controls)
        return sum(f['energy'] + sum(np.sum(f[k]*bar[k]) for k in bar)
                   for f, bar in zip(result.forwards, seeds))
    finite = (loss(1) - loss(-1)) / 2e-5
    np.testing.assert_allclose(sum(np.sum(actual[k]*directions[k]) for k in x),
                               finite, atol=3e-8, rtol=3e-7)
    with pytest.raises(ValueError, match='changed'):
        backend.pullback_orbital_matrices(dict(x, B=x['B']*1.01), batch, seeds)
    with pytest.raises(ValueError, match='count'):
        backend.pullback_orbital_matrices(x, batch, seeds[:1])


def test_orbital_w_reverse_rejects_density_replay():
    extension = native()
    x = fixture()
    controls = dict(laplace_roots=[.2, .5], laplace_weights=[.4, .6])
    forward = extension.solve_matrices(*x.values(), (0,), controls, matrix_kind='D')
    with pytest.raises(ValueError, match='Orbital W replay'):
        extension.pullback_orbital_matrices(
            *x.values(), forward['replay'], [1.], [np.eye(2)], [np.eye(3)])


def test_atom_singleton_preserves_atom_domains_and_mp2(boys_water_dimer):
    from pyscfad import dlno_stc
    mf, _, reference = boys_water_dimer
    options = dict(frozen=2, target_coeff=reference.target_reference_coeff)
    atom = dlno_stc.build_domain_selections(mf, boys_grouping='atom', **options)
    orbital = dlno_stc.build_domain_selections(mf, boys_grouping='atom_singleton', **options)
    _assert_saved_selection_equal(atom, replace(orbital, boys_grouping='atom'))
    assert _selection._cc_groups(orbital) == tuple((i,) for i in range(8))
    atom = dlno_stc.materialize_domain_selections(mf, atom)
    orbital = dlno_stc.materialize_domain_selections(mf, orbital)
    _assert_saved_selection_equal(atom, replace(orbital, boys_grouping='atom'))
    groups = _selection.build_domain_groups(orbital)
    assert any(len(g.target_indices) > 1 for g in groups)
    owned = [item for g in groups for item in _selection.physical_fragments_in_group(orbital, g)]
    assert sorted(i for i, _ in owned) == list(range(8))
    assert all(len(t) == 1 for _, t in owned)
    controls = dict(mode='deterministic', threads=1)
    ea = dlno_stc.kernel(mf, atom, controls=controls)
    eo = dlno_stc.kernel(mf, orbital, controls=controls)
    np.testing.assert_allclose(eo, ea, atol=2e-12, rtol=0)


@pytest.mark.parametrize('with_grad', [False, True])
@pytest.mark.parametrize('matrix_kind', ['W', 'D'])
def test_atom_singleton_routes_density_and_preserves_response(
        group_case, monkeypatch, with_grad, matrix_kind):
    case = group_case
    reference, _ = case.run(with_grad=with_grad, matrix_kind=matrix_kind)
    events = []
    real_prepare = ccsd.driver._prepare
    def prepare(state, static, *args, **kwargs):
        static.boys_grouping = 'atom_singleton'
        return real_prepare(state, static, *args, **kwargs)
    real_forward = backend.solve_orbital_matrices
    def forward(*args, **kwargs):
        result = real_forward(*args, **kwargs)
        events.append(('W', len(result.forwards)))
        return result
    real_matrix = backend.solve_matrices
    def matrix(*args, **kwargs):
        result = real_matrix(*args, **kwargs)
        events.append((result['matrix_kind'], args[1]['fragment_id']))
        return result
    real_solve = ccsd._solve_fragment
    def solve(*args, **kwargs):
        events.append(('CC', args[3]))
        return real_solve(*args, **kwargs)
    real_reverse = backend.pullback_orbital_matrices
    def reverse(*args, **kwargs):
        events.append(('reverse', len(args[2])))
        return real_reverse(*args, **kwargs)
    monkeypatch.setattr(ccsd.driver, '_prepare', prepare)
    monkeypatch.setattr(backend, 'solve_orbital_matrices', forward)
    monkeypatch.setattr(backend, 'solve_matrices', matrix)
    monkeypatch.setattr(ccsd, '_solve_fragment', solve)
    monkeypatch.setattr(backend, 'pullback_orbital_matrices', reverse)
    actual, _ = case.run(with_grad=with_grad, matrix_kind=matrix_kind)
    expected = ([('W', 2), ('CC', 0), ('CC', 1)]
                + ([('reverse', 2)] if with_grad else []) if matrix_kind == 'W' else
                [('D', 0), ('CC', 0), ('D', 1), ('CC', 1)])
    assert events == expected
    for key in ('energy', 'baseline_energy', 'mf_bar', 'common_bar'):
        np.testing.assert_allclose(actual[key], reference[key], atol=2e-12, rtol=2e-12)


def test_molecular_ccsdt_gradient_and_restart(tmp_path, monkeypatch):
    """Compare batched W/RI with separate orbital reverses on identical domains."""
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.lno import _df_shared
    from pyscfad.dlno_stc.test._saved_selection import save_from_run
    mol = gto.Mole(atom='O 0 0 0; H .1 0 1; H 0 1 .1', basis='sto-3g',
                   verbose=0, max_memory=10000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    def build_mf(m):
        mf = scf.RHF(m).density_fit(auxbasis='weigend')
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.kernel()
        return mf
    mf = build_mf(mol)
    domains = dlno_stc.build_domain_selections(mf, frozen=1, full_domain=True,
        boys_grouping='atom_singleton', lo_kwargs=dict(conv_tol=1e-12, conv_tol_grad=1e-9))
    plan = ccsd.build_static_selections(mf, domains, thresh_occ=1e-3, thresh_vir=1e-4)
    selected = save_from_run(mf, plan, ccsd_t=True, lis_aux_svd_tol=None)
    assert len(selected.fragments) == 4
    # Change only execution ownership: keep the exact atom domains and all
    # saved LIS labels, anchors, grids and target projections.
    reference_domains = replace(selected.mp2_static, boys_grouping='singleton',
        physical_groups=tuple((i,) for i in range(4)))
    reference_static = replace(selected,
        lis_static=replace(selected.lis_static, mp2_static=reference_domains))
    options = dict(ccsd_t=True, lis_aux_svd_tol=None, return_components=True)
    reference = ccsd.value_and_grad(mol, build_mf, static=reference_static, **options)
    receipts = []
    finish = _df_shared.LocalDFReverse.finish
    def observed(scope):
        if scope.adjoint_basis is not None:
            assert len(scope.terms) == 1
            receipts.append(dict(scope.receipt))
        return finish(scope)
    monkeypatch.setattr(_df_shared.LocalDFReverse, 'finish', observed)
    actual = ccsd.value_and_grad(mol, build_mf, static=selected,
                                checkpoint_dir=tmp_path, **options)
    assert len(receipts) == 1 and receipts[0]['consumers'] >= 5
    np.testing.assert_allclose(actual[0], reference[0], atol=3e-11, rtol=0)
    np.testing.assert_allclose(actual[1].coords, reference[1].coords, atol=3e-8, rtol=2e-6)
    for a, b in zip(actual[2].fragments, reference[2].fragments):
        assert (a.nocc_lis, a.nvir_lis) == (b.nocc_lis, b.nvir_lis)
        np.testing.assert_allclose([a.e_mp2_lis, a.e_ccsd, a.e_triples],
                                   [b.e_mp2_lis, b.e_ccsd, b.e_triples], atol=3e-11, rtol=0)
    # Fixed-label finite difference traverses SCF, localization, W, LIS and CC.
    step = 1e-4
    coords = np.asarray(mol.coords)
    direction = np.zeros_like(coords); direction[1, 2] = 1.
    def energy(sign):
        moved = mol.set_geom_(coords + sign*step*direction, unit='Bohr', inplace=False)
        return ccsd.kernel(build_mf(moved), selected, ccsd_t=True, lis_aux_svd_tol=None)
    finite = (energy(1) - energy(-1))/(2*step)
    np.testing.assert_allclose(actual[1].coords[1, 2], finite, atol=2e-6, rtol=2e-4)
    resumed = ccsd.value_and_grad(mol, build_mf, checkpoint_dir=tmp_path, resume=True, **options)
    np.testing.assert_allclose(resumed[0], actual[0], atol=2e-12, rtol=0)
    np.testing.assert_allclose(resumed[1].coords, actual[1].coords, atol=2e-10, rtol=0)


@pytest.mark.parametrize('empty', [False, True])
def test_orbital_stochastic_and_empty_replays(empty):
    native()
    x = fixture()
    if empty:
        x['B'] = x['B'][:0]
    controls = dict(mode='stochastic', threads=1, global_seed=93,
        domain_tau=1e-100, virtual_domain_tau=1e-100,
        pair_pilot_samples=16, min_samples=16, matrix_tolerance=.03)
    fragments = ((7, (1,)), (2, (0,)))
    batch = backend.solve_orbital_matrices(x, fragments, controls)
    seeds = [dict(occupied=np.eye(2)*(i+.3), virtual=np.eye(3)*(.2-i))
             for i in range(2)]
    expected = {key: np.zeros_like(value) for key, value in x.items()}
    for (index, target), actual, seed in zip(fragments, batch.forwards, seeds):
        ref = backend.solve_matrices(x, dict(fragment_id=index, target_indices=target), controls)
        np.testing.assert_allclose(actual['energy'], ref['energy'], atol=2e-13)
        bars = backend.pullback_matrices(x, ref, energy_bar=1.,
            occupied_bar=seed['occupied'], virtual_bar=seed['virtual'])
        for key in expected:
            expected[key] += bars[key]
    actual = backend.pullback_orbital_matrices(x, batch, seeds)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=3e-12, rtol=3e-11)
