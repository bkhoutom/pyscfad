"""Shared local integral response, including different frames and W feedback."""
from types import SimpleNamespace
import sys

import h5py
import jax
import numpy
import pytest

from pyscfad import gto
from pyscfad import numpy as np
from pyscfad.lno import df as lno_df
from pyscfad.lno import _df_shared
from pyscfad.lno import _df_direct
from pyscfad.dlno_stc import ccsd


ATOMS = numpy.asarray([0, 2], dtype=numpy.int32)


@pytest.fixture(autouse=True)
def no_local_df_disk(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Local DF inputs and adjoints must not open HDF5 storage')
    monkeypatch.setattr(h5py, 'File', forbidden)


def _mf(mol):
    return SimpleNamespace(mol=mol, get_ovlp=lambda: mol.intor('int1e_ovlp'),
        with_df=SimpleNamespace(auxbasis='weigend', max_memory=64),
        verbose=0, stdout=sys.stdout)


def _problem(nmo):
    mol = gto.Mole(atom='O 0 0 0; H .1 0 1; H 0 1 .1', basis='sto-3g',
                   verbose=0, max_memory=3000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    coeff = np.asarray(numpy.random.default_rng(79).normal(size=(mol.nao, nmo)) * .2)
    return mol, coeff


def _energy(mol, coeff, s1e=None):
    # Different ED OV and LIS all-pair spaces, with a downstream dependence
    # on ED factors standing in for the W -> LIS frame response.
    ed = lno_df.transform_df_to_mo(_mf(mol), coeff,
                                  (0, 2, 2, coeff.shape[1]), atmlst=ATOMS, s1e=s1e)
    lis_coeff = coeff[:, :3] * (1 + .03 * np.sum(ed**2))
    lis = lno_df.transform_df_to_mo(_mf(mol), lis_coeff, (0, 3, 0, 3),
                                   atmlst=ATOMS, s1e=s1e).reshape(-1, 3, 3)
    weight = np.asarray([[.2, -.1, .3], [.5, .7, -.2], [-.4, .2, .1]])
    return .1*np.sum(ed**2) + np.sum(lis**2*weight) - .2*np.sum(lis[:, :1, 1:]**2)


@pytest.mark.parametrize('backend', ['gemm', 'tiled'])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_shared_packed_density_matches_both_ao_legs(backend, order, monkeypatch):
    """Unequal/transposed MO blocks share the native diagonal convention."""
    monkeypatch.setenv('PYSCFAD_DF_MO_DENSITY_BACKEND', backend)
    mol, _ = _problem(7)
    scope = _df_shared.LocalDFReverse(_mf(mol), ATOMS)
    rng = numpy.random.default_rng(811)
    expected = numpy.zeros((2, scope.mol.nao, scope.mol.nao))
    try:
        for slices in ((1, 3, 2, 7), (1, 6, 3, 5)):
            k0, k1, l0, l1 = slices
            coeff = rng.normal(size=(scope.mol.nao, 7))
            z = numpy.array(rng.normal(size=(scope.auxmol.nao, (k1-k0)*(l1-l0))),
                            order=order)
            retained_z = scope.add(coeff, z, numpy.zeros((scope.auxmol.nao,)*2), slices)
            assert numpy.shares_memory(retained_z, z)
            assert scope.terms[-1].z is retained_z
            expected += numpy.einsum('uk,pkl,vl->puv', coeff[:, k0:k1],
                z[1:3].reshape(2, k1-k0, l1-l0), coeff[:, l0:l1])
        rows, cols = numpy.tril_indices(scope.mol.nao)
        packed = (expected + expected.swapaxes(1, 2))[:, rows, cols]
        packed[:, rows == cols] *= .5
        # Native packing writes C-contiguous rows; advanced indexing above
        # leaves the reference array Fortran-contiguous.
        out = numpy.empty_like(packed, order='C')
        scope._fill_packed_density(1, 3, out)
        numpy.testing.assert_allclose(out, packed, atol=2e-13, rtol=2e-13)
        assert scope.receipt['max_density_rows'] == 2
    finally:
        scope.close()
    assert not scope._packing_workspace


@pytest.mark.parametrize('nmo', [4, 7])
@pytest.mark.parametrize('seed', [.7, -.3])
def test_shared_coordinate_and_orbital_bars_match_separate_reverses(nmo, seed, monkeypatch):
    mol, coeff = _problem(nmo)
    objective = lambda m, c: seed * _energy(m, c)
    value_ref, (mol_ref, coeff_ref) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
    original_rows = _df_shared.LocalDFReverse._rows
    monkeypatch.setattr(_df_shared.LocalDFReverse, '_rows',
        lambda self, *args: min(2, original_rows(self, *args)))
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        value, (mol_bar, coeff_bar) = jax.value_and_grad(objective, argnums=(0, 1))(mol, coeff)
        assert len(scope.terms) == 2
        assert all(isinstance(term.z, numpy.ndarray) for term in scope.terms)
        retained = sum(term.z.nbytes + term.coeff.nbytes for term in scope.terms)
        assert scope.receipt['adjoint_resident_bytes'] == retained
        geometry = scope.finish()
        numpy.testing.assert_allclose(mol_bar.coords + geometry, mol_ref.coords,
                                      atol=2e-9, rtol=2e-8)
        numpy.testing.assert_allclose(coeff_bar, coeff_ref, atol=2e-10, rtol=2e-9)
        numpy.testing.assert_allclose(value, value_ref, atol=1e-13)
        assert scope.receipt['consumers'] == 2
        assert scope.receipt['geometry_passes'] == scope.receipt['metric_passes'] == 1
        assert scope.receipt['max_density_rows'] <= 2
        assert scope.receipt['adjoint_peak_bytes'] == retained
        assert scope.receipt['adjoint_resident_bytes'] == 0
        assert scope.terms == []
    assert scope.terms == []
    assert scope.metric is None and scope.low is None
    # Finite differences include the overlap projection and ED-to-LIS chain.
    step = 1e-5
    # Update libcint's geometry as well as the traced coordinate attribute.
    # Assigning only .coords leaves ordinary forward integral values unchanged.
    plus = mol.set_geom_(numpy.asarray(mol.coords.at[2, 1].add(step)),
                         unit='Bohr', inplace=False)
    minus = mol.set_geom_(numpy.asarray(mol.coords.at[2, 1].add(-step)),
                          unit='Bohr', inplace=False)
    fd = (objective(plus, coeff) - objective(minus, coeff)) / (2*step)
    numpy.testing.assert_allclose(mol_bar.coords[2, 1] + geometry[2, 1], fd,
                                  atol=2e-8, rtol=3e-6)


def test_direct_reuses_metric_but_generates_four_integral_passes(monkeypatch):
    mol, coeff = _problem(7)
    reference, ref_bars = jax.value_and_grad(_energy, argnums=(0, 1))(mol, coeff)
    counts = dict(metric=0, cholesky=0, raw=0)
    coverage = []
    intor = type(mol).intor
    cholesky = _df_shared.jsp_linalg.cholesky
    generate = _df_direct._generate_local_raw_int3c_blocks

    def metric(self, name, *args, **kwargs):
        if name in ('int2c2e', 'int2c2e_sph'):
            counts['metric'] += 1
        return intor(self, name, *args, **kwargs)

    def factor(*args, **kwargs):
        counts['cholesky'] += 1
        return cholesky(*args, **kwargs)

    def raw(*args, **kwargs):
        counts['raw'] += 1
        for p0, p1, block in generate(*args, **kwargs):
            coverage.append((p0, p1))
            yield p0, p1, block

    monkeypatch.setattr(type(mol), 'intor', metric)
    monkeypatch.setattr(_df_shared.jsp_linalg, 'cholesky', factor)
    monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', raw)
    def no_full_dataset(self, *args, **kwargs):
        pytest.fail('Shared DF inputs must be read as bounded slices')
    monkeypatch.setattr(h5py.Dataset, '__array__', no_full_dataset)
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        value, (mol_bar, coeff_bar) = jax.value_and_grad(_energy, argnums=(0, 1))(mol, coeff)
        geometry = scope.finish()
        numpy.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)
        numpy.testing.assert_allclose(mol_bar.coords + geometry, ref_bars[0].coords,
                                      atol=2e-9, rtol=2e-8)
        numpy.testing.assert_allclose(coeff_bar, ref_bars[1], atol=2e-10, rtol=2e-9)
        assert counts == dict(metric=1, cholesky=1, raw=4)
        assert scope.receipt['metric_builds'] == scope.receipt['cholesky_builds'] == 1
        assert scope.receipt['raw_integral_builds'] == scope.receipt['raw_source_passes'] == 4
        assert scope.receipt['adjoint_peak_bytes'] > 0
        assert scope.receipt['adjoint_resident_bytes'] == 0
    assert scope.terms == []
    assert scope.metric is None and scope.low is None


def test_shared_overlap_preserves_all_projection_response(monkeypatch):
    mol, coeff = _problem(4)
    reference, ref_bars = jax.value_and_grad(_energy, argnums=(0, 1))(mol, coeff)
    holder = _mf
    def no_overlap(m):
        mf = holder(m)
        def forbidden():
            pytest.fail('LIS transform must consume the supplied overlap')
        mf.get_ovlp = forbidden
        return mf
    monkeypatch.setattr(sys.modules[__name__], '_mf', no_overlap)
    def energy(m, c):
        return _energy(m, c, s1e=m.intor('int1e_ovlp'))
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        value, (mol_bar, coeff_bar) = jax.value_and_grad(energy, argnums=(0, 1))(mol, coeff)
        geometry = scope.finish()
    numpy.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)
    numpy.testing.assert_allclose(mol_bar.coords + geometry, ref_bars[0].coords,
                                  atol=2e-9, rtol=2e-8)
    numpy.testing.assert_allclose(coeff_bar, ref_bars[1], atol=2e-10, rtol=2e-9)


def test_scope_does_not_reuse_inputs_for_different_geometry():
    mol, _ = _problem(4)
    shifted = mol.set_geom_(numpy.asarray(mol.coords.at[2, 1].add(.1)),
                            unit='Bohr', inplace=False)
    local = lno_df.make_local_mol(shifted, ATOMS)
    aux = lno_df.df_addons.make_auxmol(local, 'weigend')
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        assert _df_shared.matching_scope(local, aux) is None
        assert _df_shared.matching_scope(scope.mol, scope.auxmol) is scope
        scope.finish()


def test_scope_is_closed_after_generation_failure(monkeypatch):
    mol, coeff = _problem(4)
    generate = _df_direct._generate_local_raw_int3c_blocks
    def failing(*args, **kwargs):
        first = next(generate(*args, **kwargs))
        yield first
        raise RuntimeError('interrupted integral generation')
    monkeypatch.setattr(_df_direct, '_generate_local_raw_int3c_blocks', failing)
    with pytest.raises(RuntimeError, match='interrupted integral generation'):
        with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
            jax.grad(lambda m: _energy(m, coeff))(mol)
    assert scope.terms == [] and scope.receipt['adjoint_resident_bytes'] == 0
    assert scope.metric is None and scope.low is None
    assert scope._metric_pullback is None and scope._factor_pullback is None
    assert _df_shared._ACTIVE.get() is None


def test_owner_adds_shared_geometry_once_and_preserves_other_bars(monkeypatch):
    mol, coeff = _problem(4)
    expected, reference = jax.value_and_grad(lambda m: _energy(m, coeff))(mol)
    static = SimpleNamespace(mp2_static=SimpleNamespace(
        fragments=[SimpleNamespace(extended_atoms=ATOMS, pao_center_atoms=ATOMS)]))
    group = SimpleNamespace(representative_index=0)
    initial = .123
    state = dict(mf=_mf(mol))
    def responses(state, *args, **kwargs):
        value, bar = jax.value_and_grad(lambda m: _energy(m, coeff))(mol)
        bar.coords = bar.coords + initial
        state['mf_bar'] = SimpleNamespace(mol=bar)
        return [float(value)]
    monkeypatch.setattr(ccsd, '_run_group_responses', responses)
    records = ccsd._run_group(state, static, group, {}, {}, with_grad=True,
                              verbose_imp=0, ccsd_t=True)
    numpy.testing.assert_allclose(records, [expected], atol=1e-13)
    numpy.testing.assert_allclose(state['mf_bar'].mol.coords, reference.coords+initial,
                                  atol=2e-9, rtol=2e-8)


def test_stc_replica_mean_scales_integral_response_with_shared_optimization_enabled(monkeypatch):
    """Deferred geometry must not escape the production-replica averaging."""
    mol, coeff = _problem(4)
    controls = dict(energy_tolerance=1e4, seed=17, precision_pilot_replicas=2,
                    precision_min_replicas=2, precision_max_replicas=2)
    static = SimpleNamespace(mp2_static=SimpleNamespace(
        fragments=[SimpleNamespace(extended_atoms=ATOMS, pao_center_atoms=ATOMS)]))
    group = SimpleNamespace(representative_index=0)
    state = dict(mf=_mf(mol))
    def weight(config):
        return (config['seed'] % 17 + 1) / 10.
    def responses(state, *args, **options):
        assert options['cc_solver'] == 'stc'
        def evaluate(config, need_grad):
            objective = lambda m: weight(config) * _energy(m, coeff)
            if need_grad:
                value, bar = jax.value_and_grad(objective)(mol)
            else:
                value, bar = objective(mol), None
            return value, ((0., value, 0.), {}), bar
        value, _, bar = ccsd._stc_precision.run_replicas(
            evaluate, options['stc_cc_controls'], 0, 1, with_grad=True)
        state['mf_bar'] = SimpleNamespace(mol=bar)
        return [value]
    monkeypatch.setattr(ccsd, '_run_group_responses', responses)
    records = ccsd._run_group(state, static, group, {}, {}, with_grad=True,
                              verbose_imp=0, ccsd_t=True,
                              cc_solver='stc', stc_cc_controls=controls)
    mean_weight = sum(weight(ccsd._stc_precision.replica_controls(
        controls, 0, 'production', i)) for i in range(2)) / 2
    expected, reference = jax.value_and_grad(lambda m: mean_weight * _energy(m, coeff))(mol)
    numpy.testing.assert_allclose(records, [expected], atol=1e-13)
    numpy.testing.assert_allclose(state['mf_bar'].mol.coords, reference.coords,
                                  atol=2e-9, rtol=2e-8)


def test_budget_exhaustion_and_scope_failure_do_not_leak(monkeypatch):
    mol, coeff = _problem(4)
    reference = jax.grad(lambda m: _energy(m, coeff))(mol)
    with pytest.warns(RuntimeWarning, match='minimum row exceeds the advisory workspace') as caught:
        with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
            monkeypatch.setattr(_df_shared.pyscf_lib, 'current_memory',
                                lambda: (mol.max_memory+1., 0.))
            bar = jax.grad(lambda m: _energy(m, coeff))(mol)
            geometry = scope.finish()
            assert scope.receipt['tile_rows'] == 1
            assert scope.receipt['max_density_rows'] == 1
            assert scope.receipt['max_raw_rows'] == 1
    assert len(caught) == 1
    numpy.testing.assert_allclose(bar.coords + geometry, reference.coords,
                                  atol=2e-9, rtol=2e-8)
    assert scope.terms == [] and scope.receipt['adjoint_resident_bytes'] == 0
    assert _df_shared._ACTIVE.get() is None
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='finished and added'):
        with _df_shared.reverse_scope(_mf(mol), ATOMS) as unfinished:
            jax.grad(lambda m: _energy(m, coeff))(mol)
    assert unfinished.terms == [] and unfinished.receipt['adjoint_resident_bytes'] == 0
    assert _df_shared._ACTIVE.get() is None
    # A subsequent ordinary gradient must still include its integral response.
    bar = jax.grad(lambda m: _energy(m, coeff))(mol)
    assert numpy.isfinite(bar.coords).all()


def test_shared_reverse_respects_available_block_workspace(monkeypatch):
    mol, _ = _problem(4)
    monkeypatch.setattr(_df_shared.pyscf_lib, 'current_memory', lambda: (0., 0.))
    # A small DF workspace is respected when it holds complete auxiliary rows.
    mf = _mf(mol)
    mf.with_df.max_memory = .01
    with _df_shared.reverse_scope(mf, ATOMS) as scope:
        row_bytes = 8 * (6 * scope.mol.nao**2 + 21 + 3 * scope.mol.nao + 3 * scope.mol.nao)
        rows = scope._rows(21, 3)
        assert rows == min(scope.auxmol.nao, int(.01 * 1e6 // row_bytes))
        assert rows * row_bytes <= .01 * 1e6
        assert not scope._minimum_row_warned
        scope.finish()


def test_nested_scope_and_double_finish_rejected():
    mol, _ = _problem(4)
    with _df_shared.reverse_scope(_mf(mol), ATOMS) as scope:
        with pytest.raises(RuntimeError, match='Nested'):
            with _df_shared.reverse_scope(_mf(mol), ATOMS):
                pass
        numpy.testing.assert_array_equal(scope.finish(), numpy.zeros((mol.natm, 3)))
        with pytest.raises(RuntimeError, match='already finished'):
            scope.finish()
