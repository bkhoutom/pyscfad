"""Independent references for activity-aware, blockwise DF transposes."""
import h5py
import jax
import jax.numpy as jnp
import numpy
import pytest
from pyscf import lib
from pyscfad import df, gto
from pyscfad.df import _df_jk_opt


def _case():
    mol = gto.Mole(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol)
    rng = numpy.random.default_rng(281)
    obj._cderi = rng.normal(size=(9, 3))
    obj.blockdim = 4
    return obj, rng


def _reference(packed, dm):
    nao = dm.shape[-1]
    i, j = numpy.tril_indices(nao)
    b = jnp.zeros((packed.shape[0], nao, nao))
    b = b.at[:, i, j].set(packed).at[:, j, i].set(packed)
    return (jnp.einsum('Lij,...ij,Lpq->...pq', b, dm, b),
            jnp.einsum('Lpi,...ij,Ljq->...pq', b, dm, b))


@pytest.mark.parametrize('batch', [(), (2,), (2, 2)])
@pytest.mark.parametrize('positive', [True, False])
def test_density_and_factor_bars_match_independent_tensors(batch, positive):
    obj, rng = _case()
    x = rng.normal(size=batch + (2, 1))
    dm = 2 * (x @ x.swapaxes(-1, -2))
    if not positive:
        dm = dm - 3 * numpy.eye(2)
    seeds = tuple(rng.normal(size=dm.shape) for _ in range(2))
    _, reference = jax.vjp(_reference, jnp.asarray(obj._cderi), jnp.asarray(dm))
    expected_eri, expected_dm = reference(seeds)
    obj_bar, dm_bar = _df_jk_opt.get_jk_bwd(
        1, True, True, 1e-13, (obj, dm), seeds)
    numpy.testing.assert_allclose(obj_bar._cderi, expected_eri, atol=2e-11)
    numpy.testing.assert_allclose(dm_bar, expected_dm, atol=2e-11)


def test_occupation_two_factor_bar_uses_weighted_lowrank(monkeypatch):
    obj, rng = _case()
    coeff = rng.normal(size=(2, 1))
    dm = lib.tag_array(2 * coeff @ coeff.T, mo_coeff=coeff, mo_occ=numpy.array([2.]))
    seeds = tuple(rng.normal(size=dm.shape) for _ in range(2))
    kernel = getattr(_df_jk_opt, '_DF_VK_VJP_LOWRANK', None)
    assert kernel is not None
    calls = []
    def counted(*args):
        calls.append(1)
        return kernel(*args)
    monkeypatch.setattr(_df_jk_opt, '_DF_VK_VJP_LOWRANK', counted)
    _, reference = jax.vjp(_reference, jnp.asarray(obj._cderi), jnp.asarray(dm))
    expected_eri, _ = reference(seeds)
    actual, _ = _df_jk_opt.get_jk_bwd(1, True, True, 1e-13,
                                    (obj, dm, (False, True, False, False)), seeds)
    numpy.testing.assert_allclose(actual._cderi, expected_eri, atol=2e-11)
    assert calls


@pytest.mark.parametrize('component', [0, 1])
def test_unused_output_and_inactive_factor_skip_factor_work(monkeypatch, component):
    obj, rng = _case()
    dm = jnp.asarray(rng.normal(size=(2, 2)))
    def forbidden(*args, **kwargs):
        raise AssertionError('inactive factor cotangent requested')
    monkeypatch.setattr(_df_jk_opt, '_DF_VK_VJP_LOWRANK', forbidden, raising=False)
    def objective(d):
        return jnp.sum(_df_jk_opt.get_jk(obj, d)[component])
    actual = jax.grad(objective)(dm)
    expected = jax.grad(lambda d: jnp.sum(_reference(obj._cderi, d)[component]))(dm)
    numpy.testing.assert_allclose(actual, expected, atol=2e-11)


def _finite_difference_geometry(function, mol, auxmol, step=1e-4):
    """Finite differences use ordinary PySCF integrals, outside our AD rules."""
    mol, auxmol = mol.to_pyscf(), auxmol.to_pyscf()
    result = []
    for target in (mol, auxmol):
        coords = target.atom_coords()
        bar = numpy.empty(coords.shape)
        for ia in range(target.natm):
            for xyz in range(3):
                plus, minus = coords.copy(), coords.copy()
                plus[ia, xyz] += step
                minus[ia, xyz] -= step
                target.set_geom_(plus, unit='Bohr')
                value_plus = function(mol, auxmol)
                target.set_geom_(minus, unit='Bohr')
                value_minus = function(mol, auxmol)
                target.set_geom_(coords, unit='Bohr')
                bar[ia, xyz] = (value_plus - value_minus) / (2 * step)
        result.append(bar)
    return result


def test_disk_placeholder_preserves_geometry_activity(tmp_path, monkeypatch):
    from pyscf.df import incore as ordinary_incore
    from pyscfad import config_update
    obj, rng = _case()
    obj.auxbasis = 'weigend'
    obj._cderi = None
    obj._cderi_to_save = str(tmp_path / 'factors.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    dm = jnp.asarray(numpy.array([[1.1, .2], [.2, .7]]))
    seeds = tuple(rng.normal(size=dm.shape) for _ in range(2))
    calls = []
    # The custom rule captured its bwd at registration; record its shared
    # residual extractor instead, so this observes the real AD boundary.
    residual = _df_jk_opt._jk_residual
    def record_residual(res):
        result = residual(res)
        calls.append(result[2])
        return result
    monkeypatch.setattr(_df_jk_opt, '_jk_residual', record_residual)
    def objective(mol, auxmol):
        view = df.DF(mol, auxbasis='weigend')
        view.auxmol = auxmol
        view.attach_outcore_cderi(obj._cderi_to_save)
        values = _df_jk_opt.get_jk(view, dm)
        return sum(jnp.sum(value * seed) for value, seed in zip(values, seeds))
    bars = jax.grad(objective, argnums=(0, 1))(obj.mol, obj.auxmol)
    assert calls[-1] == (False, False, True, True)
    def ordinary(mol, auxmol):
        packed = ordinary_incore.cholesky_eri(mol, auxmol=auxmol)
        b = lib.unpack_tril(packed)
        j = numpy.einsum('Lij,ij,Lpq->pq', b, dm, b)
        k = numpy.einsum('Lpi,ij,Ljq->pq', b, dm, b)
        return sum(numpy.sum(value * seed) for value, seed in zip((j, k), seeds))
    expected = _finite_difference_geometry(ordinary, obj.mol, obj.auxmol)
    for actual, reference in zip(bars, expected):
        numpy.testing.assert_allclose(actual.coords, reference, atol=2e-7, rtol=2e-7)
    def forbidden(*args, **kwargs):
        raise AssertionError('inactive disk geometry requested')
    monkeypatch.setattr(_df_jk_opt, '_cderi_mol_aux_vjp_from_block_fn', forbidden)
    actual = jax.grad(lambda d: jnp.sum(_df_jk_opt.get_jk(obj, d)[0]))(dm)
    with h5py.File(obj._cderi_to_save, 'r') as handle:
        packed = numpy.asarray(handle['j3c'])
    expected = jax.grad(lambda d: jnp.sum(_reference(packed, d)[0]))(dm)
    numpy.testing.assert_allclose(actual, expected, atol=2e-11)


@pytest.mark.parametrize('sliced', [False, True])
def test_generic_ip1_geometry_matches_finite_difference(monkeypatch, sliced):
    from pyscf.df.incore import aux_e2
    from pyscfad.df import _int3c_cross_opt
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    # Different atom counts exercise the independent auxiliary result shape.
    auxmol = gto.Mole(atom='H .1 .2 .3; H 0 0 .9', basis='weigend', verbose=0)
    auxmol.build(trace_exp=False, trace_ctr_coeff=False)
    i0 = 1 if sliced else 0
    shls = (i0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    offset = mol.ao_loc_nr()[i0]
    npair = mol.nao * (mol.nao + 1) // 2 - offset * (offset + 1) // 2
    seed = numpy.random.default_rng(283).normal(size=(npair, auxmol.nao))
    actual_cross = _int3c_cross_opt.int3c_cross
    def guarded(*args, **kwargs):
        assert 'ip2' not in kwargs.get('intor', '')
        return actual_cross(*args, **kwargs)
    monkeypatch.setattr(_int3c_cross_opt, 'int3c_cross', guarded)
    def objective(m, a):
        return jnp.sum(_int3c_cross_opt.int3c_cross(m, a, shls_slice=shls) * seed)
    bars = jax.grad(objective, argnums=(0, 1))(mol, auxmol)
    expected = _finite_difference_geometry(
        lambda m, a: numpy.sum(aux_e2(m, a, aosym='s2ij', shls_slice=(*shls[:4], 0, auxmol.nbas)) * seed),
        mol, auxmol)
    for actual, reference in zip(bars, expected):
        numpy.testing.assert_allclose(actual.coords, reference, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize('service', [False, True])
@pytest.mark.parametrize('component', [0, 1])
def test_mpi_wrappers_accept_symbolic_unused_output(service, component):
    from mpi4py import MPI
    from pyscfad._src.implicit_diff import _implicit_diff_solve_matvec
    from pyscfad.df.mpi_df_jk import MPIDFJKExecutor
    from pyscfad.dlno_stc import _mpi_scf
    obj, rng = _case()
    dm = jnp.asarray(rng.normal(size=(2, 2)))
    dm = dm + dm.T
    executor = (_mpi_scf._make_executor(MPI.COMM_SELF) if service
                else MPIDFJKExecutor(MPI.COMM_SELF))
    def objective(d):
        values = (_mpi_scf._service_get_jk(executor.token, obj, d) if service
                  else executor.get_jk(obj, d))
        return jnp.sum(values[component])
    try:
        with _implicit_diff_solve_matvec():
            actual = jax.grad(objective)(dm)
    finally:
        executor.stop_workers()
    expected = jax.grad(lambda d: jnp.sum(_reference(obj._cderi, d)[component]))(dm)
    numpy.testing.assert_allclose(actual, expected, atol=2e-11)


def test_response_factor_residual_is_relative_at_small_scale():
    obj, _ = _case()
    _df_jk_opt.set_fast_exchange_dm_data(
        obj, numpy.eye(2), numpy.array([2., 0.]), numpy.eye(2))
    # A virtual--virtual block cannot be represented by occupied response
    # factors even when its absolute size is small.
    vk = numpy.array([[[0., 0.], [0., 1e-12]]])
    assert _df_jk_opt._implicit_lowrank_exchange_factors(
        obj, -2 * vk, vk, 1, True, True) is None


@pytest.mark.parametrize('activity', [
    (True, False, False, False),
    (True, True, False, False),
    (False, True, False, False),
    (True, True, True, True),
    (False, False, True, True),
])
@pytest.mark.parametrize('symmetric_seeds', [False, True])
def test_traced_direct_backward_preserves_activity(activity, symmetric_seeds):
    """Tracing must not turn inactive residual inputs into nonzero bars."""
    obj, rng = _case()
    dm = jnp.asarray(rng.normal(size=(2, 2)))
    dm = dm + dm.T
    seeds = tuple(jnp.asarray(rng.normal(size=dm.shape)) for _ in range(2))
    if symmetric_seeds:
        seeds = tuple(bar + bar.T for bar in seeds)
    def backward(d, jbar, kbar):
        return _df_jk_opt.get_jk_bwd(
            1, True, True, 1e-13, (obj, d, activity), (jbar, kbar))
    eager_df_bar, eager_dm_bar = backward(dm, *seeds)
    traced_df_bar, traced_dm_bar = jax.jit(backward)(dm, *seeds)
    _, reference = jax.vjp(_reference, jnp.asarray(obj._cderi), dm)
    expected_eri, expected_dm = reference(seeds)
    if activity[1]:
        for bar in (eager_df_bar, traced_df_bar):
            numpy.testing.assert_allclose(bar._cderi, expected_eri, atol=2e-11)
            assert bar.mol.coords is None
    else:
        assert eager_df_bar is None
        assert traced_df_bar is None
    if activity[0]:
        numpy.testing.assert_allclose(eager_dm_bar, expected_dm, atol=2e-11)
        numpy.testing.assert_allclose(traced_dm_bar, expected_dm, atol=2e-11)
    else:
        assert eager_dm_bar is None
        assert traced_dm_bar is None


@pytest.mark.parametrize(('mol_active', 'aux_active'), [
    (True, False), (False, True), (True, True),
])
def test_traced_active_disk_geometry_raises(tmp_path, mol_active, aux_active):
    """Unsupported traced disk geometry must fail instead of returning zero."""
    from pyscfad import config_update
    obj, rng = _case()
    obj.auxbasis = 'weigend'
    obj._cderi = None
    obj._cderi_to_save = str(tmp_path / 'factors.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    dm = jnp.asarray([[1.1, .2], [.2, .7]])
    seeds = tuple(jnp.asarray(rng.normal(size=dm.shape)) for _ in range(2))
    activity = (False, False, mol_active, aux_active)
    def backward(jbar, kbar):
        return _df_jk_opt.get_jk_bwd(
            1, True, True, 1e-13, (obj, dm, activity), (jbar, kbar))
    eager_df_bar, eager_dm_bar = backward(*seeds)
    assert eager_dm_bar is None
    for subtree, active in ((eager_df_bar.mol, mol_active),
                            (eager_df_bar.auxmol, aux_active)):
        if active:
            assert numpy.linalg.norm(subtree.coords) > 1e-8
        else:
            assert subtree.coords is None
    with pytest.raises(NotImplementedError, match='traced.*geometry'):
        jax.jit(backward)(*seeds)
