"""Factor-direct regression tests for the projected LNO (T) correction."""

from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy

from pyscfad import config_update, df, gto, scf
from pyscfad.cc import dfccsd
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds
from pyscfad.lno import ccsd as lno_ccsd
from pyscfad.lno import ccsd_t


def _random_problem(seed=12, nocc=3, nvir=5, naux=7):
    rng = numpy.random.default_rng(seed)
    Lov = rng.normal(scale=0.12, size=(naux, nocc, nvir))
    nvir_pair = nvir * (nvir + 1) // 2
    Lvv = rng.normal(scale=0.12, size=(naux, nvir_pair))
    Lov_flat = Lov.reshape(naux, -1)
    ovov = (Lov_flat.T @ Lov_flat).reshape(nocc, nvir, nocc, nvir)
    ovvv = (Lov_flat.T @ Lvv).reshape(nocc, nvir, nvir_pair)

    ulo = rng.normal(scale=0.3, size=(nocc + 2, nocc))
    mat = ulo.T @ ulo
    t1T = rng.normal(scale=0.02, size=(nvir, nocc))
    t2T = rng.normal(scale=0.03, size=(nvir, nvir, nocc, nocc))
    mo_energy = numpy.concatenate((
        -numpy.linspace(1.2, 0.5, nocc),
        numpy.linspace(0.3, 1.1, nvir),
    ))
    fvo = rng.normal(scale=0.04, size=(nvir, nocc))
    ovoo = rng.normal(scale=0.08, size=(nocc, nvir, nocc, nocc))
    return SimpleNamespace(
        ulo=ulo,
        mat=mat,
        t1T=t1T,
        t2T=t2T,
        mo_energy=mo_energy,
        fvo=fvo,
        ovoo=ovoo,
        ovov=ovov,
        ovvv=ovvv,
        Lov=Lov,
        Lvv=Lvv,
    )


def _factor_args(problem):
    return (
        problem.mat,
        problem.t1T,
        problem.t2T,
        problem.mo_energy,
        problem.fvo,
        problem.ovoo,
        problem.ovov,
        problem.Lov,
        problem.Lvv,
    )


def _dense_args(problem):
    return (
        problem.mat,
        problem.t1T,
        problem.t2T,
        problem.mo_energy,
        problem.fvo,
        problem.ovoo,
        problem.ovov,
        problem.ovvv,
    )


def test_factor_direct_energy_and_vjp_match_dense(monkeypatch):
    problem = _random_problem()
    max_memory = 2000

    dense_energy = ccsd_t._ccsd_t_energy(
        *_dense_args(problem), max_memory, False)
    dense_bars = ccsd_t._ccsd_t_energy_bwd(
        max_memory, False, _dense_args(problem), numpy.asarray(1.0))

    # Explicitly exercise nonzero row/column block origins.  A one-block test
    # would not cover the rectangular cache address arithmetic in the C
    # kernels.
    monkeypatch.setattr(
        ccsd_t, '_ccsd_t_factor_block_nvir',
        lambda nocc, nvir, memory, backward=False: min(2, nvir),
    )
    monkeypatch.setattr(
        ccsd_t, '_fill_vvop',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('legacy global vvop was constructed')),
    )

    cache_shapes = []
    original_builder = ccsd_t.canonical_ccsd_t._build_df_vvop_cache

    def recording_builder(*args, **kwargs):
        cache = original_builder(*args, **kwargs)
        cache_shapes.append(cache.shape)
        return cache

    monkeypatch.setattr(
        ccsd_t.canonical_ccsd_t,
        '_build_df_vvop_cache',
        recording_builder,
    )

    factor_energy = ccsd_t._ccsd_t_energy_df(
        *_factor_args(problem), max_memory, False)
    factor_bars = ccsd_t._ccsd_t_energy_df_bwd(
        max_memory, False, _factor_args(problem), numpy.asarray(1.0))

    numpy.testing.assert_allclose(
        factor_energy, dense_energy, atol=2e-13, rtol=0.0)
    for factor_bar, dense_bar in zip(factor_bars[:7], dense_bars[:7]):
        numpy.testing.assert_allclose(
            factor_bar, dense_bar, atol=3e-12, rtol=0.0)

    dense_ovvv_bar = dense_bars[7]
    Lov_bar_reference = numpy.einsum(
        'iaq,xq->xia', dense_ovvv_bar, problem.Lvv)
    Lvv_bar_reference = numpy.einsum(
        'iaq,xia->xq', dense_ovvv_bar, problem.Lov)
    numpy.testing.assert_allclose(
        factor_bars[7], Lov_bar_reference, atol=3e-12, rtol=0.0)
    numpy.testing.assert_allclose(
        factor_bars[8], Lvv_bar_reference, atol=3e-12, rtol=0.0)

    rng = numpy.random.default_rng(31)
    dLov = rng.normal(size=problem.Lov.shape)
    dLvv = rng.normal(size=problem.Lvv.shape)
    dLov /= numpy.linalg.norm(dLov)
    dLvv /= numpy.linalg.norm(dLvv)
    directional_ad = (
        numpy.vdot(factor_bars[7], dLov).real
        + numpy.vdot(factor_bars[8], dLvv).real
    )
    step = 2e-5

    def displaced_factor_energy(scale):
        return ccsd_t._ccsd_t_energy_df(
            problem.mat, problem.t1T, problem.t2T, problem.mo_energy,
            problem.fvo, problem.ovoo, problem.ovov,
            problem.Lov + scale * dLov,
            problem.Lvv + scale * dLvv,
            max_memory, False,
        )

    directional_fd = (
        displaced_factor_energy(step) - displaced_factor_energy(-step)
    ) / (2 * step)
    numpy.testing.assert_allclose(
        directional_ad, directional_fd, atol=2e-10, rtol=0.0)

    nocc = problem.t1T.shape[1]
    nvir = problem.t1T.shape[0]
    nmo = nocc + nvir
    assert cache_shapes
    assert not any(
        shape == (nvir, nvir, nocc, nmo) for shape in cache_shapes
    )
    assert any(shape[0] == 2 for shape in cache_shapes)
    assert any(shape[1] == 2 for shape in cache_shapes)


def test_public_and_lazy_factor_paths_never_materialize_ovvv(monkeypatch):
    problem = _random_problem(seed=19, nocc=2, nvir=4, naux=6)
    monkeypatch.setattr(
        ccsd_t, '_ccsd_t_factor_block_nvir',
        lambda nocc, nvir, memory, backward=False: min(2, nvir),
    )
    monkeypatch.setattr(
        ccsd_t, '_fill_vvop',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('legacy global vvop was constructed')),
    )

    def energy(Lov, Lvv, profile_pass=None):
        def forbidden_ovvv():
            raise AssertionError('global packed ovvv was materialized')

        t1 = jnp.asarray(problem.t1T.T)
        t2 = jnp.asarray(problem.t2T.transpose(3, 2, 0, 1))
        eris = SimpleNamespace(
            ovvv=None,
            Lov=Lov,
            Lvv=Lvv,
            fock=jnp.zeros((6, 6), dtype=Lov.dtype).at[2:, :2].set(
                jnp.asarray(problem.fvo)),
            mo_energy=jnp.asarray(problem.mo_energy),
            ovoo=jnp.asarray(problem.ovoo),
            ovov=jnp.asarray(problem.ovov),
            get_ovvv_packed=forbidden_ovvv,
        )
        mycc = SimpleNamespace(
            t1=t1,
            t2=t2,
            max_memory=2000,
            profile_pass=profile_pass,
            lno_ccsd_t_timing=False,
            verbose=0,
            stdout=sys.stdout,
        )
        value = ccsd_t.kernel(
            mycc, eris, jnp.asarray(problem.ulo),
            t1=t1, t2=t2, verbose=0)
        assert eris.ovvv is None
        return value

    Lov = jnp.asarray(problem.Lov)
    Lvv = jnp.asarray(problem.Lvv)
    value, public_bars = jax.value_and_grad(
        energy, argnums=(0, 1))(Lov, Lvv)
    reference = ccsd_t._ccsd_t_energy_df_bwd(
        2000, False, _factor_args(problem), numpy.asarray(1.0))
    numpy.testing.assert_allclose(
        value,
        ccsd_t._ccsd_t_energy_df(*_factor_args(problem), 2000, False),
        atol=2e-13,
        rtol=0.0,
    )
    numpy.testing.assert_allclose(
        public_bars[0], reference[7], atol=3e-12, rtol=0.0)
    numpy.testing.assert_allclose(
        public_bars[1], reference[8], atol=3e-12, rtol=0.0)

    # Production fragment replay skips the primal (T) contraction but must
    # invoke exactly the same factor-direct pullback.
    lazy_value, lazy_bars = jax.value_and_grad(
        lambda Lov_, Lvv_: energy(Lov_, Lvv_, 'backward replay'),
        argnums=(0, 1),
    )(Lov, Lvv)
    numpy.testing.assert_allclose(lazy_value, 0.0, atol=0.0, rtol=0.0)
    numpy.testing.assert_allclose(
        lazy_bars[0], reference[7], atol=3e-12, rtol=0.0)
    numpy.testing.assert_allclose(
        lazy_bars[1], reference[8], atol=3e-12, rtol=0.0)


def test_factor_direct_outcore_water_dlno_gradient_high_cost(
        tmp_path, monkeypatch):
    """A complete one-fragment DLNO gradient keeps ovvv lazy.

    The outcore CDERI activates the same backward-replay boundary used by
    production fragment calculations.  Zero thresholds and full domains make
    the one-fragment result directly comparable to canonical DF-CCSD(T).
    """
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    cderi_path = tmp_path / 'water-cderi.h5'
    builder = df.DF(mol, auxbasis='weigend', incore=False)
    builder._cderi_to_save = str(cderi_path)
    builder.build()

    def build_mf(mol_):
        mf = scf.RHF(mol_).density_fit(auxbasis='weigend')
        mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        return mf

    def canonical_energy(mol_):
        mf = build_mf(mol_)
        cc = dfccsd.RCCSD(mf)
        cc.kernel()
        return cc.e_tot + cc.ccsd_t()

    def forbid_global_ovvv(_eris):
        raise AssertionError('LNO triples materialized global packed ovvv')

    monkeypatch.setattr(
        lno_ccsd._ChemistsERIs,
        'get_ovvv_packed',
        forbid_global_ovvv,
    )
    monkeypatch.setattr(
        ccsd_t,
        '_fill_vvop',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('LNO triples constructed global vvop')),
    )
    thresholds = IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
        pair_energy=0.0,
    )

    with (
        config_update('pyscfad_moleintor_opt', True),
        config_update('pyscfad_scf_implicit_diff', True),
        config_update('pyscfad_scf_first_order_custom', False),
        config_update('pyscfad_ccsd_implicit_diff', True),
    ):
        local_energy, local_bar = DLNOCCSD.value_and_grad(
            mol,
            build_mf=build_mf,
            thresholds=thresholds,
            pair_energy_model='all',
            force_full_domains=True,
            thresh_occ=0.0,
            thresh_vir=0.0,
            ccsd_t=True,
        )
        canonical_value, canonical_bar = jax.value_and_grad(
            canonical_energy)(mol)

    numpy.testing.assert_allclose(
        local_energy, canonical_value, atol=3e-7, rtol=0.0)
    numpy.testing.assert_allclose(
        numpy.asarray(local_bar.coords),
        numpy.asarray(canonical_bar.coords),
        atol=7e-7,
        rtol=3e-5,
    )
