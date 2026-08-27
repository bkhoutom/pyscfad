import jax
import numpy
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad import numpy as np
from pyscfad._src import implicit_diff as implicit_diff_impl
from pyscfad.df import _df_jk_opt
from pyscfad.scf import hf


def _water():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


@pytest.mark.skipif(
    _df_jk_opt._DF_VK_DM_VJP is None,
    reason="pyscfadlib was built without the density-only DF-K transpose",
)
def test_density_only_df_jk_vjp_matches_full_dm_cotangent(monkeypatch):
    rng = numpy.random.default_rng(31)
    mol = _water()
    with config_update("pyscfad_moleintor_opt", True):
        dfobj = df.DF(mol, auxbasis="weigend")
        dfobj.build()

    dm = rng.normal(size=(mol.nao, mol.nao))
    dm = np.asarray(dm + dm.T)
    vj_bar = np.asarray(rng.normal(size=dm.shape))
    vk_bar = np.asarray(rng.normal(size=dm.shape))

    with config_update("pyscfad_moleintor_opt", True):
        _, pullback = jax.vjp(
            lambda dm_: _df_jk_opt.get_jk(dfobj, dm_), dm
        )
        dm_bar_full, = pullback((vj_bar, vk_bar))

    calls = {"native_density_only": 0, "full_eri_allocation": 0}
    native_density_only = _df_jk_opt._DF_VK_DM_VJP
    numpy_zeros = _df_jk_opt.numpy.zeros
    eri_bar_shape = (
        dfobj.get_naoaux(), mol.nao * (mol.nao + 1) // 2
    )

    def record_native_density_only(*args):
        calls["native_density_only"] += 1
        native_density_only(*args)

    def record_zeros(shape, *args, **kwargs):
        shape_tuple = tuple(shape) if numpy.iterable(shape) else (shape,)
        if shape_tuple == eri_bar_shape:
            calls["full_eri_allocation"] += 1
        return numpy_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(
        _df_jk_opt, "_DF_VK_DM_VJP", record_native_density_only
    )
    monkeypatch.setattr(_df_jk_opt.numpy, "zeros", record_zeros)
    with (
        config_update("pyscfad_moleintor_opt", True),
        implicit_diff_impl._implicit_diff_solve_matvec(),
    ):
        dm_bar_fast, = pullback((vj_bar, vk_bar))

    assert calls["native_density_only"] > 0
    assert calls["full_eri_allocation"] == 0
    numpy.testing.assert_allclose(
        numpy.asarray(dm_bar_fast),
        numpy.asarray(dm_bar_full),
        atol=2e-11,
        rtol=2e-11,
    )


def test_fast_exchange_cache_is_object_local_and_density_checked(
        tmp_path, monkeypatch):
    """A shared CDERI file must not make two SCF references share orbitals."""
    mol = _water()
    cderi_path = str(tmp_path / "shared-cderi.h5")
    dfobj1 = df.DF(mol, auxbasis="weigend")
    dfobj2 = df.DF(mol, auxbasis="weigend")
    dfobj1.attach_outcore_cderi(cderi_path)
    dfobj2.attach_outcore_cderi(cderi_path)

    assert _df_jk_opt._fast_exchange_key(dfobj1) != \
        _df_jk_opt._fast_exchange_key(dfobj2)
    leaves, tree = jax.tree_util.tree_flatten(dfobj1)
    reconstructed = jax.tree_util.tree_unflatten(tree, leaves)
    assert _df_jk_opt._fast_exchange_key(reconstructed) == \
        _df_jk_opt._fast_exchange_key(dfobj1)

    nao = mol.nao
    mo_coeff = numpy.eye(nao)
    mo_occ = numpy.zeros(nao)
    mo_occ[:mol.nelectron // 2] = 2.0
    _df_jk_opt.set_fast_exchange_dm_data(
        dfobj1, mo_coeff, mo_occ, numpy.eye(nao)
    )
    dm_ref = _df_jk_opt._FAST_EXCHANGE_DM_DATA[
        _df_jk_opt._fast_exchange_key(dfobj1)
    ]["dm_ref"]

    tagged = _df_jk_opt._tag_dm_for_fast_exchange(dfobj1, dm_ref)
    assert getattr(tagged, "mo_coeff", None) is not None
    unrelated_dm = dm_ref + numpy.eye(nao)
    untagged = _df_jk_opt._tag_dm_for_fast_exchange(
        dfobj1, unrelated_dm
    )
    assert getattr(untagged, "mo_coeff", None) is None

    monkeypatch.setenv("PYSCFAD_SCF_FAST_EXCHANGE_CACHE_SIZE", "2")
    _df_jk_opt._FAST_EXCHANGE_DM_DATA.clear()
    objects = [df.DF(mol, auxbasis="weigend") for _ in range(3)]
    for item in objects:
        _df_jk_opt.set_fast_exchange_dm_data(
            item, mo_coeff, mo_occ, numpy.eye(nao)
        )
    assert len(_df_jk_opt._FAST_EXCHANGE_DM_DATA) == 2
    assert _df_jk_opt._fast_exchange_key(objects[0]) not in \
        _df_jk_opt._FAST_EXCHANGE_DM_DATA


def test_implicit_scf_lowrank_rejects_complex_orbital_factors():
    """The real native AO2MO kernel must not receive complex factors."""
    mol = _water()
    dfobj = df.DF(mol, auxbasis="weigend")
    nao = mol.nao
    mo_coeff = numpy.eye(nao, dtype=numpy.complex128)
    mo_occ = numpy.zeros(nao)
    mo_occ[:mol.nelectron // 2] = 2.0
    _df_jk_opt.set_fast_exchange_dm_data(
        dfobj, mo_coeff, mo_occ, numpy.eye(nao)
    )
    data = _df_jk_opt._FAST_EXCHANGE_DM_DATA[
        _df_jk_opt._fast_exchange_key(dfobj)
    ]
    numpy.testing.assert_allclose(data["dm_ref"], data["dm_ref"].conj().T)

    vj_bar = numpy.eye(nao)[None]
    vk_bar = -0.5 * vj_bar
    assert _df_jk_opt._implicit_lowrank_exchange_factors(
        dfobj, vj_bar, vk_bar, 1, True, True
    ) is None


def test_implicit_scf_lowrank_rejects_antisymmetric_cotangent():
    """The factor route may not discard an antisymmetric K cotangent."""
    rng = numpy.random.default_rng(41)
    mol = _water()
    with config_update("pyscfad_moleintor_opt", True):
        mf = scf.RHF(mol).density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-12
        mf.kernel()
        s1e = numpy.asarray(mf.get_ovlp())
        hf._stash_dfjk_mo_data(
            mf, mf.mo_coeff, mf.mo_occ, s1e
        )

    data = _df_jk_opt._FAST_EXCHANGE_DM_DATA[
        _df_jk_opt._fast_exchange_key(mf.with_df)
    ]
    occ_coeff = data["occ_coeff"]
    overlap_occ_coeff = data["overlap_occ_coeff"]
    response = rng.normal(size=occ_coeff.shape)
    response -= occ_coeff @ (overlap_occ_coeff.T @ response)
    symmetric = response @ occ_coeff.T + occ_coeff @ response.T
    raw = rng.normal(size=symmetric.shape)
    vk_bar = symmetric + 1e-2 * (raw - raw.T)
    vj_bar = -2.0 * vk_bar

    factors = _df_jk_opt._implicit_lowrank_exchange_factors(
        mf.with_df, vj_bar[None], vk_bar[None], 1, True, True
    )
    assert factors is None


@pytest.mark.parametrize("symmetric_seed", [True, False])
def test_implicit_scf_lowrank_df_k_matches_dense(
        monkeypatch, symmetric_seed):
    """The factorized route must reproduce the exact SCF Krylov matvec."""
    rng = numpy.random.default_rng(37)
    mol = _water()
    with config_update("pyscfad_moleintor_opt", True):
        mf = scf.RHF(mol).density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        s1e = mf.get_ovlp()
        h1e = mf.get_hcore(s1e=s1e)

        def optimality(dm_):
            return hf._scf_fixed_point(dm_, mf, s1e, h1e) - dm_

        _, pullback = jax.vjp(optimality, dm)

    seed = rng.normal(size=dm.shape)
    if symmetric_seed:
        seed = 0.5 * (seed + seed.T)
    seed = np.asarray(seed)

    monkeypatch.setenv("PYSCFAD_SCF_IMPLICIT_LOWRANK_DF_K", "0")
    with implicit_diff_impl._implicit_diff_solve_matvec():
        dense, = pullback(seed)

    calls = {"lowrank": 0}
    lowrank_kernel = _df_jk_opt._df_vk_dm_vjp_lowrank

    def record_lowrank(*args, **kwargs):
        calls["lowrank"] += 1
        return lowrank_kernel(*args, **kwargs)

    monkeypatch.setattr(
        _df_jk_opt, "_df_vk_dm_vjp_lowrank", record_lowrank
    )
    monkeypatch.setenv("PYSCFAD_SCF_IMPLICIT_LOWRANK_DF_K", "1")
    with implicit_diff_impl._implicit_diff_solve_matvec():
        lowrank, = pullback(seed)

    assert calls["lowrank"] > 0
    numpy.testing.assert_allclose(
        numpy.asarray(lowrank),
        numpy.asarray(dense),
        atol=2e-10,
        rtol=2e-10,
    )


def test_outcore_implicit_scf_matvec_avoids_df_coordinate_vjp(
        tmp_path, monkeypatch):
    """Only post-solve VJPs may create the packed DF cotangent file."""
    rng = numpy.random.default_rng(43)
    mol = _water()
    auxbasis = "weigend"
    cderi_path = tmp_path / "cderi.h5"
    builder = df.DF(mol, auxbasis=auxbasis, incore=False)
    builder._cderi_to_save = str(cderi_path)
    builder.build()

    coeff_weight = np.asarray(
        rng.normal(size=(mol.nao, mol.nao)) * 1e-2
    )
    energy_weight = np.asarray(rng.normal(size=mol.nao) * 1e-4)

    def objective(mol_, *, outcore):
        mf = scf.RHF(mol_).density_fit(auxbasis=auxbasis)
        if outcore:
            mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-6
        mf.max_cycle = 50
        mf.kernel()
        mo_projections = np.einsum(
            "pi,pi->i", coeff_weight, mf.mo_coeff
        )
        # A general, nonstationary orbital cotangent exercises the same SCF
        # response boundary used by local correlation methods.  Squaring each
        # column projection makes the scalar insensitive to arbitrary MO sign
        # choices while retaining its dependence on the individual orbitals.
        return (
            mf.e_tot
            + np.vdot(mo_projections, mo_projections).real
            + np.vdot(energy_weight, mf.mo_energy).real
        )

    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        dense_value, dense_gradient = jax.value_and_grad(
            lambda mol_: objective(mol_, outcore=False)
        )(mol)

        calls = {
            "solve_matvec": 0,
            "coordinate_vjp": 0,
            "eri_bar_file": 0,
            "lowrank_exchange": 0,
        }
        is_solve_matvec = _df_jk_opt.is_implicit_diff_solve_matvec
        coordinate_vjp = _df_jk_opt._cderi_mol_aux_vjp_from_block_fn
        mkstemp = _df_jk_opt.tempfile.mkstemp
        lowrank_exchange = _df_jk_opt._df_vk_dm_vjp_lowrank

        def record_solve_matvec():
            active = is_solve_matvec()
            calls["solve_matvec"] += int(active)
            return active

        def guarded_coordinate_vjp(*args, **kwargs):
            if is_solve_matvec():
                raise AssertionError(
                    "implicit GMRES matvec requested a DF coordinate VJP"
                )
            calls["coordinate_vjp"] += 1
            return coordinate_vjp(*args, **kwargs)

        def guarded_mkstemp(*args, **kwargs):
            prefix = kwargs.get("prefix", "")
            if prefix.startswith("pyscfad_dfjk_eri_bar_"):
                if is_solve_matvec():
                    raise AssertionError(
                        "implicit GMRES matvec created a packed eri_bar file"
                    )
                calls["eri_bar_file"] += 1
            return mkstemp(*args, **kwargs)

        def record_lowrank_exchange(*args, **kwargs):
            calls["lowrank_exchange"] += 1
            return lowrank_exchange(*args, **kwargs)

        monkeypatch.setattr(
            _df_jk_opt,
            "is_implicit_diff_solve_matvec",
            record_solve_matvec,
        )
        monkeypatch.setattr(
            _df_jk_opt,
            "_cderi_mol_aux_vjp_from_block_fn",
            guarded_coordinate_vjp,
        )
        monkeypatch.setattr(
            _df_jk_opt.tempfile, "mkstemp", guarded_mkstemp
        )
        monkeypatch.setattr(
            _df_jk_opt,
            "_df_vk_dm_vjp_lowrank",
            record_lowrank_exchange,
        )

        outcore_value, outcore_gradient = jax.value_and_grad(
            lambda mol_: objective(mol_, outcore=True)
        )(mol)

    # There must have been multiple Krylov applications; the expensive DF
    # argument VJP and its HDF scratch are instead evaluated only outside the
    # marked solve and remain necessary for the final nuclear gradient.
    assert calls["solve_matvec"] >= 2
    assert calls["coordinate_vjp"] >= 1
    assert calls["lowrank_exchange"] >= 1
    assert calls["eri_bar_file"] == calls["coordinate_vjp"]
    assert calls["coordinate_vjp"] < calls["solve_matvec"]
    assert float(outcore_value) == pytest.approx(
        float(dense_value), abs=3e-10, rel=3e-10
    )
    numpy.testing.assert_allclose(
        numpy.asarray(outcore_gradient.coords),
        numpy.asarray(dense_gradient.coords),
        atol=3e-7,
        rtol=3e-7,
    )
