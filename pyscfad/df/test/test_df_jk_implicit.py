import jax
import numpy
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad import numpy as np
from pyscfad._src import implicit_diff as implicit_diff_impl
from pyscfad.df import _df_jk_opt


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
        rng.normal(size=(mol.nao, mol.nao)) * 1e-4
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
        # A general, nonstationary orbital cotangent exercises the same SCF
        # response boundary used by local correlation methods.
        return (
            mf.e_tot
            + np.vdot(coeff_weight, mf.mo_coeff).real
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
        }
        is_solve_matvec = _df_jk_opt.is_implicit_diff_solve_matvec
        coordinate_vjp = _df_jk_opt._cderi_mol_aux_vjp_from_block_fn
        mkstemp = _df_jk_opt.tempfile.mkstemp

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

        outcore_value, outcore_gradient = jax.value_and_grad(
            lambda mol_: objective(mol_, outcore=True)
        )(mol)

    # There must have been multiple Krylov applications; the expensive DF
    # argument VJP and its HDF scratch are instead evaluated only outside the
    # marked solve and remain necessary for the final nuclear gradient.
    assert calls["solve_matvec"] >= 2
    assert calls["coordinate_vjp"] >= 1
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
