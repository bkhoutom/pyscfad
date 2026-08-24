import h5py
import jax
import numpy
import pytest

from pyscfad import config_update, df, gto, scf
from pyscfad import numpy as np
from pyscfad.ao2mo import _ao2mo
from pyscfad.df import _cderi_vjp, addons, incore
from pyscfad.mp import dfmp2


def _water():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _write_cderi(mol, auxmol, path):
    with config_update("pyscfad_moleintor_opt", True):
        cderi = numpy.asarray(
            incore.cholesky_eri(
                mol,
                auxmol=auxmol,
                int3c=mol._add_suffix("int3c2e"),
                int2c=mol._add_suffix("int2c2e"),
                aosym="s2ij",
                verbose=0,
            )
        )
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("j3c", data=cderi)
    return cderi


def _dense_nr_e2(mol, auxmol, mo_coeff, orbs_slice):
    cderi = incore.cholesky_eri(
        mol,
        auxmol=auxmol,
        int3c=mol._add_suffix("int3c2e"),
        int2c=mol._add_suffix("int2c2e"),
        aosym="s2ij",
        verbose=0,
    )
    return _ao2mo.nr_e2(cderi, mo_coeff, orbs_slice, aosym="s2")


def test_outcore_nr_e2_vjp_matches_dense_without_full_cderi_bar(
        tmp_path, monkeypatch):
    rng = numpy.random.default_rng(23)
    mol = _water()
    auxmol = addons.make_auxmol(mol, "weigend")
    cderi_path = tmp_path / "cderi.h5"
    _write_cderi(mol, auxmol, cderi_path)

    nmo = 5
    mo_coeff = np.asarray(rng.normal(size=(mol.nao, nmo)))
    orbs_slice = (0, 2, 2, nmo)
    ybar = np.asarray(
        rng.normal(size=(auxmol.nao,
                         (orbs_slice[1] - orbs_slice[0])
                         * (orbs_slice[3] - orbs_slice[2])))
    )

    dense, dense_pullback = jax.vjp(
        lambda mol_, auxmol_, coeff_: _dense_nr_e2(
            mol_, auxmol_, coeff_, orbs_slice
        ),
        mol,
        auxmol,
        mo_coeff,
    )
    dense_bars = dense_pullback(ybar)

    calls = {"forward_stream": 0, "mo_stream": 0, "direct_int3c": 0}
    original_forward = _cderi_vjp._stream_nr_e2_from_cderi_source
    original_mo = _cderi_vjp.nr_e2_mo_coeff_vjp_from_cderi_source
    original_int3c = _cderi_vjp.cholesky_eri_vjp_from_mo_coeff_ybar

    def record_forward(*args, **kwargs):
        calls["forward_stream"] += 1
        return original_forward(*args, **kwargs)

    def record_mo(*args, **kwargs):
        calls["mo_stream"] += 1
        return original_mo(*args, **kwargs)

    def record_int3c(*args, **kwargs):
        calls["direct_int3c"] += 1
        return original_int3c(*args, **kwargs)

    def forbidden_full_cderi(*args, **kwargs):
        raise AssertionError("the full packed CDERI/bar path was used")

    monkeypatch.setattr(
        _cderi_vjp, "_stream_nr_e2_from_cderi_source", record_forward
    )
    monkeypatch.setattr(
        _cderi_vjp, "nr_e2_mo_coeff_vjp_from_cderi_source", record_mo
    )
    monkeypatch.setattr(
        _cderi_vjp, "cholesky_eri_vjp_from_mo_coeff_ybar", record_int3c
    )
    monkeypatch.setattr(
        _cderi_vjp, "nr_e2_vjp_from_cderi_source", forbidden_full_cderi
    )
    monkeypatch.setattr(
        _cderi_vjp, "cholesky_eri_vjp_from_cderi_source",
        forbidden_full_cderi,
    )

    streamed, streamed_pullback = jax.vjp(
        lambda mol_, auxmol_, coeff_: dfmp2._outcore_nr_e2(
            mol_, auxmol_, coeff_, str(cderi_path), 1000,
            orbs_slice, "s2"
        ),
        mol,
        auxmol,
        mo_coeff,
    )
    streamed_bars = streamed_pullback(ybar)

    numpy.testing.assert_allclose(
        numpy.asarray(streamed), numpy.asarray(dense), atol=1e-10, rtol=1e-10
    )
    numpy.testing.assert_allclose(
        numpy.asarray(streamed_bars[0].coords),
        numpy.asarray(dense_bars[0].coords),
        atol=2e-8,
        rtol=2e-8,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(streamed_bars[1].coords),
        numpy.asarray(dense_bars[1].coords),
        atol=2e-8,
        rtol=2e-8,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(streamed_bars[2]),
        numpy.asarray(dense_bars[2]),
        atol=1e-9,
        rtol=1e-9,
    )
    assert calls["forward_stream"] >= 2
    assert calls["mo_stream"] == 1
    assert calls["direct_int3c"] == 1


def test_outcore_nr_e2_coordinate_vjp_matches_directional_finite_difference(
        tmp_path):
    rng = numpy.random.default_rng(31)
    mol = _water()
    auxmol = addons.make_auxmol(mol, "weigend")
    cderi_path = tmp_path / "cderi.h5"
    _write_cderi(mol, auxmol, cderi_path)

    nmo = 4
    mo_coeff = np.asarray(rng.normal(size=(mol.nao, nmo)))
    orbs_slice = (0, 2, 2, nmo)
    ybar = np.asarray(rng.normal(size=(auxmol.nao, 4)))
    direction = rng.normal(size=(mol.natm, 3))
    direction -= direction.mean(axis=0)
    direction /= numpy.linalg.norm(direction)

    _, pullback = jax.vjp(
        lambda mol_, auxmol_: dfmp2._outcore_nr_e2(
            mol_, auxmol_, mo_coeff, str(cderi_path), 1000,
            orbs_slice, "s2"
        ),
        mol,
        auxmol,
    )
    mol_bar, auxmol_bar = pullback(ybar)
    analytic = numpy.einsum("ix,ix->", mol_bar.coords, direction)
    analytic += numpy.einsum("ix,ix->", auxmol_bar.coords, direction)

    def displaced_objective(step):
        displaced = mol.copy()
        displaced.set_geom_(
            numpy.asarray(mol.coords) + step * direction, unit="Bohr"
        )
        displaced_aux = addons.make_auxmol(displaced, "weigend")
        cderi = incore.cholesky_eri(
            displaced,
            auxmol=displaced_aux,
            int3c=displaced._add_suffix("int3c2e"),
            int2c=displaced._add_suffix("int2c2e"),
            aosym="s2ij",
            verbose=0,
        )
        transformed = _ao2mo.nr_e2(
            cderi, mo_coeff, orbs_slice, aosym="s2"
        )
        return float(numpy.einsum("pq,pq->", transformed, ybar))

    step = 2e-4
    finite_difference = (
        displaced_objective(step) - displaced_objective(-step)
    ) / (2 * step)
    assert analytic == pytest.approx(finite_difference, abs=2e-6, rel=2e-6)


@pytest.mark.parametrize("moleintor_opt", [False, True])
def test_outcore_nr_e2_preserves_generic_float32_backend(
        tmp_path, moleintor_opt):
    rng = numpy.random.default_rng(37)
    mol = _water()
    auxmol = addons.make_auxmol(mol, "weigend")
    cderi_path = tmp_path / "cderi.h5"
    cderi = _write_cderi(mol, auxmol, cderi_path)

    nmo = 4
    mo_coeff = np.asarray(
        rng.normal(size=(mol.nao, nmo)).astype(numpy.float32)
    )
    orbs_slice = (0, 2, 2, nmo)
    ybar = np.asarray(rng.normal(size=(auxmol.nao, 4)))

    with config_update("pyscfad_moleintor_opt", moleintor_opt):
        expected, expected_pullback = jax.vjp(
            lambda coeff: _ao2mo.nr_e2_gen(
                np.asarray(cderi), coeff, orbs_slice, aosym="s2"
            ),
            mo_coeff,
        )
        actual, actual_pullback = jax.vjp(
            lambda coeff: dfmp2._outcore_nr_e2(
                mol, auxmol, coeff, str(cderi_path), 1000,
                orbs_slice, "s2"
            ),
            mo_coeff,
        )
        expected_bar, = expected_pullback(ybar)
        actual_bar, = actual_pullback(ybar)

    numpy.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    numpy.testing.assert_allclose(
        actual_bar, expected_bar, atol=2e-5, rtol=2e-5
    )


def test_dfmp2_energy_gradient_uses_streamed_outcore_transform(
        tmp_path, monkeypatch):
    mol = _water()
    auxbasis = "weigend"
    cderi_path = tmp_path / "cderi.h5"
    builder = df.DF(mol, auxbasis=auxbasis, incore=False)
    builder._cderi_to_save = str(cderi_path)
    builder.build()

    def energy(mol_, *, outcore):
        mf = scf.RHF(mol_).density_fit(auxbasis=auxbasis)
        if outcore:
            mf.with_df.attach_outcore_cderi(str(cderi_path))
        mf.conv_tol = 1e-12
        mf.kernel()
        e_corr, _ = dfmp2.MP2(mf).kernel(with_t2=False)
        return mf.e_tot + e_corr

    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        dense_energy, dense_gradient = jax.value_and_grad(
            lambda mol_: energy(mol_, outcore=False)
        )(mol)

        def forbidden_full_cderi(*args, **kwargs):
            raise AssertionError("canonical DF-MP2 used the full CDERI/bar VJP")

        monkeypatch.setattr(
            _cderi_vjp, "nr_e2_vjp_from_cderi_source", forbidden_full_cderi
        )
        monkeypatch.setattr(
            _cderi_vjp, "cholesky_eri_vjp_from_cderi_source",
            forbidden_full_cderi,
        )
        streamed_energy, streamed_gradient = jax.value_and_grad(
            lambda mol_: energy(mol_, outcore=True)
        )(mol)

    assert float(streamed_energy) == pytest.approx(
        float(dense_energy), abs=2e-10, rel=2e-10
    )
    numpy.testing.assert_allclose(
        numpy.asarray(streamed_gradient.coords),
        numpy.asarray(dense_gradient.coords),
        atol=2e-7,
        rtol=2e-7,
    )
