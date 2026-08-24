# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from types import SimpleNamespace
import numpy
import jax
import pytest
from pyscf.cc import ccsd_t as pyscf_ccsd_t
from pyscfad import scf
from pyscfad.cc import dfccsd
from pyscfad.cc import ccsd_t
from pyscfad.cc import ccsd_lambda


def test_dfccsdt_nuc_grad(get_opt_mol, monkeypatch):
    """Differentiate factor-native (T) while keeping ``ovvv`` lazy."""

    def forbid_global_ovvv(_eris):
        raise AssertionError('global packed ovvv was materialized')

    # This covers the complete CCSD(T) reverse pass, including the production
    # solve_response_lambda path, rather than only checking the triples call.
    monkeypatch.setattr(
        dfccsd._ChemistsERIs, 'get_ovvv_packed', forbid_global_ovvv
    )

    mol = get_opt_mol
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.kernel()

        mycc = dfccsd.RCCSD(mf)
        eris = mycc.ao2mo()
        mycc.kernel(eris=eris)
        # The triples caches and their cotangents are built directly from the
        # DF factors, so global packed (ov|vv) is never needed.
        assert eris.ovvv is None
        et = mycc.ccsd_t(eris=eris)
        assert eris.ovvv is None
        return mycc.e_tot + et

    e, jac = jax.value_and_grad(energy)(mol)

    e0 = -100.10156178822595
    assert abs(e - e0) < 1e-6

    g0 = numpy.array([[0., 0., -0.0860735932],
                      [0., 0.,  0.0860735932]])
    assert abs(jac.coords-g0).max() < 1e-6


def test_dfccsdt_factor_cache_energy_and_vjp(get_opt_mol):
    """Factor caches reproduce dense (T) and their directional derivative."""
    mol = get_opt_mol
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mycc = dfccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    assert eris.ovvv is None
    assert ccsd_t._can_use_df_factor_triples(
        mycc, eris, mycc.t1, mycc.t2
    )
    symmetry_cc = SimpleNamespace(mol=SimpleNamespace(symmetry=True))
    assert not ccsd_t._can_use_df_factor_triples(
        symmetry_cc, eris, mycc.t1, mycc.t2
    )
    complex_eris = copy.copy(eris)
    complex_eris.Lov = numpy.asarray(eris.Lov, dtype=numpy.complex128)
    assert not ccsd_t._can_use_df_factor_triples(
        mycc, complex_eris, mycc.t1, mycc.t2
    )

    # Zero available memory forces multiple triangular cache blocks even for
    # this small molecule, exercising row and column cache generation.
    cache_memory = 0
    e_factor = ccsd_t._ccsd_t_energy_df(
        mycc, eris, mycc.t1, mycc.t2, cache_memory)
    assert eris.ovvv is None

    dense_eris = copy.copy(eris)
    dense_eris.get_ovvv_packed()
    for name in ('fock', 'mo_energy', 'ovoo', 'ovov', 'ovvv'):
        setattr(dense_eris, name, numpy.asarray(getattr(dense_eris, name)))
    e_dense = pyscf_ccsd_t.kernel(
        mycc, dense_eris, numpy.asarray(mycc.t1),
        numpy.asarray(mycc.t2, order='C'), verbose=0)
    assert abs(e_factor - e_dense) < 1e-13

    unsupported_eris = copy.copy(dense_eris)
    unsupported_eris.ovov = numpy.asarray(
        unsupported_eris.ovov, dtype=numpy.complex128
    )
    with pytest.raises(NotImplementedError, match="real float64"):
        ccsd_t._require_supported_triples_vjp(
            mycc, unsupported_eris, mycc.t1, mycc.t2
        )
    with pytest.raises(NotImplementedError, match="C1"):
        ccsd_t._require_supported_triples_vjp(
            symmetry_cc, dense_eris, mycc.t1, mycc.t2
        )

    factor_bars = ccsd_t._ccsd_t_energy_vjp(
        eris, mycc.t1, mycc.t2, 1., cache_memory,
        use_df_factors=True)
    dense_bars = ccsd_t._ccsd_t_energy_vjp(
        dense_eris, mycc.t1, mycc.t2, 1., cache_memory,
        use_df_factors=False)

    # Cache partitioning must not change any cotangent produced directly by
    # the triples kernel.  The packed-ovvv cotangent from the dense path can
    # also be pulled analytically through ovvv[iaq] = Lov[xia] Lvv[xq].
    for factor_bar, dense_bar in zip(factor_bars[:6], dense_bars[:6]):
        assert numpy.max(numpy.abs(factor_bar - dense_bar)) < 2e-13
    dense_ovvv_bar = dense_bars[6]
    Lov_bar_reference = numpy.einsum(
        'iaq,xq->xia', dense_ovvv_bar, numpy.asarray(eris.Lvv)
    )
    Lvv_bar_reference = numpy.einsum(
        'iaq,xia->xq', dense_ovvv_bar, numpy.asarray(eris.Lov)
    )
    Lov_bar, Lvv_bar = factor_bars[7:]
    assert numpy.max(numpy.abs(Lov_bar - Lov_bar_reference)) < 2e-13
    assert numpy.max(numpy.abs(Lvv_bar - Lvv_bar_reference)) < 2e-13

    rng = numpy.random.default_rng(12)
    dLov = rng.normal(size=eris.Lov.shape)
    dLvv = rng.normal(size=eris.Lvv.shape)
    dLov /= numpy.linalg.norm(dLov)
    dLvv /= numpy.linalg.norm(dLvv)
    directional_ad = (numpy.vdot(Lov_bar, dLov).real
                      + numpy.vdot(Lvv_bar, dLvv).real)

    Lov = numpy.asarray(eris.Lov)
    Lvv = numpy.asarray(eris.Lvv)
    step = 3e-5

    def displaced_energy(scale):
        displaced_eris = copy.copy(eris)
        displaced_eris.Lov = Lov + scale * dLov
        displaced_eris.Lvv = Lvv + scale * dLvv
        return ccsd_t._ccsd_t_energy_df(
            mycc, displaced_eris, mycc.t1, mycc.t2, cache_memory)

    directional_fd = (displaced_energy(step) - displaced_energy(-step)) \
                     / (2 * step)
    assert abs(directional_ad - directional_fd) < 2e-11


def test_dfccsd_factor_lambda_matches_dense(get_opt_mol, monkeypatch):
    """Tiled DF lambda update/solve agrees with the dense ovvv equations."""
    mol = get_opt_mol
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    mycc = dfccsd.RCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)

    # Exercise nonzero virtual-tile offsets and both nested wvvov tiles; the
    # production memory heuristic uses one tile for this small test molecule.
    monkeypatch.setattr(ccsd_lambda, '_DF_OVVV_MAX_BLKSIZE', 2)
    monkeypatch.setattr(ccsd_lambda, '_DF_WVVOV_B_BLKSIZE', 2)

    dense_eris = copy.copy(eris)
    dense_eris.get_ovvv_packed()
    factor_imds = ccsd_lambda.make_intermediates(
        mycc, mycc.t1, mycc.t2, eris
    )
    dense_imds = ccsd_lambda.make_intermediates(
        mycc, mycc.t1, mycc.t2, dense_eris
    )
    assert factor_imds.wvvov is None
    assert eris.ovvv is None

    rng = numpy.random.default_rng(5)
    l1 = numpy.asarray(mycc.t1) + rng.normal(
        scale=1e-3, size=mycc.t1.shape
    )
    l2 = numpy.asarray(mycc.t2) + rng.normal(
        scale=1e-3, size=mycc.t2.shape
    )
    factor_update = ccsd_lambda.update_lambda(
        mycc, mycc.t1, mycc.t2, l1, l2, eris, factor_imds
    )
    dense_update = ccsd_lambda.update_lambda(
        mycc, mycc.t1, mycc.t2, l1, l2, dense_eris, dense_imds
    )
    for factor_value, dense_value in zip(factor_update, dense_update):
        assert numpy.max(numpy.abs(
            numpy.asarray(factor_value) - numpy.asarray(dense_value)
        )) < 1e-12

    factor_conv, factor_l1, factor_l2 = ccsd_lambda.kernel(
        mycc, eris, mycc.t1, mycc.t2,
        max_cycle=50, tol=1e-9, verbose=0
    )
    dense_conv, dense_l1, dense_l2 = ccsd_lambda.kernel(
        mycc, dense_eris, mycc.t1, mycc.t2,
        max_cycle=50, tol=1e-9, verbose=0
    )
    assert factor_conv and dense_conv
    assert numpy.max(numpy.abs(
        numpy.asarray(factor_l1) - numpy.asarray(dense_l1)
    )) < 1e-11
    assert numpy.max(numpy.abs(
        numpy.asarray(factor_l2) - numpy.asarray(dense_l2)
    )) < 1e-11
    assert eris.ovvv is None
