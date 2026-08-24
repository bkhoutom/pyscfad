import gc
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy

from pyscfad import df, gto
from pyscfad import numpy as np
from pyscfad.lno import lno_base


def _water_mol():
    mol = gto.Mole(
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        basis="sto-3g",
        verbose=0,
        max_memory=200,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _df_holder(mol, incore):
    with_df = df.DF(mol, auxbasis="weigend", incore=incore)
    with_df.max_memory = mol.max_memory
    return SimpleNamespace(mol=mol, with_df=with_df)


def _local_coeff(mol, atmlst):
    ao_idx = lno_base.dlno_util.ao_index_by_atom(mol, atmlst)
    rng = numpy.random.default_rng(14)
    return np.asarray(rng.normal(size=(ao_idx.size, 5)))


def test_local_lov_outcore_is_blocked_and_matches_incore(monkeypatch):
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    nocc = 2

    mf_incore = _df_holder(mol, incore=True)
    lov_incore = lno_base.get_local_Lov(
        mf_incore, coeff, nocc, atmlst
    )

    mf_outcore = _df_holder(mol, incore=False)
    entry = lno_base.get_local_df(mf_outcore, atmlst)
    fake_mol, local_df, _ = entry
    assert local_df._has_outcore_cderi_placeholder()
    source = local_df._get_cderi_source()
    source_path = Path(source.name)
    assert source_path.exists()

    # Force one auxiliary function per slab and record the actual transforms.
    monkeypatch.setattr(lno_base, "_outcore_nr_e2_block_mb", lambda: 1e-6)
    nr_e2 = lno_base._ao2mo.nr_e2
    slab_sizes = []

    def record_nr_e2(cderi, *args, **kwargs):
        slab_sizes.append(cderi.shape[0])
        return nr_e2(cderi, *args, **kwargs)

    monkeypatch.setattr(lno_base._ao2mo, "nr_e2", record_nr_e2)
    lov_outcore = lno_base.get_local_Lov(
        mf_outcore, coeff, nocc, atmlst
    )

    numpy.testing.assert_allclose(
        numpy.asarray(lov_outcore), numpy.asarray(lov_incore),
        atol=1e-11, rtol=1e-11,
    )
    assert len(slab_sizes) > 1
    assert max(slab_sizes) == 1

    # The unnamed local CDERI belongs to the cache entry.  Eviction releases
    # it; caller-owned named sources would not be removed by this mechanism.
    mf_outcore.with_df._lno_local_df_cache.clear()
    del entry, fake_mol, local_df, source
    gc.collect()
    assert not source_path.exists()


def test_local_lov_outcore_preserves_mo_coefficient_vjp():
    mol = _water_mol()
    atmlst = numpy.asarray([0, 1], dtype=numpy.int32)
    coeff = _local_coeff(mol, atmlst)
    nocc = 2
    mf_incore = _df_holder(mol, incore=True)
    mf_outcore = _df_holder(mol, incore=False)

    def objective(mf, coeff_):
        lov = lno_base.get_local_Lov(mf, coeff_, nocc, atmlst)
        return np.einsum("Lia,Lia->", lov, lov)

    value_incore, grad_incore = jax.value_and_grad(
        lambda coeff_: objective(mf_incore, coeff_)
    )(coeff)
    value_outcore, grad_outcore = jax.value_and_grad(
        lambda coeff_: objective(mf_outcore, coeff_)
    )(coeff)

    numpy.testing.assert_allclose(
        numpy.asarray(value_outcore), numpy.asarray(value_incore),
        atol=1e-11, rtol=1e-11,
    )
    numpy.testing.assert_allclose(
        numpy.asarray(grad_outcore), numpy.asarray(grad_incore),
        atol=1e-10, rtol=1e-10,
    )
