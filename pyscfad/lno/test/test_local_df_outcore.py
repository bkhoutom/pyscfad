import gc
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy
import pytest

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



def test_embed_local_coefficients_preserves_rows_and_rejects_invalid_pair_maps():
    coeff = numpy.arange(6., dtype=numpy.float64).reshape(3, 2)
    pairs = [0, 3, 5, 15, 17, 20]  # AO labels [0, 2, 5].
    embedded = lno_base._embed_local_mo_coeff_from_pair_idx(coeff, pairs, 6)
    numpy.testing.assert_array_equal(embedded[[0, 2, 5]], coeff)
    numpy.testing.assert_array_equal(embedded[[1, 3, 4]], 0.)
    numpy.testing.assert_array_equal(
        lno_base._embed_local_mo_coeff_from_pair_idx(coeff, range(6), 3), coeff)
    assert lno_base._embed_local_mo_coeff_from_pair_idx(
        numpy.zeros((0, 2)), [], 6).shape == (6, 2)
    for bad in (pairs[:-1], [0, 4, 5, 15, 17, 20], [-1, 3, 5, 15, 17, 20],
                [0, 3, 5, 15, 17, 21], [5, 3, 0, 17, 15, 20]):
        with pytest.raises(ValueError):
            lno_base._embed_local_mo_coeff_from_pair_idx(coeff, bad, 6)


def test_partial_global_df_preserves_overlap_projection_response(tmp_path, monkeypatch):
    import h5py
    from pyscfad.df import addons, incore
    from pyscfad.ao2mo import _ao2mo

    mol = _water_mol()
    auxmol = addons.make_auxmol(mol, 'weigend')
    ao_idx = numpy.asarray([0, 1, 2, 3, 4, 6])  # O and the second H.
    pairs = tuple(lno_base._global_pair_indices_for_local_ao(ao_idx, mol.nao))
    rows, cols = numpy.tril_indices(mol.nao)

    def packed_cderi(m, a):
        # The dense AD implementation may use s1 even when s2 is requested.
        full = incore.cholesky_eri(m, auxmol=a, aosym='s1')
        return full.reshape(a.nao, m.nao, m.nao)[:, rows, cols]

    source = str(tmp_path / 'global-cderi.h5')
    with h5py.File(source, 'w') as h5:
        h5.create_dataset('j3c', data=numpy.asarray(packed_cderi(mol, auxmol)))
    rng = numpy.random.default_rng(63)
    coeff = np.asarray(rng.normal(size=(mol.nao, 3)))
    cotangent = np.asarray(rng.normal(size=(auxmol.nao, 2)))
    orbs_slice = (0, 1, 1, 3)

    def actual(m, a, c):
        holder = SimpleNamespace(mol=m, auxmol=a, max_memory=200,
            _get_cderi_source=lambda: source, _has_outcore_cderi_placeholder=lambda: True)
        mf = SimpleNamespace(mol=m, with_df=holder, get_ovlp=lambda: m.intor('int1e_ovlp'))
        y = lno_base.transform_df_to_mo(mf, c, orbs_slice, atmlst=[0, 2])
        return np.sum(y * cotangent)

    def reference(m, a, c):
        overlap = m.intor('int1e_ovlp')
        local = jax.scipy.linalg.solve(overlap[np.ix_(ao_idx, ao_idx)], overlap[ao_idx] @ c,
                                       assume_a='pos')
        b = packed_cderi(m, a)
        y = _ao2mo.nr_e2(b[:, np.asarray(pairs)], local, orbs_slice, aosym='s2')
        return np.sum(y * cotangent)

    expected, expected_grad = jax.value_and_grad(reference, argnums=(0, 1, 2))(mol, auxmol, coeff)

    def reject_disk(*args, **kwargs):
        raise AssertionError('Projected partial domain should use the direct VJP')

    monkeypatch.setattr(lno_base._cderi_vjp, 'nr_e2_cderi_bar_packed_disk', reject_disk)
    value, grad = jax.value_and_grad(actual, argnums=(0, 1, 2))(mol, auxmol, coeff)
    numpy.testing.assert_allclose(value, expected, atol=1e-8, rtol=1e-8)
    for result, ref in zip(jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(expected_grad)):
        numpy.testing.assert_allclose(result, ref, atol=1e-8, rtol=1e-8)
