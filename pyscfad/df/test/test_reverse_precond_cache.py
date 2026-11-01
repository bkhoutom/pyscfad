"""The Coulomb response preconditioner retains bounded fixed DF factors."""
import h5py
import numpy as np
import pytest
from pyscf import lib

from pyscfad import gto, scf
from pyscfad.df import _df_jk_opt


@pytest.fixture
def problem(tmp_path, monkeypatch):
    # Test the advisory-memory policy independently of JAX allocator pages
    # retained by earlier tests in the same process. Low-memory cases below
    # still select their intended limits explicitly.
    monkeypatch.setattr(_df_jk_opt.lib, 'current_memory', lambda: (0., 0.))
    rng = np.random.default_rng(31415)
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    dfobj = scf.RHF(mol).density_fit().with_df
    nao = mol.nao
    raw = rng.normal(size=(9, nao, nao))
    factors = (raw + raw.transpose(0, 2, 1)) * .15
    path = tmp_path / 'packed.h5'
    with h5py.File(path, 'w') as handle:
        handle['j3c'] = lib.pack_tril(factors)
    dfobj.attach_outcore_cderi(str(path))
    dfobj.blockdim = 3
    orbitals, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    # All three occupied orbitals must contribute, including a hypothetical
    # frozen occupied orbital in a subsequent correlated calculation.
    occupied, virtual = orbitals[:, :3], orbitals[:, 3:]
    gaps = 1 + rng.random((virtual.shape[1], occupied.shape[1]))
    bov = np.einsum('Lpq,pa,qi->Lai', factors, virtual, occupied).reshape(9, -1)
    hessian = np.diag(gaps.ravel()) + 4 * bov.T @ bov
    rhs = [rng.normal(size=gaps.shape) for _ in range(2)]
    reads = []
    loop = dfobj.loop

    def counted_loop(*args, **kwargs):
        for block in loop(*args, **kwargs):
            reads.append(block.shape[0])
            yield block

    monkeypatch.setattr(dfobj, 'loop', counted_loop)
    return dfobj, occupied, virtual, gaps, hessian, rhs, reads


def _assert_hessian_result(coulomb, gaps, hessian, rhs):
    expected = rhs - gaps * np.linalg.solve(hessian, rhs.ravel()).reshape(gaps.shape)
    np.testing.assert_allclose(coulomb(rhs), expected, atol=2e-12, rtol=2e-12)


def test_cached_applications_match_hessian_without_further_disk_reads(problem, monkeypatch):
    monkeypatch.setenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', '1')
    dfobj, occupied, virtual, gaps, hessian, vectors, reads = problem
    coulomb = _df_jk_opt.gen_coulomb_precond(dfobj, occupied, virtual, gaps)
    assert reads
    setup_reads = len(reads)
    for rhs in vectors:
        _assert_hessian_result(coulomb, gaps, hessian, rhs)
    assert len(reads) == setup_reads


@pytest.mark.parametrize('budget', ['0', '0.000001'])
def test_insufficient_budget_streams_each_application(problem, monkeypatch, budget):
    monkeypatch.setenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', budget)
    dfobj, occupied, virtual, gaps, hessian, vectors, reads = problem
    coulomb = _df_jk_opt.gen_coulomb_precond(dfobj, occupied, virtual, gaps)
    previous_reads = len(reads)
    for rhs in vectors:
        _assert_hessian_result(coulomb, gaps, hessian, rhs)
        assert len(reads) > previous_reads
        previous_reads = len(reads)


@pytest.mark.parametrize('budget', ['', 'invalid', '-1', 'nan', 'inf'])
def test_invalid_budget_uses_safe_default(problem, monkeypatch, budget):
    monkeypatch.setenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', budget)
    dfobj, occupied, virtual, gaps, hessian, vectors, reads = problem
    coulomb = _df_jk_opt.gen_coulomb_precond(dfobj, occupied, virtual, gaps)
    setup_reads = len(reads)
    _assert_hessian_result(coulomb, gaps, hessian, vectors[0])
    assert len(reads) == setup_reads


def test_no_available_df_memory_streams_despite_positive_cache_budget(problem, monkeypatch):
    monkeypatch.setenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', '1')
    dfobj, occupied, virtual, gaps, hessian, vectors, reads = problem
    dfobj.max_memory = 0
    coulomb = _df_jk_opt.gen_coulomb_precond(dfobj, occupied, virtual, gaps)
    previous_reads = len(reads)
    for rhs in vectors:
        _assert_hessian_result(coulomb, gaps, hessian, rhs)
        assert len(reads) > previous_reads
        previous_reads = len(reads)


@pytest.mark.parametrize('remaining_mb, expect_cached', [
    (0.00216, True),   # 864-byte factors occupy 40% of remaining memory.
    (0.00144, False),  # The same factors occupy 60% and must stream.
])
def test_cache_uses_at_most_half_remaining_memory(problem, monkeypatch,
                                                 remaining_mb, expect_cached):
    monkeypatch.setenv('PYSCFAD_PRECOND_CACHE_MEMORY_MB', '1')
    dfobj, occupied, virtual, gaps, hessian, vectors, reads = problem
    # Control the process-memory measurement, while exercising the real DF
    # transform, cache/streaming paths and independent Hessian reference.
    monkeypatch.setattr(_df_jk_opt.lib, 'current_memory', lambda: (0., 0.))
    dfobj.max_memory = remaining_mb
    coulomb = _df_jk_opt.gen_coulomb_precond(dfobj, occupied, virtual, gaps)
    setup_reads = len(reads)
    for rhs in vectors:
        _assert_hessian_result(coulomb, gaps, hessian, rhs)
    if expect_cached:
        assert len(reads) == setup_reads
    else:
        assert len(reads) > setup_reads
