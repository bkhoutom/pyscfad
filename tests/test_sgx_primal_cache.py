"""Repeated root setup reuses values without suppressing their derivatives."""
import numpy as np
import pytest

from .test_sgx import mol, build, moved


def test_exact_primal_cache_reuses_components_and_protects_outputs(mol, monkeypatch):
    mf = build(mol)
    backend = mf._sgx_backend
    tol = mf.direct_scf_tol
    obj = backend.native(mol, tol)
    original = obj.get_jk
    calls = []
    def measured(*args, **kwargs):
        calls.append((kwargs['with_j'], kwargs['with_k']))
        return original(*args, **kwargs)
    monkeypatch.setattr(obj, 'get_jk', measured)
    dm = np.eye(mol.nao)
    j, _ = backend.primal_jk(mol, dm, True, False, tol)
    both = backend.primal_jk(mol, dm.copy(), True, True, tol)
    np.testing.assert_array_equal(j, both[0])
    both[0][:] = 100
    both[1][:] = 100
    repeated = backend.primal_jk(mol, dm, True, True, tol)
    np.testing.assert_array_equal(repeated[0], j)
    assert not np.all(repeated[1] == 100)
    assert calls == [(True, False), (False, True)]
    changed = dm.copy()
    changed[0, 0] += 1e-14
    backend.primal_jk(mol, changed, True, True, tol)
    assert len(calls) == 3  # Never merge merely close densities.
    assert backend.primal_cache_stats['retained_bytes'] == 6*dm.nbytes


@pytest.mark.parametrize('change', ['geometry', 'tolerance', 'capacity', 'budget', 'scratch_origin'])
def test_primal_cache_invalidates_and_obeys_bounds(mol, change):
    mf = build(mol)
    backend, tol = mf._sgx_backend, mf.direct_scf_tol
    original_workspace = backend.memory
    dm = np.eye(mol.nao)
    backend.primal_jk(mol, dm, True, True, tol)
    current = mol
    if change == 'geometry':
        current = moved(mol, 1e-3)
    elif change == 'tolerance':
        tol *= 10
    elif change == 'capacity':
        for scale in (2., 3.):
            backend.primal_jk(mol, dm*scale, True, True, tol)
    elif change == 'scratch_origin':
        from pyscf.gto.mole import AS_RINV_ORIG_ATOM
        current = mol.copy()
        current._env = mol._env.copy()
        current._env[AS_RINV_ORIG_ATOM] = 1
    else:
        backend.memory = 1e-9
    before = backend.primal_cache_stats.copy()
    actual = backend.primal_jk(current, dm, True, True, tol)
    if change == 'budget':
        assert backend.primal_cache_stats['retained_bytes'] == 0
        assert not backend._primal_cache
        # Restore calculation workspace for the independent uncached build.
        # A sub-byte allowance cannot hold an aligned generated grid panel.
        backend.memory = original_workspace
    elif change == 'scratch_origin':
        assert backend.primal_cache_stats['hits'] == before['hits']+1
    else:
        assert backend.primal_cache_stats['misses'] == before['misses']+1
    expected = backend.native(current, tol).get_jk(dm, hermi=1,
                    with_j=True, with_k=True, direct_scf_tol=tol)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
