"""DF-J Schwarz bounds, fixed response masks and cached forward tolerances."""
import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.df import df_jk
from pyscfad.sgx.sgx import _configure
from pyscfad.sgx import _dfj


@pytest.fixture(params=[False, True])
def obj(request):
    mol = gto.M(atom='O 0 0 0; H 0 -.7 .6; H 0 .7 .6; '
                     'O 0 0 14; H 0 -.7 14.6; H 0 .7 14.6',
                basis='cc-pvdz', cart=request.param, verbose=0)
    result = _configure(mol, 'weigend', 0, True, 100, 100)
    dm = scf.hf.init_guess_by_minao(mol)
    _dfj.get_j(result, dm)
    _dfj._initialize(result, dm, 1e-13)
    return result


def test_derivative_schwarz_bounds_cover_actual_shell_integrals(obj):
    opt = obj._vjopt
    n = obj.mol.nbas
    for name in ('int3c2e', 'int3c2e_ip1', 'int3c2e_ip2'):
        screen = _dfj._screen_optimizer(opt, name, 1e-13)
        q = screen.q_cond
        counts = screen._sgx_screening_counts
        assert 0 < counts['skipped_shell_triples'] < counts['candidate_shell_triples']
        for i, j in [(0, 0), (0, 2), (2, 0), (4, 3), (n-1, 0), (n-2, n-1)]:
            for k in (0, 1, 5, obj.auxmol.nbas-1):
                block = opt.fakemol.intor(name, comp=1 if name == 'int3c2e' else 3,
                    shls_slice=(i, i+1, j, j+1, n+k, n+k+1))
                bound = q[i*n+j]*q[n*n+k]
                assert np.max(np.abs(block)) <= bound*(1+1e-10)+1e-25


def test_screened_response_and_geometry_against_unscreened(obj, monkeypatch):
    n = obj.mol.nao
    rng = np.random.default_rng(98)
    dms = rng.normal(size=(2, n, n))
    dms = .5*(dms+dms.transpose(0, 2, 1))
    screen = _dfj._screen_optimizer
    screened = _dfj.density_response(obj, dms)
    grad = _dfj.geometry_many(obj, [(dms[0], dms[1])])
    tol = obj._vjopt.direct_scf_tol
    for scale in (1e-9, 1e-20, -1e-12):
        actual = _dfj.density_response(obj, scale*dms)
        np.testing.assert_allclose(actual/scale, screened, atol=3e-12, rtol=3e-12)
    assert obj._vjopt.direct_scf_tol == tol
    # Both response passes use the same integral mask: self-adjointness.
    np.testing.assert_allclose(np.sum(dms[0]*screened[1]),
                               np.sum(dms[1]*screened[0]), atol=3e-11)
    monkeypatch.setattr(_dfj, '_screen_optimizer', lambda *args: None)
    reference = _dfj.density_response(obj, dms)
    ref_grad = _dfj.geometry_many(obj, [(dms[0], dms[1])])
    np.testing.assert_allclose(screened, reference, atol=1e-10, rtol=0)
    np.testing.assert_allclose(grad, ref_grad, atol=1e-10, rtol=0)
    monkeypatch.setattr(_dfj, '_screen_optimizer', screen)


def test_forward_reapplies_requested_tolerance_and_preserves_cache(obj):
    dm = scf.hf.init_guess_by_minao(obj.mol)
    opt = obj._vjopt
    opt.direct_scf_tol = 1e-5
    actual = _dfj.get_j(obj, dm, tol=1e-13)
    assert obj._vjopt is opt and opt.direct_scf_tol == 1e-5
    opt.direct_scf_tol = 1e-13
    reference = df_jk.get_j(obj, dm, hermi=1, direct_scf_tol=1e-13)
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=0)
    # Also exercise a real loose-to-tight bound rebuild, not just a changed
    # cutoff on bounds which were already computed tightly.
    obj._vjopt = None
    _dfj.get_j(obj, dm, tol=1e-5)
    loose_opt = obj._vjopt
    actual = _dfj.get_j(obj, dm, tol=1e-13)
    assert obj._vjopt is loose_opt
    assert loose_opt._sgx_q_cond_tol == 1e-13
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=0)
