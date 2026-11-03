"""Forward Boys dipole caching: exact CIAH algebra and cache lifetime."""
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pytest
from scipy.linalg import expm
from pyscf import gto, scf
from pyscf.lo import boys as pyscf_boys
from pyscfad.lo import boys


class UncachedBoys(pyscf_boys.Boys):
    """The original optimizer, retaining PySCF integral evaluation methods."""
    get_init_guess = boys.Boys.get_init_guess
    kernel = boys.kernel


@pytest.fixture(scope='module')
def reference():
    mol = gto.M(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    return mol, mf.mo_coeff[:, 1:5]


def tight(solver):
    solver.conv_tol, solver.conv_tol_grad = 1e-12, 1e-9
    solver.ah_conv_tol = solver.ah_lindep = 1e-22
    return solver


def count_integrals(monkeypatch, mol):
    counts = Counter()
    original = mol.intor

    def counted(name, *args, **kwargs):
        counts[name] += 1
        return original(name, *args, **kwargs)

    monkeypatch.setattr(mol, 'intor', counted)
    return counts


def test_projected_objective_gradient_and_hessian_action(reference):
    mol, coeff = reference
    original = UncachedBoys(mol, coeff)
    cached = boys.Boys(mol, coeff)
    rng = np.random.default_rng(410)
    with cached._cache_dipoles():
        for scale in (0., .2, 1.):
            generator = rng.normal(size=(4, 4)) * scale
            u = expm(generator-generator.T)
            g, hop, diagonal = cached.gen_g_hop(u)
            refg, refhop, refdiag = original.gen_g_hop(u)
            direction = rng.normal(size=g.shape)
            np.testing.assert_allclose(cached.cost_function(u), original.cost_function(u), atol=1e-12)
            for actual, expected in ((g, refg), (diagonal, refdiag),
                                     (hop(direction), refhop(direction)),
                                     (cached.get_grad(u), original.get_grad(u))):
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
    assert not hasattr(cached, '_boys_dipole_cache')


def test_complete_kernel_builds_moments_once(reference, monkeypatch):
    mol, coeff = reference
    counts = count_integrals(monkeypatch, mol)
    cycles = []
    original = tight(UncachedBoys(mol, coeff))
    expected = original.kernel(callback=lambda state: cycles.append(state['imacro']))
    assert len(cycles) > 1
    assert counts['int1e_r'] > len(cycles)
    assert counts['int1e_r2'] == len(cycles)
    counts.clear()
    cycles.clear()
    cached = tight(boys.Boys(mol, coeff))
    actual = cached.kernel(callback=lambda state: cycles.append(state['imacro']))
    assert len(cycles) > 1
    assert counts['int1e_r'] == counts['int1e_r2'] == 1
    assert not hasattr(cached, '_boys_dipole_cache')
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    assert np.linalg.norm(cached.get_grad()) < cached.conv_tol_grad


def test_implicit_forward_includes_final_check_in_cache(reference, monkeypatch):
    mol, coeff = reference
    counts = count_integrals(monkeypatch, mol)
    options = dict(init_guess='atomic', conv_tol=1e-12, conv_tol_grad=1e-9)
    actual = boys._boys(None, mol, coeff, **options)
    assert counts['int1e_r'] == counts['int1e_r2'] == 1
    # Keep precisely the same forward wrapper and convergence controls, but
    # substitute the original AO integral path for an independent comparison.
    monkeypatch.setattr(boys.Boys, '_cache_dipoles', lambda *args, **kwargs: nullcontext())
    expected = boys._boys(None, mol, coeff, **options)
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-10, rtol=2e-10)
    np.testing.assert_array_equal(actual[1], expected[1])


def test_cache_is_rebuilt_for_new_coefficients_and_geometry(reference, monkeypatch):
    mol, coeff = reference
    cached = tight(boys.Boys(mol, coeff))
    cached.kernel()
    new_mol = mol.set_geom_('O 0 0 0; H .16 -.72 .56; H -.04 .85 .64', inplace=False)
    mf = scf.RHF(new_mol).run(conv_tol=1e-12)
    new_coeff = mf.mo_coeff[:, :3]  # Different subspace and cache dimensions.
    expected = tight(UncachedBoys(new_mol, new_coeff)).kernel(mo_coeff=new_coeff)
    counts = count_integrals(monkeypatch, new_mol)
    cached.mol = new_mol
    actual = cached.kernel(mo_coeff=new_coeff)
    assert counts['int1e_r'] == counts['int1e_r2'] == 1
    assert not hasattr(cached, '_boys_dipole_cache')
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)


def test_callback_failure_does_not_leave_stale_cache(reference):
    mol, coeff = reference
    cached = tight(boys.Boys(mol, coeff))

    def failed(state):
        raise RuntimeError('callback failure')

    with pytest.raises(RuntimeError, match='callback failure'):
        cached.kernel(callback=failed)
    assert not hasattr(cached, '_boys_dipole_cache')
    expected = tight(UncachedBoys(mol, coeff)).kernel()
    np.testing.assert_allclose(cached.kernel(), expected, atol=2e-10, rtol=2e-10)


def test_single_orbital_requires_no_integrals(reference, monkeypatch):
    mol, coeff = reference
    counts = count_integrals(monkeypatch, mol)
    actual = boys.Boys(mol, coeff[:, :1]).kernel()
    np.testing.assert_array_equal(actual, coeff[:, :1])
    assert not counts
