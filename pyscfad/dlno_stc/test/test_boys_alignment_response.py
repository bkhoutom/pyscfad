"""Reference alignment must remove canonical gauge motion before SCF response."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import polar

from pyscfad import gto
from pyscfad.dlno_stc import domain


@pytest.mark.parametrize('singular', [[1., 1., 1.], [0.7, 0.9, 1.1]])
def test_polar_response_at_repeated_singular_values(singular):
    rng = np.random.default_rng(123)
    left = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    right = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    matrix = left @ np.diag(singular) @ right.T
    direction = rng.normal(size=(3, 3))
    value, tangent = jax.jvp(domain._polar_rotation, (jnp.asarray(matrix),),
                            (jnp.asarray(direction),))
    np.testing.assert_allclose(value, polar(matrix)[0], atol=2e-14)
    for h in (3e-5, 1e-5):
        fd = (polar(matrix + h * direction)[0] - polar(matrix - h * direction)[0]) / (2*h)
        np.testing.assert_allclose(tangent, fd, atol=2e-8, rtol=2e-7)


def test_replay_alignment_cancels_gauge_and_keeps_coordinate_response(monkeypatch):
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    s = np.asarray(mol.intor('int1e_ovlp'))
    rng = np.random.default_rng(155)
    coeff = np.linalg.inv(np.linalg.cholesky(s).T) @ np.linalg.qr(rng.normal(size=(7, 4)))[0]
    rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    reference = coeff @ rotation
    static = SimpleNamespace(target_reference_coeff=reference,
                             target_reference_coords=np.asarray(mol.atom_coords()),
                             lo_kwargs={})
    # Hold the Boys rotation fixed to isolate the alignment response. A flat
    # localization direction must not reintroduce the canonical orbital gauge.
    monkeypatch.setattr(domain, '_build_boys_targets', lambda m, c, options: c)

    def align(m, c):
        return domain._rebuild_boys_targets(SimpleNamespace(mol=m), c, static)

    actual, reverse = jax.vjp(align, mol, jnp.asarray(coeff))
    np.testing.assert_allclose(actual, reference, atol=2e-14)
    seed = jnp.asarray(rng.normal(size=coeff.shape))
    mol_bar, coeff_bar = reverse(seed)
    orbital_bar = coeff.T @ np.asarray(coeff_bar)
    np.testing.assert_allclose(orbital_bar, orbital_bar.T, atol=3e-14)

    # Gauge invariance for a finite, unrelated occupied rotation as well.
    other = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    np.testing.assert_allclose(align(mol, coeff @ other), actual, atol=2e-14)
    # The AO cross overlap moves only on its current-geometry side.
    coords = np.asarray(mol.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 1] = 1.
    derivative = np.sum(np.asarray(mol_bar.coords) * direction)
    assert abs(derivative) > 1e-3
    for h in (1e-4, 1e-5):
        plus = mol.set_geom_(coords+h*direction, unit='Bohr', inplace=False)
        minus = mol.set_geom_(coords-h*direction, unit='Bohr', inplace=False)
        fd = jnp.vdot(seed, align(plus, coeff) - align(minus, coeff)) / (2*h)
        np.testing.assert_allclose(derivative, fd, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize('gap', [2e-7, 2e-9, 2e-11])
def test_nearly_degenerate_canonical_eigenvectors_do_not_amplify_alignment(gap):
    from pyscfad.scipy.linalg import eigh
    rng = np.random.default_rng(93)
    q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    fock = q @ np.diag([-1., -1.+gap, -.4, .5, 1.2]) @ q.T
    reference = q[:, :3] @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    direction = rng.normal(size=(5, 5))
    direction = .1 * (direction + direction.T)
    seed = jnp.asarray(rng.normal(size=(5, 3)))

    def objective(displacement):
        _, c = eigh(jnp.asarray(fock) + displacement * direction)
        occupied = c[:, :3]
        frame = occupied @ domain._polar_rotation(occupied.T @ reference)
        return jnp.vdot(seed, frame)

    derivative = jax.grad(objective)(jnp.asarray(0.))
    fd = (objective(1e-4) - objective(-1e-4)) / 2e-4
    assert abs(derivative) < 1.
    np.testing.assert_allclose(derivative, fd, atol=3e-7, rtol=1e-6)


def test_stationary_saddle_replay_keeps_labels_and_implicit_response(monkeypatch):
    """A second escape would rotate a saved saddle by 45 degrees and fail matching."""
    from contextlib import contextmanager
    from pyscfad.lo import boys
    mol = gto.Mole(atom='O 0 0 0; H .1 -.75 .57; H 0 .8 .61',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    rng = np.random.default_rng(155)
    coeff = np.linalg.inv(np.linalg.cholesky(overlap).T) @ np.linalg.qr(rng.normal(size=(7, 3)))[0]
    dipoles = np.zeros((3, 3, 3))
    dipoles[0] = [[0., 1., 0.], [1., 0., 0.], [0., 0., 3.]]
    dual = overlap @ coeff
    ao_dipoles = np.einsum('pi,aij,qj->apq', dual, dipoles, dual)

    def projected(mol, orbitals):
        return jnp.einsum('pi,apq,qj->aij', orbitals, ao_dipoles, orbitals)

    @contextmanager
    def cache(loc, mo_coeff=None):
        orbitals = loc.mo_coeff if mo_coeff is None else mo_coeff
        loc._boys_dipole_cache = (np.asarray(projected(loc.mol, orbitals)), 0.)
        try:
            yield
        finally:
            del loc._boys_dipole_cache

    monkeypatch.setattr(boys.Boys, '_cache_dipoles', cache)
    monkeypatch.setattr(boys, 'dipole_integral', projected)
    monkeypatch.delenv('BOYS_RESPONSE_VARIANT', raising=False)
    static = SimpleNamespace(target_reference_coeff=coeff,
                             target_reference_coords=np.asarray(mol.atom_coords()),
                             lo_kwargs=dict(conv_tol=1e-12, conv_tol_grad=1e-9))
    run = lambda orbitals: domain._rebuild_boys_targets(SimpleNamespace(mol=mol), orbitals, static)
    with pytest.warns(RuntimeWarning, match='disabled for branch replay'):
        actual, pullback = jax.vjp(run, jnp.asarray(coeff))
    np.testing.assert_allclose(actual, coeff, atol=1e-12)
    orbital_bar, = pullback(jnp.asarray(rng.normal(size=coeff.shape)))
    assert np.all(np.isfinite(orbital_bar))
    gauge_bar = coeff.T @ np.asarray(orbital_bar)
    np.testing.assert_allclose(gauge_bar, gauge_bar.T, atol=1e-11)
