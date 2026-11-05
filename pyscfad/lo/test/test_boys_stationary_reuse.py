"""A converged Boys frame keeps its implicit response without optimizing again."""
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag, eigh, expm, null_space

from pyscfad import gto
from pyscfad.dlno_stc.domain import _polar_rotation
from pyscfad.gto.mole import intor_cross
from pyscfad.lo import boys


OPTIONS = dict(conv_tol=1e-12, conv_tol_grad=1e-9,
               gmres_options=dict(tol=1e-10, atol=1e-12, maxiter=100))


@pytest.fixture(scope='module')
def case():
    # A real molecular occupied subspace, without an SCF calculation.
    mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                   basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    hcore = np.asarray(mol.intor('int1e_kin') + mol.intor('int1e_nuc'))
    _, coeff = eigh(hcore, overlap)
    reference = boys.boys(mol, jnp.asarray(coeff[:, 1:5]),
                          conv_tol=1e-13, conv_tol_grad=1e-11)
    return mol, reference


def _helper():
    assert hasattr(boys, '_boys_at_stationary_frame'), 'stationary Boys attachment is missing'
    return boys._boys_at_stationary_frame


def _assert_trees_close(actual, expected, *, atol=2e-9, rtol=2e-8):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for a, b in zip(jax.tree_util.tree_leaves(actual),
                    jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize('variant', ['baseline', 'projected', 'projected_precond'])
def test_stationary_frame_keeps_full_response_without_optimizer(case, monkeypatch, variant):
    attach = _helper()
    mol, frame = case
    monkeypatch.setenv('BOYS_RESPONSE_VARIANT', variant)
    expected, replay = jax.vjp(lambda m, c: boys.boys(
        m, c, init_guess=np.eye(4), max_saddle_escapes=0, **OPTIONS), mol, frame)
    seed = jnp.asarray(np.random.default_rng(730).normal(size=frame.shape))
    expected_bars = replay(seed)

    def forbidden(*args, **kwargs):
        pytest.fail('stationary response must not invoke an optimizer or curvature search')

    for name in ('_boys', 'kernel', '_lowest_curvature'):
        monkeypatch.setattr(boys, name, forbidden)
    actual, pullback = jax.vjp(lambda m, c: attach(m, c, **OPTIONS), mol, frame)
    np.testing.assert_array_equal(actual, frame)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
    _assert_trees_close(pullback(seed), expected_bars)
    assert np.linalg.norm(np.asarray(expected_bars[0].coords)) > 1e-3
    assert np.linalg.norm(np.asarray(expected_bars[1]) - seed) > 1e-3


@pytest.mark.parametrize('nonfinite', [False, True])
def test_unsuitable_frame_has_specific_stationarity_failure(case, nonfinite):
    attach = _helper()
    mol, frame = case
    perturbation = np.zeros((4, 4))
    perturbation[0, 1], perturbation[1, 0] = 1e-3, -1e-3
    displaced = frame @ expm(perturbation)
    if nonfinite:
        displaced = displaced.at[0, 0].set(jnp.nan)
    with pytest.raises(boys.BoysStationarityError, match='stationar'):
        attach(mol, displaced, **OPTIONS)


@pytest.mark.parametrize('explicit_gradient', [True, False])
def test_requested_stationarity_tolerance_is_respected(case, explicit_gradient):
    attach = _helper()
    mol, frame = case
    rotation = np.eye(4)
    angle = 1e-5
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]
    displaced = frame @ rotation
    gradient = np.linalg.norm(boys.Boys(mol, np.asarray(displaced)).get_grad(np.eye(4)))
    assert gradient > 1e-7
    tight, loose = .5 * gradient, 2 * gradient
    option = (lambda tolerance: dict(conv_tol_grad=tolerance) if explicit_gradient
              else dict(conv_tol=tolerance ** 2 / .1))
    with pytest.raises(boys.BoysStationarityError):
        attach(mol, displaced, **option(tight))
    np.testing.assert_array_equal(attach(mol, displaced, **option(loose)), displaced)


def test_nan_tolerance_cannot_certify_stationarity(case):
    attach = _helper()
    mol, frame = case
    with pytest.raises(boys.BoysStationarityError):
        attach(mol, frame, conv_tol_grad=np.nan)


def test_one_orbital_is_identity_without_integrals(case, monkeypatch):
    attach = _helper()
    mol, frame = case
    frame = frame[:, :1]

    def forbidden(*args, **kwargs):
        pytest.fail('one orbital has no localization equation')

    monkeypatch.setattr(boys, 'dipole_integral', forbidden)
    actual, reverse = jax.vjp(attach, mol, frame)
    seed = jnp.asarray(np.random.default_rng(731).normal(size=frame.shape))
    mol_bar, coeff_bar = reverse(seed)
    np.testing.assert_array_equal(actual, frame)
    np.testing.assert_array_equal(coeff_bar, seed)
    np.testing.assert_array_equal(mol_bar.coords, 0.)


def _synthetic_dipoles(monkeypatch, mol, frame, *, saddle=False):
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    dipoles = np.zeros((3, 3, 3))
    dipoles[0, 2, 2] = 3.
    if saddle:
        dipoles[0, 0, 1] = dipoles[0, 1, 0] = 1.
    dual = overlap @ np.asarray(frame)
    ao_dipoles = np.einsum('pi,aij,qj->apq', dual, dipoles, dual)

    def projected(m, c):
        return jnp.einsum('pi,apq,qj->aij', c, ao_dipoles, c)

    @contextmanager
    def cache(loc, mo_coeff=None):
        c = loc.mo_coeff if mo_coeff is None else mo_coeff
        loc._boys_dipole_cache = (np.asarray(projected(loc.mol, c)), 0.)
        try:
            yield
        finally:
            del loc._boys_dipole_cache

    monkeypatch.setattr(boys, 'dipole_integral', projected)
    monkeypatch.setattr(boys.Boys, '_cache_dipoles', cache)
    monkeypatch.setenv('BOYS_RESPONSE_VARIANT', 'projected_precond')


def test_retained_stationary_saddle_has_same_signed_response(case, monkeypatch):
    attach = _helper()
    mol, reference = case
    frame = reference[:, :3]
    _synthetic_dipoles(monkeypatch, mol, frame, saddle=True)
    with pytest.warns(RuntimeWarning, match='disabled for branch replay'):
        expected, replay = jax.vjp(lambda c: boys.boys(
            mol, c, init_guess=np.eye(3), max_saddle_escapes=0, **OPTIONS), frame)
    actual, reverse = jax.vjp(lambda c: attach(mol, c, **OPTIONS), frame)
    seed = jnp.asarray(np.random.default_rng(732).normal(size=frame.shape))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    _assert_trees_close(reverse(seed), replay(seed))


@pytest.mark.parametrize('symmetry,compatible', [(False, True), (True, False), (True, True)])
def test_stationary_attachment_preserves_null_mode_contract(case, monkeypatch,
                                                           symmetry, compatible):
    attach = _helper()
    mol, reference = case
    frame = reference[:, :3]
    _synthetic_dipoles(monkeypatch, mol, frame)
    _, reverse = jax.vjp(lambda c: attach(mol, c, symmetry=symmetry, **OPTIONS), frame)
    # The 0/1 dipoles coincide: that rotation is a genuine flat direction.
    seed = np.zeros(frame.shape)
    seed[:, 2 if compatible else 1] = np.asarray(frame[:, 0])
    if not symmetry:
        with pytest.raises(RuntimeError, match='declare symmetry=True'):
            reverse(jnp.asarray(seed))
    elif not compatible:
        with pytest.raises(RuntimeError, match='incompatible null-space component'):
            reverse(jnp.asarray(seed))
    else:
        with pytest.warns(RuntimeWarning, match='compatible range solution'):
            bars, = reverse(jnp.asarray(seed))
        assert np.isfinite(bars).all()


def _aligned(mol, coeff, reference_mol, reference):
    cross = intor_cross('int1e_ovlp', mol, reference_mol)
    return coeff @ _polar_rotation(coeff.T @ cross @ reference)


def test_aligned_attachment_handles_near_pi_and_full_ambient_response(case):
    attach = _helper()
    mol, reference = case
    angle = np.pi - 1e-4
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    coeff = reference @ block_diag(rotation, rotation)
    fixed_mol = jax.tree_util.tree_map(jax.lax.stop_gradient, mol)

    def run(m, c, *, reuse):
        aligned = _aligned(m, c, fixed_mol, reference)
        return (attach(m, aligned, **OPTIONS) if reuse else boys.boys(
            m, aligned, init_guess=np.eye(4), max_saddle_escapes=0, **OPTIONS))

    actual, reverse = jax.vjp(lambda m, c: run(m, c, reuse=True), mol, coeff)
    expected, replay = jax.vjp(lambda m, c: run(m, c, reuse=False), mol, coeff)
    seed = jnp.asarray(np.random.default_rng(733).normal(size=coeff.shape))
    bars = reverse(seed)
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
    _assert_trees_close(bars, replay(seed))
    gauge_bar = np.asarray(coeff.T @ bars[1])
    np.testing.assert_allclose(gauge_bar, gauge_bar.T, atol=2e-11, rtol=0)

    rng = np.random.default_rng(734)
    dc = rng.normal(size=coeff.shape) * .03
    dm = rng.normal(size=mol.coords.shape) * .03

    def physical(m, c, *, reuse):
        # The existing optimizer centers its dipoles while its implicit rule
        # uses the zero origin. These are equivalent on the orthonormal
        # manifold. Full ambient bars are compared above; the independent FD
        # stays on that manifold as both coordinates and the subspace move.
        metric = c.T @ m.intor('int1e_ovlp') @ c
        normalized = jnp.linalg.solve(jnp.linalg.cholesky(metric), c.T).T
        return run(m, normalized, reuse=reuse)

    _, physical_reverse = jax.vjp(lambda m, c: physical(m, c, reuse=True), mol, coeff)
    physical_bars = physical_reverse(seed)
    derivative = np.vdot(physical_bars[1], dc) + np.vdot(physical_bars[0].coords, dm)
    for step in (3e-5, 1e-5):
        plus = mol.set_geom_(np.asarray(mol.coords) + step * dm, unit='Bohr', inplace=False)
        minus = mol.set_geom_(np.asarray(mol.coords) - step * dm, unit='Bohr', inplace=False)
        finite = jnp.vdot(seed, physical(plus, coeff + step * dc, reuse=False)
                          - physical(minus, coeff - step * dc, reuse=False)) / (2 * step)
        np.testing.assert_allclose(derivative, finite, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('gap', [2e-7, 2e-11])
def test_aligned_attachment_does_not_amplify_nearly_degenerate_gauge(case, gap):
    from pyscfad.scipy.linalg import eigh as ad_eigh

    attach = _helper()
    mol, reference = case
    overlap = np.asarray(mol.intor('int1e_ovlp'))
    root = np.linalg.cholesky(overlap).T
    occupied = root @ np.asarray(reference)
    basis = np.linalg.solve(root, np.column_stack((occupied, null_space(occupied.T))))
    fock = jnp.diag(jnp.array([-1., -1. + gap, -.4, -.2, .5, 1., 1.2]))
    rng = np.random.default_rng(735)
    direction = rng.normal(size=(7, 7)) * .03
    direction += direction.T.copy()
    seed = jnp.asarray(rng.normal(size=reference.shape))

    def objective(t, *, reuse):
        _, vectors = ad_eigh(fock + t * direction)
        coeff = jnp.asarray(basis) @ vectors[:, :4]
        aligned = _aligned(mol, coeff, mol, reference)
        localized = (attach(mol, aligned, **OPTIONS) if reuse else boys.boys(
            mol, aligned, init_guess=np.eye(4), max_saddle_escapes=0, **OPTIONS))
        return jnp.vdot(seed, localized)

    derivative = jax.grad(lambda t: objective(t, reuse=True))(jnp.asarray(0.))
    replay = jax.grad(lambda t: objective(t, reuse=False))(jnp.asarray(0.))
    np.testing.assert_allclose(derivative, replay, atol=2e-6, rtol=2e-6)
    finite = (objective(1e-4, reuse=False) - objective(-1e-4, reuse=False)) / 2e-4
    assert abs(derivative) < 2.
    np.testing.assert_allclose(derivative, finite, atol=2e-6, rtol=2e-5)
