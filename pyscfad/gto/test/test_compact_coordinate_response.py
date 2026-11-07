"""Coordinate response contracts AO ownership without atom-expanded tapes."""
import importlib

import jax
import jax.numpy as jnp
import numpy
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.gto import _moleintor_jvp as integrals


@pytest.fixture(params=[False, True])
def molecule(request):
    with config_update('pyscfad_moleintor_opt', False):
        mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                       basis='cc-pvdz', cart=request.param, verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mol.set_common_origin((.3, -.2, .4))
        yield mol


@pytest.mark.parametrize('intor,hermi,sliced', [
    ('int1e_r', 0, False), ('int1e_r', 1, False),
    ('int1e_ovlp', 1, False), ('int2c2e', 1, False),
    ('int1e_kin', 0, True), ('int2c2e', 0, True),
    ('int1e_nuc', 1, False), ('int1e_nuc', 0, False),
    ('int1e_nuc', 0, True),
])
def test_coordinate_jvp_vjp_and_finite_difference(molecule, intor, hermi, sliced):
    mol = molecule
    # Unequal, nonzero shell offsets exercise bra and ket ownership separately.
    shls = (1, mol.nbas-1, 2, mol.nbas) if sliced else None
    options = dict(hermi=hermi, shls_slice=shls)

    def evaluate(coords):
        moved = mol.copy()
        moved.coords = coords
        return moved.intor(intor, **options)

    rng = numpy.random.default_rng(811)
    direction = jnp.asarray(rng.normal(size=mol.coords.shape))
    value, tangent = jax.jvp(evaluate, (mol.coords,), (direction,))
    seed = jnp.asarray(rng.normal(size=value.shape))
    _, pullback = jax.vjp(evaluate, mol.coords)
    gradient = pullback(seed)[0]
    numpy.testing.assert_allclose(jnp.vdot(gradient, direction),
                                  jnp.vdot(seed, tangent), atol=2e-10, rtol=1e-12)

    step = 1e-5
    plus = mol.set_geom_(numpy.asarray(mol.coords + step*direction),
                         unit='Bohr', inplace=False)
    minus = mol.set_geom_(numpy.asarray(mol.coords - step*direction),
                          unit='Bohr', inplace=False)
    difference = (plus.intor(intor, **options) - minus.intor(intor, **options)) / (2*step)
    numpy.testing.assert_allclose(tangent, difference, atol=2e-7, rtol=2e-7)

    # One component/Cartesian derivative tensor is enough regardless of natm.
    # The old broadcast saved an extra leading atom axis in the reverse tape.
    components = 9 if intor == 'int1e_r' else 3
    arrays = [x for x in jax.tree.leaves(pullback) if hasattr(x, 'shape')]
    assert max(x.size for x in arrays) <= components * mol.nao**2
    assert all(x.ndim <= 4 for x in arrays)


@pytest.mark.parametrize('hermi,sliced', [(1, False), (0, False), (0, True)])
def test_nuclear_centers_stream_only_during_reverse(molecule, monkeypatch, hermi, sliced):
    from pyscfad.gto import _nuclear_center_response as centers
    mol = molecule
    shls = (1, mol.nbas-1, 2, mol.nbas) if sliced else None
    evaluate = lambda m: m.intor('int1e_nuc', hermi=hermi, shls_slice=shls)
    calls = []
    original = centers._contract

    def counted(value, **options):
        calls.append(options['transpose'])
        return original(value, **options)

    monkeypatch.setattr(centers._center_response_p, 'impl', counted)
    value, pullback = jax.vjp(evaluate, mol)
    assert calls == []  # No nuclear-center derivative matrices saved forward.
    seed = jnp.asarray(numpy.random.default_rng(815).normal(size=value.shape))
    actual, = pullback(seed)
    assert calls == [True]
    with monkeypatch.context() as legacy:
        legacy.setattr(centers, 'nuclear_center_tangent', lambda *args: None)
        _, reference = jax.vjp(evaluate, mol)
        expected, = reference(seed)
    numpy.testing.assert_allclose(actual.coords, expected.coords, atol=2e-11, rtol=1e-12)


def test_nuclear_tape_has_no_per_nucleus_ao_derivatives(monkeypatch):
    from pyscfad.gto import _nuclear_center_response as centers
    with config_update('pyscfad_moleintor_opt', False):
        for natm in (2, 6):
            mol = gto.Mole(atom=[('H', (0., 0., float(i))) for i in range(natm)],
                           basis='sto-3g', verbose=0)
            mol.build(trace_exp=False, trace_ctr_coeff=False)
            _, pullback = jax.vjp(lambda m: m.intor('int1e_nuc', hermi=1), mol)
            arrays = [x for x in jax.tree.leaves(pullback) if hasattr(x, 'shape')]
            # Only the basis-center derivative remains; nuclear-center
            # derivatives are represented by the small fixed molecular data.
            assert sum(x.shape == (3, 1, mol.nao, mol.nao) for x in arrays) == 1
            with monkeypatch.context() as legacy:
                legacy.setattr(centers, 'nuclear_center_tangent', lambda *args: None)
                _, old = jax.vjp(lambda m: m.intor('int1e_nuc', hermi=1), mol)
                old_arrays = [x for x in jax.tree.leaves(old) if hasattr(x, 'shape')]
                assert sum(x.shape == (3, 1, mol.nao, mol.nao) for x in old_arrays) == natm + 1


def test_nuclear_batched_and_compiled_pullback():
    with config_update('pyscfad_moleintor_opt', False):
        mol = gto.Mole(atom='H 0 0 0; H .1 .2 .8', basis='sto-3g', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        evaluate = lambda m: m.intor('int1e_nuc', hermi=1)
        forward = jax.jacfwd(evaluate)(mol).coords
        reverse = jax.jacrev(evaluate)(mol).coords
        numpy.testing.assert_allclose(forward, reverse, atol=2e-12, rtol=1e-12)
        value, pullback = jax.vjp(evaluate, mol)
        seed = jnp.asarray(numpy.random.default_rng(816).normal(size=value.shape))
        compiled, = jax.jit(pullback)(seed)
        eager, = pullback(seed)
        numpy.testing.assert_allclose(compiled.coords, eager.coords, atol=2e-12, rtol=1e-12)


def test_auxiliary_metric_pullback_has_no_atom_expanded_residual():
    with config_update('pyscfad_moleintor_opt', False):
        mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                       basis='weigend', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)

        def factor(mol):
            return jax.scipy.linalg.cholesky(mol.intor('int2c2e', hermi=1), lower=True)

        value, pullback = jax.vjp(factor, mol)
        arrays = [x for x in jax.tree.leaves(pullback) if hasattr(x, 'shape')]
        assert all(x.ndim <= 4 for x in arrays)
        assert max(x.size for x in arrays) <= 3 * mol.nao**2
        seed = jnp.asarray(numpy.random.default_rng(812).normal(size=value.shape))
        bar = pullback(seed)[0].coords
        assert numpy.isfinite(bar).all()
        # Translation cannot change the inter-center Coulomb metric.
        numpy.testing.assert_allclose(bar.sum(axis=0), 0., atol=1e-10)


def test_complete_boys_response_matches_dense_dipole_reference(monkeypatch):
    boys = importlib.import_module('pyscfad.lo.boys')

    def dense_derivative(mol, mol_t, intor):
        derivative = -integrals.intor2c(mol, intor).reshape(3, 3, mol.nao, mol.nao)
        expanded = numpy.zeros((mol.natm, *derivative.shape))
        for atom, (_, _, begin, end) in enumerate(mol.aoslice_by_atom()):
            expanded[atom, ..., begin:end] = derivative[..., begin:end]
        tangent = jnp.einsum('npxij,nx->pij', expanded, mol_t.coords)
        return tangent + tangent.transpose(0, 2, 1)

    with config_update('pyscfad_moleintor_opt', False):
        mol = gto.Mole(atom='O 0 0 0; H .12 -.74 .57; H -.05 .83 .62',
                       basis='sto-3g', verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        coeff = jnp.asarray(mf.mo_coeff[:, 1:5])
        seed = jnp.asarray(numpy.random.default_rng(813).normal(size=coeff.shape))

        def localize(m, c):
            return boys.boys(m, c, init_guess='atomic', conv_tol=1e-12,
                             conv_tol_grad=1e-9,
                             gmres_options=dict(tol=1e-10, atol=1e-12,
                                                restart=20, maxiter=100))

        actual, pullback = jax.vjp(localize, mol, coeff)
        bars = pullback(seed)
        with monkeypatch.context() as legacy:
            legacy.setattr(integrals, '_int1e_r_jvp_r0', dense_derivative)
            expected, reference = jax.vjp(localize, mol, coeff)
            expected_bars = reference(seed)
        numpy.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
        for got, want in zip(jax.tree.leaves(bars), jax.tree.leaves(expected_bars)):
            numpy.testing.assert_allclose(got, want, atol=1e-9, rtol=1e-10)
