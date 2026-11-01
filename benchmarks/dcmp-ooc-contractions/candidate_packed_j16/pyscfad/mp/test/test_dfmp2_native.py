"""Native canonical MP2 must preserve the scan's energy and AD contract."""
import jax
import numpy
import pytest

from pyscfad import config_update, gto, scf
from pyscfad import numpy as np
from pyscfad.mp import dfmp2


def _inputs(nocc=3, nvir=4):
    rng = numpy.random.default_rng(87)
    lov = np.asarray(rng.normal(size=(9, nocc*nvir)))
    energies = np.asarray(numpy.r_[numpy.linspace(-1.5, -0.5, nocc),
                                  numpy.linspace(0.2, 1.1, nvir)])
    return lov, energies


def _assert_tree_close(actual, expected, atol=2e-10):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for got, want in zip(jax.tree_util.tree_leaves(actual),
                         jax.tree_util.tree_leaves(expected)):
        numpy.testing.assert_allclose(got, want, atol=atol, rtol=atol)


def _require_native():
    if dfmp2.libmp is None or dfmp2.libmp_vjp is None:
        pytest.skip('native MP2 libraries are not installed')


@pytest.mark.parametrize('nocc', [1, 2, 3, 4])
def test_native_energy_and_both_input_gradients_match_scan(nocc):
    # Odd/even occupied counts exercise the native triangular pair schedule.
    lov, energies = _inputs(nocc)
    assert hasattr(dfmp2, '_contract_native'), 'native MP2 contraction is missing'
    _require_native()
    reference = lambda x, e: dfmp2._contract_scan_impl(x, e, nocc, 4, False)[0]
    native = lambda x, e: dfmp2._contract_native(x, e, nocc, 4)
    expected = jax.value_and_grad(reference, argnums=(0, 1))(lov, energies)
    actual = jax.value_and_grad(native, argnums=(0, 1))(lov, energies)
    _assert_tree_close(actual, expected)
    # Non-unit cotangent also catches accidentally omitted chain-rule scaling.
    _, pullback = jax.vjp(native, lov, energies)
    _assert_tree_close(pullback(np.asarray(-0.37)),
                       jax.tree.map(lambda x: -0.37*x, expected[1]))


def test_optimized_eager_energy_and_gradient_avoid_the_scan(monkeypatch):
    _require_native()
    lov, energies = _inputs(2, 3)
    fn = lambda x, e: dfmp2._contract_energy(x, e, 2, 3, False)[0]
    ref = lambda x, e: dfmp2._contract_scan_impl(x, e, 2, 3, False)[0]
    expected = jax.value_and_grad(ref, argnums=(0, 1))(lov, energies)

    def forbidden_scan(*args, **kwargs):
        raise AssertionError('optimized eager MP2 fell back to the scan')

    monkeypatch.setattr(dfmp2, '_contract_scan_impl', forbidden_scan)
    with config_update('pyscfad_moleintor_opt', True):
        _assert_tree_close(fn(lov, energies), expected[0])
        _assert_tree_close(jax.value_and_grad(fn, argnums=(0, 1))(lov, energies),
                           expected)


def test_native_input_derivatives_match_directional_finite_difference():
    _require_native()
    lov, energies = _inputs()
    rng = numpy.random.default_rng(91)
    direction = (np.asarray(rng.normal(size=lov.shape)) * .1,
                 np.asarray(rng.normal(size=energies.shape)) * .1)
    fn = lambda x, e: dfmp2._contract_native(x, e, 3, 4)
    _, derivative = jax.jvp(fn, (lov, energies), direction)
    step = 1e-4
    plus = fn(lov+step*direction[0], energies+step*direction[1])
    minus = fn(lov-step*direction[0], energies-step*direction[1])
    assert float(derivative) == pytest.approx(float((plus-minus)/(2*step)),
                                             rel=1e-8, abs=1e-7)


@pytest.mark.parametrize('transform', [
    'jvp', 'linearize', 'forward_hessian', 'reverse_hessian',
    'jit_gradient', 'vmap_gradient',
])
def test_native_preserves_forward_staged_and_higher_derivatives(transform):
    _require_native()
    lov, energies = _inputs(2, 3)
    reference = lambda e: dfmp2._contract_scan_impl(lov, e, 2, 3, False)[0]
    native = lambda e: dfmp2._contract_native(lov, e, 2, 3)
    direction = np.asarray([0.2, -0.1, 0.3, -0.4, 0.6])

    def evaluate(fn):
        if transform == 'jvp':
            return jax.jvp(fn, (energies,), (direction,))
        if transform == 'linearize':
            value, linear = jax.linearize(fn, energies)
            return value, linear(direction), linear(-direction)
        if transform == 'forward_hessian':
            return jax.hessian(fn)(energies)
        if transform == 'reverse_hessian':
            return jax.jacrev(jax.grad(fn))(energies)
        if transform == 'jit_gradient':
            return jax.jit(jax.value_and_grad(fn))(energies)
        return jax.vmap(jax.value_and_grad(fn))(np.stack([energies, energies*1.1]))

    _assert_tree_close(evaluate(native), evaluate(reference))


@pytest.mark.parametrize('with_t2', [False, True])
@pytest.mark.parametrize('optimized', [False, True])
def test_dispatch_retains_t2_and_opt_off_derivatives(with_t2, optimized):
    lov, energies = _inputs(2, 3)
    def objective(contract, x, e):
        energy, t2 = contract(x, e, 2, 3, with_t2)
        return energy if t2 is None else energy + 0.03*np.sum(t2*t2)
    reference = lambda x, e: objective(dfmp2._contract_scan_impl, x, e)
    actual = lambda x, e: objective(dfmp2._contract_energy, x, e)
    with config_update('pyscfad_moleintor_opt', optimized):
        _assert_tree_close(jax.value_and_grad(actual, argnums=(0, 1))(lov, energies),
                           jax.value_and_grad(reference, argnums=(0, 1))(lov, energies))


@pytest.mark.parametrize('fallback', ['missing_library', 'missing_forward_library', 'float32'])
def test_dispatch_fallback_preserves_values_and_gradients(monkeypatch, fallback):
    lov, energies = _inputs(2, 3)
    if fallback == 'missing_library':
        monkeypatch.setattr(dfmp2, 'libmp_vjp', None)
    elif fallback == 'missing_forward_library':
        monkeypatch.setattr(dfmp2, 'libmp', None)
    else:
        lov, energies = lov.astype(np.float32), energies.astype(np.float32)
    reference = lambda x, e: dfmp2._contract_scan_impl(x, e, 2, 3, False)[0]
    actual = lambda x, e: dfmp2._contract_energy(x, e, 2, 3, False)[0]
    with config_update('pyscfad_moleintor_opt', True):
        _assert_tree_close(jax.value_and_grad(actual, argnums=(0, 1))(lov, energies),
                           jax.value_and_grad(reference, argnums=(0, 1))(lov, energies),
                           atol=2e-5 if fallback == 'float32' else 2e-10)


def test_zero_auxiliary_rows_have_zero_energy_and_gradients_without_blas_errors(capfd):
    _require_native()
    lov = np.zeros((0, 6), dtype=np.float64)
    energies = np.asarray([-1.2, -0.6, 0.2, 0.5, 0.9])
    fn = lambda x, e: dfmp2._contract_energy(x, e, 2, 3, False)[0]
    with config_update('pyscfad_moleintor_opt', True):
        energy, (lov_bar, energy_bar) = jax.value_and_grad(
            fn, argnums=(0, 1))(lov, energies)
    jax.block_until_ready((energy, lov_bar, energy_bar))
    captured = capfd.readouterr()
    assert 'ERROR' not in captured.out + captured.err
    assert float(energy) == 0.0
    assert lov_bar.shape == (0, 6)
    numpy.testing.assert_array_equal(energy_bar, numpy.zeros(5))


@pytest.mark.parametrize('frozen', [None, 1, [0, 6]])
def test_native_full_molecular_gradient_preserves_frozen_selection(monkeypatch, frozen):
    _require_native()
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=8000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    def energy(mol_):
        mf = scf.RHF(mol_).density_fit(auxbasis='weigend')
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf.e_tot + dfmp2.MP2(mf, frozen=frozen).kernel(with_t2=False)[0]
    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True)):
        actual = jax.value_and_grad(energy)(mol)
        monkeypatch.setattr(dfmp2, 'libmp_vjp', None)
        expected = jax.value_and_grad(energy)(mol)
    numpy.testing.assert_allclose(actual[0], expected[0], atol=1e-10, rtol=1e-10)
    numpy.testing.assert_allclose(actual[1].coords, expected[1].coords,
                                  atol=2e-8, rtol=2e-8)
