"""Native CCSD must preserve the generalized adjoint and resident storage."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag
from pyscf import lib

from pyscfad import config_update, gto, scf
from pyscfad.cc import dfccsd, ccsd_lambda
from pyscfad.cc import _dfccsd_native as native
from pyscfad.lno import ccsd as local_cc


@pytest.fixture(params=[False, True], ids=['canonical', 'noncanonical'])
def case(request):
    mol = gto.Mole(atom='H 0 0 0; H 0 0 1.4; H 0 0 3.4; H 0 0 4.8',
                   basis='sto-3g', unit='Bohr', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-13
    mf.kernel()
    if request.param:
        q = np.array([[.8, -.6], [.6, .8]])
        mf.mo_coeff = mf.mo_coeff @ block_diag(q, q.T)
    cc = dfccsd.RCCSD(mf)
    cc.incore_complete = True
    cc.diis_space = 3
    cc.max_cycle = 150
    cc.conv_tol = 1e-13
    cc.conv_tol_normt = 1e-11
    eris = cc.ao2mo()
    if request.param:
        # Exercise the occupied-virtual Fock block present in projected LISs.
        eris.fock = eris.fock.at[0, 2].add(.003)
        eris.fock = eris.fock.at[2, 0].add(.003)
    cc.kernel(eris=eris)
    assert cc.converged
    return cc, eris


def test_native_forward_response_and_integral_adjoint(case, monkeypatch):
    cc, eris = case
    t1, t2 = cc.t1, cc.t2
    options = native.settings(cc)
    host, he = native.host_objects(options, [getattr(eris, n) for n in native.ERI_NAMES], cc.nocc)
    rng = np.random.default_rng(591)
    bars = (.7, rng.normal(size=t1.shape)*.01, rng.normal(size=t2.shape)*.01)
    amp = cc.amplitudes_to_vector(t1, t2)

    def update(x):
        return cc.amplitudes_to_vector(*cc.update_amps(*cc.vector_to_amplitudes(x), eris))
    _, pullback = jax.vjp(update, amp)
    seed = rng.normal(size=amp.shape)
    expected_action = np.asarray(pullback(jnp.asarray(seed))[0])
    converged, expected = ccsd_lambda.solve_response_lambda(
        cc, eris, t1, t2, *bars, max_cycle=150, tol=1e-11)
    assert converged
    expected_bars = dfccsd.lagrangian_grad(cc, eris, t1, t2, bars[0], expected)

    def forbidden(*args, **kwargs):
        raise AssertionError('Native CCSD must not use disk scratch or JAX differentiation')
    monkeypatch.setattr(lib, 'H5TmpFile', forbidden)
    monkeypatch.setattr(jax, 'vjp', forbidden)
    monkeypatch.setattr(jax, 'grad', forbidden)
    conv, energy, a, b = native.forward(cc, eris, *cc.init_amps(eris)[1:])
    assert conv
    np.testing.assert_allclose(energy, cc.e_corr, atol=1e-11, rtol=0)
    np.testing.assert_allclose(a, t1, atol=1e-9, rtol=0)
    np.testing.assert_allclose(b, t2, atol=1e-9, rtol=0)
    action = native.response_action(host, he, np.asarray(t1), np.asarray(t2))
    np.testing.assert_allclose(action(seed), expected_action, atol=1e-9, rtol=0)
    low_memory = copy.copy(host)
    low_memory.max_memory = 0
    tiled_action = native.response_action(low_memory, he, np.asarray(t1), np.asarray(t2))
    np.testing.assert_allclose(tiled_action(seed), action(seed), atol=1e-12, rtol=0)
    conv, actual = native.solve_response(host, he, np.asarray(t1), np.asarray(t2), *bars)
    assert conv
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0)
    actual_bars = native.integral_cotangent(host, he, np.asarray(t1), np.asarray(t2), bars[0], actual)
    for name, value in zip(native.BAR_NAMES, actual_bars):
        np.testing.assert_allclose(value, getattr(expected_bars, name), atol=2e-9, rtol=0, err_msg=name)


def test_fragment_ccsdt_energy_and_geometry_gradient(get_opt_mol, record_property):
    """A projected fragment energy and triples supply a nonstandard source."""
    def energy(mol):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        s = mf.get_ovlp()
        f = mf.get_fock()
        lo = mf.mo_coeff[:, 1:3]
        _, ecc, et = local_cc._impurity_solve_core(
            mf, mf.mo_coeff, lo, f, s, frozen=[0], ccsd_t=True)
        return mf.e_tot + ecc + et
    results = []
    for enabled in (False, True):
        with config_update('pyscfad_dfccsd_native', enabled):
            value, gradient = jax.value_and_grad(energy)(get_opt_mol)
            results.append((float(value), np.asarray(gradient.coords)))
    np.testing.assert_allclose(results[1][0], results[0][0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(results[1][1], results[0][1], atol=2e-7, rtol=0)
    record_property('energy_error_eh', abs(results[1][0]-results[0][0]))
    record_property('gradient_error_eh_per_bohr', float(np.max(np.abs(results[1][1]-results[0][1]))))


def test_native_callback_under_jit(case):
    cc, eris = case
    cc = copy.copy(cc)
    cc.native_cpu = True
    t1, t2 = cc.t1, cc.t2
    def energy(fock):
        e = copy.copy(eris)
        e.fock = fock
        return dfccsd._dfccsd_kernel_custom(cc, e, t1, t2)[1]
    eager = jax.value_and_grad(energy)(eris.fock)
    compiled = jax.jit(jax.value_and_grad(energy))(eris.fock)
    np.testing.assert_allclose(compiled[0], eager[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(compiled[1], eager[1], atol=1e-10, rtol=0)


def test_native_uses_total_memory_not_df_workspace(case):
    cc, eris = case
    cc.max_memory = 900000
    cc.with_df.max_memory = 500000
    options = native.settings(cc)
    host, _ = native.host_objects(options, [getattr(eris, n) for n in native.ERI_NAMES], cc.nocc)
    assert options['max_memory'] == host.max_memory == host.mol.max_memory == 900000


def test_native_virtual_cache_policy(case, monkeypatch):
    """Default caching, explicit direct mode, budget boundary, and both ladders."""
    from pyscf.cc import dfccsd as pyscf_dfccsd
    cc, eris = case
    assert cc.vvvv_direct is False
    arrays = [getattr(eris, n) for n in native.ERI_NAMES]
    options = native.settings(cc)
    assert options['vvvv_direct'] is False
    npair = eris.Lvv.shape[-1]
    cache_mb = 8*npair**2/1e6
    monkeypatch.setattr(lib, 'current_memory', lambda: (1000., 0.))

    # The threshold is 60% of remaining total memory, using the native
    # layout's actual size. The DF workspace budget does not enter it.
    cached, ce = native.host_objects(dict(options, max_memory=1000+cache_mb/.6*1.1), arrays, cc.nocc)
    limited, le = native.host_objects(dict(options, max_memory=1000+cache_mb/.6*.9), arrays, cc.nocc)
    direct, de = native.host_objects(dict(options, max_memory=900000, vvvv_direct=True), arrays, cc.nocc)
    assert ce.vvvv.shape == (npair, npair)
    assert le.vvvv is de.vvvv is None
    for host_eris in (ce, le, de):
        assert np.shares_memory(host_eris.ovvv, np.asarray(eris.ovvv))
    tensor = np.asarray(cc.t2)
    expected = de._contract_vvvv_t2(direct, tensor)
    np.testing.assert_allclose(le._contract_vvvv_t2(limited, tensor), expected, atol=1e-13, rtol=0)

    def forbidden(*args, **kwargs):
        raise AssertionError('Cached iterations must not reconstruct DF virtual integrals')
    monkeypatch.setattr(pyscf_dfccsd, '_contract_vvvv_t2', forbidden)
    stored = ce.vvvv
    for _ in range(2):
        np.testing.assert_allclose(ce._contract_vvvv_t2(cached, tensor), expected, atol=1e-13, rtol=0)
        assert ce.vvvv is stored
