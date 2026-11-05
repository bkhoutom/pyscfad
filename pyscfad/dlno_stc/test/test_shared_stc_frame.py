"""Physical covariance and shared-core routing for a fixed local CC frame."""
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import lib
from pyscfad.dlno_stc import smooth_triples
from pyscfad.dlno_stc.test._triples_dressing_reference import dress, symmetric_exp
from pyscfad.dlno_stc.test.test_triples import (
    triples_fixture, triples_matrix_reference, triples_native,
)
from pyscfad.lno import ccsd as lno_ccsd


GRID = (np.array([.17, .63]), np.array([.31, .72]), {})


def _inputs():
    x = triples_fixture(grouped=True)
    rng = np.random.default_rng(9204)
    no, nv = x['t1'].shape
    x['foo'] = np.diag(x.pop('eocc')) + rng.normal(scale=.03, size=(no, no))
    x['fvv'] = np.diag(x.pop('evir')) + rng.normal(scale=.03, size=(nv, nv))
    x['fov'] = rng.normal(scale=.025, size=(no, nv))
    x['prjlo'] = x.pop('U').T
    return jax.tree.map(jnp.asarray, x)


def _physical(x):
    """Keep same-space perturbations physical on both sides of the comparison."""
    result = dict(x)
    for name in ('Boo', 'Bvv', 'foo', 'fvv'):
        result[name] = (x[name] + x[name].swapaxes(-1, -2)) / 2
    return result


def _rotations():
    rng = np.random.default_rng(740)
    return tuple(jnp.asarray(np.linalg.qr(rng.normal(size=(3, 3)))[0])
                 for _ in range(2))


def _rotate(x, qo, qv):
    x = _physical(x)
    return dict(
        t1=qo.T @ x['t1'] @ qv,
        t2=jnp.einsum('iI,jJ,aA,bB,ijab->IJAB', qo, qo, qv, qv, x['t2']),
        Bvo=jnp.einsum('iI,Pia,aA->PIA', qo, x['Bvo'], qv),
        Boo=jnp.einsum('iI,Pij,jJ->PIJ', qo, x['Boo'], qo),
        Bvv=jnp.einsum('aA,Pab,bB->PAB', qv, x['Bvv'], qv),
        foo=qo.T @ x['foo'] @ qo,
        fvv=qv.T @ x['fvv'] @ qv,
        fov=qo.T @ x['fov'] @ qv,
        prjlo=x['prjlo'] @ qo,
    )


def _eris(x):
    x = _physical(x)
    no, nv = x['t1'].shape
    ovvv = jnp.einsum('Pia,Pbc->iabc', x['Bvo'], x['Bvv'])
    return SimpleNamespace(
        fock=jnp.block([[x['foo'], x['fov']], [x['fov'].T, x['fvv']]]),
        Lov=x['Bvo'], Loo=x['Boo'], Lvv=lib.pack_tril(x['Bvv']),
        ovov=jnp.einsum('Pia,Pjb->iajb', x['Bvo'], x['Bvo']),
        ovoo=jnp.einsum('Pia,Pjk->iajk', x['Bvo'], x['Boo']),
        ovvv=lib.pack_tril(ovvv.reshape(no * nv, nv, nv)).reshape(no, nv, -1),
    )


def _projected_reference(x, t1, t2):
    """Explicit AO-independent contractions, without the production CC helper."""
    weight = x['prjlo'].T @ x['prjlo']
    tau = t2 + jnp.einsum('ia,jb->ijab', t1, t1)
    return (2 * jnp.einsum('pq,pa,qa->', weight, t1, x['fov'])
            + 2 * jnp.einsum('pq,pjab,Pqa,Pjb->', weight, tau, x['Bvo'], x['Bvo'])
            - jnp.einsum('pq,pjab,Pqb,Pja->', weight, tau, x['Bvo'], x['Bvo']))


def _triples_reference(x):
    """JAX matrix functions plus the independently enumerated slow numerator."""
    x = _physical(x)
    no, nv = x['t1'].shape
    total = 0.
    for node, weight in zip(*GRID[:2]):
        t1, t2, bvo, boo, bvv, fov = dress(
            *(x[name] for name in ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'fov', 'foo', 'fvv')),
            node)
        left = symmetric_exp(-x['foo'] * node / 2) @ x['prjlo'].T
        right = symmetric_exp(x['foo'] * node / 2) @ x['prjlo'].T
        # Dressing makes the same-space factors nonsymmetric; the slow
        # reference's chemist-index convention requires the explicit swaps.
        total += weight * triples_matrix_reference(
            t1, t2, bvo, boo.swapaxes(1, 2), bvv.swapaxes(1, 2),
            -jnp.ones(no) / 3, jnp.zeros(nv), left @ right.T, fov)
    return total


def _reference_energies(x):
    x = _physical(x)
    mp2 = _projected_reference(x, jnp.zeros_like(x['t1']), .23 * x['t2'])
    cc = _projected_reference(x, x['t1'], x['t2'])
    return jnp.stack((mp2, cc, _triples_reference(x)))


def _assert_energy_and_response(evaluate, reference, x):
    actual, actual_vjp = jax.vjp(evaluate, x)
    expected, expected_vjp = jax.vjp(reference, x)
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-14)
    seed = jnp.asarray([1.3, -.7, 2.1])[-actual.size:]
    actual_bar, = actual_vjp(seed)
    expected_bar, = expected_vjp(seed)
    assert np.linalg.norm(expected_bar['prjlo']) > 1e-7
    for name in x:
        np.testing.assert_allclose(actual_bar[name], expected_bar[name],
                                   rtol=3e-10, atol=3e-13, err_msg=name)


@pytest.mark.parametrize('integral_transform', ['df', 'rotate'])
def test_partial_projected_ccsd_and_triples_are_covariant(integral_transform):
    """Catch a missing tensor leg or frozen physical-projector cotangent."""
    triples_native()
    x = _inputs()
    qo, qv = _rotations()
    solver = smooth_triples.make_solver(
        dict(mode='deterministic', integral_transform=integral_transform),
        GRID, with_grad=True)

    def evaluate(x):
        local = _rotate(x, qo, qv)
        eris = _eris(local)
        cc = lno_ccsd.ccsd_fragment_energy(eris, local['t1'], local['t2'], local['prjlo'])
        triples, _ = solver(None, eris, local['prjlo'], t1=local['t1'], t2=local['t2'])
        return jnp.stack((cc, triples))

    _assert_energy_and_response(evaluate, lambda x: _reference_energies(x)[1:], x)


class _PreparedCC:
    """Replace only molecular integral generation and iterative CC solving."""
    def __init__(self, mf, **kwargs):
        self.x = mf.synthetic_inputs
        self.max_memory = 200

    def ao2mo(self, **kwargs):
        return _eris(self.x)

    def init_amps(self, *, eris):
        return 0., jnp.zeros_like(self.x['t1']), .23 * self.x['t2']

    def kernel(self, **kwargs):
        return 0., self.x['t1'], self.x['t2']


def _run_core(x, *, amplitude_solver=None, triples_solver=None, ccsd_t=True):
    no, nv = x['t1'].shape
    mf = SimpleNamespace(
        synthetic_inputs=x, mo_occ=np.r_[np.full(no, 2.), np.zeros(nv)],
        e_tot=0., verbose=0, stdout=sys.stdout, mol=SimpleNamespace(verbose=0))
    lo_coeff = jnp.concatenate((x['prjlo'].T, jnp.zeros((nv, x['prjlo'].shape[0]))))
    return lno_ccsd._impurity_solve_core(
        mf, jnp.eye(no + nv), lo_coeff, _eris(x).fock, jnp.eye(no + nv),
        amplitude_solver=amplitude_solver, triples_solver=triples_solver, ccsd_t=ccsd_t)


def _shared_callback(qo, qv):
    def solve(mcc, eris, t1, t2, *, prjlo):
        result_type = getattr(lno_ccsd, 'FragmentAmplitudeResult', None)
        assert result_type is not None, 'shared CC frames need FragmentAmplitudeResult'
        local = _rotate(dict(mcc.x, prjlo=prjlo), qo, qv)
        return result_type(t1=local['t1'], t2=local['t2'], eris=_eris(local),
                           prjlo=local['prjlo'], diagnostics={'shared_local_frame': True})
    return solve


@pytest.mark.parametrize('integral_transform', ['df', 'rotate'])
def test_shared_core_routes_physical_cc_and_triples_but_preserves_mp2(
        monkeypatch, integral_transform):
    """A frame-carrying callback must control both downstream physical energies."""
    triples_native()
    monkeypatch.setattr(lno_ccsd, 'RCCSD', _PreparedCC)
    amplitude_solver = _shared_callback(*_rotations())
    triples_solver = smooth_triples.make_solver(
        dict(mode='deterministic', integral_transform=integral_transform),
        GRID, with_grad=True)

    def evaluate(x):
        energies, diagnostics = _run_core(
            x, amplitude_solver=amplitude_solver, triples_solver=triples_solver)
        assert diagnostics['ccsd'] == {'shared_local_frame': True}
        return jnp.stack(energies)

    _assert_energy_and_response(evaluate, _reference_energies, _inputs())


def test_shared_core_rejects_semicanonical_only_triples(monkeypatch):
    """Do not silently evaluate conventional (T) on noncanonical local tensors."""
    monkeypatch.setattr(lno_ccsd, 'RCCSD', _PreparedCC)

    def forbidden(*args, **kwargs):
        pytest.fail('conventional triples received a shared local frame')

    monkeypatch.setattr(lno_ccsd.ccsd_t_mod, 'kernel', forbidden)
    with pytest.raises(ValueError, match='shared|smooth|noncanonical|frame'):
        _run_core(_inputs(), amplitude_solver=_shared_callback(*_rotations()))


def test_original_tuple_callback_and_no_callback_keep_legacy_contract(monkeypatch):
    """The optional typed result must not alter the existing callback API."""
    monkeypatch.setattr(lno_ccsd, 'RCCSD', _PreparedCC)
    x = _inputs()
    legacy = _run_core(x, ccsd_t=False)
    assert isinstance(legacy, tuple) and len(legacy) == 3

    def solve(mcc, eris, t1, t2, *, prjlo):
        return mcc.x['t1'], mcc.x['t2'], {'converged': True}

    energies, diagnostics = _run_core(x, amplitude_solver=solve, ccsd_t=False)
    expected = (_projected_reference(x, jnp.zeros_like(x['t1']), .23 * x['t2']),
                _projected_reference(x, x['t1'], x['t2']), 0.)
    np.testing.assert_allclose(energies, expected, rtol=2e-12, atol=2e-14)
    np.testing.assert_array_equal(energies, legacy)
    assert diagnostics == {'ccsd': {'converged': True}}
