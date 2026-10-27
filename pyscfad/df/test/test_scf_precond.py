"""Packed-DF SCF response preconditioning and unsupported-case fallbacks."""
from contextlib import nullcontext

import jax
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from pyscfad import config_update, dft, gto, scf
from pyscfad.df import _df_jk_opt
from pyscfad.scf import hf
from pyscfad.tools.linear_solver import eager_gmres


def _water():
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _packed_reference():
    mf = scf.RHF(_water()).density_fit(auxbasis='weigend')
    with config_update('pyscfad_moleintor_opt', True):
        mf.conv_tol = 1e-12
        mf.kernel()
        overlap = mf.get_ovlp()
    return mf, overlap, (mf.mo_energy, mf.mo_coeff, mf.mo_occ)


def test_coulomb_preconditioner_matches_independent_orbital_hessian():
    """Catch packed-pair weights and occupied/virtual flattening mistakes."""
    from pyscf import lib
    rng = np.random.default_rng(81)
    mf = scf.RHF(_water()).density_fit()
    nao = mf.mol.nao
    raw = rng.normal(size=(7, nao, nao))
    factors = raw + raw.transpose(0, 2, 1)
    mf.with_df._cderi = lib.pack_tril(factors)
    mf.with_df.blockdim = 2
    orbitals, _ = np.linalg.qr(rng.normal(size=(nao, nao)))
    occupied, virtual = orbitals[:, :3], orbitals[:, 3:]
    gaps = 1. + rng.random((nao - 3, 3))
    bov = np.einsum('Lpq,pa,qi->Lai', factors, virtual, occupied).reshape(7, -1)
    rhs = rng.normal(size=gaps.shape)
    hessian = np.diag(gaps.ravel()) + 4 * bov.T @ bov
    expected = rhs - gaps * np.linalg.solve(hessian, rhs.ravel()).reshape(gaps.shape)
    coulomb = _df_jk_opt.gen_coulomb_precond(mf.with_df, occupied, virtual, gaps)
    np.testing.assert_allclose(coulomb(rhs), expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('eager', [False, True])
def test_packed_preconditioner_supports_both_host_solver_selections(eager):
    mf, overlap, aux = _packed_reference()
    with (config_update('pyscfad_moleintor_opt', not eager),
          eager_gmres() if eager else nullcontext()):
        precond = hf._gen_precond(aux, mf, overlap, None)
        assert isinstance(precond, LinearOperator)
        assert np.isfinite(precond @ np.ones(mf.mol.nao ** 2)).all()


@pytest.mark.parametrize('captured_host', [False, True])
def test_scf_preconditioner_uses_captured_solver_across_contexts(captured_host):
    """The production wrapper uses its captured solver after contexts change."""
    from pyscfad._src.implicit_diff import make_implicit_diff
    from pyscfad.tools.linear_solver import gen_gmres
    mf, overlap, aux = _packed_reference()
    density = mf.make_rdm1()
    factors = np.asarray(aux[1])[:, np.asarray(aux[2]) > 0] * np.sqrt(2.)
    setups = []

    def preconditioner(*args, **kwargs):
        result = hf._gen_precond(*args, **kwargs)
        setups.append(result is not None)
        return result

    with config_update('pyscfad_moleintor_opt', False):
        with eager_gmres() if captured_host else nullcontext():
            solver = gen_gmres(tol=1e-12)
            wrapped = make_implicit_diff(
                lambda initial, mf, s, parameter, factors_: (parameter, factors),
                implicit_diff=True, has_aux=True, optimality_fun_has_aux=True,
                optimality_cond=lambda root, mf, s, parameter, factors_: (parameter, aux),
                solver=solver, nondiff_argnums=(4,), use_converged_args={4: 1},
                gen_precond=preconditioner, gen_precond_host_only=True)
        with nullcontext() if captured_host else eager_gmres():
            gradient = jax.grad(lambda parameter: jax.numpy.sum(
                wrapped(density, mf, overlap, parameter, None)[0]))(density)
    assert setups == [captured_host]
    np.testing.assert_allclose(gradient, np.ones_like(density), atol=1e-11, rtol=0)


def test_packed_disk_preconditioner_matches_resident_factors(tmp_path):
    import h5py
    mf, overlap, aux = _packed_reference()
    with config_update('pyscfad_moleintor_opt', True):
        resident = hf._gen_precond(aux, mf, overlap, None)
        path = tmp_path / 'packed.h5'
        with h5py.File(path, 'w') as output:
            output['j3c'] = np.asarray(mf.with_df._cderi)
        mf.with_df.attach_outcore_cderi(str(path))
        disk = hf._gen_precond(aux, mf, overlap, None)
    vector = np.random.default_rng(83).normal(size=mf.mol.nao ** 2)
    np.testing.assert_allclose(disk @ vector, resident @ vector, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('traced', ['aux', 'overlap', 'hcore', 'root_factors'])
def test_preconditioner_rejects_traced_inputs(traced):
    mf, overlap, aux = _packed_reference()
    def dispatch(value):
        arguments = dict(aux=aux, mf=mf, s1e=overlap, h1e=None, root_factors=None)
        key = dict(aux='aux', overlap='s1e', hcore='h1e', root_factors='root_factors')[traced]
        arguments[key] = (value, aux[1], aux[2]) if traced == 'aux' else value
        return hf._gen_precond(**arguments) is None
    with config_update('pyscfad_moleintor_opt', True):
        assert bool(jax.jit(dispatch)(jax.numpy.asarray(aux[0])))


@pytest.mark.parametrize('unsupported', [
    'disabled', 'jax_solver', 'full_factors', 'complex_factors',
    'complex_orbitals', 'complex_overlap', 'fractional', 'gap', 'no_df', 'dft', 'uhf',
])
def test_preconditioner_rejects_unsupported_inputs(unsupported):
    mf, overlap, aux = _packed_reference()
    energy, coeff, occ = map(np.asarray, aux)
    if unsupported == 'full_factors':
        mf.with_df._cderi = np.zeros((3, mf.mol.nao ** 2))
    elif unsupported == 'complex_factors':
        mf.with_df._cderi = np.asarray(mf.with_df._cderi).astype(complex)
    elif unsupported == 'complex_orbitals':
        coeff = coeff.astype(complex)
    elif unsupported == 'complex_overlap':
        overlap = np.asarray(overlap).astype(complex)
    elif unsupported == 'fractional':
        occ = occ.copy(); occ[0] = 1
    elif unsupported == 'gap':
        energy = np.zeros_like(energy)
    elif unsupported == 'no_df':
        mf = scf.RHF(mf.mol)
    elif unsupported == 'dft':
        mf = dft.RKS(mf.mol).density_fit(with_df=mf.with_df)
    elif unsupported == 'uhf':
        mf = scf.UHF(mf.mol).density_fit(with_df=mf.with_df)
    with (config_update('pyscfad_moleintor_opt', unsupported != 'jax_solver'),
          config_update('pyscfad_scf_df_precond', unsupported != 'disabled')):
        assert hf._gen_precond((energy, coeff, occ), mf, overlap, None) is None


def test_nonstationary_scf_response_preconditioned_gradient_agrees(monkeypatch):
    """The preconditioner must run and preserve an orbital-sensitive gradient."""
    original = hf._gen_precond
    counts = dict(setup=0, apply=0)

    def counted(aux, mf, overlap, hcore, root_factors=None, **kwargs):
        precond = original(aux, mf, overlap, hcore, root_factors, **kwargs)
        if precond is None:
            return None
        counts['setup'] += 1
        assert root_factors is not None
        _, coeff, occ = map(np.asarray, aux)
        np.testing.assert_allclose(root_factors @ root_factors.T,
                                   (coeff * occ) @ coeff.T, atol=2e-8, rtol=2e-8)
        def apply(vector):
            counts['apply'] += 1
            return precond @ vector
        return LinearOperator(precond.shape, matvec=apply, dtype=precond.dtype)

    monkeypatch.setattr(hf, '_gen_precond', counted)
    mol = _water()
    weights = jax.numpy.asarray(np.random.default_rng(82).normal(size=(mol.nao, mol.nao)) * .02)

    def objective(mol):
        mf = scf.RHF(mol).density_fit(auxbasis='weigend')
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        projections = jax.numpy.sum(weights * mf.mo_coeff, axis=0)
        return mf.e_tot + jax.numpy.vdot(projections, projections).real

    with (config_update('pyscfad_moleintor_opt', True),
          config_update('pyscfad_scf_implicit_diff', True)):
        with config_update('pyscfad_scf_df_precond', False):
            value_off, grad_off = jax.value_and_grad(objective)(mol)
        assert counts == dict(setup=0, apply=0)
        value_on, grad_on = jax.value_and_grad(objective)(mol)
    assert counts['setup'] == 1 and counts['apply'] > 0
    np.testing.assert_allclose(value_on, value_off, atol=1e-12, rtol=0)
    np.testing.assert_allclose(grad_on.coords, grad_off.coords, atol=2e-8, rtol=2e-8)
