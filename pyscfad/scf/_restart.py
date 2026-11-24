"""Reconstruct implicit RHF response at an already converged reference."""
import numpy

from pyscf.scf import hf as pyscf_hf
from pyscfad import config, numpy as np, ops
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.scf import hf
from pyscfad.scf._response_data import record


def _saved_root(dm, mf, overlap, hcore, factors):
    # The initial argument is the saved solution. The implicit wrapper supplies
    # its derivative through the usual SCF fixed-point equation, not through
    # this identity operation or any SCF iterations.
    del mf, overlap, hcore
    return dm, factors


def _fock_from_saved(mf, density, hcore, saved_fock):
    """Use saved F, but attach dH + dJ - dK/2 at this reference."""
    value = saved_fock - ops.stop_grad(hcore)
    backend = getattr(mf, '_sgx_backend', None)
    if backend is not None:
        from pyscfad.sgx.sgx import veff_from_saved
        vhf = veff_from_saved(backend, mf.mol, density, value, mf.direct_scf_tol)
    else:
        from pyscfad.df._df_jk_opt import veff_from_saved
        vhf = veff_from_saved(mf.with_df, density, value, mf.direct_scf_tol)
    live = hcore + vhf
    # Avoid even subtraction/addition roundoff in the exposed forward Fock.
    return saved_fock + (live - ops.stop_grad(live))


def restore_scf(mf, canonical, *, fock=None):
    """Install saved RHF values and attach their implicit nuclear response.

    The caller must authenticate the saved reference and its geometry/settings.
    The saved AO Fock supplies every forward value, including when setting
    up the implicit pullback. Integral data and derivative machinery are
    rebuilt; reverse mode still solves the ordinary SCF response equations.
    """
    if (not isinstance(mf, pyscf_hf.RHF) or isinstance(mf, pyscf_hf.KohnShamDFT)
            or mf.mol.spin != 0):
        raise NotImplementedError('saved SCF response requires closed-shell RHF')
    if not config.scf_implicit_diff or config.scf_first_order_custom:
        raise NotImplementedError('saved SCF response requires standard implicit differentiation')
    backend = getattr(mf, '_sgx_backend', None)
    if backend is None and (getattr(mf, 'with_df', None) is None
                            or getattr(mf, 'only_dfj', False)):
        raise NotImplementedError('saved Fock response requires DF-JK or SGX RHF')
    saved_fock = numpy.asarray(fock)
    if (saved_fock.shape != (mf.mol.nao, mf.mol.nao)
            or numpy.iscomplexobj(saved_fock) or not numpy.isfinite(saved_fock).all()):
        raise ValueError('restart requires a finite saved AO Fock matrix')
    coeff = numpy.asarray(canonical['mo_coeff'])
    occupation = numpy.asarray(canonical['mo_occ'])
    energies = numpy.asarray(canonical['mo_energy'])
    if numpy.iscomplexobj(coeff):
        raise NotImplementedError('saved SCF response currently requires real orbitals')
    if (coeff.ndim != 2 or coeff.shape[0] != mf.mol.nao
            or occupation.shape != (coeff.shape[1],) or energies.shape != occupation.shape
            or not numpy.all((occupation == 0) | (occupation == 2))
            or occupation.sum() != mf.mol.nelectron
            or any(not numpy.isfinite(x).all() for x in canonical.values())):
        raise ValueError('invalid saved RHF reference')
    mf.build(mf.mol)
    overlap = mf.get_ovlp(mf.mol)
    hcore = mf.get_hcore(mf.mol, s1e=overlap)
    factors = coeff[:, occupation > 0] * numpy.sqrt(occupation[occupation > 0])
    saved_density = (coeff * occupation) @ coeff.T
    # Materialize any lazy DF factors before the implicit primitive captures
    # mf's pytree. Building them afterwards would change its leaf structure
    # between the forward and backward passes.
    if backend is None and mf.with_df._cderi is None:
        mf.with_df.build()

    def saved_fixed_point(dm, current, s1e, h1e, root_factors=None):
        # The implicit VJP evaluates this only at the stored root. Krylov
        # trial vectors go through its derivative, never this cached primal.
        del root_factors
        fock = _fock_from_saved(current, dm, h1e, saved_fock)
        energy, coeff = current.eig(fock, s1e)
        occ = current.get_occ(energy, coeff)
        hf._stash_dfjk_mo_data(current, coeff, occ, s1e)
        nocc = current.mol.nelectron // 2
        if (backend is not None and numpy.all(occ[:nocc] == 2)
                and numpy.all(occ[nocc:] == 0)):
            dm = hf._density_from_fock_rhf(fock, s1e, energy, coeff, nocc)
        else:
            dm = hf._density_from_fock(fock, s1e, energy, coeff, occ)
        return dm, (energy, coeff, occ)

    if backend is not None:
        from pyscfad.tools.linear_solver import eager_gmres
        with eager_gmres():
            solver = hf.gen_gmres(log=hf.logger.new_logger(mf))
    else:
        solver = hf.gen_gmres(log=hf.logger.new_logger(mf))
    from pyscfad.tools._response_checkpoint import tag_solver
    solver = tag_solver(solver)
    attach_response = make_implicit_diff(
        _saved_root, True, optimality_cond=saved_fixed_point,
        solver=solver, has_aux=True, optimality_fun_has_aux=True,
        gen_precond=hf._gen_precond, gen_precond_host_only=True,
        nondiff_argnums=(4,), use_converged_args={4: 1})
    density, _ = attach_response(np.asarray(saved_density), mf, overlap, hcore, factors)
    fock = _fock_from_saved(mf, density, hcore, saved_fock)
    # Preserve the exact saved orbital gauge while differentiating the
    # generalized eigenproblem at that reference.
    mo_energy, mo_coeff = hf.mo_from_fock_eq78(fock, overlap, energies, coeff)
    hf._stash_dfjk_mo_data(mf, mo_coeff, occupation, overlap)
    final_density = mf.make_rdm1(mo_coeff, occupation)
    final_fock = _fock_from_saved(mf, final_density, hcore, saved_fock)
    energy = mf.energy_tot(final_density, hcore, final_fock - hcore)
    record(mf, overlap, final_fock)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_energy, mo_coeff, occupation
    # Keep the original scalar as well as its derivative; the replayed
    # integral reduction can differ by a few ulps.
    mf.e_tot = np.asarray(canonical['e_tot']) + (energy - ops.stop_grad(energy))
    mf.converged = True
    return mf
