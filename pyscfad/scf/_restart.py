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


def restore_scf(mf, canonical):
    """Install saved RHF values and attach their implicit nuclear response.

    The caller must authenticate the saved reference and its geometry/settings.
    Integral/Fock construction is repeated, but no SCF kernel, DIIS iteration,
    initial guess or forward orbital diagonalization is performed. Reverse
    mode still solves the ordinary SCF response equations when requested.
    """
    if (not isinstance(mf, pyscf_hf.RHF) or isinstance(mf, pyscf_hf.KohnShamDFT)
            or mf.mol.spin != 0):
        raise NotImplementedError('saved SCF response requires closed-shell RHF')
    if not config.scf_implicit_diff or config.scf_first_order_custom:
        raise NotImplementedError('saved SCF response requires standard implicit differentiation')
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
    mf.get_veff(mf.mol, np.asarray(saved_density), s1e=overlap)
    if getattr(mf, '_sgx_backend', None) is not None:
        from pyscfad.tools.linear_solver import eager_gmres
        with eager_gmres():
            solver = hf.gen_gmres()
    else:
        solver = hf.gen_gmres()
    attach_response = make_implicit_diff(
        _saved_root, True, optimality_cond=hf._scf_fixed_point,
        solver=solver, has_aux=True, optimality_fun_has_aux=True,
        gen_precond=hf._gen_precond, gen_precond_host_only=True,
        nondiff_argnums=(4,), use_converged_args={4: 1})
    density, _ = attach_response(np.asarray(saved_density), mf, overlap, hcore, factors)
    vhf = mf.get_veff(mf.mol, density, s1e=overlap)
    fock = mf.get_fock(hcore, overlap, vhf, density)
    # Preserve the exact saved orbital gauge while differentiating the
    # generalized eigenproblem at that reference.
    mo_energy, mo_coeff = hf.mo_from_fock_eq78(fock, overlap, energies, coeff)
    hf._stash_dfjk_mo_data(mf, mo_coeff, occupation, overlap)
    final_density = mf.make_rdm1(mo_coeff, occupation)
    final_vhf = hf._get_veff_from_orbitals(
        mf, mf.mol, final_density, mo_coeff, occupation,
        dm_last=density, vhf_last=vhf, s1e=overlap)
    energy = mf.energy_tot(final_density, hcore, final_vhf)
    final_fock = mf.get_fock(hcore, overlap, final_vhf, final_density)
    record(mf, overlap, final_fock)
    mf.mo_energy, mf.mo_coeff, mf.mo_occ = mo_energy, mo_coeff, occupation
    # Keep the original scalar as well as its derivative; the replayed
    # integral reduction can differ by a few ulps.
    mf.e_tot = np.asarray(canonical['e_tot']) + (energy - ops.stop_grad(energy))
    mf.converged = True
    return mf
