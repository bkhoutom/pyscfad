"""Independent normalization against exact-ERI conventional PySCF MP2."""
import jax
import numpy as np
from pyscf import gto, scf, mp, ao2mo
from pyscfad.lib._multipole import transition_moments, pair_energy
from pyscfad.dlno.multipole_numpy import multipole_pair_energy


def test_distant_pair_matches_full_spin_pyscf_mp2_and_radial_response():
    atom = gto.M(atom='He 0 0 0', basis='cc-pvdz', unit='Bohr', verbose=0)
    hf = scf.RHF(atom).run(conv_tol=1e-13)
    n = atom.nao_nr()
    distance = 20.
    mol = gto.M(atom=[('He', (0, 0, 0)), ('He', (0, 0, distance))],
                basis='cc-pvdz', unit='Bohr', verbose=0)
    occ = np.zeros((2*n, 2)); vir = np.zeros((2*n, 2*(n-1)))
    occ[:n, 0] = hf.mo_coeff[:, 0]; occ[n:, 1] = hf.mo_coeff[:, 0]
    vir[:n, :n-1] = hf.mo_coeff[:, 1:]; vir[n:, n-1:] = hf.mo_coeff[:, 1:]
    coeff = np.column_stack((occ, vir))
    energies = np.r_[np.repeat(hf.mo_energy[0], 2), np.tile(hf.mo_energy[1:], 2)]
    # Remove intra-atom blocks: retain the unordered interatomic pair only.
    ovov = ao2mo.general(mol, (occ, vir, occ, vir), compact=False).reshape(2, 2*(n-1), 2, 2*(n-1))
    ovov[0, :, 0, :] = 0.; ovov[1, :, 1, :] = 0.
    reference = scf.RHF(mol)
    reference.mo_coeff = coeff; reference.mo_energy = energies
    reference.mo_occ = np.r_[2., 2., np.zeros(2*(n-1))]
    eris = mp.mp2._ChemistsERIs()
    eris.mo_coeff = coeff; eris.mo_energy = energies
    eris.ovov = ovov.reshape(4*(n-1), -1)
    exact, _ = mp.mp2.kernel(mp.MP2(reference), eris=eris, with_t2=False)
    with mol.with_common_origin((0., 0., 0.)):
        overlap = mol.intor('int1e_ovlp'); dipole = mol.intor('int1e_r')
        second = mol.intor('int1e_rr').reshape(3, 3, 2*n, 2*n)
        third = mol.intor('int1e_rrr').reshape(3, 3, 3, 2*n, 2*n)
    np.testing.assert_allclose(coeff.T @ overlap @ coeff, np.eye(2*n), atol=1e-14)
    endpoints = []
    for i in range(2):
        center, mu, theta, omega = transition_moments(
            occ[:, i:i+1], vir[:, i*(n-1):(i+1)*(n-1)], overlap, dipole, second, third, 4)
        endpoints.append((center[0], mu[0], hf.mo_energy[1:]-hf.mo_energy[0], theta[0], omega[0]))
    actual = pair_energy(*endpoints, 4)
    np.testing.assert_allclose(actual, exact, rtol=1e-11, atol=1e-20)
    np.testing.assert_allclose(multipole_pair_energy(*endpoints, 4), exact, rtol=1e-11, atol=1e-20)
    # This s/p basis has only dipole transitions at long range: E ~ R^-6.
    def radial(r):
        right = (endpoints[1][0].at[2].set(r),) + endpoints[1][1:]
        return pair_energy(endpoints[0], right, 4)
    np.testing.assert_allclose(jax.grad(radial)(distance), -6*float(exact)/distance,
                               rtol=1e-10, atol=1e-20)


def test_halved_cutoff_preserves_screening_and_api_defaults():
    from pyscfad.dlno.domain import DLNOThresholds
    from pyscfad.dlno_stc._selection import BoysDomainThresholds
    assert DLNOThresholds().pair_energy == BoysDomainThresholds().pair_energy == 7.5e-6
    old_scores = -np.array([0., 0.9e-5, 1.e-5, 1.1e-5, 1.e-3])
    np.testing.assert_array_equal(abs(old_scores/2) > 5e-6, abs(old_scores) > 1e-5)
