"""Parent PAO cutoffs use squared S norms in both selection paths."""
import numpy as np
import pytest
from pyscfad import gto, numpy as jnp, scf
from pyscfad.dlno_stc import BoysDomainThresholds, build_domain_selections, domain, driver


def _controlled_reference(first_norm2):
    # The first orthonormal basis vector is normalized AO 0. Rotating its
    # occupied direction by theta leaves PAO 0 with S norm squared sin(theta)^2.
    mol = gto.Mole(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    overlap = np.asarray(mol.intor_symmetric('int1e_ovlp'))
    basis = np.linalg.inv(np.linalg.cholesky(overlap).T)
    sine = np.sqrt(first_norm2 / overlap[0, 0])
    cosine = np.sqrt(1. - sine**2)
    coeff = basis @ np.array([[cosine, -sine], [sine, cosine]])
    energy = np.array([-1., 1.])
    fock = overlap @ coeff @ np.diag(energy) @ coeff.T @ overlap
    mf = scf.RHF(mol).density_fit()
    # This controlled reference tests selection only, without an SCF solve.
    mf.mo_coeff, mf.mo_energy = jnp.asarray(coeff), jnp.asarray(energy)
    mf.mo_occ, mf.converged = jnp.array([2., 0.]), True
    data = jnp.asarray(overlap), jnp.asarray(fock)
    mf._stc_scf_data = data
    return mf, data


@pytest.mark.parametrize('with_response', [False, True])
@pytest.mark.parametrize('cutoff,first_norm2,parents', [
    (None, 2.5e-5, [1]),
    (None, 4e-4, [0, 1]),
    (1e-4, 2.5e-5, [1]),
    (1e-4, 4e-4, [0, 1]),
    (1e-2, 4e-4, [1]),
])
def test_parent_cutoff_is_squared_norm(with_response, cutoff, first_norm2, parents):
    mf, data = _controlled_reference(first_norm2)
    thresholds = None if cutoff is None else BoysDomainThresholds(pao_norm=cutoff)

    def select(reference):
        return build_domain_selections(reference, frozen=0, thresholds=thresholds,
                                       pair_energy_model='all', force_full_domains=True)

    if with_response:
        static, _, _, prepared = driver._select_domain_reference(mf, select, scf_data=data)
        common = prepared.common
    else:
        static = select(mf)
        common = domain.rebuild_domain_data(mf, static, scf_data=data)
    np.testing.assert_array_equal(static.pao_parent_ao_indices, parents)
    # Selection uses squared norms; retained PAOs are still normalized by norms.
    coeff = np.asarray(common.pao_coeff)
    np.testing.assert_allclose(np.diag(coeff.T @ np.asarray(data[0]) @ coeff),
                               np.ones(len(parents)), atol=2e-13)
