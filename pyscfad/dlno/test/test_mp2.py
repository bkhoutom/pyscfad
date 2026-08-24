import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from pyscfad import gto as ad_gto, scf
from pyscfad.mp import dfmp2
from pyscfad.dlno.mp2 import (
    kernel_dfmp2,
    pair_energy_multipole,
    pair_energy_multipole_cross,
)
from pyscfad.dlno.multipole_numpy import (
    multipole_orbital_data,
    multipole_pair_energy_cross,
    multipole_pair_energy_matrix,
)


def _four_distinct_orbitals():
    mol = gto.M(
        atom='''
        C  0.0  0.0  0.0
        C  3.1  0.4 -0.2
        C  6.4 -0.3  0.5
        C  9.8  0.2 -0.4
        ''',
        unit='Bohr',
        basis='sto-3g',
        spin=0,
        verbose=0,
    )
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()[:, 2:]

    occupied = []
    virtual = []
    for p0, _ in aoslices:
        occ = np.zeros(nao)
        occ[p0 + 2] = 1.0       # local 2p_x AO
        vir = np.zeros((nao, 2))
        vir[p0 + 1, 0] = 1.0   # local 2s AO
        vir[p0 + 3, 1] = 1.0   # local 2p_y AO
        occupied.append(occ)
        virtual.append(vir)

    e_occ = jnp.asarray([-0.83, -0.71, -0.62, -0.54])
    e_vir = [
        jnp.asarray([0.21, 0.47]),
        jnp.asarray([0.24, 0.51]),
        jnp.asarray([0.29, 0.56]),
        jnp.asarray([0.33, 0.61]),
    ]
    return mol, e_occ, occupied, e_vir, virtual


def test_pair_energy_multipole_cross_matches_full_cross_block():
    mol, e_occ, occupied, e_vir, virtual = _four_distinct_orbitals()

    full = pair_energy_multipole(
        mol, e_occ, occupied, e_vir, virtual, order=4
    )
    cross = pair_energy_multipole_cross(
        mol,
        e_occ[:2], occupied[:2], e_vir[:2], virtual[:2],
        e_occ[2:], occupied[2:], e_vir[2:], virtual[2:],
        order=4,
    )

    assert cross.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(cross)))
    np.testing.assert_allclose(cross, full[:2, 2:], rtol=1e-12, atol=1e-12)

    def summed_cross(left_energies):
        return pair_energy_multipole_cross(
            mol,
            left_energies, occupied[:2], e_vir[:2], virtual[:2],
            e_occ[2:], occupied[2:], e_vir[2:], virtual[2:],
            order=4,
        ).sum()

    gradient = jax.grad(summed_cross)(e_occ[:2])
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_static_numpy_multipole_matches_jax_order_four():
    mol, e_occ, occupied, e_vir, virtual = _four_distinct_orbitals()
    reference = pair_energy_multipole(
        mol, e_occ, occupied, e_vir, virtual, order=4
    )
    orbital_data = [
        multipole_orbital_data(
            mol,
            e_occ[index],
            occupied[index],
            e_vir[index],
            virtual[index],
            order=4,
        )
        for index in range(len(e_occ))
    ]

    static_matrix = multipole_pair_energy_matrix(orbital_data, order=4)
    static_cross = multipole_pair_energy_cross(
        orbital_data[:2], orbital_data[2:], order=4
    )

    np.testing.assert_allclose(
        static_matrix, np.asarray(reference), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        static_cross, np.asarray(reference)[:2, 2:],
        rtol=1e-12, atol=1e-12,
    )


def test_pair_energy_multipole_groups_heterogeneous_virtual_shapes():
    mol, e_occ, occupied, e_vir, virtual = _four_distinct_orbitals()
    e_vir = list(e_vir)
    virtual = list(virtual)
    e_vir[1] = e_vir[1][:1]
    virtual[1] = virtual[1][:, :1]

    result = pair_energy_multipole(
        mol, e_occ, occupied, e_vir, virtual, order=4
    )
    orbital_data = [
        multipole_orbital_data(
            mol,
            e_occ[index],
            occupied[index],
            e_vir[index],
            virtual[index],
            order=4,
        )
        for index in range(len(e_occ))
    ]
    reference = multipole_pair_energy_matrix(orbital_data, order=4)

    assert result.shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(result)))
    np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)


def test_kernel_dfmp2_identity_projector_matches_dfmp2():
    mol = ad_gto.Mole()
    mol.atom = "O 0 0 0; H 0 .75 .58; H 0 -.75 .58"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    nocc = mol.nelectron // 2
    projected, _ = kernel_dfmp2(mf, np.eye(nocc), with_t2=False)
    reference, _ = dfmp2.MP2(mf).kernel(with_t2=False)
    np.testing.assert_allclose(projected, reference, atol=1e-12)
