import jax
import jax.numpy as jnp
import numpy as np

from pyscfad import gto
from pyscfad.dlno import mp2 as dlno_mp2
from pyscfad.dlno.mp2 import pair_energy_multipole_cross


def _carbon_pair(coords):
    mol = gto.Mole(
        atom=[
            ('C', tuple(coords[0])),
            ('C', tuple(coords[1])),
        ],
        unit='Bohr',
        basis='sto-3g',
        verbose=0,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _fixed_orbitals(mol):
    occupied = []
    virtual = []
    for p0, p1 in mol.aoslice_by_atom()[:, 2:]:
        nao = p1 - p0
        occ = np.zeros(nao)
        occ[2] = 1.0
        vir = np.zeros((nao, 2))
        vir[1, 0] = 1.0
        vir[3, 1] = 1.0
        occupied.append(jnp.asarray(occ))
        virtual.append(jnp.asarray(vir))
    return occupied, virtual


def _two_mode_endpoint(nao):
    occupied = []
    first = np.zeros(nao)
    first[2] = 1.0
    first[1] = 0.13
    occupied.append(jnp.asarray(first))
    second = np.zeros(nao)
    second[3] = 1.0
    second[4] = -0.09
    occupied.append(jnp.asarray(second))

    shared_virtual = np.zeros((nao, 2))
    shared_virtual[1, 0] = 1.0
    shared_virtual[2, 0] = 0.07
    shared_virtual[4, 1] = 1.0
    shared_virtual[3, 1] = -0.11
    shared_virtual = jnp.asarray(shared_virtual)
    return (
        (jnp.asarray(-0.70), jnp.asarray(-0.62)),
        tuple(occupied),
        (jnp.asarray([0.20, 0.40]),) * 2,
        (shared_virtual,) * 2,
    )


def _two_mode_cross_arguments(mol):
    nao = int(mol.aoslice_by_atom()[0, 3] - mol.aoslice_by_atom()[0, 2])
    left = _two_mode_endpoint(nao)
    right = (
        (jnp.asarray(-0.66), jnp.asarray(-0.58)),
        left[1],
        (jnp.asarray([0.24, 0.46]),) * 2,
        left[3],
    )
    return left, right


def test_order_four_multipole_coordinate_vjp_matches_five_point_fd():
    coords = np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    mol = _carbon_pair(coords)
    occupied, virtual = _fixed_orbitals(mol)
    occupied_energy = (jnp.asarray(-0.70), jnp.asarray(-0.60))
    virtual_energy = (
        jnp.asarray([0.20, 0.40]),
        jnp.asarray([0.30, 0.50]),
    )

    def energy(mol_):
        return pair_energy_multipole_cross(
            mol_,
            occupied_energy[:1],
            occupied[:1],
            virtual_energy[:1],
            virtual[:1],
            occupied_energy[1:],
            occupied[1:],
            virtual_energy[1:],
            virtual[1:],
            atmlst_left=[[0]],
            atmlst_right=[[1]],
            order=4,
        ).sum()

    value, mol_bar = jax.value_and_grad(energy)(mol)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(mol_bar.coords)))
    np.testing.assert_allclose(
        value, energy(mol), rtol=1e-12, atol=1e-14
    )

    direction = np.asarray([
        [0.0, 0.0, -0.5],
        [0.0, 0.0, 0.5],
    ])
    direction /= np.linalg.norm(direction)
    ad_directional = float(np.vdot(np.asarray(mol_bar.coords), direction))

    step = 1e-3
    energies = {
        multiple: float(energy(_carbon_pair(
            coords + multiple * step * direction
        )))
        for multiple in (-2, -1, 1, 2)
    }
    fd_directional = (
        energies[-2]
        - 8.0 * energies[-1]
        + 8.0 * energies[1]
        - energies[2]
    ) / (12.0 * step)

    np.testing.assert_allclose(
        ad_directional, fd_directional, rtol=2e-6, atol=2e-10
    )
    np.testing.assert_allclose(
        np.asarray(mol_bar.coords).sum(axis=0),
        np.zeros(3),
        atol=2e-10,
    )


def test_multimode_order_four_coordinate_vjp_matches_five_point_fd():
    coords = np.asarray([
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 8.0],
    ])
    mol = _carbon_pair(coords)
    left, right = _two_mode_cross_arguments(mol)
    left_atoms = ([0], [0])
    right_atoms = ([1], [1])

    def energy(mol_):
        return pair_energy_multipole_cross(
            mol_,
            left[0], left[1], left[2], left[3],
            right[0], right[1], right[2], right[3],
            atmlst_left=left_atoms,
            atmlst_right=right_atoms,
            order=4,
        ).sum()

    value, mol_bar = jax.value_and_grad(energy)(mol)
    assert np.isfinite(float(value))

    direction = np.asarray([
        [0.13, -0.07, -0.41],
        [-0.13, 0.07, 0.41],
    ])
    direction /= np.linalg.norm(direction)
    ad_directional = float(np.vdot(np.asarray(mol_bar.coords), direction))
    step = 1e-3
    values = {
        multiple: float(energy(_carbon_pair(
            coords + multiple * step * direction
        )))
        for multiple in (-2, -1, 1, 2)
    }
    fd_directional = (
        values[-2] - 8.0 * values[-1]
        + 8.0 * values[1] - values[2]
    ) / (12.0 * step)
    np.testing.assert_allclose(
        ad_directional, fd_directional, rtol=3e-6, atol=3e-10
    )


def test_multimode_order_four_endpoint_vjps_match_directional_fd():
    mol = _carbon_pair(np.asarray([
        [0.0, 0.0, 0.0],
        [0.3, -0.2, 8.0],
    ]))
    left, right = _two_mode_cross_arguments(mol)
    arguments = (
        jnp.stack(left[0]),
        jnp.stack(left[1]),
        left[2][0],
        left[3][0],
        jnp.stack(right[0]),
        jnp.stack(right[1]).at[0, 1].add(0.021),
        right[2][0],
        right[3][0].at[0, 0].add(-0.037),
    )
    pair_weight = jnp.asarray([
        [0.37, -0.19],
        [0.23, 0.41],
    ])

    def scalar_energy(*endpoint_arrays):
        (left_e_occ, left_occ, left_e_vir, left_vir,
         right_e_occ, right_occ, right_e_vir, right_vir) = endpoint_arrays
        nleft = left_e_occ.shape[0]
        nright = right_e_occ.shape[0]
        left_values = (
            tuple(left_e_occ[index] for index in range(nleft)),
            tuple(left_occ[index] for index in range(nleft)),
            (left_e_vir,) * nleft,
            (left_vir,) * nleft,
        )
        right_values = (
            tuple(right_e_occ[index] for index in range(nright)),
            tuple(right_occ[index] for index in range(nright)),
            (right_e_vir,) * nright,
            (right_vir,) * nright,
        )
        left_data = dlno_mp2._multipole_endpoint_data(
            mol,
            *left_values,
            ([0],) * nleft,
            4,
        )
        right_data = dlno_mp2._multipole_endpoint_data(
            mol,
            *right_values,
            ([1],) * nright,
            4,
        )
        pair = dlno_mp2._multipole_cross_from_data(
            left_data, right_data, 4
        )
        return jnp.sum(pair_weight * pair)

    argnums = tuple(range(len(arguments)))
    value, bars = jax.value_and_grad(
        scalar_energy, argnums=argnums
    )(*arguments)
    assert np.isfinite(float(value))

    directions = tuple(
        jnp.linspace(-0.31, 0.27, value.size).reshape(value.shape)
        for value in arguments
    )
    norm = jnp.sqrt(sum(jnp.vdot(direction, direction)
                        for direction in directions))
    directions = tuple(direction / norm for direction in directions)
    ad_directional = sum(
        jnp.vdot(bar, direction)
        for bar, direction in zip(bars, directions)
    )
    step = 2e-5
    values = {}
    for multiple in (-2, -1, 1, 2):
        displaced = tuple(
            argument + multiple * step * direction
            for argument, direction in zip(arguments, directions)
        )
        values[multiple] = scalar_energy(*displaced)
    fd_directional = (
        values[-2] - 8.0 * values[-1]
        + 8.0 * values[1] - values[2]
    ) / (12.0 * step)
    np.testing.assert_allclose(
        ad_directional, fd_directional, rtol=3e-6, atol=3e-9
    )


def test_shared_endpoint_builds_one_raw_moment_set_per_side(monkeypatch):
    mol = _carbon_pair(np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 8.0],
    ]))
    left, right = _two_mode_cross_arguments(mol)
    calls = []
    original = dlno_mp2.multipole_ops._origin_zero_moments

    def counted(fake_mol, order):
        calls.append((fake_mol.nao, order))
        return original(fake_mol, order)

    monkeypatch.setattr(
        dlno_mp2.multipole_ops, '_origin_zero_moments', counted
    )
    energy = pair_energy_multipole_cross(
        mol,
        left[0], left[1], left[2], left[3],
        right[0], right[1], right[2], right[3],
        atmlst_left=([0], [0]),
        atmlst_right=([1], [1]),
        order=4,
    )
    assert np.all(np.isfinite(np.asarray(energy)))
    assert calls == [(5, 3), (5, 3)]


def test_differing_atom_lists_use_singleton_batches(monkeypatch):
    mol = _carbon_pair(np.asarray([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 8.0],
    ]))
    left, _ = _two_mode_cross_arguments(mol)

    calls = []
    original = dlno_mp2._multipole_orbital_data_batch

    def counted_batch(mol_, e_occ, mo_occ, e_vir, mo_vir, atmlst, order):
        calls.append((atmlst, len(e_occ)))
        return original(
            mol_, e_occ, mo_occ, e_vir, mo_vir, atmlst, order
        )

    monkeypatch.setattr(
        dlno_mp2, '_multipole_orbital_data_batch', counted_batch
    )
    data = dlno_mp2._multipole_endpoint_data(
        mol,
        left[0], left[1], left[2], left[3],
        ([0], [1]),
        4,
    )
    assert len(data) == 2
    assert all(np.all(np.isfinite(np.asarray(record[0]))) for record in data)
    assert calls == [((0,), 1), ((1,), 1)]
