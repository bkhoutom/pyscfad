import numpy as np
from pyscf import gto

from pyscfad.dlno.domain import (
    _compute_av_numpy,
    _fragment_trace_completeness_numpy,
    get_bp_domain,
    get_fragment_bp_domain,
)


def _h4_chain():
    return gto.M(
        atom='H 0 0 0; H 1.4 0 0; H 2.8 0 0; H 4.2 0 0',
        unit='Bohr',
        basis='sto-3g',
        spin=0,
        verbose=0,
    )


def _weighted_two_column_block():
    # Deliberately neither normalized nor mutually orthogonal: the fragment BP
    # criterion is defined for the weighted block P_occ A_F itself.
    return np.asarray([
        [1.00, 0.05],
        [0.35, 0.80],
        [0.08, 0.30],
        [0.01, 0.08],
    ])


def test_fragment_bp_scalar_agrees_with_orbital_bp():
    mol = _h4_chain()
    s1e = mol.intor_symmetric('int1e_ovlp')
    orbital = np.asarray([1.0, 0.3, 0.1, 0.02])[:, None]
    orbital /= np.sqrt((orbital.T @ s1e @ orbital)[0, 0])

    for atoms in ([0], [0, 1], [0, 1, 2]):
        scalar_value = _compute_av_numpy(
            mol, orbital[:, 0], s1e=s1e, atmlst=atoms
        )
        fragment_value = _fragment_trace_completeness_numpy(
            mol, orbital, s1e=s1e, atmlst=atoms
        )
        np.testing.assert_allclose(fragment_value, scalar_value, atol=1e-13)

    for bp_thr in (0.8, 0.98, 0.999):
        scalar_domain = np.asarray(
            get_bp_domain(mol, orbital, s1e=s1e, bp_thr=bp_thr)[0]
        )
        fragment_domain = np.asarray(
            get_fragment_bp_domain(
                mol, [orbital], s1e=s1e, bp_thr=bp_thr
            )[0]
        )
        np.testing.assert_array_equal(fragment_domain, scalar_domain)


def test_fragment_bp_is_rotation_invariant():
    mol = _h4_chain()
    s1e = mol.intor_symmetric('int1e_ovlp')
    occupied_block = _weighted_two_column_block()
    rotation = np.asarray([[0.6, -0.8], [0.8, 0.6]])

    domain = np.asarray(
        get_fragment_bp_domain(
            mol, [occupied_block], s1e=s1e, bp_thr=0.98,
            atmlsts=[[0]],
        )[0]
    )
    rotated_domain = np.asarray(
        get_fragment_bp_domain(
            mol, [occupied_block @ rotation], s1e=s1e, bp_thr=0.98,
            atmlsts=[[0]],
        )[0]
    )

    np.testing.assert_array_equal(rotated_domain, domain)
    completeness = _fragment_trace_completeness_numpy(
        mol, occupied_block, s1e=s1e, atmlst=domain
    )
    rotated_completeness = _fragment_trace_completeness_numpy(
        mol, occupied_block @ rotation, s1e=s1e, atmlst=domain
    )
    np.testing.assert_allclose(rotated_completeness, completeness, atol=1e-13)


def test_fragment_bp_satisfies_trace_completeness_threshold():
    mol = _h4_chain()
    s1e = mol.intor_symmetric('int1e_ovlp')
    occupied_block = _weighted_two_column_block()
    bp_thr = 0.98

    domain = np.asarray(
        get_fragment_bp_domain(
            mol, [occupied_block], s1e=s1e, bp_thr=bp_thr,
            atmlsts=[[0]],
        )[0]
    )
    completeness = _fragment_trace_completeness_numpy(
        mol, occupied_block, s1e=s1e, atmlst=domain
    )

    assert completeness >= bp_thr - 1e-12


def test_fragment_bp_domains_are_nested_as_threshold_tightens():
    mol = _h4_chain()
    s1e = mol.intor_symmetric('int1e_ovlp')
    occupied_block = _weighted_two_column_block()
    thresholds = (0.7, 0.8, 0.98, 0.999)
    domains = [
        set(np.asarray(
            get_fragment_bp_domain(
                mol, [occupied_block], s1e=s1e, bp_thr=bp_thr,
                atmlsts=[[0]],
            )[0]
        ).tolist())
        for bp_thr in thresholds
    ]

    for loose, tight in zip(domains, domains[1:]):
        assert loose <= tight
