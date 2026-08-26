"""Tests for host-only IAO local-correlation reporting helpers."""

from types import SimpleNamespace

import numpy as np

from pyscfad.dlno._output import (
    emit_lines,
    energy_summary_lines,
    fragment_energy_lines,
    lis_active_space_lines,
    lis_dimensions_from_static,
    local_correlation_settings_lines,
    mp2_prescreened_domain_lines,
    nuclear_force_lines,
)


def _static_selections():
    thresholds = SimpleNamespace(
        bp_occ=0.985,
        bp_primary=0.999,
        bp_ed=0.9998,
        bp_pao=0.98,
        pao_norm=1e-4,
        domain_pao=1e-4,
        ed_pao=0.995,
        occupied_weight=1e-4,
        metric_rank=1e-10,
        pair_energy=1.5e-5,
        near_pair_distance=3.5,
        multipole_order=4,
    )
    mp2_fragments = (
        SimpleNamespace(
            iao_indices=np.arange(3),
            fragment_atoms=np.arange(2),
            strong_fragments=np.arange(2),
            extended_atoms=np.arange(5),
            extended_ao_indices=np.arange(24),
            strong_occ_metric_keep=np.arange(6),
            strong_virtual=SimpleNamespace(metric_keep=np.arange(18)),
        ),
        SimpleNamespace(
            iao_indices=np.arange(4),
            fragment_atoms=None,
            strong_fragments=np.arange(1),
            extended_atoms=np.arange(3),
            extended_ao_indices=np.arange(15),
            strong_occ_metric_keep=np.arange(4),
            strong_virtual=SimpleNamespace(metric_keep=np.arange(11)),
        ),
    )
    mp2_static = SimpleNamespace(
        frozen=2,
        thresholds=thresholds,
        active_occ_indices=np.arange(8),
        active_vir_indices=np.arange(20),
        fragments=mp2_fragments,
    )
    lis_fragments = (
        SimpleNamespace(
            internal_occ_keep=np.arange(2),
            internal_vir_keep=np.arange(1),
            occupied_lno_keep=np.arange(3),
            virtual_lno_keep=np.arange(4),
            full_occupied_space=False,
            full_virtual_space=False,
        ),
        SimpleNamespace(
            internal_occ_keep=np.arange(3),
            internal_vir_keep=np.arange(2),
            occupied_lno_keep=np.zeros(0, dtype=int),
            virtual_lno_keep=np.zeros(0, dtype=int),
            full_occupied_space=True,
            full_virtual_space=True,
        ),
    )
    return SimpleNamespace(
        mp2_static=mp2_static,
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=1e-6,
        fragments=lis_fragments,
    )


def test_domain_and_lis_tables_report_fixed_dimensions():
    static = _static_selections()
    domain = "\n".join(mp2_prescreened_domain_lines(static))
    assert "IAO-MP2 PRESCREENED DOMAINS" in domain
    assert "target IAO/atoms" in domain
    assert "     1        3/2" in domain
    assert "     2        4/-" in domain

    occupied, virtual = lis_dimensions_from_static(static)
    assert occupied == (5, 8)
    assert virtual == (5, 20)
    lis = "\n".join(
        lis_active_space_lines(
            static, occupied, virtual, worker_ranks=(0, 1)
        )
    )
    assert "LIS ACTIVE SPACES" in lis
    assert "internal fragment orbitals + selected external LNOs" in lis
    assert "cut/cut" in lis
    assert "full/full" in lis


def test_settings_and_energy_summary_are_explicit():
    static = _static_selections()
    settings = "\n".join(
        local_correlation_settings_lines(
            static,
            ccsd_t=True,
            dcsd=False,
            nproc=4,
            pair_energy_model="multipole",
            force_full_domains=False,
        )
    )
    assert "Solver: CCSD(T)" in settings
    assert "4 MPI ranks" in settings
    assert "bp_occ=0.985" in settings
    assert "tau_occ=0.0001" in settings
    assert "model=multipole" in settings
    assert "force_full_domains=False" in settings
    assert "frozen=2" in settings

    energy = "\n".join(energy_summary_lines(
        e_hf=-10.0,
        e_iao_mp2=-0.3,
        e_mp2_lis=-0.2,
        e_ccsd=-0.25,
        e_ccsd_t=-0.01,
        e_corr=-0.36,
        e_total=-10.36,
    ))
    assert "ENERGY SUMMARY" in energy
    assert "IAO-MP2 + CCSD + (T) - MP2(LIS)" in energy
    assert "E(total)" in energy
    assert "-10.360000000000 Eh" in energy

    dcsd = "\n".join(energy_summary_lines(
        e_hf=-10.0,
        e_iao_mp2=-0.3,
        e_mp2_lis=-0.2,
        e_ccsd=-0.25,
        e_ccsd_t=0.0,
        e_corr=-0.35,
        e_total=-10.35,
        correlated_method="DCSD",
        include_triples=False,
    ))
    assert "sum_F E[DCSD(LIS_F)]" in dcsd
    assert "IAO-MP2 + DCSD - MP2(LIS)" in dcsd
    assert "(T)(LIS_F)" not in dcsd


def test_fragment_energy_table_reports_local_correction():
    records = ({
        "fragment_index": 0,
        "worker_rank": 2,
        "e_mp2_lis": -0.20,
        "e_ccsd": -0.25,
        "e_ccsd_t": -0.01,
        "wall_seconds": 12.25,
    },)
    table = "\n".join(
        fragment_energy_lines(records, correlated_method="CCSD")
    )
    assert "FRAGMENT ENERGY CONTRIBUTIONS" in table
    assert "local correction" in table
    assert "-0.0600000000" in table
    assert "12.2" in table


def test_force_table_prints_negative_gradient_and_net_force():
    class Molecule:
        natm = 2

        @staticmethod
        def atom_symbol(atom_index):
            return ("H", "O")[atom_index]

    gradient = np.asarray(((1.0, -2.0, 3.0), (-1.0, 2.0, -3.0)))
    force = "\n".join(nuclear_force_lines(Molecule(), gradient))
    assert "NUCLEAR FORCES (Eh/bohr; F = -dE/dR)" in force
    assert "-1.0000000000e+00" in force
    assert "+2.0000000000e+00" in force
    assert "-3.0000000000e+00" in force
    assert "max |sum_A F_A,k|" in force
    assert "0.0000000000e+00 Eh/bohr" in force


def test_emit_lines_uses_existing_reporter():
    observed = []
    emit_lines(observed.append, ("first", "second"))
    assert observed == ["first", "second"]
    emit_lines(None, ("ignored",))
