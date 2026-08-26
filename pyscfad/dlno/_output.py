"""Compact, host-only reporting helpers for IAO local-correlation jobs.

The functions in this module format metadata and completed scalar results.
They must only be called outside differentiated/JIT-compiled code: reporting
must never rebuild an orbital space or add work to an AD tape.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy


_SECTION_WIDTH = 78


def _section(title):
    return f" {title} ".center(_SECTION_WIDTH, "=")


def _size(values):
    if values is None:
        return None
    return int(numpy.asarray(values).size)


def _count(values):
    size = _size(values)
    return "-" if size is None else str(size)


def _threshold(value):
    return f"{float(value):.6g}"


def _frozen_selection(value):
    if value is None:
        return "none"
    if numpy.isscalar(value):
        return str(value)
    indices = numpy.asarray(value).reshape(-1)
    if indices.size <= 12:
        return "[" + ",".join(str(item) for item in indices.tolist()) + "]"
    return f"{indices.size} explicit indices"


def _field(record, name, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def emit_lines(reporter, lines):
    """Send formatted lines to an existing serial or root-only reporter."""

    if reporter is None:
        return
    for line in lines:
        reporter(line)


def local_correlation_settings_lines(
    static_selections,
    *,
    ccsd_t,
    dcsd,
    nproc=1,
    pair_energy_model=None,
    force_full_domains=None,
):
    """Summarize the cutoffs and solver mode that define a CC calculation."""

    mp2_static = static_selections.mp2_static
    thresholds = mp2_static.thresholds
    method = "DCSD" if dcsd else ("CCSD(T)" if ccsd_t else "CCSD")
    full_domain_mode = (
        "unspecified"
        if force_full_domains is None
        else str(bool(force_full_domains))
    )
    execution = (
        "serial complete-fragment solves"
        if int(nproc) == 1
        else (
            f"{int(nproc)} MPI ranks; round-robin complete-fragment solves"
        )
    )
    return (
        _section("LOCAL-CORRELATION SETTINGS"),
        f"Solver: {method}; execution: {execution}",
        (
            "MP2 domain cutoffs: "
            f"bp_occ={_threshold(thresholds.bp_occ)}, "
            f"bp_primary={_threshold(thresholds.bp_primary)}, "
            f"bp_ed={_threshold(thresholds.bp_ed)}, "
            f"bp_pao={_threshold(thresholds.bp_pao)}"
        ),
        (
            "MP2 PAO/pair cutoffs: "
            f"pao_norm={_threshold(thresholds.pao_norm)}, "
            f"domain_pao={_threshold(thresholds.domain_pao)}, "
            f"ed_pao={_threshold(thresholds.ed_pao)}, "
            f"pair_energy={_threshold(thresholds.pair_energy)}"
        ),
        (
            "MP2 screening controls: "
            f"occupied_weight={_threshold(thresholds.occupied_weight)}, "
            f"metric_rank={_threshold(thresholds.metric_rank)}, "
            "near_pair_distance="
            f"{_threshold(thresholds.near_pair_distance)}, "
            f"multipole_order={int(thresholds.multipole_order)}"
        ),
        (
            "MP2 pair/domain mode: "
            f"model={pair_energy_model or 'unspecified'}, "
            f"force_full_domains={full_domain_mode}, "
            f"frozen={_frozen_selection(mp2_static.frozen)}"
        ),
        (
            "LIS cutoffs: "
            f"tau_occ={_threshold(static_selections.thresh_occ)}, "
            f"tau_vir={_threshold(static_selections.thresh_vir)}, "
            "internal_rank="
            f"{_threshold(static_selections.internal_rank_threshold)}"
        ),
    )


def mp2_prescreened_domain_lines(static_selections):
    """Format fixed MP2 strong-extended-domain dimensions per fragment."""

    mp2_static = getattr(static_selections, "mp2_static", static_selections)
    lines = [
        _section("IAO-MP2 PRESCREENED DOMAINS"),
        (
            "All fragment labels are 1-based. Strong F includes the target; "
            "ED is the MP2 strong extended domain."
        ),
        (
            "  Frag   target IAO/atoms   strong F   ED atoms   ED AOs"
            "   ED occ   ED vir"
        ),
    ]
    for number, fragment in enumerate(mp2_static.fragments, start=1):
        lines.append(
            f"  {number:4d}"
            f"   {_count(fragment.iao_indices):>6s}/"
            f"{_count(fragment.fragment_atoms):<5s}"
            f"   {_size(fragment.strong_fragments):8d}"
            f"   {_size(fragment.extended_atoms):8d}"
            f"   {_size(fragment.extended_ao_indices):6d}"
            f"   {_size(fragment.strong_occ_metric_keep):6d}"
            f"   {_size(fragment.strong_virtual.metric_keep):6d}"
        )
    return tuple(lines)


def lis_dimensions_from_static(static_selections):
    """Return the exact total LIS ranks encoded by fixed selections."""

    mp2_static = static_selections.mp2_static
    full_occ = _size(mp2_static.active_occ_indices)
    full_vir = _size(mp2_static.active_vir_indices)
    occupied = []
    virtual = []
    for selection in static_selections.fragments:
        internal_occ = _size(selection.internal_occ_keep)
        internal_vir = _size(selection.internal_vir_keep)
        occupied.append(
            full_occ
            if selection.full_occupied_space
            else internal_occ + _size(selection.occupied_lno_keep)
        )
        virtual.append(
            full_vir
            if selection.full_virtual_space
            else internal_vir + _size(selection.virtual_lno_keep)
        )
    return tuple(occupied), tuple(virtual)


def lis_active_space_lines(
    static_selections,
    lis_occupied,
    lis_virtual,
    *,
    worker_ranks=None,
):
    """Format internal, selected-LNO, and total CC active dimensions."""

    nfragment = len(static_selections.fragments)
    lis_occupied = tuple(int(value) for value in lis_occupied)
    lis_virtual = tuple(int(value) for value in lis_virtual)
    if len(lis_occupied) != nfragment or len(lis_virtual) != nfragment:
        raise ValueError("LIS dimension arrays must contain one row per fragment")
    if worker_ranks is None:
        worker_ranks = (None,) * nfragment
    else:
        worker_ranks = tuple(worker_ranks)
    if len(worker_ranks) != nfragment:
        raise ValueError("worker_ranks must contain one row per fragment")

    lines = [
        _section("LIS ACTIVE SPACES"),
        (
            "Solver LIS = internal fragment orbitals + selected external LNOs; "
            "all dimensions are occupied/virtual."
        ),
        (
            "  Frag   rank   internal occ/vir   selected LNO occ/vir"
            "   total solver LIS occ/vir   mode occ/vir"
        ),
    ]
    for number, (selection, total_occ, total_vir, worker_rank) in enumerate(
        zip(
            static_selections.fragments,
            lis_occupied,
            lis_virtual,
            worker_ranks,
            strict=True,
        ),
        start=1,
    ):
        internal_occ = _size(selection.internal_occ_keep)
        internal_vir = _size(selection.internal_vir_keep)
        lno_occ = total_occ - internal_occ
        lno_vir = total_vir - internal_vir
        if lno_occ < 0 or lno_vir < 0:
            raise ValueError("total LIS dimensions are smaller than internal ranks")
        rank = "-" if worker_rank is None else str(int(worker_rank))
        occ_mode = "full" if selection.full_occupied_space else "cut"
        vir_mode = "full" if selection.full_virtual_space else "cut"
        lines.append(
            f"  {number:4d}   {rank:>4s}"
            f"   {internal_occ:8d}/{internal_vir:<8d}"
            f"   {lno_occ:11d}/{lno_vir:<11d}"
            f"   {total_occ:12d}/{total_vir:<12d}"
            f"   {occ_mode}/{vir_mode}"
        )
    return tuple(lines)


def fragment_energy_lines(
    records,
    *,
    correlated_method="CCSD",
    include_triples=True,
):
    """Format completed fragment energies, ownership, and wall times."""

    correlated_method = str(correlated_method)
    if include_triples:
        header = (
            f"  Frag   rank       MP2(LIS)   {correlated_method:>8s}(LIS)"
            "        (T)(LIS)     local correction    wall/s"
        )
    else:
        header = (
            f"  Frag   rank       MP2(LIS)   {correlated_method:>8s}(LIS)"
            "     local correction    wall/s"
        )
    lines = [_section("FRAGMENT ENERGY CONTRIBUTIONS"), header]
    for number, record in enumerate(records, start=1):
        index = int(_field(record, "fragment_index", number - 1))
        rank_value = _field(record, "worker_rank")
        rank = "-" if rank_value is None else str(int(rank_value))
        e_mp2 = float(_field(record, "e_mp2_lis"))
        e_ccsd = float(_field(record, "e_ccsd"))
        e_t = float(_field(record, "e_ccsd_t"))
        correction = e_ccsd + e_t - e_mp2
        wall_value = _field(record, "wall_seconds")
        wall = "-" if wall_value is None else f"{float(wall_value):.1f}"
        row = (
            f"  {index + 1:4d}   {rank:>4s}"
            f"   {e_mp2:+15.10f}   {e_ccsd:+15.10f}"
        )
        if include_triples:
            row += f"   {e_t:+15.10f}"
        row += f"   {correction:+20.10f}   {wall:>7s}"
        lines.append(row)
    return tuple(lines)


def energy_summary_lines(
    *,
    e_hf,
    e_iao_mp2,
    e_mp2_lis,
    e_ccsd,
    e_ccsd_t,
    e_corr,
    e_total,
    correlated_method="CCSD",
    include_triples=True,
):
    """Format the unique additive IAO-DLNO-CCSD(T) energy expression."""

    correlated_method = str(correlated_method)
    correlated_label = f"sum_F E[{correlated_method}(LIS_F)]"
    triples_term = " + (T)" if include_triples else ""
    formula = (
        f"E(correlation) = IAO-MP2 + {correlated_method}"
        f"{triples_term} - MP2(LIS)"
    )
    lines = [
        _section("ENERGY SUMMARY"),
        f"E(HF)                                  {float(e_hf):+.12f} Eh",
        (
            "E(IAO-DLNO-MP2; strong + weak)      "
            f"{float(e_iao_mp2):+.12f} Eh"
        ),
        f"sum_F E[MP2(LIS_F)]                    {float(e_mp2_lis):+.12f} Eh",
        f"{correlated_label:<39s}{float(e_ccsd):+.12f} Eh",
    ]
    if include_triples:
        lines.append(
            "sum_F E[(T)(LIS_F)]                    "
            f"{float(e_ccsd_t):+.12f} Eh"
        )
    lines.extend((
        f"{formula:<61s}{float(e_corr):+.12f} Eh",
        f"E(total)                               {float(e_total):+.12f} Eh",
    ))
    return tuple(lines)


def nuclear_force_lines(mol, gradient):
    """Format per-atom forces from a completed nuclear energy gradient."""

    gradient = numpy.asarray(gradient)
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError("nuclear gradient must have shape (natom, 3)")
    if int(getattr(mol, "natm", gradient.shape[0])) != gradient.shape[0]:
        raise ValueError("gradient atom count does not match the molecule")
    forces = -gradient
    lines = [
        _section("NUCLEAR FORCES (Eh/bohr; F = -dE/dR)"),
        "  Atom   Element                 Fx                 Fy                 Fz",
    ]
    for atom_index, force in enumerate(forces):
        symbol = mol.atom_symbol(atom_index)
        lines.append(
            f"  {atom_index + 1:4d}   {symbol:<7s}"
            f"   {force[0]:+18.10e}   {force[1]:+18.10e}"
            f"   {force[2]:+18.10e}"
        )
    net_force = numpy.sum(forces, axis=0)
    max_net_component = (
        float(numpy.max(numpy.abs(net_force))) if net_force.size else 0.0
    )
    max_force_component = (
        float(numpy.max(numpy.abs(forces))) if forces.size else 0.0
    )
    lines.extend((
        f"||F||_2                             {numpy.linalg.norm(forces):.10e} Eh/bohr",
        f"max |F_A,k|                         {max_force_component:.10e} Eh/bohr",
        (
            "sum_A F_A                         "
            f"({net_force[0]:+.10e}, {net_force[1]:+.10e}, "
            f"{net_force[2]:+.10e}) Eh/bohr"
        ),
        f"max |sum_A F_A,k|                   {max_net_component:.10e} Eh/bohr",
    ))
    return tuple(lines)
