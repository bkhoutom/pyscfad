"""DLNO-MP2 energies and gradients for IAO or Boys targets.

Domain construction is shared with CCSD through domain and dlno_base.
Weighted fragment contractions are implemented in fragment_mp2."""

from __future__ import annotations

from pyscfad.lno import df as lno_df
from dataclasses import dataclass, replace
import gc
import time

import jax
import numpy

from pyscfad import numpy as np

from . import multipole
from .dlno_base import build_strong_ed_domain, build_weak_multipole_screen, rebuild_domain_data
from .domain import (
    DLNOThresholds, DomainTopology, _build_fragment_domain_orbitals, _domain_lov,
    build_domain_topology,
)
from .fragment_mp2 import (
    fragment_pair_energy_from_lov, fragment_pair_energy_from_lov_jax,
)
from .targets import resolve_target_options, validate_boys_target_groups


__all__ = [
    "DLNOMP2",
    "MP2Timing",
    "FragmentDomainResult",
    "MP2Result",
    "evaluate_domain_mp2",
    "kernel",
    "FragmentDimensions",
    "MP2TermResult",
    "MP2GradientTiming",
    "MP2Decomposition",
    "strong_domain_energy",
    "weak_screen_pair_energy",
    "strong_fragment_energy",
    "correlation_energy",
    "correlation_value_and_grad_from_common",
    "correlation_value_and_grad",
    "correlation_value_and_grad_with_targets",
]


def _serial_restart_scientific_payload(
    mol,
    mf,
    *,
    frag_lolist,
    frag_atmlist,
    frozen,
    thresholds,
    pair_energy_model,
    force_full_domains,
    include_hf,
    lo_type="iao",
    lo_kwargs=None,
):
    """Return the stable scientific inputs for a serial MP2 restart."""

    import numpy
    from ._restart import df_source_fingerprint

    lo_type, lo_kwargs = resolve_target_options(lo_type, lo_kwargs)
    with_df = getattr(mf, "with_df", None)
    auxmol = None if with_df is None else getattr(with_df, "auxmol", None)

    return {
        "driver": "serial-iao-dlno-mp2-gradient",
        "system": {
            "coords_bohr": numpy.asarray(mol.atom_coords()),
            "atom_symbols": tuple(
                mol.atom_symbol(index) for index in range(mol.natm)
            ),
            "atom_charges": numpy.asarray(mol.atom_charges()),
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "basis": getattr(mol, "_basis", None),
            "ecp": getattr(mol, "_ecp", None),
            "pseudo": getattr(mol, "_pseudo", None),
            "cart": bool(getattr(mol, "cart", False)),
            "nucmod": getattr(mol, "nucmod", None),
        },
        "scf": {
            "class": f"{type(mf).__module__}.{type(mf).__qualname__}",
            "mo_occ": numpy.asarray(mf.mo_occ),
            # MO coefficients and orbital energies are intentionally omitted:
            # tiny SCF rerun noise is not a reliable exact restart identity.
            "e_tot": float(numpy.round(float(mf.e_tot), decimals=8)),
            "auxbasis": (
                None if with_df is None else getattr(with_df, "auxbasis", None)
            ),
            "auxmol_basis": (
                None if auxmol is None else getattr(auxmol, "_basis", None)
            ),
            "df_source": df_source_fingerprint(mf),
        },
        "local_correlation": {
            "lo_type": lo_type,
            "lo_kwargs": lo_kwargs,
            "frag_lolist": frag_lolist,
            "frag_atmlist": frag_atmlist,
            "frozen": frozen,
            "thresholds": thresholds,
            "pair_energy_model": str(pair_energy_model),
            "force_full_domains": bool(force_full_domains),
            "include_hf": bool(include_hf),
        },
    }


def _fix_restart_mo_phases(mf):
    """Put canonical MOs in a deterministic phase gauge inside the SCF VJP."""

    import jax.numpy as np

    coeff = np.asarray(mf.mo_coeff)
    coeff_abs = np.abs(coeff)
    largest = np.max(coeff_abs, axis=0)
    # Symmetry-related AO coefficients can tie.  Treat values within numerical
    # SCF noise as the same maximum and choose the first AO deterministically.
    near_largest = coeff_abs >= (largest[None, :] * (1.0 - 1e-10))
    pivot_rows = np.argmax(near_largest, axis=0)
    pivot = coeff[pivot_rows, np.arange(coeff.shape[1])]
    pivot_abs = np.abs(pivot)
    phase = np.where(
        pivot_abs > 0,
        np.conj(pivot) / pivot_abs,
        np.ones_like(pivot),
    )
    mf.mo_coeff = coeff * phase[None, :]
    return mf


class DLNOMP2:
    """Fixed-topology DLNO-MP2 energy and nuclear gradient for IAO or Boys targets.

    The public :meth:`value_and_grad` entry point mirrors the progressive
    interface used by :class:`pyscfad.dlno.ccsd.DLNOCCSD`: ``build_mf`` is
    traced once, the discrete ED/pair topology is constructed eagerly at the
    reference geometry, and fragment energies are differentiated one at a
    time.  Only atom/index lists, strong/weak pair classes, and retained-rank
    choices are frozen.  Targets, PAOs, local semicanonical orbitals, local RI
    factors, strong-pair MP2 energies, and weak multipole energies are rebuilt
    on the differentiable path.

    The lower-level :meth:`correlation_value_and_grad` method returns an SCF
    object cotangent rather than closing the SCF response.  DLNO-CCSD(T) can
    therefore add the same PT2 correction to its existing SCF cotangent and
    perform one final CPHF pullback.
    """

    @staticmethod
    def build_static_topology(mf, **kwargs):
        """Build the reference topology and retain only discrete choices.

        A driver that already owns a saved SCF VJP should call this eager
        helper on a detached ``mf`` (for example through ``stop_trace``), as
        :meth:`value_and_grad` does.  Besides defining the derivative
        boundary, this prevents topology-only DF setup from mutating the SCF
        object whose original pytree structure belongs to the saved VJP.
        """
        from ._selection import build_domain_selections

        reference = build_domain_topology(mf, **kwargs)
        return build_domain_selections(mf, reference)

    @staticmethod
    def correlation_value_and_grad(
        mf, topology, *, return_details=False, restart=None
    ):
        """Return ``(E_corr, mf_bar[, details])`` for fixed topology."""


        return correlation_value_and_grad(
            mf,
            topology,
            return_details=return_details,
            restart=restart,
        )

    @staticmethod
    def correlation_value_and_grad_with_targets(
        mf, iao_coeff, topology, *, return_details=False
    ):
        """Return correlation cotangents and optional scalar diagnostics."""


        return correlation_value_and_grad_with_targets(
            mf,
            iao_coeff,
            topology,
            return_details=return_details,
        )

    @classmethod
    def value_and_grad(
        cls,
        mol,
        *,
        build_mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=None,
        pair_energy_model="multipole",
        force_full_domains=False,
        topology=None,
        include_hf=True,
        return_details=False,
        checkpoint_dir=None,
        resume=False,
        lo_type="iao",
        lo_kwargs=None,
    ):
        """Return the fixed-topology local-MP2 energy and nuclear gradient.

        Parameters
        ----------
        mol
            Differentiable molecular object.
        build_mf
            Callable ``mol -> converged density-fitted RHF object``.  It is
            evaluated once under :func:`jax.vjp`.
        frag_lolist, frag_atmlist, frozen, thresholds, pair_energy_model,
        force_full_domains
            Passed to :func:`build_domain_topology` when ``topology`` is
            not supplied.
        topology
            Optional fixed selections returned by
            :func:`pyscfad.dlno._selection.build_domain_selections`.
            An energy-only :class:`DomainTopology` is also accepted and
            converted at its matching reference geometry.  Reusing the fixed
            selections for displaced geometries keeps all domain and pair
            decisions identical.
        include_hf
            Include the RHF reference energy and its response.  If false,
            return only the local-MP2 correlation energy and gradient.
        return_details
            If true, append an
            :class:`pyscfad.dlno.mp2.MP2Decomposition` containing
            the strong/weak split, term counts, ED dimensions, and timings.
            The records contain host scalars only; progressive AD tapes are
            still released one term at a time.
        checkpoint_dir
            Optional directory for an atomic progressive restart.  The fixed
            topology, cumulative completed-term cotangent, common-orbital
            closed cotangent, and final pre-SCF cotangent are retained.
        resume
            Resume a compatible calculation from ``checkpoint_dir``.  The
            converged SCF VJP is always recreated, because a Python pullback
            closure cannot be serialized.  A completed pre-SCF record then
            proceeds directly to that one implicit SCF response.  The fresh
            builder may read an SCF checkpoint, but it must represent the same
            converged implicit DF-RHF map; compatibility permits only tiny
            post-checkpoint SCF cleanup noise in the phase-fixed orbitals.

        Notes
        -----
        Domain construction is deliberately outside the AD graph, but domain
        *orbitals* are not.  In particular, the cached numerical weak-pair
        energy in :class:`DomainTopology` is never used by this method.
        The integral-direct local-RI reverse pass currently targets nuclear
        coordinates only; build ``mol`` with ``trace_exp=False`` and
        ``trace_ctr_coeff=False`` for this interface.

        The integral-direct local-RI reverse pass currently requires a
        positive-definite auxiliary metric for its Cholesky pullback.  The
        forward eigenvalue fallback for a linearly dependent auxiliary basis
        does not yet have a corresponding reverse rule and will raise rather
        than silently return an incomplete gradient.

        The saved SCF response is always built with the standard implicit
        backend.  The experimental ``pyscfad_scf_first_order_custom`` backend
        replays a finite SCF iteration history in its backward pass; for a
        nonstationary local-orbital cotangent, a small residual in that replay
        can be amplified into a large, non-reproducible nuclear gradient.
        Standard implicit response instead differentiates the converged SCF
        fixed point and is required by this interface.
        """
        import jax
        import jax.numpy as np

        from pyscfad import config_update
        from pyscfad.ops import stop_trace
        from ._restart import RestartManager, scientific_digest
        from ._selection import DomainSelections, build_domain_selections

        lo_type, lo_kwargs = resolve_target_options(lo_type, lo_kwargs)
        if lo_type == "boys" and frag_atmlist is not None:
            raise ValueError(
                "Boys targets do not accept molecular frag_atmlist overrides"
            )
        if topology is not None:
            requested = (lo_type, lo_kwargs)
            supplied = resolve_target_options(
                getattr(topology, "lo_type", "iao"),
                getattr(topology, "lo_kwargs", None),
            )
            if scientific_digest(requested) != scientific_digest(supplied):
                raise ValueError(
                    "supplied topology has incompatible localization mode or settings"
                )
            if lo_type == "boys" and frag_lolist is not None:
                groups = validate_boys_target_groups(
                    frag_lolist, len(topology.frag_lolist)
                )
                if any(not numpy.array_equal(a, b)
                       for a, b in zip(groups, topology.frag_lolist)):
                    raise ValueError(
                        "supplied topology has incompatible Boys singleton target map"
                    )

        if (
            getattr(mol, "exp", None) is not None
            or getattr(mol, "ctr_coeff", None) is not None
        ):
            raise NotImplementedError(
                "DLNOMP2.value_and_grad currently differentiates "
                "nuclear coordinates only; build mol with trace_exp=False "
                "and trace_ctr_coeff=False"
            )

        if thresholds is None:
            thresholds = DLNOThresholds()

        if resume and checkpoint_dir is None:
            raise ValueError("resume=True requires checkpoint_dir")

        scf_builder = build_mf
        if checkpoint_dir is not None:
            # Orbital cotangents are gauge covariant, so a phase flip between
            # the original SCF and its restart cannot be ignored.  Make the
            # phase convention deterministic inside the SCF VJP itself: the
            # largest AO coefficient of every MO is real and nonnegative.
            # This keeps both the saved cotangent and the fresh pullback in
            # the same gauge without serializing a live Python closure.
            def scf_builder(mol_):
                return _fix_restart_mo_phases(build_mf(mol_))

        with (
            config_update("pyscfad_scf_implicit_diff", True),
            config_update("pyscfad_scf_first_order_custom", False),
        ):
            mf, scf_pullback = jax.vjp(scf_builder, mol)

        restart = RestartManager(
            checkpoint_dir,
            resume=resume,
            method="dlno-mp2",
            scientific_payload=_serial_restart_scientific_payload(
                mol,
                mf,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                include_hf=include_hf,
                lo_type=lo_type,
                lo_kwargs=lo_kwargs,
            ),
            initialize=True,
        )

        if topology is not None and not isinstance(
            topology,
            (DomainTopology, DomainSelections),
        ):
            raise TypeError(
                "topology must be DomainTopology or "
                "DomainSelections"
            )

        saved_topology = None
        if resume and restart.enabled:
            saved_topology = restart.load_static(
                expected_type=DomainSelections
            )
        if saved_topology is not None:
            if isinstance(topology, DomainSelections):
                restart.bind_static(topology)
            elif isinstance(topology, DomainTopology):
                supplied_static = stop_trace(
                    lambda mf_: build_domain_selections(
                        mf_, topology
                    )
                )(mf)
                restart.bind_static(supplied_static)
            fixed_topology = saved_topology
        elif topology is None:
            def build_static(mf_):
                reference = build_domain_topology(
                    mf_,
                    frozen=frozen,
                    frag_lolist=frag_lolist,
                    frag_atmlist=frag_atmlist,
                    thresholds=thresholds,
                    pair_energy_model=pair_energy_model,
                    force_full_domains=force_full_domains,
                    lo_type=lo_type,
                    lo_kwargs=lo_kwargs,
                )
                return build_domain_selections(mf_, reference)

            fixed_topology = stop_trace(build_static)(mf)
        elif isinstance(topology, DomainTopology):
            fixed_topology = stop_trace(
                lambda mf_: build_domain_selections(mf_, topology)
            )(mf)
        else:
            fixed_topology = topology

        if restart.enabled and saved_topology is None:
            restart.save_static(fixed_topology)

        # Build the HF seed before looking for the pre-SCF boundary.  Besides
        # supplying the optional reference contribution, ``hf_bar`` is the
        # live MF-shaped template used to deserialize a saved total cotangent.
        e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
        hf_bar, = hf_pullback(
            np.ones((), dtype=np.asarray(e_hf).dtype)
        )
        pre_scf = None
        if resume and restart.enabled:
            pre_scf = restart.load_record(
                "pre_scf",
                templates={"mf_bar": hf_bar},
                missing_ok=True,
            )
        if pre_scf is not None:
            energy = np.asarray(
                pre_scf.scalars["energy"], dtype=np.asarray(e_hf).dtype
            )
            mf_bar = pre_scf.trees["mf_bar"]
            details = None
            if return_details:
                # This is a disk-only reconstruction from correlation_closed;
                # no local MP2 term is reevaluated in a valid pre-SCF restart.
                closed_result = cls.correlation_value_and_grad(
                    mf,
                    fixed_topology,
                    return_details=True,
                    restart=restart,
                )
                details = closed_result[2]
            mol_bar, = scf_pullback(mf_bar)
            jax.block_until_ready((energy, mol_bar))
            if return_details:
                return energy, mol_bar, details
            return energy, mol_bar

        corr_result = cls.correlation_value_and_grad(
            mf,
            fixed_topology,
            return_details=return_details,
            restart=restart,
        )
        e_corr, mf_bar = corr_result[:2]
        if include_hf:
            mf_bar = jax.tree_util.tree_map(
                _add_cotangent, mf_bar, hf_bar
            )
            energy = e_hf + e_corr
        else:
            energy = e_corr

        jax.block_until_ready((energy, mf_bar))
        if restart.enabled:
            restart.save_record(
                "pre_scf",
                scalars={"energy": float(jax.device_get(energy))},
                trees={"mf_bar": mf_bar},
                metadata={"include_hf": bool(include_hf)},
            )
        mol_bar, = scf_pullback(mf_bar)
        jax.block_until_ready((energy, mol_bar))
        if return_details:
            return energy, mol_bar, corr_result[2]
        return energy, mol_bar


@dataclass(frozen=True)
class MP2Timing:
    """Post-topology wall-time split for one fragment or their aggregate.

    The aggregate is exactly the sum of the fragment timings.  One-time
    overlap/Fock preparation is charged to the first fragment's ED-orbital
    stage, and final evaluator-owned cache cleanup is charged to the last
    fragment's bookkeeping stage.
    """

    ed_orbital_seconds: float = 0.0
    local_ri_lov_seconds: float = 0.0
    weighted_ed_mp2_seconds: float = 0.0
    weak_bookkeeping_seconds: float = 0.0

    @property
    def total_seconds(self):
        """Sum of the four profiled post-topology stages."""
        return (
            self.ed_orbital_seconds
            + self.local_ri_lov_seconds
            + self.weighted_ed_mp2_seconds
            + self.weak_bookkeeping_seconds
        )


@dataclass(frozen=True)
class FragmentDomainResult:
    """MP2 contribution and dimensions for one fragment ED.

    ``weak_multipole_opposite_spin`` follows the OS-based pair-increment
    convention of Nagy et al., including the pair-orientation prefactor in
    their Eq. (7).  It is not the conventional SCS-MP2 OS component resolved
    from a global canonical energy expression.
    """

    fragment_index: int
    strong_fragments: numpy.ndarray
    pao_center_atoms: numpy.ndarray
    extended_atoms: numpy.ndarray
    n_domain_ao: int
    n_domain_occ: int
    n_domain_vir: int
    target_weight_trace: float
    partner_weight_trace: float
    strong_total: float
    strong_opposite_spin: float
    strong_same_spin: float
    weak_multipole_opposite_spin: float
    timing: MP2Timing | None = None


@dataclass(frozen=True)
class MP2Result:
    """Summed strong-plus-weak DLNO-MP2 result.

    ``e_strong_os`` and ``e_strong_ss`` are conventional MP2 spin components.
    ``e_weak_multipole_os`` instead carries Nagy's OS-based distant-pair
    increment convention; it approximates the omitted total distant-pair
    contribution and should not be interpreted as an SCS-OS diagnostic.
    """

    e_corr: float
    e_strong: float
    e_weak_multipole_os: float
    e_strong_os: float
    e_strong_ss: float
    fragments: tuple
    topology: DomainTopology
    timing: MP2Timing | None = None


def evaluate_domain_mp2(mf, topology):
    """Evaluate exact strong-pair ED MP2 plus weak multipole OS-MP2.

    This energy-validation routine uses the reference continuous quantities
    cached in ``topology``.  It is therefore not yet a nuclear-gradient entry
    point; see :class:`DomainTopology` for the required fixed-topology
    rebuild boundary.
    """
    if not isinstance(topology, DomainTopology):
        raise TypeError("topology must be DomainTopology")

    setup_start = time.perf_counter()
    fragment_results = []
    strong_total = 0.0
    strong_os = 0.0
    strong_ss = 0.0
    weak_os = 0.0
    # Reuse the reference overlap/Fock to avoid another global J/K build.
    # The gradient driver rebuilds them through rebuild_domain_data.
    s1e = numpy.asarray(topology.s1e)
    fock = numpy.asarray(topology.fock)
    setup_seconds = time.perf_counter() - setup_start
    for fragment_index in range(len(topology.frag_lolist)):
        stage_start = time.perf_counter()
        domain = _build_fragment_domain_orbitals(
            mf,
            topology,
            fragment_index,
            s1e=s1e,
            fock=fock,
        )
        ed_orbital_seconds = time.perf_counter() - stage_start
        if fragment_index == 0:
            ed_orbital_seconds += setup_seconds

        stage_start = time.perf_counter()
        lov = _domain_lov(mf, domain)
        local_ri_lov_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        energy = fragment_pair_energy_from_lov(
            lov,
            domain["occupied_energy"],
            domain["virtual_energy"],
            domain["target_weight"],
            domain["partner_weight"],
            target_factor=domain["target_projection"],
            max_memory_mb=(
                256.0
                if topology.thresholds.mp2_block_memory_mb is None
                else topology.thresholds.mp2_block_memory_mb
            ),
        )
        weighted_ed_mp2_seconds = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        fragment_strong = float(numpy.real(energy.total))
        fragment_os = float(numpy.real(energy.opposite_spin))
        fragment_ss = float(numpy.real(energy.same_spin))

        weak_mask = ~topology.strong_mask[fragment_index]
        # ``pair_energy`` is a symmetric unordered-pair multipole estimate.
        # Half of each row assigns one half to either endpoint, so summing all
        # fragment rows counts every weak pair once.
        fragment_weak = 0.5 * float(numpy.sum(
            topology.weak_pair_energy[fragment_index, weak_mask]
        ))
        strong_total += fragment_strong
        strong_os += fragment_os
        strong_ss += fragment_ss
        weak_os += fragment_weak
        weak_bookkeeping_seconds = time.perf_counter() - stage_start
        fragment_results.append(FragmentDomainResult(
            fragment_index=fragment_index,
            strong_fragments=numpy.asarray(
                topology.strong_fragments[fragment_index]
            ),
            pao_center_atoms=domain["center_atoms"],
            extended_atoms=domain["extended_atoms"],
            n_domain_ao=int(domain["ao_idx"].size),
            n_domain_occ=int(domain["occupied_coeff"].shape[1]),
            n_domain_vir=int(domain["virtual_coeff"].shape[1]),
            target_weight_trace=float(numpy.trace(
                domain["target_weight"]
            ).real),
            partner_weight_trace=float(numpy.trace(
                domain["partner_weight"]
            ).real),
            strong_total=fragment_strong,
            strong_opposite_spin=fragment_os,
            strong_same_spin=fragment_ss,
            weak_multipole_opposite_spin=fragment_weak,
            timing=MP2Timing(
                ed_orbital_seconds=ed_orbital_seconds,
                local_ri_lov_seconds=local_ri_lov_seconds,
                weighted_ed_mp2_seconds=weighted_ed_mp2_seconds,
                weak_bookkeeping_seconds=weak_bookkeeping_seconds,
            ),
        ))

    aggregate_timing = MP2Timing(
        ed_orbital_seconds=sum(
            fragment.timing.ed_orbital_seconds
            for fragment in fragment_results
        ),
        local_ri_lov_seconds=sum(
            fragment.timing.local_ri_lov_seconds
            for fragment in fragment_results
        ),
        weighted_ed_mp2_seconds=sum(
            fragment.timing.weighted_ed_mp2_seconds
            for fragment in fragment_results
        ),
        weak_bookkeeping_seconds=sum(
            fragment.timing.weak_bookkeeping_seconds
            for fragment in fragment_results
        ),
    )

    return MP2Result(
        e_corr=strong_total + weak_os,
        e_strong=strong_total,
        e_weak_multipole_os=weak_os,
        e_strong_os=strong_os,
        e_strong_ss=strong_ss,
        fragments=tuple(fragment_results),
        topology=topology,
        timing=aggregate_timing,
    )


def kernel(mf, **kwargs):
    """Build target/domain topology and evaluate the DLNO-MP2 energy."""
    topology = build_domain_topology(mf, **kwargs)
    return evaluate_domain_mp2(mf, topology)


@dataclass(frozen=True)
class FragmentDimensions:
    """Fixed dimensions of one strong extended-domain (ED) calculation.

    ``strong_fragments`` includes the target fragment itself.  The occupied
    and virtual ranks are the fixed ranks selected at the reference geometry;
    they are therefore also the actual array dimensions rebuilt during a
    fixed-topology energy or gradient evaluation.
    """

    fragment_index: int
    strong_fragments: tuple[int, ...]
    extended_atoms: tuple[int, ...]
    n_domain_atoms: int
    n_domain_ao: int
    n_domain_occ: int
    n_domain_vir: int


@dataclass(frozen=True)
class MP2TermResult:
    """Scalar result retained after one progressive local-MP2 pullback.

    A strong term is one weighted ED row and has ``right_fragment=None``.
    A weak term is one unordered distant fragment pair.  Only host scalars
    and integer labels are retained; no orbital frame, pullback, or AD tape is
    stored in this record.

    In the serial evaluator, frame construction is part of the term VJP and
    is consequently included in ``forward_seconds``/``reverse_seconds``.  In
    MPI, root builds and replays the common-to-frame map separately, and those
    times are reported in ``frame_build_seconds`` and
    ``frame_replay_seconds``.
    """

    kind: str
    left_fragment: int
    right_fragment: int | None
    energy: float
    forward_seconds: float
    reverse_seconds: float
    frame_build_seconds: float = 0.0
    frame_replay_seconds: float = 0.0
    worker_rank: int = 0


@dataclass(frozen=True)
class MP2GradientTiming:
    """Wall/work timing of a progressive local-MP2 correlation pullback.

    On one process all entries are ordinary wall seconds.  In MPI the strong
    and weak forward/reverse entries are sums of the worker term times and are
    thus *work seconds*, while ``total_seconds`` is the collective elapsed
    wall time (the maximum over ranks).  The common and frame fields are root
    wall seconds.  This distinction exposes both parallel load and critical
    elapsed time without pretending their sum is a serial wall clock.
    """

    common_forward_seconds: float = 0.0
    strong_forward_seconds: float = 0.0
    strong_reverse_seconds: float = 0.0
    weak_forward_seconds: float = 0.0
    weak_reverse_seconds: float = 0.0
    frame_build_seconds: float = 0.0
    frame_replay_seconds: float = 0.0
    common_reverse_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class MP2Decomposition:
    """Energy, topology, dimensions, and timing from one gradient run.

    ``n_strong_pairs`` and ``n_weak_pairs`` count unordered *interfragment*
    pairs and sum to ``n_fragments * (n_fragments - 1) // 2``.  The strong
    energy is evaluated as one weighted ED row per fragment, so
    ``n_strong_ed_terms`` is instead always ``n_fragments`` and includes the
    intrafragment contribution.  Reporting both counts avoids identifying ED
    rows with pair-list entries.
    """

    e_corr: float
    e_strong: float
    e_weak: float
    n_fragments: int
    n_strong_pairs: int
    n_weak_pairs: int
    n_strong_ed_terms: int
    n_weak_pair_terms: int
    fragments: tuple[FragmentDimensions, ...]
    terms: tuple[MP2TermResult, ...]
    timing: MP2GradientTiming


def _fragment_dimensions(static):
    """Return host-only ED dimensions encoded by fixed selections."""
    rows = []
    for fragment in static.fragments:
        extended_atoms = tuple(
            int(value) for value in numpy.asarray(fragment.extended_atoms)
        )
        rows.append(FragmentDimensions(
            fragment_index=int(fragment.fragment_index),
            strong_fragments=tuple(
                int(value)
                for value in numpy.asarray(fragment.strong_fragments)
            ),
            extended_atoms=extended_atoms,
            n_domain_atoms=len(extended_atoms),
            n_domain_ao=int(numpy.asarray(
                fragment.extended_ao_indices
            ).size),
            n_domain_occ=int(numpy.asarray(
                fragment.strong_occ_metric_keep
            ).size),
            n_domain_vir=int(numpy.asarray(
                fragment.strong_virtual.metric_keep
            ).size),
        ))
    return tuple(rows)


def _make_decomposition(static, energy, terms, timing):
    """Build a host-only diagnostic result from completed scalar terms."""
    terms = tuple(terms)
    strong_terms = tuple(term for term in terms if term.kind == "strong")
    weak_terms = tuple(term for term in terms if term.kind == "weak")
    nfragment = len(static.fragments)
    strong_mask = numpy.asarray(static.strong_mask, dtype=bool)
    n_strong_pairs = int(numpy.count_nonzero(numpy.triu(
        strong_mask, k=1
    )))
    n_weak_pairs = nfragment * (nfragment - 1) // 2 - n_strong_pairs
    if len(strong_terms) != nfragment or len(weak_terms) != n_weak_pairs:
        raise RuntimeError(
            "local-MP2 term records do not match the fixed pair topology"
        )
    return MP2Decomposition(
        # Preserve the evaluator's original mixed term-addition order for the
        # reported total rather than recomputing it as strong + weak.
        e_corr=float(jax.device_get(energy)),
        e_strong=float(sum(term.energy for term in strong_terms)),
        e_weak=float(sum(term.energy for term in weak_terms)),
        n_fragments=nfragment,
        n_strong_pairs=n_strong_pairs,
        n_weak_pairs=n_weak_pairs,
        n_strong_ed_terms=len(strong_terms),
        n_weak_pair_terms=len(weak_terms),
        fragments=_fragment_dimensions(static),
        terms=terms,
        timing=timing,
    )


def strong_domain_energy(mf, domain, static, fragment_index):
    """Evaluate one exact two-sided MP2 term in a supplied strong ED frame."""
    fragment_index = int(fragment_index)
    fragment = static.fragments[fragment_index]
    nocc = domain.occupied_coeff.shape[1]
    local_coeff = np.concatenate(
        (domain.occupied_coeff, domain.virtual_coeff), axis=1
    )
    lov = lno_df.get_local_Lov(
        mf,
        local_coeff,
        nocc,
        fragment.extended_atoms,
        integral_direct=True,
    )
    lov = np.reshape(
        lov, (-1, nocc, domain.virtual_coeff.shape[1])
    )
    return fragment_pair_energy_from_lov_jax(
        lov,
        domain.occupied_energy,
        domain.virtual_energy,
        domain.target_projection,
        domain.partner_weight,
        max_memory_mb=(
            256.0
            if static.thresholds.mp2_block_memory_mb is None
            else static.thresholds.mp2_block_memory_mb
        ),
    )


def strong_fragment_energy(mf, common, static, fragment_index):
    """Differentiate the exact two-sided MP2 energy in one fixed ED.

    The atom list and every retained rank/index come from ``static``.  The
    local occupied/virtual orbitals, their semicanonical energies, the IAO
    target/partner weights, and the local density-fitting factors are rebuilt
    from the current geometry.
    """
    domain = build_strong_ed_domain(common, static, fragment_index)
    return strong_domain_energy(
        mf, domain, static, fragment_index
    )


def _multipole_screen_arguments(screen, atoms):
    """Expand one shared weak PAO space into the per-mode API."""
    nmode = int(screen.occupied_coeff.shape[1])
    return (
        screen.occupied_energy,
        tuple(screen.occupied_coeff[:, index] for index in range(nmode)),
        tuple(screen.virtual_energy for _ in range(nmode)),
        tuple(screen.virtual_coeff for _ in range(nmode)),
        tuple(atoms for _ in range(nmode)),
    )


def _validate_weak_pair(static, left_index, right_index):
    left_index = int(left_index)
    right_index = int(right_index)
    if left_index == right_index:
        raise ValueError("a weak pair must contain two distinct fragments")
    if left_index > right_index:
        left_index, right_index = right_index, left_index
    nfragment = len(static.fragments)
    if not (0 <= left_index < nfragment and 0 <= right_index < nfragment):
        raise IndexError("weak-pair fragment index is out of range")
    if bool(static.strong_mask[left_index, right_index]):
        raise ValueError(
            f"fragment pair ({left_index}, {right_index}) is classified strong"
        )
    return left_index, right_index


def weak_screen_pair_energy(
    mf,
    left,
    right,
    static,
    left_index,
    right_index,
):
    """Evaluate one weak multipole pair in supplied root-gauge screens."""
    left_index, right_index = _validate_weak_pair(
        static, left_index, right_index
    )

    if left is None:
        raise RuntimeError(
            f"fragment {left_index} has fixed weak pairs but no "
            "multipole screen"
        )
    left_fragment = static.fragments[left_index]
    left_args = _multipole_screen_arguments(
        left, left_fragment.primary_atoms
    )

    if right is None:
        raise RuntimeError(
            f"fragment {right_index} has fixed weak pairs but no "
            "multipole screen"
        )
    right_fragment = static.fragments[right_index]
    right_args = _multipole_screen_arguments(
        right, right_fragment.primary_atoms
    )
    pair = multipole.pair_energy_multipole_cross(
        mf.mol,
        left_args[0],
        left_args[1],
        left_args[2],
        left_args[3],
        right_args[0],
        right_args[1],
        right_args[2],
        right_args[3],
        atmlst_left=left_args[4],
        atmlst_right=right_args[4],
        order=static.thresholds.multipole_order,
    )
    return np.real(np.sum(
        pair * left.weights[:, None] * right.weights[None, :]
    ))


def _weak_pair_energy(mf, common, static, left_index, right_index):
    """Build both screens and evaluate one unordered weak pair."""
    left_index, right_index = _validate_weak_pair(
        static, left_index, right_index
    )
    left = build_weak_multipole_screen(common, static, left_index)
    right = build_weak_multipole_screen(common, static, right_index)
    return weak_screen_pair_energy(
        mf,
        left,
        right,
        static,
        left_index,
        right_index,
    )


def _correlation_term_specs(static):
    """Enumerate every additive strong row and unordered weak pair once."""
    specs = []
    nfragment = len(static.fragments)
    strong_mask = numpy.asarray(static.strong_mask, dtype=bool)
    for fragment_index in range(nfragment):
        specs.append(("strong", fragment_index, -1))
        for partner_index in range(fragment_index + 1, nfragment):
            if not strong_mask[fragment_index, partner_index]:
                specs.append(("weak", fragment_index, partner_index))
    return tuple(specs)


def _correlation_term_energy(mf, common, static, spec):
    """Evaluate one additive term from :func:`_correlation_term_specs`."""
    kind, left_index, right_index = spec
    if kind == "strong":
        if int(right_index) != -1:
            raise ValueError("a strong-row term must use right_index=-1")
        return strong_fragment_energy(
            mf, common, static, int(left_index)
        ).total
    if kind == "weak":
        return _weak_pair_energy(
            mf,
            common,
            static,
            int(left_index),
            int(right_index),
        )
    raise ValueError(f"unknown IAO-MP2 correlation term kind {kind!r}")


def _contains_tracer(value):
    return any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def correlation_energy(mf, static, *, iao_coeff=None):
    """Forward-only local-MP2 energy with fixed ED and pair topology.

    This scalar form is retained for energy checks and finite differences.
    Applying an outer JAX transform would place every local term on one tape,
    so traced inputs are rejected.  Use :func:`correlation_value_and_grad` for
    derivatives; it pulls back each strong ED and unordered weak pair
    separately and releases their intermediates immediately.
    """
    if _contains_tracer((mf, iao_coeff)):
        raise TypeError(
            "correlation_energy is forward-only; use "
            "correlation_value_and_grad for automatic differentiation"
        )
    common = rebuild_domain_data(
        mf, static, iao_coeff=iao_coeff
    )
    energy = np.zeros((), dtype=common.s1e.dtype)
    for spec in _correlation_term_specs(static):
        energy = energy + _correlation_term_energy(
            mf, common, static, spec
        )
    return np.real(energy)


def _add_cotangent(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if hasattr(left, "dtype") and left.dtype == jax.dtypes.float0:
        return right
    if hasattr(right, "dtype") and right.dtype == jax.dtypes.float0:
        return left
    return left + right


def _restart_is_enabled(restart):
    return restart is not None and bool(getattr(restart, "enabled", False))


def _restart_is_resuming(restart):
    return _restart_is_enabled(restart) and bool(
        getattr(restart, "resume", False)
    )


def _term_result_metadata(term):
    return {
        "kind": str(term.kind),
        "left_fragment": int(term.left_fragment),
        "right_fragment": (
            None
            if term.right_fragment is None
            else int(term.right_fragment)
        ),
        "energy": float(term.energy),
        "forward_seconds": float(term.forward_seconds),
        "reverse_seconds": float(term.reverse_seconds),
        "frame_build_seconds": float(term.frame_build_seconds),
        "frame_replay_seconds": float(term.frame_replay_seconds),
        "worker_rank": int(term.worker_rank),
    }


def _term_result_from_metadata(row):
    return MP2TermResult(
        kind=str(row["kind"]),
        left_fragment=int(row["left_fragment"]),
        right_fragment=(
            None
            if row.get("right_fragment") is None
            else int(row["right_fragment"])
        ),
        energy=float(row["energy"]),
        forward_seconds=float(row["forward_seconds"]),
        reverse_seconds=float(row["reverse_seconds"]),
        frame_build_seconds=float(row.get("frame_build_seconds", 0.0)),
        frame_replay_seconds=float(row.get("frame_replay_seconds", 0.0)),
        worker_rank=int(row.get("worker_rank", 0)),
    )


def _timing_metadata(timing):
    return {
        name: float(getattr(timing, name))
        for name in MP2GradientTiming.__dataclass_fields__
    }


def _timing_from_metadata(row):
    return MP2GradientTiming(**{
        name: float(row.get(name, 0.0))
        for name in MP2GradientTiming.__dataclass_fields__
    })


def _details_restart_metadata(details):
    return {
        "term_results": [
            _term_result_metadata(term) for term in details.terms
        ],
        "timing": _timing_metadata(details.timing),
    }


def _details_from_restart_metadata(static, energy, metadata):
    terms = tuple(
        _term_result_from_metadata(row)
        for row in metadata.get("term_results", ())
    )
    timing = _timing_from_metadata(metadata.get("timing", {}))
    return _make_decomposition(static, energy, terms, timing)


def _zero_mf_cotangent(mf, dtype):
    """Return a live MF-shaped zero tree for restart deserialization."""

    _, pullback = jax.vjp(
        lambda mf_: np.zeros((), dtype=dtype), mf
    )
    mf_bar, = pullback(np.ones((), dtype=dtype))
    return mf_bar


def _zero_term_cotangents(mf, common):
    """Return live zero trees matching one MP2 term's two inputs."""

    _, pullback = jax.vjp(
        lambda mf_, common_: np.zeros((), dtype=common_.s1e.dtype),
        mf,
        common,
    )
    return pullback(np.ones((), dtype=common.s1e.dtype))


def _progressive_correlation_pullback(
    mf, common, static, *, return_details=False, restart=None
):
    """Return correlation energy plus cotangents of ``mf`` and ``common``.

    Strong ED energies and unordered weak pairs are pulled back as separate
    scalar terms.  In particular, do not place every weak partner of one
    fragment on the same reverse-mode tape: order-four AO multipole moments
    are large, and that grouped tape grows linearly with the number of weak
    partners even though the pair energies are independent.  Immediate
    pullback and collection keep the peak tied to one ED or one weak pair.
    """
    work = _correlation_term_specs(static)
    energy = np.zeros((), dtype=common.s1e.dtype)
    mf_bar = None
    common_bar = None
    term_results = []
    completed_count = 0
    previous_elapsed = 0.0
    total_start = time.perf_counter() if return_details else None

    if _restart_is_resuming(restart):
        zero_mf_bar, zero_common_bar = _zero_term_cotangents(mf, common)
        progress = restart.load_record(
            "correlation_progress",
            templates={
                "mf_bar": zero_mf_bar,
                "common_bar": zero_common_bar,
            },
            missing_ok=True,
        )
        if progress is not None:
            completed_count = int(progress.scalars["completed_count"])
            if completed_count < 0 or completed_count > len(work):
                raise RuntimeError(
                    "serial MP2 restart completed-term count is invalid"
                )
            saved_specs = tuple(
                (str(row[0]), int(row[1]), int(row[2]))
                for row in progress.metadata.get("completed_specs", ())
            )
            if saved_specs != work[:completed_count]:
                raise RuntimeError(
                    "serial MP2 restart term prefix does not match the "
                    "fixed topology"
                )
            energy = np.asarray(
                progress.scalars["energy"], dtype=common.s1e.dtype
            )
            mf_bar = progress.trees["mf_bar"]
            common_bar = progress.trees["common_bar"]
            previous_elapsed = float(
                progress.scalars.get("elapsed_seconds", 0.0)
            )
            if return_details:
                term_results = [
                    _term_result_from_metadata(row)
                    for row in progress.metadata.get("term_results", ())
                ]
                if len(term_results) != completed_count:
                    raise RuntimeError(
                        "serial MP2 restart term diagnostics are incomplete"
                    )
        del zero_mf_bar, zero_common_bar

    def accumulate_term(term, spec, term_index):
        nonlocal energy, mf_bar, common_bar
        forward_start = time.perf_counter() if return_details else None
        term_energy, pullback = jax.vjp(term, mf, common)
        if return_details:
            jax.block_until_ready(term_energy)
            forward_seconds = time.perf_counter() - forward_start
            reverse_start = time.perf_counter()
        term_mf_bar, term_common_bar = pullback(
            np.ones((), dtype=term_energy.dtype)
        )
        energy = energy + term_energy
        if mf_bar is None:
            mf_bar = term_mf_bar
            common_bar = term_common_bar
        else:
            mf_bar = jax.tree_util.tree_map(
                _add_cotangent, mf_bar, term_mf_bar
            )
            common_bar = jax.tree_util.tree_map(
                _add_cotangent, common_bar, term_common_bar
            )
        jax.block_until_ready((energy, mf_bar, common_bar))
        if return_details:
            reverse_seconds = time.perf_counter() - reverse_start
            kind, left_index, right_index = spec
            term_results.append(MP2TermResult(
                kind=str(kind),
                left_fragment=int(left_index),
                right_fragment=(
                    None if int(right_index) == -1 else int(right_index)
                ),
                energy=float(jax.device_get(term_energy)),
                forward_seconds=float(forward_seconds),
                reverse_seconds=float(reverse_seconds),
            ))
        if _restart_is_enabled(restart):
            elapsed_seconds = previous_elapsed
            if return_details:
                elapsed_seconds += time.perf_counter() - total_start
            restart.save_record(
                "correlation_progress",
                scalars={
                    "energy": float(jax.device_get(energy)),
                    "completed_count": int(term_index + 1),
                    "elapsed_seconds": float(elapsed_seconds),
                },
                trees={
                    "mf_bar": mf_bar,
                    "common_bar": common_bar,
                },
                metadata={
                    "completed_specs": [
                        [str(kind), int(left), int(right)]
                        for kind, left, right in work[:term_index + 1]
                    ],
                    "term_results": [
                        _term_result_metadata(item) for item in term_results
                    ] if return_details else [],
                },
            )
        del pullback, term_mf_bar, term_common_bar
        # Saved JAX pullbacks contain reference cycles.  Waiting for the
        # generational collector lets several independent multipole tapes
        # coexist and leaves large native allocator arenas at their high-water
        # mark.  Collection here is cheap relative to an order-four pair VJP.
        gc.collect()

    for term_index, spec in enumerate(
        work[completed_count:], start=completed_count
    ):
        def term(mf_, common_, _spec=spec):
            return _correlation_term_energy(
                mf_, common_, static, _spec
            )

        accumulate_term(term, spec, term_index)
    if mf_bar is None:
        raise ValueError("fixed topology must contain at least one fragment")
    if return_details:
        strong_results = tuple(
            item for item in term_results if item.kind == "strong"
        )
        weak_results = tuple(
            item for item in term_results if item.kind == "weak"
        )
        timing = MP2GradientTiming(
            strong_forward_seconds=sum(
                item.forward_seconds for item in strong_results
            ),
            strong_reverse_seconds=sum(
                item.reverse_seconds for item in strong_results
            ),
            weak_forward_seconds=sum(
                item.forward_seconds for item in weak_results
            ),
            weak_reverse_seconds=sum(
                item.reverse_seconds for item in weak_results
            ),
            total_seconds=(
                previous_elapsed + time.perf_counter() - total_start
            ),
        )
        details = _make_decomposition(
            static, energy, term_results, timing
        )
        return energy, mf_bar, common_bar, details
    return energy, mf_bar, common_bar


def correlation_value_and_grad_from_common(
    mf, common, static, *, return_details=False, restart=None
):
    """Return the local-MP2 energy and open ``mf``/``common`` cotangents.

    This is the shared-localization seam used by DLNO-CCSD(T).  The caller
    owns the VJP that built ``common`` and can therefore accumulate MP2,
    LIS, and coupled-cluster cotangents in the same IAO gauge before closing
    that VJP exactly once.  Each strong ED and unordered weak pair is still
    pulled back separately, so exposing this lower boundary does not recreate
    a grouped reverse-mode tape.
    """
    result = _progressive_correlation_pullback(
        mf,
        common,
        static,
        return_details=return_details,
        restart=restart,
    )
    energy, mf_bar, common_bar = result[:3]
    jax.block_until_ready((energy, mf_bar, common_bar))
    if return_details:
        return energy, mf_bar, common_bar, result[3]
    return energy, mf_bar, common_bar


def correlation_value_and_grad(
    mf, static, *, return_details=False, restart=None
):
    """Return ``(E_corr, mf_bar)`` with progressive fixed-topology AD.

    IAOs and other common continuous arrays are rebuilt once under a saved
    VJP.  Each fragment's strong and weak energies are then evaluated and
    pulled back immediately.  This is the reusable layer for a DLNO-CCSD(T)
    PT correction: the caller can add ``mf_bar`` to its existing SCF
    cotangent and perform only one final CPHF/SCF pullback.  That saved SCF
    pullback must use the standard implicit backend.  The experimental
    first-order replay backend is not valid for this general, nonstationary
    local-orbital cotangent.
    """
    if _restart_is_resuming(restart):
        zero_mf_bar = _zero_mf_cotangent(mf, np.asarray(mf.e_tot).dtype)
        closed = restart.load_record(
            "correlation_closed",
            templates={"mf_bar": zero_mf_bar},
            missing_ok=True,
        )
        del zero_mf_bar
        if closed is not None:
            energy = np.asarray(
                closed.scalars["energy"], dtype=np.asarray(mf.e_tot).dtype
            )
            mf_bar = closed.trees["mf_bar"]
            jax.block_until_ready((energy, mf_bar))
            if return_details:
                details = _details_from_restart_metadata(
                    static, energy, closed.metadata
                )
                return energy, mf_bar, details
            return energy, mf_bar

    collect_details = return_details or _restart_is_enabled(restart)
    common_start = time.perf_counter() if collect_details else None
    common, common_pullback = jax.vjp(
        lambda mf_: rebuild_domain_data(mf_, static), mf
    )
    if collect_details:
        jax.block_until_ready(common)
        common_forward_seconds = time.perf_counter() - common_start
    result = correlation_value_and_grad_from_common(
        mf,
        common,
        static,
        return_details=collect_details,
        restart=restart,
    )
    energy, mf_bar, common_bar = result[:3]
    common_reverse_start = time.perf_counter() if collect_details else None
    common_mf_bar, = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, common_mf_bar
    )
    jax.block_until_ready((energy, mf_bar))
    if collect_details:
        common_reverse_seconds = (
            time.perf_counter() - common_reverse_start
        )
        details = result[3]
        details = replace(
            details,
            timing=replace(
                details.timing,
                common_forward_seconds=common_forward_seconds,
                common_reverse_seconds=common_reverse_seconds,
                total_seconds=(
                    details.timing.total_seconds
                    + common_forward_seconds
                    + common_reverse_seconds
                ),
            ),
        )
        if _restart_is_enabled(restart):
            restart.save_record(
                "correlation_closed",
                scalars={"energy": float(jax.device_get(energy))},
                trees={"mf_bar": mf_bar},
                metadata=_details_restart_metadata(details),
            )
        if return_details:
            return energy, mf_bar, details
    return energy, mf_bar


def correlation_value_and_grad_with_targets(
    mf, iao_coeff, static, *, return_details=False
):
    """Return ``(E_corr, mf_bar, iao_bar)`` for an externally built IAO.

    This variant lets a DLNO driver reuse its existing IAO/LO transform and
    accumulate the local-MP2 cotangent into the same saved localization VJP.
    As for :func:`correlation_value_and_grad`, the final SCF response must use
    the standard implicit backend rather than the experimental replay path.
    """
    total_start = time.perf_counter() if return_details else None
    common_start = time.perf_counter() if return_details else None
    common, common_pullback = jax.vjp(
        lambda mf_, iao_: rebuild_domain_data(
            mf_, static, iao_coeff=iao_
        ),
        mf,
        iao_coeff,
    )
    if return_details:
        jax.block_until_ready(common)
        common_forward_seconds = time.perf_counter() - common_start
    result = _progressive_correlation_pullback(
        mf, common, static, return_details=return_details
    )
    energy, mf_bar, common_bar = result[:3]
    common_reverse_start = time.perf_counter() if return_details else None
    common_mf_bar, iao_bar = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, common_mf_bar
    )
    jax.block_until_ready((energy, mf_bar, iao_bar))
    if return_details:
        details = result[3]
        details = replace(
            details,
            timing=replace(
                details.timing,
                common_forward_seconds=common_forward_seconds,
                common_reverse_seconds=(
                    time.perf_counter() - common_reverse_start
                ),
                total_seconds=time.perf_counter() - total_start,
            ),
        )
        return energy, mf_bar, iao_bar, details
    return energy, mf_bar, iao_bar
