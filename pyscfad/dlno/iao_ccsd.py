"""IAO-DLNO-CCSD(T) with MP2-selected local interacting spaces.

The correlation-energy bookkeeping implemented here is deliberately unique::

    sum_F [E_CCSD(T),F(LIS_F) - E_MP2,F(LIS_F)]
      + E_IAO-DLNO-MP2(strong ED + weak multipole).

The MP2 subtraction is the conventional full-spin MP2 fragment energy in the
same LIS used by CCSD(T).  There is no SOS variant and no domain/full
correction switch.  The molecule-wide IAO-DLNO-MP2 term is evaluated once.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp

from pyscfad import config_update
from pyscfad.lno.ccsd import _impurity_solve_core
from pyscfad.ops import stop_trace

from ._output import (
    emit_lines,
    energy_summary_lines,
    fragment_energy_lines,
    lis_active_space_lines,
    lis_dimensions_from_static,
    local_correlation_settings_lines,
    mp2_prescreened_domain_lines,
    nuclear_force_lines,
)
from .iao_lis import (
    IAO_LIS_INTERNAL_RANK_THRESHOLD,
    IAOFragmentLISStaticSelections,
    build_fragment_lis,
    build_iao_lis_static_selections,
)
from .iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from .iao_mp2_grad import (
    build_iao_mp2_static_selections,
    correlation_value_and_grad_from_common,
    rebuild_iao_mp2_common,
)
from .iao_mp2_grad import correlation_energy as iao_mp2_correlation_energy


__all__ = [
    "IAODLNOCCSDResult",
    "build_iao_dlno_ccsd_domain_selections",
    "build_iao_dlno_ccsd_static_selections",
    "kernel",
    "value_and_grad",
]


@dataclass(frozen=True)
class IAODLNOCCSDResult:
    """Energy-only IAO-DLNO-CCSD(T) result and its additive components."""

    e_hf: object
    e_corr: object
    e_total: object
    e_iao_mp2: object
    e_mp2_lis: object
    e_ccsd: object
    e_ccsd_t: object
    lis_occupied: tuple[int, ...]
    lis_virtual: tuple[int, ...]


@dataclass(frozen=True)
class _FragmentValueAndGradResult:
    """One complete LIS/CC fragment primal and local pullback."""

    value: object
    e_mp2_lis: object
    e_ccsd: object
    e_ccsd_t: object
    lis_occupied: int
    lis_virtual: int
    mf_bar: object
    common_bar: object


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


def _assemble_iao_dlno_correlation(
    e_cc_fragments,
    e_t_fragments,
    e_mp2_lis_fragments,
    e_iao_mp2,
):
    """Assemble the sole supported IAO-DLNO-CCSD(T) correction formula."""
    e_cc_fragments = jnp.asarray(e_cc_fragments)
    e_t_fragments = jnp.asarray(e_t_fragments)
    e_mp2_lis_fragments = jnp.asarray(e_mp2_lis_fragments)
    return (
        jnp.sum(e_cc_fragments)
        + jnp.sum(e_t_fragments)
        - jnp.sum(e_mp2_lis_fragments)
        + jnp.asarray(e_iao_mp2)
    )


def _validate_solver_options(*, ccsd_t, dcsd):
    if bool(ccsd_t) and bool(dcsd):
        raise ValueError("perturbative triples are not defined for DCSD")


def _zero_fragment_terms(common):
    zero = jnp.zeros((), dtype=common.s1e.dtype)
    return zero, zero, zero


def _solve_fragment(
    mf,
    common,
    static_selections,
    fragment_index,
    metadata,
    *,
    verbose_imp,
    ccsd_t,
    dcsd,
):
    """Build one LIS and solve it, treating an empty excitation space as zero."""

    lis = build_fragment_lis(
        mf, common, static_selections, fragment_index
    )
    if (
        lis.active_occupied_coeff.shape[1] == 0
        or lis.active_virtual_coeff.shape[1] == 0
    ):
        return _zero_fragment_terms(common), lis
    values = _impurity_solve_core(
        mf,
        lis.mo_coeff,
        lis.fragment_iao_coeff,
        common.fock,
        common.s1e,
        frozen=lis.frozen,
        frag_prescreen=metadata,
        verbose_imp=verbose_imp,
        ccsd_t=ccsd_t,
        dcsd=dcsd,
        profile_info=None,
        profile_pass=None,
    )
    return values, lis


def build_iao_dlno_ccsd_domain_selections(
    mf,
    *,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
):
    """Build the fixed IAO-MP2 fragment and ED-domain topology."""

    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds()
    reference = build_iao_fragment_topology(
        mf,
        frozen=frozen,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        thresholds=thresholds,
        pair_energy_model=pair_energy_model,
        force_full_domains=force_full_domains,
    )
    return build_iao_mp2_static_selections(mf, reference)


def build_iao_dlno_ccsd_static_selections(
    mf,
    *,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
):
    """Build fixed IAO-MP2 topology and fixed LIS rank selections."""
    mp2_static = build_iao_dlno_ccsd_domain_selections(
        mf,
        frag_lolist=frag_lolist,
        frag_atmlist=frag_atmlist,
        frozen=frozen,
        thresholds=thresholds,
        pair_energy_model=pair_energy_model,
        force_full_domains=force_full_domains,
    )
    common = rebuild_iao_mp2_common(mf, mp2_static)
    return build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=thresh_occ,
        thresh_vir=thresh_vir,
        internal_rank_threshold=internal_rank_threshold,
    )


def _fragment_domain_metadata(lis_static, fragment_index):
    fragment = lis_static.mp2_static.fragments[int(fragment_index)]
    return {
        "fragment_index": int(fragment_index),
        "extended_primary_domain": onp.asarray(
            fragment.extended_atoms, dtype=onp.int32
        ),
    }


def _fragment_value_and_grad(
    mf,
    common,
    static_selections,
    fragment_index,
    *,
    verbose_imp,
    ccsd_t,
    dcsd,
):
    """Differentiate one entire fragment in its locally constructed LIS.

    The LIS build, CCSD(T) solve, and matching MP2 subtraction are enclosed in
    one VJP.  Consequently every LIS-frame cotangent is closed into the shared
    ``(mf, common)`` coordinates before this helper returns.  This is also the
    safe unit of work for MPI fragment parallelism: no gauge-dependent LIS
    coefficient or cotangent needs to cross a rank boundary.
    """
    fragment_index = int(fragment_index)
    metadata = _fragment_domain_metadata(static_selections, fragment_index)

    def fragment_term(mf_, common_):
        # Raw IAOs retain any internal valence-virtual component, while their
        # occupied overlap produces precisely the additive W_F.
        (e_mp2_lis, e_cc, e_t), lis = _solve_fragment(
            mf_,
            common_,
            static_selections,
            fragment_index,
            metadata,
            verbose_imp=verbose_imp,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
        )
        value = e_cc + e_t - e_mp2_lis
        auxiliary = (
            e_mp2_lis,
            e_cc,
            e_t,
            lis.active_occupied_coeff.shape[1],
            lis.active_virtual_coeff.shape[1],
        )
        return value, auxiliary

    fragment_value, fragment_pullback, auxiliary = jax.vjp(
        fragment_term, mf, common, has_aux=True
    )
    fragment_mf_bar, fragment_common_bar = fragment_pullback(
        jnp.ones((), dtype=jnp.asarray(fragment_value).dtype)
    )
    e_mp2_lis, e_cc, e_t, lis_occupied, lis_virtual = auxiliary
    jax.block_until_ready(
        (fragment_value, fragment_mf_bar, fragment_common_bar)
    )
    result = _FragmentValueAndGradResult(
        value=fragment_value,
        e_mp2_lis=e_mp2_lis,
        e_ccsd=e_cc,
        e_ccsd_t=e_t,
        lis_occupied=int(lis_occupied),
        lis_virtual=int(lis_virtual),
        mf_bar=fragment_mf_bar,
        common_bar=fragment_common_bar,
    )
    del fragment_pullback, fragment_term, auxiliary
    return result


def kernel(
    mf,
    *,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
    ccsd_t=False,
    dcsd=False,
    verbose_imp=0,
    static_selections=None,
):
    """Evaluate the new IAO-DLNO-CCSD(T) energy without an SCF pullback."""
    _validate_solver_options(ccsd_t=ccsd_t, dcsd=dcsd)
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds()
    if static_selections is None:
        static_selections = build_iao_dlno_ccsd_static_selections(
            mf,
            frag_lolist=frag_lolist,
            frag_atmlist=frag_atmlist,
            frozen=frozen,
            thresholds=thresholds,
            pair_energy_model=pair_energy_model,
            force_full_domains=force_full_domains,
            thresh_occ=thresh_occ,
            thresh_vir=thresh_vir,
            internal_rank_threshold=internal_rank_threshold,
        )
    elif not isinstance(static_selections, IAOFragmentLISStaticSelections):
        raise TypeError(
            "static_selections must be IAOFragmentLISStaticSelections"
        )

    mp2_static = static_selections.mp2_static
    common = rebuild_iao_mp2_common(mf, mp2_static)
    e_mp2_lis_terms = []
    e_cc_terms = []
    e_t_terms = []
    lis_occ = []
    lis_vir = []
    for fragment_index in range(len(static_selections.fragments)):
        metadata = _fragment_domain_metadata(
            static_selections, fragment_index
        )
        (e_mp2_lis, e_cc, e_t), lis = _solve_fragment(
            mf,
            common,
            static_selections,
            fragment_index,
            metadata,
            verbose_imp=verbose_imp,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
        )
        e_mp2_lis_terms.append(e_mp2_lis)
        e_cc_terms.append(e_cc)
        e_t_terms.append(e_t)
        lis_occ.append(int(lis.active_occupied_coeff.shape[1]))
        lis_vir.append(int(lis.active_virtual_coeff.shape[1]))

    e_iao_mp2 = iao_mp2_correlation_energy(mf, mp2_static)
    e_corr = _assemble_iao_dlno_correlation(
        jnp.stack(e_cc_terms),
        jnp.stack(e_t_terms),
        jnp.stack(e_mp2_lis_terms),
        e_iao_mp2,
    )
    e_hf = mf.e_tot
    return IAODLNOCCSDResult(
        e_hf=e_hf,
        e_corr=e_corr,
        e_total=e_hf + e_corr,
        e_iao_mp2=e_iao_mp2,
        e_mp2_lis=jnp.sum(jnp.stack(e_mp2_lis_terms)),
        e_ccsd=jnp.sum(jnp.stack(e_cc_terms)),
        e_ccsd_t=jnp.sum(jnp.stack(e_t_terms)),
        lis_occupied=tuple(lis_occ),
        lis_virtual=tuple(lis_vir),
    )


def value_and_grad(
    mol,
    *,
    build_mf,
    frag_lolist=None,
    frag_atmlist=None,
    frozen=None,
    thresholds=None,
    pair_energy_model="multipole",
    force_full_domains=False,
    thresh_occ=1e-4,
    thresh_vir=1e-5,
    internal_rank_threshold=IAO_LIS_INTERNAL_RANK_THRESHOLD,
    ccsd_t=False,
    dcsd=False,
    verbose_imp=0,
    static_selections=None,
    progress=False,
):
    """Return the IAO-DLNO-CCSD(T) total energy and molecular gradient.

    Discrete atom domains, pair classes, PAO/rank decisions, and LIS ranks are
    fixed at the reference geometry.  All numerical IAO, ED, MP2-density,
    LIS, CCSD(T), weak-multipole, and integral quantities remain inside AD.

    The shared IAO/common frame is built once under a saved VJP.  Every CC
    fragment, strong MP2 row, and unordered weak pair is pulled back
    immediately; their common-frame cotangents are accumulated and the common
    VJP is closed once.  The resulting mean-field cotangent is finally passed
    through one standard implicit SCF response.

    Set ``progress=True`` to print flushed, structured summaries independently
    of PySC verbosity, or pass a callable that receives each formatted line.
    """
    _validate_solver_options(ccsd_t=ccsd_t, dcsd=dcsd)
    if (
        getattr(mol, "exp", None) is not None
        or getattr(mol, "ctr_coeff", None) is not None
    ):
        raise NotImplementedError(
            "IAO-DLNO-CCSD(T) currently differentiates nuclear coordinates "
            "only; build mol with trace_exp=False and trace_ctr_coeff=False"
        )
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds()

    verbose = int(getattr(mol, "verbose", 0))
    if not callable(progress) and not isinstance(progress, (bool, onp.bool_)):
        raise TypeError("progress must be a bool or callable")
    if callable(progress):
        progress_sink = progress
    elif bool(progress) or verbose >= 4:
        progress_sink = lambda message: print(message, flush=True)
    else:
        progress_sink = None
    reporter = (
        None
        if progress_sink is None
        else lambda message: progress_sink(f"[IAO-CC] {message}")
    )
    log = reporter if reporter is not None else (lambda message: None)
    method_label = "DCSD" if dcsd else ("CCSD(T)" if ccsd_t else "CCSD")
    started = time.perf_counter()
    log(f"IAO-DLNO-{method_label}: start")

    # The local-orbital cotangent is nonstationary.  It must be closed with
    # the converged fixed-point response, never the experimental finite SCF
    # replay backend.
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        mf, scf_pullback = jax.vjp(build_mf, mol)

    if static_selections is None:
        def static_builder(mf_):
            return build_iao_dlno_ccsd_static_selections(
                mf_,
                frag_lolist=frag_lolist,
                frag_atmlist=frag_atmlist,
                frozen=frozen,
                thresholds=thresholds,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
                thresh_occ=thresh_occ,
                thresh_vir=thresh_vir,
                internal_rank_threshold=internal_rank_threshold,
            )

        static_selections = stop_trace(static_builder)(mf)
    elif not isinstance(static_selections, IAOFragmentLISStaticSelections):
        raise TypeError(
            "static_selections must be IAOFragmentLISStaticSelections"
        )

    mp2_static = static_selections.mp2_static
    common, common_pullback = jax.vjp(
        lambda mf_: rebuild_iao_mp2_common(mf_, mp2_static), mf
    )
    if reporter is not None:
        emit_lines(
            reporter,
            local_correlation_settings_lines(
                static_selections,
                ccsd_t=ccsd_t,
                dcsd=dcsd,
                nproc=1,
                pair_energy_model=pair_energy_model,
                force_full_domains=force_full_domains,
            ),
        )
        emit_lines(
            reporter, mp2_prescreened_domain_lines(static_selections)
        )
        fixed_lis_occ, fixed_lis_vir = lis_dimensions_from_static(
            static_selections
        )
        emit_lines(
            reporter,
            lis_active_space_lines(
                static_selections, fixed_lis_occ, fixed_lis_vir
            ),
        )

    e_hf, hf_pullback = jax.vjp(lambda mf_: mf_.e_tot, mf)
    mf_bar, = hf_pullback(jnp.ones((), dtype=jnp.asarray(e_hf).dtype))
    common_bar = jax.tree_util.tree_map(jnp.zeros_like, common)

    e_cc_terms = []
    e_t_terms = []
    e_mp2_lis_terms = []
    fragment_records = []
    nfragment = len(static_selections.fragments)
    for fragment_index in range(nfragment):
        fragment_started = (
            time.perf_counter() if reporter is not None else None
        )
        fragment = _fragment_value_and_grad(
            mf,
            common,
            static_selections,
            fragment_index,
            verbose_imp=verbose_imp,
            ccsd_t=ccsd_t,
            dcsd=dcsd,
        )
        mf_bar = jax.tree_util.tree_map(
            _add_cotangent, mf_bar, fragment.mf_bar
        )
        common_bar = jax.tree_util.tree_map(
            _add_cotangent, common_bar, fragment.common_bar
        )
        e_mp2_lis_terms.append(fragment.e_mp2_lis)
        e_cc_terms.append(fragment.e_ccsd)
        e_t_terms.append(fragment.e_ccsd_t)
        jax.block_until_ready((mf_bar, common_bar))
        if reporter is not None:
            fragment_wall = time.perf_counter() - fragment_started
            fragment_records.append({
                "fragment_index": fragment_index,
                "worker_rank": None,
                "e_mp2_lis": float(fragment.e_mp2_lis),
                "e_ccsd": float(fragment.e_ccsd),
                "e_ccsd_t": float(fragment.e_ccsd_t),
                "wall_seconds": fragment_wall,
            })
            log(
                f"fragment {fragment_index + 1}/{nfragment} complete: "
                f"solver LIS occ/vir={fragment.lis_occupied}/"
                f"{fragment.lis_virtual}; wall={fragment_wall:.1f} s"
            )
        del fragment
        gc.collect()

    e_iao_mp2, mp2_mf_bar, mp2_common_bar = (
        correlation_value_and_grad_from_common(
            mf, common, mp2_static
        )
    )
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, mp2_mf_bar
    )
    common_bar = jax.tree_util.tree_map(
        _add_cotangent, common_bar, mp2_common_bar
    )

    common_mf_bar, = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(
        _add_cotangent, mf_bar, common_mf_bar
    )
    mol_bar, = scf_pullback(mf_bar)

    correlation = _assemble_iao_dlno_correlation(
        jnp.stack(e_cc_terms),
        jnp.stack(e_t_terms),
        jnp.stack(e_mp2_lis_terms),
        e_iao_mp2,
    )
    energy = e_hf + correlation
    jax.block_until_ready((energy, mol_bar))
    log(
        f"IAO-DLNO-{method_label}: done in "
        f"{time.perf_counter() - started:.2f} s, "
        f"E={float(energy):.12f}"
    )
    if reporter is not None:
        emit_lines(
            reporter,
            fragment_energy_lines(
                fragment_records,
                correlated_method="DCSD" if dcsd else "CCSD",
                include_triples=ccsd_t,
            ),
        )
        e_mp2_lis_total = jnp.sum(jnp.stack(e_mp2_lis_terms))
        e_ccsd_total = jnp.sum(jnp.stack(e_cc_terms))
        e_ccsd_t_total = jnp.sum(jnp.stack(e_t_terms))
        emit_lines(
            reporter,
            energy_summary_lines(
                e_hf=e_hf,
                e_iao_mp2=e_iao_mp2,
                e_mp2_lis=e_mp2_lis_total,
                e_ccsd=e_ccsd_total,
                e_ccsd_t=e_ccsd_t_total,
                e_corr=correlation,
                e_total=energy,
                correlated_method="DCSD" if dcsd else "CCSD",
                include_triples=ccsd_t,
            ),
        )
        emit_lines(
            reporter,
            nuclear_force_lines(mol, onp.asarray(mol_bar.coords)),
        )
    return energy, mol_bar
