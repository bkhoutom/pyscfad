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
import hashlib
import time
from dataclasses import asdict, dataclass, is_dataclass

import jax
import jax.numpy as jnp
import numpy as onp

from pyscfad import config_update
from pyscfad.lno.ccsd import _impurity_solve_core
from pyscfad.ops import stop_trace

from ._restart import RestartManager, df_source_fingerprint
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
    _fix_restart_mo_phases,
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
    # A few PySCFAD primitives represent a scalar parameter as a length-one
    # array in their local VJP.  Letting NumPy broadcasting choose the result
    # shape would turn an initially scalar live cotangent into shape ``(1,)``;
    # that cannot later be deserialized against the scalar MF template.  The
    # two objects contain the same single degree of freedom, so retain the
    # accumulator's (primal-compatible) shape explicitly.  No higher-rank
    # reinterpretation is valid: equal element counts do not make two
    # cotangent spaces interchangeable.
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape != right_shape:
        left_size = getattr(left, "size", None)
        right_size = getattr(right, "size", None)
        if left_size == right_size == 1:
            right = jnp.reshape(right, left_shape)
        else:
            raise ValueError(
                "incompatible cotangent leaf shapes: "
                f"{left_shape} and {right_shape}"
            )
    return left + right


def _restart_jsonable(value):
    """Convert scientific settings to a deterministic JSON-compatible tree."""

    if is_dataclass(value):
        return _restart_jsonable(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _restart_jsonable(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_restart_jsonable(item) for item in value]
    if isinstance(value, onp.ndarray):
        return value.tolist()
    if isinstance(value, onp.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        array = onp.asarray(jax.device_get(value))
    except Exception:
        return repr(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


def _restart_array_digest(value, *, decimals=10):
    array = onp.ascontiguousarray(onp.asarray(jax.device_get(value)))
    # A restarted SCF seeded from its chkfile may finish one cleanup cycle a
    # few ulps away from the original orbitals.  Quantize only the canonical
    # SCF fingerprint (not coordinates or saved cotangents) so that harmless
    # 1e-13 noise is accepted while sign changes and occupied/virtual gauge
    # rotations remain decisively incompatible.
    if array.dtype.kind in "fc":
        array = onp.ascontiguousarray(onp.round(array, decimals=decimals))
        array = onp.ascontiguousarray(
            onp.where(array == 0, onp.zeros_like(array), array)
        )
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(onp.uint8))
    return digest.hexdigest()


def _restart_scientific_payload(
    mol,
    mf,
    *,
    frag_lolist,
    frag_atmlist,
    frozen,
    thresholds,
    pair_energy_model,
    force_full_domains,
    thresh_occ,
    thresh_vir,
    internal_rank_threshold,
    ccsd_t,
    dcsd,
):
    """Return the compatibility identity for a serial gradient restart."""

    threshold_payload = asdict(thresholds)
    with_df = getattr(mf, "with_df", None)
    auxmol = None if with_df is None else getattr(with_df, "auxmol", None)
    return {
        "driver_schema": "iao-dlno-gradient-v1",
        "molecule": {
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "nelectron": int(mol.nelectron),
            "natm": int(mol.natm),
            "nao": int(mol.nao),
            "atom_charges": _restart_jsonable(mol.atom_charges()),
            "atom_coords_bohr": _restart_jsonable(
                mol.atom_coords(unit="Bohr")
            ),
            "basis": _restart_jsonable(getattr(mol, "_basis", mol.basis)),
            "ecp": _restart_jsonable(getattr(mol, "_ecp", None)),
            "pseudo": _restart_jsonable(getattr(mol, "_pseudo", None)),
            "cart": bool(getattr(mol, "cart", False)),
            "nucmod": _restart_jsonable(getattr(mol, "nucmod", None)),
        },
        "reference_scf": {
            "class": f"{type(mf).__module__}.{type(mf).__qualname__}",
            "mo_coeff_sha256": _restart_array_digest(mf.mo_coeff),
            "mo_energy_sha256": _restart_array_digest(mf.mo_energy),
            "mo_occ_sha256": _restart_array_digest(mf.mo_occ),
            "e_tot_rounded": round(float(jax.device_get(mf.e_tot)), 10),
            "auxbasis": _restart_jsonable(
                None if with_df is None else getattr(with_df, "auxbasis", None)
            ),
            "auxmol_basis": _restart_jsonable(
                None if auxmol is None else getattr(auxmol, "_basis", None)
            ),
            "df_source": df_source_fingerprint(mf),
        },
        "settings": {
            "frag_lolist": _restart_jsonable(frag_lolist),
            "frag_atmlist": _restart_jsonable(frag_atmlist),
            "frozen": _restart_jsonable(frozen),
            "thresholds": _restart_jsonable(threshold_payload),
            "pair_energy_model": str(pair_energy_model),
            "force_full_domains": bool(force_full_domains),
            "thresh_occ": float(thresh_occ),
            "thresh_vir": float(thresh_vir),
            "internal_rank_threshold": float(internal_rank_threshold),
            "ccsd_t": bool(ccsd_t),
            "dcsd": bool(dcsd),
        },
    }


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
    profile_pass=None,
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
        profile_pass=profile_pass,
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
    forward_record=None,
    save_forward=None,
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

    replay_forward = forward_record is not None

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
            profile_pass=(
                "backward replay" if replay_forward else None
            ),
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
    e_mp2_lis, e_cc, e_t, lis_occupied, lis_virtual = auxiliary
    jax.block_until_ready((fragment_value, e_mp2_lis, e_cc, e_t))

    # A JAX pullback closure is process-local and cannot be serialized.  The
    # useful checkpoint immediately after the primal is therefore deliberately
    # tiny: component energies and fixed dimensions only.  On restart we
    # rebuild the CC tape, but ask the custom triples primitive for its lazy
    # ``backward replay`` primal.  That skips the expensive real (T) energy
    # contraction while retaining the exact factor-direct triples pullback.
    if save_forward is not None and not replay_forward:
        save_forward({
            "fragment_index": fragment_index,
            "e_mp2_lis": float(jax.device_get(e_mp2_lis)),
            "e_ccsd": float(jax.device_get(e_cc)),
            "e_ccsd_t": float(jax.device_get(e_t)),
            "lis_occupied": int(lis_occupied),
            "lis_virtual": int(lis_virtual),
        })

    fragment_mf_bar, fragment_common_bar = fragment_pullback(
        jnp.ones((), dtype=jnp.asarray(fragment_value).dtype)
    )
    if replay_forward:
        if int(forward_record["fragment_index"]) != fragment_index:
            raise ValueError(
                "fragment forward checkpoint index does not match the "
                f"requested fragment: {forward_record['fragment_index']} "
                f"!= {fragment_index}"
            )
        if (
            int(forward_record["lis_occupied"]) != int(lis_occupied)
            or int(forward_record["lis_virtual"]) != int(lis_virtual)
        ):
            raise ValueError(
                "fragment forward checkpoint LIS dimensions do not match "
                "the replayed fixed selections"
            )
        dtype = jnp.asarray(fragment_value).dtype
        e_mp2_lis = jnp.asarray(forward_record["e_mp2_lis"], dtype=dtype)
        e_cc = jnp.asarray(forward_record["e_ccsd"], dtype=dtype)
        e_t = jnp.asarray(forward_record["e_ccsd_t"], dtype=dtype)
        fragment_value = e_cc + e_t - e_mp2_lis
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
    checkpoint_dir=None,
    resume=False,
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

    If ``checkpoint_dir`` is supplied, fixed selections, a tiny forward
    record for each perturbative-triples fragment, cumulative fragment
    cotangents, molecule-wide MP2 progress, and the final pre-SCF cotangent
    are written atomically.  ``resume=True`` reuses compatible records.  A
    restart may seed SCF from disk, but ``build_mf`` must still implement the
    same converged implicit DF-RHF response map.
    """
    _validate_solver_options(ccsd_t=ccsd_t, dcsd=dcsd)
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
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

    scf_builder = build_mf
    if checkpoint_dir is not None:
        def scf_builder(mol_):
            return _fix_restart_mo_phases(build_mf(mol_))

    # The local-orbital cotangent is nonstationary.  It must be closed with
    # the converged fixed-point response, never the experimental finite SCF
    # replay backend.
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
    ):
        mf, scf_pullback = jax.vjp(scf_builder, mol)

    restart = RestartManager(
        checkpoint_dir,
        resume=resume,
        method=f"iao-dlno-{method_label.lower()}-gradient",
        scientific_payload=_restart_scientific_payload(
            mol,
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
            ccsd_t=ccsd_t,
            dcsd=dcsd,
        ),
    )
    if restart.enabled:
        log(
            f"restart {'resume' if resume else 'checkpoint'} directory: "
            f"{restart.path}"
        )

    saved_static = (
        restart.load_static(expected_type=IAOFragmentLISStaticSelections)
        if resume else None
    )
    if static_selections is not None and not isinstance(
        static_selections, IAOFragmentLISStaticSelections
    ):
        raise TypeError(
            "static_selections must be IAOFragmentLISStaticSelections"
        )
    if static_selections is None and saved_static is not None:
        static_selections = saved_static
        log("restart: loaded fixed fragment topology and LIS selections")
    elif static_selections is None:
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
        if restart.enabled:
            restart.save_static(static_selections)
            log("restart: saved fixed fragment topology and LIS selections")
    # An explicitly supplied selection must agree with an existing
    # checkpoint.  ``bind_static`` performs that scientific identity check
    # without replacing the saved topology.
    if restart.enabled:
        restart.bind_static(static_selections)
        if not restart.static_path.is_file():
            restart.save_static(static_selections)
            log("restart: saved supplied fixed topology and LIS selections")

    mp2_static = static_selections.mp2_static
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
    hf_bar, = hf_pullback(jnp.ones((), dtype=jnp.asarray(e_hf).dtype))

    pre_scf = (
        restart.load_record(
            "pre_scf",
            templates={"mf_bar": hf_bar},
            on_corrupt="raise",
        )
        if resume else None
    )
    if pre_scf is not None:
        mf_bar = pre_scf.trees["mf_bar"]
        components = dict(pre_scf.scalars)
        fragment_records = list(
            pre_scf.metadata.get("fragment_records", ())
        )
        dtype = jnp.asarray(e_hf).dtype
        e_iao_mp2 = jnp.asarray(components["e_iao_mp2"], dtype=dtype)
        e_mp2_lis_total = jnp.asarray(
            components["e_mp2_lis"], dtype=dtype
        )
        e_ccsd_total = jnp.asarray(components["e_ccsd"], dtype=dtype)
        e_ccsd_t_total = jnp.asarray(
            components["e_ccsd_t"], dtype=dtype
        )
        correlation = jnp.asarray(components["e_corr"], dtype=dtype)
        energy = jnp.asarray(components["e_total"], dtype=dtype)
        log(
            "restart: loaded pre-SCF total cotangent; skipping common "
            "orbital construction, fragment solves, and local MP2"
        )
    else:
        common, common_pullback = jax.vjp(
            lambda mf_: rebuild_iao_mp2_common(mf_, mp2_static), mf
        )
        mf_bar = hf_bar
        common_bar = jax.tree_util.tree_map(jnp.zeros_like, common)

        e_cc_terms = []
        e_t_terms = []
        e_mp2_lis_terms = []
        fragment_records = []
        nfragment = len(static_selections.fragments)
        cc_progress = (
            restart.load_record(
                "cc_progress",
                templates={
                    "mf_bar": mf_bar,
                    "common_bar": common_bar,
                },
                on_corrupt="raise",
            )
            if resume else None
        )
        completed_count = 0
        if cc_progress is not None:
            completed_count = int(
                cc_progress.scalars["completed_count"]
            )
            fragment_records = list(
                cc_progress.metadata.get("fragment_records", ())
            )
            completed_ids = tuple(
                int(value) for value in
                cc_progress.metadata.get("completed_ids", ())
            )
            expected_ids = tuple(range(completed_count))
            if (
                completed_count < 0
                or completed_count > nfragment
                or completed_ids != expected_ids
                or len(fragment_records) != completed_count
            ):
                raise ValueError(
                    "cumulative CC restart record is not a complete "
                    "serial fragment prefix"
                )
            mf_bar = cc_progress.trees["mf_bar"]
            common_bar = cc_progress.trees["common_bar"]
            for record in fragment_records:
                e_mp2_lis_terms.append(jnp.asarray(
                    record["e_mp2_lis"], dtype=common.s1e.dtype
                ))
                e_cc_terms.append(jnp.asarray(
                    record["e_ccsd"], dtype=common.s1e.dtype
                ))
                e_t_terms.append(jnp.asarray(
                    record["e_ccsd_t"], dtype=common.s1e.dtype
                ))
            log(
                "restart: loaded cumulative CC fragment progress "
                f"{completed_count}/{nfragment}"
            )

        for fragment_index in range(completed_count, nfragment):
            fragment_started = time.perf_counter()
            forward_record = None
            if resume and ccsd_t:
                saved_forward = restart.load_record(
                    "fragment_forward",
                    key=fragment_index,
                    templates={},
                    on_corrupt="raise",
                )
                if saved_forward is not None:
                    forward_record = dict(saved_forward.scalars)
                    log(
                        f"restart: fragment {fragment_index + 1}/"
                        f"{nfragment} has saved (T) forward energy; "
                        "replaying only its differentiable backward path"
                    )

            def save_forward(values, _index=fragment_index):
                restart.save_record(
                    "fragment_forward",
                    key=_index,
                    scalars=values,
                )
                log(
                    f"restart: saved fragment {_index + 1}/{nfragment} "
                    "forward scalar checkpoint"
                )

            fragment = _fragment_value_and_grad(
                mf,
                common,
                static_selections,
                fragment_index,
                verbose_imp=verbose_imp,
                ccsd_t=ccsd_t,
                dcsd=dcsd,
                forward_record=forward_record,
                save_forward=(
                    save_forward
                    if restart.enabled and ccsd_t and forward_record is None
                    else None
                ),
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
            fragment_wall = time.perf_counter() - fragment_started
            fragment_records.append({
                "fragment_index": fragment_index,
                "worker_rank": None,
                "e_mp2_lis": float(fragment.e_mp2_lis),
                "e_ccsd": float(fragment.e_ccsd),
                "e_ccsd_t": float(fragment.e_ccsd_t),
                "lis_occupied": int(fragment.lis_occupied),
                "lis_virtual": int(fragment.lis_virtual),
                "wall_seconds": fragment_wall,
            })
            if restart.enabled:
                restart.save_record(
                    "cc_progress",
                    scalars={"completed_count": fragment_index + 1},
                    trees={
                        "mf_bar": mf_bar,
                        "common_bar": common_bar,
                    },
                    metadata={
                        "completed_ids": list(range(fragment_index + 1)),
                        "fragment_records": fragment_records,
                    },
                )
                log(
                    "restart: saved cumulative CC fragment progress "
                    f"{fragment_index + 1}/{nfragment}"
                )
            log(
                f"fragment {fragment_index + 1}/{nfragment} complete: "
                f"solver LIS occ/vir={fragment.lis_occupied}/"
                f"{fragment.lis_virtual}; wall={fragment_wall:.1f} s"
            )
            del fragment
            gc.collect()

        e_iao_mp2, mp2_mf_bar, mp2_common_bar = (
            correlation_value_and_grad_from_common(
                mf, common, mp2_static, restart=restart
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
        e_mp2_lis_total = jnp.sum(jnp.stack(e_mp2_lis_terms))
        e_ccsd_total = jnp.sum(jnp.stack(e_cc_terms))
        e_ccsd_t_total = jnp.sum(jnp.stack(e_t_terms))
        correlation = _assemble_iao_dlno_correlation(
            jnp.stack(e_cc_terms),
            jnp.stack(e_t_terms),
            jnp.stack(e_mp2_lis_terms),
            e_iao_mp2,
        )
        energy = e_hf + correlation
        jax.block_until_ready((mf_bar, energy))
        if restart.enabled:
            restart.save_record(
                "pre_scf",
                scalars={
                    "e_iao_mp2": float(e_iao_mp2),
                    "e_mp2_lis": float(e_mp2_lis_total),
                    "e_ccsd": float(e_ccsd_total),
                    "e_ccsd_t": float(e_ccsd_t_total),
                    "e_corr": float(correlation),
                    "e_total": float(energy),
                },
                trees={"mf_bar": mf_bar},
                metadata={"fragment_records": fragment_records},
            )
            log(
                "restart: saved total pre-SCF energy and mean-field "
                "cotangent"
            )

    mol_bar, = scf_pullback(mf_bar)
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
