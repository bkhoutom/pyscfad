"""Public API, restart, and MPI metadata coverage for Boys DLNO-CC."""

from __future__ import annotations

from pyscfad.dlno import ccsd, ccsd_mpi, dlno_base_mpi, mp2_mpi


import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


from pyscfad.dlno._restart import scientific_digest
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.domain import DLNOThresholds


def _assert_target_options(function, *, lo_type_default="iao"):
    parameters = inspect.signature(function).parameters
    assert parameters["lo_type"].default == lo_type_default
    assert parameters["lo_kwargs"].default is None


def test_public_cc_and_mpi_entry_points_expose_target_options():
    for function in (
        DLNOCCSD,
        DLNOCCSD.value_and_grad,
        ccsd.build_domain_selections_for_ccsd,
        ccsd.build_static_selections,
        ccsd.kernel,
        ccsd.value_and_grad,
        ccsd_mpi.DLNOCCSD.value_and_grad,
        mp2_mpi.DLNOMP2.value_and_grad,
    ):
        _assert_target_options(function)
    _assert_target_options(DLNOCCSD.kernel, lo_type_default=None)


def test_instance_forwards_resolved_boys_options(monkeypatch):
    captured = {}

    def fake_kernel(_mf, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(e_corr=-0.25, e_total=-1.25)

    monkeypatch.setattr(ccsd, 'kernel', fake_kernel)
    mf = SimpleNamespace(
        mol=SimpleNamespace(verbose=0),
        with_df=object(),
        mo_occ=np.asarray([2.0, 0.0]),
    )
    options = {"conv_tol": 1e-12, "symmetry": True}
    solver = DLNOCCSD(mf, lo_type="BOYS", lo_kwargs=options)
    options["conv_tol"] = 1e-3

    assert solver.lo_type == "boys"
    assert solver.lo_kwargs == {"conv_tol": 1e-12, "symmetry": True}
    assert solver.kernel() == -0.25
    assert captured["lo_type"] == "boys"
    assert captured["lo_kwargs"] == solver.lo_kwargs


def test_domain_and_lis_builders_forward_target_options(monkeypatch):
    calls = []

    def fake_topology(_mf, **kwargs):
        calls.append(("topology", kwargs))
        return object()

    static = object()

    def fake_mp2_static(_mf, topology):
        calls.append(("mp2_static", topology))
        return static

    monkeypatch.setattr(ccsd, 'build_domain_topology', fake_topology)
    monkeypatch.setattr(
        ccsd, 'build_domain_selections', fake_mp2_static
    )

    result = ccsd.build_domain_selections_for_ccsd(
        object(), lo_type="boys", lo_kwargs={"conv_tol": 1e-11}
    )
    assert result is static
    assert calls[0][1]["lo_type"] == "boys"
    assert calls[0][1]["lo_kwargs"] == {"conv_tol": 1e-11}


@pytest.mark.parametrize(
    ("requested_mode", "requested_options", "message"),
    (
        ("iao", None, "lo_type"),
        ("boys", {"conv_tol": 1e-10}, "lo_kwargs"),
    ),
)
def test_supplied_static_must_match_target_mode_and_options(
    requested_mode, requested_options, message
):
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            lo_type="boys", lo_kwargs={"conv_tol": 1e-12}
        )
    )
    with pytest.raises(ValueError, match=message):
        ccsd._validate_static_target_options(
            static,
            lo_type=requested_mode,
            lo_kwargs=requested_options,
        )


def test_supplied_boys_static_rejects_ignored_target_and_active_space_inputs():
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            lo_type="boys",
            lo_kwargs={},
            frag_lolist=(np.asarray([0]), np.asarray([1])),
            frag_atmlist=(np.asarray([0]), np.asarray([1])),
            frozen=1,
        )
    )
    cases = (
        ({"frag_lolist": [[0, 1]], "frozen": 1}, "singleton"),
        ({"frag_lolist": [[1], [0]], "frozen": 1}, "target map"),
        ({"frag_atmlist": [[0], [1]], "frozen": 1}, "frag_atmlist"),
        ({"frozen": 0}, "frozen"),
    )
    for changed, message in cases:
        with pytest.raises(ValueError, match=message):
            ccsd._validate_static_target_options(
                static,
                lo_type="boys",
                lo_kwargs=None,
                **changed,
            )


def test_static_frozen_check_allows_unspecified_and_equivalent_forms():
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            lo_type="iao",
            lo_kwargs={},
            frag_lolist=None,
            frag_atmlist=None,
            frozen=1,
        )
    )
    ccsd._validate_static_target_options(static, frozen=None)
    ccsd._validate_static_target_options(static, frozen=[0])

    no_frozen = SimpleNamespace(
        mp2_static=SimpleNamespace(
            lo_type="iao",
            lo_kwargs={},
            frag_lolist=None,
            frag_atmlist=None,
            frozen=None,
        )
    )
    ccsd._validate_static_target_options(no_frozen, frozen=0)
    ccsd._validate_static_target_options(no_frozen, frozen=[])


def test_mpi_option_identity_accepts_jax_array_initial_guess():
    semantic = mp2_mpi._semantic_tuple(
        {"init_guess": jnp.eye(2)}
    )
    assert isinstance(semantic, tuple)
    assert isinstance(semantic[0][1], tuple)
    hash(semantic)


def test_mpi_mp2_supplied_boys_topology_rejects_ignored_overrides():
    topology = SimpleNamespace(
        lo_type="boys",
        lo_kwargs={},
        frag_lolist=(np.asarray([0]), np.asarray([1])),
        frag_atmlist=(np.asarray([0]), np.asarray([1])),
        frozen=1,
    )
    with pytest.raises(ValueError, match="singleton"):
        dlno_base_mpi._validate_target_options(
            topology,
            lo_type="boys",
            lo_kwargs=None,
            frag_lolist=[[0, 1]],
            frag_atmlist=None,
            frozen=1,
        )
    with pytest.raises(ValueError, match="frag_atmlist"):
        dlno_base_mpi._validate_target_options(
            topology,
            lo_type="boys",
            lo_kwargs=None,
            frag_lolist=None,
            frag_atmlist=[[0]],
            frozen=1,
        )
    with pytest.raises(ValueError, match="frozen"):
        dlno_base_mpi._validate_target_options(
            topology,
            lo_type="boys",
            lo_kwargs=None,
            frag_lolist=None,
            frag_atmlist=None,
            frozen=0,
        )


def _restart_objects():
    coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    mol = SimpleNamespace(
        charge=0,
        spin=0,
        nelectron=2,
        natm=2,
        nao=2,
        basis="test-basis",
        _basis={"H": "test-basis"},
        _ecp={},
        _pseudo={},
        cart=False,
        nucmod={},
        atom_charges=lambda: np.asarray([1, 1]),
        atom_coords=lambda unit=None: coords,
    )
    mf = SimpleNamespace(
        mo_occ=np.asarray([2.0, 0.0]),
        e_tot=-1.0,
        with_df=None,
    )
    return mol, mf


def _cc_restart_payload(lo_type, lo_kwargs):
    mol, mf = _restart_objects()
    return ccsd._restart_scientific_payload(
        mol,
        mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=DLNOThresholds(),
        pair_energy_model="multipole",
        force_full_domains=False,
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=1e-6,
        ccsd_t=False,
        dcsd=False,
        lo_type=lo_type,
        lo_kwargs=lo_kwargs,
    )


def test_restart_abi_rejects_pre_target_metadata_records():
    from pyscfad.dlno import _restart

    assert _restart._ALGORITHM_ABI >= 2


def test_restart_payload_identifies_mode_and_resolved_options():
    iao_payload = _cc_restart_payload("iao", None)
    boys_payload = _cc_restart_payload("boys", {"conv_tol": 1e-12})

    assert iao_payload["settings"]["lo_type"] == "iao"
    assert iao_payload["settings"]["lo_kwargs"] == {}
    assert boys_payload["settings"]["lo_type"] == "boys"
    assert boys_payload["settings"]["lo_kwargs"] == {"conv_tol": 1e-12}
    assert scientific_digest(iao_payload) != scientific_digest(boys_payload)


