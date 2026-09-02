"""Small-system identities for the IAO-DLNO-CCSD(T) construction.

These tests deliberately use water dimers whose IAO-MP2 graphs exercise the
two important limits for the CC driver: a compact all-strong graph and a
separated graph with two strong self domains plus one unordered weak pair.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update, gto, scf
from pyscfad.cc import dfccsd
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno import iao_ccsd as iao_ccsd_module
from pyscfad.dlno import _restart as restart_module
from pyscfad.dlno.iao_ccsd import (
    _assemble_iao_dlno_correlation,
    _fragment_value_and_grad,
    build_iao_dlno_ccsd_static_selections,
)
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
    evaluate_iao_fragment_mp2,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    build_strong_ed_domain,
    correlation_energy,
    rebuild_iao_mp2_common,
)
from pyscfad.dlno.iao_lis import (
    IAOFragmentLISStaticSelections,
    build_fragment_lis,
    build_iao_lis_static_selections,
    strong_domain_prescreen,
)
from pyscf import lib as pyscf_lib
from pyscfad.lno.ccsd import RCCSD as ImpurityRCCSD
from pyscfad.lno.ccsd import mp2_fragment_energy
from pyscfad.mp import dfmp2


def test_correction_bookkeeping_and_cotangent_seeds_are_unique():
    """The full strong+weak IAO-MP2 scalar is added once, not per fragment."""

    e_cc = jnp.asarray([-0.31, -0.27, -0.19])
    e_t = jnp.asarray([-0.012, -0.009, -0.006])
    e_mp2_lis = jnp.asarray([-0.22, -0.18, -0.14])
    e_iao_mp2 = jnp.asarray(-0.61)

    value = _assemble_iao_dlno_correlation(
        e_cc, e_t, e_mp2_lis, e_iao_mp2
    )
    expected = (
        np.sum(np.asarray(e_cc))
        + np.sum(np.asarray(e_t))
        - np.sum(np.asarray(e_mp2_lis))
        + float(e_iao_mp2)
    )
    np.testing.assert_allclose(value, expected, atol=0.0, rtol=0.0)

    gradients = jax.grad(
        _assemble_iao_dlno_correlation,
        argnums=(0, 1, 2, 3),
    )(e_cc, e_t, e_mp2_lis, e_iao_mp2)
    np.testing.assert_array_equal(gradients[0], np.ones(3))
    np.testing.assert_array_equal(gradients[1], np.ones(3))
    np.testing.assert_array_equal(gradients[2], -np.ones(3))
    np.testing.assert_array_equal(gradients[3], np.asarray(1.0))


def test_fragment_forward_restart_replays_lazy_triples_pullback(monkeypatch):
    """A saved triples primal skips its energy but preserves its VJP."""

    @jax.custom_vjp
    def lazy_triples(mf_value, common_value):
        return jnp.zeros((), dtype=mf_value.dtype)

    def lazy_fwd(mf_value, common_value):
        del common_value
        return jnp.zeros((), dtype=mf_value.dtype), None

    def lazy_bwd(_residual, cotangent):
        return 7.0 * cotangent, 11.0 * cotangent

    lazy_triples.defvjp(lazy_fwd, lazy_bwd)
    passes = []

    def fake_solve(
        mf_value,
        common_value,
        _static,
        _fragment_index,
        _metadata,
        *,
        profile_pass=None,
        **_kwargs,
    ):
        passes.append(profile_pass)
        e_mp2 = 2.0 * mf_value + 3.0 * common_value
        e_ccsd = 5.0 * mf_value - common_value
        e_t = (
            lazy_triples(mf_value, common_value)
            if profile_pass == "backward replay"
            else 7.0 * mf_value + 11.0 * common_value
        )
        lis = SimpleNamespace(
            active_occupied_coeff=jnp.zeros((1, 2)),
            active_virtual_coeff=jnp.zeros((1, 3)),
        )
        return (e_mp2, e_ccsd, e_t), lis

    monkeypatch.setattr(iao_ccsd_module, "_solve_fragment", fake_solve)
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            fragments=(SimpleNamespace(extended_atoms=np.asarray([0])),)
        )
    )
    saved = []
    direct = _fragment_value_and_grad(
        jnp.asarray(0.4),
        jnp.asarray(-0.2),
        static,
        0,
        verbose_imp=0,
        ccsd_t=True,
        dcsd=False,
        save_forward=saved.append,
    )
    assert len(saved) == 1
    replay = _fragment_value_and_grad(
        jnp.asarray(0.4),
        jnp.asarray(-0.2),
        static,
        0,
        verbose_imp=0,
        ccsd_t=True,
        dcsd=False,
        forward_record=saved[0],
    )

    assert passes == [None, "backward replay"]
    np.testing.assert_allclose(replay.value, direct.value, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        [replay.e_mp2_lis, replay.e_ccsd, replay.e_ccsd_t],
        [direct.e_mp2_lis, direct.e_ccsd, direct.e_ccsd_t],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(replay.mf_bar, direct.mf_bar, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        replay.common_bar, direct.common_bar, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(replay.mf_bar, 10.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(replay.common_bar, 7.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("fail_pullback", [False, True])
def test_fragment_workspace_spans_forward_and_pullback_and_cleans_up(
    tmp_path, monkeypatch, fail_pullback
):
    """The rank-private Lov directory outlives the complete fragment VJP."""

    monkeypatch.setattr(pyscf_lib.param, "TMPDIR", str(tmp_path))
    phases = []
    scratch_paths = []

    def fake_solve(
        mf_value,
        common_value,
        _static,
        _fragment_index,
        _metadata,
        *,
        lov_scratch_dir,
        **_kwargs,
    ):
        scratch_path = Path(lov_scratch_dir)
        assert scratch_path.is_dir()
        scratch_paths.append(scratch_path)

        @jax.custom_vjp
        def probe(value):
            return 2.0 * value

        def probe_fwd(value):
            assert scratch_path.is_dir()
            phases.append("forward")
            return 2.0 * value, None

        def probe_bwd(_residual, cotangent):
            assert scratch_path.is_dir()
            phases.append("pullback")
            if fail_pullback:
                raise RuntimeError("injected workspace pullback failure")
            return (2.0 * cotangent,)

        probe.defvjp(probe_fwd, probe_bwd)
        values = (
            probe(mf_value) + 3.0 * common_value,
            jnp.zeros_like(mf_value),
            jnp.zeros_like(mf_value),
        )
        lis = SimpleNamespace(
            active_occupied_coeff=jnp.zeros((1, 1)),
            active_virtual_coeff=jnp.zeros((1, 1)),
        )
        return values, lis

    monkeypatch.setattr(iao_ccsd_module, "_solve_fragment", fake_solve)
    static = SimpleNamespace(
        mp2_static=SimpleNamespace(
            fragments=(SimpleNamespace(extended_atoms=np.asarray([0])),)
        )
    )
    call = lambda: _fragment_value_and_grad(
        jnp.asarray(0.4),
        jnp.asarray(-0.2),
        static,
        0,
        verbose_imp=0,
        ccsd_t=False,
        dcsd=False,
    )

    if fail_pullback:
        with pytest.raises(
            RuntimeError, match="injected workspace pullback failure"
        ):
            call()
    else:
        result = call()
        np.testing.assert_allclose(result.mf_bar, -2.0)
        np.testing.assert_allclose(result.common_bar, -3.0)

    assert phases == ["forward", "pullback"]
    assert len(scratch_paths) == 1
    assert scratch_paths[0].name.startswith("pyscfad-lov-frag0-")
    assert not scratch_paths[0].exists()
    assert list(tmp_path.glob("pyscfad-lov-frag*")) == []


def test_energy_only_kernel_gives_each_fragment_a_forward_workspace(
    tmp_path, monkeypatch
):
    """The energy-only loop also owns and removes per-fragment Lov files."""

    monkeypatch.setattr(pyscf_lib.param, "TMPDIR", str(tmp_path))
    mp2_fragments = tuple(
        SimpleNamespace(extended_atoms=np.asarray([index]))
        for index in range(2)
    )
    static = IAOFragmentLISStaticSelections(
        mp2_static=SimpleNamespace(fragments=mp2_fragments),
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=1e-6,
        fragments=tuple(
            SimpleNamespace(fragment_index=index) for index in range(2)
        ),
    )
    common = SimpleNamespace(s1e=jnp.eye(1))
    scratch_paths = []

    def fake_solve(
        _mf,
        _common,
        _static,
        fragment_index,
        _metadata,
        *,
        lov_scratch_dir,
        **_kwargs,
    ):
        scratch_path = Path(lov_scratch_dir)
        assert scratch_path.is_dir()
        (scratch_path / "local_lov.h5").touch()
        scratch_paths.append(scratch_path)
        value = jnp.asarray(float(fragment_index + 1))
        lis = SimpleNamespace(
            active_occupied_coeff=jnp.zeros((1, 1)),
            active_virtual_coeff=jnp.zeros((1, 1)),
        )
        return (value, 2.0 * value, jnp.zeros_like(value)), lis

    monkeypatch.setattr(
        iao_ccsd_module, "rebuild_iao_mp2_common", lambda *_args: common
    )
    monkeypatch.setattr(iao_ccsd_module, "_solve_fragment", fake_solve)
    monkeypatch.setattr(
        iao_ccsd_module,
        "iao_mp2_correlation_energy",
        lambda *_args: jnp.asarray(-0.25),
    )

    result = iao_ccsd_module.kernel(
        SimpleNamespace(e_tot=jnp.asarray(-10.0)),
        static_selections=static,
    )

    np.testing.assert_allclose(result.e_mp2_lis, 3.0)
    assert len(scratch_paths) == 2
    assert len({path.name for path in scratch_paths}) == 2
    assert all(path.name.startswith("pyscfad-lov-frag") for path in scratch_paths)
    assert all(not path.exists() for path in scratch_paths)
    assert list(tmp_path.glob("pyscfad-lov-frag*")) == []


def test_serial_cc_restart_after_fragment_and_from_pre_scf_high_cost(
    tmp_path, monkeypatch
):
    """Durable CC progress and pre-SCF records skip completed work."""

    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        """,
        unit="Angstrom",
        basis="6-31g",
        verbose=0,
        max_memory=500,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)

    def build_mf(mol_):
        mf = scf.RHF(mol_).density_fit(auxbasis="weigend")
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        return mf

    kwargs = dict(
        build_mf=build_mf,
        thresh_occ=1.0,
        thresh_vir=1.0,
        ccsd_t=True,
    )
    reference_energy, reference_bar = DLNOCCSD.value_and_grad(
        mol, **kwargs
    )

    class InjectedStop(RuntimeError):
        pass

    def stop_after_forward(stage, key, path):
        del key, path
        if stage == "fragment_forward":
            raise InjectedStop("durable triples forward reached")

    def stop_after_cc(stage, key, path):
        del key, path
        if stage == "cc_progress":
            raise InjectedStop("durable CC prefix reached")

    checkpoint_dir = tmp_path / "serial-cc-restart"
    monkeypatch.setattr(
        restart_module, "_CHECKPOINT_EVENT_HOOK", stop_after_forward
    )
    with pytest.raises(InjectedStop, match="durable triples forward"):
        DLNOCCSD.value_and_grad(
            mol, checkpoint_dir=checkpoint_dir, **kwargs
        )
    forward_path = checkpoint_dir / "records" / "fragment_forward" / "0.h5"
    handle, metadata, reader = restart_module._read_hdf5(forward_path)
    try:
        forward_scalars = restart_module._decode_value(
            metadata["scalars"], reader
        )
    finally:
        handle.close()
    assert int(forward_scalars["lis_virtual"]) > 0
    assert abs(float(forward_scalars["e_ccsd_t"])) > 1e-12

    replay_messages = []
    monkeypatch.setattr(
        restart_module, "_CHECKPOINT_EVENT_HOOK", stop_after_cc
    )
    with pytest.raises(InjectedStop, match="durable CC prefix"):
        DLNOCCSD.value_and_grad(
            mol,
            checkpoint_dir=checkpoint_dir,
            resume=True,
            progress=replay_messages.append,
            **kwargs,
        )
    assert any(
        "saved (T) forward energy" in line for line in replay_messages
    )

    messages = []
    monkeypatch.setattr(restart_module, "_CHECKPOINT_EVENT_HOOK", None)
    resumed_energy, resumed_bar = DLNOCCSD.value_and_grad(
        mol,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        progress=messages.append,
        **kwargs,
    )
    assert any(
        "loaded cumulative CC fragment progress 1/1" in line
        for line in messages
    )
    np.testing.assert_allclose(
        resumed_energy, reference_energy, atol=2e-11, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(resumed_bar.coords),
        np.asarray(reference_bar.coords),
        atol=2e-10,
        rtol=0.0,
    )

    def forbidden_common(*_args, **_kwargs):
        raise AssertionError("pre-SCF restart rebuilt the common orbitals")

    monkeypatch.setattr(
        iao_ccsd_module, "rebuild_iao_mp2_common", forbidden_common
    )
    messages.clear()
    final_energy, final_bar = DLNOCCSD.value_and_grad(
        mol,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        progress=messages.append,
        **kwargs,
    )
    assert any("loaded pre-SCF total cotangent" in line for line in messages)
    np.testing.assert_allclose(final_energy, resumed_energy, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(final_bar.coords),
        np.asarray(resumed_bar.coords),
        atol=2e-12,
        rtol=0.0,
    )


def test_dlno_ccsd_public_api_has_no_legacy_mp2_correction_switches():
    constructor = inspect.signature(DLNOCCSD).parameters
    parameters = inspect.signature(DLNOCCSD.value_and_grad).parameters
    for obsolete in (
        "include_mp2_correction",
        "mp2_correction_method",
        "mp2_correction_scope",
        "sos_c_os",
    ):
        assert obsolete not in parameters
        assert obsolete not in constructor
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(DLNOCCSD).parameters.values()
    )
    assert constructor["thresh"].default is None
    assert constructor["thresh_occ"].default == 1e-4
    assert constructor["thresh_vir"].default == 1e-5
    assert parameters["thresh_occ"].default == 1e-4
    assert parameters["thresh_vir"].default == 1e-5


def test_empty_fragment_excitation_space_is_a_zero_cc_increment():
    """An intentionally over-truncated LIS must not enter a zero-size CC solve."""

    mol = gto.Mole(
        atom="He 0 0 0",
        unit="Bohr",
        basis="cc-pvdz",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.kernel()

    default_local = DLNOCCSD(mf)
    assert default_local.thresh_occ == 1e-4
    assert default_local.thresh_vir == 1e-5

    local = DLNOCCSD(mf, thresh=1.0)
    assert local.thresh_occ == 1.0
    assert local.thresh_vir == 1.0
    energy = local.kernel(thresh_occ=1.0, thresh_vir=1.0)
    assert np.isfinite(float(energy))
    assert local.result.lis_virtual == (0,)
    np.testing.assert_allclose(local.result.e_ccsd, 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        local.result.e_mp2_lis, 0.0, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        local.e_corr_pt2, local.result.e_iao_mp2, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        local.e_corr_pt2_correction,
        local.result.e_iao_mp2 - local.result.e_mp2_lis,
        atol=0.0,
        rtol=0.0,
    )
    with pytest.raises(AttributeError, match="no domain-only"):
        _ = local.e_corr_pt2_domain
    with pytest.raises(AttributeError, match="already includes"):
        local.e_corr_pt2corrected(-0.1)

    def build_he_mf(mol_):
        mf_ = scf.RHF(mol_).density_fit(auxbasis="weigend")
        mf_.conv_tol = 1e-12
        mf_.conv_tol_grad = 1e-10
        mf_.kernel()
        return mf_

    total, mol_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=build_he_mf,
        thresh_occ=1.0,
        thresh_vir=1.0,
    )
    assert np.isfinite(float(total))
    np.testing.assert_allclose(
        np.asarray(mol_bar.coords), np.zeros((1, 3)), atol=1e-12, rtol=0.0
    )


def test_triples_and_dcsd_are_rejected_together():
    mol = _water_dimer(separated=False)
    with pytest.raises(ValueError, match="not defined for DCSD"):
        DLNOCCSD.value_and_grad(
            mol,
            build_mf=_build_mf,
            ccsd_t=True,
            dcsd=True,
        )


def _water_dimer(*, separated):
    if separated:
        atom = """
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
        """
    else:
        atom = """
        O  -1.485163346097 -0.114724564047  0.000000000000
        H  -1.868415346097  0.762298435953  0.000000000000
        H  -0.533833346097  0.040507435953  0.000000000000
        O   1.416468653903  0.111264435953  0.000000000000
        H   1.746241653903 -0.373945564047 -0.758561000000
        H   1.746241653903 -0.373945564047  0.758561000000
        """
    mol = gto.Mole(
        atom=atom,
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def _build_mf(mol):
    mf = scf.RHF(mol).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


def _build_local_problem(*, separated, thresholds=None,
                         pair_energy_model="multipole",
                         force_full_domains=False):
    mol = _water_dimer(separated=separated)
    mf = _build_mf(mol)
    if thresholds is None:
        thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
    topology = build_iao_fragment_topology(
        mf,
        thresholds=thresholds,
        pair_energy_model=pair_energy_model,
        force_full_domains=force_full_domains,
    )
    static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, static)
    return mol, mf, topology, static, common


@pytest.mark.parametrize(
    "separated, expected_mask, expected_atoms, expected_domain_shape",
    [
        (
            False,
            [[True, True], [True, True]],
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            (14, 10, 4),
        ),
        (
            True,
            [[True, False], [False, True]],
            [[0, 1, 2], [3, 4, 5]],
            (7, 5, 2),
        ),
    ],
)
def test_compact_and_far_water_define_expected_iao_strong_domains(
    separated,
    expected_mask,
    expected_atoms,
    expected_domain_shape,
):
    """The CC LIS must start from exactly the validated IAO-MP2 ED graph."""

    mol, mf, _, static, common = _build_local_problem(
        separated=separated
    )

    assert mol.nao == 14
    assert np.count_nonzero(np.asarray(mf.mo_occ) > 0.0) == 10
    assert common.iao_coeff.shape == (14, 14)
    assert [indices.size for indices in static.frag_lolist] == [7, 7]
    np.testing.assert_array_equal(static.strong_mask, expected_mask)

    for fragment_index, (fragment, atoms) in enumerate(
        zip(static.fragments, expected_atoms)
    ):
        np.testing.assert_array_equal(fragment.extended_atoms, atoms)
        domain = build_strong_ed_domain(common, static, fragment_index)
        nao_domain, nocc_domain, nvir_domain = expected_domain_shape
        assert domain.occupied_coeff.shape == (nao_domain, nocc_domain)
        assert domain.virtual_coeff.shape == (nao_domain, nvir_domain)
        assert domain.target_projection.shape[1] == nocc_domain

        prescreen = strong_domain_prescreen(
            common, static, fragment_index, domain=domain
        )
        np.testing.assert_array_equal(
            prescreen["extended_primary_domain"], atoms
        )
        np.testing.assert_allclose(
            prescreen["occ_prescreen_coeff"], domain.occupied_coeff,
            atol=0.0, rtol=0.0,
        )
        np.testing.assert_allclose(
            prescreen["vir_prescreen_coeff"], domain.virtual_coeff,
            atol=0.0, rtol=0.0,
        )
        np.testing.assert_allclose(
            prescreen["orbfragloc"],
            common.iao_coeff[:, static.frag_lolist[fragment_index]],
            atol=0.0, rtol=0.0,
        )
        expected_strong_iao = np.unique(np.concatenate([
            static.frag_lolist[int(partner)]
            for partner in fragment.strong_fragments
        ]))
        np.testing.assert_array_equal(
            prescreen["strong_lmo_indices"], expected_strong_iao
        )


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
        pair_energy=0.0,
    )


def _canonical_dfmp2_energy(mf):
    """Evaluate the small reference independent of prior JAX cache RSS."""

    reference = dfmp2.MP2(mf)
    reference.max_memory = max(
        float(reference.max_memory),
        float(pyscf_lib.current_memory()[0]) + 512.0,
    )
    return reference.kernel(with_t2=False)[0]


def test_far_water_pt2_is_strong_energy_plus_one_weak_pair():
    """The molecular PT2 correction contains each weak pair exactly once."""

    _, mf, topology, static, _ = _build_local_problem(separated=True)
    assert np.count_nonzero(np.triu(~static.strong_mask, k=1)) == 1

    result = evaluate_iao_fragment_mp2(mf, topology)
    assert result.e_weak_multipole_os != 0.0
    assert result.e_corr == pytest.approx(
        result.e_strong + result.e_weak_multipole_os,
        abs=2e-14,
    )

    canonical = _canonical_dfmp2_energy(mf)
    # At 8 Angstrom the fourth-order weak multipole model is already within
    # sub-microhartree accuracy of canonical DF-MP2.
    assert abs(result.e_corr - canonical) < 1e-6

    tight_topology = build_iao_fragment_topology(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    tight = evaluate_iao_fragment_mp2(mf, tight_topology)
    assert tight.e_weak_multipole_os == 0.0
    np.testing.assert_allclose(tight.e_corr, canonical, atol=2e-9, rtol=0.0)


def test_far_water_lis_rejects_distant_iao_projection_tails(tmp_path):
    """Tiny cross-monomer IAO projections must not become internal orbitals."""

    _, mf, _, mp2_static, common = _build_local_problem(separated=True)
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=1e-3,
        thresh_vir=1e-3,
    )
    for fragment_index in range(2):
        scratch_dir = tmp_path / f"far-fragment-{fragment_index}"
        scratch_dir.mkdir()
        lis = build_fragment_lis(
            mf,
            common,
            lis_static,
            fragment_index,
            lov_scratch_dir=scratch_dir,
        )
        assert lis.n_internal_occ == 5
        assert lis.n_internal_vir == 2
        assert lis.active_occupied_coeff.shape[1] == 5
        assert lis.active_virtual_coeff.shape[1] == 2


def test_full_domain_iao_pt2_exactly_cancels_lis_mp2_sum(tmp_path):
    """At zero LIS thresholds both MP2 branches recover the same energy."""

    _, mf, _, mp2_static, common = _build_local_problem(
        separated=False,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=0.0,
        thresh_vir=0.0,
    )

    fragment_mp2 = []
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0.0))
    nvir = int(mf.mo_occ.size - nocc)
    for fragment_index in range(len(mp2_static.fragments)):
        scratch_dir = tmp_path / f"full-fragment-{fragment_index}"
        scratch_dir.mkdir()
        lis = build_fragment_lis(
            mf,
            common,
            lis_static,
            fragment_index,
            lov_scratch_dir=scratch_dir,
        )
        assert lis.active_occupied_coeff.shape[1] == nocc
        assert lis.active_virtual_coeff.shape[1] == nvir
        assert lis.frozen.size == 0
        np.testing.assert_allclose(
            lis.occupied_projector, np.eye(nocc), atol=2e-10, rtol=0.0
        )
        np.testing.assert_allclose(
            lis.virtual_projector, np.eye(nvir), atol=2e-10, rtol=0.0
        )

        # Evaluate precisely the conventional full-spin MP2 fragment energy
        # that the CC driver subtracts in this LIS.  No CC iterations are
        # needed for this limiting-identity test.
        cc = ImpurityRCCSD(
            mf, mo_coeff=lis.mo_coeff, frozen=lis.frozen
        )
        eris = cc.ao2mo(fockao=common.fock)
        _, _, t2 = cc.init_amps(eris=eris)
        projector = (
            np.asarray(lis.fragment_iao_coeff).T
            @ np.asarray(common.s1e)
            @ np.asarray(lis.active_occupied_coeff)
        )
        fragment_mp2.append(float(
            mp2_fragment_energy(eris, t2, projector)
        ))

    e_mp2_lis = sum(fragment_mp2)
    e_iao_mp2 = float(correlation_energy(mf, mp2_static))
    # This is the cancellation in
    #   sum_F(E_CC,F - E_MP2,LIS,F) + E_IAO-MP2.
    np.testing.assert_allclose(
        e_mp2_lis, e_iao_mp2, atol=2e-9, rtol=0.0
    )

    canonical = _canonical_dfmp2_energy(mf)
    np.testing.assert_allclose(
        e_iao_mp2, canonical, atol=2e-9, rtol=0.0
    )


def test_compact_full_domain_matches_canonical_dfccsd_t_energy():
    """The public energy solver recovers canonical DF-CCSD(T)."""

    mol = _water_dimer(separated=False)
    mf = _build_mf(mol)
    local = DLNOCCSD(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    local.thresh_occ = 0.0
    local.thresh_vir = 0.0
    local.ccsd_t = True
    local_total = mf.e_tot + local.kernel()

    canonical = dfccsd.RCCSD(mf)
    canonical.kernel()
    canonical_total = canonical.e_tot + canonical.ccsd_t()
    np.testing.assert_allclose(
        local_total, canonical_total, atol=2e-7, rtol=0.0
    )

    # In this limit the local-MP2 correction and the sum of the LIS-MP2
    # subtractions are the same scalar, so only CCSD(T) remains.
    np.testing.assert_allclose(
        local.result.e_iao_mp2,
        local.result.e_mp2_lis,
        atol=2e-9,
        rtol=0.0,
    )


def _canonical_dfccsd_total(mol):
    mf = _build_mf(mol)
    cc = dfccsd.RCCSD(mf)
    cc.kernel()
    return cc.e_tot


def _canonical_dfccsd_t_total(mol):
    mf = _build_mf(mol)
    cc = dfccsd.RCCSD(mf)
    cc.kernel()
    return cc.e_tot + cc.ccsd_t()


def test_compact_full_domain_matches_canonical_dfccsd_high_cost():
    """Zero LIS thresholds recover canonical DF-CCSD energy and gradient."""

    mol = _water_dimer(separated=False)
    thresholds = _full_domain_thresholds()

    local_energy, local_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=0.0,
        thresh_vir=0.0,
        ccsd_t=False,
    )
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
        config_update("pyscfad_ccsd_implicit_diff", True),
    ):
        canonical_energy, canonical_bar = jax.value_and_grad(
            _canonical_dfccsd_total
        )(mol)

    np.testing.assert_allclose(
        local_energy, canonical_energy, atol=2e-7, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(local_bar.coords),
        np.asarray(canonical_bar.coords),
        atol=5e-7,
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(local_bar.coords), axis=0),
        np.zeros(3),
        atol=2e-9,
        rtol=0.0,
    )


def test_compact_full_domain_matches_canonical_dfccsd_t_gradient_high_cost():
    """The complete local triples gradient recovers canonical DF-CCSD(T)."""

    mol = _water_dimer(separated=False)
    thresholds = _full_domain_thresholds()
    local_energy, local_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="all",
        force_full_domains=True,
        thresh_occ=0.0,
        thresh_vir=0.0,
        ccsd_t=True,
    )
    with (
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", False),
        config_update("pyscfad_ccsd_implicit_diff", True),
    ):
        canonical_energy, canonical_bar = jax.value_and_grad(
            _canonical_dfccsd_t_total
        )(mol)

    np.testing.assert_allclose(
        local_energy, canonical_energy, atol=2e-7, rtol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(local_bar.coords),
        np.asarray(canonical_bar.coords),
        atol=6e-7,
        rtol=3e-5,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(local_bar.coords), axis=0),
        np.zeros(3),
        atol=2e-9,
        rtol=0.0,
    )


def test_far_water_mixed_strong_weak_gradient_smoke_high_cost():
    """The CC gradient path handles two strong EDs and one weak pair."""

    mol = _water_dimer(separated=True)
    mf = _build_mf(mol)
    thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
    static = build_iao_dlno_ccsd_static_selections(
        mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
        thresh_occ=1e-3,
        thresh_vir=1e-3,
    )
    np.testing.assert_array_equal(
        static.mp2_static.strong_mask,
        [[True, False], [False, True]],
    )
    assert np.count_nonzero(
        np.triu(~static.mp2_static.strong_mask, k=1)
    ) == 1

    energy, mol_bar = DLNOCCSD.value_and_grad(
        mol,
        build_mf=_build_mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
        thresh_occ=1e-3,
        thresh_vir=1e-3,
        ccsd_t=False,
        static_selections=static,
    )
    gradient = np.asarray(mol_bar.coords)

    assert np.isfinite(float(energy))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(
        np.sum(gradient, axis=0), np.zeros(3), atol=2e-9, rtol=0.0
    )
    # Translating the second monomer along its separation coordinate must
    # have a finite, nonzero derivative in this mixed strong/weak topology.
    intermonomer_z = float(np.sum(gradient[3:, 2]))
    assert 1e-6 < abs(intermonomer_z) < 1e-3
