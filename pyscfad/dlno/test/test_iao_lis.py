from pathlib import Path
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import gto, scf
from pyscf import lib as pyscf_lib
from pyscfad.dlno import iao_lis
from pyscfad.dlno.iao_lis import (
    _density_from_target_amplitude_block,
    _density_from_target_amplitudes,
    build_fragment_lis,
    build_iao_lis_fragment_static_selection,
    build_iao_lis_static_selections,
    strong_domain_mp2_density_from_lov,
    strong_domain_prescreen,
    target_conditioned_mp2_density_from_amplitudes,
)
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
)
from pyscfad.dlno.iao_mp2_grad import (
    FixedPAOSubspaceSelection,
    IAOFragmentMP2ContinuousData,
    IAOFragmentMP2StaticSelections,
    IAOMP2FragmentStaticSelection,
    IAOMP2StrongDomain,
    build_iao_mp2_static_selections,
    build_strong_ed_domain,
    rebuild_iao_mp2_common,
)


def test_mp2_density_block_threshold_defaults_overrides_and_validation():
    """Configuration accepts only positive optional density block settings."""
    defaults = IAOFragmentMP2Thresholds()
    assert defaults.mp2_block_memory_mb is None
    assert defaults.mp2_block_nvir is None

    exact = IAOFragmentMP2Thresholds(mp2_block_nvir=192)
    assert exact.mp2_block_nvir == 192

    budget = IAOFragmentMP2Thresholds(mp2_block_memory_mb=4096.0)
    assert budget.mp2_block_memory_mb == 4096.0

    for kwargs, field in (
        ({"mp2_block_memory_mb": 0.0}, "mp2_block_memory_mb"),
        ({"mp2_block_nvir": 0}, "mp2_block_nvir"),
        ({"mp2_block_nvir": -1}, "mp2_block_nvir"),
    ):
        with pytest.raises(ValueError, match=field):
            IAOFragmentMP2Thresholds(**kwargs)


def test_resolve_mp2_density_block_nvir_precedence_and_dimension_scaling():
    """The resolver makes a positive, dimension-aware block choice."""
    resolve = iao_lis._resolve_mp2_density_block_nvir
    common = dict(
        naux=100,
        nocc=4,
        nvir=2000,
        ntarget=3,
        dtype=np.float64,
    )

    automatic, mode, target = resolve(
        **common,
        mf_max_memory_mb=1000.0,
        configured_memory_mb=None,
        configured_block_nvir=None,
    )
    larger_automatic, larger_mode, larger_target = resolve(
        **common,
        mf_max_memory_mb=4000.0,
        configured_memory_mb=None,
        configured_block_nvir=None,
    )
    with pytest.warns(RuntimeWarning, match="one-virtual-block"):
        larger_dimensions, _, _ = resolve(
            **{
                **common,
                "naux": 2000,
                "nocc": 8,
                "nvir": 4000,
                "ntarget": 6,
            },
            mf_max_memory_mb=1000.0,
            configured_memory_mb=None,
            configured_block_nvir=None,
        )
    manual_budget, budget_mode, budget_target = resolve(
        **common,
        mf_max_memory_mb=1000.0,
        configured_memory_mb=600.0,
        configured_block_nvir=None,
    )

    assert 1 <= automatic <= common["nvir"]
    assert mode == "auto"
    assert target == 250.0
    assert larger_mode == "auto"
    assert larger_target == 400.0
    assert larger_automatic > automatic
    assert larger_dimensions < automatic
    assert budget_mode == "manual_budget"
    assert budget_target == 600.0
    assert manual_budget > automatic

    with pytest.warns(RuntimeWarning, match="manual MP2 density block width"):
        manual_width, width_mode, width_target = resolve(
            **common,
            mf_max_memory_mb=1000.0,
            configured_memory_mb=600.0,
            configured_block_nvir=9999,
        )
    assert manual_width == common["nvir"]
    assert width_mode == "manual_width"
    assert width_target == 250.0


def test_target_amplitude_block_density_matches_dense_complex_contraction():
    rng = np.random.default_rng(1907)
    u = rng.normal(size=(2, 3, 5, 5)) + 1j * rng.normal(size=(2, 3, 5, 5))
    dense = _density_from_target_amplitudes(jnp.asarray(u))

    occupied = np.zeros((3, 3), dtype=complex)
    virtual = np.zeros((5, 5), dtype=complex)
    for start in range(0, 5, 2):
        stop = min(start + 2, 5)
        a_block = u[:, :, :, start:stop]
        b_block = u[:, :, start:stop, :].transpose(0, 1, 3, 2)
        contribution = _density_from_target_amplitude_block(
            jnp.asarray(a_block), jnp.asarray(b_block)
        )
        occupied += np.asarray(contribution.occupied)
        virtual += np.asarray(contribution.virtual)

    occupied = 0.5 * (occupied + occupied.T.conj())
    virtual = 0.5 * (virtual + virtual.T.conj())
    np.testing.assert_allclose(occupied, np.asarray(dense.occupied), atol=2e-12)
    np.testing.assert_allclose(virtual, np.asarray(dense.virtual), atol=2e-12)


def test_target_amplitude_block_density_validates_rank_and_shapes():
    with pytest.raises(ValueError, match="rank-4"):
        _density_from_target_amplitude_block(
            jnp.zeros((2, 3, 5)), jnp.zeros((2, 3, 5))
        )
    with pytest.raises(ValueError, match="identical"):
        _density_from_target_amplitude_block(
            jnp.zeros((2, 3, 5, 2)), jnp.zeros((2, 3, 4, 2))
        )


def test_strong_domain_density_routes_directly_to_predictable_h5_file(
    tmp_path, monkeypatch,
):
    empty = np.zeros((0,), dtype=np.int32)
    empty_selection = FixedPAOSubspaceSelection(
        empty, empty, empty, empty, empty, empty
    )
    fragment = IAOMP2FragmentStaticSelection(
        fragment_index=0,
        iao_indices=empty,
        fragment_atoms=np.asarray([0], dtype=np.int32),
        strong_fragments=np.asarray([0], dtype=np.int32),
        extended_atoms=np.asarray([0, 1], dtype=np.int32),
        extended_ao_indices=empty,
        pao_center_atoms=empty,
        strong_occ_union_keep=empty,
        strong_occ_metric_keep=empty,
        strong_virtual=empty_selection,
        primary_atoms=empty,
        primary_ao_indices=empty,
        primary_bp_atoms=empty,
        weak_weight_eigen_indices=empty,
        weak_weight_degenerate_blocks=(),
        weak_occ_norm_keep=empty,
        weak_occ_span_metric_keep=empty,
        weak_virtual=None,
    )
    static = IAOFragmentMP2StaticSelections(
        frozen=None,
        thresholds=IAOFragmentMP2Thresholds(
            mp2_block_nvir=2,
        ),
        active_occ_indices=empty,
        active_vir_indices=empty,
        pao_projected_out_indices=empty,
        pao_parent_ao_indices=empty,
        ao2pao_map=empty,
        frag_lolist=(empty,),
        frag_atmlist=(None,),
        strong_mask=np.zeros((1, 1), dtype=bool),
        fragments=(fragment,),
    )
    domain = IAOMP2StrongDomain(
        occupied_coeff=np.arange(8.0).reshape(4, 2),
        virtual_coeff=np.arange(12.0).reshape(4, 3),
        occupied_energy=np.asarray([-1.2, -0.8]),
        virtual_energy=np.asarray([0.2, 0.7, 1.1]),
        target_projection=np.asarray([[0.8, 0.1], [0.2, 0.6]]),
        target_weight=np.eye(2),
        partner_weight=np.eye(2),
    )
    expected_density = iao_lis.IAOMP2Density(
        np.eye(2), 2.0 * np.eye(3)
    )
    events = []
    profile_token = object()
    fake_mol = SimpleNamespace(nao=4)
    auxmol = SimpleNamespace(nao=2)
    sentinel_mf = SimpleNamespace(
        mol=object(),
        max_memory=900.0,
        with_df=SimpleNamespace(auxbasis="sentinel-aux", max_memory=713.0),
    )

    def fake_start():
        events.append(("start", profile_token))
        return profile_token

    def fake_finish(phase, before, **details):
        events.append(("finish", phase, before, details))

    def fake_make_local_mol(mol, atoms):
        assert mol is sentinel_mf.mol
        np.testing.assert_array_equal(atoms, np.asarray([0, 1]))
        events.append(("local_mol",))
        return fake_mol

    def fake_make_auxmol(mol, auxbasis):
        assert mol is fake_mol
        assert auxbasis == "sentinel-aux"
        events.append(("auxmol",))
        return auxmol

    def fake_resolve(**kwargs):
        events.append(("resolve", kwargs))
        return 2, "manual_width", 225.0

    def fake_h5_density(*args):
        events.append(("h5-density", args))
        return expected_density

    monkeypatch.setattr(iao_lis.resource_profile, "start", fake_start)
    monkeypatch.setattr(iao_lis.resource_profile, "finish", fake_finish)
    monkeypatch.setattr(iao_lis.lno_base, "make_local_mol", fake_make_local_mol)
    monkeypatch.setattr(
        iao_lis.lno_base.df_addons, "make_auxmol", fake_make_auxmol
    )
    monkeypatch.setattr(
        iao_lis, "_resolve_mp2_density_block_nvir", fake_resolve
    )
    monkeypatch.setattr(
        iao_lis, "_strong_domain_mp2_density_h5", fake_h5_density
    )
    monkeypatch.setattr(
        iao_lis.lno_base,
        "get_local_Lov",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the production route must not materialize Lov")
        ),
    )

    density = iao_lis.strong_domain_mp2_density(
        sentinel_mf, domain, static, 0, lov_scratch_dir=tmp_path
    )

    assert density is expected_density
    resolve_event = next(event for event in events if event[0] == "resolve")
    assert resolve_event[1] == {
        "naux": 2,
        "nocc": 2,
        "nvir": 3,
        "ntarget": 2,
        "dtype": domain.occupied_coeff.dtype,
        "mf_max_memory_mb": 900.0,
        "configured_memory_mb": None,
        "configured_block_nvir": 2,
    }
    h5_event = next(event for event in events if event[0] == "h5-density")
    h5_args = h5_event[1]
    assert h5_args[0] is fake_mol
    assert h5_args[1] is auxmol
    np.testing.assert_array_equal(
        h5_args[2],
        np.concatenate((domain.occupied_coeff, domain.virtual_coeff), axis=1),
    )
    assert h5_args[6:] == (
        2,
        str(tmp_path / "local_lov.h5"),
        713.0,
        2,
    )
    assert [event[0] for event in events].index("resolve") < [
        event[0] for event in events
    ].index("h5-density")
    finish = next(event for event in events if event[0] == "finish")
    assert finish[1] == "iao_lis.strong_domain_mp2_density"
    assert finish[2] is profile_token
    assert finish[3]["lov_shape"] == (2, 2, 3)
    assert finish[3]["block_nvir"] == 2
    assert finish[3]["block_mode"] == "manual_width"
    assert finish[3]["block_count"] == 2
    assert finish[3]["workspace_target_mib"] == 225.0
    assert finish[3]["lov_h5_path_basename"] == "local_lov.h5"
    assert finish[3]["lov_disk_mib"] == 12 * 8 / 1024.0**2
    assert finish[3]["lov_bar_disk_mib"] == 0.0
    assert finish[3]["z_disk_mib"] == 0.0
    for key in (
        "hdf5_bytes_read",
        "hdf5_bytes_written",
        "hdf5_read_seconds",
        "hdf5_write_seconds",
    ):
        assert key in finish[3]


@pytest.mark.parametrize("fail_selection", [False, True])
def test_static_selection_owns_and_cleans_forward_workspace(
    tmp_path, monkeypatch, fail_selection
):
    empty = np.zeros((0,), dtype=np.int32)
    empty_selection = FixedPAOSubspaceSelection(
        empty, empty, empty, empty, empty, empty
    )
    fragment = IAOMP2FragmentStaticSelection(
        fragment_index=0,
        iao_indices=empty,
        fragment_atoms=np.asarray([0], dtype=np.int32),
        strong_fragments=np.asarray([0], dtype=np.int32),
        extended_atoms=np.asarray([0], dtype=np.int32),
        extended_ao_indices=empty,
        pao_center_atoms=empty,
        strong_occ_union_keep=empty,
        strong_occ_metric_keep=empty,
        strong_virtual=empty_selection,
        primary_atoms=empty,
        primary_ao_indices=empty,
        primary_bp_atoms=empty,
        weak_weight_eigen_indices=empty,
        weak_weight_degenerate_blocks=(),
        weak_occ_norm_keep=empty,
        weak_occ_span_metric_keep=empty,
        weak_virtual=None,
    )
    static = IAOFragmentMP2StaticSelections(
        frozen=None,
        thresholds=IAOFragmentMP2Thresholds(),
        active_occ_indices=empty,
        active_vir_indices=empty,
        pao_projected_out_indices=empty,
        pao_parent_ao_indices=empty,
        ao2pao_map=empty,
        frag_lolist=(empty,),
        frag_atmlist=(None,),
        strong_mask=np.ones((1, 1), dtype=bool),
        fragments=(fragment,),
    )
    common = IAOFragmentMP2ContinuousData(
        s1e=jnp.eye(1),
        fock=jnp.eye(1),
        occupied_coeff=jnp.zeros((1, 0)),
        virtual_coeff=jnp.zeros((1, 0)),
        occupied_energy=jnp.zeros((0,)),
        virtual_energy=jnp.zeros((0,)),
        iao_coeff=jnp.zeros((1, 0)),
        pao_coeff=jnp.zeros((1, 0)),
        fragment_occupied_data=(),
    )
    domain = IAOMP2StrongDomain(
        occupied_coeff=jnp.zeros((1, 0)),
        virtual_coeff=jnp.zeros((1, 0)),
        occupied_energy=jnp.zeros((0,)),
        virtual_energy=jnp.zeros((0,)),
        target_projection=jnp.zeros((0, 0)),
        target_weight=jnp.zeros((0, 0)),
        partner_weight=jnp.zeros((0, 0)),
    )
    sentinel_selection = object()
    scratch_paths = []

    def fake_density(_mf, _domain, _static, _fragment_index, *, lov_scratch_dir):
        scratch_path = Path(lov_scratch_dir)
        assert scratch_path.is_dir()
        (scratch_path / "local_lov.h5").touch()
        scratch_paths.append(scratch_path)
        return iao_lis.IAOMP2Density(jnp.zeros((0, 0)), jnp.zeros((0, 0)))

    def fake_reference(*_args, **_kwargs):
        assert scratch_paths[-1].is_dir()
        assert (scratch_paths[-1] / "local_lov.h5").is_file()
        if fail_selection:
            raise RuntimeError("injected static rank-selection failure")
        return sentinel_selection

    monkeypatch.setattr(pyscf_lib.param, "TMPDIR", str(tmp_path))
    monkeypatch.setattr(iao_lis, "strong_domain_mp2_density", fake_density)
    monkeypatch.setattr(
        iao_lis,
        "_domain_density_in_active_spaces",
        lambda *_args: (jnp.zeros((0, 0)), jnp.zeros((0, 0))),
    )
    monkeypatch.setattr(
        iao_lis, "_reference_fragment_selection", fake_reference
    )

    if fail_selection:
        with pytest.raises(
            RuntimeError, match="injected static rank-selection failure"
        ):
            build_iao_lis_fragment_static_selection(
                object(), static, 0, common=common, domain=domain
            )
    else:
        actual = build_iao_lis_fragment_static_selection(
            object(), static, 0, common=common, domain=domain
        )
        assert actual is sentinel_selection
    assert len(scratch_paths) == 1
    assert scratch_paths[0].name.startswith("pyscfad-lov-frag0-")
    assert not scratch_paths[0].exists()
    assert list(tmp_path.glob("pyscfad-lov-frag*")) == []


def _manual_ie_density(target_amplitudes):
    ntarget, nocc, nvir, _ = target_amplitudes.shape
    dmoo = np.zeros((nocc, nocc))
    dmvv = np.zeros((nvir, nvir))
    for target in range(ntarget):
        t = target_amplitudes[target].transpose(1, 0, 2)
        dmvv += t.reshape(nvir, -1) @ t.reshape(nvir, -1).T
        dmvv -= 0.5 * np.einsum("ajc,cjb->ab", t, t)
        dmvv += t.reshape(-1, nvir).T @ t.reshape(-1, nvir)
        dmvv -= 0.5 * np.einsum("cja,bjc->ab", t, t)

        dmoo += np.einsum("aib,ajb->ij", t, t)
        dmoo -= 0.5 * np.einsum("aib,bja->ij", t, t)
        dmoo += np.einsum("bia,bja->ij", t, t)
        dmoo -= 0.5 * np.einsum("bia,ajb->ij", t, t)
    return 0.5 * (dmoo + dmoo.T), 0.5 * (dmvv + dmvv.T)


def test_target_conditioned_density_matches_ie_contractions_and_rotation():
    rng = np.random.default_rng(814)
    nocc, nvir, ntarget = 4, 5, 3
    amplitudes = rng.normal(scale=0.08, size=(nocc, nocc, nvir, nvir))
    target = rng.normal(scale=0.4, size=(ntarget, nocc))

    density = target_conditioned_mp2_density_from_amplitudes(
        jnp.asarray(amplitudes), jnp.asarray(target)
    )
    target_amplitudes = np.einsum(
        "Ip,prab->Irab", target, amplitudes
    )
    dmoo_ref, dmvv_ref = _manual_ie_density(target_amplitudes)
    np.testing.assert_allclose(density.occupied, dmoo_ref, atol=2e-13)
    np.testing.assert_allclose(density.virtual, dmvv_ref, atol=2e-13)

    rotation, _ = np.linalg.qr(rng.normal(size=(ntarget, ntarget)))
    rotated = target_conditioned_mp2_density_from_amplitudes(
        jnp.asarray(amplitudes), jnp.asarray(rotation @ target)
    )
    np.testing.assert_allclose(
        rotated.occupied, density.occupied, atol=2e-13
    )
    np.testing.assert_allclose(
        rotated.virtual, density.virtual, atol=2e-13
    )


def _dense_lov_density(lov, occupied_energy, virtual_energy, target):
    integrals = jnp.einsum("Lpa,Lrb->prab", lov, lov, optimize=True)
    denominator = (
        occupied_energy[:, None, None, None]
        + occupied_energy[None, :, None, None]
        - virtual_energy[None, None, :, None]
        - virtual_energy[None, None, None, :]
    )
    return target_conditioned_mp2_density_from_amplitudes(
        integrals / denominator, target
    )


def _write_pair_major_lov(path, lov):
    naux, nocc, nvir = lov.shape
    pair_major = np.asarray(lov).transpose(1, 2, 0).reshape(
        nocc * nvir, naux
    )
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("lov", data=pair_major)


@pytest.mark.parametrize("block_nvir", [1, 2, 3, 8])
def test_blocked_lov_density_matches_dense_amplitudes(block_nvir):
    rng = np.random.default_rng(291)
    naux, nocc, nvir, ntarget = 8, 4, 5, 3
    lov = jnp.asarray(rng.normal(scale=0.12, size=(naux, nocc, nvir)))
    e_occ = jnp.asarray(-np.linspace(1.5, 0.7, nocc))
    e_vir = jnp.asarray(np.linspace(0.2, 1.3, nvir))
    target = jnp.asarray(rng.normal(scale=0.3, size=(ntarget, nocc)))

    dense = _dense_lov_density(lov, e_occ, e_vir, target)
    blocked = strong_domain_mp2_density_from_lov(
        lov, e_occ, e_vir, target, block_nvir=block_nvir
    )
    np.testing.assert_allclose(blocked.occupied, dense.occupied, atol=3e-12)
    np.testing.assert_allclose(blocked.virtual, dense.virtual, atol=3e-12)


@pytest.mark.parametrize("block_nvir", [1, 2, 3, 7])
def test_h5_primal_lov_density_matches_in_memory_blocks(tmp_path, block_nvir):
    rng = np.random.default_rng(1291)
    naux, nocc, nvir, ntarget = 7, 3, 5, 2
    lov = rng.normal(scale=0.12, size=(naux, nocc, nvir))
    e_occ = -np.linspace(1.5, 0.7, nocc)
    e_vir = np.linspace(0.2, 1.3, nvir)
    target = rng.normal(scale=0.3, size=(ntarget, nocc))
    path = tmp_path / "lov.h5"
    _write_pair_major_lov(path, lov)

    reference = strong_domain_mp2_density_from_lov(
        lov, e_occ, e_vir, target, block_nvir=block_nvir
    )
    actual = iao_lis._strong_domain_mp2_density_h5_primal(
        str(path), e_occ, e_vir, target,
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=block_nvir,
    )

    assert isinstance(actual, iao_lis.IAOMP2Density)
    np.testing.assert_allclose(actual.occupied, reference.occupied, atol=3e-12)
    np.testing.assert_allclose(actual.virtual, reference.virtual, atol=3e-12)


def test_h5_primal_lov_density_preserves_complex_algebra(tmp_path):
    rng = np.random.default_rng(1292)
    naux, nocc, nvir, ntarget = 6, 3, 5, 2
    lov = (
        rng.normal(scale=0.09, size=(naux, nocc, nvir))
        + 1j * rng.normal(scale=0.09, size=(naux, nocc, nvir))
    )
    e_occ = -np.linspace(1.4, 0.6, nocc)
    e_vir = np.linspace(0.3, 1.1, nvir)
    target = (
        rng.normal(scale=0.2, size=(ntarget, nocc))
        + 1j * rng.normal(scale=0.2, size=(ntarget, nocc))
    )
    path = tmp_path / "complex-lov.h5"
    _write_pair_major_lov(path, lov)

    reference = strong_domain_mp2_density_from_lov(
        lov, e_occ, e_vir, target, block_nvir=2
    )
    actual = iao_lis._strong_domain_mp2_density_h5_primal(
        str(path), e_occ, e_vir, target,
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=2,
    )

    np.testing.assert_allclose(actual.occupied, reference.occupied, atol=3e-12)
    np.testing.assert_allclose(actual.virtual, reference.virtual, atol=3e-12)


@pytest.mark.parametrize(
    "naux,nocc,nvir,ntarget,occupied_shape,virtual_shape",
    [
        (0, 2, 3, 1, (2, 2), (3, 3)),
        (2, 0, 3, 1, (0, 0), (3, 3)),
        (2, 2, 0, 1, (2, 2), (0, 0)),
        (2, 2, 3, 0, (2, 2), (3, 3)),
    ],
)
def test_h5_primal_lov_density_handles_empty_dimensions(
    tmp_path, naux, nocc, nvir, ntarget, occupied_shape, virtual_shape
):
    lov = np.zeros((naux, nocc, nvir))
    path = tmp_path / f"empty-{naux}-{nocc}-{nvir}-{ntarget}.h5"
    _write_pair_major_lov(path, lov)

    density = iao_lis._strong_domain_mp2_density_h5_primal(
        str(path),
        -np.arange(nocc, dtype=float) - 1.0,
        np.arange(nvir, dtype=float) + 0.2,
        np.zeros((ntarget, nocc)),
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=2,
    )

    assert density.occupied.shape == occupied_shape
    assert density.virtual.shape == virtual_shape
    np.testing.assert_array_equal(density.occupied, 0)
    np.testing.assert_array_equal(density.virtual, 0)


def test_h5_primal_reads_only_contiguous_pair_major_slabs(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(1293)
    naux, nocc, nvir, ntarget, block_nvir = 4, 3, 5, 2, 2
    lov = rng.normal(scale=0.1, size=(naux, nocc, nvir))
    path = tmp_path / "tracked-lov.h5"
    _write_pair_major_lov(path, lov)
    original_array = h5py.Dataset.__array__
    original_getitem = h5py.Dataset.__getitem__
    selections = []

    def forbidden_array(dataset, *args, **kwargs):
        if dataset.name == "/lov":
            raise AssertionError("the complete /lov dataset was converted")
        return original_array(dataset, *args, **kwargs)

    def tracked_getitem(dataset, key):
        if dataset.name == "/lov":
            selections.append(key)
        return original_getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__array__", forbidden_array)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked_getitem)

    iao_lis._strong_domain_mp2_density_h5_primal(
        str(path),
        -np.linspace(1.3, 0.7, nocc),
        np.linspace(0.2, 1.2, nvir),
        rng.normal(size=(ntarget, nocc)),
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=block_nvir,
    )

    nblock = (nvir + block_nvir - 1) // block_nvir
    selected_rows = 0
    for key in selections:
        assert isinstance(key, tuple) and len(key) == 2
        row_slice, aux_slice = key
        assert isinstance(row_slice, slice)
        assert row_slice.step in (None, 1)
        assert aux_slice == slice(None)
        row_count = row_slice.stop - row_slice.start
        assert row_count <= max(nvir, block_nvir)
        selected_rows += row_count
    assert len(selections) == 2 * nocc * nblock
    assert selected_rows * naux * lov.dtype.itemsize == (
        nocc * naux * nvir * (nblock + 1) * lov.dtype.itemsize
    )


def test_h5_primal_profiles_exact_io_and_kernel_timing(monkeypatch, tmp_path):
    rng = np.random.default_rng(1294)
    naux, nocc, nvir, ntarget, block_nvir = 4, 2, 5, 3, 2
    lov = rng.normal(size=(naux, nocc, nvir))
    path = tmp_path / "profiled-lov.h5"
    _write_pair_major_lov(path, lov)
    token = object()
    finished = []
    monkeypatch.setattr(iao_lis.resource_profile, "start", lambda: token)
    monkeypatch.setattr(
        iao_lis.resource_profile, "finish",
        lambda phase, before, **details: finished.append(
            (phase, before, details)
        ),
    )

    iao_lis._strong_domain_mp2_density_h5_primal(
        str(path),
        -np.linspace(1.4, 0.8, nocc),
        np.linspace(0.2, 1.2, nvir),
        rng.normal(size=(ntarget, nocc)),
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=block_nvir,
    )

    assert len(finished) == 1
    phase, before, details = finished[0]
    nblock = (nvir + block_nvir - 1) // block_nvir
    assert phase == "iao_lis.strong_domain_mp2_density_h5_primal"
    assert before is token
    expected_details = {
        "lov_h5_path_basename": path.name,
        "lov_disk_mib": lov.size * lov.dtype.itemsize / 1024.0**2,
        "lov_bar_disk_mib": 0.0,
        "z_disk_mib": 0.0,
        "naux": naux,
        "nocc": nocc,
        "nvir": nvir,
        "ntarget": ntarget,
        "block_nvir": block_nvir,
        "block_count": nblock,
        "hdf5_bytes_read": (
            nocc * naux * nvir * (nblock + 1) * lov.dtype.itemsize
        ),
        "hdf5_bytes_written": 0,
    }
    assert {key: details[key] for key in expected_details} == expected_details
    assert set(details) == {
        *expected_details,
        "hdf5_read_seconds",
        "hdf5_write_seconds",
        "mp2_kernel_seconds",
    }
    assert details["hdf5_read_seconds"] >= 0.0
    assert details["hdf5_write_seconds"] == 0.0
    assert details["mp2_kernel_seconds"] >= 0.0


def test_h5_primal_read_timing_excludes_host_transpose_and_assignment(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(1295)
    naux, nocc, nvir, block_nvir = 3, 2, 3, 2
    lov = rng.normal(size=(naux, nocc, nvir))
    path = tmp_path / "timed-lov.h5"
    _write_pair_major_lov(path, lov)
    original_getitem = h5py.Dataset.__getitem__
    original_empty = iao_lis.onp.empty
    clock = [0.0]
    finished = []

    class DestinationWithCostlyAssignment(np.ndarray):
        def __setitem__(self, key, value):
            clock[0] += 100.0
            return super().__setitem__(key, value)

    class FakeTime:
        @staticmethod
        def perf_counter():
            return clock[0]

    def tracked_getitem(dataset, key):
        result = original_getitem(dataset, key)
        if dataset.name == "/lov":
            clock[0] += 2.0
        return result

    def tracked_empty(*args, **kwargs):
        return original_empty(*args, **kwargs).view(
            DestinationWithCostlyAssignment
        )

    class TrackedNumpy:
        integer = np.integer
        empty = staticmethod(tracked_empty)
        asarray = staticmethod(np.asarray)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked_getitem)
    monkeypatch.setattr(iao_lis, "onp", TrackedNumpy)
    monkeypatch.setattr(iao_lis, "time", FakeTime)
    monkeypatch.setattr(iao_lis.resource_profile, "start", lambda: object())
    monkeypatch.setattr(
        iao_lis.resource_profile, "finish",
        lambda phase, before, **details: finished.append(details),
    )

    iao_lis._strong_domain_mp2_density_h5_primal(
        str(path),
        -np.linspace(1.3, 0.7, nocc),
        np.linspace(0.2, 1.0, nvir),
        rng.normal(size=(1, nocc)),
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=block_nvir,
    )

    nblock = (nvir + block_nvir - 1) // block_nvir
    nreads = 2 * nocc * nblock
    assert finished[0]["hdf5_read_seconds"] == 2.0 * nreads


def test_h5_primal_times_hermitization_and_synchronizes_return(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(1296)
    naux, nocc, nvir = 3, 2, 3
    lov = rng.normal(size=(naux, nocc, nvir))
    path = tmp_path / "synchronized-lov.h5"
    _write_pair_major_lov(path, lov)
    original_hermitize = iao_lis._hermitize
    original_block_until_ready = jax.block_until_ready
    original_perf_counter = iao_lis.time.perf_counter
    events = []
    synchronized = []

    def tracked_hermitize(array):
        events.append("hermitize")
        return original_hermitize(array)

    def tracked_block_until_ready(value):
        events.append("block_until_ready")
        synchronized.append(value)
        return original_block_until_ready(value)

    def tracked_perf_counter():
        events.append("perf_counter")
        return original_perf_counter()

    monkeypatch.setattr(iao_lis, "_hermitize", tracked_hermitize)
    monkeypatch.setattr(jax, "block_until_ready", tracked_block_until_ready)
    monkeypatch.setattr(iao_lis.time, "perf_counter", tracked_perf_counter)
    monkeypatch.setattr(iao_lis.resource_profile, "start", lambda: object())
    monkeypatch.setattr(
        iao_lis.resource_profile, "finish",
        lambda phase, before, **details: events.append("profile_finish"),
    )

    density = iao_lis._strong_domain_mp2_density_h5_primal(
        str(path),
        -np.linspace(1.2, 0.8, nocc),
        np.linspace(0.2, 1.0, nvir),
        rng.normal(size=(1, nocc)),
        naux=naux, nocc=nocc, nvir=nvir, block_nvir=2,
    )

    assert events[-6:] == [
        "perf_counter",
        "hermitize",
        "hermitize",
        "block_until_ready",
        "perf_counter",
        "profile_finish",
    ]
    assert synchronized[-1] is density


def _density_scalar(density, occupied_weight, virtual_weight):
    return (
        jnp.sum(occupied_weight * density.occupied)
        + jnp.sum(virtual_weight * density.virtual)
    )


def _symmetric_weight(rng, size):
    weight = rng.normal(size=(size, size))
    return jnp.asarray(0.5 * (weight + weight.T))


def _general_weight(rng, size):
    return jnp.asarray(rng.normal(size=(size, size)))


def _weighted_density_objective(density, occupied_weight, virtual_weight):
    return jnp.real(
        jnp.vdot(occupied_weight, density.occupied)
        + jnp.vdot(virtual_weight, density.virtual)
    )


def _assert_pytree_allclose(actual, expected, **kwargs):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        np.testing.assert_allclose(
            np.asarray(actual_leaf), np.asarray(expected_leaf), **kwargs
        )


def _actual_h5_custom_density_problem(path):
    mol = gto.Mole(
        atom="O 0 0 0; H 0 0 1; H 0 1 0",
        basis="sto-3g",
        verbose=0,
        max_memory=200,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    atmlst = np.asarray([0, 1], dtype=np.int32)
    fake_mol = iao_lis.lno_base.make_local_mol(mol, atmlst)
    auxmol = iao_lis.lno_base.df_addons.make_auxmol(
        fake_mol, "weigend"
    )
    rng = np.random.default_rng(1401)
    nocc = 2
    nvir = 3
    local_coeff = jnp.asarray(
        rng.normal(scale=0.2, size=(fake_mol.nao, nocc + nvir))
    )
    occupied_energy = jnp.asarray([-1.3, -0.8])
    virtual_energy = jnp.asarray([0.2, 0.7, 1.1])
    target_projection = jnp.asarray(
        rng.normal(scale=0.3, size=(2, nocc))
    )
    differentiable = (
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
    )
    static = (nocc, str(path), float(mol.max_memory), 2)
    return differentiable, static


def _h5_custom_density(differentiable, static):
    return iao_lis._strong_domain_mp2_density_h5(
        *differentiable, *static
    )


def _in_memory_direct_density(differentiable, static):
    (
        fake_mol,
        auxmol,
        local_coeff,
        occupied_energy,
        virtual_energy,
        target_projection,
    ) = differentiable
    nocc, _, max_memory, block_nvir = static
    nvir = local_coeff.shape[1] - nocc
    lov = iao_lis.lno_base._local_direct_nr_e2(
        fake_mol,
        auxmol,
        local_coeff,
        max_memory,
        (0, nocc, nocc, local_coeff.shape[1]),
    ).reshape(auxmol.nao, nocc, nvir)
    return strong_domain_mp2_density_from_lov(
        lov,
        occupied_energy,
        virtual_energy,
        target_projection,
        block_nvir=block_nvir,
    )


@pytest.mark.parametrize("block_nvir", [1, 2, 5])
def test_h5_custom_density_weighted_value_and_all_bars_match_in_memory(
    tmp_path, block_nvir
):
    differentiable, static = _actual_h5_custom_density_problem(
        tmp_path / f"weighted-block-{block_nvir}.h5"
    )
    static = (*static[:3], block_nvir)
    rng = np.random.default_rng(1402)
    occupied_weight = _symmetric_weight(rng, 2)
    virtual_weight = _symmetric_weight(rng, 3)

    def disk_objective(*args):
        return _weighted_density_objective(
            _h5_custom_density(args, static),
            occupied_weight,
            virtual_weight,
        )

    def memory_objective(*args):
        return _weighted_density_objective(
            _in_memory_direct_density(args, static),
            occupied_weight,
            virtual_weight,
        )

    argnums = tuple(range(len(differentiable)))
    disk_value, disk_bars = jax.value_and_grad(
        disk_objective, argnums=argnums
    )(*differentiable)
    memory_value, memory_bars = jax.value_and_grad(
        memory_objective, argnums=argnums
    )(*differentiable)

    np.testing.assert_allclose(
        disk_value, memory_value, rtol=3e-10, atol=3e-11
    )
    for disk_bar, memory_bar in zip(disk_bars, memory_bars):
        _assert_pytree_allclose(
            disk_bar, memory_bar, rtol=3e-10, atol=3e-11
        )


def test_h5_custom_density_combined_directional_finite_difference(tmp_path):
    differentiable, static = _actual_h5_custom_density_problem(
        tmp_path / "directional.h5"
    )
    fake_mol, auxmol, *array_arguments = differentiable
    rng = np.random.default_rng(1403)
    directions = tuple(
        jnp.asarray(rng.normal(size=value.shape))
        for value in array_arguments
    )
    occupied_weight = _symmetric_weight(rng, 2)
    virtual_weight = _symmetric_weight(rng, 3)

    def objective(*arrays):
        return _weighted_density_objective(
            _h5_custom_density((fake_mol, auxmol, *arrays), static),
            occupied_weight,
            virtual_weight,
        )

    argnums = tuple(range(len(array_arguments)))
    gradient = jax.grad(objective, argnums=argnums)(*array_arguments)
    analytic = sum(
        jnp.real(jnp.vdot(bar, direction))
        for bar, direction in zip(gradient, directions)
    )
    step = 2e-5
    plus = tuple(
        value + step * direction
        for value, direction in zip(array_arguments, directions)
    )
    minus = tuple(
        value - step * direction
        for value, direction in zip(array_arguments, directions)
    )
    finite_difference = (objective(*plus) - objective(*minus)) / (2 * step)
    np.testing.assert_allclose(
        analytic, finite_difference, rtol=5e-7, atol=5e-9
    )


def test_h5_custom_density_parent_coordinate_chain_and_one_output_bar(
    tmp_path,
):
    path = tmp_path / "parent-coordinate.h5"
    differentiable, static = _actual_h5_custom_density_problem(path)
    parent_mol = differentiable[0]
    _, _, local_coeff, occupied_energy, virtual_energy, target_projection = (
        differentiable
    )
    atmlst = np.arange(parent_mol.natm, dtype=np.int32)
    occupied_weight = _general_weight(np.random.default_rng(1408), 2)

    def local_arguments(mol):
        fake_mol = iao_lis.lno_base.make_local_mol(mol, atmlst)
        auxmol = iao_lis.lno_base.df_addons.make_auxmol(
            fake_mol, "weigend"
        )
        return (
            fake_mol,
            auxmol,
            local_coeff,
            occupied_energy,
            virtual_energy,
            target_projection,
        )

    def disk_objective(mol):
        density = _h5_custom_density(local_arguments(mol), static)
        return jnp.real(jnp.vdot(occupied_weight, density.occupied))

    def memory_objective(mol):
        density = _in_memory_direct_density(local_arguments(mol), static)
        return jnp.real(jnp.vdot(occupied_weight, density.occupied))

    disk_value, disk_bar = jax.value_and_grad(disk_objective)(parent_mol)
    memory_value, memory_bar = jax.value_and_grad(memory_objective)(parent_mol)
    np.testing.assert_allclose(
        disk_value, memory_value, rtol=3e-10, atol=3e-11
    )
    np.testing.assert_allclose(
        disk_bar.coords, memory_bar.coords, rtol=3e-9, atol=3e-10
    )


def test_h5_density_lov_backward_matches_dense_gradient_with_tail_block(
    tmp_path, monkeypatch,
):
    rng = np.random.default_rng(1404)
    naux, nocc, nvir, ntarget, block_nvir = 5, 3, 5, 2, 2
    lov = jnp.asarray(
        rng.normal(scale=0.12, size=(naux, nocc, nvir))
    )
    occupied_energy = jnp.asarray(-np.linspace(1.5, 0.7, nocc))
    virtual_energy = jnp.asarray(np.linspace(0.2, 1.3, nvir))
    target_projection = jnp.asarray(
        rng.normal(scale=0.3, size=(ntarget, nocc))
    )
    occupied_weight = _general_weight(rng, nocc)
    virtual_weight = _general_weight(rng, nvir)
    path = tmp_path / "lov-backward.h5"
    _write_pair_major_lov(path, lov)
    profile_token = object()
    profile_events = []
    monkeypatch.setattr(
        iao_lis.resource_profile, "start", lambda: profile_token
    )
    monkeypatch.setattr(
        iao_lis.resource_profile,
        "finish",
        lambda phase, before, **details: profile_events.append(
            (phase, before, details)
        ),
    )

    def reference_objective(*args):
        return _weighted_density_objective(
            strong_domain_mp2_density_from_lov(
                *args, block_nvir=block_nvir
            ),
            occupied_weight,
            virtual_weight,
        )

    reference_bars = jax.grad(
        reference_objective, argnums=(0, 1, 2, 3)
    )(lov, occupied_energy, virtual_energy, target_projection)
    actual_energy_projection_bars = (
        iao_lis._strong_domain_mp2_density_h5_lov_bwd(
            str(path),
            occupied_energy,
            virtual_energy,
            target_projection,
            iao_lis.IAOMP2Density(occupied_weight, virtual_weight),
            naux=naux,
            nocc=nocc,
            nvir=nvir,
            block_nvir=block_nvir,
        )
    )
    with h5py.File(path, "r") as h5file:
        lov_bar = jnp.asarray(h5file["lov_bar"][:]).reshape(
            nocc, nvir, naux
        ).transpose(2, 0, 1)

    np.testing.assert_allclose(
        lov_bar, reference_bars[0], rtol=3e-10, atol=3e-11
    )
    for actual, expected in zip(
        actual_energy_projection_bars, reference_bars[1:]
    ):
        np.testing.assert_allclose(
            actual, expected, rtol=3e-10, atol=3e-11
        )
    assert len(profile_events) == 1
    phase, before, details = profile_events[0]
    assert phase == "iao_lis.strong_domain_mp2_density_h5_lov_bwd"
    assert before is profile_token
    assert details["lov_h5_path_basename"] == path.name
    assert details["lov_disk_mib"] > 0.0
    assert details["lov_bar_disk_mib"] > 0.0
    assert details["z_disk_mib"] == 0.0
    assert details["block_nvir"] == block_nvir
    assert details["block_count"] == 3
    assert details["hdf5_bytes_read"] > 0
    assert details["hdf5_bytes_written"] > 0
    assert details["hdf5_read_seconds"] >= 0.0
    assert details["hdf5_write_seconds"] >= 0.0


def test_h5_density_lov_backward_uses_only_rows_and_virtual_blocks(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(1405)
    naux, nocc, nvir, ntarget, block_nvir = 4, 3, 5, 2, 2
    lov = rng.normal(size=(naux, nocc, nvir))
    path = tmp_path / "bounded-density-backward.h5"
    _write_pair_major_lov(path, lov)
    original_array = h5py.Dataset.__array__
    original_getitem = h5py.Dataset.__getitem__
    original_setitem = h5py.Dataset.__setitem__
    reads = {"/lov": [], "/lov_bar": []}
    writes = {"/lov_bar": []}

    def forbidden_array(dataset, *args, **kwargs):
        if dataset.name in reads:
            raise AssertionError(f"complete conversion of {dataset.name}")
        return original_array(dataset, *args, **kwargs)

    def tracked_getitem(dataset, key):
        if dataset.name in reads:
            reads[dataset.name].append(key)
        return original_getitem(dataset, key)

    def tracked_setitem(dataset, key, value):
        if dataset.name in writes:
            writes[dataset.name].append(key)
        return original_setitem(dataset, key, value)

    monkeypatch.setattr(h5py.Dataset, "__array__", forbidden_array)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked_getitem)
    monkeypatch.setattr(h5py.Dataset, "__setitem__", tracked_setitem)

    iao_lis._strong_domain_mp2_density_h5_lov_bwd(
        str(path),
        -jnp.linspace(1.4, 0.8, nocc),
        jnp.linspace(0.2, 1.1, nvir),
        jnp.asarray(rng.normal(size=(ntarget, nocc))),
        iao_lis.IAOMP2Density(
            _symmetric_weight(rng, nocc),
            _symmetric_weight(rng, nvir),
        ),
        naux=naux,
        nocc=nocc,
        nvir=nvir,
        block_nvir=block_nvir,
    )

    npair = nocc * nvir
    assert reads["/lov"]
    assert reads["/lov_bar"]
    assert writes["/lov_bar"]
    for selections in (*reads.values(), *writes.values()):
        for key in selections:
            assert isinstance(key, tuple) and len(key) == 2
            row_slice, aux_slice = key
            assert isinstance(row_slice, slice)
            assert row_slice.step in (None, 1)
            assert aux_slice == slice(None)
            row_count = np.arange(npair)[row_slice].size
            assert row_count <= nvir
            assert row_count < npair


def test_h5_density_lov_backward_flushes_only_after_all_virtual_blocks(
    monkeypatch, tmp_path
):
    rng = np.random.default_rng(1409)
    naux, nocc, nvir, block_nvir = 4, 3, 5, 2
    path = tmp_path / "single-flush-density-backward.h5"
    _write_pair_major_lov(
        path, rng.normal(size=(naux, nocc, nvir))
    )
    original_flush = h5py.File.flush
    flushes = []

    def tracked_flush(h5file):
        if h5file.filename == str(path):
            flushes.append(tuple(h5file.keys()))
        return original_flush(h5file)

    monkeypatch.setattr(h5py.File, "flush", tracked_flush)
    iao_lis._strong_domain_mp2_density_h5_lov_bwd(
        str(path),
        -jnp.linspace(1.4, 0.8, nocc),
        jnp.linspace(0.2, 1.1, nvir),
        jnp.asarray(rng.normal(size=(2, nocc))),
        iao_lis.IAOMP2Density(
            _general_weight(rng, nocc), _general_weight(rng, nvir)
        ),
        naux=naux,
        nocc=nocc,
        nvir=nvir,
        block_nvir=block_nvir,
    )

    assert flushes == [("lov", "lov_bar")]


def test_h5_density_lov_backward_zero_target_writes_zero_bar(tmp_path):
    rng = np.random.default_rng(1406)
    naux, nocc, nvir = 3, 2, 4
    path = tmp_path / "zero-target-backward.h5"
    _write_pair_major_lov(
        path, rng.normal(size=(naux, nocc, nvir))
    )
    bars = iao_lis._strong_domain_mp2_density_h5_lov_bwd(
        str(path),
        -jnp.linspace(1.2, 0.8, nocc),
        jnp.linspace(0.2, 1.0, nvir),
        jnp.zeros((0, nocc)),
        iao_lis.IAOMP2Density(jnp.ones((nocc, nocc)),
                              jnp.ones((nvir, nvir))),
        naux=naux,
        nocc=nocc,
        nvir=nvir,
        block_nvir=3,
    )
    with h5py.File(path, "r") as h5file:
        np.testing.assert_array_equal(h5file["lov_bar"][:], 0)
    for bar in bars:
        np.testing.assert_array_equal(bar, 0)


def test_h5_custom_density_residual_is_bounded_and_pullback_is_repeatable(
    tmp_path,
):
    path = tmp_path / "repeatable.h5"
    differentiable, static = _actual_h5_custom_density_problem(path)

    def density_fn(*args):
        return _h5_custom_density(args, static)

    density, pullback = jax.vjp(density_fn, *differentiable)
    residual_arrays = [
        value
        for name in ("args_res", "opaque_residuals")
        for value in jax.tree_util.tree_leaves(getattr(pullback, name, ()))
        if hasattr(value, "shape")
    ]
    assert residual_arrays
    nocc = static[0]
    nvir = differentiable[2].shape[1] - nocc
    naux = differentiable[1].nao
    assert all(value.size < naux * nocc * nvir
               for value in residual_arrays)
    ntarget = differentiable[5].shape[0]
    block_nvir = min(static[3], nvir)
    nblock = (nvir + block_nvir - 1) // block_nvir
    amplitude_block_shape = (ntarget, nocc, nvir, block_nvir)
    assert not any(
        value.ndim >= 5
        and tuple(value.shape[-4:]) == amplitude_block_shape
        and np.prod(value.shape[:-4]) >= nblock
        for value in residual_arrays
    )

    rng = np.random.default_rng(1407)
    density_bar = iao_lis.IAOMP2Density(
        _symmetric_weight(rng, density.occupied.shape[0]),
        _symmetric_weight(rng, density.virtual.shape[0]),
    )
    first_bars = pullback(density_bar)
    with h5py.File(path, "r") as h5file:
        first_lov_bar = h5file["lov_bar"][:]
        assert set(h5file) == {"lov", "lov_bar", "z"}
    second_bars = pullback(
        iao_lis.IAOMP2Density(
            2 * density_bar.occupied, 2 * density_bar.virtual
        )
    )
    with h5py.File(path, "r") as h5file:
        second_lov_bar = h5file["lov_bar"][:]
        assert set(h5file) == {"lov", "lov_bar", "z"}

    np.testing.assert_allclose(
        second_lov_bar, 2 * first_lov_bar, rtol=3e-10, atol=3e-11
    )
    for second_bar, first_bar in zip(second_bars, first_bars):
        _assert_pytree_allclose(
            second_bar, jax.tree_util.tree_map(lambda value: 2 * value,
                                               first_bar),
            rtol=3e-9, atol=3e-10,
        )


def test_h5_custom_density_failed_pullback_removes_derivative_datasets(
    monkeypatch, tmp_path
):
    path = tmp_path / "failed-pullback.h5"
    differentiable, static = _actual_h5_custom_density_problem(path)

    def density_fn(*args):
        return _h5_custom_density(args, static)

    density, pullback = jax.vjp(density_fn, *differentiable)

    def fail_local_reverse(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected local reverse failure")

    monkeypatch.setattr(
        iao_lis.lno_base,
        "_local_direct_nr_e2_h5_bwd",
        fail_local_reverse,
    )
    with pytest.raises(RuntimeError, match="injected local reverse failure"):
        pullback(
            iao_lis.IAOMP2Density(
                jnp.ones_like(density.occupied),
                jnp.ones_like(density.virtual),
            )
        )
    with h5py.File(path, "r") as h5file:
        assert set(h5file) == {"lov"}


@pytest.mark.parametrize(
    ("static_index", "invalid_value", "name", "expected_message"),
    (
        (0, 1.5, "nocc", "nocc must be a nonnegative integer"),
        (3, 1.5, "block_nvir", "block_nvir must be a positive integer"),
    ),
)
def test_h5_custom_density_rejects_fractional_static_controls(
    tmp_path, static_index, invalid_value, name, expected_message
):
    differentiable, static = _actual_h5_custom_density_problem(
        tmp_path / f"invalid-{name}.h5"
    )
    static = list(static)
    static[static_index] = invalid_value
    with pytest.raises(ValueError, match=expected_message):
        _h5_custom_density(differentiable, tuple(static))


def test_blocked_lov_density_value_and_grad_matches_dense_reference():
    rng = np.random.default_rng(977)
    naux, nocc, nvir, ntarget = 6, 3, 5, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(scale=0.12, size=(naux, nocc, nvir)),
        -np.linspace(1.4, 0.6, nocc),
        np.linspace(0.2, 1.1, nvir),
        rng.normal(scale=0.3, size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def dense_scalar(*inputs):
        return _density_scalar(
            _dense_lov_density(*inputs), occupied_weight, virtual_weight
        )

    def blocked_scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=2
            ), occupied_weight, virtual_weight
        )

    dense_value, dense_gradient = jax.value_and_grad(
        dense_scalar, argnums=(0, 1, 2, 3)
    )(*arguments)
    blocked_value, blocked_gradient = jax.value_and_grad(
        blocked_scalar, argnums=(0, 1, 2, 3)
    )(*arguments)
    np.testing.assert_allclose(blocked_value, dense_value, rtol=3e-10, atol=3e-11)
    for actual, expected in zip(blocked_gradient, dense_gradient):
        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-11)


def test_blocked_lov_density_combined_directional_finite_difference():
    rng = np.random.default_rng(978)
    naux, nocc, nvir, ntarget = 6, 3, 5, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(scale=0.12, size=(naux, nocc, nvir)),
        -np.linspace(1.4, 0.6, nocc),
        np.linspace(0.2, 1.1, nvir),
        rng.normal(scale=0.3, size=(ntarget, nocc)),
    )))
    directions = tuple(map(jnp.asarray, (
        rng.normal(size=(naux, nocc, nvir)),
        rng.normal(size=nocc),
        rng.normal(size=nvir),
        rng.normal(size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=2
            ), occupied_weight, virtual_weight
        )

    gradient = jax.grad(scalar, argnums=(0, 1, 2, 3))(*arguments)
    analytic = sum(jnp.vdot(value, direction)
                   for value, direction in zip(gradient, directions))
    step = 2e-5
    finite_difference = (
        scalar(*(value + step * direction
                 for value, direction in zip(arguments, directions)))
        - scalar(*(value - step * direction
                   for value, direction in zip(arguments, directions)))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic, finite_difference, rtol=5e-7, atol=5e-9
    )


def test_blocked_lov_density_handles_zero_sizes_and_validates_blocks():
    zero_target = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 3, 4)), jnp.arange(3.0), jnp.arange(4.0),
        jnp.zeros((0, 3)), block_nvir=2,
    )
    assert zero_target.occupied.shape == (3, 3)
    assert zero_target.virtual.shape == (4, 4)
    np.testing.assert_array_equal(zero_target.occupied, 0)
    np.testing.assert_array_equal(zero_target.virtual, 0)

    zero_occupied = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 0, 4)), jnp.zeros((0,)), jnp.arange(4.0),
        jnp.zeros((2, 0)), block_nvir=2,
    )
    assert zero_occupied.occupied.shape == (0, 0)
    assert zero_occupied.virtual.shape == (4, 4)
    np.testing.assert_array_equal(zero_occupied.virtual, 0)

    zero_virtual = strong_domain_mp2_density_from_lov(
        jnp.zeros((2, 3, 0)), jnp.arange(3.0), jnp.zeros((0,)),
        jnp.zeros((2, 3)), block_nvir=2,
    )
    assert zero_virtual.occupied.shape == (3, 3)
    assert zero_virtual.virtual.shape == (0, 0)
    np.testing.assert_array_equal(zero_virtual.occupied, 0)

    inputs = (
        jnp.zeros((2, 3, 4)),
        -jnp.arange(3.0) - 1.0,
        jnp.arange(4.0) + 0.2,
        jnp.zeros((2, 3)),
    )
    for max_memory_mb in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_memory_mb"):
            strong_domain_mp2_density_from_lov(
                *inputs, max_memory_mb=max_memory_mb
            )
    for block_nvir in (0, -1, 1.5):
        with pytest.raises(ValueError, match="block_nvir"):
            strong_domain_mp2_density_from_lov(*inputs, block_nvir=block_nvir)

    clamped = strong_domain_mp2_density_from_lov(*inputs, block_nvir=99)
    reference = _dense_lov_density(*inputs)
    np.testing.assert_allclose(clamped.occupied, reference.occupied)
    np.testing.assert_allclose(clamped.virtual, reference.virtual)


def test_blocked_lov_density_uses_nested_scans_without_full_amplitudes():
    rng = np.random.default_rng(979)
    ntarget, naux, nocc, nvir, block_nvir = 2, 7, 3, 7, 2
    arguments = tuple(map(jnp.asarray, (
        rng.normal(size=(naux, nocc, nvir)),
        -np.linspace(1.3, 0.7, nocc),
        np.linspace(0.2, 1.2, nvir),
        rng.normal(size=(ntarget, nocc)),
    )))
    occupied_weight = _symmetric_weight(rng, nocc)
    virtual_weight = _symmetric_weight(rng, nvir)

    def scalar(*inputs):
        return _density_scalar(
            strong_domain_mp2_density_from_lov(
                *inputs, block_nvir=block_nvir
            ), occupied_weight, virtual_weight
        )

    def nested_values(value):
        if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
            value = value.jaxpr
        if hasattr(value, "eqns"):
            yield value
            for equation in value.eqns:
                yield from nested_values(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                yield from nested_values(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from nested_values(item)

    forward_jaxpr = jax.make_jaxpr(scalar)(*arguments)

    def jaxpr_equations(value):
        if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
            value = value.jaxpr
        return value.eqns

    nblock = (nvir + block_nvir - 1) // block_nvir
    outer_scan, = [
        equation
        for equation in jaxpr_equations(forward_jaxpr)
        if equation.primitive.name == "scan"
        and equation.params["length"] == nblock
    ]
    outer_remat, = [
        equation
        for equation in jaxpr_equations(outer_scan.params["jaxpr"])
        if equation.primitive.name.startswith("remat")
    ]
    inner_scans = [
        equation
        for nested in nested_values(outer_remat.params)
        for equation in jaxpr_equations(nested)
        if equation.primitive.name == "scan"
        and equation.params["length"] == nocc
    ]
    assert len(inner_scans) == 1
    inner_scan = inner_scans[0]
    inner_body = jaxpr_equations(inner_scan.params["jaxpr"])
    assert len(inner_body) == 1
    assert inner_body[0].primitive.name.startswith("remat")

    gradient_jaxpr = jax.make_jaxpr(
        jax.grad(scalar, argnums=(0, 1, 2, 3))
    )(*arguments)
    all_array_shapes = {
        tuple(variable.aval.shape)
        for jaxpr in nested_values(gradient_jaxpr)
        for equation in jaxpr.eqns
        for variable in (*equation.invars, *equation.outvars)
        if hasattr(variable, "aval") and hasattr(variable.aval, "shape")
    }
    forbidden_amplitude_shapes = (
        (ntarget, nocc, nvir, nvir),
        (nocc, nocc, nvir, nvir),
    )
    for forbidden_shape in forbidden_amplitude_shapes:
        assert not any(
            shape == forbidden_shape
            or (len(shape) > len(forbidden_shape)
                and shape[-len(forbidden_shape):] == forbidden_shape)
            for shape in all_array_shapes
        )

    _, pullback = jax.vjp(scalar, *arguments)
    residuals = [
        value
        for name in ("args_res", "opaque_residuals")
        for value in getattr(pullback, name, ())
        if hasattr(value, "shape")
    ]
    lov_residuals = [
        value for value in residuals if tuple(value.shape) == arguments[0].shape
    ]
    assert len(lov_residuals) <= 1
    assert not any(tuple(value.shape) == (nocc, naux, nvir)
                   for value in residuals)
    assert not any(
        value.ndim >= 5
        and tuple(value.shape[-4:]) == (ntarget, nocc, nvir, block_nvir)
        for value in residuals
    )
    assert not any(
        value.ndim >= 4
        and tuple(value.shape[-4:]) == (nocc, ntarget, nvir, block_nvir)
        for value in residuals
    )
    pullback(jnp.ones(()))


def _water_mf(frozen=None):
    del frozen
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        """,
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _water_dimer_mf():
    mol = gto.Mole(
        atom="""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  8.0000000000
        H  0.0000000000 -0.7570000000  8.5870000000
        H  0.0000000000  0.7570000000  8.5870000000
        """,
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
    )


@pytest.mark.parametrize("frozen", [None, 1])
def test_zero_lno_threshold_recovers_full_active_hf_spaces(
    frozen, tmp_path
):
    mf = _water_mf(frozen)
    topology = build_iao_fragment_topology(
        mf,
        frozen=frozen,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    assert len(topology.frag_lolist) == 1
    mp2_static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, mp2_static)
    lis_static = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        thresh_occ=0.0,
        thresh_vir=0.0,
    )
    scratch_dir = tmp_path / "fragment-0"
    scratch_dir.mkdir()
    result = build_fragment_lis(
        mf,
        common,
        lis_static,
        0,
        lov_scratch_dir=scratch_dir,
    )

    injected = build_fragment_lis(
        mf,
        common,
        lis_static,
        0,
        domain=result.domain,
        density=iao_lis.IAOMP2Density(
            result.density_occupied_ed,
            result.density_virtual_ed,
        ),
    )
    np.testing.assert_allclose(injected.mo_coeff, result.mo_coeff)

    nocc_active = len(mp2_static.active_occ_indices)
    nvir_active = len(mp2_static.active_vir_indices)
    assert result.active_occupied_coeff.shape[1] == nocc_active
    assert result.active_virtual_coeff.shape[1] == nvir_active
    assert result.mo_coeff.shape == mf.mo_coeff.shape

    overlap = np.asarray(mf.get_ovlp())
    coeff = np.asarray(result.mo_coeff)
    np.testing.assert_allclose(
        coeff.T @ overlap @ coeff,
        np.eye(coeff.shape[1]),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        result.occupied_projector,
        np.eye(nocc_active),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        result.virtual_projector,
        np.eye(nvir_active),
        atol=2e-12,
    )

    raw_overlap = (
        result.fragment_iao_coeff.T
        @ overlap
        @ result.active_occupied_coeff
    )
    projected_overlap = (
        result.fragment_occupied_anchor.T
        @ overlap
        @ result.active_occupied_coeff
    )
    np.testing.assert_allclose(raw_overlap, projected_overlap, atol=2e-11)
    assert np.all(np.isfinite(np.asarray(result.density_occupied_ed)))
    assert np.all(np.isfinite(np.asarray(result.density_virtual_ed)))

    prescreen = strong_domain_prescreen(
        common, mp2_static, 0, domain=result.domain
    )
    np.testing.assert_array_equal(
        prescreen["strong_lmo_indices"],
        np.arange(common.iao_coeff.shape[1]),
    )
    assert prescreen["orbfragloc"] is not None
    assert prescreen["occ_prescreen_coeff"].shape[1] == nocc_active
    assert prescreen["vir_prescreen_coeff"].shape[1] == nvir_active


def test_fragment_static_selection_matches_serial_builder(monkeypatch):
    mf = _water_dimer_mf()
    topology = build_iao_fragment_topology(
        mf,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )
    assert len(topology.frag_lolist) == 2
    mp2_static = build_iao_mp2_static_selections(mf, topology)
    common = rebuild_iao_mp2_common(mf, mp2_static)
    thresholds = {
        "thresh_occ": 2e-4,
        "thresh_vir": 3e-5,
        "internal_rank_threshold": 3e-7,
    }
    serial = build_iao_lis_static_selections(
        mf,
        mp2_static,
        common=common,
        **thresholds,
    )
    domains = tuple(
        build_strong_ed_domain(common, mp2_static, fragment_index)
        for fragment_index in range(len(mp2_static.fragments))
    )

    def fail_if_domain_is_rebuilt(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the supplied ED domain must be reused")

    monkeypatch.setattr(
        "pyscfad.dlno.iao_lis.build_strong_ed_domain",
        fail_if_domain_is_rebuilt,
    )
    independent = tuple(
        build_iao_lis_fragment_static_selection(
            mf,
            mp2_static,
            fragment_index,
            common=common,
            domain=domains[fragment_index],
            **thresholds,
        )
        for fragment_index in range(len(mp2_static.fragments))
    )

    assert len(independent) == len(serial.fragments)
    for actual, expected in zip(independent, serial.fragments):
        assert actual.fragment_index == expected.fragment_index
        assert actual.full_occupied_space == expected.full_occupied_space
        assert actual.full_virtual_space == expected.full_virtual_space
        np.testing.assert_array_equal(
            actual.internal_occ_keep, expected.internal_occ_keep
        )
        np.testing.assert_array_equal(
            actual.internal_vir_keep, expected.internal_vir_keep
        )
        np.testing.assert_array_equal(
            actual.occupied_lno_keep, expected.occupied_lno_keep
        )
        np.testing.assert_array_equal(
            actual.virtual_lno_keep, expected.virtual_lno_keep
        )
