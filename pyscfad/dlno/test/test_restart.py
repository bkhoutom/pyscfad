import json
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyscfad.dlno._restart as restart_module
from pyscfad.dlno._restart import (
    RestartCorruptionError,
    RestartManager,
    RestartManifestError,
    RestartMismatchError,
    df_source_fingerprint,
)
from pyscfad.dlno.iao_ccsd import _restart_scientific_payload
from pyscfad.dlno.iao_mp2 import _serial_restart_scientific_payload
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds


@jax.tree_util.register_pytree_with_keys_class
class _OrderedTree:
    """Test pytree whose leaf order may change while named paths do not."""

    def __init__(self, values, order):
        self.values = dict(values)
        self.order = tuple(order)

    def tree_flatten_with_keys(self):
        return [
            (jax.tree_util.GetAttrKey(name), self.values[name])
            for name in self.order
        ], self.order

    @classmethod
    def tree_unflatten(cls, order, children):
        return cls(dict(zip(order, children)), order)


def _manager(path, *, resume=False, payload=None, initialize=True):
    if payload is None:
        payload = {"coords": np.arange(6.0).reshape(2, 3), "threshold": 1e-5}
    return RestartManager(
        path,
        resume=resume,
        method="test-dlno",
        scientific_payload=payload,
        initialize=initialize,
    )


def _restart_test_molecule():
    coords = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    return SimpleNamespace(
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
        atom_symbol=lambda _index: "H",
    )


def _restart_test_mf(*, coeff_shift=0.0, e_tot=-1.0, mo_occ=None):
    if mo_occ is None:
        mo_occ = np.asarray([2.0, 0.0])
    return SimpleNamespace(
        mo_coeff=np.asarray([
            [0.100000000049 + coeff_shift, 0.0],
            [0.0, 1.0 - coeff_shift],
        ]),
        mo_energy=np.asarray([-0.5 + coeff_shift, 0.2 - coeff_shift]),
        mo_occ=np.asarray(mo_occ),
        e_tot=e_tot,
        with_df=None,
    )


def _mp2_restart_payload(mol, mf):
    return _serial_restart_scientific_payload(
        mol,
        mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=IAOFragmentMP2Thresholds(),
        pair_energy_model="multipole",
        force_full_domains=False,
        include_hf=True,
    )


def _cc_restart_payload(mol, mf):
    return _restart_scientific_payload(
        mol,
        mf,
        frag_lolist=None,
        frag_atmlist=None,
        frozen=None,
        thresholds=IAOFragmentMP2Thresholds(),
        pair_energy_model="multipole",
        force_full_domains=False,
        thresh_occ=1e-4,
        thresh_vir=1e-5,
        internal_rank_threshold=1e-6,
        ccsd_t=True,
        dcsd=False,
    )


@pytest.mark.parametrize(
    "payload_builder", (_mp2_restart_payload, _cc_restart_payload),
    ids=("mp2", "cc"),
)
def test_restart_accepts_changed_scf_orbital_values(
    tmp_path, payload_builder
):
    """MO values are not a hard compatibility key for a guarded restart."""

    mol = _restart_test_molecule()
    original = RestartManager(
        tmp_path,
        method="test-dlno",
        scientific_payload=payload_builder(mol, _restart_test_mf()),
    )
    original.close()

    resumed = RestartManager(
        tmp_path,
        resume=True,
        method="test-dlno",
        scientific_payload=payload_builder(
            mol, _restart_test_mf(coeff_shift=2e-10)
        ),
    )
    resumed.close()


@pytest.mark.parametrize(
    "payload_builder", (_mp2_restart_payload, _cc_restart_payload),
    ids=("mp2", "cc"),
)
def test_restart_accepts_scf_total_energy_roundoff(
    tmp_path, payload_builder
):
    """Sub-nanohartree SCF cleanup noise is not a hard restart mismatch."""

    mol = _restart_test_molecule()
    original = RestartManager(
        tmp_path,
        method="test-dlno",
        scientific_payload=payload_builder(mol, _restart_test_mf()),
    )
    original.close()

    resumed = RestartManager(
        tmp_path,
        resume=True,
        method="test-dlno",
        scientific_payload=payload_builder(
            mol, _restart_test_mf(e_tot=-1.0 + 2e-10)
        ),
    )
    resumed.close()


@pytest.mark.parametrize(
    "payload_builder", (_mp2_restart_payload, _cc_restart_payload),
    ids=("mp2", "cc"),
)
@pytest.mark.parametrize(
    "changed_mf",
    (
        {"mo_occ": np.asarray([1.0, 1.0])},
        {"e_tot": -1.0 + 2e-7},
    ),
    ids=("occupation", "total-energy"),
)
def test_restart_rejects_changed_scf_state(
    tmp_path, payload_builder, changed_mf
):
    """Stable occupation and energy guards still reject a different SCF."""

    mol = _restart_test_molecule()
    original = RestartManager(
        tmp_path,
        method="test-dlno",
        scientific_payload=payload_builder(mol, _restart_test_mf()),
    )
    original.close()

    with pytest.raises(RestartMismatchError, match="base_digest"):
        RestartManager(
            tmp_path,
            resume=True,
            method="test-dlno",
            scientific_payload=payload_builder(
                mol, _restart_test_mf(**changed_mf)
            ),
        )


def test_manifest_creation_resume_worker_and_mismatch(tmp_path):
    manager = _manager(tmp_path)
    assert manager.enabled
    assert manager.manifest_path.is_file()

    # An MPI worker opens the root-created first-run manifest even though the
    # public resume flag is false.
    worker = _manager(tmp_path, initialize=False)
    assert worker.base_digest == manager.base_digest

    with pytest.raises(RestartManifestError):
        _manager(tmp_path)
    manager.close()
    with pytest.raises(RestartMismatchError):
        _manager(tmp_path, resume=True, payload={"coords": np.ones((2, 3))})

    resumed = _manager(tmp_path, resume=True)
    assert resumed.base_digest == manager.base_digest


def test_disabled_manager_and_invalid_resume():
    manager = _manager(None)
    assert not manager.enabled
    assert manager.save_record("progress", scalars={"n": 1}) is None
    assert manager.load_record("progress") is None
    with pytest.raises(ValueError, match="requires checkpoint_dir"):
        _manager(None, resume=True)


@pytest.mark.parametrize(
    "static",
    [
        IAOFragmentMP2Thresholds(
            pair_energy=3.25e-7,
            multipole_order=3,
        ),
        IAOFragmentMP2Thresholds(
            pair_energy=3.25e-7,
            multipole_order=3,
            mp2_block_memory_mb=19.5,
            mp2_block_nvir=17,
        ),
    ],
)
def test_static_dataclass_round_trip_and_unbound_manifest_recovery(
    tmp_path, static
):
    manager = _manager(tmp_path)
    digest = manager.save_static(static)
    assert manager.load_static(expected_type=IAOFragmentMP2Thresholds) == static

    # Emulate interruption after static.h5 was committed but before run.json
    # received its static digest.  load_static must recover and bind it.
    manifest = json.loads(manager.manifest_path.read_text("utf8"))
    manifest["static_digest"] = None
    manager.manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    manager.close()
    resumed = _manager(tmp_path, resume=True)
    assert resumed.static_digest is None
    assert resumed.load_static() == static
    assert resumed.static_digest == digest


def test_pytree_round_trip_uses_paths_not_flatten_order(tmp_path):
    manager = _manager(tmp_path)
    manager.bind_static(IAOFragmentMP2Thresholds())
    saved = _OrderedTree(
        {
            "dense": jnp.asarray([1.5, -2.0]),
            "scalar": jnp.asarray(3.25),
            "zero": jnp.zeros((2, 2)),
            "none": None,
            "float0": np.zeros((3,), dtype=jax.dtypes.float0),
        },
        ("dense", "scalar", "zero", "none", "float0"),
    )
    template = _OrderedTree(
        {
            "dense": jnp.zeros(2),
            "scalar": jnp.asarray(0.0),
            "zero": jnp.ones((2, 2)),
            "none": None,
            "float0": np.zeros((3,), dtype=jax.dtypes.float0),
        },
        ("float0", "none", "zero", "scalar", "dense"),
    )
    manager.save_record(
        "progress",
        scalars={"energy": -1.25, "completed": (0, 2)},
        trees={"bar": saved},
        metadata={"rank": np.int32(1)},
    )
    record = manager.load_record(
        "progress", templates={"bar": template}
    )
    assert record.scalars == {"energy": -1.25, "completed": (0, 2)}
    assert int(record.metadata["rank"]) == 1
    assert record.trees["bar"].order == template.order
    np.testing.assert_array_equal(
        record.trees["bar"].values["dense"], [1.5, -2.0]
    )
    assert record.trees["bar"].values["scalar"].shape == ()
    assert float(record.trees["bar"].values["scalar"]) == 3.25
    np.testing.assert_array_equal(
        record.trees["bar"].values["zero"], np.zeros((2, 2))
    )
    assert record.trees["bar"].values["none"] is None
    assert record.trees["bar"].values["float0"].dtype == jax.dtypes.float0


def test_tree_shape_and_name_mismatch_is_fatal(tmp_path):
    manager = _manager(tmp_path)
    manager.save_record(
        "progress", trees={"bar": {"x": jnp.ones(2)}}
    )
    with pytest.raises(RestartMismatchError, match="dtype/shape changed"):
        manager.load_record(
            "progress", templates={"bar": {"x": jnp.ones(3)}}
        )
    with pytest.raises(RestartMismatchError, match="tree names differ"):
        manager.load_record(
            "progress", templates={"different": {"x": jnp.ones(2)}}
        )


def test_corrupt_dataset_can_raise_or_surface_recompute_miss(tmp_path):
    manager = _manager(tmp_path)
    path = manager.save_record(
        "progress", trees={"bar": {"x": jnp.asarray([1.0, 2.0])}}
    )
    with h5py.File(path, "r+") as handle:
        dataset = handle["arrays"][sorted(handle["arrays"])[0]]
        dataset[...] = np.asarray([8.0, 9.0])
    template = {"bar": {"x": jnp.zeros(2)}}
    with pytest.raises(RestartCorruptionError, match="SHA-256"):
        manager.load_record("progress", templates=template)
    with pytest.warns(RuntimeWarning, match="will be recomputed"):
        assert manager.load_record(
            "progress", templates=template, on_corrupt="miss"
        ) is None


def test_pre_replace_failure_preserves_old_generation(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager.save_record("progress", scalars={"generation": 1})

    def interrupt(_temporary, _final):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(restart_module, "_POST_SAVE_TEST_HOOK", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        manager.save_record("progress", scalars={"generation": 2})
    monkeypatch.setattr(restart_module, "_POST_SAVE_TEST_HOOK", None)
    record = manager.load_record("progress")
    assert record.scalars["generation"] == 1


def test_durable_event_hook_runs_after_stage_commit(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    events = []

    def event(stage, key, path):
        assert path.is_file()
        events.append((stage, key, path))

    monkeypatch.setattr(restart_module, "_CHECKPOINT_EVENT_HOOK", event)
    manager.save_static(IAOFragmentMP2Thresholds())
    manager.save_record("fragment_forward", key=3, scalars={"e": -0.1})
    assert [(stage, key) for stage, key, _ in events] == [
        ("static", None),
        ("fragment_forward", 3),
    ]


def test_df_source_fingerprint_tracks_logical_hdf5_contents(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "moved.h5"
    values = np.arange(30.0).reshape(5, 6)
    for path in (first, second):
        with h5py.File(path, "w") as handle:
            handle.create_dataset("j3c", data=values)

    def fingerprint(path):
        with_df = SimpleNamespace(_get_cderi_source=lambda: str(path))
        return df_source_fingerprint(SimpleNamespace(with_df=with_df))

    reference = fingerprint(first)
    assert reference == fingerprint(second)
    with h5py.File(second, "r+") as handle:
        handle["j3c"][2, 3] += 0.25
    assert fingerprint(second)["sha256"] != reference["sha256"]
