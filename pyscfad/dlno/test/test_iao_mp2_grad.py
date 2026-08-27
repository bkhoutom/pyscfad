import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyscfad.dlno.iao_mp2_grad as iao_mp2_grad_module
from pyscfad import config, config_update, gto, scf
from pyscfad.dlno.iao_mp2 import (
    IAOFragmentMP2,
    IAOFragmentMP2Thresholds,
    build_iao_fragment_topology,
    evaluate_iao_fragment_mp2,
)
from pyscfad.dlno.iao_mp2_grad import (
    build_iao_mp2_static_selections,
    correlation_energy,
    correlation_value_and_grad,
    correlation_value_and_grad_with_iao,
)
from pyscfad.lno import lno_base
from pyscfad.mp import dfmp2
from pyscfad.ops import stop_trace


warnings.filterwarnings(
    "ignore",
    message=r"Function mol\.dumps drops attribute .* because it is not "
            r"JSON-serializable",
)

@pytest.fixture(scope="module", autouse=True)
def _gradient_config():
    with (
        config_update("pyscfad_moleintor_opt", True),
        config_update("pyscfad_scf_implicit_diff", True),
        config_update("pyscfad_scf_first_order_custom", True),
    ):
        yield


def _water():
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
    return mol


def _build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


def _full_domain_thresholds():
    return IAOFragmentMP2Thresholds(
        pao_norm=1e-10,
        domain_pao=0.0,
        ed_pao=0.0,
        occupied_weight=1e-12,
    )


@pytest.mark.parametrize("keep", [np.arange(4), np.arange(1, 4)])
def test_fixed_metric_orthonormalization_projector_jvp_matches_fd(keep):
    """The retained-space projector must be gauge independent near degeneracy."""
    rng = np.random.default_rng(812)
    nao, ncol = 9, 4
    overlap_factor = rng.normal(size=(nao, nao))
    overlap = overlap_factor.T @ overlap_factor + np.eye(nao)
    overlap_e, overlap_v = np.linalg.eigh(overlap)
    overlap_inv_sqrt = (
        overlap_v / np.sqrt(overlap_e)[None, :]
    ) @ overlap_v.T
    frame, _ = np.linalg.qr(rng.normal(size=(nao, ncol)))
    metric_e = np.asarray([
        0.983612374956,
        0.999999999990,
        0.999999999993,
        1.000000000000,
    ])
    coeff = overlap_inv_sqrt @ frame @ np.diag(np.sqrt(metric_e))
    direction = rng.normal(scale=0.03, size=coeff.shape)
    probe = rng.normal(size=(nao, nao))
    probe = 0.5 * (probe + probe.T)

    def projected_scalar(coeff_):
        orth = iao_mp2_grad_module._fixed_metric_orthonormalize(
            coeff_, overlap, keep
        )
        projector = orth @ orth.T
        return jnp.einsum("uv,uv->", probe, projector)

    _, tangent = jax.jvp(
        projected_scalar, (coeff,), (direction,)
    )
    step = 1e-5
    finite_difference = (
        projected_scalar(coeff + step * direction)
        - projected_scalar(coeff - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        tangent, finite_difference, atol=2e-9, rtol=2e-8
    )


@pytest.mark.parametrize("frozen", [None, 1])
def test_full_domain_value_and_grad_matches_canonical_dfmp2(
    frozen,
):
    mol = _water()

    def canonical_total(mol_):
        mf = _build_mf(mol_)
        e_corr, _ = dfmp2.MP2(
            mf, frozen=frozen
        ).kernel(with_t2=False)
        return mf.e_tot + e_corr

    with config_update("pyscfad_scf_first_order_custom", False):
        reference_energy, reference_bar = jax.value_and_grad(
            canonical_total
        )(mol)
    local_energy, local_bar = IAOFragmentMP2.value_and_grad(
        mol,
        build_mf=_build_mf,
        frozen=frozen,
        thresholds=_full_domain_thresholds(),
        pair_energy_model="all",
        force_full_domains=True,
    )

    np.testing.assert_allclose(
        local_energy, reference_energy, atol=2e-9, rtol=2e-9
    )
    np.testing.assert_allclose(
        np.asarray(local_bar.coords),
        np.asarray(reference_bar.coords),
        atol=2e-7,
        rtol=2e-6,
    )


def test_value_and_grad_forces_standard_implicit_scf_response():
    observed_backends = []

    def build_mf(mol):
        observed_backends.append(
            (config.scf_implicit_diff, config.scf_first_order_custom)
        )
        return _build_mf(mol)

    assert config.scf_first_order_custom
    with config_update("pyscfad_scf_implicit_diff", False):
        energy, mol_bar = IAOFragmentMP2.value_and_grad(
            _water(),
            build_mf=build_mf,
            thresholds=_full_domain_thresholds(),
            pair_energy_model="all",
            force_full_domains=True,
            include_hf=False,
        )
        # Both overrides are local to the saved SCF response construction.
        assert not config.scf_implicit_diff

    assert observed_backends
    assert all(implicit and not custom for implicit, custom in observed_backends)
    assert config.scf_implicit_diff
    assert config.scf_first_order_custom
    assert np.isfinite(float(energy))
    assert np.all(np.isfinite(np.asarray(mol_bar.coords)))


def _water_dimer(distance_angstrom):
    mol = gto.Mole(
        atom=f"""
        O  0.0000000000  0.0000000000  0.0000000000
        H  0.0000000000 -0.7570000000  0.5870000000
        H  0.0000000000  0.7570000000  0.5870000000
        O  0.0000000000  0.0000000000  {distance_angstrom:.12f}
        H  0.0000000000 -0.7570000000  {distance_angstrom + 0.587:.12f}
        H  0.0000000000  0.7570000000  {distance_angstrom + 0.587:.12f}
        """,
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
        max_memory=1000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol


def test_fixed_topology_mixed_strong_weak_gradient_matches_fd():
    distance = 8.0
    mol = _water_dimer(distance)
    thresholds = IAOFragmentMP2Thresholds(
        pair_energy=1e-4,
    )
    with config_update("pyscfad_scf_first_order_custom", False):
        reference_mf = _build_mf(mol)
    reference_topology = build_iao_fragment_topology(
        reference_mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    static = build_iao_mp2_static_selections(
        reference_mf, reference_topology
    )
    assert np.count_nonzero(~static.strong_mask) == 2
    eager_energy = evaluate_iao_fragment_mp2(
        reference_mf, reference_topology
    ).e_corr
    rebuilt_energy = float(correlation_energy(reference_mf, static))
    np.testing.assert_allclose(
        rebuilt_energy, eager_energy, atol=2e-10, rtol=2e-10
    )

    energy, mol_bar = IAOFragmentMP2.value_and_grad(
        mol,
        build_mf=_build_mf,
        topology=static,
    )
    ad_derivative = float(
        np.sum(np.asarray(mol_bar.coords)[3:, 2])
    )

    def fixed_energy(distance_):
        displaced = _water_dimer(distance_)
        with config_update("pyscfad_scf_first_order_custom", False):
            mf = _build_mf(displaced)
        return float(mf.e_tot + correlation_energy(mf, static))

    step = 2e-4
    fd_derivative_angstrom = (
        fixed_energy(distance - 2.0 * step)
        - 8.0 * fixed_energy(distance - step)
        + 8.0 * fixed_energy(distance + step)
        - fixed_energy(distance + 2.0 * step)
    ) / (12.0 * step)
    # ``mol_bar.coords`` differentiates Bohr coordinates, whereas the helper
    # above displaces the second monomer in Angstrom.
    fd_derivative_bohr = fd_derivative_angstrom * 0.529177210903

    np.testing.assert_allclose(
        ad_derivative, fd_derivative_bohr, atol=2e-6, rtol=2e-4
    )
    assert np.isfinite(float(energy))


def test_total_and_progressive_paths_evaluate_unordered_weak_pair_once(
    monkeypatch,
):
    mol = _water_dimer(8.0)
    thresholds = IAOFragmentMP2Thresholds(
        pair_energy=1e-4,
    )
    mf = _build_mf(mol)
    topology = build_iao_fragment_topology(
        mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    static = build_iao_mp2_static_selections(mf, topology)
    assert np.count_nonzero(~static.strong_mask) == 2

    real_cross = iao_mp2_grad_module.dlno_mp2.pair_energy_multipole_cross
    calls = []

    def counted_cross(*args, **kwargs):
        calls.append(None)
        return real_cross(*args, **kwargs)

    monkeypatch.setattr(
        iao_mp2_grad_module.dlno_mp2,
        "pair_energy_multipole_cross",
        counted_cross,
    )

    scalar_energy = correlation_energy(mf, static)
    assert len(calls) == 1

    calls.clear()
    progressive_energy, _ = correlation_value_and_grad(mf, static)
    assert len(calls) == 1
    np.testing.assert_allclose(
        progressive_energy, scalar_energy, atol=2e-12, rtol=2e-12
    )


def test_progressive_details_preserve_energy_and_cotangent_to_roundoff():
    mol = _water_dimer(8.0)
    mf = _build_mf(mol)
    topology = build_iao_fragment_topology(
        mf,
        thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
        pair_energy_model="multipole",
    )
    static = build_iao_mp2_static_selections(mf, topology)

    plain_energy, plain_bar = correlation_value_and_grad(mf, static)
    profiled_energy, profiled_bar, details = correlation_value_and_grad(
        mf, static, return_details=True
    )

    np.testing.assert_allclose(
        profiled_energy, plain_energy, atol=5e-14, rtol=5e-14
    )
    plain_leaves, plain_tree = jax.tree_util.tree_flatten(
        plain_bar, is_leaf=lambda value: value is None
    )
    profiled_leaves, profiled_tree = jax.tree_util.tree_flatten(
        profiled_bar, is_leaf=lambda value: value is None
    )
    assert profiled_tree == plain_tree
    for plain, profiled in zip(plain_leaves, profiled_leaves):
        if plain is None or (
            hasattr(plain, "dtype") and plain.dtype == jax.dtypes.float0
        ):
            assert profiled is None or profiled.dtype == jax.dtypes.float0
        else:
            np.testing.assert_allclose(
                np.asarray(profiled), np.asarray(plain),
                atol=5e-14, rtol=5e-14,
            )

    assert details.e_corr == float(profiled_energy)
    assert details.e_strong + details.e_weak == pytest.approx(
        details.e_corr, abs=2e-15
    )
    assert details.n_fragments == 2
    assert details.n_strong_pairs == 0
    assert details.n_weak_pairs == 1
    assert details.n_strong_ed_terms == 2
    assert details.n_weak_pair_terms == 1
    assert [term.kind for term in details.terms] == [
        "strong", "weak", "strong"
    ]
    assert all(row.n_domain_atoms > 0 for row in details.fragments)
    assert all(row.n_domain_ao > 0 for row in details.fragments)
    assert all(row.n_domain_occ > 0 for row in details.fragments)
    assert all(row.n_domain_vir > 0 for row in details.fragments)
    timing_values = vars(details.timing).values()
    assert all(value >= 0.0 for value in timing_values)
    assert details.timing.total_seconds > 0.0


def test_scalar_correlation_energy_rejects_outer_ad_trace():
    mol = _water_dimer(8.0)
    thresholds = IAOFragmentMP2Thresholds(
        pair_energy=1e-4,
    )
    mf = _build_mf(mol)
    topology = build_iao_fragment_topology(
        mf,
        thresholds=thresholds,
        pair_energy_model="multipole",
    )
    static = build_iao_mp2_static_selections(mf, topology)

    with pytest.raises(TypeError, match="correlation_value_and_grad"):
        jax.vjp(lambda mf_: correlation_energy(mf_, static), mf)


def test_external_iao_pullback_matches_internal_rebuild():
    mol = _water()
    mf, internal_scf_pullback = jax.vjp(_build_mf, mol)
    static = stop_trace(
        lambda mf_: IAOFragmentMP2.build_static_topology(
            mf_,
            thresholds=_full_domain_thresholds(),
            pair_energy_model="all",
            force_full_domains=True,
        )
    )(mf)

    internal_energy, internal_mf_bar = correlation_value_and_grad(
        mf, static
    )
    internal_mol_bar, = internal_scf_pullback(internal_mf_bar)

    # Use a fresh saved SCF pullback.  The local-DF helpers may populate
    # ignored/static cache attributes on an SCF object, so two independent
    # gradient evaluations should not share the same mutable solver instance.
    mf, external_scf_pullback = jax.vjp(_build_mf, mol)
    iao_coeff, iao_pullback = jax.vjp(
        lambda mf_: lno_base.get_iao(
            mf_.mol, mf_.mo_coeff[:, static.active_occ_indices]
        ),
        mf,
    )
    external_energy, external_mf_bar, iao_bar = (
        correlation_value_and_grad_with_iao(mf, iao_coeff, static)
    )
    iao_mf_bar, = iao_pullback(iao_bar)

    def add_cotangent(left, right):
        if left is None:
            return right
        if right is None:
            return left
        if hasattr(left, "dtype") and left.dtype == jax.dtypes.float0:
            return right
        if hasattr(right, "dtype") and right.dtype == jax.dtypes.float0:
            return left
        return left + right

    external_mf_bar = jax.tree_util.tree_map(
        add_cotangent, external_mf_bar, iao_mf_bar
    )
    external_mol_bar, = external_scf_pullback(external_mf_bar)

    np.testing.assert_allclose(
        external_energy, internal_energy, atol=2e-11, rtol=2e-11
    )
    np.testing.assert_allclose(
        np.asarray(external_mol_bar.coords),
        np.asarray(internal_mol_bar.coords),
        atol=2e-8,
        rtol=2e-7,
    )
