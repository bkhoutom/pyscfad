"""Checks of the new occupied-only target contract, not the shared solver."""

from pyscfad.dlno import dlno_base
from pyscfad.dlno import lis


from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest




def test_boys_internal_target_survives_rank_cutoff_and_virtual_noise():
    # A Boys target is a single occupied direction even when roundoff gives
    # it a tiny virtual overlap. The IAO rank cutoff must not remove it.
    target = jnp.asarray([[0.6], [0.8], [1e-11]])
    common = SimpleNamespace(
        s1e=jnp.eye(3),
        virtual_coeff=jnp.asarray([[0.0], [0.0], [1.0]]),
        fragment_occupied_data=(SimpleNamespace(
            iao_coeff=target,
            iao_occ_overlap=jnp.asarray([[0.6, 0.8]]),
        ),),
    )
    selection = lis._reference_fragment_selection(
        common, SimpleNamespace(lo_type="boys"), 0,
        jnp.zeros((2, 2)), jnp.zeros((1, 1)),
        thresh_occ=1e-4, thresh_vir=1e-5, internal_rank_threshold=2.0,
    )
    np.testing.assert_array_equal(selection.internal_occ_keep, [0])
    assert selection.internal_vir_keep.size == 0
    occ, vir = lis._internal_projection_matrices(common, 0, lo_type="boys")
    assert vir.shape == (0, 1)
    space = lis._fixed_row_space(occ, selection.internal_occ_keep)
    np.testing.assert_allclose(space @ space.T, [[0.36, 0.48], [0.48, 0.64]])


@pytest.mark.parametrize("retained", [0.8, 0.0])
def test_boys_domain_uses_selected_columns_and_retains_norm_loss(retained):
    from pyscfad.dlno._selection import FixedPAOSubspaceSelection
    from pyscfad.dlno.dlno_base import build_strong_ed_domain
    target = jnp.asarray([[np.sqrt(retained)], [np.sqrt(1.0 - retained)], [0.0]])
    overlap = target[:2].T
    common = SimpleNamespace(
        s1e=jnp.eye(3), fock=jnp.diag(jnp.array([-1.0, -0.6, 0.5])),
        occupied_coeff=jnp.eye(3)[:, :2],
        pao_coeff=jnp.eye(3)[:, 2:],
        fragment_occupied_data=(SimpleNamespace(
            iao_coeff=target, occupied_weight=overlap.T @ overlap,
        ),),
    )
    keep = np.array([0], dtype=np.int32)
    pao = FixedPAOSubspaceSelection(
        parent_columns=keep, support_ao_indices=np.array([0, 2]),
        canonical_keep=keep, overlap_keep=keep,
        completeness_keep=keep, metric_keep=keep,
    )
    static = SimpleNamespace(lo_type="boys", thresholds=SimpleNamespace(metric_rank=1e-10), fragments=(SimpleNamespace(
        extended_ao_indices=np.array([0, 2]), strong_fragments=keep,
        # These are IAO weight-eigenspace labels, never Boys column labels.
        strong_occ_union_keep=np.array([], dtype=np.int32),
        strong_occ_metric_keep=keep, strong_virtual=pao,
    ),))
    if retained == 0.0:
        with pytest.raises(RuntimeError, match="target"):
            build_strong_ed_domain(common, static, 0)
        return
    domain = build_strong_ed_domain(common, static, 0)
    np.testing.assert_allclose(domain.target_weight, [[0.8]], atol=1e-14)
    np.testing.assert_allclose(domain.partner_weight, [[0.8]], atol=1e-14)
    np.testing.assert_allclose(domain.occupied_coeff.T @ domain.occupied_coeff, [[1.0]])
    np.testing.assert_allclose(domain.virtual_coeff.T @ domain.virtual_coeff, [[1.0]])


def test_boys_reference_replay_tracks_mo_order_and_response():
    import jax
    from pyscfad import config_update, gto, scf
    from pyscfad.lo import boys
    

    with config_update("pyscfad_scf_implicit_diff", True):
        mol = gto.Mole(atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61",
                       basis="sto-3g", verbose=0)
        mol.build(trace_exp=False, trace_ctr_coeff=False)

        def mean_field(mol):
            mf = scf.RHF(mol).density_fit()
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-10
            mf.kernel()
            return mf

        mf = mean_field(mol)
        options = dict(conv_tol=1e-12, conv_tol_grad=1e-9)
        occupied = mf.mo_coeff[:, 1:5]
        reference = boys.boys(mol, occupied, **options)
        static = SimpleNamespace(
            lo_kwargs=options, target_reference_coeff=np.asarray(reference),
            target_reference_coords=np.asarray(mol.atom_coords()),
        )
        replay = dlno_base._rebuild_boys_targets
        reordered = occupied[:, [2, 0, 3, 1]] * jnp.array([-1, 1, 1, -1])
        tracked = replay(mf, reordered, static)
        np.testing.assert_allclose(reference.T @ mf.get_ovlp() @ tracked,
                                   np.eye(4), atol=2e-7)

        def observable(mol):
            current_mf = mean_field(mol)
            local = replay(current_mf, current_mf.mo_coeff[:, 1:5], static)
            probe = mol.intor("int1e_r")[2]
            return jnp.dot(jnp.arange(1., 5.), jnp.diag(local.T @ probe @ local))

        gradient = jax.grad(observable)(mol).coords
        coords = np.asarray(mol.atom_coords())
        direction = np.zeros_like(coords)
        direction[1, 1] = 1.0
        h = 1e-4
        plus = mol.set_geom_(coords + h * direction, unit="Bohr", inplace=False)
        minus = mol.set_geom_(coords - h * direction, unit="Bohr", inplace=False)
        fd = (observable(plus) - observable(minus)) / (2 * h)
        np.testing.assert_allclose(jnp.sum(gradient * direction), fd, atol=3e-6, rtol=1e-5)


def test_boys_domain_rejects_loss_of_a_noncentral_retained_direction():
    from pyscfad.dlno._selection import FixedPAOSubspaceSelection
    from pyscfad.dlno.dlno_base import build_strong_ed_domain
    central = jnp.array([[1.], [0.], [0.], [0.]])
    partner = jnp.array([[0.], [np.sqrt(1.-1e-12)], [1e-6], [0.]])
    keep = np.array([0], dtype=np.int32)
    pao = FixedPAOSubspaceSelection(
        parent_columns=keep, support_ao_indices=np.array([0, 2, 3]),
        canonical_keep=keep, overlap_keep=keep,
        completeness_keep=keep, metric_keep=keep,
    )
    common = SimpleNamespace(
        s1e=jnp.eye(4), fock=jnp.diag(jnp.array([-1., -.8, -.6, .5])),
        occupied_coeff=jnp.concatenate((central, partner), axis=1),
        pao_coeff=jnp.eye(4)[:, 3:],
        fragment_occupied_data=tuple(SimpleNamespace(iao_coeff=orbital)
                                     for orbital in (central, partner)),
    )
    static = SimpleNamespace(lo_type="boys", thresholds=SimpleNamespace(metric_rank=1e-10),
        fragments=(SimpleNamespace(
            extended_ao_indices=np.array([0, 2, 3]), strong_fragments=np.array([0, 1]),
            strong_occ_union_keep=np.array([], dtype=np.int32),
            strong_occ_metric_keep=np.array([0, 1]), strong_virtual=pao,
        ),))
    with pytest.raises(RuntimeError, match="rank"):
        build_strong_ed_domain(common, static, 0)
