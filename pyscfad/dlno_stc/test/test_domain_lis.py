"""Domain-only LNO eigenproblems, compared to independent global projectors."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad import config_update
from pyscfad.dlno_stc import ccsd, lis
from pyscfad.dlno_stc._retained_space import retained_projector
from pyscfad.dlno_stc.smooth import orthonormalize
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group


@pytest.mark.parametrize('nparent,ndomain,ntarget', [(9, 5, 2), (7, 7, 2), (2, 2, 2)])
def test_occupied_projector_and_full_response(nparent, ndomain, ntarget):
    rng = np.random.default_rng(963)
    r = rng.normal(size=(nparent, ndomain))
    t = rng.normal(size=(nparent, ntarget))
    w = np.diag(np.linspace(-1.8, 2.3, ndomain))
    probe = rng.normal(size=(nparent, nparent)); probe = (probe+probe.T)/2
    internal = orthonormalize(t, jnp.eye(nparent))
    a = lis._external_map(r, internal)
    columns = lis._range_columns(a)
    basis, k = lis._compact_mapped_density(a, w, columns)
    assert len(columns) == min(ndomain, nparent-ntarget)
    assert k.shape[0] <= ndomain
    local_keep = tuple(np.flatnonzero(np.abs(np.linalg.eigvalsh(k)) > .2))
    global_keep = tuple(np.flatnonzero(np.abs(np.linalg.eigvalsh(a@w@a.T)) > .2))
    assert len(local_keep) == len(global_keep)

    def loss(mapping, density, targets, compact):
        internal = orthonormalize(targets, jnp.eye(nparent))
        external = lis._external_map(mapping, internal)
        target_projector = internal@internal.T
        if compact:
            q, small = lis._compact_mapped_density(external, density, columns)
            selected = (q@retained_projector(local_keep)(small)@q.T
                        if local_keep else jnp.zeros_like(target_projector))
        else:
            selected = (retained_projector(global_keep)(external@density@external.T)
                        if global_keep else jnp.zeros_like(target_projector))
        return jnp.sum((target_projector+selected)*probe)

    reference, ref_bars = jax.value_and_grad(loss, argnums=(0, 1, 2))(r, w, t, False)
    actual, bars = jax.value_and_grad(loss, argnums=(0, 1, 2))(r, w, t, True)
    np.testing.assert_allclose(actual, reference, atol=2e-11, rtol=0)
    for observed, expected in zip(bars, ref_bars):
        np.testing.assert_allclose(observed, expected, atol=3e-10, rtol=3e-9)
    directions = [rng.normal(size=x.shape) for x in (r,w,t)]
    directions[1] = (directions[1]+directions[1].T)/2
    h = 1e-5
    plus = [x+h*d for x,d in zip((r,w,t),directions)]
    minus = [x-h*d for x,d in zip((r,w,t),directions)]
    fd = (loss(*plus, True)-loss(*minus, True))/(2*h)
    np.testing.assert_allclose(sum(jnp.sum(b*d) for b,d in zip(bars,directions)),
                               fd, atol=3e-7, rtol=3e-7)


def test_zero_cutoffs_keep_all_without_density_eigenproblem(molecular_group, monkeypatch):
    case = molecular_group
    common = ccsd.driver._initialize(case.mf, case.static.mp2_static, case.controls,
                                    with_grad=False)['common']
    frame = lis.build_boys_ed_frame(common, case.static.mp2_static, case.group.representative_index)
    matrices = dict(occupied=jnp.eye(frame.occupied_coeff.shape[1]),
                    virtual=jnp.eye(frame.virtual_coeff.shape[1]))
    def forbidden(*args, **kwargs):
        pytest.fail('keep-all selection must bypass density eigensystems')
    monkeypatch.setattr(lis, '_compact_mapped_density', forbidden)
    index = case.indices[0]
    selected = lis.build_fragment_lis_selection(case.mf, case.static.mp2_static, index,
        common=common, frame=frame, matrices=matrices, matrix_kind='W',
        thresh_occ=0., thresh_vir=0.)
    fragments = list(case.static.fragments); fragments[index] = selected
    static = lis.LISSelections(case.static.mp2_static, 'W', 0., 0., 1e-6, tuple(fragments))
    value = lis.build_fragment_lis(case.mf, common, static, index,
        matrices=matrices, frame=frame)
    np.testing.assert_allclose(value.occupied_projector, np.eye(common.occupied_coeff.shape[1]), atol=1e-14)
    np.testing.assert_allclose(value.virtual_projector, np.eye(common.virtual_coeff.shape[1]), atol=1e-14)
    assert value.density_occupied_active is value.density_virtual_active is None
    with pytest.raises(TypeError, match='virtual_eigenspace'):
        ccsd.build_static_selections(None, None, matrix_kind='W', thresh_occ=1e-4,
            thresh_vir=3e-5, baseline_controls={}, matrix_controls={}, virtual_eigenspace='global')
    with pytest.raises(ValueError, match='schema'):
        ccsd.kernel(case.mf, replace(case.static, schema_version=8),
                    baseline_controls=case.controls, matrix_controls=case.controls)


def test_molecular_domain_only_matches_global_oracle(molecular_group, monkeypatch):
    """Global density construction exists only here as an independent oracle."""
    case = molecular_group
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', '1')
    original = lis._compact_mapped_density
    shapes = []

    def global_oracle(mapping, density, columns=None):
        return jnp.eye(mapping.shape[0]), lis._hermitize(mapping@density@mapping.T)

    def counted_domain(mapping, density, columns=None):
        basis, compact = original(mapping, density, columns)
        shapes.append(dict(parent=mapping.shape[0], domain=mapping.shape[1],
                           eigenproblem=compact.shape[0], occupied=columns is not None))
        return basis, compact

    options = dict(baseline_controls=case.controls, matrix_controls=case.controls,
        ccsd_t=True, fragment_indices=case.indices, include_baseline=False,
        return_components=True, triples_controls=dict(mode='stochastic', nsamples=16,
            pilot_samples=16, global_seed=71, integral_transform='rotate'))
    def select(mf, single_pass=False):
        return ccsd.build_static_selections(mf, case.static.mp2_static, matrix_kind='W',
            thresh_occ=1e-4, thresh_vir=3e-5, baseline_controls=case.controls,
            matrix_controls=case.controls, fragment_indices=case.indices, single_pass=single_pass)

    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        with monkeypatch.context() as oracle_patch:
            oracle_patch.setattr(lis, '_compact_mapped_density', global_oracle)
            reference = ccsd.value_and_grad(case.mf.mol, accept.build_mf,
                build_static=select, **options)
        monkeypatch.setattr(lis, '_compact_mapped_density', counted_domain)
        actual = ccsd.value_and_grad(case.mf.mol, accept.build_mf,
            build_static=lambda mf: select(mf, single_pass=True), **options)
    assert shapes and all(x['eigenproblem'] <= x['domain'] for x in shapes)
    assert any(x['occupied'] and x['eigenproblem'] < x['parent'] for x in shapes)
    assert any(not x['occupied'] and x['eigenproblem'] < x['parent'] for x in shapes)
    np.testing.assert_allclose(actual[0], reference[0], atol=3e-11, rtol=0)
    np.testing.assert_allclose(actual[1].coords, reference[1].coords, atol=3e-9, rtol=3e-7)
    for a,b in zip(actual[2].fragments, reference[2].fragments):
        assert (a.nocc_lis,a.nvir_lis,a.triples_samples) == (b.nocc_lis,b.nvir_lis,b.triples_samples)
        np.testing.assert_allclose([a.e_ccsd,a.e_mp2_lis,a.e_triples],
                                   [b.e_ccsd,b.e_mp2_lis,b.e_triples], atol=3e-11, rtol=0)
    accept.record_result('domain_only_lis', dict(
        energy_difference_eh=float(actual[0]-reference[0]),
        max_gradient_difference_eh_per_bohr=float(np.max(np.abs(actual[1].coords-reference[1].coords))),
        eigensystem_shapes=shapes, corrected_fragments=list(case.indices)))
