"""One LIS construction, retaining the anchored physical response."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, lis
from pyscfad.dlno_stc._retained_space import retained_basis, retained_projector
from pyscfad.dlno_stc.test._cc_options import cc_options
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('keep', [(0, 1), (2, 3), (0, 1, 2, 3)])
def test_basis_response_matches_projector_at_internal_degeneracies(keep):
    rng = np.random.default_rng(420)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    matrix = q @ np.diag([-.7, -.7, 1.2, 1.2]) @ q.T
    direction = rng.normal(size=(4, 4))
    def projector(x):
        basis = retained_basis(keep)(x)
        return basis @ basis.T
    actual = jax.jvp(projector, (matrix,), (direction,))
    expected = jax.jvp(retained_projector(keep), (matrix,), (direction,))
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize('cutoff', [0., .45, 2.])
def test_prepared_lis_matches_fixed_anchor_pullback(molecular_group, monkeypatch, cutoff):
    """Compare every input cotangent, including unconstrained frame/metric bars."""
    case = molecular_group
    state = ccsd.driver._initialize(case.mf, case.static.mp2_static, case.controls,
                                    with_grad=False)
    frame = lis.build_boys_ed_frame(state['common'], case.static.mp2_static,
                                    case.group.representative_index)
    matrices = {name: jnp.diag(jnp.linspace(.1, 1., coeff.shape[1]))
                for name, coeff in [('occupied', frame.occupied_coeff),
                                    ('virtual', frame.virtual_coeff)]}
    plan = ccsd.build_static_selections(case.mf, case.static.mp2_static,
        thresh_occ=cutoff, thresh_vir=cutoff, **cc_options(case.controls, case.controls))
    index = case.indices[0]
    builds = []
    original = lis.build_fragment_lis
    def counted(*args, **kwargs):
        builds.append(args[3])
        return original(*args, **kwargs)
    monkeypatch.setattr(lis, 'build_fragment_lis', counted)
    prepared = {}
    static = ccsd._select_current_fragment(state, plan, index, case.group, frame, matrices,
                                          prepared_lis=prepared, with_grad=True)
    assert builds == [index]
    rng = np.random.default_rng(17)
    seed = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)), prepared['value'])
    actual_bars = prepared['pullback'](seed)
    assert builds == [index]  # Pullback must consume the tape, not reconstruct.
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(actual_bars))

    def reference(mf, common, blocks, coeff):
        current = frame._replace(occupied_coeff=coeff[0], virtual_coeff=coeff[1])
        value = lis.build_fragment_lis(mf, common, static.lis_static, index,
            matrices=blocks, frame=current, frame_fragment_index=case.group.representative_index)
        return ccsd._SolverLIS(value.mo_coeff, value.fragment_iao_coeff,
            value.active_occupied_coeff, value.active_virtual_coeff, tuple(value.frozen))
    # Empty virtual LIS has no solver response; compare its raw occupied gauge
    # using the same chosen anchors, which normal selection deliberately omits.
    if not prepared['value'].active_virtual_coeff.shape[1]:
        return
    expected, pullback = jax.vjp(reference, case.mf, state['common'], matrices,
                                (frame.occupied_coeff, frame.virtual_coeff))
    for a, b in zip(jax.tree.leaves(prepared['value']), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=3e-12, rtol=3e-12)
    for a, b in zip(jax.tree.leaves(actual_bars), jax.tree.leaves(pullback(seed))):
        np.testing.assert_allclose(a, b, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize('with_grad', [False, True])
def test_production_constructs_each_lis_once(molecular_group, monkeypatch, with_grad):
    case = molecular_group
    plan = ccsd.build_static_selections(case.mf, case.static.mp2_static,
        thresh_occ=1e-4, thresh_vir=3e-5, **cc_options(case.controls, case.controls))
    builds, solves = [], []
    build, solve = lis.build_fragment_lis, ccsd._solve_fragment
    def counted_build(*args, **kwargs):
        builds.append(args[3])
        return build(*args, **kwargs)
    def counted_solve(*args, **kwargs):
        assert kwargs.get('prepared_lis') is not None
        solves.append(args[3])
        return solve(*args, **kwargs)
    monkeypatch.setattr(lis, 'build_fragment_lis', counted_build)
    monkeypatch.setattr(ccsd, '_solve_fragment', counted_solve)
    state = ccsd.driver._initialize(case.mf, plan.mp2_static, case.controls,
        with_grad=with_grad, scf_pullback=case.scf_pullback if with_grad else None)
    rows = ccsd._run_group(state, plan, case.group, case.controls, case.controls,
        with_grad=with_grad, verbose_imp=0, ccsd_t=False)
    assert builds == solves == list(case.indices)
    assert len(rows) == len(case.indices)


def test_replica_average_applies_shared_lis_pullback_once(group_case, monkeypatch):
    case = group_case
    builds, reverses = [], []
    original = ccsd._solve_fragment
    def select(state, static, index, group, frame, matrices, *, prepared_lis, with_grad):
        builds.append(index)
        def build(mf, common, blocks, coeff):
            value = (jnp.sum(blocks['occupied']**2) * (1 + mf)
                     + jnp.sum(blocks['virtual']) * common
                     + jnp.sum(coeff[0] * coeff[1]))
            return ccsd._SolverLIS(value, value*.2, value*.3, value*.4, ())
        args = (state['mf'], state['common'],
                {k: jnp.asarray(matrices[k]) for k in ('occupied', 'virtual')},
                (frame.occupied_coeff, frame.virtual_coeff))
        if with_grad:
            value, pullback = jax.vjp(build, *args)
            def reverse(bar):
                reverses.append(index)
                return pullback(bar)
            prepared_lis['pullback'] = reverse
        else:
            value = build(*args)
        prepared_lis['value'] = value
        return static
    def solve(*args, **options):
        (emp2, ecc, et), _ = original(*args, **options)
        value = options['prepared_lis'].mo_coeff
        return (emp2, ecc + .03*value**2, et), {'ccsd': {}}
    monkeypatch.setattr(ccsd, '_select_current_fragment', select)
    monkeypatch.setattr(ccsd, '_solve_fragment', solve)
    reference, _ = case.run()
    builds.clear(); reverses.clear()
    actual, _ = case.run(cc_solver='stc', stc_cc_controls=dict(seed=71,
        energy_tolerance=1e3, precision_pilot_replicas=2,
        precision_min_replicas=2, precision_max_replicas=2))
    assert builds == reverses == [0, 1]
    for name in ('energy', 'mf_bar', 'common_bar'):
        np.testing.assert_allclose(actual[name], reference[name], atol=2e-12, rtol=2e-12)
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17), ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(with_grad=False, mf=.2+dm, common=-.1+dc)
        minus, _ = case.run(with_grad=False, mf=.2-dm, common=-.1-dc)
        np.testing.assert_allclose(actual[name]-initial,
            (plus['energy']-minus['energy'])/(2*step), atol=2e-8, rtol=2e-8)
