"""One domain frame producer/reverse shared by W and molecular LIS CCSD(T)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from pyscfad import config_update
from pyscfad.dlno_stc import ccsd, domain, prepare
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_backend import native


@pytest.fixture(scope='module')
def molecular_group():
    native()
    mol = accept.molecule()
    # This fixture runs several traced CC/SCF reverses in one process. Include
    # retained JAX executables in its test budget; the molecule helper's 3 GB
    # default is smaller than that residency, even for these tiny tensors.
    mol.max_memory = 8000
    with (config_update('pyscfad_scf_implicit_diff', True),
          config_update('pyscfad_scf_first_order_custom', False)):
        mf, scf_pullback = jax.vjp(accept.build_mf, mol)
    domains = accept.selection(mf)
    controls = dict(accept.controls(), threads=1)
    static = ccsd.build_static_selections(
        mf, domains, matrix_kind='W', thresh_occ=1e-4, thresh_vir=3e-5,
        baseline_controls=controls, matrix_controls=controls)
    group = ccsd.build_domain_groups(domains)[0]
    indices = tuple(i for i, _ in ccsd._group_cc_fragments(static, group))[:2]
    assert len(indices) == 2
    assert len(domains.fragments[group.representative_index].extended_ao_indices) < mf.mol.nao
    return SimpleNamespace(mf=mf, scf_pullback=scf_pullback, static=static,
                           controls=controls, group=group, indices=indices)


def counted_frames(monkeypatch):
    """Count actual frame forwards and VJPs, including occupied response."""
    counts = dict(forward=0, reverse=0)
    original = domain.build_boys_ed_frame

    def build(common, static, index, **options):
        saved = {}

        def coefficients(current):
            counts['forward'] += 1
            frame = original(current, static, index, **options)
            saved['frame'] = frame
            return frame.occupied_coeff, frame.virtual_coeff

        @jax.custom_vjp
        def wrapped(current):
            return coefficients(current)

        def forward(current):
            return jax.vjp(coefficients, current)

        def backward(pullback, bars):
            counts['reverse'] += 1
            return pullback(bars)

        wrapped.defvjp(forward, backward)
        co, cv = wrapped(common)
        return saved['frame']._replace(occupied_coeff=co, virtual_coeff=cv)

    monkeypatch.setattr(prepare, 'build_boys_ed_frame', build)
    monkeypatch.setattr(ccsd, 'build_boys_ed_frame', build)
    return counts


@pytest.mark.parametrize('shared_integrals', [False, True])
def test_molecular_shared_frame_matches_rebuilt_energy_and_gradient(
    molecular_group, monkeypatch, shared_integrals,
):
    """Real local DF, W feedback, LIS CCSD and sampled (T), with final SCF response."""
    case = molecular_group
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_REVERSE', str(int(shared_integrals)))
    counts = counted_frames(monkeypatch)
    solve = ccsd._solve_fragment
    triples = ccsd.normalize_triples_controls(
        {'mode': 'stochastic', 'nsamples': 16, 'pilot_samples': 16,
         'global_seed': 71}, ccsd_t=True)

    def run(rebuild=False, with_grad=True):
        # Reproduce the old branch: discard the supplied frame and rebuild it
        # from common inside the correction VJP. Its explicit frame seed is zero.
        def fragment(*args, **kwargs):
            if rebuild:
                kwargs['frame'] = None
            return solve(*args, **kwargs)

        monkeypatch.setattr(ccsd, '_solve_fragment', fragment)
        counts.update(forward=0, reverse=0)
        state = ccsd.driver._initialize(
            case.mf, case.static.mp2_static, case.controls, with_grad=with_grad,
            scf_pullback=case.scf_pullback if with_grad else None)
        rows = ccsd._run_group(
            state, case.static, case.group, case.controls, case.controls,
            with_grad=with_grad, verbose_imp=0, ccsd_t=True,
            triples_controls=triples, fragment_indices=case.indices)
        result = ccsd.driver._finish(state, with_grad=with_grad, include_hf=False)
        return result, rows, counts.copy(), (state['mf_bar'], state['common_bar'])

    (reference, ref_grad), ref_rows, ref_counts, ref_bars = run(rebuild=True)
    (value, gradient), rows, actual_counts, bars = run()
    assert ref_counts == dict(forward=1 + len(case.indices), reverse=1 + len(case.indices))
    assert actual_counts == dict(forward=1, reverse=1)
    np.testing.assert_allclose(value, reference, atol=2e-12, rtol=0)
    np.testing.assert_allclose(gradient.coords, ref_grad.coords, atol=2e-9, rtol=2e-7)
    for actual, expected in zip(jax.tree_util.tree_leaves(bars), jax.tree_util.tree_leaves(ref_bars)):
        if getattr(actual, 'dtype', None) == jax.dtypes.float0:
            continue
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)
    for actual, expected in zip(rows, ref_rows):
        np.testing.assert_allclose(
            (actual.e_mp2_lis, actual.e_ccsd, actual.e_triples),
            (expected.e_mp2_lis, expected.e_ccsd, expected.e_triples), atol=2e-12, rtol=0)
        assert actual.triples_samples == expected.triples_samples
        np.testing.assert_allclose(actual.triples_standard_error, expected.triples_standard_error,
                                   atol=2e-12, rtol=0)
    forward, _, energy_counts, _ = run(with_grad=False)
    np.testing.assert_allclose(forward, reference, atol=2e-12, rtol=0)
    assert energy_counts == dict(forward=1, reverse=0)
    accept.record_result('shared_domain_frame_' + str(int(shared_integrals)), {
        'energy_Eh': float(value), 'reference_energy_Eh': float(reference),
        'max_gradient_difference_Eh_per_bohr': float(np.max(np.abs(gradient.coords-ref_grad.coords))),
        'reference_frame_counts': ref_counts, 'shared_frame_counts': actual_counts,
        'shared_integral_reverse': shared_integrals,
        'corrected_fragments': list(case.indices),
    })


def test_selection_reuses_prepared_frame(molecular_group, monkeypatch):
    case = molecular_group
    counts = counted_frames(monkeypatch)
    # Select only the two physical fragments in one preparation group.
    selected = ccsd.build_static_selections(
        case.mf, case.static.mp2_static, matrix_kind='W', thresh_occ=1e-4, thresh_vir=3e-5,
        baseline_controls=case.controls, matrix_controls=case.controls,
        fragment_indices=case.indices)
    assert counts == dict(forward=1, reverse=0)
    for index in case.indices:
        assert ccsd._fragment_dimensions(selected, index) == ccsd._fragment_dimensions(case.static, index)
