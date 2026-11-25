"""Validate D selection with a four-water cluster and independent MP2 amplitudes."""

import numpy as np
import pytest

from pyscfad import config_update, dlno_stc, gto, scf
from pyscfad.dlno_stc import backend, ccsd, domain, prepare
from pyscfad.dlno_stc.test._boys_acceptance import record_result
from pyscfad.dlno_stc.test._saved_selection import save_from_run
from pyscfad.dlno_stc.test.test_backend import native


WATER4 = """
O  1.700660  1.233923  2.496518
H  2.607144  1.211629  2.189889
H  1.428477  2.144196  2.374859
O  2.639256  3.937102  2.706958
H  3.006278  4.781262  2.444430
H  3.326794  3.302385  2.502247
O  4.475299  2.478463  4.877118
H  3.709574  3.021233  4.689244
H  4.963905  2.459449  4.053477
O  0.910342  1.841863  4.591975
H  0.793674  2.791766  4.574515
H  1.858853  1.717412  4.640125
"""


def _build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.chkfile = None
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope='module', params=[
    'sto-3g', pytest.param('cc-pvdz', id='cc-pvdz_high_cost'),
])
def water4(request):
    native()
    mol = gto.Mole(atom=WATER4, basis=request.param, unit='Angstrom',
                   verbose=0, max_memory=6000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    return mol, _build_mf(mol)


def _sym(matrix):
    return (matrix + matrix.T) / 2


def _canonical_density(packet, targets):
    eo, co = np.linalg.eigh(packet['foo'])
    ev, cv = np.linalg.eigh(packet['fvv'])
    b = packet['B']
    canonical_b = np.einsum('ip,Pia,aq->Ppq', co, b, cv, optimize=True)
    eri = np.einsum('Pia,Pjb->ijab', canonical_b, canonical_b, optimize=True)
    denominator = eo[:, None, None, None] + eo[None, :, None, None]
    denominator = denominator - ev[None, None, :, None] - ev[None, None, None, :]
    t2 = eri / denominator
    # Project the target only after applying the canonical MP2 denominators.
    selected = np.einsum('si,jk,ac,bd,ikcd->sjab', co[list(targets)], co, cv, cv,
                         t2, optimize=True)
    spin = 2 * selected - selected.swapaxes(-1, -2)
    occupied = np.einsum('sjab,skab->jk', selected, spin, optimize=True)
    virtual = (np.einsum('sjac,sjbc->ab', selected, spin, optimize=True)
               + np.einsum('sjca,sjcb->ab', selected, spin, optimize=True)) / 2
    bare = np.einsum('Psa,Pjb->sjab', b[:, list(targets), :], b, optimize=True)
    energy = np.sum(bare * spin)
    return dict(energy=energy, occupied=_sym(occupied), virtual=_sym(virtual))


def test_water4_selected_d_matrices_and_pullback_match_canonical_mp2(water4):
    mol, mf = water4
    static = dlno_stc.build_domain_selections(mf, frozen=4, boys_grouping='atom')
    common = domain.rebuild_domain_data(mf, static)
    static = dlno_stc.materialize_domain_selections(mf, static, common=common)
    group = next(group for group in dlno_stc.build_domain_groups(static)
                 if len(group.target_indices) > 1)
    targets = group.target_indices[:2]
    packet = backend.host_inputs(prepare.prepare_boys_group_inputs(mf, common, static, group))
    gap = np.linalg.eigvalsh(packet['fvv'])[0] - np.linalg.eigvalsh(packet['foo'])[-1]
    assert gap > 0
    # Integrate far beyond the inverse of the smallest double-excitation gap.
    nodes, weights = np.polynomial.legendre.leggauss(80)
    scale = 10 / gap
    controls = dict(laplace_roots=scale * (nodes + 1), laplace_weights=scale * weights,
                    threads=1)
    metadata = dict(fragment_id=group.representative_index, target_indices=targets)
    actual = backend.solve_matrices(packet, metadata, controls, matrix_kind='D')
    assert actual['matrix_kind'] == 'D'
    assert actual['diagnostics']['energy_standard_error'] == 0
    reference = _canonical_density(packet, targets)
    separate = [backend.solve_matrices(packet, dict(metadata, target_indices=(target,)),
                                      controls, matrix_kind='D') for target in targets]
    value_errors = {}
    for name, expected in reference.items():
        value_errors[name] = float(np.max(np.abs(actual[name] - expected)))
        np.testing.assert_allclose(actual[name], expected, atol=2e-11, rtol=2e-9)
        np.testing.assert_allclose(actual[name], sum(result[name] for result in separate),
                                   atol=2e-12, rtol=2e-10)

    rng = np.random.default_rng(1904)
    seeds = dict(energy_bar=.37,
                 occupied_bar=rng.normal(size=reference['occupied'].shape),
                 virtual_bar=rng.normal(size=reference['virtual'].shape))
    bars = backend.pullback_matrices(packet, actual, **seeds)

    def objective(current):
        density = _canonical_density(current, targets)
        return sum(np.sum(seed * density[name.removesuffix('_bar')])
                   for name, seed in seeds.items())

    step = 1e-5
    pullback_errors = {}
    for name, value in packet.items():
        direction = rng.normal(size=value.shape)
        if name != 'B':
            direction = _sym(direction)
        direction /= np.linalg.norm(direction)
        plus = dict(packet, **{name: value + step * direction})
        minus = dict(packet, **{name: value - step * direction})
        finite = (objective(plus) - objective(minus)) / (2 * step)
        analytic = np.sum(bars[name] * direction)
        np.testing.assert_allclose(analytic, finite, atol=2e-9, rtol=2e-6, err_msg=name)
        pullback_errors[name] = float(abs(analytic - finite))
    record_result(f'water4-{mol.basis}-d-matrices', dict(value_errors=value_errors,
                  input_direction_errors=pullback_errors, energy_eh=float(actual['energy'])))


def test_water4_untruncated_d_matches_default_w_energy_and_gradient(water4):
    mol, mf = water4
    static = dlno_stc.build_domain_selections(mf, frozen=4, boys_grouping='system')
    selection = dict(thresh_occ=0., thresh_vir=0.)
    default_w = ccsd.build_static_selections(mf, static, **selection)
    explicit_d = ccsd.build_static_selections(mf, static, matrix_kind='D', **selection)
    assert default_w.lis_static.matrix_kind == 'W'
    assert explicit_d.lis_static.matrix_kind == 'D'
    # Full-space cc-pVDZ triples dominate the runtime. The truncated
    # cc-pVDZ case below exercises (T) with a D-selected LIS.
    options = dict(include_hf=True, ccsd_t=mol.basis == 'sto-3g', return_components=True)
    with config_update('pyscfad_moleintor_opt', True):
        w_energy, w_gradient, w_components = ccsd.value_and_grad(mol, _build_mf, default_w,
                                                                 **options)
        d_energy, d_gradient, d_components = ccsd.value_and_grad(mol, _build_mf, explicit_d,
                                                                 **options)
    np.testing.assert_allclose(d_energy, w_energy, atol=2e-9, rtol=0)
    np.testing.assert_allclose(d_gradient.coords, w_gradient.coords, atol=2e-7, rtol=0)
    np.testing.assert_allclose(d_components.e_mp2_base, w_components.e_mp2_base,
                               atol=2e-10, rtol=0)
    record_result(f'water4-{mol.basis}-untruncated', dict(d_energy_eh=float(d_energy),
                  w_energy_eh=float(w_energy),
                  gradient_max_error_eh_per_bohr=float(np.max(np.abs(
                      d_gradient.coords - w_gradient.coords)))))


def test_water4_truncated_d_nuclear_gradient_matches_fixed_selection_fd(water4):
    mol, mf = water4
    domains = dlno_stc.build_domain_selections(mf, frozen=4, boys_grouping='atom')
    domains = dlno_stc.materialize_domain_selections(mf, domains)
    plan = ccsd.build_static_selections(mf, domains, matrix_kind='D',
                                        thresh_occ=1e-4, thresh_vir=1e-4)
    options = dict(include_hf=True, ccsd_t=True, return_components=True)
    with config_update('pyscfad_moleintor_opt', True):
        static = save_from_run(mf, plan, **options)
        energy, gradient, components = ccsd.value_and_grad(mol, _build_mf, static, **options)
        assert any(row.nvir_lis < static.fragments[row.fragment_index].parent_dimensions[1]
                   for row in components.fragments)

        coords = np.asarray(mol.atom_coords())
        direction = np.zeros_like(coords)
        direction[1, 2], direction[4, 0] = 1., -.4
        step = 5e-5

        def displaced(sign):
            moved = mol.set_geom_(coords + sign * step * direction, unit='Bohr', inplace=False)
            value, _ = ccsd.kernel(_build_mf(moved), static, **options)
            return float(value)

        plus, minus = displaced(1), displaced(-1)
        finite = (plus - minus) / (2 * step)
    np.testing.assert_allclose(energy, (plus + minus) / 2, atol=5e-9, rtol=0)
    analytic = float(np.sum(gradient.coords * direction))
    np.testing.assert_allclose(analytic, finite, atol=2e-6, rtol=0)
    record_result(f'water4-{mol.basis}-truncated-d', dict(energy_eh=float(energy),
                  analytic_eh_per_bohr=analytic, finite_eh_per_bohr=finite,
                  error_eh_per_bohr=abs(analytic - finite), step_bohr=step,
                  lis_dimensions=[(row.nocc_lis, row.nvir_lis) for row in components.fragments]))
