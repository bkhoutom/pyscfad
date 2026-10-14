"""Selected finite-domain molecular integration, including weak and HF terms."""
from dataclasses import replace
import jax
import numpy as np
import pytest
from pyscfad import gto
from pyscfad.dlno_stc.test import _boys_acceptance as accept


def test_coordinate_only_driver_rejects_basis_leaves_before_scf():
    from pyscfad import dlno_stc
    mol = gto.Mole(atom='H 0 0 0; H 0 0 1', basis='sto-3g', verbose=0)
    mol.build()
    def forbidden_scf(mol):
        raise AssertionError('SCF must not run for unsupported basis derivatives')
    with pytest.raises(NotImplementedError, match='coordinate|nuclear'):
        dlno_stc.value_and_grad(mol, forbidden_scf, controls=accept.controls())


def test_unsupported_public_targets_are_rejected_before_scf():
    from pyscfad import dlno_stc
    from pyscfad.dlno_stc._selection import DomainSelections
    from types import SimpleNamespace
    static = DomainSelections(
        frozen=0, thresholds=None, active_occ_indices=np.array([0]),
        active_vir_indices=np.array([], dtype=int),
        pao_projected_out_indices=np.array([0]),
        pao_parent_ao_indices=np.array([], dtype=int), ao2pao_map=np.array([], dtype=int),
        frag_lolist=(np.array([0]),), frag_atmlist=(None,),
        strong_mask=np.ones((1, 1), dtype=bool),
        fragments=(SimpleNamespace(iao_indices=np.array([0]), strong_fragments=np.array([0])),),
        lo_type='boys',
    )
    def forbidden_scf(mol):
        raise AssertionError('unsupported targets reached SCF')
    mol = accept.molecule()
    for invalid in (replace(static, lo_type='iao'),
                    replace(static, frag_lolist=(np.array([0, 1]),)),
                    replace(static, frag_lolist=(np.array([.9]),))):
        with pytest.raises(ValueError, match='singleton'):
            dlno_stc.value_and_grad(mol, forbidden_scf, invalid, controls=accept.controls())


def test_finite_grouped_energy_gradient_weak_and_hf():
    from pyscfad import dlno_stc
    from pyscfad.dlno_stc._pairs import weak_pair_indices
    mol = accept.molecule()
    mf = accept.build_mf(mol)
    static = accept.selection(mf)
    assert any(len(f.extended_ao_indices) < mol.nao for f in static.fragments)
    assert tuple(weak_pair_indices(static))
    controls = accept.controls()
    calls = dict(scf=0, selection=0)

    def build_mf(current):
        calls['scf'] += 1
        return accept.build_mf(current)

    def build_static(reference):
        calls['selection'] += 1
        selected = accept.selection(reference)
        # Selection-only caches must not change the live SCF output tree
        # retained by its pullback, or remove its orbital-energy response.
        reference.mo_energy = None
        return selected

    energy, bar = dlno_stc.value_and_grad(
        mol, build_mf, build_static=build_static, controls=controls)
    assert calls == dict(scf=1, selection=1)
    singleton, singleton_bar = dlno_stc.value_and_grad(
        mol, accept.build_mf, static, controls=controls, merge_identical=False)
    np.testing.assert_allclose(energy, singleton, atol=2e-10, rtol=0)
    np.testing.assert_allclose(bar.coords, singleton_bar.coords, atol=2e-8, rtol=0)
    np.testing.assert_allclose(dlno_stc.kernel(mf, static, controls=controls), energy,
                               atol=2e-10, rtol=0)
    # Independent Phase2 per-singleton assembly closes the same three-input functional.
    np.testing.assert_allclose(energy, accept.correlation_energy(mf, static), atol=2e-10, rtol=0)
    total, total_bar = dlno_stc.value_and_grad(
        mol, accept.build_mf, static, controls=controls, include_hf=True)
    from pyscfad.dlno_stc.driver import _initialize_scf
    current, pullback = _initialize_scf(mol, accept.build_mf)
    hf, hf_pullback = jax.vjp(lambda x: x.e_tot, current)
    hf_mf_bar, = hf_pullback(jax.numpy.ones_like(hf))
    hf_bar, = pullback(hf_mf_bar)
    np.testing.assert_allclose(total-energy, hf, atol=2e-10, rtol=0)
    np.testing.assert_allclose(total_bar.coords-bar.coords, hf_bar.coords, atol=2e-7, rtol=2e-5)
    coords = np.asarray(mol.atom_coords())
    direction = np.zeros_like(coords); direction[1, 2] = 1
    step = 1e-4
    def displaced(sign):
        moved = mol.set_geom_(coords+sign*step*direction, unit='Bohr', inplace=False)
        return dlno_stc.kernel(accept.build_mf(moved), static, controls=controls)
    finite = (displaced(1)-displaced(-1))/(2*step)
    np.testing.assert_allclose(np.sum(bar.coords*direction), finite, atol=2e-6, rtol=0)
    accept.record_result('phase3_finite_driver', {
        'correlation_energy_Eh': float(energy), 'total_energy_Eh': float(total),
        'grouped_minus_singleton_energy_Eh': float(energy-singleton),
        'gradient_max_abs_Eh_per_bohr': float(np.max(np.abs(bar.coords-singleton_bar.coords))),
        'total_gradient_Eh_per_bohr': np.asarray(total_bar.coords).tolist(),
        'fd_step_bohr': step, 'fd_slope_Eh_per_bohr': float(finite),
        'analytic_slope_Eh_per_bohr': float(np.sum(bar.coords*direction)),
        'fd_absolute_error_Eh_per_bohr': float(abs(np.sum(bar.coords*direction)-finite)),
        'weak_pair_count': len(tuple(weak_pair_indices(static))),
        'ao_count': mol.nao,
        'finite_ao_support_counts': [len(f.extended_ao_indices) for f in static.fragments],
    })
    mf.mo_occ = np.asarray(mf.mo_occ).copy(); mf.mo_occ[0] = 1
    with pytest.raises(NotImplementedError, match='closed.shell'):
        dlno_stc.kernel(mf, static, controls=controls)


@pytest.mark.parametrize('options, error, match', [
    ({'build_static': object()}, TypeError, 'callable'),
    ({'static': object(), 'build_static': accept.selection}, ValueError, 'static'),
    ({'scope': 'system', 'build_static': accept.selection}, ValueError, 'domain'),
])
def test_domain_builder_options_are_rejected_before_scf(options, error, match):
    from pyscfad import dlno_stc

    def forbidden_scf(mol):
        raise AssertionError('invalid domain builder options reached SCF')

    with pytest.raises(error, match=match):
        dlno_stc.value_and_grad(
            accept.molecule(), forbidden_scf, controls=accept.controls(), **options)
