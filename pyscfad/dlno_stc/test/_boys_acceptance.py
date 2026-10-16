"""Small standalone molecular acceptance helpers shared by Phase 2/2b tests."""

import json
import os
from pathlib import Path
from functools import lru_cache

import jax
import numpy as np

from pyscfad import config_update, gto, scf


LO_OPTIONS = {"conv_tol": 1e-12, "conv_tol_grad": 1e-9}
TARGET_ORDER = (4, 0, 5, 1, 6, 2, 7, 3)


def record_result(name, values):
    """Optionally retain measured acceptance results in the experiment root."""
    directory = os.environ.get("PHASE2_RESULTS_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (name + ".json")).write_text(json.dumps(values, indent=2) + "\n")


def build_mf(mol, *, mo_coeff_init=None, mo_energy_init=None,
             mo_occ_init=None, e_tot_init=None):
    from pyscfad.dlno_stc.adjoint import fix_restart_mo_phases

    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    if mo_coeff_init is not None:
        # MPI workers install the root gauge without running SCF or fixing phases.
        mf.mo_coeff = mo_coeff_init
        mf.mo_energy = mo_energy_init
        mf.mo_occ = mo_occ_init
        mf.e_tot = e_tot_init
        mf.converged = True
        return mf
    mf.kernel()
    return fix_restart_mo_phases(mf)


def molecule():
    mol = gto.Mole(
        atom="O 0 0 0; H 0.1 -0.75 0.57; H 0 0.8 0.61; "
             "O 0 0 8; H 0.13 -0.74 8.57; H 0.02 0.81 8.61",
        basis="sto-3g", verbose=0, max_memory=3000,
    )
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    mol.incore_anyway = True
    return mol


def selection(mf):
    from pyscfad.dlno_stc import BoysDomainThresholds, build_domain_selections

    return build_domain_selections(
        mf, frozen=2, thresholds=BoysDomainThresholds(pair_energy=1e-4),
        lo_kwargs=LO_OPTIONS, frag_lolist=tuple((i,) for i in TARGET_ORDER),
    )


def controls():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    return {"mode": "deterministic", "laplace_roots": 20 * (nodes + 1),
            "laplace_weights": 20 * weights}


def selected_result(packet, static, index, *, with_grad=True, target_indices=None):
    from pyscfad.dlno_stc import _stc_mp2

    return _stc_mp2.solve_selected(
        *(np.ascontiguousarray(packet[name]) for name in ("foo", "fvv", "B")),
        ((static.fragments[index].target_index,) if target_indices is None
         else tuple(target_indices)), controls(), with_grad=with_grad,
    )


def _strong_jobs(static, groups):
    from pyscfad.dlno_stc.prepare import prepare_inputs

    if groups is None:
        return tuple((lambda mf, common, i=i: prepare_inputs(mf, common, static, i),
                      i, (static.fragments[i].target_index,))
                     for i in range(len(static.fragments)))
    from pyscfad.dlno_stc.prepare import prepare_boys_group_inputs

    return tuple((lambda mf, common, g=g: prepare_boys_group_inputs(mf, common, static, g),
                  g.representative_index, g.target_indices) for g in groups)


def correlation_energy(mf, static, groups=None):
    from pyscfad.dlno_stc.domain import rebuild_domain_data
    from pyscfad.dlno_stc._pairs import weak_pair_energy, weak_pair_indices

    common = rebuild_domain_data(mf, static)
    energy = sum(float(selected_result(
        prepare(mf, common), static, index, with_grad=False, target_indices=targets,
    )["energy"]) for prepare, index, targets in _strong_jobs(static, groups))
    return energy + sum(float(weak_pair_energy(mf, common, static, i, j))
                        for i, j in weak_pair_indices(static))


def correlation_value_and_grad(mol, static, groups=None):
    """Close independent per-job bars through one common and one SCF VJP."""
    from pyscfad.dlno_stc.adjoint import (
        add_cotangent, apply_selected_pullback, zero_term_cotangents,
    )
    from pyscfad.dlno_stc.domain import rebuild_domain_data
    from pyscfad.dlno_stc.prepare import prepare_inputs
    from pyscfad.dlno_stc._pairs import weak_pair_energy, weak_pair_indices

    with (config_update("pyscfad_scf_implicit_diff", True),
          config_update("pyscfad_scf_first_order_custom", False)):
        mf, scf_pullback = jax.vjp(build_mf, mol)
    common, common_pullback = jax.vjp(lambda x: rebuild_domain_data(x, static), mf)
    mf_bar, common_bar = zero_term_cotangents(mf, common)
    energy = 0.0
    for prepare, index, targets in _strong_jobs(static, groups):
        packet, pullback = jax.vjp(prepare, mf, common)
        result = selected_result(packet, static, index, target_indices=targets)
        energy += float(result["energy"])
        left, right = apply_selected_pullback(pullback, packet, result)
        mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, left)
        common_bar = jax.tree_util.tree_map(add_cotangent, common_bar, right)
        del pullback, result
    for i, j in weak_pair_indices(static):
        value, pullback = jax.vjp(
            lambda x, c: weak_pair_energy(x, c, static, i, j), mf, common,
        )
        energy += float(value)
        left, right = pullback(jax.numpy.ones_like(value))
        mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, left)
        common_bar = jax.tree_util.tree_map(add_cotangent, common_bar, right)
        del pullback
    (from_common,) = common_pullback(common_bar)
    mf_bar = jax.tree_util.tree_map(add_cotangent, mf_bar, from_common)
    (mol_bar,) = scf_pullback(mf_bar)
    jax.block_until_ready(mol_bar)
    return energy, mol_bar


@lru_cache(maxsize=1)
def _grouped_case():
    from pyscfad.dlno_stc import build_domain_groups
    from pyscfad.dlno_stc.domain import rebuild_domain_data

    mf = build_mf(molecule())
    static = selection(mf)
    return mf, static, rebuild_domain_data(mf, static), build_domain_groups(static)


def _write_grouped_result(directory, name, values):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + '.json')).write_text(json.dumps(values, indent=2) + '\n')
    record_result(name, values)


@lru_cache(maxsize=1)
def _grouped_response():
    mf, static, _, groups = _grouped_case()
    singleton_energy, singleton_bar = correlation_value_and_grad(mf.mol, static)
    grouped_energy, grouped_bar = correlation_value_and_grad(mf.mol, static, groups)
    return singleton_energy, np.asarray(singleton_bar.coords), grouped_energy, np.asarray(grouped_bar.coords)


def grouped_preparation_and_response(output_dir):
    """Check actual finite-support groups, identical packets, and full response."""
    import csv
    from collections import Counter
    from pyscfad.dlno_stc.prepare import prepare_inputs, prepare_boys_group_inputs
    from pyscfad.dlno_stc.adjoint import apply_selected_pullback, pullback_inputs
    from pyscfad.dlno_stc._pairs import weak_pair_energy, weak_pair_indices

    mf, static, common, groups = _grouped_case()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata, residuals = [], [], {name: 0.0 for name in ('foo', 'fvv', 'B')}
    for group_id, group in enumerate(groups):
        representative = static.fragments[group.representative_index]
        packet = prepare_boys_group_inputs(mf, common, static, group)
        naux, nocc, nvir = packet['B'].shape
        labels = [int(static.frag_lolist[i][0]) for i in group.fragment_indices]
        partners = [int(static.frag_lolist[i][0]) for i in representative.strong_fragments]
        metadata.append({'group': group_id, 'members': list(group.fragment_indices),
                         'global_targets': labels, 'local_targets': list(group.target_indices),
                         'representative': group.representative_index,
                         'ordered_partner_labels': partners,
                         'extended_atoms': representative.extended_atoms.tolist(),
                         'extended_ao_indices': representative.extended_ao_indices.tolist(),
                         'dimensions': [naux, nocc, nvir]})
        for member, label, slot in zip(group.fragment_indices, labels, group.target_indices):
            rows.append({'group': group_id, 'fragment_index': member, 'global_target': label,
                         'members': json.dumps(group.fragment_indices), 'local_slot': slot,
                         'representative': group.representative_index,
                         'ordered_partner_labels': json.dumps(partners),
                         'atom_support_size': len(representative.extended_atoms),
                         'ao_support_size': len(representative.extended_ao_indices),
                         'pao_support_size': len(representative.strong_virtual.parent_columns),
                         'naux': naux, 'nocc': nocc, 'nvir': nvir})
            singleton = prepare_inputs(mf, common, static, member)
            for name in packet:
                residuals[name] = max(residuals[name], float(np.max(np.abs(
                    np.asarray(packet[name]) - np.asarray(singleton[name])))))
                np.testing.assert_allclose(packet[name], singleton[name], atol=1e-11, rtol=1e-11)
    csv_directories = {output_dir}
    if os.environ.get('PHASE2_RESULTS_DIR'):
        csv_directories.add(Path(os.environ['PHASE2_RESULTS_DIR']))
    for directory in csv_directories:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'phase2b_groups.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    summary = {'support_regime': 'finite seven-AO domains in fourteen-AO separated-water system',
               'singleton_count': len(static.fragments), 'solver_group_count': len(groups),
               'group_size_distribution': dict(Counter(len(g.fragment_indices) for g in groups)),
               'groups': metadata, 'nonmergeable_target_dependent_constructions': [],
               'preparation_max_abs': residuals}
    _write_grouped_result(output_dir, 'group_metadata', summary)
    # Inspect the actual fixture before asserting its expected reduction.
    assert [g.fragment_indices for g in groups] == [(0, 1, 4, 5), (2, 3, 6, 7)]
    assert sorted(i for g in groups for i in g.fragment_indices) == list(range(8))
    assert all(tuple(g.target_indices) == (0, 1, 2, 3) for g in groups)
    assert all(len(static.fragments[g.representative_index].extended_ao_indices) < mf.mol.nao
               for g in groups)
    replay_residual = 0.0
    for group in groups:
        def prepare(x, c):
            return prepare_boys_group_inputs(x, c, static, group)
        packet, pullback = jax.vjp(prepare, mf, common)
        result = selected_result(packet, static, group.representative_index,
                                 target_indices=group.target_indices)
        direct = apply_selected_pullback(pullback, packet, result)
        replay = pullback_inputs(prepare, (mf, common), packet,
                                 dict(result, metadata={'cotangent_seed': 1.0}))
        for actual, expected in zip(jax.tree_util.tree_leaves(replay),
                                    jax.tree_util.tree_leaves(direct)):
            if actual.dtype == jax.dtypes.float0:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
                if actual.size:
                    replay_residual = max(replay_residual, float(np.max(np.abs(actual - expected))))
    singleton_energy, singleton_gradient, grouped_energy, grouped_gradient = _grouped_response()
    np.testing.assert_allclose(grouped_energy, singleton_energy, atol=1e-9, rtol=0)
    np.testing.assert_allclose(grouped_gradient, singleton_gradient, atol=1e-6, rtol=0)
    pairs = tuple(weak_pair_indices(static))
    assert len(pairs) == len(set(pairs)) == 16
    weak_energy = sum(float(weak_pair_energy(mf, common, static, i, j)) for i, j in pairs)
    assert abs(weak_energy) > 1e-10
    np.testing.assert_allclose(weak_energy, -3.42607799968241e-7, atol=1e-15, rtol=0)
    np.testing.assert_allclose(grouped_energy, correlation_energy(mf, static, groups), atol=1e-10, rtol=0)
    baseline_path = (Path(__file__).resolve().parents[4] / 'TEST_general/simplify_domain/phase2'
                     / 'runs/run-20261011T212042Z-ZL3Py3/results/coordinate_response.json')
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    baseline_comparison = {'available': baseline is not None}
    if baseline is not None:
        np.testing.assert_allclose(grouped_energy, baseline['correlation_energy_Eh'], atol=1e-9, rtol=0)
        np.testing.assert_allclose(grouped_gradient, baseline['gradient_Eh_per_bohr'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(-grouped_gradient, baseline['force_Eh_per_bohr'], atol=1e-6, rtol=0)
        baseline_comparison.update({'path': str(baseline_path),
            'energy_delta_Eh': grouped_energy - baseline['correlation_energy_Eh'],
            'gradient_max_abs_Eh_per_bohr': float(np.max(np.abs(
                grouped_gradient - np.asarray(baseline['gradient_Eh_per_bohr']))))})
    summary.update({'status': 'passed', 'quantity': 'correlation only; no HF term',
                    'singleton_correlation_energy_Eh': singleton_energy,
                    'grouped_correlation_energy_Eh': grouped_energy,
                    'energy_delta_Eh': grouped_energy - singleton_energy,
                    'singleton_gradient_Eh_per_bohr': singleton_gradient.tolist(),
                    'grouped_gradient_Eh_per_bohr': grouped_gradient.tolist(),
                    'grouped_force_Eh_per_bohr': (-grouped_gradient).tolist(),
                    'gradient_delta_Eh_per_bohr': (grouped_gradient - singleton_gradient).tolist(),
                    'gradient_max_abs_Eh_per_bohr': float(np.max(np.abs(grouped_gradient - singleton_gradient))),
                    'replay_cotangent_max_abs': replay_residual,
                    'weak_unordered_pairs': [list(pair) for pair in pairs],
                    'weak_energy_Eh': weak_energy, 'weak_pair_prefactor': -4,
                    'phase2_baseline_comparison': baseline_comparison})
    _write_grouped_result(output_dir, 'grouped_molecular_response', summary)
    return summary


def grouped_coordinate_response(output_dir):
    """Reuse the Phase 2 correlation-only coordinate direction and both steps."""
    mf, static, _, groups = _grouped_case()
    _, _, energy, gradient = _grouped_response()
    coords = np.asarray(mf.mol.atom_coords())
    direction = np.zeros_like(coords)
    direction[1, 2] = 1.0
    actual = float(np.sum(gradient * direction))
    differences = []
    for step in (1e-5, 5e-6):
        values = []
        for sign in (1, -1):
            moved = mf.mol.set_geom_(coords + sign * step * direction,
                                     unit='Bohr', inplace=False)
            values.append(correlation_energy(build_mf(moved), static, groups))
        finite = (values[0] - values[1]) / (2 * step)
        differences.append({'step_bohr': step, 'plus_correlation_energy_Eh': values[0],
                            'minus_correlation_energy_Eh': values[1],
                            'central_difference_Eh_per_bohr': finite,
                            'absolute_error_Eh_per_bohr': abs(actual - finite)})
        np.testing.assert_allclose(actual, finite, rtol=1e-6, atol=1e-8)
    summary = {'status': 'passed', 'quantity': 'correlation only; fixed selections and grouping worklist',
               'correlation_energy_Eh': energy, 'adjoint_Eh_per_bohr': actual,
               'gradient_Eh_per_bohr': gradient.tolist(), 'force_Eh_per_bohr': (-gradient).tolist(),
               'fixed_choice_central_differences': differences}
    baseline_path = (Path(__file__).resolve().parents[4] / 'TEST_general/simplify_domain/phase2'
                     / 'runs/run-20261011T212042Z-ZL3Py3/results/coordinate_response.json')
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        np.testing.assert_allclose(actual, baseline['adjoint_Eh_per_bohr'], atol=1e-6, rtol=0)
        for current, previous in zip(differences, baseline['fixed_choice_central_differences']):
            assert current['step_bohr'] == previous['step_bohr']
            np.testing.assert_allclose(current['central_difference_Eh_per_bohr'],
                                       previous['central_difference_Eh_per_bohr'], rtol=1e-6, atol=1e-8)
            current['phase2_central_difference_delta_Eh_per_bohr'] = (
                current['central_difference_Eh_per_bohr'] - previous['central_difference_Eh_per_bohr'])
        summary['phase2_adjoint_delta_Eh_per_bohr'] = actual - baseline['adjoint_Eh_per_bohr']
    _write_grouped_result(output_dir, 'grouped_coordinate_response', summary)
    return summary
