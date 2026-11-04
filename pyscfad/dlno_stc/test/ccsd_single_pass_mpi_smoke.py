"""Small single-pass CC owner jobs, including workers with no assigned domain."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch


def run(output, *, serial=False, gradient=False, include_baseline=False,
        invalid=False, parallel_scf_jk=False):
    import jax
    import numpy as np
    from mpi4py import MPI
    from pyscfad import config_update
    from pyscfad.df import mpi_df_jk
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.test import _boys_acceptance as accept
    from pyscfad.lno import _df_shared

    comm = MPI.COMM_WORLD
    rank = comm.rank
    mol = accept.molecule()
    # Include resident JAX executables, as in the serial molecular CC fixture.
    # This is the explicit test budget; production memory policy is unchanged.
    mol.max_memory = 8000
    controls = dict(accept.controls(), threads=1)
    selected = (0, 2)  # One physical fragment in each of the two finite domains.
    observations = dict(groups=[], selected=[], forward=[], reverse=[], scf=0,
                        skeleton=0, service_forward=0, local_df=[])
    active_group = None
    plans = []

    def build_mf(current, **initial):
        observations['skeleton' if initial else 'scf'] += 1
        return accept.build_mf(current, **initial)

    def build_static(mf):
        plan = ccsd.build_static_selections(
            mf, accept.selection(mf), matrix_kind='W', thresh_occ=1e-4,
            thresh_vir=3e-5, baseline_controls=controls, matrix_controls=controls,
            fragment_indices=selected, single_pass=True)
        assert all(item is None for item in plan.fragments)
        assert plan.mp2_fragment_energies is None
        plans.append(plan)
        return plan

    original_group = ccsd._run_group
    def group(state, static, current, *args, **kwargs):
        nonlocal active_group
        assert mpi_df_jk.get_active_executor() is None
        assert static.deferred_selection is not None
        assert all(item is None for item in static.fragments)
        active_group = current.representative_index
        observations['groups'].append(active_group)
        try:
            return original_group(state, static, current, *args, **kwargs)
        finally:
            active_group = None

    original_select = ccsd._select_current_fragment
    def select(state, static, index, current, *args):
        assert active_group == current.representative_index
        observations['selected'].append(index)
        concrete = original_select(state, static, index, current, *args)
        assert concrete.deferred_selection is None
        assert concrete.fragments[index] is not None
        assert all(item is None for i, item in enumerate(concrete.fragments) if i != index)
        assert concrete.controls_identity == static.controls_identity
        assert all(item is None for item in static.fragments)
        return concrete

    original_forward = ccsd.backend.solve_matrices
    def forward(host, metadata, *args, **kwargs):
        assert active_group is not None, 'W was built outside its owner group'
        observations['forward'].append(metadata['fragment_id'])
        return original_forward(host, metadata, *args, **kwargs)

    original_reverse = ccsd.backend.pullback_matrices
    def reverse(host, result, *args, **kwargs):
        assert active_group is not None, 'W replay escaped its owner group'
        observations['reverse'].append(result['metadata']['fragment_id'])
        return original_reverse(host, result, *args, **kwargs)

    original_service_forward = mpi_df_jk.MPIDFJKExecutor._execute_forward
    def service_forward(self, *args, **kwargs):
        result = original_service_forward(self, *args, **kwargs)
        observations['service_forward'] += 1
        return result

    original_close = _df_shared.LocalDFReverse.close
    def close_local_df(scope):
        original_close(scope)
        assert active_group is not None
        observations['local_df'].append(dict(group=active_group, **scope.receipt))

    options = dict(baseline_controls=controls, matrix_controls=controls,
                   ccsd_t=True, fragment_indices=selected,
                   include_baseline=include_baseline, return_components=True,
                   triples_controls=dict(mode='stochastic', nsamples=16,
                       pilot_samples=16, global_seed=71, integral_transform='rotate',
                       variance_allocation='occupied_count', energy_tolerance=1e-3),
                   comm=None if serial else comm)
    # Preserve the omitted-option path to exercise the root-owned default.
    if parallel_scf_jk:
        options['parallel_scf_jk'] = True
    with ExitStack() as stack:
        stack.enter_context(config_update('pyscfad_moleintor_opt', True))
        stack.enter_context(config_update('pyscfad_scf_df_precond', False))
        for obj, name, function in ((ccsd, '_run_group', group),
                (ccsd, '_select_current_fragment', select),
                (ccsd.backend, 'solve_matrices', forward),
                (ccsd.backend, 'pullback_matrices', reverse),
                (_df_shared.LocalDFReverse, 'close', close_local_df),
                (mpi_df_jk.MPIDFJKExecutor, '_execute_forward', service_forward)):
            stack.enter_context(patch.object(obj, name, function))
        if gradient:
            result = ccsd.value_and_grad(mol, build_mf, build_static=build_static, **options)
        else:
            mf = build_mf(mol) if rank == 0 else None
            static = build_static(mf) if rank == 0 else None
            if rank:
                from pyscfad import scf
                mf = scf.RHF(mol).density_fit()
            if invalid:
                cases = (
                    ('schema', 'incompatible CC selection schema'),
                    ('controls', 'scientific controls differ'),
                    ('triples', 'single-pass W requires occupied_count'),
                )
                for kind, expected in cases:
                    current = replace(static, schema_version=-1) if rank == 0 and kind == 'schema' else static
                    settings = dict(options)
                    if kind == 'controls':
                        settings['matrix_controls'] = dict(controls, global_seed=99)
                    if kind == 'triples':
                        settings['triples_controls'] = dict(options['triples_controls'], variance_allocation='mp2_energy')
                    try:
                        ccsd.kernel(mf, current, **settings)
                    except (RuntimeError, ValueError) as exc:
                        assert expected in str(exc), str(exc)
                    else:
                        raise AssertionError(f'{kind} was not rejected')
                assert not observations['groups']
                result = None
            else:
                result = ccsd.kernel(mf, static, **options)
        jax.block_until_ready(result)
    gathered = comm.allgather(observations)
    if rank == 0:
        values = dict(observations=gathered, ranks=comm.size, invalid=invalid,
                      parallel_scf_jk=parallel_scf_jk)
        assert observations['scf'] == 1
        assert all(row['scf'] == 0 for row in gathered[1:])
        service_enabled = parallel_scf_jk and not serial and comm.size > 1
        if gradient:
            assert all(row['skeleton'] == 1 + int(service_enabled) for row in gathered[1:])
        if not invalid:
            assert all((row['service_forward'] > 0) == service_enabled for row in gathered)
            expected_forward = list(range(8)) if include_baseline else list(selected)
            assert sorted(i for row in gathered for i in row['forward']) == expected_forward
            assert sorted(i for row in gathered for i in row['selected']) == list(selected)
            assert sorted(i for row in gathered for i in row['reverse']) == (expected_forward if gradient else [])
            assert sorted(i for row in gathered for i in row['groups']) == [0, 2]
            # MPI owners must inherit the serial half-transformed DF reuse,
            # not rebuild raw integrals separately for W and the selected LIS.
            receipts = [receipt for row in gathered for receipt in row['local_df']]
            assert sorted(row['group'] for row in receipts) == [0, 2]
            for receipt in receipts:
                assert receipt['cache_mode'] == 'half'
                assert receipt['raw_source_passes'] == receipt['half_builds'] == 1
                assert receipt['metric_builds'] == receipt['cholesky_builds'] == 1
                assert receipt['half_rotations'] == 1
                assert receipt['geometry_passes'] == receipt['metric_passes'] == int(gradient)
                assert receipt['half_resident_bytes'] == 0
            if comm.size > 2:
                assert all(not row['groups'] and not row['selected'] and not row['forward']
                           for row in gathered[2:])
            if comm.size > 1:
                assert gathered[1]['selected'] == [2]
            if gradient:
                energy, bar, components = result
                values['gradient'] = np.asarray(bar.coords).tolist()
                assert np.isfinite(values['gradient']).all()
            else:
                energy, components = result
            values['energy'] = float(energy)
            values['baseline'] = float(components.e_mp2_base)
            values['weak'] = float(components.e_mp2_weak)
            values['fragments'] = []
            assert np.isfinite(values['energy'])
            assert [row.fragment_index for row in components.fragments] == list(selected)
            for row in components.fragments:
                diagnostics = row.triples_diagnostics
                assert diagnostics['variance_allocation'] == 'occupied_count'
                assert diagnostics['total_active_occupied'] == 8
                assert diagnostics['fragment_occupied_count'] == 1
                assert diagnostics['fragment_variance_fraction'] == 1/8
                values['fragments'].append(dict(index=row.fragment_index,
                    energies=[row.e_mp2_lis, row.e_ccsd, row.e_triples],
                    dimensions=[row.nocc_lis, row.nvir_lis], samples=row.triples_samples,
                    target=diagnostics['target_standard_error']))
            assert all(item is None for item in plans[0].fragments)
        output.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
        print('CC single-pass MPI smoke PASS', flush=True)
    else:
        assert result is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--gradient', action='store_true')
    parser.add_argument('--include-baseline', action='store_true')
    parser.add_argument('--invalid', action='store_true')
    parser.add_argument('--parallel-scf-jk', action='store_true')
    args = parser.parse_args()
    run(**vars(args))
