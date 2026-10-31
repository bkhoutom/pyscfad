"""Validate smooth fragment/LIS forces and compare stochastic seeds.

Example: --xyz-file examples/dlno_stc/c5h6_force_compare.xyz --frozen 5
         --output-dir results/smooth_c5h6
         --samples 16446469 --seeds 71 72 73
Use --smooth-deterministic on small systems to verify exact-force recovery.
"""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--xyz-file', type=Path, required=True)
    parser.add_argument('--basis', default='cc-pvdz')
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument('--boys-grouping', choices=('singleton', 'atom', 'molecule'), default='molecule')
    parser.add_argument('--lno-thresh-occ', type=float, default=0.)
    parser.add_argument('--lno-thresh-vir', type=float, default=0.)
    parser.add_argument('--finite-domains', action='store_true')
    parser.add_argument('--selection-only', action='store_true')
    parser.add_argument('--samples', type=int, default=16446469,
                        help='production samples per S when triples energy tolerance is zero')
    parser.add_argument('--triples-energy-tolerance-mha', type=float, default=.05,
                        help='total (T) energy SE target in mEh; zero uses fixed counts')
    parser.add_argument('--triples-pilot-samples', type=int, default=10000,
                        help='independent pilot samples per triples Laplace point')
    parser.add_argument('--triples-laplace-points', type=int, default=10)
    parser.add_argument('--triples-trace-exponent', type=float, default=5.0,
                        help='exponent in the smooth triples point-variance trace model')
    parser.add_argument('--triples-budget-energy', choices=('mp2', 'ccsd'), default='mp2',
                        help='divide total variance in proportion to absolute fragment correlation energies')
    parser.add_argument('--seeds', type=int, nargs='*', default=[71, 72, 73])
    parser.add_argument('--smooth-deterministic', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--sampling-fragment', type=int,
                        help='test one zero-based physical S correction and its gradient; requires --reference or --fragment-deterministic, skips full-force comparison')
    parser.add_argument('--fragment-deterministic', action='store_true',
                        help='also compute the conventional deterministic force for --sampling-fragment')
    parser.add_argument('--fd-coordinate', type=int, nargs=2, metavar=('ATOM', 'AXIS'))
    parser.add_argument('--fd-step', type=float, default=1e-4, help='Bohr')
    parser.add_argument('--cc-conv-tol', type=float, default=1e-10)
    parser.add_argument('--cc-conv-tol-normt', type=float, default=1e-8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from benchmark_affinity import configure_cpu_workload
    configure_cpu_workload(args.threads)
    if not math.isfinite(args.triples_energy_tolerance_mha) or args.triples_energy_tolerance_mha < 0:
        parser.error('--triples-energy-tolerance-mha must be finite and nonnegative')
    if args.triples_pilot_samples < 2 or args.triples_laplace_points < 1:
        parser.error('triples pilots require at least two samples and at least one Laplace point')
    if not math.isfinite(args.triples_trace_exponent) or args.triples_trace_exponent <= 0:
        parser.error('--triples-trace-exponent must be finite and positive')
    if args.sampling_fragment is not None:
        if args.sampling_fragment < 0 or (args.reference is None and not args.fragment_deterministic):
            parser.error('--sampling-fragment requires a nonnegative index and --reference or --fragment-deterministic')
        if args.reference is None and args.triples_budget_energy == 'ccsd':
            parser.error('CCSD fragment budgeting requires --reference')
        if args.smooth_deterministic or args.fd_coordinate:
            parser.error('--sampling-fragment is a stochastic correction/gradient test')
    elif args.fragment_deterministic:
        parser.error('--fragment-deterministic requires --sampling-fragment')
    import jax
    import numpy as np
    from pyscf import gto as pgto, scf as pscf
    from pyscfad import gto, scf, dlno_stc
    from pyscfad.dlno_stc import ccsd
    from pyscfad.dlno_stc.laplace import make_laplace_grid
    from pyscfad.lno.ccsd import RCCSD
    RCCSD.conv_tol = args.cc_conv_tol
    RCCSD.conv_tol_normt = args.cc_conv_tol_normt
    RCCSD.max_cycle = 200
    basis = args.basis
    auxbasis = None
    if args.basis.lower().replace(' ', '') == 'def2-sv(p)':
        # For H/C: def2-SVP on C, its unpolarized two-s-shell basis on H.
        # Spell out the shells; do not silently substitute def2-SVP on H.
        basis = {'C': pgto.basis.load('def2-svp', 'C'),
                 'H': [shell for shell in pgto.basis.load('def2-svp', 'H') if shell[0] == 0]}
        auxbasis = 'def2-svp-jkfit'

    started = time.perf_counter()
    setup_timings = {}
    phase = started
    def report(message):
        print(f'[{time.perf_counter()-started:.1f}s] {message}', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (args.output_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    mol = gto.Mole(atom=str(args.xyz_file.resolve()), basis=basis, unit='Angstrom',
                   max_memory=800000, verbose=4)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    refmol = pgto.M(atom=str(args.xyz_file.resolve()), basis=basis, unit='Angstrom',
                    max_memory=800000, verbose=0)
    ref = pscf.RHF(refmol).density_fit(auxbasis=auxbasis)
    ref.conv_tol, ref.conv_tol_grad = 1e-9, 1e-6
    ref.kernel()
    if not ref.converged:
        raise RuntimeError('reference SCF failed')
    dm0 = np.asarray(ref.make_rdm1())
    if args.reference:
        saved_reference = json.loads(args.reference.read_text())
        if 'scf_density_ao' in saved_reference:
            dm0 = np.asarray(saved_reference['scf_density_ao'])
    setup_timings['molecule_and_reference_scf_seconds'] = time.perf_counter()-phase
    def build_mf(current):
        mf = scf.RHF(current).density_fit(auxbasis=auxbasis)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-9, 1e-6, None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            raise RuntimeError('SCF failed')
        return mf
    phase = time.perf_counter()
    mf = build_mf(mol)
    jax.block_until_ready(mf)
    setup_timings['differentiable_scf_seconds'] = time.perf_counter()-phase
    phase = time.perf_counter()
    domains = dlno_stc.build_domain_selections(mf, frozen=args.frozen,
        force_full_domains=not args.finite_domains, boys_grouping=args.boys_grouping,
        lo_kwargs=dict(init_guess='atomic', conv_tol=1e-10,
                       gmres_options=dict(restart=120, maxiter=100)))
    setup_timings['domain_selection_seconds'] = time.perf_counter()-phase
    phase = time.perf_counter()
    eo = np.asarray(mf.mo_energy)[domains.active_occ_indices]
    ev = np.asarray(mf.mo_energy)[domains.active_vir_indices]
    roots, weights, _ = make_laplace_grid(eo, ev, kind='fitted', points=6, lambda_gradient=.1)
    setup_timings['mp2_grid_fit_seconds'] = time.perf_counter()-phase
    grid = dict(laplace_roots=roots, laplace_weights=weights)
    baseline = dict(mode='deterministic', threads=args.threads, global_seed=666, **grid)
    matrix = dict(mode='deterministic', threads=args.threads, memory_limit_mb=800000, **grid)
    selection_options = dict(matrix_kind='W', thresh_occ=args.lno_thresh_occ, thresh_vir=args.lno_thresh_vir,
        baseline_controls=baseline, matrix_controls=matrix, merge_identical=True)
    phase = time.perf_counter()
    ordinary = ccsd.build_static_selections(mf, domains, smooth=False, **selection_options)
    setup_timings['conventional_lis_selection_seconds'] = time.perf_counter()-phase
    phase = time.perf_counter()
    smooth = ccsd.build_static_selections(mf, domains,
        smooth_laplace_points=args.triples_laplace_points, **selection_options)
    setup_timings['smooth_lis_and_triples_grids_seconds'] = time.perf_counter()-phase
    fragment_settings = [dict(index=i, target_orbitals=len(domains.physical_groups[i]),
        nocc_lis=ccsd._fragment_dimensions(smooth, i)[0], nvir_lis=ccsd._fragment_dimensions(smooth, i)[1],
        occupied_block_size=frames[0].anchor.shape[1] if frames else 0,
        virtual_block_size=frames[1].anchor.shape[1] if frames else 0,
        mp2_budget_energy_eh=smooth.mp2_fragment_energies[i],
        triples_quadrature=smooth.lis_static.smooth_grid[i][2] if frames else None)
        for i, frames in enumerate(smooth.lis_static.smooth_frames)]
    save('settings.json', dict(basis=args.basis, frozen=args.frozen, threads=args.threads,
         max_memory_mb=800000, hf_energy_eh=float(mf.e_tot), samples=args.samples,
         seeds=args.seeds, smooth_frame_mode="full",
         triples_energy_tolerance_meh=args.triples_energy_tolerance_mha,
         triples_pilot_samples_per_point=args.triples_pilot_samples,
         triples_laplace_points=args.triples_laplace_points,
         triples_trace_exponent=args.triples_trace_exponent,
         triples_dressing="asymmetric", triples_dressing_backend="native",
         triples_adaptive=True, setup_timings=setup_timings,
         sampling_fragment=args.sampling_fragment,
         triples_budget_energy=args.triples_budget_energy,
         cc_conv_tol=args.cc_conv_tol, cc_conv_tol_normt=args.cc_conv_tol_normt,
         boys_grouping=args.boys_grouping, force_full_domains=not args.finite_domains,
         lno_thresh_occ=args.lno_thresh_occ, lno_thresh_vir=args.lno_thresh_vir,
         nao=mol.nao, nocc_active=len(eo), nvir_active=len(ev), auxbasis=auxbasis or 'auto',
         fragments=fragment_settings))
    report(f'Physical groups / LIS dimensions: {[(x["target_orbitals"], x["nocc_lis"], x["nvir_lis"]) for x in fragment_settings]}')
    if args.selection_only:
        return
    options = dict(baseline_controls=baseline, matrix_controls=matrix,
        matrix_kind='W', ccsd_t=True, include_hf=True, return_components=True, merge_identical=True)

    def run(label, selected, controls):
        report(f'Starting {label}')
        phase = time.perf_counter()
        energy, bar, components = ccsd.value_and_grad(mol, build_mf, selected,
                                                     triples_controls=controls, **options)
        jax.block_until_ready((energy, bar))
        force = -np.asarray(bar.coords)
        if not np.isfinite(force).all():
            raise RuntimeError('nonfinite force')
        energy_key = 'total_energy_eh' if args.sampling_fragment is None else 'correction_energy_eh'
        data = dict(force_eh_per_bohr=force.tolist(),
                    components=asdict(components), elapsed_seconds=time.perf_counter()-phase,
                    setup_timings=setup_timings, setup_seconds=phase-started,
                    runner_elapsed_seconds=time.perf_counter()-started,
                    basis=args.basis, frozen=args.frozen, threads=args.threads,
                    geometry=str(args.xyz_file.resolve()), hf_energy=float(mf.e_tot),
                    triples_controls=controls,
                    boys_grouping=args.boys_grouping, lno_thresh_occ=args.lno_thresh_occ,
                    lno_thresh_vir=args.lno_thresh_vir, force_full_domains=not args.finite_domains,
                    scf_density_ao=np.asarray(mf.make_rdm1()).tolist())
        data[energy_key] = float(energy)
        if args.sampling_fragment is not None:
            data['sampling_fragment'] = args.sampling_fragment
            data['energy_scope'] = 'one S: CCSD + (T) - MP2-LIS correction'
        save(label + '.json', data)
        report(f'{label}: {float(energy):.12f} Eh, {data["elapsed_seconds"]:.1f}s')
        del energy, bar, components
        gc.collect()
        return data

    if args.reference:
        reference = json.loads(args.reference.read_text())
        if (reference['basis'] != args.basis or reference['frozen'] != args.frozen
                or abs(reference['hf_energy'] - float(mf.e_tot)) > 1e-8
                or reference.get('boys_grouping', 'molecule') != args.boys_grouping
                or reference.get('lno_thresh_occ', 0.) != args.lno_thresh_occ
                or reference.get('lno_thresh_vir', 0.) != args.lno_thresh_vir
                or reference.get('force_full_domains', True) != (not args.finite_domains)):
            raise ValueError('reference molecular settings or SCF energy disagree')
    elif args.sampling_fragment is None:
        reference = run('canonical_deterministic', ordinary, dict(mode='deterministic'))
    else:
        reference = None  # The requested fragment computes its own reference below.
    reference_force = np.asarray(reference['force_eh_per_bohr']) if reference is not None else None
    def stochastic_controls(seed):
        controls = dict(mode='stochastic', nsamples=args.samples, global_seed=seed,
            energy_tolerance=args.triples_energy_tolerance_mha / 1000,
            pilot_samples=args.triples_pilot_samples,
            trace_exponent=args.triples_trace_exponent)
        if args.triples_budget_energy == 'ccsd':
            records = {row['fragment_index']: row for row in reference['components']['fragments']}
            controls['fragment_energies'] = [records[i]['e_ccsd'] for i in range(len(smooth.fragments))]
        return controls
    if args.sampling_fragment is not None:
        index = args.sampling_fragment
        if index >= len(smooth.fragments) or smooth.fragments[index] is None:
            raise ValueError('sampling fragment index is outside the selected physical S')
        # The CLI tolerance is the target for this one requested correction.
        # For an isolated comparison, pass its original globally allocated SE.
        options.update(fragment_indices=(index,), include_baseline=False, include_hf=False)
        fragment_reference = (reference if reference is not None
                              and reference.get('sampling_fragment') == index else None)
        if args.fragment_deterministic:
            fragment_reference = run(f'canonical_deterministic_fragment_{index}', ordinary,
                                     dict(mode='deterministic'))
        fragment_comparisons = []
        for seed in args.seeds:
            label = f'smooth_stochastic_fragment_{index}_seed_{seed}'
            result = run(label, smooth, stochastic_controls(seed))
            if fragment_reference is not None:
                delta = (np.asarray(result['force_eh_per_bohr'])
                         - np.asarray(fragment_reference['force_eh_per_bohr']))
                fragment_comparisons.append(dict(label=label,
                    rms_force_error_meh_per_bohr=float(np.sqrt(np.mean(delta**2))*1000),
                    max_abs_force_error_meh_per_bohr=float(np.max(np.abs(delta))*1000),
                    standard_error_meh=float(np.sqrt(sum(f['triples_standard_error']**2
                        for f in result['components']['fragments']))*1000),
                    samples=sum(f['triples_samples'] for f in result['components']['fragments']),
                    energy_difference_meh=(result['correction_energy_eh']
                                          - fragment_reference['correction_energy_eh'])*1000))
                save('fragment_comparison.json', fragment_comparisons)
                report(str(fragment_comparisons[-1]))
        return
    if args.fd_coordinate is not None:
        index = tuple(args.fd_coordinate)
        original = np.asarray(mol.atom_coords())
        energies = {'canonical': [], 'smooth': []}
        for sign in (-1, 1):
            coords = original.copy()
            coords[index] += sign * args.fd_step
            displaced = gto.Mole(atom=[(mol.atom_symbol(i), xyz) for i, xyz in enumerate(coords)],
                unit='Bohr', basis=basis, max_memory=800000, verbose=4)
            displaced.build(trace_exp=False, trace_ctr_coeff=False)
            current_mf = build_mf(displaced)
            for label, selected in [('canonical', ordinary), ('smooth', smooth)]:
                value, _ = ccsd.kernel(current_mf, selected,
                    triples_controls=dict(mode='deterministic'), **options)
                energies[label].append(float(value))
                report(f'FD {label} {sign}: {float(value):.12f}')
        check = dict(coordinate=index, step_bohr=args.fd_step, energies=energies,
            forces_eh_per_bohr={label: -(values[1]-values[0])/(2*args.fd_step)
                               for label, values in energies.items()})
        save('finite_difference.json', check)
        report(str(check))
    comparisons = []
    def compare(label, result):
        difference = np.asarray(result['force_eh_per_bohr']) - reference_force
        item = dict(label=label, rms_force_error_meh_per_bohr=float(np.sqrt(np.mean(difference**2))*1000),
            max_abs_force_error_meh_per_bohr=float(np.max(np.abs(difference))*1000),
            energy_difference_meh=(result['total_energy_eh']-reference['total_energy_eh'])*1000,
            standard_error_meh=np.sqrt(sum(f['triples_standard_error']**2
                for f in result['components']['fragments']))*1000,
            samples=sum(f['triples_samples'] for f in result['components']['fragments']))
        comparisons.append(item)
        save('comparison.json', comparisons)
        report(str(item))
    if args.smooth_deterministic:
        compare('smooth_deterministic', run('smooth_deterministic', smooth,
                dict(mode='deterministic')))
    for seed in args.seeds:
        label = f'smooth_stochastic_seed_{seed}'
        compare(label, run(label, smooth, stochastic_controls(seed)))


if __name__ == '__main__':
    from benchmark_affinity import ensure_worker_stack
    ensure_worker_stack()
    main()
