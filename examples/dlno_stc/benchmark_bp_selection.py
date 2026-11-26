"""Compare BP selectors on sampled physical domains from a saved C65 run.

Runs baseline and current selection serially on identical saved Boys orbitals.
The PAO supports required by the sampled occupied domains are selected too.
Optionally times a fresh global Boys localization once, separately.
"""
import argparse
import ast
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

from benchmark_affinity import configure_cpu_workload, check_worker_affinity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domains', type=int, default=10)
    parser.add_argument('--skip-boys', action='store_true')
    args = parser.parse_args()
    cpus = configure_cpu_workload(32)
    import numpy as np
    import scipy
    import jax
    from threadpoolctl import threadpool_info
    from pyscfad import gto
    from pyscfad.dlno_stc import _selection as selection, _weak_numpy
    from pyscfad.dlno_stc import _ccsd_restart_reference as restart
    from pyscfad.dlno_stc.domain import DomainData, TargetOccupiedData
    from pyscfad.dlno_stc._primary_response import prepare_primary
    from pyscfad.tools import resource_profile

    args.output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        path = args.output / name
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)

    def progress(phase, **details):
        value = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(),
                     pid=os.getpid(), **details)
        save('progress.json', value)
        print(json.dumps(value), flush=True)

    timings, events = [], []
    def rss_mib(field):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(field + ':'):
                return float(line.split()[1]) / 1024
        return None

    @contextmanager
    def measure(phase, **details):
        progress(phase, **details)
        try:
            Path('/proc/self/clear_refs').write_text('5\n')
            reset = True
        except OSError:
            reset = False
        start_rss = rss_mib('VmRSS')
        wall, cpu = time.perf_counter(), time.process_time()
        with resource_profile.section(phase, **details):
            yield
        row = dict(phase=phase, wall_seconds=time.perf_counter()-wall,
                   cpu_seconds=time.process_time()-cpu,
                   start_rss_mib=start_rss, peak_rss_mib=rss_mib('VmHWM'),
                   peak_reset=reset, **details)
        timings.append(row)
        save('timings.json', timings)
        save('profile_events.json', events)
        print(json.dumps(row), flush=True)

    with resource_profile.capture_events(events.append):
        with measure('input_loading'):
            mol = gto.Mole()
            mol.atom = str(args.reference / 'inputs/geometry.xyz')
            mol.basis, mol.charge, mol.spin, mol.verbose = 'cc-pvtz', 2, 0, 0
            mol.build(trace_exp=False, trace_ctr_coeff=False)
            checkpoint = args.reference / 'checkpoint'
            reference = restart.load_scf_reference(checkpoint, mol)
            static = restart._saved_static(checkpoint, restart._manifest(checkpoint)).mp2_static
            canonical = reference['canonical']
            targets = np.asarray(static.target_reference_coeff)
            overlap = np.asarray(mol.intor_symmetric('int1e_ovlp'))
            groups = static.physical_groups
            if not 1 <= args.domains <= len(groups):
                raise ValueError('invalid sample size')
            order = sorted(range(len(groups)), key=lambda i:
                len(static.fragments[groups[i][0]].primary_ao_indices))
            chosen = [order[i] for i in np.linspace(0, len(order)-1, args.domains, dtype=int)]
            members = sorted({int(i) for g in chosen for i in groups[g]})
            positions = {i: k for k, i in enumerate(members)}

        with measure('shared_pao_and_product_preparation'):
            coeff = np.asarray(canonical['mo_coeff'])
            out = coeff[:, static.pao_projected_out_indices]
            raw = np.eye(mol.nao) - out @ (out.T @ overlap)
            parents = static.pao_parent_ao_indices
            pao = raw[:, parents]
            pao /= np.sqrt(np.sum(pao * (overlap @ pao), axis=0))
            del raw
            common = DomainData(overlap, reference['fock'], coeff[:, static.active_occ_indices],
                coeff[:, static.active_vir_indices], canonical['mo_energy'][static.active_occ_indices],
                canonical['mo_energy'][static.active_vir_indices], targets, pao,
                tuple(TargetOccupiedData(targets[:, group]) for group in static.frag_lolist))
            products = _weak_numpy.products(common)

        baseline_source = args.baseline.read_text()
        tree = ast.parse(baseline_source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('get_bp_domain', '_compute_av_numpy')]
        assert len(functions) == 2
        namespace = vars(selection).copy()
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(args.baseline), 'exec'), namespace)
        baseline = namespace['get_bp_domain']
        metadata = dict(reference=str(args.reference), baseline=str(args.baseline),
            baseline_sha256=hashlib.sha256(baseline_source.encode()).hexdigest(),
            current_source=selection.__file__,
            current_sha256=hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest(),
            executable=sys.executable, versions=dict(numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__),
            cpus=cpus, affinity=check_worker_affinity(cpus), runtimes=threadpool_info(),
            environment={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
                'OMP_DYNAMIC', 'MKL_NUM_THREADS', 'MKL_DYNAMIC', 'OPENBLAS_NUM_THREADS')},
            nao=mol.nao, physical_groups=chosen, occupied_targets=members,
            thresholds=asdict(static.thresholds),
            checkpoint_static_digest=restart._manifest(checkpoint)['static_digest'],
            repetitions=1, order=['baseline', 'optimized'],
            scope='Three occupied BP thresholds and required PAO BP supports; primary spaces for all targets in ten sampled physical groups. No pair screening or correlation calculation. Shared preparation excluded. Saved Boys coefficients used for both variants.')
        save('manifest.json', metadata)
        results = {}
        t = static.thresholds
        for name, selector in [('baseline', baseline), ('optimized', selection.get_bp_domain)]:
            gc.collect()
            with measure('stc.domain.selection.atom_support', variant=name):
                occupied_supports = [selector(mol, targets[:, members], overlap,
                    bp_thr=cutoff, atom_order='population') for cutoff in (t.bp_occ, t.bp_primary, t.bp_ed)]
                primary_bp = occupied_supports[1]
                atoms = selection._union_index_lists(primary_bp)
                pao_columns = selection._pao_columns_on_atoms(mol, atoms, static.ao2pao_map)
                selected_pao_supports = selector(mol, pao[:, pao_columns], overlap, bp_thr=t.bp_pao)
                pao_supports = [np.empty(0, dtype=np.int32) for _ in range(pao.shape[1])]
                for column, support in zip(pao_columns, selected_pao_supports):
                    pao_supports[column] = support
                primary = selection.get_primary_domain(mol, primary_bp, pao_supports, static.ao2pao_map)
                merged = []
                for g in chosen:
                    slots = [positions[i] for i in groups[g]]
                    merged.append((selection._union_index_lists([primary_bp[k] for k in slots]),
                                   selection._union_index_lists([primary[k] for k in slots])))
            screens, ranks, per_group = [], [], []
            with measure('stc.domain.selection.primary_spaces', variant=name):
                for g, (bp_atoms, pd_atoms) in zip(chosen, merged):
                    progress('primary_group', variant=name, physical_group=g)
                    start = time.perf_counter()
                    cache = {}
                    for i in groups[g]:
                        ao, norm, span, selected, preparation = prepare_primary(
                            common, products, mol, static.ao2pao_map, t, i, bp_atoms, pd_atoms, cache)
                        ranks.append(dict(target=int(i), ao=ao.tolist(), norm=norm.tolist(), span=span.tolist(),
                            virtual=None if selected is None else {
                                field: np.asarray(getattr(selected, field)).tolist()
                                for field in selected.__dataclass_fields__}))
                        if preparation is not None:
                            screen = preparation[0]
                            screens.append((int(i), np.asarray(screen.occupied_energy), np.asarray(screen.virtual_energy)))
                        del preparation
                    per_group.append(dict(group=g, atoms=len(pd_atoms), ao=len(ao),
                                          targets=len(groups[g]), seconds=time.perf_counter()-start))
                    cache.clear()
            label_data = dict(occupied=[[x.tolist() for x in supports] for supports in occupied_supports],
                pao_columns=pao_columns.tolist(), pao=[x.tolist() for x in selected_pao_supports],
                merged=[(x.tolist(), y.tolist()) for x, y in merged], ranks=ranks)
            save(name+'_labels.json', label_data)
            save(name+'_domains.json', per_group)
            results[name] = (label_data, screens)
        assert results['baseline'][0] == results['optimized'][0], 'selection labels differ'
        assert len(results['baseline'][1]) == len(results['optimized'][1])
        max_error = 0.
        for old, new in zip(results['baseline'][1], results['optimized'][1]):
            assert old[0] == new[0]
            for a, b in zip(old[1:], new[1:]):
                np.testing.assert_allclose(a, b, atol=2e-9, rtol=2e-9)
                max_error = max(max_error, float(np.max(np.abs(a-b), initial=0.)))
        save('comparison.json', dict(labels_identical=True, max_orbital_energy_error=max_error,
             physical_domains=len(chosen), occupied_targets=len(members), pao_supports=len(pao_columns),
             timings=timings, note='One baseline followed by one optimized pass; shared input preparation excluded.'))
        if not args.skip_boys:
            gc.collect()
            with measure('stc.domain.selection.boys'):
                boys = selection._host_array(selection._build_boys_targets(
                    mol, coeff[:, static.active_occ_indices], static.lo_kwargs))
            save('boys.json', dict(seconds=timings[-1]['wall_seconds'], shape=list(boys.shape),
                orthogonality_error=float(np.max(np.abs(boys.T @ overlap @ boys - np.eye(boys.shape[1])))),
                scope='Fresh global localization of all active occupied orbitals, timed once; excluded from old/new support comparison.'))
        progress('complete')


if __name__ == '__main__':
    main()
