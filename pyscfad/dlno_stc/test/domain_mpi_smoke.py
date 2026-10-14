"""Focused domain-owner acceptance: mpiexec -n 2 python -m ...domain_mpi_smoke.

Modes isolate molecular parity, owner/VJP checks and synchronized failures.
The pytest wrapper runs each potentially failing collective in a subprocess.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path

import jax
import numpy as np
from mpi4py import MPI

from pyscfad import dlno_stc, scf
from pyscfad.dlno_stc import backend, driver
from pyscfad.dlno_stc.test import _boys_acceptance as accept


@contextmanager
def patched(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield original
    finally:
        setattr(obj, name, original)


def worker_mf(mol=None):
    # kernel installs canonical root arrays on this matching DF skeleton.
    mf = scf.RHF(accept.molecule() if mol is None else mol).density_fit()
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    return mf


def fixture(comm):
    mf = accept.build_mf(accept.molecule()) if comm.rank == 0 else worker_mf()
    static = accept.selection(mf) if comm.rank == 0 else None
    return mf, static


def owner_check(comm, *, unmerged=False):
    """Old replica driver fails: rank 0 owns every preparation."""
    rank, size = comm.rank, comm.size
    mf, static = fixture(comm)
    calls = []
    original = driver._prepare

    def prepare(state, selected, group, settings, with_grad):
        calls.append(tuple(group.fragment_indices))
        return original(state, selected, group, settings, with_grad)

    with patched(driver, '_prepare', prepare):
        result = dlno_stc.kernel(mf, static, controls=accept.controls() if rank == 0 else None,
                                 merge_identical=not unmerged, comm=comm)
    observed = comm.gather(calls, root=0)
    if rank == 0:
        expected = ([(i,) for i in range(8)] if unmerged
                    else [(0, 1, 4, 5), (2, 3, 6, 7)])
        assert Counter(g for rows in observed for g in rows) == Counter(expected), observed
        assert [len(rows) for rows in observed] == [
            sum(i % size == r for i in range(len(expected))) for r in range(size)], observed
        assert sorted(i for rows in observed for g in rows for i in g) == list(range(8))
        assert np.isfinite(result)
        return {'energy': float(result), 'owners': observed}
    assert result is None


def acceptance(comm, *, serial=False, sampled=False, adaptive=False):
    """Domain-only ownership/stream parity with a fixed SCF and local frame."""
    rank = comm.rank
    mol = accept.molecule()
    counts = dict(scf=0, worker=0, selection=0, common=0, finish=0)
    captured, preparations = [], []
    settings = dict(accept.controls(), global_seed=713, energy_tolerance=0.3)
    if sampled or adaptive:
        settings.update(mode='stochastic', system_workload_cutoff=2.,
                        auxiliary_group_size=2)
        if sampled:
            settings['production_samples'] = 97
        else:
            settings.update(pilot_samples=8, min_production_samples=8,
                            max_production_samples=128)
    chosen = None
    retained = None

    def build_mf(current, **initial):
        counts['worker' if initial.get('mo_coeff_init') is not None else 'scf'] += 1
        return accept.build_mf(current, **initial)

    def build_static(reference):
        nonlocal chosen
        counts['selection'] += 1
        chosen = accept.selection(reference)
        reference.mo_energy = None  # Must be a detached selection reference.
        return chosen

    original_prepare, original_solve = driver._prepare, backend._solve_local
    original_finish = driver._finish_response
    original_common = driver._finish

    def prepare(state, selected, group, controls, with_grad):
        nonlocal retained
        # The builder executes inside the SCF VJP; state holds its concrete output.
        if rank == 0:
            retained = state['mf']
        preparations.append(tuple(group.fragment_indices))
        return original_prepare(state, selected, group, controls, with_grad)

    def solve(inputs, metadata, controls, **kwargs):
        result = original_solve(inputs, metadata, controls, **kwargs)
        captured.append(dict(fragment_id=int(metadata['fragment_id']),
            targets=list(metadata['target_indices']), owner=rank,
            seed=int(controls['global_seed']), tolerance=controls.get('energy_tolerance'),
            energy=float(result['energy']),
            count=int(result.get('diagnostics', {}).get('actual_sample_count', 0)),
            bars={key: np.asarray(value).tolist() for key, value in
                  result.get('cotangents', {}).items()}))
        return result

    def finish(*args, **kwargs):
        counts['finish'] += 1
        return original_finish(*args, **kwargs)

    def common_finish(*args, **kwargs):
        counts['common'] += 1
        return original_common(*args, **kwargs)

    # Raw backend bars and fixed-count streams belong to the saved local
    # frame. Keep SCF on root here so its reduction roundoff cannot rotate a
    # degenerate local basis; dedicated SCF tests cover default MPI physics.
    active = None if serial else comm
    root_settings = settings if rank == 0 else None
    with patched(driver, '_prepare', prepare), patched(backend, '_solve_local', solve), \
            patched(driver, '_finish_response', finish), patched(driver, '_finish', common_finish):
        result = dlno_stc.value_and_grad(mol, build_mf, build_static=build_static,
            controls=root_settings, include_hf=True, comm=active, parallel_scf_jk=False)
    observations = comm.gather((counts, preparations, captured), root=0)
    payload = None
    if rank == 0:
        energy, bar = result
        assert retained.mo_energy is not None
        assert not any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(retained))
        assert counts == dict(scf=1, worker=0, selection=1, common=1, finish=1), counts
        for r, (worker_counts, groups, records) in enumerate(observations[1:], 1):
            assert worker_counts == dict(scf=0, worker=1, selection=0, common=0, finish=0), worker_counts
        records = [row for _, _, rows in observations for row in rows]
        assert len(records) == 2, records
        records.sort(key=lambda row: row['fragment_id'])
        for gid, record in enumerate(records):
            assert record['owner'] == gid % comm.size
            assert record['tolerance'] == settings['energy_tolerance'] / np.sqrt(2)
            expected_seed = backend._replica_seed(713, record['fragment_id'], 0)
            assert record['seed'] == expected_seed, record
            assert np.isfinite(record['energy'])
            assert set(record['bars']) == {'foo', 'fvv', 'B'}
            assert all(np.isfinite(value).all() for value in
                       map(np.asarray, record['bars'].values()))
            if sampled:
                assert record['count'] > 0
        payload = dict(energy=float(energy), gradient=np.asarray(bar.coords).tolist(),
                       groups=records, counts=[row[0] for row in observations])
    else:
        assert result is None
    if not (sampled or adaptive):
        correlation = dlno_stc.value_and_grad(
            mol, accept.build_mf, chosen, controls=root_settings, comm=active,
            parallel_scf_jk=False)
        if rank == 0:
            payload['correlation_gradient'] = np.asarray(correlation[1].coords).tolist()
            payload['supplied_energy'] = float(correlation[0])
        else:
            assert correlation is None
    # Fixed selections go through the public energy path on all ranks.
    value = dlno_stc.kernel(retained if rank == 0 else worker_mf(mol), chosen,
                            controls=root_settings, comm=active, parallel_scf_jk=False)
    if rank == 0:
        payload['correlation_energy'] = float(value)
        if not (sampled or adaptive):
            np.testing.assert_allclose(payload['energy'] - float(retained.e_tot),
                                       value, atol=2e-10, rtol=0)
            np.testing.assert_allclose(payload['supplied_energy'], value, atol=2e-10, rtol=0)
    else:
        assert value is None
    return payload


def numeric_leaves(tree):
    """Serialize bars by stable paths; None and float0 are zero contributions."""
    leaves = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: x is None)[0]:
        if leaf is None or getattr(leaf, 'dtype', None) == jax.dtypes.float0:
            continue
        array = np.asarray(leaf)
        leaves[jax.tree_util.keystr(path)] = array
    return leaves


def own_vjp(comm):
    """Compare the same saved group on each installed root/worker frame."""
    assert comm.size == 2
    mf, static = fixture(comm)
    saved = None
    original = driver._prepare

    def prepare(state, selected, group, settings, with_grad):
        nonlocal saved
        saved = (state['mf'], state['common'], selected)
        return original(state, selected, group, settings, with_grad)

    with patched(driver, '_prepare', prepare):
        dlno_stc.kernel(mf, static, controls=accept.controls() if comm.rank == 0 else None,
                        comm=comm)
    assert saved is not None, 'each rank must own a group'
    local_mf, common, selected = saved
    cderi = getattr(local_mf.with_df, '_cderi', None)
    df_states = comm.allgather((local_mf.with_df.auxmol is not None,
                               cderi is None, hasattr(cderi, 'dtype')))
    assert df_states == [(True, False, True), (True, True, False)], df_states
    group = dlno_stc.build_domain_groups(selected)[0]
    from pyscfad.dlno_stc.prepare import prepare_boys_group_inputs
    prepare_same = lambda m, c: prepare_boys_group_inputs(m, c, selected, group)
    packet, pullback = jax.vjp(prepare_same, local_mf, common)
    packets = comm.gather(backend.host_inputs(packet), root=0)
    if comm.rank == 0:
        for name in ('foo', 'fvv', 'B'):
            np.testing.assert_allclose(packets[1][name], packets[0][name], atol=1e-10, rtol=1e-10)
        fixed = accept.selected_result(packet, selected, group.representative_index,
                                       target_indices=group.target_indices)['cotangents']
    else:
        fixed = None
    fixed = comm.bcast(fixed, root=0)  # Compare inputs first, then each own VJP.
    bars = pullback({key: jax.numpy.asarray(fixed[key]) for key in packet})
    gathered = comm.gather(numeric_leaves(bars), root=0)
    if comm.rank == 0:
        for path in gathered[0].keys() | gathered[1].keys():
            left, right = gathered[0].get(path), gathered[1].get(path)
            if left is None:
                left = np.zeros_like(right)
            if right is None:
                right = np.zeros_like(left)
            np.testing.assert_allclose(left, right, atol=2e-9, rtol=2e-9, err_msg=path)


def worker_failure(comm):
    mf, static = fixture(comm)
    original = driver._prepare

    def prepare(*args, **kwargs):
        if comm.rank == 1:
            raise RuntimeError('deliberate worker group preparation failure')
        return original(*args, **kwargs)

    with patched(driver, '_prepare', prepare):
        try:
            dlno_stc.kernel(mf, static, controls=accept.controls() if comm.rank == 0 else None,
                            comm=comm)
        except RuntimeError as exc:
            assert 'worker group preparation failure' in str(exc)
        else:
            raise AssertionError('worker preparation failure was not synchronized')
    assert all(comm.allgather(True))


def signed_reference(comm):
    """A true common-orbital sign change must fail the shared-frame check."""
    from pyscfad.dlno_stc import _mpi
    mf, static = fixture(comm)
    original = _mpi._validate_shared_reference
    corrupted = False

    def validate(mf, common, *args, **kwargs):
        nonlocal corrupted
        if comm.rank == 1:
            coeff = np.array(common.occupied_coeff, copy=True)
            assert np.linalg.norm(coeff[:, 0]) > 1e-3
            coeff[:, 0] *= -1
            common = common._replace(occupied_coeff=jax.numpy.asarray(coeff))
            corrupted = True
        return original(mf, common, *args, **kwargs)

    with patched(_mpi, '_validate_shared_reference', validate):
        try:
            dlno_stc.kernel(mf, static, controls=accept.controls() if comm.rank == 0 else None,
                            comm=comm)
        except RuntimeError as exc:
            assert any(word in str(exc).lower() for word in ('reference', 'signed', 'occupied'))
        else:
            raise AssertionError('signed worker common reference was accepted')
    assert comm.allgather(corrupted)[1]


def zero_send_allocation_failure(comm):
    """Fail the worker's second zero send chunk before peers enter Reduce."""
    from pyscfad.dlno_stc import _mpi
    assert comm.size >= 2
    zeros = _mpi.numpy.zeros
    allocations = 0
    reductions = 0
    rejected = False

    class CountedComm:
        def __getattr__(self, name):
            return getattr(comm, name)

        def Reduce(self, *args, **kwargs):
            nonlocal reductions
            reductions += 1
            return comm.Reduce(*args, **kwargs)

    def fail_zero_chunk(shape, *args, **kwargs):
        nonlocal allocations
        if comm.rank == 1 and shape == 2:
            allocations += 1
            if allocations == 2:
                raise MemoryError('deliberate second zero-send chunk allocation failure')
        return zeros(shape, *args, **kwargs)

    tree = {'df': np.arange(5, dtype=np.float64) if comm.rank == 0 else None}
    with patched(_mpi, '_TREE_REDUCE_CHUNK_BYTES', 16), \
            patched(_mpi.numpy, 'zeros', fail_zero_chunk):
        try:
            _mpi.tree_sum_to_root(CountedComm(), tree)
        except RuntimeError as exc:
            assert 'second zero-send chunk allocation failure' in str(exc)
            assert 'chunk 2:4 staging' in str(exc)
            rejected = True
        else:
            raise AssertionError('worker zero-send allocation failure was not synchronized')
    states = comm.allgather((rejected, allocations, reductions))
    assert all(failed and count == 1 for failed, _, count in states), states
    assert states[1][1] == 2, states


def missing_auxmol(comm):
    """Remove a real installed auxiliary subtree before actual setup validation."""
    from pyscfad.dlno_stc import _mpi
    mf, static = fixture(comm)
    original = _mpi._validate_shared_reference
    corrupted = False
    reductions = 0

    def validate(mf, common, *args, **kwargs):
        nonlocal corrupted
        assert mf.with_df.auxmol is not None
        if comm.rank == 1:
            mf.with_df.auxmol = None
            corrupted = True
        return original(mf, common, *args, **kwargs)

    def forbidden_reduce(*args, **kwargs):
        nonlocal reductions
        reductions += 1
        raise AssertionError('auxiliary subtree validation must precede reduction')

    with patched(_mpi, '_validate_shared_reference', validate), \
            patched(_mpi, 'tree_sum_to_root', forbidden_reduce):
        try:
            dlno_stc.kernel(mf, static, controls=accept.controls() if comm.rank == 0 else None,
                            comm=comm)
        except RuntimeError as exc:
            assert 'auxmol subtree differs from root' in str(exc)
        else:
            raise AssertionError('missing worker auxiliary subtree was accepted')
    states = comm.allgather((corrupted, reductions))
    assert states[1][0] and all(count == 0 for _, count in states), states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('acceptance', 'owners', 'unmerged', 'own-vjp',
                                         'worker-failure', 'signed-reference', 'sampled', 'adaptive',
                                         'zero-send-allocation', 'missing-auxmol'),
                        default='acceptance')
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    if args.serial:
        assert comm.size == 1
    if args.mode in ('acceptance', 'sampled', 'adaptive'):
        values = acceptance(comm, serial=args.serial, sampled=args.mode == 'sampled',
                            adaptive=args.mode == 'adaptive')
    elif args.mode in ('owners', 'unmerged'):
        values = owner_check(comm, unmerged=args.mode == 'unmerged')
    elif args.mode == 'own-vjp':
        values = own_vjp(comm)
    elif args.mode == 'worker-failure':
        values = worker_failure(comm)
    elif args.mode == 'zero-send-allocation':
        values = zero_send_allocation_failure(comm)
    elif args.mode == 'missing-auxmol':
        values = missing_auxmol(comm)
    else:
        values = signed_reference(comm)
    if comm.rank == 0:
        if args.output:
            args.output.write_text(json.dumps(values, indent=2) + '\n')
        print('Domain-owner MPI smoke PASS: ' + args.mode, flush=True)


if __name__ == '__main__':
    main()
