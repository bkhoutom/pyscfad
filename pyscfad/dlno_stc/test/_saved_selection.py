"""Capture fixed labels from a normal production run for finite-difference oracles."""
from dataclasses import replace
from unittest.mock import patch
from pyscfad.dlno_stc import ccsd


def save_from_run(mf, plan, **options):
    selected = {}
    original = ccsd._select_current_fragment
    def capture(state, static, index, group, frame, forward, **options):
        result = original(state, static, index, group, frame, forward, **options)
        selected[index] = (result, float(forward['energy']))
        return result
    with patch.object(ccsd, '_select_current_fragment', capture):
        ccsd.kernel(mf, plan, **options)
    lis = replace(plan.lis_static,
        fragments=tuple(selected[i][0].fragments[i] for i in range(len(plan.fragments))),
        smooth_frames=tuple(selected[i][0].lis_static.smooth_frames[i] for i in range(len(plan.fragments))),
        triples_grid=tuple(selected[i][0].lis_static.triples_grid[i] for i in range(len(plan.fragments))))
    return replace(plan, lis_static=lis, selection_options=None,
                   mp2_fragment_energies=tuple(selected[i][1] for i in range(len(plan.fragments))))
