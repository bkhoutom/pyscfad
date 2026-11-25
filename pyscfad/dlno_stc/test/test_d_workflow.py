"""D selection keeps the current shared energy/response and restart contracts."""
from dataclasses import replace

import numpy as np
import pytest

from pyscfad.dlno_stc import ccsd, _restart as storage
from pyscfad.dlno_stc.lis import LISSelections
from pyscfad.dlno_stc.test.test_full_kernel import reference
from pyscfad.dlno_stc.test.test_shared_ccsd_pullback import group_case


@pytest.mark.parametrize('kind', ['invalid', 'd', None])
def test_invalid_density_rejected_before_reference_use(kind):
    with pytest.raises(ValueError, match='matrix_kind'):
        ccsd.build_static_selections(None, None, thresh_occ=0., thresh_vir=0.,
                                     matrix_kind=kind)


def test_sequential_virtual_d_rejected_before_reference_use():
    with pytest.raises(NotImplementedError, match='W'):
        ccsd.build_static_selections(None, None, thresh_occ=0., thresh_vir=0.,
                                     matrix_kind='D', sequential_virtual=True)


def test_density_checkpoint_identity_and_legacy_w_default():
    from pyscfad.dlno_stc._ccsd_restart_reference import _selection_check
    w = LISSelections(None, 1e-4, 1e-5, 1e-6, ())
    old = storage._encode_value(w)
    assert 'matrix_kind' not in dict(old['fields'])
    restored = storage._decode_value(old, None)
    assert restored.matrix_kind == 'W'
    assert storage.scientific_digest(restored) == storage.scientific_digest(w)
    d = replace(w, matrix_kind='D')
    assert storage._decode_value(storage._encode_value(d), None).matrix_kind == 'D'
    assert storage.scientific_digest(d) != storage.scientific_digest(w)
    with pytest.raises(storage.RestartMismatchError, match='matrix_kind'):
        _selection_check(d, w)


@pytest.mark.parametrize('mode', ['deterministic', 'stochastic'])
def test_d_explicit_energy_and_density_share_preparation_response(group_case, mode):
    case = group_case
    matrix = dict(case.controls, mode=mode)
    if mode == 'stochastic':
        matrix.update(domain_tau=1e-100, virtual_domain_tau=1e-100,
                      system_workload_cutoff=1e-100, energy_tolerance=.03,
                      pair_pilot_samples=16, min_samples=16, matrix_tolerance=.03)
    options = dict(matrix_kind='D', matrix_controls=matrix)
    state, rows = case.run(**options)
    expected = reference(case.prepare(.2, -.1), case.controls)
    np.testing.assert_allclose(state['baseline_energy'], expected, atol=2e-12)
    assert all(row.matrix_diagnostics['matrix_kind'] == 'D' for row in rows)
    assert case.replay_energy_bars == [1., 1.]
    assert len(case.preparations) == len(case.preparations[0]) == 1
    step = 1e-5
    for name, dm, dc, initial in [('mf_bar', step, 0., .17),
                                  ('common_bar', 0., step, -.23)]:
        plus, _ = case.run(**options, with_grad=False, mf=.2+dm, common=-.1+dc)
        minus, _ = case.run(**options, with_grad=False, mf=.2-dm, common=-.1-dc)
        np.testing.assert_allclose(state[name]-initial,
            (plus['energy']-minus['energy'])/(2*step), atol=2e-9, rtol=2e-7)
