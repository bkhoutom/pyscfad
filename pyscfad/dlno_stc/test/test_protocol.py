"""Validation and replay identity for in-memory local tensors."""
import numpy as np
import pytest
from pyscfad.dlno_stc import protocol
from pyscfad.dlno_stc.test.test_full_kernel import full_fixture


@pytest.mark.parametrize('invalid', ['dtype', 'nan', 'shape', 'symmetry', 'extra'])
def test_local_input_contract_rejects_invalid_tensors(invalid):
    x = full_fixture()
    if invalid == 'dtype': x['B'] = x['B'].astype(np.float32)
    elif invalid == 'nan': x['B'][0, 0, 0] = np.nan
    elif invalid == 'shape': x['B'] = x['B'][:, :-1]
    elif invalid == 'symmetry': x['foo'][0, 1] += .1
    else: x['extra'] = np.zeros(())
    with pytest.raises(ValueError): protocol.validate_full_inputs(x)


@pytest.mark.parametrize('invalid', ['dtype', 'missing', 'extra', 'negative_se'])
def test_local_result_contract_rejects_invalid_cotangents(invalid):
    x = full_fixture()
    result = dict(energy=np.float64(-.1), energy_standard_error=np.float64(0.),
                  diagnostics={}, cotangents={k: np.zeros_like(v) for k, v in x.items()})
    protocol.validate_full_result(result, x)
    if invalid == 'dtype': result['cotangents']['B'] = result['cotangents']['B'].astype(np.float32)
    elif invalid == 'missing': del result['cotangents']['B']
    elif invalid == 'extra': result['cotangents']['extra'] = np.zeros(())
    else: result['energy_standard_error'] = np.float64(-.01)
    with pytest.raises(ValueError): protocol.validate_full_result(result, x)


def test_matrix_replay_identity_tracks_inputs_targets_and_controls():
    x = full_fixture()
    def identity(inputs=x, targets=(0,), seed=71):
        return protocol.matrix_request_fingerprint(inputs, {'target_indices': targets},
                                                  {'global_seed': seed}, [0, 4])
    original = identity()
    assert original == identity({k: v.copy() for k, v in x.items()})
    assert original != identity(targets=(1,))
    assert original != identity(seed=72)
    changed = {k: v.copy() for k, v in x.items()}
    changed['B'][0, 0, 0] += 1e-12
    assert original != identity(changed)
