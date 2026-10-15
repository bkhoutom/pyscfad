"""A single STC replica must not broadcast inputs or reduce cotangents."""

from functools import partial

import numpy as np
import pytest

from pyscfad.dlno_stc import backend
from pyscfad.dlno_stc.parallel import run_backend
from pyscfad.dlno_stc.test.test_backend import fixture, native


class SingleRankComm:
    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def __getattr__(self, name):
        raise AssertionError(f"single-rank solve attempted MPI operation {name}")


@pytest.mark.parametrize("entry", ["run_backend", "solve"])
@pytest.mark.parametrize("scope", ["domain", "system"])
@pytest.mark.parametrize("with_grad", [False, True])
def test_single_rank_matches_serial_without_collectives(entry, scope, with_grad):
    native()
    inputs = fixture()
    metadata = {"fragment_id": 4, "target_indices": (0,)}
    controls = {
        "mode": "stochastic", "global_seed": 17,
        "laplace_roots": [0., .3], "laplace_weights": [.4, .6],
        "production_samples": 32, "pilot_samples": 16,
        "min_production_samples": 2, "system_workload_cutoff": 2.,
        "virtual_block_size": 2, "auxiliary_group_size": 2,
    }
    solve = partial(backend.solve, scope=scope, with_grad=with_grad)
    expected = solve(inputs, metadata, controls)
    if entry == "run_backend":
        result = run_backend(inputs, metadata, controls, solve, comm=SingleRankComm())
    else:
        result = solve(inputs, metadata, controls, comm=SingleRankComm())

    assert result["energy"] == expected["energy"]
    assert result["energy_standard_error"] == expected["energy_standard_error"]
    assert result["metadata"] == expected["metadata"]
    assert result["metadata"]["replica_count"] == 1
    assert result["diagnostics"] == expected["diagnostics"]
    if with_grad:
        for key in inputs:
            np.testing.assert_array_equal(result["cotangents"][key],
                                          expected["cotangents"][key])
    else:
        assert "cotangents" not in result


@pytest.mark.parametrize("entry", ["run_backend", "solve"])
def test_single_rank_still_rejects_nonfinite_inputs(entry):
    inputs = fixture()
    inputs["B"][0, 0, 0] = np.nan
    solve = partial(backend.solve, scope="system")
    with pytest.raises(ValueError, match="finite"):
        if entry == "run_backend":
            run_backend(inputs, {}, {}, solve, comm=SingleRankComm())
        else:
            solve(inputs, {}, {}, comm=SingleRankComm())
