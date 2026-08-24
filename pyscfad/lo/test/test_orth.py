# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np

from pyscfad.lo import orth


def test_lowdin_near_degenerate_jvp_matches_five_point_fd():
    """The inverse-square-root response is finite inside a degenerate block."""
    rng = np.random.default_rng(4)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    eigenvalues = np.array([0.4, 1.0, 1.0 + 2.0e-12, 2.3])
    metric = rotation @ np.diag(eigenvalues) @ rotation.T
    dmetric = rng.normal(size=(4, 4))
    dmetric = 0.5 * (dmetric + dmetric.T)

    value, tangent = jax.jvp(
        orth.lowdin,
        (jnp.asarray(metric),),
        (jnp.asarray(dmetric),),
    )

    step = 1.0e-4
    plus_two = orth._lowdin_numpy(metric + 2.0 * step * dmetric)
    plus_one = orth._lowdin_numpy(metric + step * dmetric)
    minus_one = orth._lowdin_numpy(metric - step * dmetric)
    minus_two = orth._lowdin_numpy(metric - 2.0 * step * dmetric)
    finite_difference = (
        -plus_two + 8.0 * plus_one - 8.0 * minus_one + minus_two
    ) / (12.0 * step)

    np.testing.assert_allclose(value, orth._lowdin_numpy(metric), atol=3e-14)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-10, atol=2e-11)
