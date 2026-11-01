# Copyright 2021-2025 Xing Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial, reduce
import numpy as _onp
import scipy.linalg as _onp_sla
import jax
from pyscfad import numpy as np
from pyscfad import ops
from pyscfad import scipy


def _lowdin_impl(s, thresh):
    e, v = scipy.linalg.eigh(s)
    active = e > thresh
    safe_e = np.where(active, e, 1.0)
    inv_sqrt = np.where(active, 1.0 / np.sqrt(safe_e), 0.0)
    return np.dot(v * inv_sqrt[None, :], v.conj().T)


@partial(ops.custom_jvp, nondiff_argnums=(1,))
def _lowdin_ad(s, thresh):
    """Hermitian inverse square root with its spectral Frechet derivative."""
    return _lowdin_impl(s, thresh)


@_lowdin_ad.defjvp
def _lowdin_ad_jvp(thresh, primals, tangents):
    s, = primals
    ds, = tangents

    e, v = scipy.linalg.eigh(s)
    active = e > thresh
    safe_e = np.where(active, e, 1.0)
    sqrt_e = np.sqrt(safe_e)
    inv_sqrt = np.where(active, 1.0 / sqrt_e, 0.0)
    primal = np.dot(v * inv_sqrt[None, :], v.conj().T)

    # For f(x) = x**(-1/2), the stable divided difference is
    #
    #   (f(x) - f(y)) / (x - y)
    #       = -1 / (sqrt(x) sqrt(y) (sqrt(x) + sqrt(y))).
    #
    # Unlike an eigenvector-response formula, this remains finite and exact
    # inside a degenerate retained eigenspace.  That response is essential:
    # dropping rotations inside the block does not give the Frechet
    # derivative of the inverse square root.
    active_i = active[:, None]
    active_j = active[None, :]
    both_active = active_i & active_j
    divided_active = -1.0 / (
        sqrt_e[:, None]
        * sqrt_e[None, :]
        * (sqrt_e[:, None] + sqrt_e[None, :])
    )

    # Preserve the existing hard spectral cutoff.  Within either discarded
    # block the derivative is zero.  Across a fixed retained/discarded
    # boundary use the ordinary divided difference; the map is necessarily
    # nonsmooth only when an eigenvalue itself crosses ``thresh``.
    delta_e = e[:, None] - e[None, :]
    crosses_cutoff = active_i != active_j
    safe_delta_e = np.where(crosses_cutoff, delta_e, 1.0)
    divided_cross = (
        inv_sqrt[:, None] - inv_sqrt[None, :]
    ) / safe_delta_e
    divided = np.where(
        both_active,
        divided_active,
        np.where(crosses_cutoff, divided_cross, 0.0),
    )

    ds = 0.5 * (ds + ds.conj().T)
    ds_eigen = np.dot(v.conj().T, np.dot(ds, v))
    tangent = np.dot(v, np.dot(divided * ds_eigen, v.conj().T))
    return primal, tangent


@partial(ops.jit, static_argnums=1)
def lowdin(s, thresh=1e-15):
    return _lowdin_ad(s, thresh)


def _lowdin_numpy(s, thresh=1e-15):
    s_np = _onp.asarray(s)
    e, v = _onp_sla.eigh(s_np)
    e_sqrt = _onp.where(e > thresh, _onp.sqrt(e), _onp.inf)
    return _onp.dot(v / e_sqrt[None, :], v.conj().T)


def vec_lowdin(c, s=1):
    """Lowdin-orthonormalize ``c`` with overlap ``s``.

    Dispatches to a pure numpy path on concrete inputs to avoid XLA
    compile-cache growth from per-shape variation; otherwise uses the JAX
    path so the gradient propagates inside ``jax.value_and_grad``.
    """
    if isinstance(c, jax.core.Tracer) or (
        not isinstance(s, (int, float, complex))
        and isinstance(s, jax.core.Tracer)
    ):
        return np.dot(c, lowdin(reduce(np.dot, (c.conj().T, s, c))))
    c_np = _onp.asarray(c)
    if isinstance(s, (int, float, complex)):
        m = c_np.conj().T @ (s * c_np)
    else:
        s_np = _onp.asarray(s)
        m = c_np.conj().T @ s_np @ c_np
    return c_np @ _lowdin_numpy(m)
