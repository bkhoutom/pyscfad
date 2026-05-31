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

@partial(ops.jit, static_argnums=1)
def lowdin(s, thresh=1e-15):
    e, v = scipy.linalg.eigh(s)
    e_sqrt = np.where(e>thresh, np.sqrt(e), np.inf)
    return np.dot(v/e_sqrt[None,:], v.conj().T)


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

