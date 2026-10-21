"""Selected stochastic triples at the final LIS CCSD/DF boundary.

The fused native calculation returns the sampled energy and its unit-energy
input cotangents together. Reverse mode retains those cotangents; it never
replays the independently proposed energy-only calculation.
"""
import importlib

import jax
import numpy

from pyscfad import lib
from pyscfad import numpy as np


_INPUT_NAMES = ('t1', 't2', 'Bvo', 'Boo', 'Bvv', 'eocc', 'evir', 'U')


def _native_call(inputs, *, nsamples, seed, with_grad):
    native = importlib.import_module('pyscfad.dlno_stc._stc_mp2')
    function = (native.triples_selected_energy_vjp if with_grad
                else native.triples_selected_energy)
    result = function(*(numpy.asarray(x) for x in inputs),
                      nsamples=nsamples, seed=seed)
    diagnostics = {name: value for name, value in result.items()
                   if name not in ('energy', 'cotangents')}
    output = (np.asarray(result['energy']),
              jax.tree_util.tree_map(np.asarray, diagnostics))
    bars = (tuple(np.asarray(result['cotangents'][name]) for name in _INPUT_NAMES)
            if with_grad else None)
    return output, bars


def make_solver(controls, *, with_grad):
    """Build the optional LIS solver from normalized stochastic controls.

    ``global_seed`` is the fragment seed already derived by the CC driver.
    Diagnostics contain only numeric array leaves, so they can travel through
    the driver's ``jax.vjp(..., has_aux=True)`` path without native objects.
    """
    nsamples = controls['nsamples']
    seed = controls['global_seed']

    def native_call(*inputs):
        return _native_call(inputs, nsamples=nsamples, seed=seed,
                            with_grad=with_grad)

    if with_grad:
        @jax.custom_vjp
        def evaluate(*inputs):
            output, _ = native_call(*inputs)
            return output

        def forward(*inputs):
            return native_call(*inputs)

        def backward(unit_bars, output_bar):
            energy_bar, _ = output_bar
            return tuple(energy_bar * bar for bar in unit_bars)

        evaluate.defvjp(forward, backward)
    else:
        def evaluate(*inputs):
            output, _ = native_call(*inputs)
            return output

    def solver(mcc, eris, prjlo, *, t1, t2, verbose=0):
        del mcc, verbose
        nocc = t1.shape[0]
        return evaluate(t1, t2, eris.Lov, eris.Loo,
                        lib.unpack_tril(eris.Lvv),
                        eris.mo_energy[:nocc], eris.mo_energy[nocc:],
                        prjlo.T)

    return solver
