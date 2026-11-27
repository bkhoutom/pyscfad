"""One full-active-space MP2 baseline, independent of local CC domains.

Use the existing scalar MP2 kernels and the saved global Laplace grid.
Deterministic and stochastic evaluation use the same full active space.
Global DF factors are neither domain-truncated nor NAF-compressed. The
factor preparation and its analytic integral response are shared with domain
MP2, but no Boys target partition or pair mask enters this calculation.
"""
import jax
import numpy

from . import backend, driver
from ._mp2_preparation import project_inputs
from pyscfad.tools import resource_profile


def accumulate(state, static, controls, *, with_grad):
    domains = static.mp2_static
    mf, common = state['mf'], state['common']

    def orbital_inputs(current, data):
        return (data.fock, current.mo_coeff[:, domains.active_occ_indices],
                current.mo_coeff[:, domains.active_vir_indices])

    with resource_profile.section('stc.cc.global_mp2'):
        if with_grad:
            values, orbital_pullback = jax.vjp(orbital_inputs, mf, common)
        else:
            values = orbital_inputs(mf, common)
        atoms = numpy.arange(mf.mol.natm, dtype=numpy.int32)
        inputs, input_pullback = project_inputs(
            mf, *values, atoms, aux_atoms=atoms, with_grad=with_grad,
            correlation_auxbasis=state.get('correlation_auxbasis'))
        result = backend.solve(inputs,
            dict(fragment_id=0, target_indices=tuple(range(len(domains.active_occ_indices)))),
            controls, with_grad=with_grad)
        if with_grad:
            mf_bar, fock_bar, co_bar, cv_bar = input_pullback(result['cotangents'])
            orbital_mf_bar, common_bar = orbital_pullback((fock_bar, co_bar, cv_bar))
            mf_bar = jax.tree.map(lambda a, b: a + b, mf_bar, orbital_mf_bar)
            driver._add_bars(state, mf_bar, common_bar)
        state['energy'] = state['energy'] + result['energy']
        state['baseline_energy'] = state.get('baseline_energy', 0.) + float(result['energy'])
        state['variance'] += float(result['energy_standard_error']) ** 2
        jax.block_until_ready((state['energy'], state['mf_bar'], state['common_bar']))
