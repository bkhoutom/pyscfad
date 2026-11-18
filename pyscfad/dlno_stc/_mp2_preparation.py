"""Host-owned scalar MP2 factors with the existing analytic DF reverse.

Keep the large fitted factor out of JAX reshape residuals. Only the smaller
orbital/Fock projections and local molecule construction use saved JAX VJPs.
The factor's NumPy view, native input, and DF reverse share one allocation.
"""
import jax
import numpy
from pyscf import lib
from pyscfad import numpy as np
from pyscfad.lno import df as lno_df
from pyscfad.lno._df_direct import (_local_direct_nr_e2_impl,
                                    _local_direct_nr_e2_bwd)
from .prepare import _symmetric_fock_block
from ._profile import response


def project_inputs(mf, fock, co, cv, atoms, *, aux_atoms, with_grad,
                   correlation_auxbasis=None, group=None):
    """Return logical B[P,i,a] in native storage and an explicit input VJP."""
    nocc, nvir = co.shape[1], cv.shape[1]
    slices = (0, nocc, nocc, nocc + nvir)

    def small_inputs(current_mf, current_fock, occupied, virtual):
        mol = lno_df.make_local_mol(current_mf.mol, atoms)
        auxmol = lno_df.make_local_auxmol(current_mf, aux_atoms, correlation_auxbasis)
        coeff = np.concatenate((occupied, virtual), axis=1)
        foo, fvv = response('stc.domain.input_pullback.local_fock_blocks',
            lambda f, o, v: (_symmetric_fock_block(f, o), _symmetric_fock_block(f, v)),
            current_fock, occupied, virtual, group=group)
        return mol, auxmol, coeff, foo, fvv

    if with_grad:
        small, small_pullback = jax.vjp(small_inputs, mf, fock, co, cv)
    else:
        small = small_inputs(mf, fock, co, cv)
    mol, auxmol, coeff, foo, fvv = small
    max_memory = min(mf.mol.max_memory, lib.current_memory()[0] + mf.with_df.max_memory)
    if nocc and nvir:
        fitted = _local_direct_nr_e2_impl(mol, auxmol, coeff, max_memory, slices, host=True)
    else:
        # Empty orbital pairs have no DF dependence, even for singular metrics.
        fitted = numpy.empty((auxmol.nao, nocc * nvir), order='F')
    # This is a NumPy view: splitting the last dimension preserves its (ia)
    # order while P stays contiguous. No second full factor is retained.
    b = fitted.reshape(fitted.shape[0], nocc, nvir)
    inputs = dict(foo=foo, fvv=fvv, B=b)
    if not with_grad:
        return inputs, None

    def pullback(bars):
        if nocc and nvir:
            b_bar = numpy.asarray(bars['B']).reshape(fitted.shape)
            mol_bar, aux_bar, coeff_bar = _local_direct_nr_e2_bwd(
                max_memory, slices, (mol, auxmol, coeff, fitted), b_bar)
        else:
            mol_bar, aux_bar, coeff_bar = jax.tree_util.tree_map(np.zeros_like, (mol, auxmol, coeff))
        return small_pullback((mol_bar, aux_bar, coeff_bar, bars['foo'], bars['fvv']))

    return inputs, pullback
