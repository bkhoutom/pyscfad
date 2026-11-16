"""Project full Fock blocks and fitted OV factors into saved STC frames."""

import numpy

from pyscfad import numpy as np
from pyscfad.lno import df as lno_df

from .domain import build_boys_ed_frame
from ._profile import response
from ._df_basis import basis_kwargs


def _symmetric_fock_block(fock, coeff):
    """Preserve the real symmetric Fock contract after projection roundoff."""
    block = coeff.T @ fock @ coeff
    # Keep this projection in the AD graph so both transpose paths respond.
    return 0.5 * (block + block.T)




def _project_inputs(mf, fock, occupied_coeff, virtual_coeff, atoms, *, aux_atoms, group=None,
                    correlation_auxbasis=None):
    """Use one occupied/virtual frame for both Fock blocks and DF factors."""
    co, cv = occupied_coeff, virtual_coeff
    nocc, nvir = co.shape[1], cv.shape[1]
    coeff = np.concatenate((co, cv), axis=1)
    B = lno_df.get_local_Lov(
        mf, coeff, nocc, atoms, integral_direct=True, aux_atmlst=aux_atoms,
        **basis_kwargs(correlation_auxbasis),
    )
    foo, fvv = response('stc.domain.input_pullback.local_fock_blocks',
                        lambda f, o, v: (_symmetric_fock_block(f, o),
                                         _symmetric_fock_block(f, v)),
                        fock, co, cv, group=group)
    return {
        "foo": foo,
        "fvv": fvv,
        "B": B.reshape((-1, nocc, nvir)),
    }


def prepare_inputs(mf, common, static, fragment_index, *, return_frame=False,
                   correlation_auxbasis=None):
    """Prepare exactly three tensors in one fixed, labelled finite Boys frame.

    ``common`` is rebuilt by the independent Boys ``rebuild_domain_data``.
    The saved selection owns the target's local slot, retained virtual ranks
    and concrete PAO anchors. That discrete metadata stays outside the
    differentiable tensor packet. Local AO and
    auxiliary supports remain finite through ``get_local_Lov``.

    With ``return_frame=True``, return ``((inputs, (co, cv)), frame)``.
    The coefficient pair is a differentiable output; the full frame is VJP
    auxiliary data carrying the discrete supports and labels. A caller can
    seed both outputs to join downstream frame response with DF/Fock response
    in one preparation pullback.
    """
    frame = build_boys_ed_frame(common, static, fragment_index)
    ao = numpy.asarray(frame.extended_ao_indices)
    inputs = _project_inputs(
        mf, common.fock[numpy.ix_(ao, ao)], frame.occupied_coeff,
        frame.virtual_coeff, frame.extended_atoms, group=fragment_index,
        aux_atoms=static.fragments[fragment_index].pao_center_atoms,
        **basis_kwargs(correlation_auxbasis),
    )
    if return_frame:
        return (inputs, (frame.occupied_coeff, frame.virtual_coeff)), frame
    return inputs


def prepare_boys_group_inputs(mf, common, static, group, *, return_frame=False,
                              correlation_auxbasis=None):
    """Prepare three tensors once in the group's saved representative frame.

    The full occupied partner space is retained. Group membership and target
    slots stay outside the differentiable tensor dictionary.
    """
    return prepare_inputs(mf, common, static, group.representative_index,
                          return_frame=return_frame, **basis_kwargs(correlation_auxbasis))
