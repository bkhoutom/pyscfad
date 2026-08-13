import ctypes

import numpy

from pyscfadlib import libao2mo_vjp


def test_cderi_bar_pack_aux_block_matches_dense_reference():
    kernel = libao2mo_vjp.AO2MOnr_e2_cderi_bar_pack_aux_block
    kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    kernel.restype = ctypes.c_int

    rng = numpy.random.default_rng(812)
    naux, nao, kc, lc = 3, 5, 2, 4
    ybar = rng.normal(size=(naux, kc, lc))
    mo_k = numpy.asarray(rng.normal(size=(nao, kc)), order='C')
    mo_l = numpy.asarray(rng.normal(size=(nao, lc)), order='C')
    y2 = numpy.asarray(
        ybar.transpose(0, 2, 1).reshape(naux * lc, kc), order='C'
    )
    npair = nao * (nao + 1) // 2
    result = numpy.empty((naux, npair), order='C')

    status = kernel(
        result.ctypes.data_as(ctypes.c_void_p),
        y2.ctypes.data_as(ctypes.c_void_p),
        mo_k.ctypes.data_as(ctypes.c_void_p),
        mo_l.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(naux),
        ctypes.c_int(nao),
        ctypes.c_int(kc),
        ctypes.c_int(lc),
    )
    assert status == 0

    dense = numpy.einsum(
        'Pij,ui,vj->Puv', ybar, mo_k, mo_l, optimize=True
    )
    rows, cols = numpy.tril_indices(nao)
    reference = dense[:, rows, cols].copy()
    offdiag = rows != cols
    reference[:, offdiag] += dense[:, cols[offdiag], rows[offdiag]]
    assert numpy.allclose(result, reference, atol=1e-12, rtol=1e-12)
