import ctypes
import os

import numpy
import pytest

from pyscfadlib import libao2mo_vjp


@pytest.fixture(scope='module')
def native_library():
    """Permit testing an explicitly staged build before installation."""
    staged = os.environ.get('TEST_AO2MO_VJP_LIBRARY')
    return ctypes.CDLL(staged) if staged else libao2mo_vjp


def test_cderi_bar_pack_aux_block_matches_dense_reference(native_library):
    kernel = native_library.AO2MOnr_e2_cderi_bar_pack_aux_block
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


def _native_coefficient_and_ao_bars(native, raw, coeff, ybar, slices, coefficient_only):
    nrow, _ = raw.shape
    nao, nmo = coeff.shape
    k0, k1, l0, l1 = slices
    small_first = k1-k0 <= l1-l0
    coeff = numpy.asarray(coeff, order='F' if small_first else 'C')
    coeff_bar = numpy.zeros_like(coeff, order='C' if small_first else 'F')
    raw_bar = None if coefficient_only else numpy.zeros_like(raw)
    fmmm = getattr(native, 'AO2MOmmm_nr_vjp_s2_iltj' if small_first
                   else 'AO2MOmmm_nr_vjp_s2_igtj')
    pointer = lambda array: array.ctypes.data_as(ctypes.c_void_p)
    sizes = [ctypes.c_int(nrow), ctypes.c_int(nao), ctypes.c_int(nmo),
             (ctypes.c_int*4)(*slices)]
    if coefficient_only:
        driver = native.AO2MOnr_e2_mo_coeff_vjp_drv
        driver.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_int)]
        arguments = [ctypes.cast(fmmm, ctypes.c_void_p), pointer(coeff_bar),
                     pointer(raw), pointer(coeff), pointer(ybar), *sizes]
    else:
        driver = native.AO2MOnr_e2_vjp_drv
        driver.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_int)]
        arguments = [ctypes.cast(native.AO2MOtranse2_nr_vjp_s2, ctypes.c_void_p),
                     ctypes.cast(fmmm, ctypes.c_void_p), pointer(raw_bar),
                     pointer(coeff_bar), pointer(raw), pointer(coeff), pointer(ybar), *sizes]
    driver.restype = None
    driver(*arguments)
    return coeff_bar, raw_bar


@pytest.mark.parametrize('threads', [1, 8])
@pytest.mark.parametrize('slices', [
    (0, 2, 3, 7),  # disjoint; i_count < j_count
    (0, 4, 5, 7),  # disjoint; i_count > j_count
    (1, 3, 2, 7),  # overlapping; i_count < j_count
    (1, 6, 3, 5),  # overlapping; i_count > j_count
    (1, 5, 1, 5),  # identical slices; both bars accumulate into the same columns
])
def test_coefficient_only_and_full_vjp_match_dense_reference(native_library, slices, threads):
    """Removing the unused AO bar must preserve both coefficient contributions.

    The ordinary full VJP is checked independently, including symmetric AO
    packing. This covers both native contraction orders and overlapping slices.
    """
    native = native_library
    get_threads = native.omp_get_max_threads
    get_threads.argtypes, get_threads.restype = [], ctypes.c_int
    set_threads = native.omp_set_num_threads
    set_threads.argtypes, set_threads.restype = [ctypes.c_int], None
    previous = get_threads()
    try:
        set_threads(threads)
        rng = numpy.random.default_rng(913)
        nrow, nao, nmo = 11, 7, 8
        rows, cols = numpy.tril_indices(nao)
        raw = rng.normal(size=(nrow, len(rows)))
        dense = numpy.empty((nrow, nao, nao))
        dense[:, rows, cols] = raw
        dense[:, cols, rows] = raw
        coeff = rng.normal(size=(nao, nmo))
        k0, k1, l0, l1 = slices
        ck, cl = coeff[:, k0:k1], coeff[:, l0:l1]
        ybar3 = rng.normal(size=(nrow, k1-k0, l1-l0))
        ybar = ybar3.reshape(nrow, -1)
        saved = [x.copy() for x in (raw, coeff, ybar)]
        expected_coeff = numpy.zeros_like(coeff)
        expected_coeff[:, k0:k1] += numpy.einsum('Puv,vj,Pij->ui', dense, cl, ybar3)
        expected_coeff[:, l0:l1] += numpy.einsum('Puv,ui,Pij->vj', dense, ck, ybar3)
        dense_bar = numpy.einsum('Pij,ui,vj->Puv', ybar3, ck, cl)
        expected_raw = dense_bar[:, rows, cols].copy()
        offdiag = rows != cols
        expected_raw[:, offdiag] += dense_bar[:, cols[offdiag], rows[offdiag]]

        coefficient_only, missing_raw = _native_coefficient_and_ao_bars(
            native, raw, coeff, ybar, slices, True)
        full_coefficient, full_raw = _native_coefficient_and_ao_bars(
            native, raw, coeff, ybar, slices, False)
        assert missing_raw is None
        numpy.testing.assert_allclose(coefficient_only, expected_coeff, atol=2e-12, rtol=2e-12)
        numpy.testing.assert_allclose(full_coefficient, expected_coeff, atol=2e-12, rtol=2e-12)
        numpy.testing.assert_allclose(full_raw, expected_raw, atol=2e-12, rtol=2e-12)
        for current, original in zip((raw, coeff, ybar), saved):
            numpy.testing.assert_array_equal(current, original)
    finally:
        set_threads(previous)
