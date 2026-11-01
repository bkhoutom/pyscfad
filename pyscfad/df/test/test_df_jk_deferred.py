"""Deferred geometry must retain exact factor response and owned scratch."""

from contextlib import ExitStack

import jax
import numpy
import pytest
from pyscf import lib

from pyscfad import config_update, df, gto
from pyscfad import numpy as np
from pyscfad.df import _cderi_vjp, _df_jk_factorized as factorized, incore


@pytest.fixture
def problem(tmp_path, monkeypatch):
    monkeypatch.setattr(lib.param, 'TMPDIR', str(tmp_path))
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '32')
    monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '.001')
    mol = gto.Mole(atom='O .1 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=256)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'cderi.h5')
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    rng = numpy.random.default_rng(112026)
    terms = [(rng.normal(size=(mol.nao, rank)) * .2,
              tuple(rng.normal(size=(mol.nao, mol.nao)) * .2 for _ in range(2)))
             for rank in (1, 2, 3)]
    return obj, terms


def _dense_values(mol, auxmol, factors):
    """Reference has full differentiable integrals and no factorized VJP."""
    packed = incore.cholesky_eri(mol, auxmol=auxmol)
    nao = mol.nao
    if packed.shape[1] == nao ** 2:
        b = packed.reshape(-1, nao, nao)
    else:
        i, j = numpy.tril_indices(nao)
        b = np.zeros((packed.shape[0], nao, nao))
        b = b.at[:, i, j].set(packed).at[:, j, i].set(packed)
    d = factors @ factors.T
    return (np.einsum('Lij,ij,Lpq->pq', b, d, b),
            np.einsum('Lpi,ij,Ljq->pq', b, d, b))


@pytest.mark.parametrize(('with_j', 'with_k'), [(True, True), (True, False),
                                               (False, True)])
@pytest.mark.parametrize('force_disk', (False, True))
def test_combined_geometry_matches_independent_dense_pullbacks(
        problem, tmp_path, monkeypatch, with_j, with_k, force_disk):
    """Missing a seed, diagonal weight or metric term changes this gradient."""
    obj, terms = problem
    initial_files = set(tmp_path.iterdir())
    prepare = getattr(factorized, '_prepare_backward', None)
    assert callable(prepare), 'geometry preparation must be separable from contraction'
    calls = []
    original = _cderi_vjp._int3c_mo_deriv_coords_vjp_from_z_reader

    def observe(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_cderi_vjp, '_int3c_mo_deriv_coords_vjp_from_z_reader', observe)
    reference = []
    contributions = []
    with ExitStack() as cleanup:
        for (s, seeds), scale in zip(terms, (1., -.3, 0.)):
            seeds = tuple(scale * bar if active else numpy.zeros_like(bar)
                          for bar, active in zip(seeds, (with_j, with_k)))
            with config_update('pyscfad_moleintor_opt', True):
                _, pullback = jax.vjp(_dense_values, obj.mol, obj.auxmol, np.asarray(s))
                expected = pullback(seeds)
            contribution, sbar = prepare(
                with_j, with_k, True, (obj, s), seeds, force_disk=force_disk)
            cleanup.callback(contribution.close)
            contributions.append(contribution)
            reference.append(expected)
            assert not calls, 'preparation must not generate coordinate integrals'
            assert contribution.disk_panels == (force_disk and with_k)
            numpy.testing.assert_allclose(sbar, expected[2], atol=2e-10, rtol=2e-10)
        combined = factorized.contract_geometry(contributions)
        assert len(calls) == 1
        for actual, index in ((combined.mol.coords, 0), (combined.auxmol.coords, 1)):
            expected = sum(numpy.asarray(bar[index].coords) for bar in reference)
            numpy.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)
        # Contraction does not consume or corrupt the saved cotangent.
        again = contributions[0].contract()
        numpy.testing.assert_allclose(again.mol.coords, reference[0][0].coords,
                                      atol=2e-9, rtol=2e-10)
        numpy.testing.assert_allclose(again.auxmol.coords, reference[0][1].coords,
                                      atol=2e-9, rtol=2e-10)
    assert all(c.closed for c in contributions)
    assert set(tmp_path.iterdir()) <= initial_files
    for contribution in contributions:
        contribution.close()
        with pytest.raises(RuntimeError, match='closed'):
            contribution.contract()


@pytest.mark.parametrize(('with_j', 'with_k'), [(True, True), (True, False),
                                               (False, True), (False, False)])
def test_immediate_factor_response_needs_no_geometry_work(
        problem, monkeypatch, with_j, with_k):
    """Deferral must not block the factor cotangent needed by earlier AD nodes."""
    obj, terms = problem
    s, seeds = terms[1]
    seeds = tuple(bar if active else numpy.zeros_like(bar)
                  for bar, active in zip(seeds, (with_j, with_k)))
    routine = getattr(factorized, 'factor_pullback', None)
    assert callable(routine), 'immediate factor response must be independent of geometry'
    with config_update('pyscfad_moleintor_opt', True):
        _, pullback = jax.vjp(_dense_values, obj.mol, obj.auxmol, np.asarray(s))
        expected = pullback(seeds)[2]

    def forbidden(*args, **kwargs):
        raise AssertionError('factor-only response reached metric/geometry work')

    monkeypatch.setattr(obj.auxmol, 'intor', forbidden)
    monkeypatch.setattr(_cderi_vjp, '_int3c_mo_deriv_coords_vjp_from_z_reader', forbidden)
    actual = routine(obj, s, *seeds, with_j=with_j, with_k=with_k)
    numpy.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)


def test_failed_preparation_releases_owned_disk_panels(problem, tmp_path, monkeypatch):
    """Failure before a contribution is returned must not orphan its files."""
    obj, terms = problem
    initial_files = set(tmp_path.iterdir())
    prepare = getattr(factorized, '_prepare_backward', None)
    assert callable(prepare)

    def broken_loop(*args, **kwargs):
        raise RuntimeError('injected CDERI read failure')
        yield  # Keep the same generator protocol as DF.loop.

    monkeypatch.setattr(obj, 'loop', broken_loop)
    with pytest.raises(RuntimeError, match='injected CDERI read failure'):
        prepare(True, True, True, (obj, terms[0][0]), terms[0][1], force_disk=True)
    assert set(tmp_path.iterdir()) <= initial_files


def test_no_factor_response_and_contraction_failure_cleanup(problem, tmp_path, monkeypatch):
    """An exception in finalization must leave caller-owned cleanup usable."""
    obj, terms = problem
    initial_files = set(tmp_path.iterdir())
    prepare = getattr(factorized, '_prepare_backward', None)
    assert callable(prepare)
    contribution, sbar = prepare(
        True, True, False, (obj, terms[0][0]), terms[0][1], force_disk=True)
    assert sbar is None

    def broken_contract(*args, **kwargs):
        raise RuntimeError('injected coordinate failure')

    monkeypatch.setattr(_cderi_vjp, '_int3c_mo_deriv_coords_vjp_from_z_reader', broken_contract)
    try:
        with pytest.raises(RuntimeError, match='injected coordinate failure'):
            contribution.contract()
    finally:
        contribution.close()
    assert set(tmp_path.iterdir()) <= initial_files


@pytest.mark.parametrize('active_input', ('mol', 'auxmol', 'factors'))
@pytest.mark.parametrize('output', ('j', 'k', 'both'))
def test_direct_occupied_vjp_preserves_independent_geometry_activity(
        problem, monkeypatch, active_input, output):
    """Deferral must not inject an inactive geometry leaf into an outer scope."""
    from pyscfad.df import _ooc_cotangent

    obj, terms = problem
    s, seeds = terms[0]
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs['geometry_activity'])
        return None

    monkeypatch.setattr(_ooc_cotangent, 'defer_backward', observe)

    def select(values):
        return values if output == 'both' else values[0 if output == 'j' else 1]

    def call(value, dense):
        current, factors = jax.tree_util.tree_map(lambda x: x, obj), np.asarray(s)
        if active_input == 'factors':
            factors = value
        else:
            setattr(current, active_input, value)
        values = (_dense_values(current.mol, current.auxmol, factors) if dense
                  else factorized.get_jk_from_occ(current, factors))
        return select(values)

    primal = np.asarray(s) if active_input == 'factors' else getattr(obj, active_input)
    with config_update('pyscfad_moleintor_opt', True):
        _, reference = jax.vjp(lambda value: call(value, True), primal)
        expected = reference(select(seeds))[0]
    _, pullback = jax.vjp(lambda value: call(value, False), primal)
    actual = pullback(select(seeds))[0]
    if active_input == 'factors':
        numpy.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
        assert observed == []
    else:
        numpy.testing.assert_allclose(actual.coords, expected.coords,
                                      atol=2e-9, rtol=2e-10)
        assert observed == [(active_input == 'mol', active_input == 'auxmol')]
