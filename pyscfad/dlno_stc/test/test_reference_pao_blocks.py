"""Local PAO blocks agree with the independent expanded-AO reference."""
import numpy
import pytest
from pyscfad.lno import tools
from pyscfad.dlno_stc import _selection as selection
from pyscfad.dlno_stc._selection import (
    _pao_columns_on_atoms, _eigh_keep_numpy, _PAO_ORTH_THRESHOLD,
    _compute_completeness_numpy,
)
from pyscfad.dlno_stc.test.test_reference_pao_reuse import _inputs
from pyscfad.dlno_stc.test.test_boys_prepare import boys_water_dimer


def _expanded_reference(
    mol, pao_coeff, ao2pao_map, overlap, *, parent_atoms, support_atoms,
    representation_atoms, completeness_threshold, overlap_threshold,
):
    """Select and project the PAOs before removing any occupied target."""
    overlap = numpy.asarray(overlap)
    parent_columns = _pao_columns_on_atoms(
        mol, parent_atoms, ao2pao_map
    )
    pao_parent = numpy.asarray(pao_coeff)[:, parent_columns]

    gram = pao_parent.T.conj() @ overlap @ pao_parent
    gram_e, gram_v, canonical_keep = _eigh_keep_numpy(
        gram, _PAO_ORTH_THRESHOLD
    )
    if canonical_keep.size:
        pao_orth = pao_parent @ (
            gram_v[:, canonical_keep]
            / numpy.sqrt(numpy.real(gram_e[canonical_keep]))[None, :]
        )
    else:
        pao_orth = numpy.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    support_ao = tools.ao_index_by_atom(
        mol, numpy.asarray(support_atoms, dtype=numpy.int32)
    ).astype(numpy.int32, copy=False)
    if pao_orth.shape[1]:
        tmp = overlap[support_ao] @ pao_orth
        local_metric = overlap[numpy.ix_(support_ao, support_ao)]
        projected_overlap = tmp.T.conj() @ numpy.linalg.solve(local_metric, tmp)
        _, overlap_v, overlap_keep = _eigh_keep_numpy(
            projected_overlap, overlap_threshold
        )
        candidate = pao_orth @ overlap_v[:, overlap_keep]
    else:
        overlap_keep = numpy.zeros((0,), dtype=numpy.int32)
        candidate = numpy.zeros((mol.nao, 0), dtype=pao_parent.dtype)

    representation_ao = tools.ao_index_by_atom(
        mol, numpy.asarray(representation_atoms, dtype=numpy.int32)
    ).astype(numpy.int32, copy=False)
    completeness = _compute_completeness_numpy(
        candidate, overlap, representation_ao
    )
    completeness_keep = numpy.where(
        completeness > completeness_threshold
    )[0].astype(numpy.int32)
    candidate = candidate[:, completeness_keep]

    if candidate.shape[1]:
        candidate = tools.project_mo(
            candidate, overlap[representation_ao],
            overlap[numpy.ix_(representation_ao, representation_ao)],
        )
    else:
        candidate = numpy.zeros((representation_ao.size, 0), dtype=pao_parent.dtype)
    return (parent_columns, support_ao, representation_ao, canonical_keep,
            overlap_keep, completeness_keep, candidate)


@pytest.mark.parametrize('changed', [
    {}, {'parent_atoms': [1]}, {'support_atoms': [1]},
    {'representation_atoms': [0]}, {'support_atoms': [0, 1]},
    {'representation_atoms': [1, 0]}, {'completeness_threshold': 2.},
    {'overlap_threshold': 2.},
])
def test_local_blocks_match_expanded_ao_algebra_without_global_coefficients(changed):
    args, kwargs, _ = _inputs()
    kwargs.pop('metric_threshold')
    kwargs.update(changed)
    want = _expanded_reference(*args, **kwargs)
    products = selection._reference_pao_products(args[1], args[3])
    class ForbiddenGlobalPAOs:
        def __array__(self, *args, **kwargs):
            raise AssertionError('local preparation touched global PAO coefficients')
    local_args = (args[0], ForbiddenGlobalPAOs(), args[2], args[3])
    got = selection._reference_pao_preprojection(
        *local_args, **kwargs, pao_products=products)
    for x, y in zip(got[:-1], want[:-1]):
        numpy.testing.assert_array_equal(x, y)
    # LAPACK eigenvector phases may change with the equivalent contraction
    # order. Compare the same retained columns after their independent signs.
    if len(got[1]) < args[0].nao:
        signs = numpy.sign(numpy.sum(got[-1].conj() * want[-1], axis=0))
        numpy.testing.assert_allclose(got[-1] * signs, want[-1], rtol=2e-11, atol=2e-12)
    # Full support gives an identity projected metric: its degenerate
    # eigenvectors may rotate, so only the retained subspace is defined.
    numpy.testing.assert_allclose(got[-1] @ got[-1].T.conj(),
                                  want[-1] @ want[-1].T.conj(), rtol=2e-11, atol=2e-12)


def test_shared_products_are_not_recomputed_by_target_selection(monkeypatch):
    args, kwargs, targets = _inputs()
    products = selection._reference_pao_products(args[1], args[3])
    def forbidden(*args):
        raise AssertionError('global products repeated for an endpoint')
    monkeypatch.setattr(selection, '_reference_pao_products', forbidden)
    cache = {}
    for i in range(2):
        got = selection._reference_pao_overlap_selection(
            *args, **kwargs, occupied_for_projection=targets[:, i:i+1],
            preprojection_cache=cache, pao_products=products)
        numpy.testing.assert_allclose(
            targets[:, i:i+1].T @ args[3] @ got[1], 0., atol=1e-12)
    assert len(cache) == 1


def test_selection_builds_products_once_and_projects_targets_from_local_rhs(boys_water_dimer, monkeypatch):
    mf, _, reference = boys_water_dimer
    original = selection._reference_pao_products
    calls = []
    def products(*args):
        calls.append(1)
        return original(*args)
    def forbidden(*args):
        raise AssertionError('selection repeated a global target projection')
    monkeypatch.setattr(selection, '_reference_pao_products', products)
    monkeypatch.setattr(tools, 'project_mo', forbidden)
    selected = selection.build_domain_selections(
        mf, frozen=reference.frozen, target_coeff=reference.target_reference_coeff,
        force_full_domains=True, pair_energy_model='all-strong', boys_grouping='atom')
    assert len(selected.fragments) == len(reference.fragments)
    assert len(calls) == 1


def test_standalone_fallback_builds_only_requested_parent_products(monkeypatch):
    args, kwargs, _ = _inputs()
    kwargs.pop('metric_threshold')
    original = selection._reference_pao_products
    widths = []
    def products(pao, overlap):
        widths.append(pao.shape[1])
        return original(pao, overlap)
    monkeypatch.setattr(selection, '_reference_pao_products', products)
    selection._reference_pao_preprojection(*args, **kwargs)
    assert widths == [4]
