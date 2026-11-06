"""Shared response budgets must preserve gradients and bounded spill reads."""
from types import SimpleNamespace
import weakref

import jax
import numpy as np
import pytest

from pyscfad import config_update, df, gto
from pyscfad.df import _response_memory as memory
from pyscfad.df import _df_jk_factorized as factorized
from pyscfad.df import _ooc_cotangent as coordinator


@pytest.fixture(autouse=True)
def clean_memory_policy(monkeypatch):
    for name in ('PYSCFAD_FACTOR_PANEL_MEMORY_MB', 'PYSCFAD_SOLVE_MEMORY_MB',
                 'PYSCFAD_DF_INT3C_MO_VJP_BLOCK_MB', 'PYSCFAD_DF_OOC_COMPACT'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(memory.lib, 'current_memory', lambda: (0., 0.))


def budget(workspace=500., total=900.):
    return memory.Workspace(SimpleNamespace(
        jk_workspace_memory=workspace, precond_total_memory=total,
        mol=SimpleNamespace(max_memory=total), max_memory=workspace))


def test_spilled_solve_uses_remainder_after_retained_panels(monkeypatch):
    ws = budget()
    with ws.lease(80_000_000), ws.lease(300_000_000, panels=True):
        width = ws.solve_width(1000, 100000)
        assert width * 3 * 8 * 1000 <= 120_000_000
        assert width > 64 * 1024**2 // (3 * 8 * 1000)
        monkeypatch.setenv('PYSCFAD_SOLVE_MEMORY_MB', '16')
        assert ws.solve_width(1000, 100000) < width
    assert ws.used == ws.panel_used == 0


def test_new_allocations_respect_total_headroom_and_untouched_reservations(monkeypatch):
    monkeypatch.setattr(memory.lib, 'current_memory', lambda: (700., 0.))
    ws = budget()
    assert ws.limit == 160_000_000
    with ws.lease(100_000_000):
        assert ws.available() == 60_000_000
        monkeypatch.setattr(memory.lib, 'current_memory', lambda: (880., 0.))
        assert ws.available() == 16_000_000
        with pytest.raises(MemoryError):
            ws.lease(17_000_000)
    assert ws.used == 0


def test_panel_cap_is_shared_and_does_not_cap_spill_read_workspace(monkeypatch):
    monkeypatch.setenv('PYSCFAD_FACTOR_PANEL_MEMORY_MB', '64')
    ws = budget()
    with ws.lease(60 * 1024**2, panels=True):
        assert ws.available(panels=True) == 4 * 1024**2
        assert ws.available() > 400_000_000
        assert ws.solve_width(1000, 100000) * 3 * 8 * 1000 > 400_000_000


@pytest.fixture
def source(tmp_path):
    mol = gto.Mole(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                   basis='sto-3g', verbose=0, max_memory=1000)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj._cderi_to_save = str(tmp_path / 'source.h5')
    obj.jk_workspace_memory = 20.
    with config_update('pyscfad_moleintor_opt', True):
        obj.build()
    return obj


def test_multiple_contributions_spill_under_shared_budget_without_changing_gradient(source):
    n, a, r = source.mol.nao, source.get_naoaux(), 60
    rng = np.random.default_rng(1903)
    factors = [rng.normal(size=(n, r)) for _ in range(2)]
    seeds = (np.eye(n), -.3 * np.eye(n))
    with config_update('pyscfad_moleintor_opt', True):
        references = [factorized._backward(True, True, False, (source, x), seeds)[0]
                      for x in factors]
        fixed = memory.prepare_fixed_bytes(n, a, r)
        minimum = max(memory.transform_row_bytes(n, r) * min(16, a), 3 * 8 * a)
        source.jk_workspace_memory = (fixed + minimum + 2 * 8 * a * n * r
                                      + 4 * 8 * a*a
                                      + memory.retained_fixed_bytes(n, a, r)) / 1e6
        with coordinator.geometry_scope(source) as scope:
            explicit = jax.tree_util.tree_map(np.zeros_like, source)
            for x in factors:
                bar, _ = factorized._backward(True, True, False, (source, x), seeds)
                explicit = jax.tree_util.tree_map(np.add, explicit, bar)
            actual = scope.finish(explicit)
        expected = jax.tree_util.tree_map(np.add, *references)
    for got, want in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(got, want, atol=3e-10, rtol=3e-10)
    assert scope.stats['disk_contributions'] >= 1
    assert scope.stats['workspace_peak_reserved_bytes'] <= scope.stats['workspace_limit_bytes']
    assert scope.workspace.used == scope.workspace.panel_used == 0


def test_preparation_releases_primal_panel_and_failure_releases_reservations(source, monkeypatch, tmp_path):
    monkeypatch.setattr(memory.lib.param, 'TMPDIR', str(tmp_path))
    n, a, r = source.mol.nao, source.get_naoaux(), 3
    factors = np.random.default_rng(118).normal(size=(n, r))
    seeds = (np.eye(n), -.2 * np.eye(n))
    references = []
    original_empty = factorized.numpy.empty

    def capture(shape, *args, **kwargs):
        out = original_empty(shape, *args, **kwargs)
        if isinstance(shape, tuple) and shape == (a, n * r):
            references.append(weakref.ref(out))
        return out

    monkeypatch.setattr(factorized.numpy, 'empty', capture)
    ws = memory.Workspace(source)
    with config_update('pyscfad_moleintor_opt', True):
        contribution, _ = factorized._prepare_backward(
            True, True, False, (source, factors), seeds, workspace=ws)
    assert references[0]() is None
    assert references[1]() is not None
    assert ws.panel_used == a * n * r * 8
    contribution.close()
    assert references[1]() is None
    assert ws.used == ws.panel_used == 0
    before = set(tmp_path.iterdir())

    def fail(*args, **kwargs):
        raise RuntimeError('injected preparation failure')

    monkeypatch.setattr(factorized, '_prepare_reverse', fail)
    with pytest.raises(RuntimeError, match='injected'):
        factorized._prepare_backward(True, True, False, (source, factors), seeds,
                                     workspace=ws, force_disk=True)
    assert ws.used == ws.panel_used == 0
    assert set(tmp_path.iterdir()) == before


def test_geometry_disk_reads_respect_remainder_below_legacy_minimum(source):
    n, a, r = source.mol.nao, source.get_naoaux(), 3
    factors = np.random.default_rng(719).normal(size=(n, r))
    seeds = (np.eye(n), -.2 * np.eye(n))
    ws = memory.Workspace(source)
    with config_update('pyscfad_moleintor_opt', True):
        contribution, _ = factorized._prepare_backward(
            True, True, False, (source, factors), seeds, workspace=ws, force_disk=True)
        try:
            expected = contribution.contract()
            rows = max(1, int(np.diff(source.auxmol.ao_loc).max()))
            row_bytes = 8 * (6 * n*n + 3 * n)
            observed = []
            original = contribution.fill_packed_density_aux_block

            def read(begin, end, out):
                observed.append(end-begin)
                return original(begin, end, out)

            contribution.fill_packed_density_aux_block = read
            # Simulate unrelated retained workspace leaving only one shell's
            # scratch, far below the helper's former 1 MiB minimum.
            extra = ws.available() - 4 * 8 * a*a - rows * row_bytes
            with ws.lease(extra):
                actual = contribution.contract()
            assert observed and max(observed) <= rows
            for got, want in zip(jax.tree_util.tree_leaves(actual),
                                 jax.tree_util.tree_leaves(expected)):
                np.testing.assert_allclose(got, want, atol=3e-10, rtol=3e-10)
        finally:
            contribution.close()
    assert ws.used == 0
