"""Export auxiliary metadata is shared while each packet keeps its guard."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyscfad.dlno_stc import _workflow, driver, mp2
from pyscfad.dlno_stc.test.test_domain_grouping import grouping_fixture
from pyscfad.dlno_stc.test.test_full_kernel import grid


@pytest.mark.parametrize('group_count', [0, 2])
@pytest.mark.parametrize('correlation_auxbasis', [None, 'correlation-fit'])
def test_export_builds_auxiliary_metadata_once(tmp_path, monkeypatch, group_count,
                                               correlation_auxbasis):
    static = grouping_fixture()
    fragments = static.fragments[:group_count]
    groups = [SimpleNamespace(representative_index=i, fragment_indices=(i,),
                              target_indices=(0,)) for i in range(group_count)]
    static = replace(static, fragments=fragments)
    mf = SimpleNamespace(mol=SimpleNamespace(to_pyscf=lambda: 'molecule', max_memory=10000.),
                         max_memory=10000., with_df=SimpleNamespace(auxbasis='scf-fit'))
    builds, estimates = [], []
    def make_auxmol(mol, basis):
        builds.append((mol, basis))
        return SimpleNamespace(nao=7)
    monkeypatch.setattr(_workflow.pyscf_df.addons, 'make_auxmol', make_auxmol)
    monkeypatch.setattr(_workflow, 'assess_memory', lambda kind, estimate, **kw: estimates.append(estimate))
    monkeypatch.setattr(driver, '_validate_static', lambda *a: None)
    monkeypatch.setattr(driver, '_validate_reference', lambda *a, **kw: None)
    monkeypatch.setattr(mp2, 'build_domain_groups', lambda *a, **kw: groups)
    monkeypatch.setattr(mp2, '_budget', lambda *a: {})
    monkeypatch.setattr(mp2, '_run_payload', lambda *a, **kw: {})
    monkeypatch.setattr(mp2, '_run_identity', lambda *a, **kw: {})
    monkeypatch.setattr(mp2, 'grouping_payload', lambda *a: {})
    monkeypatch.setattr(mp2, 'RestartManager', lambda *a, **kw: SimpleNamespace(
        save_static=lambda *a: None, save_record=lambda *a, **kw: None, close=lambda: None))
    monkeypatch.setattr(mp2, 'rebuild_domain_data', lambda *a: None)
    monkeypatch.setattr(mp2, 'prepare_boys_group_inputs', lambda *a, **kw: {'B': np.ones((7, 1, 1))})
    monkeypatch.setattr(mp2, '_packet_metadata', lambda *a, **kw: {})
    monkeypatch.setattr(mp2, 'write_input', lambda *a: None)
    rows = mp2.export_run(mf, static, tmp_path, controls=grid(),
                          correlation_auxbasis=correlation_auxbasis)
    assert len(rows) == group_count
    assert builds == ([('molecule', correlation_auxbasis or 'scf-fit')] if group_count else [])
    expected = [64 * 7 * len(f.strong_occ_metric_keep) *
                len(f.strong_virtual.metric_keep) / 1e6 for f in fragments]
    assert estimates == expected
