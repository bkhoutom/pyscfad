"""The optional SCF service verifies each source once before contracting it."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from pyscfad import df, gto
from pyscfad.df.addons import make_auxmol
from pyscfad.dlno_stc import _mpi_scf


class _WorkerComm:
    def Get_rank(self):
        return 1

    def Get_size(self):
        return 2

    def allgather(self, value):
        return [value, value]


@pytest.mark.parametrize('corrupt', [False, True])
def test_worker_hashes_source_once_before_first_contraction(tmp_path, monkeypatch, corrupt):
    mol = gto.Mole(atom='He 0 0 0', basis='sto-3g', verbose=0)
    mol.build(trace_exp=False, trace_ctr_coeff=False)
    obj = df.DF(mol, auxbasis='weigend', incore=False)
    obj.auxmol = make_auxmol(mol, obj.auxbasis)
    path = tmp_path / 'source.h5'
    with h5py.File(path, 'w') as handle:
        handle['j3c'] = np.ones((3, 1))
    obj.attach_outcore_cderi(str(path))
    scans = []
    fingerprint = _mpi_scf._source_fingerprint

    def record(dfobj):
        scans.append(dfobj)
        return fingerprint(dfobj)

    monkeypatch.setattr(_mpi_scf, '_source_fingerprint', record)
    service = _mpi_scf.DomainSCFService(
        _WorkerComm(), lambda: SimpleNamespace(mol=mol, with_df=obj))
    try:
        assert not scans
        assert not service.executor.source_verified
        if corrupt:
            with h5py.File(path, 'r+') as handle:
                handle['j3c'][1, 0] = np.nan
        payload = dict(dm=np.ones((1, 1)), hermi=1, with_j=True, with_k=True,
                       direct_scf_tol=1e-13, mo_coeff=None, mo_occ=None,
                       metadata=(1, 3, 1))
        if corrupt:
            with pytest.raises(RuntimeError, match='nonfinite factors'):
                service.executor._execute_forward(service.worker_df, payload)
            assert not service.executor.source_verified
            assert service.executor.operations['forward_calls'] == 0
        else:
            for _ in range(2):
                vj, vk = service.executor._execute_forward(service.worker_df, payload)
                np.testing.assert_array_equal(vj, [[2.]])
                np.testing.assert_array_equal(vk, [[2.]])
            assert service.executor.source_verified
        assert len(scans) == 1
    finally:
        service.close_local()
