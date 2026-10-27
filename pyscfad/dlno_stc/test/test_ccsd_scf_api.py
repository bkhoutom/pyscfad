"""CC API contracts for optional shared MPI SCF and reference work."""
from types import SimpleNamespace

import numpy
import pytest

from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.test import _boys_acceptance as accept


@pytest.mark.parametrize('enabled', [True, False])
def test_energy_service_uses_supplied_worker_skeleton(monkeypatch, enabled):
    mf = SimpleNamespace()

    def run(initializer, worker_initializer, comm, *, parallel_scf_jk,
            scf_worker_initializer, **options):
        assert parallel_scf_jk is enabled
        return scf_worker_initializer()

    monkeypatch.setattr(ccsd, '_run', run)
    assert ccsd.kernel(mf, None, baseline_controls=None, matrix_controls=None,
                       parallel_scf_jk=enabled) is mf


def test_gradient_service_worker_adopts_closed_shell_placeholders_without_scf(monkeypatch):
    mol = accept.molecule()
    skeleton = object()
    supplied = {}

    def build_mf(current, *, mo_coeff_init, mo_energy_init, mo_occ_init, e_tot_init):
        assert current is mol
        supplied.update(coeff=mo_coeff_init, energy=mo_energy_init,
                        occ=mo_occ_init, total=e_tot_init)
        return skeleton

    def run(initializer, worker_initializer, comm, *, parallel_scf_jk,
            scf_worker_initializer, **options):
        assert parallel_scf_jk is True
        return scf_worker_initializer()

    monkeypatch.setattr(ccsd, '_run', run)
    assert ccsd.value_and_grad(mol, build_mf, baseline_controls=None,
                               matrix_controls=None) is skeleton
    numpy.testing.assert_array_equal(supplied['coeff'], numpy.eye(mol.nao))
    numpy.testing.assert_array_equal(supplied['energy'], numpy.zeros(mol.nao))
    expected_occ = numpy.zeros(mol.nao)
    expected_occ[:mol.nelectron // 2] = 2.
    numpy.testing.assert_array_equal(supplied['occ'], expected_occ)
    assert numpy.asarray(supplied['total']).shape == ()
    assert supplied['total'] == 0.


@pytest.mark.parametrize('comm', [None, SimpleNamespace(Get_size=lambda: 1)])
def test_serial_runner_does_not_receive_or_activate_scf_service_options(monkeypatch, comm):
    state = {'cc_fragment_indices': (1,), 'cc_triples_controls': None}

    def forbidden_worker():
        raise AssertionError('serial execution activated a worker initializer')

    def serial(state_arg, static, baseline, matrix, *, with_grad,
               fragment_indices, triples_controls, checkpoint_dir, resume):
        assert state_arg is state
        assert (static, baseline, matrix) == ('static', 'baseline', 'matrix')
        assert fragment_indices == (1,)
        assert triples_controls is None
        assert checkpoint_dir is None and resume is False
        assert with_grad is False
        return 4.25

    monkeypatch.setattr(ccsd, '_run_serial', serial)
    assert ccsd._run(lambda: (state, 'static', 'baseline', 'matrix'),
                     forbidden_worker, comm, checkpoint_dir=None, resume=False,
                     with_grad=False, parallel_scf_jk=True,
                     scf_worker_initializer=forbidden_worker) == 4.25
