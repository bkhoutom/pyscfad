"""MPI fragment distribution must not implicitly enable distributed SCF."""
import inspect

import pytest

from pyscfad import dlno_stc
from pyscfad.dlno_stc import _mpi, _mpi_ccsd, ccsd, driver


@pytest.mark.parametrize('entry_point', [
    dlno_stc.kernel,
    dlno_stc.value_and_grad,
    driver.kernel,
    driver.value_and_grad,
    ccsd.kernel,
    ccsd.value_and_grad,
    ccsd._run,
    _mpi._run_domain_mpi,
    _mpi_ccsd._run_mpi,
], ids=lambda entry_point: f'{entry_point.__module__}.{entry_point.__name__}')
def test_parallel_scf_requires_explicit_opt_in(entry_point):
    assert inspect.signature(entry_point).parameters['parallel_scf_jk'].default is False
