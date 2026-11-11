"""Native fragment CCSD retains the full DLNO geometry response."""
import numpy as np

from pyscfad import config_update
from pyscfad.dlno_stc import ccsd
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group


def test_native_dlno_ccsdt_energy_gradient(molecular_group, monkeypatch, record_property):
    case = molecular_group
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'direct')
    options = dict(baseline_controls=case.controls, matrix_controls=case.controls,
                   ccsd_t=True, fragment_indices=case.indices,
                   include_baseline=True, return_components=True,
                   triples_controls=dict(mode='deterministic', integral_transform='rotate'))
    results = []
    for enabled in (False, True):
        with config_update('pyscfad_dfccsd_native', enabled):
            results.append(ccsd.value_and_grad(case.mf.mol, accept.build_mf,
                                              case.static, **options))
    old, new = results
    energy_error = abs(float(new[0]-old[0]))
    gradient_error = float(np.max(np.abs(new[1].coords-old[1].coords)))
    record_property('energy_error_eh', energy_error)
    record_property('gradient_error_eh_per_bohr', gradient_error)
    np.testing.assert_allclose(new[0], old[0], atol=1e-9, rtol=0)
    np.testing.assert_allclose(new[1].coords, old[1].coords, atol=2e-7, rtol=0)
    for left, right in zip(new[2].fragments, old[2].fragments):
        np.testing.assert_allclose([left.e_ccsd, left.e_triples],
                                   [right.e_ccsd, right.e_triples], atol=1e-9, rtol=0)
