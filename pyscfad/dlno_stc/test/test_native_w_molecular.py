"""The deterministic native migration preserves complete DLNO-CCSD(T) forces."""
import numpy as np
from pyscfad import config_update
from pyscfad.dlno_stc import backend, ccsd, _stc_mp2
from pyscfad.dlno_stc.test._cc_options import cc_options
from pyscfad.dlno_stc.test import _boys_acceptance as accept
from pyscfad.dlno_stc.test.test_shared_domain_frame import molecular_group


def test_molecular_w_backend_equivalence(molecular_group, monkeypatch, record_property):
    case = molecular_group
    import pyscfad.dlno_stc as package
    from pyscfad.dlno_stc import _native_triples
    select = backend._matrix_extension
    scalar = backend._solve_local
    monkeypatch.setenv('PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE', 'direct')
    options = dict(**cc_options(case.controls, case.controls), ccsd_t=True,
                   triples_controls=dict(mode='deterministic'))
    results = []
    for compiled in (False, True):
        monkeypatch.setattr(backend, '_matrix_extension',
                            select if compiled else lambda *args: _stc_mp2)
        monkeypatch.setattr(package, '_native_triples', _native_triples if compiled else _stc_mp2)
        def old_scalar(inputs, metadata, controls, *, with_grad, aux_offsets):
            return _stc_mp2.solve_selected(*(inputs[k] for k in ('foo','fvv','B')),
                tuple(metadata['target_indices']),controls,with_grad=with_grad,aux_offsets=aux_offsets)
        monkeypatch.setattr(backend, '_solve_local', scalar if compiled else old_scalar)
        with config_update('pyscfad_dfccsd_native', True), config_update('pyscfad_dfccsd_cpp', True):
            results.append(ccsd.value_and_grad(case.mf.mol, accept.build_mf, case.static, **options))
    energy_error = abs(float(results[1][0]-results[0][0]))
    gradient_error = float(np.max(np.abs(results[1][1].coords-results[0][1].coords)))
    record_property('energy_error_eh', energy_error)
    record_property('gradient_error_eh_per_bohr', gradient_error)
    assert energy_error < 1e-9
    assert gradient_error < 2e-7
