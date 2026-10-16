"""STC uses retained virtual spaces without selecting an extra PAO gauge."""

import numpy as np

from pyscfad import dlno_stc
from pyscfad.dlno_stc import domain
from pyscfad.dlno_stc.test.test_boys_construction import reference
from pyscfad.dlno_stc.test.test_protocol import finite_metadata, packet
from pyscfad.dlno_stc.protocol import validate_inputs


def test_real_boys_selection_and_replay_need_no_pivoted_qr(reference, monkeypatch):
    # Pivoting here is an extra virtual-gauge selection; PAO rank selection
    # must still run and deliver an orthonormal retained ED virtual space.
    original = domain.scipy_linalg.qr

    def reject_pivoting(*args, **kwargs):
        assert not kwargs.get("pivoting", False), "virtual anchor QR was selected"
        return original(*args, **kwargs)

    monkeypatch.setattr(domain.scipy_linalg, "qr", reject_pivoting)
    static = dlno_stc.build_domain_selections(
        reference, frozen=1, target_coeff=reference.mo_coeff[:, 1:5],
        force_full_domains=True,
    )
    common = domain.rebuild_domain_data(reference, static)
    groups = dlno_stc.build_domain_groups(static)
    assert len(groups) == 1
    for index, fragment in enumerate(static.fragments):
        frame = domain.build_boys_ed_frame(common, static, index)
        overlap = np.asarray(common.s1e)[np.ix_(frame.extended_ao_indices,
                                              frame.extended_ao_indices)]
        virtual = np.asarray(frame.virtual_coeff)
        np.testing.assert_allclose(virtual.T @ overlap @ virtual, np.eye(2), atol=1e-9)
        assert not hasattr(fragment, "virtual_anchor_columns")
        assert not hasattr(frame, "virtual_anchor_columns")


def test_system_uses_active_virtual_columns_without_an_extra_rotation(reference):
    virtual = np.array([6, 5])
    frame = domain.build_system_local_frame(reference, np.arange(1, 5), virtual)
    np.testing.assert_array_equal(frame.virtual_coeff, reference.mo_coeff[:, virtual])


def test_finite_tensor_contract_does_not_require_virtual_gauge_metadata():
    metadata = finite_metadata()
    metadata.pop("virtual_anchor_parent_columns", None)
    assert validate_inputs(packet(), metadata) is None
