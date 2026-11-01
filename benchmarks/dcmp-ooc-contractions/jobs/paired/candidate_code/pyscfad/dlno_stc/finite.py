"""Compatibility entry points for the common selected-domain exchange."""
from .mp2 import (export_run as export_finite_run,
                  load_static_from_run as load_finite_static_from_run,
                  load_disk_packet as load_finite_disk_packet)
from .mp2 import imported_value_and_grad


def imported_finite_value_and_grad(mol, build_mf, static, packet_loader, *,
                                    include_hf=False, **kwargs):
    """Replay selected finite MP2 packets using the common molecular path."""
    return imported_value_and_grad(mol, build_mf, static, packet_loader,
                                   include_hf=include_hf, **kwargs)
