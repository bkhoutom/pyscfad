"""Domain memory forecasts, sampling budgets and checkpoint identity."""

import math

import numpy
from pyscf import df as pyscf_df
from pyscf import lib as pyscf_lib

from ._restart import df_source_fingerprint
from ._memory import assess_memory


def _native_workspace_mb(naux, nocc, nvir, *, with_grad,
                         mode="deterministic", target_batch_size=1, threads=None):
    """Account for bounded OpenMP scratch, without private full B tensors."""
    # Query the runtime so lib.num_threads() overrides are counted too. Using
    # its requested maximum conservatively covers a smaller OMP_THREAD_LIMIT.
    inherited_threads = max(1, int(pyscf_lib.num_threads()))
    if threads is not None:
        from .controls import _integer_control
        threads = _integer_control("threads", threads, minimum=0)
    # Preparation uses the inherited team; the native call may request more.
    threads = max(inherited_threads, threads or inherited_threads)
    owners = min(threads, nocc)
    # Exact pair GEMMs have a 64 MiB worker ceiling and a tiled fallback.
    # A single auxiliary column is the unavoidable lower bound on scratch.
    exact = owners * max(64 * 1024**2, 48 * naux)
    if with_grad:
        exact += 16 * owners * naux * nvir  # one-target U/C reductions
    transform = 8 * owners * naux * nvir
    if with_grad:
        reverse = 8 * min(threads, naux) * (
            2 * (nocc * nocc + nvir * nvir) + 3 * nocc * nvir)
        transform = max(transform, reverse, 32 * 1024**2)
    # At most 32 batches retain 131072 four-role records plus a pointer
    # partition (7 MiB together); reserve 8 MiB including small wave metadata.
    scatter = 8 * 1024**2 if with_grad else 0
    workspace = max(exact, transform) + scatter
    if mode == "deterministic":
        from .controls import _integer_control
        batch = nocc if target_batch_size is None else min(nocc, _integer_control("target_batch_size", target_batch_size))
        # Conservatively reserve the selected Gram cache and reverse scratch
        # beyond the DF/layout buffers counted by the caller. No all-target
        # cache is retained unless the caller explicitly requests that batch.
        workspace += 8 * nocc * nvir * nvir * ((2 if with_grad else 1) * batch
                                                + (4 if with_grad else 1))
    return workspace / 1e6


def run_payload(mf, frozen, *, driver, correlation_auxbasis=None):
    """Identify molecular preparation while static selections live in HDF5."""
    mol = mf.mol
    with_df = getattr(mf, "with_df", None)
    payload = {
        "driver": driver,
        "coords_bohr": numpy.asarray(mol.atom_coords()),
        "atom_symbols": tuple(mol.atom_symbol(i) for i in range(mol.natm)),
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "basis": getattr(mol, "_basis", None),
        "ecp": getattr(mol, "_ecp", None),
        "cart": bool(getattr(mol, "cart", False)),
        "frozen": frozen,
        "scf_class": f"{type(mf).__module__}.{type(mf).__qualname__}",
        "mo_occ": numpy.asarray(mf.mo_occ),
        "auxbasis": None if with_df is None else with_df.auxbasis,
        "df_source": df_source_fingerprint(mf),
    }
    if correlation_auxbasis is not None:
        payload["correlation_auxbasis"] = correlation_auxbasis
    return payload


def check_domain_memory(mf, static, fragment_index, *, with_grad,
                          virtual_block_size=16, mode="deterministic",
                          target_batch_size=1, threads=None, correlation_auxbasis=None):
    """Guard one finite domain using its local auxiliary basis and buffers.

    This estimates the local solver/RI working set, not a measured process
    peak. The shared SCF/localization tape remains resident across domains.
    """
    from pyscfad.lno import df as lno_df
    from pyscfad.lno._df_direct import _local_direct_workspace_mb

    fragment = static.fragments[fragment_index]
    nocc = len(fragment.strong_occ_metric_keep)
    nvir = len(fragment.strong_virtual.metric_keep)
    if nvir == 0:
        return 0.0
    local_mol = lno_df.make_local_mol(mf.mol, fragment.extended_atoms)
    auxbasis = mf.with_df.auxbasis if correlation_auxbasis is None else correlation_auxbasis
    auxmol = pyscf_df.addons.make_auxmol(local_mol.to_pyscf(), auxbasis)
    naux = int(auxmol.nao)
    # Preparation residual/input, host/layout buffers, dressed factors and
    # reverse bars. Target branches have only one occupied row.
    factor = 10 if with_grad else 5
    elements = factor * naux * nocc * nvir
    elements += (6 if with_grad else 3) * naux * nvir
    elements += 4 * naux * naux  # fitting metric and its reverse work
    elements += (8 if with_grad else 4) * (nocc * nocc + nvir * nvir)
    block = min(nvir, int(virtual_block_size))
    elements += 8 * block * block + 6 * naux * block
    estimate_mb = elements * 8 / 1e6
    estimate_mb += _local_direct_workspace_mb(
        mf.mol.max_memory, getattr(mf.with_df, 'max_memory', mf.mol.max_memory),
        reserve_bytes=elements * 8)
    estimate_mb += _native_workspace_mb(naux, nocc, nvir,
                                        with_grad=with_grad,
                                        mode=mode, target_batch_size=target_batch_size,
                                        threads=threads)
    limit_mb = float(getattr(mf, "max_memory", mf.mol.max_memory))
    # One local float64 B[P,i,a] is unavoidable; all copy/scratch factors above
    # remain forecasts and cannot justify a stop.
    assess_memory('domain', estimate_mb, configured_mb=limit_mb,
                  unavoidable_bytes=naux * nocc * nvir * numpy.dtype('float64').itemsize)
    return estimate_mb


def allocate_domain_controls(controls, job_count):
    """Copy controls and split total variance equally over strong jobs."""
    settings = controls.copy()
    if job_count and "energy_tolerance" in settings:
        settings["energy_tolerance"] = float(settings["energy_tolerance"]) / math.sqrt(job_count)
    return settings
