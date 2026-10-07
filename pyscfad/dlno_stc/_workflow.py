"""Host-only helpers shared by full and finite STC packet workflows."""

from pathlib import Path
import math
import subprocess

import numpy
from pyscf import df as pyscf_df
from pyscf import lib as pyscf_lib

from pyscfad.dlno._restart import df_source_fingerprint


def _native_workspace_mb(naux, nocc, nvir, *, with_grad, system):
    """Account for bounded OpenMP scratch, without private full B tensors."""
    # Query the runtime so lib.num_threads() overrides are counted too. Using
    # its requested maximum conservatively covers a smaller OMP_THREAD_LIMIT.
    threads = max(1, int(pyscf_lib.num_threads()))
    owners = min(threads, nocc)
    # Exact pair GEMMs have a 64 MiB worker ceiling and a tiled fallback.
    # A single auxiliary column is the unavoidable lower bound on scratch.
    exact = owners * max(64 * 1024**2, 48 * naux)
    if with_grad and not system:
        exact += 16 * owners * naux * nvir  # one-target U/C reductions
    transform = 8 * owners * naux * nvir
    if with_grad:
        reverse = 8 * min(threads, naux) * (
            2 * (nocc * nocc + nvir * nvir) + 3 * nocc * nvir)
        transform = max(transform, reverse, 32 * 1024**2)
    # At most 32 batches retain 131072 four-role records plus a pointer
    # partition (7 MiB together); reserve 8 MiB including small wave metadata.
    scatter = 8 * 1024**2 if with_grad else 0
    return (max(exact, transform) + scatter) / 1e6


def format_domain_dimensions(rows):
    """Return a compact per-fragment ED size table for progress output."""
    columns = ("Fragment", "ED AO", "Occ", "Vir")
    values = [
        (str(row["fragment_id"]), str(row["ed_ao"]),
         str(row["nocc"]), str(row["nvir"]))
        for row in rows
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in values))
        for index in range(4)
    ]

    def line(row):
        return "  ".join(value.rjust(width) for value, width in zip(row, widths))

    header = line(columns)
    return "\n".join((header, "-" * len(header), *(line(row) for row in values)))


def run_payload(mf, frozen, *, driver):
    """Identify molecular preparation while static selections live in HDF5."""
    mol = mf.mol
    with_df = getattr(mf, "with_df", None)
    return {
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


def revision_status():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def check_memory(mf, nocc, nvir):
    """Reject a clearly oversized in-core DF packet before its allocation.

    The factor eight allows for input, imported bar, replay, and AD temporaries;
    it is a conservative prototype estimate, not a measured peak guarantee.
    """
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise ValueError("STC packet preparation requires density fitting")
    auxmol = getattr(with_df, "auxmol", None)
    if auxmol is None:
        auxmol = pyscf_df.addons.make_auxmol(mf.mol.to_pyscf(), with_df.auxbasis)
    estimate_mb = 8 * int(auxmol.nao) * int(nocc) * int(nvir) * 8 / 1e6
    limit_mb = float(getattr(mf, "max_memory", mf.mol.max_memory))
    if estimate_mb > limit_mb:
        raise MemoryError(
            f"estimated STC packet/replay memory {estimate_mb:.1f} MB "
            f"exceeds configured {limit_mb:.1f} MB"
        )
    return estimate_mb


def check_weighted_memory(mf, static, fragment_index, *, with_grad,
                          virtual_block_size=16):
    """Guard one finite domain using its local auxiliary basis and buffers.

    This estimates the local solver/RI working set, not a measured process
    peak. The shared SCF/localization tape remains resident across domains.
    """
    from pyscfad.lno import df as lno_df
    from pyscfad.lno._df_direct import _local_direct_int3c_block_mb

    fragment = static.fragments[fragment_index]
    nocc = len(fragment.strong_occ_metric_keep)
    nvir = len(fragment.strong_virtual.metric_keep)
    if nvir == 0:
        return 0.0
    local_mol = lno_df.make_local_mol(mf.mol, fragment.extended_atoms)
    auxmol = pyscf_df.addons.make_auxmol(local_mol.to_pyscf(), mf.with_df.auxbasis)
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
    estimate_mb = elements * 8 / 1e6 + _local_direct_int3c_block_mb()
    estimate_mb += _native_workspace_mb(naux, nocc, nvir,
                                        with_grad=with_grad, system=False)
    limit_mb = float(getattr(mf, "max_memory", mf.mol.max_memory))
    if estimate_mb > limit_mb:
        raise MemoryError(
            f"estimated finite STC domain memory {estimate_mb:.1f} MB "
            f"(local naux={naux}, nocc={nocc}, nvir={nvir}) "
            f"exceeds configured {limit_mb:.1f} MB"
        )
    return estimate_mb


def check_system_memory(mf, nocc, nvir, *, with_grad, virtual_block_size=16,
                        auxiliary_group_size=32):
    """Guard one complete replica plus root preparation/SCF residency.

    Conservative counts include tensor layout/host/AD copies, the dense
    auxiliary metric, pair-conditioned proposals and bounded scatter/tiles.
    Worker replicas use less memory because they have no SCF preparation.
    This is an estimate of known arrays, not a measured peak guarantee.
    """
    from pyscfad.lno._df_direct import _local_direct_int3c_block_mb

    if nocc == 0 or nvir == 0:
        return 0.0
    for name, value in (("virtual_block_size", virtual_block_size),
                        ("auxiliary_group_size", auxiliary_group_size)):
        if (not isinstance(value, (int, numpy.integer))
                or isinstance(value, (bool, numpy.bool_)) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise ValueError("system STC preparation requires density fitting")
    # Match the direct transformation's auxiliary regeneration, without
    # installing an auxmol on (and changing the pytree of) the SCF object.
    auxmol = pyscf_df.addons.make_auxmol(mf.mol.to_pyscf(), with_df.auxbasis)
    naux, nao = int(auxmol.nao), int(mf.mol.nao)
    ngroup = (naux + int(auxiliary_group_size) - 1) // int(auxiliary_group_size)
    elements = (12 if with_grad else 6) * naux * nocc * nvir
    elements += (6 if with_grad else 4) * naux * naux
    elements += (12 if with_grad else 6) * (nocc * nocc + nvir * nvir)
    # Four residual models with occupied-pair conditional virtual tables,
    # and absolute-column/group scores plus alias tables.
    elements += 24 * nocc * nocc * nvir + 12 * ngroup * nocc * nvir
    block = min(nvir, int(virtual_block_size))
    elements += 12 * block * block + 8 * naux * block
    # Native scatter buffers are bounded independently of production draws.
    scatter_mb = 8.0 if with_grad else 0.0
    resident_bytes = 0
    for value in (mf.mo_coeff, mf.mo_occ, mf.mo_energy,
                  getattr(mf, "_eri", None), getattr(with_df, "_cderi", None)):
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            resident_bytes += math.prod(value.shape) * numpy.dtype(value.dtype).itemsize
    elements += (12 if with_grad else 4) * nao * nao
    estimate_mb = (8 * elements + (3 if with_grad else 1) * resident_bytes) / 1e6
    estimate_mb += _local_direct_int3c_block_mb() + scatter_mb
    estimate_mb += _native_workspace_mb(naux, nocc, nvir,
                                        with_grad=with_grad, system=True)
    limit_mb = float(getattr(mf, "max_memory", mf.mol.max_memory))
    if estimate_mb > limit_mb:
        raise MemoryError(
            f"estimated system STC replica/preparation/SCF memory {estimate_mb:.1f} MB "
            f"(naux={naux}, nocc={nocc}, nvir={nvir}) "
            f"exceeds configured {limit_mb:.1f} MB"
        )
    return estimate_mb
