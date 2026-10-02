"""Host-only helpers shared by full and finite STC packet workflows."""

from pathlib import Path
import subprocess

import numpy
from pyscf import df as pyscf_df

from pyscfad.dlno._restart import df_source_fingerprint


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
