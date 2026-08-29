"""Restart the C16H34 IAO-DLNO-MP2 gradient from example 24.

This run reads the converged SCF checkpoint and saved DF integrals, then
resumes the progressive DLNO energy/cotangent checkpoint.
From the repository root, run example 24 once before running

    .venv/bin/python -u examples/lno/25-c16_dlno_mp2_gradient_restart.py
"""

from pathlib import Path

import numpy
from pyscf.scf import chkfile as pyscf_scf_chkfile

from pyscfad import config, gto, scf
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2, IAOFragmentMP2Thresholds


HERE = Path(__file__).resolve().parent
SAVE_DIR = HERE.parents[1] / "output" / "c16_dlno_mp2"
SCF_CHK = SAVE_DIR / "c16_ccpvdz_rhf.chk"
CDERI = SAVE_DIR / "c16_ccpvdz_cderi.h5"
DLNO_RESTART = SAVE_DIR / "c16_dlno_mp2.restart"

BASIS = "cc-pvdz"
AUXBASIS = "cc-pvdz-ri"
FROZEN = 16
SCF_CONV_TOL = 1e-12
SCF_CONV_TOL_GRAD = 1e-10

config.update("pyscfad_moleintor_opt", True)

if (
    not SCF_CHK.is_file()
    or not CDERI.is_file()
    or not (DLNO_RESTART / "run.json").is_file()
):
    raise FileNotFoundError(
        "Run 24-c16_dlno_mp2_gradient.py first to create the checkpoint "
        "and DF integral files."
    )

mol = gto.Mole(
    atom=str(HERE / "c16h34.xyz"),
    basis=BASIS,
    max_memory=4000,
    verbose=4,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)

print(f"Reading SCF checkpoint: {SCF_CHK}", flush=True)
print(f"Reading DF integrals:     {CDERI}", flush=True)
_, scf_data = pyscf_scf_chkfile.load_scf(str(SCF_CHK))
coeff = numpy.asarray(scf_data["mo_coeff"])
occ = numpy.asarray(scf_data["mo_occ"])
if coeff.shape[0] != mol.nao_nr():
    raise ValueError("The checkpoint AO dimension does not match this molecule")
dm0 = numpy.einsum(
    "pi,i,qi->pq", coeff, occ, coeff.conj(), optimize=True
)
del coeff, occ, scf_data


def build_mf(mol_):
    """Read CDERI from disk and install the implicit SCF response."""
    mf = scf.RHF(mol_).density_fit(auxbasis=AUXBASIS)
    mf.with_df.max_memory = mol_.max_memory
    mf.with_df.attach_outcore_cderi(str(CDERI))
    mf.chkfile = None
    # Match example 24 exactly.  Restarted local-orbital cotangents are tied
    # to the canonical MO gauge, so the fresh SCF response must converge to
    # the same orbitals rather than merely to the same total energy.
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.max_cycle = 100
    mf.kernel(dm0=dm0, dump_chk=False)
    if not mf.converged:
        raise RuntimeError("Restarted DF-RHF did not converge")
    return mf


print("Running restarted IAO-DLNO-MP2 energy and gradient", flush=True)
energy, mol_bar, details = IAOFragmentMP2.value_and_grad(
    mol,
    build_mf=build_mf,
    frozen=FROZEN,
    thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
    pair_energy_model="multipole",
    include_hf=True,
    return_details=True,
    checkpoint_dir=DLNO_RESTART,
    resume=True,
)

print(f"Total energy       {float(energy):+.12f} Eh")
print(f"Correlation energy {details.e_corr:+.12f} Eh")
print(f"Strong/weak energy {details.e_strong:+.12f} / {details.e_weak:+.12f} Eh")
print(f"Fragments          {details.n_fragments}")
print(f"Strong/weak pairs  {details.n_strong_pairs}/{details.n_weak_pairs}")
print("Gradient (Eh/Bohr):")
print(numpy.asarray(mol_bar.coords))
