"""C16H34 IAO-DLNO-MP2 energy and nuclear gradient.

This first run writes the DF integrals, SCF checkpoint, and progressive DLNO
restart directory used by example 25.
Run from the repository root with

    .venv/bin/python -u examples/lno/24-c16_dlno_mp2_gradient.py
"""

from pathlib import Path

import numpy

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
SAVE_DIR.mkdir(parents=True, exist_ok=True)

mol = gto.Mole(
    atom=str(HERE / "c16h34.xyz"),
    basis=BASIS,
    max_memory=4000,
    verbose=4,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)

print(f"Building DF integrals: {CDERI}", flush=True)
df_builder = scf.RHF(mol).density_fit(auxbasis=AUXBASIS).with_df
df_builder.max_memory = mol.max_memory
df_builder._cderi_to_save = str(CDERI)
df_builder.build()
del df_builder

def build_mf(mol_):
    """Converge DF-RHF and save the SCF checkpoint used by example 25."""
    mf = scf.RHF(mol_).density_fit(auxbasis=AUXBASIS)
    mf.with_df.max_memory = mol_.max_memory
    mf.with_df.attach_outcore_cderi(str(CDERI))
    mf.chkfile = str(SCF_CHK)
    mf.conv_tol = SCF_CONV_TOL
    mf.conv_tol_grad = SCF_CONV_TOL_GRAD
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("DF-RHF did not converge")
    return mf


print(f"SCF checkpoint will be written to: {SCF_CHK}", flush=True)
print("Running DF-RHF and IAO-DLNO-MP2 gradient", flush=True)
energy, mol_bar, details = IAOFragmentMP2.value_and_grad(
    mol,
    build_mf=build_mf,
    frozen=FROZEN,
    thresholds=IAOFragmentMP2Thresholds(pair_energy=1e-4),
    pair_energy_model="multipole",
    include_hf=True,
    return_details=True,
    checkpoint_dir=DLNO_RESTART,
)

print(f"Total energy       {float(energy):+.12f} Eh")
print(f"Correlation energy {details.e_corr:+.12f} Eh")
print(f"Strong/weak energy {details.e_strong:+.12f} / {details.e_weak:+.12f} Eh")
print(f"Fragments          {details.n_fragments}")
print(f"Strong/weak pairs  {details.n_strong_pairs}/{details.n_weak_pairs}")
print("Gradient (Eh/Bohr):")
print(numpy.asarray(mol_bar.coords))
