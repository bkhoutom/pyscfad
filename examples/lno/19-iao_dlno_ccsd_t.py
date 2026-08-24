"""Small IAO-DLNO-CCSD(T) energy/gradient example.

The LIS of every fragment is selected by its target-conditioned strong-ED
MP2 density.  The reported energy is

    sum_F [CCSD(T)_F(LIS_F) - MP2_F(LIS_F)]
      + IAO-DLNO-MP2(strong ED + weak multipole),

plus the RHF reference.  There is no SOS correction option.
"""

from pathlib import Path

import numpy

from pyscfad import config, gto, scf
from pyscfad.dlno.ccsd import DLNOCCSD
from pyscfad.dlno.iao_mp2 import IAOFragmentMP2Thresholds


config.update("pyscfad_moleintor_opt", True)
config.update("pyscfad_scf_implicit_diff", True)
config.update("pyscfad_scf_first_order_custom", False)

mol = gto.Mole(
    atom=str(Path(__file__).resolve().with_name("water_dimer.xyz")),
    basis="sto-3g",
    verbose=4,
    max_memory=2000,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)


def build_mf(mol_):
    mf = scf.RHF(mol_).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf


thresholds = IAOFragmentMP2Thresholds(pair_energy=1e-4)
energy, gradient = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    thresholds=thresholds,
    thresh_occ=1e-3,
    thresh_vir=1e-3,
    ccsd_t=True,
)

print(f"IAO-DLNO-CCSD(T) total energy = {float(energy):.12f}")
print("Nuclear gradient (Eh/bohr):")
print(numpy.asarray(gradient.coords))
