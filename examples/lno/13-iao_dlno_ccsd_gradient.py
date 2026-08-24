"""Minimal serial IAO-DLNO-CCSD energy and nuclear-gradient example.

The molecule is defined in this file, so the example can be run directly
from the repository root without command-line arguments or external files::

    python examples/lno/13-iao_dlno_ccsd_gradient.py

For a production calculation, replace the small basis and increase the local
space/domain accuracy as needed. Perturbative triples can be enabled with
``ccsd_t=True`` in the call below, but are intentionally omitted here so that
this remains a quick gradient smoke test.
"""

import numpy

from pyscfad import config, gto, scf
from pyscfad.dlno.ccsd import DLNOCCSD


config.update("pyscfad_moleintor_opt", True)


mol = gto.Mole(
    atom="""
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757000  0.587000
    H  0.000000  0.757000  0.587000
    """,
    unit="Angstrom",
    basis="sto-3g",
    verbose=3,
    max_memory=1000,
)
mol.build(trace_exp=False, trace_ctr_coeff=False)


def build_mf(mol_):
    """Converge the density-fitted RHF reference differentiated below."""
    mf = scf.RHF(mol_).density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    return mf


energy, mol_bar = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    thresh_occ=1e-3,
    thresh_vir=1e-3,
    ccsd_t=False,
)

print(f"IAO-DLNO-CCSD total energy = {float(energy):.12f}")
print("IAO-DLNO-CCSD nuclear gradient (Eh/Bohr):")
print(numpy.asarray(mol_bar.coords))
