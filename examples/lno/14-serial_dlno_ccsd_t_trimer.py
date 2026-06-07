'''DLNO-CCSD(T) energy + nuclear gradient for water trimer / cc-pvdz (serial).

Run:
    python 14-serial_dlno_ccsd_t_trimer.py
'''
import numpy
from pyscfad import config, gto, scf
from pyscfad.dlno.ccsd import DLNOCCSD

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)

mol = gto.Mole(atom='water_trimer.xyz', basis='ccpvdz', max_memory=4000)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

CDERI_PATH = 'cderi_trimer_serial.h5'

# Build CDERI once outside any JAX trace; reuse via attach_outcore_cderi.
_df = scf.RHF(mol).density_fit().with_df
_df.max_memory = mol.max_memory
_df._cderi_to_save = CDERI_PATH
_df.build()
del _df


def build_mf(mol):
    mf = scf.RHF(mol).density_fit()
    mf.with_df.max_memory = mol.max_memory
    mf.with_df.attach_outcore_cderi(CDERI_PATH)
    mf.kernel()
    return mf


thr = 1e-4

e_dlno, jac_dlno = DLNOCCSD.value_and_grad(
    mol,
    build_mf=build_mf,
    ccsd_t=True,
    domain_pao_thr=thr,
    pair_energy_thr=thr,
    thresh_occ=1e-3,
    thresh_vir=1e-4,
)
e_dlno = float(e_dlno)
g_dlno = numpy.asarray(jac_dlno.coords)
print(f'DLNO-CCSD(T) energy = {e_dlno:.10f}')
print('DLNO-CCSD(T) gradient (Hartree/Bohr):')
print(g_dlno)
