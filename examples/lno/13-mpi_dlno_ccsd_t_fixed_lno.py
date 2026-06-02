'''DLNO-CCSD(T) on OMCB / aug-ccpvtz, outcore CDERI.

Builds the CDERI to disk once eagerly, then reuses it via
``attach_outcore_cderi`` inside ``build_mf`` so that the in-trace
``with_df.build()`` call is a no-op (re-building it under a JAX
trace would call ``outcore.cholesky_eri(mol, ...)`` on the traced
mol, which fails).
'''
import jax
import numpy

from pyscfad import config, gto, mp, scf
from pyscfad.lno import LNOCCSD
from pyscfad.dlno.ccsd import DLNOCCSD

config.update('pyscfad_moleintor_opt', True)
config.update('pyscfad_scf_implicit_diff', True)
config.update('pyscfad_scf_first_order_custom', True)


mol = gto.Mole(atom='omcb.xyz', basis='aug-ccpvtz', max_memory=3000)
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)


CDERI_PATH = "cderi.h5"

# Eager build of just the CDERI file once outside any JAX trace.  Skips
# the SCF iterations entirely — CDERI Cholesky only needs mol +
# auxbasis.  After this completes, ``cderi.h5`` exists on disk and
# ``build_mf`` below can attach it via the outcore-placeholder pattern
# (no re-build needed).
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
print(f'e_dlno = {e_dlno}')
print('g_dlno =')
print(g_dlno)
