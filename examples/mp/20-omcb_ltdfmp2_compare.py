import time

from pyscf import df
from pyscfad import gto, scf
from pyscfad import numpy as np
from pyscfad.mp import dfmp2, ltdfmp2

NLAP = 9
FIT_RATIO = 64.0


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def dfmp2_components(mymp):
    eris = mymp.ao2mo()
    mo_energy = eris.mo_energy
    mo_coeff = eris.mo_coeff
    nocc = mymp.nocc
    nvir = mymp.nmo - nocc
    Lov = mymp.loop_ao2mo(mo_coeff, nocc, with_t2=False).reshape(-1, nocc, nvir)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    e_direct = np.zeros((), dtype=Lov.dtype)
    e_exchange = np.zeros((), dtype=Lov.dtype)
    for i in range(nocc):
        gi = np.einsum('la,ljb->jab', Lov[:, i], Lov)
        t2i = gi / (eia[:, :, None] + eia[i, None, None, :])
        e_direct += 2.0 * np.einsum('jab,jab', t2i, gi)
        e_exchange -= np.einsum('jab,jba', t2i, gi)
    return e_direct.real, e_exchange.real


mol = gto.Mole(atom='omcb.xyz', basis='ccpvtz')
mol.verbose = 4
mol.build(trace_exp=False, trace_ctr_coeff=False)

auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
mf.with_df.incore = False
mf.kernel()

mymp = dfmp2.MP2(mf)
(e_dfmp2_direct, e_dfmp2_exchange), t_dfmp2 = timed(
    lambda: dfmp2_components(mymp)
)
e_dfmp2 = e_dfmp2_direct + e_dfmp2_exchange

myltmp = ltdfmp2.MP2(mf, nlap=NLAP, quadrature='fit', fit_ratio=FIT_RATIO)
(e_ltdfmp2, _), t_ltdfmp2 = timed(lambda: myltmp.kernel(with_t2=False))

print(f'LT quadrature points   = {NLAP}')
print(f'LT quadrature          = fitted exponential sum, R <= {FIT_RATIO:g}')
print(f'DF-MP2 time            = {t_dfmp2:.3f} s')
print(f'LT-DFMP2 time          = {t_ltdfmp2:.3f} s')
print(f'DF-MP2 direct 2J       = {e_dfmp2_direct:.12f}')
print(f'LT-DFMP2 direct 2J     = {myltmp.e_corr_direct:.12f}')
print(f'Direct error           = {myltmp.e_corr_direct - e_dfmp2_direct:.12f}')
print(f'DF-MP2 exchange -K     = {e_dfmp2_exchange:.12f}')
print(f'LT-DFMP2 exchange -K   = {myltmp.e_corr_exchange:.12f}')
print(f'Exchange error         = {myltmp.e_corr_exchange - e_dfmp2_exchange:.12f}')
print(f'DF-MP2 total           = {e_dfmp2:.12f}')
print(f'LT-DFMP2 total         = {e_ltdfmp2:.12f}')
print(f'Total error            = {e_ltdfmp2 - e_dfmp2:.12f}')
