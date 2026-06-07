import time

from pyscf import df
from pyscfad import gto, scf
from pyscfad import numpy as np
from pyscfad.lno import lno_base
from pyscfad.mp import dfmp2, ltdfmp2

NLAP = 9
FIT_RATIO = 64.0
IAO_NUMBER = 10


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def dfmp2_projected_occ_components(mymp, occ_projector):
    eris = mymp.ao2mo()
    mo_energy = eris.mo_energy
    mo_coeff = eris.mo_coeff
    nocc = mymp.nocc
    nvir = mymp.nmo - nocc
    Lov = mymp.loop_ao2mo(mo_coeff, nocc, with_t2=False).reshape(-1, nocc, nvir)
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    Lov_local = np.einsum('i,Lia->La', occ_projector, Lov)
    g_local = np.dot(Lov_local.T, Lov.reshape(Lov.shape[0], nocc*nvir))
    g_local = g_local.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
    g_local_ex = g_local.transpose(0, 2, 1)

    e_direct = np.zeros((), dtype=Lov.dtype)
    e_exchange = np.zeros((), dtype=Lov.dtype)
    for i in range(nocc):
        gi = np.dot(Lov[:, i].T, Lov.reshape(Lov.shape[0], nocc*nvir))
        gi = gi.reshape(nvir, nocc, nvir).transpose(1, 0, 2)
        t2i = gi / (eia[:, :, None] + eia[i, None, None, :])
        e_direct += 2.0 * occ_projector[i] * np.einsum('jab,jab->', t2i, g_local)
        e_exchange -= occ_projector[i] * np.einsum('jab,jab->', t2i, g_local_ex)
    return e_direct.real, e_exchange.real


mol = gto.Mole(atom='omcb.xyz', basis='ccpvdz', max_memory=2000)
mol.verbose = 3
mol.build(trace_exp=False, trace_ctr_coeff=False)

auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
mf.with_df.incore = False
mf.kernel()

mymp = dfmp2.MP2(mf)
eris = mymp.ao2mo()
nocc = mymp.nocc
orbocc = eris.mo_coeff[:, :nocc]
iaos = lno_base.get_iao(mol, orbocc)
iao_idx = IAO_NUMBER - 1
s1e = mf.get_ovlp()
occ_projector = np.linalg.multi_dot((orbocc.T, s1e, iaos[:, iao_idx]))
projector_norm = np.dot(occ_projector, occ_projector)

(e_df_direct, e_df_exchange), t_df = timed(
    lambda: dfmp2_projected_occ_components(mymp, occ_projector)
)
e_df = e_df_direct + e_df_exchange

myltmp = ltdfmp2.MP2(mf, nlap=NLAP, quadrature='fit', fit_ratio=FIT_RATIO)
(e_lt, e_lt_direct, e_lt_exchange), t_lt = timed(
    lambda: myltmp.projected_occ_energy(occ_projector)
)

print(f'IAO number             = {IAO_NUMBER}')
print(f'IAO occupied norm      = {projector_norm:.12f}')
print(f'LT quadrature points   = {NLAP}')
print(f'LT quadrature          = fitted exponential sum, R <= {FIT_RATIO:g}')
print(f'DF projected time      = {t_df:.3f} s')
print(f'LT projected time      = {t_lt:.3f} s')
print(f'DF projected direct 2J = {e_df_direct:.12f}')
print(f'LT projected direct 2J = {e_lt_direct:.12f}')
print(f'Direct error           = {e_lt_direct - e_df_direct:.12f}')
print(f'DF projected exch -K   = {e_df_exchange:.12f}')
print(f'LT projected exch -K   = {e_lt_exchange:.12f}')
print(f'Exchange error         = {e_lt_exchange - e_df_exchange:.12f}')
print(f'DF projected total     = {e_df:.12f}')
print(f'LT projected total     = {e_lt:.12f}')
print(f'Total error            = {e_lt - e_df:.12f}')
