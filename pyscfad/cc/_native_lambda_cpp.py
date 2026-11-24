# Copyright 2026 The PySCFAD Authors (Apache-2.0).
"""Resource policy and dense response iteration for native RCCSD lambda."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd
from threadpoolctl import threadpool_limits
from ._native_kernels import _ccsd_kernels as cpp
from ._native_ccsd_memory import _private

# Binary contractions that can split a long reduction into private outputs.
# Bound partial virtual tiles as well as full dimensions; reduction partition
# eligibility is not monotonic in matrix dimensions.
_PRIVATE_EXPRESSIONS = (
    'kc,jkbc->bj','jcab,jc->ba','jabc,jc->ba','jkcd,kdac->aj',
    'cibd,jkbd->ckij','lckj,klcb->bj','jlcb,lbki->cjki',
    'cijd,klcd->ijkl','ckld,ld->kc','ckja,kjcb->ba','jkbc,bikc->ij',
    'bjld,klcd->bjkc','bjld,kldc->bjkc','bljd,kldc->bjkc',
    'jabd,jkcd->abkc','acjd,kjbd->ackb','abjd,kjcd->abkc',
    'laki,klbc->abic','abkc,kibc->ia','klca,klcb->ba','ikcd,jkcd->ij',
    'ijab,jb->ia','jiab,jb->ia','ijab,bj->ia','ijba,bj->ia',
    'iabc,bc->ia','ibac,bc->ia','ijkd,kadb->ijab','ijab,ia->jb',
    'ijcd,klcd->ijkl','ijkl,aklb->ijab','ckij,jkca->ia',
    'iakc,bjkc->jiba','jkca,bikc->jiba','ijkl,klab->ijab')


def memory_bound(o, v, block, ladder, *, resident_ovvv=True, cache_wvvov=True,
                 response=False):
    """Native live-buffer bound; inputs, source and DIIS are already in RSS.

    Phase peaks are alternatives. The contraction arena keeps its maximum
    capacity across phases; private GEMM reductions coexist with that arena.
    This is a conservative allocation bound, not an allocator RSS prediction.
    """
    d, g, q, occ = 8*o*o*v*v, 8*o*v**3, 8*o**4, 8*o**3*v
    a, small = d+8*o*v, 8*(v*v+o*o+2*o*v)
    db, gb, wt = 8*o*o*block*v, 8*o*block*v*v, 8*block*block*o*v
    indices = 8*(v*v+o*(o+1)//2)
    private = 0
    choices = {1, block, v, v % block} - {0}
    for spec in _PRIVATE_EXPRESSIONS:
        operands, output = spec.split('->'); left, right = operands.split(',')
        virtual = sorted(set(operands)&set('abcd'))
        for sizes in product(choices, repeat=len(virtual)):
            dims = dict.fromkeys('ijkl',o);dims.update(zip(virtual,sizes))
            extent = lambda labels: int(np.prod([dims[x] for x in labels], dtype=np.int64))
            m = extent([x for x in left if x not in right])
            n = extent([x for x in right if x not in left])
            k = extent([x for x in left if x in right])
            private = max(private, _private(m,n,k,int(ladder['threads'])))
    # The full Wvvov operand is already a matrix for its singles contraction.
    # Only its l2 operand is packed; the vovv input is unpacked in bounded tiles.
    arena = max(int(ladder['scratch_bytes']),3*d+192,
                d+gb+wt+192, d+2*occ+192, 2*gb+wt+192)
    layouts = int(resident_ovvv)*g
    tiles = 0 if resident_ovvv else 2*gb
    base = small+indices+occ+q
    phases = {
        'build_ovvv':base+layouts+5*d+gb+tiles+arena+private,
        'build_ovoo_ovov':base+layouts+5*d+2*occ+q+arena+private,
        'build_wvvov':base+layouts+int(cache_wvvov)*g+4*d+occ+4*db+gb+wt+tiles+arena+private,
    }
    resident = base+layouts+int(cache_wvvov)*g+(3 if cache_wvvov else 4)*d
    phases['update'] = resident+(2 if response else 1)*a+d+max(d,tiles+occ)+small*4+arena+private
    if not cache_wvvov:
        phases['stream_wvvov'] = resident+(2 if response else 1)*a+d+occ+4*db+gb+2*wt+tiles+small*4+arena+private
    return dict(peak_bytes=max(phases.values()),phase_bytes=phases,
                resident_bytes=resident,scratch_bytes=arena,private_bytes=private)


@dataclass
class Intermediates:
    workspace: object
    plan: dict
    preparation_profile: dict


def supported(cc,t1,t2,eris):
    if (cpp is None or not hasattr(cpp,'lambda_prepare')
            or not getattr(cc,'cpp_kernels',False) or cc.cc2 or cc.direct
            or getattr(cc,'dcsd',False) or cc.async_io or not cc.incore_complete
            or not isinstance(eris,ccsd._ChemistsERIs)):
        return False
    arrays=(t1,t2,eris.fock,eris.mo_energy,eris.oooo,eris.ovoo,eris.oovv,
            eris.ovvo,eris.ovvv,eris.vvvv)
    return all(isinstance(x,np.ndarray) and x.dtype==np.float64 for x in arrays)


def prepare(cc,t1,t2,eris,*,reserve_bytes=0,response=False,profile=False):
    if not supported(cc,t1,t2,eris):return None
    o,v=t1.shape
    if not o or not v:return None
    available=int((cc.max_memory-lib.current_memory()[0])*1e6)-int(reserve_bytes)
    if not eris.vvvv.flags.c_contiguous:available-=eris.vvvv.nbytes
    if available<=0:return None
    ladders=[cpp.ladder_memory(o,v)]
    minimum=cpp.ladder_memory(o,v,8*v*v)
    if minimum['panel_bytes']!=ladders[0]['panel_bytes']:ladders.append(minimum)
    # Prefer Wvvov reuse; its reconstruction contains several large products.
    # The unpacked integral cache is optional and can be replaced by small tiles.
    for resident,cache in ((True,True),(False,True),(True,False),(False,False)):
        for ladder in ladders:
            for block in dict.fromkeys((min(32,v),min(16,v),min(8,v),1)):
                estimate=memory_bound(o,v,block,ladder,resident_ovvv=resident,
                    cache_wvvov=cache,response=response)
                if estimate['peak_bytes']>available:continue
                with threadpool_limits(1,user_api='blas'):
                    workspace,report=cpp.lambda_prepare([t1,t2,eris.fock,eris.mo_energy,
                        eris.oooo,eris.ovoo,eris.oovv,eris.ovvo,eris.ovvv,
                        np.ascontiguousarray(eris.vvvv)],block,ladder['panel_bytes'],resident,cache,profile)
                return Intermediates(workspace,dict(block=block,ladder=ladder,
                    resident_ovvv=resident,cache_wvvov=cache,estimate=estimate),report)
    return None


def update(cc,l1,l2,imds,*,profile=False):
    with threadpool_limits(1,user_api='blas'):
        return cpp.lambda_update(imds.workspace,l1,l2,cc.level_shift,profile)


def solve_response(cc,eris,t1,t2,bar_e,bar_t1,bar_t2):
    """Keep covectors dense; use the exact packed metric for norm and DIIS."""
    if (not supported(cc,t1,t2,eris) or cc.diis_file is not None
            or type(cc.diis) is not bool or (cc.diis and cc.diis_space<2)):
        return None
    from ._native_ccsd_solver import DenseDIIS
    # Source formation subtracts M D K(0) analytically. No zero-lambda update,
    # denominator-sized tensor, or second permanent source copy is required.
    history=2*cc.diis_space if cc.diis else 0
    imds=prepare(cc,t1,t2,eris,reserve_bytes=(history+2)*(t1.nbytes+t2.nbytes),response=True)
    if imds is None:return None
    source=cpp.lambda_source(t1,eris.fock,eris.ovov,eris.ovvo,float(bar_e),
                             np.asarray(bar_t1),np.asarray(bar_t2))
    diis=DenseDIIS(cc.diis_space) if cc.diis else None
    x1,x2=t1,t2
    converged=False
    for step in range(cc.max_cycle):
        with threadpool_limits(1,user_api='blas'):
            new1,new2=cpp.lambda_response(imds.workspace,x1,x2,*source,cc.level_shift)
        norm=cpp.difference_norm(new1,new2,x1,x2)
        x1,x2=new1,new2
        new1=new2=None
        if diis is not None and step>=cc.diis_start_cycle and cc.diis_start_energy_diff>0:
            x1,x2=diis.update(x1,x2,take_ownership=True)
        logger.info(cc,'native response cycle = %d  norm(lambda) = %.6g',step+1,norm)
        if norm<cc.conv_tol_normt:
            converged=True
            break
    if not converged:logger.warn(cc,'Native DF-CCSD response did not converge')
    # The integral adjoint still consumes one packed vector at this boundary.
    # Release iteration-only caches/history before that final conversion.
    imds=diis=source=None
    return converged,ccsd.amplitudes_to_vector(x1,x2)
