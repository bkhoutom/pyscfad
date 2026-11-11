"""Fixed-amplitude CCSD/lambda timing: isolate thread and virtual-cache effects."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs', type=Path, required=True)
p.add_argument('--amplitudes', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cpus', default='0-31')
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--threads', default='32:32,8:8,1:1,32:1,1:32')
a = p.parse_args()
lo, hi = map(int,a.cpus.split('-'))
os.sched_setaffinity(0,range(lo,hi+1))
os.environ.update(OMP_NUM_THREADS='32',OPENBLAS_NUM_THREADS='32',MKL_NUM_THREADS='32')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false')
import numpy as np
from pyscf import lib
from pyscf.cc import ccsd
from threadpoolctl import threadpool_limits
from pyscfad.cc import _dfccsd_native as native

with np.load(a.inputs) as data:
    arrays = [data[n] for n in native.ERI_NAMES]
with np.load(a.amplitudes) as data:
    t1,t2=data['t1'],data['t2']
opts=dict(max_memory=900000,max_cycle=200,conv_tol=1e-12,conv_tol_normt=1e-10,
          iterative_damping=1.,diis_space=3,diis_start_cycle=0,
          diis_start_energy_diff=1e9,level_shift=0.,verbose=0,stdout=sys.stdout,diis=True,
          vvvv_direct=True)  # this experiment installs its own packed override
rng=np.random.default_rng(515)
vector=rng.normal(size=ccsd.amplitudes_to_vector(t1,t2).shape)*.001
records=[]
result=dict(nocc=t1.shape[0],nvir=t1.shape[1],naux=arrays[-1].shape[0],
            affinity=list(os.sched_getaffinity(0)),repeats=a.repeats,records=records,
            environment={n:os.getenv(n) for n in ('OMP_WAIT_POLICY','GOMP_SPINCOUNT')})
reference={}
def timed(fn,repeat):
    ws=[];cs=[]
    for _ in range(repeat):
        w,c=time.perf_counter(),time.process_time(); value=fn()
        ws.append(time.perf_counter()-w);cs.append(time.process_time()-c)
    return value,dict(wall_s=ws,cpu_s=cs,median_wall_s=float(np.median(ws)),
                      average_cpus=sum(cs)/sum(ws))

for omp,blas in (map(int,s.split(':')) for s in a.threads.split(',')):
    with lib.with_omp_threads(omp),threadpool_limits(omp,user_api='openmp'), \
            threadpool_limits(blas,user_api='blas'):
        for mode in ('df','packed'):
            cc,eris=native.host_objects(dict(opts,threads=omp),arrays,t1.shape[0])
            record=dict(omp=omp,blas=blas,ladder=mode)
            if mode=='packed':
                packed,record['cache_setup']=timed(lambda:eris.vvL@eris.vvL.T,1)
                def contract(mycc,tensor,direct=False,out=None,verbose=None):
                    return ccsd._contract_vvvv_t2(mycc,eris.mol,packed,tensor,out,verbose)
                eris._contract_vvvv_t2=contract
            action,record['lambda_setup']=timed(lambda:native.response_action(cc,eris,t1,t2),1)
            for name,fn in [('forward_update',lambda:cc.update_amps(t1,t2,eris)),
                            ('lambda_action',lambda:action(vector))]:
                fn()  # warmup is excluded from the repeated timings
                value,record[name]=timed(fn,a.repeats)
                flat=np.concatenate([x.ravel() for x in value]) if isinstance(value,tuple) else value
                reference.setdefault(name,flat.copy())
                error=float(np.max(np.abs(flat-reference[name])))
                record[name]['max_abs_difference']=error
                assert error<1e-10,(name,error)
            records.append(record)
            a.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(record),flush=True)

# Count matrix sizes in one ordinary DF forward update. Instrumentation is
# outside the measured calls and does not change any numerical routine.
cc,eris=native.host_objects(dict(opts,threads=1),arrays,t1.shape[0])
sizes=Counter()
original=lib.numpy_helper._dgemm
def counted(ta,tb,m,n,k,*args,**kwargs):
    sizes[(m,n,k)]+=1
    return original(ta,tb,m,n,k,*args,**kwargs)
lib.numpy_helper._dgemm=counted
with lib.with_omp_threads(1),threadpool_limits(1):
    cc.update_amps(t1,t2,eris)
lib.numpy_helper._dgemm=original
result['forward_gemm_shapes']=[dict(m=m,n=n,k=k,calls=count) for (m,n,k),count in sizes.most_common()]
a.output.write_text(json.dumps(result,indent=2)+'\n')
