// Copyright 2026 The PySCFAD Authors. Apache License, Version 2.0.
// Included inside the generated driver's extern "C" block.

// All panels are grid-fast, with total as the AO stride. tmp is compact in
// both dimensions: index[u] maps a full AO to its packed row, or -1. Each
// worker owns AO rows and accumulates four small grid reductions privately.
// No density screening or precision conversion occurs in this routine.
int SGX_contract_geometry(int nao,int total,int start,int count,
        const double* weights,const double* fg,const double* gv,
        const double* dgv,const double* dao,const int* index,const double* tmp,
        double* atom_ao,double* xed,double* dxed,int threads) {
    try {
        std::vector<double> partial(size_t(threads)*4*count,0.);
        #pragma omp parallel num_threads(threads)
        {
            int tid=0;
            #ifdef _OPENMP
            tid=omp_get_thread_num();
            #endif
            double* e=partial.data()+size_t(tid)*4*count;
            double* r0=e+count;double* r1=r0+count;double* r2=r1+count;
            #pragma omp for schedule(static)
            for(int u=0;u<nao;++u) {
                const size_t off=size_t(u)*total+start;
                const size_t stride=size_t(nao)*total;
                const double* f=fg+off;const double* g=gv+off;
                const double* d0=dgv+off;const double* d1=d0+stride;const double* d2=d1+stride;
                double a0=0.,a1=0.,a2=0.;
                if(index[u]>=0) {
                    const double* t=tmp+size_t(index[u])*count;
                    const double* b0=dao+off;const double* b1=b0+stride;const double* b2=b1+stride;
                    #pragma omp simd reduction(+:a0,a1,a2)
                    for(int k=0;k<count;++k) {
                        const double w=.5*weights[start+k],wf=w*f[k],wt=w*t[k];
                        const double z0=wf*d0[k]+wt*b0[k];
                        const double z1=wf*d1[k]+wt*b1[k];
                        const double z2=wf*d2[k]+wt*b2[k];
                        e[k]+=f[k]*g[k];r0[k]-=z0;r1[k]-=z1;r2[k]-=z2;
                        a0+=z0;a1+=z1;a2+=z2;
                    }
                } else {
                    #pragma omp simd reduction(+:a0,a1,a2)
                    for(int k=0;k<count;++k) {
                        const double wf=.5*weights[start+k]*f[k];
                        const double z0=wf*d0[k],z1=wf*d1[k],z2=wf*d2[k];
                        e[k]+=f[k]*g[k];r0[k]-=z0;r1[k]-=z1;r2[k]-=z2;
                        a0+=z0;a1+=z1;a2+=z2;
                    }
                }
                atom_ao[u]-=a0;atom_ao[nao+u]-=a1;atom_ao[2*nao+u]-=a2;
            }
        }
        // Fixed thread order avoids an OpenMP critical-section reduction.
        for(int tid=0;tid<threads;++tid) {
            const double* e=partial.data()+size_t(tid)*4*count;
            #pragma omp simd
            for(int k=0;k<count;++k) {
                xed[start+k]+=e[k];
                dxed[start+k]+=e[count+k];
                dxed[total+start+k]+=e[2*count+k];
                dxed[2*total+start+k]+=e[3*count+k];
            }
        }
        return 0;
    }catch(...) {return 1;}
}
