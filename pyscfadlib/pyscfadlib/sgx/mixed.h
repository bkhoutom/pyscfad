// Copyright 2026 The PySCFAD Authors. Apache License, Version 2.0.
// Included inside namespace pyscfad_sgx after the FP64 reference engine.

// Exact powers of two keep small FP32 intermediates out of the subnormal
// range without changing their significand precision. Scale F and integrals
// independently; undo both factors only when merging G/dG into FP64.
constexpr double FLOAT_SCALE=4294967296.; // 2^32
constexpr double FLOAT_UNSCALE=5.42101086242752217004e-20; // 2^-64

struct FloatScratch {
    std::vector<float> boys, scaled, gx,gy,gz,ints,base,f,g,deriv,dg;
    std::vector<float> p,ax,ay,az,alpha,beta;
    float coeff[5][TILE], ex[TILE], delta[TILE], t[TILE];
    int index[TILE];
    FloatScratch(const Plan& plan,int ns,bool derivative):
        boys(size_t(plan.maxnp)*NBOYS*TILE),scaled(boys.size()),
        gx(size_t(plan.maxnp)*TILE),gy(gx.size()),gz(gx.size()),
        ints(size_t(plan.maxpair)*TILE),base(size_t(plan.maxbase)*TILE),
        f(size_t(ns)*plan.ncart*TILE),g(f.size()),
        deriv(derivative ? size_t(7)*plan.maxpair*TILE : 0),
        dg(derivative ? size_t(3)*ns*plan.ncart*TILE : 0),
        p(plan.maxnp),ax(plan.maxnp),ay(plan.maxnp),az(plan.maxnp),
        alpha(plan.maxnp),beta(plan.maxnp) {}
};

void boys_batch_float(const BoysTable& table,int nmax,int ng,const double* x,
                       double scale,float* out,FloatScratch& w) {
    // Fourth-degree Taylor interpolation: truncation < 8e-9 at the worst
    // half-grid displacement. Gather and argument reduction stay outside SIMD.
    // Build arguments/differences in FP64 before converting to FP32.
    const float fac=float(scale);
    double xmin=x[0];
    #pragma omp simd reduction(min:xmin)
    for(int g=0;g<ng;++g)xmin=std::min(xmin,x[g]);
    if(xmin>=64.) {
        #pragma omp simd
        for(int g=0;g<ng;++g) {
            w.t[g]=float(1./x[g]);out[g]=fac*.5f*std::sqrt(float(PI)*w.t[g]);
        }
        // 1/x can underflow before its square root does. Preserve representable
        // F0 values for extreme arguments with a rare FP64 fallback.
        for(int g=0;g<ng;++g)if(x[g]>1e30)out[g]=float(scale*.5*std::sqrt(PI/x[g]));
        for(int n=1;n<=nmax;++n) {
            #pragma omp simd
            for(int g=0;g<ng;++g)out[n*ng+g]=out[(n-1)*ng+g]*(n-.5f)*w.t[g];
        }
        return;
    }
    for(int g=0;g<ng;++g) {
        const double t=std::min(x[g],64.);
        const int j=int(t*8+.5);
        w.t[g]=float(t);w.delta[g]=float(j/8.-t);
        for(int k=0;k<5;++k)w.coeff[k][g]=float(table.f[j][nmax+k]);
        w.ex[g]=float(table.e[j]);
    }
    #pragma omp simd
    for(int g=0;g<ng;++g) {
        const float d=w.delta[g];
        const float y=((((w.coeff[4][g]*(1.f/24.f)*d+w.coeff[3][g]*(1.f/6.f))*d
                    +w.coeff[2][g]*.5f)*d+w.coeff[1][g])*d+w.coeff[0][g]);
        const float z=((((1.f/24.f)*d+1.f/6.f)*d+.5f)*d+1.f)*d+1.f;
        out[nmax*ng+g]=y*fac;w.ex[g]*=z*fac;
    }
    for(int n=nmax;n>0;--n) {
        const float inv=1.f/(2*n-1);
        #pragma omp simd
        for(int g=0;g<ng;++g)out[(n-1)*ng+g]=(2.f*w.t[g]*out[n*ng+g]+w.ex[g])*inv;
    }
    int count=0;
    for(int g=0;g<ng;++g)if(x[g]>=64.) {
        // The reciprocal is formed before conversion to avoid overflow for
        // large finite FP64 arguments. Underflow to FP32 zero is benign.
        w.t[count]=float(1./x[g]);w.index[count++]=g;
    }
    #pragma omp simd
    for(int g=0;g<count;++g)w.ex[g]=fac*.5f*std::sqrt(float(PI)*w.t[g]);
    for(int g=0;g<count;++g)if(x[w.index[g]]>1e30)
        w.ex[g]=float(scale*.5*std::sqrt(PI/x[w.index[g]]));
    for(int g=0;g<count;++g)out[w.index[g]]=w.ex[g];
    for(int n=1;n<=nmax;++n) {
        #pragma omp simd
        for(int g=0;g<count;++g)w.ex[g]*=(n-.5f)*w.t[g];
        for(int g=0;g<count;++g)out[n*ng+w.index[g]]=w.ex[g];
    }
}

bool prepare_float(const Plan& plan,const Pair& pair,int ng,Scratch& d,
                   FloatScratch& w,bool derivative,double work_scale=1.) {
    const Shell& a=plan.shells[pair.a];const Shell& b=plan.shells[pair.b];
    const int nm=a.l+b.l+int(derivative);
    for(int i=0;i<a.np;++i)w.alpha[i]=float(a.exp[i]);
    for(int i=0;i<b.np;++i)w.beta[i]=float(b.exp[i]);
    for(int k=0;k<pair.np;++k) {
        w.p[k]=float(pair.p[k]);w.ax[k]=float(pair.ax[k]);
        w.ay[k]=float(pair.ay[k]);w.az[k]=float(pair.az[k]);
        const double px=a.r[0]+pair.ax[k],py=a.r[1]+pair.ay[k],pz=a.r[2]+pair.az[k];
        #pragma omp simd
        for(int g=0;g<ng;++g) {
            const double x=px-d.xyz[g],y=py-d.xyz[TILE+g],z=pz-d.xyz[2*TILE+g];
            w.gx[k*ng+g]=float(x);w.gy[k*ng+g]=float(y);w.gz[k*ng+g]=float(z);
            d.x[g]=pair.p[k]*(x*x+y*y+z*z);
        }
        const double factor=work_scale*pair.factor[k]*(pair.nc==1 ? pair.coeff[k] : 1.);
        if(!std::isfinite(float(factor)) || !(w.p[k]>0) || !std::isfinite(w.p[k]))return false;
        boys_batch_float(plan.table,nm,ng,d.x,factor,w.boys.data()+size_t(k)*(nm+1)*ng,w);
    }
    return true;
}

void evaluate_float(const Plan& plan,const Pair& p,int cp,int ng,FloatScratch& w,bool derivative) {
    const Shell& a=plan.shells[p.a];const Shell& b=plan.shells[p.b];
    const int nm=a.l+b.l+int(derivative);
    const float* boys=w.boys.data();
    if(p.nc>1) {
        for(int k=0;k<p.np;++k) {
            const int off=k*(nm+1)*ng;const float c=float(p.coeff[cp*p.np+k]);
            #pragma omp simd
            for(int i=0;i<(nm+1)*ng;++i)w.scaled[off+i]=c*w.boys[off+i];
        }
        boys=w.scaled.data();
    }
    if(derivative) {
        fp32::select_derivative_kernel(a.l,b.l,p.same)(w.deriv.data(),boys,w.p.data(),
            w.gx.data(),w.gy.data(),w.gz.data(),w.ax.data(),w.ay.data(),w.az.data(),
            float(p.ab[0]),float(p.ab[1]),float(p.ab[2]),w.alpha.data(),w.beta.data(),b.np,p.np,ng);
    } else {
        // FP32 distinct-center d-d/f-p cross over already at one primitive.
        const bool hrr=p.hrr && ((p.np>1 && nm>3)||(!p.same&&(nm>=4||(nm==3&&p.np>=4))));
        fp32::select_kernel(a.l,b.l,p.same,hrr)(w.ints.data(),w.base.data(),boys,w.p.data(),
            w.gx.data(),w.gy.data(),w.gz.data(),w.ax.data(),w.ay.data(),w.az.data(),
            float(p.ab[0]),float(p.ab[1]),float(p.ab[2]),p.np,ng);
    }
}

void digest_float(const Plan& p,const Pair& q,int cp,int ns,int ng,FloatScratch& w,bool derivative) {
    if(derivative)digest_values<float,true>(p,q,cp,ns,ng,w);
    else digest_values<float,false>(p,q,cp,ns,ng,w);
}

int adaptive_contract(const Plan& p,int ns,int total,const double* coords,const double* fg,
        const double* weights,const double* ao_l1,double* gv,double* dgv,
        double etol,double ktol,double fp32_multiplier,int screen,int threads,uint64_t* counts) {
    const bool derivative=dgv!=nullptr,mixed=fp32_multiplier>0.;
    const bool cumulative=screen==1,per_pair=screen==2;
    #ifdef _OPENMP
    const int previous_dynamic=omp_get_dynamic();omp_set_dynamic(0);
    #endif
    std::atomic<bool> failed(false);
    std::fill(counts,counts+6,uint64_t(0));
    #pragma omp parallel num_threads(threads)
    {
        std::unique_ptr<Scratch> dp;std::unique_ptr<FloatScratch> sp;
        std::vector<double> norms, pair_energy, pair_potential;
        std::vector<uint8_t> bucket;
        uint64_t local[6]={0,0,0,0,0,0};
        try {
            dp.reset(new Scratch(p,ns,derivative));
            if(mixed)sp.reset(new FloatScratch(p,ns,derivative));
            norms.resize(p.shells.size());
            if(cumulative) {
                pair_energy.resize(p.pairs.size());pair_potential.resize(p.pairs.size());
                bucket.resize(p.pairs.size());
            }
        } catch(...) {dp.reset();sp.reset();failed=true;}
        #pragma omp for schedule(dynamic,1)
        for(int start=0;start<total;start+=TILE) {
            if(!dp)continue;
            Scratch& d=*dp;const int ng=std::min(TILE,total-start);
            for(int g=0;g<ng;++g)for(int k=0;k<3;++k)d.xyz[k*TILE+g]=coords[3*(start+g)+k];
            to_cart(p,ns,ng,start,total,fg,d);
            if(derivative)std::fill(d.dg.begin(),d.dg.begin()+size_t(3)*ns*p.ncart*ng,0.);
            bool float_safe=true;
            if(mixed) {
                const size_t size=size_t(ns)*p.ncart*ng;
                for(size_t i=0;i<size;++i) {
                    sp->f[i]=float(FLOAT_SCALE*d.f[i]);
                    if(!std::isfinite(sp->f[i]))float_safe=false;
                }
                std::fill(sp->g.begin(),sp->g.begin()+size,0.f);
                if(derivative)std::fill(sp->dg.begin(),sp->dg.begin()+3*size,0.f);
            }
            double x2=0.;
            for(int g=0;g<ng;++g)x2+=std::abs(weights[start+g])*ao_l1[start+g]*ao_l1[start+g];
            const double xn=std::sqrt(x2);
            for(size_t a=0;a<p.shells.size();++a) {
                const Shell& sh=p.shells[a];double largest=0.;
                // Take the union across all densities/adjoints in this call.
                for(int s=0;s<ns;++s) {
                    double sum=0.;
                    for(int i=0;i<sh.nc*sh.ncart;++i) {
                        const double* f=d.f.data()+(size_t(s)*p.ncart+sh.coff+i)*ng;
                        #pragma omp simd reduction(+:sum)
                        for(int g=0;g<ng;++g)sum+=std::abs(weights[start+g])*f[g]*f[g];
                    }
                    largest=std::max(largest,sum);
                }
                norms[a]=std::sqrt(largest);
            }
            const double ecut=etol*ng,kcut=ktol*ng;
            double ebin[64]={},kbin[64]={};
            for(size_t i=0;cumulative && i<p.pairs.size();++i) {
                const Pair& q=p.pairs[i];
                const double a=norms[q.a],b=norms[q.b];
                const double bound=q.bound*(derivative ? q.derivative_guard : 1.);
                const double energy=bound*a*b,potential=bound*std::hypot(a,b)*xn;
                pair_energy[i]=energy;pair_potential[i]=potential;bucket[i]=64;
                if(screen && energy<ecut && potential<kcut) {
                    const double score=std::max(energy/ecut,potential/kcut);
                    int exponent=0;if(score>0.)std::frexp(score,&exponent);
                    const int bin=score>0. ? std::max(0,std::min(63,63+exponent)) : 0;
                    bucket[i]=uint8_t(bin);ebin[bin]+=energy;kbin[bin]+=potential;
                }
            }
            // Spend the budget on the smallest contributions first. An
            // unsorted greedy pass can spend almost everything on a few
            // early pairs, retaining thousands of much smaller later pairs.
            // Logarithmic buckets give linear work and bounded scratch.
            double skipped_e=0.,skipped_k=0.;int threshold=0;
            for(;cumulative && threshold<64;++threshold) {
                if(skipped_e+ebin[threshold]>=ecut || skipped_k+kbin[threshold]>=kcut)break;
                skipped_e+=ebin[threshold];skipped_k+=kbin[threshold];
            }
            for(size_t i=0;i<p.pairs.size();++i) {
                const Pair& q=p.pairs[i];
                const double a=norms[q.a],b=norms[q.b];
                const double bound=q.bound*(derivative ? q.derivative_guard : 1.);
                const double energy=cumulative ? pair_energy[i] : bound*a*b;
                const double potential=cumulative ? pair_potential[i] : bound*std::hypot(a,b)*xn;
                int kind=2;
                if(per_pair && energy<ecut && potential<kcut)kind=0;
                else if(cumulative && bucket[i]<threshold)kind=0;
                else if(cumulative && bucket[i]==threshold && skipped_e+energy<ecut && skipped_k+potential<kcut) {
                    kind=0;skipped_e+=energy;skipped_k+=potential;
                } else if(mixed && float_safe && energy<ecut*fp32_multiplier && potential<kcut*fp32_multiplier)kind=1;
                // Choose the faster stable derivative algorithm only AFTER
                // screening/precision eligibility. Never promote skipped work,
                // demote FP64-required work, or alter forward dispatch. Count
                // the precision actually executed, including the FP64 digest.
                if(kind==1 && derivative && q.derivative_fp64_faster)kind=2;
                if(kind==1 && !prepare_float(p,q,ng,d,*sp,derivative,FLOAT_SCALE))kind=2;
                ++local[kind];local[kind+3]+=uint64_t(q.np)*q.nc*ng;
                if(kind==0)continue;
                if(kind==2)prepare(p,q,ng,d,derivative);
                for(int c=0;c<q.nc;++c) {
                    if(kind==1) {evaluate_float(p,q,c,ng,*sp,derivative);digest_float(p,q,c,ns,ng,*sp,derivative);}
                    else {
                        if(derivative)evaluate_derivative(p,q,c,ng,d);else evaluate(p,q,c,ng,0,d);
                        digest(p,q,c,ns,ng,d,derivative);
                    }
                }
            }
            if(mixed) {
                const size_t size=size_t(ns)*p.ncart*ng;
                #pragma omp simd
                for(size_t i=0;i<size;++i)d.g[i]+=FLOAT_UNSCALE*double(sp->g[i]);
                if(derivative) {
                    #pragma omp simd
                    for(size_t i=0;i<3*size;++i)d.dg[i]+=FLOAT_UNSCALE*double(sp->dg[i]);
                }
            }
            from_cart(p,ns,ng,start,total,gv,d.g.data());
            if(derivative)from_cart(p,3*ns,ng,start,total,dgv,d.dg.data());
        }
        #pragma omp critical
        {for(int i=0;i<6;++i)counts[i]+=local[i];}
    }
    #ifdef _OPENMP
    omp_set_dynamic(previous_dynamic);
    #endif
    if(failed){last_error="Unable to allocate mixed SGX thread scratch";return 1;}
    return 0;
}
