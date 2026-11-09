// Copyright 2026 The PySCFAD Authors. Apache License, Version 2.0.
// Portable SGX integrals: FP64 reference and optional adaptive mixed precision.
#include "generated/kernels.h"
#include "generated/derivatives.h"
#include "generated/hrr_derivatives.h"
#include "generated/fp32/kernels.h"
#include "generated/fp32/derivatives.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyscfad_sgx {
constexpr int LMAX = 4;
constexpr int TILE = 128;
constexpr int NBOYS = 2*LMAX+2; // One raised shell for first derivatives.
constexpr int TAYLOR = 8;
constexpr int NTABLE = 64*8+1;
constexpr double PI = 3.141592653589793238462643383279502884;
thread_local std::string last_error;

// F_n(t+d) = sum_k (-d)^k/k! F_{n+k}(t). The nearest table
// node is at most 1/16 away. Nine terms give <4e-18 truncation error.
// Positive long-double series avoid cancellation when constructing the table.
struct BoysTable {
    double f[NTABLE][NBOYS+TAYLOR];
    double e[NTABLE];
    BoysTable() {
        for (int j=0; j<NTABLE; ++j) {
            const long double t=j/8.0L;
            const long double ex=std::exp(-t);
            e[j]=double(ex);
            for (int n=0; n<NBOYS+TAYLOR; ++n) {
                long double term=1.0L/(2*n+1), sum=term;
                for (int k=1; k<1000; ++k) {
                    term *= 2*t/(2*n+2*k+1);
                    sum += term;
                    if (term < sum*1e-22L) break;
                }
                f[j][n]=double(ex*sum);
            }
        }
    }
};

struct BoysScratch {
    // Gather table entries separately from arithmetic. This lets baseline
    // targets without vector gather instructions vectorize the polynomial.
    double coeff[TAYLOR+1][TILE], ex[TILE], t[TILE];
    int idx[TILE];
};

void boys_batch(const BoysTable& tab, int nmax, int ng, const double* x,
                double scale, double* out, BoysScratch& w) {
    static const double invfact[] = {1.,1.,.5,1./6,1./24,1./120,
                                     1./720,1./5040,1./40320};
    // Separate homogeneous tiles keep SIMD loops branch-free in the common case.
    double xmin=x[0], xmax=x[0];
    #pragma omp simd reduction(min:xmin) reduction(max:xmax)
    for (int g=0; g<ng; ++g) { xmin=std::min(xmin,x[g]); xmax=std::max(xmax,x[g]); }
    if (xmin >= 64.) {
        #pragma omp simd
        for (int g=0; g<ng; ++g) {
            w.t[g]=1./x[g];
            out[g]=scale*std::sqrt(PI*w.t[g])*.5;
        }
        for (int n=1; n<=nmax; ++n) {
            #pragma omp simd
            for (int g=0; g<ng; ++g) out[n*ng+g]=out[(n-1)*ng+g]*(n-.5)*w.t[g];
        }
        return;
    }
    if (xmax<64.) {
        #pragma omp simd
        for (int g=0; g<ng; ++g) {
            w.t[g]=x[g];
            w.idx[g]=int(x[g]*8+.5);
        }
    } else {
        // Classify mixed lanes before SIMD arithmetic. Clamp before integer
        // conversion so very large finite arguments cannot overflow an index.
        for (int g=0; g<ng; ++g) {
            w.t[g]=std::min(x[g],64.);
            w.idx[g]=int(w.t[g]*8+.5);
        }
    }
    for (int g=0; g<ng; ++g) {
        const double* f=tab.f[w.idx[g]]+nmax;
        for (int k=0; k<=TAYLOR; ++k) w.coeff[k][g]=f[k];
        w.ex[g]=tab.e[w.idx[g]];
    }
    #pragma omp simd
    for (int g=0; g<ng; ++g) {
        const double d=w.idx[g]/8.-w.t[g];
        double y=w.coeff[TAYLOR][g]*invfact[TAYLOR];
        double z=invfact[TAYLOR];
        for (int k=TAYLOR-1; k>=0; --k) {
            y=y*d+w.coeff[k][g]*invfact[k];
            z=z*d+invfact[k];
        }
        out[nmax*ng+g]=y*scale;
        w.ex[g]=w.ex[g]*z*scale;
    }
    for (int n=nmax; n>0; --n) {
        const double inv=1./(2*n-1);
        #pragma omp simd
        for (int g=0; g<ng; ++g)
            out[(n-1)*ng+g]=(2*w.t[g]*out[n*ng+g]+w.ex[g])*inv;
    }
    if (xmax >= 64.) {
        // Pack the large arguments; SIMD arithmetic stays branch-free without
        // ISA-specific masks or relaxed floating-point exception semantics.
        int count=0;
        for (int g=0; g<ng; ++g) if (x[g]>=64.) {
            w.t[count]=x[g];
            w.idx[count++]=g;
        }
        #pragma omp simd
        for (int g=0; g<count; ++g) {
            w.t[g]=1./w.t[g];
            w.ex[g]=scale*std::sqrt(PI*w.t[g])*.5;
        }
        for (int g=0; g<count; ++g) out[w.idx[g]]=w.ex[g];
        for (int n=1; n<=nmax; ++n) {
            #pragma omp simd
            for (int g=0; g<count; ++g) w.ex[g]*=(n-.5)*w.t[g];
            for (int g=0; g<count; ++g) out[n*ng+w.idx[g]]=w.ex[g];
        }
    }
}

struct Shell {
    int l, np, nc, ncart, nao, off, coff;
    double r[3];
    std::vector<double> exp, coeff, c2s;
};
struct Pair {
    int a,b,np,nc;
    bool same, derivative_fp64_faster;
    double ab[3];
    double bound, derivative_guard;
    std::vector<double> p, ax, ay, az, factor, coeff;
    Kernel vrr, hrr;
    DerivativeKernel derivative;
    HRRDerivativeKernel derivative_hrr;
};
// Conservative crossovers measured with both GCC and Intel LLVM, including
// cc-pVTZ and signed general contractions. Entries are minimum primitive-pair
// counts (zero means VRR); no machine-specific dispatch or accuracy thresholds.
static bool prefer_derivative_hrr(int la,int lb,bool same,int np) {
    static const int distinct[5][5] = {
        {0,0,0,0,0}, {16,64,0,0,0}, {16,16,4,0,0},
        {64,3,4,1,0}, {64,4,1,1,1}};
    static const int coincident[5][5] = {
        {0,0,0,0,0}, {64,3,0,0,0}, {4,3,4,0,0},
        {4,3,4,4,0}, {16,4,4,4,4}};
    const int minimum=same ? coincident[la][lb] : distinct[la][lb];
    return minimum && np>=minimum;
}
// Precision eligibility is a lower bound on accuracy, not a mandate to use
// float. These derivative HRR cases beat FP32 VRR by at least 10% in both
// GCC and Intel LLVM crossover measurements. FP64 VRR had no portable wins
// over FP32 VRR. Keep marginal/compiler-dependent cases on the float path.
static bool prefer_fp64_for_mixed_derivative(int la,int lb,bool same,int np) {
    static const int distinct[5][5] = {
        {0,0,0,0,0}, {0,0,0,0,0}, {0,0,0,0,0},
        {0,0,64,16,0}, {0,64,16,4,4}};
    static const int coincident[5][5] = {
        {0,0,0,0,0}, {0,0,0,0,0}, {0,0,0,0,0},
        {0,0,64,64,0}, {0,0,64,16,16}};
    const int minimum=same ? coincident[la][lb] : distinct[la][lb];
    return minimum && np>=minimum;
}
struct Plan {
    BoysTable table;
    std::vector<Shell> shells;
    std::vector<Pair> pairs;
    int nao=0, ncart=0, maxnp=1, maxpair=1, maxbase=1, maxderivbase=1;
    size_t bytes=sizeof(Plan);
    Plan(int ns, const int* bas, const int* atm, const double* env,
         const double* transforms, const double* bounds, double cutoff,
         bool cart, size_t maxbytes) {
        size_t npairs=0;
        for(int i=0;i<ns;++i)for(int j=0;j<=i;++j)
            if(bounds[i*ns+j]>cutoff)++npairs;
        bytes+=size_t(ns)*sizeof(Shell)+npairs*sizeof(Pair);
        if(bytes>maxbytes)throw std::length_error("SGX primitive-pair cache exceeds its memory allowance");
        shells.reserve(ns);
        pairs.reserve(npairs);
        size_t toff=0;
        for (int i=0; i<ns; ++i) {
            const int* b=bas+8*i;
            Shell s;
            s.l=b[1]; s.np=b[2]; s.nc=b[3];
            if (s.l<0 || s.l>LMAX || s.np<1 || s.nc<1)
                throw std::invalid_argument("Generated SGX supports s through g shells");
            s.ncart=(s.l+1)*(s.l+2)/2;
            s.nao=cart ? s.ncart : 2*s.l+1;
            s.off=nao; s.coff=ncart;
            nao+=s.nc*s.nao; ncart+=s.nc*s.ncart;
            std::copy(env+atm[b[0]*6+1],env+atm[b[0]*6+1]+3,s.r);
            s.exp.assign(env+b[5],env+b[5]+s.np);
            s.coeff.assign(env+b[6],env+b[6]+s.np*s.nc);
            s.c2s.assign(transforms+toff,transforms+toff+s.ncart*s.nao);
            toff+=s.ncart*s.nao;
            bytes+=sizeof(double)*(s.exp.size()+s.coeff.size()+s.c2s.size());
            if(bytes>maxbytes)throw std::length_error("SGX primitive-pair cache exceeds its memory allowance");
            shells.push_back(std::move(s));
        }
        // Canonical angular order lets 15 generated pair types cover all 25.
        for (int i=0; i<ns; ++i) for (int j=0; j<=i; ++j) {
            if (bounds[i*ns+j]<=cutoff) continue;
            Pair q;
            q.bound=bounds[i*ns+j];
            q.a=i; q.b=j;
            if (shells[i].l<shells[j].l) std::swap(q.a,q.b);
            const Shell& a=shells[q.a]; const Shell& b=shells[q.b];
            // Raised/lowered derivative combinations need a stricter local
            // criterion than values. This scale is a conservative heuristic,
            // not a certified bound on the final molecular force error.
            q.derivative_guard=std::max(1.,4*(a.l+b.l+2)*std::sqrt(
                std::max(*std::max_element(a.exp.begin(),a.exp.end()),
                         *std::max_element(b.exp.begin(),b.exp.end()))));
            q.np=a.np*b.np; q.nc=a.nc*b.nc;
            double r2=0;
            for (int k=0;k<3;++k) {q.ab[k]=a.r[k]-b.r[k]; r2+=q.ab[k]*q.ab[k];}
            q.same=(r2==0.);
            q.vrr=select_kernel(a.l,b.l,q.same,false);
            q.hrr=select_kernel(a.l,b.l,q.same,true);
            q.derivative=select_derivative_kernel(a.l,b.l,q.same);
            q.derivative_hrr=prefer_derivative_hrr(a.l,b.l,q.same,q.np)
                ? select_hrr_derivative(a.l,b.l,q.same) : nullptr;
            q.derivative_fp64_faster=q.derivative_hrr &&
                prefer_fp64_for_mixed_derivative(a.l,b.l,q.same,q.np);
            bytes+=sizeof(double)*q.np*(5+q.nc);
            if (bytes>maxbytes) throw std::length_error("SGX primitive-pair cache exceeds its memory allowance");
            q.p.resize(q.np); q.ax.resize(q.np); q.ay.resize(q.np); q.az.resize(q.np);
            q.factor.resize(q.np); q.coeff.resize(q.np*q.nc);
            const auto common=[](int l) {return l==0 ? .28209479177387814347 :
                                                l==1 ? .48860251190291992159 : 1.;};
            for (int ia=0;ia<a.np;++ia) for (int ib=0;ib<b.np;++ib) {
                const int p=ia*b.np+ib;
                const double alpha=a.exp[ia], beta=b.exp[ib], z=alpha+beta;
                q.p[p]=z;
                q.ax[p]=-q.ab[0]*beta/z; q.ay[p]=-q.ab[1]*beta/z; q.az[p]=-q.ab[2]*beta/z;
                q.factor[p]=(2*PI/z)*std::exp(-alpha*beta/z*r2)*common(a.l)*common(b.l);
                for (int ca=0;ca<a.nc;++ca) for (int cb=0;cb<b.nc;++cb)
                    q.coeff[(ca*b.nc+cb)*q.np+p]=a.coeff[ca*a.np+ia]*b.coeff[cb*b.np+ib];
            }
            maxnp=std::max(maxnp,q.np);
            maxpair=std::max(maxpair,a.ncart*b.ncart);
            maxbase=std::max(maxbase,base_size(a.l,b.l,q.same));
            if(q.derivative_hrr)maxderivbase=std::max(maxderivbase,
                hrr_derivative_base_size(a.l,b.l,q.same));
            pairs.push_back(std::move(q));
        }
    }
    size_t scratch_bytes(int nset,bool derivative=false) const {
        return sizeof(BoysScratch)+sizeof(double)*TILE*(size_t(derivative ? 5 : 2)*nset*ncart
                                   +size_t(maxnp)*(2*NBOYS+3)
                                   +(derivative ? 8 : 1)*maxpair+(derivative ? std::max(maxbase,maxderivbase) : maxbase)+4);
    }
};
struct Scratch {
    BoysScratch boys_work;
    std::vector<double> boys, scaled, gx,gy,gz,ints,base,f,g,deriv,dg;
    double xyz[3*TILE],x[TILE];
    Scratch(const Plan& p,int ns,bool derivative=false): boys(size_t(p.maxnp)*NBOYS*TILE),scaled(boys.size()),
        gx(size_t(p.maxnp)*TILE),gy(gx.size()),gz(gx.size()),
        ints(size_t(p.maxpair)*TILE),base(size_t(derivative ? std::max(p.maxbase,p.maxderivbase) : p.maxbase)*TILE),
        f(size_t(ns)*p.ncart*TILE),g(f.size()),
        deriv(derivative ? size_t(7)*p.maxpair*TILE : 0),
        dg(derivative ? size_t(3)*ns*p.ncart*TILE : 0) {}
};

void prepare(const Plan& plan,const Pair& p,int ng,Scratch& w,bool derivative=false) {
    const Shell& a=plan.shells[p.a]; const Shell& b=plan.shells[p.b];
    const int nm=a.l+b.l+int(derivative);
    for (int k=0;k<p.np;++k) {
        double* gx=w.gx.data()+k*ng; double* gy=w.gy.data()+k*ng; double* gz=w.gz.data()+k*ng;
        const double px=a.r[0]+p.ax[k],py=a.r[1]+p.ay[k],pz=a.r[2]+p.az[k];
        #pragma omp simd
        for (int g=0;g<ng;++g) {
            gx[g]=px-w.xyz[g]; gy[g]=py-w.xyz[TILE+g]; gz[g]=pz-w.xyz[2*TILE+g];
            w.x[g]=p.p[k]*(gx[g]*gx[g]+gy[g]*gy[g]+gz[g]*gz[g]);
        }
        const double fac=p.factor[k]*(p.nc==1 ? p.coeff[k] : 1.);
        boys_batch(plan.table,nm,ng,w.x,fac,w.boys.data()+size_t(k)*(nm+1)*ng,w.boys_work);
    }
}

const double* contract_boys(const Pair& p,int cp,int ng,int nm,Scratch& w) {
    const double* boys=w.boys.data();
    if (p.nc>1) {
        // Reuse Boys/geometry across all general-contraction coefficient columns.
        for (int k=0;k<p.np;++k) {
            const int off=k*(nm+1)*ng;
            const double c=p.coeff[cp*p.np+k];
            #pragma omp simd
            for (int t=0;t<(nm+1)*ng;++t) w.scaled[off+t]=c*w.boys[off+t];
        }
        boys=w.scaled.data();
    }
    return boys;
}

void evaluate(const Plan& plan,const Pair& p,int cp,int ng,int algorithm,Scratch& w) {
    const int nm=plan.shells[p.a].l+plan.shells[p.b].l;
    const double* boys=contract_boys(p,cp,ng,nm,w);
    // Crossover checked with GCC and Intel LLVM. Same-center single primitives
    // favor specialized VRR; distinct high-l centers benefit from HRR even
    // without contraction. Contracted d-p pairs also amortize the HRR step.
    const bool prefer_hrr=(p.np>1 && nm>3) ||
        (!p.same && (nm>4 || (nm==3 && p.np>=4)));
    const bool hrr=p.hrr && (algorithm==2 || (algorithm==0 && prefer_hrr));
    Kernel fn=hrr ? p.hrr : p.vrr;
    fn(w.ints.data(),w.base.data(),boys,p.p.data(),w.gx.data(),w.gy.data(),w.gz.data(),
       p.ax.data(),p.ay.data(),p.az.data(),p.ab[0],p.ab[1],p.ab[2],p.np,ng);
}

void evaluate_derivative(const Plan& plan,const Pair& p,int cp,int ng,Scratch& w) {
    const Shell& a=plan.shells[p.a];const Shell& b=plan.shells[p.b];
    const double* boys=contract_boys(p,cp,ng,a.l+b.l+1,w);
    if(p.derivative_hrr) {
        p.derivative_hrr(w.deriv.data(),w.base.data(),boys,p.p.data(),
            w.gx.data(),w.gy.data(),w.gz.data(),p.ax.data(),p.ay.data(),p.az.data(),
            p.ab[0],p.ab[1],p.ab[2],a.exp.data(),b.exp.data(),b.np,p.np,ng);
        return;
    }
    p.derivative(w.deriv.data(),boys,p.p.data(),w.gx.data(),w.gy.data(),w.gz.data(),
        p.ax.data(),p.ay.data(),p.az.data(),p.ab[0],p.ab[1],p.ab[2],
        a.exp.data(),b.exp.data(),b.np,p.np,ng);
}

void to_cart(const Plan& p,int ns,int ng,int start,int total,const double* in,Scratch& w) {
    std::fill(w.f.begin(),w.f.begin()+size_t(ns)*p.ncart*ng,0.);
    std::fill(w.g.begin(),w.g.begin()+size_t(ns)*p.ncart*ng,0.);
    for (int s=0;s<ns;++s) for (const Shell& sh:p.shells) for(int c=0;c<sh.nc;++c) {
        for (int u=0;u<sh.ncart;++u) {
            double* dst=w.f.data()+(size_t(s)*p.ncart+sh.coff+c*sh.ncart+u)*ng;
            for (int v=0;v<sh.nao;++v) {
                const double t=sh.c2s[u*sh.nao+v]; if(t==0.)continue;
                const double* src=in+(size_t(s)*p.nao+sh.off+c*sh.nao+v)*total+start;
                #pragma omp simd
                for (int g=0;g<ng;++g) dst[g]+=t*src[g];
            }
        }
    }
}
void from_cart(const Plan& p,int ns,int ng,int start,int total,double* out,const double* gv) {
    for (int s=0;s<ns;++s) for(const Shell& sh:p.shells)for(int c=0;c<sh.nc;++c) {
        for(int v=0;v<sh.nao;++v) {
            double* dst=out+(size_t(s)*p.nao+sh.off+c*sh.nao+v)*total+start;
            std::fill(dst,dst+ng,0.);
            for(int u=0;u<sh.ncart;++u) {
                const double t=sh.c2s[u*sh.nao+v];if(t==0.)continue;
                const double* src=gv+(size_t(s)*p.ncart+sh.coff+c*sh.ncart+u)*ng;
                #pragma omp simd
                for(int g=0;g<ng;++g)dst[g]+=t*src[g];
            }
        }
    }
}
template<class T,bool derivative,class Work>
void digest_values(const Plan& p,const Pair& q,int cp,int ns,int ng,Work& w) {
    const Shell& a=p.shells[q.a];const Shell& b=p.shells[q.b];
    const int ca=cp/b.nc,cb=cp%b.nc;
    for(int s=0;s<ns;++s)for(int i=0;i<a.ncart;++i)for(int j=0;j<b.ncart;++j) {
        const size_t u=size_t(s)*p.ncart+a.coff+ca*a.ncart+i;
        const size_t v=size_t(s)*p.ncart+b.coff+cb*b.ncart+j;
        const T* ints=(derivative ? w.deriv.data() : w.ints.data())+(i*b.ncart+j)*ng;
        const T* fj=w.f.data()+v*ng;T* gi=w.g.data()+u*ng;
        const size_t step=size_t(a.ncart)*b.ncart*ng;
        const size_t ds=size_t(p.ncart)*ng;
        T *ga0=nullptr,*ga1=nullptr,*ga2=nullptr;
        if(derivative) {
            ga0=w.dg.data()+(size_t(s*3)*p.ncart+a.coff+ca*a.ncart+i)*ng;
            ga1=ga0+ds;ga2=ga1+ds;
        }
        if(q.a!=q.b) {
            const T* fi=w.f.data()+u*ng;T* gj=w.g.data()+v*ng;
            if(derivative) {
                T* gb0=w.dg.data()+(size_t(s*3)*p.ncart+b.coff+cb*b.ncart+j)*ng;
                T* gb1=gb0+ds;T* gb2=gb1+ds;
                #pragma omp simd
                for(int g=0;g<ng;++g) {
                    const T f_i=fi[g],f_j=fj[g],iv=ints[g];
                    gi[g]+=iv*f_j;gj[g]+=iv*f_i;
                    ga0[g]+=ints[step+g]*f_j;ga1[g]+=ints[2*step+g]*f_j;ga2[g]+=ints[3*step+g]*f_j;
                    gb0[g]+=ints[4*step+g]*f_i;gb1[g]+=ints[5*step+g]*f_i;gb2[g]+=ints[6*step+g]*f_i;
                }
            } else {
                #pragma omp simd
                for(int g=0;g<ng;++g) {const T iv=ints[g];gi[g]+=iv*fj[g];gj[g]+=iv*fi[g];}
            }
        } else if(derivative) {
            #pragma omp simd
            for(int g=0;g<ng;++g) {
                const T f_j=fj[g];gi[g]+=ints[g]*f_j;
                ga0[g]+=ints[step+g]*f_j;ga1[g]+=ints[2*step+g]*f_j;ga2[g]+=ints[3*step+g]*f_j;
            }
        } else {
            #pragma omp simd
            for(int g=0;g<ng;++g)gi[g]+=ints[g]*fj[g];
        }
    }
}
void digest(const Plan& p,const Pair& q,int cp,int ns,int ng,Scratch& w,bool derivative=false) {
    if(derivative)digest_values<double,true>(p,q,cp,ns,ng,w);
    else digest_values<double,false>(p,q,cp,ns,ng,w);
}
#include "mixed.h"
} // namespace pyscfad_sgx

extern "C" {
#include "contraction.h"
const char* SGX_generated_error() {return pyscfad_sgx::last_error.c_str();}
int SGX_generated_version() {return 4;}
void* SGX_generated_new(int ns,const int* bas,const int* atm,const double* env,
        const double* c2s,const double* bounds,double tol,int cart,size_t limit) {
    try {
        auto* plan=new pyscfad_sgx::Plan(ns,bas,atm,env,c2s,bounds,tol,cart,limit);
        pyscfad_sgx::last_error.clear();return plan;
    }catch(const std::exception& e){pyscfad_sgx::last_error=e.what();return nullptr;}
}
void SGX_generated_delete(void* p) {delete static_cast<pyscfad_sgx::Plan*>(p);}
size_t SGX_generated_bytes(void* p) {return static_cast<pyscfad_sgx::Plan*>(p)->bytes;}
size_t SGX_generated_scratch(void* p,int ns) {return static_cast<pyscfad_sgx::Plan*>(p)->scratch_bytes(ns);}
size_t SGX_generated_scratch_deriv(void* p,int ns) {return static_cast<pyscfad_sgx::Plan*>(p)->scratch_bytes(ns,true);}
size_t SGX_generated_scratch_mixed(void* ptr,int ns,int derivative) {
    const auto& p=*static_cast<pyscfad_sgx::Plan*>(ptr);
    return p.scratch_bytes(ns,derivative)+p.scratch_bytes(ns,derivative)/2
        +sizeof(pyscfad_sgx::FloatScratch)+sizeof(double)*p.shells.size()
        +(2*sizeof(double)+sizeof(uint8_t))*p.pairs.size()+128*sizeof(double)
        +sizeof(float)*6*p.maxnp;
}
int SGX_generated_contract_adaptive(void* ptr,int ns,int total,const double* coords,const double* fg,
        const double* weights,const double* ao_l1,double* gv,double* dgv,
        double etol,double ktol,double fp32_multiplier,int screen,int threads,uint64_t* counts) {
    return pyscfad_sgx::adaptive_contract(*static_cast<pyscfad_sgx::Plan*>(ptr),ns,total,coords,fg,
        weights,ao_l1,gv,dgv,etol,ktol,fp32_multiplier,screen,threads,counts);
}
static int contract_impl(void* ptr,int ns,int total,const double* coords,
                         const double* fg,double* gv,double* dgv,int algorithm,int threads) {
    using namespace pyscfad_sgx;
    const Plan& p=*static_cast<Plan*>(ptr);
    const bool derivative=dgv!=nullptr;
    std::atomic<bool> failed(false);
    // This runtime may have been loaded after Python's fixed-thread scope.
    #ifdef _OPENMP
    const int previous_dynamic=omp_get_dynamic();
    omp_set_dynamic(0);
    #endif
    #pragma omp parallel num_threads(threads)
    {
        std::unique_ptr<Scratch> w;
        try{w.reset(new Scratch(p,ns,derivative));}catch(...){failed=true;}
        #pragma omp for schedule(dynamic,1)
        for(int start=0;start<total;start+=TILE) {
            if(!w)continue;
            const int ng=std::min(TILE,total-start);
            for(int g=0;g<ng;++g)for(int k=0;k<3;++k)w->xyz[k*TILE+g]=coords[3*(start+g)+k];
            to_cart(p,ns,ng,start,total,fg,*w);
            if(derivative)std::fill(w->dg.begin(),w->dg.begin()+size_t(3)*ns*p.ncart*ng,0.);
            for(const Pair& q:p.pairs) {
                prepare(p,q,ng,*w,derivative);
                for(int c=0;c<q.nc;++c) {
                    if(derivative)evaluate_derivative(p,q,c,ng,*w);
                    else evaluate(p,q,c,ng,algorithm,*w);
                    digest(p,q,c,ns,ng,*w,derivative);
                }
            }
            from_cart(p,ns,ng,start,total,gv,w->g.data());
            if(derivative)from_cart(p,3*ns,ng,start,total,dgv,w->dg.data());
        }
    }
    #ifdef _OPENMP
    omp_set_dynamic(previous_dynamic);
    #endif
    if(failed){last_error="Unable to allocate SGX thread scratch";return 1;}
    return 0;
}

int SGX_generated_contract(void* ptr,int ns,int total,const double* coords,
                           const double* fg,double* gv,int algorithm,int threads) {
    return contract_impl(ptr,ns,total,coords,fg,gv,nullptr,algorithm,threads);
}
int SGX_generated_contract_deriv(void* ptr,int ns,int total,const double* coords,
                                 const double* fg,double* gv,double* dgv,int threads) {
    return contract_impl(ptr,ns,total,coords,fg,gv,dgv,0,threads);
}

// Shell-pair Cartesian interface for validation against an independent engine.
static int pair_impl(void* ptr,int ia,int ib,int total,const double* coords,
                     double* out,int algorithm,bool derivative,bool single=false) {
    using namespace pyscfad_sgx;
    const Plan& p=*static_cast<Plan*>(ptr);
    try {
        const Pair* pair=nullptr;
        for(const Pair& q:p.pairs)if((q.a==ia&&q.b==ib)||(q.a==ib&&q.b==ia)){pair=&q;break;}
        if(!pair)throw std::invalid_argument("Shell pair is screened out");
        const Pair& q=*pair; const Shell& a=p.shells[q.a]; const Shell& b=p.shells[q.b];
        const bool swap=q.a!=ia;
        const int noutb=swap?a.ncart*a.nc:b.ncart*b.nc;
        const size_t nout=size_t(a.ncart)*a.nc*b.ncart*b.nc;
        Scratch w(p,0,derivative);
        std::unique_ptr<FloatScratch> f;
        if(single)f.reset(new FloatScratch(p,0,derivative));
        for(int start=0;start<total;start+=TILE) {
            const int ng=std::min(TILE,total-start);
            for(int g=0;g<ng;++g)for(int k=0;k<3;++k)w.xyz[k*TILE+g]=coords[3*(start+g)+k];
            if(single) {
                if(!prepare_float(p,q,ng,w,*f,derivative))throw std::runtime_error("Unsafe FP32 integral parameters");
            } else prepare(p,q,ng,w,derivative);
            for(int cp=0;cp<q.nc;++cp) {
                if(single)evaluate_float(p,q,cp,ng,*f,derivative);
                else if(derivative)evaluate_derivative(p,q,cp,ng,w);
                else evaluate(p,q,cp,ng,algorithm,w);
                const int ca=cp/b.nc, cb=cp%b.nc;
                for(int i=0;i<a.ncart;++i)for(int j=0;j<b.ncart;++j) {
                    int u=ca*a.ncart+i,v=cb*b.ncart+j;
                    if(swap)std::swap(u,v);
                    for(int c=0;c<(derivative ? 7 : 1);++c) {
                        const int sc=(swap&&c>0) ? (c<=3 ? c+3 : c-3) : c;
                        const size_t offset=(size_t(sc)*a.ncart*b.ncart+i*b.ncart+j)*ng;
                        double* dst=out+(c*nout+size_t(u)*noutb+v)*total+start;
                        if(single) {
                            const float* src=(derivative ? f->deriv.data() : f->ints.data())+offset;
                            std::copy(src,src+ng,dst);
                        } else {
                            const double* src=(derivative ? w.deriv.data() : w.ints.data())+offset;
                            std::copy(src,src+ng,dst);
                        }
                    }
                }
            }
        }
        return 0;
    }catch(const std::exception& e){last_error=e.what();return 1;}
}
int SGX_generated_pair(void* ptr,int ia,int ib,int total,const double* coords,
                       double* out,int algorithm) {
    return pair_impl(ptr,ia,ib,total,coords,out,algorithm,false);
}
int SGX_generated_pair_deriv(void* ptr,int ia,int ib,int total,const double* coords,double* out) {
    return pair_impl(ptr,ia,ib,total,coords,out,0,true);
}
int SGX_generated_pair_float(void* ptr,int ia,int ib,int total,const double* coords,double* out,int derivative) {
    return pair_impl(ptr,ia,ib,total,coords,out,0,derivative,true);
}
int SGX_generated_boys_float(int n,int ng,const double* x,float* out) {
    using namespace pyscfad_sgx;
    if(n<0 || n>=NBOYS || ng<0 || ng>TILE)return 1;
    for(int i=0;i<ng;++i)if(!std::isfinite(x[i])||x[i]<0)return 1;
    if(ng) {
        // A minimal scratch description; this validation entry point does not
        // allocate any shell-pair or AO buffers.
        Plan p(0,nullptr,nullptr,nullptr,nullptr,nullptr,0.,false,1000000);
        FloatScratch w(p,0,false);boys_batch_float(p.table,n,ng,x,1.,out,w);
    }
    return 0;
}
int SGX_generated_boys(int n,int ng,const double* x,double* out) {
    using namespace pyscfad_sgx;
    if(n<0 || n>=NBOYS || ng<0 || ng>TILE)return 1;
    for(int i=0;i<ng;++i)if(!std::isfinite(x[i])||x[i]<0)return 1;
    if(ng) {static const BoysTable table;BoysScratch w;boys_batch(table,n,ng,x,1.,out,w);}
    return 0;
}
}
