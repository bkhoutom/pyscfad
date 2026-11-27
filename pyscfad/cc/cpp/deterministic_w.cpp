// Copyright 2026 The PySCFAD Authors (Apache-2.0).
// Deterministic selected MP2 W. Column-major views preserve the existing
// contraction layouts; GEMMs share SciPy's function pointer with RCCSD.
#include "tensor.h"
#include <pybind11/stl.h>
#include <cmath>

namespace ccsd_native { namespace dw {
struct M {
    T storage;
    Index rows=0,cols=0,ld=0;
    double* p=nullptr;
    M()=default;
    M(Index r,Index c,bool zero=false):storage(std::vector<Index>{c,r}),rows(r),cols(c),ld(r),p(storage.p) {
        if(zero) clear();
    }
    M view(Index r,Index c,Index first=0,Index leading=0) const {
        M x=*this;x.rows=r;x.cols=c;x.p=p+first;x.ld=leading?leading:ld;return x;
    }
    M columns(Index first,Index count) const {return view(rows,count,first*ld);}
    M row_panel(Index first,Index count) const {return view(count,cols,first);}
    double& operator()(Index i,Index j) const {return p[i+j*ld];}
    void clear() {
        #pragma omp parallel for if(!omp_in_parallel() && rows*cols>65536) schedule(static)
        for(Index j=0;j<cols;++j)std::fill_n(p+j*ld,rows,0.);
    }
};
int bi(Index n) {
    if(n<0 || n>INT_MAX)throw std::overflow_error("W matrix exceeds SciPy BLAS integer range");
    return int(n);
}
void product(const M& a,const M& b,M c,char ta='N',char tb='N',double alpha=1.,double beta=0.) {
    const int m=bi(c.rows),n=bi(c.cols),k=bi(ta=='N'?a.cols:a.rows);
    if(c.rows!=(ta=='N'?a.rows:a.cols) || c.cols!=(tb=='N'?b.cols:b.rows)
       || k!=(tb=='N'?b.rows:b.cols))throw std::invalid_argument("W GEMM shape mismatch");
    if(!m || !n)return;
    if(!k) {
        for(Index j=0;j<c.cols;++j)for(Index i=0;i<c.rows;++i)c(i,j)=beta==0.?0.:beta*c(i,j);
        return;
    }
    int lda=bi(a.ld),ldb=bi(b.ld),ldc=bi(c.ld),mm=m,nn=n,kk=k;
    if(omp_in_parallel())blas(&ta,&tb,&mm,&nn,&kk,&alpha,a.p,&lda,b.p,&ldb,&beta,c.p,&ldc);
    else parallel_gemm(ta,tb,m,n,k,a.p,lda,b.p,ldb,c.p,ldc,alpha,beta);
}
M multiply(const M& a,const M& b,char ta='N',char tb='N') {
    M c(ta=='N'?a.rows:a.cols,tb=='N'?b.cols:b.rows);product(a,b,c,ta,tb);return c;
}
void add(M a,const M& b,double scale=1.) {
    if(a.rows!=b.rows || a.cols!=b.cols)throw std::invalid_argument("W add shape mismatch");
    #pragma omp parallel for if(!omp_in_parallel() && a.rows*a.cols>65536) schedule(static)
    for(Index j=0;j<a.cols;++j)for(Index i=0;i<a.rows;++i)a(i,j)+=scale*b(i,j);
}
void copy(M to,const M& from) {
    for(Index j=0;j<to.cols;++j)std::copy_n(from.p+j*from.ld,to.rows,to.p+j*to.ld);
}
void symmetrize(M x) {
    #pragma omp parallel for if(!omp_in_parallel() && x.rows*x.cols>65536) schedule(static)
    for(Index j=0;j<x.cols;++j)for(Index i=0;i<j;++i)x(i,j)=x(j,i)=.5*(x(i,j)+x(j,i));
}
M spin(const M& a) {
    M b(a.rows,a.cols);
    for(Index j=0;j<a.cols;++j)for(Index i=0;i<a.rows;++i)b(i,j)=2*a(i,j)-a(j,i);
    return b;
}
M from_c(const double* a,Index r,Index c) {
    M out(r,c);
    // Tiled transpose into auxiliary-contiguous storage, one owned DF copy.
    #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
    for(Index j=0;j<c;j+=64)for(Index i=0;i<r;i+=64)
        for(Index y=j;y<std::min(j+64,c);++y)for(Index x=i;x<std::min(i+64,r);++x)
            out(x,y)=a[x*c+y];
    return out;
}
Array array(const M& a) {
    Array out({a.rows,a.cols});double* p=out.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for collapse(2) schedule(static)
        for(Index j=0;j<a.cols;j+=64)for(Index i=0;i<a.rows;i+=64)
            for(Index y=j;y<std::min(j+64,a.cols);++y)for(Index x=i;x<std::min(i+64,a.rows);++x)
                p[x*a.cols+y]=a(x,y);
    }
    return out;
}
M frame(const M& x,const M& u,bool inward) {
    M result=inward?multiply(multiply(u,x,'T'),u):multiply(multiply(u,x),u,'N','T');
    symmetrize(result);return result;
}
M divided(const double* eigen,Index n,double coefficient,double shift,const M& h) {
    M out(n,n);
    #pragma omp parallel for if(!omp_in_parallel() && n*n>65536) schedule(static)
    for(Index j=0;j<n;++j)for(Index i=0;i<n;++i) {
        const double x=coefficient*(eigen[i]-shift),y=coefficient*(eigen[j]-shift),gap=std::abs(x-y);
        out(i,j)=coefficient*std::exp(std::max(x,y))*(gap==0?1.:-std::expm1(-gap)/gap)*.5*(h(i,j)+h(j,i));
    }
    return out;
}
struct Propagator {
    Index no,nv,np;
    M b,uo,uv;
    const double *eo,*ev,*roots;
    double shift;
    std::vector<M> occupied;
    std::vector<std::vector<double>> scales;
    Propagator(const Array& ba,const Array& uoa,const Array& uva,const Array& eoa,const Array& eva,const Array& beta):
        no(ba.shape(1)),nv(ba.shape(2)),np(ba.shape(0)),b(from_c(ba.data(),np,no*nv)),
        uo(from_c(uoa.data(),no,no)),uv(from_c(uva.data(),nv,nv)),eo(eoa.data()),ev(eva.data()),roots(beta.data()),
        shift(.5*(eo[no-1]+ev[0])) {
        rotate(b,b,true);
        for(Index l=0;l<beta.size();++l) {
            M scaled(no,no);
            for(Index j=0;j<no;++j)for(Index i=0;i<no;++i)scaled(i,j)=uo(i,j)*std::exp(roots[l]*(eo[j]-shift));
            occupied.push_back(multiply(scaled,uo,'N','T'));
            scales.emplace_back(nv);
            for(Index a=0;a<nv;++a)scales.back()[a]=std::exp(-roots[l]*(ev[a]-shift));
        }
    }
    void rotate(const M& input,M output,bool inward) const {
        const Index panel=std::max<Index>(1,8*1024*1024/(8*nv)),count=(np+panel-1)/panel;
        #pragma omp parallel for schedule(static)
        for(Index task=0;task<no*count;++task) {
            const Index first=(task%count)*panel,n=std::min(np-first,panel),i=task/count;
            M temp(n,nv);product(input.columns(i*nv,nv).row_panel(first,n),uv,temp,'N',inward?'N':'T');
            copy(output.columns(i*nv,nv).row_panel(first,n),temp);
        }
    }
    void apply_occupied(Index point,M c) const {
        const M input=b.view(np*nv,no,0,np*nv),out=c.view(np*nv,no,0,np*nv);
        const Index panel=std::max<Index>(1,2*1024*1024/(8*no));
        #pragma omp parallel for schedule(static)
        for(Index first=0;first<input.rows;first+=panel) {
            const Index n=std::min(panel,input.rows-first);
            product(input.row_panel(first,n),occupied[point],out.row_panel(first,n));
        }
    }
    void apply(Index point,M c) const {
        apply_occupied(point,c);
        #pragma omp parallel for schedule(static)
        for(Index j=0;j<no*nv;++j) {
            const double factor=scales[point][j%nv];
            #pragma omp simd
            for(Index p=0;p<np;++p)c(p,j)*=factor;
        }
    }
    void reverse(Index point,M scratch,const M& gamma,M fo,M fv,M bb) const {
        apply_occupied(point,scratch);
        M hv(nv,nv,true);
        #pragma omp parallel
        {
            M local(nv,nv,true);
            #pragma omp for schedule(static)
            for(Index i=0;i<no;++i) {
                M d=scratch.columns(i*nv,nv),g=gamma.columns(i*nv,nv);
                product(d,g,local,'T','N',1.,1.);
                for(Index a=0;a<nv;++a) {
                    const double scale=scales[point][a];
                    #pragma omp simd
                    for(Index p=0;p<np;++p)d(p,a)=g(p,a)*scale;
                }
            }
            #pragma omp critical(dw_virtual_gram)
            add(hv,local);
        }
        M ho(no,no,true);
        const M input=b.view(np*nv,no,0,np*nv),g=scratch.view(np*nv,no,0,np*nv),out=bb.view(np*nv,no,0,np*nv);
        const Index panel=std::max<Index>(1,2*1024*1024/(8*no));
        #pragma omp parallel
        {
            M local(no,no,true);
            #pragma omp for schedule(static)
            for(Index first=0;first<g.rows;first+=panel) {
                const Index n=std::min(panel,g.rows-first);
                product(input.row_panel(first,n),g.row_panel(first,n),local,'T','N',1.,1.);
                product(g.row_panel(first,n),occupied[point],out.row_panel(first,n),'N','N',1.,1.);
            }
            #pragma omp critical(dw_occupied_gram)
            add(ho,local);
        }
        add(fo,frame(divided(eo,no,roots[point],shift,frame(ho,uo,true)),uo,false));
        add(fv,divided(ev,nv,-roots[point],shift,hv));
    }
};
struct Sequential {
    std::vector<M> amplitudes,grams;
    M rotation;
    Index no=0,nv=0;
    bool consumed=false;
};
struct Result {
    std::vector<M> wo,wv;
    std::vector<double> energies;
    M fo,fv,b;
    std::map<std::string,double> times;
    std::shared_ptr<Sequential> sequential;
};

Result evaluate(const Array& ba,const Array& uoa,const Array& uva,const Array& eoa,const Array& eva,
        const Array& beta,const Array& weights,const std::vector<Index>& targets,
        const Array& eb,const Array& ob,const Array& vb,bool per_target,bool sequential,bool density,bool energy_only,
        const Array& rotation,const py::object& rotation_pullback) {
    const double start=omp_get_wtime();Result out;
    for(const auto* key:{"propagation_seconds","amplitude_seconds","contraction_seconds","seed_seconds",
            "reverse_seconds","reverse_factor_seconds","amplitude_vjp_seconds","propagator_vjp_seconds",
            "propagation_allocation_seconds","propagation_occupied_seconds","propagation_virtual_seconds",
            "spin_seconds","occupied_density_seconds","bare_gram_worker_seconds","occupied_w_worker_seconds",
            "virtual_w_worker_seconds","virtual_energy_seconds"})out.times[key]=0.;
    const Index no=ba.shape(1),nv=ba.shape(2),np=ba.shape(0),vv=nv*nv,nt=targets.size();
    const bool reverse=eb.size()!=0;
    const bool sequential_reverse=reverse && rotation.ndim()==2;
    out.energies.resize(per_target?nt:1,0.);
    Propagator dress(ba,uoa,uva,eoa,eva,beta);
    out.times["eigensystem_layout_seconds"]=omp_get_wtime()-start;
    std::vector<M> amplitudes,bar_amplitudes;
    for(Index s=0;s<nt;++s)amplitudes.emplace_back(vv,no,true);
    M c(np,no*nv);
    for(Index l=0;l<beta.size();++l) {
        double tick=omp_get_wtime();dress.apply(l,c);out.times["propagation_seconds"]+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for(Index task=0;task<nt*no;++task) {
            const Index s=task/no,j=task%no;
            product(c.columns(targets[s]*nv,nv),c.columns(j*nv,nv),amplitudes[s].view(nv,nv,j*vv,nv),
                    'T','N',-weights.data()[l],1.);
        }
        out.times["amplitude_seconds"]+=omp_get_wtime()-tick;
    }
    if(reverse) {
        out.fo=M(no,no,true);out.fv=M(nv,nv,true);out.b=M(np,no*nv,true);
        for(Index s=0;s<nt;++s)bar_amplitudes.emplace_back(vv,no,true);
    } else {
        if(!energy_only)for(Index s=0;s<(per_target?nt:1);++s) {out.wo.emplace_back(no,no,true);out.wv.emplace_back(nv,nv,true);}
        if(sequential) {
            out.sequential=std::make_shared<Sequential>();out.sequential->no=no;out.sequential->nv=nv;
            out.sequential->rotation=dress.uv;
        }
    }
    // Delayed sequential reverse: build each Gram once (spin(T) for D), project both
    // factors, and accumulate the occupied-rotation cotangent before applying
    // the occupied-W seed. Only this small selection VJP crosses into Python;
    // all amplitude/DF work remains in the threaded native pass.
    std::vector<M> sequential_grams,sequential_bargrams;
    M extra_occupied,sequential_rotation;
    if(sequential_reverse) {
        const double tick=omp_get_wtime();
        const Index nr=rotation.shape(1);
        sequential_rotation=from_c(rotation.data(),no,nr);
        const M& u=sequential_rotation;M bu(no,nr,true);
        for(Index s=0;s<nt;++s) {
            M grams(vv,no);
            const M bi=dress.b.columns(targets[s]*nv,nv);
            #pragma omp parallel for schedule(static)
            for(Index j=0;j<no;++j)
                copy(grams.view(nv,nv,j*vv,nv),spin(density?amplitudes[s].view(nv,nv,j*vv,nv):
                    multiply(bi,dress.b.columns(j*nv,nv),'T')));
            M t=multiply(amplitudes[s],u),g=multiply(grams,u);
            M bt(vv,nr),bgr(vv,nr);
            M seed=frame(from_c(vb.data()+s*vv,nv,nv),dress.uv,true);
            #pragma omp parallel for schedule(static)
            for(Index j=0;j<nr;++j) {
                const M tj=t.view(nv,nv,j*vv,nv),gj=g.view(nv,nv,j*vv,nv);
                M dt=bt.view(nv,nv,j*vv,nv),dg=bgr.view(nv,nv,j*vv,nv);
                product(seed,gj,dt,'N','N',.5);product(gj,seed,dt,'N','N',.5,1.);
                product(seed,tj,dg,'N','N',.5);product(tj,seed,dg,'N','N',.5,1.);
            }
            product(amplitudes[s],bt,bu,'T','N',1.,1.);
            product(grams,bgr,bu,'T','N',1.,1.);
            product(bt,u,bar_amplitudes[s],'N','T');
            sequential_grams.push_back(std::move(grams));
            // Keep this leg in the smaller selected occupied space until
            // its target is consumed below. No additional integral replay.
            sequential_bargrams.push_back(std::move(bgr));
        }
        out.times["sequential_seed_seconds"]=omp_get_wtime()-tick;
        const double selection_start=omp_get_wtime();
        {
            py::gil_scoped_acquire acquire;
            Array correction=rotation_pullback(array(bu)).cast<Array>();
            shape_is(T{Input(correction)},{no,no});
            extra_occupied=from_c(correction.data(),no,no);
        }
        symmetrize(extra_occupied);
        out.times["occupied_selection_pullback_seconds"]=omp_get_wtime()-selection_start;
    }
    const double contractions=omp_get_wtime();
    for(Index s=0;s<nt;++s) {
        const M& t=amplitudes[s];const M bi=dress.b.columns(targets[s]*nv,nv);
        M grams,bargrams,wo,wv;
        bool occupied_active=false,virtual_active=false;
        if(reverse) {
            wo=from_c(ob.data()+s*no*no,no,no);symmetrize(wo);
            if(sequential_reverse)add(wo,extra_occupied);
            if(!density)for(Index i=0;i<no;++i)wo(i,i)+=eb.data()[s];
            for(Index j=0;j<no*no;++j)occupied_active|=wo.p[j]!=0.;
            if(!sequential_reverse) {
                wv=frame(from_c(vb.data()+s*vv,nv,nv),dress.uv,true);
                for(Index j=0;j<vv;++j)virtual_active|=wv.p[j]!=0.;
            }
            grams=sequential_reverse?std::move(sequential_grams[s]):M(vv,no);
            bargrams=M(vv,no,!sequential_reverse);
            if(sequential_reverse) {
                product(sequential_bargrams[s],sequential_rotation,bargrams,'N','T');
                sequential_bargrams[s]=M();
            }
        } else if(sequential)grams=M(vv,no);
        std::vector<double> target_energy(no,0.);
        const Index panel=reverse?1:std::max<Index>(1,std::min<Index>(4,no/omp_get_max_threads()));
        if(!sequential_reverse) {
            #pragma omp parallel
            {
                M local;
                if(!reverse && !sequential && !energy_only)local=M(nv,nv,true);
                M occupied_panel(vv,panel);
                double gram_time=0.,occ_time=0.,vir_time=0.;
                #pragma omp for schedule(static)
                for(Index first=0;first<no;first+=panel) {
                    const Index last=std::min(no,first+panel);
                    for(Index j=first;j<last;++j) {
                        const double tick=omp_get_wtime();
                        M g=multiply(bi,dress.b.columns(j*nv,nv),'T');M sg=spin(g);
                        const M tj=t.view(nv,nv,j*vv,nv);
                        {
                            double e=0.;for(Index k=0;k<vv;++k)e+=tj.p[k]*sg.p[k];
                            target_energy[j]=e;
                        }
                        if(density)sg=spin(tj);
                        gram_time+=omp_get_wtime()-tick;
                        if(reverse || sequential)copy(grams.view(nv,nv,j*vv,nv),sg);
                        if(!reverse && !energy_only) {
                            copy(occupied_panel.columns(j-first,1),sg.view(vv,1,0,vv));
                            if(!sequential) {
                                const double vstart=omp_get_wtime();const M tj=t.view(nv,nv,j*vv,nv);
                                product(tj,sg,local,'N','T',.5,1.);product(tj,sg,local,'T','N',.5,1.);
                                vir_time+=omp_get_wtime()-vstart;
                            }
                        }
                    }
                    if(!reverse && !energy_only) {
                        const double tick=omp_get_wtime();
                        product(t,occupied_panel.columns(0,last-first),out.wo[per_target?s:0].columns(first,last-first),'T','N',1.,1.);
                        occ_time+=omp_get_wtime()-tick;
                    }
                }
                #pragma omp critical(dw_forward_reduce)
                {
                    if(!reverse && !sequential && !energy_only)add(out.wv[per_target?s:0],local);
                    out.times["bare_gram_worker_seconds"]+=gram_time;
                    out.times["occupied_w_worker_seconds"]+=occ_time;
                    out.times["virtual_w_worker_seconds"]+=vir_time;
                }
            }
        }
        for(double e:target_energy)out.energies[per_target?s:0]+=e;
        if(sequential)out.sequential->grams.push_back(std::move(grams));
        if(reverse) {
            M bt=bar_amplitudes[s];
            if(occupied_active) {
                #pragma omp parallel for schedule(static)
                for(Index first=0;first<vv;first+=8192) {
                    const Index n=std::min<Index>(8192,vv-first);
                    product(grams.row_panel(first,n),wo,bt.row_panel(first,n),'N','T',1.,1.);
                    product(t.row_panel(first,n),wo,bargrams.row_panel(first,n),'N','N',1.,sequential_reverse?1.:0.);
                }
            }
            #pragma omp parallel
            {
                M local(np,nv,true);
                #pragma omp for schedule(static)
                for(Index j=0;j<no;++j) {
                    M tj=t.view(nv,nv,j*vv,nv),bar_t=bt.view(nv,nv,j*vv,nv);
                    M bg=bargrams.view(nv,nv,j*vv,nv);
                    if(virtual_active && !sequential_reverse) {
                        const M sg=grams.view(nv,nv,j*vv,nv);
                        product(wv,sg,bar_t,'N','N',.5,1.);product(sg,wv,bar_t,'N','N',.5,1.);
                        product(wv,tj,bg,'N','N',.5,1.);product(tj,wv,bg,'N','N',.5,1.);
                    }
                    M gradient=spin(bg);
                    if(density) {
                        // Both density factors are amplitudes. MP2 energy has
                        // a separate bare-integral leg and the same dressed T.
                        add(bar_t,gradient);
                        M bare=multiply(bi,dress.b.columns(j*nv,nv),'T');
                        add(bar_t,spin(bare),eb.data()[s]);
                        gradient=spin(tj);
                        for(Index k=0;k<vv;++k)gradient.p[k]*=eb.data()[s];
                    }
                    product(bi,gradient,out.b.columns(j*nv,nv),'N','N',1.,1.);
                    product(dress.b.columns(j*nv,nv),gradient,local,'N','T',1.,1.);
                }
                #pragma omp critical(dw_factor_reduce)
                add(out.b.columns(targets[s]*nv,nv),local);
            }
        }
    }
    if(!reverse) {
        for(Index s=0;s<Index(out.wo.size());++s) {
            symmetrize(out.wo[s]);symmetrize(out.wv[s]);
            if(!sequential)out.wv[s]=frame(out.wv[s],dress.uv,false);
        }
        if(sequential)out.sequential->amplitudes=std::move(amplitudes);
    }
    out.times["contraction_seconds"]=omp_get_wtime()-contractions;
    if(reverse) {
        amplitudes.clear();const double reverse_start=omp_get_wtime();M gamma(np,no*nv);
        for(Index step=0;step<beta.size();++step) {
            const Index l=beta.size()-1-step;
            double tick=omp_get_wtime();
            if(step)dress.apply(l,c);
            out.times["reverse_factor_seconds"]+=omp_get_wtime()-tick;
            tick=omp_get_wtime();
            for(Index s=0;s<nt;++s) {
                const M ci=c.columns(targets[s]*nv,nv),bt=bar_amplitudes[s];
                #pragma omp parallel
                {
                    M local(np,nv,true);
                    #pragma omp for schedule(static)
                    for(Index j=0;j<no;++j) {
                        const M bar=bt.view(nv,nv,j*vv,nv);
                        product(ci,bar,gamma.columns(j*nv,nv),'N','N',-weights.data()[l],s?1.:0.);
                        product(c.columns(j*nv,nv),bar,local,'N','T',-weights.data()[l],1.);
                    }
                    #pragma omp critical(dw_amplitude_reduce)
                    add(gamma.columns(targets[s]*nv,nv),local);
                }
            }
            out.times["amplitude_vjp_seconds"]+=omp_get_wtime()-tick;tick=omp_get_wtime();
            dress.reverse(l,c,gamma,out.fo,out.fv,out.b);
            out.times["propagator_vjp_seconds"]+=omp_get_wtime()-tick;
        }
        bar_amplitudes.clear();gamma=M();dress.b=M();
        out.fv=frame(out.fv,dress.uv,false);dress.rotate(out.b,c,false);out.b=std::move(c);
        out.times["reverse_seconds"]=omp_get_wtime()-reverse_start;
    }
    out.times["native_total_seconds"]=omp_get_wtime()-start;return out;
}

py::dict run(const Array& b,const Array& uo,const Array& uv,const Array& eo,const Array& ev,
        const Array& roots,const Array& weights,const std::vector<Index>& targets,
        const Array& eb,const Array& ob,const Array& vb,bool per_target,bool sequential,bool density,bool energy_only,
        const Array& rotation,const py::object& rotation_pullback) {
    if(b.ndim()!=3 || !b.shape(0) || !b.shape(1) || !b.shape(2))throw py::value_error("W requires nonempty rank-three factors");
    const Index no=b.shape(1),nv=b.shape(2),nt=targets.size();
    for(const auto* a:{&b,&uo,&uv,&eo,&ev,&roots,&weights,&eb,&ob,&vb})T{Input(*a)};
    shape_is(T{Input(uo)},{no,no});shape_is(T{Input(uv)},{nv,nv});
    shape_is(T{Input(eo)},{no});shape_is(T{Input(ev)},{nv});
    if(roots.ndim()!=1 || !roots.size())throw py::value_error("W requires a nonempty Laplace grid");
    shape_is(T{Input(weights)},{roots.size()});
    if(!nt)throw py::value_error("W requires selected occupied targets");
    std::vector<bool> seen(no,false);
    for(auto i:targets) {if(i<0 || i>=no || seen[i])throw py::value_error("Invalid W target");seen[i]=true;}
    if(sequential && energy_only)throw py::value_error("Sequential mode requires W or D matrices");
    if(eb.size()) {
        shape_is(T{Input(eb)},{nt});shape_is(T{Input(ob)},{nt,no,no});shape_is(T{Input(vb)},{nt,nv,nv});
        if(sequential)throw py::value_error("Sequential reverse uses an occupied rotation, not a forward workspace");
    }
    if(rotation.ndim()==2) {
        if(!eb.size() || sequential || energy_only || per_target || rotation_pullback.is_none())
            throw py::value_error("Sequential reverse requires matrix seeds and a rotation pullback");
        T{Input(rotation)};
        if(rotation.shape(0)!=no || rotation.shape(1)>no)
            throw py::value_error("Invalid occupied rotation");
    } else if(!rotation_pullback.is_none())throw py::value_error("Missing sequential occupied rotation");
    Result result;
    {py::gil_scoped_release release;result=evaluate(b,uo,uv,eo,ev,roots,weights,targets,eb,ob,vb,per_target,sequential,density,energy_only,rotation,rotation_pullback);}
    py::dict out;out["timings"]=result.times;out["openmp_dynamic"]=omp_get_dynamic()!=0;out["energy"]=result.energies;
    if(eb.size()) {
        out["foo"]=array(result.fo);out["fvv"]=array(result.fv);
        Array bars=array(result.b);bars.resize({b.shape(0),no,nv});out["B"]=bars;
    } else {
        py::list occupied,virtuals;
        for(Index i=0;i<Index(result.wo.size());++i) {
            occupied.append(array(result.wo[i]));virtuals.append(array(result.wv[i]));
        }
        out["energy"]=result.energies;out["occupied"]=occupied;out["virtual"]=virtuals;
        if(sequential)out["workspace"]=result.sequential;
    }
    return out;
}
Array finish(Sequential& cache,const Array& rotation) {
    if(cache.consumed)throw py::value_error("Sequential W workspace already consumed");
    if(rotation.ndim()!=2 || rotation.shape(0)!=cache.no || rotation.shape(1)>cache.no)
        throw py::value_error("Invalid occupied rotation");
    M result;
    {
        py::gil_scoped_release release;
        const Index nv=cache.nv,vv=nv*nv,nr=rotation.shape(1);M u=from_c(rotation.data(),cache.no,nr);
        M gram=multiply(u,u,'T');double error=0.;
        for(Index j=0;j<nr;++j)for(Index i=0;i<nr;++i)error+=std::pow(gram(i,j)-(i==j),2);
        if(error>1e-16)throw std::invalid_argument("Occupied rotation must be orthonormal");
        result=M(nv,nv,true);
        for(Index s=0;s<Index(cache.amplitudes.size());++s) {
            M t=multiply(cache.amplitudes[s],u),g=multiply(cache.grams[s],u);
            cache.amplitudes[s]=M();cache.grams[s]=M();
            const int workers=std::max(1,std::min(omp_get_max_threads(),bi(nr)));
            #pragma omp parallel num_threads(workers)
            {
                M local(nv,nv,true);
                #pragma omp for schedule(static)
                for(Index j=0;j<nr;++j) {
                    M a=t.view(nv,nv,j*vv,nv),b=g.view(nv,nv,j*vv,nv);
                    product(a,b,local,'N','T',.5,1.);product(a,b,local,'T','N',.5,1.);
                }
                #pragma omp critical(dw_sequential_reduce)
                add(result,local);
            }
        }
        result=frame(result,cache.rotation,false);cache.amplitudes.clear();cache.grams.clear();cache.rotation=M();cache.consumed=true;
    }
    return array(result);
}
} } // namespaces

void bind_deterministic_w(py::module_& m) {
    using namespace ccsd_native::dw;
    py::class_<Sequential,std::shared_ptr<Sequential>>(m,"DeterministicWWorkspace");
    m.def("deterministic_w",&run,py::arg("B").noconvert(),py::arg("Uo").noconvert(),py::arg("Uv").noconvert(),
        py::arg("eo").noconvert(),py::arg("ev").noconvert(),py::arg("roots").noconvert(),py::arg("weights").noconvert(),
        py::arg("targets"),py::arg("energy_bars").noconvert(),py::arg("occupied_bars").noconvert(),py::arg("virtual_bars").noconvert(),
        py::arg("per_target")=false,py::arg("sequential")=false,py::arg("density")=false,py::arg("energy_only")=false,
        py::arg("occupied_rotation")=ccsd_native::Array(0),py::arg("rotation_pullback")=py::none());
    m.def("finish_deterministic_w",&finish,py::arg("workspace"),py::arg("rotation").noconvert());
}
