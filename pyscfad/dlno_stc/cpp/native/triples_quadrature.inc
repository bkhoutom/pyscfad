// Exact one-sided dressing and single pair symmetrization. Included in the
// same anonymous namespace as the block GEMMs and their adjoints.
operations quadrature_describe(const triples_layout& layout,const tuple& t,size_t side) {
    const auto& x=t.x;
    const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5],s=t.selected;
    auto id=[&](size_t kind,int slot) { return slot<0 ? kind : layout.projected[side][kind][size_t(slot)]; };
    operations op;
    op.ids={{id(GV,s==0 ? 0 : -1),id(T2V,s==2 ? 0 : (s==1 ? 1 : -1)),
             id(GO,s==0 ? 0 : (s==1 ? 1 : -1)),id(T2O,s==2 ? 1 : -1),
             id(GG,s==0 ? 0 : (s==1 ? 1 : -1)),id(T1,s==2 ? 0 : -1),
             id(FOV,s==2 ? 0 : -1),id(T2,s==0 ? 0 : (s==1 ? 1 : -1))}};
    op.indices={{{{i,a,b,0}},{{k,j,c,0}},{{i,a,j,0}},{{0,k,b,c}},
                 {{i,a,j,b}},{{k,c,0,0}},{{k,c,0,0}},{{i,j,a,b}}}};
    return op;
}
std::vector<std::array<size_t,3>> quadrature_blocks(size_t no,size_t ns) {
    std::vector<std::array<size_t,3>> out;
    for(size_t s=0;s<ns;++s) for(size_t j=0;j<no;++j) for(size_t k=0;k<=j;++k)
        out.push_back({s,j,k});
    return out;
}
// Direct slices of the two disconnected outer-product factors. No integral
// gathering or L cube is needed for the energy/cache-adjoint contractions.
struct quadrature_singles {
    const double *integral,*two,*one,*fov;
    size_t stride;
    quadrature_singles(const std::vector<tensor>& bank,const operations& op) {
        const auto& g=bank[op.ids[4]];
        integral=g.value.memptr()+g.offset(op.indices[4]);
        stride=g.dims[2]*g.dims[3];
        const auto& t=bank[op.ids[7]];two=t.value.memptr()+t.offset(op.indices[7]);
        const auto& o=bank[op.ids[5]];one=o.value.memptr()+o.offset(op.indices[5]);
        const auto& f=bank[op.ids[6]];fov=f.value.memptr()+f.offset(op.indices[6]);
    }
    double half_value(size_t a,size_t b,size_t c,size_t n) const {
        return .5*(integral[a*stride+b]*one[c]+two[a*n+b]*fov[c]);
    }
};
template<size_t P> void quadrature_coordinates(size_t a,size_t b,size_t c,
        size_t& x,size_t& y,size_t& z) {
    if constexpr(P==0) {x=a;y=b;z=c;}
    if constexpr(P==1) {x=b;y=c;z=a;}
    if constexpr(P==2) {x=c;y=a;z=b;}
    if constexpr(P==3) {x=a;y=c;z=b;}
    if constexpr(P==4) {x=b;y=a;z=c;}
    if constexpr(P==5) {x=c;y=b;z=a;}
}
// P maps a raw numerator element to its cached K element. Its inverse is
// used for a contiguous cache-bar gather, avoiding a strided scatter.
template<size_t P,bool Gradient>
double quadrature_right_impl(arma::vec& raw_bar,arma::vec* cache_bar,
        const arma::vec& raw,const arma::vec& ket,const quadrature_singles& l,
        size_t n,double factor,triples_raw_workspace& workspace,triples_right_profile* profile) {
    const double* r=raw.memptr();const double* k=ket.memptr();
    double *one_bar=nullptr,*fov_bar=nullptr,*two_bar=nullptr,*integral_bar=nullptr;
    if constexpr(Gradient) {
        workspace.one_bar.zeros();workspace.fov_bar.zeros();
        one_bar=workspace.one_bar.memptr();fov_bar=workspace.fov_bar.memptr();
        two_bar=workspace.two_bar.memptr();integral_bar=workspace.integral_bar.memptr();
    }
    double* rb=Gradient ? raw_bar.memptr() : nullptr;
    double energy=0.;
    {
    triples_profile_scope timer(triples_profile_phase(profile,triples_right_profile::contract));
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
        const size_t offset=(a*n+b)*n;
        const double g=l.integral[a*l.stride+b],t=l.two[a*n+b];
        double two_sum=0.,integral_sum=0.;
        #pragma omp simd reduction(+:energy,two_sum,integral_sum)
        for(size_t c=0;c<n;++c) {
            const double weight=factor*k[triples_permuted_index<P>(a,b,c,n)];
            const double value=r[offset+c]+.5*(g*l.one[c]+t*l.fov[c]);
            energy+=weight*value;
            if constexpr(Gradient) {
                rb[offset+c]=weight;
                const double half=.5*weight;
                one_bar[c]+=half*g;fov_bar[c]+=half*t;
                two_sum+=half*l.fov[c];integral_sum+=half*l.one[c];
            }
        }
        if constexpr(Gradient) {
            two_bar[a*n+b]=two_sum;integral_bar[a*n+b]=integral_sum;
        }
    }
    }
    if constexpr(Gradient) {
        triples_profile_scope timer(triples_profile_phase(profile,triples_right_profile::ket_bar));
        constexpr size_t inverse=P==1 ? 2 : P==2 ? 1 : P;
        double* kb=cache_bar->memptr();
        for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
            const size_t offset=(a*n+b)*n;
            #pragma omp simd
            for(size_t c=0;c<n;++c) {
                size_t x,y,z;quadrature_coordinates<inverse>(a,b,c,x,y,z);
                kb[offset+c]+=factor*(r[(x*n+y)*n+z]+l.half_value(x,y,z,n));
            }
        }
    }
    return energy;
}
double quadrature_right(arma::vec& raw_bar,arma::vec* cache_bar,const arma::vec& raw,
        const arma::vec& ket,const std::vector<tensor>& bank,const operations& op,
        size_t n,size_t p,bool swap,double factor,bool gradient,
        triples_raw_workspace& workspace,triples_right_profile* profile=nullptr) {
    const quadrature_singles singles(bank,op);
    const size_t ordinary[3]={0,2,1},exchanged[3]={3,5,4};
    const size_t permutation=swap ? exchanged[p] : ordinary[p];
#define QUADRATURE_RIGHT(P) case P: return gradient \
    ? quadrature_right_impl<P,true>(raw_bar,cache_bar,raw,ket,singles,n,factor,workspace,profile) \
    : quadrature_right_impl<P,false>(raw_bar,nullptr,raw,ket,singles,n,factor,workspace,profile);
    switch(permutation) {QUADRATURE_RIGHT(0) QUADRATURE_RIGHT(1) QUADRATURE_RIGHT(2)
                         QUADRATURE_RIGHT(3) QUADRATURE_RIGHT(4) QUADRATURE_RIGHT(5)}
#undef QUADRATURE_RIGHT
    throw std::invalid_argument("quadrature: invalid cyclic permutation");
}
// The four disconnected adjoints were reduced during the weighted energy
// pass. Commit only their small vectors/matrices, under the exact tile locks.
void quadrature_singles_store(std::vector<tensor>& bank,const operations& op,
        size_t n,const triples_raw_workspace& w,triples_bar_locks& locks,
        triples_right_profile* profile=nullptr) {
    auto& t1=bank[op.ids[5]];auto& fv=bank[op.ids[6]];
    auto& t2=bank[op.ids[7]];auto& gg=bank[op.ids[4]];
    const size_t n1=t1.offset(op.indices[5]),nf=fv.offset(op.indices[6]);
    const size_t n2=t2.offset(op.indices[7]),ng=gg.offset(op.indices[4]);
    {triples_profile_lock guard(locks.at(op,5,t1),profile,4);
     arma::vec out(t1.bar.memptr()+n1,n,false,true);out+=w.one_bar;}
    {triples_profile_lock guard(locks.at(op,6,fv),profile,5);
     arma::vec out(fv.bar.memptr()+nf,n,false,true);out+=w.fov_bar;}
    {triples_profile_lock guard(locks.at(op,7,t2),profile,6);
     arma::vec out(t2.bar.memptr()+n2,n*n,false,true);out+=w.two_bar;}
    {triples_profile_lock guard(locks.at(op,4,gg),profile,7);
     const size_t stride=gg.dims[2]*gg.dims[3];
     for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b)
         gg.bar[ng+a*stride+b]+=w.integral_bar[a*n+b];}
}
// mode 0 builds K=S B X once; 1 contracts the cyclic-only right numerator;
// 2 reverses K once, after summing cotangents over all quadrature points.
stc_triples::triples_result quadrature_kernel(const stc_triples::triples_prepared& input,
        stc_triples::quadrature_cache& cache,int mode,double weight,
        stc_triples::triples_vjp* vjp) {
    const size_t no=input.no,nv=input.nv,ns=input.ns,cube=nv*nv*nv;
    if(cache.no!=no || cache.nv!=nv || cache.ns!=ns || cache.finished)
        throw std::invalid_argument("quadrature cache: incompatible or finalized cache");
    const bool gradient=mode!=0 && cache.gradient;
    const triples_layout layout(no,nv,ns);
    std::vector<tensor> bank;
    // A tensor's Armadillo members are not noexcept-movable. Vector growth
    // otherwise deep-copies borrowed primal views and existing cotangents.
    bank.reserve(layout.tensors.size());
    std::vector<std::array<size_t,3>> zero_tasks;
    constexpr size_t zero_panel=(2*1024*1024)/sizeof(double);
    for(size_t i=0;i<layout.tensors.size();++i) {
        const auto& spec=layout.tensors[i];
        bank.emplace_back(int(spec.kind),spec.dims,false,input.tensors[i].memptr());
        if(gradient) {
            auto& bar=bank.back().bar;bar.set_size(bank.back().value.n_elem);
            triples_large_page_hint(bar.memptr(),bar.n_elem);
            for(size_t first=0;first<bar.n_elem;first+=zero_panel)
                zero_tasks.push_back({i,first,std::min(zero_panel,size_t(bar.n_elem)-first)});
        }
    }
    triples_bar_locks locks(bank);
    const auto blocks=quadrature_blocks(no,ns);
    const size_t count=cache.values.size();
    if(cache.first>blocks.size() || count>blocks.size()-cache.first)
        throw std::invalid_argument("quadrature cache: invalid block range");
    const size_t workspace=(8*cube+6*no*nv*nv+8*nv*nv)*sizeof(double);
    const size_t budget=size_t(8)*1024*1024*1024;
    int workers=int(std::min({size_t(omp_get_max_threads()),count,
        std::max(size_t(1),budget/std::max(size_t(1),workspace))}));
    workers=std::max(1,workers);
    auto mkl=reinterpret_cast<triples_local_blas::Setter>(dlsym(RTLD_DEFAULT,"MKL_Set_Num_Threads_Local"));
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    const bool ob=openblas_get_parallel()==1;
    detail::scoped_openblas_threads blas_scope(ob ? 1 : 0);
    if(!mkl && !ob) workers=1;
#else
    if(!mkl) workers=1;
#endif
    stc_triples::triples_result result{};
    const double start=omp_get_wtime();
    std::vector<double> energies(count);
    std::exception_ptr failure;std::mutex error_lock;
    #pragma omp parallel num_threads(workers)
    {
        triples_local_blas local(workers>1 ? mkl : nullptr);
        arma::vec raw,tmp;
        triples_raw_workspace workspace;
        bool workspace_ready=false;
        #pragma omp single
        result.setup_threads=result.sampling_threads=omp_get_num_threads();
        // Parallel first touch avoids placing the entire shared cotangent
        // bank on the caller's NUMA node. The barrier precedes all updates.
        #pragma omp for schedule(static,1)
        for(size_t task=0;task<zero_tasks.size();++task) {
            const auto z=zero_tasks[task];
            std::fill_n(bank[z[0]].bar.memptr()+z[1],z[2],0.);
        }
        // Equal contraction counts can still have different wall times.
        // Let workers take another occupied block instead of waiting for a
        // slower worker's fixed assignment. Keep first-touch zeroing static.
        #pragma omp for schedule(dynamic,1)
        for(size_t index=0;index<count;++index) {
            try {
                if(!workspace_ready) {
                    raw.set_size(cube);tmp.set_size(cube);
                    triples_large_page_hint(raw.memptr(),cube);
                    triples_large_page_hint(tmp.memptr(),cube);
                    workspace.prepare(no,nv,gradient);
                    workspace_ready=true;
                }
                const auto b=blocks[cache.first+index];
                const tuple base{{{b[0],b[1],b[2],0,0,0}},0};
                if(mode==0) {
                    tmp.zeros();
                    for(size_t p=0;p<6;++p) {
                        auto op=quadrature_describe(layout,permute(base,p,true),0);
                        triples_raw_block(bank,op,no,nv,false,raw,nullptr,nullptr,&workspace);
                        triples_permute_add(tmp,raw,nv,p);
                    }
                    cache.values[index]=triples_spin(tmp,nv);
                    if(cache.gradient) {
                        cache.bars[index].set_size(cube);
                        triples_large_page_hint(cache.bars[index].memptr(),cube);
                        cache.bars[index].zeros();
                    }
                } else if(mode==2) {
                    arma::vec leftbar=triples_spin(cache.bars[index],nv);
                    for(size_t p=0;p<6;++p) {
                        tmp.zeros();triples_permute_add(tmp,leftbar,nv,p,true);
                        auto op=quadrature_describe(layout,permute(base,p,true),0);
                        triples_raw_block(bank,op,no,nv,false,raw,&tmp,&locks,&workspace);
                    }
                } else {
                    // Summing over all ordered j,k lets the (jk)(bc) coset
                    // supply the other three pair permutations. Triangular
                    // cache entries serve both orientations; no fragment
                    // operator is commuted through an occupied permutation.
                    const double factor=-2.*weight/3.;
                    for(size_t swap=0;swap<(b[1]==b[2] ? 1u : 2u);++swap) {
                        tuple oriented=base;
                        if(swap) std::swap(oriented.x[1],oriented.x[2]);
                        for(size_t p=0;p<3;++p) {
                            auto op=quadrature_describe(layout,permute(oriented,p,true),1);
                            triples_raw_block(bank,op,no,nv,false,raw,nullptr,nullptr,&workspace);
                            energies[index]+=quadrature_right(tmp,gradient ? &cache.bars[index] : nullptr,
                                raw,cache.values[index],bank,op,nv,p,bool(swap),factor,gradient,workspace);
                            if(gradient) {
                                quadrature_singles_store(bank,op,nv,workspace,locks);
                                triples_raw_block(bank,op,no,nv,false,raw,&tmp,&locks,&workspace,true);
                            }
                        }
                    }
                }
            } catch(...) {
                std::lock_guard<std::mutex> guard(error_lock);
                if(!failure) failure=std::current_exception();
            }
        }
    }
    if(failure) std::rethrow_exception(failure);
    for(double e:energies) result.energy+=e;
    if(!std::isfinite(result.energy)) throw std::runtime_error("quadrature cache: nonfinite energy");
    result.deterministic_blocks=count;
    result.deterministic_workspace_bytes=workspace*size_t(result.sampling_threads);
    for(const auto& b:bank) result.resident_tensor_bytes+=(b.value.n_elem+b.bar.n_elem)*sizeof(double);
    for(const auto& v:cache.values) result.resident_tensor_bytes+=v.n_elem*sizeof(double);
    for(const auto& v:cache.bars) result.resident_tensor_bytes+=v.n_elem*sizeof(double);
    if(gradient) {
        *vjp=stc_triples::triples_vjp();
        // Armadillo vectors are not noexcept-movable: growing this vector
        // otherwise copies all previously exported, potentially large bars.
        vjp->prepared.reserve(bank.size());
        for(auto& b:bank) vjp->prepared.push_back(std::move(b.bar));
        result.reverse_buffer_bytes=result.deterministic_workspace_bytes;
    }
    result.sampling_seconds=omp_get_wtime()-start;
    return result;
}
