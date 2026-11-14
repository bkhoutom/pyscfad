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
    for(size_t i=0;i<layout.tensors.size();++i) {
        const auto& spec=layout.tensors[i];
        bank.emplace_back(int(spec.kind),spec.dims,gradient,input.tensors[i].memptr());
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
        #pragma omp single
        result.setup_threads=result.sampling_threads=omp_get_num_threads();
        #pragma omp for schedule(dynamic,1)
        for(size_t index=0;index<count;++index) {
            try {
                const auto b=blocks[cache.first+index];
                const tuple base{{{b[0],b[1],b[2],0,0,0}},0};
                arma::vec raw(cube),tmp(cube);
                if(mode==0) {
                    tmp.zeros();
                    for(size_t p=0;p<6;++p) {
                        auto op=quadrature_describe(layout,permute(base,p,true),0);
                        triples_raw_block(bank,op,no,nv,false,raw);
                        triples_permute_add(tmp,raw,nv,p);
                    }
                    cache.values[index]=triples_spin(tmp,nv);
                    if(cache.gradient) cache.bars[index].zeros(cube);
                } else if(mode==2) {
                    arma::vec leftbar=triples_spin(cache.bars[index],nv);
                    for(size_t p=0;p<6;++p) {
                        tmp.zeros();triples_permute_add(tmp,leftbar,nv,p,true);
                        auto op=quadrature_describe(layout,permute(base,p,true),0);
                        triples_raw_block(bank,op,no,nv,false,raw,&tmp,&locks);
                    }
                } else {
                    // Summing over all ordered j,k lets the (jk)(bc) coset
                    // supply the other three pair permutations. Triangular
                    // cache entries serve both orientations; no fragment
                    // operator is commuted through an occupied permutation.
                    const double factor=-2.*weight/3.;
                    for(size_t swap=0;swap<(b[1]==b[2] ? 1u : 2u);++swap) {
                        tuple oriented=base;
                        arma::vec ket;
                        if(swap) {
                            std::swap(oriented.x[1],oriented.x[2]);
                            ket.zeros(cube);triples_permute_add(ket,cache.values[index],nv,3);
                        } else ket=arma::vec(cache.values[index].memptr(),cube,false,true);
                        for(size_t p=0;p<3;++p) {
                            auto op=quadrature_describe(layout,permute(oriented,p,true),1);
                            triples_raw_block(bank,op,no,nv,true,raw);
                            tmp.zeros();triples_permute_add(tmp,ket,nv,p,true);tmp*=factor;
                            energies[index]+=arma::dot(raw,tmp);
                            if(gradient) {
                                arma::vec contribution(cube,arma::fill::zeros);
                                triples_permute_add(contribution,raw,nv,p);
                                contribution*=factor;
                                if(swap) triples_permute_add(cache.bars[index],contribution,nv,3,true);
                                else cache.bars[index]+=contribution;
                                triples_raw_block(bank,op,no,nv,true,raw,&tmp,&locks);
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
        for(auto& b:bank) vjp->prepared.push_back(std::move(b.bar));
        result.reverse_buffer_bytes=result.deterministic_workspace_bytes;
    }
    result.sampling_seconds=omp_get_wtime()-start;
    return result;
}
