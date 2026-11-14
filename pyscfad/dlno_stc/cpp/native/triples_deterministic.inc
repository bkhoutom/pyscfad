// Included inside the anonymous namespace of stc_triples_asymmetric.C.
// Work on one (s,j,k) block at a time. Every numerator has a selected occupied
// leg already transformed by StableDressing; no full occupied triples tensor
// or scalar sampling/reverse-event table is constructed.

// Row-major abc storage is a column-major matrix with c rows and ab columns.
// Pair permutations of the numerator let the occupied spin permutations be
// evaluated as virtual permutations of this same block. This identity does
// not require symmetry of the input amplitudes or prepared integral tensors.
// Compile-time permutations avoid dynamic index selection in the innermost
// loop. The adjoint gathers through the inverse permutation, rather than
// scattering into strided output.
template<size_t P> void triples_permute_impl(double* out,const double* raw,size_t n) {
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
        double* dst=out+(a*n+b)*n;
        for(size_t c=0;c<n;++c) {
            size_t source;
            if constexpr(P==0) source=(a*n+b)*n+c;
            if constexpr(P==1) source=(b*n+c)*n+a;
            if constexpr(P==2) source=(c*n+a)*n+b;
            if constexpr(P==3) source=(a*n+c)*n+b;
            if constexpr(P==4) source=(b*n+a)*n+c;
            if constexpr(P==5) source=(c*n+b)*n+a;
            dst[c]+=raw[source];
        }
    }
}
void triples_permute_add(arma::vec& out,const arma::vec& raw,size_t n,size_t p,
                         bool pullback=false) {
    if(pullback && (p==1 || p==2)) p=3-p;
    switch(p) {
#define TRIPLES_PERM(P) case P: triples_permute_impl<P>(out.memptr(),raw.memptr(),n); break;
        TRIPLES_PERM(0) TRIPLES_PERM(1) TRIPLES_PERM(2)
        TRIPLES_PERM(3) TRIPLES_PERM(4) TRIPLES_PERM(5)
#undef TRIPLES_PERM
    }
}
arma::vec triples_spin(const arma::vec& input,size_t n) {
    arma::vec out(input.n_elem,arma::fill::none);
    const double* x=input.memptr();
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c)
        out[(a*n+b)*n+c]=4*x[(a*n+b)*n+c]+x[(b*n+c)*n+a]+x[(c*n+a)*n+b]
            -2*(x[(a*n+c)*n+b]+x[(b*n+a)*n+c]+x[(c*n+b)*n+a]);
    return out;
}

// Cotangents share one bank. Locks protect disjoint occupied slabs; no
// worker has a private copy of the (potentially very large) tensor bank.
struct triples_bar_locks {
    std::vector<size_t> starts;
    std::unique_ptr<std::mutex[]> locks;
    explicit triples_bar_locks(const std::vector<tensor>& bank) {
        size_t count=0;
        for(const auto& t:bank) {
            starts.push_back(count);
            count+=t.kind==T2O ? t.dims[1] : t.dims[0];
        }
        locks.reset(new std::mutex[count]);
    }
    std::mutex& at(const operations& op,size_t item,const tensor& t) {
        return locks[starts[op.ids[item]]+op.indices[item][t.kind==T2O ? 1 : 0]];
    }
};

// Twelve GEMMs per side (six virtual and six occupied contractions), plus
// the disconnected outer products on the right. Reverse uses their matrix
// adjoints and accumulates directly into the prepared tensor cotangents.
void triples_raw_block(std::vector<tensor>& bank,const operations& op,
                       size_t no,size_t nv,bool singles,arma::vec& raw,
                       const arma::vec* raw_bar=nullptr,triples_bar_locks* locks=nullptr) {
    auto& gv=bank[op.ids[0]]; auto& tv=bank[op.ids[1]];
    auto& go=bank[op.ids[2]]; auto& to=bank[op.ids[3]];
    const size_t ng=gv.offset(op.indices[0]),nt=tv.offset(op.indices[1]);
    const size_t ngo=go.offset(op.indices[2]),nto=to.offset(op.indices[3]);
    const size_t go_step=go.dims[2]*go.dims[3],to_step=to.dims[1]*nv*nv;
    arma::mat g(gv.value.memptr()+ng,nv,nv*nv,false,true);
    arma::mat t(tv.value.memptr()+nt,nv,nv,false,true);
    arma::mat h(no,nv),u(nv*nv,no);
    for(size_t a=0;a<nv;++a)
        std::copy_n(go.value.memptr()+ngo+a*go_step,no,h.colptr(a));
    for(size_t m=0;m<no;++m)
        std::copy_n(to.value.memptr()+nto+m*to_step,nv*nv,u.colptr(m));
    if(!raw_bar) {
        raw.set_size(nv*nv*nv);
        arma::mat abc(raw.memptr(),nv,nv*nv,false,true);
        abc=t.t()*g;
        arma::mat bca(raw.memptr(),nv*nv,nv,false,true);
        bca-=u*h;
    } else {
        arma::mat abc(const_cast<double*>(raw_bar->memptr()),nv,nv*nv,false,true);
        arma::mat bca(const_cast<double*>(raw_bar->memptr()),nv*nv,nv,false,true);
        arma::mat gb(gv.bar.memptr()+ng,nv,nv*nv,false,true);
        arma::mat tb(tv.bar.memptr()+nt,nv,nv,false,true);
        arma::mat delta=t*abc;
        { std::lock_guard<std::mutex> guard(locks->at(op,0,gv)); gb+=delta; }
        delta=g*abc.t();
        { std::lock_guard<std::mutex> guard(locks->at(op,1,tv)); tb+=delta; }
        arma::mat hb=-u.t()*bca,ub=-bca*h.t();
        { std::lock_guard<std::mutex> guard(locks->at(op,2,go));
          for(size_t a=0;a<nv;++a) for(size_t m=0;m<no;++m)
            go.bar[ngo+a*go_step+m]+=hb(m,a); }
        { std::lock_guard<std::mutex> guard(locks->at(op,3,to));
          for(size_t m=0;m<no;++m) for(size_t bc=0;bc<nv*nv;++bc)
            to.bar[nto+m*to_step+bc]+=ub(bc,m); }
    }
    if(!singles) return;
    auto& gg=bank[op.ids[4]]; auto& t1=bank[op.ids[5]];
    auto& fv=bank[op.ids[6]]; auto& t2=bank[op.ids[7]];
    const size_t ngg=gg.offset(op.indices[4]),n1=t1.offset(op.indices[5]);
    const size_t nf=fv.offset(op.indices[6]),n2=t2.offset(op.indices[7]);
    const size_t gg_step=gg.dims[2]*gg.dims[3];
    arma::vec integral(nv*nv);
    for(size_t a=0;a<nv;++a)
        std::copy_n(gg.value.memptr()+ngg+a*gg_step,nv,integral.memptr()+a*nv);
    arma::vec one(t1.value.memptr()+n1,nv,false,true);
    arma::vec f(fv.value.memptr()+nf,nv,false,true);
    arma::vec two(t2.value.memptr()+n2,nv*nv,false,true);
    if(!raw_bar) {
        arma::mat abc(raw.memptr(),nv,nv*nv,false,true);
        abc+=.5*(one*integral.t()+f*two.t());
    } else {
        arma::mat abc(const_cast<double*>(raw_bar->memptr()),nv,nv*nv,false,true);
        arma::vec onebar(t1.bar.memptr()+n1,nv,false,true);
        arma::vec fbar(fv.bar.memptr()+nf,nv,false,true);
        arma::vec twobar(t2.bar.memptr()+n2,nv*nv,false,true);
        arma::vec delta=.5*(abc*integral);
        { std::lock_guard<std::mutex> guard(locks->at(op,5,t1)); onebar+=delta; }
        delta=.5*(abc*two);
        { std::lock_guard<std::mutex> guard(locks->at(op,6,fv)); fbar+=delta; }
        delta=.5*(abc.t()*f);
        { std::lock_guard<std::mutex> guard(locks->at(op,7,t2)); twobar+=delta; }
        arma::vec integralbar=.5*(abc.t()*one);
        { std::lock_guard<std::mutex> guard(locks->at(op,4,gg));
          for(size_t a=0;a<nv;++a) for(size_t b=0;b<nv;++b)
            gg.bar[ngg+a*gg_step+b]+=integralbar[a*nv+b]; }
    }
}

// MKL's local setting is restored on each worker, including on exceptions.
// It does not change the embedding process's dynamic-team policy.
struct triples_local_blas {
    using Setter=int(*)(int);
    Setter setter;
    int previous=0;
    explicit triples_local_blas(Setter fn):setter(fn) { if(setter) previous=setter(1); }
    ~triples_local_blas() { if(setter) setter(previous); }
};

template<class Describe>
void triples_deterministic_blocks(std::vector<tensor>& bank,Describe describe,
        size_t no,size_t nv,size_t ns,bool gradient,double cotangent,
        stc_triples::triples_result& result) {
    const size_t cube=nv*nv*nv;
    // The six pair permutations and the central spin-permutation operator
    // make E(s,j,k)==E(s,k,j), even for independent nonsymmetric inputs.
    std::vector<std::array<size_t,3>> blocks;
    for(size_t s=0;s<ns;++s) for(size_t j=0;j<no;++j) for(size_t k=0;k<=j;++k)
        blocks.push_back({s,j,k});
    const size_t workspace=(12*cube+6*no*nv*nv+4*no*nv+8*nv*nv+4*nv)*sizeof(double);
    // Bound worker scratch at 8 GiB (one worker is always allowed). The
    // shared input/cotangent banks are accounted separately in resident bytes.
    const size_t budget=size_t(8)*1024*1024*1024;
    int workers=int(std::min({size_t(omp_get_max_threads()),blocks.size(),
                              std::max(size_t(1),budget/std::max(size_t(1),workspace))}));
    workers=std::max(1,workers);
    auto mkl_setter=reinterpret_cast<triples_local_blas::Setter>(
        dlsym(RTLD_DEFAULT,"MKL_Set_Num_Threads_Local"));
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    // Only pthread OpenBLAS supports a process-wide scope around this region.
    const bool openblas_parallel=openblas_get_parallel()==1;
    detail::scoped_openblas_threads blas_scope(openblas_parallel ? 1 : 0);
    if(!mkl_setter && !openblas_parallel) workers=1;
#else
    if(!mkl_setter) workers=1; // Unknown BLAS provider: avoid nested teams.
#endif
    triples_bar_locks locks(bank);
    std::vector<double> energies(blocks.size()),reverse_times(blocks.size());
    std::exception_ptr failure;
    std::mutex failure_lock;
    #pragma omp parallel num_threads(workers)
    {
        triples_local_blas local_blas(workers>1 ? mkl_setter : nullptr);
        #pragma omp single
        result.sampling_threads=omp_get_num_threads();
        #pragma omp for schedule(dynamic,1)
        for(size_t block=0;block<blocks.size();++block) {
            try {
                const auto indices=blocks[block];
                const size_t s=indices[0],j=indices[1],k=indices[2];
                const double multiplicity=j==k ? 1. : 2.;
                const tuple base{{{s,j,k,0,0,0}},0};
                arma::vec a(cube,arma::fill::zeros),b(cube,arma::fill::zeros),raw(cube);
                for(size_t side=0;side<2;++side) for(size_t p=0;p<6;++p) {
                    const auto op=describe(permute(base,p,true),side);
                    triples_raw_block(bank,op,no,nv,side!=0,raw);
                    triples_permute_add(side ? b : a,raw,nv,p);
                }
                arma::vec z=triples_spin(b,nv);
                const double energy=-multiplicity*arma::dot(a,z)/3.;
                if(!std::isfinite(energy)) throw std::runtime_error("deterministic (T): nonfinite block");
                energies[block]=energy;
                if(!gradient) continue;
                const double start=omp_get_wtime();
                b=triples_spin(a,nv);
                b*=-multiplicity*cotangent/3.; z*=-multiplicity*cotangent/3.;
                arma::vec rawbar(cube);
                for(size_t side=0;side<2;++side) for(size_t p=0;p<6;++p) {
                    rawbar.zeros();
                    triples_permute_add(rawbar,side ? b : z,nv,p,true);
                    const auto op=describe(permute(base,p,true),side);
                    triples_raw_block(bank,op,no,nv,side!=0,raw,&rawbar,&locks);
                }
                reverse_times[block]=omp_get_wtime()-start;
            } catch(...) {
                std::lock_guard<std::mutex> guard(failure_lock);
                if(!failure) failure=std::current_exception();
            }
        }
    }
    if(failure) std::rethrow_exception(failure);
    // Fixed block-order reduction keeps the energy independent of scheduling.
    for(double energy:energies) result.energy+=energy;
    result.deterministic_blocks=blocks.size();
    result.deterministic_workspace_bytes=workspace*size_t(result.sampling_threads);
    // Worker-seconds, since reverse contractions now overlap across blocks.
    for(double elapsed:reverse_times) result.vjp_accumulation_seconds+=elapsed;
    result.vjp_reverse_seconds=result.vjp_accumulation_seconds;
    if(gradient) result.reverse_buffer_bytes=result.deterministic_workspace_bytes;
}
