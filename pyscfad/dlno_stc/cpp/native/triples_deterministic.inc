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
template<size_t P> size_t triples_permuted_index(size_t a,size_t b,size_t c,size_t n) {
    if constexpr(P==0) return (a*n+b)*n+c;
    if constexpr(P==1) return (b*n+c)*n+a;
    if constexpr(P==2) return (c*n+a)*n+b;
    if constexpr(P==3) return (a*n+c)*n+b;
    if constexpr(P==4) return (b*n+a)*n+c;
    return (c*n+b)*n+a;
}
template<size_t P> void triples_permute_impl(double* out,const double* raw,size_t n) {
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
        double* dst=out+(a*n+b)*n;
        for(size_t c=0;c<n;++c) {
            const size_t source=triples_permuted_index<P>(a,b,c,n);
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
void triples_spin_into(arma::vec& out,const arma::vec& input,size_t n) {
    out.set_size(input.n_elem);
    const double* x=input.memptr();
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c)
        out[(a*n+b)*n+c]=4*x[(a*n+b)*n+c]+x[(b*n+c)*n+a]+x[(c*n+a)*n+b]
            -2*(x[(a*n+c)*n+b]+x[(b*n+a)*n+c]+x[(c*n+b)*n+a]);
}
// Cotangents share one bank. Locks protect disjoint occupied slabs; no
// worker has a private copy of the (potentially very large) tensor bank.
// Pair-resolved tensors lock the occupied pair actually updated by a block;
// independent j slices must not contend on a common i-only lock.
using triples_right_profile=stc_triples::right_profile;
struct triples_profile_scope {
    double* elapsed;
    double start;
    explicit triples_profile_scope(double* value):elapsed(value),start(value ? omp_get_wtime() : 0.) {}
    ~triples_profile_scope() {if(elapsed) *elapsed+=omp_get_wtime()-start;}
};
double* triples_profile_phase(triples_right_profile* p,triples_right_profile::phase phase) {
    return p ? &p->seconds[phase] : nullptr;
}
// Time waiting for acquisition separately from the writes under the lock.
// No try_lock or altered locking order: profiling preserves synchronization.
struct triples_profile_lock {
    std::mutex& mutex;
    triples_right_profile* profile;
    size_t item;
    double acquired=0.;
    triples_profile_lock(std::mutex& lock,triples_right_profile* p,size_t index)
        :mutex(lock),profile(p),item(index) {
        const double tick=profile ? omp_get_wtime() : 0.;
        mutex.lock();
        if(profile) {
            acquired=omp_get_wtime();profile->lock_wait[item]+=acquired-tick;
            ++profile->lock_calls[item];
        }
    }
    ~triples_profile_lock() {
        if(profile) profile->store[item]+=omp_get_wtime()-acquired;
        mutex.unlock();
    }
};
struct triples_bar_locks {
    std::vector<size_t> starts;
    std::unique_ptr<std::mutex[]> locks;
    explicit triples_bar_locks(const std::vector<tensor>& bank) {
        size_t count=0;
        for(const auto& t:bank) {
            starts.push_back(count);
            count+=t.kind==T2O ? t.dims[1]
                : (t.kind==T2 || t.kind==T2V) ? t.dims[0]*t.dims[1]
                : (t.kind==GO || t.kind==GG) ? t.dims[0]*t.dims[2]
                : t.dims[0];
        }
        locks.reset(new std::mutex[count]);
    }
    std::mutex& at(const operations& op,size_t item,const tensor& t) {
        const auto& x=op.indices[item];
        const size_t slab=t.kind==T2O ? x[1]
            : (t.kind==T2 || t.kind==T2V) ? x[0]*t.dims[1]+x[1]
            : (t.kind==GO || t.kind==GG) ? x[0]*t.dims[2]+x[2] : x[0];
        return locks[starts[op.ids[item]]+slab];
    }
};

// Reused by an occupied-block worker. Matrix adjoints keep their separate
// shapes, avoiding a large delta allocation followed by a smaller resize.
struct triples_raw_workspace {
    arma::mat h,u,gb,tb,hb,ub;
    arma::vec one_bar,fov_bar,two_bar,integral_bar;
    void prepare(size_t no,size_t nv,bool gradient) {
        h.set_size(no,nv);u.set_size(nv*nv,no);
        triples_large_page_hint(u.memptr(),u.n_elem);
        if(gradient) {
            gb.set_size(nv,nv*nv);tb.set_size(nv,nv);
            hb.set_size(no,nv);ub.set_size(nv*nv,no);
            triples_large_page_hint(gb.memptr(),gb.n_elem);
            triples_large_page_hint(ub.memptr(),ub.n_elem);
            one_bar.set_size(nv);fov_bar.set_size(nv);
            two_bar.set_size(nv*nv);integral_bar.set_size(nv*nv);
        }
    }
};
// MKL's local setting is restored on each worker, including on exceptions.
// It does not change the embedding process's dynamic-team policy.
struct triples_local_blas {
    using Setter=int(*)(int);
    Setter setter;
    int previous=0;
    explicit triples_local_blas(Setter fn):setter(fn) { if(setter) previous=setter(1); }
    ~triples_local_blas() { if(setter) setter(previous); }
};
