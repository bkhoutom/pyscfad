// Bounded occupied-block batches. Each output tile has one owner, so no
// adjoint GEMM or commit below needs a lock. GV and T2O adjoints use
// direct BLAS views and retained rank-four output layouts.
enum class triples_adjoint_kind { gv,tv,go,to };
struct triples_adjoint_term {
    operations op;
    triples_output_order v,o;
    size_t block;
    triples_adjoint_term(operations operation,size_t permutation,size_t b)
        : op(std::move(operation)),v(permutation,false),o(permutation,true),block(b) {}
};
struct triples_adjoint_group {
    triples_adjoint_kind kind;
    size_t id,offset;
    std::vector<size_t> terms;
};
struct triples_adjoint_job {size_t group,first,length;};
struct triples_adjoint_plan {
    std::vector<triples_adjoint_term> terms;
    std::vector<triples_adjoint_group> groups;
    std::vector<triples_adjoint_job> jobs;
    size_t largest_group=0;
};
constexpr size_t triples_adjoint_tile=2048;

triples_adjoint_plan triples_group_plan(const std::vector<tensor>& bank,
        const triples_layout& layout,const std::vector<std::array<size_t,3>>& blocks,
        size_t first,size_t count,size_t nv) {
    triples_adjoint_plan result;result.terms.reserve(6*count);
    std::map<std::tuple<size_t,size_t,size_t>,size_t> lookup;
    for(size_t b=0;b<count;++b) {
        const auto ijk=blocks[first+b];
        const tuple base{{{ijk[0],ijk[1],ijk[2],0,0,0}},0};
        for(size_t p=0;p<6;++p) {
            const auto op=quadrature_describe(layout,permute(base,p,true),0);
            const size_t term=result.terms.size();result.terms.emplace_back(op,p,b);
            for(size_t item=0;item<4;++item) {
                const size_t id=op.ids[item],offset=bank[id].offset(op.indices[item]);
                const auto key=std::make_tuple(item,id,offset);
                auto found=lookup.find(key);
                if(found==lookup.end()) {
                    const size_t group=result.groups.size();lookup.emplace(key,group);
                    result.groups.push_back({triples_adjoint_kind(item),id,offset,{term}});
                } else result.groups[found->second].terms.push_back(term);
            }
        }
    }
    for(size_t g=0;g<result.groups.size();++g) {
        const auto& group=result.groups[g];result.largest_group=std::max(result.largest_group,group.terms.size());
        if(group.kind==triples_adjoint_kind::gv || group.kind==triples_adjoint_kind::to)
            for(size_t a=0;a<nv*nv;a+=triples_adjoint_tile)
                result.jobs.push_back({g,a,std::min(triples_adjoint_tile,nv*nv-a)});
        else result.jobs.push_back({g,0,nv});
    }
    return result;
}

struct triples_group_scratch {
    dense::Matrix out;
    void prepare(size_t no,size_t nv) { out.set_size(no*nv); }
    static size_t bytes(size_t no,size_t nv) { return no*nv*sizeof(double); }
};

// Retain only rank-four output layouts. BLAS can then consume views of the
// two existing cotangent cube orderings without per-job panel permutations.
struct triples_direct_adjoint {
    std::vector<dense::Matrix> gv_swap,to_normal,to_swap;
    std::vector<size_t> gv_ids,to_ids;
    triples_direct_adjoint(const std::vector<tensor>& bank,const triples_layout& layout,bool enabled) {
        gv_swap.resize(bank.size());to_normal.resize(bank.size());to_swap.resize(bank.size());
        if(!enabled) return;
        gv_ids={GV,layout.projected[0][GV][0]};
        to_ids={T2O,layout.projected[0][T2O][1]};
        for(size_t id:gv_ids) gv_swap[id].set_size(bank[id].value.n_elem);
        for(size_t id:to_ids) {
            to_normal[id].set_size(bank[id].value.n_elem);
            to_swap[id].set_size(bank[id].value.n_elem);
        }
    }
    size_t bytes() const {
        size_t n=0;
        for(const auto* bank:{&gv_swap,&to_normal,&to_swap})
            for(const auto& x:*bank) n+=x.n_elem*sizeof(double);
        return n;
    }
    // Called collectively once per point, after all occupied batches.
    void commit(std::vector<tensor>& bank,size_t no,size_t n) const {
        for(size_t id:gv_ids) {
            const size_t count=bank[id].dims[0]*n*n;
            #pragma omp for schedule(static)
            for(size_t p=0;p<count;++p) {
                const size_t i=p/(n*n),a=p/n%n,b=p%n;
                const double* src=gv_swap[id].memptr()+((i*n+b)*n+a)*n;
                double* dst=bank[id].bar.memptr()+p*n;
                #pragma omp simd
                for(size_t f=0;f<n;++f) dst[f]+=src[f];
            }
        }
        for(size_t id:to_ids) {
            const size_t nk=bank[id].dims[1];
            #pragma omp for schedule(static)
            for(size_t k=0;k<nk;++k) for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
                const double* normal=to_normal[id].memptr()+(k*n*n+a*n+b)*no;
                const double* swapped=to_swap[id].memptr()+(k*n*n+b*n+a)*no;
                for(size_t m=0;m<no;++m)
                    bank[id].bar[(m*nk+k)*n*n+a*n+b]+=normal[m]+swapped[m];
            }
        }
    }
};

void triples_gemm_view(char ta,char tb,size_t m,size_t n,size_t k,double alpha,
        const double* a,size_t lda,const double* b,size_t ldb,double* c,size_t ldc) {
    int bm=dense::blas_int(m),bn=dense::blas_int(n),bk=dense::blas_int(k),
        ba=dense::blas_int(lda),bb=dense::blas_int(ldb),bc=dense::blas_int(ldc);
    double beta=1.;
    ccsd_native::blas(&ta,&tb,&bm,&bn,&bk,&alpha,const_cast<double*>(a),&ba,const_cast<double*>(b),&bb,&beta,c,&bc);
}

template<triples_adjoint_kind Kind>
size_t triples_group_direct(const triples_adjoint_group& group,const triples_adjoint_job& job,
        const triples_adjoint_plan& plan,std::vector<tensor>& bank,triples_direct_adjoint& retained,
        const std::vector<dense::Matrix>& main,const std::vector<dense::Matrix>& alternate,
        size_t no,size_t n,std::array<double,3>* profile) {
    constexpr bool gv=Kind==triples_adjoint_kind::gv;
    for(size_t id:group.terms) {
        const auto& term=plan.terms[id];const auto& op=term.op;
        const auto& order=gv ? term.v : term.o;
        const double* c=(order.alternate ? alternate[term.block] : main[term.block]).memptr();
        const auto& source=bank[op.ids[gv ? 1 : 2]];
        const double* a=source.value.memptr()+source.offset(op.indices[gv ? 1 : 2]);
        double* out;
        if constexpr(gv) out=(order.swap_pair ? retained.gv_swap[group.id] : bank[group.id].bar).memptr()
            +group.offset+job.first*n;
        else out=(order.swap_pair ? retained.to_swap[group.id] : retained.to_normal[group.id]).memptr()
            +(group.offset+job.first)*no;
        triples_profile_scope timer(profile ? &(*profile)[1] : nullptr);
        triples_gemm_view('N',order.singleton_first ? 'T' : 'N',gv ? n : no,job.length,n,gv ? 1. : -1.,
            a,gv ? n : source.dims[2]*source.dims[3],
            c+job.first*(order.singleton_first ? 1 : n),order.singleton_first ? n*n : n,out,gv ? n : no);
    }
    return group.terms.size();
}

// These adjoints are small matrices. Retain direct BLAS views of the large
// operands, accumulating into the owned output instead of copying and
// repacking the large reduction dimension solely to combine GEMM calls.
size_t triples_group_small(const triples_adjoint_group& group,const triples_adjoint_plan& plan,
        std::vector<tensor>& bank,const triples_left_pack& packed,
        const std::vector<dense::Matrix>& main,const std::vector<dense::Matrix>& alternate,
        size_t no,size_t n,triples_raw_workspace& workspace,triples_group_scratch& work,
        std::array<double,3>* profile=nullptr) {
    const bool tv=group.kind==triples_adjoint_kind::tv;
    auto& dest=bank[group.id];
    dense::Matrix out(tv ? dest.bar.memptr()+group.offset : work.out.memptr(),tv ? n : no,n,false,true);
    if(!tv) out.zeros();
    size_t calls=0;
    for(size_t id:group.terms) {
        const auto& term=plan.terms[id];const auto& op=term.op;const auto& order=tv ? term.v : term.o;
        const auto& cotangent=order.alternate ? alternate[term.block] : main[term.block];
        dense::Matrix c(const_cast<double*>(cotangent.memptr()),order.singleton_first ? n*n : n,
                     order.singleton_first ? n : n*n,false,true);
        if(tv) {
            auto& source=bank[op.ids[0]];
            const auto& values=order.swap_pair ? packed.value[op.ids[0]] : source.value;
            dense::Matrix g(const_cast<double*>(values.memptr())+source.offset(op.indices[0]),n,n*n,false,true);
            triples_profile_scope timer(profile ? &(*profile)[1] : nullptr);
            if(!packed.occupied_ids.empty()) {
                // Bound the active reduction panels to cache. Both operands
                // remain BLAS views, including the transposed cube case.
                for(size_t first=0;first<n*n;first+=2048) {
                    const size_t length=std::min(size_t(2048),n*n-first);
                    triples_gemm_view('N',order.singleton_first ? 'N' : 'T',n,n,length,1.,
                        g.memptr()+first*n,n,c.memptr()+first*(order.singleton_first ? 1 : n),
                        order.singleton_first ? n*n : n,out.memptr(),n);
                    ++calls;
                }
            } else {
                if(order.singleton_first) out+=g*c;
                else out+=g*c.t();
                ++calls;
            }
        } else {
            auto& source=bank[op.ids[3]];
            const size_t offset=source.offset(op.indices[3]),stride=source.dims[1]*n*n;
            const double* retained=packed.occupied_slice(op,no,n,order.swap_pair);
            {
            triples_profile_scope timer(profile ? &(*profile)[0] : nullptr);
            if(!retained) for(size_t m=0;m<no;++m) {
                const double* src=source.value.memptr()+offset+m*stride;double* target=workspace.u.colptr(m);
                if(!order.swap_pair) std::copy_n(src,n*n,target);
                else for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c) target[b*n+c]=src[c*n+b];
            }
            }
            dense::Matrix u(const_cast<double*>(retained ? retained : workspace.u.memptr()),n*n,no,false,true);
            triples_profile_scope timer(profile ? &(*profile)[1] : nullptr);
            if(order.singleton_first) out-=u.t()*c;
            else out-=u.t()*c.t();
            ++calls;
        }
    }
    if(!tv) {
        triples_profile_scope timer(profile ? &(*profile)[2] : nullptr);
        const size_t stride=dest.dims[2]*dest.dims[3];
        for(size_t a=0;a<n;++a) for(size_t m=0;m<no;++m)
            dest.bar[group.offset+a*stride+m]+=out(m,a);
    }
    return calls;
}
