// Bounded occupied-block batches. Each output tile has one owner, so no
// adjoint GEMM or commit below needs a lock. The large GV and T2O adjoints
// concatenate contributions along the reduction dimension; packing panels
// are fixed-size, not another batch of virtual cubes.
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
constexpr size_t triples_adjoint_tile=256,triples_adjoint_reduction=4096;

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
    arma::vec a,b,out;
    struct segment {size_t term,first,length,offset;};
    std::vector<segment> segments;
    void prepare(size_t no,size_t nv) {
        const size_t extent=std::max({no,nv,triples_adjoint_tile});
        a.set_size(extent*triples_adjoint_reduction);
        b.set_size(extent*triples_adjoint_reduction);
        out.set_size(std::max(no,nv)*std::max(nv,triples_adjoint_tile));
    }
    static size_t bytes(size_t no,size_t nv) {
        const size_t extent=std::max({no,nv,triples_adjoint_tile});
        return (2*extent*triples_adjoint_reduction+
                std::max(no,nv)*std::max(nv,triples_adjoint_tile))*sizeof(double);
    }
};

// Normalized GV output columns and T2O output rows refer to the original
// input frame. Pair permutations are absorbed into the packed cotangent,
// avoiding both a packed rank-four bar and a final whole-bank transpose.
template<triples_adjoint_kind Kind>
size_t triples_group_large(const triples_adjoint_group& group,const triples_adjoint_job& job,
        const triples_adjoint_plan& plan,std::vector<tensor>& bank,
        const std::vector<arma::vec>& main,const std::vector<arma::vec>& alternate,
        size_t no,size_t n,triples_group_scratch& work,size_t& largest_reduction) {
    constexpr bool gv=Kind==triples_adjoint_kind::gv;
    const size_t nr=gv ? n : job.length,nc=gv ? job.length : no;
    auto& dest=bank[group.id];
    double* output=gv ? dest.bar.memptr()+group.offset+job.first*n : work.out.memptr();
    arma::mat out(output,nr,nc,false,true);
    if constexpr(!gv) out.zeros();
    size_t next=0,kfirst=0,calls=0;
    while(next<group.terms.size()) {
        size_t width=0;work.segments.clear();
        while(next<group.terms.size() && width<triples_adjoint_reduction) {
            const size_t length=std::min(n-kfirst,triples_adjoint_reduction-width);
            work.segments.push_back({group.terms[next],kfirst,length,width});
            width+=length;kfirst+=length;
            if(kfirst==n) {++next;kfirst=0;}
        }
        largest_reduction=std::max(largest_reduction,width);
        arma::mat a(work.a.memptr(),nr,width,false,true),b(work.b.memptr(),width,nc,false,true);
        for(const auto& s:work.segments) {
            const auto& term=plan.terms[s.term];const auto& op=term.op;
            const auto& order=gv ? term.v : term.o;
            const double* c=(order.alternate ? alternate[term.block] : main[term.block]).memptr();
            if constexpr(gv) {
                const auto& tv=bank[op.ids[1]];
                const double* t=tv.value.memptr()+tv.offset(op.indices[1]);
                std::copy_n(t+s.first*n,s.length*n,a.colptr(s.offset));
                for(size_t j=0;j<nc;++j) {
                    const size_t pair=job.first+j;
                    const size_t permuted=order.swap_pair ? (pair%n)*n+pair/n : pair;
                    double* target=b.colptr(j)+s.offset;
                    if(!order.singleton_first) std::copy_n(c+permuted*n+s.first,s.length,target);
                    else for(size_t k=0;k<s.length;++k) target[k]=c[(s.first+k)*n*n+permuted];
                }
            } else {
                const auto& go=bank[op.ids[2]];
                const double* h=go.value.memptr()+go.offset(op.indices[2]);
                const size_t stride=go.dims[2]*go.dims[3];
                for(size_t k=0;k<s.length;++k) {
                    double* target=a.colptr(s.offset+k);
                    for(size_t i=0;i<nr;++i) {
                        const size_t pair=job.first+i;
                        const size_t permuted=order.swap_pair ? (pair%n)*n+pair/n : pair;
                        target[i]=order.singleton_first ? c[(s.first+k)*n*n+permuted]
                                                       : c[permuted*n+s.first+k];
                    }
                }
                for(size_t j=0;j<nc;++j) for(size_t k=0;k<s.length;++k)
                    b(s.offset+k,j)=h[(s.first+k)*stride+j];
            }
        }
        if constexpr(gv) out+=a*b;
        else out-=a*b;
        ++calls;
    }
    if constexpr(!gv) {
        const size_t stride=dest.dims[1]*n*n;
        for(size_t m=0;m<no;++m) {
            double* target=dest.bar.memptr()+group.offset+m*stride+job.first;
            const double* source=out.colptr(m);
            #pragma omp simd
            for(size_t i=0;i<nr;++i) target[i]+=source[i];
        }
    }
    return calls;
}

// These adjoints are small matrices. Retain direct BLAS views of the large
// operands, accumulating into the owned output instead of copying and
// repacking the large reduction dimension solely to combine GEMM calls.
size_t triples_group_small(const triples_adjoint_group& group,const triples_adjoint_plan& plan,
        std::vector<tensor>& bank,const triples_left_pack& packed,
        const std::vector<arma::vec>& main,const std::vector<arma::vec>& alternate,
        size_t no,size_t n,triples_raw_workspace& workspace,triples_group_scratch& work) {
    const bool tv=group.kind==triples_adjoint_kind::tv;
    auto& dest=bank[group.id];
    arma::mat out(tv ? dest.bar.memptr()+group.offset : work.out.memptr(),tv ? n : no,n,false,true);
    if(!tv) out.zeros();
    for(size_t id:group.terms) {
        const auto& term=plan.terms[id];const auto& op=term.op;const auto& order=tv ? term.v : term.o;
        const auto& cotangent=order.alternate ? alternate[term.block] : main[term.block];
        arma::mat c(const_cast<double*>(cotangent.memptr()),order.singleton_first ? n*n : n,
                     order.singleton_first ? n : n*n,false,true);
        if(tv) {
            auto& source=bank[op.ids[0]];
            const auto& values=order.swap_pair ? packed.value[op.ids[0]] : source.value;
            arma::mat g(const_cast<double*>(values.memptr())+source.offset(op.indices[0]),n,n*n,false,true);
            if(order.singleton_first) out+=g*c;
            else out+=g*c.t();
        } else {
            auto& source=bank[op.ids[3]];
            const size_t offset=source.offset(op.indices[3]),stride=source.dims[1]*n*n;
            for(size_t m=0;m<no;++m) {
                const double* src=source.value.memptr()+offset+m*stride;double* target=workspace.u.colptr(m);
                if(!order.swap_pair) std::copy_n(src,n*n,target);
                else for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c) target[b*n+c]=src[c*n+b];
            }
            if(order.singleton_first) out-=workspace.u.t()*c;
            else out-=workspace.u.t()*c.t();
        }
    }
    if(!tv) {
        const size_t stride=dest.dims[2]*dest.dims[3];
        for(size_t a=0;a<n;++a) for(size_t m=0;m<no;++m)
            dest.bar[group.offset+a*stride+m]+=out(m,a);
    }
    return group.terms.size();
}
