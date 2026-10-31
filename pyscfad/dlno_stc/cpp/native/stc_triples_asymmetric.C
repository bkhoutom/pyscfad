#include "stc_triples.h"
#include "triples_block_sampler.h"
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <omp.h>

namespace libgmbpt { namespace qchem { namespace {

using index4 = std::array<size_t,4>;
using index6 = std::array<size_t,6>;
enum { T1, T2, GV, GO, GG, FOV, NTYPES };
const size_t permutations[6][3]={{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
const double spin_weights[6]={4,1,1,-2,-2,-2};

struct tensor {
    int kind;
    index4 dims;
    arma::vec value, bar;
    tensor(int k,index4 d,bool gradient) : kind(k),dims(d) {
        size_t length=1;
        for(auto n:d) {
            if(n && length>std::numeric_limits<size_t>::max()/n)
                throw std::overflow_error("asymmetric (T): tensor dimensions overflow");
            length*=n;
        }
        value.zeros(length);
        if(gradient) bar.zeros(length);
    }
    size_t offset(const index4& x) const {
        return ((x[0]*dims[1]+x[1])*dims[2]+x[2])*dims[3]+x[3];
    }
    size_t owners() const { return kind==T2 ? dims[1]*dims[2] : dims[0]*dims[1]; }
    size_t owner(const index4& x) const {
        return kind==T2 ? x[1]*dims[2]+x[2] : x[0]*dims[1]+x[1];
    }
};

// Apply a small nocc x nS projection to one external occupied tensor axis.
// Large contiguous slices use BLAS; short slices are distributed over OpenMP
// workers without invoking a threaded BLAS inside an OpenMP region.
void project_tensor(const tensor& source,tensor& dest,size_t axis,const arma::mat& x) {
    size_t outer=1,inner=1;
    for(size_t a=0;a<axis;++a) outer*=source.dims[a];
    for(size_t a=axis+1;a<4;++a) inner*=source.dims[a];
    const size_t no=x.n_rows,ns=x.n_cols;
    if(inner<4096 && outer>1) {
        #pragma omp parallel for schedule(static)
        for(size_t a=0;a<outer;++a) for(size_t s=0;s<ns;++s) {
            double* out=dest.value.memptr()+(a*ns+s)*inner;
            for(size_t i=0;i<no;++i) {
                const double factor=x(i,s);
                const double* in=source.value.memptr()+(a*no+i)*inner;
                for(size_t b=0;b<inner;++b) out[b]+=factor*in[b];
            }
        }
    } else for(size_t a=0;a<outer;++a) {
        arma::mat in(const_cast<double*>(source.value.memptr())+a*no*inner,inner,no,false,true);
        arma::mat out(dest.value.memptr()+a*ns*inner,inner,ns,false,true);
        out=in*x;
    }
}

void project_pullback(tensor& source,const tensor& dest,size_t axis,
                      const arma::mat& x,arma::mat& xbar) {
    size_t outer=1,inner=1;
    for(size_t a=0;a<axis;++a) outer*=source.dims[a];
    for(size_t a=axis+1;a<4;++a) inner*=source.dims[a];
    const size_t no=x.n_rows,ns=x.n_cols;
    if(inner<4096 && outer>1) {
        std::vector<arma::mat> private_x(static_cast<size_t>(omp_get_max_threads()));
        for(auto& b:private_x) b.zeros(no,ns);
        #pragma omp parallel for schedule(static)
        for(size_t a=0;a<outer;++a) {
            auto& local=private_x[size_t(omp_get_thread_num())];
            for(size_t i=0;i<no;++i) {
                double* inbar=source.bar.memptr()+(a*no+i)*inner;
                const double* in=source.value.memptr()+(a*no+i)*inner;
                for(size_t s=0;s<ns;++s) {
                    const double* outbar=dest.bar.memptr()+(a*ns+s)*inner;
                    double sum=0.;
                    for(size_t b=0;b<inner;++b) {
                        inbar[b]+=x(i,s)*outbar[b];
                        sum+=in[b]*outbar[b];
                    }
                    local(i,s)+=sum;
                }
            }
        }
        for(const auto& b:private_x) xbar+=b;
    } else for(size_t a=0;a<outer;++a) {
        arma::mat in(source.value.memptr()+a*no*inner,inner,no,false,true);
        arma::mat inbar(source.bar.memptr()+a*no*inner,inner,no,false,true);
        arma::mat outbar(const_cast<double*>(dest.bar.memptr())+a*ns*inner,inner,ns,false,true);
        inbar+=outbar*x.t();
        xbar+=in.t()*outbar;
    }
}

struct tuple {
    index6 x;
    size_t selected; // position of the sole S index among the three occupied legs
};
tuple permute(const tuple& x,size_t p,bool pairs) {
    tuple y=x;
    for(size_t d=0;d<3;++d) {
        y.x[d]=x.x[permutations[p][d]];
        if(pairs) y.x[d+3]=x.x[permutations[p][d]+3];
        if(permutations[p][d]==x.selected) y.selected=d;
    }
    return y;
}

struct operations {
    std::array<size_t,8> ids;
    std::array<index4,8> indices;
};

// Factor tables are shared by every permutation/channel using the same
// tensor slice. No six-index table or complete left/right tensor copy exists.
struct factor {
    const tensor* data;
    size_t h;
    int layout; // 0: GV/GO (i,a,b/j); 1: T2(k,c,j); 2: T2(k,c,b), fixed m
    size_t rows,last,block;
    std::unique_ptr<detail::triples_block_sampler> sampler;
    factor(const tensor& t,size_t index,int form) : data(&t),h(index),layout(form) {
        if(layout==0) { rows=t.dims[0]*t.dims[1]; last=t.dims[2]; }
        else if(layout==1) { rows=t.dims[0]*t.dims[2]; last=t.dims[1]; }
        else { rows=t.dims[1]*t.dims[3]; last=t.dims[2]; }
        block=(layout==2 || (layout==0 && t.kind==GV)) ? (last+19)/20 : last;
    }
    double value(size_t index) const {
        const auto& d=data->dims;
        index4 x;
        if(layout==0) x={{index/(d[1]*d[2]),(index/d[2])%d[1],index%d[2],h}};
        else if(layout==1) x={{index/(d[2]*d[1]),index%d[1],(index/d[1])%d[2],h}};
        else x={{h,index/(d[3]*d[2]),index%d[2],(index/d[2])%d[3]}};
        return data->value[data->offset(x)];
    }
    void prepare() {
        sampler.reset(new detail::triples_block_sampler(rows,last,block,
            [&](size_t index) { return value(index); }));
    }
    size_t draw(std::mt19937_64& rng) const {
        return sampler->draw(rng,[&](size_t index) { return value(index); });
    }
    size_t bytes() const { return rows*((last+block-1)/block)*(2*sizeof(double)+sizeof(size_t)); }
};
struct channel {
    size_t left,right,side,position,h;
};
struct moments {
    size_t count=0;
    double mean=0.,m2=0.;
    void add(double value) {
        ++count;
        const double delta=value-mean;
        mean+=delta/double(count);
        m2+=delta*(value-mean);
    }
    void merge(const moments& b) {
        if(!b.count) return;
        const double delta=b.mean-mean;
        const size_t combined=count+b.count;
        m2+=b.m2+delta*delta*(double(count)*double(b.count)/double(combined));
        mean+=delta*(double(b.count)/double(combined));
        count=combined;
    }
    double error() const { return std::sqrt(std::max(0.,m2)/(double(count-1)*double(count))); }
};
struct record { index6 x; double a,z; };

}} // namespace qchem anonymous
namespace qchem {

stc_triples::triples_result stc_triples::asymmetric_triples(
    const arma::field<arma::vec>& t1,const arma::field<arma::mat>& t2,
    const arma::mat& Bvo,const arma::mat& Boo,const arma::mat& Bvv,
    const arma::mat& left,const arma::mat& right,
    size_t nsamples,std::uint64_t seed,double target,
    triples_vjp* vjp,double cotangent,const arma::mat* fov,const pilot_controls* pilot) {
    const size_t no=t1.n_elem,nv=no ? t1(0).n_elem : 0,ns=left.n_cols;
    if(!no || !nv || !ns || ns>no || left.n_rows!=no || right.n_rows!=no || right.n_cols!=ns ||
       !left.is_finite() || !right.is_finite() || Bvo.n_cols!=no*nv || Boo.n_cols!=no*no ||
       Bvv.n_cols!=nv*nv || Bvo.n_rows!=Boo.n_rows || Bvo.n_rows!=Bvv.n_rows ||
       !Bvo.is_finite() || !Boo.is_finite() || !Bvv.is_finite() ||
       t2.n_rows!=no || t2.n_cols!=no || (fov && (fov->n_rows!=no || fov->n_cols!=nv || !fov->is_finite())))
        throw std::invalid_argument("asymmetric (T): inconsistent dimensions or nonfinite inputs");
    if(nsamples==1 || !std::isfinite(target) || target<0 || (target>0 && !nsamples) || !std::isfinite(cotangent))
        throw std::invalid_argument("asymmetric (T): invalid sample count, target or cotangent");
    if(pilot && (pilot->samples<2 || nsamples<2 || pilot->seed==seed ||
                 !std::isfinite(pilot->allocation_target) || pilot->allocation_target<=0))
        throw std::invalid_argument("asymmetric (T): invalid independent pilot");
    const bool gradient=vjp && cotangent!=0., exact=nsamples==0;
    int threads=1;
    #pragma omp parallel
    {
        #pragma omp single
        threads=omp_get_num_threads();
    }
    triples_result result{};
    result.setup_threads=result.sampling_threads=threads;
    const double packing_start=omp_get_wtime();
    std::vector<tensor> bank;
    bank.reserve(32);
    bank.emplace_back(T1,index4{{no,nv,1,1}},gradient);
    bank.emplace_back(T2,index4{{no,no,nv,nv}},gradient);
    bank.emplace_back(GV,index4{{no,nv,nv,nv}},gradient);
    bank.emplace_back(GO,index4{{no,nv,no,no}},gradient);
    bank.emplace_back(GG,index4{{no,nv,no,nv}},gradient);
    bank.emplace_back(FOV,index4{{no,nv,1,1}},gradient);
    for(size_t i=0;i<no;++i) {
        if(t1(i).n_elem!=nv || !t1(i).is_finite())
            throw std::invalid_argument("asymmetric (T): invalid T1");
        for(size_t a=0;a<nv;++a) {
            bank[T1].value[i*nv+a]=t1(i)(a);
            bank[FOV].value[i*nv+a]=fov ? (*fov)(i,a) : 0.;
        }
        for(size_t j=0;j<no;++j) {
            if(t2(i,j).n_rows!=nv || t2(i,j).n_cols!=nv || !t2(i,j).is_finite())
                throw std::invalid_argument("asymmetric (T): invalid T2");
            arma::mat packed(bank[T2].value.memptr()+(i*no+j)*nv*nv,nv,nv,false,true);
            packed=t2(i,j).t();
        }
    }
    {
        arma::mat gv(bank[GV].value.memptr(),nv*nv,no*nv,false,true);
        arma::mat go(bank[GO].value.memptr(),no*no,no*nv,false,true);
        arma::mat gg(bank[GG].value.memptr(),no*nv,no*nv,false,true);
        gv=Bvv.t()*Bvo; go=Boo.t()*Bvo; gg=Bvo.t()*Bvo;
    }
    result.amplitude_packing_seconds=omp_get_wtime()-packing_start;
    const double projection_start=omp_get_wtime();
    size_t projected[2][NTYPES][2];
    struct projection { size_t source,dest,axis,side; };
    std::vector<projection> projections;
    for(size_t side=0;side<2;++side) for(size_t kind=0;kind<NTYPES;++kind)
        for(size_t slot=0;slot<2;++slot) projected[side][kind][slot]=kind;
    for(size_t side=0;side<2;++side) for(size_t kind=0;kind<NTYPES;++kind) {
        if(side==0 && (kind==T1 || kind==GG || kind==FOV)) continue;
        const size_t slots=(kind==T2 || kind==GO || kind==GG) ? 2 : 1;
        for(size_t slot=0;slot<slots;++slot) {
            const size_t axis=slot ? (kind==T2 ? 1 : 2) : 0;
            auto dims=bank[kind].dims; dims[axis]=ns;
            const size_t id=bank.size();
            bank.emplace_back(int(kind),dims,gradient);
            projected[side][kind][slot]=id;
            projections.push_back({kind,id,axis,side});
            project_tensor(bank[kind],bank[id],axis,side ? right : left);
        }
    }
    result.projection_setup_seconds=omp_get_wtime()-projection_start;
    for(const auto& b:bank) result.resident_tensor_bytes+=(b.value.n_elem+b.bar.n_elem)*sizeof(double);
    auto selected_id=[&](size_t side,size_t kind,int slot) {
        return slot<0 ? kind : projected[side][kind][size_t(slot)];
    };
    auto describe=[&](const tuple& t,size_t side) {
        const auto& x=t.x;
        const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5],s=t.selected;
        operations op;
        op.ids={{selected_id(side,GV,s==0 ? 0 : -1),
                 selected_id(side,T2,s==2 ? 0 : (s==1 ? 1 : -1)),
                 selected_id(side,GO,s==0 ? 0 : (s==1 ? 1 : -1)),
                 selected_id(side,T2,s==2 ? 1 : -1),
                 selected_id(side,GG,s==0 ? 0 : (s==1 ? 1 : -1)),
                 selected_id(side,T1,s==2 ? 0 : -1),
                 selected_id(side,FOV,s==2 ? 0 : -1),
                 selected_id(side,T2,s==0 ? 0 : (s==1 ? 1 : -1))}};
        op.indices={{{{i,a,b,0}},{{k,j,c,0}},{{i,a,j,0}},{{0,k,b,c}},
                     {{i,a,j,b}},{{k,c,0,0}},{{k,c,0,0}},{{i,j,a,b}}}};
        return op;
    };
    auto wq=[&](const operations& op,bool probability) {
        std::array<double,2> out{{0.,0.}};
        const double* g=bank[op.ids[0]].value.memptr()+bank[op.ids[0]].offset(op.indices[0]);
        const double* t=bank[op.ids[1]].value.memptr()+bank[op.ids[1]].offset(op.indices[1]);
        for(size_t f=0;f<nv;++f) { const double v=g[f]*t[f]; out[0]+=v; if(probability) out[1]+=v*v; }
        g=bank[op.ids[2]].value.memptr()+bank[op.ids[2]].offset(op.indices[2]);
        const auto& tb=bank[op.ids[3]];
        t=tb.value.memptr()+tb.offset(op.indices[3]);
        const size_t stride=tb.dims[1]*nv*nv;
        for(size_t m=0;m<no;++m) { const double v=g[m]*t[m*stride]; out[0]-=v; if(probability) out[1]+=v*v; }
        return out;
    };
    auto scalar=[&](const operations& op,size_t role) {
        const auto& b=bank[op.ids[role]];
        return b.value[b.offset(op.indices[role])];
    };
    auto evaluate=[&](const index6& x) {
        std::array<double,4> out{{0.,0.,0.,0.}}; // A_left, Z_right, q_left, q_right
        for(size_t p=0;p<6;++p) {
            const auto y=permute(tuple{x,0},p,true);
            const auto left_w=wq(describe(y,0),true);
            out[0]+=left_w[0]; out[2]+=left_w[1];
            for(size_t r=0;r<6;++r) {
                const auto op=describe(permute(y,r,false),1);
                const auto right_w=wq(op,r==0);
                if(r==0) out[3]+=right_w[1];
                const double v=scalar(op,4)*scalar(op,5)+scalar(op,6)*scalar(op,7);
                out[1]+=spin_weights[r]*(right_w[0]+.5*v);
            }
        }
        return out;
    };

    const double probability_start=omp_get_wtime();
    std::vector<std::unique_ptr<factor>> factors;
    std::map<std::tuple<size_t,size_t,int>,size_t> factor_ids;
    std::vector<channel> channels;
    auto factor_id=[&](size_t id,size_t h,int layout) {
        const auto key=std::make_tuple(id,h,layout);
        auto found=factor_ids.find(key);
        if(found!=factor_ids.end()) return found->second;
        const size_t index=factors.size();
        factors.emplace_back(new factor(bank[id],h,layout));
        factor_ids.emplace(key,index);
        return index;
    };
    if(!exact) for(size_t side=0;side<2;++side) for(size_t position=0;position<3;++position) {
        index6 zero{{0,0,0,0,0,0}};
        const auto op=describe(tuple{zero,position},side);
        for(size_t h=0;h<nv+no;++h) {
            const bool virt=h<nv;
            const size_t internal=virt ? h : h-nv;
            channels.push_back({factor_id(op.ids[virt ? 0 : 2],internal,0),
                factor_id(op.ids[virt ? 1 : 3],internal,virt ? 1 : 2),side,position,h});
        }
    }
    std::exception_ptr error;
    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0;i<factors.size();++i) {
        try { factors[i]->prepare(); }
        catch(...) {
            #pragma omp critical(asymmetric_triples_error)
            { if(!error) error=std::current_exception(); }
        }
    }
    if(error) std::rethrow_exception(error);
    std::array<double,2> norms{{0.,0.}};
    std::vector<double> masses(channels.size());
    for(size_t c=0;c<channels.size();++c) {
        const auto& h=channels[c];
        masses[c]=factors[h.left]->sampler->total*factors[h.right]->sampler->total;
        norms[h.side]+=2*masses[c]; // two pair permutations for each selected position
    }
    if(!std::isfinite(norms[0]+norms[1])) throw std::runtime_error("asymmetric (T): proposal overflow");
    const double live=double(norms[0]>0)+double(norms[1]>0);
    for(size_t c=0;c<channels.size();++c)
        masses[c]=norms[channels[c].side]>0 ? masses[c]*2/(norms[channels[c].side]*live) : 0.;
    const double uniform_fraction=live>0 ? .05 : 1.;
    std::discrete_distribution<size_t> channel_draw(masses.begin(),masses.end());
    std::array<std::array<size_t,2>,3> pair_choices;
    for(size_t position=0;position<3;++position) {
        size_t n=0;
        for(size_t p=0;p<6;++p) if(permutations[p][position]==0) pair_choices[position][n++]=p;
    }
    for(const auto& f:factors) result.proposal_bytes+=f->bytes();
    result.probability_setup_seconds=omp_get_wtime()-probability_start;

    std::vector<size_t> owner_start(bank.size()+1,0);
    for(size_t id=0;id<bank.size();++id) owner_start[id+1]=owner_start[id]+bank[id].owners();
    auto term=[&](const record& sample,size_t number) {
        const size_t side=number<6 ? 0 : 1;
        const size_t p=side ? (number-6)/6 : number;
        const size_t r=side ? (number-6)%6 : 0;
        auto t=permute(tuple{sample.x,0},p,true);
        if(side) t=permute(t,r,false);
        return describe(t,side);
    };
    // Compile permutation, projection, stride and sign metadata once per LIS.
    // An event then needs only two linear offsets, not eight rebuilt operands.
    struct update_layout {
        index6 output_stride{}, input_stride{}, owner_stride{};
        double* output=nullptr;
        const double* input=nullptr;
        size_t owner_base=0, length=0, output_step=0, input_step=0;
        double coefficient=0.;
        bool left=false;
    };
    std::array<update_layout,336> layouts;
    std::vector<size_t> active_layouts;
    const record zero{{{0,0,0,0,0,0}},0.,0.};
    for(size_t t=0;t<42;++t) {
        const auto op=term(zero,t);
        for(size_t role=0;role<(t<6 ? 4 : 8);++role) {
            const size_t id=t*8+role,partner=role^1;
            auto& layout=layouts[id];
            auto& dest=bank[op.ids[role]];
            const auto& source=bank[op.ids[partner]];
            layout.output=dest.bar.memptr(); layout.input=source.value.memptr();
            layout.owner_base=owner_start[op.ids[role]];
            layout.left=t<6;
            layout.coefficient=(t<6 ? 1. : spin_weights[(t-6)%6])
                *(role>=4 ? .5 : (role>=2 ? -1. : 1.));
            layout.length=role<2 ? nv : (role<4 ? no : 1);
            layout.output_step=role==3 ? dest.dims[1]*nv*nv : 1;
            layout.input_step=role==2 ? source.dims[1]*nv*nv : 1;
            for(size_t d=0;d<6;++d) {
                record unit=zero; unit.x[d]=1;
                const auto basis=term(unit,t);
                layout.output_stride[d]=dest.offset(basis.indices[role]);
                layout.input_stride[d]=source.offset(basis.indices[partner]);
                layout.owner_stride[d]=dest.owner(basis.indices[role]);
            }
            active_layouts.push_back(id);
        }
    }
    auto offset=[](const index6& x,const index6& stride) {
        return x[0]*stride[0]+x[1]*stride[1]+x[2]*stride[2]
             + x[3]*stride[3]+x[4]*stride[4]+x[5]*stride[5];
    };
    auto reverse=[&](const std::vector<record>& records) {
        if(records.empty()) return;
        const double start=omp_get_wtime();
        if(records.size()>std::numeric_limits<std::uint32_t>::max()/336)
            throw std::overflow_error("asymmetric (T): reverse event IDs overflow");
        const size_t owners=owner_start.back();
        // Bounded private histograms contain integers only. Each partition is
        // a contiguous sample range, preserving the old per-owner sum order.
        const size_t partitions=std::min(size_t(threads),std::max(size_t(1),
            size_t(64*1024*1024)/(owners*sizeof(std::uint32_t))));
        std::vector<std::uint32_t> cursor(partitions*owners,0);
        #pragma omp parallel for schedule(static)
        for(size_t part=0;part<partitions;++part) {
            auto* counts=cursor.data()+part*owners;
            for(size_t n=records.size()*part/partitions;n<records.size()*(part+1)/partitions;++n)
                for(auto id:active_layouts) {
                    const auto& layout=layouts[id];
                    ++counts[layout.owner_base+offset(records[n].x,layout.owner_stride)];
                }
        }
        std::vector<size_t> offsets(owners+1,0);
        for(size_t owner=0;owner<owners;++owner) {
            size_t end=offsets[owner];
            for(size_t part=0;part<partitions;++part) {
                auto& value=cursor[part*owners+owner];
                const size_t count=value; value=std::uint32_t(end); end+=count;
            }
            offsets[owner+1]=end;
        }
        std::vector<std::uint32_t> events(offsets.back());
        #pragma omp parallel for schedule(static)
        for(size_t part=0;part<partitions;++part) {
            auto* positions=cursor.data()+part*owners;
            for(size_t n=records.size()*part/partitions;n<records.size()*(part+1)/partitions;++n)
                for(auto id:active_layouts) {
                    const auto& layout=layouts[id];
                    const size_t owner=layout.owner_base+offset(records[n].x,layout.owner_stride);
                    events[positions[owner]++]=std::uint32_t(n*336+id);
                }
        }
        const double grouped=omp_get_wtime();
        result.vjp_grouping_seconds+=grouped-start;
        result.reverse_buffer_bytes=std::max(result.reverse_buffer_bytes,
            records.size()*sizeof(record)+offsets.size()*sizeof(size_t)
            +(cursor.size()+events.size())*sizeof(std::uint32_t));
        // Each tensor output tile has one writer. T2 is partitioned by (j,a),
        // including occupied contractions spanning i. Full gradients are shared.
        #pragma omp parallel for schedule(dynamic)
        for(size_t owner=0;owner<owners;++owner) {
            for(size_t e=offsets[owner];e<offsets[owner+1];++e) {
                const size_t id=events[e],n=id/336;
                const auto& layout=layouts[id%336];
                const auto& sample=records[n];
                const double weight=(layout.left ? sample.a : sample.z)*layout.coefficient;
                double* out=layout.output+offset(sample.x,layout.output_stride);
                const double* in=layout.input+offset(sample.x,layout.input_stride);
                if(layout.input_step==1 && layout.output_step==1) {
                    #pragma omp simd
                    for(size_t k=0;k<layout.length;++k) out[k]+=weight*in[k];
                } else if(layout.output_step==1) {
                    #pragma omp simd
                    for(size_t k=0;k<layout.length;++k) out[k]+=weight*in[k*layout.input_step];
                } else for(size_t k=0;k<layout.length;++k) out[k*layout.output_step]+=weight*in[k];
            }
        }
        result.vjp_accumulation_seconds+=omp_get_wtime()-grouped;
        result.vjp_reverse_seconds+=omp_get_wtime()-start;
    };

    const double volume=double(ns)*double(no)*double(no)*double(nv)*double(nv)*double(nv);
    if(exact && volume>=double(std::numeric_limits<size_t>::max()))
        throw std::overflow_error("asymmetric (T): exact enumeration too large");
    const size_t batch_size=2048;
    auto stream=[&](size_t count,std::uint64_t stream_seed,double tolerance,bool derivatives,bool enumerate) {
        moments all;
        size_t round=count,batch_offset=0;
        double first_error=0.;
        do {
            const size_t nbatches=(round-1)/batch_size+1;
            std::vector<moments> batches(nbatches);
            for(size_t first=0;first<nbatches;first+=size_t(threads)) {
                const size_t nblocks=std::min(size_t(threads),nbatches-first);
                const size_t ndraws=std::min(round-first*batch_size,nblocks*batch_size);
                std::vector<record> records(derivatives ? ndraws : 0);
                #pragma omp parallel for schedule(dynamic)
                for(size_t b=0;b<nblocks;++b) {
                    try {
                        const size_t batch=first+b;
                        std::uint64_t state=stream_seed+0x9e3779b97f4a7c15ULL*(batch_offset+batch+1);
                        state=(state^(state>>30))*0xbf58476d1ce4e5b9ULL;
                        state=(state^(state>>27))*0x94d049bb133111ebULL;
                        std::mt19937_64 rng(state^(state>>31));
                        auto draw=channel_draw;
                        std::uniform_real_distribution<double> uniform(0,1);
                        std::uniform_int_distribution<size_t> occ(0,no-1),vir(0,nv-1),selected(0,ns-1),binary(0,1);
                        const size_t draws=std::min(batch_size,round-batch*batch_size);
                        for(size_t n=0;n<draws;++n) {
                            index6 x;
                            if(enumerate) {
                                size_t index=batch*batch_size+n;
                                x[5]=index%nv; index/=nv; x[4]=index%nv; index/=nv;
                                x[3]=index%nv; index/=nv; x[2]=index%no; index/=no;
                                x[1]=index%no; index/=no; x[0]=index;
                            } else if(uniform(rng)<uniform_fraction) {
                                x={{selected(rng),occ(rng),occ(rng),vir(rng),vir(rng),vir(rng)}};
                            } else {
                                const auto& ch=channels[draw(rng)];
                                const size_t l=factors[ch.left]->draw(rng),r=factors[ch.right]->draw(rng);
                                const size_t ni=ch.position==0 ? ns : no;
                                const size_t nj=ch.position==1 ? ns : no;
                                const size_t nk=ch.position==2 ? ns : no;
                                (void)ni; (void)nk;
                                index6 y;
                                if(ch.h<nv) y={{l/(nv*nv),r%nj,r/(nv*nj),(l/nv)%nv,l%nv,(r/nj)%nv}};
                                else y={{l/(nv*nj),l%nj,r/(nv*nv),(l/nj)%nv,r%nv,(r/nv)%nv}};
                                const size_t p=pair_choices[ch.position][binary(rng)];
                                for(size_t d=0;d<3;++d) { x[permutations[p][d]]=y[d]; x[3+permutations[p][d]]=y[3+d]; }
                            }
                            const auto values=evaluate(x);
                            double probability=uniform_fraction/volume;
                            if(live>0) for(size_t side=0;side<2;++side)
                                if(norms[side]>0) probability+=(1-uniform_fraction)*values[2+side]/(norms[side]*live);
                            const double weight=enumerate ? -1./3 : -1./(3*probability);
                            const double value=weight*values[0]*values[1];
                            if(!std::isfinite(value)) throw std::runtime_error("asymmetric (T): nonfinite sample");
                            batches[batch].add(value);
                            if(derivatives) records[b*batch_size+n]={x,cotangent*weight*values[1],cotangent*weight*values[0]};
                        }
                    } catch(...) {
                        #pragma omp critical(asymmetric_triples_error)
                        { if(!error) error=std::current_exception(); }
                    }
                }
                if(error) std::rethrow_exception(error);
                reverse(records);
            }
            for(const auto& batch:batches) all.merge(batch);
            const double se=enumerate ? 0. : all.error();
            if(!batch_offset) first_error=se;
            batch_offset+=nbatches;
            if(!tolerance || se<=tolerance) break;
            const double ratio=se/tolerance;
            const double extra=std::ceil(double(all.count)*(ratio*ratio-1));
            round=size_t(std::max(double(batch_size),std::min(1000000.,extra)));
            if(all.count>std::numeric_limits<size_t>::max()-round)
                throw std::overflow_error("asymmetric (T): sample count overflow");
        } while(true);
        triples_result out{};
        out.energy=enumerate ? all.mean*double(all.count) : all.mean;
        out.standard_error=enumerate ? 0. : all.error();
        out.samples=enumerate ? 0 : all.count;
        out.pilot_samples=enumerate ? 0 : count;
        out.pilot_standard_error=first_error;
        return out;
    };
    if(pilot) {
        const double start=omp_get_wtime();
        const auto trial=stream(pilot->samples,pilot->seed,0.,false,false);
        result.independent_pilot_samples=trial.samples;
        result.independent_pilot_standard_error=trial.standard_error;
        const double ratio=trial.standard_error*std::sqrt(double(trial.samples))/pilot->allocation_target;
        const double required=std::ceil(ratio*ratio);
        if(!std::isfinite(required) || required>=double(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("asymmetric (T): pilot allocation exceeds int64 capacity");
        nsamples=std::max(nsamples,size_t(required));
        result.independent_pilot_seconds=omp_get_wtime()-start;
    }
    const double sampling_start=omp_get_wtime();
    const auto production=stream(exact ? size_t(volume) : nsamples,seed,target,gradient,exact);
    result.energy=production.energy; result.standard_error=production.standard_error;
    result.samples=production.samples; result.pilot_samples=production.pilot_samples;
    result.pilot_standard_error=production.pilot_standard_error;
    if(vjp) {
        const double start=omp_get_wtime();
        *vjp=triples_vjp();
        vjp->occupied_left.zeros(no,ns); vjp->occupied_right.zeros(no,ns);
        if(gradient) {
            const double scale=exact ? 1. : 1./double(result.samples);
            for(auto& b:bank) b.bar*=scale;
            for(const auto& p:projections)
                project_pullback(bank[p.source],bank[p.dest],p.axis,p.side ? right : left,
                                 p.side ? vjp->occupied_right : vjp->occupied_left);
        } else for(auto& b:bank) b.bar.zeros(b.value.n_elem);
        vjp->t1.set_size(no); vjp->t2.set_size(no,no); vjp->fov.zeros(no,nv);
        for(size_t i=0;i<no;++i) {
            vjp->t1(i)=bank[T1].bar.subvec(i*nv,(i+1)*nv-1);
            for(size_t a=0;a<nv;++a) vjp->fov(i,a)=bank[FOV].bar[i*nv+a];
            for(size_t j=0;j<no;++j) {
                arma::mat packed(bank[T2].bar.memptr()+(i*no+j)*nv*nv,nv,nv,false,true);
                vjp->t2(i,j)=packed.t();
            }
        }
        arma::mat gvbar(bank[GV].bar.memptr(),nv*nv,no*nv,false,true);
        arma::mat gobar(bank[GO].bar.memptr(),no*no,no*nv,false,true);
        arma::mat ggbar(bank[GG].bar.memptr(),no*nv,no*nv,false,true);
        vjp->Bvv=Bvo*gvbar.t(); vjp->Boo=Bvo*gobar.t();
        vjp->Bvo=Bvv*gvbar+Boo*gobar+Bvo*(ggbar+ggbar.t());
        result.vjp_finalize_seconds=omp_get_wtime()-start;
        result.vjp_reverse_seconds+=result.vjp_finalize_seconds;
    }
    result.sampling_seconds=omp_get_wtime()-sampling_start;
    return result;
}

}} // namespace libgmbpt::qchem
