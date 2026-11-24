// Copyright 2026 The PySCFAD Authors (Apache-2.0).
#pragma once
#include "residual.h"
#include <algorithm>
#include <array>
#include <climits>
#include <ctime>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <omp.h>

namespace py = pybind11;
namespace ccsd_native {
using Index=py::ssize_t;
using Input=py::array_t<double,0>;
using Array=ResidualArray;
extern ResidualGemm blas;
// Native owned buffers and borrowed views. No Python objects or callbacks are
// touched while evaluating the equations. Views share ownership of temporaries.
struct T {
    std::shared_ptr<double> owner;
    double* p = nullptr;
    int rank = 0;
    std::array<Index,4> n{{1,1,1,1}}, s{{0,0,0,0}};
    T() = default;
    explicit T(const Input& a) : p(const_cast<double*>(a.data())), rank(a.ndim()) {
        if (rank < 1 || rank > 4) throw std::invalid_argument("expected rank 1 through 4");
        if (reinterpret_cast<uintptr_t>(p)%alignof(double))
            throw std::invalid_argument("unaligned tensor data");
        for (int d=0; d<rank; ++d) {
            n[d]=a.shape(d);
            if (a.strides(d)%sizeof(double)) throw std::invalid_argument("unaligned tensor stride");
            s[d]=a.strides(d)/Index(sizeof(double));
        }
    }
    explicit T(const std::vector<Index>& shape,const std::shared_ptr<double>& storage={}) : rank(shape.size()) {
        if (rank < 1 || rank > 4) throw std::invalid_argument("invalid tensor rank");
        Index count=1;
        for (int d=rank-1; d>=0; --d) {
            n[d]=shape[d]; s[d]=count;
            if (n[d]<0 || (n[d] && count>PTRDIFF_MAX/n[d]/Index(sizeof(double))))
                throw std::overflow_error("tensor size overflow");
            count*=n[d];
        }
        owner=storage?storage:std::shared_ptr<double>(new double[count], std::default_delete<double[]>());
        p=owner.get();
    }
    Index size() const { Index z=1; for (int d=0;d<rank;++d) z*=n[d]; return z; }
    std::vector<Index> shape() const { return {n.begin(),n.begin()+rank}; }
    bool contiguous() const {
        Index stride=1;
        for (int d=rank-1;d>=0;--d) { if(n[d]>1 && s[d]!=stride)return false; stride*=n[d]; }
        return true;
    }
    Index offset(Index row) const {
        Index at=0;
        for (int d=rank-2;d>=0;--d) { at+=(row%n[d])*s[d]; row/=n[d]; }
        return at;
    }
    T perm(const std::vector<int>& axes) const {
        if (int(axes.size())!=rank) throw std::invalid_argument("invalid permutation");
        T v=*this;
        std::array<bool,4> used{};
        for (int d=0;d<rank;++d) {
            int a=axes[d];
            if(a<0 || a>=rank || used[a])throw std::invalid_argument("invalid permutation");
            used[a]=true; v.n[d]=n[a];v.s[d]=s[a];
        }
        return v;
    }
    T slice(int axis, Index lo, Index hi) const {
        if(axis<0 || axis>=rank || lo<0 || hi<lo || hi>n[axis])
            throw std::invalid_argument("invalid tensor slice");
        T v=*this;v.p+=lo*s[axis];v.n[axis]=hi-lo;return v;
    }
};
inline void shape_is(const T& a, const std::vector<Index>& expected) {
    if (a.shape()!=expected) throw std::invalid_argument("inconsistent CCSD tensor shapes");
}
inline void fill(T a, double value) {
    if(!a.contiguous())throw std::logic_error("fill requires contiguous storage");
    const Index count=a.size();
    #pragma omp parallel for schedule(static) if(count>4096)
    for(Index i=0;i<count;++i)a.p[i]=value;
}
inline T copy(const T& a, double alpha=1.,const T& buffer=T()) {
    T r=buffer.p?buffer:T(a.shape());shape_is(r,a.shape());
    const Index last=a.n[a.rank-1], count=a.size();
    if (!count) return r;
    #pragma omp parallel for schedule(static) if(count>4096)
    for(Index row=0;row<count/last;++row) {
        const auto off=a.offset(row),stride=a.s[a.rank-1];
        #pragma omp simd
        for(Index k=0;k<last;++k)r.p[row*last+k]=alpha*a.p[off+k*stride];
    }
    return r;
}
inline void add(T dst,const T& source,double alpha=1.,double beta=1.) {
    shape_is(source,dst.shape());
    // Traverse destination storage in physical order. In particular, wVooV
    // and contraction-result views need not be contiguous in equation order.
    std::vector<int> axes(dst.rank);std::iota(axes.begin(),axes.end(),0);
    std::stable_sort(axes.begin(),axes.end(),[&](int a,int b){return std::abs(dst.s[a])>std::abs(dst.s[b]);});
    T src=source.perm(axes);dst=dst.perm(axes);
    const Index last=dst.n[dst.rank-1],count=dst.size();
    if(!count)return;
    #pragma omp parallel for schedule(static) if(count>4096)
    for(Index row=0;row<count/last;++row) {
        const auto x=dst.offset(row),y=src.offset(row);
        const auto ds=dst.s[dst.rank-1],ss=src.s[src.rank-1];
        #pragma omp simd
        for(Index k=0;k<last;++k)dst.p[x+k*ds]=beta==0.?alpha*src.p[y+k*ss]:beta*dst.p[x+k*ds]+alpha*src.p[y+k*ss];
    }
}
inline T linear(const T& a,const T& b,double alpha,double beta,const T& buffer=T()) {
    shape_is(b,a.shape());T r=buffer.p?buffer:T(a.shape());shape_is(r,a.shape());
    const Index last=a.n[a.rank-1],count=a.size();
    if(!count)return r;
    #pragma omp parallel for schedule(static) if(count>4096)
    for(Index row=0;row<count/last;++row) {
        const auto x=a.offset(row),y=b.offset(row);
        #pragma omp simd
        for(Index k=0;k<last;++k)
            r.p[row*last+k]=alpha*a.p[x+k*a.s[a.rank-1]]+beta*b.p[y+k*b.s[b.rank-1]];
    }
    return r;
}
inline T tau(const T& t,const T& l,const T& r,double alpha=1.,double beta=1.,const T& buffer=T()) {
    shape_is(t,{l.n[0],r.n[0],l.n[1],r.n[1]});T out=buffer.p?buffer:T(t.shape());shape_is(out,t.shape());
    const Index last=t.n[3],rows=t.n[0]*t.n[1]*t.n[2];
    #pragma omp parallel for schedule(static) if(rows*last>4096)
    for(Index row=0;row<rows;++row) {
        const Index a=row%t.n[2],j=(row/t.n[2])%t.n[1],i=row/(t.n[2]*t.n[1]);
        const auto off=t.offset(row);const double x=beta*l.p[i*l.s[0]+a*l.s[1]];
        #pragma omp simd
        for(Index b=0;b<last;++b)out.p[row*last+b]=alpha*t.p[off+b*t.s[3]]+x*r.p[j*r.s[0]+b*r.s[1]];
    }
    return out;
}
// Form the exchange-weighted half-tau directly, without writing tau first.
inline T theta_fock(const T& theta,const T& left,const T& right) {
    T out(theta.shape());const Index o=theta.n[0],a=theta.n[2],v=theta.n[3];
    shape_is(theta,{o,o,a,v});shape_is(left,{o,a});shape_is(right,{o,v});
    #pragma omp parallel for schedule(static) if(out.size()>4096)
    for(Index row=0;row<o*o*a;++row) {
        Index x=row%a,j=(row/a)%o,i=row/(a*o),off=theta.offset(row);
        double lji=left.p[j*left.s[0]+x*left.s[1]],lij=.5*left.p[i*left.s[0]+x*left.s[1]];
        #pragma omp simd
        for(Index b=0;b<v;++b)out.p[row*v+b]=theta.p[off+b*theta.s[3]]
            +lji*right.p[i*right.s[0]+b*right.s[1]]-lij*right.p[j*right.s[0]+b*right.s[1]];
    }
    return out;
}
inline void add_exchange(T dst,const T& a,const T& b,double beta) {
    shape_is(a,dst.shape());shape_is(b,dst.shape());
    const Index last=dst.n[3],rows=dst.n[0]*dst.n[1]*dst.n[2];
    #pragma omp parallel for schedule(static) if(rows*last>4096)
    for(Index row=0;row<rows;++row) {
        auto x=dst.offset(row),y=a.offset(row),z=b.offset(row);
        #pragma omp simd
        for(Index k=0;k<last;++k)dst.p[x+k*dst.s[3]]+=a.p[y+k*a.s[3]]+beta*b.p[z+k*b.s[3]];
    }
}

inline Index pair(Index a,Index b) {auto hi=std::max(a,b),lo=std::min(a,b);return hi*(hi+1)/2+lo;}
inline T unpack(const T& packed,Index v) {
    if(packed.rank!=3 || packed.n[2]!=v*(v+1)/2)throw std::invalid_argument("invalid packed ovvv");
    T out({packed.n[0],packed.n[1],v,v});
    #pragma omp parallel for schedule(static) if(out.size()>4096)
    for(Index row=0;row<packed.n[0]*packed.n[1]*v;++row) {
        const Index a=row%v,j=(row/v)%packed.n[1],i=row/(v*packed.n[1]);
        const auto off=i*packed.s[0]+j*packed.s[1];
        for(Index b=0;b<v;++b)out.p[row*v+b]=packed.p[off+pair(a,b)*packed.s[2]];
    }
    return out;
}

// Packing and result lifetimes inside one accumulating contraction do not
// overlap the next contraction. Reserve one backing allocation, then carve
// non-overlapping views from it. Escaping E() results own separate storage;
// with_result() consumers finish before this scratch can be reused.
struct Scratch {
    std::shared_ptr<double> owner;
    Index capacity=0,used=0,phase_peak=0,populated=0;
    void start_phase() {phase_peak=0;}
    void begin(Index required) {
        if(owner && owner.use_count()!=1)throw std::logic_error("live contraction scratch view");
        phase_peak=std::max(phase_peak,required);
        if(required>capacity) {
            owner.reset();capacity=0;populated=0;
            owner=std::shared_ptr<double>(new double[required],std::default_delete<double[]>());
            capacity=required;
        }
        used=0;
    }
    T get(const std::vector<Index>& shape) {
        T result(shape,owner);
        used=(used+7)/8*8;
        if(!owner || result.size()>capacity-used)throw std::logic_error("contraction scratch overflow");
        result.p+=used;used+=result.size();return result;
    }
    // Credit only storage written by completed operations, not untouched
    // allocation capacity. Leave alignment gaps out of the RSS credit.
    void written() {populated=std::max(populated,std::max(Index(0),used-24));}
};

inline std::vector<Index> occupied_pair_rows(Index o) {
    std::vector<Index> rows(o*(o+1)/2);
    for(Index i=0;i<o;++i)for(Index j=0;j<=i;++j)rows[i*(i+1)/2+j]=i;
    return rows;
}

struct Timing { double pack=0,gemm=0,accumulate=0; size_t calls=0,packed_bytes=0,direct_calls=0,result_bytes=0; };

// Own parallelism in one OpenMP runtime. The caller scopes BLAS to one thread.
// Skinny results with a long reduction use private output tiles, then a fixed
// worker-order reduction. Other shapes split the longer output dimension.
// Private storage is bounded by the size of the larger input matrix.
inline void parallel_gemm(char ta,char tb,int m,int n,int k,double* a,int lda,
                   double* b,int ldb,double* c,int ldc,double alpha=1.,double beta=0.) {
    double one=1.,zero=0.;
    const Index output=Index(m)*n;
    const int workers=omp_get_max_threads();
    if(double(m)*n*k<200000. || workers==1) {
        blas(&ta,&tb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc);
    } else if(k/m>3 && k/n>3 && output*workers<=Index(k)*std::max(m,n)) {
        std::unique_ptr<double[]> scratch(new double[output*workers]);
        #pragma omp parallel
        {
            const int tid=omp_get_thread_num(),team=omp_get_num_threads();
            const Index lo=Index(k)*tid/team,hi=Index(k)*(tid+1)/team;
            int dk=int(hi-lo);double* temp=scratch.get()+output*tid;
            if(dk) {
                double* x=a+(ta=='N'?lo*lda:lo);
                double* y=b+(tb=='N'?lo:lo*ldb);
                blas(&ta,&tb,&m,&n,&dk,&one,x,&lda,y,&ldb,&zero,temp,&m);
            } else std::fill(temp,temp+output,0.);
            #pragma omp barrier
            #pragma omp for schedule(static)
            for(Index ij=0;ij<output;++ij) {
                double sum=0.;
                for(int t=0;t<team;++t)sum+=scratch[output*t+ij];
                double& dst=c[(ij/m)*ldc+ij%m];
                dst=beta==0.?alpha*sum:alpha*sum+beta*dst;
            }
        }
    } else {
        #pragma omp parallel
        {
            const int tid=omp_get_thread_num(),team=omp_get_num_threads();
            if(m>n) {
                const Index lo=Index(m)*tid/team,hi=Index(m)*(tid+1)/team;
                int dm=int(hi-lo);
                if(dm)blas(&ta,&tb,&dm,&n,&k,&alpha,a+(ta=='N'?lo:lo*lda),&lda,b,&ldb,&beta,c+lo,&ldc);
            } else {
                const Index lo=Index(n)*tid/team,hi=Index(n)*(tid+1)/team;
                int dn=int(hi-lo);
                if(dn)blas(&ta,&tb,&m,&dn,&k,&alpha,a,&lda,b+(tb=='N'?lo*ldb:lo),&ldb,&beta,c+lo*ldc,&ldc);
            }
        }
    }
}

struct Engine {
    bool profile;
    Scratch* accumulation_scratch=nullptr;
    std::map<std::string,Timing> timings;
    using LayoutKey=std::tuple<uintptr_t,std::array<Index,4>,std::array<Index,4>,int>;
    std::map<LayoutKey,T> layouts;
    size_t cache_limit=0,cache_bytes=0,cache_peak_bytes=0,cache_packed_bytes=0,cache_hits=0;
    bool cache_selected=false;
    LayoutKey selected_layout;
    // Only borrowed, immutable equation inputs are eligible. Mutable owned
    // intermediates never enter this cache; its lifetime is one ovov section.
    void cache_inputs(size_t bytes) { cache_limit=bytes;cache_selected=false; }
    void clear_layouts() { layouts.clear();cache_limit=cache_bytes=0;cache_selected=false; }
    void cache_layout(const T& view) {
        clear_layouts();cache_limit=size_t(view.size())*sizeof(double);cache_selected=true;
        selected_layout={reinterpret_cast<uintptr_t>(view.p),view.n,view.s,view.rank};
    }
    explicit Engine(bool enabled) : profile(enabled) {}
    static T order(const T& a,const std::string& from,const std::string& to) {
        std::vector<int> axes;
        for(char c:to) { auto i=from.find(c); if(i==std::string::npos)throw std::invalid_argument("missing index");axes.push_back(i); }
        return a.perm(axes);
    }
    struct Matrix { T storage; char trans; int ld; size_t packed_bytes=0; };
    // Flatten each group only if its non-singleton axes have consistent
    // strides. A dense inner group and a padded outer stride form a valid
    // BLAS matrix; tau[:, :, p:q, :] is the important resident example.
    static bool group_stride(const T& v,int lo,int hi,Index& stride,Index& count) {
        stride=1;count=1;bool first=true;
        for(int d=hi-1;d>=lo;--d) {
            if(v.n[d]==1)continue;
            if(v.s[d]<=0)return false;
            if(first) {stride=v.s[d];first=false;}
            else if(stride>PTRDIFF_MAX/count || v.s[d]!=stride*count)return false;
            if(count>PTRDIFF_MAX/v.n[d])return false;
            count*=v.n[d];
        }
        return true;
    }
    static int row_major_ld(const T& v,int split) {
        Index rs,cs,rows,cols;
        if(!group_stride(v,0,split,rs,rows) ||
           !group_stride(v,split,v.rank,cs,cols) || (cols>1 && cs!=1))return 0;
        Index ld=rows>1?rs:std::max(Index(1),cols);
        return ld>=cols && ld<=INT_MAX?int(ld):0;
    }
    static bool matrix_view(const T& a,const std::string& spec,const std::string& rows,
                            const std::string& cols,Matrix& result) {
        T v=order(a,spec,rows+cols);int ld=row_major_ld(v,int(rows.size()));
        if(ld) {result={v,'N',ld};return true;}
        T t=order(a,spec,cols+rows);ld=row_major_ld(t,int(cols.size()));
        if(ld) {result={t,'T',ld};return true;}
        return false;
    }
    static size_t packing_cost(const T& a,const std::string& spec,const std::string& rows,const std::string& cols) {
        Matrix view;return matrix_view(a,spec,rows,cols,view)?0:size_t(a.size())*8;
    }
    Matrix matrix(const T& a,const std::string& spec,const std::string& rows,const std::string& cols,Index m,Index n,Scratch* buffers=nullptr) {
        Matrix view;if(matrix_view(a,spec,rows,cols,view))return view;
        T v=order(a,spec,rows+cols);
        const size_t bytes=size_t(v.size())*sizeof(double);
        if(cache_limit && !a.owner) {
            LayoutKey key{reinterpret_cast<uintptr_t>(v.p),v.n,v.s,v.rank};
            auto found=layouts.find(key);
            if(found!=layouts.end()) {++cache_hits;return {found->second,'N',int(n)};}
            // Retained layouts must own storage independent of scratch. When
            // the cache is full, this operand can use the temporary workspace.
            if((!cache_selected || key==selected_layout) && bytes<=cache_limit-cache_bytes) {
                T packed=copy(v);layouts.emplace(key,packed);cache_bytes+=bytes;
                cache_peak_bytes=std::max(cache_peak_bytes,cache_bytes);cache_packed_bytes+=bytes;
                return {packed,'N',int(n),bytes};
            }
        }
        return {copy(v,1.,buffers?buffers->get(v.shape()):T()),'N',int(n),bytes};
    }
    static bool overlaps(const T& a,const T& b) {
        auto span=[](const T& t) {
            Index lo=0,hi=0;
            for(int d=0;d<t.rank;++d) {
                Index extent=(t.n[d]-1)*t.s[d];lo+=std::min(Index(0),extent);hi+=std::max(Index(0),extent);
            }
            return std::make_pair(reinterpret_cast<uintptr_t>(t.p+lo),reinterpret_cast<uintptr_t>(t.p+hi+1));
        };
        auto x=span(a),y=span(b);return x.first<y.second && y.first<x.second;
    }
    // All residual contractions are binary, with disjoint output groups and
    // common reduction indices. Choose reduction order by required copy bytes;
    // notably icjb,cjba uses c,j,b and views the large ovvv matrix directly.
    T E(const std::string& expr,const T& a,const T& b,const T* destination=nullptr,double alpha=1.,Scratch* buffers=nullptr,double beta=1.) {
        auto comma=expr.find(','),arrow=expr.find("->");
        if(comma==std::string::npos || arrow==std::string::npos || comma>arrow)
            throw std::invalid_argument("expected explicit binary contraction");
        const std::string as=expr.substr(0,comma),bs=expr.substr(comma+1,arrow-comma-1),cs=expr.substr(arrow+2);
        if(int(as.size())!=a.rank || int(bs.size())!=b.rank || cs.empty() || cs.size()>4)
            throw std::invalid_argument("contraction rank mismatch");
        std::array<Index,256> dims{};
        for(auto item:{std::make_pair(as,&a),std::make_pair(bs,&b)}) {
            std::string used;
            for(int d=0;d<item.second->rank;++d) {
                unsigned char c=item.first[d];Index n=item.second->n[d];
                if(n<=0 || used.find(c)!=std::string::npos || (dims[c] && dims[c]!=n))
                    throw std::invalid_argument("invalid contraction dimensions or repeated index");
                dims[c]=n;used+=c;
            }
        }
        std::string left,right,red,red_b;
        for(char c:as) {if(bs.find(c)!=std::string::npos)red+=c;else left+=c;}
        for(char c:bs) {if(as.find(c)!=std::string::npos)red_b+=c;else right+=c;}
        std::string expected=left+right,actual=cs;
        std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
        if(expected!=actual || red.empty())throw std::invalid_argument("unsupported contraction indices");
        auto cost=[&](const std::string& r){return packing_cost(a,as,left,r)+packing_cost(b,bs,r,right);};
        if(cost(red_b)<cost(red))red=red_b;
        auto extent=[&](const std::string& s){Index v=1;for(unsigned char c:s) {
            if(v>INT_MAX/dims[c])throw std::overflow_error("contraction exceeds BLAS integer range");v*=dims[c];}return v;};
        int m=extent(left),n=extent(right),k=extent(red);
        Matrix output;bool direct=false;
        if(destination) {
            std::vector<Index> shape;for(unsigned char c:cs)shape.push_back(dims[c]);
            shape_is(*destination,shape);
            direct=!overlaps(*destination,a) && !overlaps(*destination,b)
                && matrix_view(*destination,cs,left,right,output);
        }
        double start=profile?omp_get_wtime():0;
        // Reserving the uncached packing cost is conservative even when one
        // or both operands are found in (or newly added to) the layout cache.
        if(buffers)buffers->begin(Index(cost(red)/sizeof(double))+(direct?0:Index(m)*n)+24);
        Matrix x=matrix(a,as,left,red,m,k,buffers),y=matrix(b,bs,red,right,k,n,buffers);
        std::vector<Index> shape;for(unsigned char c:left+right)shape.push_back(dims[c]);
        T result;if(!direct)result=buffers?buffers->get(shape):T(shape);
        double ready=profile?omp_get_wtime():0;
        // Row-major C=A B is column-major C^T=B^T A^T. BLAS runs outside
        // Python; a single native team partitions each matrix product.
        if(direct && output.trans=='T') {
            parallel_gemm(x.trans=='N'?'T':'N',y.trans=='N'?'T':'N',m,n,k,
                x.storage.p,x.ld,y.storage.p,y.ld,output.storage.p,output.ld,alpha,beta);
        } else {
            parallel_gemm(y.trans,x.trans,n,m,k,y.storage.p,y.ld,x.storage.p,x.ld,
                direct?output.storage.p:result.p,direct?output.ld:n,direct?alpha:1.,direct?beta:0.);
        }
        if(profile) {auto& t=timings[expr];t.pack+=ready-start;t.gemm+=omp_get_wtime()-ready;++t.calls;t.packed_bytes+=x.packed_bytes+y.packed_bytes;
            t.direct_calls+=direct;t.result_bytes+=direct?0:sizeof(double)*size_t(m)*n;}
        if(buffers)buffers->written();
        // Empty means the destination was updated by GEMM; no result needs
        // copying/adding. All ordinary E() calls still return owned results.
        if(direct)return T();
        return order(result,left+right,cs);
    }
    void C(T out,const std::string& expr,const T& a,const T& b,double alpha=1.,double beta=1.) {
        T temp=E(expr,a,b,accumulation_scratch?&out:nullptr,alpha,accumulation_scratch,beta);
        if(!temp.p)return;
        double start=profile?omp_get_wtime():0;
        add(out,temp,alpha,beta);
        if(profile)timings[expr].accumulate+=omp_get_wtime()-start;
    }
    void A(T out,const std::string& expr,const T& a,const T& b,double alpha=1.) { C(out,expr,a,b,alpha,0.); }
    template<class Consumer>
    void with_result(const std::string& expr,const T& a,const T& b,Consumer consume) {
        T temp=E(expr,a,b,nullptr,1.,accumulation_scratch);
        double start=profile?omp_get_wtime():0;
        consume(temp);
        if(profile)timings[expr].accumulate+=omp_get_wtime()-start;
    }
};
struct Sections {
    bool enabled;double wall,cpu;
    std::map<std::string,std::pair<double,double>> times;
    explicit Sections(bool on):enabled(on),wall(omp_get_wtime()),cpu(double(std::clock())/CLOCKS_PER_SEC){}
    void mark(const std::string& name) {
        if(!enabled)return;
        double w=omp_get_wtime(),c=double(std::clock())/CLOCKS_PER_SEC;
        times[name]={w-wall,c-cpu};wall=w;cpu=c;
    }
};


Index ladder_scratch_elements(Index,Index,const LadderPanelPlan&);
void initialize_r2(T,const T&,const T&,const T&,Scratch&,const LadderPanelPlan&,const std::vector<size_t>&,const std::vector<Index>&,bool=true);
void finish_amplitudes(T,T,const T&,double,const std::vector<Index>&);
} // namespace ccsd_native
