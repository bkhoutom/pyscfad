#include "stc_mp2_helpers.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>

namespace pyscfad { namespace dlno_stc {
namespace {
struct Residual {
    std::size_t term=0, number=0;
    Proposal outer, group_A, group_bare;
    std::vector<Proposal> first, second;
    Moments pilot;
    double seconds_per_sample=0;
    bool empty() const { return outer.indices.empty(); }
};
Moments draw(const Residual& residual,std::size_t count,std::uint64_t seed,double weight,
             const Inputs& x,const arma::mat& A,const arma::mat& U,
             const arma::mat& C,const arma::mat& Z,arma::mat& bar_A,arma::mat& bar_U,
             arma::mat& bar_C,arma::mat& bar_Z,bool with_grad) {
    const arma::uword nv=U.n_cols;
    const double coefficient=(residual.term==0 ? -2.0:1.0)*weight;
    return draw_batched(count,seed,with_grad,x.aux_offsets,
        [&](std::mt19937_64& rng,std::vector<Update>& updates) {
        double probability=1;
        const auto k=residual.outer.draw(rng,probability);
        const auto a=residual.first[k].draw(rng,probability), b=residual.second[k].draw(rng,probability);
        const auto g=residual.group_A.draw(rng,probability), h=residual.group_bare.draw(rng,probability);
        const auto ca=residual.term==0 ? a:b, zb=k*nv+(residual.term==0 ? b:a);
        const double sA=dot_group(U,a,A,k*nv+b,x.aux_offsets,g);
        const double sB=dot_group(C,ca,Z,zb,x.aux_offsets,h);
        const double physical=coefficient/probability;
        if (with_grad) {
            const double alpha=physical/count;
            // Never skip an energy-zero sample: either factor can have a bar.
            updates.push_back({&bar_U,&A,a,k*nv+b,g,alpha*sB});
            updates.push_back({&bar_A,&U,k*nv+b,a,g,alpha*sB});
            updates.push_back({&bar_C,&Z,ca,zb,h,alpha*sA});
            updates.push_back({&bar_Z,&C,zb,ca,h,alpha*sA});
        }
        return physical*sA*sB;
    });
}
}  // namespace

KeepSets select_keep(const arma::mat& A,const arma::mat& U,const arma::mat& C,
                     const arma::mat& Z,arma::uword no,const Controls& controls) {
    const arma::uword nv=U.n_cols;
    const auto na=column_norms(A), nu=column_norms(U), nc=column_norms(C), nz=column_norms(Z);
    KeepSets keep(no);
    for (arma::uword k=0;k<no;++k) {
        std::vector<double> scores(nv);
        for (arma::uword a=0;a<nv;++a) scores[a]=nu[a]*na[k*nv+a]+nc[a]*nz[k*nv+a];
        if (controls.virtual_keep_fraction>=0) {
            std::vector<arma::uword> order(nv); std::iota(order.begin(),order.end(),0);
            std::stable_sort(order.begin(),order.end(),[&](arma::uword a,arma::uword b){return scores[a]>scores[b];});
            order.resize(static_cast<std::size_t>(std::floor(controls.virtual_keep_fraction*nv)));
            std::sort(order.begin(),order.end()); keep[k]=std::move(order);
        } else {
            const double maximum=*std::max_element(scores.begin(),scores.end());
            for (arma::uword a=0;a<nv;++a)
                if (maximum>0 && scores[a]>=controls.workload_cutoff*maximum) keep[k].push_back(a);
        }
    }
    return keep;
}

void sample_residuals(const Inputs& x,const Controls& controls,std::size_t point,
                      const KeepSets& keep,const arma::mat& A,const arma::mat& U,
                      const arma::mat& C,const arma::mat& Z,arma::mat& bar_A,
                      arma::mat& bar_U,arma::mat& bar_C,arma::mat& bar_Z,
                      bool with_grad,Result& out) {
    const arma::uword no=x.foo.n_rows,nv=U.n_cols,ng=x.aux_offsets.size()-1;
    const auto na=column_norms(A),nu=column_norms(U),nc=column_norms(C),nz=column_norms(Z);
    std::vector<arma::uword> all(nv), groups(ng); std::iota(all.begin(),all.end(),0); std::iota(groups.begin(),groups.end(),0);
    std::vector<double> gscore(ng),hscore(ng);
    for (arma::uword g=0;g<ng;++g) {
        const auto begin=x.aux_offsets[g],end=x.aux_offsets[g+1]-1;
        gscore[g]=arma::norm(U.rows(begin,end),"fro")*arma::norm(A.rows(begin,end),"fro");
        hscore[g]=arma::norm(C.rows(begin,end),"fro")*arma::norm(Z.rows(begin,end),"fro");
    }
    std::array<Residual,4> residuals;
    for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; residual.term=r/2; residual.number=r%2+1;
        residual.group_A=Proposal(groups,gscore,controls.uniform_mixture);
        residual.group_bare=Proposal(groups,hscore,controls.uniform_mixture);
        residual.first.resize(no); residual.second.resize(no);
        std::vector<arma::uword> occupied; std::vector<double> kscore(no,0);
        for (arma::uword k=0;k<no;++k) {
            std::vector<arma::uword> outside;
            std::set_difference(all.begin(),all.end(),keep[k].begin(),keep[k].end(),std::back_inserter(outside));
            const auto& allowed_a=residual.number==1 ? outside:keep[k];
            const auto& allowed_b=residual.number==1 ? all:outside;
            if (allowed_a.empty() || allowed_b.empty()) continue;
            std::vector<double> q(nv),rscore(nv);
            for (arma::uword a=0;a<nv;++a) {
                q[a]=nu[a]*(residual.term==0 ? nc[a]:nz[k*nv+a]);
                rscore[a]=na[k*nv+a]*(residual.term==0 ? nz[k*nv+a]:nc[a]);
            }
            double qsum=0,rsum=0;
            for (auto a:allowed_a) qsum+=q[a];
            for (auto b:allowed_b) rsum+=rscore[b];
            kscore[k]=qsum*rsum; occupied.push_back(k);
            residual.first[k]=Proposal(allowed_a,q,controls.uniform_mixture);
            residual.second[k]=Proposal(allowed_b,rscore,controls.uniform_mixture);
        }
        residual.outer=Proposal(occupied,kscore,controls.uniform_mixture);
    }
    // Four-way pilot allocation migrated in physical weighted-energy units.
    // Direct and Laplace coefficients already occur inside samples: multiplier1.
    double allocation_sum=0;
    if (controls.adaptive) for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; if (residual.empty()) continue;
        const auto start=std::chrono::steady_clock::now();
        residual.pilot=draw(residual,controls.pilot_samples,stream_seed(controls.global_seed,point,r,true),
                            controls.weights[point],x,A,U,C,Z,bar_A,bar_U,bar_C,bar_Z,false);
        residual.seconds_per_sample=std::max(1e-15,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/controls.pilot_samples);
        allocation_sum+=std::sqrt(residual.pilot.variance()*residual.seconds_per_sample);
    }
    for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; if (residual.empty()) continue;
        const std::size_t count=production_count(controls,residual.pilot,
                                                 residual.seconds_per_sample,allocation_sum,point);
        const auto seed=stream_seed(controls.global_seed,point,r,false);
        const Moments production=draw(residual,count,seed,controls.weights[point],x,A,U,C,Z,
                                      bar_A,bar_U,bar_C,bar_Z,with_grad);
        const double variance=std::max(0.0,production.variance()/count);
        out.energy+=production.mean;
        out.residuals.push_back({point,residual.term,residual.number,residual.pilot.count,count,
                                 stream_seed(controls.global_seed,point,r,true),seed,variance});
    }
}
}}  // namespace pyscfad::dlno_stc
