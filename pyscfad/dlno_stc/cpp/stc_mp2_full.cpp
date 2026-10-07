#include "stc_mp2_helpers.h"
#include <array>
#include <chrono>
#include <iterator>

namespace pyscfad { namespace dlno_stc {
namespace {
KeepSets full_keep(const arma::mat& T, arma::uword no, arma::uword nv,
                   double weight, const Controls& controls) {
    const auto norms=column_norms(T);
    KeepSets keep(no);
    for (arma::uword i=0;i<no;++i) {
        if (!controls.stochastic || controls.virtual_keep_fraction>=0) {
            std::vector<arma::uword> order(nv); std::iota(order.begin(),order.end(),0);
            if (controls.stochastic) {
                std::stable_sort(order.begin(),order.end(),[&](arma::uword a,arma::uword b) {
                    return norms[i*nv+a]>norms[i*nv+b];
                });
                order.resize(static_cast<std::size_t>(std::floor(controls.virtual_keep_fraction*nv)));
                std::sort(order.begin(),order.end());
            }
            keep[i]=std::move(order);
        } else {
            const double scale=std::sqrt(std::sqrt(std::abs(weight)));
            for (arma::uword a=0;a<nv;++a)
                if (scale*norms[i*nv+a]>controls.system_workload_cutoff) keep[i].push_back(a);
        }
    }
    return keep;
}
std::vector<arma::uword> intersection(const std::vector<arma::uword>& a,
                                      const std::vector<arma::uword>& b) {
    std::vector<arma::uword> result;
    std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(result));
    return result;
}
std::vector<arma::uword> complement(const std::vector<arma::uword>& all,
                                    const std::vector<arma::uword>& kept) {
    std::vector<arma::uword> result;
    std::set_difference(all.begin(),all.end(),kept.begin(),kept.end(),std::back_inserter(result));
    return result;
}
// Each occupied iteration owns every Gamma_i column. Pair symmetry folds the
// other factor roles into this owner's derivative, including restricted sets.
// Use the original full kept-set GEMMs when their workspace fits 64 MiB;
// otherwise retain bounded virtual tiles. Never form all-pair four-index data.
bool pair_fits(arma::uword np,std::size_t first,std::size_t second,double reserve=0) {
    return reserve+8.0*(2.0*np*first+double(np)*second+2.0*first*second)<=64.0*1024*1024;
}
arma::uvec occupied_columns(const std::vector<arma::uword>& kept,
                             arma::uword i,arma::uword nv) {
    return i*nv+arma::uvec(kept);
}
double exact_owned(const arma::mat& T,arma::mat& gamma,arma::uword i,arma::uword k,
                   arma::uword nv,const std::vector<arma::uword>& first,
                   const std::vector<arma::uword>& second,bool exchange,
                   double weight,arma::uword block,bool with_grad,
                   const arma::mat* cached=nullptr,double reserve=0) {
    if (first.empty() || second.empty()) return 0;
    const auto ia=occupied_columns(first,i,nv),kb=occupied_columns(second,k,nv);
    if (pair_fits(T.n_rows,first.size(),second.size(),reserve)) {
        arma::mat gathered;
        if (!cached) gathered=T.cols(ia);
        const arma::mat& X=cached ? *cached:gathered;
        const arma::mat Y=T.cols(kb),G=X.t()*Y;
        if (!exchange) {
            if (with_grad) gamma.cols(ia)+=-8*weight*(Y*G.t());
            return -2*weight*arma::accu(G%G);
        }
        if (with_grad) gamma.cols(ia)+=4*weight*(Y*G);
        return weight*arma::accu(G%G.t());
    }
    double energy=0;
    // Include both rectangular virtual panels in the exact energy. Only the
    // first occupied owner is written, so panels and occupied pairs need no locks.
    for (std::size_t a=0;a<first.size();a+=block)
        for (std::size_t b=0;b<second.size();b+=block) {
            const arma::uvec va(std::vector<arma::uword>(first.begin()+a,first.begin()+std::min<std::size_t>(first.size(),a+block)));
            const arma::uvec vb(std::vector<arma::uword>(second.begin()+b,second.begin()+std::min<std::size_t>(second.size(),b+block)));
            const arma::uvec ca=i*nv+va,cb=k*nv+vb;
            const arma::mat X=T.cols(ca),Y=T.cols(cb),G=X.t()*Y;
            if (!exchange) {
                energy-=2*weight*arma::accu(G%G);
                if (with_grad) gamma.cols(ca)+=-8*weight*(Y*G.t());
            } else {
                const arma::mat W=T.cols(k*nv+va),Z=T.cols(i*nv+vb);
                const arma::mat swapped=W.t()*Z;
                energy+=weight*arma::accu(G%swapped);
                if (with_grad) gamma.cols(ca)+=4*weight*(Y*swapped.t());
            }
        }
    return energy;
}
struct FullResidual {
    std::size_t term=0, number=0;
    Proposal pairs;
    std::vector<Proposal> first, second;
    Moments pilot;
    std::uint64_t pilot_refinement_seed=0;
    double seconds_per_sample=0;
    bool empty() const { return pairs.indices.empty(); }
};
Moments draw_full(const FullResidual& residual, const std::vector<Proposal>& groups,
                  std::size_t count, std::uint64_t seed, double weight,
                  const Inputs& x, const arma::mat& T, arma::mat& gamma, bool with_grad) {
    const arma::uword no=x.foo.n_rows,nv=x.fvv.n_rows;
    const double coefficient=(residual.term==0 ? -2.0:1.0)*weight;
    return draw_batched(count,seed,with_grad,x.aux_offsets,
        [&](std::mt19937_64& rng,std::vector<Update>& updates) {
        double probability=1;
        const auto pair=residual.pairs.draw(rng,probability), i=pair/no, k=pair%no;
        const auto a=residual.first[pair].draw(rng,probability);
        const auto b=residual.second[pair].draw(rng,probability);
        const auto ia=i*nv+a,kb=k*nv+b,ib=i*nv+b,ka=k*nv+a;
        const auto source=residual.number==1 ? kb:ia;
        const auto g=groups[residual.term==0 ? source:ia].draw(rng,probability);
        const auto h=groups[residual.term==0 ? source:ka].draw(rng,probability);
        const auto second_left=residual.term==0 ? ia:ib;
        const auto second_right=residual.term==0 ? kb:ka;
        const double s=dot_group(T,ia,T,kb,x.aux_offsets,g);
        const double t=dot_group(T,second_left,T,second_right,x.aux_offsets,h);
        const double physical=coefficient/probability;
        if (with_grad) {
            const double alpha=physical/count;
            // Four distinct factor roles, even when a product is zero or roles
            // address the same column/group. Proposal probabilities are frozen.
            updates.push_back({&gamma,&T,ia,kb,g,alpha*t});
            updates.push_back({&gamma,&T,kb,ia,g,alpha*t});
            updates.push_back({&gamma,&T,second_left,second_right,h,alpha*s});
            updates.push_back({&gamma,&T,second_right,second_left,h,alpha*s});
        }
        return physical*s*t;
    });
}
void full_residuals(const Inputs& x,const Controls& controls,std::size_t point,
                    const KeepSets& keep,const arma::mat& T,arma::mat& gamma,
                    bool with_grad,Result& out) {
    const arma::uword no=x.foo.n_rows,nv=x.fvv.n_rows,ng=x.aux_offsets.size()-1;
    const auto n=column_norms(T);
    std::vector<double> row_norm(T.n_rows),z(T.n_cols,0);
    parallel_for(T.n_rows,[&](std::size_t p) { row_norm[p]=arma::norm(T.row(p),2); });
    std::vector<arma::uword> all(nv),group_indices(ng);
    std::iota(all.begin(),all.end(),0); std::iota(group_indices.begin(),group_indices.end(),0);
    std::vector<Proposal> groups(T.n_cols);
    parallel_for(T.n_cols,[&](std::size_t col) {
        std::vector<double> scores(ng,0);
        for (arma::uword g=0;g<ng;++g)
            for (auto p=x.aux_offsets[g];p<x.aux_offsets[g+1];++p)
                scores[g]+=std::abs(T(p,col))*row_norm[p];
        z[col]=std::accumulate(scores.begin(),scores.end(),0.0);
        groups[col]=Proposal(group_indices,scores,controls.uniform_mixture);
    });
    std::array<FullResidual,4> residuals;
    for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; residual.term=r/2; residual.number=r%2+1;
        residual.first.resize(no*no); residual.second.resize(no*no);
        std::vector<arma::uword> pairs;
        std::vector<double> pair_scores(no*no,0);
        for (arma::uword i=0;i<no;++i) for (arma::uword k=0;k<no;++k) {
            const auto pair=i*no+k;
            const auto D=residual.term==1 ? intersection(keep[i],keep[k]):std::vector<arma::uword>();
            const auto& Ka=residual.term==0 ? keep[i]:D;
            const auto& Kb=residual.term==0 ? keep[k]:D;
            const auto outside_a=complement(all,Ka),outside_b=complement(all,Kb);
            const auto& allowed_a=residual.number==1 ? outside_a:Ka;
            const auto& allowed_b=residual.number==1 ? all:outside_b;
            if (allowed_a.empty() || allowed_b.empty()) continue;
            std::vector<double> first(nv),second(nv);
            for (arma::uword a=0;a<nv;++a) {
                const auto ia=i*nv+a,ka=k*nv+a;
                if (residual.term==1) { first[a]=z[ia]*z[ka]; second[a]=n[ia]*n[ka]; }
                else if (residual.number==1) { first[a]=n[ia]*n[ia]; second[a]=z[ka]*z[ka]; }
                else { first[a]=z[ia]*z[ia]; second[a]=n[ka]*n[ka]; }
            }
            double first_sum=0,second_sum=0;
            for (auto a:allowed_a) first_sum+=first[a];
            for (auto b:allowed_b) second_sum+=second[b];
            pair_scores[pair]=first_sum*second_sum; pairs.push_back(pair);
            residual.first[pair]=Proposal(allowed_a,first,controls.uniform_mixture);
            residual.second[pair]=Proposal(allowed_b,second,controls.uniform_mixture);
        }
        residual.pairs=Proposal(pairs,pair_scores,controls.uniform_mixture);
    }
    double allocation_sum=0;
    if (controls.adaptive) for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; if (residual.empty()) continue;
        const auto start=std::chrono::steady_clock::now();
        residual.pilot=draw_full(residual,groups,controls.pilot_samples,
            stream_seed(controls.global_seed,point,r,true),controls.weights[point],x,T,gamma,false);
        // Original policy: refine a difficult pilot to one million draws.
        // Samples already contain the physical coefficient and Laplace weight,
        // so both direct and exchange compare against weighted point budget/4.
        constexpr std::size_t refined_count=1000000;
        const double refinement_target=point_variance_target(controls,point)/4.;
        if (residual.pilot.count<refined_count && residual.pilot.variance()>0.
                && (refinement_target<=0. || residual.pilot.variance()/refinement_target>1000000.)) {
            residual.pilot_refinement_seed=stream_seed(controls.global_seed,point,r+4,true);
            const auto extra=draw_full(residual,groups,refined_count-residual.pilot.count,
                residual.pilot_refinement_seed,controls.weights[point],x,T,gamma,false);
            residual.pilot.merge(extra);
        }
        residual.seconds_per_sample=std::max(1e-15,std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count()/residual.pilot.count);
        allocation_sum+=std::sqrt(residual.pilot.variance()*residual.seconds_per_sample);
    }
    for (std::size_t r=0;r<4;++r) {
        auto& residual=residuals[r]; if (residual.empty()) continue;
        const auto count=production_count(controls,residual.pilot,residual.seconds_per_sample,allocation_sum,point);
        const auto seed=stream_seed(controls.global_seed,point,r,false);
        const auto production=draw_full(residual,groups,count,seed,controls.weights[point],x,T,gamma,with_grad);
        const double variance=std::max(0.0,production.variance()/count);
        out.energy+=production.mean;
        out.residuals.push_back({point,residual.term,residual.number,residual.pilot.count,count,
                                stream_seed(controls.global_seed,point,r,true),seed,variance,
                                residual.pilot_refinement_seed});
    }
}
}  // namespace
void contract_full(const Inputs& x,const Controls& controls,std::size_t point,
                   const arma::mat& T,arma::mat& gamma,bool with_grad,Result& out) {
    const arma::uword no=x.foo.n_rows,nv=x.fvv.n_rows;
    const double weight=controls.weights[point];
    const auto keep=full_keep(T,no,nv,weight,controls);
    arma::uword block=std::min(nv,controls.virtual_block_size);
    while (block>1 && !pair_fits(T.n_rows,2*block,2*block,32.0*1024*1024)) block=std::max<arma::uword>(1,block/2);
    std::vector<double> energy(no,0);
    parallel_for(no,[&](std::size_t i) {
        arma::mat cached;
        const bool cache=pair_fits(T.n_rows,nv,nv) && !keep[i].empty();
        if (cache) cached=T.cols(occupied_columns(keep[i],i,nv));
        const auto begin=with_grad ? 0:i;
        for (arma::uword k=begin;k<no;++k) {
            const double factor=(!with_grad && i!=k) ? 2.0:1.0;
            energy[i]+=factor*exact_owned(T,gamma,i,k,nv,keep[i],keep[k],false,
                                         weight,block,with_grad,cache ? &cached:nullptr);
            const auto D=intersection(keep[i],keep[k]);
            const bool reuse=cache && D==keep[i];
            energy[i]+=factor*exact_owned(T,gamma,i,k,nv,D,D,true,weight,block,with_grad,
                                         reuse ? &cached:nullptr,
                                         reuse ? 0.0:double(cached.n_elem)*sizeof(double));
        }
    });
    out.energy+=std::accumulate(energy.begin(),energy.end(),0.0);
    if (controls.stochastic) full_residuals(x,controls,point,keep,T,gamma,with_grad,out);
}
}}  // namespace pyscfad::dlno_stc
