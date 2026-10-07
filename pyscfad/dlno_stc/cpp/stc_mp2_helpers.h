#pragma once
#include "stc_mp2_kernel.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <exception>
#include <limits>
#include <omp.h>

namespace pyscfad { namespace dlno_stc {
inline int worker_limit() {
    return std::max(1,std::min(omp_get_max_threads(),omp_get_thread_limit()));
}
// Never let a worker exception escape an OpenMP region.
template<class Function>
inline void parallel_for(std::size_t count, const Function& function) {
    if (count < 2) {
        for (std::size_t i=0;i<count;++i) function(i);
        return;
    }
    std::exception_ptr error;
    #pragma omp parallel for schedule(static)
    for (std::int64_t i=0;i<static_cast<std::int64_t>(count);++i) {
        try { function(static_cast<std::size_t>(i)); }
        catch (...) {
            #pragma omp critical(stc_mp2_parallel_error)
            { if (!error) error=std::current_exception(); }
        }
    }
    if (error) std::rethrow_exception(error);
}
// Packed columns i*nv+a have contiguous auxiliary rows. The immutable input is
// packed exactly once by the binding; dressing never overwrites it.
inline arma::mat orbital_matrix(const arma::mat& packed, arma::uword P,
                                 arma::uword no, arma::uword nv) {
    arma::mat result(no, nv);
    for (arma::uword i = 0; i < no; ++i)
        for (arma::uword a = 0; a < nv; ++a)
            result(i,a) = packed(P, i*nv+a);
    return result;
}
inline void set_orbital_matrix(arma::mat& packed, arma::uword P,
                               const arma::mat& matrix) {
    for (arma::uword i = 0; i < matrix.n_rows; ++i)
        for (arma::uword a = 0; a < matrix.n_cols; ++a)
            packed(P, i*matrix.n_cols+a) = matrix(i,a);
}
inline Spectrum spectrum(const arma::mat& matrix) {
    Spectrum result;
    if (!arma::eig_sym(result.values, result.vectors, matrix))
        throw std::runtime_error("STC-MP2 Fock eigendecomposition failed");
    return result;
}
inline arma::mat project_target(const arma::mat& B, const arma::mat& M,
                                arma::uword nv) {
    arma::mat result(B.n_rows, nv, arma::fill::zeros);
    for (arma::uword i = 0; i < M.n_cols; ++i)
        result += M(0,i) * B.cols(i*nv, (i+1)*nv-1);
    return result;
}
// Migrated alias-table construction from the student's numerical helpers.
class AliasDistribution {
    std::vector<double> threshold;
    std::vector<std::size_t> alias;
public:
    AliasDistribution() = default;
    explicit AliasDistribution(const std::vector<double>& probabilities) {
        const std::size_t n = probabilities.size();
        threshold.assign(n,1.0); alias.resize(n);
        std::iota(alias.begin(),alias.end(),0);
        std::vector<double> scaled(n);
        std::vector<std::size_t> small, large;
        for (std::size_t k=0;k<n;++k) {
            scaled[k]=probabilities[k]*n;
            (scaled[k]<1 ? small:large).push_back(k);
        }
        while (!small.empty() && !large.empty()) {
            auto s=small.back(), l=large.back(); small.pop_back(); large.pop_back();
            threshold[s]=scaled[s]; alias[s]=l; scaled[l]-=1-scaled[s];
            (scaled[l]<1 ? small:large).push_back(l);
        }
    }
    std::size_t draw(std::mt19937_64& rng) const {
        const double position=std::generate_canonical<double,53>(rng)*threshold.size();
        const std::size_t k=std::min(static_cast<std::size_t>(position),threshold.size()-1);
        return position-k<threshold[k] ? k:alias[k];
    }
};
struct Proposal {
    std::vector<arma::uword> indices;
    std::vector<double> probability;
    AliasDistribution alias;
    Proposal() = default;
    Proposal(const std::vector<arma::uword>& allowed, const std::vector<double>& scores,
             double uniform) : indices(allowed) {
        if (allowed.empty()) return;
        double scale=0;
        for (auto i:allowed) scale=std::max(scale,scores[i]);
        double sum=0;
        if (scale>0) for (auto i:allowed) sum+=scores[i]/scale;
        for (auto i:allowed)
            probability.push_back(sum>0 ? (1-uniform)*(scores[i]/scale)/sum+uniform/allowed.size()
                                        : 1.0/allowed.size());
        alias=AliasDistribution(probability);
    }
    arma::uword draw(std::mt19937_64& rng, double& joint) const {
        const auto j=alias.draw(rng); joint*=probability[j]; return indices[j];
    }
};
inline std::uint64_t mix_seed(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31);
}
inline std::uint64_t stream_seed(std::uint64_t seed,std::size_t point,std::size_t stream,bool pilot) {
    return mix_seed(mix_seed(mix_seed(seed)^point)^(16+2*stream+(pilot?1:0)));
}
struct Moments {
    std::size_t count=0;
    double mean=0, m2=0;
    void add(double value) {
        ++count; const double delta=value-mean;
        mean+=delta/count; m2+=delta*(value-mean);
    }
    void merge(const Moments& other) {
        if (!other.count) return;
        if (!count) { *this=other; return; }
        const auto total=count+other.count;
        const double delta=other.mean-mean;
        m2+=other.m2+delta*delta*double(count)*double(other.count)/double(total);
        mean+=delta*double(other.count)/double(total); count=total;
    }
    double variance() const { return count>1 ? m2/(count-1):0; }
};

inline std::vector<double> column_norms(const arma::mat& m) {
    std::vector<double> norms(m.n_cols);
    parallel_for(m.n_cols,[&](std::size_t j) { norms[j]=arma::norm(m.col(j),2); });
    return norms;
}
inline double dot_group(const arma::mat& x,arma::uword colx,const arma::mat& y,arma::uword coly,
                 const std::vector<arma::uword>& offsets,arma::uword group) {
    const double* a=x.colptr(colx); const double* b=y.colptr(coly);
    double value=0;
    for (auto p=offsets[group];p<offsets[group+1];++p) value+=a[p]*b[p];
    return value;
}
// The student's column/group scatter records now distinguish the four physical
// branches. Updates live in bounded waves; no sample history or full tensor replicas.
struct Update {
    arma::mat* target; const arma::mat* source;
    arma::uword target_col, source_col, group;
    double scale;
};
// Logical batches and their streams are independent of the OpenMP team size.
// At most eight batches (32768 four-role updates) reside at once. Each scatter
// worker exclusively owns target columns; no full tensor bars or sample
// histories are duplicated. Scanning this bounded wave preserves update order.
template<class Sampler>
inline Moments draw_batched(std::size_t count,std::uint64_t seed,bool with_grad,
                            const std::vector<arma::uword>& offsets,const Sampler& sample) {
    constexpr std::size_t batch_size=1024, wave_size=8;
    const auto batches=count/batch_size+(count%batch_size!=0);
    Moments total;
    for (std::size_t first=0;first<batches;first+=wave_size) {
        const auto size=std::min(wave_size,batches-first);
        std::vector<Moments> moments(size);
        std::vector<std::vector<Update>> updates(size);
        parallel_for(size,[&](std::size_t j) {
            const auto batch=first+j, begin=batch*batch_size;
            std::mt19937_64 rng(batch ? mix_seed(seed^mix_seed(batch)):seed);
            auto& records=updates[j];
            if (with_grad) records.reserve(4*batch_size);
            const auto end=begin+std::min(batch_size,count-begin);
            for (auto n=begin;n<end;++n)
                moments[j].add(sample(rng,records));
        });
        for (const auto& moment:moments) total.merge(moment);
        if (with_grad) {
            const auto owners=std::min(8,worker_limit());
            parallel_for(owners,[&](std::size_t owner) {
                for (const auto& records:updates) for (const auto& u:records) {
                    if (u.target_col%owners!=owner) continue;
                    double* target=u.target->colptr(u.target_col);
                    const double* source=u.source->colptr(u.source_col);
                    for (auto p=offsets[u.group];p<offsets[u.group+1];++p)
                        target[p]+=u.scale*source[p];
                }
            });
        }
    }
    return total;
}
inline double point_variance_target(const Controls& controls,std::size_t point) {
    return controls.point_variance_targets.empty()
        ? controls.energy_tolerance*controls.energy_tolerance/controls.roots.size()
        : controls.point_variance_targets.at(point);
}
// Original full-system trace model, evaluated in log space. The occupied
// maximum and virtual minimum factor out scalar shifts without overflow.
inline std::vector<double> full_variance_targets(const Controls& controls,
                                                const Spectrum& eo,const Spectrum& ev) {
    const long double gap=static_cast<long double>(ev.values.min())-eo.values.max();
    std::vector<long double> log_mass(controls.roots.size(),
                                     -std::numeric_limits<long double>::infinity());
    long double maximum=-std::numeric_limits<long double>::infinity();
    for (std::size_t l=0;l<log_mass.size();++l) {
        if (controls.weights[l]==0.) continue;
        const long double beta=.5L*controls.roots[l];
        long double occupied=0.,virtuals=0.;
        for (auto value:eo.values)
            occupied+=std::exp(beta*(static_cast<long double>(value)-eo.values.max()));
        for (auto value:ev.values)
            virtuals+=std::exp(-beta*(static_cast<long double>(value)-ev.values.min()));
        log_mass[l]=std::log(std::abs(static_cast<long double>(controls.weights[l])))
            +1.4L*(-beta*gap+std::log(occupied)+std::log(virtuals));
        maximum=std::max(maximum,log_mass[l]);
    }
    std::vector<double> targets(log_mass.size(),0.);
    if (!std::isfinite(maximum)) return targets; // All weights zero.
    long double sum=0.;
    for (auto value:log_mass) sum+=std::exp(value-maximum);
    const long double variance=static_cast<long double>(controls.energy_tolerance)
                              *controls.energy_tolerance;
    for (std::size_t l=0;l<targets.size();++l)
        targets[l]=static_cast<double>(variance*std::exp(log_mass[l]-maximum)/sum);
    return targets;
}
inline std::size_t production_count(const Controls& controls, const Moments& pilot,
                                    double cost, double allocation_sum,std::size_t point=0) {
    std::size_t count=controls.production_samples;
    if (controls.adaptive) {
        const double variance=pilot.variance();
        if (!std::isfinite(variance) || variance<0 || !std::isfinite(allocation_sum)
                || allocation_sum<0 || !std::isfinite(cost) || cost<=0)
            throw std::runtime_error("Invalid pilot statistics in dynamic sampling allocation");
        // A zero-variance pilot still needs positive production draws for bars.
        double requested=0.;
        if (variance>0.) {
            const double target=point_variance_target(controls,point);
            // A positive infinite target is a very loose finite tolerance
            // whose square overflowed, and needs only the production minimum.
            requested=target==std::numeric_limits<double>::infinity() ? 0. : target>0.
                ? std::ceil(std::sqrt(variance/cost)*allocation_sum/target)
                : std::numeric_limits<double>::infinity();
        }
        if (controls.max_production_samples && requested>=controls.max_production_samples)
            count=controls.max_production_samples; // Explicit opt-in limit only.
        else {
            if (!std::isfinite(requested) || requested<0.
                    || requested>=double(std::numeric_limits<std::size_t>::max()))
                throw std::overflow_error("Dynamic sampling budget overflow");
            count=static_cast<std::size_t>(requested);
        }
    }
    const auto minimum=controls.min_production_samples ? controls.min_production_samples
                                                       : (controls.adaptive ? 10000:32);
    return std::max(count,minimum);
}
}}  // namespace pyscfad::dlno_stc
