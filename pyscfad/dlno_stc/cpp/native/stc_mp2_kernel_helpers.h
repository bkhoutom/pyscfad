#ifndef LIBGMBPT_QCHEM_STC_MP2_KERNEL_HELPERS_H
#define LIBGMBPT_QCHEM_STC_MP2_KERNEL_HELPERS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <armadillo>
#include <omp.h>

namespace libgmbpt {
namespace qchem {
namespace stc_mp2 {

inline std::vector<size_t> parse_occupied_indices(const std::string& text, size_t nocc) {
    std::vector<size_t> indices;
    size_t first = 0;
    do {
        const size_t end = text.find(',', first);
        std::string token = text.substr(first, end == std::string::npos ? end : end-first);
        const size_t begin = token.find_first_not_of(" \t");
        if (begin == std::string::npos) throw std::invalid_argument("Empty selected occupied index");
        token = token.substr(begin, token.find_last_not_of(" \t")-begin+1);
        if (token.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Selected occupied indices must be zero-based integers");
        const unsigned long long i = std::stoull(token);
        if (i >= nocc) throw std::invalid_argument("Selected occupied index out of range");
        indices.push_back(static_cast<size_t>(i));
        if (end == std::string::npos) break;
        first = end+1;
    } while (true);
    std::sort(indices.begin(), indices.end());
    if (std::adjacent_find(indices.begin(), indices.end()) != indices.end())
        throw std::invalid_argument("Duplicate selected occupied index");
    return indices;
}

class alias_distribution {
public:
    alias_distribution() = default;

    size_t storage_bytes() const {
        return sizeof(*this) + probability_.capacity()*sizeof(double)
                             + alias_.capacity()*sizeof(uint32_t);
    }

    template<class Iterator>
    alias_distribution(Iterator first, Iterator last) { reset(first, last); }

    template<class Iterator>
    void reset(Iterator first, Iterator last) {
        std::vector<double> weights(first, last);
        const size_t n = weights.size();
        if (n > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Alias distribution exceeds 32-bit index range");
        probability_.assign(n, 1.0);
        alias_.resize(n);
        for (size_t k = 0; k < n; ++k) alias_[k] = k;
        if (n == 0) return;

        double total = 0.0;
        for (double value : weights) total += value;
        if (!(total > 0.0)) return;

        std::vector<double> scaled(n);
        std::vector<size_t> small, large;
        small.reserve(n);
        large.reserve(n);
        for (size_t k = 0; k < n; ++k) {
            scaled[k] = weights[k] * double(n) / total;
            (scaled[k] < 1.0 ? small : large).push_back(k);
        }
        while (!small.empty() && !large.empty()) {
            const size_t s = small.back(); small.pop_back();
            const size_t l = large.back(); large.pop_back();
            probability_[s] = scaled[s];
            alias_[s] = l;
            scaled[l] -= 1.0-scaled[s];
            (scaled[l] < 1.0 ? small : large).push_back(l);
        }
        for (size_t k : small) probability_[k] = 1.0;
        for (size_t k : large) probability_[k] = 1.0;
    }

    template<class Generator>
    size_t operator()(Generator& generator) const {
        if (probability_.empty()) return 0;
        const double unit = (double(generator())-double(Generator::min()))
                          / (double(Generator::max())-double(Generator::min())+1.0);
        const double position = unit*double(probability_.size());
        const size_t k = std::min(static_cast<size_t>(position), probability_.size()-1);
        return position-double(k) < probability_[k] ? k : size_t(alias_[k]);
    }

private:
    std::vector<double> probability_;
    std::vector<uint32_t> alias_;
};

struct pilot_stats {
    double variance = 0.0;
    double seconds_per_sample = 0.0;
    size_t count = 0;
};

struct residual_tables {
    explicit residual_tables(size_t conditional_count) {
        for (size_t term = 0; term < 2; ++term) {
            first[term].resize(conditional_count);
            second[term].resize(conditional_count);
        }
    }

    std::array<std::vector<alias_distribution>, 2> first;
    std::array<std::vector<alias_distribution>, 2> second;
    std::array<alias_distribution, 2> outer;
    std::array<bool, 2> loaded{{false, false}};
    std::array<double, 2> normalization{{0.0, 0.0}};
};

struct moments {
    double mean = 0.0;
    double m2 = 0.0;
    double seconds = 0.0;
    size_t count = 0;
};

inline double group_dot(arma::mat& coefficients,
                        const arma::Col<size_t>& atom_offsets,
                        size_t atom, size_t x, size_t y) {
    const size_t first = atom_offsets(atom);
    const size_t count = atom_offsets(atom+1)-first;
    arma::Col<double> vx(coefficients.colptr(x)+first, count, false, true);
    arma::Col<double> vy(coefficients.colptr(y)+first, count, false, true);
    return arma::dot(vx, vy);
}

inline void merge(moments& total, const moments& batch) {
    if (batch.count == 0) return;
    if (total.count == 0) {
        total = batch;
        return;
    }
    const double delta = batch.mean-total.mean;
    const size_t next = total.count+batch.count;
    total.m2 += batch.m2 + delta*delta*double(total.count)*double(batch.count)/double(next);
    total.mean += delta*double(batch.count)/double(next);
    total.count = next;
    total.seconds += batch.seconds;
}

inline int effective_sample_threads() {
    int nthreads=omp_get_max_threads();
    return nthreads;
}

template<class Sampler>
moments draw_moments(size_t sample_count, unsigned seed, size_t point,
                     size_t stream, const Sampler& sample) {
    const int nthreads=effective_sample_threads();
    std::vector<moments> thread_moments(nthreads);
    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        std::mt19937 generator(seed + 104729u*static_cast<unsigned>(point)
                              + 1000003u*static_cast<unsigned>(stream)
                              + 7919u*static_cast<unsigned>(tid));
        moments& local = thread_moments[tid];
        const double start = omp_get_wtime();
        #pragma omp for schedule(static)
        for (size_t n = 0; n < sample_count; ++n) {
            const double value = sample(generator);
            ++local.count;
            const double delta = value-local.mean;
            local.mean += delta/double(local.count);
            local.m2 += delta*(value-local.mean);
        }
        local.seconds = omp_get_wtime()-start;
    }

    moments result;
    for (const moments& local : thread_moments) merge(result, local);
    result.seconds = 0.0;
    for (const moments& local : thread_moments)
        result.seconds = std::max(result.seconds, local.seconds);
    return result;
}

template<class Draw>
pilot_stats collect_pilot(size_t term, double refinement_variance,
                          const Draw& draw) {
    constexpr size_t initial_count = 100000;
    constexpr size_t extended_count = 1000000;
    moments result = draw(initial_count, 3*term);
    double variance = result.m2/double(result.count-1);
    if (variance/refinement_variance > 1000000.0) {
        moments extra = draw(extended_count-initial_count, 3*term+1);
        merge(result, extra);
        variance = result.m2/double(result.count-1);
    }
    pilot_stats stats;
    stats.variance = variance;
    stats.seconds_per_sample = result.seconds/double(result.count);
    stats.count = result.count;
    return stats;
}

template<class Sampler>
double run_residual(const char* label, double normalization, size_t term,
                    double refinement_variance, size_t production_count,
                    unsigned seed, size_t point, const Sampler& sample,
                    pilot_stats* pilot, double& variance_of_mean) {
    std::cout << label << " Z = " << std::scientific << std::setprecision(15)
              << normalization << std::endl;
    if (normalization == 0.0) {
        std::cout << label << " skipped (empty domain)" << std::endl;
        if (pilot) *pilot = pilot_stats{};
        return 0.0;
    }

    auto draw = [&](size_t count, size_t stream) {
        return draw_moments(count, seed, point, stream, sample);
    };
    if (pilot) {
        *pilot = collect_pilot(term, refinement_variance, draw);
        std::cout << label << " pilot_samples = " << pilot->count
                  << " pilot_var = " << std::scientific << pilot->variance
                  << " seconds_per_sample = " << pilot->seconds_per_sample
                  << std::endl;
        return 0.0;
    }

    const moments result = draw(production_count, 3*term+2);
    variance_of_mean = production_count > 1
        ? result.m2/(double(production_count)*double(production_count-1)) : 0.0;
    std::cout << label << " production_samples = " << production_count
              << " mean = " << std::setprecision(15) << result.mean
              << " se = " << std::sqrt(variance_of_mean) << std::endl;
    return result.mean;
}

// Shared dressed-factor C propagation for selected density, W, and exact MP2.
// The whole-system energy sampler uses its own in-place Laplace transform.
namespace density {

struct propagation_timing {
    double allocation_seconds=0, occupied_seconds=0, virtual_seconds=0;
};

struct reverse_propagation_timing {
    double occupied_rebuild_seconds=0, virtual_gram_seconds=0;
    double virtual_rotate_seconds=0, occupied_gram_seconds=0;
    double bare_adjoint_seconds=0, fock_adjoint_seconds=0;
};

// Exponentials are small orbital matrices; the scheduler owns the resident C
// arrays. A shared shift bounds both exponentials in a gapped SCF.
class propagator {
    const arma::mat& bare_;
    size_t no_, nv_;
    int threads_;
    arma::vec occupied_eigenvalues_, virtual_eigenvalues_, beta_;
    arma::mat occupied_eigenvectors_, virtual_eigenvectors_;
    arma::mat occupied_eigenvectors_t_, virtual_eigenvectors_t_;
    double shift_;
    std::vector<arma::mat> occupied_, virtuals_;
    static arma::mat multiply_panels(const arma::mat& left,const arma::mat& right,
            int threads) {
        arma::mat answer(left.n_rows,right.n_cols);
        const size_t panel=32;
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t first=0;first<right.n_cols;first+=panel) {
            const size_t last=std::min(size_t(right.n_cols),first+panel)-1;
            answer.cols(first,last)=left*right.cols(first,last);
        }
        return answer;
    }
    static arma::mat exponential_vjp(const arma::vec& eigenvalues,
            const arma::mat& eigenvectors,const arma::mat& eigenvectors_t,
            double coefficient,double shift,const arma::mat& adjoint,int threads) {
        const arma::mat symmetric=0.5*(adjoint+adjoint.t());
        const bool parallel=threads>1 && eigenvalues.n_elem>=512;
        const arma::mat eigen_adjoint=parallel
            ? multiply_panels(multiply_panels(eigenvectors_t,symmetric,threads),
                eigenvectors,threads)
            : eigenvectors_t*symmetric*eigenvectors;
        arma::mat divided(eigenvalues.n_elem,eigenvalues.n_elem);
        for (size_t p=0;p<eigenvalues.n_elem;++p) {
            for (size_t q=0;q<eigenvalues.n_elem;++q) {
                const double xp=coefficient*(eigenvalues(p)-shift);
                const double xq=coefficient*(eigenvalues(q)-shift);
                const double gap=std::abs(xp-xq);
                divided(p,q)=coefficient*std::exp(std::max(xp,xq))
                    *(gap==0 ? 1.0 : -std::expm1(-gap)/gap);
            }
        }
        const arma::mat result=parallel
            ? multiply_panels(multiply_panels(eigenvectors,
                divided%eigen_adjoint,threads),eigenvectors_t,threads)
            : eigenvectors*(divided%eigen_adjoint)*eigenvectors_t;
        return 0.5*(result+result.t());
    }
public:
    const arma::vec& occupied_spectrum() const { return occupied_eigenvalues_; }
    const arma::vec& virtual_spectrum() const { return virtual_eigenvalues_; }
    propagator(const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
               const arma::vec& beta, int threads): bare_(bare), no_(fo.n_rows),
               nv_(fv.n_rows), threads_(threads), beta_(beta) {
        if (!arma::eig_sym(occupied_eigenvalues_,occupied_eigenvectors_,fo) ||
            !arma::eig_sym(virtual_eigenvalues_,virtual_eigenvectors_,fv) ||
            virtual_eigenvalues_.min()<=occupied_eigenvalues_.max())
            throw std::invalid_argument("Density requires symmetric Fock blocks with a positive gap");
        occupied_eigenvectors_t_=occupied_eigenvectors_.t();
        virtual_eigenvectors_t_=virtual_eigenvectors_.t();
        shift_=(occupied_eigenvalues_.max()+virtual_eigenvalues_.min())/2;
        for (double t : beta) {
            occupied_.push_back(occupied_eigenvectors_*
                arma::diagmat(arma::exp(t*(occupied_eigenvalues_-shift_)))*occupied_eigenvectors_.t());
            virtuals_.push_back(virtual_eigenvectors_*
                arma::diagmat(arma::exp(-t*(virtual_eigenvalues_-shift_)))*virtual_eigenvectors_.t());
        }
    }
    void apply_occupied(size_t point, arma::mat& c, propagation_timing* timing=nullptr) const {
        double start=omp_get_wtime();
        c.set_size(bare_.n_rows,bare_.n_cols);
        if (timing) timing->allocation_seconds=omp_get_wtime()-start;
        start=omp_get_wtime();
        const arma::mat input(const_cast<double*>(bare_.memptr()),bare_.n_rows*nv_,no_,false,true);
        arma::mat view(c.memptr(), c.n_rows*nv_, no_, false, true);
        // Read B and write the occupied-rotated C directly: no full B->C copy.
        // Each worker owns disjoint rows and about 2 MiB per packed/result panel.
        const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(8*no_));
        #pragma omp parallel for num_threads(threads_) schedule(static)
        for (size_t first=0; first<view.n_rows; first+=panel) {
            const size_t last=std::min(size_t(view.n_rows),first+panel)-1;
            const arma::mat rotated=input.rows(first,last)*occupied_[point];
            view.rows(first,last)=rotated;
        }
        if (timing) timing->occupied_seconds=omp_get_wtime()-start;
    }
    void rotate_virtual(size_t point, arma::mat& c, propagation_timing* timing=nullptr) const {
        double start=omp_get_wtime();
        // About 8 MiB per packed/result panel per worker. Larger GEMMs amortize
        // packing the virtual rotation; at 32 threads the two panels use <=512 MiB.
        const size_t rows=std::max(size_t(1),size_t(8*1024*1024)/(8*nv_));
        const size_t panels=(c.n_rows+rows-1)/rows;
        #pragma omp parallel for num_threads(threads_) schedule(static)
        for (size_t task=0; task<no_*panels; ++task) {
            const size_t i=task/panels,first=(task%panels)*rows;
            const size_t last=std::min(size_t(c.n_rows),first+rows)-1;
            arma::mat pa(c.colptr(i*nv_),c.n_rows,nv_,false,true);
            const arma::mat rotated=pa.rows(first,last)*virtuals_[point];
            pa.rows(first,last)=rotated;
        }
        if (timing) timing->virtual_seconds=omp_get_wtime()-start;
    }
    void apply(size_t point, arma::mat& c, propagation_timing* timing=nullptr) const {
        apply_occupied(point,c,timing);
        rotate_virtual(point,c,timing);
    }
    // On entry gamma is dL/dC. Overwrite it with dL/dD, where D=B*O, and
    // accumulate input cotangents. scratch is a full-size reusable C buffer.
    void backpropagate(size_t point, arma::mat& scratch, arma::mat& gamma,
            arma::mat& bar_fo, arma::mat& bar_fv, arma::mat& bar_b,
            reverse_propagation_timing* timing=nullptr,
            bool occupied_ready=false) const {
        if (gamma.n_rows!=bare_.n_rows || gamma.n_cols!=bare_.n_cols ||
            bar_b.n_rows!=bare_.n_rows || bar_b.n_cols!=bare_.n_cols ||
            bar_fo.n_rows!=no_ || bar_fo.n_cols!=no_ ||
            bar_fv.n_rows!=nv_ || bar_fv.n_cols!=nv_)
            throw std::invalid_argument("Invalid propagator cotangent shape");
        double phase=omp_get_wtime();
        if (!occupied_ready) {
            apply_occupied(point,scratch);
            if (timing) timing->occupied_rebuild_seconds+=omp_get_wtime()-phase;
        } else if (scratch.n_rows!=bare_.n_rows || scratch.n_cols!=bare_.n_cols) {
            throw std::invalid_argument("Invalid retained occupied propagation shape");
        }
        arma::mat h_virtual(nv_,nv_,arma::fill::zeros);
        double virtual_gram=0,virtual_rotate=0;
        #pragma omp parallel num_threads(threads_)
        {
            arma::mat local(nv_,nv_,arma::fill::zeros);
            double local_gram=0,local_rotate=0;
            #pragma omp for schedule(static)
            for (size_t i=0;i<no_;++i) {
                arma::mat d(scratch.colptr(i*nv_),bare_.n_rows,nv_,false,true);
                arma::mat g(gamma.colptr(i*nv_),bare_.n_rows,nv_,false,true);
                const double gram_tick=timing ? omp_get_wtime() : 0.;
                local+=d.t()*g;
                if (timing) local_gram+=omp_get_wtime()-gram_tick;
                const double rotate_tick=timing ? omp_get_wtime() : 0.;
                // The occupied-only factor is no longer needed after its Gram
                // contribution. Reuse it for the rotated cotangent, leaving
                // gamma untouched (and potentially sparse in physical memory).
                d=g*virtuals_[point];
                if (timing) local_rotate+=omp_get_wtime()-rotate_tick;
            }
            #pragma omp critical
            {
                h_virtual+=local;
                virtual_gram+=local_gram;
                virtual_rotate+=local_rotate;
            }
        }
        if (timing) {
            timing->virtual_gram_seconds+=virtual_gram/double(threads_);
            timing->virtual_rotate_seconds+=virtual_rotate/double(threads_);
        }
        const arma::mat input(const_cast<double*>(bare_.memptr()),bare_.n_rows*nv_,no_,false,true);
        arma::mat gflat(scratch.memptr(),bare_.n_rows*nv_,no_,false,true);
        arma::mat bflat(bar_b.memptr(),bare_.n_rows*nv_,no_,false,true);
        arma::mat h_occupied(no_,no_,arma::fill::zeros);
        const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(8*no_));
        double occupied_gram=0,bare_adjoint=0;
        #pragma omp parallel num_threads(threads_)
        {
            arma::mat local(no_,no_,arma::fill::zeros);
            double local_gram=0,local_bare=0;
            #pragma omp for schedule(static)
            for (size_t first=0;first<gflat.n_rows;first+=panel) {
                const size_t last=std::min(size_t(gflat.n_rows),first+panel)-1;
                const double gram_tick=timing ? omp_get_wtime() : 0.;
                local+=input.rows(first,last).t()*gflat.rows(first,last);
                if (timing) local_gram+=omp_get_wtime()-gram_tick;
                const double bare_tick=timing ? omp_get_wtime() : 0.;
                bflat.rows(first,last)+=gflat.rows(first,last)*occupied_[point];
                if (timing) local_bare+=omp_get_wtime()-bare_tick;
            }
            #pragma omp critical
            {
                h_occupied+=local;
                occupied_gram+=local_gram;
                bare_adjoint+=local_bare;
            }
        }
        if (timing) {
            timing->occupied_gram_seconds+=occupied_gram/double(threads_);
            timing->bare_adjoint_seconds+=bare_adjoint/double(threads_);
        }
        phase=omp_get_wtime();
        bar_fo+=exponential_vjp(occupied_eigenvalues_,occupied_eigenvectors_,
            occupied_eigenvectors_t_,beta_(point),shift_,h_occupied,threads_);
        bar_fv+=exponential_vjp(virtual_eigenvalues_,virtual_eigenvectors_,
            virtual_eigenvectors_t_,-beta_(point),shift_,h_virtual,threads_);
        if (timing) timing->fock_adjoint_seconds+=omp_get_wtime()-phase;
    }
};

} // namespace density

} // namespace stc_mp2
} // namespace qchem
} // namespace libgmbpt

#endif
