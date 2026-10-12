// Selected-target port of authoritative src_code_domain numerical routines.
// Host adaptations and explicit test controls are documented in phase1/kernel_implementation.md.
#include "stc_mp2_selected.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

namespace pyscfad { namespace dlno_stc {
namespace selected_source {
namespace stc_mp2 {
class alias_distribution {
public:
    alias_distribution() = default;

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

template<class Sampler>
moments draw_moments(size_t sample_count, unsigned seed, size_t point,
                     size_t stream, const Sampler& sample) {
    int nthreads = omp_get_max_threads();
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
                          const Draw& draw, const Controls& controls) {
    const size_t initial_count = controls.pilot_samples;
    const size_t extended_count = std::max(size_t(1000000),initial_count);
    moments result = draw(initial_count, 3*term);
    double variance = result.m2/double(result.count-1);
    if (controls.adaptive && variance/refinement_variance > 1000000.0) {
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
                    pilot_stats* pilot, double& variance_of_mean,
                    const Controls& controls, Result& output, double laplace_weight) {

    if (normalization == 0.0) {

        if (pilot) *pilot = pilot_stats{};
        return 0.0;
    }

    auto draw = [&](size_t count, size_t stream) {
        return draw_moments(count, seed, point, stream, sample);
    };
    if (pilot) {
        *pilot = collect_pilot(term, refinement_variance, draw, controls);
        ResidualDiagnostic row{};
        row.point=point; row.term=std::string(label).find("direct")==0 ? 0:1;
        row.residual=term+1; row.pilot_samples=pilot->count;
        row.pilot_seed=unsigned(seed+104729u*unsigned(point)+1000003u*unsigned(3*term));
        row.pilot_refinement_seed=unsigned(seed+104729u*unsigned(point)+1000003u*unsigned(3*term+1));
        row.production_seed=unsigned(seed+104729u*unsigned(point)+1000003u*unsigned(3*term+2));
        output.residuals.push_back(row);

        return 0.0;
    }

    const moments result = draw(production_count, 3*term+2);
    variance_of_mean = production_count > 1
        ? result.m2/(double(production_count)*double(production_count-1)) : 0.0;

    const size_t kind=std::string(label).find("direct")==0 ? 0:1;
    for(auto& row:output.residuals)
        if(row.point==point && row.term==kind && row.residual==term+1) {
            row.production_samples=production_count;
            row.variance_of_mean=laplace_weight*laplace_weight*(kind==0 ? 4.:1.)*variance_of_mean;
        }
    return result.mean;
}

} // namespace stc_mp2
class stc_mp2_kernel {
public:
    stc_mp2_kernel(Inputs& inputs, const Controls& controls, Result& output, bool has_seed);

    void estimate_n_sample(const arma::vec& beta, const arma::vec& weights,
                           double energy_tolerance);
    enum class sampling_phase { pilot, production };
    // Per-point proposals and pilot statistics; never owns a copy of B.
    struct sampling_context {
        sampling_context(size_t nocc, size_t laplace_point)
            : point(laplace_point), exchange_tables(nocc*nocc),
              direct_tables(nocc), exchange_pilots(2), direct_pilots(2),
              production_counts(4, 0) {}
        size_t point;
        arma::umat keep;
        std::vector<stc_mp2::alias_distribution> atom_distributions, bare_atom_distributions;
        stc_mp2::residual_tables exchange_tables, direct_tables;
        std::vector<stc_mp2::pilot_stats> exchange_pilots, direct_pilots;
        std::vector<size_t> production_counts;
        unsigned exchange_seed = 0, direct_seed = 20260920u;
        double exchange_dense = 0.0, direct_dense = 0.0;
        double exchange_variance = 0.0, direct_variance = 0.0;
        double direct_dense_seconds = 0.0;
    };
    sampling_context prepare_sampling(size_t point);
    void allocate_samples(sampling_context& context) const;
    void sample_mp2_exchange(double& energy, sampling_context& context,
                             sampling_phase phase);
    void sample_mp2_direct(double& energy, sampling_context& context,
                           sampling_phase phase);
    void select_occupied(const std::vector<size_t>& indices);
    void prepare_selected_point(double beta);
    void begin_gradient_sampling();
    void finish_gradient_sampling(sampling_context& context, arma::mat& gamma,
                                  arma::mat* b_gradient = nullptr);
    void backpropagate_laplace_point(double beta, arma::mat& gamma,
                                    arma::mat& f_occ_gradient,
                                    arma::mat& f_virt_gradient,
                                    arma::mat& b_gradient) const;

private:
    const Controls& settings_;
    Result& output_;
    bool has_seed_;
    using sampling_pilot_stats = stc_mp2::pilot_stats;

    struct gamma_update {
        uint32_t atom;
        uint32_t target;
        uint32_t source;
        double alpha;
        bool bare;
    };

    double select_domain_tau();
    void record_gamma_update(size_t atom, size_t target, size_t source,
                             double alpha, bool bare = false);
    void scatter_gamma_updates(arma::mat& gamma, arma::mat* b_gradient);
    static arma::mat symmetric_exponential_vjp(
        const arma::mat& fock, double coefficient, const arma::mat& adjoint);

    bool selected(size_t i) const { return selected_mask_[i]; }
    double selected_dense(const arma::umat& keep, bool exchange,
                          double weight = 0, arma::mat* gamma = nullptr,
                          arma::mat* b_gradient = nullptr);
    // Four main tensors in selected mode: original_B_ia_, B_ia_ (full-time C),
    // the caller's Gamma, and the caller's accumulated B cotangent.
    std::vector<bool> selected_mask_;
    arma::mat bare_orbital_norms_, bare_atom_weights_, bare_atom_weight_sums_;
    size_t nocc_ = 0;
    size_t nvirt_ = 0;
    size_t naux_ = 0;
    size_t natom_ = 0;
    arma::Col<size_t> offset_aux_;
    arma::mat occupied_fock_;
    arma::mat virtual_fock_;
    arma::mat B_ia_;
    arma::mat original_B_ia_;
    arma::vec laplace_weights_;
    arma::vec variance_targets_;
    arma::mat orbital_norms_;
    arma::mat atom_weights_;
    arma::mat atom_weight_sums_;
    double selected_domain_tau_ = 6.5e-3;
    bool record_gradient_ = false;
    bool full_domain_ = false;
    std::vector<std::vector<gamma_update> > gradient_updates_;
};

using stc_mp2::alias_distribution;
using stc_mp2::residual_tables;
stc_mp2_kernel::stc_mp2_kernel(Inputs& inputs, const Controls& controls,
                                    Result& output, bool has_seed)
    : settings_(controls), output_(output), has_seed_(has_seed) {
    nocc_=inputs.foo.n_rows; nvirt_=inputs.fvv.n_rows; naux_=inputs.B.n_rows;
    occupied_fock_=std::move(inputs.foo); virtual_fock_=std::move(inputs.fvv);
    original_B_ia_=std::move(inputs.B);
    offset_aux_.set_size(inputs.aux_offsets.size());
    for(size_t k=0;k<inputs.aux_offsets.size();++k) offset_aux_(k)=inputs.aux_offsets[k];
    natom_=offset_aux_.n_elem-1;
    full_domain_=!settings_.stochastic;
}
void stc_mp2_kernel::estimate_n_sample(
        const arma::vec& beta, const arma::vec& weights, double energy_tolerance) {
    if (beta.is_empty() || beta.n_elem != weights.n_elem)
        throw std::runtime_error("STC-MP2 requires matching nonempty Laplace nodes and weights");
    if (!(energy_tolerance > 0.0) || !std::isfinite(energy_tolerance))
        throw std::runtime_error("STC-MP2 energy tolerance must be positive and finite");

    laplace_weights_ = weights;
    const size_t n_points = beta.n_elem;
    variance_targets_.set_size(n_points);
    if (!settings_.adaptive) {
        // Fixed-count/deterministic test modes do not allocate a variance budget.
        variance_targets_.ones();
        output_.point_variance_targets=arma::conv_to<std::vector<double>>::from(variance_targets_);
        return;
    }

    arma::vec v_estimator(n_points);

    for (size_t i = 0; i < n_points; i++){
        arma::mat m_F_occ_exp = arma::expmat_sym(beta(i) * occupied_fock_ / 2);
        arma::mat m_F_virt_exp = arma::expmat_sym(-beta(i) * virtual_fock_ / 2);

        v_estimator(i) = laplace_weights_(i) * std::pow(arma::trace(m_F_occ_exp) * arma::trace(m_F_virt_exp), 1.4);
    }
    // variance_targets_ is the target variance of the unweighted contraction E_l.
    // The Laplace energy is sum_l w_l E_l, so allocate Var(w_l E_l) and convert
    // back to Var(E_l) by dividing by w_l^2.
    double v_est_sum = arma::accu(v_estimator);
    for (size_t i = 0; i < n_points; i++){
        variance_targets_(i) = energy_tolerance * energy_tolerance * v_estimator(i)
            / v_est_sum / (laplace_weights_(i) * laplace_weights_(i));
    }
    output_.point_variance_targets=arma::conv_to<std::vector<double>>::from(variance_targets_);
}

double stc_mp2_kernel::select_domain_tau() {
    selected_domain_tau_=settings_.system_workload_cutoff;
    return selected_domain_tau_;
}

stc_mp2_kernel::sampling_context stc_mp2_kernel::prepare_sampling(size_t point) {
    sampling_context context(nocc_, point);
    // Rank-one magnitude models are separate for B and C. The full-energy
    // path uses its half-time tensor for both roles, as before.
    const bool selected_mode = true;
    auto build_weights = [&](const arma::mat& factor, arma::mat& norms,
                             arma::mat& atoms, arma::mat& sums) {
        norms.set_size(nocc_, nvirt_);
        arma::vec rows(naux_);
        #pragma omp parallel for collapse(2)
        for (size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a)
            norms(i,a)=arma::norm(factor.col(i*nvirt_+a));
        #pragma omp parallel for
        for(size_t p=0;p<naux_;++p) rows(p)=arma::norm(factor.row(p));
        atoms.zeros(natom_,nocc_*nvirt_); sums.zeros(nocc_,nvirt_);
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
            const size_t ia=i*nvirt_+a;
            for(size_t at=0;at<natom_;++at)
                for(size_t p=offset_aux_(at);p<offset_aux_(at+1);++p)
                    atoms(at,ia)+=std::abs(factor(p,ia))*rows(p);
            sums(i,a)=arma::accu(atoms.col(ia));
        }
    };
    build_weights(B_ia_,orbital_norms_,atom_weights_,atom_weight_sums_);
    if (selected_mode && bare_orbital_norms_.is_empty())
        build_weights(original_B_ia_,bare_orbital_norms_,bare_atom_weights_,bare_atom_weight_sums_);
    const arma::mat& bare_norms=selected_mode ? bare_orbital_norms_ : orbital_norms_;
    const double w_lap=laplace_weights_(point), w14=std::pow(std::abs(w_lap),.25);
    const double tau=select_domain_tau();
    auto& keep=context.keep;
    keep.zeros(nocc_,nvirt_);
    for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
        // Geometric mean of the two orbital norms; no extra factor tensor.
        const double norm=selected_mode ? std::sqrt(bare_norms(i,a)*orbital_norms_(i,a))
                                        : orbital_norms_(i,a);
        keep(i,a)=full_domain_ || w14*norm>tau;
    }

    auto make_distributions = [&](arma::mat& norms, arma::mat& atoms, arma::mat& sums,
                                 std::vector<alias_distribution>& distributions) {
        // Keep derivative support at exact zeros; no proposal mixture.
        if(selected_mode) {
            const double nf=std::max(1e-100,norms.max()*1e-12);
            const double af=std::max(1e-100,atoms.max()*1e-12);
            norms.transform([&](double x){return std::max(x,nf);});
            atoms.transform([&](double x){return std::max(x,af);});
        }
        distributions.resize(nocc_*nvirt_);
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
            const size_t ia=i*nvirt_+a;
            sums(i,a)=arma::accu(atoms.col(ia));
            distributions[ia].reset(atoms.colptr(ia),atoms.colptr(ia)+natom_);
        }
    };
    make_distributions(orbital_norms_,atom_weights_,atom_weight_sums_,context.atom_distributions);
    if(selected_mode)
        make_distributions(bare_orbital_norms_,bare_atom_weights_,bare_atom_weight_sums_,context.bare_atom_distributions);

    context.exchange_seed = has_seed_ ? unsigned(settings_.global_seed) : std::random_device{}();
    context.direct_seed = 20260920u + (has_seed_ ? unsigned(settings_.global_seed) : 0u);
    return context;
}

void stc_mp2_kernel::sample_mp2_exchange(
        double& E_corr, sampling_context& context, sampling_phase phase) {
    const size_t point = context.point;
    const double w_lap = laplace_weights_(point);
    const auto& keep = context.keep;
    const auto& dist_A_ia = context.atom_distributions;
    const auto& dist_B_ia = context.bare_atom_distributions;
    const arma::mat& bn = bare_orbital_norms_;
    const arma::mat& bw = bare_atom_weights_;
    const arma::mat& bs = bare_atom_weight_sums_;
    const double t0 = omp_get_wtime();
    auto in_D = [&](size_t i, size_t j, size_t a) -> bool {
        return keep(i, a) && keep(j, a);
    };

    if (phase == sampling_phase::pilot) {
        context.exchange_dense = selected_dense(keep, true);
    }
    arma::mat xi_ij_a, eta_ij_a;
    arma::vec w_ij;
    if (phase == sampling_phase::pilot) {
        xi_ij_a.set_size(nvirt_, nocc_*nocc_);
        eta_ij_a.set_size(nvirt_, nocc_*nocc_);
        w_ij.set_size(nocc_*nocc_);
    }
    const unsigned exchange_seed = context.exchange_seed;
    auto& exchange_tables = context.exchange_tables;

    auto fill_safe_dist = [&](alias_distribution& dist, double* col) {
        double s = 0.0;
        for (size_t k = 0; k < nvirt_; k++){
            s += col[k];
        }
        if (s == 0.0){
            col[0] = 1.0;
        }
        dist.reset(col, col + nvirt_);
        if (s == 0.0){
            col[0] = 0.0;
        }
    };
    auto run_exchange_residual = [&](bool allow_a_in, bool allow_a_out,
                         bool allow_b_in, bool allow_b_out,
                         const char* tag, size_t term_id,
                         sampling_pilot_stats* pilot_stats,
                         size_t production_count) -> double {
        const double term_start = omp_get_wtime();
        std::vector<alias_distribution>& dist_a_ij = exchange_tables.first[term_id];
        std::vector<alias_distribution>& dist_b_ij = exchange_tables.second[term_id];
        if (!exchange_tables.loaded[term_id]) {
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < nocc_; i++){
            for (size_t j = 0; j < nocc_; j++){
                size_t ij = i*nocc_ + j;
                for (size_t a = 0; a < nvirt_; a++){
                    const bool d = in_D(i, j, a);
                    const bool a_ok = (d && allow_a_in) || ((!d) && allow_a_out);
                    const bool b_ok = (d && allow_b_in) || ((!d) && allow_b_out);
                    xi_ij_a(a, ij) = a_ok ? bs(i, a) * atom_weight_sums_(j, a) : 0.0;
                    eta_ij_a(a, ij) = b_ok ? orbital_norms_(i, a) * bn(j, a) : 0.0;
                }
            }
        }
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < nocc_; i++){
            for (size_t j = 0; j < nocc_; j++){
                size_t ij = i*nocc_ + j;
                double xi = 0.0;
                double eta = 0.0;
                for (size_t a = 0; a < nvirt_; a++){
                    xi += xi_ij_a(a, ij);
                    eta += eta_ij_a(a, ij);
                }
                w_ij(ij) = selected(i) ? xi * eta : 0.0;
                fill_safe_dist(dist_a_ij[ij], xi_ij_a.colptr(ij));
                fill_safe_dist(dist_b_ij[ij], eta_ij_a.colptr(ij));
            }
        }
        exchange_tables.normalization[term_id] = arma::accu(w_ij);
        exchange_tables.outer[term_id].reset(w_ij.begin(), w_ij.end());
        exchange_tables.loaded[term_id] = true;
        }
        const double Z = exchange_tables.normalization[term_id];
        auto one_sample = [&](std::mt19937& local_rng) -> double {
            size_t ij = exchange_tables.outer[term_id](local_rng);
            size_t i = ij / nocc_;
            size_t j = ij % nocc_;
            size_t a = dist_a_ij[i*nocc_ + j](local_rng);
            size_t b = dist_b_ij[i*nocc_ + j](local_rng);
            size_t ia = i*nvirt_ + a;
            size_t ja = j*nvirt_ + a;
            size_t At = dist_B_ia[ia](local_rng);
            size_t Ct = dist_A_ia[ja](local_rng);
            double wA = bw(At, ia);
            double wC = atom_weights_(Ct, ja);
            if (wA == 0.0 || wC == 0.0 || orbital_norms_(i, b) == 0.0 || bn(j, b) == 0.0){
                return 0.0;
            }
            size_t ib = i*nvirt_ + b;
            size_t jb = j*nvirt_ + b;
            const double sP = stc_mp2::group_dot(original_B_ia_, offset_aux_, At, ia, jb);
            const double sQ = stc_mp2::group_dot(B_ia_, offset_aux_, Ct, ib, ja);
            const double denominator = orbital_norms_(i, b)*bn(j, b)*wA*wC;
            if (record_gradient_ && pilot_stats == NULL && production_count != 0) {
                const double inverse_probability = Z/denominator;
                const double scale = w_lap*inverse_probability/double(production_count);
                record_gamma_update(At, ia, jb, scale*sQ, true);
                record_gamma_update(At, jb, ia, scale*sQ, true);
                record_gamma_update(Ct, ib, ja, scale*sP);
                record_gamma_update(Ct, ja, ib, scale*sP);
            }
            return sP*sQ*Z/denominator;
        };
        double variance_of_mean = 0.0;
        const double result = stc_mp2::run_residual(
            tag, Z, term_id, 0.25*variance_targets_(point), production_count,
            exchange_seed, point, one_sample, pilot_stats, variance_of_mean, settings_, output_, w_lap);
        if (!pilot_stats) context.exchange_variance += variance_of_mean;

        return result;
    };

    if (phase == sampling_phase::pilot) {
        run_exchange_residual(false, true, true, true, "stoch a~D", 0,
                              &context.exchange_pilots[0], 0);
        run_exchange_residual(true, false, false, true, "stoch b~D", 1,
                              &context.exchange_pilots[1], 0);
        E_corr = context.exchange_dense;
        return;
    }
    const double E3 = run_exchange_residual(true, false, false, true, "stoch b~D", 1,
                                          NULL, context.production_counts[1]);
    const double E2 = run_exchange_residual(false, true, true, true, "stoch a~D", 0,
                                          NULL, context.production_counts[0]);
    E_corr = context.exchange_dense + E2 + E3;

    
    context.exchange_tables = residual_tables(0);
}

void stc_mp2_kernel::allocate_samples(sampling_context& context) const {
    const size_t point = context.point;
    const auto& exchange_pilots = context.exchange_pilots;
    const auto& direct_pilots = context.direct_pilots;
    const double budget_coefficient = 2.0;

    std::vector<sampling_pilot_stats> pilots(4);
    pilots[0] = exchange_pilots[0];
    pilots[1] = exchange_pilots[1];
    pilots[2] = direct_pilots[0];
    pilots[3] = direct_pilots[1];
    const char* labels[4] = {"exchange a~D", "exchange b~D",
                             "direct a~D", "direct b~D"};
    double coefficients[4] = {1.0, 1.0, budget_coefficient, budget_coefficient};
    double allocation_weight[4] = {0.0, 0.0, 0.0, 0.0};
    double weight_sum = 0.0;
    for (size_t k = 0; k < 4; ++k) {
        const double contribution_variance = coefficients[k]*coefficients[k]*pilots[k].variance;
        const double sample_cost = std::max(1.0e-15, pilots[k].seconds_per_sample);
        allocation_weight[k] = std::sqrt(std::max(0.0, contribution_variance*sample_cost));
        weight_sum += allocation_weight[k];
    }
    if (weight_sum < 0.0 || !std::isfinite(weight_sum))
        throw std::runtime_error("Invalid pilot statistics in four-way variance allocation");

    size_t active_estimators = 0;
    for (size_t k = 0; k < 4; ++k)
        if (pilots[k].count > 0) ++active_estimators;

    auto& production_counts = context.production_counts;

    for (size_t k = 0; k < 4; ++k) {
        const double share = weight_sum > 0.0 ? allocation_weight[k]/weight_sum
            : (pilots[k].count > 0 ? 1.0/double(active_estimators) : 0.0);
        const double contribution_target = variance_targets_(point)*share;
        const double contribution_variance = coefficients[k]*coefficients[k]*pilots[k].variance;
        double requested = contribution_target > 0.0
                         ? std::ceil(contribution_variance/contribution_target) : 0.0;
        if (!std::isfinite(requested) || requested >= double(std::numeric_limits<size_t>::max()))
            throw std::runtime_error("Dynamic sampling budget overflow");
        if (pilots[k].count > 0)
            requested = std::max(double(settings_.min_production_samples), requested);
        if (!settings_.adaptive && pilots[k].count > 0) requested=double(settings_.production_samples);
        if (settings_.max_production_samples && requested>double(settings_.max_production_samples))
            throw std::runtime_error("Selected sampling budget exceeds max_production_samples");
        production_counts[k] = static_cast<size_t>(requested);

    }

}

void stc_mp2_kernel::sample_mp2_direct(
        double& E_corr, sampling_context& context, sampling_phase phase) {
    const size_t point = context.point;
    const auto& keep = context.keep;
    const auto& atom_distributions = context.atom_distributions;
    const auto& production_counts = context.production_counts;
    const double t0 = omp_get_wtime();
    const size_t nx = nocc_ * nvirt_;
    const double w = laplace_weights_(point);

    // Direct term, with x=(ia) and y=(jb):
    //   D = sum_xyPQ R_Px R_Qx R_Py R_Qy.
    // As in the exchange sampler, use the rank-one magnitude model
    // |R_Px| ~ A_x B_P for two factors.  The importance function is
    //   pi(x,y,P,Q) ~ A_x^2 B_P B_Q |R_Py| |R_Qy|,
    // which factors as p(ij) p(a|i) p(b|j) p(P|jb) p(Q|jb).
    // Auxiliary indices are sampled as atom groups and summed exactly inside
    // each selected group.
    const arma::mat& atom_weight = atom_weights_;
    const auto& bare_distributions = context.bare_atom_distributions;
    const arma::mat& bn = bare_orbital_norms_;
    const arma::mat& bw = bare_atom_weights_;
    const arma::mat& bs = bare_atom_weight_sums_;
    arma::vec z_y(nx, arma::fill::zeros);
    for (size_t y = 0; y < nx; ++y) {
        const size_t i = y/nvirt_, a = y%nvirt_;
        z_y(y) = atom_weight_sums_(i,a);
    }

    const double t_tables = omp_get_wtime() - t0;

    double t_dense = 0.0;
    if (phase == sampling_phase::pilot) {
        const double t_dense0 = omp_get_wtime();
        context.direct_dense = selected_dense(keep, false);
        t_dense = omp_get_wtime()-t_dense0;
        context.direct_dense_seconds = t_dense;
    }
    const unsigned seed = context.direct_seed;
    double variance = 0.0;
    arma::vec sum_a(nocc_, arma::fill::zeros);
    arma::vec sum_b(nocc_, arma::fill::zeros);
    arma::mat weight_a(nvirt_, nocc_, arma::fill::zeros);
    arma::mat weight_b(nvirt_, nocc_, arma::fill::zeros);
    arma::vec weight_ij(nocc_*nocc_, arma::fill::zeros);
    auto& direct_tables = context.direct_tables;
    auto run_direct_residual = [&](bool x_strong, bool allow_y_weak,
                         bool allow_y_strong, const char* tag, size_t term_id,
                         sampling_pilot_stats* pilot_stats,
                         size_t production_count) -> double {
        const double term_start = omp_get_wtime();
        // For the strong-x/weak-y residual, exploit the x<->y symmetry of
        // (B_x^T B_y)^2 and put the atom-resolved proposal on the strong side.
        // This is the same orientation used by the local STC direct sampler.
        const bool reverse_atom_source = x_strong && !allow_y_strong;
        std::vector<alias_distribution>& dist_a_i = direct_tables.first[term_id];
        std::vector<alias_distribution>& dist_b_j = direct_tables.second[term_id];
        if (!direct_tables.loaded[term_id]) {
        sum_a.zeros();
        sum_b.zeros();
        weight_a.zeros();
        weight_b.zeros();
        for (size_t i = 0; i < nocc_; ++i) {
            for (size_t a = 0; a < nvirt_; ++a) {
                const size_t x = i*nvirt_+a;
                if (bool(keep(i,a)) == x_strong)
                    weight_a(a,i) = reverse_atom_source ? bs(i,a)*z_y(x)
                        : bn(i,a)*orbital_norms_(i,a);
                const bool y_ok = (keep(i,a) && allow_y_strong)
                               || (!keep(i,a) && allow_y_weak);
                if (y_ok)
                    weight_b(a,i) = reverse_atom_source
                        ? bn(i,a)*orbital_norms_(i,a) : bs(i,a)*z_y(x);
            }
            dist_a_i[i].reset(weight_a.colptr(i), weight_a.colptr(i)+nvirt_);
            dist_b_j[i].reset(weight_b.colptr(i), weight_b.colptr(i)+nvirt_);
            sum_a(i) = arma::accu(weight_a.col(i));
            sum_b(i) = arma::accu(weight_b.col(i));
        }
        weight_ij.zeros();
        for (size_t i = 0; i < nocc_; ++i)
            for (size_t j = 0; j < nocc_; ++j)
                weight_ij(i*nocc_+j) = selected(i) ? sum_a(i)*sum_b(j) : 0.0;
        direct_tables.normalization[term_id] = arma::accu(weight_ij);
        direct_tables.outer[term_id].reset(weight_ij.begin(), weight_ij.end());
        direct_tables.loaded[term_id] = true;
        }
        const double Z = direct_tables.normalization[term_id];
        auto one_sample = [&](std::mt19937& local_rng) -> double {
            const size_t ij = direct_tables.outer[term_id](local_rng);
            const size_t i = ij/nocc_, j = ij%nocc_;
            const size_t a = dist_a_i[i](local_rng), b = dist_b_j[j](local_rng);
            const size_t x = i*nvirt_+a, y = j*nvirt_+b;
            const size_t atom_source = reverse_atom_source ? x : y;
            const double norm_product = reverse_atom_source ? bn(j,b)*orbital_norms_(j,b) : bn(i,a)*orbital_norms_(i,a);
            const size_t at = bare_distributions[atom_source](local_rng);
            const size_t ct = atom_distributions[atom_source](local_rng);
            const double denom = norm_product*bw(at,atom_source)*
                                 atom_weight(ct,atom_source);
            if (denom == 0.0) return 0.0;
            const double sP = stc_mp2::group_dot(original_B_ia_, offset_aux_, at, x, y);
            const double sQ = stc_mp2::group_dot(B_ia_, offset_aux_, ct, x, y);
            if (record_gradient_ && pilot_stats == NULL && production_count != 0) {
                const double inverse_probability = Z/denom;
                const double scale = -2.0*w*inverse_probability/double(production_count);
                record_gamma_update(at, x, y, scale*sQ, true);
                record_gamma_update(at, y, x, scale*sQ, true);
                record_gamma_update(ct, x, y, scale*sP);
                record_gamma_update(ct, y, x, scale*sP);
            }
            return sP*sQ*Z/denom;
        };
        double term_variance = 0.0;
        const double result = stc_mp2::run_residual(
            tag, Z, term_id, variance_targets_(point)/16.0, production_count,
            seed, point, one_sample, pilot_stats, term_variance, settings_, output_, w);
        variance += term_variance;

        return result;
    };

    if (phase == sampling_phase::pilot) {
        run_direct_residual(false, true, true, "direct a~D", 0,
                            &context.direct_pilots[0], 0);
        run_direct_residual(true, true, false, "direct b~D", 1,
                            &context.direct_pilots[1], 0);
        E_corr = context.direct_dense;
        return;
    }

    // Both pilot table sets remain resident for production.
    const double direct_b = run_direct_residual(true, true, false, "direct b~D", 1, NULL,
                                      production_counts[3]);
    const double direct_a = run_direct_residual(false, true, true, "direct a~D", 0, NULL,
                                      production_counts[2]);
    // Weak a / all b plus strong a / weak b covers the complement of dense.
    E_corr = context.direct_dense + direct_a + direct_b;
    const double elapsed = omp_get_wtime() - t0;
    context.direct_variance = variance;

    
    context.direct_tables = residual_tables(0);
}

// Bound aliasing GEMM temporaries: each panel owns disjoint matrix rows.
// This keeps the selected path near four full tensors during rotations too.
static void right_multiply_panels(arma::mat& matrix, const arma::mat& right) {
    const size_t rows=std::max(size_t(1),size_t(32*1024*1024)/(sizeof(double)*size_t(matrix.n_cols)));
    for(size_t first=0;first<matrix.n_rows;first+=rows) {
        const size_t last=std::min(size_t(matrix.n_rows),first+rows)-1;
        matrix.rows(first,last)=matrix.rows(first,last)*right;
    }
}

// Selected mode uses separate bare-B and full-time-C proposal weights.
void stc_mp2_kernel::select_occupied(const std::vector<size_t>& indices) {
    if (indices.empty()) throw std::invalid_argument("Empty occupied group");
    selected_mask_.assign(nocc_, false);
    for (size_t i : indices) {
        if (i >= nocc_ || selected_mask_[i]) throw std::invalid_argument("Invalid occupied group");
        selected_mask_[i] = true;
    }
}

void stc_mp2_kernel::prepare_selected_point(double beta) {
    B_ia_ = original_B_ia_;
    const arma::mat o = arma::expmat_sym(beta*occupied_fock_);
    const arma::mat v = arma::expmat_sym(-beta*virtual_fock_);
    arma::mat pa_i(B_ia_.memptr(), naux_*nvirt_, nocc_, false, true);
    right_multiply_panels(pa_i,o);
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<nocc_; ++i) {
        arma::mat pa(B_ia_.colptr(i*nvirt_), naux_, nvirt_, false, true);
        pa = pa*v;
    }
}

// Ordered selected-i pairs. Direct retains D_i x D_j; exchange D_ij x D_ij.
// Pair products run in parallel. Only the short additive column updates are
// locked, by occupied index; no thread owns a full-size gradient copy.
double stc_mp2_kernel::selected_dense(const arma::umat& keep, bool exchange,
                                     double weight, arma::mat* gamma, arma::mat* b_gradient) {
    const double start=omp_get_wtime();
    if (gamma && !b_gradient) throw std::invalid_argument("Missing selected B cotangent");
    std::vector<omp_lock_t> locks(gamma ? nocc_ : 0);
    for (auto& lock : locks) omp_init_lock(&lock);
    double energy=0;
    #pragma omp parallel for collapse(2) reduction(+:energy) schedule(dynamic)
    for (size_t i=0; i<nocc_; ++i)
        for (size_t j=0; j<nocc_; ++j) {
            if (!selected(i)) continue;
            std::vector<arma::uword> ci,cj;
            for (size_t a=0; a<nvirt_; ++a) {
                if (keep(i,a) && (!exchange || keep(j,a))) ci.push_back(i*nvirt_+a);
                if (keep(j,a) && (!exchange || keep(i,a))) cj.push_back(j*nvirt_+a);
            }
            if (ci.empty() || cj.empty()) continue;
            const arma::uvec ix(ci),jx(cj);
            const arma::mat bi=original_B_ia_.cols(ix), bj=original_B_ia_.cols(jx);
            const arma::mat di=B_ia_.cols(ix), dj=B_ia_.cols(jx);
            const arma::mat gb=bi.t()*bj, gc=di.t()*dj;
            arma::mat db=gc, dc=gb;
            if (exchange) { db=gc.t(); dc=gb.t(); }
            energy+=arma::accu(gb%db);
            if (gamma) {
                // Compute outside the lock, then scatter to one orbital block.
                arma::mat delta_b=weight*bj*db.t(), delta_c=weight*dj*dc.t();
                omp_set_lock(&locks[i]);
                b_gradient->cols(ix)+=delta_b;
                gamma->cols(ix)+=delta_c;
                omp_unset_lock(&locks[i]);
                delta_b=weight*bi*db; delta_c=weight*di*dc;
                // Separate acquisitions also handle i==j without deadlock.
                omp_set_lock(&locks[j]);
                b_gradient->cols(jx)+=delta_b;
                gamma->cols(jx)+=delta_c;
                omp_unset_lock(&locks[j]);
            }
        }
    for (auto& lock : locks) omp_destroy_lock(&lock);

    return energy;
}

void stc_mp2_kernel::record_gamma_update(
        size_t atom, size_t target, size_t source, double alpha, bool bare) {
    if (!record_gradient_ || alpha == 0.0) return;
    if (atom > std::numeric_limits<uint32_t>::max()
            || target > std::numeric_limits<uint32_t>::max()
            || source > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("STC-MP2 gamma update exceeds 32-bit index range");
    const int thread = omp_in_parallel() ? omp_get_thread_num() : 0;
    gradient_updates_.at(static_cast<size_t>(thread)).push_back(gamma_update{
        static_cast<uint32_t>(atom), static_cast<uint32_t>(target),
        static_cast<uint32_t>(source), alpha, bare});
}

void stc_mp2_kernel::scatter_gamma_updates(arma::mat& gamma, arma::mat* b_gradient) {
    const size_t nx = nocc_*nvirt_;
    const size_t key_count = natom_*nx;
    size_t update_count = 0;
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        update_count += updates.size();

    std::vector<size_t> offsets(key_count+1, 0);
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        for (const gamma_update& update : updates)
            ++offsets[size_t(update.atom)*nx+size_t(update.target)+1];
    for (size_t key = 0; key < key_count; ++key)
        offsets[key+1] += offsets[key];

    // Store only an index permutation, rather than a second copy of every
    // update.  The records themselves remain in per-thread append buffers.
    std::vector<const gamma_update*> ordered(update_count);
    std::vector<size_t> cursor(offsets.begin(), offsets.end()-1);
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        for (const gamma_update& update : updates) {
            const size_t key = size_t(update.atom)*nx+size_t(update.target);
            ordered[cursor[key]++] = &update;
        }

    if (naux_ > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("STC-MP2 auxiliary dimension exceeds DAXPY range");
    #pragma omp parallel for schedule(dynamic,16)
    for (size_t key = 0; key < key_count; ++key) {
        const size_t atom = key/nx;
        const size_t target = key%nx;
        const size_t first = offset_aux_(atom);
        const size_t count = offset_aux_(atom+1)-first;
        for (size_t position = offsets[key]; position < offsets[key+1]; ++position) {
            const gamma_update& update = *ordered[position];
            const double* source=(update.bare ? original_B_ia_ : B_ia_).colptr(update.source)+first;
            double* destination=(update.bare ? *b_gradient : gamma).colptr(target)+first;
            for(size_t p=0;p<count;++p) destination[p]+=update.alpha*source[p];
        }
    }

}

void stc_mp2_kernel::begin_gradient_sampling() {
    if (record_gradient_)
        throw std::runtime_error("Nested STC-MP2 gradient sampling is not supported");
    const int thread_count = std::max(1, omp_get_max_threads());
    gradient_updates_.clear();
    gradient_updates_.resize(static_cast<size_t>(thread_count));
    record_gradient_ = true;
}

void stc_mp2_kernel::finish_gradient_sampling(
        sampling_context& context, arma::mat& gamma, arma::mat* b_gradient) {
    if (!record_gradient_)
        throw std::runtime_error("STC-MP2 gradient sampling was not started");
    record_gradient_ = false;
    // Sampling is complete: release proposals before allocating gamma.
    std::vector<stc_mp2::alias_distribution>().swap(context.atom_distributions);
    std::vector<stc_mp2::alias_distribution>().swap(context.bare_atom_distributions);

        gamma.zeros(naux_, nocc_*nvirt_);
        if (!b_gradient) throw std::invalid_argument("Selected gradients require the B accumulator");
        if (b_gradient->is_empty()) b_gradient->zeros(naux_, nocc_*nvirt_);
        selected_dense(context.keep, true, laplace_weights_(context.point), &gamma, b_gradient);
        selected_dense(context.keep, false, -2*laplace_weights_(context.point), &gamma, b_gradient);

    scatter_gamma_updates(gamma, b_gradient);
    gradient_updates_.clear();
}

arma::mat stc_mp2_kernel::symmetric_exponential_vjp(
        const arma::mat& fock, double coefficient, const arma::mat& adjoint) {
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    if (!arma::eig_sym(eigenvalues, eigenvectors, fock))
        throw std::runtime_error("STC-MP2 Fock eigendecomposition failed");
    const arma::mat symmetric_adjoint = 0.5*(adjoint+adjoint.t());
    arma::mat eigen_adjoint = eigenvectors.t()*symmetric_adjoint*eigenvectors;
    arma::mat divided_difference(fock.n_rows, fock.n_cols);
    for (size_t p = 0; p < fock.n_rows; ++p) {
        for (size_t q = 0; q < fock.n_cols; ++q) {
            const double xp=coefficient*eigenvalues(p), xq=coefficient*eigenvalues(q);
            const double gap=std::abs(xp-xq);
            // Avoid subtracting nearly equal exponentials for degenerate LMOs.
            divided_difference(p,q)=coefficient*std::exp(std::max(xp,xq))
                *(gap==0 ? 1.0 : -std::expm1(-gap)/gap);
        }
    }
    arma::mat result = eigenvectors*(divided_difference%eigen_adjoint)*eigenvectors.t();
    return 0.5*(result+result.t());
}

void stc_mp2_kernel::backpropagate_laplace_point(
        double beta, arma::mat& gamma, arma::mat& f_occ_gradient,
        arma::mat& f_virt_gradient, arma::mat& b_gradient) const {
    if (gamma.n_rows != naux_ || gamma.n_cols != nocc_*nvirt_)
        throw std::runtime_error("Invalid STC-MP2 gamma shape");
    if (f_occ_gradient.is_empty()) {
        f_occ_gradient.zeros(nocc_, nocc_);
        f_virt_gradient.zeros(nvirt_, nvirt_);
    }
    if (b_gradient.is_empty()) b_gradient.zeros(naux_, nocc_*nvirt_);

    const double half_beta = 0.5*beta;
    const arma::mat occupied_exp = arma::expmat_sym(half_beta*occupied_fock_);
    const arma::mat virtual_exp = arma::expmat_sym(-half_beta*virtual_fock_);
    if (original_B_ia_.n_rows != naux_
            || original_B_ia_.n_cols != nocc_*nvirt_)
        throw std::runtime_error("Retain original STC-MP2 integrals before transforming");
    arma::mat h_occ(nocc_, nocc_, arma::fill::zeros);
    arma::mat h_virt(nvirt_, nvirt_, arma::fill::zeros);

    // For each auxiliary P use virtual-by-occupied matrices X and G:
    // B_tilde_P = V X O, h_O = (V X)^T G, h_V = (X O) G^T.
    // Forward contractions preserve the strongly damped virtual components;
    // reconstructing X from B_tilde with an inverse exponential does not.
    #pragma omp parallel
    {
        arma::mat local_occ(nocc_, nocc_, arma::fill::zeros);
        arma::mat local_virt(nvirt_, nvirt_, arma::fill::zeros);
        arma::mat x(nvirt_, nocc_), g(nvirt_, nocc_);
        #pragma omp for schedule(static)
        for (size_t p = 0; p < naux_; ++p) {
            for (size_t i = 0; i < nocc_; ++i)
                for (size_t a = 0; a < nvirt_; ++a) {
                    x(a,i) = original_B_ia_(p,i*nvirt_+a);
                    g(a,i) = gamma(p,i*nvirt_+a);
                }
            const arma::mat vx = virtual_exp*x;
            const arma::mat xo = x*occupied_exp;
            local_occ += vx.t()*g;
            local_virt += xo*g.t();
        }
        #pragma omp critical
        {
            h_occ += local_occ;
            h_virt += local_virt;
        }
    }
    f_occ_gradient += symmetric_exponential_vjp(
        occupied_fock_, half_beta, h_occ);
    f_virt_gradient += symmetric_exponential_vjp(
        virtual_fock_, -half_beta, h_virt);

    // bar(B) = O Gamma V^T.  Gamma is no longer needed, so reuse its buffer.
    arma::mat gamma_pa_i(gamma.memptr(), naux_*nvirt_, nocc_, false, true);
    right_multiply_panels(gamma_pa_i,occupied_exp);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nocc_; ++i) {
        arma::mat gamma_pa(gamma.colptr(i*nvirt_), naux_, nvirt_, false, true);
        gamma_pa = gamma_pa*virtual_exp;
    }
    b_gradient += gamma;
}

} // namespace selected_source
Result solve_selected(Inputs& inputs, const std::vector<arma::uword>& targets,
                      const SelectedControls& selected_controls, bool with_grad) {
    const Controls& controls=selected_controls.numerical;
    Result result;
    const size_t no=inputs.foo.n_rows,nv=inputs.fvv.n_rows,np=inputs.B.n_rows;
    if(with_grad) {
        result.foo.zeros(no,no); result.fvv.zeros(nv,nv); result.B.zeros(np,no*nv);
    }
    if(nv==0 || np==0) return result;
    arma::vec eo,ev;
    if(!arma::eig_sym(eo,inputs.foo) || !arma::eig_sym(ev,inputs.fvv))
        throw std::runtime_error("Selected Fock eigendecomposition failed");
    if(!(ev.min()>eo.max()))
        throw std::invalid_argument("STC-MP2 requires a positive occupied-to-virtual gap");
    selected_source::stc_mp2_kernel kernel(inputs,controls,result,selected_controls.has_global_seed);
    kernel.select_occupied(std::vector<size_t>(targets.begin(),targets.end()));
    const arma::vec roots(controls.roots),weights(controls.weights);
    kernel.estimate_n_sample(roots,weights,controls.energy_tolerance>0 ? controls.energy_tolerance:1.);
    double variance=0;
    for(size_t point=0;point<roots.n_elem;++point) {
        kernel.prepare_selected_point(roots(point));
        auto context=kernel.prepare_sampling(point);
        using phase=selected_source::stc_mp2_kernel::sampling_phase;
        double direct=0,exchange=0;
        kernel.sample_mp2_exchange(exchange,context,phase::pilot);
        kernel.sample_mp2_direct(direct,context,phase::pilot);
        kernel.allocate_samples(context);
        if(with_grad) kernel.begin_gradient_sampling();
        kernel.sample_mp2_direct(direct,context,phase::production);
        kernel.sample_mp2_exchange(exchange,context,phase::production);
        result.energy+=weights(point)*(exchange-2*direct);
        variance+=weights(point)*weights(point)*(context.exchange_variance+4*context.direct_variance);
        if(with_grad) {
            arma::mat gamma;
            kernel.finish_gradient_sampling(context,gamma,&result.B);
            // Source reverse halves its argument: selected dressing uses full root.
            kernel.backpropagate_laplace_point(2*roots(point),gamma,result.foo,result.fvv,result.B);
        }
    }
    result.energy_standard_error=std::sqrt(variance);
    return result;
}

}} // namespace pyscfad::dlno_stc
