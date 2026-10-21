#ifndef libgmbpt_qchem_stc_mp2_kernel_qchem_h
#define libgmbpt_qchem_stc_mp2_kernel_qchem_h

#include <armadillo>

#include "stc_mp2_kernel_helpers.h"

namespace libgmbpt {
namespace qchem {

class stc_mp2_kernel {
public:
    stc_mp2_kernel(const arma::mat& fo, const arma::mat& fv, const arma::mat& b,
                   const arma::Col<size_t>& offsets, unsigned seed=20260920u,
                   double energy_domain_tau=6.5e-3);

    void estimate_n_sample(const arma::vec& beta, const arma::vec& weights,
                           double energy_tolerance);
    // Read-only views of the current numerical state; no tensor copies.
    const arma::mat& occupied_fock() const { return occupied_fock_; }
    const arma::mat& virtual_fock() const { return virtual_fock_; }
    const arma::mat& integrals() const { return B_ia_; }
    const arma::Col<size_t>& aux_atom_offsets() const { return offset_aux_; }
    void transform_integrals(double beta);
    void retain_original_integrals(bool release_current = false);
    const arma::mat& original_integrals() const { return original_B_ia_; }
    // Transfer ownership for a scoped evaluation on a shared propagated factor.
    void swap_integrals(arma::mat& factor) { B_ia_.swap(factor); }
    void restore_original_integrals(bool release_original = false);
    void use_full_domain(bool enabled) { full_domain_ = enabled; }
    void detransform_integrals(double beta1, double beta2);
    enum class sampling_phase { pilot, replay_setup, production };
    // Per-point proposals and pilot statistics; never owns a copy of B.
    struct sampling_context {
        sampling_context(size_t nocc, size_t laplace_point)
            : point(laplace_point), exchange_tables(nocc*nocc),
              direct_tables(nocc), exchange_pilots(2), direct_pilots(2),
              production_counts(4, 0) {}
        size_t point;
        arma::umat keep;
        std::vector<stc_mp2::alias_distribution> atom_distributions;
        stc_mp2::residual_tables exchange_tables, direct_tables;
        std::vector<stc_mp2::pilot_stats> exchange_pilots, direct_pilots;
        std::vector<size_t> production_counts;
        unsigned exchange_seed = 20260920u, direct_seed = 20260920u;
        double exchange_dense = 0.0, direct_dense = 0.0;
        double exchange_variance = 0.0, direct_variance = 0.0;
        double direct_dense_seconds = 0.0;
    };
    sampling_context prepare_sampling(size_t point, const arma::mat* norms = nullptr,
                                      const arma::mat* atom_weights = nullptr);
    void allocate_samples(sampling_context& context) const;
    void sample_mp2_exchange(double& energy, sampling_context& context,
                             sampling_phase phase);
    void sample_mp2_direct(double& energy, sampling_context& context,
                           sampling_phase phase);
    void select_occupied(const std::vector<size_t>& indices);
    void prepare_selected_point(double beta);
    void begin_gradient_sampling();
    void finish_gradient_sampling(sampling_context& context, arma::mat& gamma,
                                  arma::mat* b_gradient = nullptr,
                                  bool gamma_preinitialized = false,
                                  double energy_cotangent = 1.0);
    void backpropagate_laplace_point(double beta, arma::mat& gamma,
                                    arma::mat& f_occ_gradient,
                                    arma::mat& f_virt_gradient,
                                    arma::mat& b_gradient) const;

    double weighted_direct_energy() const { return direct_weighted_sample_; }
    double weighted_direct_variance() const { return direct_weighted_variance_; }

private:
    using sampling_pilot_stats = stc_mp2::pilot_stats;

    struct gamma_update {
        uint32_t atom;
        uint32_t target;
        uint32_t source;
        double alpha;
        bool bare;
    };

    double select_domain_tau();
    arma::mat deterministic_gamma(const arma::umat& keep,
                                  double laplace_weight) const;
    void record_gamma_update(size_t atom, size_t target, size_t source,
                             double alpha, bool bare = false);
    void scatter_gamma_updates(arma::mat& gamma, arma::mat* b_gradient,
                               double energy_cotangent);
    static arma::mat symmetric_exponential_vjp(
        const arma::mat& fock, double coefficient, const arma::mat& adjoint);

    bool selected(size_t i) const { return selected_mask_.empty() || selected_mask_[i]; }
    double selected_dense(const arma::umat& keep, bool exchange,
                          double weight = 0, arma::mat* gamma = nullptr,
                          arma::mat* b_gradient = nullptr);
    // Four main tensors in selected mode: original_B_ia_, B_ia_ (full-time C),
    // the caller's Gamma, and the caller's accumulated B cotangent.
    std::vector<bool> selected_mask_;
    arma::mat bare_orbital_norms_, bare_atom_weights_, bare_atom_weight_sums_;
    std::vector<stc_mp2::alias_distribution> bare_atom_distributions_;
    size_t nbasis_ = 0;
    size_t nocc_ = 0;
    size_t nvirt_ = 0;
    size_t naux_ = 0;
    size_t natom_ = 0;
    arma::Col<size_t> offset_aux_;
    arma::mat occupied_coefficients_;
    arma::mat virtual_coefficients_;
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
    double direct_weighted_sample_ = 0.0;
    double direct_weighted_variance_ = 0.0;
    bool record_gradient_ = false;
    bool full_domain_ = false;
    unsigned explicit_seed_ = 20260920u;
    std::vector<std::vector<gamma_update> > gradient_updates_;
};

}  // namespace qchem
}  // namespace libgmbpt

#endif
