#pragma once
#include <armadillo>
#include <cstdint>
#include <vector>
#include <string>

namespace pyscfad { namespace dlno_stc {
struct Controls {
    std::vector<double> roots, weights;
    arma::uword virtual_block_size = 16;
    std::uint64_t global_seed = 0;
    bool stochastic = false, adaptive = false;
    // Explicit system specialization: diagonal canonical energies are the only
    // differentiable Fock inputs. Offdiagonal Fock cotangents stay zero.
    bool canonical_fock = false;
    double workload_cutoff = 0.1, virtual_keep_fraction = -1.0;
    double system_workload_cutoff = 6.5e-3;
    double uniform_mixture = 0.05, energy_tolerance = 0.0;
    // Zero selects the mode's minimum: adaptive 10000, fixed-count 32.
    std::size_t production_samples = 4096, min_production_samples = 0;
    std::size_t pilot_samples = 100000, max_production_samples = 0;
    // Weighted variance budgets; empty retains the domain's equal point split.
    std::vector<double> point_variance_targets;
};
struct Inputs {
    arma::mat foo, fvv, B, target_projection, partner_weight;
    std::vector<arma::uword> aux_offsets;
};
struct ResidualDiagnostic {
    std::size_t point, term, residual, pilot_samples, production_samples;
    std::uint64_t pilot_seed, production_seed;
    double variance_of_mean;
    std::uint64_t pilot_refinement_seed = 0;
};
struct Result {
    double energy = 0.0, energy_standard_error = 0.0;
    std::vector<ResidualDiagnostic> residuals;
    std::vector<double> point_variance_targets;
    arma::mat foo, fvv, B, target_projection, partner_weight;
};
struct Spectrum { arma::vec values; arma::mat vectors; };
using KeepSets = std::vector<std::vector<arma::uword>>;
KeepSets select_keep(const arma::mat&, const arma::mat&, const arma::mat&,
                     const arma::mat&, arma::uword, const Controls&);
void sample_residuals(const Inputs&, const Controls&, std::size_t,
                      const KeepSets&, const arma::mat&, const arma::mat&,
                      const arma::mat&, const arma::mat&, arma::mat&, arma::mat&,
                      arma::mat&, arma::mat&, bool, Result&);
Result solve(const Inputs&, const Controls&, bool with_grad);
Result solve_full(const Inputs&, const Controls&, bool with_grad);
void contract_full(const Inputs&, const Controls&, std::size_t point,
                   const arma::mat& dressed, arma::mat& gamma, bool with_grad, Result&);
void dress_integrals(const arma::mat&, const arma::mat&, const arma::mat&, arma::mat&);
arma::mat exponential(const Spectrum&, double coefficient, double shift);
arma::mat exponential_vjp(const Spectrum&, double coefficient, double shift,
                          const arma::mat& adjoint);
void propagate_dressing_reverse(const Inputs&, const Spectrum&, const Spectrum&,
                       double beta, double shift, const arma::mat& O,
                       const arma::mat& V, arma::mat& bar_A, Result&);
}}  // namespace pyscfad::dlno_stc
