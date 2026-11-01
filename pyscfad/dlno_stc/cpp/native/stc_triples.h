#ifndef LIBGMBPT_QCHEM_STC_TRIPLES_H
#define LIBGMBPT_QCHEM_STC_TRIPLES_H

#include <armadillo>
#include <cstddef>
#include <cstdint>

namespace libgmbpt { namespace qchem {

// Asymmetric Laplace numerator and selected-space restricted sampling.
// The full occupied and virtual Fock response is handled by the dresser.
class stc_triples {
public:
    struct triples_result {
        double energy;
        double standard_error;
        size_t samples;
        double probability_setup_seconds;
        double sampling_seconds;
        int setup_threads;
        int sampling_threads;
        size_t pilot_samples;
        double pilot_standard_error;
        double amplitude_packing_seconds;
        double vjp_reverse_seconds;
        size_t independent_pilot_samples;
        double independent_pilot_standard_error;
        double independent_pilot_seconds;
        double projection_setup_seconds;
        size_t resident_tensor_bytes;
        size_t proposal_bytes;
        size_t reverse_buffer_bytes;
        double vjp_grouping_seconds;
        double vjp_accumulation_seconds;
        double vjp_finalize_seconds;
    };
    // Discarded pilot with its own RNG stream; shares tensors and proposal
    // tables with production. nsamples is the minimum production count.
    struct pilot_controls {
        size_t samples;
        std::uint64_t seed;
        double allocation_target;
    };
    struct triples_vjp {
        arma::field<arma::vec> t1;
        arma::field<arma::mat> t2;
        arma::mat fov;
        arma::mat occupied_left, occupied_right;
        arma::mat Bvo, Boo, Bvv;
    };

    // Denominator-free Laplace numerator, contracted directly over a small
    // selected space: sum_s (A^T L)_s (R^T Z)_s, with the physical -1/3 factor.
    // Only one external occupied index is projected; the other legs retain
    // their common half dressing. L=exp(-t Foo/2) U, R=exp(t Foo/2) U.
    static triples_result asymmetric_triples(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::mat& left, const arma::mat& right,
        size_t nsamples, std::uint64_t seed, double target_standard_error,
        triples_vjp* vjp, double cotangent, const arma::mat* fov = nullptr,
        const pilot_controls* pilot = nullptr);

};

}}
#endif
