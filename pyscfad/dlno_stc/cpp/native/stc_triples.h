#ifndef LIBGMBPT_QCHEM_STC_TRIPLES_H
#define LIBGMBPT_QCHEM_STC_TRIPLES_H

#include <armadillo>
#include <cstddef>
#include <cstdint>

namespace libgmbpt { namespace qchem {

// Self-contained numerical (T) kernel. No Q-Chem runtime or integral I/O is
// needed once the canonical amplitudes, orbital energies, and integrals exist.
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
        arma::vec eocc, evir;
        arma::mat fov;
        arma::vec occupied_projector;
        arma::mat occupied_subspace;
        arma::mat occupied_matrix;
        arma::mat occupied_left, occupied_right;
        arma::mat ovvv, ovoo, ovov;
        arma::mat Bvo, Boo, Bvv;
    };

    static double projected_triples_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::vec& occupied_projector,
        const arma::mat* fov = nullptr);
    static double projected_triples_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::mat& occupied_subspace,
        const arma::mat* fov = nullptr);
    static triples_result projected_triples_stochastic_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::vec& occupied_projector, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr);
    static triples_result projected_triples_stochastic_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::mat& occupied_subspace, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr);
    // Fused whole-system energy and optional input VJP.
    static triples_result whole_triples(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr,
        const pilot_controls* pilot = nullptr);
    static triples_result whole_triples_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr);
    static triples_result projected_triples_stochastic(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::vec& occupied_projector, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr);
    static triples_result projected_triples_stochastic(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::mat& occupied_subspace, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        triples_vjp* vjp = nullptr, double cotangent = 1.0,
        const arma::mat* fov = nullptr);

    // General (possibly nonsymmetric) occupied weight, needed after Laplace
    // similarity dressing of a fragment projector. Supports exact small VJPs.
    static triples_result weighted_triples(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir, const arma::mat& weight,
        size_t nsamples, std::uint64_t seed, double target_standard_error,
        triples_vjp* vjp, double cotangent, const arma::mat* fov = nullptr,
        const pilot_controls* pilot = nullptr);

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

    // Four explicit DF entry points for autodiff callers. The VJP calls
    // compute the energy and input cotangents together in one sampled pass.
    static triples_result whole_energy(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        const arma::mat* fov = nullptr) {
        return whole_triples(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            nsamples,seed,target_standard_error,nullptr,1.0,fov);
    }
    static triples_result whole_energy_vjp(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
        triples_vjp& bar, std::uint64_t seed = 1,
        double target_standard_error = 0, double cotangent = 1.0,
        const arma::mat* fov = nullptr) {
        return whole_triples(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            nsamples,seed,target_standard_error,&bar,cotangent,fov);
    }
    static triples_result selected_energy(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::vec& occupied_projector, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        const arma::mat* fov = nullptr) {
        return projected_triples_stochastic(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            occupied_projector,nsamples,seed,target_standard_error,nullptr,1.0,fov);
    }
    static triples_result selected_energy_vjp(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::vec& occupied_projector, size_t nsamples,
        triples_vjp& bar, std::uint64_t seed = 1,
        double target_standard_error = 0, double cotangent = 1.0,
        const arma::mat* fov = nullptr) {
        return projected_triples_stochastic(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            occupied_projector,nsamples,seed,target_standard_error,&bar,cotangent,fov);
    }
    static triples_result selected_energy(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::mat& occupied_subspace, size_t nsamples,
        std::uint64_t seed = 1, double target_standard_error = 0,
        const arma::mat* fov = nullptr) {
        return projected_triples_stochastic(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            occupied_subspace,nsamples,seed,target_standard_error,nullptr,1.0,fov);
    }
    static triples_result selected_energy_vjp(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
        const arma::vec& eocc, const arma::vec& evir,
        const arma::mat& occupied_subspace, size_t nsamples,
        triples_vjp& bar, std::uint64_t seed = 1,
        double target_standard_error = 0, double cotangent = 1.0,
        const arma::mat* fov = nullptr) {
        return projected_triples_stochastic(t1,t2,Bvo,Boo,Bvv,eocc,evir,
            occupied_subspace,nsamples,seed,target_standard_error,&bar,cotangent,fov);
    }
private:
    static triples_result evaluate_integrals(
        const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
        const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
        const arma::vec& eocc, const arma::vec& evir,
        size_t nsamples, std::uint64_t seed, double target_standard_error,
        triples_vjp* vjp, double cotangent,
        const arma::vec* occupied_projector = nullptr,
        const arma::mat* occupied_subspace = nullptr,
        const arma::mat* fov = nullptr,
        const arma::mat* occupied_matrix = nullptr,
        const pilot_controls* pilot = nullptr);
};

}}
#endif
