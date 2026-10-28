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
    };
    struct triples_vjp {
        arma::field<arma::vec> t1;
        arma::field<arma::mat> t2;
        arma::vec eocc, evir;
        arma::mat fov;
        arma::vec occupied_projector;
        arma::mat occupied_subspace;
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
        const arma::mat* fov = nullptr);
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
        const arma::mat* fov = nullptr);
};

}}
#endif
