#ifndef LIBGMBPT_QCHEM_STC_TRIPLES_H
#define LIBGMBPT_QCHEM_STC_TRIPLES_H

#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <vector>

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
        int packing_blas_threads;
        int sampling_blas_threads;
        int finalize_blas_threads;
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
        size_t deterministic_blocks;
        size_t deterministic_workspace_bytes;
    };
    // Discarded pilot with its own RNG stream; shares tensors and proposal
    // tables with production. nsamples is the minimum production count.
    struct pilot_controls {
        size_t samples;
        std::uint64_t seed;
        double allocation_target;
    };
    // Matrices use the tensor bank layout: the last two orbital indices are
    // rows, and the leading (i,a) pair indexes columns.
    struct triples_integrals {
        arma::mat ovvv, ovoo, ovov;
    };
    struct triples_vjp {
        arma::field<arma::vec> t1;
        arma::field<arma::mat> t2;
        arma::mat fov;
        arma::mat occupied_left, occupied_right;
        arma::mat Bvo, Boo, Bvv;
        triples_integrals integrals;
        std::vector<arma::vec> prepared;
    };

    struct triples_prepared {
        size_t no, nv, ns;
        std::vector<arma::vec> tensors;
    };

    // Bounded batch of spin-adapted, undressed selected numerator blocks.
    struct quadrature_cache {
        size_t no=0,nv=0,ns=0,first=0;
        bool gradient=false,finished=false;
        std::vector<arma::vec> values,bars;
    };
    static triples_result quadrature_build(const triples_prepared&,quadrature_cache&,
        size_t first,size_t count,bool gradient);
    static triples_result quadrature_step(const triples_prepared&,quadrature_cache&,
        double weight,triples_vjp*);
    static triples_result quadrature_finish(const triples_prepared&,quadrature_cache&,triples_vjp*);

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
        const pilot_controls* pilot = nullptr,
        const triples_integrals* integrals = nullptr,
        const triples_prepared* prepared = nullptr);

};

}}
#endif
