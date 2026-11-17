#ifndef LIBGMBPT_QCHEM_STC_TRIPLES_H
#define LIBGMBPT_QCHEM_STC_TRIPLES_H

#include <armadillo>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace libgmbpt { namespace qchem {

// Asymmetric Laplace numerator and selected-space restricted sampling.
// The full occupied and virtual Fock response is handled by the dresser.
class stc_triples {
public:
    // Elapsed worker times overlap across workers; only point wall times are
    // additive wall-clock stages. Fine-grained profiling is explicitly opt-in.
    struct right_profile {
        enum phase { pack, forward_virtual, forward_occupied, contract,
            ket_bar, reverse_gv, reverse_tv, reverse_go, reverse_to, phase_count };
        std::array<double,phase_count> seconds{};
        // GV, T2V, GO, T2O, T1, FOV, T2, GG, in that order.
        std::array<double,8> lock_wait{},store{};
        std::array<size_t,8> lock_calls{};
        double worker_seconds=0.,cpu_seconds=0.,barrier_seconds=0.;
        size_t blocks=0,minor_faults=0,major_faults=0,voluntary_switches=0,involuntary_switches=0;
    };
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
        size_t numerator_cache_peak_bytes;
        size_t rank_four_packing_bytes;
        double left_build_worker_seconds;
        double right_worker_seconds;
        double left_reverse_worker_seconds;
        double left_build_wall_seconds,right_wall_seconds,left_reverse_wall_seconds;
        size_t occupied_block_batch_size,occupied_block_batches;
        size_t grouped_adjoint_groups,grouped_adjoint_max_terms;
        size_t grouped_adjoint_gemm_calls,grouped_adjoint_max_reduction;
        // Worker-seconds: GV, T2V, GO, T2O by pack, GEMM, commit.
        std::array<std::array<double,3>,4> grouped_profile_seconds{};
        double grouped_spin_wall_seconds=0.,grouped_plan_wall_seconds=0.,grouped_jobs_wall_seconds=0.;
        double grouped_commit_wall_seconds=0.;
        bool right_profile_enabled=false;
        std::vector<double> point_right_wall_seconds;
        std::vector<std::vector<double>> point_right_batch_wall_seconds;
        std::vector<right_profile> point_right_profiles;
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

    static constexpr size_t maximum_streaming_blocks=128;
    static triples_result quadrature_stream(const triples_prepared&,
        const std::vector<const triples_prepared*>&,const std::vector<double>&,
        size_t workspace_limit,size_t block_batch_size,std::vector<double>&,std::vector<triples_vjp>*,
        bool profile_right=false);

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
