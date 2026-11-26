#pragma once
#include "dense_matrix.h"
namespace ccsd_native { namespace triples {
struct stc_triples {
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
        size_t workspace_budget_bytes=0;
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
struct triples_prepared {size_t no,nv,ns;std::vector<dense::Matrix> tensors;};
struct triples_vjp {std::vector<dense::Matrix> prepared;};
static constexpr size_t maximum_streaming_blocks=128;
};
}}
