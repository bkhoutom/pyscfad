#ifndef LIBGMBPT_QCHEM_STC_MP2_SELECTED_ENERGY_DENSITY_H
#define LIBGMBPT_QCHEM_STC_MP2_SELECTED_ENERGY_DENSITY_H
#include "stc_mp2_selected_energy.h"
#include "stc_mp2_stochastic_density.h"

namespace libgmbpt { namespace qchem { namespace stc_mp2 {
struct selected_density_result {
    density::combined_result densities;
    double energy=0, variance=0, seconds=0, energy_setup_seconds=0;
    double energy_table_seconds=0, energy_pilot_seconds=0, energy_production_seconds=0;
    double gradient_seconds=0;
    density::reverse_propagation_timing reverse_timing;
    size_t energy_production_samples=0;
    arma::mat bar_b, bar_occ, bar_virt;
};

// The replay state is small: no sampled tuples or propagated DF factors are kept.
// The selected group supplies the density used by that S-indexed CCSD(T).
struct selected_density_replay_plan {
    std::vector<size_t> group;
    arma::vec beta, weights;
    double energy_target=0;
    density::options occupied_options, virtual_options;
    arma::Mat<size_t> occupied_counts, virtual_counts;
    size_t occupied_replicas=0, virtual_replicas=0;
    int energy_sample_threads=0;
    std::vector<selected_energy_sample_plan> energy_points;
};

struct selected_density_forward_result {
    selected_density_result primal;
    selected_density_replay_plan replay;
};

struct fixed_sample_team {
    bool active;
    int previous;
    explicit fixed_sample_team(bool enabled): active(enabled),previous(omp_get_dynamic()) {
        if (active) omp_set_dynamic(0);
    }
    ~fixed_sample_team() { if (active) omp_set_dynamic(previous); }
};

inline selected_density_result estimate_selected_energy_density(stc_mp2_kernel& kernel,
        const arma::vec& beta, const arma::vec& weights, const std::vector<size_t>& group,
        double energy_target, const density::options& occupied_opt,
        const density::options& virtual_opt, bool gradients=false, std::ostream* log=nullptr,
        const arma::mat* bar_density_occupied=nullptr,
        const arma::mat* bar_density_virtual=nullptr,
        selected_density_replay_plan* record_plan=nullptr,
        const selected_density_replay_plan* replay_plan=nullptr,
        double energy_cotangent=1.0) {
    const double start=omp_get_wtime();
    if (bool(bar_density_occupied)!=bool(bar_density_virtual))
        throw std::invalid_argument("Both density cotangent seeds are required");
    if (record_plan && replay_plan)
        throw std::invalid_argument("Cannot record and replay the same MP2 pass");
    fixed_sample_team team(record_plan || replay_plan);
    if (replay_plan && (replay_plan->group!=group ||
        replay_plan->beta.n_elem!=beta.n_elem || replay_plan->weights.n_elem!=weights.n_elem ||
        !arma::approx_equal(replay_plan->beta,beta,"absdiff",0.) ||
        !arma::approx_equal(replay_plan->weights,weights,"absdiff",0.) ||
        replay_plan->energy_points.size()!=beta.n_elem ||
        replay_plan->energy_sample_threads!=effective_sample_threads()))
        throw std::invalid_argument("MP2 replay grid, selected indices, or sample thread count changed");
    if (!std::isfinite(energy_cotangent))
        throw std::invalid_argument("Invalid MP2 energy cotangent");
    const bool density_gradients=bar_density_occupied!=nullptr;
    density::options occupied_settings=occupied_opt, virtual_settings=virtual_opt;
    if (replay_plan) {
        occupied_settings=replay_plan->occupied_options;
        virtual_settings=replay_plan->virtual_options;
        occupied_settings.replay_replicas=replay_plan->occupied_replicas;
        virtual_settings.replay_replicas=replay_plan->virtual_replicas;
    }
    if (record_plan) {
        record_plan->group=group;
        record_plan->beta=beta;
        record_plan->weights=weights;
        record_plan->energy_target=energy_target;
        record_plan->energy_sample_threads=effective_sample_threads();
        record_plan->occupied_options=occupied_opt;
        record_plan->virtual_options=virtual_opt;
        record_plan->energy_points.resize(beta.n_elem);
    }
    selected_density_result out;
    kernel.select_occupied(group);
    kernel.estimate_n_sample(beta,weights,energy_target);
    kernel.retain_original_integrals(true);
    out.energy_setup_seconds=omp_get_wtime()-start;
    auto reverse_factor=[&](size_t point,density::point_data& data,
                            const density::propagator& dress) {
        const double tick=omp_get_wtime();
        // The retired C buffer is scratch for the parallel shared reverse.
        dress.backpropagate(point,data.c,data.bar_c,
            out.bar_occ,out.bar_virt,out.bar_b,&out.reverse_timing);
        out.gradient_seconds+=omp_get_wtime()-tick;
    };
    std::unique_ptr<density::density_cotangent_hooks> density_hooks;
    if (gradients || density_gradients) {
        const size_t no=kernel.occupied_fock().n_rows;
        const size_t nv=kernel.virtual_fock().n_rows;
        out.bar_occ.zeros(no,no);
        out.bar_virt.zeros(nv,nv);
        out.bar_b.zeros(kernel.original_integrals().n_rows,no*nv);
    }
    if (density_gradients) {
        density_hooks.reset(new density::density_cotangent_hooks(
            *bar_density_occupied,*bar_density_virtual,reverse_factor));
    }
    out.densities=density::estimate_both(kernel.occupied_fock(),kernel.virtual_fock(),
        kernel.original_integrals(),kernel.aux_atom_offsets(),beta,weights,group,
        occupied_settings,virtual_settings,[&](size_t l,density::point_data& data) {
            detail::shared_factor_scope factor(kernel,data.c);
            const auto point=detail::sample_selected_energy_point(kernel,l,weights(l),
                &data.norm,&data.atom_weight,gradients ? &data.bar_c : nullptr,
                gradients ? &out.bar_b : nullptr,
                record_plan ? &record_plan->energy_points[l] : nullptr,
                replay_plan ? &replay_plan->energy_points[l] : nullptr,energy_cotangent);
            out.energy+=point.energy;
            out.variance+=point.variance;
            out.energy_table_seconds+=point.table_seconds;
            out.energy_pilot_seconds+=point.pilot_seconds;
            out.energy_production_seconds+=point.production_seconds;
            out.gradient_seconds+=point.gradient_seconds;
            out.energy_production_samples+=point.production_samples;
        },log,density_hooks.get(),gradients && !density_gradients
            ? std::function<void(size_t,density::point_data&,const density::propagator&)>(reverse_factor)
            : std::function<void(size_t,density::point_data&,const density::propagator&)>(),
        replay_plan ? &replay_plan->occupied_counts : nullptr,
        replay_plan ? &replay_plan->virtual_counts : nullptr);
    if (record_plan) {
        record_plan->occupied_counts=out.densities.occupied.residual_samples;
        record_plan->virtual_counts=out.densities.virtuals.residual_samples;
        record_plan->occupied_replicas=out.densities.occupied.production_replicas;
        record_plan->virtual_replicas=out.densities.virtuals.production_replicas;
    }
    // Restore the input by ownership transfer, without another B-sized copy.
    kernel.restore_original_integrals(true);
    out.seconds=omp_get_wtime()-start;
    return out;
}

inline selected_density_forward_result selected_mp2_forward(stc_mp2_kernel& kernel,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,double energy_target,
        const density::options& occupied_opt,const density::options& virtual_opt,
        std::ostream* log=nullptr) {
    selected_density_forward_result out;
    out.primal=estimate_selected_energy_density(kernel,beta,weights,group,energy_target,
        occupied_opt,virtual_opt,false,log,nullptr,nullptr,&out.replay);
    return out;
}

inline selected_density_forward_result selected_mp2_forward(stc_mp2_kernel& kernel,
        const arma::vec& beta,const arma::vec& weights,size_t s,double energy_target,
        const density::options& occupied_opt,const density::options& virtual_opt,
        std::ostream* log=nullptr) {
    return selected_mp2_forward(kernel,beta,weights,std::vector<size_t>{s},energy_target,
        occupied_opt,virtual_opt,log);
}

inline selected_density_result selected_mp2_vjp(stc_mp2_kernel& kernel,
        const selected_density_replay_plan& replay,double bar_energy,
        const arma::mat& bar_occupied,const arma::mat& bar_virtual,
        std::ostream* log=nullptr) {
    return estimate_selected_energy_density(kernel,replay.beta,replay.weights,replay.group,
        replay.energy_target,replay.occupied_options,replay.virtual_options,
        bar_energy!=0.,log,&bar_occupied,&bar_virtual,nullptr,&replay,bar_energy);
}
}}}
#endif
