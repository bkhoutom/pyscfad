#ifndef LIBGMBPT_QCHEM_STC_MP2_SELECTED_ENERGY_H
#define LIBGMBPT_QCHEM_STC_MP2_SELECTED_ENERGY_H

#include "stc_mp2_deterministic_energy.h"

namespace libgmbpt { namespace qchem { namespace stc_mp2 {

struct selected_energy_sample_plan {
    std::vector<size_t> production_counts;
    unsigned exchange_seed=0, direct_seed=0;
};

namespace detail {

// Borrow the propagated factor without copying the three-index tensor.
class shared_factor_scope {
    stc_mp2_kernel& kernel_;
    arma::mat& factor_;
public:
    shared_factor_scope(stc_mp2_kernel& kernel, arma::mat& factor)
        : kernel_(kernel), factor_(factor) { kernel_.swap_integrals(factor_); }
    ~shared_factor_scope() { kernel_.swap_integrals(factor_); }
    shared_factor_scope(const shared_factor_scope&)=delete;
    shared_factor_scope& operator=(const shared_factor_scope&)=delete;
};

struct selected_energy_point_result {
    double energy=0, variance=0, table_seconds=0, pilot_seconds=0;
    double production_seconds=0, gradient_seconds=0;
    size_t production_samples=0;
};

// Shared by energy-only and E+D. The latter supplies its existing C proposal
// tables and gamma buffer; both paths run the identical energy sampler.
inline selected_energy_point_result sample_selected_energy_point(
        stc_mp2_kernel& kernel,size_t point,double weight,
        const arma::mat* norms=nullptr,const arma::mat* atom_weights=nullptr,
        arma::mat* gamma=nullptr,arma::mat* bar_b=nullptr,
        selected_energy_sample_plan* record=nullptr,
        const selected_energy_sample_plan* replay=nullptr,
        double energy_cotangent=1.0) {
    if (record && replay) throw std::invalid_argument("Cannot record and replay one energy point");
    if (bool(gamma)!=bool(bar_b))
        throw std::invalid_argument("Selected energy gradient requires gamma and B cotangent");
    selected_energy_point_result out;
    double tick=omp_get_wtime();
    auto context=kernel.prepare_sampling(point,norms,atom_weights);
    if (replay) {
        if (replay->production_counts.size()!=4)
            throw std::invalid_argument("Invalid energy replay allocation");
        context.exchange_seed=replay->exchange_seed;
        context.direct_seed=replay->direct_seed;
    }
    out.table_seconds=omp_get_wtime()-tick;
    double direct=0,exchange=0;
    using phase=stc_mp2_kernel::sampling_phase;
    tick=omp_get_wtime();
    kernel.sample_mp2_exchange(exchange,context,replay ? phase::replay_setup : phase::pilot);
    kernel.sample_mp2_direct(direct,context,replay ? phase::replay_setup : phase::pilot);
    if (replay) context.production_counts=replay->production_counts;
    else kernel.allocate_samples(context);
    if (record) {
        record->production_counts=context.production_counts;
        record->exchange_seed=context.exchange_seed;
        record->direct_seed=context.direct_seed;
    }
    out.pilot_seconds=omp_get_wtime()-tick;
    for (size_t n:context.production_counts) out.production_samples+=n;
    if (gamma) kernel.begin_gradient_sampling();
    tick=omp_get_wtime();
    kernel.sample_mp2_direct(direct,context,phase::production);
    kernel.sample_mp2_exchange(exchange,context,phase::production);
    out.production_seconds=omp_get_wtime()-tick;
    out.energy=weight*(exchange-2*direct);
    out.variance=weight*weight*(context.exchange_variance+4*context.direct_variance);
    if (gamma) {
        tick=omp_get_wtime();
        kernel.finish_gradient_sampling(context,*gamma,bar_b,true,energy_cotangent);
        out.gradient_seconds=omp_get_wtime()-tick;
    }
    return out;
}

} // namespace detail

struct selected_stochastic_energy_result {
    double energy=0, variance=0, seconds=0, setup_seconds=0;
    double table_seconds=0, pilot_seconds=0, production_seconds=0;
    double gradient_seconds=0;
    size_t production_samples=0;
    arma::mat bar_occ, bar_virt, bar_b;
};

namespace detail {

inline selected_stochastic_energy_result evaluate_selected_stochastic_energy(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,double energy_target,int threads,bool gradients) {
    if (threads<1 || beta.is_empty() || beta.n_elem!=weights.n_elem)
        throw std::invalid_argument("Invalid selected STC energy input");
    const double start=omp_get_wtime();
    kernel.select_occupied(group);
    kernel.estimate_n_sample(beta,weights,energy_target);
    kernel.retain_original_integrals(true);
    selected_stochastic_energy_result out;
    out.setup_seconds=omp_get_wtime()-start;
    const arma::mat& bare=kernel.original_integrals();
    density::propagator dress(kernel.occupied_fock(),kernel.virtual_fock(),bare,beta,threads);
    if (gradients) {
        out.bar_occ.zeros(kernel.occupied_fock().n_rows,kernel.occupied_fock().n_cols);
        out.bar_virt.zeros(kernel.virtual_fock().n_rows,kernel.virtual_fock().n_cols);
        out.bar_b.zeros(bare.n_rows,bare.n_cols);
    }
    arma::mat c;
    for (size_t l=0;l<beta.n_elem;++l) {
        dress.apply(l,c);
        arma::mat gamma;
        if (gradients) gamma.zeros(bare.n_rows,bare.n_cols);
        selected_energy_point_result point;
        {
            shared_factor_scope factor(kernel,c);
            point=sample_selected_energy_point(kernel,l,weights(l),nullptr,nullptr,
                gradients ? &gamma : nullptr,gradients ? &out.bar_b : nullptr);
        }
        out.energy+=point.energy;
        out.variance+=point.variance;
        out.table_seconds+=point.table_seconds;
        out.pilot_seconds+=point.pilot_seconds;
        out.production_seconds+=point.production_seconds;
        out.gradient_seconds+=point.gradient_seconds;
        out.production_samples+=point.production_samples;
        if (gradients) {
            const double tick=omp_get_wtime();
            dress.backpropagate(l,c,gamma,out.bar_occ,out.bar_virt,out.bar_b);
            out.gradient_seconds+=omp_get_wtime()-tick;
        }
    }
    kernel.restore_original_integrals(true);
    out.seconds=omp_get_wtime()-start;
    return out;
}

} // namespace detail

inline selected_stochastic_energy_result selected_stochastic_energy(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,double energy_target,int threads) {
    return detail::evaluate_selected_stochastic_energy(kernel,beta,weights,group,energy_target,threads,false);
}

// Fused reverse: the energy samples are differentiated while they are drawn.
inline selected_stochastic_energy_result selected_stochastic_energy_vjp(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,double energy_target,int threads) {
    return detail::evaluate_selected_stochastic_energy(kernel,beta,weights,group,energy_target,threads,true);
}

inline deterministic_energy_result selected_deterministic_energy(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads) {
    return evaluate_deterministic_energy(kernel,beta,weights,group,threads,false);
}

inline deterministic_energy_result selected_deterministic_energy_vjp(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads) {
    return evaluate_deterministic_energy(kernel,beta,weights,group,threads,true);
}

// Keep the single-index entry points as convenience overloads.
inline selected_stochastic_energy_result selected_stochastic_energy(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        size_t s,double energy_target,int threads) {
    return selected_stochastic_energy(kernel,beta,weights,std::vector<size_t>{s},energy_target,threads);
}
inline selected_stochastic_energy_result selected_stochastic_energy_vjp(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        size_t s,double energy_target,int threads) {
    return selected_stochastic_energy_vjp(kernel,beta,weights,std::vector<size_t>{s},energy_target,threads);
}
inline deterministic_energy_result selected_deterministic_energy(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        size_t s,int threads) {
    return selected_deterministic_energy(kernel,beta,weights,std::vector<size_t>{s},threads);
}
inline deterministic_energy_result selected_deterministic_energy_vjp(
        stc_mp2_kernel& kernel,const arma::vec& beta,const arma::vec& weights,
        size_t s,int threads) {
    return selected_deterministic_energy_vjp(kernel,beta,weights,std::vector<size_t>{s},threads);
}

}}}
#endif
