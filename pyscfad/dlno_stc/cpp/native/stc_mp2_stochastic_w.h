#ifndef LIBGMBPT_QCHEM_STC_MP2_STOCHASTIC_W_H
#define LIBGMBPT_QCHEM_STC_MP2_STOCHASTIC_W_H

#include "stc_mp2_stochastic_density.h"

namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {

struct stochastic_w_seed {
    double energy=0.;
    arma::mat occupied, virtuals;
    stochastic_w_seed()=default;
    stochastic_w_seed(double e,const arma::mat& o,const arma::mat& v):
        energy(e),occupied(o),virtuals(v) {}
};

struct stochastic_w_block_timing {
    double proposal_seconds=0.,pilot_seconds=0.,exact_core_seconds=0.;
    double production_seconds=0.,production_accumulation_seconds=0.;
    double production_reduction_seconds=0.,exact_vjp_seconds=0.;
    double update_reduction_seconds=0.;
    double direct_production_seconds=0.,exchange_production_seconds=0.;
    size_t pilot_draws=0,production_draws=0;
    size_t direct_production_draws=0,exchange_production_draws=0;
    strong_vjp_profile core_vjp_profile;
};

struct stochastic_w_result {
    // Empty on a replay VJP, which computes only input cotangents.
    arma::mat occupied, virtuals, occupied_se, virtuals_se;
    arma::mat bar_fo, bar_fv, bar_b;
    double energy=0., energy_se=0., seconds=0., table_seconds=0.;
    double propagation_seconds=0., contraction_seconds=0., reverse_seconds=0.;
    double occupied_frobenius_se=0., virtual_frobenius_se=0.;
    bool occupied_target_met=false, virtual_target_met=false;
    double occupied_requested_samples=0., virtual_requested_samples=0.;
    size_t pilot_draws=0, production_draws=0;
    stochastic_w_block_timing occupied_timing,virtual_timing;
    reverse_propagation_timing reverse_timing;
};

struct stochastic_w_replay_plan {
    std::vector<size_t> group;
    arma::vec beta,weights;
    options occupied_options,virtual_options;
    arma::Mat<size_t> occupied_counts,virtual_counts;
    size_t occupied_replicas=0,virtual_replicas=0;
};

struct stochastic_w_forward_result {
    stochastic_w_result primal;
    stochastic_w_replay_plan replay;
};

// Density-style integral sampling for W.  Treat B as the fixed first factor
// and each C_l as the second factor of the existing density pair kernels.
// This keeps their orbital/atom proposals, strong cores, local pilot
// allocation, and sample VJPs. The outer schedule visits (B,C_l), with -w_l.
// Fock-trace point quotas permit pilot, production, and reverse propagation
// while each C_l is resident, as in the density-plus-cotangent path.
inline stochastic_w_result evaluate_stochastic_w(const arma::mat& fo,
        const arma::mat& fv,const arma::mat& bare,
        const arma::Col<size_t>& offsets,const arma::vec& beta,
        const arma::vec& weights,const std::vector<size_t>& group,
        const options& occupied_opt,const options& virtual_opt,
        const stochastic_w_seed* seed=nullptr,std::ostream* log=nullptr,
        stochastic_w_replay_plan* record_plan=nullptr,
        const stochastic_w_replay_plan* replay_plan=nullptr) {
    const double start=omp_get_wtime();
    if (record_plan && replay_plan)
        throw std::invalid_argument("Cannot record and replay W in one pass");
    if (replay_plan && (replay_plan->group!=group ||
        replay_plan->beta.n_elem!=beta.n_elem || replay_plan->weights.n_elem!=weights.n_elem ||
        !arma::approx_equal(replay_plan->beta,beta,"absdiff",0.) ||
        !arma::approx_equal(replay_plan->weights,weights,"absdiff",0.)))
        throw std::invalid_argument("W replay grid or selected indices changed");
    options occupied_settings=replay_plan ? replay_plan->occupied_options : occupied_opt;
    options virtual_settings=replay_plan ? replay_plan->virtual_options : virtual_opt;
    if (replay_plan) {
        occupied_settings.replay_replicas=replay_plan->occupied_replicas;
        virtual_settings.replay_replicas=replay_plan->virtual_replicas;
    }
    const options& o=occupied_settings;
    const options& v=virtual_settings;
    const bool occupied_mixture=o.w_occupied_stratum_mixture &&
        o.w_direct_joint_x_strong_h;
    const bool virtual_split=v.w_virtual_reform && v.w_virtual_split;
    for(int variant:v.w_virtual_exchange_variants)
        if(variant<0 || variant>5)
            throw std::invalid_argument("Invalid W virtual exchange envelope");
    for(int variant:v.w_virtual_direct_variants)
        if(variant<-1 || variant>5)
            throw std::invalid_argument("Invalid W virtual direct envelope");
    const bool virtual_c_atom=virtual_split &&
        (std::find(v.w_virtual_exchange_variants.begin(),
                   v.w_virtual_exchange_variants.end(),5)!=v.w_virtual_exchange_variants.end() ||
         std::any_of(v.w_virtual_direct_variants.begin(),v.w_virtual_direct_variants.end(),
                     [](int variant){ return variant>=4; }));
    validate(fo,fv,bare,offsets,beta,weights,group,o);
    validate(fo,fv,bare,offsets,beta,weights,group,v);
    if (o.threads!=v.threads)
        throw std::invalid_argument("W integral sampling requires independent pair allocation");
    if (o.atom_block_l2!=v.atom_block_l2)
        throw std::invalid_argument("W occupied and virtual atom proposals must match");
    if (record_plan) {
        record_plan->group=group;
        record_plan->beta=beta;
        record_plan->weights=weights;
        record_plan->occupied_options=o;
        record_plan->virtual_options=v;
    }
    const size_t no=fo.n_rows,nv=fv.n_rows,np=bare.n_rows,nl=beta.n_elem;
    if (seed && (!std::isfinite(seed->energy) ||
            seed->occupied.n_rows!=no || seed->occupied.n_cols!=no ||
            seed->virtuals.n_rows!=nv || seed->virtuals.n_cols!=nv ||
            !seed->occupied.is_finite() || !seed->virtuals.is_finite()))
        throw std::invalid_argument("Invalid W output cotangent");
    const bool reverse=seed!=nullptr;
    // Pilot replay adds more update events to a W pass. Smaller W-only chunks
    // keep the two factor-adjoint queues below the density-mode high-water mark.
    constexpr size_t w_update_chunk_size=160000;
    // Replay uses the saved allocations and RNG keys. It needs proposal draws
    // and factor adjoints, but no second W matrix or matrix SE reduction.
    const bool reverse_only=reverse && replay_plan;
    stochastic_w_result out;
    arma::mat wo(no,no,arma::fill::zeros),wv(nv,nv,arma::fill::zeros);
    if (reverse) {
        // E is the average of the two W traces, so its adjoint is split.
        wo=.5*(seed->occupied+seed->occupied.t());
        wo.diag()+=.5*seed->energy;
        wv=.5*(seed->virtuals+seed->virtuals.t());
        wv.diag()+=.5*seed->energy;
        out.bar_b.zeros(np,no*nv);
        out.bar_fo.zeros(no,no); out.bar_fv.zeros(nv,nv);
    }
    // A zero output cotangent has no VJP contribution. In replay mode skip
    // that block's exact core, proposal tables, and regenerated samples.
    const bool occupied_active=!reverse_only || arma::accu(arma::abs(wo))!=0.;
    const bool virtual_active=!reverse_only || arma::accu(arma::abs(wv))!=0.;
    const arma::mat zero_occupied(no,no,arma::fill::zeros);
    const arma::mat zero_virtual(nv,nv,arma::fill::zeros);
    pair_estimator<occupied_policy> occupied(no,nv,nl,offsets,group,o,
        replay_plan ? &replay_plan->occupied_counts : nullptr,!reverse_only,
        o.w_direct_joint_x_strong_h);
    pair_estimator<virtual_policy> virtuals(no,nv,nl,offsets,group,v,
        replay_plan ? &replay_plan->virtual_counts : nullptr,!reverse_only,
        false);
    if (!occupied_active && !virtual_active) {
        out.seconds=omp_get_wtime()-start;
        return out;
    }
    propagator dress(fo,fv,bare,beta,o.threads);
    arma::mat point_quotas(nl,nl,arma::fill::zeros);
    point_quotas.diag()=fixed_point_weights_from_spectrum(
        dress.occupied_spectrum(),dress.virtual_spectrum(),beta,weights);
    occupied.set_quotas(point_quotas);
    virtuals.set_quotas(point_quotas);
    point_data fixed(bare),dressed;
    double tick=omp_get_wtime();
    fixed.prepare(no,nv,offsets,1.,o.domain_tau,o.threads,o.atom_block_l2,
                  o.w_direct_atom_pair || o.w_exchange_atom_pair ||
                  occupied_mixture || v.w_virtual_reform);
    out.table_seconds=omp_get_wtime()-tick; // Fixed B proposal table, built once.
    std::unique_ptr<w_occupied_b_tables> b_cache;
    if(occupied_mixture && occupied_active) {
        tick=omp_get_wtime();
        b_cache.reset(new w_occupied_b_tables(fixed,o.threads));
        out.table_seconds+=omp_get_wtime()-tick;
    }
    std::unique_ptr<w_virtual_b_tables> vb_cache;
    if(virtual_split && virtual_active) {
        fixed.set_domain(1.,v.domain_tau);
        tick=omp_get_wtime();
        vb_cache.reset(new w_virtual_b_tables(fixed,v.threads));
        out.table_seconds+=omp_get_wtime()-tick;
    }
    for (size_t l=0;l<nl;++l) {
        tick=omp_get_wtime();
        dress.apply(l,dressed.c);
        out.propagation_seconds+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        dressed.prepare(no,nv,offsets,weights(l),o.domain_tau,o.threads,o.atom_block_l2,
                        (occupied_mixture && occupied_active) ||
                        (virtual_c_atom && virtual_active));
        out.table_seconds+=omp_get_wtime()-tick; // Per-point C proposal table.
        std::unique_ptr<w_occupied_c_tables> c_cache;
        if(occupied_mixture && occupied_active) {
            tick=omp_get_wtime();
            c_cache.reset(new w_occupied_c_tables(dressed,o.threads));
            out.table_seconds+=omp_get_wtime()-tick;
        }
        std::unique_ptr<w_virtual_c_tables> vc_cache;
        if(virtual_split && virtual_active) {
            dressed.set_domain(weights(l),v.domain_tau);
            tick=omp_get_wtime();
            vc_cache.reset(new w_virtual_c_tables(dressed,v.threads,virtual_c_atom));
            out.table_seconds+=omp_get_wtime()-tick;
        }
        if (reverse) dressed.bar_c.zeros(np,no*nv);
        const double factor=-weights(l);
        const size_t table_bytes=fixed.table_bytes()+dressed.table_bytes();
        tick=omp_get_wtime();
        if (occupied_active) {
            fixed.set_domain(1.,o.domain_tau);
            dressed.set_domain(weights(l),o.domain_tau);
            typename pair_estimator<occupied_policy>::vjp_callback hook;
            std::unique_ptr<density_block_locks> b_locks,c_locks;
            std::unique_ptr<selected_density_updates> b_updates,c_updates;
            if (reverse) {
                b_locks.reset(new density_block_locks(no,offsets.n_elem-1,nv));
                c_locks.reset(new density_block_locks(no,offsets.n_elem-1,nv));
                b_updates.reset(new selected_density_updates(out.bar_b,*b_locks,
                    offsets,nv,o.threads,o.w_parallel_update_reduction,
                    o.w_column_major_updates,w_update_chunk_size));
                c_updates.reset(new selected_density_updates(dressed.bar_c,*c_locks,
                    offsets,nv,o.threads,o.w_parallel_update_reduction,
                    o.w_column_major_updates,w_update_chunk_size));
                hook.core=[&]() {
                    const double core_start=omp_get_wtime();
                    strong_pair_vjp(fixed,dressed,group,wo,zero_virtual,
                        factor,out.bar_b,dressed.bar_c,o.threads,
                        &out.occupied_timing.core_vjp_profile);
                    out.occupied_timing.exact_vjp_seconds+=omp_get_wtime()-core_start;
                };
                hook.production=[&](const proposal& table,const arma::Col<size_t>& off,
                        size_t count,size_t stream,std::mt19937& rng,arma::mat& value,
                        double pair_scale) {
                    if (reverse_only)
                        accumulate_occupied_pair_vjp<false>(table,fixed,dressed,
                            out.bar_b,dressed.bar_c,*b_locks,*c_locks,wo,off,count,
                            stream,rng,value,pair_scale,
                            b_updates.get(),c_updates.get());
                    else
                        accumulate_occupied_pair_vjp<true>(table,fixed,dressed,
                            out.bar_b,dressed.bar_c,*b_locks,*c_locks,wo,off,count,
                            stream,rng,value,pair_scale,
                            b_updates.get(),c_updates.get());
                };
            }
            occupied.process(l,l,factor,fixed,dressed,
                table_bytes+(b_cache ? b_cache->storage_bytes() : 0)
                    +(c_cache ? c_cache->storage_bytes() : 0),
                log,"W occupied ",hook,b_cache.get(),c_cache.get());
            const double reduce_start=omp_get_wtime();
            if (b_updates) b_updates->reduce(o.threads);
            if (c_updates) c_updates->reduce(o.threads);
            out.occupied_timing.update_reduction_seconds+=omp_get_wtime()-reduce_start;
        }
        if (virtual_active) {
            fixed.set_domain(1.,v.domain_tau);
            dressed.set_domain(weights(l),v.domain_tau);
            typename pair_estimator<virtual_policy>::vjp_callback hook;
            std::unique_ptr<density_block_locks> b_locks,c_locks;
            std::unique_ptr<selected_density_updates> b_updates,c_updates;
            if (reverse) {
                b_locks.reset(new density_block_locks(no,offsets.n_elem-1,nv));
                c_locks.reset(new density_block_locks(no,offsets.n_elem-1,nv));
                b_updates.reset(new selected_density_updates(out.bar_b,*b_locks,
                    offsets,nv,v.threads,v.w_parallel_update_reduction,
                    v.w_column_major_updates,w_update_chunk_size));
                c_updates.reset(new selected_density_updates(dressed.bar_c,*c_locks,
                    offsets,nv,v.threads,v.w_parallel_update_reduction,
                    v.w_column_major_updates,w_update_chunk_size));
                hook.core=[&]() {
                    const double core_start=omp_get_wtime();
                    strong_pair_vjp(fixed,dressed,group,zero_occupied,wv,
                        factor,out.bar_b,dressed.bar_c,v.threads,
                        &out.virtual_timing.core_vjp_profile);
                    out.virtual_timing.exact_vjp_seconds+=omp_get_wtime()-core_start;
                };
                hook.production=[&](const virtual_proposal& table,const arma::Col<size_t>& off,
                        size_t count,size_t,std::mt19937& rng,arma::mat& value,
                        double pair_scale) {
                    if (reverse_only)
                        accumulate_virtual_pair_vjp<false>(table,fixed,dressed,
                            out.bar_b,dressed.bar_c,*b_locks,*c_locks,wv,off,count,
                            rng,value,pair_scale,
                            b_updates.get(),c_updates.get());
                    else
                        accumulate_virtual_pair_vjp<true>(table,fixed,dressed,
                            out.bar_b,dressed.bar_c,*b_locks,*c_locks,wv,off,count,
                            rng,value,pair_scale,
                            b_updates.get(),c_updates.get());
                };
            }
            virtuals.process(l,l,factor,fixed,dressed,
                table_bytes+(vb_cache ? vb_cache->storage_bytes() : 0)
                    +(vc_cache ? vc_cache->storage_bytes() : 0),
                log,"W virtual ",hook,nullptr,nullptr,vb_cache.get(),vc_cache.get());
            const double reduce_start=omp_get_wtime();
            if (b_updates) b_updates->reduce(v.threads);
            if (c_updates) c_updates->reduce(v.threads);
            out.virtual_timing.update_reduction_seconds+=omp_get_wtime()-reduce_start;
        }
        out.contraction_seconds+=omp_get_wtime()-tick;
        if (reverse) {
            tick=omp_get_wtime();
            // Sampling is finished with C_l. Reuse its storage to regenerate
            // the occupied-only factor needed by the reverse pass.
            dress.backpropagate(l,dressed.c,dressed.bar_c,out.bar_fo,out.bar_fv,
                out.bar_b,&out.reverse_timing);
            out.reverse_seconds+=omp_get_wtime()-tick;
        }
    }
    occupied.finish(); virtuals.finish();
    out.occupied_timing.proposal_seconds=occupied.answer.pair_table_seconds;
    out.occupied_timing.pilot_draws=occupied.answer.pilot_draws;
    out.occupied_timing.production_draws=occupied.answer.production_draws;
    out.occupied_timing.pilot_seconds=occupied.answer.pilot_seconds;
    out.occupied_timing.exact_core_seconds=occupied.answer.dense_seconds;
    out.occupied_timing.production_seconds=occupied.answer.production_seconds;
    out.occupied_timing.production_accumulation_seconds=
        occupied.answer.production_accumulation_seconds;
    out.occupied_timing.production_reduction_seconds=
        occupied.answer.production_reduction_seconds;
    out.occupied_timing.direct_production_seconds=occupied.answer.direct_production_seconds;
    out.occupied_timing.exchange_production_seconds=occupied.answer.exchange_production_seconds;
    out.occupied_timing.direct_production_draws=occupied.answer.direct_production_draws;
    out.occupied_timing.exchange_production_draws=occupied.answer.exchange_production_draws;
    out.virtual_timing.proposal_seconds=virtuals.answer.pair_table_seconds;
    out.virtual_timing.pilot_draws=virtuals.answer.pilot_draws;
    out.virtual_timing.production_draws=virtuals.answer.production_draws;
    out.virtual_timing.pilot_seconds=virtuals.answer.pilot_seconds;
    out.virtual_timing.exact_core_seconds=virtuals.answer.dense_seconds;
    out.virtual_timing.production_seconds=virtuals.answer.production_seconds;
    out.virtual_timing.production_accumulation_seconds=
        virtuals.answer.production_accumulation_seconds;
    out.virtual_timing.production_reduction_seconds=
        virtuals.answer.production_reduction_seconds;
    out.virtual_timing.direct_production_seconds=virtuals.answer.direct_production_seconds;
    out.virtual_timing.exchange_production_seconds=virtuals.answer.exchange_production_seconds;
    out.virtual_timing.direct_production_draws=virtuals.answer.direct_production_draws;
    out.virtual_timing.exchange_production_draws=virtuals.answer.exchange_production_draws;
    if (record_plan) {
        record_plan->occupied_counts=occupied.answer.residual_samples;
        record_plan->virtual_counts=virtuals.answer.residual_samples;
        record_plan->occupied_replicas=occupied.answer.production_replicas;
        record_plan->virtual_replicas=virtuals.answer.production_replicas;
    }
    out.occupied=std::move(occupied.answer.matrix);
    out.virtuals=std::move(virtuals.answer.matrix);
    out.occupied_se=std::move(occupied.answer.standard_error);
    out.virtuals_se=std::move(virtuals.answer.standard_error);
    out.occupied_frobenius_se=occupied.answer.frobenius_se;
    out.virtual_frobenius_se=virtuals.answer.frobenius_se;
    out.occupied_target_met=occupied.answer.target_met;
    out.virtual_target_met=virtuals.answer.target_met;
    out.occupied_requested_samples=occupied.answer.requested_samples;
    out.virtual_requested_samples=virtuals.answer.requested_samples;
    out.pilot_draws=occupied.answer.pilot_draws+virtuals.answer.pilot_draws;
    out.production_draws=occupied.answer.production_draws+virtuals.answer.production_draws;
    if (!reverse_only) {
        out.energy=.5*(arma::trace(out.occupied)+arma::trace(out.virtuals));
        // The two W traces use the same replica RNG keys. Retain their
        // measured covariance within each Laplace point; different points
        // have independent keys and their variances add.
        double covariance=0.;
        const auto& ot=occupied.answer.trace_replicas;
        const auto& vt=virtuals.answer.trace_replicas;
        const size_t bo=ot.n_cols,bv=vt.n_cols,paired=std::min(bo,bv);
        if(paired>1) for(size_t l=0;l<nl;++l) {
            const size_t key=l*nl+l;
            double mo=0.,mv=0.;
            for(size_t replica=0;replica<paired;++replica) {
                mo+=ot(key,replica);mv+=vt(key,replica);
            }
            mo/=double(paired);mv/=double(paired);
            double cross=0.;
            for(size_t replica=0;replica<paired;++replica)
                cross+=(ot(key,replica)-mo)*(vt(key,replica)-mv);
            covariance+=cross/double(paired-1)*double(paired)/double(bo*bv);
        }
        out.energy_se=std::sqrt(std::max(0.,.25*(occupied.answer.trace_variance+
            virtuals.answer.trace_variance+2.*covariance)));
    }
    out.seconds=omp_get_wtime()-start;
    return out;
}

inline stochastic_w_forward_result selected_stochastic_w_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::Col<size_t>& offsets,const arma::vec& beta,
        const arma::vec& weights,const std::vector<size_t>& group,
        const options& occupied_opt,
        const options& virtual_opt,std::ostream* log=nullptr) {
    stochastic_w_forward_result out;
    out.primal=evaluate_stochastic_w(fo,fv,bare,offsets,beta,weights,group,
        occupied_opt,virtual_opt,nullptr,log,&out.replay);
    return out;
}

inline stochastic_w_forward_result selected_stochastic_w_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::Col<size_t>& offsets,const arma::vec& beta,
        const arma::vec& weights,size_t s,const options& occupied_opt,
        const options& virtual_opt,std::ostream* log=nullptr) {
    return selected_stochastic_w_forward(fo,fv,bare,offsets,beta,weights,
        std::vector<size_t>{s},occupied_opt,virtual_opt,log);
}

inline stochastic_w_result selected_stochastic_w_vjp(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::Col<size_t>& offsets,const stochastic_w_replay_plan& replay,
        double bar_energy,const arma::mat& bar_occupied,
        const arma::mat& bar_virtuals,std::ostream* log=nullptr) {
    const stochastic_w_seed seed{bar_energy,bar_occupied,bar_virtuals};
    return evaluate_stochastic_w(fo,fv,bare,offsets,replay.beta,replay.weights,
        replay.group,replay.occupied_options,replay.virtual_options,&seed,log,nullptr,&replay);
}

}}}}
#endif
