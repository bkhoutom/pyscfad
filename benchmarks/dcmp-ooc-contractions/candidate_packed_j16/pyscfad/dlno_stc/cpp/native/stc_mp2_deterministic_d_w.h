#ifndef LIBGMBPT_QCHEM_STC_MP2_DETERMINISTIC_D_W_H
#define LIBGMBPT_QCHEM_STC_MP2_DETERMINISTIC_D_W_H
#include "stc_mp2_stochastic_density.h"

namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {

struct deterministic_result {
    arma::mat occupied, virtuals;
    // Symmetric selected energy matrices: each trace equals the selected energy.
    arma::mat energy_occupied, energy_virtuals;
    arma::mat bar_fo, bar_fv, bar_b;
    double energy=0, seconds=0, propagation_seconds=0, amplitude_seconds=0;
    double propagation_allocation_seconds=0, propagation_occupied_seconds=0;
    double propagation_virtual_seconds=0;
    double contraction_seconds=0, reverse_seconds=0;
    double seed_seconds=0;
    double reverse_factor_seconds=0, amplitude_vjp_seconds=0, propagator_vjp_seconds=0;
    double spin_seconds=0, occupied_density_seconds=0, virtual_energy_seconds=0;
    size_t reverse_factor_rebuilds=0;
};

struct deterministic_cotangent_seed {
    // Defines L = energy*E + <occupied,D_oo> + <virtuals,D_vv>.
    // Only the symmetric parts of the matrix seeds contribute.
    double energy=0;
    arma::mat occupied, virtuals;
    deterministic_cotangent_seed(double e, const arma::mat& o, const arma::mat& v):
        energy(e), occupied(o), virtuals(v) {}
};

struct deterministic_w_cotangent_seed {
    double energy=0;
    arma::mat occupied, virtuals;
    deterministic_w_cotangent_seed(double e,const arma::mat& o,const arma::mat& v):
        energy(e),occupied(o),virtuals(v) {}
};

// Exact contractions for the supplied Laplace grid. All selected amplitudes are
// retained; each dressed DF factor is shared by the entire selected group.
inline deterministic_result evaluate_deterministic(const arma::mat& fo, const arma::mat& fv,
        const arma::mat& bare, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, int threads,
        const deterministic_cotangent_seed* seed=nullptr,
        bool energy_matrices=false, bool primal_outputs=true,
        const deterministic_w_cotangent_seed* w_seed=nullptr) {
    const size_t no=fo.n_rows, nv=fv.n_rows, np=bare.n_rows;
    if (!no || !nv || !np || fo.n_cols!=no || fv.n_cols!=nv ||
        bare.n_cols!=no*nv || beta.is_empty() || beta.n_elem!=weights.n_elem ||
        group.empty() || threads<1 || !fo.is_finite() || !fv.is_finite() ||
        !bare.is_finite() || !beta.is_finite() || !weights.is_finite())
        throw std::invalid_argument("Invalid deterministic selected density input");
    std::vector<unsigned char> seen(no,0);
    for (size_t i:group) {
        if (i>=no || seen[i]++) throw std::invalid_argument("Invalid selected occupied index");
    }
    if (seed && (!std::isfinite(seed->energy) ||
        seed->occupied.n_rows!=no || seed->occupied.n_cols!=no ||
        seed->virtuals.n_rows!=nv || seed->virtuals.n_cols!=nv ||
        !seed->occupied.is_finite() || !seed->virtuals.is_finite()))
        throw std::invalid_argument("Invalid deterministic output cotangent");
    if (w_seed && (!std::isfinite(w_seed->energy) ||
        w_seed->occupied.n_rows!=no || w_seed->occupied.n_cols!=no ||
        w_seed->virtuals.n_rows!=nv || w_seed->virtuals.n_cols!=nv ||
        !w_seed->occupied.is_finite() || !w_seed->virtuals.is_finite()))
        throw std::invalid_argument("Invalid deterministic W cotangent");
    if (seed && w_seed)
        throw std::invalid_argument("Select one deterministic output cotangent type");
    if (!primal_outputs && !energy_matrices && !seed && !w_seed)
        throw std::invalid_argument("Deterministic MP2 pass requires output or cotangent");
    const double start=omp_get_wtime();
    propagator dress(fo,fv,bare,beta,threads);
    std::vector<arma::cube> amplitudes(group.size());
    for (auto& t:amplitudes) t.zeros(nv,nv,no);
    deterministic_result out;
    arma::mat c;
    for (size_t l=0;l<beta.n_elem;++l) {
        double tick=omp_get_wtime();
        propagation_timing propagation;
        dress.apply(l,c,&propagation);
        out.propagation_seconds+=omp_get_wtime()-tick;
        out.propagation_allocation_seconds+=propagation.allocation_seconds;
        out.propagation_occupied_seconds+=propagation.occupied_seconds;
        out.propagation_virtual_seconds+=propagation.virtual_seconds;
        tick=omp_get_wtime();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t task=0;task<group.size()*no;++task) {
            const size_t s=task/no,j=task%no;
            arma::mat ci(c.colptr(group[s]*nv),np,nv,false,true);
            arma::mat cj(c.colptr(j*nv),np,nv,false,true);
            amplitudes[s].slice(j)-=weights(l)*(ci.t()*cj);
        }
        out.amplitude_seconds+=omp_get_wtime()-tick;
    }
    arma::mat wo,wv,w_wo,w_wv;
    bool occupied_active=false,virtual_active=false,energy_active=false;
    bool w_occupied_active=false,w_virtual_active=false;
    std::vector<arma::cube> bar_amplitudes;
    if (seed || w_seed) {
        const double seed_start=omp_get_wtime();
        if (seed) {
            wo=0.5*(seed->occupied+seed->occupied.t());
            wv=0.5*(seed->virtuals+seed->virtuals.t());
            occupied_active=arma::accu(arma::abs(wo))!=0;
            virtual_active=arma::accu(arma::abs(wv))!=0;
            energy_active=seed->energy!=0;
        }
        if (w_seed) {
            w_wo=0.5*(w_seed->occupied+w_seed->occupied.t());
            w_wo.diag()+=w_seed->energy;
            w_wv=0.5*(w_seed->virtuals+w_seed->virtuals.t());
            w_occupied_active=arma::accu(arma::abs(w_wo))!=0.;
            w_virtual_active=arma::accu(arma::abs(w_wv))!=0.;
        }
        out.bar_fo.zeros(no,no);
        out.bar_fv.zeros(nv,nv);
        out.bar_b.zeros(np,no*nv);
        bar_amplitudes.resize(group.size());
        for (size_t s=0;s<group.size();++s) {
            const arma::cube& t=amplitudes[s];
            arma::cube& bt=bar_amplitudes[s];
            bt.zeros(nv,nv,no);
            if (!occupied_active) continue;
            const arma::mat flat(const_cast<double*>(t.memptr()),nv*nv,no,false,true);
            arma::mat bar_flat(bt.memptr(),nv*nv,no,false,true);
            const arma::mat wo_t=wo.t();
            // Each row panel writes disjoint amplitudes. A single-threaded
            // multiply here dominates the setup on large virtual spaces.
            const size_t row_panel=8192;
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t first=0;first<flat.n_rows;first+=row_panel) {
                const size_t last=std::min(size_t(flat.n_rows),first+row_panel)-1;
                bar_flat.rows(first,last)=flat.rows(first,last)*wo_t;
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t j=0;j<no;++j) {
                const arma::mat weighted=bt.slice(j);
                bt.slice(j)=2*(2*weighted-weighted.t());
            }
        }
        out.seed_seconds=omp_get_wtime()-seed_start;
    }
    const double contraction_start=omp_get_wtime();
    if (primal_outputs) {
        out.occupied.zeros(no,no);
        out.virtuals.zeros(nv,nv);
    }
    if (energy_matrices) {
        out.energy_occupied.zeros(no,no);
        out.energy_virtuals.zeros(nv,nv);
    }
    for (size_t s=0;s<group.size();++s) {
        const arma::cube& t=amplitudes[s];
        arma::mat flat(const_cast<double*>(t.memptr()),nv*nv,no,false,true);
        double tick=omp_get_wtime();
        arma::cube spins(nv,nv,no);
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t k=0;k<no;++k) {
            arma::mat transpose=t.slice(k);
            arma::inplace_trans(transpose);
            spins.slice(k)=2*t.slice(k)-transpose;
        }
        out.spin_seconds+=omp_get_wtime()-tick;
        if (primal_outputs) {
            const arma::mat spin_flat(spins.memptr(),nv*nv,no,false,true);
            tick=omp_get_wtime();
            const size_t panel=8;
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t first=0;first<no;first+=panel) {
                const size_t last=std::min(no,first+panel)-1;
                out.occupied.cols(first,last)+=flat.t()*spin_flat.cols(first,last);
            }
            out.occupied_density_seconds+=omp_get_wtime()-tick;
        }
        tick=omp_get_wtime();
        arma::mat bi(const_cast<double*>(bare.colptr(group[s]*nv)),np,nv,false,true);
        arma::cube bare_grams;
        if (w_seed) bare_grams.set_size(nv,nv,no);
        #pragma omp parallel num_threads(threads)
        {
            arma::mat local;
            if (primal_outputs) local.zeros(nv,nv);
            arma::mat local_energy_virtuals;
            if (energy_matrices) local_energy_virtuals.zeros(nv,nv);
            arma::mat local_bi;
            if (energy_active) local_bi.zeros(np,nv);
            double local_energy=0;
            #pragma omp for schedule(static)
            for (size_t j=0;j<no;++j) {
                const arma::mat& tj=t.slice(j);
                const arma::mat& spin=spins.slice(j);
                if (primal_outputs)
                    local+=0.5*(tj*spin.t()+tj.t()*spin);
                if (primal_outputs || energy_matrices || energy_active || w_seed) {
                    arma::mat bj(const_cast<double*>(bare.colptr(j*nv)),np,nv,false,true);
                    const arma::mat g=bi.t()*bj;
                    const arma::mat spin_g=2*g-g.t();
                    if (w_seed) bare_grams.slice(j)=g;
                    if (primal_outputs) local_energy+=arma::accu(tj%spin_g);
                    if (energy_matrices) {
                        // Different partner j values own disjoint output columns.
                        // Stream the contraction without another nv*nv*no cube.
                        const arma::vec spin_column(const_cast<double*>(spin_g.memptr()),
                                                    nv*nv,false,true);
                        out.energy_occupied.col(j)+=flat.t()*spin_column;
                        // Polarize the quadratic virtual density in t and g.
                        // This treats the two amplitude positions equally and
                        // reduces to D_vv when g=t.
                        local_energy_virtuals+=0.25*(tj*spin_g.t()+tj.t()*spin_g
                                                         +g*spin.t()+g.t()*spin);
                    }
                    if (energy_active) {
                        arma::mat& bar_tj=bar_amplitudes[s].slice(j);
                        bar_tj+=seed->energy*spin_g;
                        const arma::mat bar_g=seed->energy*spin;
                        arma::mat bar_bj(out.bar_b.colptr(j*nv),np,nv,false,true);
                        bar_bj+=bi*bar_g;
                        local_bi+=bj*bar_g.t();
                    }
                }
                if (seed) {
                    arma::mat& bar_tj=bar_amplitudes[s].slice(j);
                    if (virtual_active) {
                        const arma::mat direct=0.5*(wv*spin+spin*wv);
                        const arma::mat bar_spin=0.5*(wv*tj+tj*wv);
                        bar_tj+=direct+2*bar_spin-bar_spin.t();
                    }
                }
            }
            #pragma omp critical
            {
                if (primal_outputs) { out.virtuals+=local; out.energy+=local_energy; }
                if (energy_matrices) out.energy_virtuals+=local_energy_virtuals;
                if (energy_active)
                    out.bar_b.cols(group[s]*nv,group[s]*nv+nv-1)+=local_bi;
            }
        }
        if (w_seed) {
            // W_oo = T^T spin(G). Its symmetric seed includes bar_E I because
            // the selected energy is trace(W_oo). Both matrix products use
            // disjoint row panels to keep large virtual spaces parallel.
            arma::cube spin_grams(nv,nv,no),bar_spin_grams(nv,nv,no);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t j=0;j<no;++j)
                spin_grams.slice(j)=2*bare_grams.slice(j)-bare_grams.slice(j).t();
            const arma::mat gram_flat(spin_grams.memptr(),nv*nv,no,false,true);
            arma::mat bar_flat(bar_amplitudes[s].memptr(),nv*nv,no,false,true);
            arma::mat bar_gram_flat(bar_spin_grams.memptr(),nv*nv,no,false,true);
            if (w_occupied_active) {
                const size_t row_panel=8192;
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (size_t first=0;first<flat.n_rows;first+=row_panel) {
                    const size_t last=std::min(size_t(flat.n_rows),first+row_panel)-1;
                    bar_flat.rows(first,last)+=gram_flat.rows(first,last)*w_wo.t();
                    bar_gram_flat.rows(first,last)=flat.rows(first,last)*w_wo;
                }
            } else bar_spin_grams.zeros();
            #pragma omp parallel num_threads(threads)
            {
                arma::mat local_bi(np,nv,arma::fill::zeros);
                #pragma omp for schedule(static)
                for (size_t j=0;j<no;++j) {
                    const arma::mat& tj=t.slice(j);
                    const arma::mat& g=bare_grams.slice(j);
                    const arma::mat& spin=spins.slice(j);
                    const arma::mat& spin_g=spin_grams.slice(j);
                    arma::mat& bar_t=bar_amplitudes[s].slice(j);
                    arma::mat bar_spin_g=bar_spin_grams.slice(j);
                    arma::mat bar_g=2*bar_spin_g-bar_spin_g.t();
                    if (w_virtual_active) {
                        const arma::mat bar_spin_t=.25*(w_wv*g+g*w_wv);
                        bar_t+=.25*(w_wv*spin_g+spin_g*w_wv)
                               +2*bar_spin_t-bar_spin_t.t();
                        bar_spin_g=.25*(w_wv*tj+tj*w_wv);
                        bar_g+=.25*(w_wv*spin+spin*w_wv)
                               +2*bar_spin_g-bar_spin_g.t();
                    }
                    const arma::mat bj(const_cast<double*>(bare.colptr(j*nv)),np,nv,false,true);
                    arma::mat bar_bj(out.bar_b.colptr(j*nv),np,nv,false,true);
                    bar_bj+=bi*bar_g;
                    local_bi+=bj*bar_g.t();
                }
                #pragma omp critical
                out.bar_b.cols(group[s]*nv,group[s]*nv+nv-1)+=local_bi;
            }
        }
        out.virtual_energy_seconds+=omp_get_wtime()-tick;
    }
    if (primal_outputs) {
        out.occupied=0.5*(out.occupied+out.occupied.t());
        out.virtuals=0.5*(out.virtuals+out.virtuals.t());
    }
    if (energy_matrices) {
        out.energy_occupied=0.5*(out.energy_occupied+out.energy_occupied.t());
        out.energy_virtuals=0.5*(out.energy_virtuals+out.energy_virtuals.t());
        if (!primal_outputs) out.energy=arma::trace(out.energy_occupied);
    }
    out.contraction_seconds=omp_get_wtime()-contraction_start;
    out.seconds=omp_get_wtime()-start;
    if (!seed && !w_seed) return out;

    const double reverse_start=omp_get_wtime();
    // Reuse the factor-sized cotangent across all points. The first selected
    // orbital writes every block, so no full-array zero pass is needed.
    arma::mat gamma(np,no*nv);
    // Consume the final forward C first, then rebuild one factor at a time.
    for (size_t step=0;step<beta.n_elem;++step) {
        const size_t l=beta.n_elem-1-step;
        double tick=omp_get_wtime();
        if (step) { dress.apply(l,c); ++out.reverse_factor_rebuilds; }
        out.reverse_factor_seconds+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        for (size_t s=0;s<group.size();++s) {
            const arma::cube& bt=bar_amplitudes[s];
            const arma::mat ci(c.colptr(group[s]*nv),np,nv,false,true);
            arma::mat gamma_i(gamma.colptr(group[s]*nv),np,nv,false,true);
            #pragma omp parallel num_threads(threads)
            {
                arma::mat local_i(np,nv,arma::fill::zeros);
                #pragma omp for schedule(static)
                for (size_t j=0;j<no;++j) {
                    const arma::mat cj(c.colptr(j*nv),np,nv,false,true);
                    arma::mat gamma_j(gamma.colptr(j*nv),np,nv,false,true);
                    if (s==0) gamma_j=-weights(l)*(ci*bt.slice(j));
                    else gamma_j-=weights(l)*(ci*bt.slice(j));
                    local_i-=weights(l)*(cj*bt.slice(j).t());
                }
                #pragma omp critical
                gamma_i+=local_i;
            }
        }
        out.amplitude_vjp_seconds+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        dress.backpropagate(l,c,gamma,out.bar_fo,out.bar_fv,out.bar_b);
        out.propagator_vjp_seconds+=omp_get_wtime()-tick;
    }
    out.reverse_seconds=omp_get_wtime()-reverse_start;
    out.seconds+=out.reverse_seconds;
    return out;
}

// Return the primal values and dL/dF_occ, dL/dF_virt, dL/dB. Fock inputs are
// symmetric; their cotangents are symmetric as well. The propagator's common
// spectral shift cancels between occupied and virtual exponentials.
inline deterministic_result evaluate_deterministic_cotangent(
        const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
        const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, int threads, double bar_energy,
        const arma::mat& bar_occupied, const arma::mat& bar_virtuals,
        bool energy_matrices=false) {
    const deterministic_cotangent_seed seed{bar_energy,bar_occupied,bar_virtuals};
    return evaluate_deterministic(fo,fv,bare,beta,weights,group,threads,&seed,energy_matrices);
}

// One selected-group forward density for the corresponding local CCSD(T) solve.
// No amplitude cube is retained after this call.
inline deterministic_result selected_deterministic_mp2_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads) {
    return evaluate_deterministic(fo,fv,bare,beta,weights,group,threads);
}

// Called after the local CCSD(T) reverse pass supplies the output cotangents.
// Recompute selected amplitudes, then run the existing parallel VJP without
// forming the E, D_oo or D_vv output contractions a second time.
inline deterministic_result selected_deterministic_mp2_vjp(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads,
        double bar_energy,const arma::mat& bar_occupied,
        const arma::mat& bar_virtuals) {
    const deterministic_cotangent_seed seed{bar_energy,bar_occupied,bar_virtuals};
    return evaluate_deterministic(fo,fv,bare,beta,weights,group,threads,&seed,false,false);
}

// W is the energy-weighted pair of density matrices; trace(W_oo)=E.
// Its forward pass shares the amplitude construction and bare/dressed
// contractions with the D path, while omitting the D contractions.
inline deterministic_result selected_deterministic_mp2_w_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads) {
    return evaluate_deterministic(fo,fv,bare,beta,weights,group,threads,nullptr,true,false);
}

inline deterministic_result selected_deterministic_mp2_w_vjp(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads,
        double bar_energy,const arma::mat& bar_occupied,
        const arma::mat& bar_virtuals) {
    const deterministic_w_cotangent_seed seed{bar_energy,bar_occupied,bar_virtuals};
    return evaluate_deterministic(fo,fv,bare,beta,weights,group,threads,
                                  nullptr,false,false,&seed);
}

// Preserve the single-index calls for existing callers.
inline deterministic_result selected_deterministic_mp2_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,size_t s,int threads) {
    return selected_deterministic_mp2_forward(fo,fv,bare,beta,weights,
        std::vector<size_t>{s},threads);
}
inline deterministic_result selected_deterministic_mp2_vjp(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,size_t s,int threads,
        double bar_energy,const arma::mat& bar_occupied,const arma::mat& bar_virtuals) {
    return selected_deterministic_mp2_vjp(fo,fv,bare,beta,weights,
        std::vector<size_t>{s},threads,bar_energy,bar_occupied,bar_virtuals);
}
inline deterministic_result selected_deterministic_mp2_w_forward(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,size_t s,int threads) {
    return selected_deterministic_mp2_w_forward(fo,fv,bare,beta,weights,
        std::vector<size_t>{s},threads);
}
inline deterministic_result selected_deterministic_mp2_w_vjp(
        const arma::mat& fo,const arma::mat& fv,const arma::mat& bare,
        const arma::vec& beta,const arma::vec& weights,size_t s,int threads,
        double bar_energy,const arma::mat& bar_occupied,const arma::mat& bar_virtuals) {
    return selected_deterministic_mp2_w_vjp(fo,fv,bare,beta,weights,
        std::vector<size_t>{s},threads,bar_energy,bar_occupied,bar_virtuals);
}

}}}}
#endif
