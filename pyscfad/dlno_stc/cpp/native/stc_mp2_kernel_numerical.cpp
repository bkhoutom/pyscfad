#include "stc_mp2_kernel_qchem.h"
#include <iomanip>
#include <iostream>
namespace libgmbpt { namespace qchem {
using stc_mp2::alias_distribution;
using stc_mp2::residual_tables;
void stc_mp2_kernel::estimate_n_sample(
		const arma::vec& beta, const arma::vec& weights, double energy_tolerance) {
	if (beta.is_empty() || beta.n_elem != weights.n_elem)
		throw std::runtime_error("STC-MP2 requires matching nonempty Laplace nodes and weights");
	if (!(energy_tolerance > 0.0) || !std::isfinite(energy_tolerance))
		throw std::runtime_error("STC-MP2 energy tolerance must be positive and finite");

	laplace_weights_ = weights;
	const size_t n_points = beta.n_elem;
	variance_targets_.set_size(n_points);
	
	arma::vec v_estimator(n_points);

	for (size_t i = 0; i < n_points; i++){
		arma::mat m_F_occ_exp = arma::expmat_sym(beta(i) * occupied_fock_ / 2);
		arma::mat m_F_virt_exp = arma::expmat_sym(-beta(i) * virtual_fock_ / 2);

		v_estimator(i) = laplace_weights_(i) * std::pow(arma::trace(m_F_occ_exp) * arma::trace(m_F_virt_exp), 1.4);
	}
	// variance_targets_ is the target variance of the unweighted contraction E_l.
	// The Laplace energy is sum_l w_l E_l, so allocate Var(w_l E_l) and convert
	// back to Var(E_l) by dividing by w_l^2.
	double v_est_sum = arma::accu(v_estimator);
	for (size_t i = 0; i < n_points; i++){
		variance_targets_(i) = energy_tolerance * energy_tolerance * v_estimator(i)
			/ v_est_sum / (laplace_weights_(i) * laplace_weights_(i));
	}
	variance_targets_.print("variance_targets_");
}

void stc_mp2_kernel::transform_integrals(double beta) {
	double t0 = omp_get_wtime();
	arma::mat m_F_occ_exp = arma::expmat_sym(beta * occupied_fock_ / 2);
	arma::mat m_F_virt_exp = arma::expmat_sym(-beta * virtual_fock_ / 2);

	// B_ia_ is stored as B(P, i*nvirt_+a), so its memory is also a
	// (P,a)-by-i matrix.  Apply the occupied transform in one large GEMM,
	// then apply the virtual transform to each contiguous P-by-a block.
	// This is algebraically V^T B_P O for every P, without transposing the
	// full three-index tensor or launching two small GEMMs per auxiliary.
	double t_occ = omp_get_wtime();
	arma::mat m_B_Pa_i(B_ia_.memptr(), naux_*nvirt_, nocc_, false, true);
	m_B_Pa_i = m_B_Pa_i * m_F_occ_exp;
	double t_virt = omp_get_wtime();
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < nocc_; ++i) {
		arma::mat m_B_Pa(B_ia_.colptr(i*nvirt_), naux_, nvirt_, false, true);
		m_B_Pa = m_B_Pa * m_F_virt_exp;
	}
	std::cout << "timing  Laplace occupied GEMM: " << std::fixed << std::setprecision(3)
			  << (t_virt-t_occ) << " s" << std::endl;
	std::cout << "timing  Laplace virtual GEMMs: " << std::fixed << std::setprecision(3)
			  << (omp_get_wtime()-t_virt) << " s" << std::endl;
	std::cout << "timing  Laplace transform:    " << std::fixed << std::setprecision(3)
			  << (omp_get_wtime() - t0) << " s" << std::endl;
}

void stc_mp2_kernel::detransform_integrals(double beta1, double beta2) {
	double t0 = omp_get_wtime();
	arma::mat m_F_occ_exp = arma::expmat_sym((-beta1 + beta2) * occupied_fock_ / 2);
	arma::mat m_F_virt_exp = arma::expmat_sym((beta1 - beta2) * virtual_fock_ / 2);
	double t_occ = omp_get_wtime();
	arma::mat m_B_Pa_i(B_ia_.memptr(), naux_*nvirt_, nocc_, false, true);
	m_B_Pa_i = m_B_Pa_i * m_F_occ_exp;
	double t_virt = omp_get_wtime();
	#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < nocc_; ++i) {
		arma::mat m_B_Pa(B_ia_.colptr(i*nvirt_), naux_, nvirt_, false, true);
		m_B_Pa = m_B_Pa * m_F_virt_exp;
	}
	std::cout << "timing  Laplace occupied GEMM: " << std::fixed << std::setprecision(3)
			  << (t_virt-t_occ) << " s" << std::endl;
	std::cout << "timing  Laplace virtual GEMMs: " << std::fixed << std::setprecision(3)
			  << (omp_get_wtime()-t_virt) << " s" << std::endl;
	std::cout << "timing  Laplace detransform:  " << std::fixed << std::setprecision(3)
			  << (omp_get_wtime() - t0) << " s" << std::endl;
}

double stc_mp2_kernel::select_domain_tau() {
	// The standalone constructor supplies the explicit threshold.
	if (!(selected_domain_tau_ > 0.0) || !std::isfinite(selected_domain_tau_))
		throw std::runtime_error("QCHEM_STC_E_TAU must be positive and finite");
	return selected_domain_tau_;
}

stc_mp2_kernel::sampling_context stc_mp2_kernel::prepare_sampling(
        size_t point, const arma::mat* shared_norms, const arma::mat* shared_atoms) {
    sampling_context context(nocc_, point);
    // Rank-one magnitude models are separate for B and C. The full-energy
    // path uses its half-time tensor for both roles, as before.
    const bool selected_mode = !selected_mask_.empty();
    auto build_weights = [&](const arma::mat& factor, arma::mat& norms,
                             arma::mat& atoms, arma::mat& sums) {
        norms.set_size(nocc_, nvirt_);
        arma::vec rows(naux_);
        #pragma omp parallel for collapse(2)
        for (size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a)
            norms(i,a)=arma::norm(factor.col(i*nvirt_+a));
        #pragma omp parallel for
        for(size_t p=0;p<naux_;++p) rows(p)=arma::norm(factor.row(p));
        atoms.zeros(natom_,nocc_*nvirt_); sums.zeros(nocc_,nvirt_);
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
            const size_t ia=i*nvirt_+a;
            for(size_t at=0;at<natom_;++at)
                for(size_t p=offset_aux_(at);p<offset_aux_(at+1);++p)
                    atoms(at,ia)+=std::abs(factor(p,ia))*rows(p);
            sums(i,a)=arma::accu(atoms.col(ia));
        }
    };
    if (shared_norms || shared_atoms) {
        if (!shared_norms || !shared_atoms || shared_norms->n_rows!=nocc_
            || shared_norms->n_cols!=nvirt_ || shared_atoms->n_rows!=natom_
            || shared_atoms->n_cols!=nocc_*nvirt_)
            throw std::invalid_argument("Invalid shared STC-MP2 point tables");
        // Copy only the small envelopes: support floors below must not mutate
        // the density proposals. The full C factor is borrowed without copying.
        orbital_norms_=*shared_norms; atom_weights_=*shared_atoms;
        atom_weight_sums_.set_size(nocc_,nvirt_);
    } else build_weights(B_ia_,orbital_norms_,atom_weights_,atom_weight_sums_);
    if (selected_mode && bare_orbital_norms_.is_empty())
        build_weights(original_B_ia_,bare_orbital_norms_,bare_atom_weights_,bare_atom_weight_sums_);
    const arma::mat& bare_norms=selected_mode ? bare_orbital_norms_ : orbital_norms_;
    const double w_lap=laplace_weights_(point), w14=std::pow(std::abs(w_lap),.25);
    const double tau=select_domain_tau();
    auto& keep=context.keep;
    keep.zeros(nocc_,nvirt_);
    for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
        // Geometric mean of the two orbital norms; no extra factor tensor.
        const double norm=selected_mode ? std::sqrt(bare_norms(i,a)*orbital_norms_(i,a))
                                        : orbital_norms_(i,a);
        keep(i,a)=full_domain_ || w14*norm>tau;
    }
    std::cout << "tau = " << std::scientific << tau << "  w = " << w_lap
              << "  w^{1/4} = " << w14 << std::endl;
    auto make_distributions = [&](arma::mat& norms, arma::mat& atoms, arma::mat& sums,
                                 std::vector<alias_distribution>& distributions) {
        // Keep derivative support at exact zeros; no proposal mixture.
        if(selected_mode) {
            const double nf=std::max(1e-100,norms.max()*1e-12);
            const double af=std::max(1e-100,atoms.max()*1e-12);
            norms.transform([&](double x){return std::max(x,nf);});
            atoms.transform([&](double x){return std::max(x,af);});
        }
        distributions.resize(nocc_*nvirt_);
        #pragma omp parallel for collapse(2)
        for(size_t i=0;i<nocc_;++i) for(size_t a=0;a<nvirt_;++a) {
            const size_t ia=i*nvirt_+a;
            sums(i,a)=arma::accu(atoms.col(ia));
            distributions[ia].reset(atoms.colptr(ia),atoms.colptr(ia)+natom_);
        }
    };
    make_distributions(orbital_norms_,atom_weights_,atom_weight_sums_,context.atom_distributions);
    // B is unchanged across Laplace points. Its floors and atom aliases are too.
    if(selected_mode && bare_atom_distributions_.empty())
        make_distributions(bare_orbital_norms_,bare_atom_weights_,bare_atom_weight_sums_,bare_atom_distributions_);

    context.direct_seed = explicit_seed_;
    context.exchange_seed = explicit_seed_;
    return context;
}

void stc_mp2_kernel::sample_mp2_exchange(
        double& E_corr, sampling_context& context, sampling_phase phase) {
    const size_t point = context.point;
    const double w_lap = laplace_weights_(point);
    const auto& keep = context.keep;
    const auto& dist_A_ia = context.atom_distributions;
    const auto& dist_B_ia = selected_mask_.empty() ? dist_A_ia : bare_atom_distributions_;
    const arma::mat& bn = selected_mask_.empty() ? orbital_norms_ : bare_orbital_norms_;
    const arma::mat& bw = selected_mask_.empty() ? atom_weights_ : bare_atom_weights_;
    const arma::mat& bs = selected_mask_.empty() ? atom_weight_sums_ : bare_atom_weight_sums_;
    const double t0 = omp_get_wtime();
	auto in_D = [&](size_t i, size_t j, size_t a) -> bool {
		return keep(i, a) && keep(j, a);
	};

    if (phase != sampling_phase::production) {
        if (!selected_mask_.empty()) {
            context.exchange_dense = selected_dense(keep, true);
        } else {
            // ---- dense: a,b ∈ D_ij, exact PQ via G = M_i^T M_j ----
            double t_dense = omp_get_wtime();
            double E_dense = 0.0;
            size_t nD_sum = 0;
            size_t nD_max = 0;
            size_t n_pairs = nocc_ * nocc_;
            #pragma omp parallel
            {
                double E_loc = 0.0;
                size_t nD_sum_loc = 0;
                size_t nD_max_loc = 0;
                #pragma omp for schedule(dynamic)
                for (size_t i = 0; i < nocc_; i++){
                    for (size_t j = i; j < nocc_; j++){
                        const size_t pair_factor = (i == j) ? 1 : 2;
                        std::vector<size_t> D;
                        D.reserve(nvirt_);
                        for (size_t a = 0; a < nvirt_; a++){
                            if (in_D(i, j, a)){
                                D.push_back(a);
                            }
                        }
                        const size_t nD = D.size();
                        nD_sum_loc += pair_factor*nD;
                        if (nD > nD_max_loc){
                            nD_max_loc = nD;
                        }
                        if (nD == 0){
                            continue;
                        }
                        arma::mat Mi(naux_, nD);
                        arma::mat Mj(naux_, nD);
                        for (size_t d = 0; d < nD; d++){
                            Mi.col(d) = B_ia_.col(i*nvirt_ + D[d]);
                            Mj.col(d) = B_ia_.col(j*nvirt_ + D[d]);
                        }
                        arma::mat G = Mi.t() * Mj;
                        E_loc += pair_factor*arma::accu(G % G.t());
                    }
                }
                #pragma omp atomic
                E_dense += E_loc;
                #pragma omp atomic
                nD_sum += nD_sum_loc;
                #pragma omp critical
                {
                    if (nD_max_loc > nD_max){
                        nD_max = nD_max_loc;
                    }
                }
            }
            std::cout << "dense |D_ij| mean = " << std::fixed << std::setprecision(2)
                      << (double)nD_sum / std::max((size_t)1, n_pairs)
                      << "  max = " << nD_max
                      << "  E_dense = " << std::scientific << std::setprecision(15) << E_dense
                      << std::endl;
            std::cout << "timing  exchange dense:       " << std::fixed << std::setprecision(3)
                      << (omp_get_wtime() - t_dense) << " s" << std::endl;

            context.exchange_dense = E_dense;
        }
    }
	std::vector<size_t> active_first;
	active_first.reserve(nocc_);
	for (size_t i = 0; i < nocc_; ++i)
		if (selected(i)) active_first.push_back(i);
	arma::mat xi_ij_a, eta_ij_a;
	arma::vec w_ij;
	if (phase != sampling_phase::production) {
		// Only selected first indices can have nonzero pair probability.
		xi_ij_a.set_size(nvirt_, active_first.size()*nocc_);
		eta_ij_a.set_size(nvirt_, active_first.size()*nocc_);
        w_ij.zeros(nocc_*nocc_);
	}
	const unsigned exchange_seed = context.exchange_seed;
	auto& exchange_tables = context.exchange_tables;

	auto fill_safe_dist = [&](alias_distribution& dist, double* col) {
		double s = 0.0;
		for (size_t k = 0; k < nvirt_; k++){
			s += col[k];
		}
		if (s == 0.0){
			col[0] = 1.0;
		}
		dist.reset(col, col + nvirt_);
		if (s == 0.0){
			col[0] = 0.0;
		}
	};
	auto run_exchange_residual = [&](bool allow_a_in, bool allow_a_out,
						 bool allow_b_in, bool allow_b_out,
						 const char* tag, size_t term_id,
						 sampling_pilot_stats* pilot_stats,
						 size_t production_count) -> double {
		const double term_start = omp_get_wtime();
		std::vector<alias_distribution>& dist_a_ij = exchange_tables.first[term_id];
		std::vector<alias_distribution>& dist_b_ij = exchange_tables.second[term_id];
		if (!exchange_tables.loaded[term_id]) {
		#pragma omp parallel for collapse(2)
		for (size_t s = 0; s < active_first.size(); s++){
			for (size_t j = 0; j < nocc_; j++){
				const size_t i = active_first[s], col = s*nocc_ + j;
				for (size_t a = 0; a < nvirt_; a++){
					const bool d = in_D(i, j, a);
					const bool a_ok = (d && allow_a_in) || ((!d) && allow_a_out);
					const bool b_ok = (d && allow_b_in) || ((!d) && allow_b_out);
					xi_ij_a(a, col) = a_ok ? bs(i, a) * atom_weight_sums_(j, a) : 0.0;
					eta_ij_a(a, col) = b_ok ? orbital_norms_(i, a) * bn(j, a) : 0.0;
				}
			}
		}
		#pragma omp parallel for collapse(2)
		for (size_t s = 0; s < active_first.size(); s++){
			for (size_t j = 0; j < nocc_; j++){
				const size_t i = active_first[s], ij = i*nocc_ + j;
				const size_t col = s*nocc_ + j;
				double xi = 0.0;
				double eta = 0.0;
				for (size_t a = 0; a < nvirt_; a++){
					xi += xi_ij_a(a, col);
					eta += eta_ij_a(a, col);
				}
				w_ij(ij) = xi * eta;
				fill_safe_dist(dist_a_ij[ij], xi_ij_a.colptr(col));
				fill_safe_dist(dist_b_ij[ij], eta_ij_a.colptr(col));
			}
		}
		exchange_tables.normalization[term_id] = arma::accu(w_ij);
		exchange_tables.outer[term_id].reset(w_ij.begin(), w_ij.end());
		exchange_tables.loaded[term_id] = true;
		}
		const double Z = exchange_tables.normalization[term_id];
		auto one_sample = [&](std::mt19937& local_rng) -> double {
			size_t ij = exchange_tables.outer[term_id](local_rng);
			size_t i = ij / nocc_;
			size_t j = ij % nocc_;
			size_t a = dist_a_ij[i*nocc_ + j](local_rng);
			size_t b = dist_b_ij[i*nocc_ + j](local_rng);
			size_t ia = i*nvirt_ + a;
			size_t ja = j*nvirt_ + a;
			size_t At = dist_B_ia[ia](local_rng);
			size_t Ct = dist_A_ia[ja](local_rng);
			double wA = bw(At, ia);
			double wC = atom_weights_(Ct, ja);
			if (wA == 0.0 || wC == 0.0 || orbital_norms_(i, b) == 0.0 || bn(j, b) == 0.0){
				return 0.0;
			}
			size_t ib = i*nvirt_ + b;
			size_t jb = j*nvirt_ + b;
			const double sP = stc_mp2::group_dot(selected_mask_.empty() ? B_ia_ : original_B_ia_, offset_aux_, At, ia, jb);
			const double sQ = stc_mp2::group_dot(B_ia_, offset_aux_, Ct, ib, ja);
			const double denominator = orbital_norms_(i, b)*bn(j, b)*wA*wC;
			if (record_gradient_ && pilot_stats == NULL && production_count != 0) {
				const double inverse_probability = Z/denominator;
				const double scale = w_lap*inverse_probability/double(production_count);
				record_gamma_update(At, ia, jb, scale*sQ, !selected_mask_.empty());
				record_gamma_update(At, jb, ia, scale*sQ, !selected_mask_.empty());
				record_gamma_update(Ct, ib, ja, scale*sP);
				record_gamma_update(Ct, ja, ib, scale*sP);
			}
			return sP*sQ*Z/denominator;
		};
		double variance_of_mean = 0.0;
		const double result = stc_mp2::run_residual(
			tag, Z, term_id, 0.25*variance_targets_(point), production_count,
			exchange_seed, point, one_sample, pilot_stats, variance_of_mean);
		if (!pilot_stats) context.exchange_variance += variance_of_mean;
		std::cout << "timing  " << tag << ": " << std::fixed << std::setprecision(3)
				  << (omp_get_wtime() - term_start) << " s" << std::endl;
		return result;
	};

	std::cout << "timing  exchange tables:      " << std::fixed << std::setprecision(3)
			  << (omp_get_wtime() - t0) << " s" << std::endl;

    if (phase != sampling_phase::production) {
        run_exchange_residual(false, true, true, true, "stoch a~D", 0,
                              phase==sampling_phase::pilot ? &context.exchange_pilots[0] : nullptr, 0);
        run_exchange_residual(true, false, false, true, "stoch b~D", 1,
                              phase==sampling_phase::pilot ? &context.exchange_pilots[1] : nullptr, 0);
        E_corr = context.exchange_dense;
        return;
    }
    const double E3 = run_exchange_residual(true, false, false, true, "stoch b~D", 1,
                                          NULL, context.production_counts[1]);
    const double E2 = run_exchange_residual(false, true, true, true, "stoch a~D", 0,
                                          NULL, context.production_counts[0]);
    E_corr = context.exchange_dense + E2 + E3;
    std::cout << "n_sample exchange total E_corr = " << std::scientific
              << std::setprecision(15) << E_corr << std::endl;
    std::cout << "timing  exchange sample: " << std::fixed << std::setprecision(3)
              << omp_get_wtime()-t0 << " s" << std::endl;
    context.exchange_tables = residual_tables(0);
}

void stc_mp2_kernel::allocate_samples(sampling_context& context) const {
    const size_t point = context.point;
    const auto& exchange_pilots = context.exchange_pilots;
    const auto& direct_pilots = context.direct_pilots;
    const double budget_coefficient = 2.0;

    std::vector<sampling_pilot_stats> pilots(4);
    pilots[0] = exchange_pilots[0];
    pilots[1] = exchange_pilots[1];
    pilots[2] = direct_pilots[0];
    pilots[3] = direct_pilots[1];
    const char* labels[4] = {"exchange a~D", "exchange b~D",
                             "direct a~D", "direct b~D"};
    double coefficients[4] = {1.0, 1.0, budget_coefficient, budget_coefficient};
    double allocation_weight[4] = {0.0, 0.0, 0.0, 0.0};
    double weight_sum = 0.0;
    for (size_t k = 0; k < 4; ++k) {
        const double contribution_variance = coefficients[k]*coefficients[k]*pilots[k].variance;
        const double sample_cost = std::max(1.0e-15, pilots[k].seconds_per_sample);
        allocation_weight[k] = std::sqrt(std::max(0.0, contribution_variance*sample_cost));
        weight_sum += allocation_weight[k];
    }
    if (weight_sum < 0.0 || !std::isfinite(weight_sum))
        throw std::runtime_error("Invalid pilot statistics in four-way variance allocation");

    size_t active_estimators = 0;
    for (size_t k = 0; k < 4; ++k)
        if (pilots[k].count > 0) ++active_estimators;

    auto& production_counts = context.production_counts;
    std::cout << "dynamic variance allocation point = " << point
              << " target = " << std::scientific << variance_targets_(point) << std::endl;
    for (size_t k = 0; k < 4; ++k) {
        const double share = weight_sum > 0.0 ? allocation_weight[k]/weight_sum
            : (pilots[k].count > 0 ? 1.0/double(active_estimators) : 0.0);
        const double contribution_target = variance_targets_(point)*share;
        const double contribution_variance = coefficients[k]*coefficients[k]*pilots[k].variance;
        double requested = contribution_target > 0.0
                         ? std::ceil(contribution_variance/contribution_target) : 0.0;
        if (!std::isfinite(requested) || requested >= double(std::numeric_limits<size_t>::max()))
            throw std::runtime_error("Dynamic sampling budget overflow");
        if (pilots[k].count > 0)
            requested = std::max(10000.0, requested);
        production_counts[k] = static_cast<size_t>(requested);
        std::cout << "allocation " << labels[k]
                  << " share = " << std::fixed << std::setprecision(6) << share
                  << " pilot_var = " << std::scientific << pilots[k].variance
                  << " ns_per_sample = " << pilots[k].seconds_per_sample*1.0e9
                  << " production_samples = " << production_counts[k] << std::endl;
    }

}

void stc_mp2_kernel::sample_mp2_direct(
        double& E_corr, sampling_context& context, sampling_phase phase) {
    const size_t point = context.point;
    const auto& keep = context.keep;
    const auto& atom_distributions = context.atom_distributions;
    const auto& production_counts = context.production_counts;
    const double t0 = omp_get_wtime();
    const size_t nx = nocc_ * nvirt_;
    const double w = laplace_weights_(point);

    // Direct term, with x=(ia) and y=(jb):
    //   D = sum_xyPQ R_Px R_Qx R_Py R_Qy.
    // As in the exchange sampler, use the rank-one magnitude model
    // |R_Px| ~ A_x B_P for two factors.  The importance function is
    //   pi(x,y,P,Q) ~ A_x^2 B_P B_Q |R_Py| |R_Qy|,
    // which factors as p(ij) p(a|i) p(b|j) p(P|jb) p(Q|jb).
    // Auxiliary indices are sampled as atom groups and summed exactly inside
    // each selected group.
    const arma::mat& atom_weight = atom_weights_;
    const auto& bare_distributions = selected_mask_.empty() ? atom_distributions : bare_atom_distributions_;
    const arma::mat& bn = selected_mask_.empty() ? orbital_norms_ : bare_orbital_norms_;
    const arma::mat& bw = selected_mask_.empty() ? atom_weights_ : bare_atom_weights_;
    const arma::mat& bs = selected_mask_.empty() ? atom_weight_sums_ : bare_atom_weight_sums_;
    arma::vec z_y(nx, arma::fill::zeros);
    for (size_t y = 0; y < nx; ++y) {
        const size_t i = y/nvirt_, a = y%nvirt_;
        z_y(y) = atom_weight_sums_(i,a);
    }

    const double t_tables = omp_get_wtime() - t0;

    double t_dense = 0.0;
    if (phase != sampling_phase::production) {
        const double t_dense0 = omp_get_wtime();
        if (!selected_mask_.empty()) {
            context.direct_dense = selected_dense(keep, false);
        } else {
            // Exact strong/strong virtual block.
            std::vector<arma::uvec> domains(nocc_);
            for (size_t i = 0; i < nocc_; ++i) {
                std::vector<arma::uword> d;
                for (size_t a = 0; a < nvirt_; ++a)
                    if (keep(i,a)) d.push_back(i*nvirt_+a);
                domains[i] = arma::uvec(d);
            }
            double dense = 0.0;
            #pragma omp parallel for reduction(+:dense) schedule(dynamic)
            for (size_t i = 0; i < nocc_; ++i) {
                if (domains[i].is_empty()) continue;
                arma::mat Mi = B_ia_.cols(domains[i]);
                for (size_t j = i; j < nocc_; ++j) {
                    if (domains[j].is_empty()) continue;
                    arma::mat Mj = B_ia_.cols(domains[j]);
                    arma::mat G = Mi.t()*Mj;
                    dense += (i == j ? 1.0 : 2.0)*arma::accu(G%G);
                }
            }
            context.direct_dense = dense;
        }
        t_dense = omp_get_wtime()-t_dense0;
        context.direct_dense_seconds = t_dense;
    }
    const unsigned seed = context.direct_seed;
    double variance = 0.0;
    arma::vec sum_a(nocc_, arma::fill::zeros);
    arma::vec sum_b(nocc_, arma::fill::zeros);
    arma::mat weight_a(nvirt_, nocc_, arma::fill::zeros);
    arma::mat weight_b(nvirt_, nocc_, arma::fill::zeros);
    arma::vec weight_ij(nocc_*nocc_, arma::fill::zeros);
    auto& direct_tables = context.direct_tables;
    auto run_direct_residual = [&](bool x_strong, bool allow_y_weak,
                         bool allow_y_strong, const char* tag, size_t term_id,
                         sampling_pilot_stats* pilot_stats,
                         size_t production_count) -> double {
        const double term_start = omp_get_wtime();
        // For the strong-x/weak-y residual, exploit the x<->y symmetry of
        // (B_x^T B_y)^2 and put the atom-resolved proposal on the strong side.
        // This is the same orientation used by the local STC direct sampler.
        const bool reverse_atom_source = x_strong && !allow_y_strong;
        std::vector<alias_distribution>& dist_a_i = direct_tables.first[term_id];
        std::vector<alias_distribution>& dist_b_j = direct_tables.second[term_id];
        if (!direct_tables.loaded[term_id]) {
        sum_a.zeros();
        sum_b.zeros();
        weight_a.zeros();
        weight_b.zeros();
        for (size_t i = 0; i < nocc_; ++i) {
            for (size_t a = 0; a < nvirt_; ++a) {
                const size_t x = i*nvirt_+a;
                if (bool(keep(i,a)) == x_strong)
                    weight_a(a,i) = reverse_atom_source ? bs(i,a)*z_y(x)
                        : bn(i,a)*orbital_norms_(i,a);
                const bool y_ok = (keep(i,a) && allow_y_strong)
                               || (!keep(i,a) && allow_y_weak);
                if (y_ok)
                    weight_b(a,i) = reverse_atom_source
                        ? bn(i,a)*orbital_norms_(i,a) : bs(i,a)*z_y(x);
            }
            dist_a_i[i].reset(weight_a.colptr(i), weight_a.colptr(i)+nvirt_);
            dist_b_j[i].reset(weight_b.colptr(i), weight_b.colptr(i)+nvirt_);
            sum_a(i) = arma::accu(weight_a.col(i));
            sum_b(i) = arma::accu(weight_b.col(i));
        }
        weight_ij.zeros();
        for (size_t i = 0; i < nocc_; ++i)
            for (size_t j = 0; j < nocc_; ++j)
                weight_ij(i*nocc_+j) = selected(i) ? sum_a(i)*sum_b(j) : 0.0;
        direct_tables.normalization[term_id] = arma::accu(weight_ij);
        direct_tables.outer[term_id].reset(weight_ij.begin(), weight_ij.end());
        direct_tables.loaded[term_id] = true;
        }
        const double Z = direct_tables.normalization[term_id];
        auto one_sample = [&](std::mt19937& local_rng) -> double {
            const size_t ij = direct_tables.outer[term_id](local_rng);
            const size_t i = ij/nocc_, j = ij%nocc_;
            const size_t a = dist_a_i[i](local_rng), b = dist_b_j[j](local_rng);
            const size_t x = i*nvirt_+a, y = j*nvirt_+b;
            const size_t atom_source = reverse_atom_source ? x : y;
            const double norm_product = reverse_atom_source ? bn(j,b)*orbital_norms_(j,b) : bn(i,a)*orbital_norms_(i,a);
            const size_t at = bare_distributions[atom_source](local_rng);
            const size_t ct = atom_distributions[atom_source](local_rng);
            const double denom = norm_product*bw(at,atom_source)*
                                 atom_weight(ct,atom_source);
            if (denom == 0.0) return 0.0;
            const double sP = stc_mp2::group_dot(selected_mask_.empty() ? B_ia_ : original_B_ia_, offset_aux_, at, x, y);
            const double sQ = stc_mp2::group_dot(B_ia_, offset_aux_, ct, x, y);
            if (record_gradient_ && pilot_stats == NULL && production_count != 0) {
                const double inverse_probability = Z/denom;
                const double scale = -2.0*w*inverse_probability/double(production_count);
                record_gamma_update(at, x, y, scale*sQ, !selected_mask_.empty());
                record_gamma_update(at, y, x, scale*sQ, !selected_mask_.empty());
                record_gamma_update(ct, x, y, scale*sP);
                record_gamma_update(ct, y, x, scale*sP);
            }
            return sP*sQ*Z/denom;
        };
        double term_variance = 0.0;
        const double result = stc_mp2::run_residual(
            tag, Z, term_id, variance_targets_(point)/16.0, production_count,
            seed, point, one_sample, pilot_stats, term_variance);
        variance += term_variance;
        std::cout << "timing  " << tag << ": " << std::fixed << std::setprecision(3)
                  << (omp_get_wtime() - term_start) << " s" << std::endl;
        return result;
    };

    if (phase != sampling_phase::production) {
        run_direct_residual(false, true, true, "direct a~D", 0,
                            phase==sampling_phase::pilot ? &context.direct_pilots[0] : nullptr, 0);
        run_direct_residual(true, true, false, "direct b~D", 1,
                            phase==sampling_phase::pilot ? &context.direct_pilots[1] : nullptr, 0);
        E_corr = context.direct_dense;
        return;
    }

    // Both pilot table sets remain resident for production.
    const double direct_b = run_direct_residual(true, true, false, "direct b~D", 1, NULL,
                                      production_counts[3]);
    const double direct_a = run_direct_residual(false, true, true, "direct a~D", 0, NULL,
                                      production_counts[2]);
    // Weak a / all b plus strong a / weak b covers the complement of dense.
    E_corr = context.direct_dense + direct_a + direct_b;
    const double elapsed = omp_get_wtime() - t0;
    context.direct_variance = variance;
    direct_weighted_sample_ += -2.0*w*E_corr;
    direct_weighted_variance_ += 4.0*w*w*variance;
    std::cout << "direct factorized point = " << point << " E_corr = " << std::scientific
              << std::setprecision(15) << E_corr << " se = " << std::sqrt(variance)
              << " dense = " << context.direct_dense << " seed = " << seed << std::endl;
    std::cout << "timing  direct tables: " << std::fixed << std::setprecision(6) << t_tables << " s\n"
              << "timing  direct dense: " << context.direct_dense_seconds << " s\n"
              << "timing  direct factorized stochastic: " << elapsed << " s" << std::endl;
    context.direct_tables = residual_tables(0);
}



}  // namespace qchem
}  // namespace libgmbpt
