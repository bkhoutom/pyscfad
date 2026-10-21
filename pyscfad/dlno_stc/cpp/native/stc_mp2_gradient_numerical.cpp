#include "stc_mp2_kernel_qchem.h"
extern "C" void daxpy_(const int*, const double*, const double*, const int*, double*, const int*);
inline void CL_DAXPY(int n, double a, const double* x, int ix, double* y, int iy) {
    daxpy_(&n,&a,x,&ix,y,&iy);
}
namespace libgmbpt { namespace qchem {
// Bound aliasing GEMM temporaries: each panel owns disjoint matrix rows.
// This keeps the selected path near four full tensors during rotations too.
static void right_multiply_panels(arma::mat& matrix, const arma::mat& right) {
    const size_t rows=std::max(size_t(1),size_t(32*1024*1024)/(sizeof(double)*size_t(matrix.n_cols)));
    for(size_t first=0;first<matrix.n_rows;first+=rows) {
        const size_t last=std::min(size_t(matrix.n_rows),first+rows)-1;
        matrix.rows(first,last)=matrix.rows(first,last)*right;
    }
}

// Selected mode uses separate bare-B and full-time-C proposal weights.
void stc_mp2_kernel::select_occupied(const std::vector<size_t>& indices) {
    if (indices.empty()) throw std::invalid_argument("Empty occupied group");
    selected_mask_.assign(nocc_, false);
    for (size_t i : indices) {
        if (i >= nocc_ || selected_mask_[i]) throw std::invalid_argument("Invalid occupied group");
        selected_mask_[i] = true;
    }
}

void stc_mp2_kernel::prepare_selected_point(double beta) {
    B_ia_ = original_B_ia_;
    const arma::mat o = arma::expmat_sym(beta*occupied_fock_);
    const arma::mat v = arma::expmat_sym(-beta*virtual_fock_);
    arma::mat pa_i(B_ia_.memptr(), naux_*nvirt_, nocc_, false, true);
    right_multiply_panels(pa_i,o);
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<nocc_; ++i) {
        arma::mat pa(B_ia_.colptr(i*nvirt_), naux_, nvirt_, false, true);
        pa = pa*v;
    }
}

// Ordered selected-i pairs. Direct retains D_i x D_j; exchange D_ij x D_ij.
// Pair products run in parallel. Only the short additive column updates are
// locked, by occupied index; no thread owns a full-size gradient copy.
double stc_mp2_kernel::selected_dense(const arma::umat& keep, bool exchange,
                                     double weight, arma::mat* gamma, arma::mat* b_gradient) {
    const double start=omp_get_wtime();
    if (gamma && !b_gradient) throw std::invalid_argument("Missing selected B cotangent");
    std::vector<omp_lock_t> locks(gamma ? nocc_ : 0);
    for (auto& lock : locks) omp_init_lock(&lock);
    double energy=0;
    #pragma omp parallel for collapse(2) reduction(+:energy) schedule(dynamic)
    for (size_t i=0; i<nocc_; ++i)
        for (size_t j=0; j<nocc_; ++j) {
            if (!selected(i)) continue;
            std::vector<arma::uword> ci,cj;
            for (size_t a=0; a<nvirt_; ++a) {
                if (keep(i,a) && (!exchange || keep(j,a))) ci.push_back(i*nvirt_+a);
                if (keep(j,a) && (!exchange || keep(i,a))) cj.push_back(j*nvirt_+a);
            }
            if (ci.empty() || cj.empty()) continue;
            const arma::uvec ix(ci),jx(cj);
            const arma::mat bi=original_B_ia_.cols(ix), bj=original_B_ia_.cols(jx);
            const arma::mat di=B_ia_.cols(ix), dj=B_ia_.cols(jx);
            const arma::mat gb=bi.t()*bj, gc=di.t()*dj;
            arma::mat db=gc, dc=gb;
            if (exchange) { db=gc.t(); dc=gb.t(); }
            energy+=arma::accu(gb%db);
            if (gamma) {
                // Compute outside the lock, then scatter to one orbital block.
                arma::mat delta_b=weight*bj*db.t(), delta_c=weight*dj*dc.t();
                omp_set_lock(&locks[i]);
                b_gradient->cols(ix)+=delta_b;
                gamma->cols(ix)+=delta_c;
                omp_unset_lock(&locks[i]);
                delta_b=weight*bi*db; delta_c=weight*di*dc;
                // Separate acquisitions also handle i==j without deadlock.
                omp_set_lock(&locks[j]);
                b_gradient->cols(jx)+=delta_b;
                gamma->cols(jx)+=delta_c;
                omp_unset_lock(&locks[j]);
            }
        }
    for (auto& lock : locks) omp_destroy_lock(&lock);
    std::cout << "timing  selected dense " << (exchange ? "exchange" : "direct")
              << (gamma ? " VJP: " : " energy: ") << std::fixed << std::setprecision(6)
              << omp_get_wtime()-start << " s" << std::endl;
    return energy;
}

void stc_mp2_kernel::retain_original_integrals(bool release_current) {
    // These envelopes belong to the retained input and include support floors.
    // Rebuild for a new calculation rather than reusing previously floored norms.
    bare_orbital_norms_.reset(); bare_atom_weights_.reset(); bare_atom_weight_sums_.reset();
    bare_atom_distributions_.clear();
    // Combined forward evaluation borrows C from the density scheduler, so it
    // needs no second owning copy of B. Existing gradient callers retain it.
    if (release_current) { original_B_ia_.reset(); original_B_ia_.swap(B_ia_); }
    else original_B_ia_ = B_ia_;
}

void stc_mp2_kernel::restore_original_integrals(bool release_original) {
    if (original_B_ia_.is_empty())
        throw std::runtime_error("Original STC-MP2 integrals were not retained");
    if (release_original) { B_ia_.reset(); B_ia_.swap(original_B_ia_); }
    else B_ia_ = original_B_ia_;
    if (release_original) bare_atom_distributions_.clear();
    direct_weighted_sample_ = 0.0;
    direct_weighted_variance_ = 0.0;
}

arma::mat stc_mp2_kernel::deterministic_gamma(
        const arma::umat& keep, double laplace_weight) const {
    if (keep.n_rows != nocc_ || keep.n_cols != nvirt_)
        throw std::runtime_error("Invalid STC-MP2 deterministic domain shape");
    arma::mat gamma(naux_, nocc_*nvirt_, arma::fill::zeros);
    std::vector<arma::uvec> domains(nocc_);
    for (size_t i = 0; i < nocc_; ++i) {
        std::vector<arma::uword> columns;
        for (size_t a = 0; a < nvirt_; ++a)
            if (keep(i,a)) columns.push_back(i*nvirt_+a);
        domains[i] = arma::uvec(columns);
    }

    // Use all ordered (i,j) pairs.  By i<->j symmetry, twice the first-role
    // derivative is the complete derivative, so each OpenMP iteration owns
    // every column of Gamma_i and requires neither locks nor atomics.
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < nocc_; ++i) {
        if (domains[i].is_empty()) continue;
        const arma::mat Mi = B_ia_.cols(domains[i]);
        for (size_t j = 0; j < nocc_; ++j) {
            if (domains[j].is_empty()) continue;
            const arma::mat Mj = B_ia_.cols(domains[j]);

            // Direct: D_ij=||Mi^T Mj||_F^2 and E_D=-2*w*sum_ij D_ij.
            const arma::mat G_direct = Mi.t()*Mj;
            gamma.cols(domains[i]) +=
                -8.0*laplace_weight*(Mj*G_direct.t());

            std::vector<arma::uword> columns_i;
            std::vector<arma::uword> columns_j;
            for (size_t a = 0; a < nvirt_; ++a) {
                if (keep(i,a) && keep(j,a)) {
                    columns_i.push_back(i*nvirt_+a);
                    columns_j.push_back(j*nvirt_+a);
                }
            }
            if (columns_i.empty()) continue;
            const arma::uvec index_i(columns_i);
            const arma::uvec index_j(columns_j);
            const arma::mat Xi = B_ia_.cols(index_i);
            const arma::mat Xj = B_ia_.cols(index_j);
            const arma::mat G_exchange = Xi.t()*Xj;
            // K_ij=tr(G_ij^2), E_K=w*sum_ij K_ij.  The factor four
            // includes d tr(G^2)/dG and the equal second-role contribution.
            gamma.cols(index_i) +=
                4.0*laplace_weight*(Xj*G_exchange);
        }
    }
    return gamma;
}

void stc_mp2_kernel::record_gamma_update(
        size_t atom, size_t target, size_t source, double alpha, bool bare) {
    if (!record_gradient_ || alpha == 0.0) return;
    if (atom > std::numeric_limits<uint32_t>::max()
            || target > std::numeric_limits<uint32_t>::max()
            || source > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("STC-MP2 gamma update exceeds 32-bit index range");
    const int thread = omp_in_parallel() ? omp_get_thread_num() : 0;
    gradient_updates_.at(static_cast<size_t>(thread)).push_back(gamma_update{
        static_cast<uint32_t>(atom), static_cast<uint32_t>(target),
        static_cast<uint32_t>(source), alpha, bare});
}

void stc_mp2_kernel::scatter_gamma_updates(arma::mat& gamma, arma::mat* b_gradient,
                                           double energy_cotangent) {
    const size_t nx = nocc_*nvirt_;
    const size_t key_count = natom_*nx;
    size_t update_count = 0;
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        update_count += updates.size();

    std::vector<size_t> offsets(key_count+1, 0);
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        for (const gamma_update& update : updates)
            ++offsets[size_t(update.atom)*nx+size_t(update.target)+1];
    for (size_t key = 0; key < key_count; ++key)
        offsets[key+1] += offsets[key];

    // Store only an index permutation, rather than a second copy of every
    // update.  The records themselves remain in per-thread append buffers.
    std::vector<const gamma_update*> ordered(update_count);
    std::vector<size_t> cursor(offsets.begin(), offsets.end()-1);
    for (const std::vector<gamma_update>& updates : gradient_updates_)
        for (const gamma_update& update : updates) {
            const size_t key = size_t(update.atom)*nx+size_t(update.target);
            ordered[cursor[key]++] = &update;
        }

    if (naux_ > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("STC-MP2 auxiliary dimension exceeds DAXPY range");
    #pragma omp parallel for schedule(dynamic,16)
    for (size_t key = 0; key < key_count; ++key) {
        const size_t atom = key/nx;
        const size_t target = key%nx;
        const size_t first = offset_aux_(atom);
        const size_t count = offset_aux_(atom+1)-first;
        for (size_t position = offsets[key]; position < offsets[key+1]; ++position) {
            const gamma_update& update = *ordered[position];
            CL_DAXPY(static_cast<int>(count), energy_cotangent*update.alpha,
                     (update.bare ? original_B_ia_ : B_ia_).colptr(update.source)+first, 1,
                     (update.bare ? *b_gradient : gamma).colptr(target)+first, 1);
        }
    }
    std::cout << "gamma stochastic updates = " << update_count << std::endl;
}

void stc_mp2_kernel::begin_gradient_sampling() {
    if (record_gradient_)
        throw std::runtime_error("Nested STC-MP2 gradient sampling is not supported");
    const int thread_count = std::max(1, omp_get_max_threads());
    gradient_updates_.clear();
    gradient_updates_.resize(static_cast<size_t>(thread_count));
    record_gradient_ = true;
}

void stc_mp2_kernel::finish_gradient_sampling(
        sampling_context& context, arma::mat& gamma, arma::mat* b_gradient,
        bool gamma_preinitialized, double energy_cotangent) {
    if (!record_gradient_)
        throw std::runtime_error("STC-MP2 gradient sampling was not started");
    if (!std::isfinite(energy_cotangent))
        throw std::invalid_argument("Invalid STC-MP2 energy cotangent");
    record_gradient_ = false;
    // Release the C proposals before allocating gamma; B aliases are reused.
    std::vector<stc_mp2::alias_distribution>().swap(context.atom_distributions);
    if (selected_mask_.empty()) {
        gamma = deterministic_gamma(context.keep, laplace_weights_(context.point));
        gamma *= energy_cotangent;
    } else {
        if (gamma_preinitialized) {
            if (gamma.n_rows != naux_ || gamma.n_cols != nocc_*nvirt_)
                throw std::invalid_argument("Preinitialized gamma has the wrong shape");
        } else gamma.zeros(naux_, nocc_*nvirt_);
        if (!b_gradient) throw std::invalid_argument("Selected gradients require the B accumulator");
        if (b_gradient->is_empty()) b_gradient->zeros(naux_, nocc_*nvirt_);
        selected_dense(context.keep, true, energy_cotangent*laplace_weights_(context.point), &gamma, b_gradient);
        selected_dense(context.keep, false, -2*energy_cotangent*laplace_weights_(context.point), &gamma, b_gradient);
    }
    scatter_gamma_updates(gamma, b_gradient, energy_cotangent);
    gradient_updates_.clear();
}

arma::mat stc_mp2_kernel::symmetric_exponential_vjp(
        const arma::mat& fock, double coefficient, const arma::mat& adjoint) {
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    if (!arma::eig_sym(eigenvalues, eigenvectors, fock))
        throw std::runtime_error("STC-MP2 Fock eigendecomposition failed");
    const arma::mat symmetric_adjoint = 0.5*(adjoint+adjoint.t());
    arma::mat eigen_adjoint = eigenvectors.t()*symmetric_adjoint*eigenvectors;
    arma::mat divided_difference(fock.n_rows, fock.n_cols);
    for (size_t p = 0; p < fock.n_rows; ++p) {
        for (size_t q = 0; q < fock.n_cols; ++q) {
            const double xp=coefficient*eigenvalues(p), xq=coefficient*eigenvalues(q);
            const double gap=std::abs(xp-xq);
            // Avoid subtracting nearly equal exponentials for degenerate LMOs.
            divided_difference(p,q)=coefficient*std::exp(std::max(xp,xq))
                *(gap==0 ? 1.0 : -std::expm1(-gap)/gap);
        }
    }
    arma::mat result = eigenvectors*(divided_difference%eigen_adjoint)*eigenvectors.t();
    return 0.5*(result+result.t());
}

void stc_mp2_kernel::backpropagate_laplace_point(
        double beta, arma::mat& gamma, arma::mat& f_occ_gradient,
        arma::mat& f_virt_gradient, arma::mat& b_gradient) const {
    if (gamma.n_rows != naux_ || gamma.n_cols != nocc_*nvirt_)
        throw std::runtime_error("Invalid STC-MP2 gamma shape");
    if (f_occ_gradient.is_empty()) {
        f_occ_gradient.zeros(nocc_, nocc_);
        f_virt_gradient.zeros(nvirt_, nvirt_);
    }
    if (b_gradient.is_empty()) b_gradient.zeros(naux_, nocc_*nvirt_);

    const double half_beta = 0.5*beta;
    const arma::mat occupied_exp = arma::expmat_sym(half_beta*occupied_fock_);
    const arma::mat virtual_exp = arma::expmat_sym(-half_beta*virtual_fock_);
    if (original_B_ia_.n_rows != naux_
            || original_B_ia_.n_cols != nocc_*nvirt_)
        throw std::runtime_error("Retain original STC-MP2 integrals before transforming");
    arma::mat h_occ(nocc_, nocc_, arma::fill::zeros);
    arma::mat h_virt(nvirt_, nvirt_, arma::fill::zeros);

    // For each auxiliary P use virtual-by-occupied matrices X and G:
    // B_tilde_P = V X O, h_O = (V X)^T G, h_V = (X O) G^T.
    // Forward contractions preserve the strongly damped virtual components;
    // reconstructing X from B_tilde with an inverse exponential does not.
    #pragma omp parallel
    {
        arma::mat local_occ(nocc_, nocc_, arma::fill::zeros);
        arma::mat local_virt(nvirt_, nvirt_, arma::fill::zeros);
        arma::mat x(nvirt_, nocc_), g(nvirt_, nocc_);
        #pragma omp for schedule(static)
        for (size_t p = 0; p < naux_; ++p) {
            for (size_t i = 0; i < nocc_; ++i)
                for (size_t a = 0; a < nvirt_; ++a) {
                    x(a,i) = original_B_ia_(p,i*nvirt_+a);
                    g(a,i) = gamma(p,i*nvirt_+a);
                }
            const arma::mat vx = virtual_exp*x;
            const arma::mat xo = x*occupied_exp;
            local_occ += vx.t()*g;
            local_virt += xo*g.t();
        }
        #pragma omp critical
        {
            h_occ += local_occ;
            h_virt += local_virt;
        }
    }
    f_occ_gradient += symmetric_exponential_vjp(
        occupied_fock_, half_beta, h_occ);
    f_virt_gradient += symmetric_exponential_vjp(
        virtual_fock_, -half_beta, h_virt);

    // bar(B) = O Gamma V^T.  Gamma is no longer needed, so reuse its buffer.
    arma::mat gamma_pa_i(gamma.memptr(), naux_*nvirt_, nocc_, false, true);
    if (selected_mask_.empty()) gamma_pa_i = gamma_pa_i*occupied_exp;
    else right_multiply_panels(gamma_pa_i,occupied_exp);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nocc_; ++i) {
        arma::mat gamma_pa(gamma.colptr(i*nvirt_), naux_, nvirt_, false, true);
        gamma_pa = gamma_pa*virtual_exp;
    }
    b_gradient += gamma;
}


} // namespace qchem
} // namespace libgmbpt
