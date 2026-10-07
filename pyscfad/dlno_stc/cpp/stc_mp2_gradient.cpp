#include "stc_mp2_helpers.h"
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pyscfad { namespace dlno_stc {
namespace {
void capture_transform_error(std::exception_ptr& error) {
    #pragma omp critical(stc_mp2_transform_error)
    { if (!error) error = std::current_exception(); }
}

// An in-place full tensor GEMM requires a second tensor for alias protection.
// Keep its input and output copies bounded to 16 MiB each instead.  The rows
// are packed (P,a) rows, so each batch is still a tall occupied-space GEMM.
void transform_occupied_in_place(arma::mat& packed, arma::uword no,
                                 const arma::mat& right) {
    arma::mat pa_i(packed.memptr(), packed.n_elem/no, no, false, true);
    constexpr arma::uword scratch_elements = 16*1024*1024/sizeof(double);
    const arma::uword row_batch = std::max<arma::uword>(1, scratch_elements/no);
    for (arma::uword begin = 0; begin < pa_i.n_rows; begin += row_batch) {
        const arma::uword end = std::min(pa_i.n_rows, begin+row_batch)-1;
        const arma::mat source = pa_i.rows(begin,end);
        const arma::mat transformed = source*right;
        pa_i.rows(begin,end) = transformed;
    }
}
}

void dress_integrals(const arma::mat& B, const arma::mat& O, const arma::mat& V,
                     arma::mat& dressed) {
    const arma::uword no=O.n_rows, nv=V.n_rows, np=B.n_rows;
    dressed.set_size(np,no*nv);

    // Input columns i*nv+a are also a contiguous (P,a)-by-i matrix.  Keep B
    // immutable and write the occupied transform straight into the point
    // buffer, then let each worker own one contiguous P-by-a output block.
    const arma::mat B_pa_i(const_cast<double*>(B.memptr()), np*nv, no, false, true);
    arma::mat dressed_pa_i(dressed.memptr(), np*nv, no, false, true);
    dressed_pa_i = B_pa_i*O.t();
    std::exception_ptr dressing_error;
    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(no); ++i) {
        try {
            arma::mat dressed_pa(dressed.colptr(i*nv), np, nv, false, true);
            dressed_pa = dressed_pa*V;
        } catch (...) {
            capture_transform_error(dressing_error);
        }
    }
    if (dressing_error) std::rethrow_exception(dressing_error);
}

arma::mat exponential(const Spectrum& eig, double coefficient, double shift) {
    if (coefficient == 0.0) return arma::eye(eig.values.n_elem,eig.values.n_elem);
    return eig.vectors * arma::diagmat(arma::exp(coefficient*(eig.values-shift)))
           * eig.vectors.t();
}

// Migrated from the student's symmetric_exponential_vjp. Cache spectra per
// call and evaluate the divided difference with expm1, including degeneracy.
arma::mat exponential_vjp(const Spectrum& eig, double coefficient, double shift,
                          const arma::mat& adjoint) {
    if (coefficient == 0.0) return arma::zeros(eig.values.n_elem,eig.values.n_elem);
    const arma::mat rotated = eig.vectors.t() * (0.5*(adjoint+adjoint.t()))
                              * eig.vectors;
    arma::mat divided(eig.values.n_elem, eig.values.n_elem);
    for (arma::uword p = 0; p < eig.values.n_elem; ++p)
        for (arma::uword q = 0; q < eig.values.n_elem; ++q) {
            const double x = coefficient*(eig.values(p)-shift);
            const double y = coefficient*(eig.values(q)-shift);
            if (x == y) {
                divided(p,q) = coefficient*std::exp(x);
            } else {
                const double d = std::abs(x-y);
                const double divided_factor = d < 1.0
                    ? coefficient*(-std::expm1(-d)/d)
                    : std::copysign(1.0,coefficient)*(-std::expm1(-d)) /
                      std::abs(eig.values(p)-eig.values(q));
                divided(p,q) = std::exp(std::max(x,y))*divided_factor;
            }
        }
    arma::mat result = eig.vectors * (divided % rotated) * eig.vectors.t();
    return 0.5*(result+result.t());
}

void propagate_dressing_reverse(const Inputs& x, const Spectrum& eo, const Spectrum& ev,
                       double beta, double shift, const arma::mat& O,
                       const arma::mat& V, arma::mat& bar_A, Result& out) {
    const arma::uword no = x.foo.n_rows, nv = x.fvv.n_rows;
    arma::mat bar_O(no,no,arma::fill::zeros), bar_V(nv,nv,arma::fill::zeros);
    int worker_count = 1;
    #ifdef _OPENMP
    worker_count = static_cast<int>(std::min<arma::uword>(x.B.n_rows,
                                     worker_limit()));
    #endif
    // Only Fock-sized adjoints are private to workers.  Allocate before the
    // parallel region so allocation errors cannot escape an OpenMP block.
    {
        std::vector<arma::mat> local_O(worker_count), local_V(worker_count);
        std::vector<std::exception_ptr> worker_errors(worker_count);
        for (int t = 0; t < worker_count; ++t) {
            local_O[t].zeros(no,no);
            local_V[t].zeros(nv,nv);
        }
        #pragma omp parallel for schedule(static) num_threads(worker_count)
        for (std::int64_t P = 0; P < static_cast<std::int64_t>(x.B.n_rows); ++P) {
            int thread = 0;
            #ifdef _OPENMP
            thread = omp_get_thread_num();
            #endif
            if (worker_errors[thread]) continue;
            try {
                const arma::mat Bp = orbital_matrix(x.B,P,no,nv);
                const arma::mat Gp = orbital_matrix(bar_A,P,no,nv);
                local_O[thread] += Gp*(Bp*V).t();
                local_V[thread] += (O*Bp).t()*Gp;
            } catch (...) {
                worker_errors[thread] = std::current_exception();
            }
        }
        // A fixed reduction order also makes repeats at a given thread count
        // deterministic.  No tensor-sized gradient replicas are required.
        for (int t = 0; t < worker_count; ++t) {
            if (worker_errors[t]) std::rethrow_exception(worker_errors[t]);
            bar_O += local_O[t];
            bar_V += local_V[t];
        }
    }
    out.foo += exponential_vjp(eo,beta,shift,bar_O);
    out.fvv += exponential_vjp(ev,-beta,shift,bar_V);

    // The caller has finished using this point's dressed-integral adjoint.
    // Consume its buffer for O^T Gamma V^T, with a bounded occupied GEMM
    // scratch and one P-by-a alias temporary per virtual-transform worker.
    transform_occupied_in_place(bar_A,no,O);
    std::exception_ptr reverse_error;
    #pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(no); ++i) {
        try {
            arma::mat gamma_pa(bar_A.colptr(i*nv), x.B.n_rows, nv, false, true);
            gamma_pa = gamma_pa*V.t();
            out.B.cols(i*nv,(i+1)*nv-1) += gamma_pa;
        } catch (...) {
            capture_transform_error(reverse_error);
        }
    }
    if (reverse_error) std::rethrow_exception(reverse_error);
}
}}  // namespace pyscfad::dlno_stc
