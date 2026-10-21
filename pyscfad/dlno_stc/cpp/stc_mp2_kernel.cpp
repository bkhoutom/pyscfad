#include "stc_mp2_helpers.h"
#include "stc_mp2_matrices.h"
#include <algorithm>
#include <numeric>

namespace pyscfad { namespace dlno_stc {
namespace {
bool strictly_diagonal(const arma::mat& matrix) {
    for (arma::uword i=0;i<matrix.n_rows;++i)
        for (arma::uword j=0;j<matrix.n_cols;++j)
            if (i!=j && matrix(i,j)!=0.0) return false;
    return true;
}
Spectrum diagonal_spectrum(const arma::mat& matrix) {
    Spectrum result;
    result.values=matrix.diag();
    return result;
}
void dress_canonical_integrals(const arma::mat& B,const arma::vec& occupied,
                               const arma::vec& virtuals,arma::mat& dressed) {
    const arma::uword no=occupied.n_elem,nv=virtuals.n_elem;
    dressed.set_size(B.n_rows,no*nv);
    parallel_for(no,[&](std::size_t i) {
        for (arma::uword a=0;a<nv;++a)
            dressed.col(i*nv+a)=B.col(i*nv+a)*(occupied(i)*virtuals(a));
    });
}
void propagate_canonical_reverse(const arma::mat& A,const arma::mat& bar_A,
                                 const arma::vec& occupied,const arma::vec& virtuals,
                                 double beta,Result& out) {
    const arma::uword no=occupied.n_elem,nv=virtuals.n_elem;
    arma::mat sums(no,nv);
    parallel_for(no,[&](std::size_t i) {
        double occupied_sum=0;
        for (arma::uword a=0;a<nv;++a) {
            const arma::uword ia=i*nv+a;
            const double z=arma::dot(bar_A.col(ia),A.col(ia));
            sums(i,a)=z;
            occupied_sum+=z;
            out.B.col(ia)+=bar_A.col(ia)*(occupied(i)*virtuals(a));
        }
        out.foo(i,i)+=beta*occupied_sum;
    });
    for (arma::uword a=0;a<nv;++a)
        out.fvv(a,a)-=beta*arma::accu(sums.col(a));
}
Result solve_numerical(const Inputs& x, const Controls& requested_controls, bool with_grad) {
    Controls controls=requested_controls;
    if (controls.canonical_fock) {
        if (!strictly_diagonal(x.foo) || !strictly_diagonal(x.fvv))
            throw std::invalid_argument("canonical_fock requires exactly diagonal foo and fvv");
    }
    const arma::uword no = x.foo.n_rows, nv = x.fvv.n_rows, np = x.B.n_rows;
    Result out;
    if (with_grad) {
        out.foo.zeros(no,no); out.fvv.zeros(nv,nv); out.B.zeros(np,no*nv);
    }
    if (nv == 0 || np == 0) return out;
    const Spectrum eo = controls.canonical_fock ? diagonal_spectrum(x.foo):spectrum(x.foo);
    const Spectrum ev = controls.canonical_fock ? diagonal_spectrum(x.fvv):spectrum(x.fvv);
    if (!(ev.values.min() > eo.values.max()))
        throw std::invalid_argument("STC-MP2 requires a positive occupied-to-virtual gap");
    if (controls.adaptive)
        controls.point_variance_targets=full_variance_targets(controls,eo,ev);
    if (controls.adaptive) {
        for (std::size_t l=0;l<controls.roots.size();++l)
            out.point_variance_targets.push_back(point_variance_target(controls,l));
    }
    const double shift = 0.5*eo.values.max()+0.5*ev.values.min();
    arma::mat A, bar_A;
    for (std::size_t l = 0; l < controls.roots.size(); ++l) {
        const double beta = controls.roots[l]*0.5, weight = controls.weights[l];
        if (controls.canonical_fock) {
            // The common shift cancels in each occupied/virtual product and
            // keeps both exponentials bounded by one for a positive gap.
            const arma::vec occupied=arma::exp(beta*(eo.values-shift));
            const arma::vec virtuals=arma::exp(-beta*(ev.values-shift));
            dress_canonical_integrals(x.B,occupied,virtuals,A);
            if (with_grad) bar_A.zeros(np,no*nv);
            contract_full(x,controls,l,A,bar_A,with_grad,out);
            if (with_grad)
                propagate_canonical_reverse(A,bar_A,occupied,virtuals,beta,out);
            continue;
        }
        const arma::mat O = exponential(eo,beta,shift), V = exponential(ev,-beta,shift);
        dress_integrals(x.B,O,V,A);
        if (with_grad) bar_A.zeros(np,no*nv);
        contract_full(x,controls,l,A,bar_A,with_grad,out);
        if (with_grad) propagate_dressing_reverse(x,eo,ev,beta,shift,O,V,bar_A,out);
    }
    double variance=0;
    for (const auto& residual:out.residuals) variance+=residual.variance_of_mean;
    out.energy_standard_error=std::sqrt(variance);
    return out;
}
}  // namespace
Result solve_full(const Inputs& x, const Controls& controls, bool with_grad) {
    if(!controls.stochastic) {
        std::vector<arma::uword> targets(x.foo.n_rows);
        std::iota(targets.begin(),targets.end(),0);
        return solve_deterministic_energy(x,targets,controls,with_grad);
    }
    return solve_numerical(x,controls,with_grad);
}
}}  // namespace pyscfad::dlno_stc
