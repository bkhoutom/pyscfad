#include "stc_mp2_helpers.h"
#include <algorithm>
#include <numeric>

namespace pyscfad { namespace dlno_stc {
namespace {
enum class Scope { domain, system };
// Occupied workers own bar_A_k/bar_Z_k. Only the one-target U/C bars
// require small private accumulators; no worker duplicates a complete B bar.
void contract_domain_exact(const arma::mat& A,const arma::mat& U,
                           const arma::mat& C,const arma::mat& Z,
                           const KeepSets& keep,double weight,arma::uword block,
                           bool with_grad,arma::mat& bar_A,arma::mat& bar_U,
                           arma::mat& bar_C,arma::mat& bar_Z,Result& out) {
    const arma::uword np=A.n_rows,nv=U.n_cols,no=keep.size();
    const auto owners=std::min<std::size_t>(no,worker_limit());
    std::vector<arma::mat> local_U(owners),local_C(owners);
    if (with_grad) for (std::size_t owner=0;owner<owners;++owner) {
        local_U[owner].zeros(np,nv); local_C[owner].zeros(np,nv);
    }
    std::vector<double> energy(no,0);
    while (block>1 && 8.0*(10.0*np*block+24.0*block*block)>64.0*1024*1024)
        block=std::max<arma::uword>(1,block/2);
    parallel_for(owners,[&](std::size_t owner) {
        for (std::size_t k=owner;k<no;k+=owners) {
            const auto m=keep[k].size();
            if (!m) continue;
            if (8.0*(5.0*np*m+6.0*m*m)<=64.0*1024*1024) {
                const arma::uvec va(keep[k]),oa=k*nv+va;
                const arma::mat Ua=U.cols(va),Ca=C.cols(va),Ab=A.cols(oa),Zb=Z.cols(oa);
                const arma::mat H=Ua.t()*Ab,J=Ca.t()*Zb;
                const arma::mat bar_H=weight*(J.t()-2*J);
                energy[k]+=arma::accu(H%bar_H);
                if (with_grad) {
                    const arma::mat bar_J=weight*(H.t()-2*H);
                    local_U[owner].cols(va)+=Ab*bar_H.t();
                    bar_A.cols(oa)+=Ua*bar_H;
                    local_C[owner].cols(va)+=Zb*bar_J.t();
                    bar_Z.cols(oa)+=Ca*bar_J;
                }
                continue;
            }
            for (std::size_t a=0;a<m;a+=block) for (std::size_t b=0;b<m;b+=block) {
                const auto ae=std::min<std::size_t>(m,a+block),be=std::min<std::size_t>(m,b+block);
                const arma::uvec va(std::vector<arma::uword>(keep[k].begin()+a,keep[k].begin()+ae));
                const arma::uvec vb(std::vector<arma::uword>(keep[k].begin()+b,keep[k].begin()+be));
                const arma::uvec oa=k*nv+va,ob=k*nv+vb;
                const arma::mat Ua=U.cols(va),Ca=C.cols(va),Ab=A.cols(ob),Zb=Z.cols(ob);
                const arma::mat H=Ua.t()*Ab,J=Ca.t()*Zb;
                const arma::mat Jswap=Z.cols(oa).t()*C.cols(vb);
                const arma::mat bar_H=weight*(Jswap-2*J);
                energy[k]+=arma::accu(H%bar_H);
                if (with_grad) {
                    const arma::mat Hswap=A.cols(oa).t()*U.cols(vb);
                    const arma::mat bar_J=weight*(Hswap-2*H);
                    local_U[owner].cols(va)+=Ab*bar_H.t();
                    bar_A.cols(ob)+=Ua*bar_H;
                    local_C[owner].cols(va)+=Zb*bar_J.t();
                    bar_Z.cols(ob)+=Ca*bar_J;
                }
            }
        }
    });
    out.energy+=std::accumulate(energy.begin(),energy.end(),0.0);
    if (with_grad) for (std::size_t owner=0;owner<owners;++owner) {
        bar_U+=local_U[owner]; bar_C+=local_C[owner];
    }
}
Result solve_numerical(const Inputs& x, const Controls& requested_controls, bool with_grad, Scope scope) {
    Controls controls=requested_controls;
    const bool domain=scope==Scope::domain;
    const arma::uword no = x.foo.n_rows, nv = x.fvv.n_rows, np = x.B.n_rows;
    Result out;
    if (with_grad) {
        out.foo.zeros(no,no); out.fvv.zeros(nv,nv); out.B.zeros(np,no*nv);
        if (domain) { out.target_projection.zeros(1,no); out.partner_weight.zeros(no,no); }
    }
    if (nv == 0 || np == 0) return out;
    const Spectrum eo = spectrum(x.foo), ev = spectrum(x.fvv);
    if (!(ev.values.min() > eo.values.max()))
        throw std::invalid_argument("STC-MP2 requires a positive occupied-to-virtual gap");
    if (controls.adaptive && !domain)
        controls.point_variance_targets=full_variance_targets(controls,eo,ev);
    if (controls.adaptive) {
        for (std::size_t l=0;l<controls.roots.size();++l)
            out.point_variance_targets.push_back(point_variance_target(controls,l));
    }
    const double shift = 0.5*eo.values.max()+0.5*ev.values.min();
    arma::mat C, Z, bar_C, bar_Z;
    if (domain) {
        C = project_target(x.B,x.target_projection,nv);
        Z.zeros(np,no*nv);
        parallel_for(no,[&](std::size_t k) {
            for (arma::uword i = 0; i < no; ++i)
                Z.cols(k*nv,(k+1)*nv-1) += x.partner_weight(k,i)*x.B.cols(i*nv,(i+1)*nv-1);
        });
        if (with_grad) { bar_C.zeros(np,nv); bar_Z.zeros(np,no*nv); }
    }
    arma::mat A, bar_A;
    for (std::size_t l = 0; l < controls.roots.size(); ++l) {
        const double beta = controls.roots[l]*(domain ? 1.0:0.5), weight = controls.weights[l];
        const arma::mat O = exponential(eo,beta,shift), V = exponential(ev,-beta,shift);
        dress_integrals(x.B,O,V,A);
        if (with_grad) bar_A.zeros(np,no*nv);
        if (!domain) {
            contract_full(x,controls,l,A,bar_A,with_grad,out);
            if (with_grad) propagate_dressing_reverse(x,eo,ev,beta,shift,O,V,bar_A,out);
            continue;
        }
        const arma::mat U = project_target(A,x.target_projection,nv);
        arma::mat bar_U;
        if (with_grad) bar_U.zeros(np,nv);
        const arma::uword block = controls.virtual_block_size;
        KeepSets keep;
        if (controls.stochastic) keep=select_keep(A,U,C,Z,no,controls);
        else {
            std::vector<arma::uword> all(nv); std::iota(all.begin(),all.end(),0);
            keep.assign(no,all);
        }
        contract_domain_exact(A,U,C,Z,keep,weight,block,with_grad,
                              bar_A,bar_U,bar_C,bar_Z,out);
        if (controls.stochastic)
            sample_residuals(x,controls,l,keep,A,U,C,Z,bar_A,bar_U,bar_C,bar_Z,with_grad,out);
        if (with_grad) {
            parallel_for(no,[&](std::size_t i) {
                out.target_projection(0,i) += arma::accu(bar_U % A.cols(i*nv,(i+1)*nv-1));
                bar_A.cols(i*nv,(i+1)*nv-1) += x.target_projection(0,i)*bar_U;
            });
            propagate_dressing_reverse(x,eo,ev,beta,shift,O,V,bar_A,out);
        }
    }
    if (with_grad && domain) {
        parallel_for(no,[&](std::size_t i) {
            out.target_projection(0,i) += arma::accu(bar_C % x.B.cols(i*nv,(i+1)*nv-1));
            out.B.cols(i*nv,(i+1)*nv-1) += x.target_projection(0,i)*bar_C;
            for (arma::uword k = 0; k < no; ++k) {
                const auto G = bar_Z.cols(k*nv,(k+1)*nv-1);
                out.B.cols(i*nv,(i+1)*nv-1) += x.partner_weight(k,i)*G;
                out.partner_weight(k,i) += arma::accu(G % x.B.cols(i*nv,(i+1)*nv-1));
            }
        });
        out.partner_weight = 0.5*(out.partner_weight+out.partner_weight.t());
    }
    double variance=0;
    for (const auto& residual:out.residuals) variance+=residual.variance_of_mean;
    out.energy_standard_error=std::sqrt(variance);
    return out;
}
}  // namespace
Result solve(const Inputs& x, const Controls& controls, bool with_grad) {
    return solve_numerical(x,controls,with_grad,Scope::domain);
}
Result solve_full(const Inputs& x, const Controls& controls, bool with_grad) {
    return solve_numerical(x,controls,with_grad,Scope::system);
}
}}  // namespace pyscfad::dlno_stc
