#include "native/stc_mp2_deterministic_d_w.h"
#include <iostream>
#include <iomanip>

using namespace libgmbpt::qchem::stc_mp2::density;

namespace {
double largest_error=0.,largest_fd_error=0.;
arma::mat symmetric(const arma::mat& x) { return .5*(x+x.t()); }
arma::mat orthogonal(size_t n) {
    arma::mat q,r;
    arma::qr_econ(q,r,arma::randn(n,n));
    return q;
}
void close(const arma::mat& a,const arma::mat& b,const char* label,double tol=2e-11) {
    if (!a.is_finite() || !b.is_finite()) throw std::runtime_error(label);
    const double error=arma::norm(a-b,"fro")/(1.+arma::norm(b,"fro"));
    largest_error=std::max(largest_error,error);
    if (error>tol) throw std::runtime_error(std::string(label)+": "+std::to_string(error));
}
struct Fixture {
    arma::mat fo,fv,b,u;
    arma::vec beta={.03,.17,.6,1.4},weights={.08,.21,.5,.9};
    Fixture(int spectrum,size_t no=4,size_t nv=6,size_t np=9) {
        const arma::mat qo=orthogonal(no);
        fo=qo*arma::diagmat(arma::linspace(-1.3,-.4,no))*qo.t();
        if (spectrum==3) fo=-.7*arma::eye(no,no);
        u=orthogonal(nv);
        arma::vec ev=arma::linspace(.3,1.6,nv);
        if (spectrum==1 || spectrum==2) {
            ev(1)=ev(0)+(spectrum==2 ? 1e-13 : 0.);
            ev(3)=ev(2);
        }
        if (spectrum==3) ev.fill(.5);
        fv=u*arma::diagmat(ev)*u.t();
        b=.15*arma::randn(np,no*nv);
    }
    deterministic_result run(const std::vector<size_t>& group,int threads,
            const deterministic_w_cotangent_seed* seed,bool optimized,bool forward=true) const {
        if (!optimized) return evaluate_deterministic(fo,fv,b,beta,weights,group,threads,
            nullptr,forward,false,seed,false);
        arma::mat owned=b;
        const double* allocation=owned.memptr();
        auto answer=evaluate_deterministic(fo,fv,owned,beta,weights,group,threads,
            nullptr,forward,false,seed,true,&owned);
        if (!seed && owned.memptr()!=allocation)
            throw std::runtime_error("W replaced the private input DF allocation");
        if (seed && !owned.is_empty())
            throw std::runtime_error("W reverse retained the consumed input DF");
        return answer;
    }
};
double objective(const Fixture& f,const std::vector<size_t>& group,
        const deterministic_w_cotangent_seed& seed) {
    const auto x=f.run(group,1,nullptr,true);
    return seed.energy*x.energy+arma::accu(seed.occupied%x.energy_occupied)
        +arma::accu(seed.virtuals%x.energy_virtuals);
}
void finite_difference(const Fixture& f,const std::vector<size_t>& group,
        const deterministic_w_cotangent_seed& seed,const deterministic_result& bars) {
    const double h=2e-5;
    for (int input=0;input<4;++input) {
        Fixture plus=f,minus=f;
        arma::mat direction;
        double analytic=0.;
        if (input==0) {
            direction=symmetric(arma::randn(f.fo.n_rows,f.fo.n_cols));
            direction/=arma::norm(direction,"fro");
            plus.fo+=h*direction; minus.fo-=h*direction;
            analytic=arma::accu(bars.bar_fo%direction);
        } else if (input==1 || input==3) {
            direction=symmetric(arma::randn(f.fv.n_rows,f.fv.n_cols));
            if (input==3) {
                // Off-diagonal perturbation inside a repeated eigenspace:
                // eigenvalue-only differentiation incorrectly drops this.
                direction=f.u.col(0)*f.u.col(1).t()+f.u.col(1)*f.u.col(0).t();
            }
            direction/=arma::norm(direction,"fro");
            plus.fv+=h*direction; minus.fv-=h*direction;
            analytic=arma::accu(bars.bar_fv%direction);
        } else {
            direction=arma::randn(f.b.n_rows,f.b.n_cols);
            direction/=arma::norm(direction,"fro");
            plus.b+=h*direction; minus.b-=h*direction;
            analytic=arma::accu(bars.bar_b%direction);
        }
        const double numerical=(objective(plus,group,seed)-objective(minus,group,seed))/(2*h);
        const double error=std::abs(numerical-analytic)/(1.+std::abs(analytic));
        largest_fd_error=std::max(largest_fd_error,error);
        if (error>2e-8) throw std::runtime_error("W directional finite difference failed");
    }
}
void validate() {
    arma::arma_rng::set_seed(20261113);
    for (int spectrum=0;spectrum<4;++spectrum) {
        const Fixture f(spectrum);
        for (const auto& group:std::vector<std::vector<size_t>>{{2},{3,1},{0,1,2,3}}) {
            for (int kind=0;kind<5;++kind) {
                arma::mat wo=arma::randn(4,4),wv=arma::randn(6,6);
                if (kind==1 || kind==3 || kind==4) wo.zeros();
                if (kind==1 || kind==2 || kind==4) wv.zeros();
                const deterministic_w_cotangent_seed seed(kind==0 || kind==1 ? .7 : 0.,wo,wv);
                const auto ref=f.run(group,1,&seed,false);
                for (int dynamic:{0,1}) {
                    omp_set_dynamic(dynamic);
                    for (int threads:{1,4}) {
                        const auto value=f.run(group,threads,nullptr,true);
                        const auto bars=f.run(group,threads,&seed,true,false);
                        close(value.energy_occupied,ref.energy_occupied,"occupied W");
                        close(value.energy_virtuals,ref.energy_virtuals,"virtual W");
                        close(arma::mat{value.energy},arma::mat{ref.energy},"energy");
                        close(arma::mat{arma::trace(value.energy_virtuals)},arma::mat{value.energy},"trace Wv");
                        close(bars.bar_fo,ref.bar_fo,"occupied Fock VJP");
                        close(bars.bar_fv,ref.bar_fv,"virtual Fock VJP");
                        close(bars.bar_b,ref.bar_b,"DF VJP");
                        if (dynamic==0 && threads==1 && kind==0)
                            finite_difference(f,group,seed,bars);
                    }
                }
            }
        }
    }
    // Minimal dimensions also exercise empty spin storage on the W path.
    const Fixture tiny(0,1,1,2);
    const deterministic_w_cotangent_seed seed(.7,arma::ones(1,1),arma::ones(1,1));
    const auto a=tiny.run({0},4,&seed,true),b=tiny.run({0},1,&seed,false);
    close(a.energy_virtuals,b.energy_virtuals,"scalar W");
    close(a.bar_fv,b.bar_fv,"scalar Fock VJP");
    close(a.bar_b,b.bar_b,"scalar DF VJP");
    // Two- and four-column occupied-W panels, including partial last panels.
    for (size_t no:{size_t(11),size_t(17)}) {
        const Fixture panel_case(0,no,7,13);
        const deterministic_w_cotangent_seed panel_seed(.4,
            symmetric(arma::randn(no,no)),symmetric(arma::randn(7,7)));
        const std::vector<size_t> targets={0,no-1};
        const auto reference=panel_case.run(targets,1,&panel_seed,false);
        const auto forward=panel_case.run(targets,4,nullptr,true);
        const auto reverse=panel_case.run(targets,4,&panel_seed,true,false);
        close(forward.energy_occupied,reference.energy_occupied,"panel occupied W");
        close(forward.energy_virtuals,reference.energy_virtuals,"panel virtual W");
        close(reverse.bar_fo,reference.bar_fo,"panel occupied Fock VJP");
        close(reverse.bar_fv,reference.bar_fv,"panel virtual Fock VJP");
        close(reverse.bar_b,reference.bar_b,"panel DF VJP");
        finite_difference(panel_case,targets,panel_seed,reverse);
    }
    // The final rotation must reuse a dead factor allocation, even after the
    // cached eigenbasis factors have been released. Check both memory reuse
    // and numerical equivalence to a complete forward/backward frame change.
    const Fixture workspace(0);
    propagator dress(workspace.fo,workspace.fv,workspace.b,workspace.beta,4,true);
    const arma::mat rotated=dress.factors();
    arma::mat owned=workspace.b;
    const double* owned_allocation=owned.memptr();
    propagator inplace(workspace.fo,workspace.fv,owned,workspace.beta,4,true,&owned);
    if (inplace.factors().memptr()!=owned_allocation)
        throw std::runtime_error("In-place DF transform allocated a duplicate factor");
    close(inplace.factors(),rotated,"In-place virtual DF transform");
    arma::mat scratch(workspace.b.n_rows,workspace.b.n_cols);
    const double* allocation=scratch.memptr();
    dress.release_eigenbasis_factors();
    if (!dress.factors().is_empty()) throw std::runtime_error("DF cache was not released");
    dress.rotate_factors(rotated,scratch,false);
    if (scratch.memptr()!=allocation) throw std::runtime_error("DF rotation allocated a new factor");
    close(scratch,workspace.b,"DF rotation into reused scratch");
    std::cout<<std::setprecision(6)<<"W reference/VJP max relative error: "<<largest_error
             <<"; directional finite-difference max error: "<<largest_fd_error<<'\n';
}
void benchmark() {
    arma::arma_rng::set_seed(71);
    omp_set_dynamic(0);
    const Fixture f(0,12,192,320);
    const deterministic_w_cotangent_seed seed(.7,symmetric(arma::randn(12,12)),
        symmetric(arma::randn(192,192)));
    for (bool optimized:{false,true}) {
        auto forward=f.run({1,5},4,nullptr,optimized);
        auto reverse=f.run({1,5},4,&seed,optimized,false);
        std::cout<<(optimized ? "optimized" : "reference")<<" forward="<<forward.seconds
            <<" reverse="<<reverse.seconds<<" propagation="<<forward.propagation_seconds
            <<" contraction="<<forward.contraction_seconds
            <<" propagator_vjp="<<reverse.propagator_vjp_seconds<<'\n';
    }
}
}
int main(int argc,char**) {
    try {
        if (argc>1) benchmark(); else validate();
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
