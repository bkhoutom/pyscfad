#include "stc_triples.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace libgmbpt { namespace qchem {

namespace {
// Alias-sample a row/block pair, then scan the selected final-index block.
// Store only block sums and alias metadata; read element weights from the
// original integral/amplitude tensor. No full conditional table is allocated.
class triples_block_sampler {
    size_t last_, block_size_, nblocks_;
    std::vector<double> mass_, probability_;
    std::vector<size_t> alias_;
public:
    double total;
    triples_block_sampler(size_t rows, size_t last, size_t block_size) :
        last_(last), block_size_(block_size), nblocks_((last-1)/block_size+1),
        mass_(rows*nblocks_,0.0), probability_(rows*nblocks_,1.0),
        alias_(rows*nblocks_), total(0.0) { }
    // Distinct rows may be populated concurrently; finalize after the barrier.
    void set_mass(size_t row,size_t block,double value) {
        mass_[row*nblocks_+block]=value;
    }
    template<typename Value>
    triples_block_sampler(size_t rows, size_t last, size_t block_size, Value value) :
        triples_block_sampler(rows,last,block_size) {
        for(size_t row=0;row<rows;++row) for(size_t b=0;b<nblocks_;++b) {
            double sum=0;
            for(size_t k=b*block_size_;k<std::min(last_,(b+1)*block_size_);++k) {
                const double v=value(row*last_+k); sum+=v*v;
            }
            mass_[row*nblocks_+b]=sum;
        }
        finalize();
    }
    void finalize() {
        // Preserve the original row/block summation order, independently of
        // the schedule used to populate masses.
        total=std::accumulate(mass_.begin(),mass_.end(),0.0);
        if(!std::isfinite(total)) throw std::runtime_error("(T): block weights overflowed");
        if(total==0) return; // A zero-weight channel is never sampled.
        const size_t count=mass_.size();
        std::vector<size_t> small,large;
        for(size_t k=0;k<count;++k) {
            alias_[k]=k;
            probability_[k]=(mass_[k]/total)*double(count);
            (probability_[k]<1 ? small : large).push_back(k);
        }
        while(!small.empty() && !large.empty()) {
            const size_t a=small.back(); small.pop_back();
            const size_t b=large.back(); large.pop_back();
            alias_[a]=b;
            probability_[b]=(probability_[b]+probability_[a])-1;
            (probability_[b]<1 ? small : large).push_back(b);
        }
        for(size_t k:small) probability_[k]=1;
        for(size_t k:large) probability_[k]=1;
    }
    template<typename Value>
    size_t draw(std::mt19937_64& rng, Value value) const {
        std::uniform_int_distribution<size_t> integer(0,mass_.size()-1);
        std::uniform_real_distribution<double> uniform(0,1);
        const size_t trial=integer(rng);
        const size_t selected=uniform(rng)<probability_[trial] ? trial : alias_[trial];
        const size_t row=selected/nblocks_, b=selected%nblocks_;
        double remaining=uniform(rng)*mass_[selected];
        size_t last_positive=row*last_+b*block_size_;
        for(size_t k=b*block_size_;k<std::min(last_,(b+1)*block_size_);++k) {
            const size_t index=row*last_+k;
            const double v=value(index), weight=v*v;
            if(weight>0) {
                last_positive=index;
                if(remaining<weight) return index;
            }
            remaining-=weight;
        }
        return last_positive; // Roundoff at the upper end of the block CDF.
    }
};

void pullback_df_bars(stc_triples::triples_vjp& bar,
                     const arma::mat& Bvo,const arma::mat& Boo,
                     const arma::mat& Bvv) {
    bar.Bvv=Bvo*bar.ovvv.t();
    bar.Boo=Bvo*bar.ovoo;
    bar.Bvo=Bvv*bar.ovvv+Boo*bar.ovoo.t()+Bvo*(bar.ovov+bar.ovov.t());
}
}

// Closed-shell spatial-orbital (T), following stc_qc_paper/stc_cc/pt.py.
// The permutation sum is unnormalized; the factor 2 and singles factor 1/2
// are part of this convention (there is no additional spin-orbital 1/36).

double stc_triples::projected_triples_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::vec& u, const arma::mat* fov) {
    return projected_triples_integrals(t1,t2,ovvv,ovoo,ovov,eocc,evir,
        arma::mat(u),fov);
}

double stc_triples::projected_triples_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::mat& u, const arma::mat* fov) {
    const size_t no=eocc.n_elem, nv=evir.n_elem;
    if(fov && (fov->n_rows!=no || fov->n_cols!=nv || !fov->is_finite()))
        throw std::invalid_argument("Projected (T): invalid Fov dimensions or values");
    if(u.n_rows!=no || u.n_cols==0 || u.n_cols>no || !u.is_finite() ||
       t1.n_elem!=no ||
       t2.n_rows!=no || t2.n_cols!=no ||
       ovvv.n_rows!=nv*nv || ovvv.n_cols!=no*nv ||
       ovoo.n_rows!=no*nv || ovoo.n_cols!=no*no ||
       ovov.n_rows!=no*nv || ovov.n_cols!=no*nv ||
       !eocc.is_finite() || !evir.is_finite())
        throw std::invalid_argument("Projected (T): inconsistent input dimensions");
    if(arma::norm(u.t()*u-arma::eye<arma::mat>(u.n_cols,u.n_cols),"fro")>1e-8)
        throw std::invalid_argument("Projected (T): selected orbitals must be orthonormal");
    if(no==0 || nv==0) return 0.;
    if(evir.min()<=eocc.max())
        throw std::invalid_argument("Projected (T): requires a positive orbital gap");
    for(size_t i=0;i<no;++i) {
        if(t1(i).n_elem!=nv) throw std::invalid_argument("Projected (T): singles dimensions");
        for(size_t j=0;j<no;++j)
            if(t2(i,j).n_rows!=nv || t2(i,j).n_cols!=nv)
                throw std::invalid_argument("Projected (T): doubles dimensions");
    }
    arma::field<arma::mat> t2_virtual(no,no);
    #pragma omp parallel for collapse(2)
    for(size_t i=0;i<no;++i) for(size_t j=0;j<no;++j)
        t2_virtual(i,j)=t2(i,j).t();
    const size_t perm[6][3]={{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
    const double rweight[6]={4,1,1,-2,-2,-2};
    typedef std::array<size_t,6> index_type;
    std::vector<size_t> active_rows;
    for(size_t i=0;i<no;++i) {
        for(size_t s=0;s<u.n_cols;++s) if(u(i,s)!=0.) {
            active_rows.push_back(i);
            break;
        }
    }
    auto pair_permute=[&](const index_type& x,size_t p) {
        return index_type{{x[perm[p][0]],x[perm[p][1]],x[perm[p][2]],
                           x[3+perm[p][0]],x[3+perm[p][1]],x[3+perm[p][2]]}};
    };
    auto w=[&](const index_type& x) {
        const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5];
        double value=0.;
        const double* integral=ovvv.colptr(i*nv+a)+b*nv;
        const double* amplitude=t2_virtual(k,j).colptr(c);
        for(size_t f=0;f<nv;++f) value+=integral[f]*amplitude[f];
        for(size_t m=0;m<no;++m) value-=ovoo(i*nv+a,j*no+m)*t2(m,k)(b,c);
        return value;
    };
    auto rz=[&](const index_type& x,double identity_w) {
        double value=0.;
        for(size_t p=0;p<6;++p) {
            index_type y=x;
            for(size_t d=0;d<3;++d) y[d]=x[perm[p][d]];
            const double v=ovov(y[0]*nv+y[3],y[1]*nv+y[4])*t1(y[2])(y[5])+
                (fov ? (*fov)(y[2],y[5])*t2(y[0],y[1])(y[3],y[4]) : 0.0);
            value+=rweight[p]*((p==0 ? identity_w : w(y))+0.5*v);
        }
        return value;
    };
    double energy=0.;
    #pragma omp parallel reduction(+:energy)
    {
    std::vector<double> projected_a(u.n_cols),projected_z(u.n_cols);
    #pragma omp for collapse(3) schedule(dynamic)
    for(size_t j=0;j<no;++j)
        for(size_t k=0;k<no;++k)
            for(size_t a=0;a<nv;++a)
                for(size_t b=0;b<nv;++b)
                    for(size_t c=0;c<nv;++c) {
                        std::fill(projected_a.begin(),projected_a.end(),0.);
                        std::fill(projected_z.begin(),projected_z.end(),0.);
                        for(size_t i:active_rows) {
                            const index_type x{{i,j,k,a,b,c}};
                            double pw=0.,prz=0.;
                            for(size_t p=0;p<6;++p) {
                                const auto y=pair_permute(x,p);
                                const double wy=w(y);
                                pw+=wy;
                                prz+=rz(y,wy);
                            }
                            const double d=eocc(i)+eocc(j)+eocc(k)-evir(a)-evir(b)-evir(c);
                            for(size_t s=0;s<u.n_cols;++s) {
                                projected_a[s]+=u(i,s)*pw;
                                projected_z[s]+=u(i,s)*prz/d;
                            }
                        }
                        for(size_t s=0;s<u.n_cols;++s)
                            energy+=projected_a[s]*projected_z[s]/3.;
                    }
    }
    return energy;
}

stc_triples::triples_result stc_triples::projected_triples_stochastic_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::vec& occupied_projector, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    return evaluate_integrals(t1,t2,ovvv,ovoo,ovov,eocc,evir,
        nsamples,seed,target_standard_error,vjp,cotangent,&occupied_projector,nullptr,fov);
}

stc_triples::triples_result stc_triples::projected_triples_stochastic_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::mat& occupied_subspace, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    return evaluate_integrals(t1,t2,ovvv,ovoo,ovov,eocc,evir,
        nsamples,seed,target_standard_error,vjp,cotangent,nullptr,&occupied_subspace,fov);
}

stc_triples::triples_result stc_triples::whole_triples(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
    const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    const size_t no=eocc.n_elem,nv=evir.n_elem;
    if (Bvo.n_cols!=no*nv || Boo.n_cols!=no*no || Bvv.n_cols!=nv*nv ||
        Bvo.n_rows!=Boo.n_rows || Bvo.n_rows!=Bvv.n_rows)
        throw std::invalid_argument("(T): inconsistent DF tensor dimensions");
    const auto result=whole_triples_integrals(t1,t2,Bvv.t()*Bvo,
        Bvo.t()*Boo,Bvo.t()*Bvo,eocc,evir,nsamples,seed,
        target_standard_error,vjp,cotangent,fov);
    if(vjp) pullback_df_bars(*vjp,Bvo,Boo,Bvv);
    return result;
}

stc_triples::triples_result stc_triples::whole_triples_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    return evaluate_integrals(t1,t2,ovvv,ovoo,ovov,eocc,evir,
        nsamples,seed,target_standard_error,vjp,cotangent,nullptr,nullptr,fov);
}

stc_triples::triples_result stc_triples::projected_triples_stochastic(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::vec& occupied_projector, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    const size_t no=eocc.n_elem,nv=evir.n_elem;
    if (Bvo.n_cols!=no*nv || Boo.n_cols!=no*no || Bvv.n_cols!=nv*nv ||
        Bvo.n_rows!=Boo.n_rows || Bvo.n_rows!=Bvv.n_rows)
        throw std::invalid_argument("(T): inconsistent DF tensor dimensions");
    const auto result=projected_triples_stochastic_integrals(t1,t2,
        Bvv.t()*Bvo,Bvo.t()*Boo,Bvo.t()*Bvo,eocc,evir,occupied_projector,
        nsamples,seed,target_standard_error,vjp,cotangent,fov);
    if(vjp) pullback_df_bars(*vjp,Bvo,Boo,Bvv);
    return result;
}

stc_triples::triples_result stc_triples::projected_triples_stochastic(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& Bvo, const arma::mat& Boo, const arma::mat& Bvv,
    const arma::vec& eocc, const arma::vec& evir,
    const arma::mat& occupied_subspace, size_t nsamples,
    std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent, const arma::mat* fov) {
    const size_t no=eocc.n_elem,nv=evir.n_elem;
    if (Bvo.n_cols!=no*nv || Boo.n_cols!=no*no || Bvv.n_cols!=nv*nv ||
        Bvo.n_rows!=Boo.n_rows || Bvo.n_rows!=Bvv.n_rows)
        throw std::invalid_argument("(T): inconsistent DF tensor dimensions");
    const auto result=projected_triples_stochastic_integrals(t1,t2,
        Bvv.t()*Bvo,Bvo.t()*Boo,Bvo.t()*Bvo,eocc,evir,occupied_subspace,
        nsamples,seed,target_standard_error,vjp,cotangent,fov);
    if(vjp) pullback_df_bars(*vjp,Bvo,Boo,Bvv);
    return result;
}

stc_triples::triples_result stc_triples::evaluate_integrals(
    const arma::field<arma::vec>& t1, const arma::field<arma::mat>& t2,
    const arma::mat& ovvv, const arma::mat& ovoo, const arma::mat& ovov,
    const arma::vec& eocc, const arma::vec& evir,
    size_t nsamples, std::uint64_t seed, double target_standard_error,
    triples_vjp* vjp, double cotangent,
    const arma::vec* occupied_projector,
    const arma::mat* occupied_subspace,
    const arma::mat* fov) {
    const size_t no=eocc.n_elem, nv=evir.n_elem;
    if(fov && (fov->n_rows!=no || fov->n_cols!=nv || !fov->is_finite()))
        throw std::invalid_argument("(T): invalid Fov dimensions or values");
    const bool projected=occupied_projector || occupied_subspace;
    if(occupied_projector && occupied_subspace)
        throw std::invalid_argument("(T): choose one selected-projector representation");
    arma::mat selected_orbitals;
    if(projected) {
        selected_orbitals=occupied_subspace ? *occupied_subspace :
            arma::mat(*occupied_projector);
        if(selected_orbitals.n_rows!=no || selected_orbitals.n_cols==0 ||
           selected_orbitals.n_cols>no || !selected_orbitals.is_finite() ||
           arma::norm(selected_orbitals.t()*selected_orbitals-
               arma::eye<arma::mat>(selected_orbitals.n_cols,
                                    selected_orbitals.n_cols),"fro")>1e-8)
            throw std::invalid_argument("(T): selected occupied orbitals must be orthonormal");
    }
    const arma::mat projector=projected ? selected_orbitals*selected_orbitals.t() :
        arma::mat();
    const bool whole_vjp=vjp && !projected;
    const char* vjp_proposal_setting=std::getenv("STC_TRIPLES_VJP_PROPOSAL");
    if(!vjp_proposal_setting)
        vjp_proposal_setting=std::getenv("QCHEM_LCCPLAY_TRIPLES_VJP_PROPOSAL");
    const std::string vjp_proposal=vjp_proposal_setting ? vjp_proposal_setting : "energy";
    if(vjp_proposal!="energy" && vjp_proposal!="mixed")
        throw std::invalid_argument("(T): VJP_PROPOSAL must be energy or mixed");
    const bool gradient_mixture=vjp && !projected && vjp_proposal=="mixed";
    arma::vec sampling_weights(no,arma::fill::ones);
    const double abs_weight_sum=double(no);
    if(vjp && !std::isfinite(cotangent))
        throw std::invalid_argument("(T): VJP requires a finite cotangent");
    if(projected && vjp && nsamples==0)
        throw std::invalid_argument("(T): projected VJP requires stochastic samples");
    if(!std::isfinite(target_standard_error) || target_standard_error<0 ||
        (target_standard_error>0 && nsamples==0))
        throw std::invalid_argument("(T): invalid stochastic standard-error target");
    if (nsamples == 1)
        throw std::invalid_argument("(T): use zero or at least two samples");
    if (t1.n_elem != no || t2.n_rows != no || t2.n_cols != no ||
        ovvv.n_rows != nv*nv || ovvv.n_cols != no*nv ||
        ovoo.n_rows != no*nv || ovoo.n_cols != no*no ||
        ovov.n_rows != no*nv || ovov.n_cols != no*nv)
        throw std::invalid_argument("(T): inconsistent canonical tensor dimensions");
    for (size_t i = 0; i < no; ++i) {
        if (t1(i).n_elem != nv || !t1(i).is_finite())
            throw std::invalid_argument("(T): invalid singles amplitudes");
        for (size_t j = 0; j < no; ++j)
            if (t2(i,j).n_rows != nv || t2(i,j).n_cols != nv || !t2(i,j).is_finite())
                throw std::invalid_argument("(T): invalid doubles amplitudes");
    }
    if (!eocc.is_finite() || !evir.is_finite() || !ovvv.is_finite() ||
        !ovoo.is_finite() || !ovov.is_finite())
        throw std::invalid_argument("(T): nonfinite energies or DF integrals");
    if(vjp) {
        *vjp=triples_vjp();
        vjp->t1.set_size(no); vjp->t2.set_size(no,no);
        for(size_t i=0;i<no;++i) {
            vjp->t1(i).zeros(nv);
            for(size_t j=0;j<no;++j) vjp->t2(i,j).zeros(nv,nv);
        }
        vjp->eocc.zeros(no); vjp->evir.zeros(nv); vjp->fov.zeros(no,nv);
        if(occupied_projector) vjp->occupied_projector.zeros(no);
        if(projected)
            vjp->occupied_subspace.zeros(no,selected_orbitals.n_cols);
        vjp->ovvv.zeros(nv*nv,no*nv); vjp->ovoo.zeros(no*nv,no*no);
        vjp->ovov.zeros(no*nv,no*nv);
    }
    if (no == 0 || nv == 0) return {0.0, 0.0, 0};
    if(projected && nsamples==0) {
        triples_result result={projected_triples_integrals(t1,t2,ovvv,ovoo,ovov,
            eocc,evir,selected_orbitals,fov),0.0,0};
        return result;
    }
    // A positive occupied/virtual gap bounds every denominator away from zero.
    if (evir.min() <= eocc.max())
        throw std::invalid_argument("(T): requires a positive occupied-virtual gap");
    // OVVV is supplied directly as (b*nv+f, i*nv+a): f is contiguous.
    // Keep the original T2 for sampling/occupied contractions and a transpose
    // of each virtual matrix for contiguous virtual contractions.
    const double packing_start=omp_get_wtime();
    arma::field<arma::mat> t2_virtual(no,no);
    for(size_t i=0;i<no;++i) for(size_t j=0;j<no;++j)
        t2_virtual(i,j).set_size(nv,nv);
    #pragma omp parallel for collapse(2)
    for(size_t i=0;i<no;++i) for(size_t j=0;j<no;++j)
        t2_virtual(i,j)=t2(i,j).t();
    const double packing_seconds=omp_get_wtime()-packing_start;
    const size_t perm[6][3] = {{0,1,2}, {1,2,0}, {2,0,1},
                              {0,2,1}, {1,0,2}, {2,1,0}};
    const double rweight[6] = {4, 1, 1, -2, -2, -2};
    // x = (i,j,k,a,b,c). w and q contract f/m exactly for each selected x.
    typedef std::array<size_t, 6> index_type;
    auto pair_permute = [&](const index_type& x, size_t p) {
        return index_type{{x[perm[p][0]], x[perm[p][1]], x[perm[p][2]],
                           x[3+perm[p][0]], x[3+perm[p][1]], x[3+perm[p][2]]}};
    };
    auto w = [&](const index_type& x) {
        const size_t i=x[0], j=x[1], k=x[2], a=x[3], b=x[4], c=x[5];
        double value = 0;
        const double* integrals=ovvv.colptr(i*nv+a)+b*nv;
        const double* amplitudes=t2_virtual(k,j).colptr(c);
        for (size_t f=0; f<nv; ++f) value += integrals[f]*amplitudes[f];
        for (size_t m=0; m<no; ++m) value -= ovoo(i*nv+a,j*no+m)*t2(m,k)(b,c);
        return value;
    };
    // Accumulate the signed contraction and its sampling weight together,
    // sharing every integral/amplitude load and product.
    auto wq = [&](const index_type& x) {
        const size_t i=x[0], j=x[1], k=x[2], a=x[3], b=x[4], c=x[5];
        double signed_sum=0, squared_sum=0, amplitude_square=0, integral_square=0;
        const double* integrals=ovvv.colptr(i*nv+a)+b*nv;
        const double* amplitudes=t2_virtual(k,j).colptr(c);
        for(size_t f=0;f<nv;++f) {
            const double v=integrals[f]*amplitudes[f];
            signed_sum+=v; squared_sum+=v*v;
            if(gradient_mixture) {
                amplitude_square+=amplitudes[f]*amplitudes[f];
                integral_square+=integrals[f]*integrals[f];
            }
        }
        for(size_t m=0;m<no;++m) {
            const double integral=ovoo(i*nv+a,j*no+m);
            const double amplitude=t2(m,k)(b,c);
            const double v=integral*amplitude;
            signed_sum-=v; squared_sum+=v*v;
            if(gradient_mixture) {
                amplitude_square+=amplitude*amplitude;
                integral_square+=integral*integral;
            }
        }
        return std::array<double,4>{{signed_sum,squared_sum,
            amplitude_square,integral_square}};
    };
    auto rz = [&](const index_type& x, double identity_w) {
        double value = 0;
        for (size_t p=0; p<6; ++p) {
            index_type y = x;
            for (size_t d=0; d<3; ++d) y[d] = x[perm[p][d]];
            const double v = ovov(y[0]*nv+y[3],y[1]*nv+y[4])*t1(y[2])(y[5])+
                (fov ? (*fov)(y[2],y[5])*t2(y[0],y[1])(y[3],y[4]) : 0.0);
            // The identity occupied permutation reuses the already computed w(x).
            value += rweight[p]*((p==0 ? identity_w : w(y))+0.5*v);
        }
        return value;
    };
    // Sum the selected-S permutation orbit while loading each integral fiber
    // once for the two terms that share it. The virtual and occupied fibers
    // have different pairings, so keep their partial contractions separately.
    auto rz_orbit_sum = [&](const index_type& x) {
        std::array<index_type,36> z;
        std::array<double,36> virtual_part,occupied_part;
        for(size_t p=0;p<6;++p) {
            const auto y=pair_permute(x,p);
            for(size_t q=0;q<6;++q) {
                auto& term=z[6*p+q];
                term=y;
                for(size_t leg=0;leg<3;++leg)
                    term[leg]=y[perm[q][leg]];
            }
        }
        for(size_t p=0;p<6;++p) for(size_t q=0;q<3;++q) {
            const size_t first=6*p+q,second=first+3;
            const auto& a=z[first];
            const auto& b=z[second];
            const double* integral=ovvv.colptr(a[0]*nv+a[3])+a[4]*nv;
            const double* amplitude_a=t2_virtual(a[2],a[1]).colptr(a[5]);
            const double* amplitude_b=t2_virtual(b[2],b[1]).colptr(b[5]);
            double sum_a=0,sum_b=0;
            for(size_t f=0;f<nv;++f) {
                const double v=integral[f];
                sum_a+=v*amplitude_a[f];
                sum_b+=v*amplitude_b[f];
            }
            virtual_part[first]=sum_a;
            virtual_part[second]=sum_b;
        }
        // p and p+3 exchange the last two pairs. These q partners leave
        // (first occupied, first virtual, second occupied) unchanged.
        static const size_t occupied_partner[6]={3,5,4,0,2,1};
        for(size_t p=0;p<3;++p) for(size_t q=0;q<6;++q) {
            const size_t first=6*p+q,second=6*(p+3)+occupied_partner[q];
            const auto& a=z[first];
            const auto& b=z[second];
            double sum_a=0,sum_b=0;
            for(size_t m=0;m<no;++m) {
                const double v=ovoo(a[0]*nv+a[3],a[1]*no+m);
                sum_a+=v*t2(m,a[2])(a[4],a[5]);
                sum_b+=v*t2(m,b[2])(b[4],b[5]);
            }
            occupied_part[first]=sum_a;
            occupied_part[second]=sum_b;
        }
        double value=0;
        for(size_t p=0;p<6;++p) for(size_t q=0;q<6;++q) {
            const size_t term=6*p+q;
            const auto& a=z[term];
            const double v=ovov(a[0]*nv+a[3],a[1]*nv+a[4])*t1(a[2])(a[5])+
                (fov ? (*fov)(a[2],a[5])*t2(a[0],a[1])(a[3],a[4]) : 0.0);
            value+=rweight[q]*(virtual_part[term]-occupied_part[term]+0.5*v);
        }
        return value;
    };
    auto denominator = [&](const index_type& x) {
        return eocc(x[0])+eocc(x[1])+eocc(x[2])-evir(x[3])-evir(x[4])-evir(x[5]);
    };
    // Deterministic reverse of the physical integrand. The stochastic path
    // below records samples and assigns each adjoint row to one worker.
    // The sampling law and its normalization are stop-gradient quantities.
    auto add_bar=[](double& destination,double value) {
        #pragma omp atomic update
        destination+=value;
    };
    auto backward_w=[&](const index_type& x,double bar) {
        const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5];
        for(size_t f=0;f<nv;++f) {
            add_bar(vjp->ovvv(b*nv+f,i*nv+a),bar*t2(k,j)(c,f));
            add_bar(vjp->t2(k,j)(c,f),bar*ovvv(b*nv+f,i*nv+a));
        }
        for(size_t m=0;m<no;++m) {
            add_bar(vjp->ovoo(i*nv+a,j*no+m),-bar*t2(m,k)(b,c));
            add_bar(vjp->t2(m,k)(b,c),-bar*ovoo(i*nv+a,j*no+m));
        }
    };
    auto backward_v=[&](const index_type& x,double bar) {
        const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5];
        add_bar(vjp->ovov(i*nv+a,j*nv+b),bar*t1(k)(c));
        add_bar(vjp->t1(k)(c),bar*ovov(i*nv+a,j*nv+b));
        add_bar(vjp->fov(k,c),bar*t2(i,j)(a,b));
        if(fov) add_bar(vjp->t2(i,j)(a,b),bar*(*fov)(k,c));
    };
    auto backward=[&](const index_type& x,double pw,double prz,double weight) {
        if(!vjp || cotangent==0) return;
        const double d=denominator(x), factor=cotangent*weight/d;
        const double bar_pw=factor*prz, bar_prz=factor*pw;
        const double bar_d=-factor*pw*prz/d;
        for(size_t leg=0;leg<3;++leg) {
            add_bar(vjp->eocc(x[leg]),bar_d);
            add_bar(vjp->evir(x[leg+3]),-bar_d);
        }
        for(size_t p=0;p<6;++p) {
            const auto y=pair_permute(x,p);
            backward_w(y,bar_pw);
            for(size_t r=0;r<6;++r) {
                auto z=y;
                for(size_t leg=0;leg<3;++leg) z[leg]=y[perm[r][leg]];
                const double bar=bar_prz*rweight[r];
                backward_w(z,bar);
                backward_v(z,0.5*bar);
            }
        }
    };
    if (nsamples == 0) {
        double energy = 0;
        #pragma omp parallel for collapse(3) reduction(+:energy) schedule(dynamic)
        for (size_t i=0; i<no; ++i)
            for (size_t j=0; j<no; ++j)
                for (size_t k=0; k<no; ++k)
                    for (size_t a=0; a<nv; ++a)
                        for (size_t b=0; b<nv; ++b)
                            for (size_t c=0; c<nv; ++c) {
                                const index_type x{{i,j,k,a,b,c}};
                                double pw = 0;
                                for (size_t p=0; p<6; ++p) pw += w(pair_permute(x,p));
                                if(whole_vjp) {
                                    // Sum the symmetrized integrand for the
                                    // fused whole-system VJP.
                                    double prz=0;
                                    for(size_t p=0;p<6;++p) {
                                        const auto y=pair_permute(x,p);
                                        prz+=rz(y,w(y));
                                    }
                                    energy+=pw*prz/(3*denominator(x));
                                    backward(x,pw,prz,1.0/3.0);
                                } else energy += 2*pw*rz(x,w(x))/denominator(x);
                            }
        triples_result result={energy, 0.0, 0};
        result.amplitude_packing_seconds=packing_seconds;
        return result;
    }

    // Same blocked conditional sampling as stc_qc_paper: one occupied
    // block and up to 20 virtual blocks. Handle partial virtual blocks
    // directly instead of padding orbitals. Keep iid marginal draws rather
    // than quota rounding so the standard error remains s/sqrt(N).
    const double probability_start=omp_get_wtime();
    const size_t virtual_block=(nv-1)/20+1;
    auto left_value=[&](size_t h,size_t index) {
        if(h<nv) return ovvv((index%nv)*nv+h,index/(nv*nv)*nv+(index/nv)%nv);
        return ovoo(index/(nv*no)*nv+(index/no)%nv,(index%no)*no+h-nv);
    };
    auto right_value=[&](size_t h,size_t index) {
        if(h<nv) return t2(index/(nv*no),index%no)((index/no)%nv,h);
        return t2(h-nv,index/(nv*nv))(index%nv,(index/nv)%nv);
    };
    std::vector<std::unique_ptr<triples_block_sampler> > left(nv+no),right(nv+no);
    std::vector<double> channel_weights(nv+no);
    std::exception_ptr parallel_error;
    int setup_threads=1, sampling_threads=1;
    #pragma omp parallel
    {
        #pragma omp single
        setup_threads=omp_get_num_threads();
        #pragma omp for schedule(dynamic)
        for(size_t h=0;h<nv+no;++h) {
            try {
                if(h<nv) left[h].reset(new triples_block_sampler(no*nv,nv,virtual_block));
                else left[h].reset(new triples_block_sampler(no*nv,no,no,
                    [&](size_t index) { return left_value(h,index); }));
                right[h].reset(new triples_block_sampler(no*nv,h<nv ? no : nv,h<nv ? no : virtual_block,
                    [&](size_t index) { return right_value(h,index); }));

            } catch(...) {
                #pragma omp critical(lccplay_triples_error)
                { if(!parallel_error) parallel_error=std::current_exception(); }
            }
        }
    }
    if(parallel_error) std::rethrow_exception(parallel_error);
    // Stream the sole OVVV buffer in its physical (ia,b,f) order. A worker
    // owns complete ia rows and accumulates all f channels together, so no
    // atomics or duplicate OVVV storage are needed. Only O(nv*threads) scratch.
    arma::mat mass_scratch(nv+8,omp_get_max_threads()); // separate workers by a cache line
    const size_t virtual_blocks=(nv-1)/virtual_block+1;
    #pragma omp parallel for schedule(static)
    for(size_t row=0;row<no*nv;++row) {
        double* sums=mass_scratch.colptr(omp_get_thread_num());
        const double* integrals=ovvv.colptr(row);
        for(size_t block=0;block<virtual_blocks;++block) {
            std::fill(sums,sums+nv,0.0);
            const size_t end=std::min(nv,(block+1)*virtual_block);
            for(size_t b=block*virtual_block;b<end;++b) {
                const double* values=integrals+b*nv;
                for(size_t f=0;f<nv;++f) sums[f]+=values[f]*values[f];
            }
            for(size_t f=0;f<nv;++f) left[f]->set_mass(row,block,sums[f]);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for(size_t h=0;h<nv+no;++h) {
        try {
            if(h<nv) left[h]->finalize();
            channel_weights[h]=left[h]->total*right[h]->total;
        } catch(...) {
            #pragma omp critical(lccplay_triples_error)
            { if(!parallel_error) parallel_error=std::current_exception(); }
        }
    }
    if(parallel_error) std::rethrow_exception(parallel_error);
    // The whole-system VJP keeps three occupied-leg channels for the
    // symmetrized proposal and optional mixed gradient proposal.
    const size_t nh=nv+no;
    std::vector<std::unique_ptr<triples_block_sampler> > restricted(whole_vjp ? 3*nh : 0);
    auto restrict_left=[&](size_t h,size_t leg) { return leg==0 || (h>=nv && leg==1); };
    auto restricted_value=[&](size_t h,size_t leg,size_t index) {
        return restrict_left(h,leg) ? left_value(h,index) : right_value(h,index);
    };
    if(whole_vjp) {
        channel_weights.resize(3*nh);
        #pragma omp parallel for schedule(dynamic)
        for(size_t channel_id=0;channel_id<3*nh;++channel_id) {
            try {
                const size_t h=channel_id%nh, leg=channel_id/nh;
                const bool on_left=restrict_left(h,leg);
                const size_t last=on_left ? (h<nv ? nv : no) : (h<nv ? no : nv);
                const size_t block=(on_left ? h<nv : h>=nv) ? virtual_block : no;
                restricted[channel_id].reset(new triples_block_sampler(no*nv,last,block,
                    [&](size_t index) { return restricted_value(h,leg,index); }));
                channel_weights[channel_id]=restricted[channel_id]->total*
                    (on_left ? right[h]->total : left[h]->total);
            } catch(...) {
                #pragma omp critical(lccplay_triples_error)
                { if(!parallel_error) parallel_error=std::current_exception(); }
            }
        }
        if(parallel_error) std::rethrow_exception(parallel_error);
    }
    const double norm=std::accumulate(channel_weights.begin(),channel_weights.end(),0.0);
    if (!std::isfinite(norm)) throw std::runtime_error("(T): importance weights overflowed");
    if (norm == 0 && !vjp) return {0.0, 0.0, 0, omp_get_wtime()-probability_start, 0.0, setup_threads, 0, 0, 0.0, packing_seconds};
    // Two gradient-oriented proposal families reuse the existing tables:
    // q_G=sum T2^2 (uniform integral side), q_T=sum G^2 (uniform amplitude
    // side). Each family is normalized independently before mixing.
    std::array<std::vector<double>,2> gradient_channel_weights;
    std::array<double,3> proposal_norms{{norm,0,0}};
    std::array<double,3> proposal_mix{{1,0,0}};
    if(gradient_mixture) {
        gradient_channel_weights[0].resize(3*nh);
        gradient_channel_weights[1].resize(3*nh);
        for(size_t id=0;id<3*nh;++id) {
            const size_t h=id%nh,leg=id/nh;
            const bool on_left=restrict_left(h,leg);
            const double nleft=double(h<nv ? no*nv*nv : no*nv*no);
            const double nright=double(h<nv ? no*nv*no : no*nv*nv);
            gradient_channel_weights[0][id]=nleft*(on_left ?
                right[h]->total : restricted[id]->total);
            gradient_channel_weights[1][id]=nright*(on_left ?
                restricted[id]->total : left[h]->total);
        }
        for(size_t family=1;family<3;++family) {
            proposal_norms[family]=std::accumulate(
                gradient_channel_weights[family-1].begin(),
                gradient_channel_weights[family-1].end(),0.0);
            if(!std::isfinite(proposal_norms[family]))
                throw std::runtime_error("(T): gradient proposal weights overflowed");
        }
        proposal_mix={{0.4,0.3,0.3}};
        double live=0;
        for(size_t family=0;family<3;++family) {
            if(proposal_norms[family]==0) proposal_mix[family]=0;
            live+=proposal_mix[family];
        }
        if(live>0) for(double& fraction:proposal_mix) fraction/=live;
    }
    // Retain a full-support branch for tuples with zero importance mass.
    const double uniform_fraction=vjp ?
        (proposal_norms[0]+proposal_norms[1]+proposal_norms[2]>0 ? 0.05 : 1.0) : 0.0;
    const double uniform_size=double(no)*double(no)*double(nv)*double(nv)*double(nv);
    const double projected_uniform_size=uniform_size*double(no);
    std::discrete_distribution<size_t> uniform_orbital(
        sampling_weights.begin(),sampling_weights.end());
    std::vector<double> projector_masses;
    if(projected) {
        projector_masses.resize(no);
        // Marginalize |P_ji| over the first occupied leg. For one column this
        // is proportional to |u_i|, matching the original rank-one proposal.
        for(size_t i=0;i<no;++i)
            projector_masses[i]=arma::accu(arma::abs(projector.col(i)));
        const double floor=vjp ? 0.05*std::accumulate(projector_masses.begin(),
            projector_masses.end(),0.0)/double(no) : 0.;
        for(double& mass:projector_masses) mass+=floor;
    }
    const double projector_l1=std::accumulate(projector_masses.begin(),
        projector_masses.end(),0.0);
    std::discrete_distribution<size_t> projector_orbital(
        projector_masses.begin(),projector_masses.end());
    if(norm==0) channel_weights[0]=1; // unused placeholder: uniform branch is certain
    std::discrete_distribution<size_t> channel(channel_weights.begin(),channel_weights.end());
    std::array<std::discrete_distribution<size_t>,2> gradient_channels;
    if(gradient_mixture) for(size_t family=0;family<2;++family) {
        if(proposal_norms[family+1]==0) gradient_channel_weights[family][0]=1;
        gradient_channels[family]=std::discrete_distribution<size_t>(
            gradient_channel_weights[family].begin(),
            gradient_channel_weights[family].end());
    }
    std::discrete_distribution<size_t> proposal_family(
        proposal_mix.begin(),proposal_mix.end());
    auto draw_uniform_factor=[&](size_t h,bool on_left,size_t marked_leg,
                                 std::mt19937_64& rng,
                                 std::discrete_distribution<size_t>& orbital_draw) {
        std::uniform_int_distribution<size_t> occ(0,no-1),vir(0,nv-1);
        auto draw_occ=[&](size_t leg) {
            return marked_leg==leg ? orbital_draw(rng) : occ(rng);
        };
        if(h<nv) {
            if(on_left) return (draw_occ(0)*nv+vir(rng))*nv+vir(rng); // i,a,b
            return (draw_occ(2)*nv+vir(rng))*no+draw_occ(1); // k,c,j
        }
        if(on_left) return (draw_occ(0)*nv+vir(rng))*no+draw_occ(1); // i,a,j
        return (draw_occ(2)*nv+vir(rng))*nv+vir(rng); // k,c,b
    };
    const double sampling_start=omp_get_wtime();
    const double probability_seconds=sampling_start-probability_start;
    // Fixed logical batches, each with its own SplitMix64-derived RNG seed.
    // Deterministic batch-order merging makes the result independent of the
    // OpenMP schedule and thread count; no shared RNG or atomic sample updates.
    const size_t batch_size=2048;
    struct moments { size_t count; double mean,m2; };
    struct reverse_sample {
        index_type x;
        double bar_pw,bar_prz,bar_d;
    };
    struct reverse_worker {
        arma::mat t1,fov;
        arma::vec eocc,evir;
        arma::mat occupied_subspace;
    };
    // One shared T2 adjoint in the same transposed virtual layout as the
    // forward contraction. The second event pass partitions its columns by
    // the second occupied index, so no full-size worker copies are needed.
    arma::mat t2_virtual_bar;
    std::vector<reverse_worker> reverse_workers(vjp && cotangent!=0 ?
        size_t(omp_get_max_threads()) : 0);
    if(!reverse_workers.empty()) {
        t2_virtual_bar.zeros(nv*nv,no*no);
        #pragma omp parallel for schedule(static)
        for(size_t worker=0;worker<reverse_workers.size();++worker) {
            auto& local=reverse_workers[worker];
            local.t1.zeros(nv,no);
            local.fov.zeros(no,nv);
            local.eocc.zeros(no); local.evir.zeros(nv);
            if(projected)
                local.occupied_subspace.zeros(no,selected_orbitals.n_cols);
        }
    }
    double vjp_reverse_seconds=0;
    auto reverse_orders=[](const reverse_sample& sample) {
        return sample.bar_prz!=0.0 ? size_t(6) :
            (sample.bar_pw!=0.0 || sample.bar_d!=0.0 ? size_t(1) : size_t(0));
    };
    auto reverse_chunk=[&](const std::vector<reverse_sample>& samples,
                           size_t first,size_t count) {
        if(count==0) return;
        const double started=omp_get_wtime();
        const size_t rows=no*nv;
        std::vector<size_t> offsets(rows+1,0);
        std::vector<size_t> pair_offsets(2*no+1,0);
        // The Pz pullback uses every p,r pair. A PW-only pullback needs only
        // r=0 for each p; denominator bars use just p=r=0.
        // Projected sampling creates separate PW and Pz records, so avoiding
        // their zero-adjoint events saves most of the unnecessary work here.
        for(size_t n=first;n<first+count;++n) for(size_t p=0;p<6;++p) {
            const auto y=pair_permute(samples[n].x,p);
            for(size_t r=0;r<reverse_orders(samples[n]);++r) {
                ++offsets[y[perm[r][0]]*nv+y[3]+1];
                ++pair_offsets[y[perm[r][1]]+1];
                ++pair_offsets[no+y[perm[r][2]]+1];
            }
        }
        for(size_t row=0;row<rows;++row) offsets[row+1]+=offsets[row];
        for(size_t owner=0;owner<2*no;++owner)
            pair_offsets[owner+1]+=pair_offsets[owner];
        std::vector<size_t> cursor(offsets.begin(),offsets.end()-1);
        // A reverse chunk contains at most 16*2048*4 records, so its event
        // IDs fit in 32 bits even for selected-S's 36 Pz events per record.
        std::vector<std::uint32_t> events(offsets.back());
        for(size_t n=0;n<count;++n) for(size_t p=0;p<6;++p) {
            const auto y=pair_permute(samples[first+n].x,p);
            for(size_t r=0;r<reverse_orders(samples[first+n]);++r)
                events[cursor[y[perm[r][0]]*nv+y[3]]++]=
                    std::uint32_t(n*36+p*6+r);
        }
        // Each integral adjoint row is owned by exactly one worker. Small
        // amplitude and orbital adjoints stay private until reduction.
        #pragma omp parallel for schedule(dynamic)
        for(size_t row=0;row<rows;++row) {
            auto& local=reverse_workers[size_t(omp_get_thread_num())];
            for(size_t entry=offsets[row];entry<offsets[row+1];++entry) {
                const size_t event=events[entry], n=event/36;
                const size_t p=(event%36)/6, r=event%6;
                const auto& sample=samples[first+n];
                const auto y=pair_permute(sample.x,p);
                auto z=y;
                for(size_t leg=0;leg<3;++leg) z[leg]=y[perm[r][leg]];
                const size_t j=z[1],k=z[2],b=z[4],c=z[5];
                const double bar_w=sample.bar_prz*rweight[r]+(r==0 ? sample.bar_pw : 0);
                const double bar_v=0.5*sample.bar_prz*rweight[r];
                double* integral_bar=vjp->ovvv.colptr(row)+b*nv;
                const double* amplitude=t2_virtual(k,j).colptr(c);
                for(size_t f=0;f<nv;++f)
                    integral_bar[f]+=bar_w*amplitude[f];
                for(size_t m=0;m<no;++m) {
                    vjp->ovoo(row,j*no+m)-=bar_w*t2(m,k)(b,c);
                }
                const double gv=ovov(row,j*nv+b);
                vjp->ovov(row,j*nv+b)+=bar_v*t1(k)(c);
                local.t1(c,k)+=bar_v*gv;
                local.fov(k,c)+=bar_v*t2(z[0],j)(z[3],b);
                if(p==0 && r==0) {
                    for(size_t leg=0;leg<3;++leg) {
                        local.eocc(sample.x[leg])+=sample.bar_d;
                        local.evir(sample.x[leg+3])-=sample.bar_d;
                    }
                }
            }
        }
        // The virtual contraction updates T2(k,j;c,f), owned by group j.
        // The occupied contraction updates T2(m,k;b,c), owned by group k.
        // Posting each reverse event to these two groups gives one shared,
        // race-free transposed T2 adjoint without atomics or worker copies.
        auto event_pair=[&](size_t event) {
            const auto& sample=samples[first+event/36];
            const size_t p=(event%36)/6, r=event%6;
            const auto y=pair_permute(sample.x,p);
            return std::array<size_t,2>{{y[perm[r][1]],y[perm[r][2]]}};
        };
        std::vector<size_t> pair_cursor(pair_offsets.begin(),pair_offsets.end()-1);
        std::vector<std::uint32_t> pair_events(events.size()*2);
        for(std::uint32_t event:events) {
            const auto owner=event_pair(event);
            pair_events[pair_cursor[owner[0]]++]=event;
            pair_events[pair_cursor[no+owner[1]]++]=event;
        }
        #pragma omp parallel for schedule(dynamic)
        for(size_t owner=0;owner<no;++owner) {
            for(size_t entry=pair_offsets[owner];entry<pair_offsets[owner+1];++entry) {
                const size_t event=pair_events[entry];
                const auto& sample=samples[first+event/36];
                const size_t p=(event%36)/6, r=event%6;
                const auto y=pair_permute(sample.x,p);
                auto z=y;
                for(size_t leg=0;leg<3;++leg) z[leg]=y[perm[r][leg]];
                const size_t row=z[0]*nv+z[3],b=z[4],c=z[5],k=z[2];
                const double bar_w=sample.bar_prz*rweight[r]+(r==0 ? sample.bar_pw : 0);
                const double* integral=ovvv.colptr(row)+b*nv;
                double* amplitude_bar=t2_virtual_bar.colptr(k*no+owner)+c*nv;
                for(size_t f=0;f<nv;++f)
                    amplitude_bar[f]+=bar_w*integral[f];
                if(fov) t2_virtual_bar(b+nv*z[3],z[0]*no+owner)+=
                    0.5*sample.bar_prz*rweight[r]*(*fov)(k,c);
            }
            for(size_t entry=pair_offsets[no+owner];entry<pair_offsets[no+owner+1];++entry) {
                const size_t event=pair_events[entry];
                const auto& sample=samples[first+event/36];
                const size_t p=(event%36)/6, r=event%6;
                const auto y=pair_permute(sample.x,p);
                auto z=y;
                for(size_t leg=0;leg<3;++leg) z[leg]=y[perm[r][leg]];
                const size_t row=z[0]*nv+z[3],j=z[1],b=z[4],c=z[5];
                const double bar_w=sample.bar_prz*rweight[r]+(r==0 ? sample.bar_pw : 0);
                for(size_t m=0;m<no;++m)
                    t2_virtual_bar(c+nv*b,m*no+owner)-=bar_w*ovoo(row,j*no+m);
            }
        }
        vjp_reverse_seconds+=omp_get_wtime()-started;
    };
    moments all{0,0.0,0.0};
    size_t round_samples=nsamples, batch_offset=0;
    double pilot_error=0, standard_error=0;
    do {
    const size_t nbatches=(round_samples-1)/batch_size+1;
    std::vector<moments> batches(nbatches, moments{0,0.0,0.0});
    // Reverse each fixed group of logical batches before sampling the next.
    // This bounds the saved draws and event lists even for a large adaptive
    // continuation, while the per-batch RNG streams and merge order stay fixed.
    const size_t reverse_batches=reverse_workers.empty() ? nbatches : size_t(16);
    for(size_t first_batch=0;first_batch<nbatches;first_batch+=reverse_batches) {
    const size_t block_batches=std::min(reverse_batches,nbatches-first_batch);
    const size_t block_draws=std::min(round_samples-first_batch*batch_size,
        block_batches*batch_size);
    std::vector<reverse_sample> reverse_records(reverse_workers.empty() ? 0 :
        block_draws*(projected ? 4 : 1));
    #pragma omp parallel
    {
        #pragma omp single
        sampling_threads=omp_get_num_threads();
        #pragma omp for schedule(dynamic)
        for(size_t block_batch=0;block_batch<block_batches;++block_batch) {
            try {
                const size_t batch=first_batch+block_batch;
                std::uint64_t state=seed+0x9e3779b97f4a7c15ULL*(batch_offset+batch+1);
                state=(state^(state>>30))*0xbf58476d1ce4e5b9ULL;
                state=(state^(state>>27))*0x94d049bb133111ebULL;
                state^=state>>31;
                std::mt19937_64 rng(state);
                auto channel_local=channel;
                auto gradient_channels_local=gradient_channels;
                auto proposal_family_local=proposal_family;
                auto uniform_orbital_local=uniform_orbital;
                auto projector_orbital_local=projector_orbital;
                moments stats{0,0.0,0.0};
                const size_t count=std::min(batch_size,round_samples-batch*batch_size);
                for(size_t n=0;n<count;++n) {
                    index_type x;
                    if(vjp && std::uniform_real_distribution<double>(0,1)(rng)<uniform_fraction) {
                        std::uniform_int_distribution<size_t> occ(0,no-1),vir(0,nv-1),leg(0,2);
                        x={{occ(rng),occ(rng),occ(rng),vir(rng),vir(rng),vir(rng)}};
                        if(!projected)
                            x[leg(rng)]=uniform_orbital_local(rng);
                    } else {
                    const size_t family=gradient_mixture ? proposal_family_local(rng) : 0;
                    const size_t channel_id=family==0 ? channel_local(rng) :
                        gradient_channels_local[family-1](rng);
                    const size_t h=channel_id%nh, leg=channel_id/nh;
                    const bool fixed_left=whole_vjp && restrict_left(h,leg);
                    const bool fixed_right=whole_vjp && !fixed_left;
                    const size_t l=family==1 ? draw_uniform_factor(h,true,
                        fixed_left ? leg : 3,rng,uniform_orbital_local) :
                        fixed_left ? restricted[channel_id]->draw(rng,
                            [&](size_t index) { return restricted_value(h,leg,index); }) :
                            left[h]->draw(rng,[&](size_t index) { return left_value(h,index); });
                    const size_t r=family==2 ? draw_uniform_factor(h,false,
                        fixed_right ? leg : 3,rng,uniform_orbital_local) :
                        fixed_right ? restricted[channel_id]->draw(rng,
                            [&](size_t index) { return restricted_value(h,leg,index); }) :
                            right[h]->draw(rng,[&](size_t index) { return right_value(h,index); });
                    if (h<nv) x={{l/(nv*nv), r%no, r/(nv*no), (l/nv)%nv, l%nv, (r/no)%nv}};
                    else x={{l/(nv*no), l%no, r/(nv*nv), (l/no)%nv, r%nv, (r/nv)%nv}};
                    }
                    double pw=0, pq=0, pq_amplitude=0, pq_integral=0, prz=0;
                    for (size_t p=0; p<6; ++p) {
                        const index_type y=pair_permute(x,p);
                        const auto contracted=wq(y);
                        pw += contracted[0]; pq += contracted[1];
                        if(gradient_mixture) {
                            pq_amplitude+=contracted[2];
                            pq_integral+=contracted[3];
                        }
                        if(!projected) prz += rz(y,contracted[0]);
                    }
                    // Orbit averaging preserves the expectation even though
                    // the original tuple was drawn from unsymmetrized q.
                    double gradient_weight=0, value=0;
                    if(projected) {
                        // Draw the occupied index on the second side of U*U^T
                        // from its absolute row mass, then orbit-average
                        // of the proposal tuple. The three cyclic choices are
                        // duplicated by swapping the other two pairs.
                        const size_t i=projector_orbital_local(rng);
                        double cross=0;
                        const double projector_probability=projector_masses[i]/projector_l1;
                        const double inverse_projector_probability=1.0/projector_probability;
                        const double inverse_orbit_probability=1.0/(
                            (norm>0 ? (1-uniform_fraction)*pq/norm : 0.)+
                            6*uniform_fraction/projected_uniform_size);
                        const double scale=(2.0/3.0)*inverse_orbit_probability;
                        const size_t record_offset=(block_batch*batch_size+n)*4;
                        for(size_t leg=0;leg<3;++leg) {
                            const double projector_weight=projector(x[leg],i);
                            auto y=pair_permute(x,leg);
                            y[0]=i;
                            const double prz_cross=rz_orbit_sum(y);
                            const double d=denominator(y);
                            cross+=projector_weight*prz_cross/d;
                            if(!reverse_records.empty()) {
                                const double factor=cotangent*scale*pw*
                                    inverse_projector_probability;
                                reverse_records[record_offset+leg+1]={y,0.,
                                    factor*projector_weight/d,
                                    -factor*projector_weight*prz_cross/(d*d)};
                                auto& bar_u=reverse_workers[size_t(omp_get_thread_num())].occupied_subspace;
                                for(size_t s=0;s<selected_orbitals.n_cols;++s) {
                                    const double bar_p=factor*prz_cross/d;
                                    bar_u(x[leg],s)+=bar_p*selected_orbitals(i,s);
                                    bar_u(i,s)+=bar_p*selected_orbitals(x[leg],s);
                                }
                            }
                        }
                        value=scale*pw*cross*inverse_projector_probability;
                        if(!reverse_records.empty()) {
                            reverse_records[record_offset]={x,
                                cotangent*scale*cross*inverse_projector_probability,
                                0.,0.};
                        }
                    } else {
                        const double mixed_orbit_density=vjp ?
                            (proposal_norms[0]>0 ? proposal_mix[0]*pq/proposal_norms[0] : 0)+
                            (proposal_norms[1]>0 ? proposal_mix[1]*pq_amplitude/proposal_norms[1] : 0)+
                            (proposal_norms[2]>0 ? proposal_mix[2]*pq_integral/proposal_norms[2] : 0) : 0;
                        gradient_weight=vjp ? (2.0/3)/(
                            (1-uniform_fraction)*mixed_orbit_density+
                            2*uniform_fraction/(uniform_size*abs_weight_sum)) : 0;
                        value=vjp ? gradient_weight*pw*prz/denominator(x) :
                            2*pw*prz*norm/(denominator(x)*pq);
                    }
                    if (!std::isfinite(value)) throw std::runtime_error("(T): nonfinite sample");
                    if(!projected && !reverse_records.empty()) {
                        const double d=denominator(x);
                        const double factor=cotangent*gradient_weight/d;
                        reverse_records[block_batch*batch_size+n]={x,factor*prz,factor*pw,
                            -factor*pw*prz/d};
                    }
                    ++stats.count;
                    const double delta=value-stats.mean;
                    stats.mean+=delta/double(stats.count);
                    stats.m2+=delta*(value-stats.mean);
                }
                batches[batch]=stats;
            } catch(...) {
                #pragma omp critical(lccplay_triples_error)
                { if(!parallel_error) parallel_error=std::current_exception(); }
            }
        }
    }
    if(parallel_error) std::rethrow_exception(parallel_error);
    reverse_chunk(reverse_records,0,reverse_records.size());
    }
    for(const auto& part:batches) {
        const double delta=part.mean-all.mean;
        const size_t count=all.count+part.count;
        all.m2+=part.m2+delta*delta*(double(all.count)*double(part.count)/double(count));
        all.mean+=delta*(double(part.count)/double(count));
        all.count=count;
    }
    standard_error=std::sqrt(std::max(0.0,all.m2)/(double(all.count-1)*double(all.count)));
    if(batch_offset==0) pilot_error=standard_error;
    batch_offset+=nbatches;
    if(target_standard_error==0 || standard_error<=target_standard_error) break;
    // Estimate the additional work from the pooled sample variance. Keep
    // each continuation bounded, then reassess; never redraw the pilot or
    // rebuild the probability tables. The reported SE is the usual sample
    // estimate, not a sequential confidence bound.
    const double ratio=standard_error/target_standard_error;
    const double additional=std::ceil(double(all.count)*(ratio*ratio-1));
    round_samples=size_t(std::max(double(batch_size),std::min(1000000.0,additional)));
    if(all.count>std::numeric_limits<size_t>::max()-round_samples)
        throw std::overflow_error("(T): adaptive sample count overflow");
    } while(true);
    if(vjp) {
        if(!reverse_workers.empty()) {
            const double reduction_start=omp_get_wtime();
            #pragma omp parallel for schedule(static)
            for(size_t pair=0;pair<no*no;++pair) {
                arma::mat transposed(t2_virtual_bar.colptr(pair),nv,nv,false,true);
                vjp->t2(pair/no,pair%no)=transposed.t();
            }
            for(const auto& local:reverse_workers) {
                for(size_t i=0;i<no;++i) vjp->t1(i)+=local.t1.col(i);
                vjp->eocc+=local.eocc; vjp->evir+=local.evir;
                vjp->fov+=local.fov;
                if(projected) vjp->occupied_subspace+=local.occupied_subspace;
            }
            vjp_reverse_seconds+=omp_get_wtime()-reduction_start;
        }
        const double scale=1.0/double(all.count);
        for(size_t i=0;i<no;++i) {
            vjp->t1(i)*=scale;
            for(size_t j=0;j<no;++j) vjp->t2(i,j)*=scale;
        }
        vjp->eocc*=scale; vjp->evir*=scale;
        vjp->fov*=scale;
        if(projected) {
            vjp->occupied_subspace*=scale;
            if(occupied_projector) {
                vjp->occupied_projector=arma::vec(vjp->occupied_subspace.col(0));
                vjp->occupied_subspace.reset();
            }
        }
        vjp->ovvv*=scale; vjp->ovoo*=scale; vjp->ovov*=scale;
    }
    return {all.mean, standard_error, all.count, probability_seconds,
        omp_get_wtime()-sampling_start, setup_threads, sampling_threads, nsamples,
        pilot_error, packing_seconds, vjp_reverse_seconds};
}


}} // namespace libgmbpt::qchem
