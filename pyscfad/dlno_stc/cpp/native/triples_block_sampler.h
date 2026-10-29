#ifndef PYSCFAD_TRIPLES_BLOCK_SAMPLER_H
#define PYSCFAD_TRIPLES_BLOCK_SAMPLER_H
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace libgmbpt { namespace qchem { namespace detail {
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

}}}
#endif
