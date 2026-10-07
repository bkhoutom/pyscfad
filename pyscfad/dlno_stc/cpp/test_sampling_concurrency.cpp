#include "stc_mp2_helpers.h"
#include <iostream>
#include <set>

using namespace pyscfad::dlno_stc;

struct Run { Moments moments; arma::mat bar; std::set<int> workers; };
Run run(int threads) {
    omp_set_num_threads(threads);
    Run result;
    result.bar.zeros(3,97);
    arma::mat source(3,97);
    for (arma::uword j=0;j<source.n_elem;++j) source[j]=(j+1)*.003;
    result.moments=draw_batched(65539,713,true,{0,1,3},
        [&](std::mt19937_64& rng,std::vector<Update>& records) {
            #pragma omp critical(test_sampling_workers)
            result.workers.insert(omp_get_thread_num());
            if (records.size()>=4096)
                throw std::runtime_error("logical batch retained more than 1024 samples");
            const auto column=rng()%97;
            const double value=std::generate_canonical<double,53>(rng)-.5;
            for (std::size_t role=0;role<4;++role)
                records.push_back({&result.bar,&source,(column+role)%97,
                    (column+3*role)%97,role%2,value*(role+1)});
            return value;
        });
    return result;
}
int main() {
    try {
        omp_set_dynamic(0);
        const auto serial=run(1);
        for (int threads:{2,6,16,32,40}) {
            const auto result=run(threads);
            const auto expected=static_cast<std::size_t>(std::min(threads,32));
            if (result.workers.size()!=expected)
                throw std::runtime_error("sampling did not use the bounded useful worker team");
            if (result.moments.count!=serial.moments.count ||
                    result.moments.mean!=serial.moments.mean || result.moments.m2!=serial.moments.m2 ||
                    !arma::approx_equal(result.bar,serial.bar,"absdiff",0.))
                throw std::runtime_error("team size changed moments or gradient update order");
        }
        omp_set_num_threads(32);
        int largest_team=0;
        parallel_for(2,[&](std::size_t) {
            #pragma omp critical(test_sampling_team)
            largest_team=std::max(largest_team,omp_get_num_threads());
        });
        if (largest_team!=2) throw std::runtime_error("small jobs launched idle workers");
        bool caught=false;
        try { parallel_for(16,[](std::size_t i) { if(i==3) throw std::runtime_error("worker"); }); }
        catch(const std::runtime_error&) { caught=true; }
        if(!caught) throw std::runtime_error("worker exception was lost");
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
    std::cout<<"Sampling concurrency checks passed\n";
}
