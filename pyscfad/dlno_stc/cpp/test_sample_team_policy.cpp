#include "native/stc_mp2_kernel_helpers.h"
#include <atomic>

using namespace libgmbpt::qchem::stc_mp2;

int main() {
    omp_set_num_threads(4);
    omp_set_max_active_levels(1);
    const unsigned seed=713;
    const size_t count=65539, point=2, stream=3;
    // Independent oracle for the historical four fixed-worker streams.
    moments expected;
    for (int tid=0;tid<4;++tid) {
        std::mt19937 rng(seed+104729u*point+1000003u*stream+7919u*tid);
        moments local;
        const size_t n=count/4+(size_t(tid)<count%4);
        for(size_t i=0;i<n;++i) {
            const double x=std::generate_canonical<double,53>(rng)-.5;
            ++local.count;const double delta=x-local.mean;
            local.mean+=delta/local.count;local.m2+=delta*(x-local.mean);
        }
        merge(expected,local);
    }
    for(int dynamic:{0,1}) {
        omp_set_dynamic(dynamic);
        for(bool serialized:{false,true}) {
            omp_set_max_active_levels(serialized ? 0 : 1);
            std::atomic<bool> valid(true);
            moments result;
            auto run=[&]() {
                result=draw_moments(count,seed,point,stream,[&](std::mt19937& rng) {
                    if(omp_get_dynamic()!=dynamic || (serialized && omp_get_num_threads()!=1))valid=false;
                    return std::generate_canonical<double,53>(rng)-.5;
                });
            };
            // A zero active-level budget guarantees one physical worker,
            // independently of runtime load-based dynamic-team decisions.
            run();
            if(!valid || omp_get_dynamic()!=dynamic || result.count!=expected.count
                    || result.mean!=expected.mean || result.m2!=expected.m2) {
                std::cerr<<"OpenMP team sizing changed sampled streams or caller policy\n";
                return 1;
            }
        }
    }
    std::cout<<"Sample streams preserve caller policy and survive reduced teams\n";
}
