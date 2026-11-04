#include "native/scoped_openblas_threads.h"
#include <iostream>

using namespace libgmbpt::qchem::detail;

int main() {
    try {
        if(openblas_get_parallel()!=1)
            throw std::runtime_error("test requires pthread OpenBLAS");
        openblas_set_num_threads(1);
        omp_set_dynamic(0);
        omp_set_num_threads(8);
        {
            scoped_openblas_threads outer(8);
            if(observed_blas_threads()!=8 || omp_get_max_threads()!=8)
                throw std::runtime_error("BLAS scope changed the wrong runtime");
            try {
                scoped_openblas_threads inner(2);
                if(observed_blas_threads()!=2)
                    throw std::logic_error("nested BLAS scope was not applied");
                throw std::runtime_error("expected failure");
            } catch(const std::runtime_error&) { }
            if(observed_blas_threads()!=8)
                throw std::runtime_error("exception did not restore the outer BLAS count");
        }
        if(observed_blas_threads()!=1 || omp_get_max_threads()!=8)
            throw std::runtime_error("BLAS scope did not restore the original policy");
        bool rejected=false;
        #pragma omp parallel num_threads(2)
        {
            #pragma omp single
            try { scoped_openblas_threads illegal(8); }
            catch(const std::runtime_error&) { rejected=true; }
        }
        if(!rejected || observed_blas_threads()!=1)
            throw std::runtime_error("BLAS scope accepted an OpenMP region");
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
    std::cout<<"Scoped BLAS nesting, exception restoration and OpenMP checks passed\n";
}
