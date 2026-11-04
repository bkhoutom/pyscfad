#ifndef PYSCFAD_STC_SCOPED_OPENBLAS_THREADS_H
#define PYSCFAD_STC_SCOPED_OPENBLAS_THREADS_H

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <omp.h>

#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
extern "C" {
int openblas_get_num_threads(void);
int openblas_get_parallel(void);
void openblas_set_num_threads(int);
}
#endif

namespace libgmbpt { namespace qchem { namespace detail {

inline int observed_blas_threads() {
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    return openblas_get_num_threads();
#else
    return 0; // Provider thread count is unavailable in the default build.
#endif
}

inline int requested_dense_blas_threads() {
    const char* text=std::getenv("PYSCFAD_STC_DENSE_BLAS_THREADS");
    if(!text) return 0; // Preserve the embedding process's existing policy.
    char* end=nullptr;
    errno=0;
    const long value=std::strtol(text,&end,10);
    if(errno || end==text || *end || value<1 || value>INT_MAX)
        throw std::invalid_argument("PYSCFAD_STC_DENSE_BLAS_THREADS must be a positive integer");
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    // OpenMP OpenBLAS can alter OpenMP team settings; only the independent
    // pthread worker pool is supported by this opt-in feature.
    if(openblas_get_parallel()!=1)
        throw std::runtime_error("PYSCFAD_STC_DENSE_BLAS_THREADS requires pthread OpenBLAS");
#else
    throw std::runtime_error("PYSCFAD_STC_DENSE_BLAS_THREADS requires rebuilding with "
                             "PYSCFAD_STC_OPENBLAS_THREAD_CONTROL=ON and pthread OpenBLAS");
#endif
    return static_cast<int>(value);
}

// OpenBLAS limits are process-wide, as with the existing Python BLAS scopes.
// Concurrent callers with competing BLAS policies are unsupported. Use only
// around serial GEMMs, never around an OpenMP region or the whole kernel.
class scoped_openblas_threads {
    int previous_=0;
public:
    explicit scoped_openblas_threads(int requested) {
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
        if(!requested) return;
        if(omp_in_parallel())
            throw std::runtime_error("STC dense BLAS scope cannot enter an OpenMP region");
        previous_=openblas_get_num_threads();
        openblas_set_num_threads(requested);
        if(openblas_get_num_threads()!=requested) {
            openblas_set_num_threads(previous_);
            previous_=0;
            throw std::runtime_error("OpenBLAS could not apply the requested STC dense thread count");
        }
#else
        if(requested)
            throw std::runtime_error("STC dense BLAS thread control is unavailable in this build");
#endif
    }
    ~scoped_openblas_threads() {
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
        if(previous_) openblas_set_num_threads(previous_);
#endif
    }
    scoped_openblas_threads(const scoped_openblas_threads&)=delete;
    scoped_openblas_threads& operator=(const scoped_openblas_threads&)=delete;
};

}}}
#endif
