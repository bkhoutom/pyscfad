// Copyright 2026 The PySCFAD Authors
// Licensed under the Apache License, Version 2.0 (see LICENSE).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <omp.h>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style>;
using Gemm = void (*)(char*, char*, int*, int*, int*, double*, double*, int*,
                     double*, int*, double*, double*, int*);
static Gemm gemm = nullptr;

static size_t pair_index(size_t i, size_t j) {
    const size_t hi = std::max(i, j), lo = std::min(i, j);
    return hi*(hi+1)/2 + lo;
}

struct PanelPlan {
    size_t workers, a, c;
};

static PanelPlan panel_plan(size_t v, size_t workspace_bytes, size_t threads) {
    if (!v || !threads || v > size_t(std::numeric_limits<int>::max())/v)
        throw py::value_error("invalid virtual dimension or thread count");
    const size_t capacity = workspace_bytes/sizeof(double)/(v*v);
    if (!capacity) throw py::value_error("ladder workspace cannot hold one virtual panel");
    const size_t workers = std::min({v, threads, capacity});
    // Target roughly 1K-wide GEMM dimensions, where the measured BLAS
    // throughput has plateaued. This is a performance tile size, not a
    // fixed byte cap: panels grow with v and the number of active workers.
    // Leave enough independent output panels to occupy the OpenMP team.
    const size_t extent = (1024+v-1)/v;
    const size_t a = std::min({(v+workers-1)/workers, extent, capacity/workers});
    const size_t c = std::min({v, extent, capacity/(workers*a)});
    return {workers, a, c};
}

// Independent output panels let a single OpenMP team own all parallelism.
// The caller limits SciPy BLAS to one thread for this call. Each worker owns
// a bounded panel and writes disjoint output columns, with no reductions.
static Array ladder(const Array& pairs, const Array& t, size_t workspace_bytes) {
    if (t.ndim() < 2 || t.shape(t.ndim()-1) != t.shape(t.ndim()-2))
        throw py::value_error("ladder requires square trailing virtual axes");
    const size_t v = size_t(t.shape(t.ndim()-1)), vv = v*v;
    const size_t npair = v*(v+1)/2;
    if (pairs.ndim() != 2 || size_t(pairs.shape(0)) != npair ||
        size_t(pairs.shape(1)) != npair)
        throw py::value_error("inconsistent packed virtual integrals");
    Array output(std::vector<py::ssize_t>(t.shape(), t.shape()+t.ndim()));
    if (t.size() == 0) return output;
    const size_t rows = size_t(t.size())/vv;
    if (vv > size_t(std::numeric_limits<int>::max()) ||
        rows > size_t(std::numeric_limits<int>::max()))
        throw py::value_error("ladder dimensions exceed BLAS integer range");
    const auto plan = panel_plan(v, workspace_bytes, size_t(omp_get_max_threads()));
    const size_t workers=plan.workers, ablock=plan.a, cblock=plan.c;
    const size_t panel_size = ablock*cblock*vv;
    std::unique_ptr<double[]> storage(new double[panel_size*workers]);
    std::vector<size_t> indices(vv);
    for (size_t i=0; i<v; ++i) for (size_t j=0; j<v; ++j)
        indices[i*v+j] = pair_index(i,j);
    const double* p=pairs.data();
    const double* input=t.data();
    double* result=output.mutable_data();
    py::gil_scoped_release release;
    #pragma omp parallel num_threads(workers)
    {
        double* panel=storage.get()+size_t(omp_get_thread_num())*panel_size;
        #pragma omp for schedule(static)
        for (size_t a0=0; a0<v; a0+=ablock) {
            const size_t na=std::min(ablock,v-a0);
            for (size_t c0=0; c0<v; c0+=cblock) {
                const size_t nc=std::min(cblock,v-c0);
                for (size_t cd=0; cd<nc*v; ++cd) {
                    const size_t c=c0+cd/v, d=cd%v;
                    double* dst=panel+cd*na*v;
                    for (size_t a=0; a<na; ++a) {
                        const double* src=p+indices[c*v+a0+a]*npair;
                        #pragma omp simd
                        for (size_t b=0; b<v; ++b)
                            dst[a*v+b]=src[indices[d*v+b]];
                    }
                }
                int m=int(na*v), n=int(rows), k=int(nc*v), ld=int(vv);
                char no='N';
                double one=1., beta=c0 ? 1. : 0.;
                gemm(&no,&no,&m,&n,&k,&one,panel,&m,
                     const_cast<double*>(input+c0*v),&ld,&beta,result+a0*v,&ld);
            }
        }
    }
    return output;
}

PYBIND11_MODULE(_ccsd_kernels, m) {
    // SciPy's public Cython BLAS API has a stable LP64 (int*) interface,
    // including when SciPy internally wraps an ILP64 BLAS installation.
    py::object capsule = py::module_::import("scipy.linalg.cython_blas")
        .attr("__pyx_capi__")["dgemm"];
    gemm = reinterpret_cast<Gemm>(PyCapsule_GetPointer(
        capsule.ptr(), PyCapsule_GetName(capsule.ptr())));
    if (!gemm) throw py::error_already_set();
    m.def("ladder", &ladder, py::arg("pairs").noconvert(),
          py::arg("t").noconvert(), py::arg("workspace_bytes"));
    m.def("panel_shape", [](size_t v, size_t bytes, size_t threads) {
        const auto p=panel_plan(v,bytes,threads);
        return py::make_tuple(p.workers,p.a,p.c,p.workers*p.a*p.c*v*v*sizeof(double));
    }, py::arg("nvir"), py::arg("workspace_bytes"), py::arg("threads"));
}
