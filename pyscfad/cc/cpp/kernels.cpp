// Copyright 2026 The PySCFAD Authors
// Licensed under the Apache License, Version 2.0 (see LICENSE).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <cstdint>
#include <omp.h>
#include "residual.h"

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style>;
using Gemm = void (*)(char*, char*, int*, int*, int*, double*, double*, int*,
                     double*, int*, double*, double*, int*);
static Gemm gemm = nullptr;

// Strided tensor operations used between the resident CCSD GEMMs. Each worker
// owns disjoint output rows; no BLAS pool is entered from these OpenMP teams.
using Tensor = py::array_t<double, 0>;
struct View4 {
    const double* data;
    py::ssize_t n[4], s[4];
    explicit View4(const Tensor& a) : data(a.data()) {
        if (a.ndim() != 4) throw py::value_error("expected a rank-four tensor");
        for (int d=0; d<4; ++d) {
            n[d]=a.shape(d);
            if (a.strides(d) % sizeof(double)) throw py::value_error("unaligned tensor stride");
            s[d]=a.strides(d)/py::ssize_t(sizeof(double));
        }
    }
    py::ssize_t offset(py::ssize_t row) const {
        const auto k=row%n[2]; row/=n[2];
        const auto j=row%n[1], i=row/n[1];
        return i*s[0]+j*s[1]+k*s[2];
    }
    py::ssize_t rows() const { return n[0]*n[1]*n[2]; }
};

static void same_shape(const View4& a, const View4& b) {
    for (int d=0; d<4; ++d)
        if (a.n[d] != b.n[d]) throw py::value_error("inconsistent tensor shapes");
}

static void validate_destination(const Tensor& out, const View4& dst, const View4& src) {
    if (!out.writeable()) throw py::value_error("output is not writable");
    if (out.size() == 0) return;
    std::vector<std::pair<py::ssize_t,py::ssize_t>> axes;
    for (int d=0; d<4; ++d) if (dst.n[d]>1) axes.emplace_back(std::abs(dst.s[d]),dst.n[d]);
    std::sort(axes.begin(),axes.end());
    py::ssize_t span=1;
    for (auto axis:axes) {
        if (axis.first<span) throw py::value_error("overlapping output elements");
        span+=(axis.second-1)*axis.first;
    }
    auto bounds=[](const View4& a) {
        py::ssize_t lo=0,hi=0;
        for (int d=0;d<4;++d) {
            const auto delta=(a.n[d]-1)*a.s[d];
            lo+=std::min(py::ssize_t(0),delta); hi+=std::max(py::ssize_t(0),delta);
        }
        return std::make_pair(reinterpret_cast<uintptr_t>(a.data+lo),
                              reinterpret_cast<uintptr_t>(a.data+hi+1));
    };
    auto x=bounds(dst),y=bounds(src);
    bool identical=dst.data==src.data;
    for (int d=0;d<4;++d) identical=identical && dst.s[d]==src.s[d];
    if (!identical && x.first<y.second && y.first<x.second)
        throw py::value_error("permuted or partial input/output overlap");
}

static void axpy4(Tensor out, const Tensor& input, double alpha) {
    const View4 dst(out),src(input);
    same_shape(dst,src); validate_destination(out,dst,src);
    double* result=out.mutable_data();
    const auto rows=dst.rows(),last=dst.n[3];
    py::gil_scoped_release release;
    #pragma omp parallel for schedule(static) if(rows*last>4096)
    for (py::ssize_t row=0;row<rows;++row) {
        const auto d=dst.offset(row),s=src.offset(row);
        #pragma omp simd
        for (py::ssize_t b=0;b<last;++b)
            result[d+b*dst.s[3]]+=alpha*src.data[s+b*src.s[3]];
    }
}

static Array linear4(const Tensor& left, const Tensor& right, double alpha, double beta) {
    const View4 a(left),b(right); same_shape(a,b);
    Array out(std::vector<py::ssize_t>(left.shape(),left.shape()+4));
    double* result=out.mutable_data();
    const auto rows=a.rows(),last=a.n[3];
    py::gil_scoped_release release;
    #pragma omp parallel for schedule(static) if(rows*last>4096)
    for (py::ssize_t row=0;row<rows;++row) {
        const auto x=a.offset(row),y=b.offset(row);
        #pragma omp simd
        for (py::ssize_t k=0;k<last;++k)
            result[row*last+k]=alpha*a.data[x+k*a.s[3]]+beta*b.data[y+k*b.s[3]];
    }
    return out;
}

static Array tau4(const Tensor& tensor, const Tensor& left, const Tensor& right,
                  double alpha, double beta) {
    const View4 t(tensor);
    if (left.ndim()!=2 || right.ndim()!=2 || left.shape(0)!=t.n[0] ||
        left.shape(1)!=t.n[2] || right.shape(0)!=t.n[1] || right.shape(1)!=t.n[3])
        throw py::value_error("inconsistent outer-product shapes");
    for (int d=0;d<2;++d)
        if (left.strides(d)%sizeof(double) || right.strides(d)%sizeof(double))
            throw py::value_error("unaligned matrix stride");
    Array out(std::vector<py::ssize_t>(tensor.shape(),tensor.shape()+4));
    const double *l=left.data(),*r=right.data(); double* result=out.mutable_data();
    const auto l0=left.strides(0)/8,l1=left.strides(1)/8;
    const auto r0=right.strides(0)/8,r1=right.strides(1)/8;
    const auto rows=t.rows(),last=t.n[3];
    py::gil_scoped_release release;
    #pragma omp parallel for schedule(static) if(rows*last>4096)
    for (py::ssize_t row=0;row<rows;++row) {
        const auto a=row%t.n[2],j=(row/t.n[2])%t.n[1],i=row/(t.n[2]*t.n[1]);
        const auto offset=t.offset(row);
        const double x=beta*l[i*l0+a*l1];
        #pragma omp simd
        for (py::ssize_t b=0;b<last;++b)
            result[row*last+b]=alpha*t.data[offset+b*t.s[3]]+x*r[j*r0+b*r1];
    }
    return out;
}

static size_t pair_index(size_t i, size_t j) {
    const size_t hi = std::max(i, j), lo = std::min(i, j);
    return hi*(hi+1)/2 + lo;
}

LadderPanelPlan ladder_panel_plan(size_t v, size_t workspace_bytes, size_t threads) {
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
void ladder_into(const double* p, const double* input, double* result,
                 size_t v, size_t rows, const LadderPanelPlan& plan,
                 double* storage, const size_t* indices) {
    const size_t vv=v*v;
    const size_t npair = v*(v+1)/2;
    const size_t workers=plan.workers, ablock=plan.a, cblock=plan.c;
    const size_t panel_size = ablock*cblock*vv;
    #pragma omp parallel num_threads(workers)
    {
        double* panel=storage+size_t(omp_get_thread_num())*panel_size;
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
}

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
    const auto plan = ladder_panel_plan(v, workspace_bytes, size_t(omp_get_max_threads()));
    std::unique_ptr<double[]> storage(new double[plan.elements(v)]);
    std::vector<size_t> indices(vv);
    for (size_t i=0; i<v; ++i) for (size_t j=0; j<v; ++j)
        indices[i*v+j] = pair_index(i,j);
    const double* p=pairs.data();const double* input=t.data();double* result=output.mutable_data();
    py::gil_scoped_release release;
    ladder_into(p,input,result,v,rows,plan,storage.get(),indices.data());
    return output;
}

PYBIND11_MODULE(_ccsd_kernels, m) {
    m.def("axpy4", &axpy4, py::arg("out").noconvert(), py::arg("input").noconvert(), py::arg("alpha"));
    m.def("linear4", &linear4, py::arg("left").noconvert(), py::arg("right").noconvert(), py::arg("alpha"), py::arg("beta"));
    m.def("tau4", &tau4, py::arg("tensor").noconvert(), py::arg("left").noconvert(), py::arg("right").noconvert(), py::arg("alpha"), py::arg("beta"));
    // SciPy's public Cython BLAS API has a stable LP64 (int*) interface,
    // including when SciPy internally wraps an ILP64 BLAS installation.
    py::object capsule = py::module_::import("scipy.linalg.cython_blas")
        .attr("__pyx_capi__")["dgemm"];
    gemm = reinterpret_cast<Gemm>(PyCapsule_GetPointer(
        capsule.ptr(), PyCapsule_GetName(capsule.ptr())));
    if (!gemm) throw py::error_already_set();
    bind_residual(m, gemm, ladder);
    bind_solver(m);
    bind_lambda(m);
    m.def("ladder", &ladder, py::arg("pairs").noconvert(),
          py::arg("t").noconvert(), py::arg("workspace_bytes"));
    m.def("panel_shape", [](size_t v, size_t bytes, size_t threads) {
        const auto p=ladder_panel_plan(v,bytes,threads);
        return py::make_tuple(p.workers,p.a,p.c,p.workers*p.a*p.c*v*v*sizeof(double));
    }, py::arg("nvir"), py::arg("workspace_bytes"), py::arg("threads"));
}
