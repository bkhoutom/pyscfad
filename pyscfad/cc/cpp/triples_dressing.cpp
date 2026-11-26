#include "dense_matrix.h"
#include "triples_prepared.h"
#include <pybind11/stl.h>
namespace ccsd_native { namespace triples {
void bind_point(py::module_&);
namespace {
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
using Shape=std::vector<py::ssize_t>;
struct Axis { size_t axis, matrix; };

Shape shape(const Array& a) { return Shape(a.shape(),a.shape()+a.ndim()); }
Array copy(const Array& a) {
    Array result(shape(a));
    std::memcpy(result.mutable_data(),a.data(),size_t(a.size())*sizeof(double));
    return result;
}
void check_shape(const Array& a,const Shape& expected) {
    if(shape(a)!=expected) throw py::value_error("smooth triples dressing: inconsistent input shape");
}
struct Spectrum {dense::Matrix values,vectors;};
Spectrum spectrum(const Array& fock) {
    const size_t n=size_t(fock.shape(0));
    Array sym({py::ssize_t(n),py::ssize_t(n)});
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j)
        sym.mutable_data()[i*n+j]=.5*(fock.data()[i*n+j]+fock.data()[j*n+i]);
    auto result=py::module_::import("scipy.linalg").attr("eigh")(sym,
        py::arg("check_finite")=false,py::arg("driver")="evd").cast<py::tuple>();
    auto values=Array::ensure(result[0]),vectors=Array::ensure(result[1]);
    Spectrum out;out.values.set_size(n);out.vectors.set_size(n,n);
    std::copy_n(values.data(),n,out.values.data);
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j)out.vectors(i,j)=vectors.data()[i*n+j];
    return out;
}
dense::Matrix exponential(const Spectrum& eig,double coefficient,double shift) {
    if(coefficient==0.)return dense::eye(eig.values.n_elem,eig.values.n_elem);
    dense::Matrix scaled(eig.vectors);
    for(size_t j=0;j<scaled.n_cols;++j) {
        const double factor=std::exp(coefficient*(eig.values[j]-shift));
        for(size_t i=0;i<scaled.n_rows;++i)scaled(i,j)*=factor;
    }
    return scaled*eig.vectors.t();
}
dense::Matrix exponential_vjp(const Spectrum& eig,double coefficient,double shift,const dense::Matrix& adjoint) {
    const size_t n=eig.values.n_elem;
    if(coefficient==0.)return dense::Matrix(n,n,dense::zero);
    dense::Matrix sym(n,n);
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j)sym(i,j)=.5*(adjoint(i,j)+adjoint(j,i));
    dense::Matrix rotated=eig.vectors.t()*sym*eig.vectors;
    for(size_t p=0;p<n;++p)for(size_t q=0;q<n;++q) {
        const double x=coefficient*(eig.values[p]-shift),y=coefficient*(eig.values[q]-shift);
        const double d=std::abs(x-y);
        const double factor=d==0. ? coefficient : d<1. ? coefficient*(-std::expm1(-d)/d)
            : std::copysign(1.,coefficient)*(-std::expm1(-d))/std::abs(eig.values[p]-eig.values[q]);
        rotated(p,q)*=std::exp(std::max(x,y))*factor;
    }
    dense::Matrix out=eig.vectors*rotated*eig.vectors.t();
    for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j)out(i,j)=out(j,i)=.5*(out(i,j)+out(j,i));
    return out;
}
// Copy contiguous runs along the inner dimension; a panel may cross outer
// tensor slices. The output is a column-major (other indices)-by-orbital panel.
void gather(const double* input,dense::Matrix& packed,
            size_t n,size_t inner,size_t first,size_t count) {
    packed.set_size(count,n);
    for(size_t j=0;j<n;++j) {
        size_t done=0;
        while(done<count) {
            const size_t row=first+done, run=std::min(count-done,inner-row%inner);
            std::memcpy(packed.colptr(j)+done,input+((row/inner)*n+j)*inner+row%inner,run*sizeof(double));
            done+=run;
        }
    }
}
void scatter(double* output,const dense::Matrix& packed,
             size_t n,size_t inner,size_t first,size_t count) {
    for(size_t j=0;j<n;++j) {
        size_t done=0;
        while(done<count) {
            const size_t row=first+done, run=std::min(count-done,inner-row%inner);
            std::memcpy(output+((row/inner)*n+j)*inner+row%inner,packed.colptr(j)+done,run*sizeof(double));
            done+=run;
        }
    }
}
void multiply(const dense::Matrix& input,const dense::Matrix& matrix,dense::Matrix& output,bool blas) {
    if(blas && matrix.n_rows>=16) { output=input*matrix.t(); return; }
    output.zeros(input.n_rows,matrix.n_rows);
    for(size_t i=0;i<matrix.n_rows;++i) for(size_t j=0;j<matrix.n_cols;++j) {
        const double scale=matrix(i,j);
        double* dst=output.colptr(i); const double* src=input.colptr(j);
        #pragma omp simd
        for(size_t r=0;r<input.n_rows;++r) dst[r]+=scale*src[r];
    }
}


#include "triples_stable.inc"
}
void bind_dressing(py::module_& module) {
    py::class_<StableDressing>(module,"StableTriplesDressing")
        .def(py::init<py::tuple,bool,bool>(),py::arg("inputs"),py::kw_only(),
             py::arg("full_external")=false,py::arg("mrcc_partition")=false)
        .def("occupied_diagonal_at",&StableDressing::occupied_diagonal_at,py::arg("node"))
        .def("virtual_diagonal_at",&StableDressing::virtual_diagonal_at,py::arg("node"),
             "Whether this node uses scalar virtual dressing within a roundoff-sized primal error bound.")
        .def("prepare",&StableDressing::prepare,py::arg("node"),py::arg("cache_limit_bytes")=0)
        .def("pullback",&StableDressing::pullback,py::arg("node"),py::arg("cotangents"),py::arg("cache_limit_bytes")=0)
        .def("clear_cache",&StableDressing::clear_cache)
        .def_property_readonly("cache_bytes",&StableDressing::cache_bytes)
        .def_property_readonly("cache_hits",&StableDressing::cache_hits)
        .def_property_readonly("peak_cache_bytes",&StableDressing::peak_cache_bytes)
        .def_property_readonly("cache_rebuilds",&StableDressing::cache_rebuilds)
        .def_property_readonly("selected_occupied_frame",&StableDressing::selected_occupied_frame);
}
}}
void bind_native_triples(py::module_& module) {
    ccsd_native::triples::bind_dressing(module);
    ccsd_native::triples::bind_point(module);
}
