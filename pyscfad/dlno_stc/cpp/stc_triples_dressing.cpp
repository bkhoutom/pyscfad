// Smooth Laplace dressing: MP2 matrix functions and bounded tensor panels.
#include "stc_mp2_kernel.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>
#include <cstring>
#include <memory>
#include <cmath>
#include <dlfcn.h>

namespace py=pybind11;
namespace pyscfad { namespace dlno_stc { namespace {
using Array=py::array_t<double,py::array::c_style|py::array::forcecast>;
using Shape=std::vector<py::ssize_t>;
struct Axis { size_t axis, matrix; };

// Large panels may use BLAS, with one BLAS worker per OpenMP worker. On a
// backend without a local thread control, use the SIMD panel loop instead.
struct LocalBlas {
    using Setter=int(*)(int);
    Setter setter;
    int previous=0;
    LocalBlas() {
        static auto fn=reinterpret_cast<Setter>(dlsym(RTLD_DEFAULT,"MKL_Set_Num_Threads_Local"));
        setter=fn;
        if(setter) previous=setter(1);
    }
    ~LocalBlas() { if(setter) setter(previous); }
};
Shape shape(const Array& a) { return Shape(a.shape(),a.shape()+a.ndim()); }
Array copy(const Array& a) {
    Array result(shape(a));
    std::memcpy(result.mutable_data(),a.data(),size_t(a.size())*sizeof(double));
    return result;
}
void check_shape(const Array& a,const Shape& expected) {
    if(shape(a)!=expected) throw py::value_error("smooth triples dressing: inconsistent input shape");
}
Spectrum spectrum(const Array& fock) {
    const size_t n=size_t(fock.shape(0));
    arma::mat f(n,n);
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j)
        f(i,j)=.5*(fock.data()[i*n+j]+fock.data()[j*n+i]);
    Spectrum result;
    if(!arma::eig_sym(result.values,result.vectors,f))
        throw std::runtime_error("smooth triples dressing: Fock eigensolve failed");
    return result;
}
// Copy contiguous runs along the inner dimension; a panel may cross outer
// tensor slices. The output is a column-major (other indices)-by-orbital panel.
void gather(const double* input,arma::mat& packed,
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
void scatter(double* output,const arma::mat& packed,
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
void multiply(const arma::mat& input,const arma::mat& matrix,arma::mat& output,bool blas) {
    if(blas && matrix.n_rows>=16) { output=input*matrix.t(); return; }
    output.zeros(input.n_rows,matrix.n_rows);
    for(size_t i=0;i<matrix.n_rows;++i) for(size_t j=0;j<matrix.n_cols;++j) {
        const double scale=matrix(i,j);
        double* dst=output.colptr(i); const double* src=input.colptr(j);
        #pragma omp simd
        for(size_t r=0;r<input.n_rows;++r) dst[r]+=scale*src[r];
    }
}

// In-place full-axis transform. Optional raw input computes the matrix
// cotangent before replacing the tensor cotangent. Panels stay <=2 MiB each
// (three per worker in reverse), independent of the complete tensor size.
void axis_transform(Array& data,size_t axis,const arma::mat& matrix,
                    const Array* raw=nullptr,arma::mat* adjoint=nullptr) {
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(data.ndim());++d) inner*=size_t(data.shape(d));
    const size_t n=size_t(data.shape(axis)),rows=size_t(data.size())/n;
    const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(sizeof(double)*n));
    const size_t tasks=(rows+panel-1)/panel;
    std::vector<arma::mat> local;
    if(adjoint) {
        local.resize(size_t(omp_get_max_threads()));
        for(auto& gram:local) gram.zeros(n,n);
    }
    std::exception_ptr error;
    #pragma omp parallel
    {
        LocalBlas local_blas;
        arma::mat packed,result,source;
        #pragma omp for schedule(dynamic)
        for(size_t t=0;t<tasks;++t) {
            try {
                const size_t first=t*panel,count=std::min(panel,rows-first);
                gather(data.data(),packed,n,inner,first,count);
                if(raw) {
                    gather(raw->data(),source,n,inner,first,count);
                    auto& gram=local[size_t(omp_get_thread_num())];
                    if(local_blas.setter && n>=16) gram+=packed.t()*source;
                    else for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) {
                        const double* left=packed.colptr(i); const double* right=source.colptr(j);
                        double sum=0.;
                        #pragma omp simd reduction(+:sum)
                        for(size_t r=0;r<count;++r) sum+=left[r]*right[r];
                        gram(i,j)+=sum;
                    }
                }
                // Matrix exponentials are symmetric; A^T=A in reverse.
                multiply(packed,matrix,result,bool(local_blas.setter));
                scatter(data.mutable_data(),result,n,inner,first,count);
            } catch(...) {
                #pragma omp critical(triples_dressing_error)
                { if(!error) error=std::current_exception(); }
            }
        }
    }
    if(error) std::rethrow_exception(error);
    if(adjoint) for(const auto& gram:local) *adjoint+=gram;
}

class Dressing {
    std::vector<Array> inputs_;
    Spectrum occupied_,virtual_;
    std::vector<std::vector<Axis>> axes_;
    std::vector<size_t> sources_;
    const Spectrum& spectrum_for(size_t m) const { return m%2 ? virtual_ : occupied_; }
    std::vector<arma::mat> matrices(double node) const {
        if(!std::isfinite(node)) throw py::value_error("smooth triples dressing: nonfinite node");
        std::vector<arma::mat> result(4);
        const double scale[4]={node/2,-node/2,-node/2,node/2};
        for(size_t m=0;m<4;++m)
            result[m]=exponential(spectrum_for(m),scale[m],0.);
        return result;
    }
public:
    Dressing(py::tuple inputs) {
        if(inputs.size()!=9) throw py::value_error("smooth triples dressing: expected nine inputs including physical target U");
        for(auto input:inputs) {
            auto array=Array::ensure(input);
            if(!array) throw py::value_error("smooth triples dressing: expected numeric arrays");
            for(py::ssize_t i=0;i<array.size();++i)
                if(!std::isfinite(array.data()[i])) throw py::value_error("smooth triples dressing: nonfinite input");
            inputs_.push_back(std::move(array));
        }
        if(inputs_[0].ndim()!=2 || inputs_[2].ndim()!=3) throw py::value_error("smooth triples dressing: invalid input rank");
        const auto no=inputs_[0].shape(0),nv=inputs_[0].shape(1),np=inputs_[2].shape(0);
        if(no<=0 || nv<=0 || np<=0) throw py::value_error("smooth triples dressing: empty orbital space");
        check_shape(inputs_[1],{no,no,nv,nv}); check_shape(inputs_[2],{np,no,nv});
        check_shape(inputs_[3],{np,no,no}); check_shape(inputs_[4],{np,nv,nv});
        check_shape(inputs_[5],{no,nv}); check_shape(inputs_[6],{no,no}); check_shape(inputs_[7],{nv,nv});
        occupied_=spectrum(inputs_[6]); virtual_=spectrum(inputs_[7]);
        axes_={{{0,0},{1,1}},{{0,0},{1,0},{2,1},{3,1}},{{1,0},{2,1}},
               {{1,0},{2,2}},{{1,1},{2,3}},{{0,0},{1,1}}};
        sources_={0,1,2,3,4,5};
        if(inputs_[8].ndim()!=2 || inputs_[8].shape(0)!=no ||
           inputs_[8].shape(1)<=0 || inputs_[8].shape(1)>no)
            throw py::value_error("smooth triples dressing: U must have shape (nocc, nselected)");
        axes_.push_back({{0,2}}); sources_.push_back(8);
        axes_.push_back({{0,0}}); sources_.push_back(8);
    }
    py::tuple prepare(double node) const {
        const auto transforms=matrices(node);
        py::tuple result(axes_.size());
        for(size_t i=0;i<axes_.size();++i) {
            Array output=copy(inputs_[sources_[i]]);
            { py::gil_scoped_release release;
              for(auto axis:axes_[i]) axis_transform(output,axis.axis,transforms[axis.matrix]); }
            result[i]=std::move(output);
        }
        return result;
    }
    py::tuple pullback(double node,py::tuple cotangents) const {
        if(cotangents.size()!=axes_.size()) throw py::value_error("smooth triples dressing: invalid cotangent count");
        const auto transforms=matrices(node);
        std::vector<arma::mat> matrix_bars(4);
        for(size_t m=0;m<4;++m)
            matrix_bars[m].zeros(transforms[m].n_rows,transforms[m].n_cols);
        py::tuple result(inputs_.size());
        for(size_t i=0;i<axes_.size();++i) {
            auto cotangent=Array::ensure(cotangents[i]);
            if(!cotangent) throw py::value_error("smooth triples dressing: invalid cotangent");
            const auto& input=inputs_[sources_[i]];
            check_shape(cotangent,shape(input));
            Array bar=copy(cotangent);
            // Rematerialize preceding axes in one shared staging tensor.
            // This avoids retaining four full T2 intermediates for its VJP.
            Array stage(shape(input));
            for(size_t k=axes_[i].size();k-->0;) {
                const Array* raw=&input;
                if(k) {
                    std::memcpy(stage.mutable_data(),input.data(),size_t(input.size())*sizeof(double));
                    raw=&stage;
                }
                const auto axis=axes_[i][k];
                { py::gil_scoped_release release;
                  for(size_t j=0;j<k;++j) {
                      const auto previous=axes_[i][j];
                      axis_transform(stage,previous.axis,transforms[previous.matrix]);
                  }
                  axis_transform(bar,axis.axis,transforms[axis.matrix],raw,&matrix_bars[axis.matrix]); }
            }
            if(i==7) {
                auto sum=result[8].cast<Array>();
                for(py::ssize_t j=0;j<sum.size();++j) sum.mutable_data()[j]+=bar.data()[j];
            } else result[sources_[i]]=std::move(bar);
        }
        const double scale[4]={node/2,-node/2,-node/2,node/2};
        for(size_t space=0;space<2;++space) {
            const auto& input=inputs_[6+space]; const size_t n=size_t(input.shape(0));
            Array bar(shape(input)); std::fill(bar.mutable_data(),bar.mutable_data()+bar.size(),0.);
            for(size_t m=space;m<4;m+=2) {
                const arma::mat value=exponential_vjp(spectrum_for(m),scale[m],0.,matrix_bars[m]);
                for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j)
                    bar.mutable_data()[i*n+j]+=value(i,j);
            }
            result[6+space]=std::move(bar);
        }
        return result;
    }
};
} // namespace
void bind_triples_dressing(py::module_& module) {
    py::class_<Dressing>(module,"TriplesDressing")
        .def(py::init<py::tuple>(), py::arg("inputs"))
        .def("prepare",&Dressing::prepare)
        .def("pullback",&Dressing::pullback);
}
}} // namespace pyscfad::dlno_stc
