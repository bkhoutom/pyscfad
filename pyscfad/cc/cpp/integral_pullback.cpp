// Copyright 2026 The PySCFAD Authors (Apache-2.0).
// DF factor adjoints. SciPy owns BLAS; the Python adapter scopes it to one
// thread while these kernels use the same OpenMP runtime as RCCSD lambda.
#include "tensor.h"

namespace ccsd_native {
namespace {
int blas_dimension(Index n) {
    if(n<0 || n>INT_MAX)throw std::overflow_error("DF pullback exceeds BLAS integer range");
    return int(n);
}
void require_shape(const Array& a,const std::vector<Index>& expected) {
    shape_is(T{Input(a)},expected);
}

py::tuple ovvv_vjp(const Array& lov,const Array& lvv,const Array& t1a,
        const Array& thetaa,const Array& taua,const Array& t1bara,
        const Array& fvvbara,const Array& wbara,const Array& tmpbara,
        const Array& woobara) {
    if(lov.ndim()!=3)throw py::value_error("Lov must have shape (aux,occ,vir)");
    const Index q=lov.shape(0),o=lov.shape(1),v=lov.shape(2),pair=v*(v+1)/2;
    const int nv=blas_dimension(v),no2=blas_dimension(o*o);
    const int nq=blas_dimension(q),npair=blas_dimension(pair),nrow=blas_dimension(o*v);
    require_shape(lov,{q,o,v});require_shape(lvv,{q,pair});require_shape(t1a,{o,v});
    require_shape(thetaa,{o,v,o,v});require_shape(taua,{o,o,v,v});
    require_shape(t1bara,{o,v});require_shape(fvvbara,{v,v});
    require_shape(wbara,{v,o,o,v});require_shape(tmpbara,{o,o,v,o});
    require_shape(woobara,{o,o*pair});
    Array lovbar({q,o,v}),lvvbar({q,pair});
    if(!q || !o || !v) {
        std::fill_n(lovbar.mutable_data(),lovbar.size(),0.);
        std::fill_n(lvvbar.mutable_data(),lvvbar.size(),0.);
        return py::make_tuple(lovbar,lvvbar);
    }
    const double *t1=t1a.data(),*theta=thetaa.data(),*tau=taua.data();
    const double *t1bar=t1bara.data(),*fvvbar=fvvbara.data(),*wbar=wbara.data();
    const double *tmpbar=tmpbara.data(),*woobar=woobara.data();
    auto* lovout=lovbar.mutable_data();auto* lvvout=lvvbar.mutable_data();
    auto* lovin=const_cast<double*>(lov.data());auto* lvvin=const_cast<double*>(lvv.data());
    {
        py::gil_scoped_release release;
        T packed({o*v,pair});
        const int workers=int(std::min<Index>(omp_get_max_threads(),v));
        const Index scratch_size=v*v+2*o*o*v;
        T scratch({workers,scratch_size});
        #pragma omp parallel num_threads(workers)
        {
            double* dense=scratch.p+omp_get_thread_num()*scratch_size;
            double* tau_buf=dense+v*v;double* tmp_buf=tau_buf+o*o*v;
            #pragma omp for schedule(static)
            for(Index p=0;p<v;++p) {
                for(Index ij=0;ij<o*o;++ij)
                    std::copy_n(tau+(ij*v+p)*v,v,tau_buf+ij*v);
                for(Index i=0;i<o;++i) {
                    std::fill_n(dense,v*v,0.);
                    for(Index b=0;b<v;++b)for(Index a=0;a<v;++a) {
                        double sum=0.;
                        for(Index j=0;j<o;++j)
                            sum+=theta[((j*v+p)*o+i)*v+b]*t1bar[j*v+a];
                        dense[b*v+a]+=sum;
                    }
                    for(Index a=0;a<v;++a)for(Index c=0;c<v;++c) {
                        double sum=0.;
                        for(Index j=0;j<o;++j)sum+=wbar[((p*o+i)*o+j)*v+a]*t1[j*v+c];
                        dense[a*v+c]+=sum;
                    }
                    for(Index ij=0;ij<o*o;++ij)for(Index b=0;b<v;++b)
                        tmp_buf[ij*v+b]=tmpbar[(ij*v+b)*o+i];
                    // Each worker owns a slab. No nested OpenMP GEMM here.
                    char n='N',t='T';double one=1.;int m=nv,k=no2;
                    blas(&n,&t,&m,&m,&k,&one,tmp_buf,&m,tau_buf,&m,&one,dense,&m);
                    for(Index c=0;c<v;++c)for(Index a=0;a<v;++a)
                        dense[c*v+a]-=fvvbar[a*v+p]*t1[i*v+c];
                    for(Index a=0;a<v;++a)for(Index b=0;b<v;++b)
                        dense[a*v+b]+=2*t1[i*v+p]*fvvbar[a*v+b];
                    double* row=packed.p+(i*v+p)*pair;
                    for(Index a=0;a<v;++a)for(Index b=0;b<=a;++b)
                        row[a*(a+1)/2+b]=dense[a*v+b]+(a==b?0.:dense[b*v+a]);
                    for(Index j=0;j<o;++j) {
                        const double scale=-t1[j*v+p];
                        const double* src=woobar+(j*o+i)*pair;
                        #pragma omp simd
                        for(Index z=0;z<pair;++z)row[z]+=scale*src[z];
                    }
                }
            }
        }
        // Leave the outer region before the two factor contractions. The
        // lambda GEMM helper partitions each product over one OpenMP team.
        parallel_gemm('T','N',nrow,nq,npair,packed.p,npair,lvvin,npair,lovout,nrow);
        parallel_gemm('N','N',npair,nq,nrow,packed.p,npair,lovin,nrow,lvvout,npair);
    }
    return py::make_tuple(lovbar,lvvbar);
}

Array vvvv_vjp(const Array& taua,const Array& la,const Array& bara) {
    if(taua.ndim()!=3 || taua.shape(1)!=taua.shape(2) || la.ndim()!=3)
        throw py::value_error("vvvv pullback requires square virtual axes");
    const Index r=taua.shape(0),v=taua.shape(1),q=la.shape(0);
    require_shape(taua,{r,v,v});require_shape(bara,{r,v,v});require_shape(la,{q,v,v});
    const int nv=blas_dimension(v),vv=blas_dimension(v*v);
    const int twice=blas_dimension(2*r),nq=blas_dimension(q);
    Array result({q,v,v});
    if(!q || !v || !r) {
        std::fill_n(result.mutable_data(),result.size(),0.);
        return result;
    }
    const double *tau=taua.data(),*bar=bara.data();
    auto* l=const_cast<double*>(la.data());auto* out=result.mutable_data();
    {
        py::gil_scoped_release release;
        T c({2*r,v*v}),b({v,2*r}),slab({v,v*v});
        #pragma omp parallel for schedule(static)
        for(Index ij=0;ij<r;++ij)for(Index a=0;a<v;++a)for(Index z=0;z<v;++z) {
            c.p[(ij*v+a)*v+z]=bar[(ij*v+a)*v+z];
            c.p[((ij+r)*v+a)*v+z]=bar[(ij*v+z)*v+a];
        }
        // One v^3 slab is shared across iterations, not replicated per worker.
        for(Index x=0;x<v;++x) {
            #pragma omp parallel for schedule(static)
            for(Index a=0;a<v;++a)for(Index ij=0;ij<r;++ij) {
                b.p[a*2*r+ij]=tau[(ij*v+x)*v+a];
                b.p[a*2*r+r+ij]=tau[(ij*v+a)*v+x];
            }
            parallel_gemm('N','N',vv,nv,twice,c.p,vv,b.p,twice,slab.p,vv);
            // Reinterpret the contiguous slab as (v*v,v); output[:,x,:]
            // has a leading dimension of v*v in its column-major view.
            parallel_gemm('N','N',nv,nq,vv,slab.p,nv,l,vv,out+x*v,vv);
        }
    }
    return result;
}
} // namespace
} // namespace ccsd_native

void bind_integral_pullback(pybind11::module_& m) {
    using namespace ccsd_native;
    m.def("dfccsd_ovvv_vjp",&ovvv_vjp,py::arg("Lov").noconvert(),py::arg("Lvv").noconvert(),
        py::arg("t1").noconvert(),py::arg("theta").noconvert(),py::arg("tau").noconvert(),
        py::arg("t1_bar").noconvert(),py::arg("fvv_bar").noconvert(),py::arg("w_bar").noconvert(),
        py::arg("tmp_bar").noconvert(),py::arg("woo_bar").noconvert());
    m.def("dfccsd_vvvv_vjp",&vvvv_vjp,py::arg("tau").noconvert(),
          py::arg("Lvv_full").noconvert(),py::arg("G_bar").noconvert());
}
