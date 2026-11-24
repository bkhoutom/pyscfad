// Copyright 2026 The PySCFAD Authors (Apache-2.0).
// Dense real RCCSD solver operations. No triangular pack/unpack workspaces.
#include "residual.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace py=pybind11;
namespace {
using I=py::ssize_t;
using Input=py::array_t<double,0>;
using Array=ResidualArray;
struct View {
    const double* p;
    int rank;
    std::array<I,4> n{{1,1,1,1}},s{{0,0,0,0}};
    explicit View(const Input& a):p(a.data()),rank(a.ndim()) {
        if(rank<1 || rank>4 || reinterpret_cast<uintptr_t>(p)%alignof(double))
            throw std::invalid_argument("invalid tensor rank or alignment");
        for(int d=0;d<rank;++d) {
            n[d]=a.shape(d);
            if(a.strides(d)%sizeof(double))throw std::invalid_argument("unaligned tensor stride");
            s[d]=a.strides(d)/I(sizeof(double));
        }
    }
    double at(I i,I a) const {return p[i*s[0]+a*s[1]];}
    I row(I i,I j,I a) const {return i*s[0]+j*s[1]+a*s[2];}
};
void amplitudes(const View& a,const View& b,I o,I v) {
    if(a.rank!=2 || b.rank!=4 || a.n[0]!=o || a.n[1]!=v ||
       b.n[0]!=o || b.n[1]!=o || b.n[2]!=v || b.n[3]!=v)
        throw std::invalid_argument("inconsistent amplitude shapes");
}
// Fixed row partitions and a fixed final summation order make reductions
// independent of the dynamically selected OpenMP team size.
double weighted(const Input& a1,const Input& a2,const Input& b1,const Input& b2,bool difference) {
    View a(a1),x(a2),b(b1),y(b2);I o=a.n[0],v=a.n[1];
    amplitudes(a,x,o,v);amplitudes(b,y,o,v);
    std::vector<double> partial(o*o*v);
    double total=0.;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static) if(o*o*v*v>4096)
        for(I row=0;row<o*o*v;++row) {
            I aa=row%v,j=(row/v)%o,i=row/(v*o);
            I xo=x.row(i,j,aa),yo=y.row(i,j,aa);double sum=0.;
            #pragma omp simd reduction(+:sum)
            for(I bb=0;bb<v;++bb) {
                double xx=x.p[xo+bb*x.s[3]],yy=y.p[yo+bb*y.s[3]];
                double value=difference?(xx-yy)*(xx-yy):xx*yy;
                sum+=.5*value;
            }
            if(i==j) {
                double xx=x.p[xo+aa*x.s[3]],yy=y.p[yo+aa*y.s[3]];
                sum+=.5*(difference?(xx-yy)*(xx-yy):xx*yy);
            }
            partial[row]=sum;
        }
        for(I i=0;i<o;++i)for(I aa=0;aa<v;++aa) {
            double xx=a.at(i,aa),yy=b.at(i,aa);
            total+=difference?(xx-yy)*(xx-yy):xx*yy;
        }
        total=std::accumulate(partial.begin(),partial.end(),total);
    }
    return difference?std::sqrt(total):total;
}

py::tuple combine(const Input& coefficients,const py::list& singles,const py::list& doubles) {
    if(coefficients.ndim()!=1 || coefficients.size()<1 ||
       I(singles.size())!=coefficients.size() || singles.size()!=doubles.size())
        throw std::invalid_argument("inconsistent history/coefficient sizes");
    View coefficients_view(coefficients);
    std::vector<Input> owned1,owned2;
    std::vector<View> a,b;
    std::vector<double> c;
    for(I k=0;k<coefficients.size();++k) {
        if(!Input::check_(singles[k]) || !Input::check_(doubles[k]))
            throw py::type_error("dense history requires float64 arrays");
        owned1.push_back(py::reinterpret_borrow<Input>(singles[k]));
        owned2.push_back(py::reinterpret_borrow<Input>(doubles[k]));
        a.emplace_back(owned1.back());b.emplace_back(owned2.back());
        amplitudes(a.back(),b.back(),a[0].n[0],a[0].n[1]);
        c.push_back(coefficients_view.p[k*coefficients_view.s[0]]);
    }
    I o=a[0].n[0],v=a[0].n[1],count=I(a.size());
    Array out1(std::vector<I>{o,v}),out2(std::vector<I>{o,o,v,v});
    double* r1=out1.mutable_data();double* r2=out2.mutable_data();
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static) if(o*v>4096)
        for(I i=0;i<o;++i) {
            for(I aa=0;aa<v;++aa)r1[i*v+aa]=0.;
            for(I k=0;k<count;++k) {
                #pragma omp simd
                for(I aa=0;aa<v;++aa)r1[i*v+aa]+=c[k]*a[k].at(i,aa);
            }
        }
        #pragma omp parallel for schedule(static) if(o*o*v*v>4096)
        for(I row=0;row<o*o*v;++row) {
            I aa=row%v,j=(row/v)%o,i=row/(v*o);
            double* dst=r2+row*v;
            for(I bb=0;bb<v;++bb)dst[bb]=0.;
            for(I k=0;k<count;++k) {
                const double* src=b[k].p+b[k].row(i,j,aa);I stride=b[k].s[3];
                #pragma omp simd
                for(I bb=0;bb<v;++bb)dst[bb]+=c[k]*src[bb*stride];
            }
        }
    }
    return py::make_tuple(out1,out2);
}

double energy(const Input& t1,const Input& t2,const Input& fock,const Input& ovvo) {
    View a(t1),b(t2),f(fock),g(ovvo);I o=a.n[0],v=a.n[1];
    amplitudes(a,b,o,v);
    if(f.rank!=2 || f.n[0]!=o+v || f.n[1]!=o+v || g.rank!=4 ||
       g.n[0]!=o || g.n[1]!=v || g.n[2]!=v || g.n[3]!=o)
        throw std::invalid_argument("inconsistent energy integral shapes");
    std::vector<double> partial(o*o*v);
    double total=0.;
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static) if(o*o*v*v>4096)
        for(I row=0;row<o*o*v;++row) {
            I aa=row%v,j=(row/v)%o,i=row/(v*o);
            I x=b.row(i,j,aa),y=b.row(j,i,aa),z=i*g.s[0]+aa*g.s[1]+j*g.s[3];
            double ai=a.at(i,aa),aj=a.at(j,aa),sum=0.;
            #pragma omp simd reduction(+:sum)
            for(I bb=0;bb<v;++bb) {
                double tau=b.p[x+bb*b.s[3]]+ai*a.at(j,bb);
                double exchange=b.p[y+bb*b.s[3]]+aj*a.at(i,bb);
                sum+=(2*tau-exchange)*g.p[z+bb*g.s[2]];
            }
            partial[row]=sum;
        }
        for(I i=0;i<o;++i)for(I aa=0;aa<v;++aa)total+=2*f.at(i,o+aa)*a.at(i,aa);
        total=std::accumulate(partial.begin(),partial.end(),total);
    }
    return total;
}
}

void bind_solver(py::module_& m) {
    m.def("weighted_dot",[](const Input& a,const Input& b,const Input& c,const Input& d){return weighted(a,b,c,d,false);},
        py::arg("a1").noconvert(),py::arg("a2").noconvert(),py::arg("b1").noconvert(),py::arg("b2").noconvert());
    m.def("difference_norm",[](const Input& a,const Input& b,const Input& c,const Input& d){return weighted(a,b,c,d,true);},
        py::arg("a1").noconvert(),py::arg("a2").noconvert(),py::arg("b1").noconvert(),py::arg("b2").noconvert());
    m.def("dense_combine",&combine,py::arg("coefficients").noconvert(),py::arg("singles"),py::arg("doubles"));
    m.def("ccsd_energy",&energy,py::arg("t1").noconvert(),py::arg("t2").noconvert(),py::arg("fock").noconvert(),py::arg("ovvo").noconvert());
}
