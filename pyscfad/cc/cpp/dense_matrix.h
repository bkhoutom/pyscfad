// Copyright 2026 The PySCFAD Authors (Apache-2.0).
#pragma once
#include "tensor.h"
#include <cmath>
#include <cstring>

namespace ccsd_native { namespace dense {
struct Matrix;
struct View {const Matrix* matrix; bool transpose=false; double scale=1.;};
struct Product {View left,right;};
struct Scaled {const Matrix* matrix;double scale;};
struct Zero {};
constexpr Zero zero{};

// Column-major owned buffers or strict borrowed views. Copying owns a copy;
// assignment to a borrowed view writes its destination. Transposes in GEMMs
// are descriptors, never full-sized packing temporaries.
struct Matrix {
    size_t n_rows=0,n_cols=0,n_elem=0;
    std::unique_ptr<double[]> owner;
    double* data=nullptr;
    bool borrowed=false;
    Matrix()=default;
    explicit Matrix(size_t n) {set_size(n);}
    Matrix(size_t r,size_t c) {set_size(r,c);}
    Matrix(size_t r,size_t c,Zero) {zeros(r,c);}
    Matrix(double* p,size_t r,size_t c,bool copy,bool):n_rows(r),n_cols(c),n_elem(r*c),data(p),borrowed(!copy) {
        if(copy) {owner.reset(new double[n_elem]);data=owner.get();std::copy_n(p,n_elem,data);}
    }
    Matrix(double* p,size_t n,bool copy,bool strict):Matrix(p,n,1,copy,strict) {}
    Matrix(const Matrix& x) {operator=(x);}
    Matrix(Matrix&& x) {operator=(std::move(x));}
    Matrix(View x) {operator=(x);}
    Matrix(Product x) {operator=(x);}
    Matrix(Scaled x) {operator=(x);}
    Matrix& operator=(const Matrix& x) {
        if(this!=&x) {set_size(x.n_rows,x.n_cols);std::copy_n(x.data,n_elem,data);}return *this;
    }
    Matrix& operator=(Matrix&& x) {
        if(this==&x)return *this;
        if(borrowed)return operator=(static_cast<const Matrix&>(x));
        owner=std::move(x.owner);data=x.data;borrowed=x.borrowed;
        n_rows=x.n_rows;n_cols=x.n_cols;n_elem=x.n_elem;
        x.data=nullptr;x.n_rows=x.n_cols=x.n_elem=0;x.borrowed=false;return *this;
    }
    void set_size(size_t n) {set_size(n,1);}
    void set_size(size_t r,size_t c) {
        if(r && c>size_t(PTRDIFF_MAX)/sizeof(double)/r)throw std::overflow_error("native matrix size overflow");
        if(borrowed && (r!=n_rows || c!=n_cols))throw std::invalid_argument("cannot resize a native matrix view");
        if(r*c!=n_elem) {owner.reset(new double[r*c]);data=owner.get();}
        n_rows=r;n_cols=c;n_elem=r*c;
    }
    double* memptr() {return data;}
    const double* memptr() const {return data;}
    double* colptr(size_t j) {return data+j*n_rows;}
    const double* colptr(size_t j) const {return data+j*n_rows;}
    double& operator[](size_t i) {return data[i];}
    const double& operator[](size_t i) const {return data[i];}
    double& operator()(size_t i) {return data[i];}
    const double& operator()(size_t i) const {return data[i];}
    double& operator()(size_t i,size_t j) {return data[i+j*n_rows];}
    const double& operator()(size_t i,size_t j) const {return data[i+j*n_rows];}
    void zeros() {std::fill_n(data,n_elem,0.);}
    void zeros(size_t n) {set_size(n);zeros();}
    void zeros(size_t r,size_t c) {set_size(r,c);zeros();}
    void eye(size_t r,size_t c) {zeros(r,c);for(size_t i=0;i<std::min(r,c);++i)(*this)(i,i)=1.;}
    void reset() {*this=Matrix();}
    bool is_empty() const {return !n_elem;}
    bool is_diagmat() const {
        for(size_t j=0;j<n_cols;++j)for(size_t i=0;i<n_rows;++i)if(i!=j && (*this)(i,j)!=0.)return false;
        return true;
    }
    Matrix diag() const {Matrix r(std::min(n_rows,n_cols));for(size_t i=0;i<r.n_elem;++i)r[i]=(*this)(i,i);return r;}
    double min() const {return *std::min_element(data,data+n_elem);}
    double max() const {return *std::max_element(data,data+n_elem);}
    bool is_finite() const {for(size_t i=0;i<n_elem;++i)if(!std::isfinite(data[i]))return false;return true;}
    View t() const {return {this,true,1.};}
    operator View() const {return {this,false,1.};}
    Matrix& operator=(View x) {
        if(x.matrix==this) {Matrix tmp(x);return operator=(std::move(tmp));}
        const auto& a=*x.matrix;set_size(x.transpose?a.n_cols:a.n_rows,x.transpose?a.n_rows:a.n_cols);
        for(size_t j=0;j<n_cols;++j)for(size_t i=0;i<n_rows;++i)(*this)(i,j)=x.scale*(x.transpose?a(j,i):a(i,j));
        return *this;
    }
    Matrix& operator=(Scaled x) {set_size(x.matrix->n_rows,x.matrix->n_cols);for(size_t i=0;i<n_elem;++i)data[i]=x.scale*x.matrix->data[i];return *this;}
    Matrix& operator+=(const Matrix& x) {return add(x,1.);}
    Matrix& operator-=(const Matrix& x) {return add(x,-1.);}
    Matrix& add(const Matrix& x,double alpha) {
        if(n_rows!=x.n_rows || n_cols!=x.n_cols)throw std::invalid_argument("native matrix addition shape mismatch");
        for(size_t i=0;i<n_elem;++i)data[i]+=alpha*x.data[i];return *this;
    }
    Matrix& operator=(Product x);
    Matrix& operator+=(Product x);
    Matrix& operator-=(Product x);
};
inline int blas_int(size_t n) {if(n>INT_MAX)throw std::overflow_error("native matrix exceeds SciPy BLAS range");return int(n);}
inline void gemm(View a,View b,Matrix& c,double alpha=1.,double beta=0.) {
    const size_t m=a.transpose?a.matrix->n_cols:a.matrix->n_rows;
    const size_t k=a.transpose?a.matrix->n_rows:a.matrix->n_cols;
    const size_t n=b.transpose?b.matrix->n_rows:b.matrix->n_cols;
    if(k!=(b.transpose?b.matrix->n_cols:b.matrix->n_rows))throw std::invalid_argument("native GEMM shape mismatch");
    if(beta!=0. && (c.n_rows!=m || c.n_cols!=n))throw std::invalid_argument("native GEMM output shape mismatch");
    c.set_size(m,n);if(!m || !n)return;
    if(!k) {for(size_t i=0;i<c.n_elem;++i)c[i]=beta==0.?0.:beta*c[i];return;}
    char ta=a.transpose?'T':'N',tb=b.transpose?'T':'N';
    int bm=blas_int(m),bn=blas_int(n),bk=blas_int(k),lda=blas_int(a.matrix->n_rows),ldb=blas_int(b.matrix->n_rows),ldc=bm;
    alpha*=a.scale*b.scale;
    if(omp_in_parallel())blas(&ta,&tb,&bm,&bn,&bk,&alpha,a.matrix->data,&lda,b.matrix->data,&ldb,&beta,c.data,&ldc);
    else parallel_gemm(ta,tb,bm,bn,bk,a.matrix->data,lda,b.matrix->data,ldb,c.data,ldc,alpha,beta);
}
inline Matrix& Matrix::operator=(Product x) {
    if(data && (data==x.left.matrix->data || data==x.right.matrix->data)) {
        Matrix tmp;gemm(x.left,x.right,tmp);return operator=(std::move(tmp));
    }
    gemm(x.left,x.right,*this);return *this;
}
inline Matrix& Matrix::operator+=(Product x) {gemm(x.left,x.right,*this,1.,1.);return *this;}
inline Matrix& Matrix::operator-=(Product x) {gemm(x.left,x.right,*this,-1.,1.);return *this;}
inline Product operator*(View a,View b) {return {a,b};}
inline Matrix operator*(Product a,View b) {Matrix tmp(a),out;gemm(tmp,b,out);return out;}
inline View operator-(View a) {a.scale=-a.scale;return a;}
inline Scaled operator*(double alpha,const Matrix& x) {return {&x,alpha};}
inline Matrix operator+(Matrix a,const Matrix& b) {a+=b;return a;}
inline Matrix operator-(const Matrix& a,double shift) {Matrix b(a);for(size_t i=0;i<b.n_elem;++i)b[i]-=shift;return b;}
inline Matrix exp(const Matrix& a) {Matrix b(a);for(size_t i=0;i<b.n_elem;++i)b[i]=std::exp(b[i]);return b;}
inline Matrix diagmat(const Matrix& v) {Matrix a(v.n_elem,v.n_elem,zero);for(size_t i=0;i<v.n_elem;++i)a(i,i)=v[i];return a;}
inline Matrix eye(size_t r,size_t c) {Matrix a;a.eye(r,c);return a;}
inline double dot(const Matrix& a,const Matrix& b) {
    if(a.n_elem!=b.n_elem)throw std::invalid_argument("native dot shape mismatch");
    double value=0.;for(size_t i=0;i<a.n_elem;++i)value+=a[i]*b[i];return value;
}
} }
