// Included in the dressing translation unit to share bounded panel helpers.
// Diagonal virtual factors need neither gathered panels nor dense GEMMs.
// This applies equally to a tensor cotangent; the matrix cotangent is still
// computed in full by stable_matrix_bar before that scaling in the pullback.
Array stable_scale_axis(const Array& input,size_t axis,const dense::Matrix& matrix) {
    const size_t n=size_t(input.shape(axis));
    if(matrix.n_rows!=n || matrix.n_cols!=n)
        throw py::value_error("stable triples: inconsistent diagonal transform");
    const dense::Matrix factors=matrix.diag();
    Array output(shape(input));
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(input.ndim());++d) inner*=size_t(input.shape(d));
    const size_t blocks=size_t(input.size())/inner;
    const double* source=input.data(); double* dest=output.mutable_data();
    py::gil_scoped_release release;
    #pragma omp parallel for schedule(static)
    for(size_t block=0;block<blocks;++block) {
        const double factor=factors[block%n];
        const size_t start=block*inner;
        #pragma omp simd
        for(size_t r=0;r<inner;++r) dest[start+r]=factor*source[start+r];
    }
    return output;
}

// In the original occupied LIS, U^T=[I_S,0]. Copy its selected rows and
// scatter their cotangents instead of multiplying by a dense projector.
// The independent matrix cotangent is still evaluated in full below.
Array stable_selected_axis(const Array& input,size_t axis,size_t no,size_t ns,
                           bool adjoint) {
    Shape dims=shape(input);
    const size_t ni=adjoint ? ns : no,nout=adjoint ? no : ns;
    if(size_t(dims[axis])!=ni)
        throw py::value_error("stable triples: inconsistent selected occupied axis");
    dims[axis]=py::ssize_t(nout);
    Array output(dims);
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(input.ndim());++d) inner*=size_t(input.shape(d));
    const size_t outer=size_t(input.size())/(ni*inner);
    const double* source=input.data();double* dest=output.mutable_data();
    py::gil_scoped_release release;
    if(adjoint) {
        #pragma omp parallel for simd schedule(static) if(output.size()>=32768)
        for(py::ssize_t i=0;i<output.size();++i) dest[i]=0.;
    }
    // Split a long boundary-axis copy too; occupied selection otherwise
    // exposes only one or two loop iterations for a large ovvv tensor.
    constexpr size_t panel=262144;
    const size_t selected=ns*inner,panels=(selected+panel-1)/panel;
    #pragma omp parallel for schedule(static) if(outer*selected>=32768)
    for(size_t task=0;task<outer*panels;++task) {
        const size_t block=task/panels,first=(task%panels)*panel;
        std::copy_n(source+block*ni*inner+first,std::min(panel,selected-first),
                    dest+block*nout*inner+first);
    }
    return output;
}

Array stable_transform(const Array& input,size_t axis,const dense::Matrix& matrix,
                       bool optimized) {
    Shape dims=shape(input);
    const size_t ni=size_t(dims[axis]),no=matrix.n_rows;
    if(matrix.n_cols!=ni) throw py::value_error("stable triples: inconsistent transform");
    dims[axis]=py::ssize_t(no);
    Array output(dims);
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(input.ndim());++d) inner*=size_t(input.shape(d));
    const size_t rows=size_t(input.size())/ni;
    if(optimized && (inner==1 || inner==rows)) {
        // Boundary axes are already contiguous matrices in one orientation.
        // Views avoid both packing buffers and the element-sized memcpy runs
        // of a last-axis gather. One BLAS call owns its configured thread team.
        py::gil_scoped_release release;
        if(inner==1) {
            const dense::Matrix source(const_cast<double*>(input.data()),ni,rows,false,true);
            dense::Matrix target(output.mutable_data(),no,rows,false,true);
            target=matrix*source;
        } else {
            const dense::Matrix source(const_cast<double*>(input.data()),rows,ni,false,true);
            dense::Matrix target(output.mutable_data(),rows,no,false,true);
            target=source*matrix.t();
        }
        return output;
    }
    const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(sizeof(double)*std::max(ni,no)));
    std::exception_ptr error;
    py::gil_scoped_release release;
    #pragma omp parallel
    {
        dense::Matrix packed,result;
        #pragma omp for schedule(dynamic)
        for(size_t first=0;first<rows;first+=panel) {
            try {
                const size_t count=std::min(panel,rows-first);
                gather(input.data(),packed,ni,inner,first,count);
                multiply(packed,matrix,result,true);
                scatter(output.mutable_data(),result,no,inner,first,count);
            } catch(...) {
                #pragma omp critical(stable_triples_error)
                { if(!error) error=std::current_exception(); }
            }
        }
    }
    if(error) std::rethrow_exception(error);
    return output;
}

dense::Matrix stable_matrix_bar(const Array& raw,const Array& bar,size_t axis,
                            bool optimized) {
    const size_t ni=size_t(raw.shape(axis)),no=size_t(bar.shape(axis));
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(raw.ndim());++d) inner*=size_t(raw.shape(d));
    const size_t rows=size_t(raw.size())/ni;
    if(optimized && (inner==1 || inner==rows)) {
        // Full rectangular matrix bars are required even for diagonal
        // propagators. These views have the same orientation as the forward.
        dense::Matrix result;
        py::gil_scoped_release release;
        if(inner==1) {
            const dense::Matrix source(const_cast<double*>(raw.data()),ni,rows,false,true);
            const dense::Matrix adjoint(const_cast<double*>(bar.data()),no,rows,false,true);
            result=adjoint*source.t();
        } else {
            const dense::Matrix source(const_cast<double*>(raw.data()),rows,ni,false,true);
            const dense::Matrix adjoint(const_cast<double*>(bar.data()),rows,no,false,true);
            result=adjoint.t()*source;
        }
        return result;
    }
    const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(sizeof(double)*std::max(ni,no)));
    std::vector<dense::Matrix> local(static_cast<size_t>(omp_get_max_threads()));
    for(auto& gram:local) gram.zeros(no,ni);
    std::exception_ptr error;
    py::gil_scoped_release release;
    #pragma omp parallel
    {
        dense::Matrix source,adjoint;
        auto& gram=local[size_t(omp_get_thread_num())];
        #pragma omp for schedule(dynamic)
        for(size_t first=0;first<rows;first+=panel) {
            try {
                const size_t count=std::min(panel,rows-first);
                gather(raw.data(),source,ni,inner,first,count);
                gather(bar.data(),adjoint,no,inner,first,count);
                if(true && std::min(no,ni)>=16) gram+=adjoint.t()*source;
                else for(size_t i=0;i<no;++i) for(size_t j=0;j<ni;++j) {
                    const double* a=adjoint.colptr(i); const double* b=source.colptr(j);
                    double sum=0.;
                    #pragma omp simd reduction(+:sum)
                    for(size_t r=0;r<count;++r) sum+=a[r]*b[r];
                    gram(i,j)+=sum;
                }
            } catch(...) {
                #pragma omp critical(stable_triples_error)
                { if(!error) error=std::current_exception(); }
            }
        }
    }
    if(error) std::rethrow_exception(error);
    dense::Matrix result(no,ni,dense::zero);
    for(const auto& gram:local) result+=gram;
    return result;
}

class StableDressing {
    std::vector<Array> inputs_;
    Spectrum occupied_,virtual_;
    dense::Matrix virtual_diagonal_,occupied_diagonal_;
    double occupied_offdiag_norm_=0.;
    double virtual_offdiag_norm_=0.;
    dense::Matrix u_;
    double shift_=0.,cached_node_=0.;
    bool full_external_=false,selected_occupied_frame_=false,mrcc_partition_=false;
    std::vector<ccsd_native::triples::triples_tensor_spec> specs_;
    std::vector<std::vector<Axis>> plans_;
    std::vector<std::vector<Array>> prefixes_;
    std::vector<Array> graph_prefixes_;
    bool graph_cached_=false;
    size_t tape_bytes_=0,cached_bytes_=0,peak_bytes_=0,hits_=0,rebuilds_=0;
    double external_time(double node) const { return full_external_ && !mrcc_partition_ ? node : node/2; }
    std::vector<dense::Matrix> matrices(double node,bool diagonal_virtual) const {
        if(!std::isfinite(node) || node<0) throw py::value_error("stable triples: node must be finite and nonnegative");
        // The common energy shift cancels in every sampled product. Both
        // ordinary exponentials and O^2 are bounded by one for a positive gap.
        if(full_external_ && node==0.)
            return {dense::eye(u_.n_rows,u_.n_rows),
                    dense::eye(virtual_diagonal_.n_elem,virtual_diagonal_.n_elem),u_.t()};
        dense::Matrix v;
        if(diagonal_virtual) v=dense::diagmat(dense::exp((-external_time(node))*(virtual_diagonal_-shift_)));
        else v=exponential(virtual_,-external_time(node),shift_);
        dense::Matrix o;
        if(occupied_diagonal_at(node))
            o=dense::diagmat(dense::exp(external_time(node)*(occupied_diagonal_-shift_)));
        else o=exponential(occupied_,external_time(node),shift_);
        std::vector<dense::Matrix> result={std::move(o),
                                       std::move(v),u_.t()};
        if(!full_external_) result.emplace_back(u_.t()*exponential(occupied_,node,shift_));
        return result;
    }
    Array transform(const Array& input,const Axis& axis,const dense::Matrix& matrix,
                    bool diagonal_virtual) const {
        if(axis.matrix==2 && selected_occupied_frame_)
            return stable_selected_axis(input,axis.axis,u_.n_rows,u_.n_cols,
                                        matrix.n_rows==u_.n_rows);
        // The matrix bar is computed in full before this tensor-only shortcut.
        if(axis.matrix==0 && matrix.is_diagmat())
            return stable_scale_axis(input,axis.axis,matrix);
        return diagonal_virtual && axis.matrix==1
            ? stable_scale_axis(input,axis.axis,matrix)
            : stable_transform(input,axis.axis,matrix,full_external_);
    }
#include "triples_graph.inc"
public:
    explicit StableDressing(py::tuple inputs,bool full_external=false,bool mrcc_partition=false)
        :full_external_(full_external),mrcc_partition_(mrcc_partition) {
        if(!mrcc_partition_ || !full_external_)
            throw py::value_error("MRCC partition requires the shared transform graph");
        if(inputs.size()!=9) throw py::value_error("stable triples: expected nine integral inputs");
        for(auto value:inputs) {
            auto a=Array::ensure(value);
            if(!a) throw py::value_error("stable triples: expected real arrays");
            for(py::ssize_t i=0;i<a.size();++i)
                if(!std::isfinite(a.data()[i])) throw py::value_error("stable triples: nonfinite input");
            inputs_.push_back(std::move(a));
        }
        if(inputs_[0].ndim()!=2 || inputs_[8].ndim()!=2)
            throw py::value_error("stable triples: invalid input rank");
        const auto no=inputs_[0].shape(0),nv=inputs_[0].shape(1),ns=inputs_[8].shape(1);
        if(no<=0 || nv<=0 || ns<=0 || ns>no) throw py::value_error("stable triples: invalid dimensions");
        check_shape(inputs_[1],{no,no,nv,nv}); check_shape(inputs_[2],{no,nv,nv,nv});
        check_shape(inputs_[3],{no,nv,no,no}); check_shape(inputs_[4],{no,nv,no,nv});
        check_shape(inputs_[5],{no,nv}); check_shape(inputs_[6],{no,no});
        check_shape(inputs_[7],{nv,nv}); check_shape(inputs_[8],{no,ns});
        occupied_=spectrum(inputs_[6]);
        occupied_diagonal_.set_size(size_t(no));
        for(size_t i=0;i<size_t(no);++i) {
            occupied_diagonal_[i]=inputs_[6].data()[i*size_t(no)+i];
            double row=0.;
            for(size_t j=0;j<size_t(no);++j) if(i!=j)
                row+=std::abs(.5*inputs_[6].data()[i*size_t(no)+j]
                             +.5*inputs_[6].data()[j*size_t(no)+i]);
            occupied_offdiag_norm_=std::max(occupied_offdiag_norm_,row);
        }
        virtual_diagonal_.set_size(size_t(nv));
        for(size_t i=0;i<size_t(nv);++i) {
            virtual_diagonal_[i]=inputs_[7].data()[i*size_t(nv)+i];
            double row=0.;
            for(size_t j=0;j<size_t(nv);++j) if(i!=j)
                row+=std::abs(.5*inputs_[7].data()[i*size_t(nv)+j]
                             +.5*inputs_[7].data()[j*size_t(nv)+i]);
            virtual_offdiag_norm_=std::max(virtual_offdiag_norm_,row);
        }
        if(virtual_offdiag_norm_==0.) {
            // No eigensolve or orbital reordering is needed for diagonal Fvv.
            virtual_.values=virtual_diagonal_;
            virtual_.vectors.eye(size_t(nv),size_t(nv));
        } else virtual_=spectrum(inputs_[7]);
        if(virtual_.values.min()<=occupied_.values.max()) throw py::value_error("stable triples: requires positive gap");
        shift_=.5*occupied_.values.max()+.5*virtual_.values.min();
        u_.set_size(size_t(no),size_t(ns));
        for(size_t i=0;i<size_t(no);++i) for(size_t s=0;s<size_t(ns);++s)
            u_(i,s)=inputs_[8].data()[i*size_t(ns)+s];
        double selector_error=0.;
        for(size_t i=0;i<size_t(no);++i) {
            double row=0.;
            for(size_t s=0;s<size_t(ns);++s)
                row+=std::abs(u_(i,s)-double(i==s));
            selector_error=std::max(selector_error,row);
        }
        selected_occupied_frame_=selector_error<=32*std::numeric_limits<double>::epsilon();
        specs_=ccsd_native::triples::triples_layout(size_t(no),size_t(nv),size_t(ns)).tensors;
        // MRCC: every selected external leg has U^T O(t/2), on both sides.
        if(mrcc_partition_) for(auto& spec:specs_)
            for(auto& transform:spec.transform) if(transform==3) transform=4;
        for(const auto& spec:specs_) {
            std::vector<Axis> plan;
            // Project first, reducing the size of every subsequent temporary.
            for(size_t d=0;d<spec.rank;++d) if(spec.transform[d]>=3)
                plan.push_back({d,spec.transform[d]-1});
            for(size_t d=0;d<spec.rank;++d) if(spec.transform[d]>0 && spec.transform[d]<3)
                plan.push_back({d,spec.transform[d]-1});
            auto dims=shape(inputs_[spec.source]);
            for(size_t k=0;k+1<plan.size();++k) {
                if(plan[k].matrix>=2) dims[plan[k].axis]=ns;
                size_t n=1; for(auto d:dims) n*=size_t(d);
                tape_bytes_+=n*sizeof(double);
            }
            plans_.push_back(std::move(plan));
        }
    }
    void clear_cache() {
        prefixes_.clear(); graph_prefixes_.clear(); graph_cached_=false; cached_bytes_=0;
    }
    size_t cache_bytes() const { return cached_bytes_; }
    size_t cache_hits() const { return hits_; }
    size_t peak_cache_bytes() const { return peak_bytes_; }
    size_t cache_rebuilds() const { return rebuilds_; }
    bool selected_occupied_frame() const { return selected_occupied_frame_; }
    bool occupied_diagonal_at(double node) const {
        if(!std::isfinite(node) || node<0)
            throw py::value_error("stable triples: node must be finite and nonnegative");
        // The shifted occupied generator is negative definite. The same
        // Duhamel error bound as for virtual factors permits roundoff only.
        return external_time(node)*occupied_offdiag_norm_<=32*std::numeric_limits<double>::epsilon();
    }
    bool virtual_diagonal_at(double node) const {
        if(!std::isfinite(node) || node<0)
            throw py::value_error("stable triples: node must be finite and nonnegative");
        // For symmetric Fvv, the maximum absolute off-diagonal row sum bounds
        // the spectral norm of the omitted residual. The Duhamel bound for
        // exp(-tau(Fvv-mu I)) is <= tau||residual||
        // (tau=t/2 for shared dressing, tau=t for one-sided dressing), since both the full and
        // diagonal shifted virtual blocks are positive definite. Only a
        // roundoff-sized absolute primal error is permitted at this node.
        return external_time(node)*virtual_offdiag_norm_<=32*std::numeric_limits<double>::epsilon();
    }
    py::tuple prepare(double node,size_t cache_limit_bytes=0) {
        return graph_prepare(node,cache_limit_bytes);
    }
    py::tuple pullback(double node,py::tuple cotangents,size_t cache_limit_bytes=0) {
        return graph_pullback(node,cotangents,cache_limit_bytes);
    }
};
