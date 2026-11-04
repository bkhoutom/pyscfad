// Included in the dressing translation unit to share bounded panel helpers.
Array stable_transform(const Array& input,size_t axis,const arma::mat& matrix) {
    Shape dims=shape(input);
    const size_t ni=size_t(dims[axis]),no=matrix.n_rows;
    if(matrix.n_cols!=ni) throw py::value_error("stable triples: inconsistent transform");
    dims[axis]=py::ssize_t(no);
    Array output(dims);
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(input.ndim());++d) inner*=size_t(input.shape(d));
    const size_t rows=size_t(input.size())/ni;
    const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(sizeof(double)*std::max(ni,no)));
    std::exception_ptr error;
    py::gil_scoped_release release;
    #pragma omp parallel
    {
        LocalBlas local_blas;
        arma::mat packed,result;
        #pragma omp for schedule(dynamic)
        for(size_t first=0;first<rows;first+=panel) {
            try {
                const size_t count=std::min(panel,rows-first);
                gather(input.data(),packed,ni,inner,first,count);
                multiply(packed,matrix,result,bool(local_blas.setter));
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

arma::mat stable_matrix_bar(const Array& raw,const Array& bar,size_t axis) {
    const size_t ni=size_t(raw.shape(axis)),no=size_t(bar.shape(axis));
    size_t inner=1;
    for(size_t d=axis+1;d<size_t(raw.ndim());++d) inner*=size_t(raw.shape(d));
    const size_t rows=size_t(raw.size())/ni;
    const size_t panel=std::max(size_t(1),size_t(2*1024*1024)/(sizeof(double)*std::max(ni,no)));
    std::vector<arma::mat> local(static_cast<size_t>(omp_get_max_threads()));
    for(auto& gram:local) gram.zeros(no,ni);
    std::exception_ptr error;
    py::gil_scoped_release release;
    #pragma omp parallel
    {
        LocalBlas local_blas;
        arma::mat source,adjoint;
        auto& gram=local[size_t(omp_get_thread_num())];
        #pragma omp for schedule(dynamic)
        for(size_t first=0;first<rows;first+=panel) {
            try {
                const size_t count=std::min(panel,rows-first);
                gather(raw.data(),source,ni,inner,first,count);
                gather(bar.data(),adjoint,no,inner,first,count);
                if(local_blas.setter && std::min(no,ni)>=16) gram+=adjoint.t()*source;
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
    arma::mat result(no,ni,arma::fill::zeros);
    for(const auto& gram:local) result+=gram;
    return result;
}

class StableDressing {
    std::vector<Array> inputs_;
    Spectrum occupied_,virtual_;
    arma::mat u_;
    double shift_=0.,cached_node_=0.;
    std::vector<libgmbpt::qchem::triples_tensor_spec> specs_;
    std::vector<std::vector<Axis>> plans_;
    std::vector<std::vector<Array>> prefixes_;
    size_t tape_bytes_=0,cached_bytes_=0,peak_bytes_=0,hits_=0,rebuilds_=0;
    std::vector<arma::mat> matrices(double node) const {
        if(!std::isfinite(node) || node<0) throw py::value_error("stable triples: node must be finite and nonnegative");
        // The common energy shift cancels in every sampled product. Both
        // ordinary exponentials and O^2 are bounded by one for a positive gap.
        return {exponential(occupied_,node/2,shift_),exponential(virtual_,-node/2,shift_),
                u_.t(),u_.t()*exponential(occupied_,node,shift_)};
    }
public:
    explicit StableDressing(py::tuple inputs) {
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
        occupied_=spectrum(inputs_[6]); virtual_=spectrum(inputs_[7]);
        if(virtual_.values.min()<=occupied_.values.max()) throw py::value_error("stable triples: requires positive gap");
        shift_=.5*occupied_.values.max()+.5*virtual_.values.min();
        u_.set_size(size_t(no),size_t(ns));
        for(size_t i=0;i<size_t(no);++i) for(size_t s=0;s<size_t(ns);++s)
            u_(i,s)=inputs_[8].data()[i*size_t(ns)+s];
        specs_=libgmbpt::qchem::triples_layout(size_t(no),size_t(nv),size_t(ns)).tensors;
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
    void clear_cache() { prefixes_.clear(); cached_bytes_=0; }
    size_t cache_bytes() const { return cached_bytes_; }
    size_t cache_hits() const { return hits_; }
    size_t peak_cache_bytes() const { return peak_bytes_; }
    size_t cache_rebuilds() const { return rebuilds_; }
    py::tuple prepare(double node,size_t cache_limit_bytes=0) {
        clear_cache();
        const auto mats=matrices(node);
        const bool retain=cache_limit_bytes>0 && tape_bytes_<=cache_limit_bytes;
        if(retain) { prefixes_.resize(specs_.size()); cached_node_=node; }
        py::tuple out(specs_.size());
        try {
            for(size_t i=0;i<specs_.size();++i) {
                Array value=inputs_[specs_[i].source];
                for(size_t k=0;k<plans_[i].size();++k) {
                    const auto a=plans_[i][k];
                    value=stable_transform(value,a.axis,mats[a.matrix]);
                    if(retain && k+1<plans_[i].size()) prefixes_[i].push_back(value);
                }
                out[i]=std::move(value);
            }
        } catch(...) { clear_cache(); throw; }
        cached_bytes_=retain ? tape_bytes_ : 0;
        peak_bytes_=std::max(peak_bytes_,cached_bytes_);
        return out;
    }
    py::tuple pullback(double node,py::tuple cotangents,size_t cache_limit_bytes=0) {
        if(cotangents.size()!=specs_.size()) { clear_cache(); throw py::value_error("stable triples: invalid cotangent count"); }
        if((prefixes_.empty() || cached_node_!=node) && cache_limit_bytes>0 && tape_bytes_<=cache_limit_bytes) {
            prepare(node,cache_limit_bytes); ++rebuilds_;
        }
        auto prefixes=std::move(prefixes_);
        const bool cached=!prefixes.empty() && cached_node_==node;
        clear_cache(); hits_+=size_t(cached);
        const auto mats=matrices(node);
        std::vector<arma::mat> mbars;
        for(const auto& m:mats) mbars.emplace_back(m.n_rows,m.n_cols,arma::fill::zeros);
        py::tuple out(inputs_.size());
        for(size_t i=0;i<inputs_.size();++i) {
            Array zero(shape(inputs_[i])); std::fill(zero.mutable_data(),zero.mutable_data()+zero.size(),0.);
            out[i]=std::move(zero);
        }
        for(size_t i=0;i<specs_.size();++i) {
            const auto& spec=specs_[i];
            auto bar=Array::ensure(cotangents[i]);
            if(!bar) throw py::value_error("stable triples: invalid cotangent");
            check_shape(bar,Shape(spec.dims.begin(),spec.dims.begin()+spec.rank));
            for(size_t k=plans_[i].size();k-->0;) {
                Array raw=inputs_[spec.source];
                if(k && cached) raw=prefixes[i][k-1];
                else for(size_t j=0;j<k;++j) {
                    const auto a=plans_[i][j];
                    raw=stable_transform(raw,a.axis,mats[a.matrix]);
                }
                const auto a=plans_[i][k];
                mbars[a.matrix]+=stable_matrix_bar(raw,bar,a.axis);
                bar=stable_transform(bar,a.axis,mats[a.matrix].t());
            }
            auto sum=out[spec.source].cast<Array>();
            for(py::ssize_t j=0;j<sum.size();++j) sum.mutable_data()[j]+=bar.data()[j];
        }
        const arma::mat o2=exponential(occupied_,node,shift_);
        const arma::mat ubar=mbars[2].t()+o2*mbars[3].t();
        const arma::mat fobar=exponential_vjp(occupied_,node/2,shift_,mbars[0])
                             +exponential_vjp(occupied_,node,shift_,u_*mbars[3]);
        const arma::mat fvbar=exponential_vjp(virtual_,-node/2,shift_,mbars[1]);
        const arma::mat* matrix_bars[3]={&fobar,&fvbar,&ubar};
        for(size_t k=0;k<3;++k) {
            auto value=out[6+k].cast<Array>(); const auto& b=*matrix_bars[k];
            for(size_t i=0;i<b.n_rows;++i) for(size_t j=0;j<b.n_cols;++j)
                value.mutable_data()[i*b.n_cols+j]=b(i,j);
        }
        return out;
    }
};
