// Private implementation of StableDressing's full-external transform graph.
// Every edge is a single axis transform. Public slots remain independent;
// only internal primals and accumulated consumer cotangents are shared.
struct GraphNode {
    size_t parent,source;
    Axis axis;
    Shape dims;
};
struct Graph {
    std::vector<GraphNode> nodes;
    std::vector<size_t> outputs,children;
    size_t tape_bytes=0;
};
Graph graph(bool diagonal_virtual,bool identity) const {
    using namespace ccsd_native::triples;
    Graph result;
    for(size_t source=0;source<6;++source)
        result.nodes.push_back({source,source,{0,0},shape(inputs_[source])});
    auto add=[&](size_t parent,size_t axis,size_t matrix) {
        if(identity && matrix<2) return parent;
        for(size_t i=6;i<result.nodes.size();++i) {
            const auto& n=result.nodes[i];
            if(n.parent==parent && n.axis.axis==axis && n.axis.matrix==matrix) return i;
        }
        auto dims=result.nodes[parent].dims;
        if(matrix==2) dims[axis]=inputs_[8].shape(1);
        result.nodes.push_back({parent,result.nodes[parent].source,{axis,matrix},std::move(dims)});
        return result.nodes.size()-1;
    };
    result.outputs.resize(specs_.size());
    for(size_t kind=0;kind<NTYPES;++kind) {
        if(triples_doubles_kind(kind)) continue;
        size_t current=specs_[kind].source;
        for(const auto& axis:plans_[kind]) current=add(current,axis.axis,axis.matrix);
        result.outputs[kind]=current;
    }
    // O_1 and V_2 occur in all three T2 forms. In the dense case the last
    // shared edge is occupied; scalar virtual scaling is cheaper otherwise.
    const size_t prefix=add(add(1,1,0),2,1);
    result.outputs[T2V]=add(prefix,0,0);
    result.outputs[T2O]=add(prefix,3,1);
    result.outputs[T2]=diagonal_virtual ? add(result.outputs[T2V],3,1)
                                      : add(result.outputs[T2O],0,0);
    for(size_t slot=NTYPES;slot<specs_.size();++slot) {
        const auto& spec=specs_[slot];
        size_t selected_axis=spec.rank;
        for(size_t axis=0;axis<spec.rank;++axis)
            if(spec.transform[axis]==4) selected_axis=axis;
        if(selected_axis<spec.rank) {
            // Full-external ordinary occupied legs already contain O(t).
            // Thus U^T applied to the dressed base is exactly U^T O(t).
            result.outputs[slot]=add(result.outputs[spec.kind],selected_axis,2);
        } else {
            size_t current=spec.source;
            for(const auto& axis:plans_[slot]) current=add(current,axis.axis,axis.matrix);
            result.outputs[slot]=current;
        }
    }
    result.children.assign(result.nodes.size(),0);
    for(size_t i=6;i<result.nodes.size();++i) ++result.children[result.nodes[i].parent];
    // Reverse needs only the distinct non-root parents. Public leaf outputs
    // and immutable caller inputs are never part of the retained tape.
    for(size_t i=6;i<result.nodes.size();++i) if(result.children[i]) {
        size_t count=1;
        for(auto dim:result.nodes[i].dims) count*=size_t(dim);
        result.tape_bytes+=count*sizeof(double);
    }
    return result;
}
py::tuple graph_forward(const Graph& plan,const std::vector<dense::Matrix>& mats,
                        bool diagonal_virtual,bool retain) {
    std::vector<Array> values(plan.nodes.size());
    std::vector<size_t> remaining=plan.children,public_uses(plan.nodes.size(),0);
    for(auto output:plan.outputs) { ++remaining[output]; ++public_uses[output]; }
    for(size_t i=0;i<6;++i) values[i]=inputs_[i];
    if(retain) graph_prefixes_.resize(plan.nodes.size());
    for(size_t i=6;i<plan.nodes.size();++i) {
        const auto& n=plan.nodes[i];
        values[i]=transform(values[n.parent],n.axis,mats[n.axis.matrix],diagonal_virtual);
        if(retain && plan.children[i]) graph_prefixes_[i]=values[i];
        if(--remaining[n.parent]==0) values[n.parent]=Array();
    }
    py::tuple out(plan.outputs.size());
    for(size_t slot=0;slot<plan.outputs.size();++slot) {
        const auto id=plan.outputs[slot];
        // Outputs may be overwritten by the caller. Do not expose input or
        // tape memory, or alias distinct public slots at the identity node.
        out[slot]=(id<6 || public_uses[id]>1 || (retain && plan.children[id]))
                   ? copy(values[id]) : values[id];
        if(--remaining[id]==0) values[id]=Array();
    }
    return out;
}
py::tuple graph_prepare(double node,size_t cache_limit_bytes) {
    clear_cache();
    const bool diagonal_virtual=virtual_diagonal_at(node);
    const auto mats=matrices(node,diagonal_virtual);
    const auto plan=graph(diagonal_virtual,node==0.);
    const bool retain=cache_limit_bytes>0 && plan.tape_bytes<=cache_limit_bytes;
    try {
        auto out=graph_forward(plan,mats,diagonal_virtual,retain);
        graph_cached_=retain;
        cached_node_=node;
        cached_bytes_=retain ? plan.tape_bytes : 0;
        peak_bytes_=std::max(peak_bytes_,cached_bytes_);
        return out;
    } catch(...) { clear_cache(); throw; }
}
Array graph_replay(const Graph& plan,size_t id,const std::vector<dense::Matrix>& mats,
                   bool diagonal_virtual) const {
    // Each edge has one primal parent: replay one ancestor chain using two
    // staging tensors, never an unbounded tape when the cache is disabled.
    std::vector<size_t> chain;
    for(;id>=6;id=plan.nodes[id].parent) chain.push_back(id);
    Array value=inputs_[id];
    for(auto i=chain.rbegin();i!=chain.rend();++i) {
        const auto& n=plan.nodes[*i];
        value=transform(value,n.axis,mats[n.axis.matrix],diagonal_virtual);
    }
    return value;
}
static void graph_add(Array& sum,const Array& value) {
    if(!sum.size()) { sum=copy(value); return; }
    const double* source=value.data(); double* dest=sum.mutable_data();
    const auto count=sum.size();
    py::gil_scoped_release release;
    #pragma omp parallel for simd schedule(static) if(count>=32768)
    for(py::ssize_t i=0;i<count;++i) dest[i]+=source[i];
}
py::tuple graph_pullback(double node,py::tuple cotangents,size_t cache_limit_bytes) {
    if(!std::isfinite(node) || node<0) {
        clear_cache(); throw py::value_error("stable triples: node must be finite and nonnegative");
    }
    if(cotangents.size()!=specs_.size()) {
        clear_cache(); throw py::value_error("stable triples: invalid cotangent count");
    }
    // Validate all independent slots before consuming the tape or doing work.
    std::vector<Array> bars;
    try {
        for(size_t i=0;i<specs_.size();++i) {
            auto bar=Array::ensure(cotangents[i]);
            if(!bar) throw py::value_error("stable triples: invalid cotangent");
            const auto& spec=specs_[i];
            check_shape(bar,Shape(spec.dims.begin(),spec.dims.begin()+spec.rank));
            bars.push_back(std::move(bar));
        }
    } catch(...) { clear_cache(); throw; }
    const bool diagonal_virtual=virtual_diagonal_at(node);
    const auto mats=matrices(node,diagonal_virtual);
    const auto plan=graph(diagonal_virtual,node==0.);
    if((!graph_cached_ || cached_node_!=node) && cache_limit_bytes>0
       && plan.tape_bytes<=cache_limit_bytes) {
        graph_prepare(node,cache_limit_bytes); ++rebuilds_;
    }
    const bool cached=graph_cached_ && cached_node_==node;
    auto prefixes=std::move(graph_prefixes_);
    clear_cache(); hits_+=size_t(cached);
    if(!cached) prefixes.clear();
    std::vector<dense::Matrix> mbars;
    for(const auto& m:mats) mbars.emplace_back(m.n_rows,m.n_cols,dense::zero);
    py::tuple out(inputs_.size());
    for(size_t i=0;i<inputs_.size();++i) {
        Array zero(shape(inputs_[i]));
        std::fill(zero.mutable_data(),zero.mutable_data()+zero.size(),0.);
        out[i]=std::move(zero);
    }
    std::vector<Array> pending(plan.nodes.size());
    for(size_t slot=0;slot<plan.outputs.size();++slot) if(plan.outputs[slot]<6) {
        auto sum=out[plan.outputs[slot]].cast<Array>();
        graph_add(sum,bars[slot]);
    }
    for(size_t id=plan.nodes.size();id-->6;) {
        Array bar=std::move(pending[id]);
        for(size_t slot=0;slot<plan.outputs.size();++slot)
            if(plan.outputs[slot]==id) graph_add(bar,bars[slot]);
        if(!bar.size()) continue;
        const auto& n=plan.nodes[id];
        Array raw=n.parent<6 ? inputs_[n.parent]
                            : cached ? prefixes[n.parent]
                                     : graph_replay(plan,n.parent,mats,diagonal_virtual);
        mbars[n.axis.matrix]+=stable_matrix_bar(raw,bar,n.axis.axis,true);
        auto parent_bar=transform(bar,n.axis,mats[n.axis.matrix].t(),diagonal_virtual);
        if(n.parent<6) {
            auto sum=out[n.parent].cast<Array>();
            graph_add(sum,parent_bar);
        } else {
            // Descending topological order visits a shared parent only after
            // every consumer and every independent public slot has contributed.
            auto& sum=pending[n.parent];
            if(!sum.size()) sum=std::move(parent_bar);
            else graph_add(sum,parent_bar);
        }
        if(cached) prefixes[id]=Array();
    }
    const dense::Matrix fobar=exponential_vjp(occupied_,external_time(node),shift_,mbars[0]);
    const dense::Matrix fvbar=exponential_vjp(virtual_,-external_time(node),shift_,mbars[1]);
    const dense::Matrix ubar=mbars[2].t();
    const dense::Matrix* matrix_bars[3]={&fobar,&fvbar,&ubar};
    for(size_t k=0;k<3;++k) {
        auto value=out[6+k].cast<Array>(); const auto& b=*matrix_bars[k];
        for(size_t i=0;i<b.n_rows;++i) for(size_t j=0;j<b.n_cols;++j)
            value.mutable_data()[i*b.n_cols+j]=b(i,j);
    }
    return out;
}
