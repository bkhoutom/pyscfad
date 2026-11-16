// Group the twelve connected GEMMs into abc and bac output orderings.
// A matrix transpose puts the singleton virtual index first or last for
// free. Only terms whose singleton is b need the second cube. One final
// (ab) permutation combines them. This uses no integral/T2 symmetry: the
// independent prepared-input cotangents remain exactly defined.
struct triples_left_pack {
    std::vector<arma::vec> value,bar;
    std::vector<size_t> ids;
    explicit triples_left_pack(const std::vector<tensor>& bank,
            const triples_layout& layout,bool gradient) {
        value.resize(bank.size());bar.resize(bank.size());
        ids={GV,layout.projected[0][GV][0]};
        for(auto id:ids) {
            value[id].set_size(bank[id].value.n_elem);
            if(gradient) bar[id].set_size(bank[id].value.n_elem);
        }
    }
    size_t bytes() const {
        size_t n=0;
        for(const auto& v:value) n+=v.n_elem*sizeof(double);
        for(const auto& v:bar) n+=v.n_elem*sizeof(double);
        return n;
    }
};

// Called collectively by the worker team. All packing and its reverse are
// rank-four operations, performed once per prepared left input.
void triples_left_pack_forward(triples_left_pack& pack,const std::vector<tensor>& bank,size_t n) {
    for(auto id:pack.ids) {
        const size_t pairs=bank[id].dims[0]*n*n;
        #pragma omp for schedule(static)
        for(size_t pair=0;pair<pairs;++pair) {
            const size_t i=pair/(n*n),a=pair/n%n,b=pair%n;
            std::copy_n(bank[id].value.memptr()+pair*n,n,
                        pack.value[id].memptr()+((i*n+b)*n+a)*n);
            if(pack.bar[id].n_elem) std::fill_n(pack.bar[id].memptr()+pair*n,n,0.);
        }
    }
}
void triples_left_pack_reverse(const triples_left_pack& pack,std::vector<tensor>& bank,size_t n) {
    for(auto id:pack.ids) {
        const size_t pairs=bank[id].dims[0]*n*n;
        #pragma omp for schedule(static)
        for(size_t pair=0;pair<pairs;++pair) {
            const size_t i=pair/(n*n),a=pair/n%n,b=pair%n;
            const double* src=pack.bar[id].memptr()+((i*n+b)*n+a)*n;
            double* dst=bank[id].bar.memptr()+pair*n;
            #pragma omp simd
            for(size_t d=0;d<n;++d) dst[d]+=src[d];
        }
    }
}

struct triples_output_order {
    bool alternate,singleton_first,swap_pair;
    triples_output_order(size_t p,bool occupied) {
        const auto* axes=permutations[p];
        const size_t singleton=axes[occupied ? 0 : 2];
        alternate=singleton==1;
        const size_t order[3]={alternate ? 1u : 0u,alternate ? 0u : 1u,2};
        singleton_first=singleton==order[0];
        swap_pair=axes[occupied ? 1 : 0]!=order[singleton_first ? 1 : 0];
    }
};

void triples_left_virtual(std::vector<tensor>& bank,const operations& op,
        triples_left_pack& pack,size_t n,const triples_output_order& order,
        arma::vec& output,triples_raw_workspace& w,triples_bar_locks* locks) {
    auto& gv=bank[op.ids[0]];auto& tv=bank[op.ids[1]];
    const size_t ng=gv.offset(op.indices[0]),nt=tv.offset(op.indices[1]);
    auto& gvalue=order.swap_pair ? pack.value[op.ids[0]] : gv.value;
    arma::mat g(gvalue.memptr()+ng,n,n*n,false,true);
    arma::mat t(tv.value.memptr()+nt,n,n,false,true);
    arma::mat c(output.memptr(),order.singleton_first ? n*n : n,
                order.singleton_first ? n : n*n,false,true);
    if(!locks) {
        if(order.singleton_first) c+=g.t()*t;
        else c+=t.t()*g;
    } else {
        if(order.singleton_first) { w.gb=t*c.t();w.tb=g*c; }
        else { w.gb=t*c;w.tb=g*c.t(); }
        auto& gbar=order.swap_pair ? pack.bar[op.ids[0]] : gv.bar;
        arma::mat gb(gbar.memptr()+ng,n,n*n,false,true);
        arma::mat tb(tv.bar.memptr()+nt,n,n,false,true);
        { std::lock_guard<std::mutex> guard(locks->at(op,0,gv));gb+=w.gb; }
        { std::lock_guard<std::mutex> guard(locks->at(op,1,tv));tb+=w.tb; }
    }
}

void triples_left_occupied(std::vector<tensor>& bank,const operations& op,
        size_t no,size_t n,const triples_output_order& order,arma::vec& output,
        triples_raw_workspace& w,triples_bar_locks* locks) {
    auto& go=bank[op.ids[2]];auto& to=bank[op.ids[3]];
    const size_t ng=go.offset(op.indices[2]),nt=to.offset(op.indices[3]);
    const size_t gs=go.dims[2]*go.dims[3],ts=to.dims[1]*n*n;
    for(size_t a=0;a<n;++a) std::copy_n(go.value.memptr()+ng+a*gs,no,w.h.colptr(a));
    for(size_t m=0;m<no;++m) {
        const double* src=to.value.memptr()+nt+m*ts;double* dst=w.u.colptr(m);
        if(!order.swap_pair) std::copy_n(src,n*n,dst);
        else for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c) dst[b*n+c]=src[c*n+b];
    }
    arma::mat c(output.memptr(),order.singleton_first ? n*n : n,
                order.singleton_first ? n : n*n,false,true);
    if(!locks) {
        if(order.singleton_first) c-=w.u*w.h;
        else c-=w.h.t()*w.u.t();
    } else {
        if(order.singleton_first) {w.hb=-w.u.t()*c;w.ub=-c*w.h.t();}
        else {w.hb=-w.u.t()*c.t();w.ub=-c.t()*w.h.t();}
        { std::lock_guard<std::mutex> guard(locks->at(op,2,go));
          for(size_t a=0;a<n;++a) for(size_t m=0;m<no;++m) go.bar[ng+a*gs+m]+=w.hb(m,a); }
        { std::lock_guard<std::mutex> guard(locks->at(op,3,to));
          for(size_t m=0;m<no;++m) for(size_t b=0;b<n;++b) for(size_t c=0;c<n;++c)
              to.bar[nt+m*ts+b*n+c]+=w.ub(order.swap_pair ? c*n+b : b*n+c,m); }
    }
}

void triples_left_one_permutation(std::vector<tensor>& bank,const triples_layout& layout,
        const tuple& base,triples_left_pack& pack,size_t no,size_t n,
        arma::vec& main,arma::vec& alternate,triples_raw_workspace& workspace,
        triples_bar_locks* locks=nullptr) {
    if(!locks) {main.zeros();alternate.zeros();}
    else {
        // The (ab) permutation is self-adjoint; assignment avoids zero+add.
        for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b)
            std::copy_n(main.memptr()+(b*n+a)*n,n,alternate.memptr()+(a*n+b)*n);
    }
    for(size_t p=0;p<6;++p) {
        const auto op=quadrature_describe(layout,permute(base,p,true),0);
        const triples_output_order v(p,false),o(p,true);
        triples_left_virtual(bank,op,pack,n,v,v.alternate ? alternate : main,workspace,locks);
        triples_left_occupied(bank,op,no,n,o,o.alternate ? alternate : main,workspace,locks);
    }
    if(!locks) triples_permute_add(main,alternate,n,4);
}
