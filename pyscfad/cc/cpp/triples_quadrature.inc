// MRCC spin contractions and disconnected numerator adjoints. Included in the
// same anonymous namespace as the block GEMMs and their adjoints.
operations quadrature_describe(const triples_layout& layout,const tuple& t,size_t side) {
    const auto& x=t.x;
    const size_t i=x[0],j=x[1],k=x[2],a=x[3],b=x[4],c=x[5],s=t.selected;
    auto id=[&](size_t kind,int slot) { return slot<0 ? kind : layout.projected[side][kind][size_t(slot)]; };
    operations op;
    op.ids={{id(GV,s==0 ? 0 : -1),id(T2V,s==2 ? 0 : (s==1 ? 1 : -1)),
             id(GO,s==0 ? 0 : (s==1 ? 1 : -1)),id(T2O,s==2 ? 1 : -1),
             id(GG,s==0 ? 0 : (s==1 ? 1 : -1)),id(T1,s==2 ? 0 : -1),
             id(FOV,s==2 ? 0 : -1),id(T2,s==0 ? 0 : (s==1 ? 1 : -1))}};
    op.indices={{{{i,a,b,0}},{{k,j,c,0}},{{i,a,j,0}},{{0,k,b,c}},
                 {{i,a,j,b}},{{k,c,0,0}},{{k,c,0,0}},{{i,j,a,b}}}};
    return op;
}
std::vector<std::array<size_t,3>> quadrature_blocks(size_t no,size_t ns) {
    std::vector<std::array<size_t,3>> out;
    for(size_t s=0;s<ns;++s) for(size_t j=0;j<no;++j) for(size_t k=0;k<=j;++k)
        out.push_back({s,j,k});
    return out;
}
// Direct slices of the two disconnected outer-product factors. No integral
// gathering or L cube is needed for the energy/cache-adjoint contractions.
struct quadrature_singles {
    const double *integral,*two,*one,*fov;
    size_t stride;
    quadrature_singles(const std::vector<tensor>& bank,const operations& op) {
        const auto& g=bank[op.ids[4]];
        integral=g.value.memptr()+g.offset(op.indices[4]);
        stride=g.dims[2]*g.dims[3];
        const auto& t=bank[op.ids[7]];two=t.value.memptr()+t.offset(op.indices[7]);
        const auto& o=bank[op.ids[5]];one=o.value.memptr()+o.offset(op.indices[5]);
        const auto& f=bank[op.ids[6]];fov=f.value.memptr()+f.offset(op.indices[6]);
    }
    double half_value(size_t a,size_t b,size_t c,size_t n) const {
        return .5*(integral[a*stride+b]*one[c]+two[a*n+b]*fov[c]);
    }
};
template<size_t P> void quadrature_coordinates(size_t a,size_t b,size_t c,
        size_t& x,size_t& y,size_t& z) {
    if constexpr(P==0) {x=a;y=b;z=c;}
    if constexpr(P==1) {x=b;y=c;z=a;}
    if constexpr(P==2) {x=c;y=a;z=b;}
    if constexpr(P==3) {x=a;y=c;z=b;}
    if constexpr(P==4) {x=b;y=a;z=c;}
    if constexpr(P==5) {x=c;y=b;z=a;}
}
// P maps a raw numerator element to its cached K element. Its inverse is
// used for a contiguous cache-bar gather, avoiding a strided scatter.
template<size_t P,bool Gradient>
double mrcc_singles_contract(const dense::Matrix& ket,const quadrature_singles& l,
        size_t n,double factor,triples_raw_workspace& workspace,triples_right_profile* profile) {
    const double* k=ket.memptr();
    double *one_bar=nullptr,*fov_bar=nullptr,*two_bar=nullptr,*integral_bar=nullptr;
    if constexpr(Gradient) {
        workspace.one_bar.zeros();workspace.fov_bar.zeros();
        one_bar=workspace.one_bar.memptr();fov_bar=workspace.fov_bar.memptr();
        two_bar=workspace.two_bar.memptr();integral_bar=workspace.integral_bar.memptr();
    }
    double energy=0.;
    {
    triples_profile_scope timer(triples_profile_phase(profile,triples_right_profile::contract));
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
        const size_t offset=(a*n+b)*n;
        const double g=l.integral[a*l.stride+b],t=l.two[a*n+b];
        double two_sum=0.,integral_sum=0.;
        #pragma omp simd reduction(+:energy,two_sum,integral_sum)
        for(size_t c=0;c<n;++c) {
            const double weight=factor*k[triples_permuted_index<P>(a,b,c,n)];
            const double value=.5*(g*l.one[c]+t*l.fov[c]);
            energy+=weight*value;
            if constexpr(Gradient) {
                const double half=.5*weight;
                one_bar[c]+=half*g;fov_bar[c]+=half*t;
                two_sum+=half*l.fov[c];integral_sum+=half*l.one[c];
            }
        }
        if constexpr(Gradient) {
            two_bar[a*n+b]=two_sum;integral_bar[a*n+b]=integral_sum;
        }
    }
    }
    return energy;
}

template<size_t P>
double quadrature_singles_seed(const quadrature_singles& l,size_t a,size_t b,size_t c,size_t n) {
    size_t x,y,z;quadrature_coordinates<P>(a,b,c,x,y,z);
    return l.half_value(x,y,z,n);
}

// The MRCC connected seed needs the sum of six singles permutations.
// Gather all of them into one contiguous K-bar pass, instead of six RMWs.
void mrcc_singles_seed(dense::Matrix& bar,const std::array<quadrature_singles,6>& l,
        size_t n,double factor) {
    double* target=bar.memptr();
    for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b) {
        const size_t offset=(a*n+b)*n;
        #pragma omp simd
        for(size_t c=0;c<n;++c) {
            const double value=quadrature_singles_seed<0>(l[0],a,b,c,n)
                +quadrature_singles_seed<1>(l[1],a,b,c,n)
                +quadrature_singles_seed<2>(l[2],a,b,c,n)
                +quadrature_singles_seed<3>(l[3],a,b,c,n)
                +quadrature_singles_seed<4>(l[4],a,b,c,n)
                +quadrature_singles_seed<5>(l[5],a,b,c,n);
            target[offset+c]+=factor*value;
        }
    }
}
void quadrature_singles_store(std::vector<tensor>& bank,const operations& op,
        size_t n,const triples_raw_workspace& w,triples_bar_locks& locks,
        triples_right_profile* profile=nullptr) {
    auto& t1=bank[op.ids[5]];auto& fv=bank[op.ids[6]];
    auto& t2=bank[op.ids[7]];auto& gg=bank[op.ids[4]];
    const size_t n1=t1.offset(op.indices[5]),nf=fv.offset(op.indices[6]);
    const size_t n2=t2.offset(op.indices[7]),ng=gg.offset(op.indices[4]);
    {triples_profile_lock guard(locks.at(op,5,t1),profile,4);
     dense::Matrix out(t1.bar.memptr()+n1,n,false,true);out+=w.one_bar;}
    {triples_profile_lock guard(locks.at(op,6,fv),profile,5);
     dense::Matrix out(fv.bar.memptr()+nf,n,false,true);out+=w.fov_bar;}
    {triples_profile_lock guard(locks.at(op,7,t2),profile,6);
     dense::Matrix out(t2.bar.memptr()+n2,n*n,false,true);out+=w.two_bar;}
    {triples_profile_lock guard(locks.at(op,4,gg),profile,7);
     const size_t stride=gg.dims[2]*gg.dims[3];
     for(size_t a=0;a<n;++a) for(size_t b=0;b<n;++b)
         gg.bar[ng+a*stride+b]+=w.integral_bar[a*n+b];}
}
// mode 0 builds K=S B X once; 1 contracts the cyclic-only right numerator;
// 2 reverses K once, after summing cotangents over all quadrature points.
