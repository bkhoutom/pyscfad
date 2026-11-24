// Copyright 2026 The PySCFAD Authors (Apache-2.0).
// Restricted, real CCSD equations adapted from PySCF ccsd.update_amps:
// Copyright 2014-2021 The PySCF Developers (Apache-2.0).
#include "tensor.h"
#include <algorithm>
#include <array>
#include <climits>
#include <ctime>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>
#include <omp.h>

namespace py = pybind11;
namespace ccsd_native {
using Index = py::ssize_t;
using Input = py::array_t<double, 0>;
using Array = ResidualArray;
ResidualGemm blas;
ResidualLadder virtual_ladder;

// One resident solve owns immutable integral layouts and reusable amplitude
// workspaces. Only the layouts retain values: tau/theta are overwritten on
// every update. Keep input owners alive and serialize uses of the workspaces.
struct ResidualWorkspace {
    Input packed_owner,occupied_owner;
    T packed_source,occupied_source,vovv,occupied,tau_buffer,theta_buffer;
    Scratch scratch;
    std::vector<size_t> virtual_pairs;
    std::vector<Index> occupied_rows;
    bool amplitudes_populated=false;
    std::mutex mutex;
    ResidualWorkspace(const Input& packed,const Input& ovoo)
        :packed_owner(packed),occupied_owner(ovoo),packed_source(packed),occupied_source(ovoo) {
        if(packed_source.rank!=3 || packed_source.n[0]<1 || packed_source.n[1]<1)
            throw std::invalid_argument("invalid resident ovvv shape");
        Index o=packed_source.n[0],v=packed_source.n[1];
        shape_is(packed_source,{o,v,v*(v+1)/2});shape_is(occupied_source,{o,v,o,o});
        py::gil_scoped_release release;
        vovv=unpack(packed_source.perm({1,0,2}),v);
        occupied=copy(occupied_source.perm({2,0,1,3})); // k,j,b,i for the oooo singles term
        tau_buffer=T({o,o,v,v});theta_buffer=T({o,o,v,v});
        virtual_pairs.resize(v*v);
        for(Index a=0;a<v;++a)for(Index b=0;b<v;++b)virtual_pairs[a*v+b]=pair(a,b);
        occupied_rows=occupied_pair_rows(o);
    }
    static bool same_view(const T& a,const T& b) {
        return a.p==b.p && a.rank==b.rank && a.n==b.n && a.s==b.s;
    }
    void validate(const T& packed,const T& ovoo) const {
        if(!same_view(packed,packed_source) || !same_view(ovoo,occupied_source))
            throw std::invalid_argument("residual workspace belongs to different integral views");
    }
};

// Flatten occupied-pair/virtual rows instead of distributing triangular i
// rows: every task packs exactly v elements, including occupied diagonals.
void pack_ladder_tau(double* packed,const T& t1,const T& t2,const std::vector<Index>& occupied_rows,bool singles=true) {
    const Index v=t1.n[1],op=occupied_rows.size();
    #pragma omp parallel for schedule(static) if(op*v*v>4096)
    for(Index row=0;row<op*v;++row) {
        const Index ij=row/v,a=row%v,i=occupied_rows[ij],j=ij-i*(i+1)/2;
        const double ti=singles?t1.p[i*t1.s[0]+a*t1.s[1]]:0.;
        const Index off=i*t2.s[0]+j*t2.s[1]+a*t2.s[2];
        #pragma omp simd
        for(Index b=0;b<v;++b)
            packed[row*v+b]=t2.p[off+b*t2.s[3]]+ti*t1.p[j*t1.s[0]+b*t1.s[1]];
    }
}

void expand_ladder(T out,const double* packed,double scale) {
    const Index o=out.n[0],v=out.n[2];
    #pragma omp parallel for schedule(static) if(o*o*v*v>4096)
    for(Index row=0;row<o*o*v;++row) {
        const Index a=row%v,j=(row/v)%o,i=row/(o*v),ij=pair(i,j);
        #pragma omp simd
        for(Index b=0;b<v;++b)
            out.p[row*v+b]=scale*packed[(ij*v+(i>=j?a:b))*v+(i>=j?b:a)];
    }
}

// The ladder uses serial BLAS inside its OpenMP team. It is a separate entry
// so the driver can time it separately and bound its panel workspace.
Array ladder_residual(const Input& t1a,const Input& t2a,const Array& pairs,size_t workspace) {
    T t1(t1a),t2(t2a);
    if(t1.rank!=2 || t1.n[0]<1 || t1.n[1]<1)throw std::invalid_argument("invalid amplitudes");
    Index o=t1.n[0],v=t1.n[1],op=o*(o+1)/2;
    shape_is(t2,{o,o,v,v});
    Array packed(std::vector<Index>{op,v,v}),out(std::vector<Index>{o,o,v,v});
    double* p=packed.mutable_data();T output{Input(out)};
    {
        py::gil_scoped_release release;
        pack_ladder_tau(p,t1,t2,occupied_pair_rows(o));
    }
    Array value=virtual_ladder(pairs,packed,workspace);
    const double* x=value.data();
    {
        py::gil_scoped_release release;
        expand_ladder(output,x,1.);
    }
    return out;
}

Index ladder_scratch_elements(Index o,Index v,const LadderPanelPlan& plan) {
    if(o<1 || v<1 || o>INT_MAX || o*(o+1)/2>INT_MAX)
        throw std::overflow_error("ladder dimensions exceed BLAS integer range");
    const size_t limit=PTRDIFF_MAX/sizeof(double),panels=plan.elements(v);
    const size_t op=size_t(o)*(o+1)/2,vv=size_t(v)*v;
    if(panels>limit-24 || op>(limit-panels-24)/2/vv)
        throw std::overflow_error("ladder scratch size overflow");
    return Index(2*op*vv+panels+24);
}

// Tau and the triangular result live only through expansion. The same arena
// can then be overwritten by ovvv/ovov; no returned array borrows its storage.
void initialize_r2(T r2,const T& t1,const T& t2,const T& pairs,Scratch& scratch,
                   const LadderPanelPlan& plan,const std::vector<size_t>& indices,
                   const std::vector<Index>& occupied_rows,bool singles) {
    const Index o=t1.n[0],v=t1.n[1],op=o*(o+1)/2;
    scratch.begin(ladder_scratch_elements(o,v,plan));
    T packed=scratch.get({op,v,v}),value=scratch.get({op,v,v});
    T panels=scratch.get({Index(plan.elements(v))});
    pack_ladder_tau(packed.p,t1,t2,occupied_rows,singles);
    ladder_into(pairs.p,packed.p,value.p,v,op,plan,panels.p,indices.data());
    // Initialize every r2 element, including its 1/2 coefficient, directly
    // from the triangular result. No full seed, zero-fill or subsequent add.
    expand_ladder(r2,value.p,.5);
    // Packed tau/output are fully written. Panel capacity can exceed the
    // dynamically selected team, so do not credit all panels as resident.
    scratch.populated=std::max(scratch.populated,2*op*v*v);
}

// One task owns a virtual tile and its symmetry partner. Diagonal occupied
// pairs use only the lower virtual triangle, so no two tasks write an element.
void finish_amplitudes(T r1,T r2,const T& energy,double shift,const std::vector<Index>& occupied_rows) {
    const Index o=r1.n[0],v=r1.n[1],tile=32,nt=(v+tile-1)/tile;
    const Index tasks=Index(occupied_rows.size())*nt*nt;
    std::vector<double> eps(o+v);
    for(Index k=0;k<o+v;++k)eps[k]=energy.p[k*energy.s[0]];
    #pragma omp parallel if(o*o*v*v>4096)
    {
        #pragma omp for schedule(static) nowait
        for(Index ia=0;ia<o*v;++ia)
            r1.p[ia]/=eps[ia/v]-eps[o+ia%v]-shift;
        #pragma omp for schedule(static)
        for(Index task=0;task<tasks;++task) {
            const Index ij=task/(nt*nt),at=(task/nt)%nt,bt=task%nt;
            const Index i=occupied_rows[ij],j=ij-i*(i+1)/2;
            if(i==j && bt>at)continue;
            const Index a0=at*tile,a1=std::min(v,a0+tile),b0=bt*tile,b1=std::min(v,b0+tile);
            double* x=r2.p+(i*o+j)*v*v;
            double* y=r2.p+(j*o+i)*v*v;
            const double eij=eps[i]+eps[j];
            for(Index a=a0;a<a1;++a) {
                const Index end=i==j && at==bt?std::min(b1,a+1):b1;
                const double eija=eij-eps[o+a];
                #pragma omp simd
                for(Index b=b0;b<end;++b) {
                    const double denom=eija-eps[o+b]-2*shift;
                    const double value=(x[a*v+b]+y[b*v+a])/denom;
                    x[a*v+b]=value;y[b*v+a]=value;
                }
            }
        }
    }
}

py::tuple residual_impl(const Input& t1a,const Input& t2a,const Input& focka,const Input& energya,
                   const Input& ooooa,const Input& ovooa,const Input& oovva,const Input& ovvoa,
                   const Input& ovvva,const Input& laddera,double shift,Index block,bool profile,
                   const std::shared_ptr<ResidualWorkspace>& workspace,const LadderPanelPlan* ladder_plan) {
    T t1(t1a),t2(t2a),fock(focka),energy(energya),oooo(ooooa),ovoo(ovooa),oovv(oovva),ovvo(ovvoa),ovvv(ovvva),seed(laddera);
    if(t1.rank!=2 || t1.n[0]<1 || t1.n[1]<1 || block<1)throw std::invalid_argument("invalid amplitudes or block size");
    const Index o=t1.n[0],v=t1.n[1],np=v*(v+1)/2;
    shape_is(t2,{o,o,v,v});shape_is(fock,{o+v,o+v});shape_is(energy,{o+v});
    if(ladder_plan) {
        shape_is(seed,{np,np});
        if(!seed.contiguous())throw std::invalid_argument("packed vvvv must be contiguous");
        if(o*(o+1)/2>INT_MAX)throw std::overflow_error("ladder dimensions exceed BLAS integer range");
    } else shape_is(seed,{o,o,v,v});
    shape_is(oooo,{o,o,o,o});shape_is(ovoo,{o,v,o,o});shape_is(oovv,{o,o,v,v});shape_is(ovvo,{o,v,v,o});shape_is(ovvv,{o,v,np});
    if(workspace)workspace->validate(ovvv,ovoo);
    Array r1a(std::vector<Index>{o,v}),r2a(std::vector<Index>{o,o,v,v});
    T r1{Input(r1a)},r2{Input(r2a)};Engine e(profile);Sections sections(profile),finish_sections(profile);
    Index vvvv_scratch_bytes=0,ovvv_scratch_bytes=0,ovov_scratch_bytes=0,oooo_scratch_bytes=0,finish_scratch_bytes=0,shared_scratch_bytes=0;
    Index ovov_tau_peak_bytes=0,tau_half_builds=0,tau_response_builds=0;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> workspace_lock;
        if(workspace)workspace_lock=std::unique_lock<std::mutex>(workspace->mutex);
        Scratch local_scratch;
        Scratch& shared_scratch=workspace?workspace->scratch:local_scratch;
        std::vector<Index> local_occupied_rows;
        if(!workspace)local_occupied_rows=occupied_pair_rows(o);
        const auto& occupied_rows=workspace?workspace->occupied_rows:local_occupied_rows;
        if(ladder_plan) {
            std::vector<size_t> local_pairs;
            if(!workspace) {
                local_pairs.resize(v*v);
                for(Index a=0;a<v;++a)for(Index b=0;b<v;++b)local_pairs[a*v+b]=pair(a,b);
            }
            shared_scratch.start_phase();
            initialize_r2(r2,t1,t2,seed,shared_scratch,*ladder_plan,
                          workspace?workspace->virtual_pairs:local_pairs,occupied_rows);
            vvvv_scratch_bytes=Index(sizeof(double))*shared_scratch.phase_peak;
            sections.mark("vvvv");
        } else {fill(r2,0);add(r2,seed,.5);}
        fill(r1,0);
        T fov=copy(fock.slice(0,0,o).slice(1,o,o+v));add(r1,fov);
        T foo=copy(fock.slice(0,0,o).slice(1,0,o)),fvv=copy(fock.slice(0,o,o+v).slice(1,o,o+v));
        for(Index i=0;i<o;++i)foo.p[i*o+i]-=energy.p[i*energy.s[0]];
        for(Index a=0;a<v;++a)fvv.p[a*v+a]-=energy.p[(o+a)*energy.s[0]]+shift;
        e.C(foo,"ia,ja->ij",fov,t1,.5);e.C(fvv,"ia,ib->ab",t1,fov,-.5);
        sections.mark("fock");
        e.accumulation_scratch=&shared_scratch;
        e.accumulation_scratch->start_phase();
        T wVOov({v,o,o,v}),wVooV;
        T tau_full=tau(t2,t1,t1,1.,1.,workspace?workspace->tau_buffer:T());
        T theta_base=linear(t2.perm({1,0,2,3}),t2,2.,-1.,workspace?workspace->theta_buffer:T());
        if(workspace)workspace->amplitudes_populated=true;
        {
            T woo({o,o,np});fill(woo,0);
            T theta=theta_base.perm({0,2,1,3});
            T tmp_acc({o,o,v,o});fill(tmp_acc,0);fill(wVOov,0);
            // Packed resident integrals are unpacked in bounded virtual tiles.
            const Index tile=std::min(Index(32),block);
            for(Index p=0;p<v;p+=tile) {
                Index end=std::min(v,p+tile);
                T packed=ovvv.slice(1,p,end).perm({1,0,2});
                T vovv=workspace?workspace->vovv.slice(0,p,end):unpack(packed,v);
                T t1p=t1.slice(1,p,end);
                e.C(woo,"ic,cjp->ijp",t1p,packed,-1.);
                e.C(fvv,"kc,ckab->ab",t1p,vovv,2.);
                e.C(fvv.slice(1,p,end),"kc,bkca->ab",t1,vovv,-1.);
                e.C(tmp_acc,"ijcd,cdbk->ijbk",tau_full.slice(2,p,end),vovv.perm({0,2,3,1}));
                e.C(wVOov.slice(0,p,end),"biac,jc->bija",vovv,t1);
                e.C(r1,"icjb,cjba->ia",theta.slice(1,p,end),vovv);
            }
            e.C(r2,"ka,ijbk->ijab",t1,tmp_acc,-1.);
            wVooV=unpack(woo,v).perm({2,1,0,3});
        }
        ovvv_scratch_bytes=Index(sizeof(double))*e.accumulation_scratch->phase_peak;
        sections.mark("ovvv");
        e.accumulation_scratch->start_phase();
        T woooo=copy(oooo.perm({0,2,1,3}));
        // Build these only at their first consumer, and release at their last.
        // Partial p blocks still share one construction of each full tensor.
        T tau_half,tau_response;
        for(Index p=0;p<v;p+=block) {
            Index end=std::min(v,p+block);
            T wA=wVOov.slice(0,p,end),wB=wVooV.slice(0,p,end);
            T oo=ovoo.slice(1,p,end),vv=oovv.slice(2,p,end),tp=t1.slice(1,p,end),r2p=r2.slice(2,p,end);
            e.C(foo,"kc,kcji->ij",tp,oo,2.);e.C(foo,"kc,icjk->ij",tp,oo,-1.);
            {T tmp=e.E("la,jaik->lkji",tp,oo);add(woooo,tmp);add(woooo,tmp.perm({1,0,3,2}));}
            e.C(wA,"jbik,ka->bjia",oo,t1,-1.);add(r2p,wA.perm({1,2,0,3}));
            e.C(wB,"kbij,ka->bija",oo,t1);
            e.C(r1.slice(1,p,end),"jb,jiab->ia",t1,vv,-1.);
            add(wB,vv.perm({2,0,1,3}),-1.);add(wA,wB,.5);
            T voov=ovvo.slice(1,p,end).perm({1,0,3,2});
            add(r2p,voov.perm({1,2,0,3}),.5);
            T singles_voov=e.E("jb,aijb->ia",t1,voov);
            add(r1.slice(1,p,end),singles_voov,2.);
            // Only this packed layout has known reuse: bjkc -> (bjk,c)
            // below and cjib -> (cji,b) in the fvv contraction. Other inputs
            // use ordinary temporaries or the existing shared scratch arena.
            e.cache_layout(voov);
            {T tmp=e.E("ic,kjbc->ibkj",t1,vv);e.C(tmp,"bjkc,ic->jbki",voov,t1);
             e.C(r2p,"ka,jbki->jiba",t1,tmp,-1.);}
            add(fov.slice(1,p,end),singles_voov,2.);
            e.C(fov.slice(1,p,end),"kc,akic->ia",t1,voov,-1.);
            {T theta=theta_fock(theta_base.slice(2,p,end),tp,t1);
             e.C(fvv,"cjia,cjib->ab",theta.perm({2,1,0,3}),voov,-1.);
             e.clear_layouts();
             e.C(foo,"aikb,kjab->ij",voov,theta);}
            e.C(woooo,"ijab,aklb->ijkl",tau_full.slice(2,p,end),voov);
            if(!tau_half.p) {
                tau_half=linear(tau_full,t2,1.,-.5);++tau_half_builds;
                ovov_tau_peak_bytes=std::max(ovov_tau_peak_bytes,
                    Index(sizeof(double))*(tau_half.size()+(tau_response.p?tau_response.size():0)));
            }
            for(Index q=0;q<v;q+=block) {
                Index qe=std::min(v,q+block);
                e.C(wB,"bkic,jkca->bija",voov.slice(3,q,qe),tau_half.slice(2,q,qe));
            }
            if(end==v)tau_half=T();
            for(Index q=0;q<v;q+=block) {
                Index qe=std::min(v,q+block);
                e.with_result("jkca,ckib->jaib",t2.slice(2,p,end).slice(3,q,qe),wB,[&](const T& tmp) {
                    add_exchange(r2.slice(2,q,qe),tmp.perm({2,0,1,3}),tmp.perm({0,2,1,3}),.5);
                });
            }
            add(wA,voov);
            T VOov=linear(voov.perm({0,2,1,3}),voov,-.5,1.);
            if(!tau_response.p) {
                tau_response=tau(theta_base,t1,t1,1.,-2.).perm({0,3,1,2});++tau_response_builds;
                ovov_tau_peak_bytes=std::max(ovov_tau_peak_bytes,
                    Index(sizeof(double))*(tau_response.size()+(tau_half.p?tau_half.size():0)));
            }
            for(Index q=0;q<v;q+=block) {
                Index qe=std::min(v,q+block);
                T ta=tau_response.slice(3,q,qe);
                e.C(wA.slice(3,q,qe),"aikc,kcjb->aijb",VOov,ta,.5);
            }
            if(end==v)tau_response=T();
            for(Index q=0;q<v;q+=block) {
                Index qe=std::min(v,q+block);
                T theta=theta_base.perm({1,0,2,3}).slice(2,p,end).slice(3,q,qe);
                e.C(r2.slice(2,q,qe),"kica,ckjb->ijab",theta,wA);
            }
        }
        wVOov=T();wVooV=T();tau_half=T();tau_response=T();e.clear_layouts();
        ovov_scratch_bytes=Index(sizeof(double))*e.accumulation_scratch->phase_peak;
        shared_scratch_bytes=Index(sizeof(double))*e.accumulation_scratch->capacity;
        sections.mark("ovov");
        shared_scratch.start_phase();
        for(Index p=0;p<v;p+=block) {
            Index end=std::min(v,p+block);T t=t2.slice(2,p,end);
            T theta=theta_base.slice(2,p,end);
            e.C(r1,"jb,ijba->ia",fov.slice(1,p,end),theta);
            T occupied=workspace?workspace->occupied.perm({1,2,0,3}):ovoo;
            e.C(r1,"jbki,kjba->ia",occupied.slice(1,p,end),theta,-1.);
            e.C(r2.slice(2,p,end),"ijkl,klab->ijab",woooo,tau_full.slice(2,p,end),.5);
        }
        theta_base=T();tau_full=T();
        oooo_scratch_bytes=sizeof(double)*shared_scratch.phase_peak;
        sections.mark("oooo");
        finish_sections=Sections(profile);
        e.accumulation_scratch=&shared_scratch;
        shared_scratch.start_phase();
        T ftij=copy(foo),ftab=copy(fvv);
        e.C(ftij,"ja,ia->ij",t1,fov,.5);e.C(ftab,"ia,ib->ab",t1,fov,-.5);
        e.C(r2,"ijac,bc->ijab",t2,ftab);e.C(r2,"ki,kjab->ijab",ftij,t2,-1.);
        e.C(r1,"ib,ab->ia",t1,fvv);e.C(r1,"ja,ji->ia",t1,foo,-1.);
        finish_sections.mark("contractions");
        finish_amplitudes(r1,r2,energy,shift,occupied_rows);
        finish_sections.mark("symmetry_and_denominators");
        finish_scratch_bytes=sizeof(double)*shared_scratch.phase_peak;
        shared_scratch_bytes=std::max(shared_scratch_bytes,Index(sizeof(double))*shared_scratch.capacity);
        e.accumulation_scratch=nullptr;
        sections.mark("finish");
    }
    if(!profile)return py::make_tuple(r1a,r2a);
    py::dict report,contractions,regions;
    for(const auto& item:e.timings) {
        const auto& t=item.second;py::dict d;
        d["pack_seconds"]=t.pack;d["gemm_seconds"]=t.gemm;d["accumulate_seconds"]=t.accumulate;
        d["calls"]=t.calls;d["packed_bytes"]=t.packed_bytes;contractions[py::str(item.first)]=d;
        d["direct_calls"]=t.direct_calls;d["result_bytes"]=t.result_bytes;
    }
    for(const auto& item:sections.times) {
        py::dict d;d["wall_seconds"]=item.second.first;d["cpu_seconds"]=item.second.second;
        regions[py::str(item.first)]=d;
    }
    report["contractions"]=contractions;report["sections"]=regions;
    py::dict finish_detail;
    for(const auto& item:finish_sections.times) {
        py::dict d;d["wall_seconds"]=item.second.first;d["cpu_seconds"]=item.second.second;
        finish_detail[py::str(item.first)]=d;
    }
    report["finish_detail"]=finish_detail;
    py::dict cache;cache["bytes"]=e.cache_peak_bytes;cache["hits"]=e.cache_hits;
    cache["packed_bytes"]=e.cache_packed_bytes;cache["end_bytes"]=e.cache_bytes;
    report["ovov_layout_cache"]=cache;
    py::dict tau_lifetimes;tau_lifetimes["peak_bytes"]=ovov_tau_peak_bytes;
    tau_lifetimes["half_builds"]=tau_half_builds;tau_lifetimes["response_builds"]=tau_response_builds;
    report["ovov_tau_temporaries"]=tau_lifetimes;
    py::dict reuse;reuse["enabled"]=bool(workspace);
    reuse["integral_bytes"]=workspace?8*(workspace->vovv.size()+workspace->occupied.size()):0;
    reuse["workspace_bytes"]=workspace?8*(workspace->tau_buffer.size()+workspace->theta_buffer.size()):0;
    reuse["vvvv_scratch_bytes"]=vvvv_scratch_bytes;
    reuse["vvvv_pair_index_bytes"]=workspace?sizeof(size_t)*workspace->virtual_pairs.size():0;
    reuse["occupied_pair_index_bytes"]=workspace?sizeof(Index)*workspace->occupied_rows.size():0;
    reuse["ovvv_scratch_bytes"]=ovvv_scratch_bytes;
    reuse["ovov_scratch_bytes"]=ovov_scratch_bytes;
    reuse["oooo_scratch_bytes"]=oooo_scratch_bytes;
    reuse["finish_scratch_bytes"]=finish_scratch_bytes;
    reuse["shared_scratch_bytes"]=shared_scratch_bytes;
    report["resident_reuse"]=reuse;
    return py::make_tuple(r1a,r2a,report);
}

py::tuple residual(const Input& t1,const Input& t2,const Input& fock,const Input& energy,
                   const Input& oooo,const Input& ovoo,const Input& oovv,const Input& ovvo,
                   const Input& ovvv,const Input& seed,double shift,Index block,bool profile,
                   const std::shared_ptr<ResidualWorkspace>& workspace) {
    return residual_impl(t1,t2,fock,energy,oooo,ovoo,oovv,ovvo,ovvv,seed,shift,block,profile,workspace,nullptr);
}

py::tuple residual_from_integrals(const Input& t1,const Input& t2,const Input& fock,const Input& energy,
                   const Input& oooo,const Input& ovoo,const Input& oovv,const Input& ovvo,
                   const Input& ovvv,const Array& pairs,size_t panel_bytes,double shift,Index block,bool profile,
                   const std::shared_ptr<ResidualWorkspace>& workspace) {
    if(t1.ndim()!=2 || t1.shape(0)<1 || t1.shape(1)<1)throw std::invalid_argument("invalid amplitudes");
    const auto plan=ladder_panel_plan(t1.shape(1),panel_bytes,omp_get_max_threads());
    return residual_impl(t1,t2,fock,energy,oooo,ovoo,oovv,ovvo,ovvv,pairs,shift,block,profile,workspace,&plan);
}
} // namespace

void bind_residual(py::module_& m,ResidualGemm gemm,ResidualLadder ladder) {
    using namespace ccsd_native;
    blas=gemm;virtual_ladder=ladder;
    m.attr("ovov_reuse_version")=1;
    py::class_<ResidualWorkspace,std::shared_ptr<ResidualWorkspace>>(m,"ResidualWorkspace")
        .def(py::init<const Input&,const Input&>(),py::arg("ovvv").noconvert(),py::arg("ovoo").noconvert())
        .def("memory_info",[](ResidualWorkspace& w) {
            size_t integrals,amplitudes,indices,capacity,populated;bool ready;
            {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(w.mutex);
             integrals=sizeof(double)*(w.vovv.size()+w.occupied.size());
             amplitudes=sizeof(double)*(w.tau_buffer.size()+w.theta_buffer.size());
             indices=sizeof(size_t)*w.virtual_pairs.size()+sizeof(Index)*w.occupied_rows.size();
             capacity=sizeof(double)*w.scratch.capacity;populated=sizeof(double)*w.scratch.populated;
             ready=w.amplitudes_populated;}
            py::dict out;out["integral_bytes"]=integrals;out["amplitude_bytes"]=amplitudes;
            out["index_bytes"]=indices;out["scratch_capacity_bytes"]=capacity;
            out["scratch_populated_bytes"]=populated;out["amplitudes_populated"]=ready;
            out["allocated_bytes"]=integrals+amplitudes+indices+capacity;
            out["populated_bytes"]=integrals+(ready?amplitudes:0)+indices+populated;
            return out;
        });
    m.def("ladder_memory",[](Index o,Index v,size_t budget) {
        if(o<1 || v<1)throw std::invalid_argument("invalid amplitude dimensions");
        const int threads=omp_get_max_threads();
        const auto plan=ladder_panel_plan(v,budget,threads);
        py::dict out;out["panel_bytes"]=sizeof(double)*plan.elements(v);
        out["scratch_bytes"]=sizeof(double)*ladder_scratch_elements(o,v,plan);
        out["threads"]=threads;return out;
    },py::arg("nocc"),py::arg("nvir"),py::arg("panel_bytes")=std::numeric_limits<size_t>::max());
    m.def("ladder_residual",&ladder_residual,py::arg("t1").noconvert(),py::arg("t2").noconvert(),
          py::arg("pairs").noconvert(),py::arg("workspace_bytes"));
    m.def("residual",&residual,py::arg("t1").noconvert(),py::arg("t2").noconvert(),
          py::arg("fock").noconvert(),py::arg("mo_energy").noconvert(),py::arg("oooo").noconvert(),
          py::arg("ovoo").noconvert(),py::arg("oovv").noconvert(),py::arg("ovvo").noconvert(),
          py::arg("ovvv").noconvert(),py::arg("ladder").noconvert(),py::arg("level_shift"),
          py::arg("block_size"),py::arg("profile")=false,py::arg("workspace")=nullptr);
    m.def("residual_from_integrals",&residual_from_integrals,py::arg("t1").noconvert(),py::arg("t2").noconvert(),
          py::arg("fock").noconvert(),py::arg("mo_energy").noconvert(),py::arg("oooo").noconvert(),
          py::arg("ovoo").noconvert(),py::arg("oovv").noconvert(),py::arg("ovvo").noconvert(),
          py::arg("ovvv").noconvert(),py::arg("pairs").noconvert(),py::arg("workspace_bytes"),py::arg("level_shift"),
          py::arg("block_size"),py::arg("profile")=false,py::arg("workspace")=nullptr);
    m.def("ladder_scratch_bytes",[](Index o,Index v) {
        if(o<1 || v<1)throw std::invalid_argument("invalid amplitude dimensions");
        const auto plan=ladder_panel_plan(v,std::numeric_limits<size_t>::max(),omp_get_max_threads());
        return sizeof(double)*ladder_scratch_elements(o,v,plan);
    },py::arg("nocc"),py::arg("nvir"));
    m.def("_finish_amplitudes",[](const Input& singles,const Input& doubles,const Input& energy,double shift) {
        T a(singles),b(doubles),eps(energy);
        if(a.rank!=2 || a.n[0]<1 || a.n[1]<1)throw std::invalid_argument("invalid amplitudes");
        Index o=a.n[0],v=a.n[1];shape_is(b,{o,o,v,v});shape_is(eps,{o+v});
        Array first({o,v}),second({o,o,v,v});T r1{Input(first)},r2{Input(second)};
        {py::gil_scoped_release release;copy(a,1.,r1);copy(b,1.,r2);
         finish_amplitudes(r1,r2,eps,shift,occupied_pair_rows(o));}
        return py::make_tuple(first,second);
    },py::arg("singles").noconvert(),py::arg("doubles").noconvert(),py::arg("energy").noconvert(),py::arg("shift"));
    // Diagnostic entry point for validating GEMM planning independently.
    m.def("_contract",[](const std::string& spec,const Input& a,const Input& b) {
        T x(a),y(b),z;{py::gil_scoped_release release;Engine e(false);z=copy(e.E(spec,x,y));}
        Array out(z.shape());std::copy(z.p,z.p+z.size(),out.mutable_data());return out;
    },py::arg("subscripts"),py::arg("left").noconvert(),py::arg("right").noconvert());
    m.def("_matrix_layout",[](const Input& input,int split) {
        T a(input);if(split<0 || split>a.rank)throw std::invalid_argument("invalid matrix split");
        std::string spec=std::string("ijkl").substr(0,a.rank);Engine::Matrix view;
        py::dict result;
        bool borrowed=Engine::matrix_view(a,spec,spec.substr(0,split),spec.substr(split),view);
        result["packed_bytes"]=borrowed?0:size_t(a.size())*sizeof(double);
        result["trans"]=borrowed?std::string(1,view.trans):std::string("packed");
        result["leading_dimension"]=borrowed?view.ld:0;return result;
    },py::arg("input").noconvert(),py::arg("split"));
    m.def("_contract_accumulate",[](const Input& out,const std::string& spec,const Input& a,const Input& b,double alpha,size_t cache_bytes,int repeat) {
        if(!out.writeable())throw std::invalid_argument("output must be writable");
        if(repeat<1)throw std::invalid_argument("repeat must be positive");
        T dst(out),x(a),y(b);Engine e(true);Scratch buffers;e.accumulation_scratch=&buffers;
        if(cache_bytes && (Engine::overlaps(dst,x) || Engine::overlaps(dst,y)))
            throw std::invalid_argument("cached inputs must not overlap output");
        e.cache_inputs(cache_bytes);
        {py::gil_scoped_release release;for(int i=0;i<repeat;++i)e.C(dst,spec,x,y,alpha);}
        auto t=e.timings.at(spec);py::dict result;
        result["direct_calls"]=t.direct_calls;result["packed_bytes"]=t.packed_bytes;
        result["result_bytes"]=t.result_bytes;result["scratch_bytes"]=size_t(buffers.capacity)*sizeof(double);
        result["cache_bytes"]=e.cache_bytes;result["cache_hits"]=e.cache_hits;
        return result;
    },py::arg("output").noconvert(),py::arg("subscripts"),py::arg("left").noconvert(),py::arg("right").noconvert(),py::arg("alpha")=1.,py::arg("cache_bytes")=0,py::arg("repeat")=1);
}
