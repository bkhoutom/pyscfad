// Copyright 2026 The PySCFAD Authors (Apache-2.0).
// Real RCCSD lambda equations adapted from PySCF/PySCFAD (Apache-2.0).
#include "tensor.h"
#include <pybind11/stl.h>

namespace ccsd_native {
namespace {
void outer_add(T out,const T& a,const T& b,double alpha=1.) {
    shape_is(out,{a.n[0],b.n[0],a.n[1],b.n[1]});
    const Index rows=out.n[0]*out.n[1]*out.n[2],v=out.n[3];
    #pragma omp parallel for schedule(static) if(rows*v>4096)
    for(Index row=0;row<rows;++row) {
        Index x=row%out.n[2],j=row/out.n[2]%out.n[1],i=row/(out.n[2]*out.n[1]);
        double f=alpha*a.p[i*a.s[0]+x*a.s[1]];auto off=out.offset(row);
        #pragma omp simd
        for(Index y=0;y<v;++y)out.p[off+y*out.s[3]]+=f*b.p[j*b.s[0]+y*b.s[1]];
    }
}
py::dict lambda_profile(const Engine& e,const Sections& sections) {
    py::dict report,regions,contractions;
    for(const auto& x:sections.times) {
        py::dict d;d["wall_seconds"]=x.second.first;d["cpu_seconds"]=x.second.second;regions[py::str(x.first)]=d;
    }
    for(const auto& x:e.timings) {
        const auto& t=x.second;py::dict d;d["pack_seconds"]=t.pack;d["gemm_seconds"]=t.gemm;
        d["accumulate_seconds"]=t.accumulate;d["packed_bytes"]=t.packed_bytes;d["calls"]=t.calls;
        d["direct_calls"]=t.direct_calls;d["result_bytes"]=t.result_bytes;contractions[py::str(x.first)]=d;
    }
    report["sections"]=regions;report["contractions"]=contractions;return report;
}
struct LambdaWorkspace {
    // Inputs belong to this solve and are immutable. Retain their Python owners,
    // but perform all tensor operations with native views after releasing GIL.
    std::vector<Input> owners;
    T t1,t2,fock,energy,oooo,ovoo,oovv,ovvo,packed,pairs;
    T vovv,theta,tau_full,w1,w2,w3,w4,woooo,wvooo,mixed,wvOOv,wvvov;
    T response_l1,response_l2,result_buffer;
    Scratch scratch;
    std::vector<size_t> virtual_pairs;
    std::vector<Index> occupied_rows;
    Index o,v,block;
    size_t panel_bytes;
    std::mutex mutex;
    LambdaWorkspace(const std::vector<Input>& arrays,Index tile,size_t panels)
        :owners(arrays),t1(arrays.at(0)),t2(arrays.at(1)),fock(arrays.at(2)),energy(arrays.at(3)),
         oooo(arrays.at(4)),ovoo(arrays.at(5)),oovv(arrays.at(6)),ovvo(arrays.at(7)),packed(arrays.at(8)),pairs(arrays.at(9)),
         o(t1.n[0]),v(t1.n[1]),block(tile),panel_bytes(panels) {
        if(t1.rank!=2 || o<1 || v<1 || block<1 || block>v)throw std::invalid_argument("invalid lambda dimensions");
        shape_is(t2,{o,o,v,v});shape_is(fock,{o+v,o+v});shape_is(energy,{o+v});
        shape_is(oooo,{o,o,o,o});shape_is(ovoo,{o,v,o,o});shape_is(oovv,{o,o,v,v});
        shape_is(ovvo,{o,v,v,o});shape_is(packed,{o,v,v*(v+1)/2});shape_is(pairs,{v*(v+1)/2,v*(v+1)/2});
        if(!pairs.contiguous())throw std::invalid_argument("lambda ladder pairs must be contiguous");
    }
    size_t resident_bytes() const {
        size_t n=0;for(const auto* t:{&vovv,&theta,&tau_full,&w1,&w2,&w3,&w4,&woooo,&wvooo,&mixed,&wvOOv,&wvvov})
            if(t->p)n+=sizeof(double)*t->size();
        return n+sizeof(size_t)*virtual_pairs.size()+sizeof(Index)*occupied_rows.size();
    }
    size_t working_bytes() const {
        size_t n=0;for(const auto* t:{&response_l1,&response_l2,&result_buffer})
            if(t->p)n+=sizeof(double)*t->size();return n;
    }
    void tiles(Index lo,Index hi,T& a,T& b,T& vv) const {
        if(vovv.p) {
            a=vovv.slice(0,lo,hi).perm({1,0,2,3});
            b=vovv.slice(2,lo,hi).perm({1,0,2,3});
        } else {
            a=unpack(packed.slice(1,lo,hi),v);b=T({o,v,hi-lo,v});
            #pragma omp parallel for schedule(static) if(b.size()>4096)
            for(Index row=0;row<o*v*(hi-lo);++row) {
                Index x=row%(hi-lo)+lo,c=row/(hi-lo)%v,j=row/((hi-lo)*v);
                Index off=j*packed.s[0]+c*packed.s[1];
                #pragma omp simd
                for(Index y=0;y<v;++y)b.p[row*v+y]=packed.p[off+pair(x,y)*packed.s[2]];
            }
        }
        vv=b.perm({2,3,0,1});
    }
    void composite(Engine& e,Index lo,Index hi,T& vo,T& oo) const {
        T x=ovvo.perm({1,0,3,2}).slice(0,lo,hi);
        vo=T({hi-lo,o,o,v});oo=T(vo.shape());
        {T g=linear(x,x.perm({0,2,1,3}),2.,-1.);e.A(vo,"bjld,klcd->bjkc",g,t2);}
        e.C(vo,"bjld,kldc->bjkc",x,t2,-1.);add(vo,x);
        e.A(oo,"bljd,kldc->bjkc",x,t2);add(oo,oovv.perm({2,1,0,3}).slice(0,lo,hi),-1.);
    }
    // Build W in two virtual tiles. All products accumulate directly into the
    // cached slice where its leading dimensions permit; no full-size W copies.
    void wvvov_block(Engine& e,Index lo,Index hi,const T& vo,const T& oo,
                     const T& g2ovoo,const T& l2,const T& singles) {
        T a,b,vv;tiles(lo,hi,a,b,vv);
        T g=linear(vv,a.perm({1,2,0,3}),2.,-1.);
        T one=linear(oo,vo,2.,1.),two=linear(vo,oo,2.,1.);
        T temp({hi-lo,v,o,block}),local;
        if(!wvvov.p)local=T({hi-lo,block,o,v});
        for(Index q=0;q<v;q+=block) {
            Index end=std::min(v,q+block),width=end-q;
            T out=wvvov.p?wvvov.slice(0,lo,hi).slice(1,q,end):local.slice(1,0,width);
            e.A(out,"jabd,jkcd->abkc",a.slice(2,q,end),t2,-1.5);
            add(out,vv.slice(3,q,end).perm({0,3,2,1}),2.);add(out,vv.slice(1,q,end),-1.);
            T t=temp.slice(3,0,width);
            e.A(t,"acjd,kjbd->ackb",g,theta.slice(2,q,end));add(out,t.perm({0,3,2,1}));
            e.C(out,"abjd,kjcd->abkc",g.slice(1,q,end),theta,-.5);
            e.C(out,"ajkc,jb->abkc",one,t1.slice(1,q,end));
            e.C(out,"ajkb,jc->abkc",two.slice(3,q,end),t1,-1.);
            e.C(out,"laki,klbc->abic",g2ovoo.slice(1,lo,hi),tau_full.slice(2,q,end));
            if(!wvvov.p)e.C(singles.slice(1,lo,hi),"abkc,kibc->ia",out,l2.slice(2,q,end));
        }
    }
    void build(Engine& e,Sections& sections,bool resident,bool cache_w) {
        occupied_rows=occupied_pair_rows(o);virtual_pairs.resize(v*v);
        for(Index a=0;a<v;++a)for(Index b=0;b<v;++b)virtual_pairs[a*v+b]=pair(a,b);
        if(resident)vovv=unpack(packed.perm({1,0,2}),v);
        theta=linear(t2,t2.perm({0,1,3,2}),2.,-1.);tau_full=tau(t2,t1,t1);
        T foo=fock.slice(0,0,o).slice(1,0,o),fov=fock.slice(0,0,o).slice(1,o,o+v);
        w1=copy(fock.slice(0,o,o+v).slice(1,o,o+v));w2=copy(foo);w3=copy(fov.perm({1,0}));w4=copy(fov);
        e.C(w1,"ja,jb->ba",fov,t1,-1.);e.C(w2,"ib,jb->ij",fov,t1);
        e.C(w3,"kc,jkbc->bj",fov,t2,2.);e.C(w3,"kc,kjbc->bj",fov,t2,-1.);
        {T t=e.E("kc,kb->cb",fov,t1);e.C(w3,"cb,jc->bj",t,t1);}
        wvooo=T({v,o,o,o});mixed=T({v,o,o,v});wvOOv=T({v,o,o,v});
        T theta_o=linear(t2,t2.perm({1,0,2,3}),2.,-1.);
        sections.mark("initialize");
        for(Index p=0;p<v;p+=block) {
            Index end=std::min(v,p+block);T a,b,vv;tiles(p,end,a,b,vv);
            e.C(w1.slice(1,p,end),"jcab,jc->ba",b,t1,2.);e.C(w1.slice(1,p,end),"jabc,jc->ba",a,t1,-1.);
            e.C(w3.slice(0,p,end),"jkcd,kdac->aj",theta_o,b);
            e.A(mixed.slice(0,p,end),"jbcd,kd->bjkc",a,t1);
            e.A(wvOOv.slice(0,p,end),"cbjd,kd->cjkb",vv,t1,-1.);
            T g=linear(vv.perm({0,2,1,3}),vv.perm({0,2,3,1}),2.,-1.);
            e.A(wvooo.slice(0,p,end),"cibd,jkbd->ckij",g,tau_full);
        }
        sections.mark("ovvv");
        e.C(w2,"kbij,kb->ij",ovoo,t1,2.);e.C(w2,"ibkj,kb->ij",ovoo,t1,-1.);
        e.C(w3,"lckj,klcb->bj",ovoo,theta_o.perm({1,0,2,3}),-1.);
        woooo=copy(oooo.perm({0,2,1,3}));
        {T tmp=e.E("lc,jcik->ijkl",t1,ovoo);add(woooo,tmp);add(woooo,tmp.perm({1,0,3,2}));}
        e.C(wvOOv,"lbjk,lc->bjkc",ovoo,t1);e.C(mixed,"jbkl,lc->bjkc",ovoo,t1,-1.);
        add(wvooo,ovoo.perm({1,3,2,0}),2.);add(wvooo,ovoo.perm({1,0,2,3}),-1.);
        e.C(wvooo,"klbc,iblj->ckij",t2,ovoo,-1.5);
        T g2ovoo=linear(ovoo,ovoo.perm({2,1,0,3}),2.,-1.);
        {T t=e.E("jlcb,lbki->cjki",theta_o,g2ovoo);add(wvooo,t.perm({0,3,2,1}));add(wvooo,t,-.5);}
        theta_o=T();
        sections.mark("ovoo");
        T voov=ovvo.perm({1,0,3,2});
        e.C(woooo,"cijd,klcd->ijkl",voov,tau_full);
        {T g=linear(voov,voov.perm({0,2,1,3}),2.,-1.);
         T tmp=e.E("ckld,ld->kc",g,t1);
         e.C(w1,"ckja,kjcb->ba",g,t2,-1.);e.C(w1,"ja,jb->ba",tmp,t1,-1.);
         e.C(w2,"jkbc,bikc->ij",t2,g);e.C(w2,"ib,jb->ij",tmp,t1);
         T x=e.E("ia,ib->ab",t1,tmp);e.C(w3,"ab,jb->aj",x,t1);add(w4,tmp);}
        {T t=e.E("bljd,kd->bljk",voov,t1);e.C(wvOOv,"bljk,lc->bjkc",t,t1);}
        {T t=e.E("bjld,kd->bjlk",voov,t1);e.C(mixed,"bjlk,lc->bjkc",t,t1,-1.);}
        if(cache_w)wvvov=T({v,v,o,v});
        for(Index p=0;p<v;p+=block) {
            Index end=std::min(v,p+block);T vo,oo;composite(e,p,end,vo,oo);
            add(mixed.slice(0,p,end),vo);add(wvOOv.slice(0,p,end),oo);
            {T one=linear(oo,vo,2.,1.),two=linear(vo,oo,2.,1.);
             e.C(wvooo,"ja,aikc->ckij",t1.slice(1,p,end),one,-1.);
             e.C(wvooo,"ka,aijc->ckij",t1.slice(1,p,end),two);
             e.C(w3,"akjb,ka->bj",two,t1.slice(1,p,end));}
            if(cache_w)wvvov_block(e,p,end,vo,oo,g2ovoo,T(),T());
        }
        // The update needs this combination only. Reuse its storage instead
        // of retaining both the original and a third ovov-sized layout.
        linear(mixed,wvOOv,2.,1.,mixed);
        e.C(w3,"bc,jc->bj",w1,t1);e.C(w3,"kj,kb->bj",w2,t1,-1.);
        if(cache_w)tau_full=T(); // single-use update tau is contracted in factored form
        sections.mark("ovov_and_wvvov");
    }
    void update(const T& l1,const T& l2,T r1,T r2,double shift,Engine& e,Sections& sections) {
        shape_is(l1,{o,v});shape_is(l2,{o,o,v,v});
        auto plan=ladder_panel_plan(v,panel_bytes,omp_get_max_threads());
        initialize_r2(r2,t1,l2,pairs,scratch,plan,virtual_pairs,occupied_rows,false);
        // initialize_r2 writes half the ladder directly into the output.
        e.A(r1,"ijab,jb->ia",r2,t1,4.);e.C(r1,"jiab,jb->ia",r2,t1,-2.);
        sections.mark("vvvv");
        T mba=e.E("klca,klcb->ba",l2,theta),mij=e.E("ikcd,jkcd->ij",l2,theta);
        T mba1=copy(mba),mij1=copy(mij),mia1=copy(t1);
        e.C(mba1,"jc,jb->bc",l1,t1);e.C(mij1,"kb,jb->kj",l1,t1);
        e.C(mia1,"kc,jkbc->jb",l1,t2,2.);e.C(mia1,"kc,jkcb->jb",l1,t2,-1.);
        {T x=e.E("ic,jc->ij",t1,l1);e.C(mia1,"ij,jb->ib",x,t1,-1.);}
        e.C(mia1,"bd,jd->jb",mba,t1,-1.);e.C(mia1,"lj,lb->jb",mij,t1,-1.);
        T v1=copy(w1),o2=copy(w2);
        for(Index a=0;a<v;++a)v1.p[a*v+a]-=energy.p[(o+a)*energy.s[0]]+shift;
        for(Index i=0;i<o;++i)o2.p[i*o+i]-=energy.p[i*energy.s[0]];
        add(r1,fock.slice(0,0,o).slice(1,o,o+v));
        e.C(r1,"ib,ba->ia",l1,v1);e.C(r1,"ja,ij->ia",l1,o2,-1.);
        e.C(r1,"ik,ka->ia",mij,w4,-1.);e.C(r1,"ca,ic->ia",mba,w4,-1.);
        e.C(r1,"ijab,bj->ia",l2,w3,2.);e.C(r1,"ijba,bj->ia",l2,w3,-1.);
        outer_add(r2,l1,w4);
        e.C(r2,"jibc,ca->jiba",l2,v1);e.C(r2,"jk,kiba->jiba",o2,l2,-1.);
        e.C(r1,"iajk,kj->ia",ovoo,mij1,-2.);e.C(r1,"jaik,kj->ia",ovoo,mij1);
        e.C(r2,"jbki,ka->jiba",ovoo,l1,-1.);
        e.C(r1,"jb,jiab->ia",l1,oovv,-1.);
        sections.mark("fock_and_density");
        // Lazily reserve one reusable result allocation for ovvv, ovov and
        // oooo. Keep its capacity across updates, like the contraction arena.
        if(!result_buffer.p)result_buffer=T({o,o,v,v});
        T m4=result_buffer;
        {T l2t1=e.E("jidc,kc->ijkd",l2,t1);
         for(Index p=0;p<v;p+=block) {
            Index end=std::min(v,p+block);T a,b,vv;tiles(p,end,a,b,vv);
            e.C(r1.slice(1,p,end),"iabc,bc->ia",a,mba1,2.);
            e.C(r1.slice(1,p,end),"ibac,bc->ia",b,mba1,-1.);
            e.C(r2.slice(3,p,end),"jbac,ic->jiba",b,l1);
            T tile=m4.slice(2,0,end-p);
            e.A(tile,"ijkd,kadb->ijab",l2t1,a);add(r2.slice(2,p,end),tile,-1.);
            e.C(r1.slice(1,p,end),"ijab,jb->ia",tile,t1,-2.);
            e.C(r1.slice(1,p,end),"jiab,jb->ia",tile,t1);
            e.C(r1,"ijab,ia->jb",tile,t1.slice(1,p,end),-2.);
            e.C(r1,"jiab,ia->jb",tile,t1.slice(1,p,end));
         }}
        if(wvvov.p)e.C(r1,"abkc,kibc->ia",wvvov,l2);
        else {
            T g=linear(ovoo,ovoo.perm({2,1,0,3}),2.,-1.);
            for(Index p=0;p<v;p+=block) {
                Index end=std::min(v,p+block);T vo,oo;composite(e,p,end,vo,oo);
                wvvov_block(e,p,end,vo,oo,g,l2,r1);
            }
        }
        sections.mark("ovvv");
        T voov=ovvo.perm({1,0,3,2});
        e.C(r1,"jb,aijb->ia",l1,voov,2.);add(r2,voov.perm({1,2,0,3}),.5);
        e.C(r2,"bjic,ca->jiba",voov,mba1,-1.);e.C(r2,"bjka,ik->jiba",voov,mij1,-1.);
        e.C(r1,"aijb,jb->ia",voov,mia1,2.);e.C(r1,"bija,jb->ia",voov,mia1,-1.);
        {T l2tau=e.E("ijcd,klcd->ijkl",l2,t2);
         {T t=e.E("ijcd,kc->ijkd",l2,t1);e.C(l2tau,"ijkd,ld->ijkl",t,t1);}
         e.A(m4,"ijkl,aklb->ijab",l2tau,voov);}
        add(r2,m4,.5);e.C(r1,"ijab,jb->ia",m4,t1,2.);e.C(r1,"ijba,jb->ia",m4,t1,-1.);
        e.C(r1,"ckij,jkca->ia",wvooo,l2,-1.);
        {T tmp({o,v,o,v},m4.owner);
         linear(l2.perm({0,2,1,3}),l2.perm({0,3,1,2}),1.,-.5,tmp);
         e.C(r2,"iakc,bjkc->jiba",tmp,mixed);}
        // Give the reusable result buffer the GEMM's natural (ja,bi) layout.
        // This removes an ovov-sized result in the packing arena and its copy.
        {T product({o,v,v,o},m4.owner);
         e.A(product,"jkca,bikc->jabi",l2,wvOOv);
         T view=product.perm({0,3,2,1});add(r2,view);add(r2,view.perm({1,0,2,3}),.5);}
        sections.mark("ovov");
        e.A(m4,"ijkl,klab->ijab",woooo,l2);add(r2,m4,.5);
        e.C(r1,"ijab,jb->ia",m4,t1,2.);e.C(r1,"ijba,jb->ia",m4,t1,-1.);
        sections.mark("oooo");
        finish_amplitudes(r1,r2,energy,shift,occupied_rows);
        sections.mark("finish");
    }
};

inline double value4(const T& t,Index i,Index j,Index a,Index b) {
    return t.p[i*t.s[0]+j*t.s[1]+a*t.s[2]+b*t.s[3]];
}
void inverse_response_metric(const T& x1,const T& x2,T l1,T l2,const T& energy,double shift) {
    const Index o=x1.n[0],v=x1.n[1];shape_is(x2,{o,o,v,v});shape_is(energy,{o+v});
    #pragma omp parallel if(o*o*v*v>4096)
    {
        #pragma omp for schedule(static) nowait
        for(Index ia=0;ia<o*v;++ia) {
            Index i=ia/v,a=ia%v;double d=energy.p[i*energy.s[0]]-energy.p[(o+a)*energy.s[0]]-shift;
            l1.p[ia]=.5*x1.p[i*x1.s[0]+a*x1.s[1]]/d;
        }
        #pragma omp for schedule(static)
        for(Index row=0;row<o*o*v;++row) {
            Index a=row%v,j=row/v%o,i=row/(o*v);
            double d=energy.p[i*energy.s[0]]+energy.p[j*energy.s[0]]-energy.p[(o+a)*energy.s[0]]-2*shift;
            #pragma omp simd
            for(Index b=0;b<v;++b) {
                double weight=i==j && a==b?1.:.5;
                l2.p[row*v+b]=weight*(2*value4(x2,i,j,a,b)+value4(x2,i,j,b,a))/(3*(d-energy.p[(o+b)*energy.s[0]]));
            }
        }
    }
}
void response_metric_inplace(T r1,T r2,const T& source1,const T& source2,const T& energy,double shift) {
    const Index o=r1.n[0],v=r1.n[1],tile=32,nt=(v+tile-1)/tile;
    const auto rows=occupied_pair_rows(nt);const Index vp=rows.size();
    std::vector<double> eps(o+v);
    for(Index i=0;i<o+v;++i)eps[i]=energy.p[i*energy.s[0]];
    #pragma omp parallel if(o*o*v*v>4096)
    {
        #pragma omp for schedule(static) nowait
        for(Index ia=0;ia<o*v;++ia) {
            Index i=ia/v,a=ia%v;double d=eps[i]-eps[o+a]-shift;
            r1.p[ia]=2*d*r1.p[ia]+source1.p[i*source1.s[0]+a*source1.s[1]];
        }
        // Every task owns a small tile and its virtual-exchange partner.
        // The inner loops have unit-stride reads/writes and no index division.
        #pragma omp for schedule(static)
        for(Index task=0;task<o*o*vp;++task) {
            Index ab=task%vp,at=rows[ab],bt=ab-at*(at+1)/2,ij=task/vp,i=ij/o,j=ij%o;
            Index a0=at*tile,a1=std::min(v,a0+tile),b0=bt*tile,b1=std::min(v,b0+tile);
            double* x=r2.p+ij*v*v;double eij=eps[i]+eps[j]-2*shift;
            for(Index a=a0;a<a1;++a) {
                Index end=at==bt?std::min(b1,a+1):b1;double eija=eij-eps[o+a];
                #pragma omp simd
                for(Index b=b0;b<end;++b) {
                    double weight=i==j && a==b?1.:2.,d=eija-eps[o+b];
                    double left=x[a*v+b],right=x[b*v+a];
                    x[a*v+b]=weight*d*(2*left-right)+value4(source2,i,j,a,b);
                    x[b*v+a]=weight*d*(2*right-left)+value4(source2,i,j,b,a);
                }
            }
        }
    }
}
py::tuple lambda_response(LambdaWorkspace& w,const Input& x1a,const Input& x2a,
                         const Input& s1a,const Input& s2a,double shift,bool profile) {
    T x1(x1a),x2(x2a),s1(s1a),s2(s2a);
    shape_is(x1,{w.o,w.v});shape_is(x2,{w.o,w.o,w.v,w.v});shape_is(s1,x1.shape());shape_is(s2,x2.shape());
    Array first({w.o,w.v}),second({w.o,w.o,w.v,w.v});T r1{Input(first)},r2{Input(second)};
    Engine e(profile);Sections sections(profile);e.accumulation_scratch=&w.scratch;
    {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(w.mutex);
     if(!w.response_l1.p)w.response_l1=T(x1.shape());
     if(!w.response_l2.p)w.response_l2=T(x2.shape());
     T l1=w.response_l1,l2=w.response_l2;inverse_response_metric(x1,x2,l1,l2,w.energy,shift);
     sections.mark("inverse_metric");w.update(l1,l2,r1,r2,shift,e,sections);
     l1=T();l2=T();response_metric_inplace(r1,r2,s1,s2,w.energy,shift);sections.mark("metric_and_source");}
    if(!profile)return py::make_tuple(first,second);
    py::dict report=lambda_profile(e,sections);report["scratch_bytes"]=sizeof(double)*w.scratch.capacity;
    report["resident_bytes"]=w.resident_bytes();report["working_bytes"]=w.working_bytes();
    return py::make_tuple(first,second,report);
}
py::tuple lambda_source(const Input& t1a,const Input& focka,const Input& ovova,const Input& ovvoa,
                        double bar_e,const Input& b1a,const Input& b2a) {
    T t1(t1a),fock(focka),ovov(ovova),ovvo(ovvoa),b1(b1a),b2(b2a);Index o=t1.n[0],v=t1.n[1];
    if(t1.rank!=2 || o<1 || v<1)throw std::invalid_argument("invalid lambda source");
    shape_is(fock,{o+v,o+v});shape_is(ovov,{o,v,o,v});shape_is(ovvo,{o,v,v,o});
    shape_is(b1,{o,v});shape_is(b2,{o,o,v,v});
    Array first({o,v}),second({o,o,v,v});T out1{Input(first)},out2{Input(second)};
    {py::gil_scoped_release release;
     #pragma omp parallel if(o*o*v*v>4096)
     {
        #pragma omp for schedule(static)
        for(Index ia=0;ia<o*v;++ia) {
            Index i=ia/v,a=ia%v;double f=fock.p[i*fock.s[0]+(o+a)*fock.s[1]],grad=2*f,zero=f;
            for(Index j=0;j<o;++j) {
                double gsum=0.,zsum=0.;
                #pragma omp simd reduction(+:gsum,zsum)
                for(Index b=0;b<v;++b) {
                    double t=t1.p[j*t1.s[0]+b*t1.s[1]];
                    double g=2*value4(ovov,i,a,j,b)-value4(ovov,i,b,j,a);
                    double gt=2*value4(ovov,j,b,i,a)-value4(ovov,j,a,i,b);
                    gsum+=(g+gt)*t;zsum+=(2*value4(ovvo,i,a,b,j)-value4(ovvo,i,b,a,j))*t;
                }
                grad+=gsum;zero+=zsum;
            }
            out1.p[ia]=bar_e*grad+b1.p[i*b1.s[0]+a*b1.s[1]]-2*zero;
        }
        #pragma omp for schedule(static)
        for(Index row=0;row<o*o*v;++row) {
            Index a=row%v,j=row/v%o,i=row/(o*v);
            #pragma omp simd
            for(Index b=0;b<v;++b) {
                double weight=i==j && a==b?.5:1.;
                double g=2*value4(ovov,i,a,j,b)-value4(ovov,i,b,j,a);
                double gt=2*value4(ovov,j,b,i,a)-value4(ovov,j,a,i,b);
                double za=.5*(value4(ovvo,i,a,b,j)+value4(ovvo,j,b,a,i));
                double zb=.5*(value4(ovvo,i,b,a,j)+value4(ovvo,j,a,b,i));
                out2.p[row*v+b]=weight*(bar_e*(g+gt)+value4(b2,i,j,a,b)+value4(b2,j,i,b,a)-2*(2*za-zb));
            }
        }
     }}
    return py::make_tuple(first,second);
}

py::tuple prepare_lambda(const std::vector<Input>& arrays,Index block,size_t panels,bool resident,bool cache_w,bool profile) {
    auto w=std::make_shared<LambdaWorkspace>(arrays,block,panels);Engine e(profile);Sections sections(profile);
    e.accumulation_scratch=&w->scratch;
    {py::gil_scoped_release release;w->build(e,sections,resident,cache_w);}
    return py::make_tuple(w,lambda_profile(e,sections));
}
py::tuple update_lambda(LambdaWorkspace& w,const Input& l1a,const Input& l2a,double shift,bool profile) {
    T l1(l1a),l2(l2a);Array first({w.o,w.v}),second({w.o,w.o,w.v,w.v});T r1{Input(first)},r2{Input(second)};
    Engine e(profile);Sections sections(profile);e.accumulation_scratch=&w.scratch;
    {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(w.mutex);
     w.response_l1=T();w.response_l2=T();w.update(l1,l2,r1,r2,shift,e,sections);}
    if(!profile)return py::make_tuple(first,second);
    py::dict report=lambda_profile(e,sections);report["scratch_bytes"]=sizeof(double)*w.scratch.capacity;
    report["resident_bytes"]=w.resident_bytes();report["working_bytes"]=w.working_bytes();
    return py::make_tuple(first,second,report);
}
} // namespace
} // namespace ccsd_native

void bind_lambda(py::module_& m) {
    using namespace ccsd_native;
    py::class_<LambdaWorkspace,std::shared_ptr<LambdaWorkspace>>(m,"LambdaWorkspace")
        .def("memory_info",[](LambdaWorkspace& w) {
            size_t resident,capacity,populated,working;
            {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(w.mutex);
             resident=w.resident_bytes();working=w.working_bytes();capacity=8*w.scratch.capacity;populated=8*w.scratch.populated;}
            py::dict d;d["resident_bytes"]=resident;d["scratch_capacity_bytes"]=capacity;
            d["working_bytes"]=working;d["populated_bytes"]=resident+working+populated;d["allocated_bytes"]=resident+working+capacity;
            d["resident_ovvv"]=bool(w.vovv.p);d["cached_wvvov"]=bool(w.wvvov.p);return d;
        })
        .def("intermediates",[](LambdaWorkspace& w) {
            py::dict d;
            for(auto x:{std::make_pair("w1",&w.w1),{"w2",&w.w2},{"w3",&w.w3},{"w4",&w.w4},
                        {"woooo",&w.woooo},{"wvooo",&w.wvooo},{"mixed",&w.mixed},{"wvOOv",&w.wvOOv},{"wvvov",&w.wvvov}}) {
                if(!x.second->p){d[py::str(x.first)]=py::none();continue;}
                Array a(x.second->shape());T dst{Input(a)};
                {py::gil_scoped_release release;std::lock_guard<std::mutex> lock(w.mutex);copy(*x.second,1.,dst);}
                d[py::str(x.first)]=a;
            }
            return d;
        });
    m.def("lambda_response",&lambda_response,py::arg("workspace"),py::arg("x1").noconvert(),py::arg("x2").noconvert(),
          py::arg("source1").noconvert(),py::arg("source2").noconvert(),py::arg("level_shift")=0.,py::arg("profile")=false);
    m.def("lambda_source",&lambda_source,py::arg("t1").noconvert(),py::arg("fock").noconvert(),py::arg("ovov").noconvert(),
          py::arg("ovvo").noconvert(),py::arg("bar_e"),py::arg("bar_t1").noconvert(),py::arg("bar_t2").noconvert());
    m.def("lambda_prepare",&prepare_lambda,py::arg("arrays"),py::arg("block_size"),py::arg("panel_bytes"),
          py::arg("resident_ovvv")=true,py::arg("cache_wvvov")=true,py::arg("profile")=false);
    m.def("lambda_update",&update_lambda,py::arg("workspace"),py::arg("l1").noconvert(),py::arg("l2").noconvert(),
          py::arg("level_shift")=0.,py::arg("profile")=false);
}
