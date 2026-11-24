// Shared orbital tensors have at most four indices. A constant-size occupied
// batch retains K/K-bar through all supplied points, then groups its adjoints
// by destination. No domain-wide K or K-bar exists.
#include "triples_one_permutation.inc"
#include "triples_grouped_adjoint.inc"

struct triples_thread_usage {
#if defined(__linux__)
    rusage usage{};
    explicit triples_thread_usage(bool enabled) {if(enabled) getrusage(RUSAGE_THREAD,&usage);}
    void accumulate(const triples_thread_usage& start,triples_right_profile& out) const {
        auto seconds=[](timeval t) {return double(t.tv_sec)+double(t.tv_usec)*1e-6;};
        out.cpu_seconds+=seconds(usage.ru_utime)+seconds(usage.ru_stime)
            -seconds(start.usage.ru_utime)-seconds(start.usage.ru_stime);
        out.minor_faults+=usage.ru_minflt-start.usage.ru_minflt;
        out.major_faults+=usage.ru_majflt-start.usage.ru_majflt;
        out.voluntary_switches+=usage.ru_nvcsw-start.usage.ru_nvcsw;
        out.involuntary_switches+=usage.ru_nivcsw-start.usage.ru_nivcsw;
    }
#else
    explicit triples_thread_usage(bool) {}
    void accumulate(const triples_thread_usage&,triples_right_profile&) const {}
#endif
};

struct triples_stream_bank {
    std::vector<tensor> tensors;
    std::unique_ptr<triples_bar_locks> locks;
    explicit triples_stream_bank(const stc_triples::triples_prepared& input,
            const triples_layout& layout,bool gradient) {
        tensors.reserve(layout.tensors.size());
        for(size_t i=0;i<layout.tensors.size();++i) {
            const auto& spec=layout.tensors[i];
            tensors.emplace_back(int(spec.kind),spec.dims,false,input.tensors[i].memptr());
            if(gradient) tensors.back().bar.set_size(tensors.back().value.n_elem);
        }
        locks=std::make_unique<triples_bar_locks>(tensors);
    }
};

stc_triples::triples_result triples_stream_kernel(const stc_triples::triples_prepared& left,
        const std::vector<const stc_triples::triples_prepared*>& points,
        const std::vector<double>& weights,size_t workspace_limit,size_t requested_batch,
        std::vector<double>& point_energies,std::vector<stc_triples::triples_vjp>* vjps,
        bool profile_right) {
    const double started=omp_get_wtime();
    const bool gradient=vjps!=nullptr;
    const size_t no=left.no,nv=left.nv,ns=left.ns,cube=nv*nv*nv;
    if(points.empty() || points.size()!=weights.size())
        throw std::invalid_argument("streaming triples: inconsistent quadrature points");
    for(size_t q=0;q<points.size();++q)
        if(points[q]->no!=no || points[q]->nv!=nv || points[q]->ns!=ns || !std::isfinite(weights[q]))
            throw std::invalid_argument("streaming triples: invalid point dimensions or weight");
    if(!requested_batch || requested_batch>stc_triples::maximum_streaming_blocks)
        throw std::invalid_argument("streaming triples: block_batch_size must be between 1 and 128");
    if(points.size()!=1 || points[0]!=&left)
        throw std::invalid_argument("MRCC triples requires one shared point bank");
    const triples_layout layout(no,nv,ns);
    std::vector<std::unique_ptr<triples_stream_bank>> banks;
    banks.push_back(std::make_unique<triples_stream_bank>(left,layout,gradient));
    triples_left_pack packed(banks[0]->tensors,layout,false,true);
    triples_direct_adjoint retained(banks[0]->tensors,layout,gradient);
    struct zero_task {double* data;size_t length;};
    std::vector<zero_task> zero_tasks;
    constexpr size_t panel=(2*1024*1024)/sizeof(double);
    if(gradient) for(auto& bank:banks) for(auto& t:bank->tensors)
        for(size_t first=0;first<t.bar.n_elem;first+=panel)
            zero_tasks.push_back({t.bar.memptr()+first,std::min(panel,size_t(t.bar.n_elem)-first)});
    for(auto* values:{&retained.gv_swap,&retained.to_normal,&retained.to_swap})
        for(auto& t:*values) for(size_t first=0;first<t.n_elem;first+=panel)
            zero_tasks.push_back({t.memptr()+first,std::min(panel,size_t(t.n_elem)-first)});
    const auto blocks=quadrature_blocks(no,ns);
    const size_t workspace=(8*cube+6*no*nv*nv+8*nv*nv)*sizeof(double)
        +(gradient ? triples_group_scratch::bytes(no,nv) : 0);
    const size_t block_bytes=cube*sizeof(double)*(gradient ? 2 : 1);
    // The caller combines its configurable workspace cap with available headroom.
    const size_t budget=workspace_limit;
    // Preserve one worker under a zero advisory budget. All other allocations
    // obey the available scratch/cache headroom and the hard 128-block cap.
    const size_t cache_budget=budget>workspace ? budget-workspace : 0;
    const size_t batch_size=std::min({requested_batch,blocks.size(),
        std::max(size_t(1),cache_budget/std::max(size_t(1),block_bytes))});
    const size_t scratch_budget=budget>batch_size*block_bytes ? budget-batch_size*block_bytes : 0;
    int workers=int(std::min({size_t(omp_get_max_threads()),batch_size,
        std::max(size_t(1),scratch_budget/std::max(size_t(1),workspace))}));
    workers=std::max(1,workers);
    auto mkl=reinterpret_cast<triples_local_blas::Setter>(dlsym(RTLD_DEFAULT,"MKL_Set_Num_Threads_Local"));
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    const bool ob=openblas_get_parallel()==1;
    detail::scoped_openblas_threads blas_scope(ob ? 1 : 0);
    if(!mkl && !ob) workers=1;
#else
    if(!mkl) workers=1;
#endif
    stc_triples::triples_result result{};
    result.workspace_budget_bytes=budget;
    result.right_profile_enabled=profile_right;
    result.point_right_wall_seconds.resize(points.size(),0.);
    result.point_right_batch_wall_seconds.resize(points.size());
    result.point_right_profiles.resize(points.size());
    std::vector<std::vector<triples_right_profile>> point_profiles(
        size_t(workers),std::vector<triples_right_profile>(points.size()));
    std::vector<double> energies(blocks.size()*points.size());
    std::vector<std::array<double,3>> worker_times(size_t(workers),{0.,0.,0.});
    std::vector<std::array<std::array<double,3>,4>> grouped_profiles{size_t(workers)};
    std::vector<arma::vec> ket(batch_size),ket_bar(gradient ? batch_size : 0);
    for(auto& value:ket) value.set_size(cube);
    for(auto& value:ket_bar) value.set_size(cube);
    triples_adjoint_plan plan;
    std::vector<size_t> calls(size_t(workers),0);
    double stage_start=0.,point_start=0.;
    double grouped_start=0.;
    std::exception_ptr failure;std::mutex error_lock;std::atomic<bool> failed{false};
    auto record_failure=[&]() {
        std::lock_guard<std::mutex> guard(error_lock);
        if(!failure) failure=std::current_exception();
        failed=true;
    };
    #pragma omp parallel num_threads(workers)
    {
        triples_local_blas local(workers>1 ? mkl : nullptr);
        const size_t worker=size_t(omp_get_thread_num());
        auto& times=worker_times[worker];
        #pragma omp single
        result.setup_threads=result.sampling_threads=omp_get_num_threads();
        #pragma omp for schedule(static,1)
        for(size_t z=0;z<zero_tasks.size();++z) std::fill_n(zero_tasks[z].data,zero_tasks[z].length,0.);
        triples_left_pack_forward(packed,banks[0]->tensors,nv);
        arma::vec raw,tmp;
        triples_raw_workspace workspace;
        triples_group_scratch grouped;
        try {
            raw.set_size(cube);tmp.set_size(cube);workspace.prepare(no,nv,gradient);
            if(gradient) grouped.prepare(no,nv);
        } catch(...) {record_failure();}
        for(size_t first=0;first<blocks.size();first+=batch_size) {
            const size_t count=std::min(batch_size,blocks.size()-first);
            #pragma omp single
            stage_start=omp_get_wtime();
            #pragma omp for schedule(dynamic,1)
            for(size_t b=0;b<count;++b) {
                if(failed) continue;
                try {
                    const auto ijk=blocks[first+b];
                    const tuple base{{{ijk[0],ijk[1],ijk[2],0,0,0}},0};
                    const double tick=omp_get_wtime();
                    triples_left_one_permutation(banks[0]->tensors,layout,base,packed,
                                                 no,nv,raw,tmp,workspace);
                    triples_spin_into(ket[b],raw,nv);
                    {
                        const double factor=-(ijk[1]==ijk[2] ? 1. : 2.)*weights[0]/3.;
                        energies[first+b]=factor*arma::dot(raw,ket[b]);
                        // Combined connected seed: A(2 f W + f L/2).
                        // Store the pre-spin seed until singles are added below.
                        if(gradient) ket_bar[b]=2.*factor*raw;
                    }
                    times[0]+=omp_get_wtime()-tick;
                } catch(...) {record_failure();}
            }
            #pragma omp single
            {result.left_build_wall_seconds+=omp_get_wtime()-stage_start;stage_start=omp_get_wtime();}
            for(size_t q=0;q<points.size();++q) {
                #pragma omp single
                point_start=omp_get_wtime();
                auto& point_profile=point_profiles[worker][q];
                auto* profile=profile_right ? &point_profile : nullptr;
                const triples_thread_usage usage_start(profile_right);
                #pragma omp for schedule(dynamic,1) nowait
                for(size_t index=0;index<count;++index) {
                    if(failed) continue;
                    try {
                        const auto b=blocks[first+index];
                        const tuple base{{{b[0],b[1],b[2],0,0,0}},0};
                        const double tick=omp_get_wtime();
                        auto& selected_bank=*banks[0];
                        auto& bank=selected_bank.tensors;auto& locks=*selected_bank.locks;
                        double energy=energies[first+index];
                        {
                            const double f=-(b[1]==b[2] ? 1. : 2.)*weights[q]/3.;
                            // Singles-only specialization neither reads the dummy raw
                            // numerator nor writes an unused connected cotangent cube.
                            std::array<operations,6> ops;
                            for(size_t p=0;p<6;++p)
                                ops[p]=quadrature_describe(layout,permute(base,p,true),1);
                            const std::array<quadrature_singles,6> singles{{
                                {bank,ops[0]},{bank,ops[1]},{bank,ops[2]},
                                {bank,ops[3]},{bank,ops[4]},{bank,ops[5]}}};
                            for(size_t p=0;p<6;++p) {
                                const auto& op=ops[p];
                                const size_t inverse=p==1 ? 2 : p==2 ? 1 : p;
#define MRCC_SINGLES(P) case P: energy+=gradient \
    ? mrcc_singles_contract<P,true>(ket[index],singles[p],nv,f,workspace,profile) \
    : mrcc_singles_contract<P,false>(ket[index],singles[p],nv,f,workspace,profile); break;
                                switch(inverse) {
                                    MRCC_SINGLES(0) MRCC_SINGLES(1) MRCC_SINGLES(2)
                                    MRCC_SINGLES(3) MRCC_SINGLES(4) MRCC_SINGLES(5)
                                }
#undef MRCC_SINGLES
                                if(gradient) quadrature_singles_store(bank,op,nv,workspace,locks,profile);
                            }
                            if(gradient) mrcc_singles_seed(ket_bar[index],singles,nv,f);
                        }
                        energies[(first+index)*points.size()+q]=energy;
                        const double elapsed=omp_get_wtime()-tick;
                        times[1]+=elapsed;point_profile.worker_seconds+=elapsed;
                        ++point_profile.blocks;
                    } catch(...) {record_failure();}
                }
                const double barrier_start=profile_right ? omp_get_wtime() : 0.;
                #pragma omp barrier
                if(profile_right) {
                    point_profile.barrier_seconds+=omp_get_wtime()-barrier_start;
                    triples_thread_usage(true).accumulate(usage_start,point_profile);
                }
                #pragma omp single
                {
                    const double elapsed=omp_get_wtime()-point_start;
                    result.point_right_wall_seconds[q]+=elapsed;
                    result.point_right_batch_wall_seconds[q].push_back(elapsed);
                    result.right_wall_seconds+=elapsed;
                }
            }
            #pragma omp single
            stage_start=omp_get_wtime();
            if(gradient) {
                // K is dead after the last point. Reuse its storage for the
                // main cotangent and K-bar storage for the alternate order.
                #pragma omp for schedule(dynamic,1)
                for(size_t b=0;b<count;++b) {
                    if(failed) continue;
                    const double tick=omp_get_wtime();
                    triples_spin_into(ket[b],ket_bar[b],nv);
                    for(size_t a=0;a<nv;++a) for(size_t c=0;c<nv;++c)
                        std::copy_n(ket[b].memptr()+(c*nv+a)*nv,nv,
                                    ket_bar[b].memptr()+(a*nv+c)*nv);
                    times[2]+=omp_get_wtime()-tick;
                }
                #pragma omp single
                {
                    result.grouped_spin_wall_seconds+=omp_get_wtime()-stage_start;
                    grouped_start=omp_get_wtime();
                    try {
                        if(!failed) {
                            plan=triples_group_plan(banks[0]->tensors,layout,blocks,first,count,nv);
                            result.grouped_adjoint_groups+=plan.groups.size();
                            result.grouped_adjoint_max_terms=std::max(result.grouped_adjoint_max_terms,plan.largest_group);
                        }
                    } catch(...) {record_failure();}
                    result.grouped_plan_wall_seconds+=omp_get_wtime()-grouped_start;
                    grouped_start=omp_get_wtime();
                }
                #pragma omp for schedule(dynamic,1)
                for(size_t j=0;j<plan.jobs.size();++j) {
                    if(failed) continue;
                    try {
                        const double tick=omp_get_wtime();
                        const auto& job=plan.jobs[j];const auto& group=plan.groups[job.group];
                        auto* gp=profile_right ? &grouped_profiles[worker][size_t(group.kind)] : nullptr;
                        if(group.kind==triples_adjoint_kind::gv)
                            calls[worker]+=triples_group_direct<triples_adjoint_kind::gv>(group,job,plan,
                                banks[0]->tensors,retained,ket,ket_bar,no,nv,gp);
                        else if(group.kind==triples_adjoint_kind::to)
                            calls[worker]+=triples_group_direct<triples_adjoint_kind::to>(group,job,plan,
                                banks[0]->tensors,retained,ket,ket_bar,no,nv,gp);
                        else calls[worker]+=triples_group_small(group,plan,banks[0]->tensors,packed,
                                ket,ket_bar,no,nv,workspace,grouped,gp);
                        times[2]+=omp_get_wtime()-tick;
                    } catch(...) {record_failure();}
                }
                #pragma omp single
                result.grouped_jobs_wall_seconds+=omp_get_wtime()-grouped_start;
            }
            #pragma omp single
            result.left_reverse_wall_seconds+=gradient ? omp_get_wtime()-stage_start : 0.;
        }
        if(gradient) {
            #pragma omp single
            stage_start=omp_get_wtime();
            retained.commit(banks[0]->tensors,no,nv);
            #pragma omp single
            {
                result.grouped_commit_wall_seconds=omp_get_wtime()-stage_start;
                result.left_reverse_wall_seconds+=result.grouped_commit_wall_seconds;
            }
        }
    }
    if(failure) std::rethrow_exception(failure);
    point_energies.assign(points.size(),0.);
    for(size_t i=0;i<blocks.size();++i) for(size_t q=0;q<points.size();++q)
        point_energies[q]+=energies[i*points.size()+q];
    for(double e:point_energies) result.energy+=e;
    if(!std::isfinite(result.energy)) throw std::runtime_error("streaming triples: nonfinite energy");
    for(const auto& t:worker_times) {
        result.left_build_worker_seconds+=t[0];result.right_worker_seconds+=t[1];
        result.left_reverse_worker_seconds+=t[2];
    }
    for(const auto& worker:point_profiles) for(size_t q=0;q<points.size();++q) {
        const auto& src=worker[q];auto& dst=result.point_right_profiles[q];
        dst.worker_seconds+=src.worker_seconds;dst.cpu_seconds+=src.cpu_seconds;
        dst.barrier_seconds+=src.barrier_seconds;dst.blocks+=src.blocks;
        dst.minor_faults+=src.minor_faults;dst.major_faults+=src.major_faults;
        dst.voluntary_switches+=src.voluntary_switches;dst.involuntary_switches+=src.involuntary_switches;
        for(size_t i=0;i<src.seconds.size();++i) dst.seconds[i]+=src.seconds[i];
        for(size_t i=0;i<8;++i) {
            dst.lock_wait[i]+=src.lock_wait[i];dst.store[i]+=src.store[i];dst.lock_calls[i]+=src.lock_calls[i];
        }
    }
    result.deterministic_blocks=blocks.size();
    result.deterministic_workspace_bytes=workspace*size_t(result.sampling_threads);
    result.numerator_cache_peak_bytes=block_bytes*batch_size;
    result.occupied_block_batch_size=batch_size;
    result.occupied_block_batches=(blocks.size()+batch_size-1)/batch_size;
    for(size_t c:calls) result.grouped_adjoint_gemm_calls+=c;
    for(const auto& p:grouped_profiles) for(size_t k=0;k<4;++k) for(size_t s=0;s<3;++s)
        result.grouped_profile_seconds[k][s]+=p[k][s];
    result.rank_four_packing_bytes=packed.bytes()+retained.bytes();
    for(const auto& bank:banks) for(const auto& t:bank->tensors)
        result.resident_tensor_bytes+=(t.value.n_elem+t.bar.n_elem)*sizeof(double);
    result.resident_tensor_bytes+=packed.bytes()+retained.bytes();
    if(gradient) {
        vjps->resize(banks.size());
        for(size_t q=0;q<banks.size();++q) {
            auto& out=(*vjps)[q].prepared;out.reserve(layout.tensors.size());
            for(auto& t:banks[q]->tensors) out.push_back(std::move(t.bar));
        }
        result.reverse_buffer_bytes=result.deterministic_workspace_bytes;
    }
    result.sampling_seconds=omp_get_wtime()-started;
    return result;
}
