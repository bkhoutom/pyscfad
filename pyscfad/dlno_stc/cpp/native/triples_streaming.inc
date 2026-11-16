// Shared orbital tensors have at most four indices. A constant-size occupied
// batch retains K/K-bar through all supplied points, then groups its adjoints
// by destination. No domain-wide K or K-bar exists.
#include "triples_one_permutation.inc"
#include "triples_grouped_adjoint.inc"

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
        std::vector<double>& point_energies,std::vector<stc_triples::triples_vjp>* vjps) {
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
    const triples_layout layout(no,nv,ns);
    std::vector<std::unique_ptr<triples_stream_bank>> banks;
    banks.push_back(std::make_unique<triples_stream_bank>(left,layout,gradient));
    for(const auto* point:points) banks.push_back(std::make_unique<triples_stream_bank>(*point,layout,gradient));
    triples_left_pack packed(banks[0]->tensors,layout,false);
    struct zero_task {double* data;size_t length;};
    std::vector<zero_task> zero_tasks;
    constexpr size_t panel=(2*1024*1024)/sizeof(double);
    if(gradient) for(auto& bank:banks) for(auto& t:bank->tensors)
        for(size_t first=0;first<t.bar.n_elem;first+=panel)
            zero_tasks.push_back({t.bar.memptr()+first,std::min(panel,size_t(t.bar.n_elem)-first)});
    const auto blocks=quadrature_blocks(no,ns);
    const size_t workspace=(8*cube+6*no*nv*nv+8*nv*nv)*sizeof(double)
        +(gradient ? triples_group_scratch::bytes(no,nv) : 0);
    const size_t block_bytes=cube*sizeof(double)*(gradient ? 2 : 1);
    const size_t budget=std::min(size_t(16)*1024*1024*1024,workspace_limit);
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
    std::vector<double> energies(blocks.size()*points.size());
    std::vector<std::array<double,3>> worker_times(size_t(workers),{0.,0.,0.});
    std::vector<arma::vec> ket(batch_size),ket_bar(gradient ? batch_size : 0);
    for(auto& value:ket) value.set_size(cube);
    for(auto& value:ket_bar) value.set_size(cube);
    triples_adjoint_plan plan;
    std::vector<size_t> calls(size_t(workers),0),reductions(size_t(workers),0);
    double stage_start=0.;
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
                    if(gradient) ket_bar[b].zeros();
                    times[0]+=omp_get_wtime()-tick;
                } catch(...) {record_failure();}
            }
            #pragma omp single
            {result.left_build_wall_seconds+=omp_get_wtime()-stage_start;stage_start=omp_get_wtime();}
            for(size_t q=0;q<points.size();++q) {
                #pragma omp for schedule(dynamic,1)
                for(size_t index=0;index<count;++index) {
                    if(failed) continue;
                    try {
                        const auto b=blocks[first+index];
                        const tuple base{{{b[0],b[1],b[2],0,0,0}},0};
                        const double tick=omp_get_wtime();
                        auto& bank=banks[q+1]->tensors;auto& locks=*banks[q+1]->locks;
                        const double factor=-2.*weights[q]/3.;
                        double energy=0.;
                        for(size_t swap=0;swap<(b[1]==b[2] ? 1u : 2u);++swap) {
                            tuple oriented=base;
                            if(swap) std::swap(oriented.x[1],oriented.x[2]);
                            for(size_t p=0;p<3;++p) {
                                const auto op=quadrature_describe(layout,permute(oriented,p,true),1);
                                triples_raw_block(bank,op,no,nv,false,raw,nullptr,nullptr,&workspace);
                                energy+=quadrature_right(tmp,gradient ? &ket_bar[index] : nullptr,raw,ket[index],
                                    bank,op,nv,p,bool(swap),factor,gradient,workspace);
                                if(gradient) {
                                    quadrature_singles_store(bank,op,nv,workspace,locks);
                                    triples_raw_block(bank,op,no,nv,false,raw,&tmp,&locks,&workspace,true);
                                }
                            }
                        }
                        energies[(first+index)*points.size()+q]=energy;
                        times[1]+=omp_get_wtime()-tick;
                    } catch(...) {record_failure();}
                }
            }
            #pragma omp single
            {result.right_wall_seconds+=omp_get_wtime()-stage_start;stage_start=omp_get_wtime();}
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
                    try {
                        if(!failed) {
                            plan=triples_group_plan(banks[0]->tensors,layout,blocks,first,count,nv);
                            result.grouped_adjoint_groups+=plan.groups.size();
                            result.grouped_adjoint_max_terms=std::max(result.grouped_adjoint_max_terms,plan.largest_group);
                        }
                    } catch(...) {record_failure();}
                }
                #pragma omp for schedule(dynamic,1)
                for(size_t j=0;j<plan.jobs.size();++j) {
                    if(failed) continue;
                    try {
                        const double tick=omp_get_wtime();
                        const auto& job=plan.jobs[j];const auto& group=plan.groups[job.group];
                        if(group.kind==triples_adjoint_kind::gv)
                            calls[worker]+=triples_group_large<triples_adjoint_kind::gv>(group,job,plan,
                                banks[0]->tensors,ket,ket_bar,no,nv,grouped,reductions[worker]);
                        else if(group.kind==triples_adjoint_kind::to)
                            calls[worker]+=triples_group_large<triples_adjoint_kind::to>(group,job,plan,
                                banks[0]->tensors,ket,ket_bar,no,nv,grouped,reductions[worker]);
                        else calls[worker]+=triples_group_small(group,plan,banks[0]->tensors,packed,
                                ket,ket_bar,no,nv,workspace,grouped);
                        times[2]+=omp_get_wtime()-tick;
                    } catch(...) {record_failure();}
                }
            }
            #pragma omp single
            result.left_reverse_wall_seconds+=gradient ? omp_get_wtime()-stage_start : 0.;
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
    result.deterministic_blocks=blocks.size();
    result.deterministic_workspace_bytes=workspace*size_t(result.sampling_threads);
    result.numerator_cache_peak_bytes=block_bytes*batch_size;
    result.occupied_block_batch_size=batch_size;
    result.occupied_block_batches=(blocks.size()+batch_size-1)/batch_size;
    for(size_t c:calls) result.grouped_adjoint_gemm_calls+=c;
    for(size_t r:reductions) result.grouped_adjoint_max_reduction=std::max(result.grouped_adjoint_max_reduction,r);
    result.rank_four_packing_bytes=packed.bytes();
    for(const auto& bank:banks) for(const auto& t:bank->tensors)
        result.resident_tensor_bytes+=(t.value.n_elem+t.bar.n_elem)*sizeof(double);
    result.resident_tensor_bytes+=packed.bytes();
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
