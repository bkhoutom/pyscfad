#ifndef LIBGMBPT_QCHEM_STC_MP2_STOCHASTIC_DENSITY_H
#define LIBGMBPT_QCHEM_STC_MP2_STOCHASTIC_DENSITY_H

// Selected-group density scheduler and occupied kernel, Notes Eq. (28).
// The estimator samples tuples, auxiliary atom blocks and the second Laplace index.
// No sign probes, amplitude tensor, or per-selected-orbital density is stored.
#include "stc_mp2_kernel_helpers.h"
#include <memory>
#include <array>
#include <functional>
#include <numeric>
#include <utility>

namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {

struct options {
    double domain_tau = 0.003;
    double frobenius_se_target = 1e-4;
    bool atom_block_l2 = false; // W option: use the column norm within each auxiliary atom.
    bool w_direct_joint_x_strong_h = true; // W only; density retains its four-leg direct split.
    bool w_direct_atom_pair = false; // W direct: share one B atom, weighted by unweighted column L1 sums.
    bool w_occupied_stratum_mixture = true; // Reuse B conditionals across W points.
    bool w_exchange_atom_pair = false; // W exchange: same B-atom proposal.
    bool w_virtual_reform = false; // W virtual: atom-paired three-term envelopes.
    bool w_virtual_split = true; // W virtual: shared B tables and exchange envelopes.
    bool w_parallel_update_reduction = true; // W VJP: reduce thread chunks concurrently.
    bool w_column_major_updates = true; // W VJP: sort column before auxiliary atom.
    // Exchange 0..3: four h/n orientations; 4: B atom pair; 5: C atom pair.
    // The default h/n mix avoids atom-conditioned exchange proposal tables.
    std::array<int,4> w_virtual_exchange_variants{{2,0,0,3}};
    // Direct -1: B atom pair. For 0..3, bit 1 puts h_B on the shared x leg,
    // bit 0 puts h_C on z; the complementary leg of each factor uses n.
    // 4/5: C atom pair on x,z, with h_B on y/x respectively. Entries are
    // term-major, with three residual strata per direct term.
    std::array<int,6> w_virtual_direct_variants{{-1,-1,-1,-1,-1,-1}};
    size_t pair_pilot_samples = 100000; // Match the energy pilot per active residual.
    size_t min_samples = 64;
    size_t replay_replicas = 0; // Internal replay count, never a sampling control.
    unsigned seed = 20261009;
    int threads = 1;
    size_t resident_points = 3; // Six/seven-point schedules reduce repeated propagations.
    bool symmetric_pairs = true;
};

inline std::array<int,4> parse_virtual_exchange_variants(const std::string& raw) {
    std::vector<int> values;
    size_t begin=0;
    while(begin<=raw.size()) {
        const size_t end=raw.find(',',begin);
        const std::string token=raw.substr(begin,end==std::string::npos ? end : end-begin);
        if(token.size()!=1 || token[0]<'0' || token[0]>'5')
            throw std::invalid_argument("Virtual exchange envelope must be 0..5");
        values.push_back(token[0]-'0');
        if(end==std::string::npos) break;
        begin=end+1;
    }
    if(values.size()!=1 && values.size()!=4)
        throw std::invalid_argument("Virtual exchange needs one or four envelopes");
    std::array<int,4> result{{0,0,0,0}};
    for(size_t k=0;k<4;++k) result[k]=values[values.size()==1 ? 0 : k];
    return result;
}

inline std::array<int,6> parse_virtual_direct_variants(const std::string& raw) {
    std::vector<int> values;
    size_t begin=0;
    while(begin<=raw.size()) {
        const size_t end=raw.find(',',begin);
        const std::string token=raw.substr(begin,end==std::string::npos ? end : end-begin);
        if(token!="-1" && (token.size()!=1 || token[0]<'0' || token[0]>'5'))
            throw std::invalid_argument("Virtual direct envelope must be -1 or 0..5");
        values.push_back(token=="-1" ? -1 : token[0]-'0');
        if(end==std::string::npos) break;
        begin=end+1;
    }
    if(values.size()!=1 && values.size()!=6)
        throw std::invalid_argument("Virtual direct needs one or six envelopes");
    std::array<int,6> result;
    for(size_t k=0;k<6;++k) result[k]=values[values.size()==1 ? 0 : k];
    return result;
}

struct result {
    arma::mat matrix, standard_error;
    // Per-pair trace samples retain occupied/virtual covariance for W energy.
    arma::mat trace_replicas;
    double frobenius_se = 0, trace_variance = 0, seconds = 0;
    double setup_seconds = 0, dense_seconds = 0, sampling_seconds = 0;
    // Detailed timers for the all-pair estimator; subsets of setup/sampling.
    double dressing_seconds = 0, point_table_seconds = 0, pair_table_seconds = 0;
    double pilot_seconds = 0, production_seconds = 0;
    double production_accumulation_seconds = 0, production_reduction_seconds = 0;
    // Mean worker elapsed time inside the direct/exchange production kernels.
    // Their sum excludes the production matrix reduction and thread launch.
    double direct_production_seconds = 0, exchange_production_seconds = 0;
    size_t direct_production_draws = 0, exchange_production_draws = 0;
    size_t production_replicas = 0, production_accumulator_bytes = 0;
    size_t pilot_draws = 0, production_draws = 0, batches = 0;
    size_t peak_table_bytes = 0, pair_preparations = 0, dense_pair_count = 0;
    arma::Mat<size_t> production_pair_batches;
    arma::Mat<size_t> pilot_pair_draws, production_pair_draws, residual_samples;
    arma::mat pair_stream_variance, pair_stream_cost, pair_variance_quota;
    size_t propagated_points = 0;
    double requested_samples = 0., predicted_se = 0.;
    bool target_met = false;
};

inline size_t matrix_bytes(const arma::mat& x) { return x.n_elem*sizeof(double); }
inline double sample_variance(const moments& x) {
    return x.count > 1 ? x.m2/double(x.count-1) : 0.;
}
inline void add_moment(moments& x, double value) {
    ++x.count;
    const double delta=value-x.mean;
    x.mean += delta/double(x.count);
    x.m2 += delta*(value-x.mean);
}
inline std::mt19937 generator(unsigned seed, size_t stage, size_t point,
                              size_t batch, size_t stream, size_t thread=0) {
    std::seed_seq sequence{seed, unsigned(stage), unsigned(point), unsigned(batch),
                          unsigned(stream), unsigned(thread)};
    return std::mt19937(sequence);
}

struct compact_integrals {
    size_t occupied=0, bytes=0;
    std::vector<arma::uvec> domains;
    std::vector<arma::mat> values;
    bool covers(const std::vector<arma::uvec>& requested) const {
        if (domains.size()!=requested.size()) return false;
        for (size_t j=0;j<domains.size();++j)
            if (!std::includes(domains[j].begin(),domains[j].end(),requested[j].begin(),requested[j].end()))
                return false;
        return true;
    }
    bool exact(const std::vector<arma::uvec>& requested) const {
        // Called only after covers(): equal sizes then imply identical domains.
        for (size_t j=0;j<domains.size();++j)
            if (domains[j].n_elem!=requested[j].n_elem) return false;
        return true;
    }
};

struct point_data {
    arma::mat c, bar_c, norm, atom_weight, atom_l1, sum;
    arma::umat strong;
    std::vector<alias_distribution> atoms;
    // Cache only selected-i compact integrals, never an additional C. The cap
    // covers all selected orbitals together, including domain/matrix metadata.
    size_t integral_cache_limit=size_t(1024)*1024*1024;
    mutable std::vector<std::shared_ptr<const compact_integrals>> integral_cache;
    mutable size_t integral_cache_builds=0, integral_cache_hits=0, integral_cache_subset_hits=0;
    mutable size_t integral_cache_skips=0;
    point_data() = default;
    // The fixed W factor only reads c. Borrow B for the lifetime of the W
    // estimator instead of allocating another full B-shaped matrix.
    explicit point_data(const arma::mat& borrowed)
        : c(const_cast<double*>(borrowed.memptr()),borrowed.n_rows,
            borrowed.n_cols,false,true) {}
    size_t integral_cache_bytes() const {
        size_t bytes=0;
        for (const auto& entry : integral_cache) bytes+=entry->bytes;
        return bytes;
    }
    std::shared_ptr<const compact_integrals> cached_integrals(size_t i,
            const std::vector<arma::uvec>& domains,int threads) const {
        if (integral_cache_bytes()>integral_cache_limit) integral_cache.clear();
        for (const auto& entry : integral_cache) if (entry->occupied==i && entry->covers(domains)) {
            ++integral_cache_hits;
            if (!entry->exact(domains)) ++integral_cache_subset_hits;
            return entry;
        }
        const size_t no=domains.size(),nv=norm.n_cols;
        // Overflow-safe accounting before allocating integral matrices.
        size_t bytes=sizeof(compact_integrals)+sizeof(std::shared_ptr<const compact_integrals>);
        bool fits=bytes<=integral_cache_limit;
        auto add=[&](size_t count,size_t width) {
            if (!fits || (width && count>(integral_cache_limit-bytes)/width)) fits=false;
            else bytes+=count*width;
        };
        add(no,sizeof(arma::uvec)+sizeof(arma::mat));
        for (const auto& domain : domains) {
            add(domain.n_elem,sizeof(arma::uword));
            add(domain.n_elem,domains[i].n_elem*sizeof(double));
        }
        size_t retained=0;
        for (const auto& entry : integral_cache) if (entry->occupied!=i) retained+=entry->bytes;
        if (!fits || retained>integral_cache_limit-bytes) { ++integral_cache_skips; return {}; }
        // Discard the old domain before building its replacement. Callers hold
        // entries only for the current selected i, never across point reloads.
        integral_cache.erase(std::remove_if(integral_cache.begin(),integral_cache.end(),
            [&](const std::shared_ptr<const compact_integrals>& x){return x->occupied==i;}),integral_cache.end());
        auto entry=std::make_shared<compact_integrals>();
        entry->occupied=i; entry->bytes=bytes; entry->domains=domains; entry->values.resize(no);
        // One selected-column pack shared by all partner-j workers.
        const arma::mat selected=c.cols(i*nv+domains[i]);
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (size_t j=0;j<no;++j) {
            if (!domains[i].n_elem || !domains[j].n_elem) continue;
            if (i==j) entry->values[j]=selected.t()*selected;
            else entry->values[j]=selected.t()*c.cols(j*nv+domains[j]);
        }
        integral_cache.push_back(entry); ++integral_cache_builds;
        return entry;
    }
    void prepare(size_t no, size_t nv, const arma::Col<size_t>& offsets,
                 double weight, double tau, int threads, bool atom_block_l2=false,
                 bool store_atom_l1=false) {
        const size_t na=offsets.n_elem-1;
        // prepare() follows every replacement of c; cached integrals belong to
        // the previous factor even when the allocation and dimensions match.
        integral_cache.clear();
        norm.set_size(no,nv); sum.set_size(no,nv); strong.set_size(no,nv);
        if (atom_block_l2) atom_weight.set_size(na,no*nv);
        else atom_weight.zeros(na,no*nv);
        if (store_atom_l1) atom_l1.zeros(na,no*nv);
        else atom_l1.reset();
        atoms.resize(no*nv);
        if (atom_block_l2) {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t ia=0;ia<no*nv;++ia) {
                const double* column=c.colptr(ia);
                double squared=0.;
                #pragma omp simd reduction(+:squared)
                for (size_t p=0;p<c.n_rows;++p) squared+=column[p]*column[p];
                norm(ia/nv,ia%nv)=squared>0 && std::isfinite(squared)
                    ? std::sqrt(squared) : arma::norm(c.col(ia));
            }
        }
        else {
            arma::mat row_squares(c.n_rows,threads,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                double* rows=row_squares.colptr(omp_get_thread_num());
                #pragma omp for schedule(static)
                for (size_t ia=0;ia<no*nv;++ia) {
                    const double* column=c.colptr(ia);
                    double squared=0.;
                    #pragma omp simd reduction(+:squared)
                    for (size_t p=0;p<c.n_rows;++p) {
                        const double x=column[p]*column[p]; rows[p]+=x; squared+=x;
                    }
                    norm(ia/nv,ia%nv)=squared>0 && std::isfinite(squared)
                        ? std::sqrt(squared) : arma::norm(c.col(ia));
                }
            }
            arma::vec rho=arma::sum(row_squares,1);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t p=0;p<c.n_rows;++p)
                rho(p)=rho(p)>0 && std::isfinite(rho(p)) ? std::sqrt(rho(p)) : arma::norm(c.row(p));
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t ia=0; ia<no*nv; ++ia) {
                const size_t i=ia/nv, a=ia%nv;
                strong(i,a)=std::sqrt(weight)*norm(i,a)>tau;
                for (size_t at=0; at<na; ++at)
                    for (size_t p=offsets(at); p<offsets(at+1); ++p) {
                        const double absolute=std::abs(c(p,ia));
                        atom_weight(at,ia)+=absolute*rho(p);
                        if (store_atom_l1) atom_l1(at,ia)+=absolute;
                    }
                sum(i,a)=arma::accu(atom_weight.col(ia));
                atoms[ia].reset(atom_weight.colptr(ia),atom_weight.colptr(ia)+na);
            }
        }
        if (atom_block_l2) {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t ia=0;ia<no*nv;++ia) {
                const size_t i=ia/nv,a=ia%nv;
                strong(i,a)=std::sqrt(weight)*norm(i,a)>tau;
                for (size_t at=0;at<na;++at) {
                    double squared=0.;
                    for (size_t p=offsets(at);p<offsets(at+1);++p) {
                        squared+=c(p,ia)*c(p,ia);
                        if (store_atom_l1) atom_l1(at,ia)+=std::abs(c(p,ia));
                    }
                    atom_weight(at,ia)=std::sqrt(squared);
                }
                sum(i,a)=arma::accu(atom_weight.col(ia));
                atoms[ia].reset(atom_weight.colptr(ia),atom_weight.colptr(ia)+na);
            }
        }
    }
    // Change only the deterministic-domain mask. Factors and sampling envelopes
    // can be shared by density blocks using different numerical cutoffs.
    void set_domain(double weight, double tau) {
        strong=(std::sqrt(weight)*norm>tau);
    }
    size_t table_bytes() const {
        size_t bytes=matrix_bytes(norm)+matrix_bytes(atom_weight)
                     +matrix_bytes(atom_l1)+matrix_bytes(sum)
                     +strong.n_elem*sizeof(arma::uword);
        for (const auto& a : atoms) bytes+=a.storage_bytes();
        return bytes;
    }
};

struct tuple { size_t i,j,k,a,b,A,T; };

// Fixed-B conditionals shared by every W Laplace point and by direct and
// exchange occupied streams. The masks are fixed because B's domain is fixed.
struct w_occupied_b_tables {
    size_t no,nv,na;
    std::array<std::vector<alias_distribution>,3> j_atom;
    arma::mat j_atom_total; // (T*nv+b, all/strong/weak)
    std::vector<alias_distribution> j_norm_weak,j_sum_all;
    arma::vec j_norm_weak_total,j_sum_all_total;
    explicit w_occupied_b_tables(const point_data& b,int threads):
        no(b.norm.n_rows),nv(b.norm.n_cols),na(b.atom_l1.n_rows),
        j_atom_total(na*nv,3,arma::fill::zeros),
        j_norm_weak_total(nv,arma::fill::zeros),
        j_sum_all_total(nv,arma::fill::zeros) {
        if(b.atom_l1.n_rows!=b.atom_weight.n_rows)
            throw std::invalid_argument("W occupied B tables require atom L1 weights");
        for(auto& tables:j_atom) tables.resize(na*nv);
        #pragma omp parallel num_threads(threads)
        {
            arma::vec all(no),strong(no),weak(no);
            #pragma omp for schedule(static)
            for(size_t tb=0;tb<na*nv;++tb) {
                const size_t t=tb/nv,v=tb%nv;
                for(size_t j=0;j<no;++j) {
                    const double value=b.atom_l1(t,j*nv+v);
                    all(j)=value;
                    strong(j)=b.strong(j,v) ? value : 0.;
                    weak(j)=b.strong(j,v) ? 0. : value;
                }
                j_atom_total(tb,0)=arma::accu(all);
                j_atom_total(tb,1)=arma::accu(strong);
                j_atom_total(tb,2)=arma::accu(weak);
                j_atom[0][tb].reset(all.begin(),all.end());
                j_atom[1][tb].reset(strong.begin(),strong.end());
                j_atom[2][tb].reset(weak.begin(),weak.end());
            }
        }
        j_norm_weak.resize(nv);j_sum_all.resize(nv);
        #pragma omp parallel num_threads(threads)
        {
            arma::vec weak(no),all(no);
            #pragma omp for schedule(static)
            for(size_t v=0;v<nv;++v) {
                for(size_t j=0;j<no;++j) {
                    weak(j)=b.strong(j,v) ? 0. : b.norm(j,v);
                    all(j)=b.sum(j,v);
                }
                j_norm_weak_total(v)=arma::accu(weak);
                j_sum_all_total(v)=arma::accu(all);
                j_norm_weak[v].reset(weak.begin(),weak.end());
                j_sum_all[v].reset(all.begin(),all.end());
            }
        }
    }
    size_t storage_bytes() const {
        size_t n=matrix_bytes(j_atom_total)+matrix_bytes(j_norm_weak_total)
            +matrix_bytes(j_sum_all_total);
        for(const auto& variant:j_atom) for(const auto& table:variant)
            n+=table.storage_bytes();
        for(const auto& table:j_norm_weak) n+=table.storage_bytes();
        for(const auto& table:j_sum_all) n+=table.storage_bytes();
        return n;
    }
};

// C_l-only conditionals. One atom-L1 k table serves direct stream 1 and
// exchange stream 0; norm/sum k tables serve the other occupied streams.
struct w_occupied_c_tables {
    size_t no,nv,na;
    std::vector<alias_distribution> k_atom,k_norm_all,k_norm_weak,k_sum_all;
    arma::vec k_atom_total,k_norm_all_total,k_norm_weak_total,k_sum_all_total;
    explicit w_occupied_c_tables(const point_data& c,int threads):
        no(c.norm.n_rows),nv(c.norm.n_cols),na(c.atom_l1.n_rows),
        k_atom_total(na*nv,arma::fill::zeros),
        k_norm_all_total(nv,arma::fill::zeros),
        k_norm_weak_total(nv,arma::fill::zeros),
        k_sum_all_total(nv,arma::fill::zeros) {
        if(c.atom_l1.n_rows!=c.atom_weight.n_rows)
            throw std::invalid_argument("W occupied C tables require atom L1 weights");
        k_atom.resize(na*nv);
        #pragma omp parallel num_threads(threads)
        {
            arma::vec weights(no);
            #pragma omp for schedule(static)
            for(size_t av=0;av<na*nv;++av) {
                const size_t a=av/nv,v=av%nv;
                for(size_t k=0;k<no;++k)
                    weights(k)=c.atom_l1(a,k*nv+v);
                k_atom_total(av)=arma::accu(weights);
                k_atom[av].reset(weights.begin(),weights.end());
            }
        }
        k_norm_all.resize(nv);k_norm_weak.resize(nv);k_sum_all.resize(nv);
        #pragma omp parallel num_threads(threads)
        {
            arma::vec norm(no),weak(no),sum(no);
            #pragma omp for schedule(static)
            for(size_t v=0;v<nv;++v) {
                for(size_t k=0;k<no;++k) {
                    norm(k)=c.norm(k,v);
                    weak(k)=c.strong(k,v) ? 0. : norm(k);
                    sum(k)=c.sum(k,v);
                }
                k_norm_all_total(v)=arma::accu(norm);
                k_norm_weak_total(v)=arma::accu(weak);
                k_sum_all_total(v)=arma::accu(sum);
                k_norm_all[v].reset(norm.begin(),norm.end());
                k_norm_weak[v].reset(weak.begin(),weak.end());
                k_sum_all[v].reset(sum.begin(),sum.end());
            }
        }
    }
    size_t storage_bytes() const {
        size_t n=matrix_bytes(k_atom_total)+matrix_bytes(k_norm_all_total)
            +matrix_bytes(k_norm_weak_total)+matrix_bytes(k_sum_all_total);
        for(const auto& table:k_atom) n+=table.storage_bytes();
        for(const auto& table:k_norm_all) n+=table.storage_bytes();
        for(const auto& table:k_norm_weak) n+=table.storage_bytes();
        for(const auto& table:k_sum_all) n+=table.storage_bytes();
        return n;
    }
};

// B-only virtual conditionals. Direct terms share a|(occupied,B atom),
// and all four h/n exchange orientations share a|occupied. Strong masks of
// B are fixed, so these aliases serve every W Laplace point in each pass.
struct w_virtual_b_tables {
    size_t no,nv,na;
    std::array<std::vector<alias_distribution>,3> atom_a;
    arma::mat atom_total; // (i*na+A, all/weak/strong)
    std::array<std::vector<alias_distribution>,6> exchange_a;
    arma::mat exchange_total; // (i, 3*kind+mask), kind=sum or norm
    explicit w_virtual_b_tables(const point_data& b,int threads):
        no(b.norm.n_rows),nv(b.norm.n_cols),na(b.atom_l1.n_rows),
        atom_total(no*na,3,arma::fill::zeros),
        exchange_total(no,6,arma::fill::zeros) {
        if(na!=b.atom_weight.n_rows)
            throw std::invalid_argument("W virtual B tables require atom L1 weights");
        for(auto& v:atom_a) v.resize(no*na);
        #pragma omp parallel num_threads(threads)
        {
            std::array<arma::vec,3> w{arma::vec(nv),arma::vec(nv),arma::vec(nv)};
            #pragma omp for schedule(static)
            for(size_t iA=0;iA<no*na;++iA) {
                const size_t i=iA/na,A=iA%na;
                for(size_t a=0;a<nv;++a) {
                    const double x=b.atom_l1(A,i*nv+a);
                    w[0](a)=x;
                    w[1](a)=b.strong(i,a) ? 0. : x;
                    w[2](a)=b.strong(i,a) ? x : 0.;
                }
                for(size_t mask=0;mask<3;++mask) {
                    atom_total(iA,mask)=arma::accu(w[mask]);
                    atom_a[mask][iA].reset(w[mask].begin(),w[mask].end());
                }
            }
        }
        for(auto& v:exchange_a) v.resize(no);
        #pragma omp parallel num_threads(threads)
        {
            std::array<arma::vec,6> w{arma::vec(nv),arma::vec(nv),arma::vec(nv),
                arma::vec(nv),arma::vec(nv),arma::vec(nv)};
            #pragma omp for schedule(static)
            for(size_t i=0;i<no;++i) {
                for(size_t a=0;a<nv;++a) {
                    const double sum=b.sum(i,a),norm=b.norm(i,a);
                    const bool strong=b.strong(i,a);
                    w[0](a)=sum; w[1](a)=strong ? 0. : sum; w[2](a)=strong ? sum : 0.;
                    w[3](a)=norm; w[4](a)=strong ? 0. : norm; w[5](a)=strong ? norm : 0.;
                }
                for(size_t variant=0;variant<6;++variant) {
                    exchange_total(i,variant)=arma::accu(w[variant]);
                    exchange_a[variant][i].reset(w[variant].begin(),w[variant].end());
                }
            }
        }
    }
    size_t storage_bytes() const {
        size_t n=matrix_bytes(atom_total)+matrix_bytes(exchange_total);
        for(const auto& v:atom_a) for(const auto& a:v) n+=a.storage_bytes();
        for(const auto& v:exchange_a) for(const auto& a:v) n+=a.storage_bytes();
        return n;
    }
};

// C_l-only b conditionals, shared by both virtual direct terms and all
// exchange orientations. The atom b tables are needed only for C-paired q.
struct w_virtual_c_tables {
    size_t no,nv,na;
    std::array<std::vector<alias_distribution>,4> b;
    arma::mat b_total; // (j, norm-all/norm-weak/sum-all/sum-weak)
    std::array<std::vector<alias_distribution>,2> atom_b;
    arma::mat atom_b_total; // (j*na+T, all/weak)
    explicit w_virtual_c_tables(const point_data& c,int threads,bool need_atom):
        no(c.norm.n_rows),nv(c.norm.n_cols),na(c.atom_weight.n_rows),
        b_total(no,4,arma::fill::zeros),
        atom_b_total(need_atom ? no*na : 0,need_atom ? 2 : 0,arma::fill::zeros) {
        for(auto& v:b) v.resize(no);
        #pragma omp parallel num_threads(threads)
        {
            std::array<arma::vec,4> w{arma::vec(nv),arma::vec(nv),arma::vec(nv),arma::vec(nv)};
            #pragma omp for schedule(static)
            for(size_t j=0;j<no;++j) {
                for(size_t a=0;a<nv;++a) {
                    const double norm=c.norm(j,a),sum=c.sum(j,a);
                    const bool strong=c.strong(j,a);
                    w[0](a)=norm; w[1](a)=strong ? 0. : norm;
                    w[2](a)=sum; w[3](a)=strong ? 0. : sum;
                }
                for(size_t variant=0;variant<4;++variant) {
                    b_total(j,variant)=arma::accu(w[variant]);
                    b[variant][j].reset(w[variant].begin(),w[variant].end());
                }
            }
        }
        if(need_atom) {
            if(c.atom_l1.n_rows!=na)
                throw std::invalid_argument("W virtual C atom tables require atom L1 weights");
            for(auto& v:atom_b) v.resize(no*na);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec all(nv),weak(nv);
                #pragma omp for schedule(static)
                for(size_t jT=0;jT<no*na;++jT) {
                    const size_t j=jT/na,T=jT%na;
                    for(size_t bidx=0;bidx<nv;++bidx) {
                        const double x=c.atom_l1(T,j*nv+bidx);
                        all(bidx)=x; weak(bidx)=c.strong(j,bidx) ? 0. : x;
                    }
                    atom_b_total(jT,0)=arma::accu(all);
                    atom_b_total(jT,1)=arma::accu(weak);
                    atom_b[0][jT].reset(all.begin(),all.end());
                    atom_b[1][jT].reset(weak.begin(),weak.end());
                }
            }
        }
    }
    size_t storage_bytes() const {
        size_t n=matrix_bytes(b_total)+matrix_bytes(atom_b_total);
        for(const auto& v:b) for(const auto& a:v) n+=a.storage_bytes();
        for(const auto& v:atom_b) for(const auto& a:v) n+=a.storage_bytes();
        return n;
    }
};

// Stratum -1 is the unrestricted proposal (also used by algebra tests).
// Density uses four first-weak-factor strata. W may combine B_x and C_x into
// one direct x leg, leaving three active direct strata and four exchange strata.
class proposal {
    const point_data &l_, &m_;
    const std::vector<size_t>& group_;
    size_t no_, nv_;
    bool exchange_;
    int stratum_;
    bool joint_x_direct_, strong_x_atom_source_, atom_pair_direct_, atom_pair_exchange_;
    alias_distribution root_, b_direct_;
    std::vector<alias_distribution> a_given_i_, b_given_i_, j_given_b_, k_given_v_;
    std::vector<alias_distribution> b_given_k_, jp_given_b_, ia_given_atom_;
    std::vector<alias_distribution> exchange_a_given_root_, exchange_b_given_root_;
    std::vector<alias_distribution> exchange_j_given_atom_b_, exchange_k_given_a_;
    const w_occupied_b_tables* b_cache_=nullptr;
    const w_occupied_c_tables* c_cache_=nullptr;
    int mixture_=0;
    std::vector<alias_distribution> mix_first_,mix_second_;
    double gate(size_t leg, size_t i, size_t a) const {
        if (joint_x_direct_) {
            if (stratum_<0) return 1.;
            if (stratum_>=3) return 0.;
            // Direct: x=(i,a) appears in both factors, while y=(j,b) and
            // z=(k,b) appear once. Split the residual by first weak slot:
            // !strong_x, strong_x&&!strong_y, strong_x&&strong_y&&!strong_z.
            if (leg==0) {
                const bool strong=l_.strong(i,a) && m_.strong(i,a);
                return stratum_==0 ? double(!strong) : double(strong);
            }
            if (leg==1)
                return stratum_==1 ? double(!l_.strong(i,a))
                    : stratum_==2 ? double(l_.strong(i,a)) : 1.;
            if (leg==2) return 1.;
            return stratum_==2 ? double(!m_.strong(i,a)) : 1.;
        }
        if (stratum_<0 || int(leg)>stratum_) return 1.;
        const bool h=(leg<2 ? l_.strong : m_.strong)(i,a);
        return int(leg)==stratum_ ? double(!h) : double(h);
    }
    static void reset(alias_distribution& target, const arma::vec& weight) {
        target.reset(weight.begin(),weight.end());
    }
    void build_mixture(int threads) {
        const size_t na=l_.atom_weight.n_rows;
        const bool direct=mixture_<=3;
        const size_t roots=(direct ? na : group_.size()*na);
        mix_first_.resize(roots); mix_second_.resize(roots);
        arma::vec root_weights(roots,arma::fill::zeros);
        #pragma omp parallel num_threads(threads)
        {
            arma::vec first(direct ? group_.size()*nv_ : nv_);
            arma::vec second(nv_);
            #pragma omp for schedule(static)
            for(size_t root=0;root<roots;++root) {
                const size_t atom=direct ? root : root%na;
                const size_t i=direct ? 0 : group_[root/na];
                if(direct) {
                    for(size_t ha=0;ha<group_.size()*nv_;++ha) {
                        const size_t x=group_[ha/nv_]*nv_+ha%nv_;
                        const bool sx=l_.strong(x/nv_,x%nv_) && m_.strong(x/nv_,x%nv_);
                        if(mixture_==1) // D4: weak joint x, C atom on z.
                            first(ha)=double(!sx)*l_.atom_l1(atom,x)*m_.norm(x/nv_,x%nv_);
                        else if(mixture_==2) // D2: strong x, C shared atom.
                            first(ha)=sx*l_.sum(x/nv_,x%nv_)*m_.atom_l1(atom,x);
                        else // D1: strong x and strong B_y.
                            first(ha)=sx*l_.atom_l1(atom,x)*m_.sum(x/nv_,x%nv_);
                    }
                    for(size_t b=0;b<nv_;++b) {
                        if(mixture_==1)
                            second(b)=b_cache_->j_atom_total(atom*nv_+b,0)*c_cache_->k_sum_all_total(b);
                        else if(mixture_==2)
                            second(b)=b_cache_->j_norm_weak_total(b)*c_cache_->k_atom_total(atom*nv_+b);
                        else
                            second(b)=b_cache_->j_atom_total(atom*nv_+b,1)*c_cache_->k_norm_weak_total(b);
                    }
                } else {
                    for(size_t a=0;a<nv_;++a) {
                        const size_t ia=i*nv_+a;
                        if(mixture_==4) // X3: weak B_x.
                            first(a)=double(!l_.strong(i,a))*l_.norm(i,a)*c_cache_->k_atom_total(atom*nv_+a);
                        else if(mixture_==5) // X1: B_x strong, B_y weak.
                            first(a)=l_.strong(i,a)*l_.atom_l1(atom,ia)*c_cache_->k_norm_all_total(a);
                        else if(mixture_==6) // X4: both B pairs strong, C_x weak.
                            first(a)=l_.strong(i,a)*l_.atom_l1(atom,ia)*c_cache_->k_sum_all_total(a);
                        else // X1: B,C_x strong, C_z weak.
                            first(a)=l_.strong(i,a)*l_.atom_l1(atom,ia)*c_cache_->k_norm_weak_total(a);
                    }
                    for(size_t b=0;b<nv_;++b) {
                        const size_t ib=i*nv_+b;
                        if(mixture_==4)
                            second(b)=b_cache_->j_sum_all_total(b)*m_.atom_l1(atom,ib);
                        else if(mixture_==5)
                            second(b)=b_cache_->j_atom_total(atom*nv_+b,2)*m_.sum(i,b);
                        else if(mixture_==6)
                            second(b)=b_cache_->j_atom_total(atom*nv_+b,1)
                                *double(!m_.strong(i,b))*m_.norm(i,b);
                        else
                            second(b)=b_cache_->j_atom_total(atom*nv_+b,1)*m_.strong(i,b)*m_.sum(i,b);
                    }
                }
                root_weights(root)=arma::accu(first)*arma::accu(second);
                reset(mix_first_[root],first);
                reset(mix_second_[root],second);
            }
        }
        normalization=arma::accu(root_weights);
        reset(root_,root_weights);
    }
public:
    double normalization=0.;
    proposal(const point_data& l, const point_data& m, const std::vector<size_t>& group,
             bool exchange, int stratum, bool joint_x_direct=false,
             bool atom_pair_direct=false, bool atom_pair_exchange=false,
             int threads=1,
             const w_occupied_b_tables* b_cache=nullptr,
             const w_occupied_c_tables* c_cache=nullptr): l_(l), m_(m), group_(group),
             no_(l.norm.n_rows), nv_(l.norm.n_cols), exchange_(exchange), stratum_(stratum),
             joint_x_direct_(joint_x_direct && !exchange),
             strong_x_atom_source_(joint_x_direct_ && stratum>=1 && stratum<=2),
             atom_pair_direct_(joint_x_direct_ && atom_pair_direct),
             atom_pair_exchange_(exchange && atom_pair_exchange),
             b_cache_(b_cache),c_cache_(c_cache) {
        if(b_cache_ && c_cache_) {
            if(!joint_x_direct)
                throw std::invalid_argument("W occupied mixture requires AND three-leg direct split");
            mixture_=exchange_ ? (stratum_==0 ? 4 : stratum_==1 ? 5 : stratum_==2 ? 6 : 7)
                : (stratum_==0 ? 1 : stratum_==1 ? 2 : stratum_==2 ? 3 : 0);
            if(mixture_) { build_mixture(threads); return; }
        }
        if (joint_x_direct_ && stratum_>=3) return;
        if (atom_pair_direct_) {
            // u_B,T(ia) u_B,T(jb) h_C,A(ia) ||C(kb)||, where u is the
            // unweighted atom L1 sum.  The C atom remains row weighted.
            const size_t na=l.atom_weight.n_rows;
            ia_given_atom_.resize(na);
            arma::vec atom_total(na,arma::fill::zeros);
            for (size_t T=0;T<na;++T) {
                arma::vec ia(group.size()*nv_);
                for (size_t h=0;h<group.size();++h) {
                    const size_t i=group[h];
                    for (size_t a=0;a<nv_;++a)
                        ia(h*nv_+a)=l.atom_l1(T,i*nv_+a)*m.sum(i,a)
                            *gate(0,i,a)*gate(2,i,a);
                }
                atom_total(T)=arma::accu(ia);
                reset(ia_given_atom_[T],ia);
            }
            jp_given_b_.resize(nv_);
            arma::vec b_total(nv_,arma::fill::zeros);
            for (size_t b=0;b<nv_;++b) {
                arma::vec jp(no_*na);
                for (size_t j=0;j<no_;++j)
                    for (size_t T=0;T<na;++T)
                        jp(j*na+T)=l.atom_l1(T,j*nv_+b)*atom_total(T)
                            *gate(1,j,b);
                b_total(b)=arma::accu(jp);
                reset(jp_given_b_[b],jp);
            }
            b_given_k_.resize(no_);
            arma::vec k_total(no_,arma::fill::zeros);
            for (size_t k=0;k<no_;++k) {
                arma::vec bw(nv_);
                for (size_t b=0;b<nv_;++b)
                    bw(b)=m.norm(k,b)*gate(3,k,b)*b_total(b);
                k_total(k)=arma::accu(bw);
                reset(b_given_k_[k],bw);
            }
            normalization=arma::accu(k_total);
            reset(root_,k_total);
            return;
        }
        if (atom_pair_exchange_) {
            // q = u_B,T(ia) u_B,T(jb) h_C,A(ib) ||C(ka)||.  Marginalize
            // the C atom before drawing b, then restore it as A|i,b.
            // Rooting at (selected i,B atom T) avoids an O(|S|*na*no*nv)
            // alias table for the equivalent k-first draw order.
            const size_t na=l.atom_weight.n_rows;
            exchange_k_given_a_.resize(nv_);
            arma::vec k_total(nv_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec kw(no_);
                #pragma omp for schedule(static)
                for (size_t a=0;a<nv_;++a) {
                    for (size_t k=0;k<no_;++k)
                        kw(k)=m.norm(k,a)*gate(3,k,a);
                    k_total(a)=arma::accu(kw);
                    reset(exchange_k_given_a_[a],kw);
                }
            }
            exchange_j_given_atom_b_.resize(na*nv_);
            arma::vec j_total(na*nv_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec jw(no_);
                #pragma omp for schedule(static)
                for (size_t tb=0;tb<na*nv_;++tb) {
                    const size_t T=tb/nv_,b=tb%nv_;
                    for (size_t j=0;j<no_;++j)
                        jw(j)=l.atom_l1(T,j*nv_+b)*gate(1,j,b);
                    j_total(tb)=arma::accu(jw);
                    reset(exchange_j_given_atom_b_[tb],jw);
                }
            }
            exchange_a_given_root_.resize(group.size()*na);
            exchange_b_given_root_.resize(group.size()*na);
            arma::vec root_weight(group.size()*na,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec aw(nv_),bw(nv_);
                #pragma omp for schedule(static)
                for (size_t root=0;root<group.size()*na;++root) {
                    const size_t h=root/na,T=root%na,i=group[h];
                    for (size_t a=0;a<nv_;++a) {
                        aw(a)=l.atom_l1(T,i*nv_+a)*gate(0,i,a)*k_total(a);
                        bw(a)=m.sum(i,a)*gate(2,i,a)*j_total(T*nv_+a);
                    }
                    root_weight(root)=arma::accu(aw)*arma::accu(bw);
                    reset(exchange_a_given_root_[root],aw);
                    reset(exchange_b_given_root_[root],bw);
                }
            }
            normalization=arma::accu(root_weight);
            reset(root_,root_weight);
            return;
        }
        arma::mat left(no_,nv_), right(no_,nv_);
        j_given_b_.resize(nv_); k_given_v_.resize(nv_);
        for (size_t b=0; b<nv_; ++b) {
            for (size_t j=0; j<no_; ++j) {
                left(j,b)=(exchange || strong_x_atom_source_ ? l.norm(j,b) : l.sum(j,b))*gate(1,j,b);
                right(j,b)=(strong_x_atom_source_ ? m.norm(j,b) : m.sum(j,b))*gate(3,j,b);
            }
            reset(j_given_b_[b],left.col(b)); reset(k_given_v_[b],right.col(b));
        }
        arma::vec ls=arma::sum(left,0).t(), rs=arma::sum(right,0).t();
        if (!exchange) {
            arma::vec ia(group.size()*nv_);
            for (size_t h=0; h<group.size(); ++h)
                for (size_t a=0; a<nv_; ++a) {
                    const size_t i=group[h];
                    ia(h*nv_+a)=(strong_x_atom_source_
                        ? l.sum(i,a)*m.sum(i,a) : l.norm(i,a)*m.norm(i,a))
                        *gate(0,i,a)*gate(2,i,a);
                }
            arma::vec b=ls%rs;
            normalization=arma::accu(ia)*arma::accu(b);
            reset(root_,ia); reset(b_direct_,b);
        } else {
            a_given_i_.resize(group.size()); b_given_i_.resize(group.size());
            arma::vec iw(group.size()), aw(nv_), bw(nv_);
            for (size_t h=0; h<group.size(); ++h) {
                const size_t i=group[h];
                for (size_t a=0; a<nv_; ++a) {
                    aw(a)=l.sum(i,a)*gate(0,i,a)*rs(a);
                    bw(a)=m.norm(i,a)*gate(2,i,a)*ls(a);
                }
                iw(h)=arma::accu(aw)*arma::accu(bw);
                reset(a_given_i_[h],aw); reset(b_given_i_[h],bw);
            }
            normalization=arma::accu(iw); reset(root_,iw);
        }
    }
    bool contains(const tuple& x) const {
        return std::find(group_.begin(),group_.end(),x.i)!=group_.end()
            && gate(0,x.i,x.a) && gate(1,x.j,x.b)
            && gate(2,x.i,exchange_ ? x.b : x.a)
            && gate(3,x.k,exchange_ ? x.a : x.b);
    }
    double probability(const tuple& x) const {
        if (!(normalization>0) || !contains(x)) return 0.;
        if(mixture_) {
            const size_t ia=x.i*nv_+x.a,jb=x.j*nv_+x.b;
            const size_t ib=x.i*nv_+x.b,kb=x.k*nv_+x.b,ka=x.k*nv_+x.a;
            const double bx=l_.atom_l1(x.A,ia),by=l_.atom_l1(x.A,jb);
            switch(mixture_) {
                case 1: return bx*by*m_.norm(x.i,x.a)*m_.atom_weight(x.T,kb)/normalization;
                case 2: return l_.atom_weight(x.A,ia)*l_.norm(x.j,x.b)
                    *m_.atom_l1(x.T,ia)*m_.atom_l1(x.T,kb)/normalization;
                case 3: return bx*by*m_.atom_weight(x.T,ia)*m_.norm(x.k,x.b)/normalization;
                case 4: return l_.norm(x.i,x.a)*l_.atom_weight(x.A,jb)
                    *m_.atom_l1(x.T,ib)*m_.atom_l1(x.T,ka)/normalization;
                case 6: return bx*by*m_.norm(x.i,x.b)*m_.atom_weight(x.T,ka)/normalization;
                default: return bx*by*m_.atom_weight(x.T,ib)*m_.norm(x.k,x.a)/normalization;
            }
        }
        if (atom_pair_direct_)
            return l_.atom_l1(x.A,x.i*nv_+x.a)
                *l_.atom_l1(x.A,x.j*nv_+x.b)
                *m_.atom_weight(x.T,x.i*nv_+x.a)
                *m_.norm(x.k,x.b)/normalization;
        if (atom_pair_exchange_)
            return l_.atom_l1(x.A,x.i*nv_+x.a)
                *l_.atom_l1(x.A,x.j*nv_+x.b)
                *m_.atom_weight(x.T,x.i*nv_+x.b)
                *m_.norm(x.k,x.a)/normalization;
        if (exchange_)
            return l_.norm(x.j,x.b)*m_.norm(x.i,x.b)
                *l_.atom_weight(x.A,x.i*nv_+x.a)*m_.atom_weight(x.T,x.k*nv_+x.a)/normalization;
        if (strong_x_atom_source_)
            return l_.norm(x.j,x.b)*m_.norm(x.k,x.b)
                *l_.atom_weight(x.A,x.i*nv_+x.a)*m_.atom_weight(x.T,x.i*nv_+x.a)/normalization;
        return l_.norm(x.i,x.a)*m_.norm(x.i,x.a)
            *l_.atom_weight(x.A,x.j*nv_+x.b)*m_.atom_weight(x.T,x.k*nv_+x.b)/normalization;
    }
    tuple draw(std::mt19937& rng) const {
        tuple x;
        const size_t h=root_(rng);
        if(mixture_) {
            const size_t na=l_.atom_weight.n_rows;
            if(mixture_<=3) {
                const size_t ia=mix_first_[h](rng);
                x.i=group_[ia/nv_]; x.a=ia%nv_; x.b=mix_second_[h](rng);
                if(mixture_==2) {
                    x.T=h;
                    x.j=b_cache_->j_norm_weak[x.b](rng);
                    x.k=c_cache_->k_atom[x.T*nv_+x.b](rng);
                    x.A=l_.atoms[x.i*nv_+x.a](rng);
                } else {
                    x.A=h;
                    x.j=b_cache_->j_atom[mixture_==1 ? 0 : 1][x.A*nv_+x.b](rng);
                    x.k=(mixture_==1 ? c_cache_->k_sum_all[x.b]
                        : c_cache_->k_norm_weak[x.b])(rng);
                    x.T=m_.atoms[(mixture_==1 ? x.k*nv_+x.b : x.i*nv_+x.a)](rng);
                }
            } else {
                x.i=group_[h/na];
                x.a=mix_first_[h](rng); x.b=mix_second_[h](rng);
                if(mixture_==4) {
                    x.T=h%na;
                    x.j=b_cache_->j_sum_all[x.b](rng);
                    x.k=c_cache_->k_atom[x.T*nv_+x.a](rng);
                    x.A=l_.atoms[x.j*nv_+x.b](rng);
                } else {
                    x.A=h%na;
                    x.j=b_cache_->j_atom[mixture_==5 ? 2 : 1][x.A*nv_+x.b](rng);
                    x.k=(mixture_==6 ? c_cache_->k_sum_all[x.a]
                        : mixture_==7 ? c_cache_->k_norm_weak[x.a]
                        : c_cache_->k_norm_all[x.a])(rng);
                    x.T=m_.atoms[(mixture_==6 ? x.k*nv_+x.a : x.i*nv_+x.b)](rng);
                }
            }
            return x;
        }
        if (atom_pair_direct_) {
            const size_t na=l_.atom_weight.n_rows;
            x.k=h;
            x.b=b_given_k_[x.k](rng);
            const size_t jp=jp_given_b_[x.b](rng);
            x.j=jp/na; x.A=jp%na; // x.A is the B atom; x.T is the C atom.
            const size_t ia=ia_given_atom_[x.A](rng);
            x.i=group_[ia/nv_]; x.a=ia%nv_;
            x.T=m_.atoms[x.i*nv_+x.a](rng);
            return x;
        }
        if (atom_pair_exchange_) {
            const size_t na=l_.atom_weight.n_rows;
            x.i=group_[h/na]; x.A=h%na;
            x.a=exchange_a_given_root_[h](rng);
            x.b=exchange_b_given_root_[h](rng);
            x.j=exchange_j_given_atom_b_[x.A*nv_+x.b](rng);
            x.k=exchange_k_given_a_[x.a](rng);
            x.T=m_.atoms[x.i*nv_+x.b](rng);
            return x;
        }
        if (exchange_) { x.i=group_[h]; x.a=a_given_i_[h](rng); x.b=b_given_i_[h](rng); }
        else { x.i=group_[h/nv_]; x.a=h%nv_; x.b=b_direct_(rng); }
        x.j=j_given_b_[x.b](rng); x.k=k_given_v_[exchange_ ? x.a : x.b](rng);
        x.A=l_.atoms[(exchange_ || strong_x_atom_source_ ? x.i*nv_+x.a : x.j*nv_+x.b)](rng);
        x.T=m_.atoms[strong_x_atom_source_ ? x.i*nv_+x.a
            : x.k*nv_+(exchange_ ? x.a : x.b)](rng);
        return x;
    }
    size_t table_bytes() const {
        size_t n=root_.storage_bytes()+b_direct_.storage_bytes();
        for (const auto* v : {&a_given_i_,&b_given_i_,&j_given_b_,&k_given_v_,
                              &b_given_k_,&jp_given_b_,&ia_given_atom_,
                              &exchange_a_given_root_,&exchange_b_given_root_,
                              &exchange_j_given_atom_b_,&exchange_k_given_a_,
                              &mix_first_,&mix_second_})
            for (const auto& a : *v) n+=a.storage_bytes();
        return n;
    }
};

inline double block_dot(const arma::mat& c, const arma::Col<size_t>& offsets,
                        size_t atom, size_t x, size_t y) {
    const size_t first=offsets(atom), count=offsets(atom+1)-first;
    if (!count) return 0.;
    return arma::dot(c.col(x).rows(first,first+count-1),c.col(y).rows(first,first+count-1));
}
inline double numerator(const point_data& l, const point_data& m,
                        const arma::Col<size_t>& offsets, const tuple& x, bool exchange) {
    const size_t nv=l.norm.n_cols;
    return (exchange ? -1. : 2.)
        *block_dot(l.c,offsets,x.A,x.i*nv+x.a,x.j*nv+x.b)
        *block_dot(m.c,offsets,x.T,x.i*nv+(exchange ? x.b : x.a),
                                       x.k*nv+(exchange ? x.a : x.b));
}

// Exact all-strong core. Virtual-pair panels bound workspace independently of
// nvirt^2; each thread accumulates one nocc-by-nocc matrix, not an amplitude tensor.
inline arma::mat dense_pair_panels(const point_data& l, const point_data& m,
                            const std::vector<size_t>& group, int threads) {
    const size_t no=l.norm.n_rows, nv=l.norm.n_cols, panel=128;
    const size_t panels=(nv*nv+panel-1)/panel;
    std::vector<arma::mat> partial(threads,arma::mat(no,no,arma::fill::zeros));
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (size_t task=0; task<group.size()*panels; ++task) {
        const size_t i=group[task/panels], first=(task%panels)*panel;
        const size_t count=std::min(panel,nv*nv-first);
        arma::mat u(no,count,arma::fill::zeros), spin(no,count,arma::fill::zeros);
        for (size_t q=0; q<count; ++q) {
            const size_t a=(first+q)/nv, b=(first+q)%nv;
            if (!l.strong(i,a) || (!m.strong(i,a) && !m.strong(i,b))) continue;
            for (size_t j=0; j<no; ++j) {
                if (l.strong(j,b)) u(j,q)=arma::dot(l.c.col(i*nv+a),l.c.col(j*nv+b));
                if (m.strong(i,a) && m.strong(j,b))
                    spin(j,q)=2*arma::dot(m.c.col(i*nv+a),m.c.col(j*nv+b));
                if (m.strong(i,b) && m.strong(j,a))
                    spin(j,q)-=arma::dot(m.c.col(i*nv+b),m.c.col(j*nv+a));
            }
        }
        partial[omp_get_thread_num()]+=u*spin.t();
    }
    arma::mat result(no,no,arma::fill::zeros);
    for (const auto& p : partial) result+=p;
    return result;
}

struct domain_overlap {
    arma::uvec left,right;
    domain_overlap() = default;
    domain_overlap(const arma::uvec& a,const arma::uvec& b) {
        std::vector<arma::uword> ai,bi;
        size_t i=0,j=0;
        while (i<a.n_elem && j<b.n_elem) {
            if (a(i)<b(j)) ++i;
            else if (b(j)<a(i)) ++j;
            else { ai.push_back(i++); bi.push_back(j++); }
        }
        left=arma::uvec(ai); right=arma::uvec(bi);
    }
};

// A cached virtual domain may cover a smaller occupied domain. Extract only
// the required entries; exact-domain hits return the cached matrix directly.
inline const arma::mat& compact_block(const compact_integrals& cache,size_t j,
        const std::vector<arma::uvec>& domains,arma::mat& scratch) {
    const size_t i=cache.occupied;
    if (domains[i].n_elem==cache.domains[i].n_elem && domains[j].n_elem==cache.domains[j].n_elem)
        return cache.values[j];
    if (!domains[i].n_elem || !domains[j].n_elem) { scratch.reset(); return scratch; }
    const domain_overlap rows(cache.domains[i],domains[i]),cols(cache.domains[j],domains[j]);
    scratch=cache.values[j].submat(rows.left,cols.left);
    return scratch;
}

inline const std::vector<arma::mat>& compact_blocks(const point_data& point,size_t i,
        const std::vector<arma::uvec>& domains,int threads,std::vector<arma::mat>& scratch) {
    const auto cached=point.cached_integrals(i,domains,threads);
    if (cached && cached->exact(domains)) return cached->values;
    const size_t no=domains.size(),nv=point.norm.n_cols;
    scratch.resize(no);
    arma::mat selected;
    if (!cached) selected=point.c.cols(i*nv+domains[i]);
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (size_t j=0;j<no;++j) {
        if (!domains[i].n_elem || !domains[j].n_elem) continue;
        if (cached) scratch[j]=compact_block(*cached,j,domains,scratch[j]);
        else if (i==j) scratch[j]=selected.t()*selected;
        else scratch[j]=selected.t()*point.c.cols(j*nv+domains[j]);
    }
    return scratch;
}

// Build compact (ia|jb) blocks once per selected i, then contract only matching
// virtual-domain entries. Selected factor columns are packed once per point/pair
// and shared by the workers. At l=m the right integral blocks alias the left.
// Bound the integral cache; large/all-strong domains retain the panel fallback.
inline arma::mat dense_pair(const point_data& l, const point_data& m,
        const std::vector<size_t>& group, int threads,
        size_t max_integral_bytes=size_t(256)*1024*1024) {
    const size_t no=l.norm.n_rows,nv=l.norm.n_cols;
    const bool diagonal=&l==&m;
    std::vector<arma::uvec> ld(no),md(diagonal ? 0 : no);
    size_t left_columns=0,right_columns=0;
    for (size_t j=0;j<no;++j) {
        ld[j]=arma::find(l.strong.row(j).t()); left_columns+=ld[j].n_elem;
        if (!diagonal) {
            md[j]=arma::find(m.strong.row(j).t()); right_columns+=md[j].n_elem;
        }
    }
    const auto& right=diagonal ? ld : md;
    arma::mat answer(no,no,arma::fill::zeros);
    for (size_t i : group) {
        const auto &li=ld[i], &mi=right[i];
        if (!li.n_elem || !mi.n_elem) continue;
        const size_t integral_bytes=sizeof(double)*(li.n_elem*left_columns
            +(diagonal ? 0 : mi.n_elem*right_columns));
        if (integral_bytes>max_integral_bytes) {
            answer+=dense_pair_panels(l,m,{i},threads);
            continue;
        }
        std::vector<arma::mat> left_scratch,right_scratch;
        const auto& u=compact_blocks(l,i,ld,threads,left_scratch);
        const auto& v=diagonal ? u : compact_blocks(m,i,right,threads,right_scratch);
        const domain_overlap selected_common(li,mi);
        std::vector<domain_overlap> exchange_a(no),exchange_b(no);
        for (size_t j=0;j<no;++j) {
            exchange_a[j]=domain_overlap(li,right[j]);
            exchange_b[j]=domain_overlap(ld[j],mi);
        }
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (size_t j=0;j<no;++j) {
            if (u[j].is_empty()) continue;
            for (size_t k=diagonal ? j : 0;k<no;++k) {
                if (v[k].is_empty()) continue;
                double value=0.;
                if (selected_common.left.n_elem) {
                    const domain_overlap common(ld[j],right[k]);
                    if (common.left.n_elem)
                        value=2*arma::accu(u[j].submat(selected_common.left,common.left)
                            %v[k].submat(selected_common.right,common.right));
                }
                const auto &a=exchange_a[k], &b=exchange_b[j];
                if (a.left.n_elem && b.left.n_elem)
                    value-=arma::accu(u[j].submat(a.left,b.left)
                        %v[k].submat(b.right,a.right).t());
                answer(j,k)+=value;
                if (diagonal && j!=k) answer(k,j)+=value;
            }
        }
    }
    return answer;
}

struct residual_result {
    arma::mat mean;
    double frobenius_variance=0., draw_seconds=0.;
};
inline residual_result sample_residual(const proposal& table, const point_data& l,
        const point_data& m, const arma::Col<size_t>& offsets, bool exchange,
        size_t count, int threads, unsigned seed, size_t stage, size_t point,
        size_t batch, size_t stream) {
    const size_t no=l.norm.n_rows;
    residual_result answer; answer.mean.zeros(no,no);
    if (!(table.normalization>0)) return answer;
    std::vector<arma::mat> partial(threads,arma::mat(no,no,arma::fill::zeros));
    std::vector<double> square(threads,0.);
    const double draw_start=omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int shard=0;shard<threads;++shard) {
        const size_t tid=size_t(shard);
        auto rng=generator(seed,stage,point,batch,stream,tid);
        const size_t begin=count*tid/size_t(threads);
        const size_t end=count*(tid+1)/size_t(threads);
        for (size_t n=begin; n<end; ++n) {
            const tuple x=table.draw(rng);
            const double value=numerator(l,m,offsets,x,exchange)/table.probability(x);
            partial[tid](x.j,x.k)+=value;
            square[tid]+=(x.j==x.k ? 1. : .5)*value*value;
        }
    }
    answer.draw_seconds=omp_get_wtime()-draw_start;
    for (const auto& value : partial) answer.mean+=value;
    answer.mean/=double(count);
    const arma::mat sym=(answer.mean+answer.mean.t())/2.;
    answer.frobenius_variance=std::max(0.,(std::accumulate(square.begin(),square.end(),0.)
        -double(count)*arma::accu(arma::square(sym)))/double(count-1));
    return answer;
}

struct occupied_policy {
    using table_type=proposal;
    static constexpr size_t streams=8;
    static constexpr size_t direct_streams=4;
    static size_t dimension(size_t no, size_t) { return no; }
    static table_type* table(const point_data& l, const point_data& m,
            const std::vector<size_t>& group, size_t stream, bool joint_x_direct,
            bool atom_pair_direct, bool atom_pair_exchange, bool, int threads,
            const w_occupied_b_tables* b_cache,
            const w_occupied_c_tables* c_cache,
            const w_virtual_b_tables*,const w_virtual_c_tables*,int,int) {
        return new proposal(l,m,group,stream>=4,int(stream%4),
                            joint_x_direct,atom_pair_direct,atom_pair_exchange,
                            threads,b_cache,c_cache);
    }
    static arma::mat core(const point_data& l, const point_data& m,
            const std::vector<size_t>& group, int threads) {
        return dense_pair(l,m,group,threads);
    }
    // One independent production estimate, accumulated into a persistent
    // thread-owned matrix. Pilot statistics still use sample() below.
    static void accumulate(const table_type& table, const point_data& l, const point_data& m,
            const arma::Col<size_t>& offsets, size_t count, size_t stream,
            std::mt19937& rng, arma::mat& value,double weight=1.) {
        if(!count) return;
        const double scale=weight/double(count);
        for (size_t n=0;n<count;++n) {
            const auto x=table.draw(rng);
            value(x.j,x.k)+=scale*numerator(l,m,offsets,x,stream>=4)/table.probability(x);
        }
    }
    static residual_result sample(const table_type& p, const point_data& l, const point_data& m,
            const arma::Col<size_t>& offsets, size_t count, const options& opt,
            size_t stage, size_t point, size_t batch, size_t stream) {
        return sample_residual(p,l,m,offsets,stream>=4,count,opt.threads,opt.seed,
                               stage,point,batch,stream);
    }
};

inline void validate(const arma::mat& fo, const arma::mat& fv, const arma::mat& b,
        const arma::Col<size_t>& offsets, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, const options& opt) {
    if (!fo.n_rows || !fv.n_rows || fo.n_rows!=fo.n_cols || fv.n_rows!=fv.n_cols
        || b.n_cols!=fo.n_rows*fv.n_rows || !b.n_rows || !b.is_finite()
        || !fo.is_finite() || !fv.is_finite() || arma::norm(fo-fo.t(),"fro")>1e-10
        || arma::norm(fv-fv.t(),"fro")>1e-10 || offsets.n_elem<2 || offsets(0)!=0
        || offsets(offsets.n_elem-1)!=b.n_rows || !beta.n_elem || beta.n_elem!=weights.n_elem
        || !beta.is_finite() || !weights.is_finite() || beta.min()<0 || weights.min()<=0)
        throw std::invalid_argument("Invalid density inputs");
    for (size_t a=1; a<offsets.n_elem; ++a)
        if (offsets(a)<offsets(a-1)) throw std::invalid_argument("Invalid auxiliary atom offsets");
    std::vector<size_t> sorted=group; std::sort(sorted.begin(),sorted.end());
    if (sorted.empty() || sorted.back()>=fo.n_rows
        || std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
        throw std::invalid_argument("Invalid occupied density group");
    if (!std::isfinite(opt.domain_tau) || opt.domain_tau<0
        || !std::isfinite(opt.frobenius_se_target) || opt.frobenius_se_target<=0
        || opt.min_samples<2 || opt.pair_pilot_samples<2
        || (opt.replay_replicas && opt.replay_replicas<2)
        || opt.threads<1 ||
            (opt.resident_points!=2 && opt.resident_points!=3))
        throw std::invalid_argument("Invalid density sampling options");
}

// Independent Laplace-pair strata remove random-partner variance entirely.
struct pair_allocation {
    arma::Mat<size_t> per_batch;
    double requested_samples=0., predicted_se=0.;
};

inline pair_allocation allocate_pair_samples(const arma::mat& variance,
        const arma::umat& active, const options& opt, size_t batches,
        const arma::mat& cost) {
    pair_allocation out; out.per_batch.zeros(variance.n_rows,variance.n_cols);
    if (!variance.is_finite() || variance.min()<0.)
        throw std::runtime_error("Nonfinite density pilot variance");
    const arma::uvec indices=arma::find(active);
    if (!indices.n_elem) return out;
    const double target=opt.frobenius_se_target;
    if (cost.n_rows!=variance.n_rows || cost.n_cols!=variance.n_cols || !cost.is_finite())
        throw std::invalid_argument("Invalid density pilot draw costs");
    arma::vec sd(indices.n_elem), units(indices.n_elem), root_cost(indices.n_elem,arma::fill::ones);
    for (size_t k=0;k<indices.n_elem;++k) sd(k)=std::sqrt(variance(indices(k)));
    for (size_t k=0;k<indices.n_elem;++k) {
        if (!(cost(indices(k))>0.)) throw std::invalid_argument("Density pilot draw cost must be positive");
        root_cost(k)=std::sqrt(cost(indices(k)));
    }
    const double weighted_sd=arma::dot(sd,root_cost);
    // Minimize estimated draw time at the requested Frobenius variance.
    for (size_t k=0;k<indices.n_elem;++k)
        units(k)=std::max(double(opt.min_samples),std::ceil(sd(k)/root_cost(k)*weighted_sd/(double(batches)*target*target)));
    out.requested_samples=double(batches)*arma::accu(units);
    if (!std::isfinite(out.requested_samples) ||
        out.requested_samples>=double(std::numeric_limits<size_t>::max()))
        throw std::runtime_error("Density requested sample budget overflow");
    double predicted=0.;
    for (size_t k=0;k<indices.n_elem;++k) {
        if (units(k)>=double(std::numeric_limits<size_t>::max()))
            throw std::runtime_error("Density pair sample count overflow");
        out.per_batch(indices(k))=size_t(units(k));
        predicted+=variance(indices(k))/(double(batches)*units(k));
    }
    out.predicted_se=std::sqrt(predicted);
    return out;
}

// Fixed point weights use the same pre-pilot Fock/weight heuristic as energy.
inline arma::vec fixed_point_weights_from_spectrum(const arma::vec& eo,const arma::vec& ev,
        const arma::vec& beta, const arma::vec& weights) {
    auto log_trace=[](const arma::vec& x) {
        const double largest=x.max();
        return largest+std::log(arma::accu(arma::exp(x-largest)));
    };
    arma::vec log_weight(beta.n_elem);
    for (size_t l=0;l<beta.n_elem;++l)
        log_weight(l)=std::log(weights(l))+1.4*(log_trace(.5*beta(l)*eo)+log_trace(-.5*beta(l)*ev));
    arma::vec point=arma::exp(log_weight-log_weight.max());
    point=arma::clamp(point,1e-150,std::numeric_limits<double>::max());
    point/=arma::accu(point);
    return point;
}
inline arma::vec fixed_point_weights(const arma::mat& fo, const arma::mat& fv,
        const arma::vec& beta, const arma::vec& weights) {
    arma::vec eo,ev;
    if (!arma::eig_sym(eo,fo) || !arma::eig_sym(ev,fv))
        throw std::runtime_error("Cannot construct Laplace point weights");
    return fixed_point_weights_from_spectrum(eo,ev,beta,weights);
}

// For unordered density pairs, q_lm=(2-delta_lm) p_l p_m; their sum is one.
inline arma::mat fixed_pair_quotas(const arma::mat& fo, const arma::mat& fv,
        const arma::vec& beta, const arma::vec& weights, bool symmetric) {
    const arma::vec point=fixed_point_weights(fo,fv,beta,weights);
    arma::mat quota(beta.n_elem,beta.n_elem,arma::fill::zeros);
    for (size_t l=0;l<beta.n_elem;++l) for (size_t m=symmetric ? l : 0;m<beta.n_elem;++m)
        quota(l,m)=(symmetric && l!=m ? 2. : 1.)*point(l)*point(m);
    return quota/arma::accu(quota);
}

// Serpentine upper-triangle traversal: consecutive off-diagonal pairs share a
// vertex. A diagonal is processed on its factor's first load. Ordered estimates
// process both orientations together. The three-point path owns three C arrays.
template<class Prepare, class Visit, class Evict>
inline void visit_pairs_once(size_t nl, bool symmetric, Prepare prepare, Visit visit,
        Evict evict, size_t resident_points=2) {
    if (resident_points!=2 && resident_points!=3)
        throw std::invalid_argument("Density traversal requires two or three resident factors");
    if (resident_points==3) {
        point_data data[3];
        size_t resident[3]={nl,nl,nl};
        std::vector<bool> visited(nl*nl,false);
        auto load=[&](size_t point,size_t slot) {
            if (resident[slot]<nl) evict(resident[slot],data[slot]);
            prepare(point,data[slot]); resident[slot]=point;
            size_t bytes=0;
            for (const auto& p : data) bytes+=p.table_bytes();
            // Complete every new pair exposed by this load, including its diagonal.
            for (size_t other=0;other<3;++other) if (resident[other]<nl) {
                const size_t l=std::min(point,resident[other]),m=std::max(point,resident[other]);
                if (visited[l*nl+m]) continue;
                const size_t a=point==l ? slot : other,b=point==l ? other : slot;
                visit(l,m,data[a],data[b],bytes);
                if (!symmetric && l!=m) visit(m,l,data[b],data[a],bytes);
                visited[l*nl+m]=true;
            }
        };
        for (size_t p=0;p<std::min(nl,size_t(3));++p) load(p,p);
        // Optimal six/seven-point schedules, given as (new point, evicted point).
        std::vector<std::pair<size_t,size_t>> replacements;
        if (nl==6) replacements={{3,0},{4,1},{5,3},{1,2},{0,1},{3,4}};
        if (nl==7) replacements={{3,0},{4,1},{0,2},{5,0},{6,4},{0,3},{2,0},{1,2},{4,5}};
        for (const auto& step : replacements)
            for (size_t slot=0;slot<3;++slot) if (resident[slot]==step.second) {
                load(step.first,slot); break;
            }
        // Other quadrature sizes use a bounded three-slot traversal. Retain
        // either endpoint when loading its missing partner; no pair repeats.
        for (size_t l=0;l<nl;++l) for (size_t m=l+1;m<nl;++m) if (!visited[l*nl+m]) {
            for (size_t point : {l,m}) {
                if (std::find(resident,resident+3,point)!=resident+3) continue;
                const size_t retain=point==l ? m : l;
                size_t slot=0;
                while (resident[slot]==retain) ++slot;
                load(point,slot);
            }
        }
        for (size_t slot=0;slot<3;++slot) if (resident[slot]<nl)
            evict(resident[slot],data[slot]);
        return;
    }
    point_data data[2];
    size_t resident[2]={nl,nl};
    std::vector<bool> diagonal(nl,false);
    auto load=[&](size_t point,size_t retain) {
        for (size_t slot=0;slot<2;++slot) if (resident[slot]==point) return slot;
        const size_t slot=resident[0]==retain ? 1 : 0;
        if (resident[slot]<nl) evict(resident[slot],data[slot]);
        prepare(point,data[slot]); resident[slot]=point;
        if (!diagonal[point]) {
            visit(point,point,data[slot],data[slot],data[0].table_bytes()+data[1].table_bytes());
            diagonal[point]=true;
        }
        return slot;
    };
    auto pair=[&](size_t l,size_t m) {
        const size_t a=load(l,m),b=load(m,l);
        const size_t bytes=data[0].table_bytes()+data[1].table_bytes();
        visit(l,m,data[a],data[b],bytes);
        if (!symmetric) visit(m,l,data[b],data[a],bytes);
    };
    if (nl==1) load(0,nl);
    for (size_t l=0;l+1<nl;++l) {
        if (l%2==0) for (size_t m=l+1;m<nl;++m) pair(l,m);
        else for (size_t m=nl;m-- > l+1;) pair(l,m);
    }
    for (size_t slot=0;slot<2;++slot) if (resident[slot]<nl)
        evict(resident[slot],data[slot]);
}
template<class Prepare, class Visit>
inline void visit_pairs_once(size_t nl, bool symmetric, Prepare prepare, Visit visit,
        size_t resident_points=2) {
    visit_pairs_once(nl,symmetric,prepare,visit,[](size_t,point_data&){},resident_points);
}

// Independent block state; the shared scheduler supplies the same point data to
// both blocks. RNG keys and per-block allocation are unchanged by interleaving.
template<class Policy>
class pair_estimator {
    const options& opt;
    const std::vector<size_t>& group;
    const arma::Col<size_t>& offsets;
    const size_t nl, dimension, batches;
    const bool primal_outputs;
    const bool joint_x_direct;
    const bool atom_pair_direct;
    const bool atom_pair_exchange;
    const bool virtual_reform;
    arma::umat active;
    std::vector<arma::mat> production_values;
    const arma::Mat<size_t>* replay_counts;
public:
    using table_list=std::vector<std::unique_ptr<typename Policy::table_type>>;
    struct vjp_callback {
        std::function<void()> core;
        std::function<void(const typename Policy::table_type&,
            const arma::Col<size_t>&,size_t,size_t,std::mt19937&,arma::mat&,double)> production;
    };
    result answer;
    pair_estimator(size_t no, size_t nv, size_t points,
            const arma::Col<size_t>& off, const std::vector<size_t>& selected, const options& settings,
            const arma::Mat<size_t>* saved_counts=nullptr,bool output_matrices=true,
            bool three_leg_direct=false)
        : opt(settings), group(selected), offsets(off), nl(points),
          dimension(Policy::dimension(no,nv)),
          // One replica per worker, at least two to measure matrix SE.
          batches(opt.replay_replicas ? opt.replay_replicas : std::max(size_t(2),size_t(opt.threads))),
          primal_outputs(output_matrices),
          joint_x_direct(three_leg_direct),
          atom_pair_direct(three_leg_direct && opt.w_direct_atom_pair),
          atom_pair_exchange(three_leg_direct && opt.w_exchange_atom_pair),
          virtual_reform(opt.w_virtual_reform),
          replay_counts(saved_counts) {
        if (replay_counts && (!opt.replay_replicas ||
            replay_counts->n_rows!=Policy::streams || replay_counts->n_cols!=nl*nl))
            throw std::invalid_argument("Invalid density replay allocation");
        if (!primal_outputs && !replay_counts)
            throw std::invalid_argument("Reverse-only density pass requires saved allocation");
        answer.production_replicas=batches;
        if (primal_outputs) {
            answer.matrix.zeros(dimension,dimension);
            answer.standard_error.zeros(dimension,dimension);
            answer.trace_replicas.zeros(nl*nl,batches);
        }
        answer.production_pair_batches.zeros(nl,nl);
        answer.pilot_pair_draws.zeros(nl,nl); answer.production_pair_draws.zeros(nl,nl);
        answer.pair_stream_variance.zeros(Policy::streams,nl*nl);
        answer.pair_stream_cost.zeros(Policy::streams,nl*nl);
        answer.residual_samples.zeros(Policy::streams,nl*nl); active.zeros(Policy::streams,nl*nl);
        if (replay_counts) answer.residual_samples=*replay_counts;
    }
    void set_quotas(const arma::mat& quota) {
        answer.pair_variance_quota=quota;
    }
    void allocate_local(size_t l,size_t m,std::ostream* log,const char* label) {
        const size_t key=l*nl+m;
        const double share=answer.pair_variance_quota(l,m);
        options local=opt;
        local.frobenius_se_target*=std::sqrt(share);
        const auto allocation=allocate_pair_samples(answer.pair_stream_variance.col(key),
            active.col(key),local,batches,answer.pair_stream_cost.col(key));
        answer.residual_samples.col(key)=allocation.per_batch.col(0);
        answer.requested_samples+=allocation.requested_samples;
        answer.predicted_se=std::hypot(answer.predicted_se,allocation.predicted_se);
        if (log) *log << label << "Density local allocation " << l << ',' << m
            << " variance_quota = " << share << " requested_samples = " << allocation.requested_samples
            << " allocated_samples = " << double(batches)*double(arma::accu(allocation.per_batch))
            << " predicted_pair_se = " << allocation.predicted_se << '\n';
    }
    void process(size_t l, size_t m, double factor,
            const point_data& left, const point_data& right, size_t point_bytes, std::ostream* log, const char* label="",
            const vjp_callback& vjp=vjp_callback(),
            const w_occupied_b_tables* b_cache=nullptr,
            const w_occupied_c_tables* c_cache=nullptr,
            const w_virtual_b_tables* vb_cache=nullptr,
            const w_virtual_c_tables* vc_cache=nullptr) {
        const size_t key=l*nl+m, streams=Policy::streams;
        const double pair_table_start=omp_get_wtime();
        std::vector<std::unique_ptr<typename Policy::table_type>> tables;
        size_t bytes=point_bytes;
        bool any=false;
        for (size_t stream=0;stream<streams;++stream) {
            tables.emplace_back(Policy::table(left,right,group,stream,
                                              joint_x_direct,atom_pair_direct,
                                              atom_pair_exchange,virtual_reform,opt.threads,
                                              b_cache,c_cache,vb_cache,vc_cache,
                                              opt.w_virtual_exchange_variants[stream%4],
                                              stream<8 && stream%4<3 ?
                                                opt.w_virtual_direct_variants[(stream/4)*3+stream%4] : -1));
            bytes+=tables.back()->table_bytes();
            active(stream,key)=tables.back()->normalization>0;
            any=any || active(stream,key);
        }
        answer.peak_table_bytes=std::max(answer.peak_table_bytes,bytes);
        ++answer.pair_preparations;
        answer.pair_table_seconds+=omp_get_wtime()-pair_table_start;
        answer.setup_seconds+=omp_get_wtime()-pair_table_start;
        if (replay_counts) {
            for (size_t stream=0;stream<streams;++stream)
                if (!active(stream,key) && answer.residual_samples(stream,key))
                    throw std::invalid_argument("Density replay proposal support changed");
        }
        if (!replay_counts) {
            for (size_t stream=0;stream<streams;++stream) if (active(stream,key)) {
                    const double sample_start=omp_get_wtime();
                    const auto r=Policy::sample(*tables[stream],left,right,offsets,
                        opt.pair_pilot_samples,opt,0,key,0,stream);
                    const double elapsed=omp_get_wtime()-sample_start;
                    answer.sampling_seconds+=elapsed;
                    answer.pilot_seconds+=elapsed;
                    answer.pair_stream_variance(stream,key)+=factor*factor*r.frobenius_variance;
                    answer.pair_stream_cost(stream,key)+=std::max(r.draw_seconds,1e-15)
                        /double(opt.pair_pilot_samples);
                    if(log && std::getenv("QCHEM_STC_W_LOG_PILOT_STREAMS"))
                        *log << label << "Pilot point=" << l
                            << " stream=" << stream << " variance="
                            << r.frobenius_variance << " cost="
                            << r.draw_seconds/double(opt.pair_pilot_samples) << '\n';
                    answer.pilot_pair_draws(l,m)+=opt.pair_pilot_samples;
                    answer.pilot_draws+=opt.pair_pilot_samples;
                }
            allocate_local(l,m,log,label);
        }
        // Integrate every pair core exactly once, reusing resident factors and
        // selected compact integrals. No random-partner 1/p(m|l) factor.
        const double dense_start=omp_get_wtime();
        if (primal_outputs) {
            const arma::mat core=Policy::core(left,right,group,opt.threads);
            answer.matrix+=factor*arma::mat((core+core.t())/2.);
        }
        ++answer.dense_pair_count; answer.dense_seconds+=omp_get_wtime()-dense_start;
        if (vjp.core) vjp.core();
        if (!any) return;
        if (!primal_outputs && !vjp.production)
            throw std::invalid_argument("Reverse-only density pass requires a sample VJP");
        const double production_start=omp_get_wtime();
        if (primal_outputs && production_values.empty()) production_values.resize(batches);
        // One parallel region for all production streams of this pair. No
        // per-stream matrices, sparse-update buffers, or nested chunk launches.
        #pragma omp parallel num_threads(opt.threads)
        {
            double direct_local=0.,exchange_local=0.;
            #pragma omp for schedule(static)
            for (size_t replica=0;replica<batches;++replica) {
                arma::mat unused;
                arma::mat& value=primal_outputs ? production_values[replica] : unused;
                if (primal_outputs) value.zeros(dimension,dimension);
                for (size_t stream=0;stream<streams;++stream) if (active(stream,key)) {
                    const size_t count=answer.residual_samples(stream,key);
                    if(!count) continue;
                    auto rng=generator(opt.seed,1,key,replica,stream);
                    const double stream_start=omp_get_wtime();
                    if (vjp.production)
                        vjp.production(*tables[stream],offsets,count,stream,rng,value,
                            factor/double(batches));
                    else Policy::accumulate(*tables[stream],left,right,offsets,count,
                                            stream,rng,value,1.);
                    const double elapsed=omp_get_wtime()-stream_start;
                    if (stream<Policy::direct_streams) direct_local+=elapsed;
                    else exchange_local+=elapsed;
                }
            }
            const double workers=double(omp_get_num_threads());
            #pragma omp atomic
            answer.direct_production_seconds+=direct_local/workers;
            #pragma omp atomic
            answer.exchange_production_seconds+=exchange_local/workers;
        }
        const double reduction_start=omp_get_wtime();
        answer.production_accumulation_seconds+=reduction_start-production_start;
        if (primal_outputs) {
            answer.production_accumulator_bytes=batches*dimension*dimension*sizeof(double);
            for (const auto& value : production_values) {
                if (!value.is_finite()) throw std::runtime_error("Nonfinite density sample");
            }
            // The trace uses the same independent replica matrices as the
            // entrywise SE. Taking each replica's trace first retains the
            // covariance between diagonal entries, without another draw.
            moments trace_samples;
            for (size_t replica=0;replica<batches;++replica) {
                const double trace=factor*arma::trace(production_values[replica]);
                answer.trace_replicas(key,replica)=trace;
                add_moment(trace_samples,trace);
            }
            answer.trace_variance+=sample_variance(trace_samples)/double(batches);
            // Symmetrize only while reducing. Entrywise variance must include the
            // covariance between transposed entries from each independent estimate.
            #pragma omp parallel for num_threads(opt.threads) schedule(static)
            for (size_t col=0;col<dimension;++col) for (size_t row=0;row<=col;++row) {
                moments entry;
                for (const auto& value : production_values)
                    add_moment(entry,.5*factor*(value(row,col)+value(col,row)));
                const double variance=sample_variance(entry)/double(batches);
                answer.matrix(row,col)+=entry.mean;
                answer.standard_error(row,col)+=variance;
                if (row!=col) {
                    answer.matrix(col,row)+=entry.mean;
                    answer.standard_error(col,row)+=variance;
                }
            }
        }
        const double production_end=omp_get_wtime();
        answer.production_reduction_seconds+=production_end-reduction_start;
        answer.production_seconds+=production_end-production_start;
        answer.sampling_seconds+=production_end-production_start;
        const size_t draws=batches*arma::accu(answer.residual_samples.col(key));
        for (size_t stream=0;stream<streams;++stream) {
            const size_t stream_draws=batches*answer.residual_samples(stream,key);
            if (stream<Policy::direct_streams) answer.direct_production_draws+=stream_draws;
            else answer.exchange_production_draws+=stream_draws;
        }
        answer.production_pair_draws(l,m)+=draws; answer.production_draws+=draws;
        answer.production_pair_batches(l,m)=batches; answer.batches+=batches;
        if (log) *log << label << "Density pair " << l << ',' << m << " samples = "
            << answer.production_pair_draws(l,m) << '\n';
    }
    void finish() {
        if (!primal_outputs) return;
        answer.standard_error=arma::sqrt(arma::clamp(answer.standard_error,0.,std::numeric_limits<double>::max()));
        answer.frobenius_se=arma::norm(answer.standard_error,"fro");
        answer.target_met=answer.frobenius_se<=opt.frobenius_se_target;

    }
};

template<class Policy>
inline result estimate_pairs(const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
        const arma::Col<size_t>& offsets, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, const options& opt, std::ostream* log) {
    const double start=omp_get_wtime();
    pair_estimator<Policy> state(fo.n_rows,fv.n_rows,beta.n_elem,offsets,group,opt);
    result& answer=state.answer;
    propagator dress(fo,fv,bare,beta,opt.threads);
    state.set_quotas(fixed_pair_quotas(fo,fv,beta,weights,opt.symmetric_pairs));
    answer.setup_seconds=omp_get_wtime()-start;
    auto prepare=[&](size_t point, point_data& data) {
        double tick=omp_get_wtime(); dress.apply(point,data.c);
        const double rotation=omp_get_wtime()-tick;
        answer.dressing_seconds+=rotation; ++answer.propagated_points;
        tick=omp_get_wtime();
        data.prepare(fo.n_rows,fv.n_rows,offsets,weights(point),opt.domain_tau,opt.threads,
                     opt.atom_block_l2,opt.w_virtual_reform);
        const double tables=omp_get_wtime()-tick;
        answer.point_table_seconds+=tables; answer.setup_seconds+=rotation+tables;
    };
    visit_pairs_once(beta.n_elem,opt.symmetric_pairs,prepare,
        [&](size_t l,size_t m,const point_data& a,const point_data& b,size_t bytes) {
            const double factor=(opt.symmetric_pairs && l!=m ? 2. : 1.)*weights(l)*weights(m);
            state.process(l,m,factor,a,b,bytes,log);
        },opt.resident_points);
    state.finish();
    answer.seconds=omp_get_wtime()-start;
    return std::move(answer);
}

template<class Policy>
inline result estimate_block(const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
        const arma::Col<size_t>& offsets, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, const options& opt, std::ostream* log=nullptr) {
    validate(fo,fv,bare,offsets,beta,weights,group,opt);
    return estimate_pairs<Policy>(fo,fv,bare,offsets,beta,weights,group,opt,log);
}

inline result estimate(const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
        const arma::Col<size_t>& offsets, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, const options& opt, std::ostream* log=nullptr) {
    return estimate_block<occupied_policy>(fo,fv,bare,offsets,beta,weights,group,opt,log);
}

}}}} // namespace libgmbpt::qchem::stc_mp2::density
// Virtual density
namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {

// Notes Eq. (29): retain two direct terms and merge the two exchange terms
// using occupied interchange and s_i+s_j. Four first-weak strata per term.
struct virtual_tuple { size_t i,j,a,b,c,A,T; };
struct virtual_update { size_t a,b; double value; };
inline std::array<std::pair<size_t,size_t>,4> virtual_legs(const virtual_tuple& x, size_t term) {
    if (term==0) return {{{x.i,x.a},{x.j,x.c},{x.i,x.b},{x.j,x.c}}};
    if (term==1) return {{{x.i,x.c},{x.j,x.a},{x.i,x.c},{x.j,x.b}}};
    if (term==2) return {{{x.i,x.a},{x.j,x.c},{x.i,x.c},{x.j,x.b}}};
    return {{{x.i,x.c},{x.j,x.a},{x.i,x.b},{x.j,x.c}}};
}

// Direct terms 0 and 1 retain i in S. Each direct term has one orbital-pair
// column shared by B and C, so its residual uses three first-weak slots.
// Occupied interchange combines exchange terms 2+3 into term 2 weighted by
// s_i+s_j, including both Laplace orderings; exchange retains four slots.
// Marginalize this weight into conditional alias tables (no mixture draw).
// In virtual_tuple, A names the left/B atom and T the right/C atom; this is
// opposite to the T(B), A(C) labels in the atom-paired envelope equations.
class virtual_proposal {
    const point_data &l_, &m_;
    const w_virtual_b_tables* b_cache_=nullptr;
    const w_virtual_c_tables* c_cache_=nullptr;
    size_t no_, nv_, na_;
    size_t term_;
    int stratum_;
    int exchange_variant_=0;
    int direct_variant_=-1;
    bool reform_;
    std::vector<unsigned char> selected_;
    alias_distribution root_;
    std::array<alias_distribution,2> center_;
    std::vector<alias_distribution> a_, b_, i_given_c_, partner_, c_given_root_;
    std::array<std::vector<alias_distribution>,2> j_given_c_;
    size_t b_ia_mask() const { return stratum_==0 ? 1 : 2; }
    size_t b_jc_mask() const { return stratum_==0 ? 0 : stratum_==1 ? 1 : 2; }
    size_t c_jb_mask() const { return stratum_==3 ? 1 : 0; }
    bool b_ia_h() const { return exchange_variant_!=2 && exchange_variant_!=3; }
    bool c_jb_h() const { return exchange_variant_!=1 && exchange_variant_!=3; }
    size_t b_a_kind() const { return b_ia_h() ? 0 : 1; }
    size_t c_b_kind() const { return c_jb_h() ? 2 : 0; }
    bool direct_b_x_h() const { return direct_variant_==2 || direct_variant_==3 || direct_variant_==5; }
    bool direct_c_pair() const { return direct_variant_>=4; }
    bool direct_c_z_h() const { return direct_variant_%2==1; }
    size_t direct_b_y_kind() const { return direct_b_x_h() ? 1 : 0; }
    size_t direct_c_z_kind() const { return direct_c_z_h() ? 2 : 0; }
    double raw_direct_q(const virtual_tuple& x) const {
        const size_t bx=term_==0 ? x.j*nv_+x.c : x.i*nv_+x.c;
        const size_t by=term_==0 ? x.i*nv_+x.a : x.j*nv_+x.a;
        const size_t cz=term_==0 ? x.i*nv_+x.b : x.j*nv_+x.b;
        const double b=direct_b_x_h() ? l_.atom_weight(x.A,bx)*l_.norm(by/nv_,by%nv_)
                                       : l_.norm(bx/nv_,bx%nv_)*l_.atom_weight(x.A,by);
        if(direct_c_pair()) return b*m_.atom_l1(x.T,bx)*m_.atom_l1(x.T,cz);
        return b*(direct_c_z_h() ? m_.norm(bx/nv_,bx%nv_)*m_.atom_weight(x.T,cz)
                                 : m_.atom_weight(x.T,bx)*m_.norm(cz/nv_,cz%nv_));
    }
    double raw_exchange_q(const virtual_tuple& x) const {
        const size_t ia=x.i*nv_+x.a,jc=x.j*nv_+x.c,
            ic=x.i*nv_+x.c,jb=x.j*nv_+x.b;
        if(exchange_variant_==4)
            return l_.atom_l1(x.A,ia)*l_.atom_l1(x.A,jc)
                *m_.norm(x.i,x.c)*m_.atom_weight(x.T,jb);
        if(exchange_variant_==5)
            return l_.atom_weight(x.A,ia)*l_.norm(x.j,x.c)
                *m_.atom_l1(x.T,ic)*m_.atom_l1(x.T,jb);
        return (b_ia_h() ? l_.atom_weight(x.A,ia)*l_.norm(x.j,x.c)
                         : l_.norm(x.i,x.a)*l_.atom_weight(x.A,jc))
            *(c_jb_h() ? m_.norm(x.i,x.c)*m_.atom_weight(x.T,jb)
                       : m_.atom_weight(x.T,ic)*m_.norm(x.j,x.b));
    }
    double gate(size_t leg, size_t i, size_t a) const {
        if (reform_ && term_<2) {
            if (stratum_<0) return 1.;
            if (stratum_>=3) return 0.;
            if (term_==0) {
                // x=(j,c): B_x C_x, y=(i,a): B_y, z=(i,b): C_z.
                if (leg==1) {
                    const bool strong=l_.strong(i,a) && m_.strong(i,a);
                    return stratum_==0 ? double(!strong) : double(strong);
                }
                if (leg==3) return 1.;
                if (leg==0) return stratum_==0 ? 1.
                    : stratum_==1 ? double(!l_.strong(i,a)) : double(l_.strong(i,a));
                return stratum_==2 ? double(!m_.strong(i,a)) : 1.;
            }
            // x=(i,c): B_x C_x, y=(j,a): B_y, z=(j,b): C_z.
            if (leg==0) {
                const bool strong=l_.strong(i,a) && m_.strong(i,a);
                return stratum_==0 ? double(!strong) : double(strong);
            }
            if (leg==2) return 1.;
            if (leg==1) return stratum_==0 ? 1.
                : stratum_==1 ? double(!l_.strong(i,a)) : double(l_.strong(i,a));
            return stratum_==2 ? double(!m_.strong(i,a)) : 1.;
        }
        if (stratum_<0 || int(leg)>stratum_) return 1.;
        const bool h=(leg<2 ? l_.strong : m_.strong)(i,a);
        return int(leg)==stratum_ ? double(!h) : double(h);
    }
    static void reset(alias_distribution& table, const arma::vec& w) {
        table.reset(w.begin(),w.end());
    }
    void build_reform(int threads) {
        if (term_<2 && stratum_>=3) return;
        const size_t roots=no_*na_;
        if (term_==0) {
            // i,T -> a and b; T -> (j,c); (j,c) -> C atom A.
            a_.resize(roots); b_.resize(no_); partner_.resize(na_);
            arma::vec b_total(no_), partner_total(na_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i=0;i<no_;++i) {
                arma::vec w(nv_);
                for (size_t b=0;b<nv_;++b) w(b)=m_.norm(i,b)*gate(2,i,b);
                b_total(i)=arma::accu(w); reset(b_[i],w);
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t T=0;T<na_;++T) {
                arma::vec w(no_*nv_);
                for (size_t jc=0;jc<no_*nv_;++jc) {
                    const size_t j=jc/nv_, c=jc%nv_;
                    w(jc)=l_.atom_l1(T,jc)*gate(1,j,c)*m_.sum(j,c)*gate(3,j,c);
                }
                partner_total(T)=arma::accu(w); reset(partner_[T],w);
            }
            arma::vec root(roots,arma::fill::zeros);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t iT=0;iT<roots;++iT) {
                const size_t i=iT/na_, T=iT%na_;
                if (!selected_[i]) continue;
                arma::vec w(nv_);
                for (size_t a=0;a<nv_;++a)
                    w(a)=l_.atom_l1(T,i*nv_+a)*gate(0,i,a);
                root(iT)=arma::accu(w)*b_total(i)*partner_total(T);
                reset(a_[iT],w);
            }
            normalization=arma::accu(root); reset(root_,root);
        } else if (term_==1) {
            // i,T -> c; T -> j; (T,j) -> a; j -> b.
            c_given_root_.resize(roots); partner_.resize(na_);
            a_.resize(na_*no_); b_.resize(no_);
            arma::vec b_total(no_),partner_total(na_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t j=0;j<no_;++j) {
                arma::vec w(nv_);
                for (size_t b=0;b<nv_;++b) w(b)=m_.norm(j,b)*gate(3,j,b);
                b_total(j)=arma::accu(w); reset(b_[j],w);
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t T=0;T<na_;++T) {
                arma::vec jw(no_);
                for (size_t j=0;j<no_;++j) {
                    arma::vec aw(nv_);
                    for (size_t a=0;a<nv_;++a)
                        aw(a)=l_.atom_l1(T,j*nv_+a)*gate(1,j,a);
                    jw(j)=arma::accu(aw)*b_total(j);
                    reset(a_[T*no_+j],aw);
                }
                partner_total(T)=arma::accu(jw); reset(partner_[T],jw);
            }
            arma::vec root(roots,arma::fill::zeros);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t iT=0;iT<roots;++iT) {
                const size_t i=iT/na_,T=iT%na_;
                if (!selected_[i]) continue;
                arma::vec cw(nv_);
                for (size_t c=0;c<nv_;++c)
                    cw(c)=l_.atom_l1(T,i*nv_+c)*gate(0,i,c)
                         *m_.sum(i,c)*gate(2,i,c);
                root(iT)=arma::accu(cw)*partner_total(T);
                reset(c_given_root_[iT],cw);
            }
            normalization=arma::accu(root); reset(root_,root);
        } else {
            // c -> i,j with s_i+s_j; i -> a,T; j -> b,A.
            a_.resize(no_); b_.resize(no_); i_given_c_.resize(nv_);
            for (auto& tables:j_given_c_) tables.resize(nv_);
            arma::vec a_total(no_),b_total(no_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i=0;i<no_;++i) {
                arma::vec aw(nv_),bw(nv_);
                for (size_t a=0;a<nv_;++a) {
                    aw(a)=l_.sum(i,a)*gate(0,i,a);
                    bw(a)=m_.sum(i,a)*gate(3,i,a);
                }
                a_total(i)=arma::accu(aw); b_total(i)=arma::accu(bw);
                reset(a_[i],aw); reset(b_[i],bw);
            }
            arma::vec cw(nv_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t c=0;c<nv_;++c) {
                arma::vec left(no_),right(no_),iw(no_),jw(no_);
                double lt=0.,ls=0.,rt=0.,rs=0.;
                for (size_t i=0;i<no_;++i) {
                    left(i)=a_total(i)*m_.norm(i,c)*gate(2,i,c);
                    right(i)=b_total(i)*l_.norm(i,c)*gate(1,i,c);
                    lt+=left(i); ls+=selected_[i]*left(i);
                    rt+=right(i); rs+=selected_[i]*right(i);
                }
                for (size_t i=0;i<no_;++i) {
                    iw(i)=left(i)*(selected_[i]*rt+rs);
                    jw(i)=selected_[i]*right(i);
                }
                cw(c)=lt*rs+ls*rt;
                reset(i_given_c_[c],iw); reset(j_given_c_[0][c],jw);
                for (size_t j=0;j<no_;++j) jw(j)=(1+selected_[j])*right(j);
                reset(j_given_c_[1][c],jw);
            }
            normalization=arma::accu(cw); reset(root_,cw);
        }
    }
    void build_split_direct(int threads) {
        if(stratum_>=3) return;
        const size_t mask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
        const size_t c_mask=stratum_==2 ? 1 : 0;
        arma::vec root(no_*na_,arma::fill::zeros);
        partner_.resize(na_);
        if(term_==0) {
            arma::vec partner_total(na_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(no_*nv_);
                #pragma omp for schedule(static)
                for(size_t A=0;A<na_;++A) {
                    for(size_t jc=0;jc<no_*nv_;++jc) {
                        const size_t j=jc/nv_,c=jc%nv_;
                        w(jc)=l_.atom_l1(A,jc)*gate(1,j,c)
                            *m_.sum(j,c)*gate(3,j,c);
                    }
                    partner_total(A)=arma::accu(w);
                    reset(partner_[A],w);
                }
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t iA=0;iA<no_*na_;++iA) {
                const size_t i=iA/na_,A=iA%na_;
                if(selected_[i])
                    root(iA)=b_cache_->atom_total(iA,mask)
                        *c_cache_->b_total(i,c_mask)*partner_total(A);
            }
        } else {
            c_given_root_.resize(no_*na_);
            arma::vec partner_total(na_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(no_);
                #pragma omp for schedule(static)
                for(size_t A=0;A<na_;++A) {
                    for(size_t j=0;j<no_;++j)
                        w(j)=b_cache_->atom_total(j*na_+A,mask)
                            *c_cache_->b_total(j,c_mask);
                    partner_total(A)=arma::accu(w);
                    reset(partner_[A],w);
                }
            }
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(nv_);
                #pragma omp for schedule(static)
                for(size_t iA=0;iA<no_*na_;++iA) {
                    const size_t i=iA/na_,A=iA%na_;
                    if(!selected_[i]) continue;
                    for(size_t c=0;c<nv_;++c)
                        w(c)=l_.atom_l1(A,i*nv_+c)*gate(0,i,c)
                            *m_.sum(i,c)*gate(2,i,c);
                    root(iA)=arma::accu(w)*partner_total(A);
                    reset(c_given_root_[iA],w);
                }
            }
        }
        normalization=arma::accu(root); reset(root_,root);
    }
    void build_split_direct_hn(int threads) {
        if(stratum_>=3) return;
        const size_t bmask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
        const size_t cmask=stratum_==2 ? 1 : 0;
        const size_t by_kind=direct_b_y_kind()*3+bmask;
        const size_t cz_kind=direct_c_z_kind()+cmask;
        partner_.resize(1);
        if(term_==0) {
            arma::vec root(no_,arma::fill::zeros),partner(no_*nv_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t i=0;i<no_;++i)
                if(selected_[i]) root(i)=b_cache_->exchange_total(i,by_kind)
                    *c_cache_->b_total(i,cz_kind);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t jc=0;jc<no_*nv_;++jc) {
                const size_t j=jc/nv_,c=jc%nv_;
                partner(jc)=(direct_b_x_h() ? l_.sum(j,c) : l_.norm(j,c))
                    *(direct_c_z_h() ? m_.norm(j,c) : m_.sum(j,c))
                    *gate(1,j,c)*gate(3,j,c);
            }
            normalization=arma::accu(root)*arma::accu(partner);
            reset(root_,root);reset(partner_[0],partner);
        } else {
            arma::vec root(no_*nv_,arma::fill::zeros),partner(no_);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t ic=0;ic<no_*nv_;++ic) {
                const size_t i=ic/nv_,c=ic%nv_;
                if(selected_[i]) root(ic)=(direct_b_x_h() ? l_.sum(i,c) : l_.norm(i,c))
                    *(direct_c_z_h() ? m_.norm(i,c) : m_.sum(i,c))
                    *gate(0,i,c)*gate(2,i,c);
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t j=0;j<no_;++j)
                partner(j)=b_cache_->exchange_total(j,by_kind)
                    *c_cache_->b_total(j,cz_kind);
            normalization=arma::accu(root)*arma::accu(partner);
            reset(root_,root);reset(partner_[0],partner);
        }
    }
    void build_split_direct_c_pair(int threads) {
        if(stratum_>=3) return;
        const size_t bmask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
        const size_t cmask=stratum_==2 ? 1 : 0;
        const size_t by_kind=direct_b_y_kind()*3+bmask;
        arma::vec root(no_*na_,arma::fill::zeros);
        partner_.resize(na_);
        if(term_==0) {
            arma::vec partner_total(na_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(no_*nv_);
                #pragma omp for schedule(static)
                for(size_t T=0;T<na_;++T) {
                    for(size_t jc=0;jc<no_*nv_;++jc) {
                        const size_t j=jc/nv_,c=jc%nv_;
                        w(jc)=(direct_b_x_h() ? l_.sum(j,c) : l_.norm(j,c))
                            *m_.atom_l1(T,jc)*gate(1,j,c)*gate(3,j,c);
                    }
                    partner_total(T)=arma::accu(w);
                    reset(partner_[T],w);
                }
            }
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(size_t iT=0;iT<no_*na_;++iT) {
                const size_t i=iT/na_,T=iT%na_;
                if(selected_[i]) root(iT)=b_cache_->exchange_total(i,by_kind)
                    *c_cache_->atom_b_total(iT,cmask)*partner_total(T);
            }
        } else {
            c_given_root_.resize(no_*na_);
            arma::vec partner_total(na_,arma::fill::zeros);
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(no_);
                #pragma omp for schedule(static)
                for(size_t T=0;T<na_;++T) {
                    for(size_t j=0;j<no_;++j)
                        w(j)=b_cache_->exchange_total(j,by_kind)
                            *c_cache_->atom_b_total(j*na_+T,cmask);
                    partner_total(T)=arma::accu(w);
                    reset(partner_[T],w);
                }
            }
            #pragma omp parallel num_threads(threads)
            {
                arma::vec w(nv_);
                #pragma omp for schedule(static)
                for(size_t iT=0;iT<no_*na_;++iT) {
                    const size_t i=iT/na_,T=iT%na_;
                    if(!selected_[i]) continue;
                    for(size_t c=0;c<nv_;++c)
                        w(c)=(direct_b_x_h() ? l_.sum(i,c) : l_.norm(i,c))
                            *m_.atom_l1(T,i*nv_+c)*gate(0,i,c)*gate(2,i,c);
                    root(iT)=arma::accu(w)*partner_total(T);
                    reset(c_given_root_[iT],w);
                }
            }
        }
        normalization=arma::accu(root);reset(root_,root);
    }
    void build_split_exchange(int threads) {
        const bool b_pair=exchange_variant_==4,c_pair=exchange_variant_==5;
        const size_t roots=(b_pair || c_pair ? na_*nv_ : nv_);
        i_given_c_.resize(roots);
        for(auto& v:j_given_c_) v.resize(roots);
        arma::vec root_weight(roots,arma::fill::zeros);
        const size_t bmask=b_ia_mask(),jmask=b_jc_mask(),cmask=c_jb_mask();
        const size_t akind=b_a_kind()*3+bmask;
        const size_t bkind=c_b_kind()+cmask;
        #pragma omp parallel num_threads(threads)
        {
            arma::vec left(no_),right(no_),iw(no_),jw(no_);
            #pragma omp for schedule(static)
            for(size_t root=0;root<roots;++root) {
                const size_t atom=(b_pair || c_pair ? root/nv_ : 0),c=root%nv_;
                double lt=0.,ls=0.,rt=0.,rs=0.;
                for(size_t i=0;i<no_;++i) {
                    if(b_pair) {
                        left(i)=b_cache_->atom_total(i*na_+atom,bmask)
                            *m_.norm(i,c)*gate(2,i,c);
                        right(i)=l_.atom_l1(atom,i*nv_+c)*gate(1,i,c)
                            *c_cache_->b_total(i,2+cmask);
                    } else if(c_pair) {
                        left(i)=b_cache_->exchange_total(i,akind)
                            *m_.atom_l1(atom,i*nv_+c)*gate(2,i,c);
                        right(i)=l_.norm(i,c)*gate(1,i,c)
                            *c_cache_->atom_b_total(i*na_+atom,cmask);
                    } else {
                        left(i)=b_cache_->exchange_total(i,akind)
                            *(c_jb_h() ? m_.norm(i,c) : m_.sum(i,c))*gate(2,i,c);
                        right(i)=(b_ia_h() ? l_.norm(i,c) : l_.sum(i,c))*gate(1,i,c)
                            *c_cache_->b_total(i,bkind);
                    }
                    lt+=left(i);ls+=selected_[i]*left(i);
                    rt+=right(i);rs+=selected_[i]*right(i);
                }
                for(size_t i=0;i<no_;++i) {
                    iw(i)=left(i)*(selected_[i]*rt+rs);
                    jw(i)=selected_[i]*right(i);
                }
                root_weight(root)=lt*rs+ls*rt;
                reset(i_given_c_[root],iw);
                reset(j_given_c_[0][root],jw);
                for(size_t j=0;j<no_;++j) jw(j)=(1+selected_[j])*right(j);
                reset(j_given_c_[1][root],jw);
            }
        }
        normalization=arma::accu(root_weight);reset(root_,root_weight);
    }
public:
    double normalization=0.;
    virtual_proposal(const point_data& l, const point_data& m,
            const std::vector<size_t>& group, size_t term, int stratum,
            bool reform=false,int threads=1,
            const w_virtual_b_tables* b_cache=nullptr,
            const w_virtual_c_tables* c_cache=nullptr,int exchange_variant=0,
            int direct_variant=-1):
            l_(l),m_(m),b_cache_(b_cache),c_cache_(c_cache),
            no_(l.norm.n_rows),nv_(l.norm.n_cols),na_(l.atom_weight.n_rows),
            term_(term),stratum_(stratum),exchange_variant_(exchange_variant),
            direct_variant_(direct_variant),
            reform_(reform),selected_(no_,0) {
        if (term_>2) throw std::invalid_argument("Invalid virtual density term");
        if(exchange_variant_<0 || exchange_variant_>5)
            throw std::invalid_argument("Invalid W virtual exchange envelope");
        if(direct_variant_<-1 || direct_variant_>5)
            throw std::invalid_argument("Invalid W virtual direct envelope");
        for (size_t i : group) selected_[i]=1;
        if(reform_ && b_cache_ && c_cache_) {
            if(term_<2) {
                if(direct_variant_<0) build_split_direct(threads);
                else if(direct_c_pair()) build_split_direct_c_pair(threads);
                else build_split_direct_hn(threads);
            }
            else build_split_exchange(threads);
            return;
        }
        if (reform_) { build_reform(threads); return; }
        // Norms on canonical legs 0,2; atom envelopes on legs 1,3.
        std::array<arma::mat,4> f;
        for (size_t leg=0; leg<4; ++leg) {
            f[leg]=(leg==0 ? l.norm : leg==1 ? l.sum : leg==2 ? m.norm : m.sum);
            for (size_t i=0;i<no_;++i) for (size_t a=0;a<nv_;++a)
                f[leg](i,a)*=gate(leg,i,a);
        }
        const size_t aleg=term_==1 ? 1 : 0, bleg=term_==0 ? 2 : 3;
        arma::vec as=arma::sum(f[aleg],1), bs=arma::sum(f[bleg],1);
        a_.resize(no_); b_.resize(no_);
        for (size_t i=0;i<no_;++i) {
            reset(a_[i],f[aleg].row(i).t()); reset(b_[i],f[bleg].row(i).t());
        }
        if (term_<2) {
            arma::vec iw(no_), center(no_*nv_);
            for (size_t h=0;h<no_;++h) {
                iw(h)=as(h)*bs(h)*(term_==0 ? selected_[h] : 1);
                for (size_t c=0;c<nv_;++c)
                    center(h*nv_+c)=term_==0 ? f[1](h,c)*f[3](h,c)
                        : selected_[h]*f[0](h,c)*f[2](h,c);
            }
            normalization=arma::accu(iw)*arma::accu(center);
            reset(root_,iw); reset(center_[0],center);
        } else {
            i_given_c_.resize(nv_);
            for (auto& tables : j_given_c_) tables.resize(nv_);
            arma::vec cw(nv_), iw(no_), jw(no_);
            for (size_t c=0;c<nv_;++c) {
                double total=0.,selected_total=0.;
                for (size_t j=0;j<no_;++j) {
                    const double value=f[1](j,c)*bs(j);
                    total+=value; selected_total+=selected_[j]*value;
                    jw(j)=selected_[j]*value;
                }
                reset(j_given_c_[0][c],jw);
                for (size_t j=0;j<no_;++j)
                    jw(j)=(1+selected_[j])*f[1](j,c)*bs(j);
                reset(j_given_c_[1][c],jw);
                for (size_t i=0;i<no_;++i)
                    iw(i)=f[2](i,c)*as(i)*(selected_[i]*total+selected_total);
                cw(c)=arma::accu(iw); reset(i_given_c_[c],iw);
            }
            normalization=arma::accu(cw); reset(root_,cw);
        }
    }
    size_t selection_weight(const virtual_tuple& x) const {
        return term_<2 ? selected_[x.i] : selected_[x.i]+selected_[x.j];
    }
    size_t term() const { return term_; }
    bool contains(const virtual_tuple& x) const {
        if (!selection_weight(x)) return false;
        const auto pairs=virtual_legs(x,term());
        for (size_t leg=0;leg<4;++leg)
            if (!gate(leg,pairs[leg].first,pairs[leg].second)) return false;
        return true;
    }
    double probability(const virtual_tuple& x) const {
        if (!(normalization>0) || !contains(x)) return 0.;
        if (reform_) {
            const size_t ia=x.i*nv_+x.a,ic=x.i*nv_+x.c,
                jc=x.j*nv_+x.c,ja=x.j*nv_+x.a,
                ib=x.i*nv_+x.b,jb=x.j*nv_+x.b;
            double q=0.;
            if (term_<2 && b_cache_ && direct_variant_>=0)
                q=raw_direct_q(x);
            else if (term_==0)
                q=l_.atom_l1(x.A,ia)*l_.atom_l1(x.A,jc)
                    *m_.atom_weight(x.T,jc)*m_.norm(x.i,x.b);
            else if (term_==1)
                q=l_.atom_l1(x.A,ic)*l_.atom_l1(x.A,ja)
                    *m_.atom_weight(x.T,ic)*m_.norm(x.j,x.b);
            else
                q=b_cache_ ? raw_exchange_q(x) :
                    l_.atom_weight(x.A,ia)*m_.norm(x.i,x.c)
                    *l_.norm(x.j,x.c)*m_.atom_weight(x.T,jb);
            return selection_weight(x)*q/normalization;
        }
        const auto p=virtual_legs(x,term());
        return selection_weight(x)*l_.norm(p[0].first,p[0].second)*m_.norm(p[2].first,p[2].second)
            *l_.atom_weight(x.A,p[1].first*nv_+p[1].second)
            *m_.atom_weight(x.T,p[3].first*nv_+p[3].second)/normalization;
    }
    virtual_tuple draw(std::mt19937& rng) const {
        if (reform_) {
            virtual_tuple x;
            if(b_cache_ && c_cache_) {
                if(term_<2 && direct_c_pair()) {
                    const size_t iT=root_(rng),bmask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
                    x.i=iT/na_;x.T=iT%na_;
                    if(term_==0) {
                        const size_t jc=partner_[x.T](rng);
                        x.j=jc/nv_;x.c=jc%nv_;
                    } else {
                        x.c=c_given_root_[iT](rng);
                        x.j=partner_[x.T](rng);
                    }
                    x.a=b_cache_->exchange_a[direct_b_y_kind()*3+bmask]
                        [term_==0 ? x.i : x.j](rng);
                    x.b=c_cache_->atom_b[stratum_==2 ? 1 : 0]
                        [(term_==0 ? x.i : x.j)*na_+x.T](rng);
                    x.A=l_.atoms[direct_b_x_h() ? (term_==0 ? x.j : x.i)*nv_+x.c
                        : (term_==0 ? x.i : x.j)*nv_+x.a](rng);
                } else if(term_<2 && direct_variant_>=0) {
                    const size_t root=root_(rng);
                    const size_t bmask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
                    const size_t cmask=stratum_==2 ? 1 : 0;
                    if(term_==0) {
                        x.i=root;
                        const size_t jc=partner_[0](rng);
                        x.j=jc/nv_;x.c=jc%nv_;
                    } else {
                        x.i=root/nv_;x.c=root%nv_;
                        x.j=partner_[0](rng);
                    }
                    x.a=b_cache_->exchange_a[direct_b_y_kind()*3+bmask]
                        [term_==0 ? x.i : x.j](rng);
                    x.b=c_cache_->b[direct_c_z_kind()+cmask]
                        [term_==0 ? x.i : x.j](rng);
                    x.A=l_.atoms[direct_b_x_h() ? (term_==0 ? x.j : x.i)*nv_+x.c
                        : (term_==0 ? x.i : x.j)*nv_+x.a](rng);
                    x.T=m_.atoms[direct_c_z_h() ? (term_==0 ? x.i : x.j)*nv_+x.b
                        : (term_==0 ? x.j : x.i)*nv_+x.c](rng);
                } else if(term_==0) {
                    const size_t iA=root_(rng),mask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
                    x.i=iA/na_;x.A=iA%na_;
                    x.a=b_cache_->atom_a[mask][iA](rng);
                    x.b=c_cache_->b[stratum_==2 ? 1 : 0][x.i](rng);
                    const size_t jc=partner_[x.A](rng);
                    x.j=jc/nv_;x.c=jc%nv_;x.T=m_.atoms[jc](rng);
                } else if(term_==1) {
                    const size_t iA=root_(rng),mask=stratum_==0 ? 0 : stratum_==1 ? 1 : 2;
                    x.i=iA/na_;x.A=iA%na_;
                    x.c=c_given_root_[iA](rng);x.j=partner_[x.A](rng);
                    x.a=b_cache_->atom_a[mask][x.j*na_+x.A](rng);
                    x.b=c_cache_->b[stratum_==2 ? 1 : 0][x.j](rng);
                    x.T=m_.atoms[x.i*nv_+x.c](rng);
                } else {
                    const size_t root=root_(rng);
                    x.c=root%nv_;
                    x.i=i_given_c_[root](rng);
                    x.j=j_given_c_[selected_[x.i]][root](rng);
                    if(exchange_variant_==4) {
                        x.A=root/nv_;
                        x.a=b_cache_->atom_a[b_ia_mask()][x.i*na_+x.A](rng);
                        x.b=c_cache_->b[2+c_jb_mask()][x.j](rng);
                        x.T=m_.atoms[x.j*nv_+x.b](rng);
                    } else if(exchange_variant_==5) {
                        x.T=root/nv_;
                        x.a=b_cache_->exchange_a[b_ia_mask()][x.i](rng);
                        x.A=l_.atoms[x.i*nv_+x.a](rng);
                        x.b=c_cache_->atom_b[c_jb_mask()][x.j*na_+x.T](rng);
                    } else {
                        x.a=b_cache_->exchange_a[b_a_kind()*3+b_ia_mask()][x.i](rng);
                        x.b=c_cache_->b[c_b_kind()+c_jb_mask()][x.j](rng);
                        x.A=l_.atoms[(b_ia_h() ? x.i*nv_+x.a : x.j*nv_+x.c)](rng);
                        x.T=m_.atoms[(c_jb_h() ? x.j*nv_+x.b : x.i*nv_+x.c)](rng);
                    }
                }
                return x;
            }
            if (term_==0) {
                const size_t iT=root_(rng);
                x.i=iT/na_; x.A=iT%na_;
                x.a=a_[iT](rng); x.b=b_[x.i](rng);
                const size_t jc=partner_[x.A](rng);
                x.j=jc/nv_; x.c=jc%nv_;
                x.T=m_.atoms[jc](rng);
            } else if (term_==1) {
                const size_t iT=root_(rng);
                x.i=iT/na_; x.A=iT%na_;
                x.c=c_given_root_[iT](rng); x.j=partner_[x.A](rng);
                x.a=a_[x.A*no_+x.j](rng); x.b=b_[x.j](rng);
                x.T=m_.atoms[x.i*nv_+x.c](rng);
            } else {
                x.c=root_(rng); x.i=i_given_c_[x.c](rng);
                x.j=j_given_c_[selected_[x.i]][x.c](rng);
                x.a=a_[x.i](rng); x.b=b_[x.j](rng);
                x.A=l_.atoms[x.i*nv_+x.a](rng);
                x.T=m_.atoms[x.j*nv_+x.b](rng);
            }
            return x;
        }
        virtual_tuple x;
        if (term_<2) {
            const size_t h=root_(rng), center=center_[0](rng);
            x.i=term_==0 ? h : center/nv_;
            x.j=term_==0 ? center/nv_ : h; x.c=center%nv_;
        } else {
            x.c=root_(rng); x.i=i_given_c_[x.c](rng);
            x.j=j_given_c_[selected_[x.i]][x.c](rng);
        }
        x.a=a_[term_==1 ? x.j : x.i](rng);
        x.b=b_[term_==0 ? x.i : x.j](rng);
        const auto p=virtual_legs(x,term());
        x.A=l_.atoms[p[1].first*nv_+p[1].second](rng);
        x.T=m_.atoms[p[3].first*nv_+p[3].second](rng);
        return x;
    }
    // Same aliases and RNG order as draw(), specialized once per stream.
    // The tables already impose the selection/domain gates. On their support,
    // s*f / (s*q/Z) = Z*f/q, also when the merged exchange has s=2.
    template<size_t Term>
    virtual_update draw_weighted(std::mt19937& rng, const arma::Col<size_t>& offsets,
            virtual_tuple* drawn=nullptr,double* left_dot_out=nullptr,
            double* right_dot_out=nullptr,double* raw_q_out=nullptr) const {
        static_assert(Term<3,"Invalid virtual density term");
        if (reform_) {
            const virtual_tuple x=draw(rng);
            const size_t ia=x.i*nv_+x.a,ic=x.i*nv_+x.c,
                jc=x.j*nv_+x.c,ja=x.j*nv_+x.a,
                ib=x.i*nv_+x.b,jb=x.j*nv_+x.b;
            const size_t p0=Term==1 ? ic : ia;
            const size_t p1=Term==1 ? ja : jc;
            const size_t p2=Term==0 ? ib : ic;
            const size_t p3=Term==0 ? jc : jb;
            const double q=Term<2 && b_cache_ && direct_variant_>=0 ? raw_direct_q(x) : Term==0 ?
                l_.atom_l1(x.A,ia)*l_.atom_l1(x.A,jc)
                    *m_.atom_weight(x.T,jc)*m_.norm(x.i,x.b) :
                Term==1 ? l_.atom_l1(x.A,ic)*l_.atom_l1(x.A,ja)
                    *m_.atom_weight(x.T,ic)*m_.norm(x.j,x.b) :
                b_cache_ ? raw_exchange_q(x) :
                    l_.atom_weight(x.A,ia)*m_.norm(x.i,x.c)
                    *l_.norm(x.j,x.c)*m_.atom_weight(x.T,jb);
            const double left_dot=block_dot(l_.c,offsets,x.A,p0,p1);
            const double right_dot=block_dot(m_.c,offsets,x.T,p2,p3);
            if (drawn) *drawn=x;
            if (left_dot_out) *left_dot_out=left_dot;
            if (right_dot_out) *right_dot_out=right_dot;
            if (raw_q_out) *raw_q_out=q;
            return {x.a,x.b,normalization*(Term<2 ? 1. : -.5)*left_dot*right_dot/q};
        }
        size_t i,j,c;
        if (Term<2) {
            const size_t h=root_(rng), center=center_[0](rng);
            i=Term==0 ? h : center/nv_;
            j=Term==0 ? center/nv_ : h; c=center%nv_;
        } else {
            c=root_(rng); i=i_given_c_[c](rng);
            j=j_given_c_[selected_[i]][c](rng);
        }
        const size_t a=a_[Term==1 ? j : i](rng);
        const size_t b=b_[Term==0 ? i : j](rng);
        const size_t p0=i*nv_+(Term==1 ? c : a);
        const size_t p1=j*nv_+(Term==1 ? a : c);
        const size_t p2=i*nv_+(Term==0 ? b : c);
        const size_t p3=Term==0 ? p1 : j*nv_+b;
        const size_t A=l_.atoms[p1](rng), T=m_.atoms[p3](rng);
        if (drawn) *drawn={i,j,a,b,c,A,T};
        const double q=l_.norm(i,Term==1 ? c : a)*m_.norm(i,Term==0 ? b : c)
            *l_.atom_weight(A,p1)*m_.atom_weight(T,p3);
        const double left_dot=block_dot(l_.c,offsets,A,p0,p1);
        const double right_dot=block_dot(m_.c,offsets,T,p2,p3);
        if (left_dot_out) *left_dot_out=left_dot;
        if (right_dot_out) *right_dot_out=right_dot;
        if (raw_q_out) *raw_q_out=q;
        const double f=(Term<2 ? 1. : -.5)*left_dot*right_dot;
        return {a,b,normalization*f/q};
    }
    size_t table_bytes() const {
        size_t bytes=root_.storage_bytes()+selected_.size()*sizeof(unsigned char);
        for (const auto& a : center_) bytes+=a.storage_bytes();
        for (const auto* v : {&a_,&b_,&i_given_c_,&j_given_c_[0],&j_given_c_[1],
                              &partner_,&c_given_root_})
            for (const auto& a : *v) bytes+=a.storage_bytes();
        return bytes;
    }
};

inline double virtual_numerator(const point_data& l, const point_data& m,
        const arma::Col<size_t>& offsets, const virtual_tuple& x, size_t term) {
    const auto p=virtual_legs(x,term); const size_t nv=l.norm.n_cols;
    return (term<2 ? 1. : -.5)
        *block_dot(l.c,offsets,x.A,p[0].first*nv+p[0].second,p[1].first*nv+p[1].second)
        *block_dot(m.c,offsets,x.T,p[2].first*nv+p[2].second,p[3].first*nv+p[3].second);
}

// Domain indices are sorted. Contract their intersection without embedding
// either integral block in a zero-filled nvirt x nvirt matrix.
inline void virtual_core_product(arma::mat& out, const arma::mat& u,
        const arma::uvec& ur, const arma::uvec& uc, bool ut,
        const arma::mat& v, const arma::uvec& vr, const arma::uvec& vc, bool vt,
        double scale, bool add_transpose=false) {
    const arma::uvec &a=ut ? ur : uc, &b=vt ? vc : vr;
    std::vector<arma::uword> ai,bi;
    size_t i=0,j=0;
    while (i<a.n_elem && j<b.n_elem) {
        if (a(i)<b(j)) ++i;
        else if (b(j)<a(i)) ++j;
        else { ai.push_back(i++); bi.push_back(j++); }
    }
    if (ai.empty()) return;
    const arma::uvec ia(ai),ib(bi);
    const arma::uvec &rows=ut ? uc : ur, &cols=vt ? vr : vc;
    arma::mat product;
    if (ut) {
        if (vt) product=u.rows(ia).t()*v.cols(ib).t();
        else product=u.rows(ia).t()*v.rows(ib);
    } else {
        if (vt) product=u.cols(ia)*v.cols(ib).t();
        else product=u.cols(ia)*v.rows(ib);
    }
    out.submat(rows,cols)+=scale*product;
    if (add_transpose) out.submat(cols,rows)+=scale*product.t();
}
inline arma::mat virtual_dense_pair(const point_data& l, const point_data& m,
        const std::vector<size_t>& group, int threads) {
    const size_t no=l.norm.n_rows, nv=l.norm.n_cols;
    const bool diagonal=&l==&m;
    std::vector<arma::uvec> ld(no),md(diagonal ? 0 : no);
    for (size_t i=0;i<no;++i) {
        ld[i]=arma::find(l.strong.row(i).t());
        if (!diagonal) md[i]=arma::find(m.strong.row(i).t());
    }
    const auto& right=diagonal ? ld : md;
    std::vector<arma::mat> partial(threads,arma::mat(nv,nv,arma::fill::zeros));
    for (size_t i : group) {
        const auto &li=ld[i], &mi=right[i];
        if (!li.n_elem || !mi.n_elem) continue;
        const auto left_cache=l.cached_integrals(i,ld,threads);
        const auto right_cache=diagonal ? left_cache : m.cached_integrals(i,right,threads);
        // If the integral cache is full, still pack selected columns once and
        // share them across partner-j workers. Only one selected i is live.
        arma::mat packed_left,packed_other;
        if (!left_cache) packed_left=l.c.cols(i*nv+li);
        if (!diagonal && !right_cache) packed_other=m.c.cols(i*nv+mi);
        const auto& packed_right=diagonal ? packed_left : packed_other;
        #pragma omp parallel for num_threads(threads) schedule(dynamic)
        for (size_t j=0;j<no;++j) {
            const auto &lj=ld[j], &mj=right[j];
            if (!lj.n_elem || !mj.n_elem) continue;
            arma::mat scratch_u,scratch_v;
            if (!left_cache) scratch_u=packed_left.t()*l.c.cols(j*nv+lj);
            const auto& u=left_cache ? compact_block(*left_cache,j,ld,scratch_u) : scratch_u;
            if (!diagonal && !right_cache) scratch_v=packed_right.t()*m.c.cols(j*nv+mj);
            const auto& v=diagonal ? u : right_cache ? compact_block(*right_cache,j,right,scratch_v) : scratch_v;
            auto& out=partial[omp_get_thread_num()];
            virtual_core_product(out,u,li,lj,false,v,mi,mj,true,1.);
            virtual_core_product(out,u,li,lj,true,v,mi,mj,false,1.);
            // At l=m the second exchange product is the transpose of the first.
            virtual_core_product(out,u,li,lj,false,v,mi,mj,false,-.5,diagonal);
            if (!diagonal) virtual_core_product(out,u,li,lj,true,v,mi,mj,true,-.5);
        }
    }
    arma::mat answer(nv,nv,arma::fill::zeros);
    for (const auto& x : partial) answer+=x;
    return answer;
}

struct virtual_policy {
    using table_type=virtual_proposal;
    static constexpr size_t streams=12;
    static constexpr size_t direct_streams=8;
    static size_t dimension(size_t, size_t nv) { return nv; }
    static table_type* table(const point_data& l, const point_data& m,
            const std::vector<size_t>& group, size_t stream, bool, bool, bool, bool reform,
            int threads,const w_occupied_b_tables*,const w_occupied_c_tables*,
            const w_virtual_b_tables* b_cache,const w_virtual_c_tables* c_cache,
            int variant,int direct_variant) {
        return new virtual_proposal(l,m,group,stream/4,int(stream%4),reform,threads,
                                    b_cache,c_cache,variant,direct_variant);
    }
    static arma::mat core(const point_data& l, const point_data& m,
            const std::vector<size_t>& group, int threads,int=0) {
        return virtual_dense_pair(l,m,group,threads);
    }
    static void accumulate(const table_type& table, const point_data& l, const point_data& m,
            const arma::Col<size_t>& offsets, size_t count, size_t stream,
            std::mt19937& rng, arma::mat& value,double weight=1.) {
        switch (table.term()) {
            case 0: accumulate_term<0>(table,offsets,count,rng,value,weight); break;
            case 1: accumulate_term<1>(table,offsets,count,rng,value,weight); break;
            case 2: accumulate_term<2>(table,offsets,count,rng,value,weight); break;
        }
    }
    template<size_t Term>
    static void accumulate_term(const table_type& table, const arma::Col<size_t>& offsets,
            size_t count, std::mt19937& rng, arma::mat& value,double weight=1.) {
        if(!count) return;
        const double scale=weight/double(count);
        for (size_t n=0;n<count;++n) {
            const auto x=table.draw_weighted<Term>(rng,offsets);
            value(x.a,x.b)+=scale*x.value;
        }
    }
    static residual_result sample(const table_type& table, const point_data& l, const point_data& m,
            const arma::Col<size_t>& offsets, size_t count, const options& opt,
            size_t stage, size_t point, size_t batch, size_t stream) {
        switch (table.term()) {
            case 0: return sample_term<0>(table,l,offsets,count,opt,stage,point,batch,stream);
            case 1: return sample_term<1>(table,l,offsets,count,opt,stage,point,batch,stream);
            default: return sample_term<2>(table,l,offsets,count,opt,stage,point,batch,stream);
        }
    }
    template<size_t Term>
    static residual_result sample_term(const table_type& table, const point_data& l,
            const arma::Col<size_t>& offsets, size_t count, const options& opt,
            size_t stage, size_t point, size_t batch, size_t stream) {
        const size_t nv=l.norm.n_cols;
        residual_result answer; answer.mean.zeros(nv,nv);
        if (!(table.normalization>0)) return answer;
        // Samples touch one entry each. Store their sparse updates instead of
        // allocating/clearing an nvirt^2 matrix per sampling thread and stream.
        std::vector<size_t> rows(count), columns(count);
        std::vector<double> values(count);
        std::vector<double> square(opt.threads,0.);
        const double draw_start=omp_get_wtime();
        #pragma omp parallel for num_threads(opt.threads) schedule(static)
        for(int shard=0;shard<opt.threads;++shard) {
            const size_t tid=size_t(shard);
            auto rng=generator(opt.seed,stage,point,batch,stream,tid);
            const size_t begin=count*tid/size_t(opt.threads);
            const size_t end=count*(tid+1)/size_t(opt.threads);
            for (size_t n=begin;n<end;++n) {
                const auto x=table.draw_weighted<Term>(rng,offsets);
                const double value=x.value;
                rows[n]=x.a; columns[n]=x.b; values[n]=value;
                square[tid]+=(x.a==x.b ? 1. : .5)*value*value;
            }
        }
        for (size_t n=0;n<count;++n) {
            const size_t a=rows[n],b=columns[n];
            const double value=values[n];
            answer.mean(a,b)+=value;
        }
        // Include the per-draw scatter, but exclude fixed matrix setup/reductions.
        answer.draw_seconds=omp_get_wtime()-draw_start;
        answer.mean/=double(count);
        const arma::mat sym=(answer.mean+answer.mean.t())/2.;
        answer.frobenius_variance=std::max(0.,(std::accumulate(square.begin(),square.end(),0.)
            -double(count)*arma::accu(arma::square(sym)))/double(count-1));
        return answer;
    }
};

inline result estimate_virtual(const arma::mat& fo, const arma::mat& fv, const arma::mat& bare,
        const arma::Col<size_t>& offsets, const arma::vec& beta, const arma::vec& weights,
        const std::vector<size_t>& group, const options& opt, std::ostream* log=nullptr) {
    return estimate_block<virtual_policy>(fo,fv,bare,offsets,beta,weights,group,opt,log);
}

}}}}
// Density reverse pass

#include <cstdint>

namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {

// Proposal probabilities, pilot allocations, samples, and strong-domain masks
// are fixed in this VJP. The resulting sample average differentiates the same
// realized density estimate returned by the forward pair estimator.
struct density_cotangent_hooks {
    arma::mat occupied, virtuals;
    std::function<void(size_t,point_data&,const propagator&)> evict;
    density_cotangent_hooks(const arma::mat& o,const arma::mat& v,
            std::function<void(size_t,point_data&,const propagator&)> on_evict)
        : occupied(0.5*(o+o.t())),virtuals(0.5*(v+v.t())),evict(std::move(on_evict)) {}
};

struct strong_vjp_profile {
    double setup_seconds=0, gram_seconds=0, backward_seconds=0;
    // These four counters sum elapsed seconds across worker threads.
    double occupied_adjoint_seconds=0,virtual_adjoint_seconds=0;
    double factor_product_seconds=0,selected_merge_seconds=0;
    // Full-cube capacity for comparison with the compact active-block count.
    size_t dense_gram_elements=0, active_gram_elements=0;
};

inline std::pair<arma::uvec,arma::uvec> matching_domain_positions(
        const arma::uvec& source,const arma::uvec& target) {
    std::vector<arma::uword> s,t;
    size_t i=0,j=0;
    while (i<source.n_elem && j<target.n_elem) {
        if (source(i)<target(j)) ++i;
        else if (target(j)<source(i)) ++j;
        else { s.push_back(i++); t.push_back(j++); }
    }
    return {arma::uvec(s),arma::uvec(t)};
}

struct compact_domain_match {
    arma::uvec source,target;
};

inline void add_occupied_pair_adjoint_compact(arma::mat& target,
        const arma::mat& g,double weight,
        const compact_domain_match& direct_rows,
        const compact_domain_match& direct_cols,
        const compact_domain_match& transpose_rows,
        const compact_domain_match& transpose_cols) {
    if (weight==0) return;
    if (direct_rows.source.n_elem && direct_cols.source.n_elem)
        target.submat(direct_rows.target,direct_cols.target)+=
            2*weight*g.submat(direct_rows.source,direct_cols.source);
    if (transpose_rows.source.n_elem && transpose_cols.source.n_elem)
        target.submat(transpose_rows.target,transpose_cols.target)-=
            weight*g.submat(transpose_cols.source,transpose_rows.source).t();
}

// G stores only source_i x source_j. Restrict each W G or G W product to
// the requested target block without embedding G in an nv x nv matrix.
inline void add_virtual_pair_adjoint_compact(arma::mat& target,
        const arma::mat& g,const arma::mat& w,
        const arma::uvec& source_i,const arma::uvec& source_j,
        const arma::uvec& target_i,const arma::uvec& target_j,double scale=1.) {
    if (source_i.is_empty() || source_j.is_empty() ||
            target_i.is_empty() || target_j.is_empty()) return;
    const auto ii=matching_domain_positions(source_i,target_i);
    const auto jj=matching_domain_positions(source_j,target_j);
    const auto ij=matching_domain_positions(source_i,target_j);
    const auto ji=matching_domain_positions(source_j,target_i);
    if (jj.first.n_elem)
        target.cols(jj.second)+=scale*(w.submat(target_i,source_i)*g.cols(jj.first));
    if (ii.first.n_elem)
        target.rows(ii.second)+=scale*(g.rows(ii.first)*w.submat(source_j,target_j));
    if (ij.first.n_elem)
        target.cols(ij.second)-=0.5*scale*(w.submat(target_i,source_j)*g.rows(ij.first).t());
    if (ji.first.n_elem)
        target.rows(ji.second)-=0.5*scale*(g.cols(ji.first).t()*w.submat(source_i,target_j));
}

inline void strong_compact_pair_vjp(const point_data& l,const point_data& m,
        const std::vector<size_t>& group,const arma::mat& wo,const arma::mat& wv,double factor,
        arma::mat& bar_l,arma::mat& bar_m,int threads,strong_vjp_profile* profile) {
    double tick=omp_get_wtime();
    const size_t no=l.norm.n_rows,nv=l.norm.n_cols,np=l.c.n_rows;
    const bool diagonal=&l==&m;
    std::vector<arma::uvec> ld(no),md_storage(diagonal ? 0 : no);
    for (size_t j=0;j<no;++j) {
        ld[j]=arma::find(l.strong.row(j).t());
        if (!diagonal) md_storage[j]=arma::find(m.strong.row(j).t());
    }
    const auto& md=diagonal ? ld : md_storage;
    const bool occupied_active=arma::accu(arma::abs(wo))!=0;
    const bool virtual_active=arma::accu(arma::abs(wv))!=0;
    std::vector<compact_domain_match> m_to_l, l_to_m_storage;
    if (occupied_active) {
        m_to_l.resize(no*no);
        if (!diagonal) l_to_m_storage.resize(no*no);
        for (size_t k=0;k<no;++k) for (size_t j=0;j<no;++j) {
            const auto ml=matching_domain_positions(md[k],ld[j]);
            m_to_l[k*no+j]={ml.first,ml.second};
            if (!diagonal) {
                const auto lm=matching_domain_positions(ld[k],md[j]);
                l_to_m_storage[k*no+j]={lm.first,lm.second};
            }
        }
    }
    const auto& l_to_m=diagonal ? m_to_l : l_to_m_storage;
    if (profile) profile->setup_seconds+=omp_get_wtime()-tick;
    for (size_t i:group) {
        if (ld[i].is_empty() && md[i].is_empty()) continue;
        const double gram_tick=omp_get_wtime();
        std::vector<arma::mat> gl_scratch,gm_scratch;
        // The forward exact core has just built these selected-i Gram blocks.
        // Reuse its bounded cache; compact_blocks also handles a superset hit
        // or rebuilds panels when the cache cap prevents retention.
        const auto& gl=compact_blocks(l,i,ld,threads,gl_scratch);
        const auto& gm=diagonal ? gl : compact_blocks(m,i,md,threads,gm_scratch);
        if (profile) profile->gram_seconds+=omp_get_wtime()-gram_tick;
        tick=omp_get_wtime();
        // Pack once per selected i: every j product below reuses these columns.
        const arma::mat ci_l=ld[i].is_empty() ? arma::mat(np,0) : l.c.cols(i*nv+ld[i]);
        const arma::mat ci_m=(diagonal || md[i].is_empty())
            ? arma::mat(np,0) : m.c.cols(i*nv+md[i]);
        if (profile) {
            profile->dense_gram_elements+=(diagonal ? 1 : 2)*no*nv*nv;
            for (size_t j=0;j<no;++j)
                profile->active_gram_elements+=ld[i].n_elem*ld[j].n_elem
                    +(diagonal ? 0 : md[i].n_elem*md[j].n_elem);
            profile->setup_seconds+=omp_get_wtime()-tick;
        }
        std::vector<arma::mat> selected_l_workers(threads),selected_m_workers;
        if (!diagonal) selected_m_workers.resize(threads);
        tick=omp_get_wtime();
        #pragma omp parallel num_threads(threads)
        {
            const int worker=omp_get_thread_num(),workers=omp_get_num_threads();
            double occupied_time=0.,virtual_time=0.,product_time=0.;
            arma::mat selected_l(np,ld[i].n_elem,arma::fill::zeros);
            arma::mat selected_m;
            if (!diagonal) selected_m.zeros(np,md[i].n_elem);
            #pragma omp for schedule(static)
            for (size_t j=0;j<no;++j) {
                arma::mat adj_l(ld[i].n_elem,ld[j].n_elem,arma::fill::zeros);
                arma::mat adj_m;
                if (!diagonal) adj_m.zeros(md[i].n_elem,md[j].n_elem);
                double stage_tick=profile ? omp_get_wtime() : 0.;
                if (occupied_active) for (size_t k=0;k<no;++k) {
                    add_occupied_pair_adjoint_compact(adj_l,gm[k],
                        diagonal ? wo(j,k)+wo(k,j) : wo(j,k),
                        m_to_l[i*no+i],m_to_l[k*no+j],
                        m_to_l[k*no+i],m_to_l[i*no+j]);
                    if (!diagonal) add_occupied_pair_adjoint_compact(adj_m,gl[k],wo(k,j),
                        l_to_m[i*no+i],l_to_m[k*no+j],
                        l_to_m[k*no+i],l_to_m[i*no+j]);
                }
                if (profile) { occupied_time+=omp_get_wtime()-stage_tick; stage_tick=omp_get_wtime(); }
                if (virtual_active) {
                    add_virtual_pair_adjoint_compact(adj_l,gm[j],wv,
                        md[i],md[j],ld[i],ld[j],diagonal ? 2. : 1.);
                    if (!diagonal) add_virtual_pair_adjoint_compact(adj_m,gl[j],wv,
                        ld[i],ld[j],md[i],md[j]);
                }
                if (profile) { virtual_time+=omp_get_wtime()-stage_tick; stage_tick=omp_get_wtime(); }
                if (ld[i].n_elem && ld[j].n_elem) {
                    const arma::mat a=factor*adj_l;
                    selected_l+=l.c.cols(j*nv+ld[j])*a.t();
                    const arma::mat partner=ci_l*a;
                    if (j==i) selected_l+=partner;
                    else bar_l.cols(j*nv+ld[j])+=partner;
                }
                if (!diagonal && md[i].n_elem && md[j].n_elem) {
                    const arma::mat a=factor*adj_m;
                    selected_m+=m.c.cols(j*nv+md[j])*a.t();
                    const arma::mat partner=ci_m*a;
                    if (j==i) selected_m+=partner;
                    else bar_m.cols(j*nv+md[j])+=partner;
                }
                if (profile) product_time+=omp_get_wtime()-stage_tick;
            }
            const double merge_tick=profile ? omp_get_wtime() : 0.;
            selected_l_workers[worker]=std::move(selected_l);
            if (!diagonal) selected_m_workers[worker]=std::move(selected_m);
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (size_t a=0;a<ld[i].n_elem;++a)
                for (int t=0;t<workers;++t)
                    bar_l.col(i*nv+ld[i](a))+=selected_l_workers[t].col(a);
            if (!diagonal) {
                #pragma omp for schedule(static)
                for (size_t a=0;a<md[i].n_elem;++a)
                    for (int t=0;t<workers;++t)
                        bar_m.col(i*nv+md[i](a))+=selected_m_workers[t].col(a);
            }
            if (profile) {
                #pragma omp atomic
                profile->occupied_adjoint_seconds+=occupied_time;
                #pragma omp atomic
                profile->virtual_adjoint_seconds+=virtual_time;
                #pragma omp atomic
                profile->factor_product_seconds+=product_time;
                #pragma omp atomic
                profile->selected_merge_seconds+=omp_get_wtime()-merge_tick;
            }
        }
        if (profile) profile->backward_seconds+=omp_get_wtime()-tick;
    }
}

// Differentiate the exactly contracted all-strong core. A core is the exact
// Laplace-pair density of C with weak columns set to zero; masks are fixed.
inline void strong_pair_vjp(const point_data& l,const point_data& m,
        const std::vector<size_t>& group,const arma::mat& wo,const arma::mat& wv,
        double factor,arma::mat& bar_l,arma::mat& bar_m,int threads,
        strong_vjp_profile* profile=nullptr) {
    if (arma::accu(arma::abs(wo))==0 && arma::accu(arma::abs(wv))==0) return;
    strong_compact_pair_vjp(l,m,group,wo,wv,factor,bar_l,bar_m,threads,profile);
}

class density_block_locks {
    // Each update touches one atom segment of one occupied column. Distinct
    // segments can be updated concurrently, and separate cache lines keep
    // unrelated OpenMP locks from bouncing between worker cores.
    struct padded_lock {
        omp_lock_t value;
        char padding[64-sizeof(omp_lock_t)];
    };
    static_assert(sizeof(omp_lock_t)<=64,"OpenMP lock exceeds cache line");
    static_assert(sizeof(padded_lock)==64,"Lock padding changed");
    size_t atoms_,stripes_;
    std::vector<padded_lock> locks_;
public:
    density_block_locks(size_t no,size_t atoms,size_t nv=1):
        atoms_(atoms),stripes_(std::max<size_t>(1,std::min<size_t>(16,nv))),
        locks_(no*atoms*stripes_) {
        for (auto& lock:locks_) omp_init_lock(&lock.value);
    }
    ~density_block_locks() { for (auto& lock:locks_) omp_destroy_lock(&lock.value); }
    density_block_locks(const density_block_locks&)=delete;
    density_block_locks& operator=(const density_block_locks&)=delete;
    omp_lock_t* at(size_t i,size_t atom) {
        return &locks_[(i*atoms_+atom)*stripes_].value;
    }
    omp_lock_t* at_column(size_t column,size_t nv,size_t atom) {
        return &locks_[((column/nv)*atoms_+atom)*stripes_
                       +(column%nv)%stripes_].value;
    }
};

// Like the selected-energy cotangent, record atom/target/source/scale, then
// group updates by destination and scatter in parallel. The bounded per-thread
// chunks also cover dense W cotangents without retaining all drawn updates.
class selected_density_updates {
    arma::mat& output_;
    density_block_locks& locks_;
    const arma::Col<size_t>& offsets_;
    const size_t nv_,ncolumns_;
    const bool parallel_reduction_;
    const bool column_major_order_;
    struct update {
        uint32_t column,atom;
        const double* source;
        double scale;
    };
    static_assert(sizeof(update)==24,"Recorded density update size changed");
    // At most two factor channels are live. Account for both record buffers
    // (2*24 bytes/event) and two pointer arrays in the final radix sort
    // (2*8 bytes/event). Reserve the remainder of the 512 MiB allowance for
    // striped locks, counters, and allocator overhead.
    static constexpr size_t update_budget_bytes_=480ull*1024*1024;
    static constexpr size_t max_chunk_size_=260000;
    const size_t chunk_size_;
    std::vector<std::vector<update>> pending_;
    static size_t chunk_size(int threads,size_t max_chunk_size) {
        if (threads<1) throw std::invalid_argument("Recorded density updates need threads >= 1");
        if (!max_chunk_size) throw std::invalid_argument("Recorded density updates need a nonempty chunk");
        const size_t bytes_per_event=2*sizeof(update)+2*sizeof(size_t);
        return std::max<size_t>(1,std::min(max_chunk_size,
            update_budget_bytes_/(bytes_per_event*size_t(threads))));
    }
    uint64_t key(const update& event) const {
        return column_major_order_
            ? uint64_t(event.column)*(offsets_.n_elem-1)+event.atom
            : uint64_t(event.atom)*ncolumns_+event.column;
    }
    // LSD radix grouping needs linear work and bounded scratch.  A local
    // flush sorts 32-bit indices; the final cross-thread reduction sorts
    // pointers, avoiding a second copy of the 24-byte update records.
    template<class Ref,class Access>
    void radix_group(std::vector<Ref>& order,std::vector<Ref>& scratch,
            const Access& access) const {
        std::vector<uint32_t> count(65536);
        const unsigned passes=uint64_t(ncolumns_)*offsets_.n_elem
            <=uint64_t(std::numeric_limits<uint32_t>::max()) ? 2 : 4;
        for (unsigned pass=0;pass<passes;++pass) {
            std::fill(count.begin(),count.end(),0);
            const unsigned shift=pass*16;
            for (const auto& ref:order) ++count[(key(access(ref))>>shift)&65535];
            uint32_t position=0;
            for (auto& size:count) { const uint32_t next=size; size=position; position+=next; }
            for (const auto& ref:order)
                scratch[count[(key(access(ref))>>shift)&65535]++]=ref;
            order.swap(scratch);
        }
    }
    void flush(std::vector<update>& events) {
        std::vector<uint32_t> order(events.size()),scratch(events.size());
        std::iota(order.begin(),order.end(),0);
        radix_group(order,scratch,[&](uint32_t index)->const update& { return events[index]; });
        size_t first=0;
        while (first<order.size()) {
            const size_t column=events[order[first]].column,atom=events[order[first]].atom;
            size_t last=first+1;
            while (last<order.size() && events[order[last]].column==column &&
                   events[order[last]].atom==atom) ++last;
            omp_lock_t* lock=locks_.at_column(column,nv_,atom);
            omp_set_lock(lock);
            double* target=output_.colptr(column);
            for (size_t n=first;n<last;++n) {
                const update& event=events[order[n]];
                const double* source=event.source;
                const double scale=event.scale;
                for (size_t p=offsets_(atom);p<offsets_(atom+1);++p)
                    target[p]+=scale*source[p];
            }
            omp_unset_lock(lock);
            first=last;
        }
        events.clear();
    }
public:
    selected_density_updates(arma::mat& output,density_block_locks& locks,
            const arma::Col<size_t>& offsets,size_t nv,int threads,
            bool parallel_reduction=false,bool column_major_order=false,
            size_t max_chunk_size=max_chunk_size_)
        : output_(output),locks_(locks),offsets_(offsets),nv_(nv),
          ncolumns_(output.n_cols),parallel_reduction_(parallel_reduction),
          column_major_order_(column_major_order),
          chunk_size_(chunk_size(threads,max_chunk_size)),pending_(threads) {
        if (output.n_cols>std::numeric_limits<uint32_t>::max() ||
            offsets.n_elem-1>std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Recorded density update index exceeds 32 bits");
    }
    void add(size_t column,size_t atom,const double* source,double scale) {
        auto& local=pending_[omp_get_thread_num()];
        if (local.size()==local.capacity()) {
            // Jump from at most one-quarter of the final capacity. Even when
            // every worker reallocates both channels together, old and new
            // record storage stays below the final scatter's memory bound.
            size_t next=local.capacity() ? local.capacity()*2 : 64;
            if (next>chunk_size_/4) next=chunk_size_;
            local.reserve(next);
        }
        local.push_back({uint32_t(column),uint32_t(atom),source,scale});
        if (local.size()==chunk_size_) flush(local);
    }
    void reduce(int threads) {
        size_t count=0;
        for (const auto& local:pending_) count+=local.size();
        if (!count) return;
        if(parallel_reduction_ && threads>1) {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int worker=0;worker<threads;++worker)
                flush(pending_[worker]);
            return;
        }
        std::vector<const update*> ordered,scratch;
        ordered.reserve(count);
        for (const auto& local:pending_)
            for (const auto& event:local) ordered.push_back(&event);
        scratch.resize(count);
        radix_group(ordered,scratch,[](const update* event)->const update& { return *event; });
        // Each worker owns a disjoint sorted key range. Adjust cuts to key
        // boundaries so no two workers ever write the same destination block.
        #pragma omp parallel num_threads(threads)
        {
            const size_t thread=omp_get_thread_num(),workers=omp_get_num_threads();
            auto cut=[&](size_t worker) {
                size_t at=count*worker/workers;
                if (at && at<count) {
                    const uint64_t previous=key(*ordered[at-1]);
                    while (at<count && key(*ordered[at])==previous) ++at;
                }
                return at;
            };
            size_t n=cut(thread),end=cut(thread+1);
            while (n<end) {
                const update& first=*ordered[n];
                const uint64_t destination=key(first);
                double* target=output_.colptr(first.column);
                const size_t begin=offsets_(first.atom),finish=offsets_(first.atom+1);
                do {
                    const update& event=*ordered[n++];
                    for (size_t p=begin;p<finish;++p)
                        target[p]+=event.scale*event.source[p];
                } while (n<end && key(*ordered[n])==destination);
            }
        }
    }
};

inline void add_block_dot_vjp(const point_data& point,arma::mat& bar,
        density_block_locks& locks,const arma::Col<size_t>& offsets,size_t atom,
        size_t x,size_t y,double scale,selected_density_updates* selected=nullptr) {
    const size_t first=offsets(atom),last=offsets(atom+1),nv=point.norm.n_cols;
    if (!scale || first==last) return;
    if (selected) {
        selected->add(x,atom,point.c.colptr(y),scale);
        selected->add(y,atom,point.c.colptr(x),scale);
    } else {
        omp_set_lock(locks.at(x/nv,atom));
        for (size_t p=first;p<last;++p) bar(p,x)+=scale*point.c(p,y);
        omp_unset_lock(locks.at(x/nv,atom));
        omp_set_lock(locks.at(y/nv,atom));
        for (size_t p=first;p<last;++p) bar(p,y)+=scale*point.c(p,x);
        omp_unset_lock(locks.at(y/nv,atom));
    }
}

template<bool RecordPrimal=true>
inline void accumulate_occupied_pair_vjp(const proposal& table,const point_data& l,
        const point_data& m,arma::mat& bar_l,arma::mat& bar_m,
        density_block_locks& left,density_block_locks& right,
        const arma::mat& seed,const arma::Col<size_t>& offsets,size_t count,
        size_t stream,std::mt19937& rng,arma::mat& value,double pair_scale,
        selected_density_updates* selected_l=nullptr,
        selected_density_updates* selected_m=nullptr) {
    if (!count) return;
    const size_t nv=l.norm.n_cols;
    const bool exchange=stream>=4;
    const double density_scale=1./double(count);
    for (size_t draw=0;draw<count;++draw) {
        const tuple x=table.draw(rng);
        const double p=table.probability(x);
        const size_t la=x.i*nv+x.a,lb=x.j*nv+x.b;
        const size_t ma=x.i*nv+(exchange ? x.b : x.a);
        const size_t mb=x.k*nv+(exchange ? x.a : x.b);
        const double left_dot=block_dot(l.c,offsets,x.A,la,lb);
        const double right_dot=block_dot(m.c,offsets,x.T,ma,mb);
        const double coefficient=(exchange ? -1. : 2.)/p;
        if (RecordPrimal)
            value(x.j,x.k)+=density_scale*coefficient*left_dot*right_dot;
        const double adjoint=pair_scale*density_scale*seed(x.j,x.k)*coefficient;
        add_block_dot_vjp(l,bar_l,left,offsets,x.A,la,lb,adjoint*right_dot,selected_l);
        add_block_dot_vjp(m,bar_m,right,offsets,x.T,ma,mb,adjoint*left_dot,selected_m);
    }
}

template<size_t Term,bool RecordPrimal=true>
inline void accumulate_virtual_term_vjp(const virtual_proposal& table,
        const point_data& l,const point_data& m,arma::mat& bar_l,arma::mat& bar_m,
        density_block_locks& left,density_block_locks& right,
        const arma::mat& seed,const arma::Col<size_t>& offsets,size_t count,
        std::mt19937& rng,arma::mat& value,double pair_scale,
        selected_density_updates* selected_l=nullptr,
        selected_density_updates* selected_m=nullptr) {
    if (!count) return;
    const size_t nv=l.norm.n_cols;
    const double density_scale=1./double(count);
    for (size_t draw=0;draw<count;++draw) {
        virtual_tuple x;double left_dot,right_dot,q;
        const virtual_update update=table.draw_weighted<Term>(rng,offsets,
            &x,&left_dot,&right_dot,&q);
        if (RecordPrimal) value(update.a,update.b)+=density_scale*update.value;
        const auto legs=virtual_legs(x,Term);
        const size_t la=legs[0].first*nv+legs[0].second;
        const size_t lb=legs[1].first*nv+legs[1].second;
        const size_t ma=legs[2].first*nv+legs[2].second;
        const size_t mb=legs[3].first*nv+legs[3].second;
        const double adjoint=pair_scale*density_scale*seed(x.a,x.b)
            *table.normalization*(Term<2 ? 1. : -.5)/q;
        add_block_dot_vjp(l,bar_l,left,offsets,x.A,la,lb,adjoint*right_dot,selected_l);
        add_block_dot_vjp(m,bar_m,right,offsets,x.T,ma,mb,adjoint*left_dot,selected_m);
    }
}

template<bool RecordPrimal=true>
inline void accumulate_virtual_pair_vjp(const virtual_proposal& table,
        const point_data& l,const point_data& m,arma::mat& bar_l,arma::mat& bar_m,
        density_block_locks& left,density_block_locks& right,
        const arma::mat& seed,const arma::Col<size_t>& offsets,size_t count,
        std::mt19937& rng,arma::mat& value,double pair_scale,
        selected_density_updates* selected_l=nullptr,
        selected_density_updates* selected_m=nullptr) {
    switch (table.term()) {
        case 0: accumulate_virtual_term_vjp<0,RecordPrimal>(table,l,m,bar_l,bar_m,left,right,
                    seed,offsets,count,rng,value,pair_scale,selected_l,selected_m); break;
        case 1: accumulate_virtual_term_vjp<1,RecordPrimal>(table,l,m,bar_l,bar_m,left,right,
                    seed,offsets,count,rng,value,pair_scale,selected_l,selected_m); break;
        default: accumulate_virtual_term_vjp<2,RecordPrimal>(table,l,m,bar_l,bar_m,left,right,
                    seed,offsets,count,rng,value,pair_scale,selected_l,selected_m); break;
    }
}

inline void sampled_occupied_pair_vjp(const point_data& l,const point_data& m,
        arma::mat& bar_l,arma::mat& bar_m,const arma::mat& seed,
        const arma::Col<size_t>& offsets,double factor,
        const std::vector<std::unique_ptr<proposal>>& tables,
        const arma::Col<size_t>& counts,size_t batches,size_t key,
        unsigned random_seed,int threads) {
    if (arma::accu(arma::abs(seed))==0) return;
    density_block_locks left(l.norm.n_rows,offsets.n_elem-1),right(m.norm.n_rows,offsets.n_elem-1);
    density_block_locks& right_locks=&bar_l==&bar_m ? left : right;
    const size_t nv=l.norm.n_cols;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t replica=0;replica<batches;++replica) {
        for (size_t stream=0;stream<tables.size();++stream) {
            const size_t count=counts(stream);
            if (!count || !(tables[stream]->normalization>0)) continue;
            auto rng=generator(random_seed,1,key,replica,stream);
            const bool exchange=stream>=4;
            for (size_t draw=0;draw<count;++draw) {
                const tuple x=tables[stream]->draw(rng);
                const double p=tables[stream]->probability(x);
                if (!(p>0)) continue;
                const size_t la=x.i*nv+x.a,lb=x.j*nv+x.b;
                const size_t ma=x.i*nv+(exchange ? x.b : x.a);
                const size_t mb=x.k*nv+(exchange ? x.a : x.b);
                const double left_dot=block_dot(l.c,offsets,x.A,la,lb);
                const double right_dot=block_dot(m.c,offsets,x.T,ma,mb);
                const double scale=factor*seed(x.j,x.k)*(exchange ? -1. : 2.)
                    /(double(batches)*double(count)*p);
                add_block_dot_vjp(l,bar_l,left,offsets,x.A,la,lb,scale*right_dot);
                add_block_dot_vjp(m,bar_m,right_locks,offsets,x.T,ma,mb,scale*left_dot);
            }
        }
    }
}

inline void sampled_virtual_pair_vjp(const point_data& l,const point_data& m,
        arma::mat& bar_l,arma::mat& bar_m,const arma::mat& seed,
        const arma::Col<size_t>& offsets,double factor,
        const std::vector<std::unique_ptr<virtual_proposal>>& tables,
        const arma::Col<size_t>& counts,size_t batches,size_t key,
        unsigned random_seed,int threads) {
    if (arma::accu(arma::abs(seed))==0) return;
    density_block_locks left(l.norm.n_rows,offsets.n_elem-1),right(m.norm.n_rows,offsets.n_elem-1);
    density_block_locks& right_locks=&bar_l==&bar_m ? left : right;
    const size_t nv=l.norm.n_cols;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t replica=0;replica<batches;++replica) {
        for (size_t stream=0;stream<tables.size();++stream) {
            const size_t count=counts(stream);
            if (!count || !(tables[stream]->normalization>0)) continue;
            auto rng=generator(random_seed,1,key,replica,stream);
            const size_t term=tables[stream]->term();
            for (size_t draw=0;draw<count;++draw) {
                const virtual_tuple x=tables[stream]->draw(rng);
                const double p=tables[stream]->probability(x);
                if (!(p>0)) continue;
                const auto legs=virtual_legs(x,term);
                const size_t la=legs[0].first*nv+legs[0].second;
                const size_t lb=legs[1].first*nv+legs[1].second;
                const size_t ma=legs[2].first*nv+legs[2].second;
                const size_t mb=legs[3].first*nv+legs[3].second;
                const double left_dot=block_dot(l.c,offsets,x.A,la,lb);
                const double right_dot=block_dot(m.c,offsets,x.T,ma,mb);
                const double scale=factor*seed(x.a,x.b)*(term<2 ? 1. : -.5)
                    *double(tables[stream]->selection_weight(x))
                    /(double(batches)*double(count)*p);
                add_block_dot_vjp(l,bar_l,left,offsets,x.A,la,lb,scale*right_dot);
                add_block_dot_vjp(m,bar_m,right_locks,offsets,x.T,ma,mb,scale*left_dot);
            }
        }
    }
}

}}}}
// Combined occupied and virtual estimator

#include <cstdlib>
#include <map>
#include <new>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace libgmbpt { namespace qchem { namespace stc_mp2 { namespace density {
struct combined_result {
    result occupied, virtuals;
    double seconds=0, initial_setup_seconds=0, dressing_seconds=0, point_table_seconds=0;
    double point_callback_seconds=0;
    double strong_vjp_seconds=0, occupied_strong_vjp_seconds=0;
    double virtual_strong_vjp_seconds=0, virtual_update_reduce_seconds=0;
    strong_vjp_profile occupied_strong_profile;
    strong_vjp_profile virtual_strong_profile;
    double cotangent_backprop_seconds=0;
    size_t propagated_points=0, peak_table_bytes=0;
    size_t peak_integral_cache_bytes=0, integral_cache_builds=0, integral_cache_hits=0;
    size_t integral_cache_subset_hits=0, integral_cache_skips=0;
};

// Anonymous mappings start as zero pages and acquire physical memory only for
// cotangent columns that are updated. Keep ownership outside point_data: its
// Armadillo view is destroyed by the pair scheduler before these mappings.
struct mapped_zero_cotangent {
    void* storage=nullptr;
    size_t bytes=0;
    mapped_zero_cotangent()=default;
    mapped_zero_cotangent(const mapped_zero_cotangent&)=delete;
    mapped_zero_cotangent& operator=(const mapped_zero_cotangent&)=delete;
    ~mapped_zero_cotangent() { release(); }
    void release() {
#ifdef __linux__
        if (storage) munmap(storage,bytes);
#endif
        storage=nullptr; bytes=0;
    }
    void reset(arma::mat& matrix,size_t rows,size_t cols) {
        matrix.reset();
        release();
#ifdef __linux__
        if (rows && cols>std::numeric_limits<size_t>::max()/rows/sizeof(double))
            throw std::bad_alloc();
        bytes=rows*cols*sizeof(double);
        storage=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (storage==MAP_FAILED) { storage=nullptr; bytes=0; throw std::bad_alloc(); }
        arma::mat view(static_cast<double*>(storage),rows,cols,false,false);
        matrix.swap(view);
#else
        matrix.zeros(rows,cols);
#endif
    }
};

// B and two (optionally three) C factors are resident. Both blocks share every propagated
// factor, orbital norm and auxiliary-atom table. The optional callback evaluates
// energy once on each distinct factor. Each pair completes pilot and production
// while its factors are resident.
// Shared times live here, while each block reports only its incremental cost.
template<class PointCallback>
inline combined_result estimate_both(const arma::mat& fo, const arma::mat& fv,
        const arma::mat& bare, const arma::Col<size_t>& offsets,
        const arma::vec& beta, const arma::vec& weights, const std::vector<size_t>& group,
        const options& occupied_opt, const options& virtual_opt, PointCallback point_callback,
        std::ostream* log=nullptr,density_cotangent_hooks* cotangent=nullptr,
        std::function<void(size_t,point_data&,const propagator&)> factor_adjoint={},
        const arma::Mat<size_t>* occupied_replay=nullptr,
        const arma::Mat<size_t>* virtual_replay=nullptr) {
    const double start=omp_get_wtime();
    validate(fo,fv,bare,offsets,beta,weights,group,occupied_opt);
    validate(fo,fv,bare,offsets,beta,weights,group,virtual_opt);
    if (occupied_opt.threads!=virtual_opt.threads
        || occupied_opt.symmetric_pairs!=virtual_opt.symmetric_pairs
        || occupied_opt.resident_points!=virtual_opt.resident_points)
        throw std::invalid_argument("Combined densities require pair allocation and matching threads, symmetry and allocation mode");
    if (cotangent && (cotangent->occupied.n_rows!=fo.n_rows || cotangent->occupied.n_cols!=fo.n_cols ||
        cotangent->virtuals.n_rows!=fv.n_rows || cotangent->virtuals.n_cols!=fv.n_cols ||
        !cotangent->occupied.is_finite() || !cotangent->virtuals.is_finite()))
        throw std::invalid_argument("Invalid combined density cotangent configuration");
    if (bool(occupied_replay)!=bool(virtual_replay))
        throw std::invalid_argument("Both density replay allocations are required");
    combined_result out;
    pair_estimator<occupied_policy> occupied(fo.n_rows,fv.n_rows,beta.n_elem,offsets,group,occupied_opt,occupied_replay);
    pair_estimator<virtual_policy> virtuals(fo.n_rows,fv.n_rows,beta.n_elem,offsets,group,virtual_opt,virtual_replay);
    propagator dress(fo,fv,bare,beta,occupied_opt.threads);
    const auto quota=fixed_pair_quotas(fo,fv,beta,weights,occupied_opt.symmetric_pairs);
    occupied.set_quotas(quota); virtuals.set_quotas(quota);
    // These pointers are inspected only inside prepare/process callbacks,
    // while the scheduler's two/three point_data objects are still alive.
    std::vector<const point_data*> resident;
    auto measure_cache=[&]() {
        size_t bytes=0;
        out.integral_cache_builds=out.integral_cache_hits=0;
        out.integral_cache_subset_hits=out.integral_cache_skips=0;
        for (const auto* point : resident) {
            bytes+=point->integral_cache_bytes();
            out.integral_cache_builds+=point->integral_cache_builds;
            out.integral_cache_hits+=point->integral_cache_hits;
            out.integral_cache_subset_hits+=point->integral_cache_subset_hits;
            out.integral_cache_skips+=point->integral_cache_skips;
        }
        out.peak_integral_cache_bytes=std::max(out.peak_integral_cache_bytes,bytes);
    };
    std::vector<bool> callback_done(beta.n_elem,false);
    std::vector<bool> factor_adjoint_pending(beta.n_elem,false);
    const bool sparse_cotangent=false; // Explicit standalone dense cotangent storage.
    std::map<point_data*,mapped_zero_cotangent> mapped_cotangents;
    const arma::mat zero_occupied(fo.n_rows,fo.n_cols,arma::fill::zeros);
    const arma::mat zero_virtual(fv.n_rows,fv.n_cols,arma::fill::zeros);
    out.initial_setup_seconds=omp_get_wtime()-start;
    auto prepare=[&](size_t point, point_data& data) {
        if (std::find(resident.begin(),resident.end(),&data)==resident.end()) resident.push_back(&data);
        double tick=omp_get_wtime(); dress.apply(point,data.c);
        out.dressing_seconds+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        data.prepare(fo.n_rows,fv.n_rows,offsets,weights(point),occupied_opt.domain_tau,
                     occupied_opt.threads,occupied_opt.atom_block_l2,
                     virtual_opt.w_virtual_reform);
        out.point_table_seconds+=omp_get_wtime()-tick;
        ++out.propagated_points;
        if (cotangent || (factor_adjoint && !callback_done[point])) {
            if (sparse_cotangent)
                mapped_cotangents[&data].reset(data.bar_c,data.c.n_rows,data.c.n_cols);
            else data.bar_c.zeros(data.c.n_rows,data.c.n_cols);
        }
        if (!callback_done[point]) {
            tick=omp_get_wtime(); point_callback(point,data);
            out.point_callback_seconds+=omp_get_wtime()-tick;
            if (factor_adjoint) factor_adjoint_pending[point]=true;
            callback_done[point]=true;
        }
    };
    // Each block's proposals are destroyed before switching masks. Only the
    // small strong/weak masks change; no C, norm or auxiliary table is copied.
    auto process=[&](size_t l,size_t m,point_data& a,point_data& b,size_t bytes) {
        const double factor=(occupied_opt.symmetric_pairs && l!=m ? 2. : 1.)*weights(l)*weights(m);
        double tick=omp_get_wtime();
        a.set_domain(weights(l),occupied_opt.domain_tau);
        if (&a!=&b) b.set_domain(weights(m),occupied_opt.domain_tau);
        typename pair_estimator<occupied_policy>::vjp_callback ovjp;
        std::unique_ptr<density_block_locks> occupied_left_locks,occupied_right_locks;
        std::unique_ptr<selected_density_updates> occupied_left_updates,occupied_right_updates;
        if (cotangent) {
            occupied_left_locks.reset(new density_block_locks(fo.n_rows,offsets.n_elem-1,fv.n_rows));
            if (&a!=&b) occupied_right_locks.reset(new density_block_locks(fo.n_rows,offsets.n_elem-1,fv.n_rows));
            occupied_left_updates.reset(new selected_density_updates(a.bar_c,
                *occupied_left_locks,offsets,fv.n_rows,occupied_opt.threads));
            if (&a!=&b) occupied_right_updates.reset(new selected_density_updates(b.bar_c,
                *occupied_right_locks,offsets,fv.n_rows,occupied_opt.threads));
            ovjp.core=[&]() {
                const double vjp_tick=omp_get_wtime();
                strong_pair_vjp(a,b,group,cotangent->occupied,zero_virtual,factor,
                                a.bar_c,b.bar_c,occupied_opt.threads,
                                &out.occupied_strong_profile);
                const double elapsed=omp_get_wtime()-vjp_tick;
                out.strong_vjp_seconds+=elapsed;
                out.occupied_strong_vjp_seconds+=elapsed;
            };
            ovjp.production=[&](const proposal& table,const arma::Col<size_t>& off,
                    size_t count,size_t stream,std::mt19937& rng,arma::mat& value,
                    double pair_scale) {
                density_block_locks& right_locks=occupied_right_locks
                    ? *occupied_right_locks : *occupied_left_locks;
                selected_density_updates* right_updates=occupied_right_updates
                    ? occupied_right_updates.get() : occupied_left_updates.get();
                accumulate_occupied_pair_vjp(table,a,b,a.bar_c,b.bar_c,
                    *occupied_left_locks,right_locks,cotangent->occupied,off,count,
                    stream,rng,value,pair_scale,
                    occupied_left_updates.get(),right_updates);
            };
        }
        occupied.process(l,m,factor,a,b,bytes,log,"Occupied ",ovjp);
        if (occupied_left_updates) occupied_left_updates->reduce(occupied_opt.threads);
        if (occupied_right_updates) occupied_right_updates->reduce(occupied_opt.threads);
        occupied_left_updates.reset(); occupied_right_updates.reset();
        occupied_left_locks.reset(); occupied_right_locks.reset();
        occupied.answer.seconds+=omp_get_wtime()-tick;
        measure_cache();
        tick=omp_get_wtime();
        if (virtual_opt.domain_tau!=occupied_opt.domain_tau) {
            a.set_domain(weights(l),virtual_opt.domain_tau);
            if (&a!=&b) b.set_domain(weights(m),virtual_opt.domain_tau);
        }
        typename pair_estimator<virtual_policy>::vjp_callback vvjp;
        std::unique_ptr<density_block_locks> virtual_left_locks,virtual_right_locks;
        std::unique_ptr<selected_density_updates> virtual_left_updates,virtual_right_updates;
        if (cotangent) {
            virtual_left_locks.reset(new density_block_locks(fo.n_rows,offsets.n_elem-1,fv.n_rows));
            if (&a!=&b) virtual_right_locks.reset(new density_block_locks(fo.n_rows,offsets.n_elem-1,fv.n_rows));
            virtual_left_updates.reset(new selected_density_updates(a.bar_c,
                *virtual_left_locks,offsets,fv.n_rows,virtual_opt.threads));
            if (&a!=&b) virtual_right_updates.reset(new selected_density_updates(b.bar_c,
                *virtual_right_locks,offsets,fv.n_rows,virtual_opt.threads));
            vvjp.core=[&]() {
                const double vjp_tick=omp_get_wtime();
                strong_pair_vjp(a,b,group,zero_occupied,cotangent->virtuals,factor,
                                a.bar_c,b.bar_c,virtual_opt.threads,
                                &out.virtual_strong_profile);
                const double elapsed=omp_get_wtime()-vjp_tick;
                out.strong_vjp_seconds+=elapsed;
                out.virtual_strong_vjp_seconds+=elapsed;
            };
            vvjp.production=[&](const virtual_proposal& table,const arma::Col<size_t>& off,
                    size_t count,size_t,std::mt19937& rng,arma::mat& value,
                    double pair_scale) {
                density_block_locks& right_locks=virtual_right_locks
                    ? *virtual_right_locks : *virtual_left_locks;
                selected_density_updates* right_updates=virtual_right_updates
                    ? virtual_right_updates.get() : virtual_left_updates.get();
                accumulate_virtual_pair_vjp(table,a,b,a.bar_c,b.bar_c,
                    *virtual_left_locks,right_locks,cotangent->virtuals,off,count,
                    rng,value,pair_scale,
                    virtual_left_updates.get(),right_updates);
            };
        }
        virtuals.process(l,m,factor,a,b,bytes,log,"Virtual ",vvjp);
        const double reduce_tick=omp_get_wtime();
        if (virtual_left_updates) virtual_left_updates->reduce(virtual_opt.threads);
        if (virtual_right_updates) virtual_right_updates->reduce(virtual_opt.threads);
        out.virtual_update_reduce_seconds+=omp_get_wtime()-reduce_tick;
        virtuals.answer.seconds+=omp_get_wtime()-tick;
        measure_cache();
    };
    visit_pairs_once(beta.n_elem,occupied_opt.symmetric_pairs,prepare,
            [&](size_t l,size_t m,point_data& a,point_data& b,size_t bytes) {
                process(l,m,a,b,bytes);
            },[&](size_t point,point_data& data) {
                if (cotangent || (factor_adjoint && factor_adjoint_pending[point])) {
                    const double tick=omp_get_wtime();
                    if (cotangent) cotangent->evict(point,data,dress);
                    else {
                        factor_adjoint(point,data,dress);
                        factor_adjoint_pending[point]=false;
                    }
                    out.cotangent_backprop_seconds+=omp_get_wtime()-tick;
                }
            },occupied_opt.resident_points);
    occupied.finish(); virtuals.finish();
    out.peak_table_bytes=std::max(occupied.answer.peak_table_bytes,virtuals.answer.peak_table_bytes);
    out.occupied=std::move(occupied.answer); out.virtuals=std::move(virtuals.answer);
    out.seconds=omp_get_wtime()-start;
    return out;
}
inline combined_result estimate_both(const arma::mat& fo, const arma::mat& fv,
        const arma::mat& bare, const arma::Col<size_t>& offsets,
        const arma::vec& beta, const arma::vec& weights, const std::vector<size_t>& group,
        const options& occupied_opt, const options& virtual_opt, std::ostream* log=nullptr) {
    return estimate_both(fo,fv,bare,offsets,beta,weights,group,occupied_opt,virtual_opt,
                         [](size_t,point_data&){},log);
}
}}}}
#endif
