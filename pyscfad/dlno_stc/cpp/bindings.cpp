#include "stc_mp2_kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <string>
#include <limits>
#include <algorithm>
#include <array>
#include <omp.h>
#ifdef __linux__
#include <dlfcn.h>
#include <sched.h>
#include <cerrno>
#endif

namespace py = pybind11;
namespace stc = pyscfad::dlno_stc;
namespace {
#ifdef __linux__
struct WorkerPlacement {
    int thread=-1, cpu=-1, error=0;
    cpu_set_t affinity;
};
struct RuntimeProbe {
    std::array<WorkerPlacement,32> workers;
    int (*thread_number)();
};
// This callback is C++ only and cannot throw across the GNU OpenMP boundary.
void collect_placement(void* data) noexcept {
    auto& probe=*static_cast<RuntimeProbe*>(data);
    const auto thread=probe.thread_number();
    if (thread<0 || thread>=32) return;
    auto& worker=probe.workers[thread];
    worker.thread=thread;
    CPU_ZERO(&worker.affinity);
    if (sched_getaffinity(0,sizeof(worker.affinity),&worker.affinity)!=0)
        worker.error=errno;
    worker.cpu=sched_getcpu();
    if (worker.cpu<0 && !worker.error) worker.error=errno;
}
#endif
py::dict parallel_runtime_info(py::object runtime_path) {
#ifndef __linux__
    throw std::runtime_error("OpenMP worker affinity diagnostics require Linux");
#else
    using Parallel=void (*)(void (*)(void*),void*,unsigned,unsigned);
    using Integer=int (*)();
    void* handle=nullptr;
    Parallel external=nullptr;
    Integer maximum=&omp_get_max_threads, limit=&omp_get_thread_limit;
    RuntimeProbe probe;
    probe.thread_number=&omp_get_thread_num;
    if (!runtime_path.is_none()) {
        const auto path=py::cast<std::string>(runtime_path);
        handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
        if (!handle)
            throw std::runtime_error("OpenMP runtime must be an already loaded library: "+path);
        external=reinterpret_cast<Parallel>(dlsym(handle,"GOMP_parallel"));
        maximum=reinterpret_cast<Integer>(dlsym(handle,"omp_get_max_threads"));
        limit=reinterpret_cast<Integer>(dlsym(handle,"omp_get_thread_limit"));
        probe.thread_number=reinterpret_cast<Integer>(dlsym(handle,"omp_get_thread_num"));
        if (!external || !maximum || !limit || !probe.thread_number) {
            dlclose(handle);
            throw std::runtime_error("Unsupported OpenMP runtime: GNU GOMP_parallel and omp worker queries are required");
        }
    }
    const auto threads=std::max(1,std::min(32,std::min(maximum(),limit())));
    {
        py::gil_scoped_release release;
        if (external) external(&collect_placement,&probe,static_cast<unsigned>(threads),0);
        else {
            #pragma omp parallel num_threads(threads)
            collect_placement(&probe);
        }
    }
    if (handle) dlclose(handle);
    py::list workers;
    for (const auto& worker:probe.workers) {
        if (worker.thread<0) continue;
        if (worker.error)
            throw std::runtime_error("Cannot read OpenMP worker CPU affinity (errno "+std::to_string(worker.error)+")");
        py::dict row;
        row["thread"]=worker.thread; row["cpu"]=worker.cpu;
        std::vector<int> affinity;
        for (int cpu=0;cpu<CPU_SETSIZE;++cpu)
            if (CPU_ISSET(cpu,&worker.affinity)) affinity.push_back(cpu);
        row["affinity"]=affinity;
        workers.append(row);
    }
    py::dict result;
    result["threads"]=py::len(workers); result["workers"]=workers;
    return result;
#endif
}
py::array checked_array(py::handle value, const char* name, int ndim) {
    if (!py::isinstance<py::array>(value))
        throw py::type_error(std::string(name)+" must be a NumPy array");
    auto a = py::reinterpret_borrow<py::array>(value);
    if (!a.dtype().is(py::dtype::of<double>()))
        throw py::type_error(std::string(name)+" must have float64 dtype");
    if (!(a.flags() & py::array::c_style))
        throw py::value_error(std::string(name)+" must be C-contiguous");
    if (a.ndim() != ndim)
        throw py::value_error(std::string(name)+" has incorrect rank");
    const auto* data = static_cast<const double*>(a.data());
    for (py::ssize_t j = 0; j < a.size(); ++j)
        if (!std::isfinite(data[j]))
            throw py::value_error(std::string(name)+" must contain finite values");
    return a;
}
arma::mat matrix(const py::array& a) {
    arma::mat m(a.shape(0),a.shape(1));
    const auto* p = static_cast<const double*>(a.data());
    for (arma::uword i = 0; i < m.n_rows; ++i)
        for (arma::uword j = 0; j < m.n_cols; ++j) m(i,j) = p[i*m.n_cols+j];
    return m;
}
void symmetric(const arma::mat& m, const char* name) {
    if (!arma::approx_equal(m,m.t(),"both",1e-12,1e-12))
        throw py::value_error(std::string(name)+" must be symmetric");
}
void strictly_diagonal(const arma::mat& m, const char* name) {
    for (arma::uword i=0; i<m.n_rows; ++i)
        for (arma::uword j=0; j<m.n_cols; ++j)
            if (i!=j && m(i,j)!=0.0)
                throw py::value_error(std::string("canonical_fock requires exactly diagonal ")+name);
}
py::array_t<double> array(const arma::mat& m) {
    py::array_t<double> a({static_cast<py::ssize_t>(m.n_rows),static_cast<py::ssize_t>(m.n_cols)});
    auto* p = a.mutable_data();
    for (arma::uword i = 0; i < m.n_rows; ++i)
        for (arma::uword j = 0; j < m.n_cols; ++j) p[i*m.n_cols+j] = m(i,j);
    return a;
}
std::size_t integer_control(const py::dict& controls,const char* key,std::size_t fallback,
                            std::size_t minimum=1) {
    if (!controls.contains(key)) return fallback;
    const auto value=controls[key];
    if (py::isinstance<py::bool_>(value)) throw py::value_error(std::string(key)+" must be an integer");
    auto index=py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index) throw py::error_already_set();
    const auto number=py::cast<std::int64_t>(index);
    if (number<0 || static_cast<std::size_t>(number)<minimum)
        throw py::value_error(std::string(key)+" is below its minimum");
    return static_cast<std::size_t>(number);
}
double real_control(const py::dict& controls,const char* key,double fallback,
                    double minimum,double maximum,bool strict_min=false) {
    if (!controls.contains(key)) return fallback;
    const auto number=py::cast<double>(controls[key]);
    if (!std::isfinite(number) || number>maximum || (strict_min ? number<=minimum:number<minimum))
        throw py::value_error(std::string(key)+" is out of range");
    return number;
}
py::dict solve_binding(bool full, py::handle foo, py::handle fvv, py::handle B,
               py::handle target_projection, py::handle partner_weight,
               py::dict controls, py::object aux_offsets, bool with_grad) {
    bool canonical_fock=false;
    if (controls.contains("canonical_fock")) {
        if (!py::isinstance<py::bool_>(controls["canonical_fock"]))
            throw py::type_error("canonical_fock must be a bool");
        canonical_fock=py::cast<bool>(controls["canonical_fock"]);
    }
    if (canonical_fock && !full)
        throw py::value_error("canonical_fock is supported only by solve_full");
    const auto fo = checked_array(foo,"foo",2), fv = checked_array(fvv,"fvv",2);
    const auto bp = checked_array(B,"B",3);
    const py::ssize_t no = fo.shape(0), nv = fv.shape(0), np = bp.shape(0);
    if (no == 0) throw py::value_error("STC-MP2 requires at least one active occupied orbital");
    if (fo.shape(1)!=no || fv.shape(1)!=nv || bp.shape(1)!=no || bp.shape(2)!=nv)
        throw py::value_error("inconsistent Fock/B shapes");
    stc::Inputs x;
    x.foo=matrix(fo); x.fvv=matrix(fv);
    symmetric(x.foo,"foo"); symmetric(x.fvv,"fvv");
    if (canonical_fock) {
        strictly_diagonal(x.foo,"foo"); strictly_diagonal(x.fvv,"fvv");
    }
    if (!full) {
        const auto m=checked_array(target_projection,"target_projection",2);
        const auto w=checked_array(partner_weight,"partner_weight",2);
        if (m.shape(0)!=1 || m.shape(1)!=no || w.shape(0)!=no || w.shape(1)!=no)
            throw py::value_error("inconsistent five-input shapes");
        x.target_projection=matrix(m); x.partner_weight=matrix(w);
        symmetric(x.partner_weight,"partner_weight");
    }
    stc::Controls settings;
    settings.canonical_fock=canonical_fock;
    const std::string mode=controls.contains("mode") ? py::cast<std::string>(controls["mode"]):"deterministic";
    if (mode!="deterministic" && mode!="stochastic")
        throw py::value_error("mode must be deterministic or stochastic");
    settings.stochastic=mode=="stochastic";
    if (!controls.contains("laplace_roots") || !controls.contains("laplace_weights"))
        throw py::value_error("explicit laplace_roots and laplace_weights are required");
    settings.roots = py::cast<std::vector<double>>(controls["laplace_roots"]);
    settings.weights = py::cast<std::vector<double>>(controls["laplace_weights"]);
    if (settings.roots.empty() || settings.roots.size()!=settings.weights.size())
        throw py::value_error("Laplace roots and weights must have equal nonzero lengths");
    for (double v : settings.roots)
        if (!std::isfinite(v) || v<0) throw py::value_error("laplace_roots must be finite and nonnegative");
    for (double v : settings.weights)
        if (!std::isfinite(v)) throw py::value_error("laplace_weights must be finite");
    settings.adaptive=controls.contains("energy_tolerance") && !controls.contains("production_samples");
    settings.virtual_block_size=integer_control(controls,"virtual_block_size",16);
    settings.production_samples=integer_control(controls,"production_samples",4096,2);
    settings.min_production_samples=integer_control(controls,"min_production_samples",settings.adaptive ? 10000:32,2);
    settings.pilot_samples=integer_control(controls,"pilot_samples",100000,2);
    settings.max_production_samples=integer_control(controls,"max_production_samples",0,0);
    if (settings.max_production_samples && settings.max_production_samples<settings.min_production_samples)
        throw py::value_error("max_production_samples is below min_production_samples");
    settings.system_workload_cutoff=real_control(controls,"system_workload_cutoff",6.5e-3,
                                                  0.,std::numeric_limits<double>::max(),true);
    settings.workload_cutoff=real_control(controls,"workload_cutoff",0.1,0.,1.);
    settings.virtual_keep_fraction=real_control(controls,"virtual_keep_fraction",-1.,0.,1.);
    settings.uniform_mixture=real_control(controls,"uniform_mixture",0.05,0.,1.,true);
    settings.energy_tolerance=real_control(controls,"energy_tolerance",0.,0.,std::numeric_limits<double>::max(),true);
    if (controls.contains("global_seed")) {
        const auto value=controls["global_seed"];
        if (py::isinstance<py::bool_>(value)) throw py::value_error("global_seed must be an integer");
        auto index=py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
        if (!index) throw py::error_already_set();
        settings.global_seed=py::cast<std::uint64_t>(index);
    }
    x.B.set_size(np,no*nv);
    const auto* source = static_cast<const double*>(bp.data());
    for (py::ssize_t P=0; P<np; ++P)
        for (py::ssize_t i=0; i<no; ++i)
            for (py::ssize_t a=0; a<nv; ++a) x.B(P,i*nv+a)=source[(P*no+i)*nv+a];
    if (aux_offsets.is_none()) x.aux_offsets={0,static_cast<arma::uword>(np)};
    else {
        auto offsets = py::array::ensure(aux_offsets);
        if (!offsets || offsets.ndim()!=1 ||
            (offsets.dtype().kind()!='i' && offsets.dtype().kind()!='u'))
            throw py::value_error("aux_offsets must be a one-dimensional integer array");
        const auto values=py::cast<std::vector<std::int64_t>>(offsets);
        if (values.size()<2 || values.front()!=0 || values.back()!=np)
            throw py::value_error("aux_offsets must span [0,naux]");
        for (std::size_t j=0; j<values.size(); ++j) {
            if (values[j]<0 || (j>0 && values[j]<=values[j-1] && !(np==0 && values.size()==2)))
                throw py::value_error("aux_offsets must be strictly increasing");
            x.aux_offsets.push_back(static_cast<arma::uword>(values[j]));
        }
    }
    stc::Result out;
    { py::gil_scoped_release release; out=full ? stc::solve_full(x,settings,with_grad):stc::solve(x,settings,with_grad); }
    if (!std::isfinite(out.energy) || !std::isfinite(out.energy_standard_error))
        throw std::runtime_error("nonfinite STC-MP2 energy or uncertainty");
    if (with_grad && (!out.foo.is_finite() || !out.fvv.is_finite() || !out.B.is_finite() ||
                      !out.target_projection.is_finite() || !out.partner_weight.is_finite()))
        throw std::runtime_error("nonfinite STC-MP2 cotangents");
    py::dict result, diagnostics;
    result["energy"]=out.energy; result["energy_standard_error"]=out.energy_standard_error;
    diagnostics["mode"]=mode; diagnostics["global_seed"]=settings.global_seed;
    diagnostics["canonical_fock"]=settings.canonical_fock;
    if (settings.canonical_fock)
        diagnostics["fock_derivative_kind"]="canonical_diagonal_energies";
    diagnostics["max_production_samples"]=settings.max_production_samples;
    diagnostics["point_variance_targets"]=out.point_variance_targets;
    py::int_ sample_count(0);
    py::list residuals;
    for (const auto& row:out.residuals) {
        py::dict entry;
        entry["point"]=row.point; entry["term"]=row.term==0 ? "direct":"exchange";
        entry["residual"]=row.residual; entry["pilot_samples"]=row.pilot_samples;
        entry["production_samples"]=row.production_samples;
        entry["pilot_seed"]=row.pilot_seed; entry["production_seed"]=row.production_seed;
        if (full && row.pilot_samples>settings.pilot_samples)
            entry["pilot_refinement_seed"]=row.pilot_refinement_seed;
        entry["variance_of_mean"]=row.variance_of_mean;
        sample_count=py::reinterpret_borrow<py::int_>(sample_count.attr("__add__")(py::int_(row.production_samples)));
        residuals.append(entry);
    }
    diagnostics["actual_sample_count"]=sample_count; diagnostics["residuals"]=residuals;
    diagnostics["laplace_point_count"]=settings.roots.size();
    result["diagnostics"]=diagnostics;
    if (with_grad) {
        py::dict bars;
        bars["foo"]=array(out.foo); bars["fvv"]=array(out.fvv);
        if (!full) {
            bars["target_projection"]=array(out.target_projection);
            bars["partner_weight"]=array(out.partner_weight);
        }
        py::array_t<double> bar_B({np,no,nv});
        auto* target=bar_B.mutable_data();
        for (py::ssize_t P=0; P<np; ++P)
            for (py::ssize_t i=0; i<no; ++i)
                for (py::ssize_t a=0; a<nv; ++a) target[(P*no+i)*nv+a]=out.B(P,i*nv+a);
        bars["B"]=bar_B; result["cotangents"]=bars;
    }
    return result;
}
py::dict solve(py::handle foo, py::handle fvv, py::handle B,
               py::handle target_projection, py::handle partner_weight,
               py::dict controls, py::object aux_offsets, bool with_grad) {
    return solve_binding(false,foo,fvv,B,target_projection,partner_weight,controls,aux_offsets,with_grad);
}
py::dict solve_full(py::handle foo, py::handle fvv, py::handle B,
                    py::dict controls, py::object aux_offsets, bool with_grad) {
    return solve_binding(true,foo,fvv,B,py::none(),py::none(),controls,aux_offsets,with_grad);
}
}  // namespace
PYBIND11_MODULE(_stc_mp2,m) {
    m.def("parallel_runtime_info",&parallel_runtime_info,py::arg("runtime_path")=py::none());
    m.doc()="STC-MP2 with bounded pair contractions and an explicit canonical-system specialization";
    m.def("solve",&solve,py::arg("foo"),py::arg("fvv"),py::arg("B"),
          py::arg("target_projection"),py::arg("partner_weight"),py::arg("controls"),
          py::arg("aux_offsets")=py::none(),py::arg("with_grad")=false);
    m.def("solve_full",&solve_full,py::arg("foo"),py::arg("fvv"),py::arg("B"),
          py::arg("controls"),py::arg("aux_offsets")=py::none(),py::arg("with_grad")=false);
}
