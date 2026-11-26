// Deterministic MRCC triples; shared SciPy BLAS, no external matrix runtime.
#include "triples_types.h"
#include "triples_prepared.h"
#include <pybind11/stl.h>
#include <atomic>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
namespace ccsd_native { namespace triples { namespace {
using index4 = std::array<size_t,4>;
using index6 = std::array<size_t,6>;

const size_t permutations[6][3]={{0,1,2},{1,2,0},{2,0,1},{0,2,1},{1,0,2},{2,1,0}};
const double spin_weights[6]={4,1,1,-2,-2,-2};

// Process-private allocation hint only; never alters host THP policy.
// Ignore unsupported/failed hints and retain ordinary-page correctness.
void triples_large_page_hint(double* data,size_t length) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    static const bool enabled=[] {
        const char* flag=std::getenv("PYSCFAD_TRIPLES_HUGEPAGES");
        return flag && std::strcmp(flag,"0")!=0;
    }();
    const size_t bytes=length*sizeof(double);
    if(!enabled || bytes<size_t(4)*1024*1024) return;
    static const size_t page=size_t(sysconf(_SC_PAGESIZE));
    const uintptr_t begin=reinterpret_cast<uintptr_t>(data);
    // Leave allocation boundary pages untouched: they may contain allocator
    // metadata or another allocation. Huge pages form in aligned interiors.
    const uintptr_t first=(begin+page-1)/page*page;
    const uintptr_t last=(begin+bytes)/page*page;
    if(last>first) (void)madvise(reinterpret_cast<void*>(first),last-first,MADV_HUGEPAGE);
#else
    (void)data;(void)length;
#endif
}

struct tensor {
    int kind;
    index4 dims;
    dense::Matrix value, bar;
    tensor(int k,index4 d,bool gradient,const double* values=nullptr) : kind(k),dims(d) {
        size_t length=1;
        for(auto n:d) {
            if(n && length>std::numeric_limits<size_t>::max()/n)
                throw std::overflow_error("asymmetric (T): tensor dimensions overflow");
            length*=n;
        }
        if(values) value=dense::Matrix(const_cast<double*>(values),length,false,true);
        else value.zeros(length);
        if(gradient) {
            bar.set_size(length);triples_large_page_hint(bar.memptr(),length);bar.zeros();
        }
    }
    size_t offset(const index4& x) const {
        return ((x[0]*dims[1]+x[1])*dims[2]+x[2])*dims[3]+x[3];
    }
    size_t owners() const { return triples_doubles_kind(kind) ? dims[1]*dims[2] : dims[0]*dims[1]; }
    size_t owner(const index4& x) const {
        return triples_doubles_kind(kind) ? x[1]*dims[2]+x[2] : x[0]*dims[1]+x[1];
    }
};

struct tuple {
    index6 x;
    size_t selected; // position of the sole S index among the three occupied legs
};
tuple permute(const tuple& x,size_t p,bool pairs) {
    tuple y=x;
    for(size_t d=0;d<3;++d) {
        y.x[d]=x.x[permutations[p][d]];
        if(pairs) y.x[d+3]=x.x[permutations[p][d]+3];
        if(permutations[p][d]==x.selected) y.selected=d;
    }
    return y;
}

struct operations {
    std::array<size_t,8> ids;
    std::array<index4,8> indices;
};


#include "triples_deterministic.inc"
#include "triples_quadrature.inc"
#include "triples_streaming.inc"
using Triples=stc_triples;
struct InputArray {
    py::array array;
    py::buffer_info info;
    std::string name;
    InputArray(py::handle value, const char* label, int rank) : name(label) {
        if (!py::isinstance<py::array>(value))
            throw py::type_error(name + " must be a NumPy float64 array");
        array = py::reinterpret_borrow<py::array>(value);
        if (!array.dtype().is(py::dtype::of<double>()))
            throw py::type_error(name + " must have real float64 dtype");
        info = array.request();
        if (info.ndim != rank)
            throw py::value_error(name + " has incorrect rank");
    }
    py::ssize_t size(size_t axis) const { return info.shape[axis]; }
    double get(std::initializer_list<py::ssize_t> index) const {
        const char* ptr = static_cast<const char*>(info.ptr);
        size_t axis = 0;
        for (auto i : index) ptr += i * info.strides[axis++];
        double value;
        std::memcpy(&value, ptr, sizeof(value));
        if (!std::isfinite(value))
            throw py::value_error(name + " must contain finite values");
        return value;
    }
    void shape(std::initializer_list<py::ssize_t> expected) const {
        size_t axis = 0;
        for (auto length : expected)
            if (size(axis++) != length)
                throw py::value_error(name + " has inconsistent dimensions");
    }
};

std::uint64_t unsigned_integer(py::handle value, const char* name) {
    if (PyBool_Check(value.ptr()) || !PyIndex_Check(value.ptr()))
        throw py::type_error(std::string(name) + " must be an integer");
    auto integer = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!integer) throw py::error_already_set();
    const auto number = PyLong_AsUnsignedLongLong(integer.ptr());
    if (PyErr_Occurred()) throw py::error_already_set();
    return number;
}

struct PreparedInput {
    Triples::triples_prepared value;
    std::vector<py::array_t<double,py::array::c_style>> owners;
    explicit PreparedInput(py::tuple values) {
        if(values.size()!=26) throw py::value_error("stable triples: expected 26 tensors");
        InputArray t1(values[0],"t1",2), selected(values[8],"selected ovvv",4);
        if(t1.size(0)<=0 || t1.size(1)<=0 || selected.size(0)<=0 || selected.size(0)>t1.size(0))
            throw py::value_error("stable triples: invalid dimensions");
        value.no=size_t(t1.size(0));value.nv=size_t(t1.size(1));value.ns=size_t(selected.size(0));
        const ccsd_native::triples::triples_layout layout(value.no,value.nv,value.ns);
        owners.reserve(values.size());value.tensors.reserve(values.size());
        for(size_t i=0;i<values.size();++i) {
            const auto& spec=layout.tensors[i];InputArray input(values[i],"prepared tensor",int(spec.rank));
            for(size_t d=0;d<spec.rank;++d) if(input.size(d)!=py::ssize_t(spec.dims[d]))
                throw py::value_error("stable triples: inconsistent tensor shape");
            owners.push_back(py::array_t<double,py::array::c_style>::ensure(values[i]));
            const auto& a=owners.back();
            if(!a) throw py::value_error("stable triples: invalid array");
            for(py::ssize_t j=0;j<a.size();++j) if(!std::isfinite(a.data()[j]))
                throw py::value_error("stable triples: nonfinite tensor");
            value.tensors.emplace_back(const_cast<double*>(a.data()),size_t(a.size()),false,true);
        }
    }
};
py::dict mrcc_result(const Triples::triples_result& result) {
    py::dict out;
    out["energy"]=result.energy;out["sampling_seconds"]=result.sampling_seconds;
    out["sampling_threads"]=result.sampling_threads;out["setup_threads"]=result.setup_threads;
    out["resident_tensor_bytes"]=result.resident_tensor_bytes;
    out["reverse_buffer_bytes"]=result.reverse_buffer_bytes;
    out["deterministic_blocks"]=result.deterministic_blocks;
    out["deterministic_workspace_bytes"]=result.deterministic_workspace_bytes;
    out["workspace_budget_bytes"]=result.workspace_budget_bytes;
    return out;
}
py::tuple owned_prepared_bars(Triples::triples_vjp& bars,const ccsd_native::triples::triples_layout& layout) {
    py::tuple out(layout.tensors.size());
    for(size_t i=0;i<layout.tensors.size();++i) {
        const auto& spec=layout.tensors[i];
        auto owner=std::make_unique<dense::Matrix>(std::move(bars.prepared[i]));
        auto* ptr=owner.get();
        py::capsule lifetime(ptr,[](void* p) {delete static_cast<dense::Matrix*>(p);});
        owner.release();
        out[i]=py::array_t<double>(std::vector<py::ssize_t>(spec.dims.begin(),spec.dims.begin()+spec.rank),
                                  ptr->memptr(),lifetime);
    }
    return out;
}

}
void bind_point(py::module_& module) {
    module.attr("TRIPLES_DETERMINISTIC_BLOCKS")=true;
    module.attr("TRIPLES_DIAGONAL_VIRTUAL_DRESSING")=true;
    module.attr("TRIPLES_DIAGONAL_OCCUPIED_DRESSING")=true;
    // Bump the native checkpoint identity when the numerical algorithm changes.
    module.attr("TRIPLES_VERSION")=
        "stc_triples:stable-external-legs:deterministic-blocks-v3:quadrature-reuse-v1:diagonal-virtual-v1:shared-dressing-v3:fused-cubes-v1:direct-L-v1:dynamic-blocks-v1:reverse-memory-v3:fused-singles-adjoint-v1:original-lis-occupied-v1:returned-bars-reserve-v1:streaming-quartic-memory-v1:one-permutation-left-v1:grouped-adjoint-128-v1:point-profile-v1:diagonal-occupied-v1:mrcc-partition-v1:mrcc-singles-only-v2:retained-t2-v1:grouped-profile-v1:blocked-group-pack-v1:direct-adjoint-layouts-v1:t2v-kblocked-v1:mrcc-deterministic-only-v1";
    module.attr("TRIPLES_QUADRATURE_STREAMING")=true;
    module.attr("TRIPLES_GROUPED_ADJOINT")=true;
    module.attr("TRIPLES_RIGHT_PROFILE")=true;
    module.attr("TRIPLES_MAX_STREAMING_BLOCKS")=Triples::maximum_streaming_blocks;
    module.attr("TRIPLES_MRCC_PARTITION")=true;
    module.attr("TRIPLES_CONFIGURABLE_WORKSPACE")=true;
    module.def("triples_mrcc_point",[](py::tuple values,double weight,bool with_grad,
            py::handle workspace_limit,py::handle block_batch_size,bool profile_grouped) {
        PreparedInput input(values);
        const auto budget=unsigned_integer(workspace_limit,"workspace_limit_bytes");
        const auto batch=unsigned_integer(block_batch_size,"block_batch_size");
        Triples::triples_result result;std::vector<Triples::triples_vjp> bars;
        std::vector<double> energies;
        { py::gil_scoped_release release;
          result=triples_stream_kernel(input.value,{&input.value},{weight},size_t(budget),
                size_t(batch),energies,with_grad ? &bars : nullptr,profile_grouped); }
        auto out=mrcc_result(result);
        out["connected_build_blocks"]=result.deterministic_blocks;
        out["connected_gemm_calls"]=12*result.deterministic_blocks;
        out["numerator_cache_peak_bytes"]=result.numerator_cache_peak_bytes;
        out["rank_four_packing_bytes"]=result.rank_four_packing_bytes;
        out["occupied_block_batch_size"]=result.occupied_block_batch_size;
        out["occupied_block_batches"]=result.occupied_block_batches;
        out["grouped_adjoint_gemm_calls"]=result.grouped_adjoint_gemm_calls;
        out["connected_build_seconds"]=result.left_build_wall_seconds;
        out["singles_seconds"]=result.right_wall_seconds;
        out["connected_reverse_seconds"]=result.left_reverse_wall_seconds;
        out["grouped_profile_enabled"]=profile_grouped;
        out["grouped_profile_worker_seconds"]=result.grouped_profile_seconds;
        out["grouped_spin_wall_seconds"]=result.grouped_spin_wall_seconds;
        out["grouped_plan_wall_seconds"]=result.grouped_plan_wall_seconds;
        out["grouped_jobs_wall_seconds"]=result.grouped_jobs_wall_seconds;
        out["grouped_commit_wall_seconds"]=result.grouped_commit_wall_seconds;
        if(with_grad) out["cotangents"]=owned_prepared_bars(bars[0],
            ccsd_native::triples::triples_layout(input.value.no,input.value.nv,input.value.ns));
        return out;
    },py::arg("values"),py::arg("weight")=1.,py::kw_only(),py::arg("with_grad")=false,
      py::arg("workspace_limit_bytes")=size_t(16)*1024*1024*1024,py::arg("block_batch_size")=128,
      py::arg("profile_grouped")=false);

}
}}
