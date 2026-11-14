// Packing and validation for the native triples kernels.
#include "native/stc_triples.h"
#include "native/triples_prepared.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <memory>

namespace py = pybind11;
namespace {
using Triples = libgmbpt::qchem::stc_triples;

// Read through byte strides, including negative/zero/unaligned NumPy strides.
// No force-cast or contiguous conversion is allowed at this boundary.
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

double standard_error_target(py::handle value) {
    if (PyBool_Check(value.ptr()) ||
        !py::isinstance(value, py::module_::import("numbers").attr("Real")))
        throw py::type_error("target_standard_error must be a real number");
    const py::int_ zero(0);
    const auto negative=PyObject_RichCompareBool(value.ptr(),zero.ptr(),Py_LT);
    if (negative<0) throw py::error_already_set();
    if (negative)
        throw py::value_error("target_standard_error must be finite and nonnegative");
    const auto target=py::cast<double>(value);
    if (!std::isfinite(target) || target<0)
        throw py::value_error("target_standard_error must be finite and nonnegative");
    if (target==0) {
        const auto nonzero=PyObject_RichCompareBool(value.ptr(),zero.ptr(),Py_NE);
        if (nonzero<0) throw py::error_already_set();
        if (nonzero)
            throw py::value_error("target_standard_error is too small for float64");
    }
    return target;
}

arma::mat matrix(const InputArray& input) {
    arma::mat result(input.size(0), input.size(1));
    for (py::ssize_t i=0; i<input.size(0); ++i)
        for (py::ssize_t j=0; j<input.size(1); ++j)
            result(i,j) = input.get({i,j});
    return result;
}
arma::mat df_matrix(const InputArray& input) {
    const auto n0=input.size(0), n1=input.size(1), n2=input.size(2);
    arma::mat result(n0,n1*n2);
    for (py::ssize_t p=0; p<n0; ++p)
        for (py::ssize_t i=0; i<n1; ++i)
            for (py::ssize_t j=0; j<n2; ++j)
                result(p,i*n2+j) = input.get({p,i,j});
    return result;
}
py::array_t<double> array(const arma::mat& input) {
    py::array_t<double> result({py::ssize_t(input.n_rows), py::ssize_t(input.n_cols)});
    auto out=result.mutable_unchecked<2>();
    for (arma::uword i=0; i<input.n_rows; ++i)
        for (arma::uword j=0; j<input.n_cols; ++j) out(i,j)=input(i,j);
    return result;
}
py::array_t<double> df_array(const arma::mat& input, py::ssize_t n1, py::ssize_t n2) {
    py::array_t<double> result({py::ssize_t(input.n_rows), n1, n2});
    auto out=result.mutable_unchecked<3>();
    for (arma::uword p=0; p<input.n_rows; ++p)
        for (py::ssize_t i=0; i<n1; ++i)
            for (py::ssize_t j=0; j<n2; ++j) out(p,i,j)=input(p,i*n2+j);
    return result;
}

arma::mat integral_matrix(const InputArray& input) {
    const auto n0=input.size(0), n1=input.size(1), n2=input.size(2), n3=input.size(3);
    arma::mat result(n2*n3,n0*n1);
    for(py::ssize_t i=0;i<n0;++i) for(py::ssize_t a=0;a<n1;++a)
        for(py::ssize_t b=0;b<n2;++b) for(py::ssize_t c=0;c<n3;++c)
            result(b*n3+c,i*n1+a)=input.get({i,a,b,c});
    return result;
}
py::array_t<double> integral_array(const arma::mat& input,
        py::ssize_t n0,py::ssize_t n1,py::ssize_t n2,py::ssize_t n3) {
    py::array_t<double> result({n0,n1,n2,n3});
    // Column-major matrix storage is the C-order (i,a,b,c) tensor layout.
    std::memcpy(result.mutable_data(),input.memptr(),input.n_elem*sizeof(double));
    return result;
}

py::dict asymmetric(py::handle t1_value, py::handle t2_value,
                    py::handle bvo_value, py::handle boo_value, py::handle bvv_value,
                    py::handle left_value, py::handle right_value,
                    py::handle sample_value, py::handle seed_value,
                    py::handle target_value, py::handle fov_value, bool with_vjp,
                    py::object pilot_value, bool integral_inputs=false, py::object prepared_value=py::none()) {
    const auto samples=unsigned_integer(sample_value,"nsamples");
    const auto seed=unsigned_integer(seed_value,"seed");
    const auto target=standard_error_target(target_value);
    if (target>0 && samples==0)
        throw py::value_error("positive target_standard_error requires stochastic nsamples");
    if (samples > std::numeric_limits<size_t>::max())
        throw py::value_error("nsamples is too large");
    if (samples==1)
        throw py::value_error("nsamples must be zero for exact enumeration or at least two for sampling");
    Triples::pilot_controls pilot{};
    const Triples::pilot_controls* pilot_ptr=nullptr;
    if(!pilot_value.is_none()) {
        if(!py::isinstance<py::dict>(pilot_value))
            throw py::type_error("pilot must be a dict or None");
        const auto settings=pilot_value.cast<py::dict>();
        if(settings.size()!=3 || !settings.contains("samples") ||
           !settings.contains("seed") || !settings.contains("allocation_target"))
            throw py::value_error("pilot requires samples, seed and allocation_target");
        const auto count=unsigned_integer(settings["samples"],"pilot samples");
        const auto pilot_seed=unsigned_integer(settings["seed"],"pilot seed");
        const double allocation_target=standard_error_target(settings["allocation_target"]);
        if(count<2 || count>std::numeric_limits<size_t>::max() || samples<2 ||
           allocation_target<=0 || pilot_seed==seed)
            throw py::value_error("pilot requires distinct seeds, positive allocation_target and sample counts >=2");
        pilot={size_t(count),pilot_seed,allocation_target};
        pilot_ptr=&pilot;
    }
    InputArray t1(t1_value,"t1",2), t2(t2_value,"t2",4);
    InputArray left(left_value,"L",2), right(right_value,"R",2);
    const auto no=t1.size(0), nv=t1.size(1);
    if (no<=0 || nv<=0)
        throw py::value_error("triples requires nonempty occupied and virtual dimensions");
    t2.shape({no,no,nv,nv});
    arma::mat Bvo,Boo,Bvv;
    Triples::triples_integrals integrals;
    if(!prepared_value.is_none()) {
        // Raw integral packing is bypassed for already prepared stable tensors.
    } else if(integral_inputs) {
        InputArray ovvv(bvo_value,"ovvv",4), ovoo(boo_value,"ovoo",4), ovov(bvv_value,"ovov",4);
        ovvv.shape({no,nv,nv,nv}); ovoo.shape({no,nv,no,no}); ovov.shape({no,nv,no,nv});
        integrals.ovvv=integral_matrix(ovvv);
        integrals.ovoo=integral_matrix(ovoo); integrals.ovov=integral_matrix(ovov);
    } else {
        InputArray bvo(bvo_value,"Bvo",3), boo(boo_value,"Boo",3), bvv(bvv_value,"Bvv",3);
        const auto naux=bvo.size(0);
        if(naux<=0) throw py::value_error("triples requires nonempty auxiliary dimensions");
        bvo.shape({naux,no,nv}); boo.shape({naux,no,no}); bvv.shape({naux,nv,nv});
        Bvo=df_matrix(bvo); Boo=df_matrix(boo); Bvv=df_matrix(bvv);
    }
    arma::mat fov;
    const arma::mat* fov_ptr=nullptr;
    if (!fov_value.is_none()) {
        InputArray input(fov_value,"fov",2);
        input.shape({no,nv});
        fov=matrix(input);
        fov_ptr=&fov;
    }
    if (left.size(0)!=no || left.size(1)<1 || left.size(1)>no)
        throw py::value_error("L must have shape (nocc, nselected) with 1 <= nselected <= nocc");
    right.shape({no,left.size(1)});
    const auto L=matrix(left), R=matrix(right);
    arma::field<arma::vec> singles(no);
    arma::field<arma::mat> doubles(no,no);
    if(prepared_value.is_none()) for (py::ssize_t i=0; i<no; ++i) {
        singles(i).set_size(nv);
        for (py::ssize_t a=0; a<nv; ++a) singles(i)(a)=t1.get({i,a});
        for (py::ssize_t j=0; j<no; ++j) {
            doubles(i,j).set_size(nv,nv);
            for (py::ssize_t a=0; a<nv; ++a)
                for (py::ssize_t b=0; b<nv; ++b) doubles(i,j)(a,b)=t2.get({i,j,a,b});
        }
    }
    Triples::triples_prepared prepared;
    std::vector<py::array_t<double,py::array::c_style>> retained;
    if(!prepared_value.is_none()) {
        prepared.no=size_t(no); prepared.nv=size_t(nv); prepared.ns=size_t(left.size(1));
        const auto values=prepared_value.cast<py::tuple>();
        const libgmbpt::qchem::triples_layout layout(prepared.no,prepared.nv,prepared.ns);
        if(values.size()!=layout.tensors.size()) throw py::value_error("stable triples: inconsistent tensor count");
        retained.reserve(values.size()); prepared.tensors.reserve(values.size());
        for(size_t i=0;i<values.size();++i) {
            const auto& spec=layout.tensors[i];
            InputArray input(values[i],"prepared tensor",int(spec.rank));
            for(size_t d=0;d<spec.rank;++d)
                if(input.size(d)!=py::ssize_t(spec.dims[d]))
                    throw py::value_error("stable triples: inconsistent tensor shape");
            retained.push_back(py::array_t<double,py::array::c_style>::ensure(values[i]));
            const auto& a=retained.back();
            if(!a) throw py::value_error("stable triples: invalid array");
            for(py::ssize_t j=0;j<a.size();++j)
                if(!std::isfinite(a.data()[j])) throw py::value_error("stable triples: nonfinite tensor");
            prepared.tensors.emplace_back(const_cast<double*>(a.data()),size_t(a.size()),false,true);
        }
    }
    Triples::triples_result result;
    Triples::triples_vjp bars;
    {
        py::gil_scoped_release release;
        result=Triples::asymmetric_triples(singles,doubles,Bvo,Boo,Bvv,
            L,R,size_t(samples),seed,target,
            with_vjp ? &bars : nullptr,1.0,fov_ptr,pilot_ptr,
            integral_inputs ? &integrals : nullptr,
            prepared_value.is_none() ? nullptr : &prepared);
    }
    py::dict out;
    out["energy"]=result.energy; out["standard_error"]=result.standard_error;
    out["samples"]=result.samples; out["setup_threads"]=result.setup_threads;
    out["sampling_threads"]=result.sampling_threads;
    out["packing_blas_threads"]=result.packing_blas_threads;
    out["sampling_blas_threads"]=result.sampling_blas_threads;
    out["finalize_blas_threads"]=result.finalize_blas_threads;
    out["pilot_samples"]=result.pilot_samples;
    out["pilot_standard_error"]=result.pilot_standard_error;
    out["probability_setup_seconds"]=result.probability_setup_seconds;
    out["sampling_seconds"]=result.sampling_seconds;
    out["amplitude_packing_seconds"]=result.amplitude_packing_seconds;
    out["vjp_reverse_seconds"]=result.vjp_reverse_seconds;
    out["independent_pilot_samples"]=result.independent_pilot_samples;
    out["independent_pilot_standard_error"]=result.independent_pilot_standard_error;
    out["independent_pilot_seconds"]=result.independent_pilot_seconds;
    out["projection_setup_seconds"]=result.projection_setup_seconds;
    out["resident_tensor_bytes"]=result.resident_tensor_bytes;
    out["proposal_bytes"]=result.proposal_bytes;
    out["reverse_buffer_bytes"]=result.reverse_buffer_bytes;
    out["deterministic_blocks"]=result.deterministic_blocks;
    out["deterministic_workspace_bytes"]=result.deterministic_workspace_bytes;
    out["vjp_grouping_seconds"]=result.vjp_grouping_seconds;
    out["vjp_accumulation_seconds"]=result.vjp_accumulation_seconds;
    out["vjp_finalize_seconds"]=result.vjp_finalize_seconds;

    py::dict timings;
    timings["probability_setup_seconds"]=result.probability_setup_seconds;
    timings["sampling_seconds"]=result.sampling_seconds;
    timings["amplitude_packing_seconds"]=result.amplitude_packing_seconds;
    timings["vjp_reverse_seconds"]=result.vjp_reverse_seconds;
    timings["independent_pilot_seconds"]=result.independent_pilot_seconds;
    timings["projection_setup_seconds"]=result.projection_setup_seconds;
    timings["vjp_grouping_seconds"]=result.vjp_grouping_seconds;
    timings["vjp_accumulation_seconds"]=result.vjp_accumulation_seconds;
    timings["vjp_finalize_seconds"]=result.vjp_finalize_seconds;
    out["timings"]=timings;
    if(with_vjp && !prepared_value.is_none()) {
        const libgmbpt::qchem::triples_layout layout(prepared.no,prepared.nv,prepared.ns);
        py::tuple cotangents(layout.tensors.size());
        for(size_t i=0;i<layout.tensors.size();++i) {
            const auto& spec=layout.tensors[i];
            std::vector<py::ssize_t> dims(spec.dims.begin(),spec.dims.begin()+spec.rank);
            py::array_t<double> value(dims);
            std::memcpy(value.mutable_data(),bars.prepared[i].memptr(),bars.prepared[i].n_elem*sizeof(double));
            cotangents[i]=std::move(value);
        }
        out["cotangents"]=std::move(cotangents);
    } else if (with_vjp) {
        py::dict cotangents;
        py::array_t<double> t1bar({no,nv}), t2bar({no,no,nv,nv});
        auto s=t1bar.mutable_unchecked<2>(); auto d=t2bar.mutable_unchecked<4>();
        for (py::ssize_t i=0; i<no; ++i) {
            for (py::ssize_t a=0; a<nv; ++a) s(i,a)=bars.t1(i)(a);
            for (py::ssize_t j=0; j<no; ++j)
                for (py::ssize_t a=0; a<nv; ++a)
                    for (py::ssize_t b=0; b<nv; ++b) d(i,j,a,b)=bars.t2(i,j)(a,b);
        }
        cotangents["t1"]=t1bar; cotangents["t2"]=t2bar;
        if(integral_inputs) {
            cotangents["ovvv"]=integral_array(bars.integrals.ovvv,no,nv,nv,nv);
            cotangents["ovoo"]=integral_array(bars.integrals.ovoo,no,nv,no,no);
            cotangents["ovov"]=integral_array(bars.integrals.ovov,no,nv,no,nv);
        } else {
            cotangents["Bvo"]=df_array(bars.Bvo,no,nv);
            cotangents["Boo"]=df_array(bars.Boo,no,no);
            cotangents["Bvv"]=df_array(bars.Bvv,nv,nv);
        }
        cotangents["L"]=array(bars.occupied_left);
        cotangents["R"]=array(bars.occupied_right);
        cotangents["fov"]=array(bars.fov);
        out["cotangents"]=cotangents;
    }
    return out;
}
// Retain any contiguous conversions until the released-GIL native call ends.
struct PreparedInput {
    Triples::triples_prepared value;
    std::vector<py::array_t<double,py::array::c_style>> owners;
    explicit PreparedInput(py::tuple values) {
        if(values.size()!=26) throw py::value_error("stable triples: expected 26 tensors");
        InputArray t1(values[0],"t1",2), selected(values[8],"selected ovvv",4);
        if(t1.size(0)<=0 || t1.size(1)<=0 || selected.size(0)<=0 || selected.size(0)>t1.size(0))
            throw py::value_error("stable triples: invalid dimensions");
        value.no=size_t(t1.size(0));value.nv=size_t(t1.size(1));value.ns=size_t(selected.size(0));
        const libgmbpt::qchem::triples_layout layout(value.no,value.nv,value.ns);
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
py::dict cache_result(const Triples::triples_result& result,const Triples::triples_vjp* bars,
        const Triples::quadrature_cache& cache) {
    py::dict out;
    out["energy"]=result.energy;out["sampling_seconds"]=result.sampling_seconds;
    out["sampling_threads"]=result.sampling_threads;out["setup_threads"]=result.setup_threads;
    out["resident_tensor_bytes"]=result.resident_tensor_bytes;
    out["reverse_buffer_bytes"]=result.reverse_buffer_bytes;
    out["deterministic_blocks"]=result.deterministic_blocks;
    out["deterministic_workspace_bytes"]=result.deterministic_workspace_bytes;
    if(bars) {
        const libgmbpt::qchem::triples_layout layout(cache.no,cache.nv,cache.ns);
        py::tuple values(layout.tensors.size());
        for(size_t i=0;i<layout.tensors.size();++i) {
            const auto& spec=layout.tensors[i];
            py::array_t<double> a(std::vector<py::ssize_t>(spec.dims.begin(),spec.dims.begin()+spec.rank));
            std::memcpy(a.mutable_data(),bars->prepared[i].memptr(),bars->prepared[i].n_elem*sizeof(double));
            values[i]=std::move(a);
        }
        out["cotangents"]=std::move(values);
    }
    return out;
}
}  // namespace

namespace pyscfad { namespace dlno_stc {
void bind_triples_dressing(py::module_& module);
void bind_triples(py::module_& module) {
    bind_triples_dressing(module);
#ifdef PYSCFAD_STC_OPENBLAS_THREAD_CONTROL
    module.attr("TRIPLES_OPENBLAS_THREAD_CONTROL")=true;
#else
    module.attr("TRIPLES_OPENBLAS_THREAD_CONTROL")=false;
#endif
    module.attr("TRIPLES_DETERMINISTIC_BLOCKS")=true;
    module.attr("TRIPLES_DIAGONAL_VIRTUAL_DRESSING")=true;
    // Bump the native checkpoint identity when the numerical algorithm changes.
    module.attr("TRIPLES_VERSION")=
        "stc_triples:stable-external-legs:deterministic-blocks-v3:quadrature-reuse-v1:diagonal-virtual-v1";
    module.attr("TRIPLES_QUADRATURE_REUSE")=true;
    py::class_<Triples::quadrature_cache>(module,"TriplesQuadratureCache")
        .def(py::init([](py::tuple values,py::handle first,py::handle count,bool with_grad) {
            PreparedInput input(values);
            auto cache=std::make_unique<Triples::quadrature_cache>();
            const auto begin=unsigned_integer(first,"first"),length=unsigned_integer(count,"count");
            { py::gil_scoped_release release;
              Triples::quadrature_build(input.value,*cache,size_t(begin),size_t(length),with_grad); }
            return cache;
        }),py::arg("tensors"),py::kw_only(),py::arg("first"),py::arg("count"),py::arg("with_grad")=false)
        .def_property_readonly("cache_bytes",[](const Triples::quadrature_cache& cache) {
            size_t n=0;for(const auto& v:cache.values)n+=v.n_elem*sizeof(double);
            for(const auto& v:cache.bars)n+=v.n_elem*sizeof(double);return n;
        })
        .def("step",[](Triples::quadrature_cache& cache,py::tuple values,double weight) {
            PreparedInput input(values);Triples::triples_vjp bars;Triples::triples_result result;
            { py::gil_scoped_release release;
              result=Triples::quadrature_step(input.value,cache,weight,cache.gradient ? &bars : nullptr); }
            return cache_result(result,cache.gradient ? &bars : nullptr,cache);
        },py::arg("tensors"),py::arg("weight"))
        .def("finish",[](Triples::quadrature_cache& cache,py::tuple values) {
            PreparedInput input(values);Triples::triples_vjp bars;Triples::triples_result result;
            { py::gil_scoped_release release;result=Triples::quadrature_finish(input.value,cache,&bars); }
            return cache_result(result,&bars,cache);
        },py::arg("tensors"));
    for (bool vjp : {false,true}) {
        module.def(vjp ? "triples_prepared_energy_vjp" : "triples_prepared_energy",
            [vjp](py::tuple tensors,py::handle nsamples,py::handle seed,
                  py::handle target,py::object pilot) {
                if(tensors.size()!=26) throw py::value_error("stable triples: expected 26 tensors");
                InputArray t1(tensors[0],"t1",2), selected(tensors[8],"selected ovvv",4);
                py::array_t<double> placeholder({t1.size(0),selected.size(0)});
                std::fill(placeholder.mutable_data(),placeholder.mutable_data()+placeholder.size(),0.);
                return asymmetric(tensors[0],tensors[1],tensors[2],tensors[3],tensors[4],
                    placeholder,placeholder,nsamples,seed,target,py::none(),vjp,pilot,true,tensors);
            },py::arg("tensors"),py::arg("nsamples"),py::arg("seed")=71,
              py::arg("target_standard_error")=0.,py::kw_only(),py::arg("pilot")=py::none(),
              "Stable Laplace numerator with undressed internal legs and direct physical projections. "
              "nsamples=0 uses deterministic selected-occupied block GEMMs, including their VJP.");
        module.def(vjp ? "triples_asymmetric_energy_vjp" : "triples_asymmetric_energy",
            [vjp](py::handle t1,py::handle t2,py::handle Bvo,py::handle Boo,
                  py::handle Bvv,py::handle L,py::object R,py::handle nsamples,
                  py::handle seed,py::handle target_standard_error,py::handle fov,py::object pilot) {
                return asymmetric(t1,t2,Bvo,Boo,Bvv,L,R,nsamples,seed,
                    target_standard_error,fov,vjp,pilot);
            },py::arg("t1"),py::arg("t2"),py::arg("Bvo"),py::arg("Boo"),py::arg("Bvv"),
              py::arg("L"),py::arg("R"),py::arg("nsamples"),py::arg("seed")=71,
              py::arg("target_standard_error")=0.,py::kw_only(),py::arg("fov")=py::none(),
              py::arg("pilot")=py::none(),
              "Laplace numerator with direct S contraction and M=L R^T; denominator is -1. "
              "nsamples=0 enumerates an exact small-system reference.");
        module.def(vjp ? "triples_asymmetric_integrals_energy_vjp" : "triples_asymmetric_integrals_energy",
            [vjp](py::handle t1,py::handle t2,py::handle ovvv,py::handle ovoo,
                  py::handle ovov,py::handle L,py::object R,py::handle nsamples,
                  py::handle seed,py::handle target_standard_error,py::handle fov,py::object pilot) {
                return asymmetric(t1,t2,ovvv,ovoo,ovov,L,R,nsamples,seed,
                    target_standard_error,fov,vjp,pilot,true);
            },py::arg("t1"),py::arg("t2"),py::arg("ovvv"),py::arg("ovoo"),py::arg("ovov"),
              py::arg("L"),py::arg("R"),py::arg("nsamples"),py::arg("seed")=71,
              py::arg("target_standard_error")=0.,py::kw_only(),py::arg("fov")=py::none(),
              py::arg("pilot")=py::none(),
              "Laplace numerator from dressed four-index integrals; no auxiliary factors required.");
    }
}
}} // namespace pyscfad::dlno_stc
