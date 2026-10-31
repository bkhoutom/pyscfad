// Packing and validation for the native triples kernels.
#include "native/stc_triples.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

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

py::dict asymmetric(py::handle t1_value, py::handle t2_value,
                    py::handle bvo_value, py::handle boo_value, py::handle bvv_value,
                    py::handle left_value, py::handle right_value,
                    py::handle sample_value, py::handle seed_value,
                    py::handle target_value, py::handle fov_value, bool with_vjp,
                    py::object pilot_value) {
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
    InputArray bvo(bvo_value,"Bvo",3), boo(boo_value,"Boo",3), bvv(bvv_value,"Bvv",3);
    InputArray left(left_value,"L",2), right(right_value,"R",2);
    const auto no=t1.size(0), nv=t1.size(1), naux=bvo.size(0);
    if (no<=0 || nv<=0 || naux<=0)
        throw py::value_error("triples requires nonempty occupied, virtual, and auxiliary dimensions");
    t2.shape({no,no,nv,nv}); bvo.shape({naux,no,nv});
    boo.shape({naux,no,no}); bvv.shape({naux,nv,nv});
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
    for (py::ssize_t i=0; i<no; ++i) {
        singles(i).set_size(nv);
        for (py::ssize_t a=0; a<nv; ++a) singles(i)(a)=t1.get({i,a});
        for (py::ssize_t j=0; j<no; ++j) {
            doubles(i,j).set_size(nv,nv);
            for (py::ssize_t a=0; a<nv; ++a)
                for (py::ssize_t b=0; b<nv; ++b) doubles(i,j)(a,b)=t2.get({i,j,a,b});
        }
    }
    const auto Bvo=df_matrix(bvo), Boo=df_matrix(boo), Bvv=df_matrix(bvv);
    Triples::triples_result result;
    Triples::triples_vjp bars;
    {
        py::gil_scoped_release release;
        result=Triples::asymmetric_triples(singles,doubles,Bvo,Boo,Bvv,
            L,R,size_t(samples),seed,target,
            with_vjp ? &bars : nullptr,1.0,fov_ptr,pilot_ptr);
    }
    py::dict out;
    out["energy"]=result.energy; out["standard_error"]=result.standard_error;
    out["samples"]=result.samples; out["setup_threads"]=result.setup_threads;
    out["sampling_threads"]=result.sampling_threads;
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
    if (with_vjp) {
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
        cotangents["Bvo"]=df_array(bars.Bvo,no,nv);
        cotangents["Boo"]=df_array(bars.Boo,no,no);
        cotangents["Bvv"]=df_array(bars.Bvv,nv,nv);
        cotangents["L"]=array(bars.occupied_left);
        cotangents["R"]=array(bars.occupied_right);
        cotangents["fov"]=array(bars.fov);
        out["cotangents"]=cotangents;
    }
    return out;
}
}  // namespace

namespace pyscfad { namespace dlno_stc {
void bind_triples_dressing(py::module_& module);
void bind_triples(py::module_& module) {
    bind_triples_dressing(module);
    // Immutable identity covers the extended native numerical sources and binding.
    module.attr("TRIPLES_VERSION")=
        "stc_triples:asymmetric-only:sha256:header=962b3da2176b7a267fad82179681cde04a0495310f9ec58eae1027aef38c4adb:asymmetric=73b292face29e1c712a06cee5901e48ac3618a03cfb021962d38bfea7f9532b1:sampler=3cb84a2fd873ab706d3b67b9f0f9f3119981d22a9ca291535075dd78072f7a96:dressing=ed94a891ccc42d53224305ad48fa298036af5e155753d10f697fc5755318455a:mp2_gradient=496fea31c8e455f6d6e6d84238d6aff9915607cde3649c3434fbe21f0efcc573:binding-10";
    for (bool vjp : {false,true}) {
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
    }
}
}} // namespace pyscfad::dlno_stc
