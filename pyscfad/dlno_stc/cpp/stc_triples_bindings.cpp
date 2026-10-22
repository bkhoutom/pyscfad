// Packing and validation only; the supplied numerical kernel is unchanged.
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
    const auto target=py::cast<double>(value);
    if (!std::isfinite(target) || target<0)
        throw py::value_error("target_standard_error must be finite and nonnegative");
    return target;
}

arma::mat matrix(const InputArray& input) {
    arma::mat result(input.size(0), input.size(1));
    for (py::ssize_t i=0; i<input.size(0); ++i)
        for (py::ssize_t j=0; j<input.size(1); ++j)
            result(i,j) = input.get({i,j});
    return result;
}
arma::vec vector(const InputArray& input) {
    arma::vec result(input.size(0));
    for (py::ssize_t i=0; i<input.size(0); ++i) result(i) = input.get({i});
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
py::array_t<double> array(const arma::vec& input) {
    py::array_t<double> result(py::ssize_t(input.n_elem));
    std::memcpy(result.mutable_data(), input.memptr(), input.n_elem*sizeof(double));
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

py::dict selected(py::handle t1_value, py::handle t2_value,
                  py::handle bvo_value, py::handle boo_value, py::handle bvv_value,
                  py::handle eo_value, py::handle ev_value, py::handle u_value,
                  py::handle sample_value, py::handle seed_value,
                  py::handle target_value, bool with_vjp) {
    const auto samples=unsigned_integer(sample_value,"nsamples");
    const auto seed=unsigned_integer(seed_value,"seed");
    const auto target=standard_error_target(target_value);
    if (target>0 && samples==0)
        throw py::value_error("positive target_standard_error requires stochastic nsamples");
    if (samples > std::numeric_limits<size_t>::max())
        throw py::value_error("nsamples is too large");
    if (samples==1 || (with_vjp && samples==0))
        throw py::value_error("nsamples must be at least two (zero is energy reference only)");
    InputArray t1(t1_value,"t1",2), t2(t2_value,"t2",4);
    InputArray bvo(bvo_value,"Bvo",3), boo(boo_value,"Boo",3), bvv(bvv_value,"Bvv",3);
    InputArray eo(eo_value,"eocc",1), ev(ev_value,"evir",1), u(u_value,"U",2);
    const auto no=t1.size(0), nv=t1.size(1), naux=bvo.size(0);
    if (no<=0 || nv<=0 || naux<=0)
        throw py::value_error("triples requires nonempty occupied, virtual, and auxiliary dimensions");
    t2.shape({no,no,nv,nv}); bvo.shape({naux,no,nv});
    boo.shape({naux,no,no}); bvv.shape({naux,nv,nv});
    eo.shape({no}); ev.shape({nv});
    if (u.size(0)!=no || u.size(1)<1 || u.size(1)>no)
        throw py::value_error("U must have shape (nocc, nselected) with 1 <= nselected <= nocc");
    const auto eocc=vector(eo), evir=vector(ev);
    if (evir.min()<=eocc.max())
        throw py::value_error("triples requires a positive occupied-virtual gap");
    const auto subspace=matrix(u);
    if (arma::norm(subspace.t()*subspace-arma::eye<arma::mat>(u.size(1),u.size(1)),"fro")>1e-8)
        throw py::value_error("U columns must be orthonormal (Frobenius tolerance 1e-8)");
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
        if (with_vjp)
            result=Triples::selected_energy_vjp(singles,doubles,Bvo,Boo,Bvv,
                eocc,evir,subspace,size_t(samples),bars,seed,target,1.0);
        else
            result=Triples::selected_energy(singles,doubles,Bvo,Boo,Bvv,
                eocc,evir,subspace,size_t(samples),seed,target);
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
    py::dict timings;
    timings["probability_setup_seconds"]=result.probability_setup_seconds;
    timings["sampling_seconds"]=result.sampling_seconds;
    timings["amplitude_packing_seconds"]=result.amplitude_packing_seconds;
    timings["vjp_reverse_seconds"]=result.vjp_reverse_seconds;
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
        cotangents["eocc"]=array(bars.eocc); cotangents["evir"]=array(bars.evir);
        cotangents["U"]=array(bars.occupied_subspace);
        out["cotangents"]=cotangents;
    }
    return out;
}
}  // namespace

namespace pyscfad { namespace dlno_stc {
void bind_triples(py::module_& module) {
    // str is immutable; identity is fixed by the two byte-identical vendored sources.
    module.attr("TRIPLES_VERSION")=
        "stc_triples:sha256:h=273f7818f5cde71f7c07f62127bbdcca37e42b045748434424a737c7de569f2c:"
        "C=a7fc2b8488255ca4fa36968a632f817b88fb7c60e8f67de336d30ab297eec384:binding-2";
    module.def("triples_selected_energy", [](py::handle t1, py::handle t2,
        py::handle Bvo, py::handle Boo, py::handle Bvv, py::handle eocc,
        py::handle evir, py::handle U, py::handle nsamples, py::handle seed,
        py::handle target_standard_error) {
        return selected(t1,t2,Bvo,Boo,Bvv,eocc,evir,U,nsamples,seed,target_standard_error,false);
    }, py::arg("t1"),py::arg("t2"),py::arg("Bvo"),py::arg("Boo"),py::arg("Bvv"),
       py::arg("eocc"),py::arg("evir"),py::arg("U"),py::arg("nsamples"),py::arg("seed")=71,
       py::arg("target_standard_error")=0.0,
       "Selected stochastic (T); positive target_standard_error makes nsamples a pilot count. "
       "nsamples=0 is a small deterministic reference only.");
    module.def("triples_selected_energy_vjp", [](py::handle t1, py::handle t2,
        py::handle Bvo, py::handle Boo, py::handle Bvv, py::handle eocc,
        py::handle evir, py::handle U, py::handle nsamples, py::handle seed,
        py::handle target_standard_error) {
        return selected(t1,t2,Bvo,Boo,Bvv,eocc,evir,U,nsamples,seed,target_standard_error,true);
    }, py::arg("t1"),py::arg("t2"),py::arg("Bvo"),py::arg("Boo"),py::arg("Bvv"),
       py::arg("eocc"),py::arg("evir"),py::arg("U"),py::arg("nsamples"),py::arg("seed")=71,
       py::arg("target_standard_error")=0.0,
       "Fused sampled selected (T) energy and unit-energy VJP of the eight DF inputs; "
       "positive target_standard_error makes nsamples a pilot count.");
}
}}  // namespace pyscfad::dlno_stc
