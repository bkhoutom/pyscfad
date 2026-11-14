#include <mutex>
#include "kernel_nanobind_helpers.h"
#include "lapack/lapack_kernels.h"

namespace pyscfad {

namespace nb = nanobind;

using ::xla::ffi::DataType;

void GetLapackKernelsFromScipy() {
    // Use SciPy's public Cython LAPACK API: its wrappers provide the LP64
    // signatures below regardless of the vendor BLAS symbol naming. This
    // shares SciPy's thread pool instead of the wheel's serial BLAS copy.
    // Import before call_once so no Python import can release the GIL while
    // holding the once lock (another importing thread could then deadlock).
    nb::dict capi = nb::module_::import_("scipy.linalg.cython_lapack")
                        .attr("__pyx_capi__");
    auto kernel = [&](const char* name) {
        return nb::cast<nb::capsule>(capi[name]).data();
    };
    static std::once_flag initialized;

    std::call_once(initialized, [&](){
        // Resolve every capsule before assigning any function pointers.
        void* s = kernel("ssygvd");
        void* d = kernel("dsygvd");
        void* c = kernel("chegvd");
        void* z = kernel("zhegvd");
        AssignKernelFn<RealSygvd<float>>(s);
        AssignKernelFn<RealSygvd<double>>(d);
        AssignKernelFn<ComplexHegvd<std::complex<float>>>(c);
        AssignKernelFn<ComplexHegvd<std::complex<double>>>(z);

        AssignKernelFn<EighReal<DataType::F32>>(s);
        AssignKernelFn<EighReal<DataType::F64>>(d);
        AssignKernelFn<EighComplex<DataType::C64>>(c);
        AssignKernelFn<EighComplex<DataType::C128>>(z);
    });
}

nb::dict Registrations() {
    nb::dict dict;

    dict["lapack_ssygvd"] = EncapsulateFunction(RealSygvd<float>::Kernel);
    dict["lapack_dsygvd"] = EncapsulateFunction(RealSygvd<double>::Kernel);
    dict["lapack_chegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<float>>::Kernel);
    dict["lapack_zhegvd"] =
        EncapsulateFunction(ComplexHegvd<std::complex<double>>::Kernel);

    dict["lapack_ssygvd_ffi"] = EncapsulateFfiHandler(lapack_ssygvd_ffi);
    dict["lapack_dsygvd_ffi"] = EncapsulateFfiHandler(lapack_dsygvd_ffi);
    dict["lapack_chegvd_ffi"] = EncapsulateFfiHandler(lapack_chegvd_ffi);
    dict["lapack_zhegvd_ffi"] = EncapsulateFfiHandler(lapack_zhegvd_ffi);
    return dict;
}

NB_MODULE(pyscfad_lapack, m) {
    m.def("initialize", GetLapackKernelsFromScipy);
    m.attr("lapack_backend") = "scipy.linalg.cython_lapack";
    m.def("registrations", &Registrations);

    m.def("sygvd_work_size", &SygvdWorkSize, nb::arg("n"));
    m.def("sygvd_iwork_size", &SygvdIworkSize, nb::arg("n"));
    m.def("hegvd_work_size", &HegvdWorkSize, nb::arg("n"));
    m.def("hegvd_rwork_size", &HegvdRworkSize, nb::arg("n"));
}

} // namespace pyscfad
