#pragma once
#include "stc_mp2_kernel.h"
#include <pybind11/pybind11.h>

namespace pyscfad {
namespace dlno_stc {

// Energy-only or fused unit-energy input bars, with bounded selected batches.
Result solve_deterministic_energy(const Inputs& inputs,
                                  const std::vector<arma::uword>& targets,
                                  const Controls& controls, bool with_grad);

void bind_matrices(pybind11::module_& module);

} // namespace dlno_stc
} // namespace pyscfad
