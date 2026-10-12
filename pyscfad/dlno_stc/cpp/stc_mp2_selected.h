#pragma once
#include "stc_mp2_kernel.h"

namespace pyscfad { namespace dlno_stc {
// Selected labels are fixed input-frame columns, with every occupied partner.
struct SelectedControls {
    Controls numerical;
    bool has_global_seed = false;
};
// The binding owns inputs exclusively; move its arrays into the source kernel.
Result solve_selected(Inputs&, const std::vector<arma::uword>&,
                      const SelectedControls&, bool with_grad);
}}  // namespace pyscfad::dlno_stc
