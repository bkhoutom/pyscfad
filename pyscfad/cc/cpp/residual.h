// Copyright 2026 The PySCFAD Authors (Apache-2.0).
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
using ResidualArray = pybind11::array_t<double, pybind11::array::c_style>;
using ResidualGemm = void (*)(char*, char*, int*, int*, int*, double*, double*, int*,
                             double*, int*, double*, double*, int*);
using ResidualLadder = ResidualArray (*)(const ResidualArray&, const ResidualArray&, size_t);
struct LadderPanelPlan {
    size_t workers, a, c;
    size_t elements(size_t v) const { return workers*a*c*v*v; }
};
LadderPanelPlan ladder_panel_plan(size_t v, size_t workspace_bytes, size_t threads);
// Allocation-free numerical core. Caller owns buffers and releases the GIL.
void ladder_into(const double* pairs, const double* input, double* output,
                 size_t v, size_t rows, const LadderPanelPlan& plan,
                 double* panels, const size_t* indices);
void bind_residual(pybind11::module_&, ResidualGemm, ResidualLadder);
void bind_solver(pybind11::module_&);

void bind_lambda(pybind11::module_&);
void bind_integral_pullback(pybind11::module_&);
void bind_deterministic_w(pybind11::module_&);

void bind_native_triples(pybind11::module_&);
