// Thin standalone array adapter for the supplied selected energy/D/W kernels.
#include "stc_mp2_matrices.h"
#include "native/stc_mp2_deterministic_energy.h"
#include "native/stc_mp2_deterministic_d_w.h"
#include "native/stc_mp2_selected_energy_density.h"
#include "native/stc_mp2_stochastic_w.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <set>

namespace libgmbpt {
namespace qchem {

stc_mp2_kernel::stc_mp2_kernel(
        const arma::mat& fo, const arma::mat& fv, const arma::mat& b,
        const arma::Col<size_t>& offsets, unsigned seed, double energy_domain_tau)
    : nocc_(fo.n_rows), nvirt_(fv.n_rows), naux_(b.n_rows),
      natom_(offsets.n_elem - 1), offset_aux_(offsets), occupied_fock_(fo),
      virtual_fock_(fv), B_ia_(b), selected_domain_tau_(energy_domain_tau),
      explicit_seed_(seed) {}

} // namespace qchem
} // namespace libgmbpt

namespace pyscfad {
namespace dlno_stc {

namespace py = pybind11;
namespace source = libgmbpt::qchem::stc_mp2;
namespace density = source::density;
using NativeKernel = libgmbpt::qchem::stc_mp2_kernel;

namespace {

constexpr const char* matrix_kernel_version =
    "src_code_rdm_v2:427459d9b91fc70f:standalone-1";

// OpenMP ICVs belong to this calling thread. Forward and replay use the saved
// explicit count; the embedding process's previous settings are restored.
struct ThreadScope {
    int previous = omp_get_max_threads();
    int dynamic = omp_get_dynamic();

    explicit ThreadScope(int threads) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
    }

    ~ThreadScope() {
        omp_set_num_threads(previous);
        omp_set_dynamic(dynamic);
    }
};

unsigned native_seed(std::uint64_t seed) {
    return unsigned(seed) ^ unsigned(seed >> 32);
}

arma::Col<size_t> offsets_array(const Inputs& inputs) {
    arma::Col<size_t> offsets(inputs.aux_offsets.size());
    for (size_t j = 0; j < inputs.aux_offsets.size(); ++j) {
        offsets(j) = inputs.aux_offsets[j];
    }
    return offsets;
}

void positive_gap(const Inputs& inputs) {
    if (!inputs.foo.n_rows || !inputs.fvv.n_rows || !inputs.B.n_rows) {
        return;
    }
    arma::vec occupied, virtuals;
    if (!arma::eig_sym(occupied, inputs.foo) ||
        !arma::eig_sym(virtuals, inputs.fvv) ||
        !(virtuals.min() > occupied.max())) {
        throw std::invalid_argument(
            "STC-MP2 requires a positive occupied-to-virtual gap");
    }
}

py::array checked_array(py::handle value, const char* name, int rank) {
    if (!py::isinstance<py::array>(value)) {
        throw py::type_error(std::string(name) + " must be a NumPy array");
    }
    auto array = py::reinterpret_borrow<py::array>(value);
    if (!array.dtype().is(py::dtype::of<double>())) {
        throw py::type_error(std::string(name) + " must have float64 dtype");
    }
    if (array.ndim() != rank || !(array.flags() & py::array::c_style)) {
        throw py::value_error(
            std::string(name) + " has incorrect rank or is not C-contiguous");
    }
    const auto* data = static_cast<const double*>(array.data());
    for (py::ssize_t j = 0; j < array.size(); ++j) {
        if (!std::isfinite(data[j])) {
            throw py::value_error(std::string(name) + " must contain finite values");
        }
    }
    return array;
}

arma::mat from_array(const py::array& array) {
    arma::mat matrix(array.shape(0), array.shape(1));
    const auto* data = static_cast<const double*>(array.data());
    for (arma::uword i = 0; i < matrix.n_rows; ++i) {
        for (arma::uword j = 0; j < matrix.n_cols; ++j) {
            matrix(i, j) = data[i * matrix.n_cols + j];
        }
    }
    return matrix;
}

py::array_t<double> as_array(const arma::mat& matrix) {
    py::array_t<double> array({py::ssize_t(matrix.n_rows), py::ssize_t(matrix.n_cols)});
    auto* data = array.mutable_data();
    for (arma::uword i = 0; i < matrix.n_rows; ++i) {
        for (arma::uword j = 0; j < matrix.n_cols; ++j) {
            data[i * matrix.n_cols + j] = matrix(i, j);
        }
    }
    return array;
}

py::array_t<double> as_df_array(const arma::mat& matrix, size_t no, size_t nv) {
    py::array_t<double> array({py::ssize_t(matrix.n_rows), py::ssize_t(no),
                               py::ssize_t(nv)});
    auto* data = array.mutable_data();
    for (arma::uword p = 0; p < matrix.n_rows; ++p) {
        for (size_t i = 0; i < no; ++i) {
            for (size_t a = 0; a < nv; ++a) {
                data[(p * no + i) * nv + a] = matrix(p, i * nv + a);
            }
        }
    }
    return array;
}

Inputs input_arrays(py::handle foo, py::handle fvv, py::handle b) {
    const auto occupied = checked_array(foo, "foo", 2);
    const auto virtuals = checked_array(fvv, "fvv", 2);
    const auto integrals = checked_array(b, "B", 3);
    const auto no = occupied.shape(0);
    const auto nv = virtuals.shape(0);
    const auto np = integrals.shape(0);
    if (!no || !nv || !np || occupied.shape(1) != no || virtuals.shape(1) != nv ||
        integrals.shape(1) != no || integrals.shape(2) != nv) {
        throw py::value_error("inconsistent or empty Fock/B shapes");
    }
    Inputs inputs;
    inputs.foo = from_array(occupied);
    inputs.fvv = from_array(virtuals);
    inputs.B.set_size(np, no * nv);
    if (!arma::approx_equal(inputs.foo, inputs.foo.t(), "both", 1e-12, 1e-12) ||
        !arma::approx_equal(inputs.fvv, inputs.fvv.t(), "both", 1e-12, 1e-12)) {
        throw py::value_error("foo and fvv must be symmetric");
    }
    const auto* data = static_cast<const double*>(integrals.data());
    for (py::ssize_t p = 0; p < np; ++p) {
        for (py::ssize_t i = 0; i < no; ++i) {
            for (py::ssize_t a = 0; a < nv; ++a) {
                inputs.B(p, i * nv + a) = data[(p * no + i) * nv + a];
            }
        }
    }
    inputs.aux_offsets = {0, arma::uword(np)};
    return inputs;
}

void read_offsets(Inputs& inputs, py::object offsets) {
    if (offsets.is_none()) {
        return;
    }
    const auto array = py::array::ensure(offsets);
    if (!array || array.ndim() != 1 ||
        (array.dtype().kind() != 'i' && array.dtype().kind() != 'u')) {
        throw py::value_error("aux_offsets must be a one-dimensional integer array");
    }
    const auto values = py::cast<std::vector<std::int64_t>>(array);
    if (values.size() < 2 || values.front() != 0 ||
        values.back() != std::int64_t(inputs.B.n_rows)) {
        throw py::value_error("aux_offsets must span [0,naux]");
    }
    inputs.aux_offsets.clear();
    for (size_t j = 0; j < values.size(); ++j) {
        if (values[j] < 0 || (j && values[j] < values[j - 1])) {
            throw py::value_error("aux_offsets must be nondecreasing");
        }
        inputs.aux_offsets.push_back(arma::uword(values[j]));
    }
}

std::vector<size_t> read_targets(py::handle target_indices, size_t no) {
    if (!py::isinstance<py::tuple>(target_indices)) {
        throw py::type_error("target_indices must be a nonempty tuple of integers");
    }
    std::vector<size_t> targets;
    std::set<size_t> seen;
    for (auto item : py::reinterpret_borrow<py::tuple>(target_indices)) {
        if (py::isinstance<py::bool_>(item) || !PyIndex_Check(item.ptr()) ||
            py::isinstance(item, py::module_::import("numpy").attr("bool_"))) {
            throw py::type_error(
                "target_indices must contain integers excluding booleans");
        }
        auto index = py::reinterpret_steal<py::object>(PyNumber_Index(item.ptr()));
        if (!index) {
            throw py::error_already_set();
        }
        const auto value = py::cast<std::int64_t>(index);
        if (value < 0 || size_t(value) >= no || !seen.insert(size_t(value)).second) {
            throw py::value_error("invalid or duplicate target index");
        }
        targets.push_back(size_t(value));
    }
    if (targets.empty()) {
        throw py::value_error("target_indices must be nonempty");
    }
    return targets;
}

size_t integer_control(const py::dict& controls, const char* key, size_t fallback,
                       size_t minimum = 1) {
    if (!controls.contains(key)) {
        return fallback;
    }
    const auto item = controls[key];
    if (py::isinstance<py::bool_>(item) || !PyIndex_Check(item.ptr())) {
        throw py::type_error(std::string(key) + " must be an integer");
    }
    auto index = py::reinterpret_steal<py::object>(PyNumber_Index(item.ptr()));
    if (!index) {
        throw py::error_already_set();
    }
    const auto value = py::cast<std::uint64_t>(index);
    if (value < minimum || value > std::numeric_limits<size_t>::max()) {
        throw py::value_error(std::string(key) + " is out of range");
    }
    return size_t(value);
}

double real_control(const py::dict& controls, const char* key, double fallback,
                    bool allow_zero = false) {
    const double value = controls.contains(key) ? py::cast<double>(controls[key]) : fallback;
    if (!std::isfinite(value) || (allow_zero ? value < 0 : value <= 0)) {
        throw py::value_error(std::string(key) + " is out of range");
    }
    return value;
}

// Owned record, independent of Python arrays. No registry or retained B tensor.
struct MatrixReplay {
    bool stochastic = false;
    std::string kind;
    std::string native_version = matrix_kernel_version;
    std::uint64_t fingerprint = 0;
    std::uint64_t seed = 0;
    int threads = 1;
    double energy_tau = .0065;
    arma::vec beta, weights;
    std::vector<size_t> targets;
    std::vector<arma::uword> offsets;
    density::options occupied_options, virtual_options;
    double energy_target = 3e-4;
    source::selected_density_replay_plan d_plan;
    density::stochastic_w_replay_plan w_plan;
};

std::uint64_t fingerprint(const Inputs& inputs) {
    std::uint64_t value = 1469598103934665603ULL;
    auto add = [&](const void* data, size_t bytes) {
        const auto* characters = static_cast<const unsigned char*>(data);
        for (size_t j = 0; j < bytes; ++j) {
            value ^= characters[j];
            value *= 1099511628211ULL;
        }
    };
    for (const arma::mat* matrix : {&inputs.foo, &inputs.fvv, &inputs.B}) {
        add(&matrix->n_rows, sizeof(matrix->n_rows));
        add(&matrix->n_cols, sizeof(matrix->n_cols));
        add(matrix->memptr(), matrix->n_elem * sizeof(double));
    }
    add(inputs.aux_offsets.data(), inputs.aux_offsets.size() * sizeof(arma::uword));
    return value;
}

void check_control_keys(const py::dict& controls, const MatrixReplay& replay) {
    std::set<std::string> allowed = {
        "mode", "laplace_roots", "laplace_weights", "global_seed", "threads"};
    if (replay.stochastic) {
        allowed.insert({"matrix_tolerance", "domain_tau", "virtual_domain_tau",
                        "pair_pilot_samples", "min_samples"});
        if (replay.kind == "D") {
            allowed.insert({"energy_tolerance", "system_workload_cutoff", "resident_points"});
        }
    }
    for (auto item : controls) {
        const auto key = py::cast<std::string>(item.first);
        if (!allowed.count(key)) {
            throw py::value_error("unsupported native matrix control: " + key);
        }
    }
}

void read_grid(const py::dict& controls, MatrixReplay& replay) {
    if (!controls.contains("laplace_roots") || !controls.contains("laplace_weights")) {
        throw py::value_error("explicit laplace_roots and laplace_weights are required");
    }
    replay.beta = arma::vec(py::cast<std::vector<double>>(controls["laplace_roots"]));
    replay.weights = arma::vec(py::cast<std::vector<double>>(controls["laplace_weights"]));
    if (replay.beta.is_empty() || replay.beta.n_elem != replay.weights.n_elem ||
        !replay.beta.is_finite() || !replay.weights.is_finite() || replay.beta.min() < 0 ||
        (replay.stochastic && replay.weights.min() <= 0)) {
        throw py::value_error("invalid Laplace grid");
    }
}

struct MatrixSettings {
    std::shared_ptr<MatrixReplay> replay;
    density::options occupied, virtuals;
    double energy_target = 3e-4;
};

MatrixSettings read_settings(const Inputs& inputs, py::handle targets,
                             const py::dict& controls, const std::string& kind) {
    MatrixSettings settings;
    settings.replay = std::make_shared<MatrixReplay>();
    auto& replay = *settings.replay;
    replay.kind = kind;
    replay.targets = read_targets(targets, inputs.foo.n_rows);
    replay.offsets = inputs.aux_offsets;
    const auto mode = controls.contains("mode")
        ? py::cast<std::string>(controls["mode"]) : "deterministic";
    if (mode != "deterministic" && mode != "stochastic") {
        throw py::value_error("invalid mode");
    }
    replay.stochastic = mode == "stochastic";
    check_control_keys(controls, replay);
    read_grid(controls, replay);
    const auto threads = integer_control(controls, "threads", 1);
    if (threads > size_t(std::numeric_limits<int>::max())) {
        throw py::value_error("threads is too large");
    }
    replay.threads = int(threads);
    replay.seed = integer_control(controls, "global_seed", 0, 0);
    replay.fingerprint = fingerprint(inputs);
    if (!replay.stochastic) {
        return settings;
    }
    auto& occupied = settings.occupied;
    occupied.domain_tau = real_control(controls, "domain_tau", .003, true);
    occupied.frobenius_se_target = real_control(controls, "matrix_tolerance", 1e-4);
    occupied.pair_pilot_samples = integer_control(controls, "pair_pilot_samples", 100000, 2);
    occupied.min_samples = integer_control(controls, "min_samples", 64, 2);
    occupied.resident_points = integer_control(controls, "resident_points", 3);
    occupied.seed = native_seed(replay.seed);
    occupied.threads = replay.threads;
    if (occupied.resident_points != 2 && occupied.resident_points != 3) {
        throw py::value_error("resident_points must be 2 or 3");
    }
    settings.virtuals = occupied;
    settings.virtuals.domain_tau = real_control(controls, "virtual_domain_tau", .0003, true);
    if (kind == "W") {
        settings.virtuals.w_virtual_reform = true;
    }
    if (kind == "D") {
        settings.energy_target = real_control(controls, "energy_tolerance", 3e-4);
        replay.energy_tau = real_control(controls, "system_workload_cutoff", .0065);
    }
    replay.occupied_options = settings.occupied;
    replay.virtual_options = settings.virtuals;
    replay.energy_target = settings.energy_target;
    return settings;
}

struct MatrixOutput {
    double energy = 0, energy_se = 0;
    double occupied_frobenius_se = 0, virtual_frobenius_se = 0;
    bool occupied_target_met = true, virtual_target_met = true;
    size_t pilot_draws = 0, production_draws = 0;
    arma::mat occupied, virtuals, occupied_se, virtual_se;
};

MatrixOutput deterministic_forward(const Inputs& inputs, const MatrixReplay& replay) {
    const bool density_output = replay.kind == "D";
    auto result = density_output ? density::selected_deterministic_mp2_forward(
        inputs.foo, inputs.fvv, inputs.B, replay.beta, replay.weights,
        replay.targets, replay.threads) : density::selected_deterministic_mp2_w_forward(
        inputs.foo, inputs.fvv, inputs.B, replay.beta, replay.weights,
        replay.targets, replay.threads);
    MatrixOutput output;
    output.energy = result.energy;
    output.occupied = density_output
        ? std::move(result.occupied) : std::move(result.energy_occupied);
    output.virtuals = density_output
        ? std::move(result.virtuals) : std::move(result.energy_virtuals);
    output.occupied_se.zeros(output.occupied.n_rows, output.occupied.n_cols);
    output.virtual_se.zeros(output.virtuals.n_rows, output.virtuals.n_cols);
    return output;
}

MatrixOutput stochastic_density_forward(const Inputs& inputs, MatrixSettings& settings) {
    auto& replay = *settings.replay;
    NativeKernel kernel(inputs.foo, inputs.fvv, inputs.B, offsets_array(inputs),
                        native_seed(replay.seed), replay.energy_tau);
    auto forward = source::selected_mp2_forward(
        kernel, replay.beta, replay.weights, replay.targets, settings.energy_target,
        replay.occupied_options, replay.virtual_options);
    replay.d_plan = std::move(forward.replay);
    auto& result = forward.primal;
    auto& occupied = result.densities.occupied;
    auto& virtuals = result.densities.virtuals;
    MatrixOutput output;
    output.energy = result.energy;
    output.energy_se = std::sqrt(result.variance);
    output.occupied = std::move(occupied.matrix);
    output.virtuals = std::move(virtuals.matrix);
    output.occupied_se = std::move(occupied.standard_error);
    output.virtual_se = std::move(virtuals.standard_error);
    output.occupied_frobenius_se = occupied.frobenius_se;
    output.virtual_frobenius_se = virtuals.frobenius_se;
    output.occupied_target_met = occupied.target_met;
    output.virtual_target_met = virtuals.target_met;
    output.pilot_draws = occupied.pilot_draws + virtuals.pilot_draws;
    output.production_draws = occupied.production_draws + virtuals.production_draws
        + result.energy_production_samples;
    return output;
}

MatrixOutput stochastic_w_forward(const Inputs& inputs, MatrixSettings& settings) {
    auto& replay = *settings.replay;
    auto forward = density::selected_stochastic_w_forward(
        inputs.foo, inputs.fvv, inputs.B, offsets_array(inputs), replay.beta,
        replay.weights, replay.targets, replay.occupied_options, replay.virtual_options);
    replay.w_plan = std::move(forward.replay);
    auto& result = forward.primal;
    MatrixOutput output;
    output.energy = result.energy;
    output.energy_se = result.energy_se;
    output.occupied = std::move(result.occupied);
    output.virtuals = std::move(result.virtuals);
    output.occupied_se = std::move(result.occupied_se);
    output.virtual_se = std::move(result.virtuals_se);
    output.occupied_frobenius_se = result.occupied_frobenius_se;
    output.virtual_frobenius_se = result.virtual_frobenius_se;
    output.occupied_target_met = result.occupied_target_met;
    output.virtual_target_met = result.virtual_target_met;
    output.pilot_draws = result.pilot_draws;
    output.production_draws = result.production_draws;
    return output;
}

void finite_outputs(double energy, const arma::mat& occupied, const arma::mat& virtuals) {
    if (!std::isfinite(energy) || !occupied.is_finite() || !virtuals.is_finite()) {
        throw std::runtime_error("nonfinite native matrix result");
    }
}

// Report every effective source option, including defaults used to rebuild
// proposals during replay. The standalone boundary exposes only supported controls.
py::dict option_values(const density::options& options) {
    py::dict values;
    values["domain_tau"] = options.domain_tau;
    values["frobenius_se_target"] = options.frobenius_se_target;
    values["atom_block_l2"] = options.atom_block_l2;
    values["w_direct_joint_x_strong_h"] = options.w_direct_joint_x_strong_h;
    values["w_direct_atom_pair"] = options.w_direct_atom_pair;
    values["w_occupied_stratum_mixture"] = options.w_occupied_stratum_mixture;
    values["w_exchange_atom_pair"] = options.w_exchange_atom_pair;
    values["w_virtual_reform"] = options.w_virtual_reform;
    values["w_virtual_split"] = options.w_virtual_split;
    values["w_parallel_update_reduction"] = options.w_parallel_update_reduction;
    values["w_column_major_updates"] = options.w_column_major_updates;
    values["w_virtual_exchange_variants"] = options.w_virtual_exchange_variants;
    values["w_virtual_direct_variants"] = options.w_virtual_direct_variants;
    values["pair_pilot_samples"] = options.pair_pilot_samples;
    values["min_samples"] = options.min_samples;
    values["replay_replicas"] = options.replay_replicas;
    values["seed"] = options.seed;
    values["threads"] = options.threads;
    values["resident_points"] = options.resident_points;
    values["symmetric_pairs"] = options.symmetric_pairs;
    return values;
}

py::dict matrix_result(const MatrixOutput& output,
                       const std::shared_ptr<MatrixReplay>& replay) {
    finite_outputs(output.energy, output.occupied, output.virtuals);
    py::dict diagnostics;
    diagnostics["mode"] = replay->stochastic ? "stochastic" : "deterministic";
    diagnostics["matrix_kind"] = replay->kind;
    diagnostics["target_indices"] = replay->targets;
    diagnostics["global_seed"] = replay->seed;
    diagnostics["native_seed"] = native_seed(replay->seed);
    diagnostics["threads"] = replay->threads;
    diagnostics["native_version"] = replay->native_version;
    py::dict effective_options;
    effective_options["occupied"] = replay->stochastic
        ? option_values(replay->occupied_options) : py::dict();
    effective_options["virtual"] = replay->stochastic
        ? option_values(replay->virtual_options) : py::dict();
    if (replay->stochastic && replay->kind == "D") {
        effective_options["energy_tolerance"] = replay->energy_target;
        effective_options["system_workload_cutoff"] = replay->energy_tau;
    }
    diagnostics["effective_options"] = effective_options;
    diagnostics["energy_standard_error"] = output.energy_se;
    diagnostics["occupied_standard_error"] = as_array(output.occupied_se);
    diagnostics["virtual_standard_error"] = as_array(output.virtual_se);
    diagnostics["occupied_frobenius_se"] = output.occupied_frobenius_se;
    diagnostics["virtual_frobenius_se"] = output.virtual_frobenius_se;
    diagnostics["occupied_target_met"] = output.occupied_target_met;
    diagnostics["virtual_target_met"] = output.virtual_target_met;
    diagnostics["pilot_draws"] = output.pilot_draws;
    diagnostics["production_draws"] = output.production_draws;
    py::dict result;
    result["energy"] = output.energy;
    result["occupied"] = as_array(output.occupied);
    result["virtual"] = as_array(output.virtuals);
    result["replay"] = replay;
    result["diagnostics"] = diagnostics;
    return result;
}

py::dict solve_matrices(py::handle foo, py::handle fvv, py::handle b,
                        py::handle targets, const py::dict& controls,
                        const std::string& kind, py::object offsets) {
    if (kind != "D" && kind != "W") {
        throw py::value_error("matrix_kind must be D or W");
    }
    auto inputs = input_arrays(foo, fvv, b);
    read_offsets(inputs, offsets);
    positive_gap(inputs);
    auto settings = read_settings(inputs, targets, controls, kind);
    MatrixOutput output;
    {
        py::gil_scoped_release release;
        ThreadScope team(settings.replay->threads);
        if (!settings.replay->stochastic) {
            output = deterministic_forward(inputs, *settings.replay);
        } else if (kind == "D") {
            output = stochastic_density_forward(inputs, settings);
        } else {
            output = stochastic_w_forward(inputs, settings);
        }
    }
    return matrix_result(output, settings.replay);
}

Result replay_pullback(const Inputs& inputs, const MatrixReplay& replay,
                       double energy_bar, const arma::mat& occupied_bar,
                       const arma::mat& virtual_bar) {
    Result bars;
    if (!replay.stochastic) {
        auto result = replay.kind == "D" ? density::selected_deterministic_mp2_vjp(
            inputs.foo, inputs.fvv, inputs.B, replay.beta, replay.weights,
            replay.targets, replay.threads, energy_bar, occupied_bar, virtual_bar)
            : density::selected_deterministic_mp2_w_vjp(
            inputs.foo, inputs.fvv, inputs.B, replay.beta, replay.weights,
            replay.targets, replay.threads, energy_bar, occupied_bar, virtual_bar);
        bars.foo = std::move(result.bar_fo);
        bars.fvv = std::move(result.bar_fv);
        bars.B = std::move(result.bar_b);
    } else if (replay.kind == "D") {
        // Tau and seed affect proposal reconstruction. Counts and production
        // seeds come from d_plan; restore the forward settings as well.
        NativeKernel kernel(inputs.foo, inputs.fvv, inputs.B, offsets_array(inputs),
                            native_seed(replay.seed), replay.energy_tau);
        auto result = source::selected_mp2_vjp(
            kernel, replay.d_plan, energy_bar, occupied_bar, virtual_bar);
        bars.foo = std::move(result.bar_occ);
        bars.fvv = std::move(result.bar_virt);
        bars.B = std::move(result.bar_b);
    } else {
        auto result = density::selected_stochastic_w_vjp(
            inputs.foo, inputs.fvv, inputs.B, offsets_array(inputs), replay.w_plan,
            energy_bar, occupied_bar, virtual_bar);
        bars.foo = std::move(result.bar_fo);
        bars.fvv = std::move(result.bar_fv);
        bars.B = std::move(result.bar_b);
    }
    return bars;
}

py::dict pullback_matrices(py::handle foo, py::handle fvv, py::handle b,
                           const MatrixReplay& replay, double energy_bar,
                           py::handle occupied_bar, py::handle virtual_bar) {
    if (replay.native_version != matrix_kernel_version) {
        throw py::value_error("matrix replay native version has changed");
    }
    auto inputs = input_arrays(foo, fvv, b);
    inputs.aux_offsets = replay.offsets;
    if (fingerprint(inputs) != replay.fingerprint) {
        throw py::value_error("matrix replay inputs have changed");
    }
    const auto occupied = checked_array(occupied_bar, "occupied_bar", 2);
    const auto virtuals = checked_array(virtual_bar, "virtual_bar", 2);
    if (occupied.shape(0) != py::ssize_t(inputs.foo.n_rows) ||
        occupied.shape(1) != occupied.shape(0) ||
        virtuals.shape(0) != py::ssize_t(inputs.fvv.n_rows) ||
        virtuals.shape(1) != virtuals.shape(0) || !std::isfinite(energy_bar)) {
        throw py::value_error("invalid matrix output cotangents");
    }
    const auto occupied_seed = from_array(occupied);
    const auto virtual_seed = from_array(virtuals);
    Result bars;
    {
        py::gil_scoped_release release;
        ThreadScope team(replay.threads);
        bars = replay_pullback(inputs, replay, energy_bar, occupied_seed, virtual_seed);
    }
    finite_outputs(0, bars.foo, bars.fvv);
    if (!bars.B.is_finite()) {
        throw std::runtime_error("nonfinite native B cotangent");
    }
    py::dict result;
    result["foo"] = as_array(bars.foo);
    result["fvv"] = as_array(bars.fvv);
    result["B"] = as_df_array(bars.B, inputs.foo.n_rows, inputs.fvv.n_rows);
    return result;
}

} // namespace

Result solve_deterministic_energy(const Inputs& inputs,
                                  const std::vector<arma::uword>& targets,
                                  const Controls& controls, bool with_grad) {
    Result output;
    const auto no = inputs.foo.n_rows;
    const auto nv = inputs.fvv.n_rows;
    const auto np = inputs.B.n_rows;
    if (with_grad) {
        output.foo.zeros(no, no);
        output.fvv.zeros(nv, nv);
        output.B.zeros(np, no * nv);
    }
    if (!nv || !np) {
        return output;
    }
    positive_gap(inputs);
    ThreadScope team(controls.threads > 0 ? controls.threads : omp_get_max_threads());
    NativeKernel kernel(inputs.foo, inputs.fvv, inputs.B, offsets_array(inputs));
    const arma::vec roots(controls.roots), weights(controls.weights);
    for (size_t first = 0; first < targets.size(); first += controls.target_batch_size) {
        const auto last = std::min(first + controls.target_batch_size, targets.size());
        std::vector<size_t> group(targets.begin() + first, targets.begin() + last);
        auto result = with_grad ? source::selected_deterministic_energy_vjp(
            kernel, roots, weights, group, omp_get_max_threads())
            : source::selected_deterministic_energy(
            kernel, roots, weights, group, omp_get_max_threads());
        output.energy += result.energy;
        if (with_grad) {
            output.foo += result.bar_occ;
            output.fvv += result.bar_virt;
            output.B += result.bar_b;
        }
    }
    if (with_grad && controls.canonical_fock) {
        output.foo = arma::diagmat(output.foo.diag());
        output.fvv = arma::diagmat(output.fvv.diag());
    }
    return output;
}

void bind_matrices(py::module_& module) {
    module.attr("matrix_kernel_version") = matrix_kernel_version;
    py::class_<MatrixReplay, std::shared_ptr<MatrixReplay>>(module, "MatrixReplay");
    module.def("solve_matrices", &solve_matrices,
               py::arg("foo"), py::arg("fvv"), py::arg("B"), py::arg("target_indices"),
               py::arg("controls"), py::arg("matrix_kind") = "D",
               py::arg("aux_offsets") = py::none());
    module.def("pullback_matrices", &pullback_matrices,
               py::arg("foo"), py::arg("fvv"), py::arg("B"), py::arg("forward_replay"),
               py::arg("energy_bar"), py::arg("occupied_bar"), py::arg("virtual_bar"));
}

} // namespace dlno_stc
} // namespace pyscfad
