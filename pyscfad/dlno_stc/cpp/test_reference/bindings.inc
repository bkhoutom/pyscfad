// Independent full-system oracle, compiled only with BUILD_TESTING.
py::dict solve_full_reference(py::handle foo, py::handle fvv, py::handle B,
               py::dict controls, py::object aux_offsets, bool with_grad) {
    check_energy_controls(controls);
    bool canonical_fock=false;
    if (controls.contains("canonical_fock")) {
        if (!py::isinstance<py::bool_>(controls["canonical_fock"]))
            throw py::type_error("canonical_fock must be a bool");
        canonical_fock=py::cast<bool>(controls["canonical_fock"]);
    }
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
    deterministic_controls(controls,settings);
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
    { py::gil_scoped_release release; out=stc::solve_full(x,settings,with_grad); }
    if (!std::isfinite(out.energy) || !std::isfinite(out.energy_standard_error))
        throw std::runtime_error("nonfinite STC-MP2 energy or uncertainty");
    if (with_grad && (!out.foo.is_finite() || !out.fvv.is_finite() || !out.B.is_finite()))
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
        if (row.pilot_samples>settings.pilot_samples)
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
        py::array_t<double> bar_B({np,no,nv});
        auto* target=bar_B.mutable_data();
        for (py::ssize_t P=0; P<np; ++P)
            for (py::ssize_t i=0; i<no; ++i)
                for (py::ssize_t a=0; a<nv; ++a) target[(P*no+i)*nv+a]=out.B(P,i*nv+a);
        bars["B"]=bar_B; result["cotangents"]=bars;
    }
    return result;
}
