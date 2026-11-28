# Domain MP2 and DLNO-CCSD/CCSD(T)

This is the user guide for `pyscfad.dlno_stc`. It covers the four supported
workflows and their shared domain construction. For the mathematical details,
see [smooth LIS frames and Laplace triples](../../doc/source/user_guide/smooth_lis_triples.md).

The library supports four workflows:

| Calculation | Forward energy | Energy and nuclear gradient |
| --- | --- | --- |
| Domain MP2 | `dlno_stc.kernel` | `dlno_stc.value_and_grad` |
| DLNO-CCSD/CCSD(T) | `ccsd.kernel` | `ccsd.value_and_grad` |

Forces are `-gradient.coords`. Set `include_hf=True` on gradient calls or CC
energy calls for total molecular results. MP2 `kernel` returns correlation
energy; add `mf.e_tot` for total energy. Forward calls do not create response
tapes. Use `ccsd_t=True` to include triples.

For CC, W selection is the default; D selection is available with
`ccsd.build_static_selections(..., matrix_kind="D")`. Both use smooth anchored
LIS frames and the same fragment, CC correction and force workflow.
Direct Coulomb V selection is available with `matrix_kind="V"` and
`baseline_kind="global"`, using deterministic V and deterministic or stochastic global MP2.
Every solver mode defaults to deterministic. Stochastic MP2, D/W, CCSD and
triples remain available explicitly.

Deterministic MP2, D/W and MRCC triples use the `_ccsd_kernels` extension built with
`pyscfad/cc/cpp`: SciPy LAPACK supplies its symmetric eigensolves, and the
native contractions share CCSD's SciPy BLAS pointer and OpenMP runtime.
OpenMP owns kernel parallelism with serial BLAS inside workers; the requested
thread budget is scoped to the call and the caller's dynamic-team policy is
preserved throughout forward and reverse.
This covers scalar MP2, summed D/W, per-orbital W and the forward-only sequential W option.
The delayed reverse retains the full matrix-exponential derivative, including
degenerate eigenspaces, and returns cotangents in the original ED frame.
The deterministic DLNO-CCSD(T) route does not import `_stc_mp2`, Armadillo,
or `libcc_vjp`. Build only `pyscfad/cc/cpp` for this route. The optional
`dlno_stc/cpp` extension remains necessary for stochastic MP2, D/W and triples.
An older CCSD extension must be rebuilt; deterministic requests do not silently
fall back to Armadillo.

All library kernels preserve the caller's OpenMP dynamic-team policy,
including the Armadillo compatibility paths and SGX. `OMP_DYNAMIC=TRUE`
permits smaller teams; `FALSE` disables that adjustment. Neither choice is
overridden internally. Stochastic energy replay assigns its saved streams
to logical tasks so a change in actual team size does not change the samples
(the replay's configured thread budget must still match). BLAS thread limits
and OpenMP loop scheduling are separate from this policy.

## Fragments and domains

```python
from pyscfad import dlno_stc
from pyscfad.dlno_stc import ccsd

domains = dlno_stc.build_domain_selections(
    mf, frozen=1, boys_grouping="atom", full_domain=False,
    mp2_laplace_points=6,
)
```

`boys_grouping="atom_singleton"` uses the same domain construction, strong/weak
partition and MP2 baseline as `atom`, with one W pair, LIS and CCSD(T) correction
per original occupied target orbital. Set it in `build_domain_selections`;
the existing CC selection and energy/gradient APIs then use it automatically.
The saved `physical_groups` remain the atom partition used for domains; CC/LIS
records are indexed by individual orbital records.

With ordinary virtual selection (`sequential_virtual=False`), all per-orbital
W matrices are constructed before the group's CC solves. Deterministic W shares
factor propagation and amplitude construction across targets, bounded by
`target_batch_size`; no summed atom W is constructed. Stochastic W uses the
existing independent target streams and replay plans. Each target's W trace
contributes its MP2 energy once; the weak-domain correction is unchanged.

With `matrix_kind="D"`, each orbital uses the D provider and its explicit MP2
scalar. Domain preparation and integral geometry response remain shared.

Each CC/LIS reverse completes before the next solver. Its W cotangents are
retained until the group's combined MP2 reverse; coefficient and Fock responses
are accumulated immediately. Shared RI geometry adjoints are transformed into
one common domain-basis accumulator, so completed solvers do not retain separate
RI adjoints or CC tapes. This buffer scales as auxiliary size times the square
of the common domain rank, rather than as the number of LIS solves. Temporary
current-solver buffers are still required. The shared integral geometry response
and domain-preparation response are each applied once. Stochastic CC replica
precision retains its existing separate RI-response handling.

`boys_grouping="system"` puts all active occupied targets in one fragment,
uses full-system domains, and runs the same preparation, solver and response
pipeline. `full_domain=True` expands every fragment's
occupied, primary, PAO and extended domains to the entire system while
preserving physical fragment membership. Frozen orbitals remain excluded;
LNO cutoffs still apply.

`boys_grouping` accepts `singleton`, `atom`, `atom_singleton` and `system`; `molecule` remains a
compatibility alias for `system`. The separate `scope` keyword has been removed.
Pair screening and
`BoysDomainThresholds` control finite domains. Domain selection fixes discrete
labels; production reconstructs continuous orbitals and their responses.
`merge_identical=True` shares preparation among identical domains while each
physical fragment retains its own target projector and CC correction.

The parent PAO cutoff `pao_norm` applies to its squared overlap norm:
retain a raw PAO `p` when `p.T @ S @ p > pao_norm`. The default `1e-4`
preserves the physical cutoff of the previous `1e-2` norm default. Retained
PAOs are normalized by the square root of their squared norm. To reproduce
another old explicit norm cutoff, square its value. Saved selections and
checkpoints using the old convention must be rebuilt; restart algorithm ABI
6 rejects them.

## Calculation interface

For deterministic sequential W or D selection, pass `sequential_virtual=True` to
`ccsd.build_static_selections`. This selects the occupied LIS first, projects
the cached full-domain MP2 amplitudes and bare integral contractions onto that
occupied space, and constructs the virtual matrix there. D projects both amplitude legs;
W projects the amplitude and bare-integral legs. The physical target block is
preserved, and the domain MP2 energy and downstream CCSD/(T) treatment are
unchanged. The default remains independent occupied/virtual W selection.
The forward retains target amplitudes and integral slices only until occupied
selection, so it needs additional memory. Gradients replay those tensors once
and include the occupied-selection response through the occupied W/D matrix, the physical targets,
and the ED frame. Ranks and selection labels stay fixed during differentiation;
cutoff crossings require new selections. The reverse contracts the virtual
seed in the retained occupied space before a small selection VJP, then combines
its occupied-matrix seed with the direct CC response in the same native replay.
Native GEMMs use outer OpenMP teams and serial BLAS inside workers.
Stochastic matrix construction remains unsupported; this option requires
`matrix_kind="W"` or `"D"`.

Set `scale_lis_by_target_count=True` in `ccsd.build_static_selections` to
interpret both cutoffs per original physical target LMO. A four-target
fragment then uses four times each specified cutoff. The occupied cutoff
is scaled before sequential virtual W is built; domain MP2 energies remain
unscaled. This option is recorded in checkpoints and defaults to false.

This small deterministic example runs all four workflows. `build_static`
callbacks let force calculations share the differentiated SCF with screening
and selection. `build_mf` can run cold SCF or use a caller-supplied warm density.
It must return a converged restricted density-fitted reference.

```python
from pyscfad import gto, scf, dlno_stc
from pyscfad.dlno_stc import ccsd

mol = gto.Mole(atom="H 0 0 0; H 0 0 0.8; H 0 0 2.2; H 0 0 3.0",
               basis="sto-3g", verbose=0)
mol.build(trace_exp=False, trace_ctr_coeff=False)

def build_mf(current):
    mf = scf.RHF(current).density_fit()
    mf.conv_tol = 1e-11
    mf.chkfile = None
    mf.kernel()  # mf.kernel(dm0=...) can supply a warm starting density.
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf

def select_domains(mf):
    return dlno_stc.build_domain_selections(
        mf, frozen=0, boys_grouping="singleton", full_domain=True)

controls = {"mode": "deterministic", "threads": 1}

def select_cc(mf):
    return ccsd.build_static_selections(
        mf, select_domains(mf), thresh_occ=1e-4, thresh_vir=1e-5,
        baseline_controls=controls, matrix_controls=controls)

mf = build_mf(mol)
mp2_corr = dlno_stc.kernel(mf, select_domains(mf), controls=controls)
mp2_total = mf.e_tot + mp2_corr
mp2_total_grad, mp2_bar = dlno_stc.value_and_grad(
    mol, build_mf, build_static=select_domains, controls=controls, include_hf=True)

options = dict(baseline_controls=controls, matrix_controls=controls,
               ccsd_t=True, include_hf=True)
cc_total = ccsd.kernel(mf, select_cc(mf), **options)
cc_total_grad, cc_bar = ccsd.value_and_grad(
    mol, build_mf, build_static=select_cc, **options)
mp2_forces, cc_forces = -mp2_bar.coords, -cc_bar.coords
```

Set `ccsd_t=False` for CCSD. Energy is in Eh; coordinate gradients and forces
are in Eh/bohr. Use `value_and_grad` for derivatives; the forward-only `kernel`
interfaces are not outer JAX differentiation targets. Alternatively, pass
explicit `domains` or a CC plan as `static` instead of `build_static`.
The MP2 entry points also accept `boys_grouping="system", frozen=...` with no
`static` to select one system-wide fragment automatically. Explicit `singleton`, `atom` or `atom_singleton` grouping also builds the
corresponding selections automatically.
When supplying `static` or `build_static`, choose grouping, frozen orbitals and
localization options in that selection instead of passing them to the solver.

With the default `baseline_kind="domain"`, CC includes every physical fragment
and the complete domain MP2 baseline, including weak-pair completion:

`E_corr = E_MP2_domain + sum_f (E_CCSD,f + E_(T),f - E_MP2_LIS,f)`.

### Global MP2 baseline and direct Coulomb V selection

```python
domain_thresholds = dlno_stc.BoysDomainThresholds()
domains = dlno_stc.build_domain_selections(
    mf, frozen=1, boys_grouping="atom", thresholds=domain_thresholds)
plan = ccsd.build_static_selections(
    mf, domains, baseline_kind="global", matrix_kind="V",
    thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls={"mode": "deterministic", "threads": 8},
    matrix_controls={"mode": "deterministic", "threads": 8})
energy, parts = ccsd.kernel(
    mf, plan, ccsd_t=True, return_components=True,
    baseline_controls={"mode": "deterministic", "threads": 8},
    matrix_controls={"mode": "deterministic", "threads": 8})
```

The cutoffs above illustrate the API; V thresholds need their own accuracy
assessment. Define `domain_thresholds` with the existing `BoysDomainThresholds`.
Domain construction, pair-energy screening, grouping and LIS cutoffs are
unchanged. Pair energies define the strong-pair graph only; no weak-pair energy
or weak-pair response is added in this mode. The correlation energy is

`E_corr = E_MP2_global + sum_f (E_CCSD,f + E_(T),f - E_MP2_LIS,f)`.

Global MP2 covers the full active occupied/virtual space, honors frozen
orbitals and `correlation_auxbasis`, includes exchange, and uses the saved
`mp2_grid`. The existing `baseline_controls` chooses deterministic (default)
or stochastic evaluation. The scalar uses the existing Laplace kernels, so
quadrature error is controlled by `mp2_laplace_points` or an explicit grid.
Global factors are not compressed by `mp2_aux_svd_tol`. That option continues
to control local ED factors used for LIS construction.

To sample global MP2 while keeping V deterministic:

```python
baseline_controls = {
    "mode": "stochastic",
    "energy_tolerance": 3e-4,  # target global MP2 sampling SE, in Eh
    "global_seed": 19,
}
plan = ccsd.build_static_selections(
    mf, domains, baseline_kind="global", matrix_kind="V",
    thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls=baseline_controls)
energy, parts = ccsd.kernel(
    mf, plan, ccsd_t=True, baseline_controls=baseline_controls,
    return_components=True)
mp2_sampling_se = parts.baseline_standard_error
```

The optional stochastic `_stc_mp2` extension is required. One full-system
scalar solve estimates direct and exchange residuals across all active
occupied targets; the tolerance is not divided over local CC fragments.
Omitting both `energy_tolerance` and `production_samples` defaults to `3e-4` Eh.
For fixed sampling, pass `production_samples` instead of `energy_tolerance`.
Existing scalar controls such as `pilot_samples`, `min_production_samples`,
`max_production_samples`, `system_workload_cutoff` and `auxiliary_group_size`
remain available. A sample cap that prevents the requested allocation raises
an error. The measured SE is a statistical estimate, not a bound on quadrature,
LIS truncation or force error.

Stochastic scalar MP2 follows the existing OpenMP thread settings (for example
`OMP_NUM_THREADS`); its native interface does not accept `threads` or
`target_batch_size`. Matrix threads remain independent in `matrix_controls`.
Fixed-count runs reproduce the sample stream with the same seed, inputs and
configured OpenMP thread count. Adaptive allocation also uses pilot timings,
so production counts may vary between fresh runs. A completed baseline and
its variance are saved once in a serial progress checkpoint and restored
without resampling. MPI evaluates this one baseline on rank zero and reduces
its variance once. Sampled gradients use the scalar solver's product replay.

The existing stochastic-CC replica precision mode still requires deterministic
baseline, matrices and triples because its reported replica error excludes
their sampling uncertainty.

V uses bare real Coulomb integrals `g[i,j,a,b]=(ia|jb)`, summed over physical
target indices `i` in the domain input frame:

```
V_occ[j,k] = sum(i,a,b) g[i,j,a,b] * g[i,k,a,b]
V_vir[a,c] = 0.5 * sum(i,j,b) (
    g[i,j,a,b] * g[i,j,c,b] + g[i,j,b,a] * g[i,j,b,c])
```

There is no exchange contraction or additional spin prefactor in V. Both
matrices are positive semidefinite, with eigenvalues in Eh squared. The same
`thresh_occ`, `thresh_vir` and optional `scale_lis_by_target_count` apply;
zero cutoffs still retain the full ED. Density-fitting Gram contractions
give quartic dense scaling without four-index integrals or MP2 amplitudes.
V has no Laplace-grid dependence and currently accepts deterministic matrix
controls only. It returns matrices and their response, without an MP2 scalar;
fragment `matrix_energy_diagnostic` is `None`.

Global baselines also support D/W selection; their domain MP2 scalars are then
diagnostic only. `baseline_kind="domain"` remains the default for D/W, and V
requires `baseline_kind="global"`. Both choices are saved and checked on
restart. Serial checkpoints commit global MP2 once before local groups; MPI
evaluates it once on rank zero. `value_and_grad` includes global MP2 and V
response through the existing shared SCF response pipeline.

### D/W selection

For the domain baseline, W construction supplies the strong-domain MP2 energy through
`(trace(W_occ) + trace(W_vir))/2`. D instead evaluates the MP2 scalar explicitly
by contracting the selected amplitudes with the bare spin-adapted integrals;
the trace of D is not an energy. The D matrices use the historical positive
occupied/virtual selection-density convention. The provider shares its
amplitudes and quadrature between energy and density construction. One delayed
pullback combines the energy seed with the LIS-response cotangents before
returning to the shared preparation. The shared `mp2_grid=(roots, weights)`
supplies MP2 and either density. `return_components=True` returns fragment energies and uncertainty
and resource diagnostics.

D and W eigenvalues have different scales and meanings. `thresh_occ` and
`thresh_vir` apply directly to the chosen matrix; no automatic conversion is
made. With stochastic D, `matrix_controls` additionally accepts
`energy_tolerance` (default `3e-4` Eh), `system_workload_cutoff` (default
`0.0065`) and `resident_points` (`2` or `3`, default `3`). Energy and matrix
sampling use their respective error targets. In deterministic D, the scalar
is evaluated without sampling. As for W, `baseline_controls` does not launch
an additional strong-domain MP2 calculation.

The matrix choice is saved in selections and checked on restart. Saved
W selections from the current schema continue to load with W as the default;
requesting D for a W checkpoint is rejected. The benchmark runner exposes
the same choice as `--matrix-kind D` and defaults to W.

There is one pipeline and no `single_pass` option:

1. Run cold or warm SCF.
2. Screen weak pairs and group Boys orbitals into physical fragments; retain
   the weak-pair response when screening runs inside the force callback.
3. Prepare each fragment domain. Standalone MP2 computes the scalar energy
   directly, without a density. CCSD/(T) computes D or W once, selects its smooth LIS and
   solves the correction. Consume the pullbacks on that owner.
4. Accumulate the common-data response and solve the SCF response for forces.

For concrete real float64 inputs, weak-pair energies and their first reverse
pass use NumPy/SciPy with explicit adjoints. This includes primary PAO
selection/replay, overlap projections, normalization, semicanonicalization,
transition moments through octupoles, and pair contractions. Shared PAO
products and projections are reversed once after summing their endpoint
cotangents. AO-moment coordinate response uses the existing native shell
contraction, followed by a NumPy scatter to the parent atoms. No JAX reverse
tapes or shape-specialized weak algebra executables are created on this path.
SCF, Boys localization and the common-data producer retain their existing
response implementations.

Weak inputs must be concrete real float64 arrays. Traced weak inputs and
basis-parameter pullbacks are rejected; use the explicit coordinate VJP.
The host kernels use actual endpoint dimensions without JAX padding buckets.
Rank labels and eigengap regularization are unchanged.


The CC builder returns a plan, without computing a density. Its production D/W selects
ranks, anchors and the independent (T) grid, held fixed in that fragment's
energy and pullback. The LIS orbitals are constructed once while choosing
the anchors, then passed directly to CCSD/(T). Force calculations retain that
construction's pullback; they do not rebuild the LIS to obtain its response.
Solver replicas share these orbitals and apply the LIS pullback once to their
averaged cotangent. This reuse is local to the current fragment evaluation.
`correlation_auxbasis`
selects a separate correlation fitting basis; selection and execution must
use the same basis. HF/Fock continue using the SCF fit.

## Driver options

### SGX settings

Use `pyscfad.sgx.sgx_fit(mf, ...)` for integral-direct DF-J and SGX exchange
in SCF and its force response. The correlation DF object remains separate.
Apply the same SGX settings to cold and warm SCF. The
[SGX README](../../pyscfadlib/pyscfadlib/sgx/README.md#build-and-use)
covers native builds, backend selection, screening, mixed precision and
geometry response.

| Setting | Library default | Meaning |
| --- | --- | --- |
| `integral_backend` | `"auto"` | Prefer generated kernels when supported, with libcint fallback. `"generated"` requires the native kernels; `"libcint"` selects the reference engine. |
| `grid_level` | `2` | Numerical integration grid level (0–7); higher levels use denser grids. |
| `fit_ovlp` | `True` | Apply the overlap-fitting correction to reduce grid-integration error. |
| `screening` | `"per_pair"` | Static density-independent screening; `"adaptive"` distributes cumulative density-dependent budgets over tiles; `"per_pair"` tests each pair independently. |
| `mixed_precision` | `False` | FP64 throughout; `True` selects FP32/FP64 for forward, density-response and geometry kernels. |
| `energy_tol` | `"auto"` | Threshold for estimated exchange-energy contributions in density-dependent screening and FP32 classification. |
| `potential_tol` | `"auto"` | Threshold for estimated exchange-potential contributions; both energy and potential tests must pass to skip a contribution. |
| `fp32_multiplier` | `1e5` | Multiply the classification thresholds to select FP32 work. Larger values allow more FP32 work without changing which contributions are skipped. |
| `df_memory_mb` | `4000` MB | DF/J/K workspace budget, shared with the driver's DF setting; there is no separate SGX `max_memory` option. |
| `mf.direct_scf_tol` | `1e-13` | DF-J integral-screening threshold, set on the SCF object rather than passed to `sgx_fit`. Also sets the base SGX integral tolerance. |

**DF-J screening** is enabled independently of the SGX-K `screening` mode.
Set `mf.direct_scf_tol` on both cold and warm SCF objects before running SCF.
Forward J uses Schwarz shell bounds and density-dependent tests in both DF
passes. Density response uses a fixed Schwarz mask, with density-dependent
tests disabled so tiny or signed response vectors preserve linearity. Force
integrals use separate AO-derivative and auxiliary-derivative Schwarz bounds;
metric derivatives are evaluated in memory-bounded tiles. See the
[DF-J implementation](../sgx/_dfj.py).

`direct_scf_tol` is not the same threshold as SGX `energy_tol`: DF-J screens
integral bounds (and density-weighted bounds in the forward passes), while
SGX screens density- and grid-weighted estimates of exchange contributions.
The inspected local Q-Chem implementation also keeps them separate: RI-J uses
`10**(-THRESH)`, with ordinary SCF default `THRESH=SCF_CONVERGENCE+4` and
CC/EOM default `THRESH=14`. Thus SCF convergence exponent 9 gives `1e-13`,
while SNK independently defaults to energy/potential cutoffs `1e-10`/`1e-8`.
This supports `1e-13` as a conservative DF-J default for tight SCF/forces;
it does not establish that a looser cutoff would preserve a particular
energy or force accuracy. Q-Chem source locations are
`libgen/rem_setup.C` (integral defaults),
`libgscf/libgscf/qchem/fock/setup_fock_builder.C` (RI-J threshold), and
`libgscf/gen_scfman/get_scf_params.C` (SNK defaults).

The default `per_pair` screening requires generated SGX kernels (library
version 4 or later). Use `screening="static"` for the libcint reference path.

`"auto"` resolves according to the screening mode:

- **`per_pair`:** `energy_tol=1e-10` and `potential_tol=1e-8`, independently.
- **`adaptive`:** energy uses the SCF integral screening tolerance
  (`direct_scf_tol`, normally `1e-13`); potential uses the square root of the
  resolved energy tolerance (normally about `3.16e-7`). Thus explicitly setting
  `energy_tol=1e-10` while leaving potential automatic gives `potential_tol=1e-5`.
- **`static`:** with the default FP64 precision, these two tolerances are unused.
  With mixed precision enabled, the same automatic values as `adaptive` control
  FP32 classification, while density-dependent skipping stays disabled.

These are integral-screening/classification thresholds, not SCF convergence
criteria or guaranteed bounds on total molecular energy or force error.
Explicit numeric tolerances must be finite and positive.

The general benchmark runner defaults to integral-direct DF-J/SGX-K and exposes
its accuracy settings as `--sgx-*` flags. Use `--scf-jk df` to select conventional
DF SCF. It passes
`--df-memory-mb` to `sgx_fit(df_memory_mb=...)` for both cold and warm SCF.
The [bulk-water16 launcher](../../examples/dlno_stc/run_water_16_bulk_sgx.sh)
sets SGX explicitly. SGX precision and screening are independent of whether
the correlation solvers use deterministic or stochastic evaluation.

### Domain selection

`build_domain_selections(mf, ...)` accepts the following settings. The
[selection implementation](_selection.py) defines and validates their defaults.

| Keyword | Default | Meaning |
| --- | --- | --- |
| `frozen` | Required | Frozen-space specification; `0` retains all occupied orbitals. |
| `boys_grouping` | `"singleton"` | Domain/CC grouping: `singleton`, `atom`, `atom_singleton` or `system`. |
| `domain_link` | `False` | Enable occupied-pair linking and shared virtual support during Boys domain selection. |
| `full_domain` | `False` | Expand domains without combining fragments; system grouping enables it automatically. |
| `thresholds` | `BoysDomainThresholds()` | Domain completeness, ranks and weak-pair screening thresholds. |
| `lo_kwargs` | `None` | [Boys or lambda-Boys localization options](#boys-and-lambda-boys-localization). |
| `target_coeff` | `None` | Optional reference target coefficients. |
| `frag_lolist` | `None` | Optional complete singleton occupied-column partition; use `boys_grouping` for physical grouping. |
| `pair_energy_model` | `"multipole"` | Also `multipole-os`/`os-multipole`, `exact`, and `all`/`all-strong`. |
| `correlation_auxbasis` | `None` | Correlation fit override; defaults to the SCF fit. |
| `mp2_aux_svd_tol` | `1e-3` | ED OV auxiliary compression before MP2; `None` or zero disables it. |
| `mp2_laplace_points`, `mp2_grid` | `6`, `None` | Global fitted MP2 quadrature size or an explicit pair of arrays. |

With `domain_link=True`, two occupied Boys orbitals are connected when their compact
Boughton–Pulay atom lists share at least two atoms. In a connected component containing
a weak pair at the reference cutoff of `1e-5` Eh, internal pairs use one hundredth of
the requested pair cutoff. A target group containing a member also draws PAO centers
and extended AO support from the whole component. Other target groups use their usual
strong partners. Pass the option to `build_domain_selections`, or directly to
`kernel` or `value_and_grad` when using automatic `boys_grouping`.

`BoysDomainThresholds` defaults:

| Fields | Defaults | Role |
| --- | --- | --- |
| `bp_occ`, `bp_primary`, `bp_ed` | `0.985`, `0.999`, `0.9998` | Occupied BP completeness thresholds. |
| `bp_pao` | `0.98` | PAO BP completeness. |
| `pao_norm` | `1e-4` | Strict cutoff on the parent PAO squared overlap norm. |
| `domain_pao` | `1e-4` | Domain PAO screening. |
| `ed_pao`, `ed_pao_rank` | `0.995`, `1e-6` | ED PAO completeness and normalized local Gram rank cutoff. |
| `metric_rank` | `1e-10` | Occupied metric rank cutoff. |
| `pair_energy` | `5e-6` Eh | Strong-pair energy threshold. |
| `near_pair_distance`, `multipole_order` | `3.5`, `4` | Near-pair distance in Bohr and multipole order. |

### Boys and lambda-Boys localization

Ordinary Boys is the default. To enable lambda-Boys, pass `lo_kwargs` during
domain selection; the saved options apply to both energies and forces:

```python
domains = dlno_stc.build_domain_selections(
    mf, frozen=1, boys_grouping="atom",
    lo_kwargs={"method": "lambda_boys", "lambda_": 100.0},
)
```

[Lambda-Boys](../lo/lambda_boys.py) minimizes the sum of Boys spreads plus
`lambda_ * sum(i != j, |F_ij|**2)`, with F in the localized occupied basis.
Both ordered entries `F_ij` and `F_ji` count. `lambda_` defaults to `100`
in bohr²/Eh², must be finite and nonnegative, and `lambda_=0` recovers
ordinary Boys. Larger values penalize off-diagonal Fock elements more strongly.
The current SCF Fock is fixed during orbital rotations; its response is
included when computing forces.

Both methods accept `init_guess`, `conv_tol`, `conv_tol_grad`, `max_cycle`,
`max_saddle_escapes`, `symmetry` and `gmres_options` in `lo_kwargs`.
`gmres_options` controls the localization response solve (default `tol=1e-7`,
`atol=1e-7`); it is separate from the CCSD lambda-equation controls.

### LIS cutoffs

Set the occupied and virtual cutoffs explicitly on
`ccsd.build_static_selections(mf, domains, thresh_occ=..., thresh_vir=...)`.
These select the CC local interacting space (LIS) from D or W and apply to both
CCSD and CCSD(T), for energies and forces. Standalone domain MP2 does not use
these cutoffs or build selection matrices.

| Keyword | Public API default | Meaning |
| --- | --- | --- |
| `matrix_kind` | `"W"` | Use energy-weighted W or amplitude-density D for LIS selection. |
| `thresh_occ` | Required | Retain external occupied density eigenvectors with `abs(eigenvalue) > thresh_occ`; the internal target span is retained. |
| `thresh_vir` | Required | Retain virtual density eigenvectors with `abs(eigenvalue) > thresh_vir`. |

Both cutoffs are nonnegative: W thresholds are energies in Eh; D thresholds
are dimensionless. The quickstart and
general water-16 runner use `thresh_occ=1e-4`, `thresh_vir=1e-5`; the
[archived-run example](../../examples/dlno_stc/README_water_16.md#bulk-water16-from-the-completed-sgx-benchmark)
explicitly uses `1e-4`/`3e-5`. A zero cutoff retains
the full corresponding domain parent space, including zero-eigenvalue modes.
Smaller positive cutoffs retain more orbitals. `full_domain=True` expands the
parent domains but still applies these LIS cutoffs. Ranks and labels are
selected once from each fragment's density and held fixed during its pullback.

### CC plan and execution

`ccsd.build_static_selections(mf, domains, ...)` requires `thresh_occ` and
`thresh_vir`, the [LIS cutoffs](#lis-cutoffs) described above. Its other settings
are `matrix_kind="W"`, `baseline_controls=None`, `matrix_controls=None`, `mp2_grid=None`,
`internal_rank_threshold=1e-6`, `triples_laplace_points=4` and
`correlation_auxbasis=None`. `merge_identical=True` is accepted by this builder
for compatibility; actual preparation grouping occurs during execution.

| Execution keyword | Default | Applies to |
| --- | --- | --- |
| `controls` | Required; `{}` selects deterministic | Standalone MP2. |
| `baseline_controls`, `matrix_controls`, `mp2_grid` | `None` | CC; use the settings associated with the plan. |
| `ccsd_t`, `triples_controls` | `False`, `None` | CC; enable triples explicitly. |
| `include_hf` | `False` | Both gradient drivers and CC energy; MP2 energy always returns correlation energy. |
| `return_components` | `False` | CC; append fragment energies, dimensions, SEs and diagnostics. |
| `merge_identical` | `True` | Both methods; share identical-domain preparation. |
| `comm`, `parallel_scf_jk` | `None`, `True` | Both methods; MPI and shared DF J/K work. |
| `correlation_auxbasis` | `None` | Both methods; use the same fit for selection and execution. |
| `checkpoint_dir`, `resume` | `None`, `False` | Serial CC progress checkpoints. |
| `vvvv_direct`, `lis_aux_svd_tol` | `False`, `None` | CC virtual-ladder storage and LIS auxiliary compression (disabled by default). |
| `mp2_aux_svd_tol` | `1e-3` | ED OV auxiliary compression before MP2/W and LIS selection. |
| `cc_solver`, `stc_cc_controls` | `"deterministic"`, `None` | CC amplitude solver and optional stochastic controls. |
| `verbose_imp` | `0` | CC impurity solver verbosity. |

Gradient drivers accept either `static` or `build_static(mf)`. The molecule must
have coordinate derivatives enabled with
`mol.build(trace_exp=False, trace_ctr_coeff=False)`. With `return_components=True`,
CC returns `(energy, components)` or `(energy, mol_bar, components)`.

## Separate MP2 and triples quadratures

`build_domain_selections` automatically fits **one global MP2/density grid** from
all active canonical occupied/virtual energies, excluding frozen orbitals.
`mp2_laplace_points=6` sets its size. Every domain uses the same saved grid;
`boys_grouping="system"` follows the same rule. Nodes/weights are saved as
`domains.mp2_grid`, with reproducible fit diagnostics in
`domains.mp2_grid_diagnostics`. The fit accounts for both energy and derivative
accuracy and runs outside differentiation.

CC selection inherits this grid. Pass `mp2_grid=(roots, weights)` to domain
or CC selection for an explicit override; CC saves the override too, so it
need not be repeated in energy/force calls. A grid supplied at CC execution
must match the saved grid. `baseline_controls` and `matrix_controls` contain
execution/sampling settings only. Standalone MP2 uses its saved grid when
`controls` omits quadrature keys; explicit `laplace_roots`/`laplace_weights`
remain available as caller-owned overrides.

`triples_laplace_points=4` on CC selection independently fits a positive grid
for each nonempty LIS using its **triple-excitation** denominator range.
The saved `lis_static.triples_grid` holds these nodes, weights and fit diagnostics.
The global MP2 grid is saved before execution. Each (T) grid is saved in the
fragment-local selection after W and remains fixed during its response.
Independent runs select new LIS labels and (T) grids; fixed-label finite
differences must reuse labels captured from a reference run. The LIS MP2 subtraction
solves the noncanonical MP2 equation directly.

## Deterministic and stochastic controls

Empty control dictionaries select deterministic defaults. Explicit stochastic
choices are independent:

- MP2: `controls={"mode": "stochastic", ...}`; choose `energy_tolerance`
  (default `3e-4` Eh) or fixed `production_samples`. Standalone MP2 never builds W.
- W: `matrix_controls={"mode": "stochastic", ...}`; `matrix_tolerance` controls
  occupied/virtual matrix uncertainty. Its energy and uncertainty use the same
  W samples, including trace covariance.
- CCSD: `cc_solver="stc"`, with `stc_cc_controls`. This requires the optional
  external `stc_cc_jax` package. Deterministic runs do not import it.
- Triples: `triples_controls={"mode": "stochastic", "nsamples": 10000, ...}`;
  `energy_tolerance` enables adaptive allocation. The fragment pipeline uses
  `variance_allocation="occupied_count"` across all physical targets.

MP2 seeds depend on preparation-group identity; W and CC seeds depend on
physical fragment identity. Sampling streams do not depend on MPI rank. Saved
production samples are reused for response. Reported energy standard errors
do not bound force errors.

### Standalone MP2 controls

Pass `controls` explicitly, using `{}` for deterministic defaults. The high-level
stochastic default is an adaptive total strong-domain energy SE target of
`3e-4` Eh, divided in variance across the preparation jobs. Explicit
`production_samples` selects fixed-count sampling instead. Forces reuse the
scalar solver's production samples and input cotangents; no W is constructed.

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `"deterministic"` | Explicit `"stochastic"` enables sampling. |
| `global_seed` | `0` | Root seed for rank-independent group streams. |
| `energy_tolerance` | `3e-4` Eh in adaptive mode | Total strong-domain energy SE target. |
| `production_samples` | Omitted | Fixed production count; overrides adaptive allocation when supplied. |
| `pilot_samples` | `100000` | Independent pilot budget in adaptive mode. |
| `min_production_samples` | `10000` | Adaptive production minimum. |
| `max_production_samples` | `0` | Optional adaptive cap; zero means uncapped. |
| `laplace_roots`, `laplace_weights` | Saved global grid | Supply both to override standalone MP2 quadrature. |

Advanced scalar sampling and execution controls are defined in the
[native selected-solver parser](cpp/bindings.cpp). W sampling controls below
belong to a different solver and should not be used for standalone MP2.

### W controls for CC

Use `matrix_controls` for W. In the CC API, `baseline_controls` supplies
preparation settings; it does not launch an additional scalar MP2 solve.
Scientific controls must match those used to construct the CC plan. Pass
quadrature through `mp2_grid`, not either control dictionary.

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `"deterministic"` | Explicit `"stochastic"` enables W sampling. |
| `global_seed`, `threads` | `0`, `1` | Sampling root seed and native thread count. |
| `target_batch_size` | `None` | All targets in a physical fragment; a positive cap reduces batch storage. |
| `auxiliary_group_size` | `32` | Auxiliary grouping size. |
| `memory_limit_mb` | Omitted | Optional memory budget for staged W calls. |
| `matrix_tolerance` | `1e-4` | Stochastic occupied/virtual matrix precision target. |
| `domain_tau`, `virtual_domain_tau` | `0.003`, `0.0003` | Stochastic compact-domain cutoffs. |
| `pair_pilot_samples`, `min_samples` | `100000`, `64` | Stochastic pilot budget and minimum production allocation. |

The last three rows apply only to stochastic W. Its MP2 energy SE includes
occupied/virtual trace covariance. It is not an estimate of the total CCSD(T)
uncertainty caused by selecting a LIS from sampled W.

### Triples controls

`triples_controls=None` selects deterministic MRCC triples when `ccsd_t=True`.
Leave it `None` when triples are disabled. Accepted keys are:

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `"deterministic"` | `"stochastic"` samples the asymmetric selected-space partition. |
| `partition` | `"mrcc"` for deterministic, `"asymmetric"` for stochastic | These are the only supported mode/partition combinations. |
| `workspace_gib` | `16` | Deterministic (T) numerator-cache plus worker-scratch budget in GiB (1024³ bytes), further bounded by `max_memory` headroom. Finite and positive. |
| `nsamples` | Required for stochastic mode | Integer ≥ 2; nominal production budget per fragment in fixed-count mode. Still required with a tolerance. |
| `energy_tolerance` | Omitted or `0` | A positive value targets total triples energy SE in Eh; `5e-5` Eh is `0.05` mEh. |
| `pilot_samples` | `10000` | Independent discarded pilot draws per Laplace point per fragment. |
| `global_seed` | `71` | Root seed for fragment, point, pilot and production streams. |
| `trace_exponent` | `5.0` | Positive exponent for point-variance allocation in tolerance mode. |
| `variance_allocation` | `"occupied_count"` for fragment plans | Fragment variance shares follow original occupied target counts. |
| `fragment_energies` | Omitted | Only for `mp2_energy` allocation with previously captured concrete labels; incompatible with plan-based occupied-count allocation. |

Deterministic triples accept `mode`, `partition="mrcc"`, and `workspace_gib`.
For example, `triples_controls={"mode": "deterministic", "workspace_gib": 64}`
raises the default 16 GiB workspace allowance to 64 GiB when headroom permits.
This is an advisory workspace budget, not a total-process memory limit:
shared tensors and library allocations are additional, and the minimum one
worker/one block is retained even when it exceeds the budget. The existing
cache/worker allocation policy remains in effect, so a larger budget does not
guarantee use of every requested thread. `workspace_budget_bytes` reports the
largest effective native budget across quadrature points. Changing this
execution setting preserves compatible completed-domain checkpoints. Rebuild
the native extension to obtain `TRIPLES_CONFIGURABLE_WORKSPACE` support.
For energies and forces, triples automatically reuse available cached CCSD
integral blocks and construct only missing blocks from DF factors once per
fragment. There is no integral-transform setting.
The former asymmetric deterministic implementation has been removed. The MRCC
and asymmetric partitions can assign different energies to a selected fragment
when its occupied projector does not commute with the occupied Fock matrix.
Their complete-system sums agree.
For ordinary fragment plans, retain `variance_allocation="occupied_count"`;
energy-proxy allocation would require results before the fragment loop.
A positive tolerance enables adaptive production, with at least 10000 draws
per point. Without it, pilot variances allocate `nsamples` across points;
rounding can increase the total. Pilot draws are additional and do not enter
the energy or force. Quadrature error is separate from sampling SE.

```python
triples = dict(mode="stochastic", nsamples=10000, pilot_samples=10000,
               global_seed=71, energy_tolerance=5e-5,
               variance_allocation="occupied_count")
energy, bar, components = ccsd.value_and_grad(
    mol, build_mf, build_static=select_cc, triples_controls=triples,
    return_components=True, **options)
forces = -bar.coords
```

### CCSD solver and convergence

`cc_solver="deterministic"` is the default. `cc_solver="stc"` enables the
optional external `stc_cc_jax` solver and accepts `stc_cc_controls`.
Its [adapter](stc_ccsd.py) defines accepted keys, units and precision-replica
controls; deterministic calculations do not import that external package.

The deterministic fragment solver inherits RCCSD defaults `conv_tol=1e-7` Eh,
`conv_tol_normt=1e-5` and `max_cycle=50`. Set
`pyscfad.lno.ccsd.RCCSD.conv_tol`, `.conv_tol_normt` and `.max_cycle` before
calling the driver to override them. Both forward convergence tests must pass.
The default custom response uses the amplitude tolerance and iteration limit
for lambda too; a force request does not automatically tighten them.
`PYSCFAD_LNO_CCSD_DIIS_SPACE=3` is the impurity DIIS default; nonpositive values
disable it. These settings are independent of `triples_controls`.

Native CPU DF-CCSD is the deterministic fragment default. The optional C++
residual extension and the JAX reference implementation are described
under [build and verification](#build-and-verification) and in the
[implementation notes](../../doc/source/user_guide/smooth_lis_triples.md#deterministic-ccsd-and-response).

## Integral storage and memory

Set `PYSCFAD_LNO_SHARED_LOCAL_DF_CACHE_MODE` to:

- `half`: retain the current group's half-transformed `H_ED[P,mu,p]`. Obtain
  each LIS half transform by rotating H. Release LIS buffers after their
  correction and ED buffers after the W/MP2 response.
- `direct`: regenerate ordinary three-center integrals for each forward
  transform and coefficient pullback, reducing resident storage.

Gradients and CC energy default to `half`; MP2 energy defaults to `direct`.
For one ED and `k` LIS solves, direct CC energy requires `1+k` ordinary passes
and its gradient requires `2*(1+k)`. MP2 requires one ordinary forward pass or
two for a gradient. Half mode uses one ordinary pass per group. Gradients in
both modes additionally contract derivative integrals and metric response
once per group.

`vvvv_direct=True` separately controls the CC virtual-ladder integral cache.
`lis_aux_svd_tol` controls LIS auxiliary-space compression (default `None`,
disabled; a positive threshold enables it, and zero also disables it).
These memory/accuracy choices are independent of
H storage. Memory forecasts warn when budgets are exceeded; proven impossible
new allocations are rejected. No global AO-pair tensor is required by local
correlation preparation.

`mp2_aux_svd_tol` independently compresses the metric-whitened ED `(ia|P)`
factors before MP2/W (default `1e-3`, retaining eigenvalues strictly above
the cutoff). OV products carry no symmetric-pair weights. LIS auxiliary
compression still starts in the original auxiliary basis. Weak-pair screening
and completion use their existing factors. Set `mp2_aux_svd_tol=None` or zero
to reproduce the uncompressed ED calculation.

The MP2 NAF pullback includes the retained-subspace response at fixed rank,
including the omitted left null space for tall factors. It transforms the
summed MP2/W factor cotangent before the existing ED DF pullback. This adds
no AO integral pass: direct mode retains one ordinary three-center pass for
the ED coefficient reverse and one derivative contraction pass; half-cache
mode reuses the resident intermediates for the coefficient reverse. The shared
geometry and auxiliary-metric responses are each evaluated once per group.
Gradients are local derivatives of the truncated model, away from cutoff
crossings; they are not gradients of uncompressed MP2. CC progress checkpoints
include the MP2 NAF cutoff and reject incompatible resumed calculations.

## MPI and checkpoints

Pass `comm` to the high-level energy or gradient entry point. MPI distributes
complete domain groups; each owner prepares and differentiates its fragments
locally. `parallel_scf_jk=True` also distributes SCF/common DF J/K and their
responses. Set it false for root-owned SCF and shared response.

MPI gradient builders must accept `mo_coeff_init`, `mo_energy_init`,
`mo_occ_init` and `e_tot_init`, install these values and skip SCF on workers.
MPI energy calls require a matching MF skeleton on each rank. Attached
out-of-core SCF factors must be accessible on every rank.

Serial CC supports `checkpoint_dir` and `resume`. Checkpoints save completed
numerical contributions, selections, scientific settings and implementation
identity. Native replay handles and AD tapes are never serialized. CC
selection schema 19 records the fragment plan and requires all fragments; older selections must be rebuilt.

Resume reuses the saved Boys anchors, discrete domain/LIS selections and
quadratures before rebuilding shared response. Molecular inputs, occupations,
selection thresholds, solver controls and implementation versions remain exact
compatibility checks. New checkpoints also save numerical SCF and shared-frame
reference arrays in `records/cc-reference/state.h5`. SCF total/orbital energies
use an absolute `1e-8` Eh tolerance; coefficients and shared arrays use
`atol=rtol=1e-8`. Unconverged SCF, nonfinite values and larger orbital/frame
changes are rejected. Every resume compares against the original reference;
the tolerance does not accumulate across restarts. Degenerate orbital rotations
that exceed these bounds require a fresh calculation, not unchecked cotangent reuse.

Intermediate BP construction lists may differ when tied populations produce
the same effective fragment domains, retained ranks, pair graph and physical
grouping. Resume validates those final choices and restores the original saved
selections. This also supports a compatible SCF density-checkpoint restart;
SCF/SGX settings and the stricter numerical tolerances above still apply.
Memory/workspace limits may change. The integrated restart implementation keeps
the `cc-reference` format and the SCF-restoration/interruption-recovery behavior
described below. Checkpoints written in the separate `cc-scf-reference` branch
format require a new checkpoint directory; this merge does not convert them.

The `benchmark_lno_w_triples.py` runner restores the saved converged SCF on
`--resume`, bypassing both cold and warm SCF iterations. It rebuilds integrals
and the implicit SCF response at the saved density, preserving the original
orbital gauge and its derivatives. The AO Fock is read from `common.fock`
in the CC reference (or from the early SCF checkpoint) and reused as the
forward value, including during pullback setup. No forward J/K construction
at the saved density is needed. The DF/SGX density-response and geometry
derivative operations still execute when requested; a saved Fock does not
replace these derivatives. Early checkpoints created before Fock saving was
added cannot use this path and produce an explicit missing-Fock error.
The output records
`reference_scf_mode="restored"` and zero reference-SCF solve time. This replaces
the earlier warm-start behavior. Legacy checkpoints without numerical orbitals
produce an explicit error rather than silently starting cold.

When CC selections are present, the runner also loads the saved Boys anchors,
BP/domain choices, LIS plan and quadratures directly. Geometry, basis, frozen
space, grouping, thresholds, localization options and LIS settings are checked
against the request; SCF/DF/SGX, solver and implementation checks remain active.
It records `selection_source="checkpoint"` and zero selection-construction
times. Boys response is attached to the saved stationary frame without running
the localization optimizer. An insufficiently stationary frame is rejected;
it is not silently refined. Continuous integrals, PAO/domain projections and
response needed for unfinished work are still rebuilt.
API callers that explicitly reconverge SCF rather than restore its saved values
may still need continuous Boys refinement at that slightly different SCF root;
their saved discrete domain choices are reused.

The API exposes `ccsd.load_static_selections(checkpoint_dir, mol,
domain_options={...}, selection_options={...})`. Pass requested domain options
such as `frozen`, `thresholds`, `boys_grouping`, `lo_kwargs` and
`pair_energy_model`, and LIS options `thresh_occ`, `thresh_vir`,
`internal_rank_threshold` and `triples_laplace_points`, to validate them without
selection. Omitted options adopt the saved configuration. Serial
`ccsd.kernel(..., resume=True)` and `ccsd.value_and_grad(..., resume=True)`
automatically load it when neither explicit selections nor a selection callback
is supplied. An explicitly supplied callback still runs; use the loader and
pass its result as `static` to avoid rebuilding inside that callback.

If only the early SCF checkpoint exists, no domain choices have been committed
yet, so selection runs once before correlation can resume.

Immediately after a successful warm SCF, the runner atomically saves
`.scf-reference.h5` in the checkpoint directory. This independent reference
survives interruptions during domain selection or CC checkpoint initialization.
Once the CC checkpoint is complete, its `records/cc-reference/state.h5` is the
authoritative reference. Both paths check molecular inputs and SCF/SGX settings.
These serial RHF restore paths require standard implicit differentiation.

If interrupted before checkpoint initialization completes, `resume=True`
restarts preparation. Partial manifest/static metadata is preserved in the
checkpoint's hidden `.initialization-history/` directory. Once a progress
record exists it is never reset by this recovery: completed weak work and
domain groups are restored, and an unfinished group is recalculated. A crash
during an atomic progress write retains the preceding committed group boundary.

The final SCF adjoint now checkpoints lambda automatically whenever a serial
CC checkpoint is active. `records/scf-lambda-iterate/state.h5` holds the latest
GMRES restart-cycle iterate and RHS. A cycle contains at most the solver's
`restart` number of Arnoldi iterations (20 by default); saving does not change
that setting or the convergence tolerances. Restart supplies this lambda as
`x0`, rebuilding the Krylov basis. It does not resume the exact old Krylov state.
`records/scf-lambda-converged/state.h5` is committed separately, immediately
after convergence and before the external derivative contractions. On restart,
the current operator checks its true residual against the current RHS and
tolerance. A valid solution skips GMRES and preconditioner construction; a
solution that no longer meets tolerance becomes an initial guess for refinement.
Convergence failures never create a new converged record. JAX-only solver
configurations use host GMRES while this checkpoint scope is active, preserving
their tolerances and restart length. Higher-order traced VJPs are not checkpointed.

Force assembly checkpoints completed shared SGX Coulomb and exchange geometry
contractions separately (`records/scf-force-sgx-j/` and `scf-force-sgx-k/`). The
shared disk-backed DF thin-factor geometry contraction is checkpointed in
`records/scf-force-df-geometry/`. These small coordinate results are reused only
when their derivative seeds match exactly. Common-response setup and seed
assembly are reconstructed; an interrupted contraction stage repeats from its
start. Individual integral blocks and raw transformed-MO disk panels are not
checkpointed. All these records use the same checksums, atomic replacement and
exclusive writer lease as the fragment checkpoints.

Older checkpoints contain canonical hashes but no numerical reference. Exact
reference resumes still work. For a completed old run, its exact `hf_energy`
can be supplied using `_ccsd_restart_reference.recover_legacy_energy(directory,
energy)`; the value must match the original checkpoint's checksum. This limited
recovery additionally checks the saved active occupied space and Boys frame,
and rejects nonzero canonical/virtual-frame cotangents whose original gauge
cannot be verified. Keep a copy of the original checkpoint when recovering it.
Missing original energy or incompatible frames cannot be repaired by changing
the manifest. New runs need no such recovery step.

Disk request/result exchange, export/import/replay drivers, per-domain MPI
replica averaging and public fragment-subset/correction-only modes have been
removed. The old `force_full_domains` alias is removed; use `full_domain`.
The fixed-value `smooth` and `lis_space` arguments are removed.
`matrix_kind` is selectable again, with W as the default.
Use `triples_laplace_points` in place of `smooth_laplace_points`.
Restart checkpoints remain supported independently of disk solver exchange.

## Build and verification

Standalone deterministic MP2 uses the same native propagator as D/W, with
energy-only contractions rather than W output matrices. Forward plus input
cotangents share one amplitude construction. `target_batch_size` bounds the
selected-target amplitude storage (one target by default for scalar MP2).
The adapter owns a virtual-eigenbasis DF copy and may need a C-layout staging
copy for strided factors. Fock spectra and their full derivatives use SciPy
LAPACK and the native Frechet pullback. No JAX compilation cache is cleared.

The old `_stc_mp2.solve_selected` and `_solve_selected_borrowed` APIs remain
available for legacy tests and stochastic users. Their four-factor/pair-Gram
memory diagnostics describe that legacy implementation, not the new scalar
propagator. Process memory also includes host factors, SCF/localization state,
DF response, worker scratch and retained JAX executables.

For **stochastic** MP2, D/W or triples only, build the optional legacy extension
with PySCFAD, pybind11, Armadillo and OpenMP available:

```bash
cmake -S pyscfad/dlno_stc/cpp -B build/dlno_stc \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/pyscfad/dlno_stc"
cmake --build build/dlno_stc -j 4
ctest --test-dir build/dlno_stc --output-on-failure
python -m pytest pyscfad/dlno_stc/test/test_calculation_modes.py \
  pyscfad/dlno_stc/test/test_selected_kernel.py \
  pyscfad/dlno_stc/test/test_w_delayed_pullback.py
```

For deterministic DLNO-CCSD(T), build the native CCSD/MP2/triples extension.
It requires CMake, pybind11, an OpenMP C++ compiler and the active Python/SciPy
environment; Armadillo and MKL are not build dependencies.

`python -m pip install .` now builds this extension automatically using the
compiler's OpenMP runtime. CMake and pybind11 are declared build
dependencies. The commands below remain available for an explicit developer
build. MPI is separate and optional: install/load the host MPI implementation
and install `pyscfad[mpi]` (or `'.[mpi]'` from this checkout). Binary wheels may
bundle OpenMP; separate runtime copies are allowed and are not rewritten.

```bash
cmake -S pyscfad/cc/cpp -B build/ccsd -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/pyscfad/cc"
cmake --build build/ccsd -j 2
```

For resident real RCCSD with cached virtual integrals, the extension evaluates
the complete amplitude update: the virtual ladder, tiled `ovvv`, `ovov` and
`oooo` contractions, intermediates, symmetrization and denominators. Python
retains DIIS, convergence and memory/thread policy. The extension also supplies lambda response, DF integral pullbacks,
deterministic MP2/D/W and the MRCC triples numerator/dressing pullback. Unsupported modes and tight
memory budgets use the Python reference equations. See the
[native residual implementation notes](../cc/cpp/README.md) for the kernel
boundary and diagnostic timing interface.

`config.update('pyscfad_dfccsd_cpp', False)`
disables the optional CCSD residual/lambda kernels (MP2 and deterministic
triples still require the native extension); `config.update('pyscfad_dfccsd_native', False)`
selects the JAX fragment reference implementation. Higher derivatives require
the JAX path; the native RCCSD adapter supports real float64 energies and first
derivatives. Imports do not build or download dependencies.

The native `solve_full` entry point delegates to `solve_selected` with every
occupied target. Independent reference sources in `cpp/test_reference` compile
only with `BUILD_TESTING=ON`, exposing `_solve_full_reference` for tests.
Use `-DBUILD_TESTING=OFF` for a production-only STC extension.

## Command-line examples

Run from the repository root with this checkout installed. The scripts expose
machine-specific memory and threading policies; their defaults can differ from
the library. Inspect their help before choosing a workload:

```bash
python examples/dlno_stc/water_tetramer.py --help
python examples/dlno_stc/smooth_force_compare.py --help
python examples/dlno_stc/benchmark_lno_w_triples.py --help
python examples/dlno_stc/water_16_ccsd_t_gradient.py --help
```

| Runner | Purpose and current behavior |
| --- | --- |
| [water_tetramer.py](../../examples/dlno_stc/water_tetramer.py) | Standalone scalar domain MP2; `--energy-only` skips forces and `--mode stochastic` selects scalar sampling. |
| [water_tetramer_full.py](../../examples/dlno_stc/water_tetramer_full.py) | Standalone system-scope MP2 with scalar energy/response. |
| [smooth_force_compare.py](../../examples/dlno_stc/smooth_force_compare.py) | General smooth CCSD(T) force comparison. Deterministic unless `--seeds` is supplied. Defaults to molecular grouping, zero LIS cutoffs, 32 threads and an 800000 MB process budget. |
| [benchmark_lno_w_triples.py](../../examples/dlno_stc/benchmark_lno_w_triples.py) | C5H6 molecular or atom-grouped alkane/water runner. Deterministic by default; `--compare-solvers` or `--stochastic-only` requires `--targets-meh` and `--seeds`. Exposes process/DF budgets, half/direct storage and a `--domain-thresholds` JSON input. |
| [water_16_ccsd_t_gradient.py](../../examples/dlno_stc/water_16_ccsd_t_gradient.py) | Water-cluster force runner with explicit stochastic triples; see the [water example guide](../../examples/dlno_stc/README_water_16.md) for its launcher defaults. |
| [run_water_16_bulk_sgx.sh](../../examples/dlno_stc/run_water_16_bulk_sgx.sh) | Archived bulk-water16 geometry and domain parameters, `pair_energy=5e-6`, LIS cutoffs `1e-4`/`3e-5`, generated SGX and packed CC ladder. Explicitly deterministic W, CCSD and (T), with three fitted (T) points per LIS. |

The full water16 energy-and-force example runs with:

```bash
PYTHON_BIN=.venv/bin/python THREADS=32 \
  examples/dlno_stc/run_water_16_bulk_sgx.sh \
  --output-dir output/water_16_bulk_deterministic
```

See the [water16 input and provenance](../../examples/dlno_stc/README_water_16.md#bulk-water16-from-the-completed-sgx-benchmark)
for the domain-default comparison and memory settings.

For example, this explicitly requests a deterministic reference and sampled
triples forces on C5H6, with full domains and one untruncated LIS:

```bash
python examples/dlno_stc/benchmark_lno_w_triples.py \
  --xyz-file examples/dlno_stc/c5h6_force_compare.xyz \
  --basis cc-pvdz --frozen 5 --boys-grouping molecule \
  --lno-thresh-occ 0 --lno-thresh-vir 0 --threads 4 \
  --max-memory-mb 10000 --df-memory-mb 1000 \
  --triples-laplace-points 4 --pilot-samples 10000 \
  --compare-solvers --targets-meh 0.05 --seeds 71 \
  --output-dir output/c5h6_triples_comparison
```

The benchmark runner has no `--single-pass-w`, `--selection-only` or `--reference`
option. The comparison runner still accepts `--selection-only` (a domain/CC
plan, not completed W/LIS selection) and `--reference`; its
`--triples-budget-energy ccsd` option is incompatible with ordinary fragment
plans, so retain the default. Its `--fd-coordinate` refits LIS/(T) choices on
displaced evaluations; this is not a fixed-label derivative check.

The two comparison runners set `RCCSD.max_cycle=200`. Their energy/amplitude
tolerances differ: `smooth_force_compare.py` uses `1e-10`/`1e-8`, while
`benchmark_lno_w_triples.py` uses `1e-7`/`1e-5`. The benchmark's default pilots
are 20000, the comparison runner's 10000. Output includes settings and individual
energy/force records; the comparison runner also writes `comparison.json`.
Force comparisons use Cartesian-component RMS in mEh/bohr. These diagnostics
are separate from the stochastic energy SE target.

Weak-pair forward and pullback use NumPy/SciPy and explicit real-float64
adjoints. The JAX weak-pair fallback has been removed. Use the explicit
`weak_pair_energy_and_grad` API; tracing weak-pair forward functions with JAX
is unsupported. Coordinate integral response uses the native moment VJP.
