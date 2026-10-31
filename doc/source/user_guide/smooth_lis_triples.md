# Full-LIS CCSD(T) forces and selected Laplace sampling

For runnable C5H6 and C6H14 examples and a short defaults table, start with the
[smooth CCSD(T) README](../../../README_SMOOTH_CCSD_T.md). This reference describes
the current public Python drivers and both run scripts. Defaults below are
source defaults; run-script overrides are listed separately.

The supported smooth implementation has three parts:

1. One full occupied block and one full virtual block in each retained LIS.
   The internal Fock projector is the identity.
2. Native asymmetric Laplace `(T)` dressing with full off-diagonal Fock response.
3. Restricted sampling over the physical selected occupied target S, with a
   fused sampled energy and input derivative calculation.

A deterministic conventional reference remains available with `smooth=False`.
Stochastic calculations require smooth selections. Canonical stochastic
triples, equal-weight dressing, the JAX production dressing backend, explicit-S
frames and the older adaptive Gaussian triples grid have been removed.

## Frames and physical selection

`build_static_selections` chooses LIS ranks, AO anchors and Laplace grids once,
outside differentiation. `select_frame` saves semicanonical reference anchors;
`rebuild_frame` transports those anchors with the current LIS and AO overlap.
Its input contains no Fock matrix, so the Fock-frame cotangent is zero.

Writing the current metric-orthonormal retained coefficients as C, the saved AO
anchor columns as A0, and the AO overlap as S_AO, the transport is

```text
X = C (C^T S_AO A0)
G = X^T S_AO X = L L^T
Q = X L^-T,               Q^T S_AO Q = I.
```

A0 is held fixed inside the derivative. C and S_AO remain differentiable.
G equals the identity at the reference geometry, up to numerical error, but is
not defined to be constant. The Cholesky step enforces metric orthonormality
during the response. There is no selectable QR or alternative normalization mode.

The retained D/W subspace and the AO overlap still respond to geometry. Their
response is physical and remains differentiated. Metric normalization uses a
Cholesky factor and a triangular solve; there is no Fock-block projector
construction or inverse internal orbital-gap response.

The identity **frame** projector does not remove physical fragment selection.
For a target coefficient matrix U in the occupied LIS, the fragment energy
weight is U U^T. Its response remains part of the force. When S spans the whole
occupied LIS, this weight is the identity, and the code uses the equivalent
representative U = I. The physical fragment coefficients themselves need not
equal the identity in a semicanonical frame.

CCSD and the MP2 subtraction retain the full occupied and virtual Fock matrices.
The MP2 resolvent uses an implicit derivative of its full Fock operator.

At the reference, Q coincides with the saved semicanonical anchors. Keeping that
forward basis preserves the usual CCSD diagonal-Fock preconditioner; the reverse
calculation includes off-diagonal Fock response. Internal occupied–occupied or
virtual–virtual Fock degeneracies do not require individual-eigenvector
derivatives. A truncated retained/discarded D/W boundary still has its own
physical subspace response; the anchored frame does not remove that response
or make a changing LIS rank differentiable.

## Asymmetric Laplace triples

At time t define

```
O = exp(t F_oo / 2),  V = exp(-t F_vv / 2)
L = O^-1 U,          R = O U
```

External numerator legs receive half exponentials. Reciprocal factors on
contracted internal legs cancel. The physical weight becomes
`O^-1 U U^T O = L R^T`; the native kernel contracts the selected S index directly.
It evaluates the denominator-free numerator, including its physical sign and
normalization. The quadrature sum supplies the energy denominator.

Both partial and whole-LIS targets use this asymmetric kernel. Whole-LIS
sampling therefore follows the selected-space proposal as well; individual
seed realizations from the removed whole/canonical kernel are not preserved.
The deterministic energy and its full derivative remain the validation target.

The native `TriplesDressing(inputs)` constructor takes nine arrays:
`t1, t2, Bvo, Boo, Bvv, fov, Foo, Fvv, U`. It returns six dressed tensors plus L
and R. Reverse mode includes derivatives of both Fock blocks and of U. Stable
matrix-function derivatives handle internal Fock degeneracies without
individual-eigenvector derivatives. Tensor panels bound dressing scratch, and
reverse mode rematerializes panels instead of saving every T2 intermediate.

The only native triples entry points are `triples_asymmetric_energy` and
`triples_asymmetric_energy_vjp`. With `nsamples=0` they enumerate a small exact
reference. Positive counts use selected-space restricted sampling. The VJP
returns derivatives from the same draws as the energy; reverse does not resample.

## Controls

The following assumes a converged density-fitted `mf`, its molecule `mol`, a
`build_mf(mol)` callback returning a converged differentiable MF, domain selections
`domains`, and separate MP2 `baseline` / D/W `matrix` control dictionaries.

```python
from pyscfad.dlno_stc import ccsd

static = ccsd.build_static_selections(
    mf, domains, matrix_kind="W", thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls=baseline, matrix_controls=matrix,
    smooth_laplace_points=10,
)
triples = dict(
    mode="stochastic", nsamples=10000, pilot_samples=10000,
    global_seed=71, energy_tolerance=5e-5, trace_exponent=5.0,
)
energy, mol_bar, components = ccsd.value_and_grad(
    mol, build_mf, static, baseline_controls=baseline, matrix_controls=matrix,
    ccsd_t=True, triples_controls=triples, include_hf=True, return_components=True,
)
forces = -mol_bar.coords  # Eh/bohr
```

`ccsd.kernel` is an energy-only entry point and cannot be wrapped in JAX AD.
Use `ccsd.value_and_grad` for forces. With `return_components=False`, they return
`energy` and `(energy, mol_bar)`, respectively. With it enabled, they return
`(energy, components)` and `(energy, mol_bar, components)`.

The tolerance is in Hartree and controls the system `(T)` energy standard error.
Fragment variance budgets use absolute MP2 fragment energy proxies, or explicit
`fragment_energies`. Point budgets use a trace model with configurable positive
`trace_exponent` (default 5.0 in the API and both runners). The MP2 trace model
uses exponent 1.4.

Independent discarded pilots choose production counts, with at least 10000
production draws per point in tolerance mode. Production extends until each
point meets its allocated SE. The pilot and production share prepared tensors
and proposal tables; pilot draws are excluded from production statistics and
forces. Zero or omitted energy tolerance selects a fixed production budget.

The fitted triples grid uses ten positive points by default, fitting both inverse
first and second powers of denominators. Grid diagnostics report energy and
derivative approximation errors separately from stochastic SE. Point count,
seeds, sample counts, tolerances and trace exponents remain configurable.
This triples grid is separate from the MP2 baseline / D/W grid. The latter
defaults to the existing six fixed points; the example runners instead fit six
MP2 points with `lambda_gradient=0.1`. The triples fitter also fixes its
derivative weight at `0.1`; this is not a separate public triples option.

### Forward choices and backward response

The controls configure the calculation before evaluation. Ranks, grouping,
anchors, quadrature rules, proposal probabilities, random draws, and allocation
decisions are not differentiated. Sampled numerator derivatives use the same
production samples and importance weights as the energy; there is no independent
gradient seed or gradient sample count. Pilots do not contribute to the returned
energy or gradient. The physical derivatives through integrals, amplitudes,
full Fock matrices, fragment coefficients, retained D/W subspaces, AO overlap,
Boys localization, and SCF response remain included.

Thus these are not generally “forward-only options.” In particular, the default
DF-CCSD custom response uses `conv_tol_normt` and `max_cycle` for its backward
lambda solve as well as its forward solve. The `(T)` SE target controls estimated
energy uncertainty; it does not impose a force uncertainty or RMS-error target.

### Domain and LIS selection options

`dlno_stc.build_domain_selections(mf, ...)`:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `frozen` | Required | Frozen occupied specification; use `0` for no frozen core. |
| `boys_grouping` | `"singleton"` | `singleton`: one S per active Boys orbital; `atom`: assign Boys orbitals to heavy-atom/H groups by Löwdin population; `molecule`: all targets in one S. |
| `force_full_domains` | `False` | Full extended domains; automatically true for molecule grouping. It does not set LIS cutoffs to zero. |
| `thresholds` | `None` → `BoysDomainThresholds()` | Domain completeness, PAO rank, and pair-screen controls below. |
| `lo_kwargs` | `None` → `{}` | Boys options: `init_guess`, `conv_tol`, `conv_tol_grad`, `max_cycle`, `symmetry`, `gmres_options`. Unspecified options inherit the Boys implementation defaults. |
| `target_coeff` | `None` | Build Boys targets; supply coefficients to use reference target orbitals. |
| `frag_lolist` | `None` | Default singleton occupied-column partition; explicit input must also be a complete disjoint singleton partition. Use `boys_grouping` to combine physical S. |
| `pair_energy_model` | `"multipole"` | Also accepts `multipole-os` / `os-multipole`, `exact`, and `all` / `all-strong`. |

`BoysDomainThresholds` defaults: `bp_occ=0.985`, `bp_primary=0.999`,
`bp_ed=0.9998`, `bp_pao=0.98`, `pao_norm=1e-4`, `domain_pao=1e-4`,
`ed_pao=0.995`, `metric_rank=1e-10`, `pair_energy=1.5e-5`,
`near_pair_distance=3.5`, and `multipole_order=4`.
See the [domain and MP2 guide](../../../pyscfad/dlno_stc/README.md) for their definitions.

`ccsd.build_static_selections(mf, mp2_static, ...)`:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `matrix_kind` | Required | `"D"` or `"W"`; no automatic conversion between their thresholds. |
| `thresh_occ`, `thresh_vir` | Required | Nonnegative absolute-eigenvalue thresholds for retained external occupied/virtual directions. Zero retains the full corresponding space. |
| `baseline_controls`, `matrix_controls` | Required dictionaries | Independent MP2 baseline and D/W precision choices; `{}` selects their deterministic defaults. |
| `internal_rank_threshold` | `1e-6` | Numerical rank threshold for constructing discarded complements when completing the full MO layout; all original S orbitals are retained. |
| `merge_identical` | `True` | Reuse identical-domain preparation while retaining physical S identities. |
| `fragment_indices` | `None` | Select all physical fragments; otherwise select only the listed zero-based S indices. |
| `smooth` | `True` | Full occupied/virtual anchored frames. `False` retains the conventional deterministic reference. |
| `smooth_laplace_points` | `10` | Positive integer; fit a separate positive triples quadrature for each nonempty LIS. Used with `smooth=True`. |

The baseline and matrix dictionaries must match the scientific settings used to
build selections. D/W matrices can independently use `mode="deterministic"`
(default) or `"stochastic"`. Matrix defaults include `global_seed=0`, `threads=1`,
`target_batch_size=1`, `auxiliary_group_size=32`, and no `memory_limit_mb`.
Stochastic matrix defaults and separate MP2 energy controls are listed in the
[matrix control table](../../../pyscfad/dlno_stc/README.md#native-dw-matrices-and-delayed-reverse).
Those controls are not accepted in `triples_controls`.

### Energy and gradient driver options

These keywords are common to `ccsd.kernel` and `ccsd.value_and_grad`:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `baseline_controls`, `matrix_controls` | Required dictionaries | MP2 baseline and D/W settings matching the saved selections. |
| `matrix_kind` | `None` | Infer from selections; an explicit value must match. |
| `ccsd_t` | `False` | Enable perturbative triples when true. |
| `triples_controls` | `None` | Deterministic triples if `ccsd_t=True`; must remain `None` if triples are disabled. |
| `include_hf` | `False` | Include HF once for total molecular energy and gradient. |
| `include_baseline` | `True` | Include the complete MP2 baseline, even for a subset of S. False returns only selected `CCSD + (T) - MP2-LIS` corrections and requires `include_hf=False`. |
| `fragment_indices` | `None` | Evaluate all saved LIS selections, or the requested zero-based S subset. |
| `return_components` | `False` | Append `CCSDComponents` to the result. |
| `merge_identical` | `True` | Share identical-domain preparation. |
| `verbose_imp` | `0` | Impurity solver verbosity. |
| `comm` | `None` | Serial execution; optionally supply an MPI communicator. |
| `parallel_scf_jk` | `True` | For multiple MPI ranks, distribute shared DF J/K work and its applicable response; no effect on serial calls. |
| `checkpoint_dir` | `None` | Optional serial progress checkpoint directory. |
| `resume` | `False` | Resume compatible progress; requires `checkpoint_dir`. Multiple-rank CC progress checkpoints are not supported. |

`kernel(mf, static, ...)` requires a converged MF and selections.
`value_and_grad(mol, build_mf, static=None, ..., build_static=None)` accepts either
saved `static` or a `build_static(mf)` callback, which selects once on the detached
SCF reference. Supply one of these, not both. Its molecule must be configured
for coordinate derivatives (`mol.build(trace_exp=False, trace_ctr_coeff=False)`).
With multiple ranks, `build_mf` must also accept the supplied `mo_coeff_init`,
`mo_energy_init`, `mo_occ_init`, and `e_tot_init`, adopting them without another SCF
on workers. The comparison runners below use one process.

### Triples dictionary: complete accepted keys

| Key | Default | Meaning / allowed values |
| --- | --- | --- |
| `mode` | `"deterministic"` | `deterministic` or `stochastic`. With smooth selections, deterministic means exact numerator enumeration on the fitted Laplace grid, not exact integration of the denominator. |
| `nsamples` | Required in stochastic mode | Integer ≥ 2. Nominal production samples per S in fixed-count mode. Still required with a positive tolerance; it does not cap adaptive production. |
| `global_seed` | `71` | Nonnegative integer smaller than `2**64`. |
| `energy_tolerance` | Omitted; `0` is equivalent | Total requested triples SE in Eh. Positive values enable adaptive production. |
| `pilot_samples` | `10000` | Integer ≥ 2, per Laplace point per S; independent pilots are discarded. |
| `trace_exponent` | `5.0` | Finite positive exponent for triples point variance allocation in tolerance mode. MP2 uses `1.4`. Fixed-count allocation uses measured pilot variances instead. |
| `fragment_energies` | Saved MP2 proxies | Optional finite one-dimensional vector, one value for every physical S, even when running a subset; absolute values allocate variance. CCSD proxies must be supplied explicitly. |

The last six keys are sampling controls and are rejected with deterministic
mode. High-level stochastic `nsamples=0` is invalid: use deterministic mode for
enumeration. The `nsamples=0` convention belongs only to the native entry points.

### Variance allocation and sample accounting

For total requested SE T, requested fragment set I, and saved/overridden energies
e_S, the fragment allocation is

```text
a_S = max(|e_S| / max_{R in I}|e_R|, 1e-12)
f_S = a_S / sum_{R in I} a_R
T_S^2 = T^2 f_S.
```

All-zero proxies use equal shares. Legacy selections without proxies also fall
back to equal shares. For a subset of S, the total target is allocated over that
subset; a one-S test gives that S the full requested target.

For node t_q, quadrature weight w_q, and trace exponent p:

```text
score_q = |w_q| [Tr exp(t_q F_oo/2) Tr exp(-t_q F_vv/2)]^p
f_q     = score_q / sum_r score_r
T_Sq^2  = T_S^2 f_q
native unweighted SE target = T_S sqrt(f_q) / |w_q|.
```

Thus `sum_S sum_q T_Sq^2 = T^2`. The trace model fixes variance shares before
sampling. Independent pilots estimate how many production draws each share
needs; they do not change these shares. The positive-target path completes one
point at a time, reusing its prepared tensors and probability tables, and starts
with at least 10000 production draws per point. Production extends using its
measured SE until the allocated point target is met. The reported total SE is
the square root of the sum of weighted production variances. This checks an
estimated SE, not a bound on the unknown exact error.

In fixed-count mode, pilot estimates instead give
`N_q = max(2, ceil(nsamples * sigma_q / sum_r sigma_r))`, with a small support
floor; `sigma_q` is the estimated single-draw standard deviation including
`|w_q|`. Counts are per S; rounding can exceed `nsamples`. All pilot draws are
additional. This mode prepares all points for pilot-based allocation, whereas
positive-target mode can finish and release one point at a time.

### CCSD convergence

The inherited RCCSD defaults, absent user configuration, are `conv_tol=1e-7`
Eh, `conv_tol_normt=1e-5`, and `max_cycle=50`. Both the energy-change and
amplitude-update-norm tests must pass. The usual diagonal-Fock single/double
excitation denominators precondition the update; off-diagonal Fock entries remain
in the equations. The impurity solver uses `PYSCFAD_LNO_CCSD_DIIS_SPACE=3` by
default; a nonpositive value disables DIIS.

The default `pyscfad_dfccsd_custom_response=True` path solves the response lambda
equations with tolerance `mycc.conv_tol_normt` and limit `mycc.max_cycle`, using
DIIS when enabled. These are not separate triples-dictionary options. Set
`pyscfad.lno.ccsd.RCCSD.conv_tol`, `.conv_tol_normt`, and `.max_cycle` before
calling the driver; the example runners expose the first two as CLI flags and
set `.max_cycle=200`. Force requests do not automatically tighten them.

### Removed controls and saved selections

`dressing`, `dressing_backend`, `adaptive`, `smooth_explicit_s`, `smooth_gap`,
`smooth_full_blocks` and a `None` triples point count are not accepted options.
The CC selection schema is 6. Rebuild selections/checkpoints after this cleanup;
the native implementation version also participates in checkpoint identity.

## Command-line runners

Run from the repository root with this checkout installed and the native module
built. [Build instructions](../../../pyscfad/dlno_stc/README.md#build-and-checks)
list CMake, pybind11, Armadillo, and OpenMP requirements. Both scripts configure
CPU affinity and numerical workers before importing JAX. Use `--threads 32` for
the intended host; Python matrix controls otherwise default to one thread.
The native triples loop processes up to the OpenMP worker count of sample blocks
at once. Neither script exposes a separate batch count or backward sample count.

### `examples/dlno_stc/smooth_force_compare.py`

Compares conventional deterministic forces with smooth stochastic forces. This
is the general geometry runner, including C5H6, C6H14, and square H4.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--xyz-file` | Required | XYZ geometry, in Angstrom. |
| `--output-dir` | Required | JSON output directory. |
| `--threads` | `32` | CPU affinity and numerical worker budget. |
| `--basis` | `cc-pvdz` | Orbital basis; H/C `def2-SV(P)` is handled explicitly. |
| `--frozen` | `0` | Frozen occupied count. |
| `--boys-grouping` | `molecule` | `singleton`, `atom`, or `molecule`. |
| `--lno-thresh-occ`, `--lno-thresh-vir` | Both `0` | W cutoffs; defaults are untruncated. |
| `--finite-domains` | Off | Use finite domains; molecule grouping still forces full domains. |
| `--selection-only` | Off | Stop after selections and settings output. |
| `--triples-energy-tolerance-mha` | `0.05` | Total requested `(T)` SE in mEh; zero selects fixed counts. |
| `--samples` | `16446469` | Nominal production draws per S when tolerance is zero. |
| `--triples-pilot-samples` | `10000` | Independent pilot draws per point per S. |
| `--triples-laplace-points` | `10` | Fitted triples point count. |
| `--triples-trace-exponent` | `5.0` | Triples trace-model exponent. |
| `--triples-budget-energy` | `mp2` | `mp2` uses selection proxies; `ccsd` uses fragment energies from the deterministic reference. |
| `--seeds` | `71 72 73` | One stochastic calculation per seed; an empty list skips stochastic runs. |
| `--smooth-deterministic` | Off | Also enumerate smooth deterministic triples; expensive beyond small systems. |
| `--reference` | None | Reuse a compatible deterministic result JSON instead of computing it. |
| `--sampling-fragment` | None | Zero-based S correction only, without the full baseline/HF or full-force comparison; requires `--reference` or `--fragment-deterministic`. |
| `--fragment-deterministic` | Off | Compute a conventional deterministic reference for `--sampling-fragment`. |
| `--fd-coordinate` | None | Two zero-based integers `ATOM AXIS`; central finite differences of conventional and smooth deterministic energies. |
| `--fd-step` | `1e-4` | Finite-difference displacement in bohr. |
| `--cc-conv-tol` | `1e-10` | Forward CCSD energy-change threshold in Eh. |
| `--cc-conv-tol-normt` | `1e-8` | Forward amplitude and backward response update-norm threshold. |

The one-S mode cannot be combined with `--smooth-deterministic` or
`--fd-coordinate`. With `--triples-budget-energy ccsd`, it also requires a
reference providing all fragment energies. Its tolerance is the full target for
that one requested correction; pass the original allocated fragment SE if
matching a full-molecule allocation.

### `examples/dlno_stc/benchmark_lno_w_triples.py`

Runs a queue of target/seed pairs, saving each result. This specialized runner
currently asserts C5H6 for molecule grouping (including C-then-H atom order) and
C20H42 for atom grouping, with one or twenty physical S respectively. Use the
comparison runner above for other molecules.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--threads`, `--xyz-file`, `--basis`, `--frozen`, `--output-dir` | Required | Worker budget and molecular/output settings. |
| `--boys-grouping` | Required | `molecule` or `atom`, subject to the restrictions above. |
| `--targets-meh` | Required | Positive distinct total triples SE targets in mEh. |
| `--seeds` | Required | Distinct nonnegative seeds. |
| `--job-indices` | All target/seed jobs | Zero-based subset of the target-major, seed-minor job list. |
| `--pilot-samples` | `20000` | Pilot draws per triples point per S; pass `10000` to match the chosen production setup. |
| `--max-memory-mb` | `800000` | Molecular memory setting and advisory matrix planning budget, in MB. |
| `--cc-conv-tol` | `1e-7` | Forward CCSD energy-change threshold in Eh. |
| `--cc-conv-tol-normt` | `1e-5` | Forward amplitude and backward response update-norm threshold. |
| `--triples-method` | `smooth` | `smooth` or deterministic-only `conventional`. |
| `--smooth-laplace-points` | `10` | Fitted triples points per LIS. |
| `--smooth-trace-exponent` | `5.0` | Trace-model exponent. |
| `--selection-only` | Off | Stop after selections and settings output. |
| `--deterministic-only` | Off | Compute only the deterministic reference for the selected method. |
| `--stochastic-only` | Off | Run only requested stochastic jobs. Mutually exclusive with `--deterministic-only`. |
| `--reference` | None | Reuse a compatible deterministic JSON; invalid with `--deterministic-only`. |

This runner fixes `force_full_domains=True`, `matrix_kind="W"`,
`thresh_occ=1e-4`, and `thresh_vir=1e-5`. `conventional` requires
`--deterministic-only`; `smooth` deterministic enumeration is principally a
small-system check. Required target/seed arguments are still parsed for
selection-only and deterministic-only invocations.

### Settings shared by the runners

Both use deterministic MP2 baseline and W matrices, a six-point fitted MP2 grid
with `lambda_gradient=0.1`, baseline seed 666, and automatic auxiliary-basis
selection. For H/C `def2-SV(P)`, the scripts explicitly use def2-SVP on carbon,
only its s shells on hydrogen, and `def2-svp-jkfit` as the auxiliary basis.
Both use DF-RHF thresholds `conv_tol=1e-9` and `conv_tol_grad=1e-6`, atomic Boys
initialization with `conv_tol=1e-10`, and Boys derivative GMRES settings
`restart=120`, `maxiter=100`. These are runner settings, not triples controls.
Both fix CCSD `max_cycle=200`. The comparison script fixes memory at 800000 MB;
the benchmark exposes it as a flag. This setting leaves room on the approximately
900 GB host; STC planning budgets are advisory rather than memory reservations.

## Results and verification

`benchmark_lno_w_triples.py --triples-method smooth` runs the retained path.
`--triples-method conventional` requires `--deterministic-only`.
`smooth_force_compare.py` compares retained smooth calculations with a
conventional deterministic energy/force reference, including finite differences.
CPU affinity is derived from the requested thread count before JAX imports.

`return_components=True` provides each fragment's `e_mp2_lis`, `e_ccsd`, `e_triples`,
`triples_samples`, `triples_standard_error`, and `triples_diagnostics`. The latter
contains per-point nodes/weights, variance fractions and targets, pilot counts,
initial/adaptive/total production counts, weighted energies/variances,
`target_met`, and grid-independent sampling diagnostics. The saved selection's
`smooth_grid` holds the quadrature-fit diagnostics. The MP2 baseline SE is
separate from triples SE.

The comparison runner writes `settings.json`, the individual energy/force result
JSONs, and `comparison.json` (or `fragment_comparison.json` for a one-S test).
It reports

```text
RMS force error [mEh/bohr]
    = 1000 sqrt(sum_{A=1..N} sum_{alpha=x,y,z}
                    (F_stochastic[A,alpha] - F_reference[A,alpha])^2 / (3N)).
```

This is the RMS over Cartesian components, without alignment or removal of net
translation/rotation. Forces are the negative coordinate gradients. Changing
seeds probes sampling scatter; the energy SE target alone does not specify this
force error.

Scientific tests compare whole and partial target energies and every tensor
cotangent with an independent slow deterministic reference. Other checks cover
noncanonical Fock response, degenerate matrix functions, moving/truncated LISs,
sampled derivative means, pilot reuse, SE targets and checkpoint rejection.
The independent JAX dressing implementation exists only in test reference code.
