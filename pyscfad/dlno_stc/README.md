# Domain MP2 and DLNO-CCSD/CCSD(T)

The library supports four workflows:

| Calculation | Forward energy | Energy and nuclear gradient |
| --- | --- | --- |
| Domain MP2 | `dlno_stc.kernel` | `dlno_stc.value_and_grad` |
| DLNO-CCSD/CCSD(T) | `ccsd.kernel` | `ccsd.value_and_grad` |

Forces are `-gradient.coords`. Set `include_hf=True` on gradient calls or CC
energy calls for total molecular results. MP2 `kernel` returns correlation
energy; add `mf.e_tot` for total energy. Forward calls do not create response
tapes. Use `ccsd_t=True` to include triples.

W selection and smooth anchored LIS frames are the only supported choices.
Every solver mode defaults to deterministic. Stochastic MP2, W, CCSD and
triples remain available explicitly.

## Fragments and domains

```python
from pyscfad import dlno_stc
from pyscfad.dlno_stc import ccsd

domains = dlno_stc.build_domain_selections(
    mf, frozen=1, boys_grouping="atom", scope="domain", full_domain=False,
    mp2_laplace_points=6,
)
```

`scope="system"` puts all active occupied targets in one molecular fragment
and uses the same preparation, solver and response pipeline. It is equivalent
to `boys_grouping="molecule"`. `full_domain=True` expands every fragment's
occupied, primary, PAO and extended domains to the entire system while
preserving physical fragment membership. Frozen orbitals remain excluded;
LNO cutoffs still apply.

`boys_grouping` accepts `singleton`, `atom` and `molecule`. Pair screening and
`BoysDomainThresholds` control finite domains. Domain selection fixes discrete
labels; production reconstructs continuous orbitals and their responses.
`merge_identical=True` shares preparation among identical domains while each
physical fragment retains its own target projector and CC correction.

## Calculation interface

```python
controls = {"mode": "deterministic", "threads": 1}
mp2_energy = dlno_stc.kernel(mf, domains, controls=controls)
mp2_energy, mp2_gradient = dlno_stc.value_and_grad(
    mol, build_mf, domains, controls=controls, include_hf=True,
)

selected = ccsd.build_static_selections(
    mf, domains, thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls=controls, matrix_controls=controls,
)
options = dict(baseline_controls=controls, matrix_controls=controls,
               ccsd_t=True, include_hf=True)
cc_energy = ccsd.kernel(mf, selected, **options)
cc_energy, cc_gradient = ccsd.value_and_grad(mol, build_mf, selected, **options)
```

`build_mf(mol)` must return a converged restricted density-fitted reference.
For gradients, `build_static(mf)` can replace explicit selections so selection
and production share the differentiated SCF. The MP2 entry points also accept
`scope="system", frozen=...` directly to select the molecular fragment.

CC always includes every physical fragment and the complete domain MP2
baseline, including weak-pair completion:

`E_corr = E_MP2_domain + sum_f (E_CCSD,f + E_(T),f - E_MP2_LIS,f)`.

W construction supplies the strong-domain MP2 energy through
`(trace(W_occ) + trace(W_vir))/2`. One delayed W pullback combines its energy
seed with the LIS-response cotangents. The shared `mp2_grid=(roots, weights)` supplies both MP2 and W. `return_components=True` returns fragment energies and uncertainty
and resource diagnostics.

There is one pipeline and no `single_pass` option:

1. Run cold or warm SCF.
2. Screen weak pairs and group Boys orbitals into physical fragments; retain
   the weak-pair response when screening runs inside the force callback.
3. Prepare each fragment domain. Standalone MP2 computes the scalar energy
   directly, without W. CCSD/(T) computes W once, selects its smooth LIS and
   solves the correction. Consume the pullbacks on that owner.
4. Accumulate the common-data response and solve the SCF response for forces.

The CC builder returns a plan, without computing W. Its production W selects
ranks, anchors and the independent (T) grid, held fixed in that fragment's
energy and pullback. `correlation_auxbasis`
selects a separate correlation fitting basis; selection and execution must
use the same basis. HF/Fock continue using the SCF fit.

## Separate MP2 and triples quadratures

`build_domain_selections` automatically fits **one global MP2/W grid** from
all active canonical occupied/virtual energies, excluding frozen orbitals.
`mp2_laplace_points=6` sets its size. Every domain uses the same saved grid;
`scope="system"` follows the same rule. Nodes/weights are saved as
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

Seeds depend on physical fragment and sampling stream, not MPI rank. Saved
production samples are reused for response. Reported energy standard errors
do not bound force errors.

See the [smooth solver guide](../../doc/source/user_guide/smooth_lis_triples.md)
for equations, quadrature, stochastic controls, CC convergence and diagnostics.
The [smooth CCSD(T) examples](../../README_SMOOTH_CCSD_T.md) cover practical runs.

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
`lis_aux_svd_tol` controls LIS auxiliary-space compression (default `1e-5`;
`None` or zero disables it). These memory/accuracy choices are independent of
H storage. Memory forecasts warn when budgets are exceeded; proven impossible
new allocations are rejected. No global AO-pair tensor is required by local
correlation preparation.

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
selection schema 18 records the fragment plan and requires all fragments; older selections must be rebuilt.

Disk request/result exchange, export/import/replay drivers, per-domain MPI
replica averaging and public fragment-subset/correction-only modes have been
removed. The old `force_full_domains` alias is removed; use `full_domain`.
The fixed-value `matrix_kind`, `smooth` and `lis_space` arguments are removed.
Use `triples_laplace_points` in place of `smooth_laplace_points`.
Restart checkpoints remain supported independently of disk solver exchange.

## Build and verification

With PySCFAD, pybind11, Armadillo and OpenMP available, build from the repo root:

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

Imports do not build or download dependencies. Examples include
`water_tetramer.py`, `water_tetramer_full.py`, `benchmark_lno_w_triples.py` and
`smooth_force_compare.py`. Benchmark runners use deterministic calculations by
default; sampling comparisons require explicit stochastic options.

The native `solve_full` entry point delegates to `solve_selected` with every
occupied target, for both deterministic and stochastic calculations and their
responses. Independent full-system reference sources live in `cpp/test_reference`
and are compiled only with `BUILD_TESTING=ON`, exposing `_solve_full_reference`
for numerical tests. Use `-DBUILD_TESTING=OFF` for a production-only extension.
