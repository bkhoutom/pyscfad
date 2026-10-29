# Smooth CCSD(T) forces for physical groups and truncated LIS spaces

The opt-in `smooth_gap` argument of `dlno_stc.ccsd.build_static_selections`
constructs a smooth block semicanonical frame for each physical group. It supports
atom, singleton and molecule grouping, truncated occupied/virtual LIS spaces,
and partial occupied energy targets. The semicanonical path remains the default.

```python
selected = ccsd.build_static_selections(
    mf, domains, matrix_kind="W", thresh_occ=1e-4, thresh_vir=1e-5,
    baseline_controls=baseline, matrix_controls=matrix,
    smooth_gap=0.01,  # Hartree; cluster adjacent Fock eigenvalues
)
```

## Differentiated quantities

The reference selection saves AO anchors and clusters of adjacent Fock
eigenvalues. The derivative keeps those clusters and all LIS ranks fixed.
Current D/W-selected occupied/virtual projectors first project the anchors into
the retained LIS. All required internal target directions are retained. Only
retained/discarded W gaps enter its projector derivative; W, target and parent
orbital response are included. Each Fock cluster then uses its spectral projector and a
Cholesky-normalized anchored frame. The projector JVP contains only gaps between
clusters; it has no divisions by gaps inside a cluster. A retained manifold or
cluster boundary still needs a finite gap, and the projected anchors must remain
linearly independent. Rebuild selections when leaving the neighborhood of the
reference geometry.

CCSD and its existing implicit Lambda response use the full Fock blocks in this
frame. The MP2 subtraction solves the full two-particle Fock resolvent with a
custom linear-solve derivative. Initial diagonal denominators are not used as a
noncanonical MP2 energy approximation.

## Laplace triples

For a positive triples excitation energy D, use

    1/D = integral_0^infinity exp(-t D) dt.

At each node define O=exp(t F_oo/2), V=exp(-t F_vv/2). All occupied/virtual
external legs of both triples numerators receive O/V. For the existing native
DF contraction interface, this is implemented by

    t1 -> O t1 V
    t2 -> (O tensor O tensor V tensor V) t2
    Bvo[P] -> O Bvo[P] V
    Boo[P] -> O Boo[P] O^-1
    Bvv[P] -> V Bvv[P] V^-1
    Fov -> O Fov V

The reciprocal factors cancel the dressing on the internal contracted legs of
t2. The transformed Boo/Bvv are generally nonsymmetric; they must not be
repacked as symmetric tensors. The numerical native kernel evaluates
these dressed numerators with the constant denominator -1. Quadrature weights
then sum the node energies. For a partial target M=U U^T, additionally transform
M -> O^-1 M O. By default, `triples_controls['dressing']='asymmetric'` factors
this weight as L R^T with L=O^-1 U and R=O U. Native S-projected tensor blocks
contract directly over `(s,j,k,a,b,c)`, s in S, without forming a dense weight
or sampling an extra occupied projector index. The selected leg has zero/full
dressing on the left/right; the other legs retain equal half dressing.
Both projections, the full Fock exponentials and target orbitals are
differentiated. All S/environment Fock couplings are retained. Choose
`dressing='equal'` to use the earlier full nonsymmetric weighted kernel.
The full-target identity case retains
the original whole-space path. MP2 and CCSD energies use the same target weight.

Matrix-exponential JVPs use divided differences with their analytic coincident
eigenvalue limit. Numerical diagonalization inside that matrix function does not
differentiate individual eigenvectors. Exponentials are evaluated within Fock
clusters. Inter-cluster Fock entries are analytically zero in this frame, so
their roundoff remnants are excluded. Small cluster widths limit conditioning
of the reciprocal factors. Very large spectral ranges can still overflow and
are not covered by the present benchmark.

### Native dressing and gradient accumulation

Native dressing is the selected production backend for calculations that usually
run once. It avoids the initial JAX compilation cost and bounds worker scratch
memory. Production comparisons should therefore include first-use costs.

Smooth triples now defaults to `dressing_backend='native'`. This reuses the
MP2 `exponential` and `exponential_vjp` routines, caches each Fock cluster's
spectrum once per LIS evaluation, and applies block transforms through bounded
panels. It preserves the same transforms and full matrix-function derivatives.
The original JAX implementation remains available with `dressing_backend='jax'`
(or `--triples-dressing-backend jax` in the comparison runner).

For an axis transform `Y=A X`, reverse accumulation uses `bar_X=A^T bar_Y`
and `bar_A=bar_Y X^T`. Preceding transforms are rematerialized into one shared
staging tensor, avoiding a saved full T2 array for every intermediate. Each
OpenMP worker uses at most three 2 MiB tensor panels plus small Fock-block
Gram matrices. The complete output cotangents and one staging tensor remain
shared; these bounds do not include the input/output tensors or the CCSD tape.
Large blocks use one MKL BLAS worker inside each OpenMP worker when the C local
thread-control entry point is available; other backends use SIMD panel loops.

The asymmetric triples reverse kernel compiles permutation, projection, stride
and sign metadata once per call. Parallel integer histograms and scatters group
312 update events per draw by output tile. Contiguous sample partitions retain
the previous summation order within each tile. One worker writes a tile at a
time, so full T2/integral cotangents remain shared without atomics or a copy per
thread. Histogram scratch is limited to 64 MiB unless a single owner histogram
itself exceeds that size. The event buffer remains bounded by the active team's
2048-draw batches. Diagnostics separate `vjp_grouping_seconds`,
`vjp_accumulation_seconds` and `vjp_finalize_seconds`; these sum to
`vjp_reverse_seconds` up to timer overhead. Neither optimization changes the
proposal distribution, RNG streams, independent pilots or SE stopping rule.

The positive quadrature is fixed before AD. Its 1/D and 1/D^2 errors are checked
at 10,001 denominators over bounds covering the full reference triples spectrum.
These checks measure quadrature error only, not stochastic force uncertainty.

## Sampling and comparison

`mode="deterministic"` enumerates the whole numerator and its VJP at each node;
it is intended for small validation systems and is much slower than the
optimized conventional deterministic (T) kernel.

The default grid now has exactly **10 fitted points** per LIS, controlled by
`smooth_laplace_points` at selection. `None` retains the older adaptive Gaussian
grid. Fitted energy and denominator-derivative errors are recorded.

With a positive `energy_tolerance` (total Hartree SE), variance is divided among
S in proportion to the magnitudes of their reference MP2 correlation energies
saved during D/W selection. `fragment_energies` can override those proxies with
one saved CCSD/MP2 correlation energy per S. Within each S, the variance shares
are proportional to `abs(w_q) * (trace(exp(t_q F_oo/2)) *
trace(exp(-t_q F_vv/2)))**p`. Set `triples_controls['trace_exponent']` to choose
`p`; its default is **2.1**, while the historical ten-point benchmark used 1.4.
The proposed 3/2 scaling from the MP2 exponent is heuristic: denominator gap
count alone does not determine the sampling variance. The resolved exponent
is recorded in checkpoint controls and diagnostics. It changes only the smooth
stochastic triples allocation, not MP2 or the physical energy/force expressions.

Each point uses **10000 independent pilot samples**, configurable through
`pilot_samples`. Its initial production count is `ceil(w_q**2 * pilot_variance / V_q)`,
with a minimum of 10000. Pilots use the same proposal as production, including
the fused VJP proposal for force calculations. Their estimates are discarded;
production uses independent streams. With the default `adaptive=True`, native
production continues until its measured SE is at most `sqrt(V_q)/abs(w_q)`.
Energy and cotangents include every production extension and use the same final
normalization. Thus the summed reported SE meets the total target, although
sample SE remains an estimate rather than a bound on actual error. Set
`adaptive=False` to retain the historical fixed-after-pilot count behavior.
The flag is saved in checkpoint identity. Per-point target/observed variances,
initial/final counts, extension counts and energies are recorded.

For positive tolerances, each point's dressed tensors and pullback are prepared
once. One native call builds its four-index integrals and probability tables,
runs the discarded pilot, then reuses that preparation for production and all
extensions. The pilot uses the force proposal without accumulating discarded
cotangents. The point's pullback is applied and its intermediates released
before the next point is prepared. Diagnostics mark this as
`point_preparation_reused=True`. Gradient rounds have one 2048-draw batch per
thread in the observed OpenMP setup team; the T2 adjoint remains shared.

The asymmetric proposal mixes normalized left/right squared contraction-term
scores (95%) with uniform tuples (5%). Three-index factor tables are shared
between permutation channels; no six-index table is formed. Four additional
S-projected T2 blocks and their adjoints are shared, with output-tile ownership
for race-free reverse accumulation. `resident_tensor_bytes`, `proposal_bytes`
and `reverse_buffer_bytes` describe native arrays only; process RSS also
includes binding, JAX, CCSD and runtime storage. `asymmetric_dressing` records
which kernel ran. The CLI option is `--triples-dressing equal|asymmetric`.

Without a positive tolerance, `nsamples` remains the production budget per S,
distributed by estimated weighted standard deviations, with a two-sample
minimum and rounding. This legacy allocation requires all point pilots and
retains its older preparation path. Energy SE combines independent point/fragment variances;
it does not estimate force errors.

The stochastic VJP estimates the derivative of the summed energy. Proposal
probabilities and pilot allocations are sampling controls; their derivatives
are excluded from the physical force estimator. Validate the estimator against
deterministic forces and across independent seeds.

`examples/dlno_stc/smooth_force_compare.py` runs at 32 threads with an 800,000 MB
PySCF memory cap. It records the quadrature, clusters, each force array, actual
sample counts, and Cartesian force RMS errors in mEh/bohr against conventional
deterministic CCSD(T). Independent seeds are required to assess variability.
For a single-S comparison, use `--sampling-fragment INDEX
--fragment-deterministic`; the saved full reference still supplies the common
setup, while the new deterministic calculation supplies that S correction's
force. `--seeds` with no values runs only the requested fragment reference.

No gap cutoff is enlarged and no physical orbital response is frozen by this
option. The change is an alternative basis and a consistent full-Fock energy
and derivative formulation, with separately controlled quadrature error.
