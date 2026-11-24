# Smooth LIS frames, Laplace triples and response

This is the theory and implementation companion to the
[DLNO-STC user guide](../../../pyscfad/dlno_stc/README.md). Installation, the four
calculation workflows, Python controls, defaults, storage choices, MPI and
command-line examples are maintained there.

The CC calculation uses W-selected retained spaces, one smooth occupied frame
and one smooth virtual frame per LIS, and MRCC deterministic or asymmetric stochastic Laplace triples with full
Fock response. “Full frame” means the complete retained LIS; it does not disable
LNO truncation or physical fragment weighting. Standalone domain MP2 uses its
scalar energy solver without constructing W.

Parent PAOs use the squared-overlap-norm cutoff `p.T @ S @ p > pao_norm`,
with default `pao_norm=1e-4`. This matches the former `1e-2` norm default.
Square other old norm cutoffs to preserve their physical selection, and
rebuild saved selections/checkpoints from the previous convention.

## Energy and shared W response

For physical fragment S, the correlation energy is

```text
E_corr = E_MP2,weak + sum_S [E_MP2,strong,S + E_CCSD,S + E_(T),S - E_MP2,LIS,S].
E_MP2,strong,S = (Tr W_occ,S + Tr W_vir,S) / 2.
```

In CC calculations, one W forward supplies the strong-domain MP2 contribution
and the LIS selection. Its delayed reverse receives unit MP2 energy cotangent
and the occupied/virtual W cotangents from the correction. Direct orbital and
Fock cotangents accumulate through the shared preparation. The weak-pair
response and the final common-data/SCF response complete the derivative.
With stochastic W, energy and response reuse the saved production samples;
the trace covariance enters the reported MP2 baseline SE.

## Frames and physical selection

LNO eigenproblems use only the orthonormal extended-domain occupied and virtual
frames. Occupied selection retains normalized domain projections of the original
Boys fragment targets and diagonalizes the density in their orthogonal
complement. Every LIS orbital stays within its corresponding domain span;
zero cutoffs retain the full corresponding domain space. Domain frames,
projected targets and retained W spaces remain differentiated.

CCSD receives the local occupied count and only the active LIS columns. The
global AO Fock is projected into this basis, including occupied-virtual entries.
A retained occupied/virtual subspace is transported against fixed AO anchors.
`select_frame` saves semicanonical reference anchors; `rebuild_frame` transports
them with the current LIS and AO overlap. The transport itself contains no
Fock matrix. Discrete ranks, anchors and quadratures are held fixed for the
fragment's derivative. Their lifecycle is documented in the
[user guide](../../../pyscfad/dlno_stc/README.md#separate-mp2-and-triples-quadratures).

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

The retained W subspace and the AO overlap still respond to geometry. Their
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
derivatives. A truncated retained/discarded W boundary still has its own
physical subspace response; the anchored frame does not remove that response
or make a changing LIS rank differentiable.

## Asymmetric Laplace triples

The default implementation dresses only external numerator legs. At time t,
`O = exp(t (F_oo - mu I)/2)` and `V = exp(-t (F_vv - mu I)/2)`, with a common
shift mu inside the occupied–virtual gap. The shift cancels in each sampled
product and keeps the exponentials bounded.

Internal contracted indices are left undressed. Three T2 forms serve the
virtual contraction, occupied contraction, and disconnected terms. The first
two omit the transform on their internal index. Likewise, ovvv and ovoo omit
the internal transform. This cancels reciprocal factors algebraically before
floating-point evaluation, avoiding instability for broad core–valence spectra.

The physical selected-space projection is applied directly to each raw tensor:
`U^T` on the left and `U^T O^2` on the right. No inverse exponential is formed.
Both partial and whole-LIS targets use this formulation. Reverse mode includes
full off-diagonal Fock response and U response through matrix-function divided
differences, including degenerate spectra. It differentiates the simplified
expression, so canceled internal factors do not introduce unstable derivative
cancellations.

The smooth LIS anchors are semicanonical when first selected. Transport with
reused anchors does not recanonicalize them, so later evaluations can have
non-diagonal Fock blocks. At each node, the dresser uses scalar virtual factors
`exp(-tau (epsilon_a - mu))` whenever `tau ||Fvv - diag(Fvv)||_inf` is at most
32 float64 machine epsilons (using the symmetric Fock block). Here `tau=t/2`
for both MRCC and stochastic half-dressing. This bounds the
absolute operator error in the virtual exponential at roundoff level, without
an energy-scale-dependent cutoff. Exactly diagonal blocks also avoid the
virtual eigensolve. Noncanonical nodes use the original dense matrix transform.
The same scalar scaling is used for tensor cotangents and tape reconstruction.
The matrix cotangent remains **full**, and its divided-difference pullback
retains off-diagonal virtual Fock derivatives, including at degeneracies.
`diagonal_virtual_points` reports how many quadrature nodes used this path.
The optimization is automatic for both smooth triples modes and requires the
rebuilt native extension with `TRIPLES_DIAGONAL_VIRTUAL_DRESSING` support.

`StableTriplesDressing(inputs)` takes nine arrays:
`t1, t2, ovvv, ovoo, ovov, fov, Foo, Fvv, U`. It prepares eight base tensors
(including three T2 forms) and eighteen smaller projected tensors. Relative to
the old numerator, this adds two full T2 forms and four T2 projections. It
requires no additional full ovvv copy or six-index triples tensor. Only the
current quadrature node retains a reverse tape, bounded by available memory;
otherwise the dresser recomputes intermediate transforms during reverse.

Deterministic evaluation uses the MRCC half-Laplace partition through
`triples_mrcc_point`. Each point builds its connected numerator once and uses
a combined connected cotangent for the grouped reverse contractions. Occupied
blocks are cached in batches of at most 128, bounded further by
`PYSCFAD_TRIPLES_NUMERATOR_CACHE_MB`. No fifth-order global numerator is retained.
The combined numerator-cache and worker-scratch allowance defaults to 16 GiB.
Set `triples_controls={"mode": "deterministic", "workspace_gib": 64}` to request
64 GiB, still bounded by the `max_memory` headroom calculated before the native
call. `workspace_gib` must be finite and positive; it is not a total-process
memory limit. Shared tensors and library allocations are additional, and the
minimum one-worker/one-block allocation may exceed a very small allowance.
The `workspace_budget_bytes` diagnostic records the largest effective budget.
This requires rebuilding the extension with `TRIPLES_CONFIGURABLE_WORKSPACE`.
The benchmark CLI exposes the same setting as `--triples-workspace-gib 64`.
The leading connected forward work is
`12 * nq * nS * no*(no+1)/2 * nv**3 * (nv+no)` floating-point operations.
The implementation preserves full occupied and virtual Fock response, including
when forward semicanonical factors use scalar scaling.

The old asymmetric deterministic and quadrature-cache APIs have been removed.
`triples_controls=None` now selects MRCC. Stochastic evaluation retains the
asymmetric partition and the `triples_prepared_energy[_vjp]` sampler. Its proposal
probabilities are held fixed during differentiation, with the same draws for
energy and cotangents. Scalar exhaustive evaluation remains a small-system
reference for this sampler, not a deterministic production backend.

MRCC and asymmetric selected-fragment energies need not agree when the occupied
projector and Fock matrix do not commute; their complete-system sums agree.

## Independent quadratures

The MP2 fit uses double-excitation denominator bounds from the full active
canonical spectrum, excluding frozen orbitals. Every fragment uses that same
saved rule. The triples fit uses the triple-excitation denominator range of
each nonempty selected LIS. These are separate approximation problems.

Both fits account for energy and derivative accuracy. In particular, the
positive triples fit approximates inverse first and second denominator powers;
its derivative weight is fixed at `0.1`. Fit diagnostics report these errors
separately from stochastic SE. The LIS MP2 subtraction instead solves the
noncanonical MP2 equation with an implicit derivative of the full Fock operator.

## Differentiation boundaries

The controls configure the calculation before evaluation. Ranks, grouping,
anchors, quadrature rules, proposal probabilities, random draws, and allocation
decisions are not differentiated. Sampled numerator derivatives use the same
production samples and importance weights as the energy; there is no independent
gradient seed or gradient sample count. Pilots do not contribute to the returned
energy or gradient. The physical derivatives through integrals, amplitudes,
full Fock matrices, fragment coefficients, retained W subspaces, AO overlap,
Boys localization, and SCF response remain included.

Thus these are not generally “forward-only options.” In particular, the default
DF-CCSD custom response uses `conv_tol_normt` and `max_cycle` for its backward
lambda solve as well as its forward solve. The `(T)` SE target controls estimated
energy uncertainty; it does not impose a force uncertainty or RMS-error target.

## Variance allocation and sample accounting

For total requested SE T and requested fragment set I, the streaming-selection
allocation (`variance_allocation="occupied_count"`) is

```text
f_S = n_occ,S / n_active,total
T_S^2 = T^2 f_S.
```

The counts refer to original occupied fragment targets. The denominator includes
all active occupied orbitals in the molecule. I contains every physical fragment.

For previously captured concrete labels, the internal
`variance_allocation="mp2_energy"` alternative uses saved/overridden energies e_S:

```text
a_S = max(|e_S| / max_{R in I}|e_R|, 1e-12)
f_S = a_S / sum_{R in I} a_R
T_S^2 = T^2 f_S.
```

In `mp2_energy` mode, all-zero proxies use equal shares. Legacy selections without
proxies also fall back to equal shares. The total target is allocated over all fragments.

For node t_q, quadrature weight w_q, and trace exponent p:

```text
score_q = |w_q| [Tr exp(t_q F_oo/2) Tr exp(-t_q F_vv/2)]^p
f_q     = score_q / sum_r score_r
T_Sq^2  = T_S^2 f_q
native unweighted SE target = T_S sqrt(f_q) / |w_q|.
```

Thus `sum_{S in I} sum_q T_Sq^2 = T^2 sum_{S in I} f_S`. This equals `T^2`
because all fragments are evaluated. The trace model fixes variance shares before
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

## Deterministic CCSD and response

Deterministic LNO/DLNO fragments use PySCF CPU DF-CCSD iterations and NumPy/BLAS
lambda response by default. Local Fock, DF and ERI tensors pass in memory;
the PySCF disk integral builder is bypassed. JAX connects the host solver to
orbital and geometry response through the custom derivative boundary.

The lambda source includes the projected fragment energy and triples response,
with explicit spin-metric and packed-amplitude conversions. Integral adjoints
are accumulated with the orbital/Fock cotangents. Both forward convergence
criteria must pass; the amplitude tolerance and iteration limit also govern the
lambda solve. The diagonal-Fock denominators are preconditioners, while full
off-diagonal Fock entries remain in the equations.

`vvvv_direct` controls virtual-ladder caching independently of the local
half-transformed DF storage. The native solver caches a packed pair matrix
when allowed by available memory; otherwise it contracts DF factors in blocks.
Forward iterations share their cache. Backward builds a cache shared by lambda
and the integral adjoint, releasing it after the host solve. Packed raw `ovvv`
integrals remain resident with either ladder policy. The optional C++ ladder
kernel obtains BLAS from SciPy and uses per-worker GEMM panels; other CCSD
contractions use the host implementation.

The native RCCSD adapter supports real float64 energies and first derivatives;
the JAX reference path is needed for higher derivatives. DCSD and CC2 do not
use that adapter. See the [user guide's build section](../../../pyscfad/dlno_stc/README.md#build-and-verification)
for the optional extension and configuration switches, and
[SCF/DF performance notes](../../../README_SERIAL_DF_SCF.md) for the outer response.

## Diagnostics and verification

`return_components=True` provides each fragment's `e_mp2_lis`, `e_ccsd`, `e_triples`,
`triples_samples`, `triples_standard_error`, and `triples_diagnostics`. The latter
contains per-point nodes/weights, variance fractions and targets, pilot counts,
initial/adaptive/total production counts, weighted energies/variances,
`target_met`, and grid-independent sampling diagnostics. The saved selection's
`triples_grid` holds the quadrature-fit diagnostics. The MP2 baseline SE is
separate from triples SE.

The comparison runner reports

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
