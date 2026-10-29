# Smooth-basis CCSD(T): implementation, theory, and force tests

## 1. Purpose and scope

This work implements and tests a smooth orbital basis for deterministic and
stochastic CCSD(T) forces in the `dlno-stc` code. The starting source snapshot is
commit `b01b2739a54be7ea77447eb5098de392b4bf5a38`; its provenance is recorded in
[SOURCE_SNAPSHOT.txt](SOURCE_SNAPSHOT.txt).

The implemented option supports **multiple physical groups S and truncated
occupied/virtual LIS spaces**, as well as a full molecular LIS. Frozen core
orbitals are supported. `boys_grouping="atom"` attaches hydrogens to heavy atoms
and assigns Boys orbitals by Löwdin populations. Each nonempty group has its own
LIS, reference anchors, Fock clusters, triples quadrature, and random stream.
The original C5H6 benchmark uses one full LIS with five frozen core orbitals,
13 active occupied orbitals, and 82 virtual orbitals.

The main changes are:

1. Construct smooth orbital frames from spectral projectors and reference anchors.
2. Use the full Fock matrices in that basis for CCSD response and MP2 subtraction.
3. Evaluate (T) through a Laplace representation with stable matrix-exponential derivatives.
4. Reuse the existing native triples contractions and their reverse derivatives.
5. Compare stochastic nuclear forces with a corrected deterministic reference.

The performance extension reuses MP2's stable exponential/VJP routines for
native block dressing and parallelizes asymmetric triples gradient grouping.
`triples_controls['dressing_backend']='native'` is the new default;
`'jax'` retains the previous dressing implementation. Both differentiate the
same full Fock and fragment-orbital response. Full T2 cotangents remain shared,
with one staging tensor and bounded per-worker panels. See the
[implementation and timing report](results/smooth_c6h14_s4_optimized/README.md).

All numerical benchmark runs use **32 threads** and an **800,000 MB memory cap**,
leaving room within the approximately 900 GB available on the host.

## 2. Why nearly degenerate orbitals can amplify force noise

For a symmetric Fock matrix in an orthonormal basis,

$$
F u_j=\epsilon_j u_j,
\qquad
u_i^T\,du_j=\frac{u_i^T(dF)u_j}{\epsilon_j-\epsilon_i},\quad i\ne j.
$$

Individual eigenvectors can therefore rotate rapidly when two eigenvalues are
close. The molecular energy can remain smooth: in an orbital-invariant,
deterministic formulation, the terms associated with internal orbital rotations
cancel when all contributions and responses are included.

A finite stochastic estimate has noisy derivatives. Incomplete cancellation
of those terms can amplify the noise through the inverse orbital-energy gap.
This concerns the response of the orbital basis; the physical triples excitation
denominator can still be well separated from zero.

The C5H6 diagnostic found a virtual-orbital gap of approximately
`6.60466e-5 Eh` between virtual orbitals 19 and 20, counting from one. Its inverse
is approximately `1.51e4 Eh^-1`. This made orbital-response noise a plausible
explanation to test, rather than a conclusion from the small gap alone.

The existing semicanonical derivative cutoff was `1e-8 Eh` in this diagnostic.
It suppresses inverse-gap rotation terms below that cutoff, so it does not
suppress this pair. The new `smooth_gap=0.01` parameter has a different role:
it groups orbitals into blocks whose **full internal Fock response is retained**.

## 3. Smooth orbital frames

### Group eigenvalues and differentiate their subspaces

At the reference geometry, adjacent occupied or virtual Fock eigenvalues are
grouped when their separation is less than `smooth_gap`. Group membership and
reference anchors are held fixed during differentiation. Chained adjacent gaps
can produce a block whose total spectral width exceeds the grouping threshold.

For a block B, construct its spectral projector

$$
P_B=U\,\mathrm{diag}(p)\,U^T,
\qquad p_i=1\text{ for }i\in B,\quad p_i=0\text{ otherwise}.
$$

In the eigenbasis, the implemented first derivative is

$$
(U^T dP_B U)_{ij}=
\begin{cases}
\dfrac{p_i-p_j}{\epsilon_i-\epsilon_j}(U^T dF U)_{ij},&p_i\ne p_j,\\
0,&p_i=p_j.
\end{cases}
$$

Only gaps across the block boundary enter this expression. Degeneracies inside
a block are harmless provided the block remains separated from its complement.

### Choose a smooth basis inside each subspace

The implementation first projects the saved AO coefficient anchors into the
current retained LIS using `C C^T S`, where C spans that LIS, `C^T S C = I`, and
S is the AO overlap matrix. This makes the construction independent of arbitrary
rotations among orbitals spanning the retained space. For an untruncated LIS,
this is the entire parent active manifold.

Within the resulting orthonormal active-space coordinates, it projects anchor
columns into each Fock block and normalizes them by Cholesky factorization:

$$
X_B=P_B A_B,\qquad L_B L_B^T=X_B^T X_B,\qquad Q_B=X_B L_B^{-T}.
$$

These columns define the smooth frame. Their normalization and the response of
the active subspaces are differentiated. Inside a block, the Fock matrix is
allowed to become non-diagonal as the geometry changes.

### Truncated LIS: retain the response of the W-selected subspace

For each physical group, all its original occupied Boys targets are retained.
The external part of W is `(I-P_internal) W (I-P_internal)`. The selected
eigenvalue labels are fixed at the reference, and a custom spectral-projector
derivative reconstructs the current retained subspace. It contains only
retained/discarded gaps; rotations between retained W eigenvectors cancel.
The internal target projector, W, the parent orbitals and AO overlap all respond.
The code also supports D selection through the same projector construction.

Let `C_L` span the current retained LIS. Reference anchors may be semicanonical
orbitals **within the reference retained LIS**. Their coefficients are saved
once. Transport them by `X = C_L C_L^T S A_0`, orthonormalize X, and then construct
the Fock-cluster frames described above. The anchors choose coordinates inside
the retained space; they do not replace W as the subspace-selection criterion.

A small retained/discarded W gap still produces a real subspace response.
This implementation neither drops that response nor increases an inverse-gap
cutoff to hide it. Splitting a degenerate W cluster at the selection boundary is
ill-conditioned. Rank changes and reselection across a hard cutoff are outside
the fixed-selection derivative.

## 4. CCSD, its response, and the MP2 subtraction

The existing CCSD equations and implicit response solver are reused in the new
frame. They receive the full occupied and virtual Fock blocks.

The amplitude updates retain the existing diagonal Jacobi preconditioner and
DIIS. Its energies are the diagonal Fock elements in the current smooth basis:
`D_ia = F_ii - F_aa` and `D_ijab = F_ii + F_jj - F_aa - F_bb` (zero level shift).
Off-diagonal Fock terms remain in the CCSD residual equations. At the anchor
geometry, the forward basis is semicanonical to roundoff. The original C5H6,
H4 and water tests therefore used the same respective CCSD iteration counts
as their conventional calculations: 38, 48 and 19. This does not establish
unchanged convergence far from the anchor geometry.

Writing the converged amplitude equations as `R(T,x)=0`, with amplitudes `T`
and molecular inputs `x`, the adjoint formulation is

$$
R_T^T\lambda=-E_T,
\qquad
\frac{dE}{dx}=E_x+\lambda^T R_x.
$$

For CCSD(T) forces, the triples derivative with respect to the CCSD amplitudes
contributes to `E_T`. The resulting response propagates through the orbitals,
Fock matrix, density-fitting integrals, and SCF response to the nuclear forces.

The DLNO energy also subtracts an MP2 energy inside the LIS. In a noncanonical
basis, this subtraction must use the full two-particle Fock operator:

$$
\mathcal D_2 t_2=-v,
\qquad
\mathcal D_2=F_{vv}^{(1)}+F_{vv}^{(2)}-F_{oo}^{(1)}-F_{oo}^{(2)}.
$$

Here each superscript identifies the tensor leg on which the matrix acts.
The new solver differentiates this linear system implicitly. Diagonalization
is used internally to solve it, while its eigenvectors are excluded from the
derivative calculation. Diagonal denominators remain useful as iteration guesses.

For this one-LIS calculation, the reported workflow energy is

$$
E=E_{HF}+E_{MP2}^{base}+
\bigl(E_{CCSD}^{LIS}-E_{MP2}^{LIS}\bigr)+E_{(T)}^{LIS}.
$$

The common deterministic MP2 baseline uses the original six-point fitted
Laplace grid. Its baseline-minus-LIS difference in the reference is
`-7.56535e-8 Eh`. Thus the benchmark reference is this complete workflow,
including that small quadrature residual. The separate triples grid described
below has 44 points.

## 5. Laplace triples in a noncanonical basis

For positive triples excitation energy

$$
D_{ijk}^{abc}=\epsilon_a+\epsilon_b+\epsilon_c-\epsilon_i-\epsilon_j-\epsilon_k,
\qquad
\frac1D=\int_0^\infty e^{-tD}\,dt.
$$

Laplace transformation provides an established route to orbital-invariant
triples expressions; see [Nagy and Kállay, JCP 146, 214106 (2017)](https://real.mtak.hu/54596/1/LaplaceT.pdf).
The following tensor dressing describes the implementation in this workspace.

At each quadrature node, define

$$
O(t)=e^{tF_{oo}/2},\qquad V(t)=e^{-tF_{vv}/2}.
$$

Applying these matrices to the three occupied and three virtual external legs
of **each** triples numerator supplies the complete factor `exp(-t D)` in their
bilinear contraction. Matrix exponentials retain the non-diagonal Fock couplings.

To reuse the native density-fitting kernel, the inputs are transformed as follows.
`Bvo[P]` denotes the code's occupied-by-virtual matrix for auxiliary index P.

| Input | Transformation at one Laplace node |
| --- | --- |
| `t1` | `O t1 V` |
| `t2` | `O` on each occupied leg and `V` on each virtual leg |
| `Bvo[P]` | `O Bvo[P] V` |
| `Boo[P]` | `O Boo[P] O^-1` |
| `Bvv[P]` | `V Bvv[P] V^-1` |
| `Fov` | `O Fov V` |

The reciprocal factors cancel the exponentials on internally contracted legs.
The transformed `Boo` and `Bvv` are generally nonsymmetric and are passed as full
matrices. The `Fov` contribution is included in both energy and derivative.

The native contraction is then evaluated with the constant denominator `-1`,
obtained by setting its placeholder occupied energies to `-1/3` and virtual
energies to zero. Physical Fock dependence is carried by the matrix exponentials.
Quadrature weights sum the node contributions.

### Partial occupied targets: similarity-dress the energy projector

For a group S, let `U` be its target-orbital coefficients in the LIS and
`M = U U^T`. The fragment energy contracts the first occupied leg of the left
numerator with the denominator-bearing right numerator through M. At node t,
the correctly dressed weight is

$$
M(t)=O^{-1} M O.
$$

In semicanonical coordinates its element is
`M_pi(t) = exp(-t epsilon_p/2) M_pi exp(t epsilon_i/2)`. Multiplying by the half
exponentials already on the two numerator legs gives
`M_pi exp(t epsilon_i)`: the denominator belongs to the right occupied index i.
Using the undressed M would give the wrong energy and Fock response whenever
M and F_oo do not commute.

M(t) is generally nonsymmetric. The `dressing='equal'` weighted-triples entry
point accepts that full matrix and returns its full cotangent. Python
differentiates the similarity transform and `M = U U^T`, including target response.
For a complete orthonormal occupied target, M is the identity and the original
whole-space native path is retained. Both CCSD and MP2 fragment energies use
the same physical target partition; MP2 solves the full Fock resolvent before
contracting its pair-energy matrix with M.

The LIS already retains the full physical S occupied span. Keeping these
orbitals as explicit columns would make the undressed M a coordinate
projector, but does not remove the similarity dressing when F_oo couples S to
the remaining LIS orbitals. An equivalent factorized expression is
`M(t) = (O^-1 U) (U^T O)`. For two numerator vectors A and B, the contraction is
`sum_s (A^T O^-1 U)_s (U^T O B)_s`, with s restricted to S. The new default
`dressing='asymmetric'` kernel implements this factorization with distinct
left/right projected tensors and their responses. Replacing M(t) by the undressed coordinate
projector would change the energy and forces unless `[F_oo, M] = 0`.

### Direct S contraction with asymmetric dressing

Define `L=O^-1 U` and `R=O U`. The common tensors retain half dressing;
additional blocks project their selected external occupied index with L or R.
Consequently that leg has **no exponential on the left** and **the full
exponential on the right**. The other five external legs retain equal half
dressing. All S/environment Fock mixing is retained, including its derivative.
No new LIS selection or rotation is needed: the existing LIS contains S.

At one node, with `A=P W` and `Z=P R_spin(W+V_singles/2)` denoting the existing
permutation sums, the contraction is

$$
E_q=-\frac13\sum_{s\in S,j,k,a,b,c}
 A_L(s,j,k,a,b,c)\,Z_R(s,j,k,a,b,c).
$$

The S index follows the selected occupied leg through every permutation.
Both the usual singles term and the Fov*T2 term are included. The native VJP
returns derivatives of the common tensors, L and R; the Python pullback then
differentiates the exponentials and physical S coefficients.

Samples now draw `(s,j,k,a,b,c)`, without an extra occupied projector index.
The proposal is a 95% mixture of normalized left/right squared contraction-term
scores and 5% uniform tuples. Each score sums squares of the individual
virtual/occupied contraction terms over the six pair permutations. Factor
tables choose a side, permutation position and internal contraction index,
then two independent three-index factors; shared block alias tables avoid a
six-index probability table. The uniform component preserves derivative
support when a numerator vanishes. The sampled energy is the integrand divided
by its full mixture probability. Probabilities are held fixed in the VJP.

Writing those scores as `q_L(x), q_R(x)` and their sums over all tuples as
`Z_L, Z_R`, the probability when both scores have nonzero mass is

$$
p(x)=\frac{0.95}{2}\left(\frac{q_L(x)}{Z_L}+\frac{q_R(x)}{Z_R}\right)
     +\frac{0.05}{n_S n_o^2 n_v^3}.
$$

A zero-mass side is omitted and the remaining score mixture is renormalized;
if both vanish, all draws are uniform. Sampling a contraction index is only a
way to generate the tuple: its internal virtual/occupied sums are subsequently
evaluated exactly for the energy and gradient.

Only S-projected blocks are added to the shared full LIS tensors. There are
four S-projected T2 blocks (two occupied positions, two sides), with shared
adjoints, rather than a full T2 gradient per thread. Reverse updates are grouped
by output tensor tile so each tile has one writer. A round contains one
2048-draw batch per active OpenMP thread; at 32 threads, at most 65536 sample
records are resident. Each point's integral/projection/proposal setup is reused
for its discarded pilot, production and adaptive extensions.

Select `triples_controls['dressing']='equal'` for the earlier estimator, or
`--triples-dressing equal` in `smooth_force_compare.py`. The default is
`asymmetric` for partial S targets in the smooth path. A whole-LIS target
continues to use the existing whole-space kernel. The choice is recorded in
checkpoint controls; `asymmetric_dressing` identifies the actual kernel in
diagnostics. `resident_tensor_bytes`, `proposal_bytes` and
`reverse_buffer_bytes` report native storage, excluding Python/JAX arrays,
binding copies, BLAS scratch and CCSD intermediates. These are not process RSS.

### Stable derivatives of the exponentials

For a symmetric matrix `A = U diag(a) U^T`, the implementation uses

$$
d(e^A)=U\left[\Phi\odot(U^T dA U)\right]U^T,
\qquad
\Phi_{ij}=\begin{cases}
\dfrac{e^{a_i}-e^{a_j}}{a_i-a_j},&a_i\ne a_j,\\
e^{a_i},&a_i=a_j.
\end{cases}
$$

The coincident-eigenvalue limit is finite. Close arguments use a stable series
evaluation. This differentiates the matrix function as a whole; general
background is given by [Al-Mohy and Higham (2009)](https://eprints.maths.manchester.ac.uk/1218/).

The matrices are evaluated block by block. Fock elements between different
spectral blocks vanish analytically; the frame derivative carries the response
between those blocks.

## 6. Stochastic estimation and force calculation

The current tolerance-controlled sampler uses **10 fitted triples Laplace
points per S and 10,000 independent pilot samples per point**. The positive
grid is fitted over each retained LIS's full triples-denominator interval;
both `1/D` and `1/D^2` enter the fit. The actual quadrature errors are recorded
separately from sampling SE. The earlier benchmarks below used the older
adaptive Gaussian grids (40 or 44 points).

For a total energy SE target `epsilon`, first divide the **variance** among
the globally requested physical S:

$$
f_S=\frac{|E_{\mathrm{MP2},S}|}{\sum_R|E_{\mathrm{MP2},R}|},\qquad
V_S=\epsilon^2 f_S.
$$

The default proxy is the selected-target MP2 energy returned during D/W
selection, before LIS truncation; it is saved with the reference selections.
It is used only for sampling budgets, not molecular energy accounting.
`triples_controls['fragment_energies']` may instead supply saved CCSD (or other
MP2) correlation energies, one per physical S. Signed values are converted to
magnitudes. The requested subset is normalized once, before MPI distribution
or checkpoint restoration. A tiny relative floor preserves support for zero
proxies; all-zero proxies use equal shares. These fixed reference weights are
not differentiated.

Within each S, use a configurable power of the MP2 trace-model score:

$$
b_{Sq}=|w_{Sq}|\left[\operatorname{Tr}e^{t_{Sq}F_{oo,S}/2}
                    \operatorname{Tr}e^{-t_{Sq}F_{vv,S}/2}\right]^p,\qquad
g_{Sq}=\frac{b_{Sq}}{\sum_r b_{Sr}},\qquad V_{Sq}=V_Sg_{Sq}.
$$

The default is now `p = 2.1`, configurable with
`triples_controls['trace_exponent']` or `--triples-trace-exponent` in the example
driver. The historical ten-point benchmark below used `p = 1.4`. Scaling the
MP2 exponent by `3/2` reflects the three gaps in a triples denominator versus
two in MP2, but is a heuristic: the sampling variance also depends on amplitudes,
integrals and the sampling proposal. It does not establish an optimal exponent.
Log-space evaluation removes sensitivity to a common orbital-energy shift.
The resolved exponent is saved in checkpoint controls and triples diagnostics.
This control applies to smooth stochastic triples; the MP2 model is unchanged.

Each pilot estimates the single-sample variance `s_pilot,Sq^2` of the
**unweighted** node contraction. Its initial independent production count is

$$
N_{Sq}=\max\!\left(10000,
 \left\lceil\frac{w_{Sq}^2s_{\mathrm{pilot},Sq}^2}{V_{Sq}}\right\rceil\right).
$$

Pilot and production use the same native proposal type (energy-only or fused
VJP), with distinct random streams; pilot energies and derivatives are not
included in the production estimator. The total estimate and measured variance are

$$
\widehat E=\sum_{S,q} w_{Sq}\widehat\mu_{Sq},
\qquad
\widehat{\mathrm{Var}}(\widehat E)=\sum_{S,q} w_{Sq}^2\frac{s_{Sq}^2}{N_{Sq}}.
$$

For a positive tolerance, each Laplace point is now completed in one visit:
prepare its dressed tensors and pullback, form its four-index integrals and
probability tables once, run the independent pilot, run production, apply the
pullback, then release that point's intermediates. Pilot sampling retains the
force proposal but skips accumulation of discarded tensor cotangents. The
native production stream reuses both integrals and tables through adaptive
extensions. Only one point's dressed tensors and differentiation intermediates
are retained by this path. The exponent and variance-allocation model are
unchanged; `point_preparation_reused` records the new execution policy.

Gradient sampling uses 2048 draws per logical batch and one batch per thread
in the observed OpenMP setup team per round (32 for a 32-thread team). Partial
final rounds are allowed. Batch seeds and scalar-statistics merge order stay
fixed; gradient reductions can differ by floating-point roundoff. The T2
adjoint remains shared rather than being copied in full for every worker.

The sum of target variances is exactly `epsilon^2`, up to floating-point
roundoff. Smooth tolerance-mode sampling now defaults to `adaptive=True`.
Each native production stream continues until its measured **unweighted** SE
is at most `sqrt(V_Sq)/abs(w_Sq)`. The native loop pools sample moments and
cotangents over the initial block and every extension, then normalizes energy
and derivatives by the final count. It does not rebuild proposal tables or
reuse discarded external pilots. Consequently the summed reported (T) SE is
at or below `epsilon` (usually slightly below due to minimum and batch sizes).
These are ordinary sample-SE estimates, not sequential confidence bounds or
guarantees on actual energy/force error.

Set `triples_controls['adaptive']=False` or `--no-triples-adaptive` to reproduce
the historical pilot-only production allocation. The resolved flag is part of
the checkpoint controls. Per-point nodes, weights, targets, pilot SEs, initial
and final production counts, extension counts, energies and measured variances
are saved in `triples_diagnostics`. Timings separate dressing/pullback
preparation, independent pilot sampling, the remaining native call (including
shared integral/table setup and production), and Laplace pullbacks. Native
reverse timings are nested within production and must not be added again.
The native fields `independent_pilot_*` describe the discarded stream;
`pilot_samples` and `pilot_standard_error` retain their historical meaning of
the initial production block. The smooth wrapper records these separately.

Without a positive energy tolerance, the older fixed-count mode remains:
`nsamples` is the production budget **per physical group**, allocated as
`N_q proportional to |w_q| sigma_q`, with a two-sample minimum and rounding.
`pilot_samples` is explicit and defaults to 10000 per point in the smooth path.
This legacy fixed-count allocation still needs pilots from all points and
retains the older preparation path; it does not use per-point reuse.
The original C5H6 runs used 45,056 pilot samples per calculation.

The comparison runner defaults to total `0.05 mEh` and accepts
`--triples-energy-tolerance-mha`, `--triples-laplace-points`,
`--triples-pilot-samples`, and `--triples-budget-energy mp2|ccsd`.
CCSD weights come from its deterministic reference's fragment CCSD energies.

The native reverse calculation supplies derivatives with respect to its tensor
inputs. A custom vector-Jacobian product propagates them through the tensor
dressing, matrix exponentials, CCSD response, and smooth frame construction.

Sampling probabilities and allocation decisions are treated as sampling controls.
For a sum `E = sum_alpha f_alpha(x)`, sampling with probability `q_alpha` estimates
its derivative through `(partial f_alpha / partial x) / q_alpha`. Its expectation
is the derivative of the sum, assuming the proposal covers every contribution.
Consequently, the force estimator targets the derivative of the summed energy;
finite-differencing one adaptive stochastic realization is a different test.

**Energy SE does not measure force uncertainty.** We assess force accuracy
against deterministic forces and across independent seeds. The exact summed
formulation is orbital invariant up to numerical approximations, while finite
sampling variance can depend strongly on the chosen orbital frame.

## 7. C5H6 results

Configuration: user-supplied C5H6 geometry, cc-pVDZ, automatic auxiliary basis,
five frozen core orbitals, one full LIS, deterministic MP2 baseline and W
matrices. Both LNO thresholds are zero to explicitly retain the full active
space. SCF tolerances are `1e-9 Eh` and `1e-6` for its gradient criterion; CCSD
tolerances are `1e-10 Eh` and `1e-8` for its amplitude norm.

The smooth grouping threshold is `0.01 Eh`, producing 12 occupied and 69 virtual
blocks. The largest virtual block contains four orbitals. The control uses
`1e-6 Eh`, producing singleton blocks for all active orbitals.

Forces are `-dE/dR`. For the 11 atoms, the reported component RMS is

$$
\mathrm{RMS}_{F}=1000\sqrt{\frac1{33}
\sum_{A=1}^{11}\sum_{\alpha=x,y,z}
\left(F_{A\alpha}^{stoch}-F_{A\alpha}^{det}\right)^2}
\quad\mathrm{mEh/bohr}.
$$

| Calculation | Production samples | Force RMS (mEh/bohr) | Energy SE (mEh) |
| --- | ---: | ---: | ---: |
| Conventional stochastic (T), seed 71 | 16,446,469 | 5.16533950 | 0.05000179 |
| Laplace (T), singleton blocks, seed 71 | 16,446,508 | 1.22271312 | 0.02254147 |
| Smooth blocks, seed 71 | 16,446,507 | 0.07303981 | 0.02254155 |
| Smooth blocks, seed 72 | 16,446,503 | 0.07406564 | 0.02253537 |
| Smooth blocks, seed 73 | 16,446,507 | 0.07832477 | 0.02260073 |

The mean of the three smooth force arrays has RMS **0.03178465 mEh/bohr**,
using approximately three times the production sampling work of one run.

With the same Laplace sampler, nominal sample budget, and seed, changing from
singleton to smooth blocks lowers the observed force RMS by about **16.7 times**.
This supports near-degenerate orbital response as a major noise source. Several
orbital groups change, so this control does not identify virtual pair 19/20 as
the sole cause.

Relative to the conventional stochastic calculation, the observed reduction is
approximately **66–71 times**. That comparison includes changes to both the
orbital frame and the triples sampling formulation. These are measured errors
for this geometry and sample budget, rather than guaranteed force-error bounds.

## 8. Validation and the corrected reference

| Validation | Result |
| --- | --- |
| H4/cc-pVDZ: smooth versus conventional deterministic forces | RMS `3.48365e-7 mEh/bohr` |
| H4: smooth stochastic (T) and its gradient, 262,183 production samples | RMS `0.14340960 mEh/bohr` |
| Distorted water/STO-3G, frozen core: deterministic force comparison | RMS `1.62355e-8 mEh/bohr` |
| Focused smooth-method derivative tests | 6 passed |
| Selected existing regression tests | 113 passed |

The focused tests cover projector response at internal degeneracy, the matrix
exponential derivative, response through the moving frame, noncanonical MP2,
all physical triples input derivatives for non-diagonal Fock matrices with
nonzero Fov, and the conventional native triples derivative. An existing test
was adjusted to tolerate `4e-19` parallel-summation roundoff; sample counts and
scalar diagnostics still require exact equality.

The 44-point triples quadrature was checked at 10,001 denominators over the
reference spectrum. Maximum checked relative errors were `1.01e-11` for `1/D`
and `1.82e-10` for `1/D^2`. These checks address quadrature error separately
from stochastic uncertainty.

**Reference correction:** my original local native-library build mixed two
OpenMP runtimes. Its deterministic triples derivative reduction failed to
accumulate all workers. I rebuilt the libraries against a consistent runtime
and checked the derivative against an independent reference. The previously
reported `6.688929 mEh/bohr` comparison is superseded. The corrected deterministic
workflow reference energy is `-193.4464206792542 Eh`.

The corrected full C5H6 smooth calculation was tested stochastically. Direct
smooth deterministic enumeration was validated on H4, water, and small tensor
fixtures; it was not run for the full C5H6 system.

The multi-group extension additionally passes a two-group water-dimer test with
both occupied and virtual truncation and four targets in each five-occupied LIS.
Smooth deterministic forces agree with conventional deterministic forces to
`9.32013e-6 mEh/bohr` RMS. A coordinate finite difference agrees within
`2.29960e-5 mEh/bohr`. See the [partial-target validation report](results/smooth_water_dimer_partial/README.md).

For **C6H14/def2-SV(P)**, six heavy-atom groups retain 12–18 occupied and 44–70
virtual orbitals per LIS, from 19 active occupied / 87 virtual orbitals. With
W cutoffs `1e-4/1e-5` and 262144 nominal production samples per group, seed 71
gives **0.739035 mEh/bohr force RMS error** against the deterministic reference
using the same truncated LISs. The total actual production count is 1,573,060.
All six native samplers report 32 threads. See the
[hexane report and reproduction command](results/smooth_c6h14/README.md), including
the generated geometry, fixed-budget qualification and Boys saddle diagnostic.

The updated **10-point, energy-weighted variance allocation**, using trace
exponent **1.4**, was also run on the same six C6H14 LISs, with 10000 pilots per
point and a total (T) SE target
of 0.05 mEh. Seed 71 achieved **0.04185143 mEh** total energy SE and
**0.21506693 mEh/bohr** force RMS error, using 39,653,122 production samples
and 600,000 pilot samples. See the
[allocation report and per-point variance data](results/smooth_c6h14_t005_10pt/README.md).

An isolated **S4 exponent-2.1 test** then completed its correction and gradient
at the same S4 target of 0.02421051 mEh. It used 7,947,172 production samples
(30.95% fewer than 1.4) and measured a (T) SE of 0.02394365 mEh. The other five S
were not rerun. A cheap scan of saved pilots predicts a sample-count minimum
near p=5.1 for S4, or p=5.36 shared across all six S. See the
[single-S report, exponent scan and reproduction command](results/smooth_c6h14_s4_p21/README.md).

A subsequent **S4 exponent-5.0 production test** completed its correction and
gradient with 4,917,293 production samples, exactly matching the saved-pilot
prediction. This is 38.13% fewer than p=2.1, but the measured (T) SE of
**0.02597251 mEh exceeded the S4 target by 7.28%**. Counts were fixed after the
pilots; no adaptive production extension was performed. Rescaling the observed
variance predicts approximately 5.66 million samples for the target, which has
not been run. The default remains 2.1. See the
[p=5.0 result and comparison](results/smooth_c6h14_s4_p50/README.md).

The **full six-S p=5.0 rerun with adaptive production** then met every point/S
budget and the total target: reported (T) SE **0.04895394 mEh**, force-component
RMS error **0.27090431 mEh/bohr**, and 17,637,941 production samples (including
1,062,205 extensions) plus 600,000 external pilots. All calculations used 32
threads. Complete process wall time was **22 min 15.32 s**, with 267.42 s in
the outer triples phase and 387.54 s in CCSD lambda response. Compared with
the p=1.4 full run, production sampling fell by 55.52% and force-calculation
time by 13.54%; the energy target does not control force error. See the
[adaptive full-molecule report and timing breakdown](results/smooth_c6h14_p50_adaptive/README.md).

The **asymmetric S4 implementation** was compared with a fresh equal-dressing
run after the batching and preparation-reuse changes. With the same p=5.0,
ten-point grid, seed 71 and S4 target, production decreased from **5,659,399 to
620,323 samples**. Reported SEs were **0.02408279 and 0.02313872 mEh**, both
below the **0.02421051 mEh** target. The triples energy/gradient phase fell from
**74.69 to 18.33 seconds** (4.07x faster), while peak process RSS remained
similar, **33.34 vs 32.63 GiB**. The S4 nuclear correction/force took 342.03 vs
292.99 seconds; CCSD took the same 22 iterations. These are isolated S4,
single-seed measurements. Against a newly computed deterministic S4 force,
the force-component RMS was **0.06605219 mEh/bohr for equal dressing** and
**0.10758395 mEh/bohr for asymmetric dressing**. Thus the energy-driven sample
reduction gave a higher force error in this seed; the energy SE target does
not enforce equal force accuracy. See the
[asymmetric implementation report](results/smooth_c6h14_s4_asymmetric/README.md)
for the matching deterministic force comparison, native memory breakdown and
reproduction command.

## 9. Code map

| File | Role |
| --- | --- |
| [smooth.py](pyscfad/dlno_stc/smooth.py) | Projectors, anchored frames, matrix-exponential derivative, full-Fock MP2 |
| [smooth_triples.py](pyscfad/dlno_stc/smooth_triples.py) | Laplace grid, tensor dressing, sample allocation, triples pullback |
| [lis.py](pyscfad/dlno_stc/lis.py) | Smooth-frame metadata and orbital assembly |
| [ccsd.py](pyscfad/dlno_stc/ccsd.py) | Opt-in selection and solver integration |
| [LNO CCSD core](pyscfad/lno/ccsd.py) | Hooks for the MP2 and triples solvers |
| [Native bindings](pyscfad/dlno_stc/cpp/stc_triples_bindings.cpp) | Whole-LIS and general occupied-weight energy/VJP entry points |
| [Asymmetric native kernel](pyscfad/dlno_stc/cpp/native/stc_triples_asymmetric.C) | Direct S projection, factor sampling and shared-gradient reverse |
| [Comparison runner](examples/dlno_stc/smooth_force_compare.py) | Deterministic reference, stochastic seeds, force output and RMS |
| [Focused tests](pyscfad/dlno_stc/test/test_smooth.py) | Numerical derivative validation |

Restart serialization stores per-group frame and quadrature metadata. CC
selection schema 4 requires rebuilding older CC selections/checkpoints.
The original whole-LIS implementation reused the supplied native numerical
sources unchanged. The multi-group extension modifies `stc_triples.C/.h` to
accept nonsymmetric occupied weights, differentiate them, and provide a small
deterministic weighted reference. Existing selected-orbital entry points retain
their original interface; the binding identity includes the new source hashes.

## 10. Running the calculation

The Python API enables the option when selecting the LIS:

```python
selected = ccsd.build_static_selections(
    mf, domains,
    matrix_kind="W", thresh_occ=0.0, thresh_vir=0.0,
    baseline_controls=baseline, matrix_controls=matrix,
    smooth_gap=0.01,
)
```

For the configured environment on this host (these commands use the current
10-point, 0.05 mEh defaults; historical fixed-count results above are unchanged):

```bash
cd /home/zhenling/scratch/pyscfad
export LD_LIBRARY_PATH=/tmp/pyscfad-armadillo/lib:/home/zhenling/intel/oneapi/mkl/2025.1/lib/intel64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c5h6_force_compare.xyz \
  --basis cc-pvdz --frozen 5 --smooth-gap 0.01 \
  --canonical-stochastic --samples 16446469 --seeds 71 72 73 \
  --output-dir results/smooth_c5h6_repeat
```

This generates a new deterministic reference, a conventional stochastic result,
and three smooth stochastic results. The driver sets the numerical thread
limits to 32 and the memory cap to 800,000 MB.

For the singleton-block control using the existing corrected reference:

```bash
/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c5h6_force_compare.xyz \
  --basis cc-pvdz --frozen 5 --smooth-gap 1e-6 \
  --reference results/smooth_c5h6/canonical_deterministic.json \
  --samples 16446469 --seeds 71 \
  --output-dir results/smooth_c5h6_singletons_repeat
```

For the square H4 validation:

```bash
/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/square_h4.xyz --basis cc-pvdz \
  --smooth-deterministic --samples 262144 --seeds 71 \
  --output-dir results/smooth_h4_repeat
```

## 11. Current limits

- Multiple S and truncated LIS spaces are supported with fixed ranks. The new
  smooth path has serial molecular validation; distributed execution has not
  been benchmarked in this extension.
- Cluster membership, anchors, and quadrature are fixed locally. Cluster
  boundaries need finite spectral gaps, and projected anchors must remain
  independent. Longer geometry changes require reassessing these selections.
- Positive occupied–virtual excitation gaps are required for the Laplace form.
- Broad blocks can make reciprocal exponential factors poorly conditioned;
  very large spectral ranges have not been validated.
- Deterministic enumeration at every Laplace node is expensive and is intended
  mainly for small-system validation.
- Three stochastic seeds provide a useful comparison, not a comprehensive
  statistical characterization or a force-error guarantee.
- These runs establish numerical behavior for the tested cases. Concurrent
  validation jobs and cache reuse prevent interpreting their timings as a
  controlled performance benchmark.

## 12. Saved results

- [C5H6 report](results/smooth_c5h6/README.md)
- [Combined summary](results/smooth_c5h6/summary.json)
- [All force components and errors](results/smooth_c5h6/force_components.csv)
- [Singleton-block control](results/smooth_c5h6_small_blocks/comparison.json)
- [H4 validation](results/smooth_h4_verified/comparison.json)
- [Water validation](results/smooth_water_verified/comparison.json)
- [Multiple-group, truncated-LIS validation](results/smooth_water_dimer_partial/README.md)
- [C6H14/def2-SV(P), six truncated LISs](results/smooth_c6h14/README.md)
- [Native-library correction and rebuild instructions](results/REFERENCE_CORRECTION.md)

The measured outcome is that smooth blocks substantially reduce stochastic
force noise while preserving deterministic forces in the validation cases.
