# Smooth-basis CCSD(T): theory and options

Updated 2026-10-29. This README explains the formulation, the options introduced
during development, and the selected production configuration. Section 7 is the
options reference; section 10 gives runnable examples.

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

The selectable formulations include conventional or smooth orbital frames,
equal or asymmetric triples dressing, and native or JAX dressing backends.
Sampling supports fixed counts or a total energy SE target, with MP2 or CCSD
fragment weights and configurable Laplace-point allocation.

All numerical benchmark runs use **32 threads** and an **800,000 MB memory cap**,
leaving room within the approximately 900 GB available on the host.

### Current production choices

| Setting | Selected configuration |
| --- | --- |
| Physical groups and LIS | Heavy-atom S groups; occupied and virtual truncation supported |
| Partial-S triples contraction | `dressing='asymmetric'`, direct contraction over S |
| Dressing backend | `dressing_backend='native'` |
| Triples quadrature | 10 fitted Laplace points per S |
| Pilot | 10,000 independent samples per point |
| Total molecular (T) energy SE target | 0.05 mEh, allocated as variances over S and points |
| Variance proxy across S | Absolute reference MP2 energies; saved CCSD energies can override |
| Point trace exponent | Explicitly set `trace_exponent=5.0`; the library/runner default remains 2.1 |
| Production stopping | `adaptive=True`, extend until each reported point SE meets its budget |
| Gradient rounds | One 2048-draw batch per active OpenMP worker, up to 32 batches per round |
| Threads / PySCF memory cap | 32 / 800,000 MB |

Native is the selected production backend because a calculation usually runs
once. Avoiding the initial JAX compilation cost and bounding worker scratch are
useful for that workload. Both backends retain the full Fock and target-orbital
response described in section 5.

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

The **saved AO coefficient anchors** are the reference orbital coefficient
matrix: rows label atomic basis functions and columns label reference orbitals.
These numerical coefficients are saved once and held constant during the local
derivative. The current orbitals still respond through the projectors, AO
overlap and normalization.

The implementation first projects these anchors into the
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

### Convergence when forces are requested

Requesting forces does **not** automatically tighten the forward CCSD
convergence settings in this checkout. The CCSD, DF-CCSD and LNO RCCSD classes
inherit the installed PySCF class defaults. The comparison runner explicitly
overrides them:

| Criterion | Library default | Current comparison runner |
| --- | ---: | ---: |
| Energy change, `conv_tol` | `1e-7 Eh` | `1e-10 Eh` |
| Packed amplitude-update norm, `conv_tol_normt` | `1e-5` | `1e-8` |
| Maximum iterations, `max_cycle` | 50 | 200 |
| DF-CCSD Lambda/adjoint update norm | Uses `conv_tol_normt`: `1e-5` | Uses `conv_tol_normt`: `1e-8` |

Both forward conditions must pass: `abs(delta_E) < conv_tol` and
`norm(delta_amplitudes) < conv_tol_normt`. The latter is the Euclidean norm of
the packed t1/t2 update before DIIS, with no division by the number of
amplitudes. The DF custom-response path also compares its packed Lambda update
norm against `conv_tol_normt`. It is enabled by default in this checkout.
There is no separate convergence threshold expressed as a nuclear-force error
in mEh/bohr, and the stochastic energy SE target does not provide one.

The runner accepts `--cc-conv-tol` and `--cc-conv-tol-normt`. See the
[forward convergence check](pyscfad/cc/ccsd.py),
[DF response settings](pyscfad/cc/dfccsd.py), and
[runner overrides](examples/dlno_stc/smooth_force_compare.py).
The tighter runner settings leave the inherited library defaults unchanged.

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
including that small quadrature residual. This original C5H6 benchmark used a
44-point triples grid; the current production grid has ten fitted points per S.

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
point accepts that full matrix and returns its full cotangent. The selected
dressing backend differentiates the similarity transform, and the outer
pullback differentiates `M = U U^T`, including target response.
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
returns derivatives of the common tensors, L and R; the selected dressing
backend then differentiates the exponentials and physical S coefficients.

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

The coincident-eigenvalue limit is finite. The native implementation evaluates
close differences with `expm1`; the JAX reference uses a stable series near
coincidence. This differentiates the matrix function as a whole; general
background is given by [Al-Mohy and Higham (2009)](https://eprints.maths.manchester.ac.uk/1218/).

The matrices are evaluated block by block. Fock elements between different
spectral blocks vanish analytically; the frame derivative carries the response
between those blocks.

### Native and JAX dressing backends

`dressing_backend='native'` is the production default. It reuses MP2's stable
matrix-exponential and reverse-derivative routines. `dressing_backend='jax'`
retains the earlier implementation as an alternative and reference. Both
implement the same tensor transforms and full Fock/target derivatives, for
whole-LIS, equal and asymmetric contractions.

**Warmed JAX** means an executable has already been compiled for compatible
shapes, dtypes and static cluster layouts. It still evaluates fresh tensors
and gradients on each call. Rebuilding the solver or changing these layouts
can require compilation again. Warmed is an execution state, not a third
backend or a different approximation. Native is our selected option for
calculations that usually run once.

Full T2 cotangents remain shared. Native dressing uses one shared staging tensor
and bounded worker panels; the asymmetric reverse assigns output tiles to
workers. Neither requires a full T2 gradient copy per thread. This does not
remove the memory cost of the shared full tensors and CCSD intermediates.

## 6. Stochastic estimation and force calculation

The current tolerance-controlled sampler uses **10 fitted triples Laplace
points per S and 10,000 independent pilot samples per point**. The positive
grid is fitted over each retained LIS's full triples-denominator interval;
both `1/D` and `1/D^2` enter the fit. The actual quadrature errors are recorded
separately from sampling SE. Earlier saved benchmarks used the older
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
driver. The historical ten-point benchmark used `p = 1.4`. Scaling the
MP2 exponent by `3/2` reflects the three gaps in a triples denominator versus
two in MP2, but is a heuristic: the sampling variance also depends on amplitudes,
integrals and the sampling proposal. It does not establish an optimal exponent.
Log-space evaluation removes sensitivity to a common orbital-energy shift.
The resolved exponent is saved in checkpoint controls and triples diagnostics.
The selected production setting is explicitly `p = 5.0`, based on the
C6H14 scan; it is not a universal optimum. This control applies to smooth
stochastic triples; the MP2 model is unchanged.

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
are retained by this path. This reuse works with any supported trace exponent;
`point_preparation_reused` records the execution policy.

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
are saved in `triples_diagnostics`.
The native fields `independent_pilot_*` describe the discarded stream;
`pilot_samples` and `pilot_standard_error` retain their historical meaning of
the initial production block. The smooth wrapper records these separately.

Without a positive energy tolerance, the older fixed-count mode remains:
`nsamples` is the production budget **per physical group**, allocated as
`N_q proportional to |w_q| sigma_q`, with a two-sample minimum and rounding.
`pilot_samples` is explicit and defaults to 10000 per point in the smooth path.
This legacy fixed-count allocation still needs pilots from all points and
retains the older preparation path; it does not use per-point reuse.

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

## 7. Options and when to use them

The tables distinguish Python API defaults from the comparison runner's
settings. The runner is
[`examples/dlno_stc/smooth_force_compare.py`](examples/dlno_stc/smooth_force_compare.py).
Its smooth calculation uses stochastic (T); it also computes or loads a
conventional deterministic reference with matching LIS truncation.

### Orbital spaces and physical groups

| Option | Choices and meaning |
| --- | --- |
| `smooth_gap` in `build_static_selections` | API default `None`: ordinary semicanonical orbitals. A positive value enables anchored smooth Fock blocks. The runner's `--smooth-gap` defaults to `0.01 Eh`; zero is invalid. |
| `--boys-grouping` | `molecule` (runner default): one physical S; `atom`: heavy-atom groups with attached hydrogens; `singleton`: one Boys occupied target per S. |
| `matrix_kind` | API supports `W` or `D` selection. This comparison runner uses deterministic W. |
| `thresh_occ`, `thresh_vir` | API thresholds for retained selection eigenvalues. Runner flags are `--lno-thresh-occ` and `--lno-thresh-vir`, both default zero for untruncated LISs. C6H14 uses `1e-4` and `1e-5`. All physical S occupied directions are retained. |
| `--finite-domains` | Use finite Boys preparation domains. Without it, the runner uses full preparation domains. This choice is separate from subsequent LNO truncation. |
| `--frozen` | Frozen occupied count; runner default zero. Examples use 5 for C5H6 and 6 for C6H14. |

A small positive `smooth_gap` that leaves every Fock block a singleton is a
useful control for the effect of internal orbital rotations. It retains the
Laplace formulation. The original semicanonical inverse-gap derivative cutoff
and `smooth_gap` have different meanings, as explained in section 2.

### Triples formulation and backend

These API fields belong to `triples_controls` passed to `ccsd.kernel` with
`ccsd_t=True`.

| API option | Choices, defaults and runner equivalent |
| --- | --- |
| `mode` | `deterministic` is the API default; `stochastic` samples the numerator. In a smooth frame, deterministic mode enumerates each Laplace node. The runner adds this small-system check with `--smooth-deterministic`. |
| `dressing` | Smooth default `asymmetric`: contract directly over S with distinct left/right projected blocks. `equal`: use the full similarity-dressed weight matrix. CLI: `--triples-dressing`. |
| `dressing_backend` | Smooth default `native`; `jax` retains the earlier backend. CLI: `--triples-dressing-backend`. |
| `smooth_laplace_points` | Selection argument, **not** a triples-control field. Default `10` fits a positive grid to each LIS. API value `None` selects the older adaptive Gaussian grid. CLI: `--triples-laplace-points`, a positive integer. |

Equal and asymmetric dressing give the same exactly summed expression. Their
finite-sample estimators and variances differ, so equal energy SE does not imply
equal force error. The full occupied-target identity case uses the whole-space
kernel. Native/JAX is an independent implementation choice, available with
either dressing. Deterministic controls accept only `mode`, `dressing` and
`dressing_backend`; sampling fields must be omitted.

### Sampling and total SE control

| API option | Default or selected value | Runner flag and effect |
| --- | --- | --- |
| `energy_tolerance` | API omission/zero: fixed count. Selected target `5e-5 Eh = 0.05 mEh`. | `--triples-energy-tolerance-mha`, default `0.05`; controls the total reported (T) energy SE over the requested S. |
| `nsamples` | Required integer at least 2 in stochastic API controls, including tolerance mode. | `--samples`, default `16446469`; sets the nominal production count **per S only when the tolerance is zero**. Positive tolerance determines counts from variance instead. |
| `pilot_samples` | Smooth default `10000` per point. | `--triples-pilot-samples`; independent pilots are discarded from production estimates. |
| `fragment_energies` | Default: absolute reference MP2 energies saved during selection. | `--triples-budget-energy mp2` (default) or `ccsd`. CCSD proxies come from the deterministic reference. API overrides supply one reference energy per physical S. |
| `trace_exponent` | Library/runner default `2.1`; selected production setting `5.0`. | `--triples-trace-exponent`; changes the allocation of variance among Laplace points. |
| `adaptive` | Smooth default `True`. | `--triples-adaptive` extends production until measured point SEs satisfy their budgets. `--no-triples-adaptive` fixes counts after pilots and can miss the target. |
| `global_seed` | API default `71`, with fragment-derived streams. | `--seeds`, runner default `71 72 73`; runs separate force estimates. |

The exponent choices made during development were:

- **1.4:** reuse the original MP2 trace-model exponent.
- **2.1:** scale 1.4 by 3/2 for the three excitation gaps in (T). This is a heuristic.
- **5.0:** explicit production choice informed by the C6H14 pilot scan. Changing
  molecule, LIS or sampling proposal can change the preferred value.

The trace allocation assigns all point budgets before preparing their
integrals. In positive-tolerance mode, each point's preparation is reused for
its pilot, production and adaptive extensions for **any** of these exponents.
The selected 5.0 setting does not require a second integral transformation.

Batch size is currently an implementation setting: 2048 draws per active
OpenMP worker per round, so 32 workers give up to 32 batches per round.
The runner fixes numerical threads at 32 and PySCF's memory cap at 800,000 MB;
these are not additional CLI flags in this runner.

### Reference, subset and convergence controls

| Runner option | Meaning |
| --- | --- |
| `--reference PATH` | Load a matching deterministic reference; omit it to compute one. Match geometry, basis, frozen core, grouping and truncation. |
| `--sampling-fragment INDEX` | Evaluate one zero-based physical S correction and gradient. Requires a saved full reference for setup; skips the full-molecule force comparison. |
| `--fragment-deterministic` | Also compute the matching deterministic force for the selected S. Use with `--sampling-fragment`. |
| `--canonical-stochastic` | Add the conventional semicanonical stochastic estimator using the first seed. |
| `--selection-only` | Save selections and metadata without CCSD or force calculations. |
| `--seeds` with no values | Skip stochastic evaluations, for example when generating a fragment reference. |
| `--fd-coordinate ATOM AXIS`, `--fd-step` | Central finite-difference check of deterministic energies using fixed selections; zero-based indices and default step `1e-4 Bohr`. |
| `--cc-conv-tol`, `--cc-conv-tol-normt` | Runner defaults `1e-10 Eh` and `1e-8`. See section 4 for forward and response convergence. |

The single-S mode cannot be combined with `--smooth-deterministic`,
`--canonical-stochastic` or `--fd-coordinate`. Its energy tolerance applies
to the **selected S**, because allocation normalizes over the requested subset.
To reproduce S4's share of the original molecular 0.05 mEh budget, use
`--sampling-fragment 3 --triples-energy-tolerance-mha 0.02421051058826991`.
Setting 0.05 for that subset instead gives S4 the entire 0.05 mEh target.

## 8. Validation and the corrected reference

Forces are `-dE/dR`. The reported force error is the RMS over all Cartesian
components, with energies in Hartree and coordinates in Bohr:

$$
\mathrm{RMS}_{F}=1000\sqrt{\frac1{3N_{\mathrm{atoms}}}
\sum_{A,\alpha}\left(F_{A\alpha}^{stoch}-F_{A\alpha}^{det}\right)^2}
\quad\mathrm{mEh/bohr}.
$$

The saved checks include square H4, frozen-core water, a water dimer with
multiple partial S targets and truncated occupied/virtual LISs, and six
heavy-atom S groups for C6H14/def2-SV(P). Smooth and conventional deterministic
forces agree in the small-system checks. The C5H6 singleton-versus-smooth
comparison supports orbital-response noise as a major source of force error.
Energy SE remains distinct from force uncertainty.

The latest saved focused regression run passed **264 tests**. Coverage includes
projector response at internal degeneracy, moving frames, full-Fock MP2,
nonzero Fov, partial targets, and native/JAX dressing values and derivatives.
See the [validation log](results/smooth_c6h14_s4_optimized/validation.log) and
[identical-input derivative comparison](results/smooth_c6h14_s4_optimized/solver_bar_comparison.json).
The latest native nuclear-force comparison covers S4; earlier full-molecule
comparisons are linked in section 12. The independent complete S4 force runs
have a small difference not yet attributed to a specific upstream response
stage; identical-input triples derivatives agree to numerical precision.

### Deterministic (T) pullback: OpenMP linkage correction

The original local `libcc_vjp.so` linked the system `libgomp.so.1`, while its
reduction helper in `libvjp_util.so` used the wheel's
`libgomp-e985bcbb.so.1.0.0`. During a **32-thread** triples pullback, the helper
saw **one thread** and failed to combine other workers' gradient contributions.
Forward energies were unaffected; deterministic forces were incorrect.

The repair changed the build/link configuration and rebuilt the shared
libraries against the same OpenMP runtime. **No deterministic C/C++ numerical
kernel or Python pullback implementation was changed to fix this bug.**
A Python regression test, `test_existing_deterministic_triples_vjp_reference`,
was added against an independent JAX reference. The smooth-frame, asymmetric
triples and native dressing features are separate numerical source changes.

Corrected force references were regenerated. The original
`6.688929 mEh/bohr` force-error comparison is invalid and superseded. See the
[correction record and rebuild instructions](results/REFERENCE_CORRECTION.md)
for affected result directories and the verification procedure.

## 9. Code map

| File | Role |
| --- | --- |
| [smooth.py](pyscfad/dlno_stc/smooth.py) | Projectors, anchored frames, matrix-exponential derivative, full-Fock MP2 |
| [smooth_triples.py](pyscfad/dlno_stc/smooth_triples.py) | Laplace grid, tensor dressing, sample allocation, triples pullback |
| [lis.py](pyscfad/dlno_stc/lis.py) | Smooth-frame metadata and orbital assembly |
| [ccsd.py](pyscfad/dlno_stc/ccsd.py) | Opt-in selection and solver integration |
| [LNO CCSD core](pyscfad/lno/ccsd.py) | Hooks for the MP2 and triples solvers |
| [Native bindings](pyscfad/dlno_stc/cpp/stc_triples_bindings.cpp) | Whole-LIS and general occupied-weight energy/VJP entry points |
| [Native dressing](pyscfad/dlno_stc/cpp/stc_triples_dressing.cpp) | Cached cluster spectra, bounded tensor panels and explicit dressing pullback |
| [MP2 exponential/VJP helpers](pyscfad/dlno_stc/cpp/stc_mp2_gradient.cpp) | Shared stable matrix-exponential and derivative routines |
| [Asymmetric native kernel](pyscfad/dlno_stc/cpp/native/stc_triples_asymmetric.C) | Direct S projection, factor sampling and shared-gradient reverse |
| [Comparison runner](examples/dlno_stc/smooth_force_compare.py) | Deterministic reference, stochastic seeds, force output and RMS |
| [Focused tests](pyscfad/dlno_stc/test/test_smooth.py) | Numerical derivative validation |
| [Native dressing tests](pyscfad/dlno_stc/test/test_triples_dressing.py) | All input derivatives, degeneracy, cluster layouts and panel boundaries |

Restart serialization stores per-group frame and quadrature metadata. CC
selection schema 4 requires rebuilding older CC selections/checkpoints.
The original whole-LIS implementation reused the supplied native numerical
sources unchanged. The multi-group extension modifies `stc_triples.C/.h` to
accept nonsymmetric occupied weights, differentiate them, and provide a small
deterministic weighted reference. Existing selected-orbital entry points retain
their original interface; the binding identity includes the new source hashes,
including native dressing and the reused MP2 gradient helpers. Resolved dressing
backend, exponent and adaptive controls participate in checkpoint identity.

## 10. Running the calculation

For an existing mean-field calculation, domains and MP2 controls, the Python
API enables the smooth frame and chooses its triples grid during LIS selection:

```python
selected = ccsd.build_static_selections(
    mf, domains,
    matrix_kind="W", thresh_occ=0.0, thresh_vir=0.0,
    baseline_controls=baseline, matrix_controls=matrix,
    smooth_gap=0.01, smooth_laplace_points=10,
)

# Pass this dictionary to ccsd.kernel(..., ccsd_t=True,
#                                     triples_controls=triples_controls).
triples_controls = dict(
    mode="stochastic", nsamples=262144,  # Required; target mode chooses counts.
    energy_tolerance=5e-5,              # Hartree: 0.05 mEh total over requested S.
    pilot_samples=10000, global_seed=71,
    trace_exponent=5.0, adaptive=True,
    dressing="asymmetric", dressing_backend="native",
)
```

For the configured environment on this host:

```bash
cd /home/zhenling/scratch/pyscfad
export LD_LIBRARY_PATH=/tmp/pyscfad-armadillo/lib:/home/zhenling/intel/oneapi/mkl/2025.1/lib/intel64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
```

The runner sets 32 numerical threads and an 800,000 MB PySCF memory cap.
Use a fresh output directory to preserve earlier results.

### Current C6H14 production configuration

After the environment exports above, the following runs all six physical S
with the current selected options and the saved matching deterministic force
reference:

```bash
/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c6h14.xyz \
  --basis 'def2-SV(P)' --frozen 6 --boys-grouping atom \
  --lno-thresh-occ 1e-4 --lno-thresh-vir 1e-5 --smooth-gap 0.01 \
  --triples-dressing asymmetric --triples-dressing-backend native \
  --triples-laplace-points 10 --triples-pilot-samples 10000 \
  --triples-trace-exponent 5.0 --triples-budget-energy mp2 \
  --triples-energy-tolerance-mha 0.05 --triples-adaptive \
  --cc-conv-tol 1e-10 --cc-conv-tol-normt 1e-8 --seeds 71 \
  --reference results/smooth_c6h14/canonical_deterministic.json \
  --output-dir results/smooth_c6h14_native_production
```

For the matching single-S comparison, add `--sampling-fragment 3
--fragment-deterministic`, change the target to `0.02421051058826991`, and choose
a separate output directory. This computes the S4 correction and its force
error against a deterministic S4 reference.

### One full LIS and a singleton-block control

For C5H6, generate a deterministic reference, a conventional stochastic result
and three smooth stochastic results at the current total SE target:

```bash
/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/c5h6_force_compare.xyz \
  --basis cc-pvdz --frozen 5 --boys-grouping molecule --smooth-gap 0.01 \
  --triples-energy-tolerance-mha 0.05 --triples-trace-exponent 5.0 \
  --canonical-stochastic --seeds 71 72 73 \
  --output-dir results/smooth_c5h6_repeat
```

For the singleton-block control, change `--smooth-gap` to `1e-6`, omit
`--canonical-stochastic`, and use a separate output directory. The corrected
reference can be reused with `--reference
results/smooth_c5h6/canonical_deterministic.json`.

### Square H4 with fixed production counts

This explicitly disables the energy-tolerance allocator and adds a smooth
deterministic comparison:

```bash
/tmp/pyscfad-venv/bin/python examples/dlno_stc/smooth_force_compare.py \
  --xyz-file examples/dlno_stc/square_h4.xyz --basis cc-pvdz \
  --smooth-deterministic --triples-energy-tolerance-mha 0 \
  --samples 262144 --seeds 71 \
  --output-dir results/smooth_h4_repeat
```

The nominal 262144 production samples are allocated among Laplace points for
each S; rounding and point minima can increase the total. The default 10000
independent pilot draws per point are additional.

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
- The latest native backend was checked on S4 and small derivative fixtures.
  Full six-S validation with that backend and memory requirements for
  300–500-orbital LISs remain to be established.

## 12. Saved results

- [C5H6 report](results/smooth_c5h6/README.md)
- [Combined summary](results/smooth_c5h6/summary.json)
- [All force components and errors](results/smooth_c5h6/force_components.csv)
- [Singleton-block control](results/smooth_c5h6_small_blocks/comparison.json)
- [H4 validation](results/smooth_h4_verified/comparison.json)
- [Water validation](results/smooth_water_verified/comparison.json)
- [Multiple-group, truncated-LIS validation](results/smooth_water_dimer_partial/README.md)
- [C6H14/def2-SV(P), six truncated LISs](results/smooth_c6h14/README.md)
- [Ten-point variance allocation](results/smooth_c6h14_t005_10pt/README.md)
- [Full-molecule exponent-5.0 adaptive run](results/smooth_c6h14_p50_adaptive/README.md)
- [Direct-S asymmetric dressing comparison](results/smooth_c6h14_s4_asymmetric/README.md)
- [Native dressing derivative comparison](results/smooth_c6h14_s4_optimized/solver_bar_comparison.json)
- [Native-library correction and rebuild instructions](results/REFERENCE_CORRECTION.md)

The measured outcome is that smooth blocks substantially reduce stochastic
force noise while preserving deterministic forces in the validation cases.
