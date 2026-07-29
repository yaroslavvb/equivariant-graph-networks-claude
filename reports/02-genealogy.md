# The genealogy of an idea

Machine-learned interatomic potentials have a founding equation and a founding problem. The
equation is Behler and Parrinello's

$$E = \sum_i E_i,$$

the decomposition of a total energy into transferable, atom-centred contributions, each depending
only on what lies inside a cutoff sphere. The problem is that $E_i$ is a function of a set of
neighbour vectors $\{\vec{r}_{ij}\}$ that has no canonical ordering and no canonical orientation,
while a neural network wants a fixed-length vector in a fixed frame. Everything in the following
twenty years is an answer to that problem, and there are exactly three families of answer:
**destroy the orientation information immediately** (build invariant scalars and regress on them),
**fix a frame** (choose axes from the neighbourhood itself), or **carry the orientation
information along as a typed object that transforms correctly** (equivariance).

The third answer won a decisive victory in 2021 and is now, in 2026, one design among several at
the top of the field's main leaderboard. This report traces how that happened, in five braided
strands. The strands are not independent: the descriptor strand supplied the completeness
arguments that the equivariance strand later needed, the invariant-graph strand supplied the
radial bases and smoothness discipline that every modern model still uses, and the two lineages
were formally proved to be one design space in 2022. What follows tries to be precise about what
each work actually demonstrated, as opposed to what it is now popularly credited with.

If you want the mechanisms rendered interactively rather than described, the companion tutorial
covers the same material with live computation: [the question](../tutorial/index.html#question),
[what symmetry is](../tutorial/index.html#symmetry), [descriptor
blindness](../tutorial/index.html#blindness), [the group-theoretic
machinery](../tutorial/index.html#machinery), [one equivariant
layer](../tutorial/index.html#layer), [the $\ell_{\max}$ ablation](../tutorial/index.html#lmax),
[conservative vs direct forces](../tutorial/index.html#forces), [locality and receptive
fields](../tutorial/index.html#locality), [the convex-hull decision](../tutorial/index.html#hull),
and [the genealogy itself](../tutorial/index.html#genealogy).

---

## I. The descriptor strand, 2007–2019: invariance by construction, and its ceiling

### Behler–Parrinello: the decomposition that made the field possible

Behler and Parrinello's 2007 letter (Phys. Rev. Lett. **98**, 146401, DOI
10.1103/PhysRevLett.98.146401) did two things at once, and the field has never really separated
them. The first was structural: write $E = \sum_i E_i$ and give every atom of a given element an
identical copy of a small feed-forward network. Size transferability and permutation invariance
among like atoms then come for free, from weight sharing, and the model applies unchanged whether
an atom has four neighbours or twelve. The paper is explicit that earlier neural-network potential
work (Lorenz–Groß–Scheffler; Blank et al.) had been stuck in low dimension precisely because a
fixed input layer cannot absorb a varying number of atoms.

The second was representational, and is where the ceiling was set. Cartesian coordinates are
replaced by a fixed-length vector of hand-designed atom-centred symmetry functions, damped by the
cosine cutoff $f_c(R_{ij}) = \tfrac{1}{2}[\cos(\pi R_{ij}/R_c) + 1]$: a radial family
$G_i^{1} = \sum_{j \neq i} e^{-\eta (R_{ij} - R_s)^2} f_c(R_{ij})$ and an angular family
$G_i^{2} = 2^{1-\zeta}\sum_{j,k \neq i}(1 + \lambda \cos\theta_{ijk})^{\zeta} e^{-\eta(R_{ij}^2 + R_{ik}^2 + R_{jk}^2)} f_c(R_{ij})f_c(R_{ik})f_c(R_{jk})$.

Rotational invariance here is achieved by hand, not by group theory: the only geometric primitives
ever exposed to the network are interatomic distances and bond-angle cosines, which are already
$O(3)$ invariants of the neighbour set. The bulk-silicon demonstration used 48 symmetry functions,
$R_c = 6$ Å, roughly 9,000 DFT energies split 8,200/800, and reached a test RMSE of 5–6 meV/atom
with force RMSE around 0.2 eV/Å, about five orders of magnitude faster than the underlying DFT for
a 64-atom cell. It reproduced the liquid-silicon radial distribution function at 3,000 K where
Tersoff and Lenosky visibly deviate.

But note what the construction costs. The descriptor is frozen — expressiveness is capped by a
human's choice of $\eta$, $R_s$, $\zeta$, $\lambda$. It is strictly two- and three-body: genuine
four-body correlations must be hallucinated by the MLP from an incomplete scalar summary. And,
most consequentially, **the environment is collapsed to rotation-invariant scalars at the very
first step**, so directional structure is destroyed before any learning happens and cannot be
propagated. Behler and Parrinello flagged two of these in their own conclusion (poor extrapolation
and the absence of long-range terms); the third took thirteen years to prove.

### GAP and SOAP: systematic harmonic analysis, and the first crack

Bartók, Payne, Kondor and Csányi replaced the hand-picked functions with harmonic analysis of the
neighbour density and the neural network with Gaussian process regression (GAP: arXiv:0910.1019,
Phys. Rev. Lett. **104**, 136403 (2010); SOAP: arXiv:1209.3140, Phys. Rev. B **87**, 184115
(2013)). GAP builds
$\rho_i(\vec r) = \delta(\vec r) + \sum_j \delta(\vec r - \vec r_{ij}) f_{\text{cut}}(r_{ij})$,
projects the ball inside $r_{\text{cut}}$ onto the 3-sphere so that radial information is carried
by a third polar angle, expands in 4D hyperspherical harmonics, and forms the $SO(4)$ bispectrum.

SOAP then inverts the logic in a way that turned out to be conceptually decisive: rather than
picking a descriptor and then a kernel, define the similarity between two environments directly,
by integrating the overlap of their smoothed densities over all relative rotations,

$$k(\rho, \rho') = \int d\hat{R}\, \left| \int \rho(\vec r)\, \rho'(\hat R \vec r)\, d\vec r
\right|^{n}.$$

For $n = 2$ this Haar integral collapses exactly to the dot product of **power spectra**,
$p_{nn'l} = \sum_m c_{nlm} c^{*}_{n'lm}$; for $n = 3$ to the dot product of **bispectra**. This is
where the field's vocabulary was fixed: the power spectrum is exactly three-body, the bispectrum
exactly four-body. On C, Si and Ge, GAP reached sub-meV/atom energy RMSE and matched DFT elastic
constants closely (diamond $C_{11}$: DFT 1118, GAP 1081, Tersoff 1072 GPa) where Brenner and
Tersoff were off by more than 25% on at least one constant, at roughly $2\times10^4$ times the
speed of CASTEP for a 216-atom cell.

GAP 2010 also made a claim that the field then leaned on for a decade: that the bispectrum "can
provide an almost one-to-one representation of the atomic neighbourhood." The SOAP paper itself
already contained the seed of its refutation — the explicit pair
$f_1 = Y_{22} + Y_{2,-2} + Y_{33} + Y_{3,-3}$ and $f_2 = Y_{21} + Y_{2,-1} + Y_{32} + Y_{3,-2}$,
which have identical power spectra $p_2 = p_3 = 2$.

### The refutation, and why it matters for everything after

Pozdnyakov, Willatt, Bartók, Ortner, Csányi and Ceriotti closed this in 2020 (arXiv:2001.11696,
Phys. Rev. Lett. **125**, 166001). They constructed explicit pairs of atomic environments that are
genuinely distinct — not related by rotation, reflection, translation or permutation — yet have
**identical three-body descriptors**, and further pairs degenerate even at four-body order. The
degeneracy manifolds have finite dimension, so these are not measure-zero curiosities that random
data avoids. Two of the SOAP authors are co-authors here: this is the field auditing its own
foundational claim, and the paper is candid that the field had gotten away with it because
degeneracy is partly masked when atom-centred contributions are summed into a total energy.

The consequence is structural and inescapable. If $E_i = f(\text{descriptor})$ for any $f$
whatsoever, degenerate configurations receive identical predicted energies and, by
differentiation, inconsistent forces. No amount of network capacity repairs this; the information
was destroyed before $f$ was applied.

The [blindness chapter](../tutorial/index.html#blindness) of the tutorial exhibits an exact
instance, computed rather than asserted. On a twelve-fold ring, the neighbour sets
$A = \{0, 1, 4, 6\}$ and $B = \{0, 1, 3, 7\}$ are a homometric (cyclotomically $Z$-related) pair:
they have the same interval vector $(1,1,1,1,1,1)$, bitwise-identical sorted radii, sorted
pairwise distances agreeing to $2.2 \times 10^{-16}$, sorted bond cosines agreeing to
$2.2 \times 10^{-16}$, and power spectra agreeing to $3.1 \times 10^{-15}$ out to $\ell = 6$ — yet
their minimum RMSD over all of $O(3)$ and all permutations is exactly $(\sqrt3 - 1)/2 \approx
0.36603$, so they are genuinely different clusters. Every descriptor in this strand, and every
model built on one, assigns them the same energy. A simpler warm-up in the same chapter shows
three neighbours at $120^\circ$ versus three neighbours clustered at $0^\circ/30^\circ/60^\circ$:
the two-body descriptor $G^{2}$ is *bitwise* identical (maximum absolute difference exactly 0.0),
while a Stillinger–Weber three-body energy differs by a factor of 42.86.

### Moment Tensor Potentials: the Cartesian route

Shapeev's Moment Tensor Potentials (arXiv:1512.06054; Multiscale Model. Simul. **14**, 1153
(2016), DOI 10.1137/15M1054183) did the same job in Cartesian tensors and proved theorems about
it. The representation is the moment tensor
$M_{\mu,\nu}(u) = \sum_i |u_i|^{2\mu}\, u_i^{\otimes \nu}$, a rank-$\nu$ Cartesian tensor;
invariants are obtained by fully contracting products of these against each other. Under
$Q \in O(d)$, $M_{\mu,\nu} \to Q^{\otimes\nu} M_{\mu,\nu}$, and contracting every free index
leaves the scalar unchanged because $Q^{T}Q = I$. This is the same physics as Clebsch–Gordan
coupling, executed with Kronecker deltas instead of Wigner symbols.

MTP is the first MLIP with both a completeness theorem and a convergence rate. Theorem 3.1 shows
the contractions span all rotation- and permutation-invariant polynomials in the neighbour
vectors; Theorem 3.2 proves exponential convergence for a prototypical tight-binding site energy,
$\sup |V^{q}(u) - p_m(u)| < C\rho^{-m}$ with $\rho > 1$. The model is linear in a graded basis
truncated by a single "level" knob, $\text{lev}\, M_{\mu,\nu} = 2 + 4\mu + \nu$.

Two things about this matter downstream. First, MTP demonstrates that Cartesian tensors are a
complete alternative to spherical harmonics, which is exactly the bet the current leaderboard's
top model makes. Second, the tensors $M_{\mu,\nu}$ exist only transiently inside a contraction:
they are never persistent, learned, per-atom equivariant features refined layer by layer.
Everything that survives to the energy is a scalar. Keeping those $\ell > 0$ objects alive is
precisely what NequIP would later do.

### SNAP: the moment MLIPs became engineering

Thompson, Swiler, Trott, Foiles and Tucker (arXiv:1409.3880; J. Comput. Phys. **285**, 316 (2015))
took GAP's $SO(4)$ bispectrum essentially verbatim — the paper says so — and replaced the Gaussian
process with a plain linear model, $E^{i} = \beta_0^{\alpha_i} + \beta^{\alpha_i}\cdot B^{i}$.
Because the model is linear, energy, forces *and* the stress tensor are all linear in the same
coefficient vector, so potential generation reduces to one weighted least-squares solve. They also
proved a new bispectrum reordering identity,
$B_{j_1 j_2 j}/(2j+1) = B_{j j_2 j_1}/(2j_1+1) = B_{j_1 j j_2}/(2j_2+1)$, cutting force-evaluation
cost by more than an order of magnitude. The bcc tantalum potential reproduced screw-dislocation
core structures and migration barriers that were deliberately excluded from training. SNAP is the
point at which MLIPs stopped being research demonstrations and became production LAMMPS
deliverables — and, being linear in a truncated four-body invariant, it is also the most exposed
member of the family to the Pozdnyakov result.

### ACE: stop proposing descriptors and derive one

Drautz's Atomic Cluster Expansion (Phys. Rev. B **99**, 014104 (2019), DOI
10.1103/PhysRevB.99.014104; erratum Phys. Rev. B **100**, 249901) closed the descriptor era, and
its completeness argument matters enormously for what comes later. Start from a complete
orthonormal single-bond basis $\{\varphi_v\}$ with $\int \varphi_v^{*}\varphi_u = \delta_{vu}$ and
$\sum_v \varphi_v^{*}(\vec r)\varphi_v(\vec r\,') = \delta(\vec r - \vec r\,')$. Build cluster
basis functions as products over the bonds of a cluster. Orthogonality and completeness then
*transfer*, and the expansion

$$E_i(\sigma) = J_0 + \sum_{\alpha v} J_{\alpha v} \Phi_{\alpha v}(\sigma), \qquad J_{\alpha v} =
\langle \Phi_{\alpha v} | E_i \rangle$$

is not an ansatz: the coefficients are literal projections. Convergence to any $E_i$ holds by
construction rather than by empirical fit quality.

The second contribution is computational and is what made high body order possible at all. Define
the atomic base $A_{iv} = \langle \varrho_i | \varphi_v \rangle = \sum_j \varphi_v(\vec r_{ji})$.
The whole expansion becomes a polynomial in $A_{iv}$:

$$E_i = \sum_v c^{(1)}_{v} A_{iv} + \sum_{v_1 \ge v_2} c^{(2)}_{v_1 v_2} A_{iv_1}A_{iv_2} +
\sum_{v_1 \ge v_2 \ge v_3} c^{(3)} A_{iv_1}A_{iv_2}A_{iv_3} + \cdots$$

Constructing $A_{iv}$ is linear in the neighbour count; evaluating the polynomial is *independent*
of it. This "density trick" defeats the $N_c^{K}$ scaling that had capped the field at $K = 3$.
Rotational invariance is imposed by coupling products of the atomic base with Wigner 3j symbols so
that only the trivial irrep survives, e.g.
$B^{(2)}_{i n_1 n_2 l} = \sum_m (-1)^m A_{i n_1 l m} A_{i n_2 l, -m}$ and $B^{(3)}$ with the 3j
symbol enforcing $m_1 + m_2 + m_3 = 0$.

Section III of that paper is a unification map, and reading it is the quickest way to understand
why the "descriptor" and "graph" literatures were never really two subjects. Steinhardt order
parameters are $Q_l \propto (B^{(2)})^{1/2}$ and $W_l \propto B^{(3)}$ with a constant radial
function. Behler–Parrinello's $G^{3}, G^{4}$ are "closely related to $B^{(2)}$" via the Legendre
addition theorem, so BP's $(1 + \lambda\cos\theta)^{\zeta}$ is merely a particular Legendre
expansion. SOAP's kernels are $\sum B^{(2)}B^{(2)}$ for $n=2$ and $\sum B^{(3)}B^{(3)}$ for $n=3$.
SNAP's bispectrum "corresponds to the four-body basis functions $B^{(3)}$." And "the moment tensor
potentials may also be cast in the form of an atomic cluster expansion." The copper demonstration
used 55,289 configurations drawn from more than 100,000 DFT calculations, sampled exhaustively
over 2-, 3- and 4-atom cluster configuration spaces rather than from MD, at $\sim 10^{-6}$ s/atom
for energies.

After ACE, "invent a better invariant descriptor" was no longer an open question. What remained
open was the practical truncation — real ACE models stop at $K \approx 3$–5 — and the fact that a
single-cutoff, single-site expansion has no mechanism for information beyond $r_c$.

---

## II. The invariant graph strand, 2017–2021: learned representations, still scalar

### Gilmer's MPNN: the vocabulary

Gilmer, Schoenholz, Riley, Vinyals and Dahl (arXiv:1704.01212; ICML 2017) reformulated a family of
prior graph models into one framework: $T$ steps of
$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$ and $h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$,
followed by a permutation-invariant readout. The key innovation for chemistry was the *edge
network* message $M(h_v, h_w, e_{vw}) = A(e_{vw})h_w$, which maps a vector-valued edge feature to
a $d \times d$ matrix and so lets continuous distances into a model whose ancestors assumed
discrete bond types. On QM9 the best single model reached chemical accuracy on 11 of 13 targets,
average error ratio 0.68, and 0.52 for a five-model ensemble — versus
2.59 for graph convolutions and 3.37 for the Coulomb matrix.

It was also, geometrically, a chemistry model with 3D bolted on. Distances entered as 10 bins over
$[2,6]$ Å combined with bond types into an alphabet of size 14; the resulting predicted PES is
discontinuous, and Schütt et al. noted bluntly that the MPNN "cannot be used on this dataset
[MD17] due to discontinuities in its inferred potential energy surface." A model that cannot be
differentiated is not a potential.

### SchNet: the model that made "GNN interatomic potential" mean something

Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko and Müller (arXiv:1706.08566, NeurIPS 2017;
arXiv:1712.06113, J. Chem. Phys. **148**, 241722 (2018)) replaced the discrete-grid convolution of
CNNs with a *continuous-filter* convolution,

$$x_i^{l+1} = \sum_j x_j^{l} \circ W^{l}(\vec r_i - \vec r_j),$$

with $W$ restricted to depend only on the scalar distance. Energy is then rotationally invariant
by construction and forces, taken as $-\partial E/\partial \vec r_i$, are equivariant and
conservative. Everything is smooth — Gaussian RBFs on a 0.1 Å grid, shifted softplus
$\ln(\tfrac12 e^x + \tfrac12)$ — so it can be trained on forces and used for MD, which the MPNN
could not.

SchNet is the canonical distances-only model, and the paper says so: "we restrict our filters for
the cfconv layers to be rotationally invariant. The rotational invariance is obtained by using
interatomic distances." All angular structure must therefore be reconstructed indirectly by
composing distance information across interaction blocks. On MD17 at $N = 1{,}000$ this costs a
great deal: aspirin 0.37 kcal/mol energy and 1.35 kcal/mol/Å forces. Also worth recording, because
it is routinely forgotten in the equivariance debate: training on energies alone at $N = 1{,}000$
is catastrophic (aspirin forces 23.54 kcal/mol/Å), so a large share of every result in this
literature comes from *force supervision*, not from the representation.

### PhysNet: smoothness and the long-range half of the argument

Unke and Meuwly (arXiv:1902.08408; J. Chem. Theory Comput. **15**, 3678 (2019)) contributed two
things that survive in essentially every model discussed later. First, an exponentially-scaled
radial basis, on the argument that "bound state wave functions in two-body systems decay
exponentially." Second, the smooth polynomial cutoff

$$\phi(r) = 1 - 6(r/r_{\text{cut}})^5 + 15(r/r_{\text{cut}})^4 - 10(r/r_{\text{cut}})^3,$$

adopted essentially verbatim as DimeNet's envelope and, through it, by NequIP, MACE and eSEN.
PhysNet also added predicted partial charges with explicit conservation and a damped Coulomb
kernel, making the case that a cutoff-local model has no access to $1/r$ physics no matter how
expressive it is — the SN2 experiment shows qualitatively wrong asymptotics without it. On ISO17
(known molecules, unknown conformations) PhysNet reached 0.12 kcal/mol/Å forces against SchNet's
1.00, roughly an eightfold improvement.

### DimeNet: angles enter the graph

Gasteiger (then Klicpera), Groß and Günnemann (arXiv:2003.03123; ICLR 2020) stopped embedding
atoms and started embedding *directed edges*. A message $m_{ji}$ carries a direction in real
space, so the angle between two messages is a rotation-invariant scalar that can modulate their
interaction:

$$m_{ji}^{(l+1)} = f_{\text{update}}\Big(m_{ji}^{(l)},\ \sum_{k \in N(j)\setminus\{i\}}
f_{\text{int}}\big(m_{kj}^{(l)},\, e_{\text{RBF}}^{(ji)},\, a_{\text{SBF}}^{(kj,ji)}\big)\Big).$$

The motivation is stated as the classical force-field decomposition
$E = E_{\text{bonds}} + E_{\text{angle}} + E_{\text{torsion}} + E_{\text{nonbonded}}$:
distance-only GNNs "lack the second and third terms." The bases are derived rather than chosen —
assume $V = 0$ inside the cutoff and $\infty$ outside, solve the Helmholtz equation, separate
variables, and read off spherical Bessel functions times spherical harmonics, giving the radial
basis $\tilde e_{\text{RBF},n}(d) = \sqrt{2/c}\,\sin(n\pi d/c)/d$ and the 2D spherical
Fourier–Bessel basis $\tilde a_{\text{SBF},ln}(d,\alpha)$. These are orthogonal and band-limited,
and 4–20 times smaller than their predecessors: 6 and 16 basis functions against PhysNet's 64 and
SchNet's 300.

DimeNet cut QM9 mean standardised MAE by 31% over the second-best model (1.05% vs PhysNet's 1.37%,
SchNet's 1.76%) and MD17 mean standardised force MAE from SchNet's 2.38% to 1.10% — the "76%
better" headline. Its ablations are unusually informative: replacing the Bessel basis with 64
Gaussians costs 10%, collapsing the angular basis costs 26%, and reverting to node embeddings
costs 68%. It also gave the field its favourite illustration of cutoff-induced blindness: at a 2 Å
cutoff, a regular GNN cannot distinguish cyclohexane from two cyclopropanes.

DimeNet++ (arXiv:2011.14115) then measured the cost precisely — on QM9 at 5 Å, "DimeNet uses
around 15× as many message embeddings as there are atoms and again around 15× as many triplet
representations" — and fixed the resulting FLOP misallocation, coming out 8× faster and about 10%
more accurate (QM9 $U_0$ 6.32 meV).

### GemNet: the strongest defence of the invariant paradigm

Gasteiger, Becker and Günnemann (arXiv:2106.08903; NeurIPS 2021) pushed the invariant approach to
its logical endpoint: quadruplets, with the dihedral $\theta_{cabd}$ joining the two angles and
three distances, completing $E = E_{\text{bonds}} + E_{\text{angle}} + E_{\text{torsion}}$. They
also proved the two theorems that make this a principled position rather than an expedient one:
spherical ($S^2$) representations are universal approximators for rotation-invariant, permutation-
equivariant, translation-invariant functions, and any such equivariant $h$ decomposes as
$h_a(X,H) = \sum_{c \neq a} f_a^{(c)}(X,H)\, \vec x_{ca}$ with $f$ *invariant*. In other words:
full $SO(3)$ representations are not necessary, and equivariant vectors can be predicted from
invariant scalars.

Their diagnosis of what invariance costs — the "Picasso problem" — is worth quoting because it is
the same diagnosis the equivariance strand makes: "if our model uses a rotationally invariant
layer we lose the relative information between components... An image model with rotationally
invariant layers cannot detect whether a person's eyes are rotated correctly." GemNet's answer is
to preserve relative rotational information through directed edges and dihedrals rather than
through $SO(3)$ irreps.

The numbers matter for a claim made later. On rMD17 at 1,000 training configurations, GemNet-T/Q
reaches an aspirin force MAE of **9.5 meV/Å** — better, on average, by 41% than everything before
it. Hold that number.

### DeePMD and DeepPot-SE: the frame-based parallel lineage

Zhang, Han, Wang, Car and E (arXiv:1707.09571; Phys. Rev. Lett. **120**, 143001 (2018)) came from
the AIMD community rather than the ML-on-graphs community, and took the second of the three
answers: construct a *local reference frame* per atom, sort neighbours by species and $1/R_{ij}$,
and feed full 3D coordinates in that frame. No hand-designed symmetry functions at all. The cost
is smoothness — the authors concede that "discontinuities are present in the forces, due to
adoption of a sharp cutoff radius, limitation of angular information to a fixed number of atoms,
and abrupt changes in the atomic lists due to sorting."

DeepPot-SE (arXiv:1805.09003; NeurIPS 2018) removed the frame, the sorting and the sharp cutoff
while keeping linear scaling, via the factorised descriptor
$D^{i} = (G^{i1})^{T}\tilde R^{i}(\tilde R^{i})^{T} G^{i2}$. Because
$\tilde R^{i}(\tilde R^{i})^{T}$ contains every pairwise inner product of neighbour directions,
this is a complete rotation-invariant description including every bond angle — but by
associativity it is *never materialised*, costing $O(N_i(M_1 + M_2) + M_1 M_2)$ rather than
$O(N_i^2)$. Angular information at pair cost, at the price of a rank-4 bottleneck. This lineage
carries forward through DPA-1 (arXiv:2208.08236), DPA-2 (arXiv:2312.15492) and DPA-3 to DPA-4, and
reappears near the top of the 2026 leaderboard in a form its 2018 authors would not recognise.

---

## III. The equivariance strand, 2016–2021: the machinery gets solved

### The theorem that makes the design space finite

Everything in this strand rests on a piece of classical representation theory imported into
machine learning between 2016 and 2018. Every finite-dimensional representation of $SO(3)$
decomposes into irreducibles indexed by $\ell = 0, 1, 2, \dots$ of dimension $2\ell+1$, realised
by the Wigner matrices $D^{\ell}$; $O(3)$ adds parity, so its irreps are pairs $(\ell, p)$ and
spherical harmonics carry parity $(-1)^{\ell}$ — in e3nn's notation `0e, 1o, 2e, 3o, …`.

The load-bearing fact has two halves. For **linear** maps,
$\dim \operatorname{Hom}_{SO(3)}(V_{\ell_1}, V_{\ell_2}) = \delta_{\ell_1 \ell_2}$, so an
equivariant linear layer can only mix channels of identical $(\ell, p)$ and can *never create a
new $\ell$*. For **bilinear** maps, Schur's lemma plus the multiplicity-free Clebsch–Gordan series
$D^{\ell_1}\otimes D^{\ell_2} \cong \bigoplus_{L = |\ell_1 - \ell_2|}^{\ell_1 + \ell_2} D^{L}$
gives

$$\dim \operatorname{Hom}(V_{\ell_1}\otimes V_{\ell_2}, V_{\ell_3}) = \begin{cases} 1 & |\ell_1 -
\ell_2| \le \ell_3 \le \ell_1 + \ell_2 \\ 0 & \text{otherwise,}\end{cases}$$

so the equivariant bilinear map *exists* iff the triangle rule holds and is then *unique up to one
real scalar*. That map is the Clebsch–Gordan contraction
$(u \otimes v)^{(\ell_3)}_{m_3} = \sum_{m_1 m_2} C^{(\ell_3,m_3)}_{(\ell_1,m_1)(\ell_2,m_2)} u^{(\ell_1)}_{m_1} v^{(\ell_2)}_{m_2}$,
and for $\ell = 1$ the CG coefficients literally *are* the dot product ($\propto \delta_{ij}$) and
the cross product ($\propto \epsilon_{ijk}$).

Put together: the CG product is not one option among many for a nonlinearity — it is the only way
to move information across $\ell$, and every trainable degree of freedom in an equivariant
bilinear layer is one scalar per allowed path. Architecture design stops being a search and
becomes a solve. The [machinery](../tutorial/index.html#machinery) and
[layer](../tutorial/index.html#layer) chapters work this through with live numbers.

### Cohen and Welling: equivariance as a design constraint

Group-equivariant CNNs (arXiv:1602.07576; ICML 2016) generalised translation equivariance to
arbitrary discrete groups by making feature maps functions on the group. Because $G$ then acts by
*permuting* the activation array (the regular representation), ordinary pointwise ReLU is trivially
equivariant. Rotated MNIST error fell from 3.98% to 2.28%; CIFAR10+ from 5.27% to 4.19% in an
otherwise identical ResNet44. The limitation is fatal for our purposes: regular representations
scale as $|G|$, which is infinite for $SO(3)$.

Steerable CNNs (arXiv:1612.08498; ICLR 2017) fixed that by replacing the group-indexed feature map
with a *fiber* carrying an arbitrary representation, moving the entire equivariance burden onto a
linear constraint on the filter bank, $\rho(h)\Psi = \Psi\pi(h)$. The admissible filters form the
intertwiner space, whose dimension is the character inner product
$\dim \operatorname{Hom}_H(\pi,\rho) = \langle \chi_\pi, \chi_\rho\rangle$ — literally the
finite-group form of the Schur argument above. This paper introduced the **type system**: features
are typed by which irreps occur with what multiplicity, layers are intertwiners, and the parameter
count is determined by the types alone. It is the direct ancestor of e3nn's `Irreps` strings and
of the $\ell$-bookkeeping in TFN and NequIP. A depth-14 steerable ResNet with 9.1M parameters
reached
3.65% on CIFAR10+, against DenseNet-100's 3.74% with 27.2M.

Kondor and Trivedi (arXiv:1802.03690; ICML 2018) then proved the converse: for compact-group
actions, convolutional structure is not merely sufficient for equivariance but **necessary**.
After this, the posture of the whole field changed — you write down the constraint and compute a
basis for its solution space.

### Tensor Field Networks: the architectural parent

Thomas, Smidt, Kearnes, Yang, Li, Kohlhoff and Riley (arXiv:1802.08219) is the direct
architectural parent of NequIP, and the correspondence is essentially equation-for-equation.
Features on points are decomposed into irreps, stored as $V^{(\ell)}_{acm}$ indexed by [point,
channel, $m$]. Filters are constrained to the product of a *learnable radial function* and a
*fixed spherical harmonic*,

$$F^{(\ell_f, \ell_i)}_{cm}(\vec r) = R^{(\ell_f,\ell_i)}_c(r)\, Y^{(\ell_f)}_m(\hat r),$$

and combined with input features by CG contraction. Self-interaction
$\sum_{c'} W^{(\ell)}_{cc'} V^{(\ell)}_{ac'm}$ is exactly the only linear equivariant operation
Schur permits. All learnable weights live in $R(r)$ and are rotation-invariant scalars; the
angular structure is entirely fixed.

Kondor's N-body Networks (arXiv:1803.01588) stated the weight constraint as a theorem (Proposition
1: weights may only mix fragments sharing the same $\ell$) and fixed the target — atomic
potentials for MD with equivariant internal features — three years before NequIP. Clebsch–Gordan
Nets (arXiv:1806.09231; NeurIPS 2018) then proved, by construction, that a network can be built
with the CG product as its *only* nonlinearity: on rotated spherical MNIST it scored
96.4/96.0/96.6 across train/test rotation regimes where a planar CNN collapsed from 97.67 to 12.

### 3D Steerable CNNs: the general solution, and the gate

Weiler, Geiger, Welling, Boomsma and Cohen (arXiv:1807.02547; NeurIPS 2018) derived the space of
$SE(3)$-equivariant linear maps between fields on $\mathbb{R}^3$ in closed form. Their Theorem 2:
"A linear map from $F_n$ to $F_{n+1}$ is equivariant if and only if it is a cross-correlation with
a rotation-steerable kernel." Translation equivariance collapses the kernel to one argument, the
rotational constraint $\kappa(r x) = \rho_2(r)\kappa(x)\rho_1(r)^{-1}$ is linear in $\kappa$,
vectorising and CG-decomposing reduces it to $\eta^{jl,J}(rx) = D^{J}(r)\eta^{jl,J}(x)$ — "a
famous equation for which the unique and complete solution is well-known to be given by the
spherical harmonics," with the radial part completely free.

This is the retroactive justification for TFN's filter factorisation: **radial-learnable,
angular-fixed is not a modelling choice, it is the general solution of the equivariance
constraint.** The paper also introduced the **gated nonlinearity** that NequIP uses: learn a
second steerable kernel producing a scalar field and multiply,
$f^{i}_n(x)\cdot\sigma(\gamma^i_n \star f_{n-1}(x))$, which in irrep language is the
$(\ell,p)\otimes(0,e)\to(\ell,p)$ path with a data-dependent coefficient. The data-efficiency
evidence was striking: rotated Tetris trained on a single orientation gave 99 ± 2% against 27 ± 7%
for a conventional 3D CNN, and CATH protein classification was won by a steerable network with
143,560 parameters over a conventional 3D CNN with 15,878,764 — more than a hundredfold fewer.

### Cormorant and SE(3)-Transformers

Cormorant (Anderson, Hy and Kondor, arXiv:1906.04015; NeurIPS 2019) carried the Fourier-space
covariant program into a practical molecular architecture with per-atom and per-pair covariant
activations and the channel-wise CG product as the nonlinearity. It reported MD-17 conformational
energies (aspirin 0.098 kcal/mol) and led on six QM9 targets. **It is the paper that most
complicates NequIP's priority claim**, and we return to that below: it already had equivariant
atom-wise message passing with $\ell \ge 1$ features. What it did not have was forces.

SE(3)-Transformers (Fuchs, Worrall, Fischer and Welling, arXiv:2006.10503; NeurIPS 2020) made the
TFN kernel data-dependent by inserting attention, with a clean separation: attention *weights* are
built to be invariant scalars, value *messages* are equivariant TFN kernels, and a linear
combination of equivariant messages with invariant coefficients is equivariant. They also
introduced the diagnostic the field now uses — the equivariance error $\Delta_{\text{EQ}}$, about
$3 \times 10^{-7}$ for a genuinely equivariant model against 0.167 for an unconstrained Set
Transformer.

### e3nn: the substrate

The e3nn library (Geiger and Smidt, arXiv:2207.09453; first Zenodo release March 2020) packaged
all of it as a PyTorch type system in which $O(3)$ irreps are types and every layer is a typed,
provably equivariant operation. Two primitives: spherical harmonics, and a fully general
`TensorProduct` that "can represent any bi-linear equivariant operations combining two sets of
irreps into irreps," with path selection rules $|\ell_1-\ell_2| \le \ell_3 \le \ell_1+\ell_2$
**and** $p_1 p_2 = p_3$. Note the authorship overlap that makes this lineage so tight: Smidt from
TFN, Geiger, Weiler and Boomsma from 3D Steerable CNNs. e3nn is what made NequIP a
several-hundred-line model rather than a multi-year exercise in implementing the Wigner–Racah
algebra, and it is the substrate on which NequIP, Allegro, MACE, Equiformer, DiffDock and most of
the ecosystem were built.

---

## IV. The confluence: NequIP, 2021/2022, and what came directly out of it

### What NequIP is, precisely

Batzner, Musaelian, Sun, Geiger, Mailoa, Kornbluth, Molinari, Smidt and Kozinsky
(arXiv:2101.03164, 8 Jan 2021; Nature Communications **13**, 2453 (2022), DOI
10.1038/s41467-022-29939-5) applied $E(3)$-equivariant convolutions over geometric tensors to
interatomic potentials. Total energy is a sum of atomic energies; forces are the analytic negative
gradient, so energy conservation and force equivariance are guaranteed.

The architectural inheritance is exact rather than approximate. NequIP's filter equation is TFN's
filter equation; NequIP's convolution is TFN's point-convolution with parity indices attached and
the neighbour sum divided by $\sqrt{\bar N}$; the gated nonlinearity is Weiler et al.'s; the
primitives are e3nn's. The paper says so: "Here, we build on the layers introduced in Tensor-Field
Networks." What NequIP genuinely added was (i) systematic parity, making the model $E(3)$- rather
than $SE(3)$-equivariant, with even and odd scalars processed by SiLU and tanh respectively; (ii)
a learnable radial MLP on a Bessel basis with a polynomial envelope, with the $n\pi$ frequencies
themselves trained by backpropagation; and (iii) the empirical result that made the field pivot.

NequIP is also admirably explicit about what the invariant models are, in its own language:
"Omitting all higher-order interactions that go beyond the $0 \otimes 0 \to 0$ interaction will
result in a conventional GNN-IP with invariant convolutions over scalar features, similar to e.g.
SchNet." Invariant GNN potentials are the $\ell_{\max} = 0$ special case.

### The priority claim, stated correctly

NequIP is routinely described as the first equivariant interatomic potential. That is too strong.
The defensible statement, after adversarial checking against primary sources, is this: **NequIP
was the first *neural message-passing* interatomic potential to combine local atomic energies,
$\ell \ge 1$ $E(3)$-equivariant tensor features, and gradient-derived conservative forces.** Every
clause is doing work, because:

- it builds directly on TFN/e3nn layers and invents none of the equivariance machinery;
- **Cormorant (2019)** already had equivariant atom-wise message passing with $\ell \ge 1$ features
  — energies only, no forces, no parity, no periodic boundary conditions;
- **Townshend et al.** had already used TFN for forces, but predicted directly, so not conservative;
- non-neural **MTP (2016)** and **ACE (2019)** already paired equivariant tensor bases with local
  atomic energies and analytic conservative forces;
- **PaiNN** (Schütt, Unke and Gastegger, arXiv:2102.03150, Feb 2021) was near-concurrent.

NequIP itself never claims to be first. The claim is a community artifact.

### The evidence, and the confound that must be stated

NequIP's headline is data efficiency. On revised MD-17 with 1,000 combined train+validation
configurations at $\ell = 3$: aspirin 8.5 meV/Å force MAE and 2.3 meV energy MAE, against ACE's
17.9/6.1, FCHL19's 20.9/6.2, GAP's 44.9/17.7 and ANI's 40.6/16.6. On liquid water and ice at
PBE0-TS, NequIP trained on **133 structures** matched DeepMD trained on **133,500** from the same
trajectories — a thousandfold reduction that made CCSD(T)-quality reference data practical.

The causal evidence offered for equivariance specifically is the rotation-order scan, on rMD17
aspirin force MAE:

| $\ell_{\max}$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| force MAE (meV/Å) | 41.9 | 12.9 | 8.7 | 8.5 |

Two corrections are needed here, and they belong in the main text rather than a footnote.

**First, a dataset point.** The *capacity-controlled* ablation — weight- and feature-matched
$\ell = 0$ baselines — and the data-efficiency learning curves are on **original MD-17** (Appendix
B, Fig. 11), not revised MD17. The $\ell = 0..3$ scan in Table II is on rMD17 but is
single-training-budget accuracy with **no capacity control**. The two are often conflated.

**Second, and more important, a confound.** From NequIP's own Table II, on the same rMD17 aspirin
task: **ACE**, an invariant *linear* model, scores 17.9 meV/Å, and **GemNet-T/Q**, an invariant
model with directional (angle and dihedral) features, scores **9.5 meV/Å**. So an invariant model
beats NequIP at $\ell = 1$ (12.9) and comes within 12% of NequIP at $\ell = 3$ (8.5). The same
pattern appears in Table I on original MD-17: SchNet, which sees only distances, scores 58.5,
while DimeNet, which is fully invariant but *has* angular features, scores 21.6.

The reason is that NequIP's $\ell = 0$ baseline is, by the paper's own description, a SchNet-style
model with no angular information at all. The $41.9 \to 12.9$ jump therefore conflates two
distinct variables: adding equivariance, and adding *any* directional/many-body resolution.
Invariant models with angular features fill most of that gap. The honest reading of the
$\ell_{\max}$ scan — which the [lmax chapter](../tutorial/index.html#lmax) reproduces in miniature
with explicit feature-count and radial-capacity controls — is that $\ell \ge 1$ tensor features
are an unusually *cheap and systematic* way to obtain angular resolution and body order, not that
equivariance per se accounts for the whole effect.

### Allegro: strict locality, and why it buys parallel scaling

NequIP left behind a specific engineering problem. With $T$ message-passing layers the receptive
field is $T \cdot r_c$, so a domain-decomposed MD code needs a ghost region $T$ cutoffs deep.
Musaelian, Batzner, Johansson, Sun, Owen, Kornbluth and Kozinsky attacked it by deleting message
passing entirely (arXiv:2204.05249; Nature Communications **14**, 579 (2023)).

Allegro decomposes the energy over ordered *pairs*, $E = \sum_i \sum_{j \in N(i)} E_{ij}$, and
every layer for edge $(ij)$ reads only atoms inside the single cutoff sphere of atom $i$.
Many-body correlation is built by iterating CG tensor products *within* that one environment,
using a two-track latent space — an invariant scalar track of plain MLPs gating an equivariant
tensor track — and an ACE-style density trick, one pooled tensor product per layer per edge
against $\sum_{k \in N(i)} Y_\ell(\hat r_{ik})$ rather than one per neighbour pair. **Depth no
longer grows the receptive field.**

The parallel consequence is the point. Each MD rank needs exactly one ghost-atom position exchange
per timestep with *zero* inter-layer communication, so the model is embarrassingly parallel under
spatial decomposition and $O(N)$ in atoms. The demonstrations: 100,640,512 atoms of bulk Ag at
1.539 ns/day on 128 A100s; strong scaling on 421,824-atom Li₃PO₄ from 0.552 to 0.018
µs/atom·step across 1→64 GPUs; and, in the SC'23 paper (arXiv:2304.10061, DOI
10.1145/3581784.3627041), a 44-million-atom fully solvated all-atom HIV capsid on Perlmutter with
70% weak-scaling efficiency out to 5,120 A100 GPUs. Accuracy did not suffer: rMD17 aspirin 2.3 meV
/ 7.3 meV/Å, better than NequIP's 8.2. And a single Allegro tensor-product layer reached QM9 $U_0$
of 5.7 meV, beating PaiNN's 5.9 and every deep message-passing baseline — direct evidence that
NequIP's depth was buying expressivity, not merely range. The [locality
chapter](../tutorial/index.html#locality) makes the receptive-field accounting explicit.

### SevenNet: the opposite answer to the same problem

Park, Kim, Hwang and Han at Seoul National University's Materials Data and Informatics Laboratory
(arXiv:2402.03789; J. Chem. Theory Comput. **20**, 4857 (2024)) kept NequIP's architecture almost
unmodified and solved the parallelism problem by *communicating* instead of by restricting. The
trained model is serialised into $T$ separate TorchScript modules, one per message-passing layer,
and LAMMPS drives them in sequence with $T-1$ forward exchanges of ghost node features and $T-1$
matching reverse exchanges for force backpropagation. The ghost region stays one cutoff deep;
range is recovered by MPI rather than by replication.

Weak scaling on SiO₂ at 4,608 atoms/GPU holds 67–84% efficiency at 32 GPUs; 112,000-atom amorphous
Si₃N₄ runs 60 ps in 12.7 hours at 0.94 eight-GPU weak-scaling efficiency. SevenNet-0, the
universal potential trained on MPtrj, used only **0.84M parameters** against GNoME's 16.24M. And
there is a design consequence that shows up later on the leaderboard: because depth is now cheap
to parallelise, SevenNet can afford *more* of it — the Omni-i8 and Omni-i12 checkpoints use 8 and
12 message-passing layers, a design point that would be unusable otherwise.

### The kernel line: when the accuracy race became an engineering race

NequIP's real bottleneck is one kernel: the Clebsch–Gordan tensor product, a sparse, irregular,
small-block contraction that e3nn expresses as many tiny einsums and that maps badly onto GPU
tensor cores. Three overlapping efforts industrialised it, and by 2026 they are as
outcome-determining as any architectural idea in this report.

The **nequip** package itself was rewritten (Tan, Descoteaux, Kotak, Nascimento, Kavanagh, Zichi,
Wang, Saluja, Hu, Smidt, Johansson, Witt, Kozinsky and Musaelian, arXiv:2504.16068; Digital
Discovery **5**, 1558 (2026)) around PyTorch Lightning, multi-node distributed training, full
`torch.compile` coverage and AOT Inductor compiled inference — the first end-to-end AOTI
deployment for MLIPs — plus a custom kernel for Allegro's dominant tensor product, giving up to
**18×** MD speedup at practically relevant system sizes. **OpenEquivariance** (Bharadwaj, Glover,
Buluç and Demmel at Berkeley/LBNL, arXiv:2501.13986; SIAM ACDA25) is an open JIT kernel generator
with e3nn-identical semantics that additionally fuses the gather/scatter into the tensor product
so the per-edge intermediate is never materialised: up to **10×** over e3nn, **1.3×** over
cuEquivariance, and 6.2× end-to-end for MACE in FP64. NVIDIA's **cuEquivariance** generalises the
operation into segmented polynomials with hand-tuned kernels, reporting ~10× end-to-end MACE
speedup and ~200× on the symmetric-contraction operation alone.

This line matters for reading the 2026 leaderboard. The current #2 model, EquFlashV2, is not a new
representation-theoretic idea at all — it is a fused tensor-product kernel (FlashTP, ICML 2025)
reported at up to 41.6× over e3nn and 60.8× over cuEquivariance with 6× lower memory, wrapped
around plain CG convolution in a transformer-shaped block *without attention*.

### MACE: diagnose the cost as body order, not locality

Batatia, Kovács, Simm, Ortner and Csányi (arXiv:2206.07697; NeurIPS 2022) gave the third answer to
NequIP's scaling problem, and the one that turned out to be most influential. Their diagnosis: a
NequIP message is a *two-body* object — it takes one neighbour at a time — so the only way to
build $n$-body correlation is to stack layers, which welds expressivity to receptive field and to
communication depth.

MACE imports ACE's product basis into the message-passing layer. Per layer: (i) form one-particle
features
$A_{i,k\ell m} = \sum_{j \in N(i)} R_{k\ell}(r_{ij}) Y^m_\ell(\hat r_{ij}) \cdot (\text{a learned mix of } h_j)$,
a pooled sum retaining full angular information; (ii) take $\nu$-fold tensor products of $A_i$
with itself, symmetrised with generalised Clebsch–Gordan coefficients, so $\nu = 3$ gives
**four-body messages**; (iii) take linear combinations with element-dependent weights. The crucial
trick — inherited straight from Drautz's density trick — is that summing over neighbours *before*
taking products makes the cost independent of neighbour count, defeating the apparent exponential
scaling in body order. Because $A$ is a node quantity, the expensive symmetric contraction runs on
nodes ($O(N)$) rather than on edges.

Two layers then suffice where NequIP used five or six. rMD17 aspirin: 2.2 meV / 6.6 meV/Å. 3BPA
extrapolated to 1200 K: 29.8 meV / 62.0 meV/Å against NequIP's 40.8 / 86.4. Speed: 24.3 ms per
structure at $L=2$ against NequIP's 103.5 ms. MACE-MP-0 (arXiv:2401.00096), trained on MPtrj's
1.58M structures across 89 elements, became the first widely-used general-purpose foundation MLIP
in this lineage and is the historical baseline for everything on the current leaderboard: F1
0.669, $\kappa_{\mathrm{SRME}}$ 0.682, CPS 0.637.

### Multi-ACE: the two strands were one design space all along

The unification was written deliberately, by both sides together. Batatia, Batzner, Kovács,
Musaelian, Simm, Drautz, Ortner, Kozinsky and Csányi (arXiv:2205.06643; Nature Machine
Intelligence
**7**, 56 (2025), DOI 10.1038/s42256-024-00956-x) is a joint Cambridge + Harvard/MIR + Bochum + UBC
paper — the NequIP authors and the ACE/MACE authors in one author list — and it contains ACE,
NequIP, MACE, BOTNet, PaiNN, SchNet and Tensor Field Networks as special cases of a single
four-stage framework.

The central results: **ACE is exactly one layer of a multi-layer architecture**, and the
linearised version of NequIP is a particular *sparsification* of a much larger polynomial model —
NequIP was implicitly making a sparsity choice nobody had named. The differences between
architectures then become explicit design axes: number of layers $T$, correlation order $\nu$,
maximum equivariance order $L$, coupled versus uncoupled channels, discrete versus learned element
embedding, linear versus nonlinear update, radial parameterisation. The placements are compact:
linear ACE is $T=1$ with coupled channels and discrete elements; SchNet is $T \ge 2$, $L = 0$;
NequIP is $T \ge 2$, $L \ge 1$, uncoupled channels, learned radial MLP, $\nu = 1$; MACE is
$T = 2$, $\nu = 3$.

The paper also produced BOTNet — "body-ordered NequIP," a controlled baseline that removes all
pointwise nonlinearities from the update except the last layer so that each $h^{(t)}$ has exactly
correlation order $t$ — and a finding that has aged extremely well: **design choices matter far
more out-of-domain than in-domain.** Extrapolation performance varies much more across the design
space than interpolation does. On an O–O dimer dissociation curve absent from training, NequIP and
BOTNet give chemically sensible curves while linear ACE predicts identically zero.

After Multi-ACE, the claim "the ACE people and the equivariant-GNN people are doing different
things" stopped being true. It is why GRACE, TACE and TECE — which reach the top of the 2026
leaderboard from Drautz's side of the family — look like siblings rather than strangers.

---

## V. Scaling and pluralisation, 2022–2026

### The first universal potentials

**M3GNet** (Chen and Ong, Materials Virtual Lab, UC San Diego; arXiv:2202.02450; Nature
Computational Science **2**, 718 (2022)) was the first single graph-network potential fit to
essentially the whole periodic table (89 elements). Architecturally it is invariant by
construction: a MEGNet-style atom/bond/state graph plus explicit three-body angular features from
spherical Bessel × spherical harmonic products, with forces and stresses by autodifferentiation.
Its real innovation was data. It mined the *ionic steps* of Materials Project relaxations — until
then discarded as computational waste — into MPF.2021.2.8: 187,687 frames from 62,783 compounds,
carrying 16,875,138 force components and 1,689,183 stress components. With 227,549 learnable
weights it reached 0.035 eV/atom, 0.072 eV/Å and 0.41 GPa, and screened 31M hypothetical
structures down to 1.8M predicted stable, of which the top 2,000 by predicted hull distance
yielded 1,578 DFT confirmations.

**CHGNet** (Deng, Zhong, Jun, Riebesell, Han, Bartel and Ceder at Berkeley/LBNL; arXiv:2302.14231;
Nature Machine Intelligence **5**, 1031 (2023)) followed the same template and added on-site
magnetic moments as a fourth regression target, forcing the latent space to carry oxidation-state
information and enabling charge-informed MD. But its enduring contribution is the dataset it
shipped: **MPtrj**, 1,580,395 configurations from ~146,000 materials over 94 elements, with
1,580,395 energies, 49,295,660 forces, 14,223,555 stresses and 7,944,833 magnetic moments. MPtrj
became the canonical "compliant" training set for Matbench Discovery and therefore the field's de
facto control condition. The architecture was overtaken within two years; the dataset was not.

**GNoME** (Merchant, Batzner, Schoenholz, Aykol, Cheon and Cubuk at Google DeepMind; Nature **624**,
80 (2023)) is not primarily a potential paper — it is a scaling-of-active-learning paper, and it
is where the NequIP line and the scaling line meet, Simon Batzner being a co-author. Two candidate
generators (element substitution with symmetry-aware partial substitutions; composition guessing
with 100 AIRSS random structures each) feed a GNN energy filter whose survivors are DFT-relaxed
and folded back into training, six times over. Results: 2.2M structures below the previously known
hull, 381,000 new stable entries against ~48,000 known before, 45,500 novel structural prototypes,
and hit rates improving from under 6% to over 80% (structural) and ~3% to 33% (compositional).
Energy MAE fell from 21 to 11 meV/atom across rounds.

Two honest caveats. First, GNoME hedged its architectural bet: the discovery filter is
*deliberately non-equivariant* — the paper's thesis is that at this data scale you buy more from
more data than from more symmetry machinery — while the downstream dynamics potential is an
equivariant NequIP-type model in e3nn-jax. Second, Cheetham and Seshadri (Chem. Mater. **36**
(2024), DOI 10.1021/acs.chemmater.4c00643) argue that few of the "discoveries" are genuinely new
compounds.

### The transformer branch: Equiformer → eSCN → EquiformerV2 → eSEN → UMA → EquiformerV3

**Equiformer** (Liao and Smidt, MIT Atomic Architects; arXiv:2206.11990; ICLR 2023) ported the
Transformer block to irreps features: pre-normalisation, multi-head equivariant graph attention,
equivariant feed-forward network, residual stream, equivariant layer normalisation. The two
substantive novelties were *MLP attention* (attention weights from the scalar part of the message
rather than a dot product) and *non-linear messages* (gate activation applied inside the message
before aggregation). It did not solve the complexity problem: the tensor product still scales
$O(L^6)$, capping $L_{\max}$ at 2 or 3.

**eSCN** (Passaro and Zitnick, FAIR at Meta; arXiv:2302.03655; ICML 2023) removed that cap, and it
is the single most consequential efficiency result in the branch. The observation: you are free to
choose the frame in which you evaluate the tensor product. Rotate the node embeddings with a
Wigner-D matrix so the edge vector is aligned with the primary axis, and the edge's spherical
harmonics collapse to their $m_f = 0$ components. Proposition 3.1 then shows
$C^{(\ell_o, m_o)}_{(\ell_i, m_i),(\ell_f, 0)}$ is non-zero only for $m_i = \pm m_o$, and the
surviving structure is a set of independent $SO(2)$ linear operations on $\pm m$ pairs.
Physically: once the axis is pinned to the edge, only the roll about it remains, so
$SO(2)$-equivariance about that axis guarantees full $SO(3)$-equivariance. Cost falls from
$O(L^6)$ to $O(L^3)$, and $L = 6$ becomes trainable at scale. eSCN inherited SCN's *direct* force
head unchanged, which mattered later.

**EquiformerV2** (Liao, Wood, Das and Smidt; arXiv:2306.12059; ICLR 2024) is where the MIT
Equiformer line and the Meta FAIR eSCN line formally merge. It swapped Equiformer's $O(L^6)$
convolutions for eSCN's, and added three stabilisers needed at high degree: attention
re-normalisation, separable $S^2$ activation, and separable layer normalisation. At 153M
parameters on OC20 S2EF-All+MD it reached 219 meV energy and 14.2 meV/Å force MAE, with AdsorbML
success rate
88.90% at $k=2$ against GemNet-OC's 77.29% and a 2× reduction in DFT calls. It is the high-water
mark of the direct-force era, and it added nothing on energy conservation or PES smoothness — a
gap that would cost it dearly.

**eSEN** (Fu, Wood, Barroso-Luque, Levine, Gao, Dzamba and Zitnick, FAIR; arXiv:2502.12147; ICML
2025 spotlight) is the corrective paper of the branch, and one of the most useful papers in this
whole genealogy. Its central empirical claim is that lower held-out S2EF test error does not
translate into better downstream physics, and it proposes a concrete gate: run NVE molecular
dynamics at fixed timestep and measure energy drift. It then diagnoses three failure families and
fixes each:

1. **Direct force prediction.** A direct head gives $\hat F \neq -\nabla_r \hat E$, so work around a
   closed path need not vanish. The paper is blunt: "strictly speaking, direct-force models are not
   truly 'potentials', but rather (non-conservative) 'force fields'." Fix: gradients.
2. **Representation discretisation.** Projecting spherical-harmonic channels onto an $S^2$ grid and
   applying a pointwise nonlinearity — as SCN, eSCN and EquiformerV2 all do — injects content above
   the grid's Nyquist frequency, breaking strict equivariance and conservation. Fix: nodewise layers
   that never discretise, using a SiLU gate directly in the spherical-harmonic representation.
3. **Non-smooth PES.** A maximum-neighbour cap makes the neighbour set jump under infinitesimal
   displacement; a hard cutoff without an envelope makes the radial embedding not twice
   continuously differentiable; 512 Gaussian basis functions make the PES hypersensitive. Fixes: a
   pure 6 Å cutoff with no neighbour limit, a polynomial envelope, and ~10 basis functions.

The paper also notes that conservative forces are necessary but *not sufficient* — the PES also
needs bounded higher derivatives for a finite-$\Delta t$ integrator to conserve energy over long
times. And it supplied the training recipe now used verbatim by UMA and EquiformerV3: 60 epochs of
direct-force pre-training (with the DeNS denoising auxiliary task), then remove the head and
fine-tune 40 epochs conservatively — reaching a *lower* validation loss than 100 epochs of
from-scratch conservative training at 40% less wallclock. eSEN-30M-MP became the first model to
lead on both F1 (0.831) and $\kappa_{\mathrm{SRME}}$ (0.340), where "all previous models only
achieve SOTA performance on one or the other." The [forces chapter](../tutorial/index.html#forces)
works through the conservative/direct distinction directly.

**UMA** (Wood, Dzamba, Fu, Gao, Shuaibi, Barroso-Luque, Abdelmaqsoud, Gharakhanyan, Kitchin,
Levine, Michel, Sriram, Cohen, Das, Rizvi, Sahoo, Ulissi and Zitnick, FAIR; arXiv:2506.23971;
NeurIPS 2025) trained one eSEN-based model across materials, molecules, catalysts, MOFs and
molecular crystals at once — close to 500 million training examples. Its capacity mechanism is a
Mixture of Linear Experts: $y = \sum_k \alpha_k (W_k x)$ with routing coefficients depending only
on time-invariant global properties (composition, total charge, spin multiplicity, DFT task).
Because the experts are linear, the sum collapses to $W^{*} = \sum_k \alpha_k W_k$, precomputable
once per system, so a 1.4B-parameter model runs at ~50M-parameter speed during MD. The reasons for
choosing linear + dense + globally-routed are all MLIP-specific and worth noting: dense use of all
experts avoids routing discontinuities that would break energy conservation; linear experts
preserve rotational equivariance inside the eSCN convolution; global-only routing enables the
precomputation. Positions are *deliberately excluded* from the routing input.

**EquiformerV3** (Liao, Hoffman, Shen, Duval, Norwood and Smidt, MIT + Mirror Physics;
arXiv:2604.09130, 10 Apr 2026) brought the Equiformer line back to the front by absorbing eSEN's
conservation lessons into an attention architecture. Three contributions. *SwiGLU-$S^2$*: apply
the nonlinearity only to scalars and use a bilinear grid–grid product, which is equivalent to a
self tensor product $x \otimes x$ in irreps space — raising body order while injecting nothing
above Nyquist, so a much coarser grid suffices (at $L_{\max} = 6$, attention grid points drop from
324 to 160, a 50.6% reduction). *Smooth-cutoff attention*: softmax normalisation is inherently
non-local, so "when atoms enter or leave the cutoff radius, the denominator changes abruptly," and
an envelope on messages alone is insufficient — the fix folds an envelope into the softmax itself.
*Merged layer normalisation*, partially reversing V2's separable version.

The most interesting result in that paper is a reversal: pre-trained on OMat24, $L_{\max} = 6$
gives "almost no improvement" over $L_{\max} = 4$. The bottleneck is data, not representation
degree — directly contradicting the 2023 assumption that motivated eSCN and EquiformerV2 in the
first place.

### The unconstrained branch and the data branch

**ORB** (Neumann, Gin, Rhodes, Bennett, Li, Choubisa, Hussey and Godwin at Orbital Materials;
arXiv:2410.22570) is the clearest deliberate rejection of built-in equivariance in the
universal-MLIP literature: an encoder-processor-decoder Graph Network Simulator with MLP message
passing, no spherical harmonics, no irreps, no tensor products, with rotational invariance learned
from random-rotation augmentation. Its second bet was denoising-diffusion pretraining as a
label-free objective that lets mutually incompatible DFT sources be mixed. ORB-v2 reached F1 0.880
— but $\kappa_{\mathrm{SRME}} = 1.7338$ and CPS 0.528, because direct forces are not the gradient
of anything and the second derivatives are garbage. **Orb-v3** (arXiv:2504.06231) treated
equivariance, conservatism and graph sparsity as three traversable axes of a Pareto surface and
shipped a matrix of checkpoints; the conservative variant cut $\kappa_{\mathrm{SRME}}$ from 1.73
to
0.21. It also produced a data-quality finding with teeth: training on all of OMat24, 45% of which is
synthetically rattled, produced unphysical homonuclear diatomic curves, so they train only on the
~55M-structure AIMD subset.

**GRACE** (Bochkarev, Lysogorskiy and Drautz, ICAMS Bochum; arXiv:2311.16326; Phys. Rev. X **14**,
021036 (2024); foundation models arXiv:2508.17936, npj Comput. Mater. **12**, 114 (2026)) extends
ACE with *graph* basis functions, making the expansion semilocal while remaining a controlled
expansion rather than an ad hoc network. Its sharpest result is structural: applying a tensor
decomposition to the graph expansion collapses it into an iterative procedure that *is* message
passing — so current equivariant MPNN potentials fall out as a truncation of graph ACE. That
reframes MACE and its relatives as approximations to a known complete basis rather than as
independent architectures. GRACE is equivariant by construction, but from the "complete basis"
school rather than the "stacked e3nn tensor products" school: Chebyshev radial basis, spherical
harmonics to $\ell_{\max} = 4$, product basis to fourth order, sparse CG coupling, recursive ACE
evaluation in FP64.

**PET** (Pozdnyakov and Ceriotti, COSMO/EPFL; arXiv:2305.19302; NeurIPS 2023) makes two claims. The
architecture puts hidden representations on every *edge* within the cutoff and runs a transformer
over each atom's neighbour set, so depth is decoupled from receptive field. And — conceptually
more important — it supplies a general symmetrisation protocol that takes *any* smooth model and
makes it exactly rotationally equivariant a posteriori, preserving smoothness, translation
invariance and permutation invariance. Applied to PET it costs almost no accuracy. The rhetorical
consequence is that equivariance stops being an architectural prerequisite and becomes a
post-processing option.

**PET-MAD** (Mazitov, Bigi, Kellner, Pegolo, Tisi, Fraux, Pozdnyakov, Loche and Ceriotti;
arXiv:2503.14118) attacked the data axis. MAD — Massive Atomic Diversity — is small, deliberately
diverse and above all *internally consistent*: 95,595 structures over 85 elements in eight subsets
(bulk crystals, rattled, randomly substituted, surfaces, clusters, 2D, molecular crystals,
molecular fragments), all computed at one level of theory (PBEsol in Quantum ESPRESSO) rather than
stitched from heterogeneous settings. A 3.3M-parameter PET trained on it reached 31.3 meV/atom
hull-distance MAE, beating MACE-MP-0-L (15.8M parameters, 1.58M structures). The paper also
audited its own approximate equivariance quantitatively, measuring rotational discrepancy on
Lebedev–Laikov quadrature grids and finding symmetry breaking **one to two orders of magnitude
below the model's own prediction error**.

The scale-up paper (Bigi, Pegolo, Mazitov, Schmidt and Ceriotti, arXiv:2601.16195) makes the
sharpest version of the argument, and it deserves to be taken seriously rather than dismissed:
rotational symmetry can be learned with roughly **20,000 augmentations**, which is affordable,
whereas permutational symmetry would require $30!$ and therefore must stay hard-coded. The two
symmetries are not on the same footing, so it is coherent to impose one and learn the other. The
[symmetry chapter](../tutorial/index.html#symmetry) explores this trade-off directly.

**MatterSim** (Yang et al., Microsoft Research AI for Science; arXiv:2405.04967) took M3GNet's
architecture essentially unchanged — Microsoft explicitly did not innovate on architecture — and
replaced the data with a 17M-structure uncertainty-driven active-learning corpus spanning 0–5000 K
and up to 1000 GPa. The point is thermodynamics: a potential trained only on relaxation paths has
never seen a hot, compressed, anharmonic solid. It reports Gibbs free energies agreeing with
experiment to 15 meV/atom up to 1000 K. The corpus is closed.

**OMat24** (Barroso-Luque, Shuaibi, Fu, Wood, Dzamba, Gao, Rizvi, Zitnick and Ulissi, FAIR;
arXiv:2410.12771) is the open answer to that closed corpus and the single biggest data-side lever
on the current leaderboard. It takes Alexandria structures and perturbs them hard: rattled
Boltzmann sampling (500 candidates per seed at $\sigma = 0.5$ Å, of which 5 are drawn Boltzmann at
300, 500 and 1000 K), 50-step AIMD at 1000 K and 3000 K in NVT and NPT, and re-relaxation of
mildly rattled structures. The result is 100,824,585 training structures over at most 3,227,606
materials — two orders of magnitude more force and stress labels than MPtrj, deliberately
concentrated off-equilibrium where MP data is empty.

It also introduced a bookkeeping fact that governs how the leaderboard must be read. OMat24 uses
VASP PAW version 54 pseudopotentials where MP uses version 52, and the mean formation-energy
difference is 13.5 meV/atom. An OMat24-only model therefore sits on a shifted reference and cannot
be scored against the MP convex hull, which is why the recipe is always *pretrain on OMat24, then
fine-tune on MPtrj + sAlex* — abbreviated **OAM** in every model name at the top of the board.

---

## The family tree

```
LEGEND     ═══>  direct architectural descent (equations, layers or code carried over)
           --->  conceptual influence (idea borrowed, machinery not)


ROOT I — DESCRIPTORS              ROOT II — INVARIANT GRAPHS       ROOT III — EQUIVARIANCE
────────────────────              ──────────────────────────       ───────────────────────
Behler-Parrinello 2007            MPNN 2017  (Gilmer, Google)      G-CNN 2016 (Cohen-Welling)
 E = Σ E_i ; ACSF ; f_c(R)         message/update/readout           regular rep, weight sharing
   ║                                ║                                ║
   ╠═> GAP 2010 / SOAP 2013         ╠═> SchNet 2017 (Schutt)         ╠═> Steerable CNN 2016
   ║    density, bispectrum         ║    cfconv W(‖r_ij‖) ; smooth   ║    IRREP TYPE SYSTEM
   ║    GP regression               ║      ║                         ║      ║
   ║      ║                         ║      ╠═> PhysNet 2019 (Unke)   ║      ╚═> Kondor-Trivedi 2018
   ║      ╚═> SNAP 2015 (Sandia)    ║      ║    poly envelope        ║          convolution NECESSARY
   ║           linear on bispectrum ║      ║    + Coulomb + D3       ║
   ╠═> MTP 2016 (Shapeev)           ║      ╠═> DimeNet 2020 (TUM)    ╠═> TFN 2018 (Thomas, Smidt...)
   ║    Cartesian moment tensors    ║      ║    triplets; Bessel     ║    F = R(r)·Y_l(rhat) ; CG
   ║    completeness + conv. rate   ║      ║    + envelope           ║      ║
   ║                                ║      ╚═> GemNet 2021           ║      ╠═> 3D Steerable 2018
   ╚═> ACE 2019 (Drautz, ICAMS)     ║           quadruplets +        ║      ║    GENERAL SOLUTION
        complete basis; density     ║           dihedrals; proof     ║      ║    + GATE nonlinearity
        trick, O(N_c) at any        ║           "invariant suffices" ║      ╠═> N-body/CG Nets 2018
        body order                  ╚═> DeePMD 2018 ═> DeepPot-SE    ║      ╠═> Cormorant 2019
             ║                            local frames; smooth       ║      ║    equivariant MP,
             ║                                 ║                     ║      ║    ENERGIES ONLY
   Pozdnyakov et al. 2020:                     ║                     ╠═> SE(3)-Transformer 2020
   power spectrum is INCOMPLETE                ║                     ╚═> e3nn 2020 (library)
   (3-body AND 4-body degeneracies)            ║                          THE SUBSTRATE
             ┆                                 ║                              ║
             ┆ (motivates)                     ║                              ║
             └ - - - - - - - - - - - - - - - - ╫ - - - - - - - - - - - -┐     ║
                                               ║                        ▼     ▼
                                               ║        ╔═══════════════════════════════╗
                    ACE product basis - - - -┐ ║        ║   NequIP  2021 / 2022         ║
                                             ┆ ║        ║   Kozinsky + Smidt            ║
                                             ┆ ║        ║   TFN layers + parity +       ║
                                             ┆ ║        ║   Bessel radial MLP +         ║
                                             ┆ ║        ║   gradient forces             ║
                                             ┆ ║        ╚═══════════════════════════════╝
                                             ┆ ║              ║      ║       ║       ║
       DimeNet Bessel basis - - - - - - - - -┼-╫--------------┘      ║       ║       ║
                                             ┆ ║                     ║       ║       ║
                          ┌──────────────────┴─╫─────────────────────┘       ║       ║
                          ▼                    ║                             ║       ║
                  MACE 2022 (Csanyi)           ║              Allegro 2022 ═══╝       ║
                  ACE product basis inside     ║              STRICT LOCALITY,        ║
                  the MP loop; nu=3 =>         ║              no message passing,     ║
                  4-body msgs; T=2             ║              10^8-atom MD            ║
                     ║                         ║                                     ║
                     ║        Multi-ACE 2022/25 (BOTH GROUPS + DRAUTZ)      SevenNet 2024 (SNU)
                     ║        "one design space": ACE = one layer;          layer-wise MPI ghost
                     ║        NequIP = a sparsification                     exchange => deep MP
                     ║             ┆                                            ║
                     ║             ┆ (reframes)                                 ║
                     ▼             ▼                                            ▼
         MACE-MP-0 2023      ┌────────────────┐                          SevenNet-0 / -Omni
         (foundation MLIP)   │  2022-2026:    │                          EquFlash / EquFlashV2
                             │  SCALING AND   │                          (Samsung, FlashTP kernel)
                             │  PLURALISATION │
                             └────────────────┘

TRANSFORMER BRANCH (TFN ═> Equiformer)              DATA / SCALING BRANCH
  Equiformer 2022 (Liao, Smidt, MIT)                  M3GNet 2022 (Chen, Ong) ═> MatterSim 2024
    ║  O(L^6) cap                                        mines MP ionic steps
    ╠═> eSCN 2023 (Passaro, Zitnick, FAIR)             CHGNet 2023 (Ceder) ══> MPtrj (the corpus)
    ║     SO(3) -> SO(2), O(L^6) -> O(L^3)             GNoME 2023 (DeepMind) — AL flywheel
    ╠═> EquiformerV2 2024  [DIRECT FORCES]             OMat24 2024 (FAIR) — 101M structures
    ╠═> eSEN 2025 (Fu, FAIR) [CONSERVATIVE]              => the "-OAM" recipe
    ║     smoothness + NVE gate; the corrective
    ╠═> UMA 2025 (FAIR) — MoLE, 5 domains            UNCONSTRAINED BRANCH
    ╚═> EquiformerV3 2026 (MIT + Mirror Physics)       PET 2023 (Pozdnyakov, Ceriotti)
          SwiGLU-S^2, smooth attention                   edge transformer, no equivariance
                                                         + exact a-posteriori symmetrisation
ACE BRANCH (independent of TFN)                          ║
  ACE 2019 ═> PACE ═> GRACE 2024 (Bochkarev,             ╠═> PET-MAD 2025 (95,595 structs)
              Lysogorskiy, Drautz) ═> GRACE-2L/3L        ╚═> PET-OAM-XL 2026 (730M params)
  ACE 2019 ═> TACE 2026 (Xu, Xie, Hu, ShanghaiTech)    ORB v2 2024 ═> ORB v3 2025 (Orbital)
              irreducible CARTESIAN tensors              GNS + diffusion pretrain + augmentation
              ═> TECE-OAM-RRA 2026  [CPS #1]
                                                      INVARIANT BRANCH (still alive)
  DeepPot-SE 2018 ═> DPA-1/2/3 ═> DPA-4 2026            M3GNet/CHGNet ---> MatRIS 2026 (ICT CAS)
              EMFA + eSCN's SO(2) trick                   invariant 3-body attention, O(N)
```

---

## The culmination: what the 2026 leaderboard actually says

Matbench Discovery (Riebesell, Goodall, Benner, Chiang, Deng, Ceder, Asta, Lee, Jain and Persson;
arXiv:2308.14920; Nature Machine Intelligence **7**, 836 (2025), DOI 10.1038/s42256-025-01055-1)
is the benchmark that matters here. Models are trained on Materials Project data and tested on the
WBM substitution set — 215,488 unique prototypes after de-duplication, 15.3% of them stable — by
predicting relaxed energies, converting to convex-hull distances, and thresholding at 0 eV/atom.
The composite CPS aggregates stability classification (F1), geometry-optimisation quality (RMSD),
and $\kappa_{\mathrm{SRME}}$, the symmetric relative mean error on phonon thermal conductivity,
which is bounded on $[0, 2]$ with 2 meaning complete disagreement. The [hull
chapter](../tutorial/index.html#hull) explains why stability is a decision problem rather than a
regression problem.

Here is the CPS top ten as of 2026-07-29, with lineage:

| # | Model | CPS | F1 | $\kappa_{\mathrm{SRME}}$ | RMSD | Params | Lineage | Group |
|---|---|---|---|---|---|---|---|---|
| 1 | TECE-OAM-RRA-1.0 | 0.908 | 0.929 | 0.0927 | 0.0575 | 222M | ACE, irreducible Cartesian tensors + rotary attention | ShanghaiTech (Xu, Xie, Hu) |
| 2 | EquFlashV2-45M-OAM | 0.907 | 0.929 | 0.0941 | 0.0577 | 44.9M | TFN → NequIP → SevenNet-0 → EquFlash | Samsung Electronics |
| 3 | EquiformerV3+DeNS-OAM | 0.902 | **0.931** | 0.1178 | 0.0595 | 30.3M | TFN → Equiformer → eSCN → V2 → V3 | MIT + Mirror Physics |
| 4 | GRACE-3L-OAM-L | 0.900 | 0.925 | 0.1211 | 0.0575 | 42.1M | ACE → graph ACE | ICAMS Bochum |
| 5 | PET-OAM-XL | 0.898 | 0.924 | 0.1192 | 0.0596 | 730M | **unconstrained** edge transformer | COSMO / EPFL |
| 6 | TACE-OAM-L | 0.889 | 0.910 | 0.1260 | 0.0606 | 82.9M | ACE, irreducible Cartesian tensors | ShanghaiTech |
| 7 | eSEN-30M-OAM | 0.888 | 0.925 | 0.1704 | 0.0608 | 30.2M | eSCN / Equiformer | FAIR at Meta |
| 8 | EquFlash-29M-OAM | 0.888 | 0.919 | 0.1583 | 0.0602 | 28.7M | SevenNet-0 → NequIP | Samsung Electronics |
| 9 | NequIP-OAM-XL | 0.886 | 0.906 | 0.1252 | 0.0630 | 32.1M | NequIP itself | MIR group, Harvard |
| 10 | MatRIS-10M-OAM | 0.877 | 0.921 | 0.2183 | 0.0601 | 10.4M | **invariant** three-body attention | ICT, Chinese Academy of Sciences |

(TACE-OAM-RRA-Preview sits at CPS 0.905 but is explicitly flagged as a preview superseded by TECE
and carries no rank, which is why a naive sort of the models table misleads.)

### Finding 1: the top tier is architecturally plural

The story most often told about this leaderboard is that equivariance won. That story was true
when it was first told and is now only half true.

**About half** the CPS top ten descends from the Tensor Field Networks / e3nn lineage: EquFlash and
EquFlashV2 (via SevenNet-0, from NequIP), NequIP-OAM-XL, and — on the eSCN/Equiformer transformer
branch — eSEN-30M-OAM and EquiformerV3+DeNS-OAM. The rest come from genuinely independent
lineages:

- **GRACE-3L-OAM-L (#4)** descends from Drautz's Atomic Cluster Expansion, not from TFN. Its
  equivariance comes from ACE's own spherical-harmonic product basis. The group's own PRX paper is
  titled "Graph Atomic Cluster Expansion for Semilocal Interactions **beyond** Equivariant Message
  Passing."
- **TACE-OAM-L (#6)** and **TECE-OAM-RRA-1.0 (#1)** are irreducible-**Cartesian**-tensor ACE
  variants whose defining move is to *avoid Clebsch–Gordan spherical tensor products* — the
  intellectual descendants of Shapeev's moment tensors as much as of anything in the e3nn family.
  (Honest complication worth recording: the published metadata for TECE also lists Wigner-D
  rotations to $SO(2)$ frames in its Radial Rotary Attention, imported from eSCN. The lineage is
  braided, not pure — but the body-order machinery is Cartesian ACE, not spherical CG.)
- **PET-OAM-XL (#5)** is, in its own authors' words, "a rotationally unconstrained and
  transformer-based graph neural network." The PET-MAD paper states plainly that "the PET
  architecture imposes no explicit rotational symmetry constraints, but learns to be equivariant
  through data augmentation." It is not in the TFN/e3nn lineage at all; it is not equivariant by
  construction; and at 730M parameters it is the largest model on the board.
- **MatRIS-10M-OAM (#10)** is explicitly an **invariant** MLIP — "an invariant MLIP that introduces
  attention-based modeling of three-body interactions," with separable $O(N)$ attention — from Zhou,
  Hu, Tan and Jia at the Institute of Computing Technology, CAS. It sits in the
  Behler–Parrinello → SchNet → DimeNet/GemNet → M3GNet/CHGNet invariant lineage, and its paper
  positions it *against* equivariant models rather than as their descendant. It is also the cheapest
  top-ten model at 10.4M parameters and reaches F1 0.921, seventh best on the entire board.

So the two strongest counterexamples are not merely non-e3nn; they are **non-equivariant by
design**. This is the single most important correction to the received story. Note also that the
invariant and unconstrained models pay for it in a specific, legible place: MatRIS's
$\kappa_{\mathrm{SRME}}$ of 0.218 is the worst in the top ten, roughly 2.4× the leader's, and its
MPtrj-only sibling sits at 0.489. The prior still buys something; it just no longer buys the
ranking.

### Finding 2: the 2024–26 jump was more architecture than data

The second received story is that the leap of the last two years was bought with data. On the
leaderboard's own composite metric, the architecture/recipe term is the larger one.

Matbench Discovery maintains a **frozen-data control track**: models trained only on MPtrj's 1.58M
structures. Watching it is the closest thing to a natural experiment the field has.

| Date | Model (MPtrj only) | CPS | F1 | $\kappa_{\mathrm{SRME}}$ |
|---|---|---|---|---|
| 2023-07 | MACE-MP-0 | 0.637 | 0.669 | 0.682 |
| 2024-07 | SevenNet-0 | 0.697 | 0.724 | 0.762 |
| 2024-12 | SevenNet-l3i5 | 0.714 | 0.760 | 0.550 |
| 2025-03 | eSEN-30M-MP | 0.797 | 0.831 | 0.340 |
| 2025-10 | MatRIS-10M-MP | 0.778 | 0.847 | 0.489 |
| 2026-04 | EquiformerV3+DeNS-MP | 0.830 | 0.863 | 0.275 |
| 2026-06 | DPA-4.0.1-Pro-MPtrj | 0.840 | 0.857 | 0.211 |

Decomposing the overall 0.637 → 0.908 CPS gain: **+0.203 (about 75%) is achieved at frozen MPtrj
data**, leaving roughly +0.068 (25%) as the residual data contribution. On F1 the split is 72/28.
On $\kappa_{\mathrm{SRME}}$ the frozen-data track alone delivers 0.682 → 0.211, a factor of 3.2,
against the further 0.211 → 0.093 (factor 2.3) from data.

Two decisive individual data points:

- **DPA-4.0.1-Pro-MPtrj** (22.8M parameters, MPtrj only, 1.58M structures) reaches CPS 0.840, F1
  0.857 and $\kappa_{\mathrm{SRME}}$ 0.211 — matching or beating OAM-trained models with roughly
  **71× more data** (GRACE-2L-OAM at CPS 0.837, SevenNet-MF-ompa at 0.844). Architecture and
  recipe alone erased a 71-fold data deficit. It descends, remarkably, from DeepPot-SE by way of
  DPA-1/2/3, with eSCN's $SO(2)$ trick grafted on.
- Within a fixed architecture, swapping MPtrj for OAM is worth about **+0.09 to +0.16 CPS**
  (GRACE-2L 0.681 → 0.837; eSEN 0.797 → 0.888; EquiformerV3 0.830 → 0.902) — real and consistent,
  but smaller than the **+0.33 CPS spread among models trained on that same OAM corpus**.

And the data effect is **not primarily a scale effect**. The OMat24 paper's own Table 2
(arXiv:2410.12771v2) holds the eqV2-S architecture and the MPtrj fine-tune fixed and varies only
the pretraining corpus: pretraining on **OC20 (134M structures)** gives F1 0.837 and 33 meV/atom,
while pretraining on **OMat24 (101M structures — fewer)** gives F1 0.890 and 26 meV/atom. The
larger corpus is worse. What OMat24 bought was non-equilibrium and compositional diversity plus
correction of systematic softening, not raw size. PET-MAD makes the same point from the other end
with 95,595 structures.

The apparent narrowness of architectural differences at the very top is survivorship bias. The
counterexample is sitting on the board: **eqV2 M**, trained on the full OAM corpus, scores F1
0.917 (9th of 51) but CPS 0.558 (36th of 41) and $\kappa_{\mathrm{SRME}}$ 1.771 — dead last, 88.5%
of the way to the metric's worst attainable value — because it uses a direct force head.
eSEN-30M-MP, trained on 70× less data but energy-conserving, beats it on CPS by +0.239.

### Finding 3: the κ_SRME story is about second derivatives, not about a label

It is tempting to compress the previous paragraph into "direct forces break phonons." The
association is strong: on the leaderboard every submitted direct-force model sits above
$\kappa_{\mathrm{SRME}}$ 1.6, and every model below 1.0 is energy-gradient-based. But
non-conservativeness is neither necessary nor sufficient:

- **CHGNet is conservative** (`targets: EFS_GM`, G = gradient) and scores
  $\kappa_{\mathrm{SRME}} = 1.7167$ — essentially identical to eqV2 M's 1.7707. That refutes
  non-conservativeness as *necessary*.
- **M3GNet** (conservative) scores 1.4094; **AlphaNet-v1-MPtrj** (conservative) scores 1.3046 with
  the highest imaginary-mode rate on the board.
- **Orb-v3** reports *direct*-force variants with good scores: orb-v3-direct-inf-mpa at F1 0.883 and
  $\kappa_{\mathrm{SRME}} = 0.348$, better than conservative SevenNet-0 (0.762), MACE-MP-0 (0.682)
  and M3GNet (1.409). That refutes it as *sufficient*.

The underlying quantity is the accuracy of the **second derivatives** of the energy surface — the
force constants that phonons depend on. Direct-force training does not constrain them, but a
conservative model with a jagged PES can get them wrong too. This is exactly eSEN's diagnosis:
envelopes, absence of maximum-neighbour caps, and modest radial bases matter as much as the
`EFS_G` label. eqV2, notably, uses `max_neighbors: 20`.

---

## What equivariance actually bought, and what separates the leaders now

**It bought data efficiency, and the evidence for that is older and cleaner than NequIP.** 3D
Steerable CNNs classified CATH protein architectures with 143,560 parameters where a conventional
3D CNN needed 15,878,764, and kept the lead as training data was cut by up to 16×. Rotated Tetris,
trained on a single orientation, went from 27% to 99%. NequIP matched DeepMD on liquid water and
ice using 133 structures against 133,500. The mechanism is not mysterious: an equivariant layer is
a constrained layer, and the constraint is exactly the symmetry the data obeys, so weight sharing
across the group replaces examples.

**It bought a strong and correct inductive prior, and exactness.** Equivariance error for a
properly constructed model is $\sim 10^{-7}$ against $\sim 10^{-1}$ for an unconstrained one.
Symmetric Hessians, exactly degenerate modes at symmetric geometries, forces that rotate with the
molecule by construction rather than by approximation — these come free rather than being learned
to within tolerance. And because the CG product is the *unique* equivariant bilinear map, the
design space is small enough to search exhaustively; Multi-ACE could enumerate it on one page.

**That prior is less decisive at 2026 data scales, and the field has measured why.** PET measures
its own rotational discrepancy at one to two orders of magnitude below its own prediction error —
once the residual symmetry violation is far under the irreducible error, exactness stops paying.
Ceriotti's group quantified the asymmetry that makes this coherent: rotational symmetry can be
learned from roughly 20,000 augmentations, whereas permutational symmetry would need $30!$, so
hard-coding one and learning the other is a defensible engineering position rather than a
concession. From the other side, EquiformerV3 found that once pretrained on OMat24, $L_{\max} = 6$
gives "almost no improvement" over $L_{\max} = 4$ — the marginal value of *more* equivariant
capacity has gone to roughly zero. And the empirical decomposition says the same thing: the spread
among top models trained on the identical OAM corpus (+0.33 CPS) exceeds the value of the corpus
itself (+0.09 to +0.16).

**What separates the leaders now is a different list.** Energy-conserving gradient forces, yes —
but as a proxy for what actually matters, which is accurate second derivatives, and therefore
smooth potential energy surfaces: envelope functions on messages *and* inside attention softmaxes,
no maximum-neighbour caps, modest radial bases, no grid discretisation above Nyquist. Training
corpus *diversity and consistency* rather than size, since 101M diverse structures beat 134M less
diverse ones and 95,595 internally consistent ones go a remarkably long way. And, increasingly,
kernel engineering: EquFlashV2 reached CPS 0.907 at 44.9M parameters essentially by making the
tensor product 41.6× faster than e3nn and spending the savings on scale, while EquiformerV3
reached CPS 0.902 in 5.7k GPU-hours against UMA-M-1.1's 0.889 in more than 129k — a 22.6× compute
ratio at higher accuracy.

There is also a visible cost to every specialisation, which is worth stating because leaderboard
summaries hide it. The #1 model, TECE, ranks 31st of 33 on diatomic-curve behaviour (CDS 0.270).
Allegro's strict locality gives it the best diatomics score in the top group (CDS 0.828) but a
$\kappa_{\mathrm{SRME}}$ of 0.319, roughly 3.4× the leader's. GRACE-2L-OAM-L has the best MD
stability score among the top models (CMDS 0.759) while sitting 13th on CPS. Nobody is winning
everywhere.

### The groups

The work described here comes from a surprisingly small number of places, and knowing which is a
genuine aid to reading the literature.

The **Kozinsky group (MIR, Harvard SEAS)** and **Tess Smidt's Atomic Architects (MIT)** are one
extended circle and account for a startling fraction of the tree: TFN (Smidt, at Google
Accelerated Science), e3nn (Geiger and Smidt), NequIP, Allegro, the nequip framework, Equiformer,
EquiformerV2 and EquiformerV3. **Gábor Csányi's group at Cambridge** produced GAP and SOAP at one
end and MACE and the MACE-MP foundation models at the other, with Christoph Ortner (Warwick/UBC)
as the recurring mathematical collaborator. **Ralf Drautz at ICAMS Bochum** produced ACE and, with
Bochkarev and Lysogorskiy, GRACE. **Michele Ceriotti's COSMO lab at EPFL** produced the
incompleteness proof (with Pozdnyakov, and with Bartók and Csányi as co-authors), PET, PET-MAD and
PET-OAM — and is the field's most articulate advocate for *not* imposing equivariance. **FAIR
Chemistry at Meta** produced OC20, SCN and eSCN, EquiformerV2 (with MIT), OMat24, eSEN and UMA,
and is the source of most of the field's data infrastructure. **MDIL at Seoul National
University** (Seungwu Han's group, with Park, Kim and Hwang) produced SevenNet and the parallel-MD
communication scheme that made deep message passing practical.

Beyond those: **Shapeev at Skoltech** (MTP), **Thompson and colleagues at Sandia** (SNAP),
**Schütt, Müller and colleagues at TU Berlin/FHI** (SchNet), **Unke and Meuwly at Basel**
(PhysNet),
**Gasteiger/Klicpera and Günnemann at TU Munich** (DimeNet, GemNet), **Zhang, Wang, Car and E** and
the DeepModeling/AISI ecosystem (DeePMD → DPA-4), **Chen and Ong at UC San Diego** (M3GNet),
**Ceder's group at Berkeley/LBNL** (CHGNet and MPtrj), **Google DeepMind** (GNoME), **Microsoft
Research AI for Science** (MatterSim), **Orbital Materials** (ORB), **Samsung Electronics**
(EquFlash and FlashTP), **Xu, Xie and Hu at ShanghaiTech** (TACE, TECE), **Zhou, Hu, Tan and Jia
at ICT, CAS** (MatRIS), **Cohen and Welling at Amsterdam** (G-CNNs, Steerable CNNs), **Kondor at
Chicago** (N-body Nets, CG Nets, Cormorant), **Weiler, Geiger and Boomsma** (3D Steerable CNNs),
and
**Riebesell, Goodall and the Matbench Discovery authors**, whose benchmark is the reason any of this
can be compared at all.

The reading-group observation that started this project — that the current leader is a graph
attention transformer incorporating rotational symmetry — was accurate when it was made and is now
half right. The better summary is this. Built-in equivariance was the decisive idea of 2021–2023
and remains the majority design at the top of the field. But by 2026 the leaderboard has become
architecturally plural: ACE-derived Cartesian-tensor models, a rotationally unconstrained
transformer, and an explicitly invariant network all sit in the top ten. What equivariance bought
was data efficiency and a strong, correct prior. At today's data scales that prior is less
decisive, and other things — smooth energy surfaces, accurate second derivatives, training-corpus
diversity and raw kernel throughput — now separate the leaders.
