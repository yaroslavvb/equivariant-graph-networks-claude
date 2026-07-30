---
title: "NequIP, read closely"
---

# NequIP, read closely

Batzner, Musaelian, Sun, Geiger, Mailoa, Kornbluth, Molinari, Smidt and Kozinsky,
*E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials*,
Nature Communications **13**, 2453 (2022); preprint arXiv:2101.03164 (v1 8 January 2021, v3
16 December 2021); doi:10.1038/s41467-022-29939-5. Code at `github.com/mir-group/nequip`.

This is a close reading of one paper. It is the paper that most people point to when they say the
machine-learned-interatomic-potential field "went equivariant," and it is worth reading at the level
of its actual equations, because the popular summary of it — *equivariance buys a thousand times the
data efficiency, therefore equivariance is what matters* — is a compression that loses the two things
an expert most wants to know: which experiment supports which clause, and what was already true
before the paper appeared. Sections 1 through 8 below reconstruct the model and the results.
Section 9 is the part that disagrees with the popular summary; it is not a footnote, and the numbers
in it come from the paper's own tables.

Numbers quoted as "the paper's" are transcribed from the arXiv v3 tables unless stated otherwise;
where the published Nature Communications version and the preprint appear to differ, that is flagged.

---

## 1. What the model is for

A machine-learned interatomic potential (MLIP) is a surrogate for the Born–Oppenheimer potential
energy surface. Given the Cartesian positions $\vec{r}_1,\dots,\vec{r}_N$ and atomic numbers
$Z_1,\dots,Z_N$ of a configuration, it returns a total energy $E$, and from that the forces
$\vec{F}_i = -\partial E/\partial \vec{r}_i$ and the virial stress. It is trained on labels produced
by an electronic-structure method — almost always Kohn–Sham density functional theory — and it exists
because that method is too expensive to run inside a molecular dynamics loop.

The cost gap is the whole motivation, and it is worth stating in units rather than adjectives.
Kohn–Sham DFT with a plane-wave or Gaussian basis costs $\mathcal{O}(N_e^3)$ in the number of
electrons through the orthogonalisation and diagonalisation of the Hamiltonian, with a large prefactor
from the self-consistency loop. Behler and Parrinello's original 2007 paper measured their neural
network as roughly **five orders of magnitude** faster than the underlying DFT for a 64-atom silicon
cell (Phys. Rev. Lett. **98**, 146401 (2007), doi:10.1103/PhysRevLett.98.146401). Drautz's Atomic
Cluster Expansion paper puts a linear-scaling analytic potential at $\approx 10^{-6}$ s/atom for
energies and $\approx 10^{-4}$ s/atom for forces (Phys. Rev. B **99**, 014104 (2019)). Ab-initio
molecular dynamics at 0.5 fs timesteps needs $2\times 10^5$ force evaluations per 100 ps of
simulated time; at DFT cost, on a few hundred atoms, that is a supercomputer allocation, and the
system sizes and timescales at which interesting things happen — nucleation, ionic transport,
amorphous relaxation, catalytic turnover — are further out still. An MLIP is an attempt to buy DFT
accuracy at classical-force-field cost, with the accuracy defined *relative to the reference method*,
not relative to experiment.

There is a second, subtler requirement that the field learned the hard way and which NequIP inherits
without much comment: the surrogate has to be *smooth*, because it is going to be differentiated and
then integrated for millions of steps. A model that reproduces energies well but has discontinuous or
kinked derivatives will heat, drift, or blow up in NVE dynamics. This is why every element of the
architecture below is built from smooth functions with smooth cutoffs, and why the choice of an
envelope function with vanishing first and second derivatives at the cutoff is not cosmetic.

[Chapter 1 of the tutorial](../tutorial/index.html#problem) develops this at length and makes the
cost gap adjustable: what the Born–Oppenheimer surface is, where the labels come from and why
reproducing a functional perfectly also reproduces its errors, and the six features that make this an
unusual regression problem rather than ordinary supervised learning with an expensive labeller. The
[opening chapter](../tutorial/index.html#question) frames why the question came up at all.

---

## 2. The Behler–Parrinello decomposition, and what it commits you to

Every model discussed here starts from one move, made by Behler and Parrinello in 2007:

$$E = \sum_{i=1}^{N} \varepsilon_i, \qquad \varepsilon_i = \varepsilon_{Z_i}\!\left(\{\vec{r}_{ij}, Z_j\} : r_{ij} < r_c\right).$$

The total energy is a sum of *atomic* energies; each atomic energy is a function only of that atom's
neighbourhood inside a cutoff radius $r_c$; and all atoms of the same element share one function
$\varepsilon_{Z}$. This is not a physical truth — the exact many-body energy is not a sum of local
terms — but it is an approximation that buys four things at once, and they are the four properties
that make an MLIP usable:

1. **Size extensivity and transferability.** The model is defined for any $N$. Train on 64 atoms,
   run on $10^6$. Nothing in the parameterisation refers to the number of atoms.
2. **Linear cost.** Evaluating $E$ costs $\mathcal{O}(N)$ given a neighbour list, because each site
   sees a bounded number of neighbours.
3. **Permutation invariance, for free.** $\varepsilon_{Z}$ is the *same function* for every atom of
   element $Z$, and it consumes the neighbour set through symmetric aggregations (sums), so
   relabelling atoms cannot change $E$. There is no combinatorial sorting step and no canonical
   ordering to get wrong. Behler and Parrinello obtained this by weight sharing across per-element
   subnetworks; NequIP obtains it by weight sharing plus sum-pooling over neighbours.
4. **Parallelism.** Sites are independent given the neighbour list, so the model domain-decomposes.

The costs are equally concrete. The local decomposition truncates long-range electrostatics and
dispersion at $r_c$, so charged defects and polar interfaces need extra machinery. And the atomic
energies $\varepsilon_i$ are unobservable: only their sum is supervised. That degeneracy is real —
different $\{\varepsilon_i\}$ decompositions fit the same $E$ — and it is why force labels, which
constrain $3N$ numbers per frame instead of one, carry so much more information than energy labels.
The paper notes this directly: there are $3N$ force components per training frame but only one energy
target. This asymmetry is the reason force-weighted losses dominate MLIP training, and it is the
reason the data-efficiency claim in §7 has to be read carefully — 133 water frames is 133 energies
but $\sim\!76{,}500$ force components.

NequIP changes none of this. It keeps $E = \sum_i \varepsilon_i$, keeps the cutoff, keeps the weight
sharing. What it changes is the *representation* of the neighbourhood that $\varepsilon_i$ consumes.

---

## 3. The symmetry requirements, stated precisely

### 3.1 The group

The relevant symmetry group of a molecule or a crystal in free space is
$E(3) = \mathbb{R}^3 \rtimes O(3)$: translations, semidirect with the full orthogonal group in three
dimensions. $O(3) = SO(3) \times \{\pm I\}$ contains proper rotations and the inversion $-I$; the
distinction between $E(3)$ and $SE(3)$ — whether inversion is included — is not decorative. A model
that is only $SE(3)$-equivariant cannot in principle distinguish enantiomers' *response* properties
correctly, and, more relevantly here, it can build a pseudoscalar (an inversion-odd scalar such as
$\vec{a}\cdot(\vec{b}\times\vec{c})$) and mistakenly use it as an energy contribution. The Born–
Oppenheimer energy is a true scalar, invariant under inversion. NequIP tracks parity explicitly to
enforce this, which is one of its few genuine departures from Tensor Field Networks.

Separately, and not part of $E(3)$, there is the permutation group $S_N$ acting on identical atoms.

### 3.2 Invariance of the energy, equivariance of the forces

The energy is invariant. For $g = (R, \vec{t}) \in E(3)$ acting on positions as
$\vec{r}_i \mapsto R\vec{r}_i + \vec{t}$,

$$E\big(\{R\vec{r}_i + \vec{t}\}\big) = E\big(\{\vec{r}_i\}\big).$$

The forces are *not* invariant. They are equivariant — specifically, they transform as vectors — and
this is a theorem, not a design choice, as long as the forces are defined as gradients of an
invariant energy. Write $\vec{r}\,' = R\vec{r} + \vec{t}$ and $E'(\{\vec{r}\,'\}) = E(\{\vec{r}\})$.
Then by the chain rule, using $\partial r'^{\,\alpha}_j / \partial r^{\beta}_i = \delta_{ij}R^{\alpha\beta}$,

$$\frac{\partial E}{\partial r_i^{\beta}} = \sum_{j,\alpha} \frac{\partial E'}{\partial r_j'^{\,\alpha}} \frac{\partial r_j'^{\,\alpha}}{\partial r_i^{\beta}} = \sum_{\alpha} R^{\alpha\beta}\, \frac{\partial E'}{\partial r_i'^{\,\alpha}},$$

that is, $\nabla_{\vec r_i} E = R^{\mathsf T} \nabla_{\vec r\,'_i} E'$, and therefore

$$\vec{F}_i\big(\{R\vec{r} + \vec{t}\}\big) = R\, \vec{F}_i\big(\{\vec{r}\}\big).$$

So: **if you get the energy's invariance right and take an exact gradient, the forces' equivariance is
automatic.** A model that predicts forces as a separate output head has to earn the same property
some other way, and — more importantly — has no guarantee that its force field is a gradient field at
all. See §6.

### 3.3 The general condition

The condition that generalises both cases is the one NequIP writes as its Eq. (1). A map
$f: X \to Y$ between vector spaces carrying representations $\rho_{\text{in}}$ and $\rho_{\text{out}}$
of a group $G$ is equivariant if

$$f\big(\rho_{\text{in}}(g)\,x\big) = \rho_{\text{out}}(g)\, f(x) \qquad \forall\, g \in G,\ \forall\, x \in X.$$

Invariance is the special case $\rho_{\text{out}} = \mathbb{1}$. The design problem is then: choose
$\rho_{\text{in}}$ and $\rho_{\text{out}}$ for every intermediate layer, and build layers that satisfy
the identity exactly rather than approximately. Because equivariance composes — if $f_1$ is
$(\rho_0,\rho_1)$-equivariant and $f_2$ is $(\rho_1,\rho_2)$-equivariant then $f_2 \circ f_1$ is
$(\rho_0,\rho_2)$-equivariant — a deep network is equivariant as soon as every layer is.

For $O(3)$ every finite-dimensional representation decomposes into irreducibles labelled by a
rotation order $\ell = 0,1,2,\dots$ and a parity $p \in \{+1,-1\}$ (conventionally written `e`/`o`).
The $(\ell,p)$ irrep has dimension $2\ell+1$ and acts by the Wigner D-matrix
$D^{(\ell)}(R) \in \mathbb{R}^{(2\ell+1)\times(2\ell+1)}$, times $p$ under inversion. So NequIP's
features carry four indices,

$$V^{(\ell,p)}_{a\,c\,m}, \qquad a = \text{atom},\ c = \text{channel},\ m \in \{-\ell,\dots,\ell\},$$

and a layer's type is fully specified by the multiset of $(\ell, p)$ it consumes and produces —
e3nn's `Irreps` string, e.g. `64x0e + 64x0o + 64x1o + 64x1e`. $\ell=0$ is a scalar, $\ell=1$ a
vector, $\ell=2$ a symmetric traceless rank-2 tensor, and so on. The
[symmetry chapter](../tutorial/index.html#symmetry) makes the group action concrete, and the
[machinery chapter](../tutorial/index.html#machinery) builds the Wigner matrices and Clebsch–Gordan
coefficients numerically.

### 3.4 How each symmetry is obtained

| Symmetry | How NequIP gets it | Exact or approximate |
| --- | --- | --- |
| Translation | The network never sees absolute positions, only relative vectors $\vec{r}_{ij} = \vec{r}_j - \vec{r}_i$ | Exact, by construction |
| Permutation of like atoms | Weight sharing per element + sum-pooling over neighbours | Exact, by construction |
| Rotation $SO(3)$ | Features are typed by $\ell$; every operation is built from Clebsch–Gordan products and $\ell$-preserving linear maps | Exact, up to floating point |
| Inversion (parity) | Features carry $p$; products obey $p_o = p_i p_f$ (Eq. 7) | Exact, by construction |
| Energy conservation | $\vec F = -\nabla E$ via autodiff | Exact, up to integrator error |

The point of the table is that nothing here is learned or regularised. The symmetries are structural
properties of the function class. That is the claim being made when a model is called "equivariant by
construction," and it is the property that distinguishes this design from data augmentation, which
buys only an approximate and distribution-dependent version of the same thing.

---

## 4. Why scalars are not enough

Before NequIP, essentially every competitive MLIP was *invariant*: the neighbourhood was reduced to a
list of $O(3)$-invariant scalars — distances $r_{ij}$, angles $\cos\theta_{ijk}$, sometimes dihedrals
— and a regressor was fitted on top. Behler–Parrinello symmetry functions, SOAP/GAP, SNAP, MTP,
ACE, SchNet, DimeNet and GemNet are all in this family (they differ in *which* invariants and how
many). Invariance is achieved by throwing angular structure away early and never recovering the
relative orientation of different pieces of the environment.

The failure this induces is not subtle. Two distinct neighbour sets can share every distance and
still be different environments — the *homometric* pairs of crystallography. The repository's own
check (`results/descriptor_blindness.json`, generated by
`python/experiments/descriptor_blindness.py`) constructs a minimal instance: three unit-length
neighbours at $0^\circ, 120^\circ, 240^\circ$ versus three at $0^\circ, 30^\circ, 60^\circ$. The
sorted radii are bitwise identical; the Behler $G_2$ radial symmetry functions on a 24-point $\mu$
grid at $\eta = 20$ are bitwise identical (max absolute difference exactly $0.0$); but a
Stillinger–Weber three-body energy with $\lambda = 21$, $\cos\theta_0 = -1/3$ evaluates to $1.75$ on
the first and $74.999$ on the second, a factor of $42.9$. The $\ell = 1$ power of the two neighbour
densities differs by everything: $9.5\times10^{-32}$ versus $1.782$. A distance-only descriptor is
provably blind to a physically enormous difference. That is the content of the
[blindness chapter](../tutorial/index.html#blindness).

DimeNet made the same argument in words in 2020 (arXiv:2003.03123): a GNN restricted to distances
within a cutoff cannot distinguish certain molecules, so directional information has to be put back.
And NequIP's own Table 1 measures the cost of not doing so — on original MD-17 aspirin at a
1,000-configuration budget, SchNet (distances only) gives a force MAE of 58.5 meV/Å while DimeNet
(invariant, but with angles) gives 21.6 meV/Å, a factor of 2.7. Hold onto that number; it comes back
in §9.

NequIP's answer is different from DimeNet's. Rather than enumerating triplets and quadruplets of
hand-chosen invariants, it keeps the *equivariant* objects — the $\ell \geq 1$ tensors — as internal
features, and only takes invariants at the very end when the scalar energy is read out. Relative
orientation is therefore preserved through the depth of the network, and many-body angular
information arises from products of tensors rather than from explicit $k$-tuple enumeration.

---

## 5. The architecture, block by block

### 5.1 The graph

For each configuration, build a radius graph: atom $i$ is connected to atom $j$ if
$r_{ij} = \|\vec{r}_j - \vec{r}_i\| < r_c$, with periodic images included for crystals. The cutoff
used in the paper is $r_c \in [4.0, 6.0]$ Å depending on system (4.0 Å for the MD-17 family, 4.5 Å
for the Cheng *et al.* water set, 5.0 Å for Cu(110), Li$_4$P$_2$O$_7$ and LiPS, 6.0 Å for the
water/ice set compared against DeepMD). Node features start as a one-hot encoding of $Z_i$ passed
through a trainable linear self-interaction, i.e. a learned element embedding of type `Nx0e`.
Edge attributes are computed once per edge from $\vec{r}_{ij}$ and never learned: a radial part and
an angular part, described next.

### 5.2 The radial basis: Bessel functions with a polynomial envelope

The radial dependence is expanded in the basis DimeNet introduced, obtained by taking the
$\ell = m = 0$ solution of the spherical Bessel equation, $j_0(x) = \sin(x)/x$, whose roots are at
$n\pi$. NequIP's Eq. (6):

$$B_n(r_{ij}) = \frac{2}{r_c}\,\frac{\sin\!\left(\dfrac{n\pi}{r_c} r_{ij}\right)}{r_{ij}}\; f_{\text{env}}(r_{ij}, r_c), \qquad n = 1,\dots,8 .$$

Two details matter. First, the basis is small: **8** functions, against DimeNet's 6–16 and SchNet's
$\sim$300 Gaussians. Second — and this is a NequIP-specific change — the coefficients $n\pi$ are
*trainable*, initialised at the Bessel roots and then optimised by backpropagation. The basis stops
being a fixed orthogonal system and becomes an initialisation.

The envelope is DimeNet's polynomial cutoff with $p = 6$, in the variable $d = r/r_c$:

$$u(d) = 1 - \frac{(p+1)(p+2)}{2}d^{\,p} + p(p+2)\,d^{\,p+1} - \frac{p(p+1)}{2}d^{\,p+2},$$

which satisfies $u(1) = u'(1) = u''(1) = 0$. This is the smoothness guarantee referred to in §1: the
energy, the forces *and* the force constants are continuous as an atom crosses the cutoff. A cosine
cutoff (Behler–Parrinello Eq. 3) gives continuous forces but a discontinuous second derivative; a
hard cutoff gives discontinuous forces. In 2026 hindsight this choice looks more important than it
did in 2021 — see the $\kappa_{\text{SRME}}$ discussion in the benchmark report.

The radial function that actually multiplies the filter is an MLP on this basis (Eq. 5):

$$R(r_{ij}) = W_n\,\sigma\!\left(\cdots \sigma\!\left(W_2\,\sigma\!\left(W_1 B(r_{ij})\right)\right)\right),$$

three hidden layers of 64 neurons with SiLU activations. This MLP is a *scalar* function of a scalar
argument, which is the whole trick: because $r_{ij}$ is invariant, an arbitrarily nonlinear
$R(r_{ij})$ cannot break equivariance. As TFN put it and NequIP repeats, all the learnable weights in
the filter live in the rotationally invariant radial function.

### 5.3 The angular projection

The direction $\hat{r}_{ij} = \vec{r}_{ij}/r_{ij}$ is projected onto real spherical harmonics
$Y^{(\ell_f)}_{m_f}(\hat{r}_{ij})$ for $\ell_f = 0,\dots,\ell_{\max}$. Under a rotation
$\hat{r} \mapsto R\hat{r}$ these transform as
$Y^{(\ell)}(R\hat r) = D^{(\ell)}(R)\, Y^{(\ell)}(\hat r)$ — that is, they *are* the $(\ell,(-1)^\ell)$
irrep, evaluated at a point on the sphere. This is the only place geometry enters the angular
channel, and it enters exactly, with no learned parameters.

Combining §5.2 and §5.3 gives the filter, NequIP Eq. (4), identical to TFN Eq. (2):

$$F^{(\ell_f,\ell_i)}_{c\,m_f}(\vec{r}_{ij}) = R^{(\ell_f,\ell_i)}_{c}(r_{ij})\; Y^{(\ell_f)}_{m_f}(\hat{r}_{ij}).$$

### 5.4 The convolution: a Clebsch–Gordan product summed over neighbours

Now the two equivariant objects — the filter, of type $\ell_f$, and the neighbour's feature, of type
$\ell_i$ — have to be combined into something of a definite type $\ell_o$. The only bilinear map that
does this equivariantly is the Clebsch–Gordan tensor product,

$$(u \otimes v)^{(\ell)}_{m} = \sum_{m_1 = -\ell_1}^{\ell_1}\ \sum_{m_2 = -\ell_2}^{\ell_2} C^{(\ell,m)}_{(\ell_1,m_1)(\ell_2,m_2)}\, u^{(\ell_1)}_{m_1} v^{(\ell_2)}_{m_2},$$

which is nonzero only when the **triangle rule** $|\ell_1 - \ell_2| \le \ell \le \ell_1 + \ell_2$
holds. For $\ell_1 = \ell_2 = 1$ this reproduces exactly the vector algebra one already knows:
$C^{(0,0)}_{(1,i)(1,j)} \propto \delta_{ij}$ is the dot product, $C^{(1,i)}_{(1,j)(1,k)} \propto
\epsilon_{ijk}$ is the cross product, and $\ell = 2$ is the symmetric traceless outer product. There
is no freedom here: Kondor and Trivedi's analysis of convolution for compact groups
(arXiv:1802.03690, ICML 2018) shows the only linear operation covariant with the group action is
per-$\ell$ mixing by a learnable matrix, so all the nonlinearity in the geometric channel must come
from the CG product.

NequIP's convolution, Eq. (8), is TFN's Eq. (3) with parity indices attached:

$$L^{(\ell_o,p_o,\ell_f,\ell_i,p_f,p_i)}_{a\,c\,m_o} = \sum_{m_f,\, m_i} C^{(\ell_o,m_o)}_{(\ell_i,m_i)(\ell_f,m_f)} \sum_{b \,\in\, S} R^{(\ell_f,\ell_i,p_f,p_i)}_{c}(r_{ab})\, Y^{(\ell_f)}_{m_f}(\hat{r}_{ab})\, V^{(\ell_i,p_i)}_{b\,c\,m_i}.$$

Three things to notice.

**Parity bookkeeping (Eq. 7).** $p_o = p_i\,p_f$. Since $Y^{(\ell_f)}$ has parity $(-1)^{\ell_f}$,
the allowed output types are determined jointly by the triangle rule and this sign rule. Tracking
parity separately from $\ell$ is what upgrades TFN's $SE(3)$-equivariance to $E(3)$-equivariance, and
it is why NequIP's irreps strings contain both `0e` and `0o` — even and odd scalars — rather than
just scalars.

**Channel structure.** The product is taken channel-wise ($c$ appears on both factors), as in
Cormorant, rather than across all channel pairs; a full outer product over channels would be
quadratic in width. The mixing across channels is done afterwards by self-interaction (§5.5).

**Normalisation.** The neighbour sum is divided by $\sqrt{\bar N}$, where $\bar N$ is the average
number of neighbours in the training set, so that feature magnitudes do not scale with coordination
number. This is a variance-preserving choice, not a physical one, but it matters for training
stability across systems whose densities differ.

The [layer chapter](../tutorial/index.html#layer) assembles exactly this object step by step and
checks its equivariance numerically.

### 5.5 Self-interaction

Between and after convolutions, features of the same $(\ell,p)$ are mixed across channels by an
atom-wise linear map,

$$V^{(\ell,p)}_{a\,c\,m} \;\longmapsto\; \sum_{c'} W^{(\ell,p)}_{c c'}\, V^{(\ell,p)}_{a\,c'\,m},$$

with weights independent of $m$. Because $W$ acts on the channel index and is constant across $m$, it
commutes with $D^{(\ell)}(R)$ and is trivially equivariant. This is the $1\times1$ convolution of the
equivariant world: it is where most of the parameters live and where the different CG paths that
produced the same $(\ell_o,p_o)$ get recombined. NequIP uses species-dependent weights in these
layers.

### 5.6 Nonlinearity: why componentwise activation is illegal, and what a gate does

This is the point at which equivariant networks stop looking like ordinary ones. Take an $\ell=1$
feature $\vec{v} \in \mathbb{R}^3$, which transforms as $\vec{v} \mapsto R\vec{v}$, and apply ReLU
componentwise. Let $\vec{v} = (1,0,0)$ and let $R$ be the rotation by $\pi$ about $z$, so
$R\vec{v} = (-1,0,0)$. Then

$$\sigma(R\vec{v}) = (0,0,0) \qquad \text{but} \qquad R\,\sigma(\vec{v}) = R(1,0,0) = (-1,0,0).$$

They differ. Componentwise nonlinearity applied to a non-scalar irrep breaks equivariance for every
$\sigma$ that is not linear, because $\sigma$ acts in a particular basis and $D^{(\ell)}(R)$ mixes
that basis. The $2\ell+1$ components of an irrep are not independent features; they are coordinates
of one geometric object.

The standard fix, which NequIP takes from the 3D Steerable CNNs of Weiler, Geiger, Welling, Boomsma
and Cohen (arXiv:1807.02547, NeurIPS 2018), is the **gate**. For each non-scalar feature
$V^{(\ell,p)}_{c}$ with $\ell > 0$, the layer also produces a scalar $s_c$ of type `0e`, and the
nonlinearity is a multiplication:

$$V^{(\ell,p)}_{c\,m} \;\longmapsto\; \sigma(s_c)\; V^{(\ell,p)}_{c\,m}.$$

Since $\sigma(s_c)$ is invariant, the product still transforms by $D^{(\ell)}(R)$ — equivariance is
preserved exactly, while the *magnitude* of each geometric feature is modulated nonlinearly by a
learned invariant function of the environment. The direction is untouched; only the length is gated.

Scalars need no gate and are activated directly, but with a parity-aware choice that is easy to miss
and is exactly right: **even scalars go through SiLU, odd scalars through tanh.** The reason is that
an odd scalar $s$ satisfies $s \mapsto -s$ under inversion, and a nonlinearity applied to it must
therefore be an odd function to preserve that transformation law. $\tanh(-x) = -\tanh(x)$, so tanh
qualifies. SiLU, $\sigma_{\text{SiLU}}(x) = x\,\mathrm{sigmoid}(x)$, does not:
$\sigma_{\text{SiLU}}(-x) \ne -\sigma_{\text{SiLU}}(x)$. Using SiLU on a `0o` channel would silently
destroy parity equivariance while leaving rotation equivariance intact — a bug that no rotation test
would catch.

### 5.7 Residual update, readout, pooling

Interaction blocks are stacked with a ResNet-style update,

$$x^{(k+1)} = f\!\left(x^{(k)}\right) + \mathrm{SelfInteraction}\!\left(x^{(k)}\right),$$

where $f$ is the sequence self-interaction $\to$ convolution $\to$ concatenation $\to$
self-interaction, with the gate nonlinearity applied to the result. The skip branch is a linear
self-interaction rather than the identity because the input and output irreps of a block generally
differ; a linear equivariant map is the only legal way to reconcile them. The paper uses 5
interaction blocks for molecules and 6 for periodic systems.

The readout is deliberately minimal. The $\ell = 0$ (scalar) features from the final convolution pass
through two atom-wise self-interaction layers down to a single scalar per atom, which is interpreted
as $\varepsilon_i$, and then

$$E = \sum_{i} \varepsilon_i .$$

The global pooling is a plain sum — the only aggregation consistent with extensivity — and it is the
*only* place where information from different atoms is combined without going through the graph. Note
where the invariance is enforced: everything before the readout is equivariant, carrying $\ell \geq 1$
tensors; invariance is imposed exactly once, at the end, by selecting the $\ell = 0$ channel. That
ordering is the architectural thesis of the paper. An invariant network takes the invariants at layer
zero and can never recover what it discarded; NequIP defers the projection to the last step. Because
the readout keeps only `0e` and discards `0o`, the pseudoscalar channels cannot leak into the energy.

### 5.8 The settings actually used

| Data set | $\ell_{\max}$ | Features | $r_c$ (Å) | $\lambda_E$ | $\lambda_F$ |
| --- | --- | --- | --- | --- | --- |
| MD-17 | 3 | 64 | 4.0 | 1 | 1,000 |
| revised MD-17 | $\{0,1,2,3\}$ | 64 | 4.0 | 1 | 1,000 |
| CCSD / CCSD(T) | 3 | 64 | 4.0 | 1 | 1,000 |
| Water + ices (vs DeepMD) | 2 | 32 | 6.0 | see §7.4 | see §7.4 |
| Formate on Cu(110) | 2 | 32 | 5.0 | 1 | 2,704 |
| Li$_4$P$_2$O$_7$ | 2 | 32 | 5.0 | 1 | 43,264 |
| LiPS | 2 | 32 | 5.0 | 1 | 6,889 |
| Water (Cheng *et al.*) | $\{0,1,2,3\}$ | 32 | 4.5 | 1 | 36,864 |

Common settings: Bessel basis of size 8; polynomial envelope $p=6$; radial MLP of 3 hidden layers
$\times$ 64 neurons with SiLU; Adam with AMSGrad; learning rate 0.01 for molecules and 0.005 for
periodic systems; batch size 5 for molecules, 1 for periodic systems.

The force loss weights in that table look arbitrary until you square-root them. The paper states a
208-atom Li$_4$P$_2$O$_7$ cell and an 83-atom LiPS cell, and $208^2 = 43{,}264$, $83^2 = 6{,}889$
exactly; the Cheng *et al.* water figure $36{,}864 = 192^2$ is consistent with that dataset's
64-molecule (192-atom) cell. So for the periodic systems $\lambda_F = N^2$, which is what you get if
the energy term is normalised per atom (introducing $1/N^2$ in the squared error) and you want the
force term to enter with unit relative weight. The molecular values (1,000) do not follow the pattern
and appear to have been tuned. This is an inference from the tabulated numbers, not a statement the
paper makes.

---

## 6. Forces as $-\nabla E$

NequIP predicts one scalar per configuration and differentiates it:

$$\vec{F}_i = -\frac{\partial E}{\partial \vec{r}_i} = -\sum_{j}\frac{\partial \varepsilon_j}{\partial \vec{r}_i},$$

evaluated by reverse-mode automatic differentiation through the entire network. The paper is explicit
about why: forces obtained this way ensure "both energy conservation and rotation-equivariant
forces." The second half of that is §3.2. The first half is the more interesting one.

A vector field $\vec{F}(\{\vec r\})$ on configuration space is a gradient field if and only if its
Jacobian is symmetric,

$$\frac{\partial F_i^{\alpha}}{\partial r_j^{\beta}} = \frac{\partial F_j^{\beta}}{\partial r_i^{\alpha}},$$

equivalently if and only if the work done around every closed loop in configuration space vanishes.
For a gradient field these hold identically, because the Jacobian *is* $-\nabla^2 E$, the Hessian,
which is symmetric by equality of mixed partials. For a directly predicted force field there is no
such constraint: the model has $3N$ independent outputs and nothing ties them together. In
microcanonical dynamics the consequence is a systematic energy drift, because the integrator is
accumulating work from a field that is not conservative.

The repository's controlled experiment (`results/conservative_vs_direct.json`, from
`python/experiments/conservative_vs_direct.py`) isolates this cleanly, because it uses *the same*
$O(3)$-equivariant feature basis for both models — the conservative model's parameter space is
literally a 28-dimensional subspace of the direct model's 79-dimensional one. Both are exactly
equivariant: max relative deviation under rotation is $1.5\times10^{-14}$ for the conservative model
and $2.6\times10^{-14}$ for the direct one. Both fit the noisy forces about equally well; on noise-free
test forces the direct model is in fact slightly *better*, RMSE $0.0114$ against $0.0129$, a ratio of
$0.88$, which is what you would expect from having $2.8\times$ the parameters. What separates them is
everything that force RMSE does not
measure: the RMS relative asymmetry of the force Jacobian is $1.4\times10^{-8}$ (conservative, i.e.
autodiff noise) versus $7.3\times10^{-3}$ (direct) — a ratio of $5.2\times10^{5}$ — and the median
relative net work around closed loops is $3.1\times10^{-16}$ versus $3.6\times10^{-3}$. Equivariance
and conservativeness are orthogonal properties, and only one of them is visible in the headline
metric. This is the [forces chapter](../tutorial/index.html#forces).

The paper draws the contrast with a specific target: models that use TFN layers to predict force
*vectors* directly (Townshend *et al.*), of which it says the predicted forces "are not guaranteed by
construction to conserve energy since they are not obtained as gradients of the total potential
energy." That is a precise and correct criticism, and it is worth noting for later that this argument
was, in 2021, made against an equivariant model — so "equivariance" and "conservative forces" were
already understood as separable design axes at the time.

The price of the gradient formulation is one backward pass per force evaluation, roughly doubling to
tripling inference cost, and the requirement that the whole network be twice differentiable if you
also want Hessians (phonons, force constants). Both costs are why direct-force models keep being
proposed.

---

## 7. The experiments

### 7.1 Revised MD-17 and the rotation-order scan

The headline table. Revised MD-17 (rMD17) is Christensen and von Lilienfeld's recomputation of the
original MD17 trajectories at PBE/def2-SVP with very tight SCF convergence and a very dense
integration grid, precisely because the original MD17 labels carried numerical noise large enough to
corrupt benchmark conclusions (arXiv:2007.09593; Mach. Learn.: Sci. Technol. **1**, 045018 (2020)).
The dataset's own documentation warns against training on more than 1,000 of its samples, since the
structures come from a time series and are not independent. The standard budget — used here — is 950
training plus 50 validation configurations, quoted as "1,000 reference configurations."

Energy MAE in meV, force MAE in meV/Å:

| Molecule | | FCHL19 | UNiTE | GAP | ANI | ACE | GemNet-(T/Q) | NequIP $\ell{=}0$ | $\ell{=}1$ | $\ell{=}2$ | $\ell{=}3$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aspirin | E | 6.2 | 2.4 | 17.7 | 16.6 | 6.1 | – | 25.2 | 3.8 | 2.4 | **2.3** |
| | F | 20.9 | 7.6 | 44.9 | 40.6 | 17.9 | 9.5 | 41.9 | 12.9 | 8.7 | **8.5** |
| Azobenzene | E | 2.8 | 1.1 | 8.5 | 15.9 | 3.6 | – | 20.3 | 1.1 | 0.8 | **0.7** |
| | F | 10.8 | 4.2 | 24.5 | 35.4 | 10.9 | – | 42.3 | 5.6 | 4.2 | **3.6** |
| Benzene | E | 0.3 | 0.07 | 0.75 | 3.3 | **0.04** | – | 3.2 | 0.09 | 0.06 | **0.04** |
| | F | 2.6 | 0.73 | 6.0 | 10.0 | 0.5 | 0.5 | 10.3 | 0.4 | 0.4 | **0.3** |
| Ethanol | E | 0.9 | 0.62 | 3.5 | 2.5 | 1.2 | – | 2.0 | 1.0 | 0.5 | **0.4** |
| | F | 6.2 | 3.7 | 18.1 | 13.4 | 7.3 | 3.6 | 13.7 | 7.6 | 4.2 | **3.4** |
| Malonaldehyde | E | 1.5 | 1.1 | 4.8 | 4.6 | 1.7 | – | 4.4 | 1.6 | 0.9 | **0.8** |
| | F | 10.2 | 6.6 | 26.4 | 24.5 | 11.1 | 6.6 | 23.4 | 10.4 | 6.0 | **5.2** |
| Naphthalene | E | 1.2 | 0.46 | 3.8 | 11.3 | 0.9 | – | 14.7 | 0.4 | 0.3 | **0.2** |
| | F | 6.5 | 2.6 | 16.5 | 29.2 | 5.1 | 1.9 | 20.1 | 2.0 | 1.3 | **1.2** |
| Paracetamol | E | 2.9 | 1.9 | 8.5 | 11.5 | 4.0 | – | 17.5 | 2.1 | **1.4** | **1.4** |
| | F | 12.2 | 7.1 | 28.9 | 30.4 | 12.7 | – | 37.6 | 10.8 | **6.9** | **6.9** |
| Salicylic acid | E | 1.8 | 0.73 | 5.6 | 9.2 | 1.8 | – | 11.4 | 1.0 | 0.8 | **0.7** |
| | F | 9.5 | 3.8 | 24.7 | 29.7 | 9.3 | 5.3 | 28.7 | 5.7 | 4.2 | **4.0** |
| Toluene | E | 1.6 | 0.45 | 4.0 | 7.7 | 1.1 | – | 9.7 | 0.5 | **0.3** | **0.3** |
| | F | 8.8 | 2.5 | 17.8 | 24.3 | 6.5 | 2.2 | 27.2 | 2.7 | 1.8 | **1.6** |
| Uracil | E | 0.6 | 0.58 | 3.0 | 5.1 | 1.1 | – | 10.0 | 0.6 | **0.4** | **0.4** |
| | F | 4.2 | 3.8 | 17.6 | 21.4 | 6.6 | 3.8 | 25.8 | 4.1 | **3.0** | 3.2 |

The rotation-order scan is the four right-hand columns. Reading force MAEs, the $\ell = 0 \to \ell=1$
step is worth a factor of $3.25$ on aspirin, $10.1$ on toluene, $10.1$ on naphthalene and $25.8$ on
benzene; the geometric mean across the ten molecules is $5.6$. The subsequent steps are much smaller:
$\ell=1 \to \ell=2$ has geometric mean $1.45$ (aspirin $1.48$), and $\ell=2 \to \ell=3$ is
within noise ($1.02$ on aspirin; uracil actually gets *worse*, 3.0 $\to$ 3.2). Total $\ell=0 \to
\ell=3$: geometric mean $8.9$, from $4.0$ on ethanol to $34.3$ on benzene. This "big first step, then
saturation" shape is the empirical fact that persuaded the field, and it is also the fact that §9.3
takes apart.

The [$\ell_{\max}$ chapter](../tutorial/index.html#lmax) reruns a controlled miniature of this scan.

### 7.2 Original MD-17

Same 1,000-configuration budget, older and noisier labels, and a different comparison set. NequIP at
$\ell = 3$:

| Molecule | | SchNet | DimeNet | sGDML | PaiNN | SpookyNet | GemNet-(T/Q) | NewtonNet | UNiTE | NequIP $\ell{=}3$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aspirin | E | 16.0 | 8.8 | 8.2 | 6.9 | 6.5 | – | 7.3 | – | **5.7** |
| | F | 58.5 | 21.6 | 29.5 | 14.7 | 11.2 | 9.4 | 15.1 | **6.8** | 8.2 |
| Ethanol | E | 3.5 | 2.8 | 3.0 | 2.7 | 2.3 | – | 2.6 | – | **2.2** |
| | F | 16.9 | 10.0 | 14.3 | 9.7 | 4.1 | 3.7 | 9.1 | 4.0 | **3.1** |
| Malonaldehyde | E | 5.6 | 4.5 | 4.3 | 3.9 | 3.4 | – | 4.2 | – | **3.3** |
| | F | 28.6 | 16.6 | 17.8 | 13.8 | 7.2 | 6.7 | 14.0 | 6.9 | **5.6** |
| Naphthalene | E | 6.9 | 5.3 | 5.2 | 5.0 | 5.0 | – | 5.1 | – | **4.9** |
| | F | 25.2 | 9.3 | 4.8 | 3.3 | 3.9 | 2.2 | 3.6 | 2.8 | **1.7** |
| Salicylic acid | E | 8.7 | 5.8 | 5.2 | 4.9 | 4.9 | – | 5.0 | – | **4.6** |
| | F | 36.9 | 16.2 | 12.1 | 8.5 | 7.8 | 5.4 | 8.5 | 4.2 | **3.9** |
| Toluene | E | 5.2 | 4.4 | 4.3 | 4.1 | 4.1 | – | 4.1 | – | **4.0** |
| | F | 24.7 | 9.4 | 6.1 | 4.1 | 3.8 | 2.6 | 3.8 | 3.1 | **2.0** |
| Uracil | E | 6.1 | 5.0 | 4.8 | **4.5** | 4.6 | – | 4.6 | – | **4.5** |
| | F | 24.3 | 13.1 | 10.4 | 6.0 | 5.2 | 4.2 | 6.5 | 4.2 | **3.3** |

NequIP wins six of seven force columns and does not win aspirin, where UNiTE's direct-learning result
(6.8) is better. The SchNet-versus-DimeNet contrast noted in §4 lives in this table.

### 7.3 CCSD and CCSD(T)

The most consequential experiment for practitioners, because it is the one where data scarcity is not
a benchmark convention but a physical fact: coupled-cluster reference data is expensive enough that
1,000 configurations is genuinely all you get. Aspirin at CCSD, the rest at CCSD(T), 1,000
configurations (950/50):

| Molecule | sGDML E / F | GemNet-(T/Q) F | NequIP $\ell{=}3$ E / F |
| --- | --- | --- | --- |
| Aspirin | 6.9 / 33.0 | 10.3 | **2.0 / 8.3** |
| Benzene | 0.17 / 1.7 | 0.7 | **0.05 / 0.26** |
| Ethanol | 2.2 / 15.2 | 3.1 | **0.36 / 3.0** |
| Malonaldehyde | 2.6 / 16.0 | 5.9 | **0.72 / 4.5** |
| Toluene | 1.3 / 9.1 | 2.7 | **0.27 / 1.7** |

A caution on this table that is easy to miss: GemNet's own paper (arXiv:2106.08903, NeurIPS 2021)
reports NequIP's CCSD(T) force MAEs as 14.7 (aspirin), 9.4 (ethanol), 16.0 (malonaldehyde), 4.4
(toluene) and 0.8 (benzene) — roughly $1.8$–$3\times$ worse than the values above — while its sGDML
and GemNet columns agree with NequIP's to the digit. The discrepancy is in the NequIP column only,
and the natural explanation is that GemNet (June 2021) quoted an earlier NequIP preprint than the
December 2021 v3 whose numbers are tabulated here. Benchmark tables in this literature move between
versions, and a comparison assembled from two papers' tables is not necessarily comparing the same
model.

### 7.4 Liquid water and ice, against DeepMD

The most-cited result and the one most often misquoted. The dataset is the water/ice set used in the
DeepMD PRL (Zhang, Han, Wang, Car, E, Phys. Rev. Lett. **120**, 143001 (2018)), computed at PBE0-TS.
DeepMD's published models were trained on 133,500 structures. NequIP was trained on **133** —
0.0996% of that, a factor of 1,004.

Energy RMSE in meV/molecule, force RMSE in meV/Å. NequIP is shown at three loss weightings:
(a) $\lambda_F = 1, \lambda_E = 0$; (b) $\lambda_F = 100, \lambda_E = 1$; (c) $\lambda_F = 10^5,
\lambda_E = 1$.

| System | NequIP (a) E / F | NequIP (b) E / F | NequIP (c) E / F | DeepMD E / F |
| --- | --- | --- | --- | --- |
| Liquid water | – / 12.5 | 1.6 / 51.4 | 1.7 / 12.2 | 1.0 / 40.4 |
| Ice I$_h$ (b) | – / 10.8 | 2.5 / 57.8 | 4.3 / 10.4 | 0.7 / 43.3 |
| Ice I$_h$ (c) | – / 12.5 | 3.9 / 29.1 | 10.2 / 12.2 | 0.7 / 26.8 |
| Ice I$_h$ (d) | – / 10.3 | 2.6 / 24.1 | 12.7 / 10.1 | 0.8 / 25.4 |

On forces the result is what it is advertised to be: 12.2 versus 40.4 meV/Å on liquid water and 10.4
versus 43.3 on ice I$_h$ (b), a factor of $3$–$4$ better on 0.1% of the data. On **energies it is
the other way around**: DeepMD is better on all four systems, by 1.0 against 1.6–1.7 on liquid water
and 0.7 against 4.3 on ice I$_h$ (b), reaching 0.8 against 12.7 on ice I$_h$ (d). The paper does not
hide this — the table prints it — but the popular summary does. The comparison is a force-accuracy
comparison, and the loss-weighting rows make clear that within NequIP itself the two objectives trade
against each other: setting (b) has the best energies among the NequIP variants and the worst forces
by a factor of $4$–$5$.

Three further qualifications belong in the main text rather than a footnote. First, this is a
cross-implementation, cross-recipe comparison: NequIP was trained here, DeepMD's numbers are quoted
from its own paper, with a different code, a different loss, a different optimiser and a different
data split. Nothing was held fixed except the reference data. Second, the paper itself notes that
"the version of DeepMD published [in the PRL] is not smooth, and a smooth version has since been
proposed" — that is DeepPot-SE (arXiv:1805.09003, NeurIPS 2018), which improved DeePMD's force errors
by roughly $1.5$–$2.5\times$ on the molecular benchmarks — "however, [it] does not report results on
the water/ice systems." So the baseline is a known-superseded version, and the paper says so. Third,
133 structures is not 133 numbers: a 192-atom cell contributes 576 force components per frame, so the
supervision is $\sim\!7.7\times10^4$ force components plus 133 energies. The thousand-fold reduction
is in *frames*, which is the right unit for "how much DFT do I have to run," but not the right unit
for "how many constraints does the fit see."

### 7.5 Formate dehydrogenation on Cu(110)

A reactive surface-chemistry problem with a metal present — a much less forgiving setting than
gas-phase organics, because metallic screening makes the local energy decomposition work harder and
because the dataset contains bond-breaking. 2,500 training structures drawn from a 6,855-structure
AIMD dataset. Per-element force MAE:

| C | O | H | Cu | mean | energy |
| --- | --- | --- | --- | --- | --- |
| 19.9 | 73.1 | 13.0 | 47.6 | **38.4 meV/Å** | **0.50 meV/atom** |

The 38.4 figure is the unweighted mean over the four species, $(19.9+73.1+13.0+47.6)/4 = 38.4$, not a
sample-weighted average — worth knowing, since oxygen is the worst species by a wide margin and
oxygen is not the most abundant.

### 7.6 Amorphous Li$_4$P$_2$O$_7$

A 208-atom cell, trained on 1,000 structures drawn from a 3000 K molten trajectory, then evaluated on
a quench to 600 K:

| | Energy MAE | Force RMSE |
| --- | --- | --- |
| Melt test set | 0.4 meV/atom | 62.7 meV/Å |
| Quenched trajectory | 0.5 meV/atom | 38.1 meV/Å |

The interesting part is not the error but the extrapolation: the model was trained *only* on the
high-temperature melt and reproduced the radial distribution function and the angular distribution
functions of the quenched amorphous phase — the P–O–O tetrahedral angle and the O–P–P bridging angle
— against AIMD. Angular distribution functions are a genuine three-body observable; matching them
after training on a structurally different phase is a stronger statement than any MAE.

### 7.7 Li$_{6.75}$P$_3$S$_{11}$ and superionic transport

An 83-atom cell of a superionic solid electrolyte, $r_c = 5.0$ Å, $\ell_{\max} = 2$:

| Training structures | Energy MAE (meV/atom) | Force RMSE (meV/Å) |
| --- | --- | --- |
| 10 | 2.03 | 142.0 |
| 100 | 0.44 | 36.8 |
| 1,000 | 0.12 | 11.2 |
| 2,500 | 0.08 | 6.6 |

And the observable that actually matters: the Li$^+$ self-diffusivity from the model's own molecular
dynamics, against AIMD.

$$D_{\text{NequIP}} = 1.25\times10^{-5}\ \text{cm}^2/\text{s}, \qquad D_{\text{AIMD}} = 1.37\times10^{-5}\ \text{cm}^2/\text{s},$$

a relative error of $8.8\%$ (the paper rounds this to 9%), from 2,500 training structures. Diffusivity
is a mean-squared-displacement
observable extracted from long trajectories; getting it to within 9% is a much more demanding test of
the potential than a force MAE, because it requires the *barriers* to be right, not just the forces
near sampled configurations, and because errors compound over the trajectory rather than averaging
out. This is, in practice, the result that convinced materials people the thing was usable.

### 7.8 The learning curves

The data-efficiency claim rests on the shape of the error-versus-training-set-size curve, not on any
single point. The abstract says NequIP "outperforms existing models with up to three orders of
magnitude fewer training data, challenging the widely held belief that deep neural networks require
massive training sets."

The two concrete supports are:

- **Bulk liquid water** (the Cheng *et al.* dataset, PNAS 2019). A Behler–Parrinello network trained
  on 1,303 structures reaches $\approx 120$ meV/Å force RMSE evaluated on the remaining 290. NequIP
  at $\ell = 2$ reaches **129.8 meV/Å with 100 training points and 103.4 meV/Å with 250** — matching
  a 1,303-structure BPNN at roughly a tenth to a fifth of the data, and beating it at a fifth.
  (The published Nature Communications version appears to give 123.3 and 98.3 for these two figures;
  the arXiv v3 values are quoted here.)
- **The log-log slope argument.** On the same water curves and on MD-17 aspirin, the paper's claim is
  not merely that the equivariant curve sits lower but that it *falls faster*: "The equivariant
  NequIP networks break this pattern. Instead they follow a log-log slope with larger magnitude,
  meaning that they learn faster as new data become available." A shifted curve is a constant-factor
  advantage; a steeper curve is an advantage that grows. The distinction is the entire strategic
  content of the paper, and the paper's own further observation — "further increasing the rotation
  order $\ell$ beyond $\ell = 1$ again only shifts the learning curve and does not result in an
  additional change in log-log slope" — narrows the load-bearing evidence to the single contrast
  $\ell = 0$ versus $\ell = 1$.

---

## 8. The hypothesis-class reading

Why would equivariance help at all? The clean argument is a statistical one and has nothing to do
with neural networks specifically.

Let $\mathcal{H}$ be the hypothesis class of all functions the architecture can express, and let
$\mathcal{H}^G \subset \mathcal{H}$ be the subclass of $G$-equivariant ones. The true target
$f^\star$ lies in $\mathcal{H}^G$, because physics says so. Training an unconstrained model means
searching $\mathcal{H}$ and spending data to discover, approximately and only on the training
distribution, the constraint you already knew exactly. Building the constraint in means searching
$\mathcal{H}^G$ directly. Two consequences follow:

1. **Sample complexity falls** roughly with the "size" of the group being quotiented out — for
   $SO(3)$, a three-parameter continuous group, this is not a constant factor but a reduction in the
   effective dimension of the function being estimated.
2. **Generalisation is exact off-distribution in the group directions.** An equivariant model's error
   on a rotated test configuration equals its error on the unrotated one, identically, for every
   rotation, including rotations no training example ever came near. An augmented model's does not.

The repository's augmentation experiment (`results/augmentation_vs_equivariance.json`) makes the
second point measurable. Built-in invariance gives invariance error at the level of floating-point
noise: mean $\sim\!4\times10^{-16}$, worst case at the $10^{-15}$ level. Augmentation never gets below
$0.115$ for any fit that is not near-trivial, even at 256 random rotations per sample. And the
off-distribution probe is decisive: a model augmented only with rotations about $z$ scores nRMSE
$0.688$ on canonically-oriented test data and $2.329$ on rotated test data — worse than predicting
the mean. Augmentation buys invariance on the augmentation distribution and nothing outside it.

The corresponding statement for NequIP specifically is that the $\ell$-truncation defines a nested
family $\mathcal{H}^G_{\ell_{\max}=0} \subset \mathcal{H}^G_{\ell_{\max}=1} \subset \cdots$, all
equivariant, differing in *angular resolution*. That framing is correct and useful. It is also
exactly where the paper's causal claim runs into trouble.

---

## 9. What the paper did not show

This section is the one the popular summary omits. Every number in it is from the paper's own tables
or from primary sources, and the conclusions are the product of an adversarial check against those
sources.

### 9.1 Priority: what was actually first

NequIP never claims to be first. Its self-description is scoped and hedged: "the introduction of a
deep learning energy-conserving interatomic potential for both molecules and materials built on
E(3)-equivariant convolutions over geometric tensors." The defensible version of the priority claim
is narrow:

> NequIP was the first **neural message-passing** interatomic potential to combine local atomic
> energies, $\ell \geq 1$ E(3)-equivariant tensor features, **and** gradient-derived conservative
> forces.

Every clause in that sentence is doing work, because each of the three legs existed separately, and
most pairs existed together:

- **The equivariant machinery is not NequIP's.** The paper says so in as many words: "we build on the
  layers introduced in Tensor-Field Networks [Thomas *et al.* 2018], primitives for which are
  implemented in e3nn." Eq. (4) is TFN Eq. (2); Eq. (8) is TFN Eq. (3) with parity indices added. The
  gate is from 3D Steerable CNNs (arXiv:1807.02547). The uniqueness of per-$\ell$ linear mixing is
  Kondor and Trivedi (arXiv:1802.03690). NequIP's genuine additions are systematic parity, the
  trainable-Bessel radial MLP, and — most of all — the empirical program.
- **Cormorant (Anderson, Hy, Kondor, NeurIPS 2019, arXiv:1906.04015)** already did equivariant
  atom-wise message passing with $\ell \geq 1$ spherical-tensor activations ($L = 3$, 16 channels)
  and already benchmarked on MD-17. Its missing leg is forces: Table 1(b) reports "conformational
  energies (in units of kcal/mol) on MD-17," energies only, no gradients, no parity, no periodic
  boundary conditions.
- **Townshend *et al.*** already applied TFN layers to forces — but predicted them directly, so the
  missing leg was conservativeness, not equivariance. NequIP itself makes this criticism (§6).
- **Non-neural prior art already had all three legs.** Moment Tensor Potentials (Shapeev,
  arXiv:1512.06054; Multiscale Model. Simul. **14**, 1153 (2016)) build rotationally covariant
  Cartesian moment tensors $M_{\mu,\nu}(u) = \sum_i f_{\mu,\nu}(|u_i|)\, u_i^{\otimes\nu}$, contract
  them to invariants, decompose the energy into atomic contributions and differentiate analytically.
  ACE (Drautz, Phys. Rev. B **99**, 014104 (2019)) does the same with a spherical-harmonic
  single-bond basis $\phi_{nlm}(\vec r) = \sqrt{4\pi}\,R_{nl}(r)\,Y_l^m(\hat r)$ and a provably
  complete body-order expansion. What is new in NequIP is the *message-passing neural* instantiation,
  not the ingredient list. This matters for reading the 2026 leaderboard, where ACE-derived models
  sit in the top five.
- **PaiNN (Schütt, Unke, Gastegger, arXiv:2102.03150, 5 February 2021)** satisfies all three legs and
  is four weeks later than NequIP v1 — effectively concurrent. NequIP's own text acknowledges it:
  "Both of these methods [PaiNN, NewtonNet] were proposed after NequIP and only make use of $\ell=1$
  tensors."
- **The invariant-versus-equivariant ablation was not first either.** Miller, Geiger, Smidt and Noé,
  "Relevance of Rotationally Equivariant Convolutions for Predicting Molecular Properties"
  (arXiv:2008.08461, ML4Molecules @ NeurIPS 2020), already ran an angular-features-on/off comparison
  on QM9, reporting $\approx 23\%$ average test-error reduction. NequIP's scan was the first on
  forces and data efficiency, not the first of its kind.

The [genealogy chapter](../tutorial/index.html#genealogy) lays out the full dependency graph.

### 9.2 Which ablation lives on which dataset

This is a factual error that has propagated widely, including into talks and survey papers, and it is
worth stating flatly.

**The capacity-controlled ablation is on original MD-17, not revised MD-17.** The rMD17 table in §7.1
does contain $\ell = 0,1,2,3$ rows, but it is a single-budget accuracy table at 1,000 configurations
with **no capacity control and no learning curves**. The experiments that do the causal work live
elsewhere: the learning curves are on bulk liquid water and on "the aspirin molecule in MD-17" — the
original set — and the only capacity controls are in Appendix B, on original MD-17 aspirin. rMD17
appears in Appendix C only as label histograms.

The controls themselves are real and worth describing, because they rule out the most obvious
objection. Two $\ell=0$ baselines were built:

- a **weight-controlled** one, with "increased feature size that matches the number of weights up to
  approx. 0.1% of the $\ell=1$ network";
- a **feature-controlled** one, using "the same number of features as the $\ell=1$ network, i.e.
  $4\times$ more features than the original $\ell = 0$ network." Concretely, the $\ell=1$ network is
  `64x0o + 64x0e + 64x1o + 64x1e`, the original $\ell=0$ network is `64x0e`, and the
  feature-controlled $\ell=0$ network is `512x0e`.

So parameter count and feature count are both controlled. That disposes of "the equivariant model
just has more capacity." It does not dispose of the next objection, which is worse.

### 9.3 The capacity/angular-information confound

NequIP's $\ell = 0$ baseline has **no angular information at all**. The paper says so: "Omitting all
higher-order interactions that go beyond the $0\otimes 0 \to 0$ interaction will result in a
conventional GNN-IP with invariant convolutions over scalar features, similar to e.g. SchNet." A
SchNet-style model sees distances and nothing else.

Therefore the $\ell = 0 \to \ell = 1$ jump — the one contrast on which the log-log-slope argument
rests — confounds **two** changes: the model gains equivariance, *and* it gains angular resolution.
Invariant models can have the second without the first, and NequIP never runs that comparison.

The evidence that the confound is doing most of the work is in NequIP's own tables. Reading the
aspirin force-MAE row of §7.1:

| Model | Angular info? | Equivariant? | Force MAE (meV/Å) |
| --- | --- | --- | --- |
| NequIP $\ell = 0$ | no | no | 41.9 |
| ACE (linear, invariant) | yes | no | 17.9 |
| GemNet-(T/Q) (invariant, directional) | yes | no | **9.5** |
| NequIP $\ell = 1$ | yes | yes | 12.9 |
| NequIP $\ell = 2$ | yes | yes | 8.7 |
| NequIP $\ell = 3$ | yes | yes | 8.5 |

An *invariant* model with good directional features — GemNet, at 9.5 — **beats NequIP $\ell=1$**
(12.9) and comes within 12% of NequIP $\ell=3$ (8.5). A linear, invariant ACE at 17.9 is $2.3\times$
better than NequIP's own $\ell = 0$ baseline. Most of the $41.9 \to 12.9$ gap is therefore
attributable to the presence of angular and body-order information, which invariant architectures
also have, not to equivariance as such.

The same pattern appears in the original-MD-17 table (§7.2), in the same molecule: SchNet
(distances only) 58.5, DimeNet (invariant, *with* angles) 21.6. Adding angular information to a fully
invariant model closes most of the gap between the distance-only baseline and the equivariant models,
which is precisely the variable the $\ell = 0$ baseline lacks.

The repository's own controlled miniature (`results/lmax_learning_curves.json`, from
`python/experiments/lmax_learning_curves.py`) sharpens the point in two directions. First, the
$\ell_{\max} = 0$ deficit is genuinely *informational*, not a capacity artefact: on a Morse +
Stillinger–Weber target, a purely radial invariant model's test RMSE is flat at $0.1696$, $0.1688$,
$0.1688$, $0.1687$, $0.1693$, $0.1698$ as its feature count grows $14 \to 44 \to 90 \to 152 \to 324
\to 560$, while $\ell_{\max}=1$ with 24 features reaches $0.1123$ and $\ell_{\max}=2$ with 34
features reaches $0.0086$. Adding radial capacity to an angular-blind model buys nothing; $12.3\%$ of
the target's variance is unreachable with 560 radial invariants. Second, the paper's "$0\to1$ is the
biggest step" ordering is **target-dependent, not universal**. For the Stillinger–Weber target, whose
angular kernel $(\cos\theta - \cos\theta_0)^2$ has exactly zero Legendre content above $\ell = 2$
(max $|a_\ell|$ for $\ell \geq 3$ is $2.9\times10^{-14}$ against $a_2 = 0.400$), the gains run
$\times 1.48$, then $\times 16.7$, then $\times 0.93$ — the $1\to2$ step is by far the largest and
$\ell_{\max}=3$ is slightly *worse* than $\ell_{\max}=2$, as it should be, since the extra irreps add
estimation variance and no signal. For a neighbour–neighbour repulsion target with $10.8\%$ of its
angular power above $\ell = 2$, the paper's diminishing-returns pattern does hold: $\times 2.23$,
$\times 2.01$, $\times 1.88$. The shape of the $\ell_{\max}$ curve is a property of the target's
angular spectrum, not a universal law about equivariance.

None of this makes the ablation worthless. It makes it *suggestive rather than decisive*. What it
supports is: "$\ell \geq 1$ internal tensor features are a very effective way to get angular and
many-body resolution into a message-passing potential, at a parameter count where distance-only
models are hopeless." What it does not support is: "equivariance per se, as opposed to angular
resolution, is what buys the data efficiency."

### 9.4 The water comparison, restated

Collecting the qualifications from §7.4, because "NequIP beat DeepMD with 1000$\times$ less data" is
the single most-repeated sentence about this paper:

1. It is a **force** result. On energy, DeepMD is better on all four systems, by up to a factor of
   $16$ (0.8 vs 12.7 meV/molecule on ice I$_h$ (d) at NequIP's best-force setting).
2. It is **cross-implementation and cross-recipe**. NequIP's numbers were produced here; DeepMD's
   were quoted from a different paper, code, loss and optimiser. No variable was controlled except
   the reference dataset.
3. The DeepMD baseline is the **non-smooth published version**, which the paper explicitly flags, and
   whose smooth successor (DeepPot-SE) did not report water/ice numbers — so the honest reading is
   "better than the only DeepMD numbers available for these systems," not "better than DeepMD."
4. "133 structures" is 133 *frames*, i.e. $\sim\!7.7\times10^4$ supervised force components. The
   right claim is a thousand-fold reduction in DFT calculations, which is the expensive resource, but
   the phrase "1000$\times$ less data" invites a reading in which the fit sees a thousand times fewer
   constraints, and it does not.

### 9.5 What survives

A fair summary of what NequIP established, with the caveats folded in:

- A **working, reproducible, open** E(3)-equivariant message-passing potential with conservative
  forces, for molecules *and* periodic materials, validated not only on MAEs but on physical
  observables — radial and angular distribution functions of a quenched amorphous phase, and a Li$^+$
  diffusivity within $8.8\%$ of AIMD from 2,500 structures.
- A large, real, repeatedly reproduced **data-efficiency effect** for $\ell \geq 1$ tensor features:
  geometric-mean factor $5.6$ in force MAE from $\ell=0$ to $\ell=1$ across ten rMD17 molecules, and
  BPNN-competitive water forces at a fifth of the training frames.
- **Saturation above $\ell = 1$** on these benchmarks: geometric-mean $1.45\times$ from $\ell=1$ to
  $\ell=2$, and nothing beyond. Later work (MACE, eSEN, the Equiformer line) revisits this, and the
  repository's own experiment shows why the saturation point depends on the target.
- A design vocabulary — irreps-typed features, CG convolution, gates, parity bookkeeping, atomic
  energies, gradient forces — that became the default and is still the majority design at the top of
  Matbench Discovery, though by 2026 no longer the only one.

What it did not establish is that equivariance, isolated from angular resolution, is the cause. The
experiment that would settle it — an invariant-but-angular baseline at matched capacity, inside
NequIP's own codebase and training recipe — is not in the paper, and the paper's own tables show an
invariant directional model (GemNet, 9.5 meV/Å) outperforming NequIP $\ell=1$ (12.9). That is the
honest state of the evidence, and it is the right frame for reading what happened to the leaderboard
afterwards, where the [convex-hull chapter](../tutorial/index.html#hull) and the
[locality chapter](../tutorial/index.html#locality) pick up the story.

---

## 10. Sources

**Primary.** Batzner *et al.*, arXiv:2101.03164 (v1 8 Jan 2021, v3 16 Dec 2021); Nature Communications
**13**, 2453 (2022), doi:10.1038/s41467-022-29939-5.

**Directly inherited.** Thomas *et al.*, Tensor Field Networks, arXiv:1802.08219 (2018). Weiler,
Geiger, Welling, Boomsma, Cohen, 3D Steerable CNNs, arXiv:1807.02547, NeurIPS 2018. Kondor and
Trivedi, arXiv:1802.03690, ICML 2018. Geiger and Smidt *et al.*, e3nn, arXiv:2207.09453 (2022).
Klicpera (Gasteiger) *et al.*, DimeNet, arXiv:2003.03123, ICLR 2020 (Bessel basis, polynomial
envelope). Schütt *et al.*, SchNet, arXiv:1706.08566, NeurIPS 2017; J. Chem. Phys. **148**, 241722
(2018).

**Priority context.** Behler and Parrinello, Phys. Rev. Lett. **98**, 146401 (2007),
doi:10.1103/PhysRevLett.98.146401. Shapeev, MTP, arXiv:1512.06054; Multiscale Model. Simul. **14**,
1153 (2016), doi:10.1137/15M1054183. Drautz, ACE, Phys. Rev. B **99**, 014104 (2019),
doi:10.1103/PhysRevB.99.014104. Anderson, Hy, Kondor, Cormorant, arXiv:1906.04015, NeurIPS 2019.
Schütt, Unke, Gastegger, PaiNN, arXiv:2102.03150, ICML 2021. Miller, Geiger, Smidt, Noé,
arXiv:2008.08461 (2020).

**Benchmarks and baselines.** Christensen and von Lilienfeld, rMD17, arXiv:2007.09593; Mach. Learn.:
Sci. Technol. **1**, 045018 (2020), doi:10.1088/2632-2153/abba6f. Klicpera (Gasteiger) *et al.*,
GemNet, arXiv:2106.08903, NeurIPS 2021. Zhang, Han, Wang, Car, E, DeePMD, Phys. Rev. Lett. **120**,
143001 (2018), doi:10.1103/PhysRevLett.120.143001. Zhang *et al.*, DeepPot-SE, arXiv:1805.09003,
NeurIPS 2018. Kovács *et al.*, "Linear Atomic Cluster Expansion Force Fields for Organic Molecules:
Beyond RMSE", J. Chem. Theory Comput. (2021), doi:10.1021/acs.jctc.1c00647 — the source of the ACE
column in the rMD17 table.

**This repository.** `results/descriptor_blindness.json`, `results/lmax_learning_curves.json`,
`results/conservative_vs_direct.json`, `results/augmentation_vs_equivariance.json`, each produced by
the correspondingly named script in `python/experiments/`. All numbers attributed to the repository
above are read from those files.
