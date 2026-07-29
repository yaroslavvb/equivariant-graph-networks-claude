# equivariant-graph-networks

The genealogy of one idea — **make every hidden feature of a neural network transform correctly
under rotation** — from Behler–Parrinello (2007) through
[NequIP](https://www.nature.com/articles/s41467-022-29939-5) (Batzner et al., *Nature
Communications* **13**, 2453, 2022) to the models at the top of
[Matbench Discovery](https://matbench-discovery.materialsproject.org/) today.

Written as a report suite plus a **clickable tutorial** in which every displayed number is either
computed live in the browser or read from JSON written by a Python script in this repository. The
browser [re-derives and checks its own representation theory](tutorial/selftest.html) against values
exported from scipy; no table of Clebsch–Gordan coefficients is shipped anywhere.

**Why this repo exists.** A reading group met on 23 July 2026, read the leaderboard out loud, and
moved on from the sentence *"it's called Graph Attention Transformer that is now the leading one …
which incorporates rotational symmetry."* Two hours of crystallographic group theory happened in the
same room and the connection to network architecture was never made. This is that connection, worked
out — see [the meeting notes](reports/04-meeting-notes.md).

---

## Start here

- **[The clickable tutorial](https://yaroslavvb.github.io/equivariant-graph-networks-claude/tutorial/)** —
  ten chapters, arrow keys or click to move. Rotate a molecule and watch the equivariance residual;
  derive the Clebsch–Gordan couplings in your browser; break a network by choosing the obvious
  nonlinearity; drag noise onto a convex hull and watch F1 collapse while MAE barely moves.
- **[The genealogy of an idea](reports/02-genealogy.md)** — the centrepiece. Five braided strands
  and where they land.
- **[Reading the leaderboard](reports/03-leaderboard.md)** — what Matbench Discovery measures, and
  why ranking by F1 and by CPS produces different winners.

## The headline finding

The obvious story — *equivariance won, and everything at the top of the leaderboard is a NequIP
descendant* — is **false**, and four claims were adversarially fact-checked against primary sources
before this was written:

| Claim | Verdict |
|---|---|
| NequIP was first to combine local-energy message passing, $\ell\ge1$ tensor features, and gradient forces | true, with real priority caveats (Cormorant, TFN-for-forces, MTP, ACE, near-concurrent PaiNN) |
| Every top-10 Matbench model descends from Tensor Field Networks / e3nn | **false** — about half does |
| The 2024–26 jump was driven mainly by training-data scale | **false** — architecture and recipe account for roughly 75% of the CPS gain |
| $\kappa_{\mathrm{SRME}}$ failure is caused by direct (non-conservative) forces | true in association, but neither necessary nor sufficient |

The top tier is **architecturally plural**. GRACE is Atomic-Cluster-Expansion-derived; TACE and the
current CPS leader TECE-OAM-RRA-1.0 use irreducible *Cartesian* tensors and explicitly avoid
Clebsch–Gordan spherical products; **PET-OAM-XL (rank 5) is rotationally unconstrained** and learns
symmetry from augmentation; MatRIS is explicitly invariant. What built-in equivariance bought was
data efficiency and a strong, exact prior. At 2026 data scales that prior is less decisive, and
other things — smooth energy surfaces, accurate second derivatives, corpus diversity — now separate
the leaders.

---

## The toy experiments

Six simulation experiments, each a deterministic Python script that ends by printing its own
pass/fail ledger and writing the JSON the tutorial reads.

| Experiment | What it shows |
|---|---|
| [`descriptor_blindness`](python/experiments/descriptor_blindness.py) | Two atomic environments that **every** three-body descriptor is provably blind to — built from the all-interval tetrachords $\{0,1,4,6\}$ and $\{0,1,3,7\}$ on $\mathbb{Z}_{12}$. Minimum RMSD over all permutations, rotations and the reflection is exactly $(\sqrt3-1)/2$. The invariant power spectrum cannot separate them at *any* degree up to 6; the equivariant features separate them at $\ell=2$, with a certified lower bound. |
| [`lmax_learning_curves`](python/experiments/lmax_learning_curves.py) | The $\ell_{\max}$ ablation in miniature, with a feature-count-matched fairness control — and an honest negative: the toy does **not** reproduce the paper's qualitative pattern, because the Stillinger–Weber target is exactly complete at $\ell=2$. |
| [`conservative_vs_direct`](python/experiments/conservative_vs_direct.py) | Two surrogates with matched one-step force error; one is a gradient field and one is not. Curl, closed-loop work, and energy drift under velocity-Verlet. |
| [`convex_hull_decision`](python/experiments/convex_hull_decision.py) | Why stability is a decision problem. MAE degrades linearly in noise while F1 collapses; two explicit predictors where one has **lower MAE and lower F1** than the other. |
| [`augmentation_vs_equivariance`](python/experiments/augmentation_vs_equivariance.py) | Built-in symmetry against learned symmetry, including the off-distribution probe: train with rotations about one axis only, test on the whole group. |
| [`receptive_field`](python/experiments/receptive_field.py) | Message passing grows the receptive field linearly in depth and cubically in atom count — the cost Allegro's strict locality removes — plus exponential versus power-law truncation error, the quantitative form of "does locality suffice?" |

Plus [`python/e3.py`](python/e3.py), the representation-theory core, which builds real spherical
harmonics, Wigner $D$ matrices and Clebsch–Gordan couplings from scratch and verifies each against
the property that defines it — recovering the dot product and the cross product as the only
equivariant bilinear maps $1\otimes1\to0$ and $1\otimes1\to1$.

---

## Reproducing

Python is managed by [uv](https://docs.astral.sh/uv/) (3.12; numpy, scipy, matplotlib):

```bash
uv run python python/e3.py
```

```bash
for f in python/experiments/*.py; do uv run python "$f"; done
```

Every run is deterministic (seeds fixed) and reproduces the committed `results/*.json`.

To read the tutorial locally it must be served over HTTP, because the chapters `fetch` their data:

```bash
uv run python -m http.server 8731
```

then open `http://localhost:8731/tutorial/`.

---

## Reports

1. [NequIP, read closely](reports/01-nequip-anatomy.md) — the architecture in full, every
   experimental number, and a critical section on the confounds in its headline ablation.
2. [The genealogy of an idea](reports/02-genealogy.md) — the descriptor strand, the invariant-graph
   strand, the equivariance strand, the confluence, and the pluralisation era.
3. [Reading the leaderboard](reports/03-leaderboard.md) — WBM, MPtrj and OMat24, $E_{\mathrm{hull}}$,
   F1, DAF, $\kappa_{\mathrm{SRME}}$, CPS, and the current standings by two different orderings.
4. [The meeting that prompted this](reports/04-meeting-notes.md) — the technical extract, with the
   three critiques the group arrived at without the vocabulary for them.

## Layout

- `python/` — `e3.py` (representation theory) and `experiments/` (six toy simulations + the scipy
  reference exporter)
- `results/` — JSON written by those scripts; the tutorial reads these
- `tutorial/` — the clickable tutorial: `index.html`, `js/chapters/*.js`, and
  [`selftest.html`](tutorial/selftest.html)
- `reports/` — the written report suite

---

## A companion implementation

A separate, independently built treatment of the same brief lives at
[yaroslavvb/equivariant-graph-networks](https://github.com/yaroslavvb/equivariant-graph-networks)
("Geometry, Learned", a Next.js single-page field guide, live at
<https://yaroslavvb.github.io/equivariant-graph-networks/>). It was not modified in building this
one. This repository takes the static, verification-first approach of
[poisson-solvers](https://github.com/yaroslavvb/poisson-solvers) instead: no build step, plain ES
modules, and a Python check ledger behind every figure.
