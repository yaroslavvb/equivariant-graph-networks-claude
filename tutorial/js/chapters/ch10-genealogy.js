import { h, segmented, checkLine, Plot, PALETTE } from '../ui.js';

// ---------------------------------------------------------------------------
// Data. Everything displayed in this chapter is derived from these tables at
// render time — the lineage tallies, the percentages, the ranges and the check
// ledger are all computed in the page, never typed into the prose.
//
// Provenance: the leaderboard rows are the Matbench Discovery model YAMLs
// (github.com/janosh/matbench-discovery/models/*/*.yml) as of 2026-07-29, on
// the "unique prototypes" discovery subset and the kappa_103 phonon subset.
// The genealogy entries are the primary papers cited in each record.
// ---------------------------------------------------------------------------

const NS = 'http://www.w3.org/2000/svg';
const el = (t, a = {}) => {
  const e = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) if (v != null) e.setAttribute(k, String(v));
  return e;
};

const LIN = {
  desc: { name: 'the descriptor era', short: 'descriptors', color: PALETTE[5] },
  inv:  { name: 'invariant graph networks', short: 'invariant GNNs', color: PALETTE[3] },
  equi: { name: 'equivariance / TFN / e3nn', short: 'e3nn / TFN', color: PALETTE[0] },
  ace:  { name: 'the cluster expansion', short: 'ACE', color: PALETTE[2] },
  unc:  { name: 'deliberately unconstrained', short: 'unconstrained', color: PALETTE[4] },
  data: { name: 'data and benchmark', short: 'data & scale', color: PALETTE[1] },
};
const LANES = ['desc', 'inv', 'equi', 'ace', 'unc', 'data'];
const YEARS = [2007, 2010, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];

const NODES = [
  // ---- the descriptor era -------------------------------------------------
  { id: 'bp', lane: 'desc', year: 2007, slot: 0, l1: 'Behler–', l2: 'Parrinello',
    name: 'Behler–Parrinello high-dimensional neural network potential',
    group: 'Jörg Behler and Michele Parrinello, ETH Zürich / USI Lugano',
    cite: 'Phys. Rev. Lett. 98, 146401 (2007); catalogue J. Chem. Phys. 134, 074106 (2011)',
    core: 'The founding move of the whole field: write the total energy as a sum of atomic energies, give every atom of an element an identical small network, and replace Cartesian coordinates with a fixed-length vector of hand-designed, rotation-invariant atom-centred symmetry functions — Gaussians over distances and (1 + λ cos θ)^ζ over triplets, both damped by a cosine cutoff.',
    inh: 'The atomic-energy decomposition from empirical many-body potentials such as Tersoff and EAM; low-dimensional neural-network fits of potential energy surfaces from Lorenz–Groß–Scheffler and Blank et al.',
    add: 'Size transferability and permutation invariance for free, a coordination-number-independent descriptor, analytic forces, and an active-learning loop. Every model in this chapter still uses its first equation. Its weakness is that geometry is compressed into invariant scalars at the very first step and never recovered.' },
  { id: 'gap', lane: 'desc', year: 2010, slot: 0, l1: 'GAP / SOAP',
    name: 'Gaussian Approximation Potentials and the SOAP kernel',
    group: 'Albert Bartók, Risi Kondor, Mike Payne, Gábor Csányi — Cambridge and Caltech',
    cite: 'Phys. Rev. Lett. 104, 136403 (2010); Phys. Rev. B 87, 184115 (2013)',
    core: 'Replace hand-picked scalars with harmonic analysis of the neighbour density, and the neural network with Gaussian process regression. SOAP then inverts the logic: define the similarity between two environments first, by integrating the overlap of their smoothed densities over all relative rotations, and only afterwards recognise it as a power-spectrum or bispectrum inner product.',
    inh: 'Behler–Parrinello’s atomic-energy decomposition, cited as reference [9] of the GAP letter; Steinhardt bond-order parameters, which the SOAP paper proves are a sub-slice of the bispectrum.',
    add: 'The vocabulary the field still uses — the power spectrum is exactly three-body, the bispectrum exactly four-body — plus principled uncertainty and training on forces as linear functionals of the process. It also contains, in its own appendix, the first admission that the power spectrum is not complete.' },
  { id: 'mtp', lane: 'desc', year: 2016, slot: 0, l1: 'Moment tensor', l2: 'potentials',
    name: 'Moment Tensor Potentials',
    group: 'Alexander Shapeev, Skoltech',
    cite: 'arXiv:1512.06054; Multiscale Model. Simul. 14, 1153 (2016)',
    core: 'Do GAP’s job in Cartesian tensors rather than spherical harmonics, and prove theorems about it. Encode the neighbourhood in moment tensors — sums of a radial function times the ν-fold outer product of the neighbour vector — and obtain invariants by fully contracting products of them. The same physics as Clebsch–Gordan coupling, done with Kronecker deltas instead of Wigner symbols.',
    inh: 'The atomic decomposition, taken as an axiom; the radial times angular structure of Behler–Parrinello symmetry functions, acknowledged explicitly; the completeness framing from GAP and SOAP.',
    add: 'The first machine-learned potential with both a spanning-set theorem and an exponential approximation rate. And the Cartesian-tensor route itself — which looked like a historical footnote for a decade, and then reappeared in 2026 at the top of the leaderboard as TACE and TECE.' },
  { id: 'pozd', lane: 'desc', year: 2020, slot: 0, l1: 'Degeneracy', l2: 'proof',
    name: 'On the completeness of atomic structure representations',
    group: 'Pozdnyakov, Willatt, Ceriotti (EPFL); Bartók, Ortner (Warwick); Csányi (Cambridge)',
    cite: 'arXiv:2001.11696; Phys. Rev. Lett. 125, 166001 (2020)',
    core: 'Not a potential — the paper that demolished the premise the descriptor era rested on. It constructs explicit pairs of genuinely distinct atomic environments with identical three-body descriptors, and further pairs that are degenerate even at four-body order. Two of the SOAP authors are co-authors: this is the field auditing its own foundational claim.',
    inh: 'The faithfulness methodology from the SOAP paper, which had already exhibited a power-spectrum counterexample and run numerical reconstruction experiments on silicon clusters.',
    add: 'It converted a suspected weakness into a proved one. Any model of the form E = f(invariant descriptor) must assign degenerate configurations identical energies no matter how expressive f is. Chapter 3 rebuilds one of these counterexamples from scratch and measures exactly where it bites.' },

  // ---- invariant graph networks ------------------------------------------
  { id: 'schnet', lane: 'inv', year: 2017, slot: 0, l1: 'SchNet',
    name: 'SchNet — continuous-filter convolutional network',
    group: 'Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller — TU Berlin, FHI, Luxembourg',
    cite: 'NeurIPS 2017, arXiv:1706.08566; J. Chem. Phys. 148, 241722 (2018)',
    core: 'Learn the descriptor instead of designing it. Atom features are updated by continuous-filter convolutions whose filters are neural functions of the interatomic distance alone, so the network is invariant by construction and smooth enough to differentiate for forces.',
    inh: 'Gilmer et al.’s message-passing framework, which supplied the message, update and readout vocabulary, and Behler–Parrinello’s atomic-energy sum.',
    add: 'Learned continuous filters and a force-trainable energy surface. Its blind spot is the one chapter 3 is about: with distances only and no angles, aspirin forces come in at 58.5 meV/Å where an invariant model that does see angles reaches 21.6.' },
  { id: 'deeppot', lane: 'inv', year: 2018, slot: 0, l1: 'DeepPot-SE',
    name: 'Deep Potential and DeepPot-SE',
    group: 'Linfeng Zhang, Jiequn Han, Han Wang, Roberto Car, Weinan E — Princeton and IAPCM',
    cite: 'Phys. Rev. Lett. 120, 143001 (2018), arXiv:1707.09571; DeepPot-SE arXiv:1805.09003',
    core: 'Build a smooth, symmetry-preserving local embedding by learning it: the environment matrix of each atom is contracted with a learned embedding network so that the result is invariant, then handed to a fitting network. No hand-chosen symmetry functions and no spherical harmonics anywhere.',
    inh: 'Behler–Parrinello’s atomic-energy decomposition and locality assumption; the general deep-learning practice of learning representations end to end.',
    add: 'The descriptor line that runs to DPA-1, DPA-2, DPA-3 and DPA-4 — a family that stayed outside the equivariance programme for the better part of a decade and then, in 2026, produced the single best model trained on Materials Project data alone.' },
  { id: 'dimenet', lane: 'inv', year: 2020, slot: 0, l1: 'DimeNet',
    name: 'DimeNet — directional message passing',
    group: 'Johannes Gasteiger (Klicpera), Janek Groß, Stephan Günnemann — TU Munich',
    cite: 'arXiv:2003.03123, ICLR 2020; successor GemNet arXiv:2106.08903, NeurIPS 2021',
    core: 'Pass messages between directed edges rather than between atoms, so the angle between two bonds enters the update explicitly. Angular information is expanded in spherical Bessel functions and spherical harmonics — genuine directional resolution, still packaged as invariant scalars.',
    inh: 'The message-passing framework from Gilmer et al. and SchNet; the angular symmetry functions of Behler–Parrinello, generalised to a complete orthogonal basis.',
    add: 'The demonstration that most of SchNet’s gap to the state of the art was missing angular information rather than missing equivariance. Its successor GemNet added quadruplets and reached 9.5 meV/Å on revised MD17 aspirin — better than NequIP at ℓ = 1. That comparison is the caveat chapter 6 has to live with.' },
  { id: 'm3gnet', lane: 'inv', year: 2022, slot: 0, l1: 'M3GNet',
    name: 'M3GNet',
    group: 'Chi Chen and Shyue Ping Ong — Materials Virtual Lab, UC San Diego',
    cite: 'Nat. Comput. Sci. 2, 718 (2022); arXiv:2202.02450',
    core: 'The first single graph network fitted to essentially the whole periodic table. An explicit three-body term updates bond features on top of a MEGNet-style graph; everything the network carries is a rotational scalar, so the model is invariant by construction, with forces and stresses from autodifferentiation.',
    inh: 'MEGNet’s atom, bond and global graph; three-body descriptors from Behler–Parrinello and DimeNet; the Materials Project archive as the label source.',
    add: 'Universality — one parameter set for 89 elements — and, at least as consequential, the idea of mining the ionic steps of relaxation trajectories that had until then been discarded as computational waste. That idea is what makes MPtrj, and therefore this leaderboard, possible at all.' },
  { id: 'chgnet', lane: 'inv', year: 2023, slot: 0, l1: 'CHGNet',
    name: 'CHGNet, and the MPtrj dataset',
    group: 'Deng, Zhong, Jun, Riebesell, Han, Bartel, Ceder — UC Berkeley and LBNL',
    cite: 'Nat. Mach. Intell. 5, 1031 (2023); arXiv:2302.14231',
    core: 'M3GNet’s template plus one substantive addition: the DFT on-site magnetic moment as an extra regression target, which forces the latent atom features to carry local oxidation state and enables charge-informed molecular dynamics.',
    inh: 'The M3GNet architecture almost wholesale — invariant convolution, angle features, autodifferentiated forces, universality — and the Materials Project labels and settings.',
    add: 'MPtrj: 1,580,395 configurations over roughly 146,000 materials, cleaned and released. The architecture was overtaken within a year. The dataset became the canonical compliant training set for the entire benchmark, and is still the frozen-data control track this chapter leans on.' },
  { id: 'matris', lane: 'inv', year: 2025, slot: 0, l1: 'MatRIS', board: true, rank: 10,
    name: 'MatRIS-10M-OAM',
    group: 'Zhou, Hu, Zhang, Wang, Tan, Jia — Institute of Computing Technology, Chinese Academy of Sciences',
    cite: 'arXiv:2603.02002',
    core: 'An explicitly invariant machine-learned potential that recovers equivariant-level accuracy through separable, linear-complexity attention over three-body interactions. Its paper positions it against equivariant models rather than as their descendant.',
    inh: 'The invariant three-body lineage — Behler–Parrinello, SchNet, DimeNet and GemNet, M3GNet and CHGNet. Neither Tensor Field Networks, nor Equiformer, nor ACE.',
    add: 'The highest-ranked invariant architecture on the board and the cheapest model in the top ten, at 10.4M parameters. Its thermal conductivity lags the equivariant leaders — κ_SRME 0.218 against 0.093 for the leader — which is exactly where an invariant model would be expected to pay.' },
  { id: 'dpa4', lane: 'inv', year: 2026, slot: 0, l1: 'DPA-4.0.1', l2: 'Pro-MPtrj', board: true, rank: 16,
    name: 'DPA-4.0.1-Pro-MPtrj',
    group: 'Li, Li, Peng, Xue, Zhang, Zhang, Wang — AI for Science Institute Beijing, Peking University, DP Technology',
    cite: 'arXiv:2606.02419',
    core: 'The most striking single data point on the leaderboard. Trained on MPtrj alone — 1.58M structures, no OMat24, no Alexandria — it reaches CPS 0.840, matching models trained on roughly seventy times more data. Its core is an edge-conditioned, multi-focus, envelope-gated SO(2)-equivariant convolution with Lebedev-grid projection.',
    inh: 'Two lineages fused: the DeepPot-SE and DPA descriptor line, and the SO(2)-convolution trick that originates in eSCN. Its paper benchmarks itself against eSEN rather than against NequIP.',
    add: 'The demonstration that architecture and training recipe can erase a seventyfold data deficit. It is the load-bearing counterexample to the story that the 2024–26 jump was mainly about training-set size.' },

  // ---- equivariance / TFN / e3nn ------------------------------------------
  { id: 'steer', lane: 'equi', year: 2016, slot: 0, l1: 'Steerable', l2: 'CNNs',
    name: 'G-CNNs, Steerable CNNs, and the necessity theorem',
    group: 'Taco Cohen and Max Welling (Amsterdam); Risi Kondor and Shubhendu Trivedi (Chicago)',
    cite: 'arXiv:1602.07576 (ICML 2016); arXiv:1612.08498 (ICLR 2017); arXiv:1802.03690 (ICML 2018)',
    core: 'Equivariance becomes an operational design constraint rather than a property you hope for. G-CNNs make feature maps functions on a group; Steerable CNNs replace that with a fibre carrying an arbitrary representation and push the whole burden onto a linear constraint on the filter bank. Kondor and Trivedi then prove the converse: under a compact group, convolutional structure is necessary and not merely sufficient.',
    inh: 'Ordinary CNN translation equivariance; classical harmonic analysis on groups.',
    add: 'The type system — features are indexed by which irreducible representations occur and with what multiplicity, and the number of weights in an equivariant layer is fixed by the types alone. This is the direct ancestor of e3nn’s Irreps strings. After the necessity theorem, architecture design stops being a search and becomes a solve.' },
  { id: 'tfn', lane: 'equi', year: 2018, slot: 0, l1: 'Tensor field', l2: 'networks',
    name: 'Tensor Field Networks',
    group: 'Thomas, Smidt, Kearnes, Yang, Li, Kohlhoff, Riley — Stanford and Google Accelerated Science',
    cite: 'arXiv:1802.08219',
    core: 'The direct architectural parent of NequIP. Features on points in three dimensions are decomposed into SO(3) irreducible representations; filters are constrained to a learnable radial function times a fixed spherical harmonic; input and filter are combined with Clebsch–Gordan coefficients so the output is again a direct sum of irreps.',
    inh: 'Irrep-typed features and the constrained-filter posture from Steerable CNNs; spherical harmonics, Wigner matrices and Clebsch–Gordan coefficients from angular-momentum theory; the continuous-filter point convolution and the self-interaction layer from SchNet.',
    add: 'The exact factorisation NequIP uses verbatim. Weiler et al.’s 3D Steerable CNNs, five months later, proved that this factorisation is not a modelling choice but the general solution of the equivariance constraint, and contributed the gated nonlinearity NequIP also uses. Chapter 4 rebuilds both pieces in the browser.' },
  { id: 'cormorant', lane: 'equi', year: 2019, slot: 1, l1: 'Cormorant',
    name: 'Cormorant — covariant molecular neural networks',
    group: 'Brandon Anderson, Truong-Son Hy, Risi Kondor — University of Chicago',
    cite: 'arXiv:1906.04015; NeurIPS 2019',
    core: 'Kondor’s Fourier-space programme carried into a practical molecular architecture. Activations are SO(3)-covariant vectors of a declared type, the network keeps both vertex and edge activations at every layer, and the only nonlinearity is the channel-wise Clebsch–Gordan product.',
    inh: 'N-body Networks and Clebsch–Gordan Nets, both Kondor 2018: covariant activation types, and the theorem that a learnable weight in an equivariant network can only mix channels within a fixed ℓ.',
    add: 'The first strong empirical showing for irrep-based networks on real quantum-chemistry benchmarks, including potential energy surfaces. It matters here for one specific reason: Cormorant already had equivariant atom-wise message passing with ℓ ≥ 1 features in 2019. What it lacked was gradient-derived forces — which is why NequIP’s priority claim needs the word “conservative” in it.' },
  { id: 'e3nn', lane: 'equi', year: 2020, slot: 0, l1: 'e3nn',
    name: 'e3nn — Euclidean neural networks',
    group: 'Mario Geiger, Tess Smidt, and contributors (Miller, Boomsma, Lapchevskyi, Weiler, Tyszkiewicz, Frellsen)',
    cite: 'arXiv:2207.09453; library from March 2020, Zenodo 10.5281/zenodo.3724963',
    core: 'Packages the entire lineage as a differentiable framework in which O(3) irreps are a type system and every layer is a typed, provably equivariant operation. Two primitives suffice: spherical harmonics, and a fully general tensor product that can represent any bilinear equivariant map between irreps.',
    inh: 'The type system from Steerable CNNs, the tensor layout and the radial times spherical-harmonic filter from TFN, the kernel basis and gated nonlinearity from 3D Steerable CNNs, the Clebsch–Gordan product from Kondor.',
    add: 'Systematic parity, which upgrades SE(3)-equivariant architectures to E(3) for free; the Irreps notation that became the field’s lingua franca; and correct, fast real spherical harmonics and Wigner matrices. Practically, e3nn is what made NequIP a few hundred lines of model rather than a representation-theory implementation project.' },
  { id: 'nequip', lane: 'equi', year: 2021, slot: 0, l1: 'NequIP',
    name: 'NequIP',
    group: 'Batzner, Musaelian, Sun, Geiger, Mailoa, Kornbluth, Molinari, Smidt, Kozinsky — Harvard, MIT, Bosch',
    cite: 'arXiv:2101.03164; Nat. Commun. 13, 2453 (2022)',
    core: 'E(3)-equivariant convolutions over geometric tensors, applied to interatomic potentials, at a moment when nearly every competitive potential used invariant features only. Node features are irreps rather than scalars; the convolution is a Clebsch–Gordan product between neighbour features and spherical harmonics of the interatomic unit vector; energy is a sum of site energies and forces are its analytic gradient.',
    inh: 'TFN for the filter form and the convolution, 3D Steerable CNNs for the gate, e3nn for the primitives, SchNet for the atomic-energy-sum and analytic-force template, DimeNet for the Bessel radial basis. NequIP invents none of the equivariance machinery, and says so.',
    add: 'Systematic parity, a learnable Bessel radial network, and the empirical result that made the field pivot: an ℓ_max ablation showing large monotone gains, and a data-efficiency claim of up to three orders of magnitude. The honest caveat is that its ℓ = 0 baseline has no angular information at all, so the ablation conflates equivariance with directional resolution — chapter 6 works through exactly this.' },
  { id: 'allegro', lane: 'equi', year: 2022, slot: 0, l1: 'Allegro',
    name: 'Allegro',
    group: 'Musaelian, Batzner, Johansson, Sun, Owen, Kornbluth, Kozinsky — Harvard',
    cite: 'arXiv:2204.05249; Nat. Commun. 14, 579 (2023)',
    core: 'Keep NequIP’s equivariant representation and delete message passing entirely. Energy is decomposed over ordered pairs, and every layer for an edge reads only atoms inside one cutoff sphere, so depth no longer grows the receptive field. Many-body correlation is recovered by iterating tensor products within that single environment.',
    inh: 'From NequIP: e3nn irreps, Clebsch–Gordan products, spherical harmonics of interatomic vectors, the learnable Bessel radial network, the site-energy and analytic-force setup, and the codebase itself.',
    add: 'Strict locality, and with it a parallel cost independent of depth — a hundred million atoms of equivariant molecular dynamics. A single Allegro layer beat every deep message-passing network on QM9, which is direct evidence that depth in NequIP was buying expressivity rather than range. Chapter 8 measures that distinction.' },
  { id: 'equiformer', lane: 'equi', year: 2022, slot: 1, l1: 'Equiformer',
    name: 'Equiformer',
    group: 'Yi-Lun Liao and Tess Smidt — MIT, Atomic Architects',
    cite: 'arXiv:2206.11990; ICLR 2023',
    core: 'Take the transformer block and replace every operation with its equivariant counterpart acting on irreps features, then introduce equivariant graph attention: attention weights computed from the scalar part of the message, and messages that are themselves nonlinear. This is the architecture family the reading group’s sentence is about.',
    inh: 'The irreps formalism and Clebsch–Gordan tensor product from TFN, e3nn and the SE(3)-Transformer; the block layout — pre-norm, multi-head attention, feed-forward, residual — from the vanilla transformer; the gate from 3D Steerable CNNs.',
    add: 'The demonstration that the transformer template transfers to atomistic graphs once its operations are made equivariant, plus equivariant layer normalisation. What it did not solve is cost: the tensor product still scales as ℓ⁶, which caps it at ℓ_max of 2 or 3.' },
  { id: 'escn', lane: 'equi', year: 2023, slot: 1, l1: 'eSCN',
    name: 'eSCN — reducing SO(3) convolutions to SO(2)',
    group: 'Saro Passaro and C. Lawrence Zitnick — FAIR at Meta',
    cite: 'arXiv:2302.03655; ICML 2023',
    core: 'You are free to choose the frame in which you evaluate the tensor product. Rotate the node features so the edge vector aligns with the primary axis and the edge’s spherical harmonics collapse to their m = 0 components; the Clebsch–Gordan matrix becomes extremely sparse and what survives is a set of independent SO(2) operations. Cost falls from ℓ⁶ to ℓ³.',
    inh: 'The spherical-channel architecture, the message-aggregation block and — importantly for what happens next — the direct force head, from SCN (Zitnick et al. 2022).',
    add: 'The single most consequential efficiency result in this branch, which made ℓ = 6 trainable at scale. It is also the piece of machinery that has travelled furthest outside its own lineage: the SO(2) reduction now appears inside DPA-4 and inside the ShanghaiTech Cartesian-tensor models at the top of the board.' },
  { id: 'eqv2', lane: 'equi', year: 2023, slot: 2, l1: 'Equiformer', l2: 'V2',
    name: 'EquiformerV2',
    group: 'Yi-Lun Liao, Brandon Wood, Abhishek Das, Tess Smidt — MIT and FAIR at Meta',
    cite: 'arXiv:2306.12059; ICLR 2024',
    core: 'Swap Equiformer’s ℓ⁶ SO(3) convolutions for eSCN’s ℓ³ SO(2) convolutions and ask what breaks as ℓ_max rises from 2–3 up to 6–8. Higher degrees alone destabilise training, so add attention re-normalisation, a separable spherical activation and separable layer normalisation.',
    inh: 'The whole Equiformer block layout; the SO(2) convolution and the spherical-grid activation from eSCN; and, from eSCN’s training culture, the direct-force objective.',
    add: 'The paper that made ℓ_max = 6 standard, and the high-water mark of the direct-force era. What it did not add is any notion of energy conservation or smoothness — a decision whose bill arrives at the next node along.' },
  { id: 'sevennet', lane: 'equi', year: 2024, slot: 0, l1: 'SevenNet',
    name: 'SevenNet',
    group: 'Yutack Park, Jaesun Kim, Seungwoo Hwang, Seungwu Han — Seoul National University',
    cite: 'arXiv:2402.03789; J. Chem. Theory Comput. 20, 4857 (2024). Omni: arXiv:2510.11241',
    core: 'Take NequIP’s architecture essentially unmodified and solve the parallel-molecular-dynamics problem the opposite way from Allegro: keep message passing, but make it communicate. Split the network at layer boundaries and exchange ghost-atom features between layers, so the ghost region stays one cutoff deep instead of T.',
    inh: 'The NequIP architecture itself — the paper says so. Equivariant convolution by Clebsch–Gordan tensor products, e3nn node features, Bessel radial basis, gated nonlinearities, site energies. The contribution is systems, not representation theory.',
    add: 'Depth becomes cheap to parallelise, so SevenNet can afford to spend it: the current Omni-i12 checkpoint runs twelve message-passing layers. It is also the direct ancestor of Samsung’s EquFlash line, which is why a systems paper sits on the critical path to the second-place model on the board.' },
  { id: 'eqv2m', lane: 'equi', year: 2024, slot: 2, l1: 'eqV2 M', board: true, rank: 36,
    name: 'eqV2 M (EquiformerV2-Medium, OMat24 + sAlex + MPtrj)',
    group: 'Barroso-Luque, Shuaibi, Fu, Wood, Dzamba, Gao, Rizvi, Zitnick, Ulissi — FAIR at Meta',
    cite: 'arXiv:2410.12771 (the OMat24 paper); architecture arXiv:2306.12059',
    core: 'The model that defined the 2024–25 state of the art on discovery and, simultaneously, the cautionary tale of the benchmark. EquiformerV2 trained on OMat24 and fine-tuned on sAlex and MPtrj, with a direct force head rather than an energy gradient.',
    inh: 'The Equiformer and eSCN transformer branch entire, plus the OMat24 corpus the same paper introduced.',
    add: 'F1 of 0.917 — still ninth best on the board — alongside κ_SRME of 1.771 against a worst possible value of 2.0. Its failure is what the Combined Performance Score was invented to catch, and what motivated eSEN, Orb-v3’s conservative variant and EquiformerV3’s move to gradient forces. Chapter 7 measures the mechanism, and finds it is smoothness as much as conservativeness.' },
  { id: 'nequipoam', lane: 'equi', year: 2025, slot: 0, l1: 'NequIP-OAM', l2: 'XL', board: true, rank: 9,
    name: 'Nequip-OAM-XL',
    group: 'MIR Group, Harvard (Boris Kozinsky), with Cambridge and Mirian Technologies',
    cite: 'Framework arXiv:2504.16068; original NequIP arXiv:2101.03164',
    core: 'NequIP itself, scaled into a foundation model: 32.1M parameters, six layers, retrained on the OMat24 + sAlex + MPtrj mixture using the rewritten, multi-node, compiled NequIP framework.',
    inh: 'It is the root of this branch rather than a descendant of anything else on the board.',
    add: 'Scale and infrastructure rather than a new equivariance mechanism — distributed training, compiled inference, a fused tensor-product kernel worth up to eighteenfold in molecular dynamics. Its real result is the null one: the 2021 architecture, unchanged in principle, is still within 0.022 CPS of the 2026 state of the art.' },
  { id: 'esen', lane: 'equi', year: 2025, slot: 1, l1: 'eSEN', board: true, rank: 7,
    name: 'eSEN-30M-OAM',
    group: 'Xiang Fu, Brandon Wood, Barroso-Luque, Levine, Gao, Dzamba, Zitnick — FAIR at Meta',
    cite: 'arXiv:2502.12147; ICML 2025',
    core: 'The corrective paper of the branch. Its claim is that smoothness of the learned energy surface, not expressivity, determines whether a model can do phonons — and it proposes a concrete gate: run constant-energy molecular dynamics and measure the energy drift. It notes bluntly that direct-force models are not potentials but non-conservative force fields.',
    inh: 'The SO(2) convolution from eSCN; the block layout and normalisation from Equiformer; the envelope idea from DimeNet; the direct-pretrain-then-conservative-finetune recipe from Bigi et al.',
    add: 'A systematic ablation identifying grid discretisation, neighbour caps and missing envelopes as conservation-breakers, and a conservative head that drops κ_SRME from EquiformerV2’s 1.77 to 0.17. It is the proximate cause of the whole leaderboard swinging back to gradient forces during 2025.' },
  { id: 'ev3', lane: 'equi', year: 2026, slot: 1, l1: 'Equiformer', l2: 'V3+DeNS', board: true, rank: 3,
    name: 'EquiformerV3+DeNS-OAM',
    group: 'Liao, Hoffman, Shen, Duval, Norwood, Smidt — MIT Atomic Architects and Mirror Physics',
    cite: 'arXiv:2604.09130',
    core: 'The third generation of the equivariant graph-attention transformer, and the best classification F1 on the entire board at 0.931. Three advances over EquiformerV2 plus the switch to conservative energy-gradient forces and higher-order derivative support — at 30.3M parameters, roughly a seventh of the model that outranks it on the composite.',
    inh: 'Equiformer → eSCN → EquiformerV2, plus the DeNS denoising auxiliary objective from the same group, plus eSEN’s diagnosis that grid-based spherical activations break strict equivariance and energy conservation.',
    add: 'SwiGLU-S², an activation that raises effective body order, restores strict equivariance and permits a coarser spherical grid all at once; smooth-cutoff attention, which fixes a conservation leak specific to softmax denominators; and a 1.75-fold pure-implementation speedup. This is the model the reading group’s sentence describes — and it is not the one at the top of the composite table.' },
  { id: 'equflash', lane: 'equi', year: 2026, slot: 0, l1: 'EquFlashV2', board: true, rank: 2,
    name: 'EquFlashV2 (EquFlashV2-45M-OAM)',
    group: 'Vertical AI 2 and Materials AI Lab, Samsung Electronics',
    cite: 'FlashTP: Lee et al., ICML 2025 (PMLR 267); lineage via SevenNet-0, arXiv:2402.03789',
    core: 'An E(3)-equivariant force field that adopts the transformer block layout — pre-norm residual blocks, feed-forward sublayers — and deliberately uses no attention at all. Message passing is plain Clebsch–Gordan tensor-product convolution with a bilinear gate. The thesis is that if the tensor product is fast enough, attention is unnecessary.',
    inh: 'The NequIP branch: Tensor Field Networks → NequIP → SevenNet-0, on which its predecessor EquFlash is explicitly based. It borrows the transformer macro-architecture from the Equiformer line without adopting its attention.',
    add: 'FlashTP, a fused tensor-product kernel reported at up to 41.6 times faster than e3nn with six times lower memory, which is what let them scale far past SevenNet-0. Second by CPS at 44.9M parameters. The accuracy race has partly become a kernel-engineering race.' },

  // ---- the cluster expansion ---------------------------------------------
  { id: 'ace', lane: 'ace', year: 2019, slot: 0, l1: 'Atomic cluster', l2: 'expansion',
    name: 'ACE — the Atomic Cluster Expansion',
    group: 'Ralf Drautz, ICAMS, Ruhr-Universität Bochum',
    cite: 'Phys. Rev. B 99, 014104 (2019)',
    core: 'Stop proposing descriptors and derive one. Start from a complete orthonormal single-bond basis, build cluster basis functions as products over the bonds of a cluster, and observe that the coefficients are literal projections rather than an ansatz. Then the density trick: reorder the sums into an atomic base and the cost of an arbitrarily high body order collapses to linear in the neighbour count.',
    inh: 'The lattice cluster expansion generalised from discrete to continuous degrees of freedom; the Finnis–Sinclair embedding; the spherical-harmonic density expansion from GAP and SOAP; Behler–Parrinello’s atomic-energy sum.',
    add: 'A complete, orthogonal, hierarchical basis, and a unification map showing that Steinhardt parameters, Behler–Parrinello symmetry functions, SOAP kernels, SNAP and moment tensor potentials are all truncations or reparametrisations of it. After ACE, “invent a better invariant descriptor” stopped being an open question. This is the second root of the tree, and it is entirely independent of Tensor Field Networks.' },
  { id: 'mace', lane: 'ace', year: 2022, slot: 0, l1: 'MACE',
    name: 'MACE, and the Multi-ACE design space',
    group: 'Batatia, Kovács, Simm, Ortner, Csányi — Cambridge and UBC. Multi-ACE additionally with Batzner, Musaelian, Drautz, Kozinsky',
    cite: 'arXiv:2206.07697 (NeurIPS 2022); Multi-ACE arXiv:2205.06643, Nat. Mach. Intell. 7, 56 (2025)',
    core: 'Diagnose NequIP’s cost as a body-order problem rather than a locality problem. A NequIP message takes one neighbour at a time, so building n-body correlation requires stacking layers, which couples expressivity to receptive field. MACE imports ACE’s product basis into the message-passing layer, so messages are intrinsically four-body and two layers suffice.',
    inh: 'From NequIP: e3nn irreps, Clebsch–Gordan products, spherical harmonics, the Bessel radial network, site energies, and the benchmark suite. From ACE: the density trick and generalised symmetrisation. It is the child of both roots.',
    add: 'Higher body-order messages, and a halved receptive field with them. Multi-ACE, its companion, then proved the deeper point: ACE, NequIP, MACE, BOTNet, PaiNN, SchNet and Tensor Field Networks are all special cases of one framework, and linearised NequIP is a particular sparsification of a larger polynomial model. The two roads were one design space all along.' },
  { id: 'grace', lane: 'ace', year: 2024, slot: 0, l1: 'Graph ACE', l2: '(GRACE)',
    name: 'GRACE — graph atomic cluster expansion',
    group: 'Anton Bochkarev, Yury Lysogorskiy, Ralf Drautz — ICAMS, Ruhr-Universität Bochum',
    cite: 'Phys. Rev. X 14, 021036 (2024); arXiv:2311.16326',
    core: 'Extend ACE’s basis with graph basis functions so the expansion becomes semilocal — information propagates beyond the first coordination shell — while remaining a controlled expansion rather than an ad hoc network. Its sharpest result is structural: a tensor decomposition of the graph expansion collapses into an iterative procedure that is message passing.',
    inh: 'ACE for the complete product basis and the Clebsch–Gordan coupling machinery; the performant-ACE implementation for LAMMPS-grade speed.',
    add: 'The reframing that equivariant message-passing potentials fall out as a truncation of a known complete basis, rather than standing as independent architectures. The paper’s title says it plainly: semilocal interactions beyond equivariant message passing. Implemented in TensorFlow, not e3nn.' },
  { id: 'grace3l', lane: 'ace', year: 2026, slot: 0, l1: 'GRACE-3L', l2: 'OAM-L', board: true, rank: 4,
    name: 'GRACE-3L-OAM-L',
    group: 'Yury Lysogorskiy, Anton Bochkarev, Ralf Drautz — ICAMS, Ruhr-Universität Bochum',
    cite: 'npj Comput. Mater., DOI 10.1038/s41524-026-01979-1 (2026); arXiv:2508.17936',
    core: 'A third message-passing layer and a scale-up of graph ACE, trained on the OMat24 + sAlex + MPtrj mixture at 42.1M parameters. Fourth by CPS, best root-mean-square error and best coefficient of determination in the top ten, and tied for the best geometry optimisation on the whole board.',
    inh: 'The ACE branch, purely: Drautz 2019 → performant ACE → graph ACE → GRACE-2L → GRACE-3L. It does not descend from NequIP or Equiformer; its equivariance comes from ACE’s own spherical-harmonic product basis.',
    add: 'The best diatomic-curve behaviour among the top models, and a new accuracy-versus-efficiency Pareto front. It is the plainest counterexample to a tidy story in which everything at the top descends from Tensor Field Networks.' },
  { id: 'tece', lane: 'ace', year: 2026, slot: 1, l1: 'TACE / TECE', board: true, rank: 1,
    name: 'TECE-OAM-RRA-1.0, and the TACE family before it',
    group: 'Zemin Xu, Wenbo Xie, P. Hu — ShanghaiTech University and Nanjing University',
    cite: 'arXiv:2607.10664; predecessor TACE arXiv:2509.14961',
    core: 'Number one on the Combined Performance Score, with the best thermal conductivity and best geometry optimisation on the board while remaining fully conservative. Three ideas stacked on the Atomic Cluster Expansion: an edge cluster expansion forming many-body products on edge features, an improved atom-centred ACE module, and radial rotary attention.',
    inh: 'The ACE branch — Drautz 2019 and MACE’s equivariant product basis — plus, from its own predecessor TACE, irreducible Cartesian tensor representations, whose stated purpose is to avoid the axis dependence and Clebsch–Gordan overhead of spherical-tensor methods. It also borrows the SO(2) and Wigner-D machinery that originates in eSCN, and rotary embeddings from the language-model literature.',
    add: 'Genuinely hybrid, and honest about it: the TACE paper concedes it draws on e3nn’s path formulation, and the model card lists both a spherical and an irreducible-Cartesian feature path. But the trunk is the cluster expansion rather than Tensor Field Networks — and the direction it revives, Cartesian tensor contraction, is Shapeev’s from 2016.' },

  // ---- deliberately unconstrained ----------------------------------------
  { id: 'pet', lane: 'unc', year: 2023, slot: 0, l1: 'PET',
    name: 'PET — the Point Edge Transformer',
    group: 'Sergey Pozdnyakov and Michele Ceriotti — COSMO, EPFL',
    cite: 'arXiv:2305.19302; NeurIPS 2023',
    core: 'Two claims in one paper. An architecture that puts hidden states on every edge within the cutoff and runs an ordinary transformer over each atom’s neighbour set, so depth can be stacked without the receptive field ballooning — and which is not rotationally equivariant. And a general protocol that makes any smooth model exactly equivariant after the fact, at almost no accuracy cost.',
    inh: 'The attention stack from mainstream deep learning rather than from the chemistry literature; the smoothness and exactness requirements from the physics side; radius graphs from the Behler–Parrinello tradition.',
    add: 'The rhetorical consequence, which is the one that matters here: equivariance stops being an architectural prerequisite and becomes a post-processing option you may decline. This is the formal underpinning of the whole argument that unconstrained models are fine.' },
  { id: 'orbv3', lane: 'unc', year: 2025, slot: 0, l1: 'Orb-v3', board: true, rank: 14,
    name: 'Orb-v3 (conservative, infinite-neighbour checkpoint)',
    group: 'Rhodes, Vandenhaute, Simkus, Gin, Godwin, Duignan, Neumann — Orbital Materials',
    cite: 'arXiv:2504.06231; Orb-v2 arXiv:2410.22570',
    core: 'A graph network simulator with attention, pretrained as a denoising diffusion model over structures and then fine-tuned on energies, forces and stresses. No spherical harmonics, no irreps, no tensor products; rotational invariance is learned from augmentation. Orb-v3 treats equivariance, conservatism and graph sparsity as three traversable axes of a Pareto surface rather than as requirements.',
    inh: 'The learned-simulator line from DeepMind rather than any equivariant lineage; MPtrj, Alexandria and OMat24 as labels; the conservative-force formulation from the tradition it otherwise argues against.',
    add: 'The clearest before-and-after on the board: an explicitly conservative head and an unlimited-neighbour graph cut κ_SRME from Orb-v2’s 1.734 to 0.210 within one architecture family. That is chapter 7’s thesis, run as a controlled experiment by the people least motivated to prove it.' },
  { id: 'petoam', lane: 'unc', year: 2026, slot: 0, l1: 'PET-OAM-XL', board: true, rank: 5,
    name: 'PET-OAM-XL',
    group: 'Bigi, Pegolo, Mazitov, Schmidt, Ceriotti — COSMO, EPFL',
    cite: 'Nat. Commun., DOI 10.1038/s41467-025-65662-7 (2025); scaling paper arXiv:2601.16195',
    core: 'The Point Edge Transformer scaled to 730M parameters — by far the largest model on the board — trained on the OMat24 + sAlex + MPtrj mixture with a conservative energy-gradient head. Fifth by CPS. The architecture imposes no explicit rotational symmetry constraints; it learns to be equivariant through data augmentation.',
    inh: 'Not NequIP, not Equiformer, not ACE. PET → PET-MAD → the universal PET scaling work, with standard multi-head self-attention rather than SO(3) graph attention.',
    add: 'A competitive κ_SRME of 0.119 with no hard equivariance, which was historically the failure mode of unconstrained models. Its companion paper makes the sharpest version of the argument: rotational symmetry can be learned with on the order of twenty thousand augmentations, which is affordable, whereas permutational symmetry would need 30! and therefore has to stay hard-coded. The two symmetries are not on the same footing.' },

  // ---- data and benchmark -------------------------------------------------
  { id: 'mptrj', lane: 'data', year: 2023, slot: 0, l1: 'MPtrj',
    name: 'MPtrj — the sanctioned training set',
    group: 'Released with CHGNet by the Ceder group; upstream, the Materials Project',
    cite: 'MP release v2022.10.28, snapshot 2023-03-15; MPtrj with Nat. Mach. Intell. 5, 1031 (2023)',
    core: 'Every ionic relaxation step in the Materials Project release of October 2022, cleaned of unrealistic energies and forces and subsampled roughly every tenth step: 1,580,395 structures over 145,923 materials. A model is benchmark-compliant if it uses nothing outside that release.',
    inh: 'M3GNet’s insight that relaxation trajectories are free labels, applied to the whole archive.',
    add: 'The frozen-data control track this chapter depends on. Because MPtrj is fixed by definition, the MPtrj-only column of the leaderboard is a natural experiment: architecture and training recipe advance while the data stands still.' },
  { id: 'mbd', lane: 'data', year: 2023, slot: 1, l1: 'Matbench', l2: 'Discovery',
    name: 'Matbench Discovery and the Combined Performance Score',
    group: 'Janosh Riebesell and collaborators',
    cite: 'arXiv:2308.14920; Nat. Mach. Intell. 7, 836 (2025). CPS introduced March 2025',
    core: 'Frame crystal stability as a decision problem on the WBM test set rather than as a regression problem, then refuse to rank on classification alone. CPS is a weighted mean of discovery F1, phonon thermal conductivity and geometry-optimisation quality, so thermal conductivity gets almost as much say as discovery itself.',
    inh: 'The WBM substitution test set; the Materials Project convex hull; the κ_SRME phonon metric of Póta et al. (2024).',
    add: 'The mechanism by which the benchmark punishes a model that wins the classification task with an unphysical energy surface. It is also why the ranking used throughout this chapter differs from ranking by F1 — and the site itself warns that CPS is not a stable metric and should be quoted with a version.' },
  { id: 'omat24', lane: 'data', year: 2024, slot: 0, l1: 'OMat24 /', l2: 'the OAM recipe',
    name: 'OMat24, sAlex, and the OAM training mixture',
    group: 'Barroso-Luque, Shuaibi, Fu, Wood, Dzamba, Gao, Rizvi, Zitnick, Ulissi — FAIR Chemistry, Meta',
    cite: 'arXiv:2410.12771; dataset CC-BY-4.0',
    core: 'Take Alexandria structures and perturb them hard — rattled Boltzmann sampling, short high-temperature molecular dynamics, re-relaxation of mildly rattled cells — to produce 100,824,585 non-equilibrium structures deliberately concentrated where the Materials Project data is empty. Almost every leader now trains on the OAM mixture: OMat24 pretraining, then sAlex and MPtrj fine-tuning.',
    inh: 'Alexandria as the structural seed set; the Materials Project protocol as a starting point; the free-labels insight from M3GNet and CHGNet, pushed to synthetic perturbation.',
    add: 'Scale under an open licence — and a systematic pseudopotential incompatibility with the Materials Project worth 13.5 meV/atom, which is exactly why the recipe is always pretrain-then-fine-tune. What the corpus bought, though, was diversity rather than size: under an identical architecture and fine-tune, pretraining on the larger OC20 is worse.' },
];

// Edges. kind 'direct' means the later work states it builds on the earlier one
// architecturally; 'influence' is a conceptual debt, a diagnosis, or borrowed
// machinery from a lineage the model does not belong to. 'fan' edges are the
// training-corpus links, drawn only when the data lineage is selected because
// they reach almost everything.
const EDGES = [
  ['bp', 'gap', 'direct'], ['bp', 'mtp', 'direct'], ['bp', 'schnet', 'direct'],
  ['bp', 'deeppot', 'direct'], ['gap', 'ace', 'direct'], ['gap', 'mtp', 'influence'],
  ['gap', 'pozd', 'influence'], ['mtp', 'ace', 'influence'], ['pozd', 'nequip', 'influence'],
  ['mtp', 'tece', 'influence'],
  ['schnet', 'dimenet', 'direct'], ['schnet', 'tfn', 'direct'],
  ['dimenet', 'm3gnet', 'direct'], ['m3gnet', 'chgnet', 'direct'],
  ['chgnet', 'matris', 'direct'], ['deeppot', 'dpa4', 'direct'],
  ['steer', 'tfn', 'direct'], ['steer', 'e3nn', 'direct'], ['steer', 'cormorant', 'influence'],
  ['tfn', 'e3nn', 'direct'], ['tfn', 'nequip', 'direct'], ['e3nn', 'nequip', 'direct'],
  ['cormorant', 'nequip', 'influence'], ['tfn', 'equiformer', 'direct'],
  ['nequip', 'allegro', 'direct'], ['nequip', 'sevennet', 'direct'], ['nequip', 'mace', 'direct'],
  ['nequip', 'nequipoam', 'direct'], ['sevennet', 'equflash', 'direct'],
  ['equiformer', 'eqv2', 'direct'], ['escn', 'eqv2', 'direct'],
  ['eqv2', 'eqv2m', 'direct'], ['eqv2', 'esen', 'direct'], ['eqv2', 'ev3', 'direct'],
  ['esen', 'ev3', 'influence'], ['eqv2m', 'esen', 'influence'], ['eqv2m', 'orbv3', 'influence'],
  ['escn', 'dpa4', 'influence'], ['escn', 'tece', 'influence'],
  ['ace', 'mace', 'direct'], ['ace', 'grace', 'direct'], ['grace', 'grace3l', 'direct'],
  ['ace', 'tece', 'direct'], ['mace', 'tece', 'direct'],
  ['pet', 'petoam', 'direct'], ['pet', 'orbv3', 'influence'],
  ['chgnet', 'mptrj', 'direct'], ['mptrj', 'mbd', 'direct'], ['mptrj', 'omat24', 'influence'],
  ['omat24', 'eqv2m', 'direct'],
  ['omat24', 'esen', 'fan'], ['omat24', 'nequipoam', 'fan'], ['omat24', 'ev3', 'fan'],
  ['omat24', 'equflash', 'fan'], ['omat24', 'grace3l', 'fan'], ['omat24', 'tece', 'fan'],
  ['omat24', 'petoam', 'fan'], ['omat24', 'matris', 'fan'], ['omat24', 'orbv3', 'fan'],
];

// The leaderboard, 2026-07-29. e3nn = descends from Tensor Field Networks,
// through either the NequIP message-passing branch or the Equiformer/eSCN
// transformer branch.
const BOARD = [
  { rank: 1,  name: 'TECE-OAM-RRA-1.0',      org: 'ShanghaiTech / Nanjing',  set: 'OAM',    f1: 0.929, k: 0.0927, rmsd: 0.0575, cps: 0.908, lin: 'ace',  e3nn: false, note: 'irreducible-Cartesian-tensor ACE' },
  { rank: 2,  name: 'EquFlashV2-45M-OAM',    org: 'Samsung Electronics',     set: 'OAM',    f1: 0.929, k: 0.0941, rmsd: 0.0577, cps: 0.907, lin: 'equi', e3nn: true,  note: 'via SevenNet-0, from NequIP' },
  { rank: 3,  name: 'EquiformerV3+DeNS-OAM', org: 'MIT / Mirror Physics',    set: 'OAM',    f1: 0.931, k: 0.1178, rmsd: 0.0595, cps: 0.902, lin: 'equi', e3nn: true,  note: 'Equiformer → eSCN → V2 → V3' },
  { rank: 4,  name: 'GRACE-3L-OAM-L',        org: 'ICAMS Bochum',            set: 'OAM',    f1: 0.925, k: 0.1211, rmsd: 0.0575, cps: 0.900, lin: 'ace',  e3nn: false, note: 'graph atomic cluster expansion' },
  { rank: 5,  name: 'PET-OAM-XL',            org: 'COSMO, EPFL',             set: 'OAM',    f1: 0.924, k: 0.1192, rmsd: 0.0596, cps: 0.898, lin: 'unc',  e3nn: false, note: 'rotationally unconstrained' },
  { rank: 6,  name: 'TACE-OAM-L',            org: 'ShanghaiTech / Nanjing',  set: 'OAM',    f1: 0.910, k: 0.1260, rmsd: 0.0606, cps: 0.889, lin: 'ace',  e3nn: false, note: 'irreducible-Cartesian-tensor ACE' },
  { rank: 7,  name: 'eSEN-30M-OAM',          org: 'FAIR at Meta',            set: 'OAM',    f1: 0.925, k: 0.1704, rmsd: 0.0608, cps: 0.888, lin: 'equi', e3nn: true,  note: 'eSCN / EquiformerV2 backbone' },
  { rank: 8,  name: 'EquFlash',              org: 'Samsung Electronics',     set: 'OAM',    f1: 0.919, k: 0.1583, rmsd: 0.0602, cps: 0.888, lin: 'equi', e3nn: true,  note: 'via SevenNet-0, from NequIP' },
  { rank: 9,  name: 'Nequip-OAM-XL',         org: 'MIR Group, Harvard',      set: 'OAM',    f1: 0.906, k: 0.1252, rmsd: 0.0630, cps: 0.886, lin: 'equi', e3nn: true,  note: 'NequIP itself, scaled up' },
  { rank: 10, name: 'MatRIS-10M-OAM',        org: 'ICT, Chinese Acad. Sci.', set: 'OAM',    f1: 0.921, k: 0.2183, rmsd: 0.0601, cps: 0.877, lin: 'inv',  e3nn: false, note: 'explicitly invariant' },
  { rank: 11, name: 'SevenNet-Omni-i12',     org: 'Seoul National Univ.',    set: 'COSMOS', f1: 0.906, k: 0.1917, rmsd: 0.0618, cps: 0.873, lin: 'equi', e3nn: true,  note: 'NequIP reimplemented, 12 layers' },
  { rank: 12, name: 'Nequip-OAM-L',          org: 'MIR Group, Harvard',      set: 'OAM',    f1: 0.893, k: 0.1657, rmsd: 0.0647, cps: 0.870, lin: 'equi', e3nn: true,  note: 'NequIP itself, 9.6M params' },
  { rank: 13, name: 'GRACE-2L-OAM-L',        org: 'ICAMS Bochum',            set: 'OAM',    f1: 0.883, k: 0.1688, rmsd: 0.0639, cps: 0.865, lin: 'ace',  e3nn: false, note: 'graph atomic cluster expansion' },
  { rank: 14, name: 'ORB v3',                org: 'Orbital Materials',       set: 'MP+Alex+OMat', f1: 0.905, k: 0.2102, rmsd: 0.0750, cps: 0.861, lin: 'unc', e3nn: false, note: 'unconstrained, diffusion-pretrained' },
  { rank: 15, name: 'Allegro-OAM-L',         org: 'MIR Group, Harvard',      set: 'OAM',    f1: 0.895, k: 0.3186, rmsd: 0.0651, cps: 0.840, lin: 'equi', e3nn: true,  note: 'NequIP with message passing deleted' },
  { rank: 16, name: 'DPA-4.0.1-Pro-MPtrj',   org: 'AISI Beijing / PKU',      set: 'MPtrj',  f1: 0.857, k: 0.2114, rmsd: 0.0687, cps: 0.840, lin: 'inv',  e3nn: false, note: 'DeepPot-SE line plus eSCN SO(2)' },
  { rank: 36, name: 'eqV2 M',                org: 'FAIR at Meta',            set: 'OAM',    f1: 0.917, k: 1.7707, rmsd: 0.0691, cps: 0.558, lin: 'equi', e3nn: true,  note: 'direct forces — see chapter 7' },
];

// The MPtrj-only control track: data frozen at 1,580,395 structures, with only
// the architecture and the training recipe moving.
const FROZEN = [
  { name: 'MACE-MP-0',            date: '2023-07', f1: 0.669, cps: 0.637, k: 0.682 },
  { name: 'SevenNet-0',           date: '2024-07', f1: 0.724, cps: 0.697, k: 0.762 },
  { name: 'SevenNet-l3i5',        date: '2024-12', f1: 0.760, cps: 0.714, k: 0.550 },
  { name: 'eSEN-30M-MP',          date: '2025-03', f1: 0.831, cps: 0.797, k: 0.340 },
  { name: 'MatRIS-10M-MP',        date: '2025-10', f1: 0.847, cps: 0.778, k: 0.489 },
  { name: 'EquiformerV3+DeNS-MP', date: '2026-04', f1: 0.863, cps: 0.830, k: 0.2753 },
  { name: 'DPA-4.0.1-Pro-MPtrj',  date: '2026-06', f1: 0.857, cps: 0.840, k: 0.211 },
];

// Same architecture family, MPtrj-only checkpoint versus OAM-trained checkpoint.
const SWAP = [
  { fam: 'MACE (MP-0 → MPA-0)', mp: 0.637, oam: 0.795 },
  { fam: 'GRACE-2L (MPtrj → OAM)', mp: 0.681, oam: 0.837 },
  { fam: 'SevenNet (0 → MF-ompa)', mp: 0.697, oam: 0.844 },
  { fam: 'eSEN-30M (MP → OAM)', mp: 0.797, oam: 0.888 },
];

// OMat24 paper, Table 2: identical eqV2 architecture, identical MPtrj fine-tune,
// only the pretraining corpus changes. Sizes in millions of structures.
const PRETRAIN = [
  { model: 'eqV2-S', corpus: 'OC20',   nStruct: 134, f1: 0.837, mae: 33 },
  { model: 'eqV2-S', corpus: 'OMat24', nStruct: 101, f1: 0.890, mae: 26 },
  { model: 'eqV2-L', corpus: 'OC20',   nStruct: 134, f1: 0.860, mae: null },
  { model: 'eqV2-L', corpus: 'OMat24', nStruct: 101, f1: 0.915, mae: null },
];

// The four hypotheses that were adversarially fact-checked against primary
// sources before this chapter was written. Two did not survive.
const AUDIT = [
  { survived: true,  claim: 'NequIP was the first neural message-passing potential to combine local atomic energies, ℓ ≥ 1 equivariant features and conservative forces', detail: 'true with caveats' },
  { survived: false, claim: 'every model in the CPS top ten descends from Tensor Field Networks / e3nn', detail: 'REFUTED' },
  { survived: false, claim: 'training-data scale was the biggest driver of the 2024–26 jump', detail: 'REFUTED' },
  { survived: true,  claim: 'κ_SRME exposes a failure invisible to F1, associated with direct forces', detail: 'true with caveats' },
];

const CPS_W = { f1: 0.5, kappa: 0.4, rmsd: 0.1 };
const RMSD_BASELINE = 0.15;
const cpsOf = (r) => CPS_W.f1 * r.f1
  + CPS_W.kappa * (1 - r.k / 2)
  + CPS_W.rmsd * Math.min(Math.max(1 - r.rmsd / RMSD_BASELINE, 0), 1);
const dateX = (s) => { const [y, m] = s.split('-').map(Number); return y + (m - 0.5) / 12; };

// ---------------------------------------------------------------------------

function buildTree(onPick, highlight) {
  const W = 1100, rowH = 53, laneGap = 18, padL = 108, padR = 26, padT = 42, padB = 18;
  const colW = (W - padL - padR) / (YEARS.length - 1);
  const colX = (y) => padL + YEARS.indexOf(y) * colW;

  const rows = {};
  for (const l of LANES) rows[l] = 1;
  for (const n of NODES) rows[n.lane] = Math.max(rows[n.lane], n.slot + 1);
  const laneTop = {};
  let cursor = padT;
  for (const l of LANES) { laneTop[l] = cursor; cursor += rows[l] * rowH + laneGap; }
  const H = cursor - laneGap + padB;
  const nodeY = (n) => laneTop[n.lane] + n.slot * rowH + rowH / 2 - 7;

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img',
    'aria-label': 'Family tree of machine-learned interatomic potentials, 2007 to 2026' });
  const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
  const lit = (n) => highlight === 'all' || n.lane === highlight;

  for (const l of LANES) {
    svg.appendChild(el('rect', { x: padL - 68, y: laneTop[l] - 8, width: W - padL - padR + 82,
      height: rows[l] * rowH - 2, rx: 7,
      fill: highlight === l ? LIN[l].color : '#1B2733',
      opacity: highlight === l ? 0.055 : 0.022 }));
    const t = el('text', { x: 12, y: laneTop[l] + (rows[l] * rowH) / 2 - 2, 'font-size': 11,
      fill: LIN[l].color, 'font-family': 'ui-sans-serif, system-ui, sans-serif',
      'font-weight': 700, opacity: highlight === 'all' || highlight === l ? 1 : 0.35 });
    t.textContent = LIN[l].name;
    svg.appendChild(t);
  }

  for (const y of YEARS) {
    const x = colX(y);
    svg.appendChild(el('line', { x1: x, x2: x, y1: padT - 18, y2: H - padB,
      stroke: '#E8ECF0', 'stroke-width': 1 }));
    const t = el('text', { x, y: padT - 24, 'text-anchor': 'middle', 'font-size': 11,
      fill: '#5A6773', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    t.textContent = String(y);
    svg.appendChild(t);
  }

  for (const [a, b, kind] of EDGES) {
    const na = byId[a], nb = byId[b];
    if (!na || !nb) continue;
    if (kind === 'fan' && highlight !== 'data') continue;
    const x1 = colX(na.year), y1 = nodeY(na), x2 = colX(nb.year), y2 = nodeY(nb);
    const touched = highlight === 'all' || lit(na) || lit(nb);
    const mx = (x1 + x2) / 2;
    svg.appendChild(el('path', {
      d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`,
      fill: 'none',
      stroke: touched ? LIN[na.lane].color : '#9AA3AC',
      'stroke-width': kind === 'direct' ? 1.7 : 1.2,
      'stroke-dasharray': kind === 'direct' ? null : '4 4',
      opacity: touched ? (kind === 'direct' ? 0.72 : 0.5) : 0.08 }));
  }

  for (const n of NODES) {
    const x = colX(n.year), y = nodeY(n), on = lit(n);
    const g = el('g', { style: 'cursor:pointer' });
    const title = el('title');
    title.textContent = `${n.name} — ${n.year}`;
    g.appendChild(title);
    g.appendChild(el('rect', { x: x - colW / 2 + 3, y: y - 16, width: colW - 6, height: rowH - 4,
      fill: 'transparent' }));
    if (n.board) {
      g.appendChild(el('circle', { cx: x, cy: y, r: 11, fill: 'none',
        stroke: LIN[n.lane].color, 'stroke-width': 1.2, opacity: on ? 0.6 : 0.12 }));
    }
    g.appendChild(el('circle', { cx: x, cy: y, r: n.board ? 7 : 5.5,
      fill: on ? LIN[n.lane].color : '#C3C9D0', opacity: on ? 1 : 0.5 }));
    const t = el('text', { x, y: y + 21, 'text-anchor': 'middle', 'font-size': 10.5,
      fill: on ? '#1B2733' : '#9AA3AC',
      'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    const s1 = el('tspan', { x, dy: 0 });
    s1.textContent = n.l1;
    t.appendChild(s1);
    if (n.l2) {
      const s2 = el('tspan', { x, dy: 11 });
      s2.textContent = n.l2;
      t.appendChild(s2);
    }
    g.appendChild(t);
    g.addEventListener('click', () => onPick(n));
    svg.appendChild(g);
  }
  return svg;
}

function nodePanel(n) {
  return h('div', { class: 'card', style: { borderTop: `3px solid ${LIN[n.lane].color}` } },
    h('div', { class: 'controls', style: { margin: '0 0 6px', gap: '10px' } },
      h('span', { class: 'pill', style: { background: LIN[n.lane].color, color: '#fff' } },
        LIN[n.lane].short),
      h('span', { class: 'pill' }, String(n.year)),
      n.board ? h('span', { class: 'pill ok' }, `on the board — #${n.rank} by CPS`) : null),
    h('h4', { style: { color: '#1B2733', textTransform: 'none', letterSpacing: '0',
      fontSize: '15.5px', margin: '2px 0 2px' } }, n.name),
    h('p', { style: { fontSize: '13px', color: '#5A6773', margin: '0 0 8px',
      fontFamily: 'ui-sans-serif, system-ui, sans-serif' } }, n.group),
    h('p', {}, n.core),
    h('p', {}, h('strong', {}, 'What it inherited. '), n.inh),
    h('p', {}, h('strong', {}, 'What it added. '), n.add),
    h('p', { class: 'mono', style: { fontSize: '11.8px', color: '#5A6773', marginTop: '10px' } },
      n.cite));
}

export default {
  id: 'genealogy',
  title: 'The genealogy, and where it lands',
  render(root) {
    // ---- live derived quantities -----------------------------------------
    const top10 = BOARD.filter((r) => r.rank <= 10).sort((a, b) => a.rank - b.rank);
    const nE3nn = top10.filter((r) => r.e3nn).length;
    const nAce = top10.filter((r) => r.lin === 'ace').length;
    const nFree = top10.filter((r) => r.lin === 'unc' || r.lin === 'inv').length;
    const cpsErr = Math.max(...BOARD.map((r) => Math.abs(cpsOf(r) - r.cps)));

    const frozenLo = FROZEN[0];
    const frozenHi = FROZEN.reduce((m, r) => (r.cps > m.cps ? r : m), FROZEN[0]);
    const boardBest = BOARD.reduce((m, r) => (r.cps > m.cps ? r : m), BOARD[0]);
    const totalGain = boardBest.cps - frozenLo.cps;
    const frozenGain = frozenHi.cps - frozenLo.cps;
    const frozenShare = (100 * frozenGain) / totalGain;
    const bestF1 = BOARD.reduce((m, r) => (r.f1 > m.f1 ? r : m), BOARD[0]);
    const frozenF1Hi = FROZEN.reduce((m, r) => (r.f1 > m.f1 ? r : m), FROZEN[0]);
    const f1Share = (100 * (frozenF1Hi.f1 - frozenLo.f1)) / (bestF1.f1 - frozenLo.f1);

    const swapDeltas = SWAP.map((s) => s.oam - s.mp);
    const swapMean = swapDeltas.reduce((a, b) => a + b, 0) / swapDeltas.length;
    const swapMin = Math.min(...swapDeltas);
    const swapMax = Math.max(...swapDeltas);
    const oamRows = BOARD.filter((r) => r.set === 'OAM');
    const oamLo = Math.min(...oamRows.map((r) => r.cps));
    const oamHi = Math.max(...oamRows.map((r) => r.cps));
    const oamSpread = oamHi - oamLo;

    const oc20 = PRETRAIN.find((p) => p.model === 'eqV2-S' && p.corpus === 'OC20');
    const omat = PRETRAIN.find((p) => p.model === 'eqV2-S' && p.corpus === 'OMat24');
    const corpusShrink = 100 * (1 - omat.nStruct / oc20.nStruct);

    const f1Hi = Math.max(...top10.map((r) => r.f1));
    const f1Lo = Math.min(...top10.map((r) => r.f1));
    const kHi = Math.max(...top10.map((r) => r.k));
    const kLo = Math.min(...top10.map((r) => r.k));
    const f1Span = f1Hi - f1Lo;
    const kSpan = kHi - kLo;
    const f1Contrib = CPS_W.f1 * f1Span;
    const kContrib = CPS_W.kappa * (kSpan / 2);

    const nCited = NODES.filter((n) => n.cite && n.cite.length > 8).length;
    const dataDelta = totalGain - frozenGain;

    // ---- prose ------------------------------------------------------------
    root.append(
      h('p', { class: 'eyebrow syn' }, 'Chapter 10'),
      h('h1', {}, 'The genealogy, and where it lands'),
      h('p', { class: 'lede' },
        'Eight chapters have argued one idea from first principles. This one asks where the idea ' +
        'came from, who carried it, and whether it is still what is winning. The answer to the ' +
        'last question is more interesting than this tutorial’s own framing would suggest, and it ' +
        'is not the answer the reading group assumed.'),

      h('h2', {}, 'Five roads, and a sixth that opened late'),
      h('p', { class: 'prose', html:
        `The map below carries ${NODES.length} works, arranged by year across and by lineage ` +
        'down. Five roads leave 2007. The <em>descriptor</em> road compresses an atom’s ' +
        'neighbourhood into rotation-invariant numbers and regresses on those; it runs from ' +
        'Behler–Parrinello through GAP and SOAP and moment tensor potentials, and arrives — ' +
        'fittingly — at a proof that its own descriptors are degenerate. The <em>invariant graph ' +
        'network</em> road learns the descriptor instead of designing it, while keeping ' +
        'everything it carries a scalar. The <em>equivariance</em> road, which begins in computer ' +
        'vision rather than in chemistry, arrives at Tensor Field Networks and then at NequIP. ' +
        'The <em>cluster expansion</em> road is Drautz’s, entirely independent of the third, and ' +
        'derives a complete basis rather than proposing one. And the <em>data</em> road is not an ' +
        'architecture at all: it is the corpora and the benchmark that decide what counts as ' +
        'winning.' }),
      h('p', { class: 'prose' },
        'The sixth road opened around 2023 and is the reason this chapter cannot end tidily. It ' +
        'is the deliberate refusal of built-in symmetry — models that could impose equivariance ' +
        'and choose not to, learning it from augmentation instead. In 2021 that would have been ' +
        'a curiosity. Today one such model sits fifth on the leaderboard.'),
      h('p', { class: 'prose' },
        'Solid edges are architectural descent: the later work states that it builds on the ' +
        'earlier one. Dashed edges are conceptual debts — a diagnosis, a borrowed kernel, a piece ' +
        'of machinery taken from a lineage the model does not belong to. Those dashed cross-lane ' +
        'edges are where most of the interesting history is. A ringed node is a model on the ' +
        'current Matbench Discovery leaderboard.'));

    // ---- the tree ---------------------------------------------------------
    const treeDemo = h('div', { class: 'demo' });
    treeDemo.append(h('h3', {}, 'The family tree — click any node'),
      h('p', { class: 'hint' },
        'Selecting a lineage dims everything that does not touch it. Selecting “data & scale” ' +
        'additionally draws the training-corpus edges, which are hidden by default because they ' +
        'reach nearly every model built after 2024.'));
    const treeHolder = h('div');
    const panelHolder = h('div', { style: { marginTop: '12px' } });
    let highlight = 'all';
    const showNode = (n) => { panelHolder.innerHTML = ''; panelHolder.append(nodePanel(n)); };
    const redraw = () => {
      treeHolder.innerHTML = '';
      treeHolder.append(buildTree(showNode, highlight));
    };
    treeDemo.append(treeHolder,
      h('div', { class: 'controls' },
        segmented({
          label: 'highlight lineage',
          options: [{ label: 'all', value: 'all' },
            ...LANES.map((l) => ({ label: LIN[l].short, value: l }))],
          value: 'all',
          onPick: (v) => { highlight = v; redraw(); },
        })),
      h('div', { class: 'controls', style: { gap: '20px', fontSize: '12.5px', color: '#5A6773' } },
        h('span', {}, '——  direct architectural descent'),
        h('span', {}, '- - -  conceptual influence or borrowed machinery'),
        h('span', {}, '◎  on the current leaderboard')),
      panelHolder);
    root.append(treeDemo);
    redraw();
    showNode(NODES.find((n) => n.id === 'nequip'));

    root.append(
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Two roots, not one'),
        h('div', {},
          'Notice that MACE has two incoming solid edges — one from NequIP and one from ACE — ' +
          'and that its companion paper, Multi-ACE, proves the two roads were a single design ' +
          'space the whole time: ACE, NequIP, MACE, BOTNet, PaiNN, SchNet and Tensor Field ' +
          'Networks all fall out of one framework, and linearised NequIP turns out to be a ' +
          'particular sparsification of a much larger polynomial model. That result is why the ' +
          'question “which lineage does this model belong to?” has a real answer for each model ' +
          'and no answer at all for the field.')),

      h('h2', {}, 'The leaderboard, and the column that matters'),
      h('p', { class: 'prose', html:
        'Matbench Discovery ranks on a composite. Its Combined Performance Score is ' +
        '$\\mathrm{CPS} = 0.5\\,F_1 + 0.4\\,\\bigl(1 - \\kappa_{\\mathrm{SRME}}/2\\bigr) + ' +
        '0.1\\,\\mathrm{clamp}\\bigl(1 - \\mathrm{RMSD}/0.15\\bigr)$, so a model that classifies ' +
        'stability beautifully while producing an unphysical energy surface is punished rather ' +
        'than crowned. The table below is that ranking. Every CPS shown in it is recomputed in ' +
        'your browser from the three components beside it, which is how you can tell the last ' +
        'column is a claim about ancestry and not about arithmetic.' }));

    const tableDemo = h('div', { class: 'demo' });
    tableDemo.append(h('h3', {}, 'Matbench Discovery, snapshot 2026-07-29'),
      h('p', { class: 'hint' },
        'Click a column heading to sort. Rank is the site’s own CPS rank among 41 active models; ' +
        'the highlighted row is included because of how far it falls. “OAM” means pretraining on ' +
        'OMat24 followed by fine-tuning on sAlex and MPtrj.'));
    const tbody = h('tbody');
    const head = h('thead');
    let sortKey = 'rank';
    let sortDir = 1;
    const fill = () => {
      head.innerHTML = '';
      const th = (label, key, num) => h('th', {
        class: num ? 'num' : null,
        style: { cursor: 'pointer', userSelect: 'none' },
        onClick: () => {
          if (sortKey === key) sortDir = -sortDir;
          else { sortKey = key; sortDir = (key === 'rank' || key === 'k' || key === 'rmsd') ? 1 : -1; }
          fill();
        },
      }, label + (sortKey === key ? (sortDir > 0 ? ' ▲' : ' ▼') : ''));
      head.append(h('tr', {},
        th('#', 'rank', false), th('model', 'name', false), th('organisation', 'org', false),
        th('training set', 'set', false), th('F1', 'f1', true), th('κ_SRME', 'k', true),
        th('RMSD', 'rmsd', true), th('CPS', 'cps', true), th('descends from', 'lin', false)));
      tbody.innerHTML = '';
      const rows = [...BOARD].sort((a, b) => {
        const va = a[sortKey];
        const vb = b[sortKey];
        if (typeof va === 'string') return sortDir * va.localeCompare(vb);
        return sortDir * (va - vb);
      });
      for (const r of rows) {
        tbody.append(h('tr', { class: r.rank > 20 ? 'hi' : null },
          h('td', {}, String(r.rank)),
          h('td', {}, r.name),
          h('td', {}, r.org),
          h('td', {}, r.set),
          h('td', { class: 'num' }, r.f1.toFixed(3)),
          h('td', { class: 'num' }, r.k.toFixed(3)),
          h('td', { class: 'num' }, r.rmsd.toFixed(4)),
          h('td', { class: 'num' }, cpsOf(r).toFixed(3)),
          h('td', {},
            h('span', { class: 'pill', style: { background: LIN[r.lin].color, color: '#fff' } },
              LIN[r.lin].short),
            h('span', { style: { color: '#5A6773', marginLeft: '7px' } }, r.note))));
      }
    };
    fill();
    tableDemo.append(h('table', {}, head, tbody), h('div', { class: 'readout' },
      `CPS recomputed from F1, kappa_SRME and RMSD for all ${BOARD.length} rows\n` +
      `  largest disagreement with the published value:  ${cpsErr.toExponential(1)}\n\n` +
      'top ten by CPS, by lineage:\n' +
      `  Tensor Field Networks / e3nn      ${nE3nn} of ${top10.length}` +
      `   (${((100 * nE3nn) / top10.length).toFixed(0)}%)\n` +
      `  atomic cluster expansion          ${nAce} of ${top10.length}\n` +
      `  unconstrained, or invariant       ${nFree} of ${top10.length}`));
    root.append(tableDemo);

    root.append(
      h('h2', {}, 'The punchline: the top tier is architecturally plural'),
      h('p', { class: 'prose' },
        'It would be a tidy story, and it is the story this tutorial has spent eight chapters ' +
        'setting up, to say that equivariance won and that everything at the top is a descendant ' +
        'of Tensor Field Networks. That claim is false, and it fails by a wide enough margin that ' +
        'the failure is itself the finding.'),
      h('p', { class: 'prose' },
        `Of the ten models in the CPS top ten, ${nE3nn} descend from the TFN and e3nn lineage: ` +
        'EquFlash and EquFlashV2 through SevenNet-0 from NequIP, Nequip-OAM-XL which is NequIP ' +
        'itself, and eSEN-30M-OAM and EquiformerV3+DeNS-OAM on the eSCN and EquiformerV2 ' +
        `transformer branch. The other ${top10.length - nE3nn} come from somewhere else, and two ` +
        'of them are not equivariant architectures at all.'),
      h('div', { class: 'grid2' },
        h('div', { class: 'card', style: { borderTop: `3px solid ${LIN.ace.color}` } },
          h('h4', { style: { color: LIN.ace.color } }, 'the cluster-expansion counterexamples'),
          h('p', { html:
            'GRACE-3L-OAM-L at rank 4 is graph ACE from Drautz’s own group in Bochum — the same ' +
            'Drautz whose 2019 paper is the second root of the tree — implemented in TensorFlow, ' +
            'not e3nn. Its equivariance comes from ACE’s own spherical-harmonic product basis ' +
            'rather than from tensor-field-network convolution layers.' }),
          h('p', { html:
            'TACE at rank 6 and TECE at rank 1 go further: their distinguishing idea is the ' +
            'irreducible <em>Cartesian</em> tensor, whose stated purpose is to avoid the axis ' +
            'dependence and Clebsch–Gordan overhead of spherical-tensor methods. They are ' +
            'genuinely hybrid, and honest about it — the TACE paper concedes that it draws on ' +
            'e3nn’s path formulation, and the radial rotary attention borrows Wigner-D machinery ' +
            'that originates in eSCN — but the trunk is the cluster expansion, and the Cartesian ' +
            'direction is Shapeev’s from 2016.' })),
        h('div', { class: 'card', style: { borderTop: `3px solid ${LIN.unc.color}` } },
          h('h4', { style: { color: LIN.unc.color } }, 'the two that decline the premise'),
          h('p', { html:
            'PET-OAM-XL at rank 5 is a Point Edge Transformer from Ceriotti’s group at EPFL, ' +
            'scaled to 730M parameters. Its own authors describe it as rotationally ' +
            'unconstrained: the architecture imposes no explicit rotational symmetry and learns ' +
            'to be equivariant from data augmentation. The companion paper is titled <em>Pushing ' +
            'the limits of unconstrained machine-learned interatomic potentials</em>. A model ' +
            'whose selling point is the absence of hard-coded symmetry cannot be a descendant of ' +
            'the equivariant lineage.' }),
          h('p', { html:
            'MatRIS-10M-OAM at rank 10 is explicitly <em>invariant</em> rather than equivariant: ' +
            'attention-based three-body modelling at linear complexity, from the Institute of ' +
            'Computing Technology in Beijing, at 10.4M parameters. Its paper positions it ' +
            'against equivariant models rather than as their descendant. Rank 14, Orb-v3, is ' +
            'unconstrained too.' }))),
      h('p', { class: 'prose' },
        `So the honest count is ${nE3nn} of ${top10.length} inside the lineage this tutorial ` +
        `describes, ${nAce} on the cluster-expansion road, and ${nFree} that decline built-in ` +
        'rotational symmetry entirely. Half the top tier is not where the story says it should ' +
        'be. Set the highlight above to “unconstrained” and you can see how thin that lineage ' +
        'is — three nodes, none of them before 2023 — and how high it has climbed anyway.'),

      h('h2', {}, 'And it was not mainly the data, either'),
      h('p', { class: 'prose' },
        'The second tidy story is that none of this matters, because the 2024–26 jump was really ' +
        'about training-set size: OMat24 arrived with a hundred million structures and everyone ' +
        'who used it moved up. That is also false, and the benchmark supplies its own control ' +
        'experiment for testing it.'),
      h('p', { class: 'prose' },
        'Matbench Discovery keeps a compliant track in which the training corpus is frozen at ' +
        'MPtrj — 1,580,395 structures, fixed by definition — while architecture and training ' +
        'recipe advance. That is a natural experiment holding the data constant. Here is what it ' +
        'did.'));

    const frozenDemo = h('div', { class: 'demo' });
    frozenDemo.append(h('h3', {}, 'The frozen-data track'),
      h('p', { class: 'hint' },
        'Every point on the solid line trains on exactly the same 1.58M structures. The dashed ' +
        'line is the best model on the board, which trains on roughly seventy times more.'));
    const fp = new Plot({ width: 700, height: 330, xLabel: 'date the model was submitted',
      yLabel: 'Combined Performance Score' });
    fp.add({ points: FROZEN.map((r) => [dateX(r.date), r.cps]), color: PALETTE[0], width: 2.4,
      markers: true, r: 4, label: 'MPtrj only — 1,580,395 structures' });
    fp.add({ points: [[2023.3, boardBest.cps], [2026.75, boardBest.cps]],
      color: PALETTE[1], width: 1.8, dash: '6 4',
      label: `best on the board: ${boardBest.name}, OMat24 + sAlex + MPtrj` });
    fp.setLimits([2023.3, 2026.75], [0.6, 0.94]);
    frozenDemo.append(fp.render(), fp.legend(), h('div', { class: 'readout' },
      `frozen-data track, ${frozenLo.name} -> ${frozenHi.name}\n` +
      `  CPS          ${frozenLo.cps.toFixed(3)} -> ${frozenHi.cps.toFixed(3)}` +
      `   (+${frozenGain.toFixed(3)}) on identical data\n` +
      `  kappa_SRME   ${frozenLo.k.toFixed(3)} -> ${frozenHi.k.toFixed(3)}` +
      `   (a factor of ${(frozenLo.k / frozenHi.k).toFixed(1)}) on identical data\n\n` +
      'decomposing the whole 2023 -> 2026 gain in CPS:\n' +
      `  total, ${frozenLo.name} -> ${boardBest.name}   = +${totalGain.toFixed(3)}\n` +
      `  available at frozen data                       = +${frozenGain.toFixed(3)}` +
      `   -> ${frozenShare.toFixed(0)}%\n` +
      `  residual, everything 111M extra structures buy = +${dataDelta.toFixed(3)}` +
      `   -> ${(100 - frozenShare).toFixed(0)}%\n\n` +
      `the same decomposition on F1 alone: architecture accounts for ${f1Share.toFixed(0)}%`));
    root.append(frozenDemo);

    root.append(
      h('p', { class: 'prose' },
        `Roughly ${frozenShare.toFixed(0)} per cent of the composite gain was available without a ` +
        'single new training structure. The most striking point is the last one: ' +
        'DPA-4.0.1-Pro-MPtrj, at 22.8M parameters and trained on Materials Project data alone, ' +
        `reaches CPS ${FROZEN[FROZEN.length - 1].cps.toFixed(3)} — matching OAM-trained models ` +
        'that saw about seventy times more data, and beating the MPtrj-only checkpoint of the ' +
        'very transformer that holds the best F1 on the whole board.'),
      h('p', { class: 'prose' },
        'The data effect is real. It is simply smaller than the architecture effect, and smaller ' +
        'than the spread among models that share the corpus. Both of those are worth measuring ' +
        'rather than asserting.'),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'same architecture, MPtrj checkpoint → OAM checkpoint'),
          h('th', { class: 'num' }, 'CPS on MPtrj'), h('th', { class: 'num' }, 'CPS on OAM'),
          h('th', { class: 'num' }, 'Δ'))),
        h('tbody', {},
          ...SWAP.map((s) => h('tr', {},
            h('td', {}, s.fam),
            h('td', { class: 'num' }, s.mp.toFixed(3)),
            h('td', { class: 'num' }, s.oam.toFixed(3)),
            h('td', { class: 'num' }, `+${(s.oam - s.mp).toFixed(3)}`))),
          h('tr', { class: 'hi' },
            h('td', {}, h('strong', {}, 'mean value of adding the extra 111M structures')),
            h('td', { class: 'num' }, '—'),
            h('td', { class: 'num' }, '—'),
            h('td', { class: 'num' }, h('strong', {}, `+${swapMean.toFixed(3)}`))),
          h('tr', { class: 'hi' },
            h('td', {}, h('strong', {}, 'spread among models sharing the OAM corpus')),
            h('td', { class: 'num' }, oamLo.toFixed(3)),
            h('td', { class: 'num' }, oamHi.toFixed(3)),
            h('td', { class: 'num' }, h('strong', {}, oamSpread.toFixed(3)))))),
      h('p', { class: 'prose' },
        `Swapping MPtrj for the full OAM mixture is worth between +${swapMin.toFixed(3)} and ` +
        `+${swapMax.toFixed(3)} CPS depending on the architecture, averaging ` +
        `+${swapMean.toFixed(3)}. The spread among models trained on that same corpus is ` +
        `${oamSpread.toFixed(3)}, about ${(oamSpread / swapMean).toFixed(1)} times larger. So the ` +
        'claim that architecture differences at the top are small compared with the data effect ' +
        'is exactly backwards on the benchmark’s own headline metric. That the leaders now look ' +
        'similar is survivorship: the ones that kept direct forces are still on the board, near ' +
        'the bottom of it.'),

      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'And the data effect is not a scale effect'),
        h('div', {},
          'The OMat24 paper runs the clean version of this experiment itself. Identical ' +
          `${oc20.model} architecture, identical MPtrj fine-tune, only the pretraining corpus ` +
          `changes: OC20 with ${oc20.nStruct} million structures gives F1 ${oc20.f1.toFixed(3)} ` +
          `and formation-energy MAE ${oc20.mae} meV/atom, while OMat24 with ${omat.nStruct} ` +
          `million — ${corpusShrink.toFixed(0)} per cent fewer — gives F1 ${omat.f1.toFixed(3)} ` +
          `and ${omat.mae} meV/atom. The larger corpus is worse by ` +
          `${(omat.f1 - oc20.f1).toFixed(3)} in F1, and the same ordering holds at the large ` +
          `model size (${PRETRAIN[2].f1.toFixed(3)} against ${PRETRAIN[3].f1.toFixed(3)}). What ` +
          'OMat24 bought was non-equilibrium sampling, compositional diversity and the ' +
          'correction of a systematic softening. It did not buy volume, because it has less of ' +
          'it.')),

      h('h2', {}, 'Back to the sentence'),
      h('blockquote', { class: 'prose' },
        '“It’s called Graph Attention Transformer that is now the leading one… which ' +
        'incorporates rotational symmetry.”',
        h('span', { class: 'attrib' }, 'Reading group, 23 July 2026')),
      h('p', { class: 'prose' },
        'It deserves a fair hearing, because it was substantially right when it was said. There ' +
        'is a graph attention transformer near the top — EquiformerV3+DeNS-OAM holds the best ' +
        `classification F1 on the entire board at ${bestF1.f1.toFixed(3)} — and it is exactly ` +
        'what the sentence describes: an SE(3)-equivariant transformer whose attention weights ' +
        'are invariant scalars and whose messages are equivariant tensors. Rotational symmetry ' +
        'is not decoration in that model. It is the reason its layers have the form they have, ' +
        'and chapters 1 through 4 derive that form from nothing but the symmetry requirement.'),
      h('p', { class: 'prose', html:
        'The honest 2026 answer is that the sentence is now about half true, and that the half ' +
        'which fails is the interesting half. Ranked by the composite the benchmark itself uses, ' +
        'the graph attention transformer is third rather than first. The model above it is ' +
        'equivariant but uses no attention at all, deliberately and by design. The model above ' +
        '<em>that</em> is an edge cluster expansion built on Cartesian tensors. Fifth place ' +
        'imposes no rotational symmetry by construction, and tenth place is invariant rather ' +
        'than equivariant.' }),
      h('p', { class: 'prose', html:
        'So what does separate the leaders now? Not classification accuracy: the top ten agree ' +
        `on $F_1$ to within ${f1Span.toFixed(3)}, about ` +
        `${((100 * f1Span) / f1Hi).toFixed(1)} per cent in relative terms. What they disagree ` +
        `about is $\\kappa_{\\mathrm{SRME}}$, which spans ${kSpan.toFixed(3)} across the same ten ` +
        `models — a factor of ${(kHi / kLo).toFixed(1)} between best and worst. Weighted into ` +
        `the composite, the classification term contributes ${f1Contrib.toFixed(3)} of spread ` +
        `and the thermal-conductivity term ${kContrib.toFixed(3)}, which is ` +
        `${(kContrib / f1Contrib).toFixed(1)} times as much. The metric that decides the ranking ` +
        'is not the one the field talks about.' }),
      h('p', { class: 'prose' },
        'That is a statement about second derivatives of the energy surface, not about ' +
        'representation theory. Every model in the top fifteen now derives its forces as an ' +
        'analytic gradient; the two famous direct-force models sit at rank 36 and below despite ' +
        'excellent F1. Smoothness of the potential — envelopes, no hard neighbour caps, no grid ' +
        'discontinuities — turns out to matter as much as the symmetry group, and chapter 7 ' +
        'measures why while also showing that conservativeness alone is neither necessary nor ' +
        'sufficient. Add training-corpus diversity, and add the kernel engineering that lets a ' +
        '45M-parameter model take second place, and you have most of what actually differs ' +
        'between the leaders.'),
      h('div', { class: 'note syn' },
        h('span', { class: 'tag' }, 'The version of the sentence that survives 2026'),
        h('div', {},
          'Built-in equivariance was the decisive idea of 2021 to 2023 and remains the majority ' +
          'design at the top of the board. What it bought was data efficiency and a strong ' +
          'inductive prior — which is precisely the claim NequIP made, and precisely the claim ' +
          'chapters 2 and 5 test. At today’s data scales that prior is less decisive than it ' +
          'was, and the leaderboard has become architecturally plural: cluster-expansion models ' +
          'built on Cartesian tensors, and an explicitly unconstrained transformer, sit in the ' +
          'top five. Symmetry did not win outright. It set the terms of a design space that ' +
          'several roads now reach.')));

    // ---- check ledger ------------------------------------------------------
    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Check ledger — recomputed in your browser'),
      h('p', { class: 'hint' },
        'This chapter has no Python twin: its raw material is the benchmark’s own model records ' +
        'plus the primary papers. A [PASS] line is a proposition that survived checking; a ' +
        '[FAIL] line is one that did not. Two of the popular claims about this leaderboard do ' +
        'not survive, and those two failures are the chapter.'));
    const checks = [
      { pass: cpsErr < 2e-3,
        name: 'CPS reproduces from F1, κ_SRME and RMSD for every tabulated model',
        detail: `${BOARD.length} rows, max |Δ| = ${cpsErr.toExponential(1)}` },
      { pass: nE3nn === top10.length,
        name: 'claim: every CPS top-ten model descends from TFN / e3nn',
        detail: `REFUTED — ${nE3nn} of ${top10.length}` },
      { pass: nFree === 0,
        name: 'claim: every top-ten model imposes rotational symmetry by construction',
        detail: `REFUTED — ${nFree} of ${top10.length} do not (PET-OAM-XL, MatRIS-10M-OAM)` },
      { pass: frozenShare < 50,
        name: 'claim: training-data scale drove most of the 2024–26 jump',
        detail: `REFUTED — ${frozenShare.toFixed(0)}% of the CPS gain is at frozen data` },
      { pass: oamSpread > swapMean,
        name: 'architecture spread on one corpus exceeds the mean data effect',
        detail: `${oamSpread.toFixed(3)} vs +${swapMean.toFixed(3)} CPS` },
      { pass: omat.f1 > oc20.f1 && omat.nStruct < oc20.nStruct,
        name: 'a larger pretraining corpus is not automatically the better one',
        detail: `OC20 ${oc20.nStruct}M → F1 ${oc20.f1.toFixed(3)}; OMat24 ${omat.nStruct}M → ${omat.f1.toFixed(3)}` },
      { pass: nCited === NODES.length,
        name: 'every node in the tree carries a primary citation',
        detail: `${nCited}/${NODES.length}` },
    ];
    const inner = h('div');
    for (const c of checks) inner.append(checkLine(c.pass, c.name, c.detail));
    const nPass = checks.filter((c) => c.pass).length;
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${nPass === checks.length ? 'ok' : 'bad'}` },
        `${nPass}/${checks.length}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);

    ledger.append(h('h3', { style: { marginTop: '22px' } }, 'The four hypotheses, audited'),
      h('p', { class: 'hint' },
        'Each was fact-checked independently against primary sources before this chapter was ' +
        'written. Two were refuted, and the refutations are what the two sections above are ' +
        'about; the two that survived carry caveats that chapters 5 and 6 spend their length on.'));
    const auditBox = h('div');
    for (const a of AUDIT) auditBox.append(checkLine(a.survived, a.claim, a.detail));
    const nSurv = AUDIT.filter((a) => a.survived).length;
    auditBox.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${nSurv === AUDIT.length ? 'ok' : 'bad'}` },
        `${nSurv}/${AUDIT.length}`),
      h('span', {}, 'SURVIVED')));
    ledger.append(auditBox);

    ledger.append(h('div', { class: 'readout' },
      'source of truth for every leaderboard number above:\n\n' +
      '  git clone https://github.com/janosh/matbench-discovery\n' +
      '  cat matbench-discovery/models/*/*.yml       # snapshot taken 2026-07-29\n\n' +
      'discovery metrics are the "unique prototypes" subset (the site default);\n' +
      'kappa_SRME is the kappa_103 phonon subset; RMSD is geo_opt at symprec 1e-2.\n' +
      'CPS is recomputed above from the site\'s own formula in\n' +
      '  site/src/lib/combined-scores.svelte.ts\n' +
      '    CPS = 0.5*F1 + 0.4*(1 - kSRME/2) + 0.1*clamp(1 - RMSD/0.15, 0, 1)'));
    root.append(ledger);
  },
};
