import { h, slider, segmented, checkLine } from '../ui.js';
import {
  realSH, wignerD, envelope, besselBasis, randomRotation, mulberry32,
} from '../e3.js';
import { matvec, norm } from '../linalg.js';
import { filterExplorer } from '../filter-explorer.js';
import { pathExplorer } from '../path-explorer.js';

// A fixed toy molecule: a central atom with five neighbours inside the cutoff.
const R_CUT = 2.6;
const MOL = [
  [1.05, 0.20, -0.15],
  [-0.62, 1.02, 0.30],
  [-0.40, -0.92, 0.66],
  [0.35, -0.55, -1.35],
  [1.72, -1.05, 0.60],
];

const LMAX = 2;
const N_RADIAL = 4;

// Fixed "learned" weights — random but seeded, because the point of this chapter
// is the transformation law, which holds for ANY weights. That is exactly the
// claim: equivariance is a property of the architecture, not of the fit.
function makeWeights(seed) {
  const rng = mulberry32(seed);
  const g = () => 2 * rng() - 1;
  return {
    radial: Array.from({ length: LMAX + 1 }, () => Array.from({ length: N_RADIAL }, g)),
    selfInt: Array.from({ length: LMAX + 1 }, () => 0.5 + rng()),
    gateW: Array.from({ length: LMAX }, () => Array.from({ length: 3 }, g)),
    readout: Array.from({ length: 3 }, g),
  };
}

/** One equivariant convolution: a message from every neighbour, summed.
 *  The input features here are scalars, so the Clebsch-Gordan product with the
 *  l=0 input degenerates into a scale and the message is R_l(r) * Y_l(rhat). */
function convolve(pos, W) {
  const out = [];
  for (let l = 0; l <= LMAX; l++) out.push(new Array(2 * l + 1).fill(0));
  for (const d of pos) {
    const r = norm(d);
    if (r >= R_CUT) continue;
    const env = envelope(r, R_CUT);
    for (let l = 0; l <= LMAX; l++) {
      const Y = realSH(l, d);
      const w = W.radial[l].reduce((s, c, n) => s + c * besselBasis(r, n + 1, R_CUT), 0) * env;
      for (let i = 0; i < Y.length; i++) out[l][i] += w * Y[i];
    }
  }
  return out;
}

/** Atom-wise linear map: mixes channels WITHIN each degree. It cannot mix across
 *  degrees, because that would not commute with rotation. */
function selfInteraction(feat, W) {
  return feat.map((f, l) => f.map((v) => v * W.selfInt[l]));
}

/** Gated nonlinearity (Weiler et al.): a scalar built from invariants multiplies
 *  each higher-degree feature. Direction untouched, so the transformation law
 *  survives; magnitude becomes a nonlinear function of the input. */
function gatedNonlinearity(feat, W) {
  const out = [feat[0].map((v) => Math.tanh(v))];
  for (let l = 1; l <= LMAX; l++) {
    const inv = norm(feat[l]);
    const gate = Math.tanh(
      W.gateW[l - 1][0] * feat[0][0] + W.gateW[l - 1][1] * inv + W.gateW[l - 1][2]);
    out.push(feat[l].map((v) => gate * v));
  }
  return out;
}

/** The obvious alternative: tanh on every component. Destroys equivariance,
 *  because tanh does not commute with a rotation matrix. */
function componentwiseNonlinearity(feat) {
  return feat.map((f) => f.map((v) => Math.tanh(v)));
}

/** Scalar readout. A single block would have nothing but its own l=0 channel to
 *  read from, and tanh applied to a scalar is invariant whatever else is broken —
 *  so a one-layer toy reading only l=0 cannot see a damaged l>=1 feature at all.
 *  A real network never does that: the next block couples the higher degrees back
 *  down to scalars. The power-spectrum terms below stand in for that coupling,
 *  which is what makes the two nonlinearities distinguishable here. */
function readout(feat, W) {
  return W.readout[0] * feat[0][0]
       + W.readout[1] * norm(feat[1]) ** 2
       + W.readout[2] * norm(feat[2]) ** 2;
}

function forward(pos, W, mode) {
  const conv = convolve(pos, W);
  const si = selfInteraction(conv, W);
  const nl = mode === 'gated' ? gatedNonlinearity(si, W) : componentwiseNonlinearity(si);
  return { conv, si, nl, energy: readout(nl, W) };
}

export default {
  id: 'layer',
  title: 'One NequIP layer, assembled',
  render(root) {
    // Swept first, so the prose below can quote the measurement instead of a
    // number someone once typed in.
    const sweepRng = mulberry32(24601);
    const sweepW = makeWeights(5);
    const worst = { gated: 0, naive: 0 };
    for (let t = 0; t < 300; t++) {
      const R = randomRotation(sweepRng);
      const rotated = MOL.map((p) => matvec(R, p));
      for (const m of ['gated', 'naive']) {
        const a = forward(MOL, sweepW, m), b = forward(rotated, sweepW, m);
        worst[m] = Math.max(worst[m], Math.abs(b.energy - a.energy) / (1 + Math.abs(a.energy)));
      }
    }
    const ratio = worst.naive / Math.max(worst.gated, 1e-18);
    const decades = Math.round(Math.log10(ratio));

    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 6'),
      h('h1', {}, 'One NequIP layer, assembled'),
      h('p', { class: 'lede' },
        'Everything from chapter 4, wired into an actual interaction block. The interesting part ' +
        'is not that it works — it is which pieces you are forbidden from choosing freely, and ' +
        'what happens the moment you choose one of them the obvious way.'),

      h('h2', {}, 'The block'),
      h('p', { class: 'prose' },
        'A NequIP interaction block takes a feature on each atom, exchanges messages along bonds ' +
        'within a cutoff radius, and returns updated features. The convolution is'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$L_{\\ell_o}^{i} \\;=\\; \\sum_{j \\in \\mathcal{N}(i)} \\; ' +
        'R_{\\ell_o \\ell_f \\ell_i}\\!\\left(\\lVert \\vec{r}_{ij} \\rVert\\right)\\; ' +
        'Y_{\\ell_f}\\!\\left(\\hat{r}_{ij}\\right) \\;\\otimes_{\\mathrm{CG}}\\; V^{j}_{\\ell_i}$$'),
      h('p', { class: 'prose', html:
        'Read it right to left: take neighbour $j$’s feature, couple it to the spherical harmonic ' +
        'of the bond direction with the Clebsch–Gordan product, weight the result by a function of ' +
        'the bond <em>length</em> alone, and sum over neighbours. ' +
        '<strong>Every learnable parameter sits in $R$</strong> — the radial part, which takes a ' +
        'scalar and returns a scalar. The angular part has no parameters, because chapter 4 showed ' +
        'there is nothing to choose: the coupling is unique.' }),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'The design is mostly forced'),
        h('div', { html:
          'Sum over neighbours rather than concatenate — otherwise permuting identical atoms would ' +
          'change the answer. Let $R$ see $\\lVert\\vec r_{ij}\\rVert$ and not $\\vec r_{ij}$ — ' +
          'otherwise the weights themselves would rotate. Mix channels only within a degree — ' +
          'mixing across degrees does not commute with rotation. Use a smooth cutoff envelope — ' +
          'otherwise an atom crossing the cutoff makes the energy jump and the forces meaningless. ' +
          'Nearly the whole architecture is a consequence of the symmetry requirement plus ' +
          'differentiability. That is the sense in which NequIP is less an invention than a ' +
          'derivation.' })),

      h('h2', {}, 'The filter, taken apart'),
      h('p', { class: 'prose', html:
        'The paper states the constraint in one sentence: <em>“the convolution filters ' +
        '$S^{(l)}_{m}(\\vec r_{ij})$ are constrained to be products of learnable radial functions ' +
        'and spherical harmonics, which are equivariant under SO(3).”</em> It is worth slowing ' +
        'down, because that sentence contains the entire reason the architecture works.' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.2em 0' } },
        '$$S^{(l)}_{m}\\!\\left(\\vec r_{ij}\\right) \\;=\\; ' +
        '\\underbrace{R\\!\\left(\\lVert \\vec r_{ij}\\rVert\\right)}_{\\text{learnable, scalar}} ' +
        '\\;\\cdot\\; ' +
        '\\underbrace{Y^{(l)}_{m}\\!\\left(\\hat r_{ij}\\right)}_{\\text{fixed, carries the geometry}}$$'),
      h('p', { class: 'prose', html:
        'A filter is a function of a full three-dimensional displacement $\\vec r_{ij}$, which ' +
        'carries both a length and a direction. The constraint says: <strong>factorise it, and put ' +
        'every free parameter on the length side.</strong> $R$ receives a single number, ' +
        '$\\lVert\\vec r_{ij}\\rVert$, and returns a single number. It is structurally incapable of ' +
        'knowing which way the bond points, so a rotation cannot change what it computes — the ' +
        'rotation is invisible to it. Meanwhile $Y^{(l)}$ sees only the unit vector ' +
        '$\\hat r_{ij}$ and has no parameters at all; its transformation law is fixed by ' +
        'representation theory before any training happens.' }),
      h('p', { class: 'prose', html:
        'So under a rotation $g$ the two factors do completely different things, and that is the ' +
        'point:' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$S^{(l)}\\!\\left(g\\,\\vec r_{ij}\\right) \\;=\\; ' +
        'R\\!\\left(\\lVert g\\,\\vec r_{ij}\\rVert\\right)\\, Y^{(l)}\\!\\left(g\\,\\hat r_{ij}\\right) ' +
        '\\;=\\; R\\!\\left(\\lVert \\vec r_{ij}\\rVert\\right)\\, D^{(l)}(g)\\, Y^{(l)}\\!\\left(\\hat r_{ij}\\right) ' +
        '\\;=\\; D^{(l)}(g)\\, S^{(l)}\\!\\left(\\vec r_{ij}\\right)$$'),
      h('p', { class: 'prose', html:
        'The middle step is where the whole thing turns. A rotation preserves length, so ' +
        '$\\lVert g\\vec r\\rVert = \\lVert\\vec r\\rVert$ and the scalar $R$ comes through ' +
        'untouched — it commutes with $D^{(l)}(g)$ trivially, being a number. The harmonic picks up ' +
        'exactly the Wigner matrix, by the defining property from chapter 4. The filter is ' +
        'therefore equivariant <em>for any $R$ whatsoever</em>. Training cannot break it, because ' +
        'training only moves $R$.' }),
      h('p', { class: 'prose' },
        'The explorer below is that argument made tangible. Drag the radial coefficients as ' +
        'violently as you like — invert them, zero them, make the function oscillate — and watch ' +
        'the equivariance residual at the bottom refuse to move off machine precision. Then turn ' +
        'on the last slider, which lets $R$ peek at direction, and watch it break immediately.'),
    );

    const explorer = h('div', { class: 'demo' });
    explorer.append(
      h('h3', {}, 'The convolution filter, factorised'),
      h('p', { class: 'hint' },
        'Left: the learnable half, a function of bond length alone. Right: how a rotation mixes ' +
        'the 2ℓ+1 components. Below: their product, the filter itself, on a plane through the origin.'),
      filterExplorer());
    root.append(explorer);

    root.append(
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Three things the explorer is showing at once'),
        h('div', { html:
          '<strong>The radial factor is blind to rotation.</strong> The readout prints the change ' +
          'in every bond length under the rotation: it is zero to machine precision, because ' +
          'rotations preserve length. $R$ literally receives identical inputs before and after.<br><br>' +
          '<strong>Rotation mixes the components.</strong> The matrix on the right is $D^{(l)}(g)$, ' +
          'and it is dense. The $m$ panels do not each rotate on their own — they turn into linear ' +
          'combinations of one another. That is what it means for the $2\\ell+1$ numbers to be one ' +
          'geometric object rather than $2\\ell+1$ separate features, and it is why the ' +
          'self-interaction layer is allowed to mix channels within a degree but never across ' +
          'degrees.<br><br>' +
          '<strong>Blank panels are real.</strong> At some slice azimuths a component vanishes ' +
          'identically, because that harmonic is odd in the direction perpendicular to the slice. ' +
          'Sweep the azimuth and it comes back. The information is in the full three-dimensional ' +
          'field, not in any one plane through it.' })),

      h('h2', {}, 'What the constraint costs'),
      h('p', { class: 'prose', html:
        'It is fair to ask what is given up. A general filter $\\mathbb{R}^3 \\to \\mathbb{R}$ can ' +
        'be any function of three numbers. The constrained one is a product of a function of one ' +
        'number with a fixed basis of $2\\ell+1$ angular patterns. That is a genuine restriction — ' +
        'but a much smaller one than it looks, because the network does not use a single filter. It ' +
        'uses one radial function per $(\\ell_o, \\ell_f, \\ell_i)$ path and per channel, and sums ' +
        'over $\\ell_f$ up to $\\ell_{\\max}$. Summing products of radial functions with harmonics ' +
        'across degrees is precisely a spherical-harmonic expansion of an arbitrary filter, so in ' +
        'the limit $\\ell_{\\max}\\to\\infty$ nothing is lost at all. Truncating at finite ' +
        '$\\ell_{\\max}$ is the only real restriction — and chapter 7 is the measurement of what ' +
        'that truncation costs.' }),
    );

    // ---- equation 8, in full ------------------------------------------------
    root.append(
      h('h2', {}, 'Equation 8, index by index'),
      h('p', { class: 'prose', html:
        'The filter of equation 4 is only half a layer. Equation 8 of the paper is the whole ' +
        'convolution — the thing that actually consumes neighbour features and produces new ones — ' +
        'and it is where the notation gets forbidding. It carries six superscripts and five indices, ' +
        'and it is worth taking apart slowly, because every one of them is doing a job.' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.3em 0', overflowX: 'auto' } },
        '$$\\mathcal{L}^{l_o,p_o,l_f,p_f,l_i,p_i}_{a\\,c\\,m_o}\\!\\left(\\vec r_a,\\, ' +
        'V^{l_i,p_i}_{a\\,c\\,m_i}\\right) \\;=\\; ' +
        '\\sum_{m_f,\\,m_i} C^{l_o,m_o}_{l_i,m_i,l_f,m_f} ' +
        '\\sum_{b\\,\\in\\,S} \\bigl(R(r_{ab})_{c,\\,l_o,\\,p_o,\\,l_f,\\,p_f,\\,l_i,\\,p_i}\\bigr)\\, ' +
        'Y^{l_f}_{m_f}\\!\\left(\\hat r_{ab}\\right)\\, V^{l_i,p_i}_{b\\,c\\,m_i}$$'),

      h('h3', {}, 'Read it from the inside out'),
      h('p', { class: 'prose', html:
        'Start at the far right with $V^{(l_i,p_i)}_{b\\,c\\,m_i}$ — the existing feature sitting on ' +
        'neighbour $b$, in channel $c$, component $m_i$, of degree $l_i$ and parity $p_i$. Multiply ' +
        'it by two things evaluated on the bond from $a$ to $b$: the spherical harmonic ' +
        '$Y^{(l_f)}_{m_f}(\\hat r_{ab})$ of the bond direction, and the learnable radial function ' +
        '$R_c(r_{ab})$ of the bond length. That product is exactly the filter from equation 4, now ' +
        'applied to a neighbour’s feature rather than standing alone.' }),
      h('p', { class: 'prose', html:
        'Then $\\sum_{b \\in S}$ gathers over the neighbour set. This is the message-passing step, ' +
        'and being a <em>sum</em> rather than a concatenation is what makes the layer indifferent to ' +
        'the order neighbours happen to be stored in — permutation symmetry, obtained for free.' }),
      h('p', { class: 'prose', html:
        'Finally $\\sum_{m_f, m_i} C^{(l_o,m_o)}_{(l_i,m_i)(l_f,m_f)}$ contracts the filter’s ' +
        'angular index against the feature’s angular index and deposits the result at output ' +
        'component $m_o$. Those are the Clebsch–Gordan coefficients chapter 4 derived, and chapter 4 ' +
        'is also why there is no freedom here: the coupling of degree $l_i$ with degree $l_f$ into ' +
        'degree $l_o$ is unique up to scale, so once the degrees are chosen the numbers are ' +
        'determined. $m_f$ and $m_i$ are summed away; $m_o$ is left free and indexes the output.' }),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'The index that is easiest to miss'),
        h('div', { html:
          'Look at $c$. It appears on the input $V_{b\\,c\\,m_i}$, on the radial function $R_c$, and ' +
          'on the output $\\mathcal{L}_{a\\,c\\,m_o}$ — and it is <strong>never summed over</strong>. ' +
          'The convolution does not mix channels at all. Each channel is carried through with its ' +
          'own private radial function, exactly like a depthwise convolution in an ordinary CNN, and ' +
          'all channel mixing is deferred to the cheap atom-wise linear layer that follows. That is a ' +
          'deliberate cost decision: a channel-mixing tensor product would cost a factor of the ' +
          'channel count more, and the separable form loses very little because the linear layer ' +
          'recovers the mixing immediately afterwards.' })),

      h('h3', {}, 'What the six superscripts are for'),
      h('p', { class: 'prose', html:
        'A layer does not compute one of these. It computes one for every combination of ' +
        '$(l_i, p_i)$ input, $l_f$ filter and $(l_o, p_o)$ output that the selection rules allow, ' +
        'and then — the paper is explicit about this, and it is easy to assume otherwise — ' +
        '<strong>concatenates</strong> them rather than summing them. Each such combination is a ' +
        '<em>path</em>, and each path gets its own learnable radial function, which is what the ' +
        'subscripts on $R(r_{ab})_{c,\\,l_o,p_o,l_f,p_f,l_i,p_i}$ are recording: one function per ' +
        'path <em>and</em> per channel. The parameter count of the layer is essentially the number of ' +
        'paths times the number of channels times the size of the little radial network. In the ' +
        'paper’s words, “there can be multiple $\\mathcal{L}^{l_o,p_o}_{acm_o}$ tensors for a given ' +
        'output rotation order and parity resulting from different combinations of $(l_i, p_i)$ and ' +
        '$(l_f, p_f)$; we take all such possible output tensors with $l_o \\le l_{\\max}$ and ' +
        'concatenate them.”' }),
      h('p', { class: 'prose', html:
        'Which paths survive is fixed by two rules and no taste at all. The triangle rule from ' +
        'chapter 4, $|l_i - l_f| \\le l_o \\le l_i + l_f$, and equation 7’s parity rule, ' +
        '$p_o = p_i\\,p_f$. And $p_f$ is not free either: the filter’s angular part is a degree-$l_f$ ' +
        'spherical harmonic, whose parity is $(-1)^{l_f}$. So the whole path structure follows from ' +
        'choosing $\\ell_{\\max}$ and which parities to carry. Enumerate them:' }),
    );

    const paths = h('div', { class: 'demo' });
    paths.append(
      h('h3', {}, 'Every interaction path equation 8 allows'),
      h('p', { class: 'hint' },
        'Rows are input irreps, columns are filter degrees, cells are the outputs the rules permit. ' +
        'Click any cell to build that path’s coupling and verify it. Notation: 1o is a vector ' +
        '(odd under inversion), 1e a pseudovector.'),
      pathExplorer());
    root.append(paths);

    root.append(
      h('p', { class: 'prose', html:
        'Two things are worth noticing in that table. The path count grows quickly with ' +
        '$\\ell_{\\max}$ — which is the real cost of raising it, and the reason chapter 7’s ablation ' +
        'is not free accuracy. And every cell you click reports $\\sigma_2$ of order one, meaning the ' +
        'coupling for that path is unique: there is exactly one way to combine those two degrees ' +
        'into that output, so the only thing left to learn is <em>how much</em> of it to use. That ' +
        'scalar amount is the radial function, evaluated at the bond length.' }),

      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'Where this came from, and where it went'),
        h('div', { html:
          'Equation 8 is the Tensor Field Network convolution of Thomas et al. (2018), ' +
          '<a href="https://arxiv.org/abs/1802.08219">arXiv:1802.08219</a>, with two additions: ' +
          'parity is tracked explicitly through $p_i, p_f, p_o$, and the radial function is indexed ' +
          'per path and per channel rather than shared. NequIP’s contribution at this equation is ' +
          'less the form than the demonstration of what it buys on interatomic potentials.<br><br>' +
          'It is also the layer’s bottleneck. The double sum over $m_f, m_i$ for every path scales ' +
          'steeply in $\\ell_{\\max}$, which is what made higher degrees expensive and what eSCN ' +
          '(<a href="https://arxiv.org/abs/2302.03655">arXiv:2302.03655</a>) later attacked by ' +
          'rotating each edge into a frame where the harmonic collapses to its $m = 0$ component and ' +
          'most of the Clebsch–Gordan matrix becomes zero. Chapter 11 follows that thread — it is ' +
          'how the transformer branch reached $\\ell = 6$ at practical cost.' })),

      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'Three details in the paper’s own commentary'),
        h('div', { html:
          'Worth carrying, because each answers a question the equation provokes. ' +
          '<strong>The index placement means nothing.</strong> The paper says so outright — “the ' +
          'placement of indices into sub- and superscript does not carry specific meaning” — so do ' +
          'not look for variance and covariance here; it is typography.<br><br>' +
          '<strong>The couplings ignore parity.</strong> “The Clebsch–Gordan coefficients do not ' +
          'depend on the parity of the arguments.” Parity is enforced entirely by the equation-7 ' +
          'selection rule deciding which paths exist; once a path is allowed, the coefficients are ' +
          'the ordinary $SO(3)$ ones.<br><br>' +
          '<strong>There is a normalisation the equation omits.</strong> The output of the sum over ' +
          'neighbours is divided by $\\sqrt{N}$, with $N$ the average number of neighbours. Without ' +
          'it the message magnitude would scale with coordination number, so an atom in a dense ' +
          'environment would produce systematically larger features than one in a sparse environment ' +
          'for no physical reason — and the effect would compound with depth.' })),

      h('h3', {}, 'What breaks if you change any of it'),
      h('p', { class: 'prose', html:
        'The quickest way to see that equation 8 is derived rather than designed is to try removing ' +
        'pieces. Replace $\\sum_b$ with a concatenation and permutation symmetry dies. Let $R$ take ' +
        '$\\vec r_{ab}$ instead of $r_{ab}$ and rotation equivariance dies — that is the slider in ' +
        'the explorer above. Replace $C$ with a learned tensor and equivariance dies unless the ' +
        'learned tensor happens to land on the one-dimensional invariant subspace, which training ' +
        'will not do. Drop the parity index and $1o \\otimes 1o \\to 1o$ becomes reachable, letting ' +
        'the network build a pseudovector where a vector belongs and quietly breaking reflection ' +
        'symmetry. Sum over $c$ and the layer still works but costs a factor of the channel count ' +
        'more for very little gain. Of the whole equation, essentially the only free choices are ' +
        '$\\ell_{\\max}$, the number of channels, and the shape of the little network inside $R$.' }),

      h('h2', {}, 'Watch it run'),
      h('p', { class: 'prose' },
        'Below, one block runs on a five-atom neighbourhood with fixed random weights. Rotate the ' +
        'input and watch each stage. The $\\ell = 0$ row must not move at all. The $\\ell = 1$ and ' +
        '$\\ell = 2$ rows must move, and must move exactly the way the Wigner matrices predict — ' +
        'that residual is printed live.'),
    );

    const demo = h('div', { class: 'demo' });
    demo.append(h('h3', {}, 'One interaction block, stage by stage'),
      h('p', { class: 'hint' },
        'Weights are random and fixed. That is deliberate: equivariance is a property of the ' +
        'architecture and holds before any training at all.'));
    const stageBox = h('div', { class: 'readout' });
    let rotIdx = 0, mode = 'gated', seed = 5;

    function run() {
      const W = makeWeights(seed);
      const Rot = rotIdx === 0
        ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        : randomRotation(mulberry32(1000 + rotIdx));
      const rotated = MOL.map((p) => matvec(Rot, p));

      const base = forward(MOL, W, mode);
      const rot = forward(rotated, W, mode);

      const lines = [];
      const label = ['convolution', 'self-interaction (channel mixing)', 'nonlinearity'];
      ['conv', 'si', 'nl'].forEach((key, li) => {
        lines.push(`<span class="dim">── ${label[li]} ──</span>`);
        for (let l = 0; l <= LMAX; l++) {
          const a = base[key][l], b = rot[key][l];
          const expect = matvec(wignerD(l, Rot), a);
          const err = Math.max(...b.map((v, i) => Math.abs(v - expect[i])));
          const rel = err / Math.max(norm(expect), 1e-12);
          lines.push(`  l=${l}  f(x)    [${a.map((v) => v.toFixed(5).padStart(9)).join(' ')}]`);
          lines.push(`        f(gx)   [${b.map((v) => v.toFixed(5).padStart(9)).join(' ')}]`);
          lines.push(`        D·f(x)  [${expect.map((v) => v.toFixed(5).padStart(9)).join(' ')}]` +
            `   <span class="${rel < 1e-9 ? 'ok' : 'bad'}">rel err ${rel.toExponential(1)}</span>`);
        }
      });
      const dE = Math.abs(rot.energy - base.energy);
      lines.push('<span class="dim">── readout (l=0 channel plus the invariant norms of l=1, l=2) ──</span>');
      lines.push(`  E(x)   ${base.energy.toFixed(12)}`);
      lines.push(`  E(gx)  ${rot.energy.toFixed(12)}`);
      lines.push(`  |ΔE|   <span class="${dE < 1e-9 ? 'ok' : 'bad'}">${dE.toExponential(2)}</span>` +
        '   <span class="dim">the energy must be invariant</span>');
      stageBox.innerHTML = lines.join('\n');
    }

    demo.append(
      h('div', { class: 'controls' },
        slider({ label: 'rotation', min: 0, max: 60, step: 1, value: 0,
          format: (v) => (v === 0 ? 'identity' : `random #${v}`),
          onInput: (v) => { rotIdx = v; run(); } }),
        slider({ label: 'weight seed', min: 1, max: 40, step: 1, value: 5,
          format: String, onInput: (v) => { seed = v; run(); } }),
        segmented({
          label: 'nonlinearity',
          options: [
            { label: 'gated (NequIP)', value: 'gated' },
            { label: 'componentwise tanh', value: 'naive' },
          ],
          value: 'gated',
          onPick: (v) => { mode = v; run(); },
        })),
      stageBox);
    root.append(demo);
    run();

    root.append(
      h('h2', {}, 'The nonlinearity is the trap'),
      h('p', { class: 'prose' },
        'Set the rotation to anything but identity, then flip the switch to “componentwise tanh”. ' +
        'The convolution and self-interaction rows stay at machine precision, because a linear map ' +
        'commutes with a rotation. The nonlinearity row does not, and the energy stops being ' +
        `invariant — by about ${decades} orders of magnitude, measured just below.`),
      h('p', { class: 'prose', html:
        'The reason is worth stating carefully, because it is the most common way to break one of ' +
        'these networks by accident. A degree-$\\ell$ feature with $\\ell \\ge 1$ is not ' +
        '$2\\ell+1$ independent numbers. It is one geometric object whose components are ' +
        'coordinates in a chosen frame. Applying $\\tanh$ to each component applies it ' +
        '<em>in that frame</em>, and $\\tanh(Dv) \\neq D\\tanh(v)$ for a general rotation $D$. The ' +
        'activation has quietly picked a preferred orientation for space.' }),
      h('p', { class: 'prose', html:
        'The fix, due to Weiler and colleagues and adopted by NequIP, is the <em>gated</em> ' +
        'nonlinearity: build a scalar out of the invariant parts, push that scalar through the ' +
        'nonlinearity, and use the result to rescale the higher-degree feature. Scaling commutes ' +
        'with rotation, so the transformation law survives, while the magnitude still becomes a ' +
        'genuinely nonlinear function of the input. Degree-0 features can take any nonlinearity ' +
        'you like — they do not transform, so there is nothing to break.' }));

    const sweep = h('div', { class: 'demo' });
    sweep.append(h('h3', {}, 'Both variants, over 300 random rotations'),
      h('p', { class: 'hint' }, 'Swept when you opened this chapter — not a stored number.'));
    const box = h('div');
    box.append(
      checkLine(worst.gated < 1e-12,
        'gated nonlinearity: max relative |E(gx) − E(x)|', worst.gated.toExponential(2)),
      checkLine(worst.naive > 1e-3,
        'componentwise tanh: same quantity, demonstrably broken', worst.naive.toExponential(2)),
      checkLine(ratio > 1e9, 'ratio between the two', `${ratio.toExponential(1)}×`));
    sweep.append(box);
    root.append(sweep);

    root.append(
      h('h2', {}, 'Stacking blocks, and two things that come free'),
      h('p', { class: 'prose', html:
        'Stack several of these and an atom’s representation reaches further with every layer — ' +
        'chapter 9 measures exactly how far, and what it costs. Two properties come free from the ' +
        'construction. Permutation symmetry, because the message is a <em>sum</em> over ' +
        'neighbours, so relabelling identical atoms changes nothing. Translation symmetry, because ' +
        'only relative displacements $\\vec r_{ij}$ ever enter and the origin is never referenced.' }),
      h('p', { class: 'prose' },
        'That leaves the second equation from the overture — forces as the gradient of the energy ' +
        'rather than a separate output. It costs one backward pass. Chapter 8 shows what happens ' +
        'to models that skip it, and why that turned into a visible column on the leaderboard.'),
    );
  },
};
