import { h, slider, segmented, checkLine } from '../ui.js';
import {
  realSH, wignerD, envelope, besselBasis, randomRotation, mulberry32,
} from '../e3.js';
import { matvec, norm } from '../linalg.js';
import { filterExplorer } from '../filter-explorer.js';

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
      h('p', { class: 'eyebrow geo' }, 'Chapter 5'),
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
        '$\\ell_{\\max}$ is the only real restriction — and chapter 6 is the measurement of what ' +
        'that truncation costs.' }),

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
        'chapter 8 measures exactly how far, and what it costs. Two properties come free from the ' +
        'construction. Permutation symmetry, because the message is a <em>sum</em> over ' +
        'neighbours, so relabelling identical atoms changes nothing. Translation symmetry, because ' +
        'only relative displacements $\\vec r_{ij}$ ever enter and the origin is never referenced.' }),
      h('p', { class: 'prose' },
        'That leaves the second equation from the overture — forces as the gradient of the energy ' +
        'rather than a separate output. It costs one backward pass. Chapter 7 shows what happens ' +
        'to models that skip it, and why that turned into a visible column on the leaderboard.'),
    );
  },
};
