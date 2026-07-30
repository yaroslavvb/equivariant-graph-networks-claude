import { h, slider, segmented, fmt, PALETTE } from '../ui.js';

// Order-of-magnitude reference points, all user-adjustable in the demo below.
// These are deliberately round numbers: the argument is about exponents, not
// about any particular machine or code.
const DEFAULTS = {
  nAtoms: 200,
  dftSecondsAt100: 60,     // wall-clock for one self-consistent force evaluation, 100 atoms
  mlipMicrosAt100: 300,    // wall-clock for one MLIP force evaluation, 100 atoms
  timestepFs: 1,
  nanoseconds: 1,
};

const YEAR_S = 365.25 * 24 * 3600;

function humanTime(seconds) {
  if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(0)} µs`;
  if (seconds < 1) return `${(seconds * 1e3).toFixed(0)} ms`;
  if (seconds < 120) return `${seconds.toFixed(1)} s`;
  if (seconds < 7200) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 2 * 86400) return `${(seconds / 3600).toFixed(1)} hours`;
  if (seconds < 2 * YEAR_S) return `${(seconds / 86400).toFixed(1)} days`;
  const y = seconds / YEAR_S;
  if (y < 1e6) return `${y.toLocaleString(undefined, { maximumFractionDigits: 0 })} years`;
  return `${y.toExponential(1)} years`;
}

export default {
  id: 'problem',
  title: 'The modelling problem',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 1'),
      h('h1', {}, 'The modelling problem'),
      h('p', { class: 'lede' },
        'Before any of the architecture makes sense, it is worth being precise about what is being ' +
        'modelled, what the training labels actually are, and why the obvious approach — just run ' +
        'the physics — is not available. The constraints that follow are not stylistic. They are ' +
        'forced by the problem.'),

      h('h2', {}, 'The object being learned'),
      h('p', { class: 'prose', html:
        'Atoms are nuclei plus electrons. The electrons are thousands of times lighter and move ' +
        'thousands of times faster, so to an excellent approximation they relax instantaneously ' +
        'into their ground state around whatever arrangement the nuclei currently have. That is the ' +
        '<strong>Born–Oppenheimer approximation</strong>, and it buys an enormous simplification: ' +
        'the electrons stop being dynamical variables and become a <em>function</em> of the nuclear ' +
        'positions. What they leave behind is a single scalar field over nuclear configuration ' +
        'space, the <strong>potential energy surface</strong>:' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$E \\;:\\; \\bigl(\\vec r_1, \\dots, \\vec r_N;\\; Z_1, \\dots, Z_N\\bigr) \\;\\longmapsto\\; \\mathbb{R}$$'),
      h('p', { class: 'prose', html:
        'with $\\vec r_i \\in \\mathbb{R}^3$ the positions and $Z_i$ the atomic numbers. Everything ' +
        'chemistry and materials science cares about is a property of this one surface. Its minima ' +
        'are stable structures. Its gradients are forces, so its shape determines every trajectory. ' +
        'Its second derivatives are the force constants that set vibrational frequencies and ' +
        'phonons — which is why chapter 7 turns out to matter. Its saddle points are transition ' +
        'states and therefore reaction rates. Relative depths of minima decide which phase is ' +
        'stable, which is the subject of chapter 9.' }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'The modelling problem, in one line'),
        h('div', { html:
          'Learn a function $E_\\theta$ that approximates the Born–Oppenheimer potential energy ' +
          'surface well enough, and cheaply enough, that you can afford to evaluate it the billions ' +
          'of times a simulation needs — and whose <em>derivatives</em> are good, not just its ' +
          'values.' })),

      h('h2', {}, 'Where the labels come from, and what they are not'),
      h('p', { class: 'prose', html:
        'The surface is defined by the electronic Schrödinger equation, which is not solvable at ' +
        'useful scale. In practice the labels come from <strong>density functional theory</strong>. ' +
        'Kohn and Sham’s reformulation replaces the interacting many-electron problem with a ' +
        'self-consistent system of single-particle equations,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.05em 0' } },
        '$$\\Bigl[-\\tfrac{1}{2}\\nabla^2 + v_{\\text{ext}} + v_H[n] + v_{\\text{xc}}[n]\\Bigr]\\phi_i ' +
        '= \\epsilon_i \\phi_i, \\qquad n(\\vec r) = \\sum_i |\\phi_i(\\vec r)|^2,$$'),
      h('p', { class: 'prose', html:
        'which is nonlinear in a specific way: the operator depends on the density, and the density ' +
        'is built from the operator’s own eigenfunctions. You solve, rebuild, mix, and iterate to a ' +
        'fixed point. Cost scales roughly as the cube of the number of electrons, and the exchange–' +
        'correlation functional $v_{\\text{xc}}$ is not known exactly — so even a perfectly ' +
        'converged calculation contains a modelling approximation of its own.' }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'This is a surrogate, not a solver'),
        h('div', { html:
          'A machine-learned interatomic potential is not doing quantum mechanics. It is regressing ' +
          'the output of one particular functional at one particular level of convergence. If that ' +
          'functional is wrong for your system — and for strongly correlated cases such as the ' +
          'chromium dimer it is qualitatively wrong — then a model that reproduces it perfectly is ' +
          'perfectly reproducing an error. Every leaderboard number in this repository is agreement ' +
          'with PBE, not agreement with nature. Keeping those two apart is the single most ' +
          'important habit when reading this literature.' })),

      h('h2', {}, 'Why a surrogate is needed at all'),
      h('p', { class: 'prose' },
        'The gap is not a factor of two. Molecular dynamics advances in steps of about a ' +
        'femtosecond, because that is what it takes to resolve a bond vibration. A nanosecond of ' +
        'simulated time is therefore a million force evaluations, and a nanosecond is short — ' +
        'diffusion, nucleation and phase transitions live in microseconds and beyond. Set the ' +
        'numbers yourself below.'),
    );

    // ---- the cost gap ------------------------------------------------------
    const demo = h('div', { class: 'demo' });
    demo.append(h('h3', {}, 'The gap you are trying to close'),
      h('p', { class: 'hint' },
        'Order-of-magnitude arithmetic, not a benchmark. The per-evaluation costs are yours to set; ' +
        'the defaults are deliberately generous to DFT. What matters is the exponent, not the constant.'));
    const out = h('div', { class: 'readout' });
    const st = { ...DEFAULTS };

    function recompute() {
      // DFT scales ~cubically in system size; an MLIP is linear.
      const dft = st.dftSecondsAt100 * Math.pow(st.nAtoms / 100, 3);
      const mlip = (st.mlipMicrosAt100 * 1e-6) * (st.nAtoms / 100);
      const steps = (st.nanoseconds * 1e6) / st.timestepFs;
      const tDft = dft * steps;
      const tMlip = mlip * steps;
      const ratio = tDft / tMlip;
      out.innerHTML =
        `system                       ${st.nAtoms} atoms\n` +
        `trajectory                   ${st.nanoseconds} ns at ${st.timestepFs} fs  ` +
        `= ${steps.toExponential(2)} force evaluations\n` +
        `\n` +
        `one DFT evaluation           ${humanTime(dft)}      ` +
        `<span class="dim">(cubic scaling from ${st.dftSecondsAt100} s at 100 atoms)</span>\n` +
        `one surrogate evaluation     ${humanTime(mlip)}      ` +
        `<span class="dim">(linear scaling from ${st.mlipMicrosAt100} µs at 100 atoms)</span>\n` +
        `\n` +
        `whole trajectory by DFT      <span class="bad">${humanTime(tDft)}</span>\n` +
        `whole trajectory by surrogate <span class="ok">${humanTime(tMlip)}</span>\n` +
        `\n` +
        `speed-up                     ${ratio.toExponential(1)}×   ` +
        `<span class="dim">this ratio is the entire reason the field exists</span>`;
    }

    demo.append(
      h('div', { class: 'controls' },
        slider({ label: 'atoms', min: 20, max: 2000, step: 10, value: DEFAULTS.nAtoms,
          format: String, onInput: (v) => { st.nAtoms = v; recompute(); } }),
        slider({ label: 'simulated time', min: 1, max: 1000, step: 1, value: DEFAULTS.nanoseconds,
          format: (v) => `${v} ns`, onInput: (v) => { st.nanoseconds = v; recompute(); } })),
      h('div', { class: 'controls' },
        slider({ label: 'DFT s / eval at 100 atoms', min: 1, max: 600, step: 1,
          value: DEFAULTS.dftSecondsAt100, format: (v) => `${v} s`,
          onInput: (v) => { st.dftSecondsAt100 = v; recompute(); } }),
        slider({ label: 'surrogate µs / eval at 100 atoms', min: 10, max: 3000, step: 10,
          value: DEFAULTS.mlipMicrosAt100, format: (v) => `${v} µs`,
          onInput: (v) => { st.mlipMicrosAt100 = v; recompute(); } })),
      out);
    root.append(demo);
    recompute();

    root.append(
      h('p', { class: 'prose' },
        'Push the atom count up and the cubic term does the rest. This is the whole motivation, and ' +
        'it is why the reading group’s sharpest remark — that a learned surrogate is only worth it ' +
        'where the incumbent solver leaves a gap — lands exactly here. Linear systems have fast ' +
        'classical solvers and no gap. This has a gap of many orders of magnitude.'),

      h('h2', {}, 'What makes it an unusual regression problem'),
      h('p', { class: 'prose' },
        'It would be easy to read all this as ordinary supervised learning with an expensive ' +
        'labelling function. It is not, and five specific features of the problem drive everything ' +
        'that follows in this tutorial.'),

      h('div', { class: 'grid2' },
        h('div', { class: 'card geo' }, h('h4', {}, '1. The input has no fixed shape'),
          h('p', { html:
            'The domain is not $\\mathbb{R}^d$ for a fixed $d$. It is the set of point clouds of ' +
            '<em>any</em> size, with labelled species, sometimes periodic. A model trained on ' +
            '64-atom cells must run on a million atoms. That rules out flattening coordinates into ' +
            'a fixed vector, and is why the field settled on graphs.' })),
        h('div', { class: 'card geo' }, h('h4', {}, '2. The target has exact symmetries'),
          h('p', { html:
            'Translate, rotate or reflect a configuration and the energy is unchanged; permute ' +
            'identical atoms and it is unchanged. These are not statistical regularities to be ' +
            'discovered — they are exact, known in advance, and free. Chapter 2 makes the ' +
            'distinction precise; chapter 3 measures what a model throws away by enforcing them ' +
            'the crude way.' })),
        h('div', { class: 'card dat' }, h('h4', {}, '3. Extensivity is a hard requirement'),
          h('p', { html:
            'Energy is <em>extensive</em>: double the system and the energy roughly doubles. The ' +
            'decomposition $E = \\sum_i \\varepsilon_i$ into atomic contributions builds this in, ' +
            'and it is what makes size transfer possible at all. It also smuggles in an assumption ' +
            '— that an atom’s contribution depends only on its neighbourhood — which chapter 8 ' +
            'examines and finds is not always true.' })),
        h('div', { class: 'card dat' }, h('h4', {}, '4. The derivatives are the product'),
          h('p', { html:
            'Almost nothing uses $E$ directly. Dynamics uses $-\\nabla E$; phonons and thermal ' +
            'conductivity use second derivatives; elastic response uses the stress. So the model ' +
            'must be smooth and differentiable, and it is judged on quantities it was never ' +
            'directly trained on. Chapter 7 is what happens when that is forgotten.' })),
        h('div', { class: 'card syn' }, h('h4', {}, '5. Labels are the binding constraint'),
          h('p', { html:
            'Every training point costs a DFT calculation. Sample efficiency is therefore not a ' +
            'nicety, it is the currency of the field — and it is precisely the axis on which NequIP ' +
            'made its claim. Chapter 6 is the measurement.' })),
        h('div', { class: 'card syn' }, h('h4', {}, '6. Errors compound'),
          h('p', { html:
            'A held-out force error is a one-step quantity. A simulation applies the model millions ' +
            'of times in sequence, visiting configurations shaped by its own previous mistakes. ' +
            'Small biases become wrong diffusion constants and spurious phases. This is why ' +
            'benchmarks moved from regression error toward downstream observables.' }))),

      h('h2', {}, 'The learning problem, stated'),
      h('p', { class: 'prose' },
        'Putting it together, what is actually fitted is'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$E_\\theta(\\mathbf{R}, \\mathbf{Z}) = \\sum_i \\varepsilon_\\theta\\bigl(\\mathcal{N}_i\\bigr), ' +
        '\\qquad \\vec F_i = -\\nabla_{\\vec r_i} E_\\theta, \\qquad ' +
        '\\sigma = \\tfrac{1}{V}\\,\\partial E_\\theta / \\partial \\varepsilon_{\\text{strain}}$$'),
      h('p', { class: 'prose', html:
        'where $\\mathcal{N}_i$ is the neighbourhood of atom $i$ within a cutoff, trained against a ' +
        'weighted loss over energies, forces and stresses,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.05em 0' } },
        '$$\\mathcal{L} = \\lambda_E \\bigl\\lVert E_\\theta - E^{\\text{DFT}} \\bigr\\rVert^2 ' +
        '+ \\lambda_F \\sum_i \\bigl\\lVert \\vec F_{i,\\theta} - \\vec F_i^{\\text{DFT}} \\bigr\\rVert^2 ' +
        '+ \\lambda_\\sigma \\bigl\\lVert \\sigma_\\theta - \\sigma^{\\text{DFT}} \\bigr\\rVert^2,$$'),
      h('p', { class: 'prose', html:
        'subject to constraints that are <em>not</em> in the loss and never appear as penalty ' +
        'terms: $E_\\theta$ invariant under $E(3)$ and under permutation of like atoms, ' +
        '$\\vec F_\\theta$ equivariant, the whole thing smooth enough to differentiate twice, and ' +
        'extensive. Those are architectural. The rest of this tutorial is about how each one gets ' +
        'built in, and what it costs when one is left out.' }),

      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'The accuracy target, and why it is slippery'),
        h('div', { html:
          'The usual target is “chemical accuracy”, about 1 kcal/mol ≈ 43 meV, with force errors in ' +
          'the tens of meV/Å. But the honest target is not a number on a held-out split — it is ' +
          'whether the derived observable comes out right. NequIP’s lithium-transport result is the ' +
          'good example: a diffusivity within 9% of the reference is a far stronger claim than any ' +
          'force MAE, because diffusion is a collective, trajectory-level quantity that a model ' +
          'cannot fake by being locally smooth. Chapter 9 shows the same lesson on the benchmark ' +
          'side, where a model can have an excellent regression error and still make bad decisions.' })),

      h('p', { class: 'prose' },
        'With the problem stated, the first question is which of those constraints is free and which ' +
        'must be paid for. Symmetry is the one that turns out to be free — if you build it in — and ' +
        'ruinously expensive if you do not. That is the next chapter.'),
    );
  },
};
