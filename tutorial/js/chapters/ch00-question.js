import { h } from '../ui.js';

export default {
  id: 'question',
  title: 'The question a reading group left open',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow syn' }, 'Overture'),
      h('h1', {}, 'The question a reading group left open'),
      h('p', { class: 'lede' },
        'A group met to work out what machine learning has actually done for materials science. ' +
        'Halfway through, someone read the leaderboard out loud and moved on. The sentence they ' +
        'moved on from is the subject of this tutorial.'),

      h('blockquote', { class: 'prose' },
        '“It’s called Graph Attention Transformer that is now the leading one… which incorporates ' +
        'rotational symmetry. So we’re going to read this overview… we should keep in mind that ' +
        'graph attention transformers are the leading ones right now.”',
        h('span', { class: 'attrib' },
          'Reading group, 23 July 2026 — auto-transcribed, lightly cleaned')),

      h('p', { class: 'prose' },
        'Two claims are packed in there. One is a fact about a leaderboard. The other — ' +
        '“incorporates rotational symmetry” — is a claim about architecture that nobody in the ' +
        'room unpacked, and that turns out to be the single most consequential design decision ' +
        'in the field. Where did it come from, why does it work, and is it still what is doing ' +
        'the work at the top of the table today?'),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'What this tutorial is'),
        h('div', { html:
          'A clickable companion to <a href="https://www.nature.com/articles/s41467-022-29939-5">' +
          'Batzner et al. (2022), <em>E(3)-equivariant graph neural networks for data-efficient and ' +
          'accurate interatomic potentials</em></a> — the NequIP paper — and to the lineage that ' +
          'runs from it to the current ' +
          '<a href="https://matbench-discovery.materialsproject.org/">Matbench Discovery</a> ' +
          'leaderboard. Eleven chapters, each with a toy you can break.' })),

      h('h2', {}, 'The setup, in one paragraph'),
      h('p', { class: 'prose' },
        'Quantum mechanics gives you, in principle, the energy of any arrangement of atoms. ' +
        'In practice you get it from density functional theory, which costs roughly cubically in ' +
        'the number of electrons and has to be redone at every step of a simulation. That is the ' +
        'bottleneck. A machine-learned interatomic potential replaces the expensive calculation ' +
        'with a function you fit once and evaluate a billion times:'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.2em 0' } },
        '$$E_\\theta(\\mathbf{R}, \\mathbf{Z}) \\;=\\; \\sum_i \\varepsilon_i(\\mathbf{R}, \\mathbf{Z}),' +
        '\\qquad \\mathbf{F}_i \\;=\\; -\\nabla_{\\mathbf{r}_i} E_\\theta.$$'),
      h('p', { class: 'prose' },
        'The first equation says the total energy is a sum of local atomic contributions — an idea ' +
        'from Behler and Parrinello in 2007 that every model in this story still uses. The second ' +
        'says forces are the gradient of that energy rather than a separate prediction. Chapter 7 ' +
        'is about what goes wrong when you ignore the second equation, and it is the reason two ' +
        'models with identical force errors can behave completely differently in a simulation. ' +
        'Chapter 1 unpacks the modelling problem properly: what the training labels actually ' +
        'are, why the obvious approach of just running the physics is unavailable, and which ' +
        'constraints the problem forces on any solution.'),

      h('h2', {}, 'Where the tutorial goes'),
      h('div', { class: 'grid3' },
        h('div', { class: 'card geo' }, h('h4', {}, 'Chapters 1–5'),
          h('p', {}, 'The problem and the geometry. What is actually being modelled and why a ' +
            'surrogate is needed at all; what symmetry demands of an energy model; what invariant ' +
            'descriptors throw away; and the machinery — spherical harmonics, Wigner matrices, ' +
            'Clebsch–Gordan couplings — assembled into one NequIP layer. Derived in your browser, ' +
            'not tabulated.')),
        h('div', { class: 'card dat' }, h('h4', {}, 'Chapters 6–9'),
          h('p', {}, 'The consequences, each a toy experiment with a Python twin in this ' +
            'repository: the $\\ell_{\\max}$ ablation, conservative versus direct forces, how far ' +
            'locality reaches, and why crystal stability is a decision problem rather than a ' +
            'regression problem.')),
        h('div', { class: 'card syn' }, h('h4', {}, 'Chapter 10'),
          h('p', {}, 'The genealogy: an interactive family tree from Behler–Parrinello through ' +
            'Tensor Field Networks and NequIP to the models sitting at the top of Matbench ' +
            'Discovery right now, with what each one actually contributed.'))),

      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'A promise about the numbers'),
        h('div', { html:
          'Nothing here is a number I typed in from a paper and asked you to trust. Figures are ' +
          'either computed live in this page or read from JSON written by a script in ' +
          '<code>python/experiments/</code>, each of which ends by printing its own pass/fail ' +
          'ledger. The browser even ' +
          '<a href="selftest.html">re-derives and checks its own representation theory</a> against ' +
          'values exported from scipy. Where a toy fails to reproduce the paper’s finding, the ' +
          'chapter says so.' })),
    );
  },
};
