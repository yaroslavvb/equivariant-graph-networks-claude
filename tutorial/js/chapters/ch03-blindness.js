import { h, segmented, checkLine, loadResults, Plot, PALETTE } from '../ui.js';

function ringFigure(setA, setB, nRing, highlight) {
  const W = 620, H = 300, R = 96;
  const el = (t, a) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', t);
    for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v));
    return e;
  };
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' });

  [[setA, W * 0.27, `A = {${setA.join(', ')}}`, PALETTE[0]],
   [setB, W * 0.73, `B = {${setB.join(', ')}}`, PALETTE[1]]].forEach(([set, cx, label, color]) => {
    const cy = H / 2 - 8;
    svg.appendChild(el('circle', { cx, cy, r: R, fill: 'none', stroke: '#DDE3E8', 'stroke-width': 1.5 }));
    for (let k = 0; k < nRing; k++) {
      const a = (2 * Math.PI * k) / nRing;
      svg.appendChild(el('circle', {
        cx: cx + R * Math.cos(a), cy: cy - R * Math.sin(a), r: 2.5, fill: '#C9CFD6' }));
    }
    for (let i = 0; i < set.length; i++) {
      for (let j = i + 1; j < set.length; j++) {
        const ai = (2 * Math.PI * set[i]) / nRing, aj = (2 * Math.PI * set[j]) / nRing;
        const raw = Math.abs(set[i] - set[j]);
        const gap = Math.min(raw, nRing - raw);
        const on = highlight === 0 || highlight === gap;
        svg.appendChild(el('line', {
          x1: cx + R * Math.cos(ai), y1: cy - R * Math.sin(ai),
          x2: cx + R * Math.cos(aj), y2: cy - R * Math.sin(aj),
          stroke: on ? color : '#E8ECF0', 'stroke-width': on ? 2 : 1, opacity: on ? 0.8 : 0.45 }));
      }
    }
    for (const k of set) {
      const a = (2 * Math.PI * k) / nRing;
      svg.appendChild(el('line', {
        x1: cx, y1: cy, x2: cx + R * Math.cos(a), y2: cy - R * Math.sin(a),
        stroke: '#B0B7BF', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
      svg.appendChild(el('circle', {
        cx: cx + R * Math.cos(a), cy: cy - R * Math.sin(a), r: 8, fill: color }));
    }
    svg.appendChild(el('circle', { cx, cy, r: 5.5, fill: '#1B2733' }));
    const t = el('text', { x: cx, y: H - 8, 'text-anchor': 'middle', 'font-size': 13,
      fill: '#1B2733', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    t.textContent = label;
    svg.appendChild(t);
  });
  return svg;
}

export default {
  id: 'blindness',
  title: 'What invariant descriptors throw away',
  async render(root) {
    const D = await loadResults('descriptor_blindness');
    const A = D.part_a_two_body_blindness;
    const B = D.part_b_three_body_blindness;
    const F = D.features;
    const RQ = D.rotation_quotient;
    const BS = D.bispectrum;
    const best = BS.entries.reduce((m, e) => (e.abs_diff > m.abs_diff ? e : m), BS.entries[0]);

    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 3'),
      h('h1', {}, 'What invariant descriptors throw away'),
      h('p', { class: 'lede' },
        'Before NequIP, the standard move was to compress an atom’s neighbourhood into ' +
        'rotation-invariant numbers and regress on those. This chapter builds two atomic ' +
        'environments that are provably different, shows exactly which descriptors cannot tell ' +
        'them apart, and finds the point at which the equivariant construction can.'),

      h('h2', {}, 'Level 1: radial descriptors are blind to angle'),
      h('p', { class: 'prose' },
        'Take a central atom with three neighbours, all at distance exactly 1. Arrange them as an ' +
        'equilateral triangle, then slide them into a cluster. Every centre-to-neighbour distance ' +
        'is unchanged, so any descriptor built only from those distances — the radial part of a ' +
        'Behler–Parrinello symmetry function, say — returns an identical vector. Not approximately: ' +
        'bitwise.'),

      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'Quantity'), h('th', { class: 'num' }, 'Equilateral'),
          h('th', { class: 'num' }, 'Clustered'), h('th', {}, 'Difference'))),
        h('tbody', {},
          h('tr', {},
            h('td', {}, 'sorted centre–neighbour distances'),
            h('td', { class: 'num' }, A.sorted_radii_equilateral.map((v) => v.toFixed(3)).join(', ')),
            h('td', { class: 'num' }, A.sorted_radii_clustered.map((v) => v.toFixed(3)).join(', ')),
            h('td', {}, h('span', { class: 'pill' }, 'bitwise identical'))),
          h('tr', {},
            h('td', {}, `${A.g2_mu_grid.length}-channel Gaussian two-body descriptor`),
            h('td', { class: 'num' }, '—'), h('td', { class: 'num' }, '—'),
            h('td', {}, h('span', { class: 'pill' },
              `max |Δ| = ${A.g2_max_abs_diff.toExponential(0)}`))),
          h('tr', { class: 'hi' },
            h('td', {}, 'the Stillinger–Weber three-body angular term'),
            h('td', { class: 'num' }, A.sw_energy_equilateral.toFixed(4)),
            h('td', { class: 'num' }, A.sw_energy_clustered.toFixed(4)),
            h('td', {}, h('span', { class: 'pill bad' },
              `${A.sw_energy_ratio.toFixed(1)}× apart`))))),

      h('p', { class: 'prose' },
        `A radial-only model is being asked to produce two numbers a factor of ` +
        `${A.sw_energy_ratio.toFixed(1)} apart from one identical input. This is why angular ` +
        `information entered the field early — Behler–Parrinello’s angular symmetry functions, and ` +
        `later DimeNet’s explicit three-body messages.`),

      h('h2', {}, 'Level 2: three-body descriptors have their own blind spot'),
      h('p', { class: 'prose', html:
        'The real question is whether adding angles fixes the problem. It does not, and the ' +
        'counterexample is exact rather than numerical. It comes from a piece of combinatorics that ' +
        'music theorists know as the <em>all-interval tetrachords</em>.' }),
      h('p', { class: 'prose', html:
        `On the cyclic group $\\mathbb{Z}_{${B.n_ring}}$, the sets ` +
        `$A = \\{${B.z12_set_A.join(', ')}\\}$ and $B = \\{${B.z12_set_B.join(', ')}\\}$ have the ` +
        `same multiset of pairwise cyclic differences — both have interval vector ` +
        `$\\langle ${B.interval_vector_A.join(',')} \\rangle$, exactly one of every interval — and ` +
        `yet they are <strong>not</strong> related by any rotation or reflection of the circle. ` +
        `Put four neighbours on the unit circle at those positions and you have two atomic ` +
        `environments that are genuinely different but agree on every distance in sight.` }));

    const ringDemo = h('div', { class: 'demo' });
    ringDemo.append(h('h3', {}, 'The homometric pair'),
      h('p', { class: 'hint' },
        'Chords are neighbour–neighbour distances. Step through the interval classes: each occurs ' +
        'exactly once in both configurations, which is what makes the distance multisets agree.'));
    const ringHolder = h('div');
    let hl = 0;
    const redrawRing = () => {
      ringHolder.innerHTML = '';
      ringHolder.appendChild(ringFigure(B.z12_set_A, B.z12_set_B, B.n_ring, hl));
    };
    ringDemo.append(ringHolder, h('div', { class: 'controls' },
      segmented({
        label: 'highlight interval',
        options: [{ label: 'all', value: 0 },
          ...[1, 2, 3, 4, 5, 6].map((k) => ({ label: String(k), value: k }))],
        value: 0,
        onPick: (v) => { hl = v; redrawRing(); },
      })));
    root.append(ringDemo);
    redrawRing();

    root.append(
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'What a descriptor could look at'),
          h('th', { class: 'num' }, 'Agreement between A and B'), h('th', {}, 'Verdict'))),
        h('tbody', {},
          h('tr', {}, h('td', {}, 'centre–neighbour distances'),
            h('td', { class: 'num' }, 'exact'),
            h('td', {}, h('span', { class: 'pill bad' }, 'blind'))),
          h('tr', {}, h('td', {}, 'all neighbour–neighbour distances (homometry)'),
            h('td', { class: 'num' }, `max |Δ| = ${B.homometry_max_abs_diff.toExponential(1)}`),
            h('td', {}, h('span', { class: 'pill bad' }, 'blind'))),
          h('tr', {}, h('td', {}, 'all bond-angle cosines — every three-body term'),
            h('td', { class: 'num' }, `max |Δ| = ${B.bond_cosine_max_abs_diff.toExponential(1)}`),
            h('td', {}, h('span', { class: 'pill bad' }, 'blind'))),
          h('tr', { class: 'hi' }, h('td', {}, 'the configurations themselves'),
            h('td', { class: 'num' }, `min RMSD = ${B.rmsd_min_scan.toFixed(6)}`),
            h('td', {}, h('span', { class: 'pill ok' }, 'genuinely different'))))),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'That last row is a proof, not a search that gave up'),
        h('div', { html:
          `The minimum was taken over all $4! = 24$ ways of matching neighbours, a scan of 3601 ` +
          `in-plane rotations, and the reflection — then confirmed by a full $O(3)$ Kabsch ` +
          `alignment, which rules out escaping through the third dimension. All three agree to ` +
          `$10^{-16}$, and the value is the closed form $(\\sqrt{3}-1)/2 = ` +
          `${B.rmsd_analytic_sqrt3_minus_1_over_2.toFixed(12)}$. The two environments sit a fixed, ` +
          `exactly known distance apart, and no amount of three-body description can see it.` })),

      h('h2', {}, 'So what does see it?'),
      h('p', { class: 'prose' },
        'Now build the features NequIP would build: for each degree $\\ell$, the atom-centred sum ' +
        '$f_\\ell = \\sum_j Y_\\ell(\\hat{r}_{ij})$ over neighbours. Two questions have to be kept ' +
        'apart, and keeping them apart is the point of the chapter.'),
      h('p', { class: 'prose', html:
        'Asking whether the raw features differ is not quite right, because the two configurations ' +
        'might simply be rotated relative to one another. The honest question is whether they still ' +
        'differ <em>after the best possible alignment</em> — whether ' +
        '$\\min_R \\lVert D^\\ell(R) f_\\ell^A - f_\\ell^B \\rVert$ is zero.' }),
      h('p', { class: 'prose', html:
        'The second question is what the <em>invariants</em> see. The natural invariant at degree ' +
        '$\\ell$ is the power spectrum $p_\\ell = \\lVert f_\\ell \\rVert^2$, which is essentially ' +
        'what SOAP and the descriptor tradition compute. Here are both answers at once.' }));

    const sepDemo = h('div', { class: 'demo' });
    sepDemo.append(h('h3', {}, 'Invariants versus equivariants, degree by degree'));
    const plot = new Plot({ width: 660, height: 320, yLog: true,
      xLabel: 'maximum degree ℓ retained', yLabel: 'separation between A and B' });
    plot.add({ points: RQ.search_min_delta_by_Lmax.map((v, i) => [i, Math.max(v, 1e-17)]),
      color: PALETTE[0], width: 2.4, markers: true,
      label: 'equivariant features, best-aligned:  minᴿ ‖D(R)f^A − f^B‖' });
    plot.add({ points: F.power_spectrum_abs_diff.map((v, i) => [i, Math.max(v, 1e-17)]),
      color: PALETTE[1], width: 2.4, markers: true, dash: '5 4',
      label: 'invariant power spectrum:  |p_ℓ^A − p_ℓ^B|' });
    plot.setLimits([-0.3, 6.3], [1e-17, 1e1]);
    sepDemo.append(plot.render(), plot.legend(),
      h('div', { class: 'readout' },
        `invariant power spectrum, worst disagreement over all l = 0..6:  ` +
        `${F.power_spectrum_max_abs_diff.toExponential(2)}\n` +
        `  → indistinguishable at every single degree, to machine precision\n\n` +
        `equivariant features, minimised over all rotations:\n` +
        `  l <= 1   ${RQ.search_min_delta_by_Lmax[1].toExponential(2)}` +
        `   an explicit R* aligns them exactly (residual ${RQ.l1_residual_at_R_star.toExponential(2)})\n` +
        `  l <= 2   ${RQ.search_min_delta_by_Lmax[2].toFixed(6)}` +
        `   certified lower bound ${RQ.certified_lower_bound_L2.toFixed(6)} > 0\n` +
        `           that same R* leaves l = 2 mismatched by ${RQ.l2_residual_at_R_star.toFixed(6)}`));
    root.append(sepDemo);

    root.append(
      h('p', { class: 'prose', html:
        `Read the blue curve first. At $\\ell \\le 1$ it is zero. The plotted ` +
        `${RQ.search_min_delta_by_Lmax[1].toExponential(1)} is the resolution of the rotation ` +
        `search, not a real gap — an explicitly constructed $R^{*}$ aligns the degree-1 features to ` +
        `${RQ.l1_residual_at_R_star.toExponential(1)}, which is machine zero. Vectors alone are ` +
        `genuinely not enough. At ` +
        '$\\ell \\le 2$ it jumps to a positive value, and the positivity is <em>certified</em>: a ' +
        `Lipschitz bound gives ${RQ.certified_lower_bound_L2.toFixed(4)}, so no rotation anywhere ` +
        'in $SO(3)$ can bring them together. Degree 2 is the first degree that separates this pair, ' +
        'and that is a statement about the group, not about a search budget.' }),
      h('p', { class: 'prose', html:
        'Now read the red curve, which is the chapter’s reason for existing. The invariant power ' +
        'spectrum sits at machine zero for <em>every</em> degree up to 6. The information that ' +
        'separates A from B is present in the equivariant features and absent from their norms. It ' +
        'lives in the relative orientation between different degrees — exactly the thing that ' +
        'taking a norm destroys.' }),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'NequIP’s argument, in one measurement'),
        h('div', { html:
          'A scalar-only network computes invariants at every layer, and each time it does it ' +
          'discards orientation and keeps magnitude. An equivariant network carries orientation ' +
          'forward and collapses to a scalar exactly once, at the readout. The gap between those ' +
          'two curves is what that buys — here, the difference between literally zero information ' +
          'and enough to separate two structures.' })),

      h('h2', {}, 'The route the descriptor tradition took instead'),
      h('p', { class: 'prose', html:
        `The power spectrum is not the only invariant available. Contracting <em>three</em> ` +
        `features through a Clebsch–Gordan coupling gives the <em>bispectrum</em>, and that does ` +
        `separate the pair: the largest disagreement is ${BS.max_abs_diff_even_parity.toFixed(4)} ` +
        `at $(\\ell_1,\\ell_2,\\ell_3) = (${best.l1},${best.l2},${best.l3})$. This is what ` +
        `higher-body-order descriptors do, and it is the road that runs through SOAP and the ` +
        `Atomic Cluster Expansion and arrives, years later, at MACE. It works. The price is that ` +
        `you must choose in advance which contractions to compute, and their number grows fast.` }),
      h('p', { class: 'prose' },
        'NequIP’s alternative is to not choose. Keep the equivariant features, let the network ' +
        'learn which couplings matter, and stack layers so that products of products become ' +
        'reachable. The Multi-ACE analysis of 2022 later showed these are two views of one design ' +
        'space — chapter 10 picks that thread up, and it matters more than it sounds, because ' +
        'several models at the top of today’s leaderboard came down the ACE road rather than this one.'),
      h('p', { class: 'prose', html:
        `Two honest caveats about this example. The parity-odd bispectrum entries all vanish for ` +
        `both configurations (largest value ${BS.max_abs_value_odd_parity.toExponential(1)}), ` +
        `because both are planar and therefore achiral. But that is weaker evidence than the row ` +
        `count suggests: of the ${BS.entries.filter((e) => e.parity < 0).length} parity-odd triples ` +
        `tabulated, most are identically zero for <em>any</em> geometry, because the coupling ` +
        `tensor is antisymmetric under exchanging two equal degrees and the bispectrum contracts ` +
        `the same feature into both slots. Only one genuinely distinct pseudo-scalar is being ` +
        `tested here. The conclusion — planar implies achiral implies vanishing pseudo-scalars — ` +
        `is right; the evidence is one measurement, not fifteen.` }),
      h('p', { class: 'prose', html:
        `Second, the even-parity table is also over-counted as printed: ` +
        `$C^{\\ell_1\\ell_2\\ell_3}$ is the same invariant under permutation of its indices up to ` +
        `sign, so the ${BS.entries.filter((e) => e.parity > 0).length} ordered even-parity rows ` +
        `collapse to substantially fewer distinct invariants — the maximum quoted above reappears ` +
        `under a permuted index order. The separation is real; the number of independent witnesses ` +
        `to it is smaller than the table length.` }));

    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Check ledger — python/experiments/descriptor_blindness.py'),
      h('p', { class: 'hint' },
        'Reproduce with:  uv run python python/experiments/descriptor_blindness.py'));
    const inner = h('div');
    for (const c of D.checks) inner.append(checkLine(c.passed, c.name, c.detail));
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${D.meta.n_pass === D.meta.n_total ? 'ok' : 'bad'}` },
        `${D.meta.n_pass}/${D.meta.n_total}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);
    root.append(ledger);
  },
};
