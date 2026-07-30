import { h, slider, segmented, checkLine, loadResults, fmt, chRef, Plot, PALETTE } from '../ui.js';
import { realSH, wignerD, clebschGordan, tensorProduct, randomRotation, mulberry32 } from '../e3.js';
import { matvec, norm } from '../linalg.js';

const dim = (l) => 2 * l + 1;

/** V_l1 (x) V_l2 decomposes into a direct sum over the triangle range. The
 *  dimension bookkeeping is a genuine check, not a restatement: the product
 *  space and the direct sum have to have the same size. */
function decompose(l1, l2) {
  const out = [];
  for (let l = Math.abs(l1 - l2); l <= l1 + l2; l++) out.push(l);
  return out;
}

export default {
  id: 'couplings',
  title: 'Clebsch–Gordan and the bispectrum',
  async render(root) {
    const D = await loadResults('descriptor_blindness');
    const BS = D.bispectrum;
    const F = D.features;
    const B = D.part_b_three_body_blindness;

    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 5'),
      h('h1', {}, 'Clebsch–Gordan and the bispectrum'),
      h('p', { class: 'lede' },
        'One algebraic object sits underneath both halves of this field. The descriptor tradition ' +
        'uses it to build invariants by hand; the message-passing tradition uses it to multiply ' +
        'features inside a network. This chapter is that object, worked properly, and then the ' +
        'invariant it generates.'),

      h('h2', {}, 'Part 1 · What Clebsch–Gordan coefficients actually are'),
      h('p', { class: 'prose', html:
        `${chRef('machinery', { capital: true })} derived them as “the unique equivariant bilinear ` +
        `map” and moved on. That is the right operational description, but it hides where they come ` +
        `from. The honest starting point is a question about representations: if I have a degree-` +
        `$\\ell_1$ object and a degree-$\\ell_2$ object, what kind of thing is the pair?` }),
      h('p', { class: 'prose', html:
        'The pair lives in the tensor product $V_{\\ell_1} \\otimes V_{\\ell_2}$, which has ' +
        'dimension $(2\\ell_1+1)(2\\ell_2+1)$, and a rotation acts on it by $D^{\\ell_1} \\otimes ' +
        'D^{\\ell_2}$. But that action is <em>reducible</em>: the big space splits into subspaces ' +
        'that rotations never mix. The decomposition is the central fact of the whole subject,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.15em 0' } },
        '$$V_{\\ell_1} \\otimes V_{\\ell_2} \\;\\cong\\; \\bigoplus_{\\ell = |\\ell_1 - \\ell_2|}^{\\ell_1 + \\ell_2} V_{\\ell}$$'),
      h('p', { class: 'prose', html:
        'and the <strong>Clebsch–Gordan coefficients are precisely the change-of-basis matrix that ' +
        'realises it</strong>. They are not a formula someone invented; they are the coordinates of ' +
        'one basis written in another. Each irreducible piece appears exactly once, which is the ' +
        'algebraic reason the coupling is unique up to scale, and therefore the reason the network ' +
        'has nothing to learn in the angular part.' }),

      h('h3', {}, 'The dimension identity, as a check'),
      h('p', { class: 'prose', html:
        'A decomposition claim is falsifiable: the two sides must have the same dimension, so ' +
        '$(2\\ell_1+1)(2\\ell_2+1) = \\sum_\\ell (2\\ell+1)$ over the triangle range. That identity ' +
        'is easy to state and easy to get wrong, so the panel below computes both sides.' }),
    );

    // ---- decomposition explorer -------------------------------------------
    const dec = h('div', { class: 'demo' });
    dec.append(h('h3', {}, 'Decomposing a product of two irreps'),
      h('p', { class: 'hint' }, 'Both sides counted independently and compared.'));
    const decOut = h('div', { class: 'readout' });
    let a1 = 1, a2 = 1;
    function drawDec() {
      const parts = decompose(a1, a2);
      const lhs = dim(a1) * dim(a2);
      const rhs = parts.reduce((s, l) => s + dim(l), 0);
      const named = { '1,1': ['dot product', 'cross product', 'traceless symmetric part'] };
      const key = `${Math.min(a1, a2)},${Math.max(a1, a2)}`;
      decOut.innerHTML =
        `V_${a1} ⊗ V_${a2}  ≅  ${parts.map((l) => `V_${l}`).join('  ⊕  ')}\n\n` +
        `left  dimension   (2·${a1}+1)(2·${a2}+1) = ${dim(a1)} × ${dim(a2)} = ${lhs}\n` +
        `right dimension   ${parts.map((l) => dim(l)).join(' + ')} = ${rhs}\n` +
        `                  <span class="${lhs === rhs ? 'ok' : 'bad'}">` +
        `${lhs === rhs ? 'identity holds' : 'MISMATCH'}</span>\n\n` +
        (named[key]
          ? `<span class="dim">the three pieces of a vector times a vector, by name:\n  ` +
            parts.map((l, i) => `V_${l} (${dim(l)} number${dim(l) === 1 ? '' : 's'}) — ${named[key][i]}`).join('\n  ') +
            `\n  3 × 3 = 9 = 1 + 3 + 5, which is the familiar decomposition of a 3×3 matrix into\n  ` +
            `trace, antisymmetric part and traceless symmetric part.</span>`
          : `<span class="dim">multiplicity is one for every piece, which is why each coupling is ` +
            `unique up to scale.</span>`);
    }
    dec.append(h('div', { class: 'controls' },
      segmented({ label: 'ℓ₁', options: [0, 1, 2, 3].map((v) => ({ label: String(v), value: v })),
        value: 1, onPick: (v) => { a1 = v; drawDec(); } }),
      segmented({ label: 'ℓ₂', options: [0, 1, 2, 3].map((v) => ({ label: String(v), value: v })),
        value: 1, onPick: (v) => { a2 = v; drawDec(); } })), decOut);
    root.append(dec);
    drawDec();

    root.append(
      h('p', { class: 'prose', html:
        'The $\\ell_1 = \\ell_2 = 1$ case is the one everybody already knows without knowing they ' +
        'know it. A vector times a vector gives nine numbers, and they fall apart into one scalar, ' +
        'three components of a vector, and five more: the dot product, the cross product, and the ' +
        'traceless symmetric part. That is exactly the decomposition of a $3\\times3$ matrix into ' +
        'trace, antisymmetric and traceless-symmetric parts — a fact from linear algebra that turns ' +
        'out to be a statement about $SO(3)$ representations.' }),

      h('h3', {}, 'Getting the numbers without a table'),
      h('p', { class: 'prose', html:
        `${chRef('machinery', { capital: true })} showed the trick and it is worth restating as ` +
        `algebra rather than as code. A coupling tensor $C$ is equivariant exactly when it is an ` +
        `<em>invariant vector</em> in the triple product $V_{\\ell_1} \\otimes V_{\\ell_2} \\otimes ` +
        `V_{\\ell_3}$ — that is, when` }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.05em 0' } },
        '$$\\bigl(D^{\\ell_1}(g) \\otimes D^{\\ell_2}(g) \\otimes D^{\\ell_3}(g)\\bigr)\\,' +
        '\\mathrm{vec}(C) \\;=\\; \\mathrm{vec}(C) \\qquad \\text{for every } g.$$'),
      h('p', { class: 'prose', html:
        'Stack that linear constraint for a handful of generic rotations and take the null space. ' +
        'Because generic rotations generate a dense subgroup, a finite handful already pins down the ' +
        'exact invariant subspace — and the decomposition above guarantees that subspace is ' +
        'one-dimensional whenever the triangle rule is satisfied. So the smallest singular vector ' +
        '<em>is</em> the coupling, and the second-smallest singular value being of order one is a ' +
        'numerical certificate that the multiplicity really is one.' }),
    );

    // ---- CG tensor viewer --------------------------------------------------
    const cg = h('div', { class: 'demo' });
    cg.append(h('h3', {}, 'The coupling tensor itself'),
      h('p', { class: 'hint' },
        'Each panel is one slice $C[\\cdot,\\cdot,m_3]$ of the tensor. Blue positive, red negative. ' +
        'Solved in your browser, then tested against the property that defines it.'));
    const cgRow = h('div', { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } });
    const cgOut = h('div', { class: 'readout' });
    let c1 = 1, c2 = 1, c3 = 1;

    function drawCG() {
      const allowed = decompose(c1, c2);
      if (!allowed.includes(c3)) c3 = allowed[0];
      const t0 = performance.now();
      const { C, sigmaMin, sigmaSecond } = clebschGordan(c1, c2, c3);
      const ms = performance.now() - t0;
      let maxAbs = 0;
      for (const p of C) for (const r of p) for (const v of r) maxAbs = Math.max(maxAbs, Math.abs(v));

      cgRow.innerHTML = '';
      for (let m3 = 0; m3 < dim(c3); m3++) {
        const cell = 22;
        const cv = h('canvas', { width: dim(c1) * cell, height: dim(c2) * cell,
          style: { border: '1px solid #E2E6EA', borderRadius: '4px' } });
        const ctx = cv.getContext('2d');
        for (let i = 0; i < dim(c1); i++) {
          for (let j = 0; j < dim(c2); j++) {
            const v = C[i][j][m3] / (maxAbs || 1);
            const t = Math.abs(v);
            const [r0, g0, b0] = v >= 0 ? [31, 78, 121] : [181, 68, 60];
            ctx.fillStyle = `rgb(${Math.round(255 + t * (r0 - 255))},${Math.round(255 + t * (g0 - 255))},${Math.round(255 + t * (b0 - 255))})`;
            ctx.fillRect(i * cell, j * cell, cell - 1, cell - 1);
          }
        }
        cgRow.appendChild(h('div', { style: { textAlign: 'center' } }, cv,
          h('div', { class: 'sans', style: { fontSize: '11.5px', color: '#5A6773', marginTop: '2px' } },
            `m₃ = ${m3 - c3 > 0 ? '+' : ''}${m3 - c3}`)));
      }

      const rng = mulberry32(777);
      let worst = 0;
      for (let t = 0; t < 8; t++) {
        const R = randomRotation(rng);
        const u = Array.from({ length: dim(c1) }, () => 2 * rng() - 1);
        const w = Array.from({ length: dim(c2) }, () => 2 * rng() - 1);
        const lhs = tensorProduct(C, matvec(wignerD(c1, R), u), matvec(wignerD(c2, R), w));
        const rhs = matvec(wignerD(c3, R), tensorProduct(C, u, w));
        const sc = Math.max(norm(rhs), 1e-12);
        worst = Math.max(worst, Math.max(...lhs.map((v, i) => Math.abs(v - rhs[i]) / sc)));
      }
      let sparsity = 0, total = 0;
      for (const p of C) for (const r of p) for (const v of r) { total++; if (Math.abs(v) < 1e-12) sparsity++; }

      cgOut.innerHTML =
        `C for (${c1}, ${c2}) → ${c3}    shape ${dim(c1)} × ${dim(c2)} × ${dim(c3)} = ${total} numbers` +
        `   <span class="dim">solved in ${ms.toFixed(0)} ms</span>\n` +
        `structurally zero   ${sparsity} of ${total} (${(100 * sparsity / total).toFixed(0)}%)` +
        `   <span class="dim">the sparsity eSCN later exploits</span>\n` +
        `uniqueness          σ_min ${sigmaMin.toExponential(2)},  σ₂ ${sigmaSecond.toExponential(2)}\n` +
        `equivariance        max relative deviation ` +
        `<span class="${worst < 1e-8 ? 'ok' : 'bad'}">${worst.toExponential(2)}</span>`;
    }

    cg.append(h('div', { class: 'controls' },
      segmented({ label: 'ℓ₁', options: [0, 1, 2].map((v) => ({ label: String(v), value: v })),
        value: 1, onPick: (v) => { c1 = v; drawCG(); } }),
      segmented({ label: 'ℓ₂', options: [0, 1, 2].map((v) => ({ label: String(v), value: v })),
        value: 1, onPick: (v) => { c2 = v; drawCG(); } }),
      segmented({ label: 'ℓ₃', options: [0, 1, 2, 3, 4].map((v) => ({ label: String(v), value: v })),
        value: 1, onPick: (v) => { c3 = v; drawCG(); } })), cgRow, cgOut);
    root.append(cg);
    drawCG();

    root.append(
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Why the sparsity matters later'),
        h('div', { html:
          'Notice how much of the tensor is exactly zero. In the complex basis the reason is a hard ' +
          'selection rule, $m_3 = m_1 + m_2$, which kills every entry that does not satisfy it; the ' +
          'real basis used here mixes $\\pm m$ but the same structure survives. That sparsity is not ' +
          'cosmetic — the cost of a tensor product is the count of surviving terms, and the whole ' +
          'point of eSCN was to make it far sparser still by rotating each edge into a frame where ' +
          'the filter has only its $m = 0$ component.' })),

      h('h2', {}, 'Part 2 · The bispectrum'),
      h('p', { class: 'prose', html:
        'Now use the coupling for something. Take an atom’s neighbourhood and write it as a density,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.05em 0' } },
        '$$\\rho_i(\\vec r) \\;=\\; \\sum_{j \\in \\mathcal{N}(i)} \\delta(\\vec r - \\vec r_{ij}) \\ \\ (\\text{smoothed}), ' +
        '\\qquad c^{(i)}_{\\ell m} \\;=\\; \\sum_j R(r_{ij})\\, Y^{(\\ell)}_{m}(\\hat r_{ij}).$$'),
      h('p', { class: 'prose', html:
        'Those coefficients $c_{\\ell m}$ are exactly the equivariant atom-centred features from ' +
        `${chRef('blindness')} — the same object, arrived at from the density side. They are not ` +
        `invariant: a rotation mixes the $m$ within each $\\ell$. To get something a regression can ` +
        `consume you have to contract the $m$ away, and there is a ladder of ways to do it.` }),
      h('p', { class: 'prose', html:
        'The cheapest contraction pairs a feature with itself — the <strong>power spectrum</strong>,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1em 0' } },
        '$$p_\\ell \\;=\\; \\sum_m \\bigl| c_{\\ell m} \\bigr|^2 \\;=\\; \\lVert c_\\ell \\rVert^2,$$'),
      h('p', { class: 'prose', html:
        'which is invariant because rotations act by orthogonal matrices and orthogonal matrices ' +
        'preserve length. This is the SOAP power spectrum, and it captures three-body information: ' +
        'through the spherical-harmonic addition theorem it is a sum over <em>pairs</em> of ' +
        'neighbours of a Legendre polynomial in the angle between them, which is a central atom plus ' +
        'two neighbours.' }),
      h('p', { class: 'prose', html:
        'The next rung contracts <em>three</em> features through a coupling. That is the ' +
        '<strong>bispectrum</strong>,' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1em 0' } },
        '$$b_{\\ell_1 \\ell_2 \\ell_3} \\;=\\; \\sum_{m_1 m_2 m_3} ' +
        'C^{(\\ell_3, m_3)}_{(\\ell_1, m_1)(\\ell_2, m_2)}\\; c_{\\ell_1 m_1}\\, c_{\\ell_2 m_2}\\, c_{\\ell_3 m_3},$$'),
      h('p', { class: 'prose', html:
        'invariant for the same reason the coupling is equivariant: contract an equivariant object ' +
        'all the way down to $\\ell = 0$ and nothing is left to rotate. It carries four-body ' +
        'information — a central atom plus three neighbours — and it is the descriptor at the heart ' +
        'of GAP’s SOAP kernel, of SNAP, and of the Atomic Cluster Expansion’s low-order terms.' }),

      h('h3', {}, 'The rung that matters'),
      h('p', { class: 'prose', html:
        `${chRef('blindness', { capital: true })} built two atomic environments that every ` +
        `three-body descriptor is provably blind to: the all-interval tetrachords ` +
        `$\\{${B.z12_set_A.join(',')}\\}$ and $\\{${B.z12_set_B.join(',')}\\}$ on ` +
        `$\\mathbb{Z}_{${B.n_ring}}$, separated by exactly ` +
        `$(\\sqrt3-1)/2 = ${B.rmsd_analytic_sqrt3_minus_1_over_2.toFixed(6)}$. This is where the ` +
        `ladder earns its keep. Below, the power spectrum and the bispectrum are evaluated on that ` +
        `same pair.` }),
    );

    // ---- power spectrum vs bispectrum on the homometric pair ---------------
    const bs = h('div', { class: 'demo' });
    bs.append(h('h3', {}, 'Power spectrum versus bispectrum, on a pair nothing three-body can separate'),
      h('p', { class: 'hint' },
        'Numbers from python/experiments/descriptor_blindness.py. Sorted by how well each invariant ' +
        'distinguishes the two environments.'));
    const even = BS.entries.filter((e) => e.parity > 0).slice().sort((x, y) => y.abs_diff - x.abs_diff);
    const top = even.slice(0, 8);
    bs.append(
      h('div', { class: 'readout', html:
        `power spectrum p_l, every degree l = 0..6\n` +
        F.power_spectrum_abs_diff.map((v, i) =>
          `  l = ${i}   |p_l(A) − p_l(B)| = ${v.toExponential(2)}`).join('\n') +
        `\n  worst over all degrees: <span class="bad">${F.power_spectrum_max_abs_diff.toExponential(2)}</span>` +
        `   <span class="dim">machine zero — completely blind</span>` }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'bispectrum entry'), h('th', { class: 'num' }, 'b(A)'),
          h('th', { class: 'num' }, 'b(B)'), h('th', { class: 'num' }, '|Δ|'))),
        h('tbody', {}, top.map((e) => h('tr', { class: e.abs_diff > 1e-6 ? 'hi' : '' },
          h('td', {}, h('code', {}, `b(${e.l1},${e.l2},${e.l3})`)),
          h('td', { class: 'num' }, fmt(e.b_A, 4)),
          h('td', { class: 'num' }, fmt(e.b_B, 4)),
          h('td', { class: 'num' }, e.abs_diff.toExponential(2)))))));
    root.append(bs);

    root.append(
      h('p', { class: 'prose', html:
        `That is the whole argument for body order in one table. Every power-spectrum entry agrees ` +
        `to machine precision, at every degree up to six. Bispectrum entries disagree by as much as ` +
        `${BS.max_abs_diff_even_parity.toFixed(4)}. Going from three-body to four-body is not a ` +
        `refinement here — it is the difference between zero information and enough to separate two ` +
        `structures.` }),
      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'Two honest caveats, both worth carrying'),
        h('div', { html:
          `First, the table as printed over-counts. $C^{\\ell_1\\ell_2\\ell_3}$ is the same invariant ` +
          `under permuting its indices up to sign, so the ${even.length} ordered even-parity rows ` +
          `collapse to substantially fewer distinct invariants, and the largest entry reappears under ` +
          `a permuted order. The separation is real; the number of independent witnesses is smaller ` +
          `than the row count.<br><br>` +
          `Second, every parity-odd entry vanishes for both configurations (largest ` +
          `${BS.max_abs_value_odd_parity.toExponential(1)}), because both are planar and therefore ` +
          `achiral — and most odd entries are identically zero for <em>any</em> geometry, since the ` +
          `coupling is antisymmetric under exchanging two equal degrees while the bispectrum feeds ` +
          `the same feature into both slots. Only one genuinely distinct pseudo-scalar is being ` +
          `tested there.` })),

      h('h2', {}, 'Part 3 · Why this is the paper’s hinge'),
      h('p', { class: 'prose', html:
        'Here is the thing worth carrying away. The bispectrum and NequIP’s convolution use the ' +
        '<em>same</em> Clebsch–Gordan coefficients, on the <em>same</em> atom-centred features, and ' +
        'they differ in exactly one decision: <strong>when to contract down to a scalar.</strong>' }),
      h('div', { class: 'grid2' },
        h('div', { class: 'card dat' }, h('h4', {}, 'The descriptor road'),
          h('p', { html:
            'Contract immediately. Choose in advance which invariants to compute — power spectrum, ' +
            'then bispectrum, then higher — hand them to a regressor, and be done. Every step is ' +
            'interpretable and the body order is explicit. The cost is that you must decide the ' +
            'contractions up front, and their number grows steeply. This is GAP/SOAP, SNAP, and the ' +
            'Atomic Cluster Expansion.' })),
        h('div', { class: 'card geo' }, h('h4', {}, 'The message-passing road'),
          h('p', { html:
            'Do not contract. Keep the equivariant features, use the coupling to multiply them by ' +
            'new bond directions, stack layers so products of products become reachable, and collapse ' +
            'to a scalar exactly once at the readout. The network decides which contractions matter. ' +
            'This is Tensor Field Networks and NequIP.' }))),
      h('p', { class: 'prose', html:
        `Both roads climb body order; they differ in whether a human or the optimiser picks the ` +
        `route. And the two turn out to be the same design space seen from different sides — which ` +
        `is what the Multi-ACE analysis (<a href="https://arxiv.org/abs/2205.06643">` +
        `arXiv:2205.06643</a>) established, placing NequIP, MACE, ACE and the descriptor methods in ` +
        `one framework. MACE is the clearest hybrid: it computes ACE-style higher-body-order products ` +
        `<em>inside</em> a message-passing layer, which is why it reaches high body order in two ` +
        `layers where NequIP needs more.` }),
      h('p', { class: 'prose', html:
        `This is not a settled question, and ${chRef('genealogy')} shows why it matters right now. ` +
        `The current Matbench Discovery leaders include models from both roads: GRACE descends from ` +
        `the Atomic Cluster Expansion, and TACE and the CPS leader TECE use irreducible ` +
        `<em>Cartesian</em> tensors that deliberately avoid Clebsch–Gordan spherical products ` +
        `altogether. The coupling in this chapter is the standard tool, not the only one.` }),

      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'Where each object shows up in the paper'),
        h('div', { html:
          `Equation 4 is the filter, $R(r_{ij})\\,Y^{(\\ell)}_m(\\hat r_{ij})$ — the density ` +
          `expansion above, one neighbour at a time. Equation 7 is the parity selection rule. ` +
          `Equation 8 is the convolution, and the $C^{(\\ell_o,m_o)}_{(\\ell_i,m_i)(\\ell_f,m_f)}$ ` +
          `sitting at its front is the coefficient this chapter built. The bispectrum never appears ` +
          `in NequIP at all — that is precisely the paper’s point of departure from the tradition it ` +
          `grew out of. ${chRef('layer', { capital: true })} assembles equation 8; ` +
          `${chRef('lmax')} measures what truncating $\\ell_{\\max}$ costs.` })),
    );

    // ---- ledger -------------------------------------------------------------
    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Checks behind this chapter'),
      h('p', { class: 'hint' },
        'The bispectrum numbers come from  uv run python python/experiments/descriptor_blindness.py; ' +
        'the couplings are rebuilt and retested in your browser each time you change a control.'));
    const inner = h('div');
    for (const c of D.checks.filter((c) => /bispectrum|power spectrum|equivarian|invariant/i.test(c.name))) {
      inner.append(checkLine(c.passed, c.name, c.detail));
    }
    // and a live one: the dimension identity across every pair up to l=3
    let dimOk = true, cases = 0;
    for (let x = 0; x <= 3; x++) {
      for (let y = 0; y <= 3; y++) {
        const lhs = dim(x) * dim(y);
        const rhs = decompose(x, y).reduce((s, l) => s + dim(l), 0);
        cases++;
        if (lhs !== rhs) dimOk = false;
      }
    }
    inner.append(checkLine(dimOk,
      'dimension identity (2ℓ₁+1)(2ℓ₂+1) = Σ(2ℓ+1) over the triangle range',
      `${cases}/${cases} pairs up to ℓ = 3, checked live`));
    ledger.append(inner);
    root.append(ledger);
  },
};
