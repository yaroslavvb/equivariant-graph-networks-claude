import { h, slider, segmented, checkLine, loadResults, Plot, PALETTE } from '../ui.js';

// ---------------------------------------------------------------------------
// Live lattice machinery. The receptive field in the first demo is recomputed
// in the page by breadth-first search on the integer lattice, so its numbers
// are measured here rather than transcribed; the counts stored in the results
// JSON are used to cross-check the browser's answer.
// ---------------------------------------------------------------------------

const KEY = (x, y, z) => ((x + 64) * 128 + (y + 64)) * 128 + (z + 64);

function offsets(rCut) {
  const R = Math.floor(rCut + 1e-12);
  const out = [];
  for (let x = -R; x <= R; x++) {
    for (let y = -R; y <= R; y++) {
      for (let z = -R; z <= R; z++) {
        const r = Math.sqrt(x * x + y * y + z * z);
        if (r > 1e-12 && r <= rCut + 1e-12) out.push([x, y, z, r]);
      }
    }
  }
  return out;
}

const FIELD_CACHE = new Map();

/** Breadth-first hop shells of the infinite lattice out to depth lMax. */
function buildField(rCut, lMax) {
  const ck = `${rCut}:${lMax}`;
  if (FIELD_CACHE.has(ck)) return FIELD_CACHE.get(ck);

  const off = offsets(rCut);
  let c = 0;
  for (const o of off) c = Math.max(c, o[3]);

  const hop = new Map([[KEY(0, 0, 0), 0]]);
  let frontier = [[0, 0, 0]];
  const nByL = [1];
  const rByL = [0];
  let n = 1;
  let rMax = 0;

  for (let L = 1; L <= lMax; L++) {
    const next = [];
    for (const p of frontier) {
      for (const o of off) {
        const x = p[0] + o[0], y = p[1] + o[1], z = p[2] + o[2];
        const k = KEY(x, y, z);
        if (hop.has(k)) continue;
        hop.set(k, L);
        next.push([x, y, z]);
        n += 1;
        const r = Math.sqrt(x * x + y * y + z * z);
        if (r > rMax) rMax = r;
      }
    }
    frontier = next;
    nByL.push(n);
    rByL.push(rMax);
  }
  const F = { rCut, lMax, k: off.length, c, hop, nByL, rByL };
  FIELD_CACHE.set(ck, F);
  return F;
}

// ---------------------------------------------------------------------------
// The z = 0 slice of the receptive field.
// ---------------------------------------------------------------------------

const NS = 'http://www.w3.org/2000/svg';
const sv = (t, a) => {
  const e = document.createElementNS(NS, t);
  for (const [k, v] of Object.entries(a)) if (v != null) e.setAttribute(k, String(v));
  return e;
};

function hopColour(L, lMax) {
  if (L === 0) return '#1B2733';
  const f = lMax <= 1 ? 0 : (L - 1) / (lMax - 1);
  return `hsl(206, ${Math.round(62 - 18 * f)}%, ${Math.round(31 + 40 * f)}%)`;
}

function sliceFigure(F, L) {
  const W = 660, H = 430;
  const M = Math.max(1, Math.ceil(L * F.rCut + 1e-9));
  const cell = Math.min(392 / (2 * M + 1), 44);
  const S = cell * (2 * M + 1);
  const ox = 18 + S / 2, oy = H / 2 - 6;
  const px = (x) => ox + x * cell;
  const py = (y) => oy - y * cell;
  const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' });

  for (let x = -M; x <= M; x++) {
    for (let y = -M; y <= M; y++) {
      const hh = F.hop.get(KEY(x, y, 0));
      if (hh != null && hh <= L) continue;
      svg.appendChild(sv('circle', { cx: px(x), cy: py(y), r: Math.max(1.1, cell * 0.09),
        fill: '#C9CFD6', opacity: 0.75 }));
    }
  }
  svg.appendChild(sv('circle', { cx: px(0), cy: py(0), r: L * F.rCut * cell, fill: 'none',
    stroke: PALETTE[1], 'stroke-width': 1.3, 'stroke-dasharray': '5 4', opacity: 0.9 }));
  svg.appendChild(sv('circle', { cx: px(0), cy: py(0), r: F.rByL[L] * cell, fill: 'none',
    stroke: PALETTE[0], 'stroke-width': 1.3, 'stroke-dasharray': '3 3', opacity: 0.9 }));
  for (let x = -M; x <= M; x++) {
    for (let y = -M; y <= M; y++) {
      const hh = F.hop.get(KEY(x, y, 0));
      if (hh == null || hh > L) continue;
      svg.appendChild(sv('circle', { cx: px(x), cy: py(y),
        r: Math.max(2.2, Math.min(cell * 0.3, 10)), fill: hopColour(hh, Math.max(L, 1)) }));
    }
  }
  svg.appendChild(sv('circle', { cx: px(0), cy: py(0),
    r: Math.max(5, Math.min(cell * 0.44, 13)), fill: 'none',
    stroke: '#1B2733', 'stroke-width': 1.6 }));

  const tx = 18 + S + 26;
  const label = (y, s, size = 12, fill = '#5A6773', weight = 400) => {
    const t = sv('text', { x: tx, y, 'font-size': size, fill, 'font-weight': weight,
      'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    t.textContent = s;
    svg.appendChild(t);
  };
  label(30, 'hops from the centre', 11.5, '#1B2733', 700);
  for (let l = 0; l <= L; l++) {
    const y = 50 + l * 21;
    svg.appendChild(sv('circle', { cx: tx + 6, cy: y - 4, r: 6, fill: hopColour(l, Math.max(L, 1)) }));
    label(y, `      ${l}${l === 0 ? '   the atom itself' : ''}`, 12);
  }
  const yb = 50 + (L + 1) * 21 + 16;
  svg.appendChild(sv('line', { x1: tx, x2: tx + 22, y1: yb - 4, y2: yb - 4,
    stroke: PALETTE[0], 'stroke-width': 1.6, 'stroke-dasharray': '3 3' }));
  label(yb, `      R_eff = ${F.rByL[L].toFixed(3)} d`, 12);
  svg.appendChild(sv('line', { x1: tx, x2: tx + 22, y1: yb + 18, y2: yb + 18,
    stroke: PALETTE[1], 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }));
  label(yb + 22, `      L r_cut = ${(L * F.rCut).toFixed(3)} d`, 12);
  label(H - 8, 'slice through z = 0; the field itself is three-dimensional', 11.5);
  return svg;
}

export default {
  id: 'locality',
  title: 'How far does locality reach',
  async render(root) {
    const D = await loadResults('receptive_field');
    const RF = D.receptive_field;
    const SW = D.growth_sweep;
    const CS = D.cubic_scaling;
    const FS = D.finite_size;
    const DV = D.depth_vs_cutoff;
    const NA = D.nequip_vs_allegro;
    const T = D.truncation;
    const SH = D.shell_harmonics;
    const LA = D.lattice;

    const ratios = SW.map((s) => s.c_over_r_cut);
    const worstOverstatement = (1 / Math.min(...ratios) - 1) * 100;
    const exps = CS.loglog_exponent_L_to_2L;
    const deepest = exps[exps.length - 1];
    const first = DV.configs[0];
    const last = DV.configs[DV.configs.length - 1];

    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 7'),
      h('h1', {}, 'How far does locality reach'),
      h('p', { class: 'lede' },
        'Depth is the only way a message-passing network acquires reach. One interaction layer ' +
        'lets an atom see inside its own cutoff sphere; two layers let it see its neighbours’ ' +
        'spheres; after L layers its prediction depends on everything within L hops of it in the ' +
        'cutoff graph. This chapter measures that reach on a crystal where every quantity is ' +
        'exactly computable, prices the ghost atoms it obliges a parallel code to own — which is ' +
        'precisely the price Allegro was designed to delete — and then asks the separate and ' +
        'harder question: whether locality is enough to represent the physics at all.'),

      h('h2', {}, 'Reach grows one hop at a time'),
      h('p', { class: 'prose', html:
        `The test structure is rock salt: a simple cubic lattice of alternating $\\pm 1$ ions with ` +
        `nearest-neighbour spacing $d = 1$, ${LA.n_atoms} sites in a ` +
        `${LA.n_cells}&times;${LA.n_cells}&times;${LA.n_cells} periodic box. It does two jobs — ` +
        `first it is the graph whose hop neighbourhoods we enumerate, later it is the charge ` +
        `distribution whose lattice sums we truncate. Nothing about it is approximate, which is ` +
        `the reason for choosing it: every count below is an integer and every radius is a surd.` }),
      h('p', { class: 'prose', html:
        `Draw an edge between two atoms whenever they are closer than $r_{\\mathrm{cut}}$. At ` +
        `$r_{\\mathrm{cut}} = ${RF.r_cut}\\,d$ every atom has exactly ${RF.coordination_min} ` +
        `neighbours — checked against an independent periodic KD-tree, ` +
        `${RF.pair_set_mismatches} disagreements over ${RF.brute_force_pairs} pairs — and the ` +
        `$L$-hop neighbourhood of an atom is the set of sites reachable in $L$ steps. The panel ` +
        `below rebuilds that set in your browser by breadth-first search and compares the count ` +
        `with the one the Python script stored.` }));

    // ---- interactive receptive field -------------------------------------
    const rfDemo = h('div', { class: 'demo' });
    rfDemo.append(h('h3', {}, 'The receptive field of a depth-L model'),
      h('p', { class: 'hint' },
        'Pick a cutoff and a depth. Colour is hop distance from the central atom; grey sites are ' +
        'in the crystal but outside the model’s reach. The blue circle is the measured effective ' +
        'radius, the red one the L × r_cut estimate everybody quotes.'));

    const figHolder = h('div');
    const readout = h('div', { class: 'readout' });
    let rCut = SW[0].r_cut;
    let depth = 3;
    const lMaxFor = (rc) => SW.find((s) => s.r_cut === rc).L.length - 1;

    const redrawField = () => {
      const lm = lMaxFor(rCut);
      if (depth > lm) depth = lm;
      const F = buildField(rCut, lm);
      const stored = SW.find((s) => s.r_cut === rCut);
      const n = F.nByL[depth];
      const rEff = F.rByL[depth];
      const bound = depth * rCut;
      const sphere = (4 * Math.PI / 3) * rEff ** 3;
      const seq = [];
      for (let L = 1; L <= lm; L++) seq.push((F.nByL[L] / L ** 3).toFixed(3));

      figHolder.innerHTML = '';
      figHolder.appendChild(sliceFigure(F, depth));
      readout.textContent =
        `r_cut = ${rCut.toFixed(2)} d      k = ${F.k} neighbours per atom      L = ${depth}\n\n` +
        `reach   measured R_eff              ${rEff.toFixed(6)}\n` +
        `        exact law    L x c          ${(depth * F.c).toFixed(6)}    ` +
        `c = ${F.c.toFixed(6)}, the longest lattice vector <= r_cut\n` +
        `        quoted bound L x r_cut      ${bound.toFixed(6)}    ` +
        `overstates the reach by ${((bound / rEff - 1) * 100).toFixed(1)} %\n\n` +
        `size    atoms in the field N(L)     ${n}` +
        `${stored.n_atoms[depth] === n ? `   (results JSON: ${stored.n_atoms[depth]}, match)`
          : `   MISMATCH against ${stored.n_atoms[depth]}`}\n` +
        `        N(L) / L^3                  ${(n / depth ** 3).toFixed(3)}    ` +
        `over L = 1..${lm}:  ${seq.join('  ')}\n` +
        `        sphere (4pi/3) R_eff^3      ${sphere.toFixed(1)}    ` +
        `the graph ball fills ${(100 * n / sphere).toFixed(1)} % of it\n\n` +
        `cost    ghost atoms to own          ${n - 1}\n` +
        `        edge evaluations per atom   L x k = ${depth * F.k}\n` +
        `        communication rounds        ${depth}`;
    };

    const depthSlider = slider({
      label: 'depth L', min: 1, max: lMaxFor(rCut), step: 1, value: depth,
      format: (v) => String(v), onInput: (v) => { depth = v; redrawField(); },
    });
    rfDemo.append(figHolder, readout, h('div', { class: 'controls' },
      segmented({
        label: 'cutoff r_cut',
        options: SW.map((s) => ({ label: s.r_cut.toFixed(1), value: s.r_cut })),
        value: rCut,
        onPick: (v) => {
          rCut = v;
          const lm = lMaxFor(v);
          depthSlider.querySelector('input').max = String(lm);
          if (depth > lm) depthSlider.setValue(lm); else redrawField();
        },
      }), depthSlider));
    root.append(rfDemo);
    redrawField();

    const exactCuts = SW.filter((s) => s.c_over_r_cut > 1 - 1e-12).map((s) => s.r_cut.toFixed(1));
    const shortCuts = SW.filter((s) => s.c_over_r_cut <= 1 - 1e-12).map((s) => s.r_cut.toFixed(1));

    root.append(
      h('p', { class: 'prose', html:
        `Two things in that picture are worth more than the numbers beside them. The field is not ` +
        `a sphere — it is the ball of the graph metric, an octahedron at the smallest cutoff and a ` +
        `cube at the largest, and the sphere it is inscribed in has a good deal more room in it ` +
        `than the model actually uses. And the blue circle reaches the red one at only one of the ` +
        `six cutoffs offered, $r_{\\mathrm{cut}} = ${exactCuts.join(', ')}$; at the other ` +
        `${shortCuts.length} it falls short.` }),

      h('h2', {}, 'The bound everybody quotes is not the law'),
      h('p', { class: 'prose', html:
        `"The receptive field grows like $L \\times r_{\\mathrm{cut}}$" is an upper bound from the ` +
        `triangle inequality, and it is attained only when the cutoff happens to be a distance the ` +
        `lattice actually realises. What holds exactly is ` +
        `$$R_{\\mathrm{eff}}(L) = L\\, c(r_{\\mathrm{cut}}), \\qquad c(r_{\\mathrm{cut}}) = ` +
        `\\max\\{\\,|n| \\;:\\; n \\in \\mathbb{Z}^3,\\ 0 < |n| \\le r_{\\mathrm{cut}}\\,\\},$$ ` +
        `because a single lattice vector can be chained collinearly with itself and no longer step ` +
        `exists to chain instead. Across the six cutoffs tested this is an identity to floating ` +
        `point, the largest deviation anywhere being ` +
        `${Math.max(...SW.map((s) => s.R_eff_minus_L_c_max)).toExponential(1)}.` }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'r_cut'), h('th', { class: 'num' }, 'neighbours k'),
          h('th', { class: 'num' }, 'c(r_cut)'), h('th', { class: 'num' }, 'c / r_cut'),
          h('th', { class: 'num' }, 'deepest R_eff measured'),
          h('th', { class: 'num' }, 'atoms there'))),
        h('tbody', {}, SW.map((s) => {
          const lm = s.L.length - 1;
          return h('tr', s.c_over_r_cut === Math.min(...ratios) ? { class: 'hi' } : {},
            h('td', {}, `${s.r_cut.toFixed(1)} d`),
            h('td', { class: 'num' }, String(s.k_neighbours)),
            h('td', { class: 'num' }, s.longest_edge_c.toFixed(6)),
            h('td', { class: 'num' }, s.c_over_r_cut.toFixed(4)),
            h('td', { class: 'num' }, `${s.R_eff[lm].toFixed(3)}   (L = ${lm})`),
            h('td', { class: 'num' }, String(s.n_atoms[lm])));
        }))),
      h('p', { class: 'prose', html:
        `The ratio $c/r_{\\mathrm{cut}}$ runs from ${Math.min(...ratios).toFixed(4)} to ` +
        `${Math.max(...ratios).toFixed(4)}. It reaches 1 exactly at ` +
        `$r_{\\mathrm{cut}} = ${exactCuts.join(', ')}$, where the cutoff is itself a lattice ` +
        `distance and the quoted estimate is right; at the other ${shortCuts.length} cutoffs it ` +
        `overstates the true reach, by as much as ${worstOverstatement.toFixed(1)}&nbsp;%. More ` +
        `awkwardly, the shortfall is not monotone ` +
        `in the cutoff. Going from $r_{\\mathrm{cut}} = ${SW[2].r_cut}$, where the ratio is ` +
        `${SW[2].c_over_r_cut.toFixed(4)}, up to $r_{\\mathrm{cut}} = ${SW[3].r_cut}$ makes it ` +
        `<em>worse</em> — back to ${SW[3].c_over_r_cut.toFixed(4)} — because the next shell at ` +
        `$\\sqrt{5} \\approx 2.236$ has not yet been reached and the longest available step is ` +
        `still ${SW[3].longest_edge_c.toFixed(0)}. Reach is a sawtooth in the cutoff rather than a ` +
        `smooth function of it, and where the teeth fall is a property of the structure, not of ` +
        `the model.` }),

      h('h2', {}, 'What depth costs'),
      h('p', { class: 'prose', html:
        `The atoms in the receptive field are exactly the atoms a domain-decomposed molecular ` +
        `dynamics code must own as ghosts before it can evaluate one atom’s energy. For ` +
        `$r_{\\mathrm{cut}}$ between $1$ and $\\sqrt{2}$ the graph is the six-neighbour cubic ` +
        `graph and the count is the octahedral number $(2L+1)(2L^2+2L+3)/3$; for ` +
        `$r_{\\mathrm{cut}}$ between $\\sqrt{3}$ and $2$ it is the 26-neighbour graph and the ` +
        `count is exactly $(2L+1)^3$. Both hold as integer identities. A cubic fit through the ` +
        `first six values of the former has residual ` +
        `${CS.cubic_fit_max_residual.toExponential(1)} and leading coefficient ` +
        `${CS.leading_coefficient.toFixed(9)}, against the predicted $4/3 = ` +
        `${CS.leading_coefficient_theory.toFixed(9)}$ — the volume of the unit $\\ell_1$ ball ` +
        `times the number density.` }));

    const costDemo = h('div', { class: 'demo' });
    costDemo.append(h('h3', {}, 'Ghost atoms against depth'),
      h('p', { class: 'hint' },
        'Both axes logarithmic. The dashed line is the asymptotic cubic term alone; the measured ' +
        'counts sit above it and approach it only slowly.'));
    const costPlot = new Plot({ width: 660, height: 330, xLog: true, yLog: true,
      xLabel: 'depth L  (interaction layers)', yLabel: 'atoms in the receptive field' });
    costPlot.add({
      points: CS.n_atoms.map((n, L) => [L, n]).filter((p) => p[0] >= 1),
      color: PALETTE[0], width: 2.4, markers: true,
      label: `measured, r_cut = ${CS.r_cut} d  (6-neighbour graph)` });
    costPlot.add({
      points: CS.cube_case.n_atoms.map((n, L) => [L, n]).filter((p) => p[0] >= 1),
      color: PALETTE[3], width: 2.2, markers: true,
      label: `measured, r_cut = ${CS.cube_case.r_cut} d  (26-neighbour graph, = (2L+1)³)` });
    costPlot.add({
      points: CS.n_atoms.map((_, L) => [L, CS.leading_coefficient * L ** 3]).filter((p) => p[0] >= 1),
      color: PALETTE[1], width: 1.8, dash: '5 4',
      label: 'asymptotic term alone,  (4/3) L³' });
    costPlot.setLimits([0.9, 13], [1, 4000]);
    costDemo.append(costPlot.render(), costPlot.legend(),
      h('div', { class: 'readout' },
        `local log-log exponent, measured by doubling the depth:\n` +
        exps.map((e) => `   L = ${e[0]} -> ${e[1]}      ${e[2].toFixed(3)}`).join('\n') +
        `\n\nthe asymptotic slope is 3; at L = ${deepest[1]} it is still ` +
        `${deepest[2].toFixed(3)}\n` +
        `N(5) = ${CS.n_atoms[5]} against its own leading term (4/3) 5^3 = ` +
        `${CS.leading_term_only_at_L5.toFixed(1)}:   ratio ` +
        `${CS.actual_over_leading_term_at_L5.toFixed(4)}`));
    root.append(costDemo);

    root.append(
      h('p', { class: 'prose', html:
        `That readout is the honest version of "the cost is cubic". It is cubic asymptotically, ` +
        `but at the depths anyone actually uses the asymptote has not arrived: the measured ` +
        `log-log exponent is ${exps[0][2].toFixed(2)} between $L = ${exps[0][0]}$ and ` +
        `$L = ${exps[0][1]}$, and has only crawled to ${deepest[2].toFixed(2)} by ` +
        `$L = ${deepest[1]}$. Surface terms still dominate — at $L = 5$ the true count is ` +
        `${CS.actual_over_leading_term_at_L5.toFixed(2)}&times; its own leading cubic term. ` +
        `Budgeting for $L^3$ growth will understate what the first few layers cost.` }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'the cell can end up smaller than the model'),
        h('div', { html:
          `A ${LA.n_cells}&times;${LA.n_cells}&times;${LA.n_cells} periodic cell hosts this ` +
          `receptive field faithfully up to $L = ${FS.first_L_where_cell_too_small - 1}$: the ` +
          `minimum-image hop counts agree with the infinite-lattice counts exactly. At ` +
          `$L = ${FS.first_L_where_cell_too_small}$ they part company, ` +
          `${FS.n_atoms_pbc[FS.first_L_where_cell_too_small]} against ` +
          `${FS.n_atoms_infinite[FS.first_L_where_cell_too_small]}, because an atom’s own ` +
          `periodic image has re-entered its receptive field. Past that depth the model is no ` +
          `longer evaluating the structure you think you handed it. Depth is not only a compute ` +
          `budget; it is a lower bound on how large a simulation cell has to be.` })),

      h('h2', {}, 'Depth against cutoff, and the decision Allegro made'),
      h('p', { class: 'prose', html:
        `Reach can be bought two ways: several cheap layers, or one expensive one. Fix the target ` +
        `at $R_{\\mathrm{eff}} \\approx ${DV.target_R_eff}\\,d$ and four configurations hit it to ` +
        `within ${(DV.R_eff_max_rel_spread * 100).toFixed(1)}&nbsp;%. What they cost is nowhere ` +
        `near equal.` }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'configuration'), h('th', { class: 'num' }, 'R_eff'),
          h('th', { class: 'num' }, 'edges per atom k'),
          h('th', { class: 'num' }, 'edge evals per atom'),
          h('th', { class: 'num' }, 'ghost atoms'),
          h('th', { class: 'num' }, 'comm. rounds'))),
        h('tbody', {}, DV.configs.map((c, i) => h('tr', i === 0 ? { class: 'hi' } : {},
          h('td', {}, `L = ${c.L},  r_cut = ${c.r_cut.toFixed(2)} d`),
          h('td', { class: 'num' }, c.R_eff.toFixed(3)),
          h('td', { class: 'num' }, String(c.k_edges_per_atom)),
          h('td', { class: 'num' }, String(c.edge_evals_per_atom)),
          h('td', { class: 'num' }, String(c.halo_atoms)),
          h('td', { class: 'num' }, String(c.comm_rounds)))))),
      h('p', { class: 'prose', html:
        `Deep and narrow wins on both of the quantities a single processor cares about. Moving ` +
        `from $L = ${first.L}$ at $r_{\\mathrm{cut}} = ${first.r_cut.toFixed(2)}$ to $L = ${last.L}$ ` +
        `at $r_{\\mathrm{cut}} = ${last.r_cut.toFixed(2)}$ multiplies the edge evaluations per atom ` +
        `by ` +
        `${DV.edge_eval_ratio_shallow_over_deep.toFixed(1)} ` +
        `(${first.edge_evals_per_atom} to ${last.edge_evals_per_atom}) and the ghost-atom halo ` +
        `by ${DV.halo_ratio_shallow_over_deep.toFixed(2)} (${first.halo_atoms} to ` +
        `${last.halo_atoms}). The only thing shallowness buys is the column on the right: ` +
        `${first.comm_rounds} rounds of communication become ${last.comm_rounds}.` }),
      h('p', { class: 'prose' },
        'That last column is the whole argument, and it is worth being exact about why. A ' +
        'message-passing model does not exchange its halo once. Every layer writes new features ' +
        'onto ghost atoms, and each layer’s features depend on atoms one hop further out, so a ' +
        'domain-decomposed run exchanges ghost features at every layer of the forward pass and ' +
        'again at every layer of the backward pass that produces the forces. Those exchanges are ' +
        'latency-bound and they synchronise: with L layers there are L barriers per timestep, and ' +
        'no amount of arithmetic throughput removes them. A strictly local model exchanges ' +
        'positions once, computes, and is finished.'),
      h('p', { class: 'prose', html:
        `Held instead at a fixed physical cutoff — $r_{\\mathrm{cut}} = ${NA.r_cut}\\,d$, which is ` +
        `${NA.k_edges_per_atom} edges per atom on this lattice — the trade shows its shape. Going ` +
        `from a strictly local single layer to a NequIP-like depth of ${NA.nequip_L} multiplies ` +
        `the reach by ${NA.reach_ratio.toFixed(1)} and the halo by ${NA.halo_ratio.toFixed(1)}, ` +
        `from ${NA.allegro_halo_atoms} ghost atoms to ${NA.nequip_halo_atoms}. That is the cubic ` +
        `law seen from the message-passing side: reach is linear in depth and what you must own ` +
        `in order to have it is cubic. Allegro’s design decision was to refuse the trade — keep ` +
        `$L = 1$ permanently, put every feature on an edge rather than on an atom, and buy ` +
        `expressive power per edge instead of reach. It is NequIP’s representation theory with ` +
        `the message passing taken out, and the reason to want that sits entirely in the ` +
        `right-hand columns of the table above.` }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'what the ghost-atom column really measures'),
        h('div', {},
          'The count here is the number of atoms one atom’s prediction depends on. A real ' +
          'domain-decomposed code pays a related but not identical price: its halo is a shell of ' +
          'thickness R_eff around each subdomain, so the absolute cost scales as subdomain ' +
          'surface area times R_eff rather than as R_eff cubed. What carries over unchanged is ' +
          'the ratio between depths, which is what the table is used for, and the fact that the ' +
          'shell thickens linearly in L.')),

      h('h2', {}, 'Does locality suffice?'),
      h('p', { class: 'prose', html:
        `Everything so far has been about the cost of reach. The prior question is whether reach ` +
        `was the thing missing. A model with cutoff $r_{\\mathrm{cut}}$ can at best represent the ` +
        `part of the interaction living inside $r_{\\mathrm{cut}}$, so whatever the truncated tail ` +
        `is worth is an error it inherits before it has made a single fitting mistake. On this ` +
        `lattice the tail can be computed rather than guessed. Three pair potentials, the same ` +
        `geometry, the same truncation.` }));

    // ---- truncation demo -------------------------------------------------
    const grid = T.r_cut;
    const idxOf = (rc) => {
      let best = 0, bd = Infinity;
      for (let i = 0; i < grid.length; i++) {
        const d = Math.abs(grid[i] - rc);
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    };
    const crossIdx = T.vdw_r6.rel_err.findIndex((v, i) => v > T.yukawa.rel_err[i]);

    const truncDemo = h('div', { class: 'demo' });
    truncDemo.append(h('h3', {}, 'Truncation error against cutoff radius'),
      h('p', { class: 'hint' },
        'Relative error of the truncated lattice sum against the exact one, both axes ' +
        'logarithmic. Move the cutoff to read the three errors off at that radius.'));
    const tPlot = new Plot({ width: 660, height: 340, xLog: true, yLog: true,
      xLabel: 'cutoff radius r_cut  (units of d)', yLabel: 'relative truncation error' });
    tPlot.add({ points: grid.map((r, i) => [r, T.yukawa.rel_err[i]]),
      color: PALETTE[0], width: 2.2, label: `exp(−κr)/r,  κ = ${T.kappa}  (short-ranged)` });
    tPlot.add({ points: grid.map((r, i) => [r, T.vdw_r6.rel_err[i]]),
      color: PALETTE[3], width: 2.2, label: '1/r⁶  (dispersion)' });
    tPlot.add({ points: grid.map((r, i) => [r, T.coulomb.rel_err[i]]),
      color: PALETTE[1], width: 2.2, label: '1/r with alternating charges  (Coulomb)' });
    const marker = { points: [[6, 1e-8], [6, 2e1]], color: '#5A6773', width: 1, dash: '4 4',
      opacity: 0.75 };
    tPlot.add(marker);
    tPlot.setLimits([1, 20], [1e-8, 2e1]);
    const tSvg = tPlot.render();
    const tRead = h('div', { class: 'readout' });

    const setCut = (rc) => {
      const i = idxOf(rc);
      marker.points = [[grid[i], 1e-8], [grid[i], 2e1]];
      tPlot.render();
      const y = T.yukawa.rel_err[i], v = T.vdw_r6.rel_err[i], c = T.coulomb.rel_err[i];
      tRead.textContent =
        `r_cut = ${grid[i].toFixed(2)} d\n\n` +
        `   exp(-r)/r          relative truncation error    ${y.toExponential(3)}\n` +
        `   1/r^6                                           ${v.toExponential(3)}\n` +
        `   1/r alternating                                 ${c.toExponential(3)}\n\n` +
        `   Coulomb error / short-range error               ${(c / y).toExponential(2)} x`;
    };
    truncDemo.append(tSvg, tPlot.legend(), tRead,
      h('div', { class: 'controls' }, slider({
        label: 'cutoff r_cut', min: grid[0], max: grid[grid.length - 1], step: 0.25, value: 6,
        format: (v) => `${v.toFixed(2)} d`, onInput: setCut })));
    root.append(truncDemo);
    setCut(6);

    root.append(
      h('p', { class: 'prose', html:
        `The two convergent curves behave as the continuum estimates say they should. Fitting the ` +
        `truncated tail of $e^{-\\kappa r}/r$ over $r_{\\mathrm{cut}} \\ge 4$ recovers a decay ` +
        `rate of ${T.yukawa.fitted_decay_rate.toFixed(5)} against $-\\kappa = -${T.kappa}$ with ` +
        `$R^2 = ${T.yukawa.fit_r2.toFixed(6)}$, and a prefactor ` +
        `${T.yukawa.fitted_prefactor.toFixed(4)} against $4\\pi\\rho = ` +
        `${T.yukawa.prefactor_theory.toFixed(4)}$. Fitting the $1/r^6$ tail gives a power of ` +
        `${T.vdw_r6.fitted_power.toFixed(5)} against $-3$, and a prefactor ` +
        `${T.vdw_r6.fitted_prefactor.toFixed(4)} against $4\\pi\\rho/3 = ` +
        `${T.vdw_r6.prefactor_theory.toFixed(4)}$. Exponential convergence and power-law ` +
        `convergence respectively — and both of them are convergence, which is the point. For a ` +
        `short-ranged interaction locality is not an approximation anyone need worry about; for ` +
        `dispersion it is an approximation whose size can be quoted.` }),
      h('p', { class: 'prose', html:
        `The crossing is worth noticing. At $r_{\\mathrm{cut}} = 6\\,d$ the exponential tail costs ` +
        `${(100 * T.yukawa.rel_err_at_6d).toFixed(2)}&nbsp;% and the dispersion tail only ` +
        `${(100 * T.vdw_r6.rel_err_at_6d).toFixed(3)}&nbsp;%, so at short range the power law ` +
        `looks like the harmless one. Push the cutoff out and the exponential collapses past it: ` +
        `by $r_{\\mathrm{cut}} = ${grid[crossIdx]}\\,d$ the dispersion error is the larger of the ` +
        `two, and at $12\\,d$ it is ` +
        `${(T.vdw_r6.rel_err_at_12d / T.yukawa.rel_err_at_12d).toFixed(1)}&times; larger ` +
        `(${T.vdw_r6.rel_err_at_12d.toExponential(2)} against ` +
        `${T.yukawa.rel_err_at_12d.toExponential(2)}). A power law never stops mattering; it only ` +
        `stops mattering much.` }),

      h('h2', {}, 'The case that does not converge'),
      h('p', { class: 'prose', html:
        `The third curve answers the reading group’s question. Truncating the alternating $1/r$ ` +
        `lattice sum at a sphere does not converge to anything. At $r_{\\mathrm{cut}} = 6\\,d$ the ` +
        `relative error is ${(100 * T.coulomb.rel_err_at_6d).toFixed(1)}&nbsp;%; at $12\\,d$ it ` +
        `has gone <em>up</em>, to ${(100 * T.coulomb.rel_err_at_12d).toFixed(1)}&nbsp;%. The error ` +
        `changes sign ${T.coulomb.n_sign_changes_of_error} times over the swept range and its ` +
        `envelope grows rather than shrinks: the effective decay exponent is ` +
        `${T.coulomb.effective_decay_exponent.toFixed(3)}, a negative number where the ` +
        `short-ranged case gives ${T.yukawa.effective_decay_exponent.toFixed(1)} and the ` +
        `dispersion case ${T.vdw_r6.effective_decay_exponent.toFixed(3)}. Three regimes, ` +
        `separated cleanly by one number each.` }));

    const coulDemo = h('div', { class: 'demo' });
    coulDemo.append(h('h3', {}, 'The Madelung sum under a spherical cutoff'),
      h('p', { class: 'hint' },
        'The truncated energy per ion against the exact Ewald value. Linear axes; the dashed ' +
        'horizontal line is the answer it is supposed to be converging to.'));
    const cPlot = new Plot({ width: 660, height: 320,
      xLabel: 'cutoff radius r_cut  (units of d)', yLabel: 'truncated energy per ion' });
    cPlot.add({ points: grid.map((r, i) => [r, T.coulomb.U_trunc[i]]),
      color: PALETTE[1], width: 2, markers: true, r: 2.4,
      label: 'spherically truncated pair sum' });
    cPlot.add({ points: [[grid[0], T.coulomb.U_exact_ewald],
      [grid[grid.length - 1], T.coulomb.U_exact_ewald]],
      color: '#1B2733', width: 1.6, dash: '6 4',
      label: `exact Ewald value, −M = ${T.coulomb.U_exact_ewald.toFixed(6)}` });
    cPlot.setLimits([1, 20], [-15, 13]);
    coulDemo.append(cPlot.render(), cPlot.legend(),
      h('div', { class: 'readout' },
        `Ewald reference, written from scratch in the script:\n` +
        `   Madelung constant obtained       ${T.coulomb.madelung_ewald.toFixed(15)}\n` +
        `   published value                  ${T.coulomb.madelung_reference.toFixed(15)}\n` +
        `   absolute error                   ${T.coulomb.madelung_abs_error.toExponential(1)}\n` +
        `   spread over the splitting alpha  ${T.coulomb.alpha_spread.toExponential(2)}\n\n` +
        `truncated sum over r_cut in [10, 20]:\n` +
        `   peak-to-peak swing               ${T.coulomb.U_ptp_over_rc_10_to_20.toFixed(4)}\n` +
        `   magnitude of the answer itself   ${Math.abs(T.coulomb.U_exact_ewald).toFixed(4)}\n` +
        `   swing / answer                   ` +
        `${(T.coulomb.U_ptp_over_rc_10_to_20 / Math.abs(T.coulomb.U_exact_ewald)).toFixed(1)} x`));
    root.append(coulDemo);

    root.append(
      h('p', { class: 'prose', html:
        `Between $r_{\\mathrm{cut}} = 10\\,d$ and $20\\,d$ the truncated sum swings over a range ` +
        `${(T.coulomb.U_ptp_over_rc_10_to_20 / Math.abs(T.coulomb.U_exact_ewald)).toFixed(1)} ` +
        `times larger than the quantity it is estimating. The cause is not slow convergence but ` +
        `no convergence: each coordination shell added by growing the sphere carries a net charge ` +
        `that does not shrink, and the value of a conditionally convergent sum depends on the ` +
        `shape one sums in. Strip out the alternation and the failure is blunter still — the same ` +
        `truncated sum over like charges grows as $2\\pi\\rho\\,r_{\\mathrm{cut}}^2$, matched to ` +
        `${(100 * T.coulomb_same_sign.ratio_max_dev_rc_ge_6).toFixed(1)}&nbsp;% beyond ` +
        `$r_{\\mathrm{cut}} = 6\\,d$, and has no limit whatsoever. What licenses calling any of ` +
        `this an <em>error</em> is the reference: an Ewald sum written from scratch in the ` +
        `script, which returns the published rock-salt Madelung constant ` +
        `$M = ${T.coulomb.madelung_reference.toFixed(15)}$ to an absolute error of ` +
        `${T.coulomb.madelung_abs_error.toExponential(1)} — the identical double — and is ` +
        `independent of its own splitting parameter $\\alpha$ to ` +
        `${T.coulomb.alpha_spread.toExponential(1)}.` }),
      h('p', { class: 'prose', html:
        `At $r_{\\mathrm{cut}} = 12\\,d$ the Coulomb truncation error exceeds the short-range one ` +
        `by a factor ${T.locality_ratio_at_12d.toExponential(2)}. That gap is the quantitative ` +
        `answer to "does locality suffice?". For the interactions a short-ranged potential ` +
        `describes, emphatically yes: the residual truncation error at any sensible cutoff is far ` +
        `below the fitting error of any model. For charged or strongly polar systems, no — and no ` +
        `amount of receptive field fixes it, because the failure is in the summation order rather ` +
        `than in the radius.` }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'what this does and does not prove'),
        h('div', {},
          'The measurement condemns naive spherical truncation of a raw pair sum, which is what a ' +
          'plain cutoff model implements. It does not prove that nothing local can work: ' +
          'charge-neutralised and damped truncation schemes converge perfectly well at modest ' +
          'radii, and machine-learned potentials that need long-range physics bolt on explicit ' +
          'machinery — charge equilibration, or an Ewald or multipole term evaluated outside the ' +
          'network. The lesson is that the remedy has to come from knowing the physics. Adding ' +
          'layers is the one remedy that provably does not work, since the receptive field would ' +
          'have to grow to the size of the crystal.')),
      h('p', { class: 'prose' },
        'This constraint is also architecture-independent, which is worth saying because it is ' +
        'easy to read the last few years as a story about one lineage winning. Every model near ' +
        'the top of today’s leaderboards is local in exactly the sense measured here, whether it ' +
        'descends from the tensor-field-network line, from the Atomic Cluster Expansion, or from ' +
        'an unconstrained transformer that learns its symmetry from augmentation. They differ in ' +
        'what they compute inside the cutoff, not in whether they have one. The truncation error ' +
        'measured above is a floor all of them share.'),

      h('h2', {}, 'A shell is not an environment'),
      h('p', { class: 'prose', html:
        `One last measurement, using the repository’s own $O(3)$ code, which explains a design ` +
        `choice rather than a cost. Every hop shell of this lattice is octahedrally symmetric, so ` +
        `the atom-centred sum $\\sum_j Y_\\ell(\\hat{r}_j)$ taken over a whole shell vanishes ` +
        `identically for $\\ell = 1, 2, 3$ — the largest value over all five shells is ` +
        `${SH.max_norm_l123.toExponential(1)} — and is first non-zero at $\\ell = 4$, where the ` +
        `smallest value over the shells is ${SH.min_norm_l4.toFixed(4)}. That quantity is a ` +
        `genuine invariant, not an accident of orientation: rotating the shell changes its norm ` +
        `by ${SH.max_l4_rotation_dev.toExponential(1)}.` }),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'hop shell'), h('th', { class: 'num' }, 'atoms'),
          h('th', { class: 'num' }, '|Σ Y₁|'), h('th', { class: 'num' }, '|Σ Y₂|'),
          h('th', { class: 'num' }, '|Σ Y₃|'), h('th', { class: 'num' }, '|Σ Y₄|'))),
        h('tbody', {}, SH.shells.map((s) => h('tr', {},
          h('td', {}, `L = ${s.L}`),
          h('td', { class: 'num' }, String(s.n)),
          h('td', { class: 'num' }, s.norm_l1.toExponential(1)),
          h('td', { class: 'num' }, s.norm_l2.toExponential(1)),
          h('td', { class: 'num' }, s.norm_l3.toExponential(1)),
          h('td', { class: 'num' }, s.norm_l4.toFixed(4)))))),
      h('p', { class: 'prose', html:
        `A model that summarised each shell and passed the summary along would be blind to the ` +
        `anisotropy of this environment until it carried $\\ell = 4$ features, and there is no ` +
        `chemistry in that number — it is a fact about the octahedron. This is the argument for ` +
        `keeping features on edges rather than on shells, which is what NequIP and Allegro both ` +
        `do, and it rhymes with chapter 2: what an aggregation destroys depends on the symmetry ` +
        `of the thing being aggregated, and a crystal is exactly the case where that symmetry is ` +
        `maximal.` }),

      h('h2', {}, 'Two checks that were reframed rather than loosened'),
      h('p', { class: 'prose', html:
        `The asymptotic tail laws deserve a note, because the obvious way to check them fails. ` +
        `Comparing the measured tail pointwise against the continuum prediction gives a ratio ` +
        `that oscillates about 1 rather than sitting on it: up to ` +
        `${(100 * T.yukawa.tail_over_pred_max_dev).toFixed(2)}&nbsp;% deviation for the ` +
        `exponential and ${(100 * T.vdw_r6.tail_over_pred_max_dev).toFixed(2)}&nbsp;% for ` +
        `$1/r^6$. Those two numbers failed a tolerance in an earlier version of the script. They ` +
        `are not evidence of a wrong law. They are a lattice being compared with an integral: the ` +
        `crystal adds whole coordination shells at discrete radii while the continuum formula ` +
        `$4\\pi\\rho\\int r^2\\phi(r)\\,\\mathrm{d}r$ is smooth in $r_{\\mathrm{cut}}$.` }),
      h('p', { class: 'prose', html:
        `Rather than widen the tolerance until a pointwise comparison passed, the checks now test ` +
        `the three things the discretisation cannot touch. The fitted decay <em>rate</em> matches, ` +
        `with $R^2 = ${T.yukawa.fit_r2.toFixed(6)}$ for the exponential and ` +
        `$R^2 = ${T.vdw_r6.fit_r2.toFixed(6)}$ for $1/r^6$. The mean ratio is unbiased — ` +
        `${T.yukawa.tail_over_pred_mean.toFixed(4)} and ` +
        `${T.vdw_r6.tail_over_pred_mean.toFixed(4)}, both within ` +
        `${(100 * Math.max(Math.abs(T.yukawa.tail_over_pred_mean - 1), Math.abs(T.vdw_r6.tail_over_pred_mean - 1))).toFixed(1)}` +
        `&nbsp;% of 1 — so the oscillation is scatter and not an offset. And the scatter shrinks ` +
        `as the shells become fine relative to the radius: for $1/r^6$ it falls from ` +
        `${(100 * T.vdw_r6.tail_over_pred_max_dev_near).toFixed(2)}&nbsp;% over ` +
        `$r_{\\mathrm{cut}} \\in [4, 10)$ to ` +
        `${(100 * T.vdw_r6.tail_over_pred_max_dev_far).toFixed(2)}&nbsp;% beyond $10\\,d$, which ` +
        `is what discreteness does and what a wrong exponent would not. All three are in the ` +
        `ledger below, and the raw scatter is quoted in the check text rather than hidden.` }));

    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Check ledger — python/experiments/receptive_field.py'),
      h('p', { class: 'hint' },
        'Reproduce with:  uv run python python/experiments/receptive_field.py'));
    const inner = h('div');
    for (const c of D.checks) inner.append(checkLine(c.passed, c.name, c.detail));
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${D.meta.n_pass === D.meta.n_total ? 'ok' : 'bad'}` },
        `${D.meta.n_pass}/${D.meta.n_total}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);
    root.append(ledger);

    root.append(
      h('p', { class: 'prose', html:
        `Two numbers to carry forward. Reach is linear in depth while what you must own in order ` +
        `to have it is cubic: at a fixed cutoff, ${NA.reach_ratio.toFixed(1)}&times; the radius ` +
        `costs ${NA.halo_ratio.toFixed(1)}&times; the ghost atoms. And the physics is content ` +
        `with a cutoff right up until it is not — ` +
        `${(100 * T.yukawa.rel_err_at_6d).toFixed(2)}&nbsp;% truncation error at $6\\,d$ for a ` +
        `short-ranged interaction, against ${(100 * T.coulomb.rel_err_at_6d).toFixed(0)}&nbsp;% ` +
        `for an unscreened Coulomb one that never improves. The first number is why message ` +
        `passing is worth its cost; the second is why depth is not the axis along which that ` +
        `particular problem gets solved.` }));
  },
};
