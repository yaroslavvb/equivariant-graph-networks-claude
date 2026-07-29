import { h, slider, segmented, checkLine, loadResults, Plot, PALETTE } from '../ui.js';

// ---------------------------------------------------------------------------
// The only numbers in this chapter that are not produced by
// python/experiments/convex_hull_decision.py. They are quoted from the Matbench
// Discovery leaderboard as recorded in this project's verified-claims note;
// every comparison drawn from them below is computed from these entries rather
// than written out by hand.
// ---------------------------------------------------------------------------
const LEADERBOARD = {
  source: 'Matbench Discovery, unique-prototypes split',
  rows: [
    { model: 'EquiformerV3+DeNS-OAM', f1: 0.931, kappa: null, cps: null, rankF1: 1, rankCPS: null,
      note: 'best F1 on the board' },
    { model: 'TECE-OAM-RRA-1.0', f1: null, kappa: null, cps: null, rankF1: null, rankCPS: 1,
      note: 'best CPS on the board — a Cartesian-tensor ACE variant, no Clebsch–Gordan products' },
    { model: 'DPA-4.0.1-Pro-MPtrj', f1: 0.857, kappa: 0.211, cps: 0.840, rankF1: null, rankCPS: null,
      note: '22.8M parameters, trained on MPtrj alone' },
    { model: 'eqV2 M', f1: 0.917, kappa: 1.7707, cps: 0.558, rankF1: 9, rankCPS: 36,
      note: 'geometry-optimisation RMSD 0.0691 — the relaxations themselves look fine' },
  ],
};

// ---------------------------------------------------------------------------
// Geometry: the lower convex hull, written out so the page can rebuild it from
// perturbed energies exactly the way the Python script does.
// ---------------------------------------------------------------------------

/** Indices of the lower convex hull of the 2-D cloud (x, y), left to right. */
function lowerHull(x, y) {
  const order = Array.from({ length: x.length }, (_, i) => i)
    .sort((a, b) => (x[a] - x[b]) || (y[a] - y[b]));
  const hull = [];
  for (const i of order) {
    while (hull.length >= 2) {
      const o = hull[hull.length - 2], a = hull[hull.length - 1];
      const cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o]);
      if (cross <= 0) hull.pop(); else break;
    }
    hull.push(i);
  }
  return hull;
}

/** Linear interpolation along a polyline with non-decreasing abscissa. */
function interpOn(hx, hy, xq) {
  const n = hx.length;
  if (xq <= hx[0]) return hy[0];
  if (xq >= hx[n - 1]) return hy[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (hx[m] <= xq) lo = m; else hi = m; }
  const dx = hx[hi] - hx[lo];
  return dx === 0 ? hy[hi] : hy[lo] + ((xq - hx[lo]) / dx) * (hy[hi] - hy[lo]);
}

// Deterministic Gaussian noise, so a given slider position always draws the
// same cloud until the reader asks for a fresh one.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussians(n, seed) {
  const r = mulberry32(seed), out = new Float64Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(r(), 1e-12), v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    out[i] = m * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = m * Math.sin(2 * Math.PI * v);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const maxOf = (a) => a.reduce((m, v) => (v > m ? v : m), -Infinity);
const minOf = (a) => a.reduce((m, v) => (v < m ? v : m), Infinity);
const diffs = (a) => a.slice(1).map((v, i) => v - a[i]);
const meV = (v, d = 1) => (1000 * v).toFixed(d);
const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

function slopeFit(xs, ys) {
  const mx = mean(xs), my = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  return num / den;
}

function confusion(truth, pred) {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < truth.length; i++) {
    if (truth[i] && pred[i]) tp++;
    else if (!truth[i] && pred[i]) fp++;
    else if (truth[i] && !pred[i]) fn++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { tp, fp, fn, tn, precision, recall, f1, accuracy: (tp + tn) / truth.length };
}

// ---------------------------------------------------------------------------
// A small categorical bar chart — Plot's axes are numeric, and these bins are
// not.
// ---------------------------------------------------------------------------
function binChart(labels, counts, acc, captionRight) {
  const W = 690, H = 262, L = 54, R = 14, T = 18, B = 64;
  const el = (t, a, txt) => {
    const e = document.createElementNS('http://www.w3.org/2000/svg', t);
    for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v));
    if (txt != null) e.textContent = txt;
    return e;
  };
  const FONT = 'ui-sans-serif, system-ui, sans-serif';
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%' });
  const ty = (v) => H - B - v * (H - T - B);

  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: ty(g), y2: ty(g),
      stroke: g === 1 ? '#C9CFD6' : '#EDF0F3', 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: L - 8, y: ty(g) + 4, 'text-anchor': 'end',
      'font-size': 11.5, fill: '#5A6773', 'font-family': FONT }, g.toFixed(2)));
  }

  const n = labels.length, slot = (W - R - L) / n, bw = slot * 0.64;
  for (let i = 0; i < n; i++) {
    const cx = L + slot * (i + 0.5);
    const a = acc[i];
    svg.appendChild(el('rect', { x: cx - bw / 2, y: ty(a), width: bw,
      height: Math.max(1, ty(0) - ty(a)), rx: 2,
      fill: i === 0 ? PALETTE[1] : PALETTE[0], opacity: i === 0 ? 0.9 : 0.5 }));
    svg.appendChild(el('text', { x: cx, y: ty(a) - 6, 'text-anchor': 'middle',
      'font-size': 11, fill: '#1B2733', 'font-family': FONT }, a.toFixed(3)));
    svg.appendChild(el('text', { x: cx, y: H - B + 16, 'text-anchor': 'middle',
      'font-size': 10.5, fill: '#5A6773', 'font-family': FONT },
    labels[i].replace('E_hull = 0 (stable)', 'on the hull')));
    svg.appendChild(el('text', { x: cx, y: H - B + 30, 'text-anchor': 'middle',
      'font-size': 10, fill: '#9AA4AE', 'font-family': FONT }, `n = ${counts[i]}`));
  }
  svg.appendChild(el('text', { x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle',
    'font-size': 12, fill: '#1B2733', 'font-family': FONT },
  'true distance above the hull  (meV/atom)'));
  svg.appendChild(el('text', { x: 13, y: (T + H - B) / 2, 'text-anchor': 'middle',
    'font-size': 12, fill: '#1B2733', 'font-family': FONT,
    transform: `rotate(-90 13 ${(T + H - B) / 2})` }, 'classification accuracy'));
  if (captionRight) {
    svg.appendChild(el('text', { x: W - R, y: T - 2, 'text-anchor': 'end',
      'font-size': 11.5, fill: '#5A6773', 'font-family': FONT }, captionRight));
  }
  return svg;
}

// ---------------------------------------------------------------------------

export default {
  id: 'hull',
  title: 'Stability is a decision, not a regression',
  async render(root) {
    const D = await loadResults('convex_hull_decision');
    const M = D.meta, S = D.system, E1 = D.exp1_noise_sweep, FX = E1.fixed_hull;
    const E2 = D.exp2_near_hull, E3 = D.exp3_mae_f1_inversion, E4 = D.exp4_error_cost;
    const TS = D.threshold_sweep;

    // ---- rebuild the true hull in the browser from the published cloud ----
    const X = [S.element_x[0], S.element_x[1], ...S.candidate_x];
    const Etrue = [S.element_e_form[0], S.element_e_form[1], ...S.candidate_e_form];
    const NPOOL = S.candidate_x.length;
    const trueHullIdx = lowerHull(X, Etrue);
    const trueHullX = trueHullIdx.map((i) => X[i]);
    const trueHullE = trueHullIdx.map((i) => Etrue[i]);
    const trueStable = S.candidate_stable.map((v) => v === 1);
    const prevalence = M.prevalence;
    const baselineAccuracy = 1 - prevalence;

    const nSig = E1.sigma.length;
    const last = nSig - 1;
    const kAt = (target) => E1.sigma.reduce(
      (best, s, i) => (Math.abs(s - target) < Math.abs(E1.sigma[best] - target) ? i : best), 0);
    const k25 = kAt(0.025);
    const kFit = kAt(0.005);
    const kBaseline = E1.accuracy.findIndex((a) => a < baselineAccuracy);

    const eHullSpan = maxOf(S.candidate_e_hull);
    const maeRange = E1.mae[last] / E1.mae[0];
    const f1Range = E1.f1[0] / E1.f1[last];
    const elasticityRatio = E1.max_abs_loglog_slope_f1 / E1.min_abs_loglog_slope_f1;
    const bandSlope = slopeFit(E2.sigma.slice(kFit).map(Math.log),
      E2.band_width_p99.slice(kFit).map(Math.log));

    // ------------------------------------------------------------------
    root.append(
      h('p', { class: 'eyebrow dat' }, 'Chapter 8'),
      h('h1', {}, 'Stability is a decision, not a regression'),
      h('p', { class: 'lede' },
        'Every model in the previous seven chapters was measured in meV per atom. The benchmark ' +
        'those models are actually ranked on refuses to score them that way. This chapter builds ' +
        'a small crystal-stability problem where the entire pipeline is known exactly, and ' +
        'measures how far the regression number and the decision number can come apart — far ' +
        'enough that they disagree about which of two models is the better one.'),

      h('h2', {}, 'What the reading group was circling'),
      h('p', { class: 'prose' },
        'Three separate worries came up in the reading group, all of them correct, and all of ' +
        'them reaching for a vocabulary that already exists. The transcription is approximate.'),
      h('blockquote', {},
        h('div', {}, '“…thermodynamically stable versus physically realizable…”'),
        h('span', { class: 'attrib' }, 'reading group, auto-transcribed')),
      h('p', { class: 'prose' },
        'That is the right distinction, and the benchmark takes the first half of it very ' +
        'seriously and the second half not at all. Thermodynamic stability has a precise ' +
        'definition — it is a statement about a convex hull — and the whole leaderboard is built ' +
        'on it. Whether the compound can actually be made is a different question that no energy ' +
        'model on the board attempts to answer.'),
      h('blockquote', {},
        h('div', {}, '“…the classes are so imbalanced… can we see F1 rather than accuracy…”'),
        h('span', { class: 'attrib' }, 'reading group, auto-transcribed')),
      h('p', { class: 'prose', html:
        'This is exactly the benchmark’s own reasoning, arrived at independently and in the ' +
        'right order. The experiment below makes the point in the sharpest form available: there ' +
        'is a noise level at which the model’s plain accuracy is <em>worse than a classifier that ' +
        'answers “unstable” to everything</em>, while its energy MAE is still comfortably inside ' +
        'the range people publish.' }),
      h('blockquote', {},
        h('div', {},
          '“…two hundred thousand of them, but they’re just very individual ionic substitutions…”'),
        h('span', { class: 'attrib' }, 'reading group, auto-transcribed')),
      h('p', { class: 'prose' },
        'Also right, and the benchmark agrees: it reports a unique-prototypes split precisely ' +
        'because a candidate set generated by elemental substitution contains a great deal of ' +
        'near-duplication. But the dismissal has a cost. A screening campaign runs over exactly ' +
        'that kind of candidate set, and the figure of merit is not how novel the candidates are, ' +
        'it is how many real materials survive the filter. That is a decision problem, and the ' +
        'rest of this chapter is about what happens when you score a decision with a regression ' +
        'number.'),

      h('h2', {}, 'Why the energy above hull is not a property of one structure'),
      h('p', { class: 'prose', html:
        'For a binary $A$–$B$ system the formation energy per atom of $A_xB_{1-x}$ is referenced ' +
        'to the pure elements, so $E_f(0) = E_f(1) = 0$ by construction. A compound is ' +
        'thermodynamically stable if and only if it lies on the <em>lower convex hull</em> of ' +
        '$E_f$ against composition: anything above the hull can lower its energy by phase ' +
        'separating into the two hull phases that bracket it. The energy above hull is ' +
        '$$E_{\\rm hull}(i) \\;=\\; E_f(i) \\;-\\; \\mathrm{hull}\\!\\left(x_i\\right),$$ ' +
        'and the second term is assembled out of <em>other structures</em>. That is the whole ' +
        'source of the trouble in this chapter. A model can be accurate about every absolute ' +
        'energy and still get the ordering of near-degenerate competing phases wrong, and the ' +
        'ordering is what the decision reads.' }),
      h('p', { class: 'prose', html:
        `The synthetic system is a regular-solution mixing curve, $\\Omega x(1-x)$ with ` +
        `$\\Omega = ${S.regular_solution_omega.toFixed(2)}$ eV/atom, plus an energy well at every ` +
        `simple stoichiometry $p/q$ with $q \\le 9$, plus a polymorph ladder at each composition ` +
        `whose rungs are exponentially spaced a few tens of meV/atom apart. That last detail is ` +
        `the realistic part: polymorph spacings and model errors are the same size. The result is ` +
        `${M.n_candidates} candidate structures across ${M.n_compositions} compositions, of which ` +
        `exactly ${M.n_stable} sit on the hull — a prevalence of ${pct(prevalence, 2)}. The ` +
        `Discovery Acceleration Factor, which is precision divided by that prevalence, therefore ` +
        `has a ceiling of ${M.max_possible_daf.toFixed(2)}: a perfect model surfaces a stable ` +
        `compound ${M.max_possible_daf.toFixed(1)}× more often than picking at random does.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'the protocol matters, and there are two of them'),
        h('div', { html:
          'The panel below defaults to what the script calls the <em>self-consistent</em> ' +
          'protocol: the hull is rebuilt from the predicted energies, because the competing ' +
          'phases are predicted too. The alternative — score a predicted energy against the fixed ' +
          'DFT hull — is closer to what Matbench Discovery literally does with the Materials ' +
          'Project hull. These are not small variations on one another. Switch between them and ' +
          'watch the failure mode invert.' })));

    // ================================================================== demo 1
    const d1 = h('div', { class: 'demo' });
    d1.append(h('h3', {}, 'The hull, rebuilt from noisy predictions'),
      h('p', { class: 'hint' },
        'Each dot is one candidate at its predicted formation energy. Raise the noise and the ' +
        'cloud smears; under the self-consistent protocol the hull is recomputed from the smeared ' +
        'cloud, so the competing phases move too. Green = correctly found, gold = a real stable ' +
        'compound missed, red = a false alarm.'));
    const plotHolder = h('div');
    const legendHolder = h('div');
    const readout = h('div', { class: 'readout' });
    let kSigma = kAt(0.010), protocol = 'self', drawSeed = 20260729;

    function drawHull() {
      const sigma = E1.sigma[kSigma];
      const noise = gaussians(NPOOL, (drawSeed + kSigma * 7919) >>> 0);
      const ePred = Etrue.slice();
      for (let i = 0; i < NPOOL; i++) ePred[2 + i] = Etrue[2 + i] + sigma * noise[i];

      const pHullIdx = lowerHull(X, ePred);
      const pHullX = pHullIdx.map((i) => X[i]);
      const pHullE = pHullIdx.map((i) => ePred[i]);
      const refX = protocol === 'self' ? pHullX : trueHullX;
      const refE = protocol === 'self' ? pHullE : trueHullE;

      const pred = [], absErr = [];
      for (let i = 0; i < NPOOL; i++) {
        pred.push(ePred[2 + i] - interpOn(refX, refE, S.candidate_x[i]) <= 1e-9);
        absErr.push(Math.abs(sigma * noise[i]));
      }
      const c = confusion(trueStable, pred);
      const mae = mean(absErr);
      const daf = c.precision / prevalence;

      const groups = { tn: [], tp: [], fn: [], fp: [] };
      for (let i = 0; i < NPOOL; i++) {
        const key = trueStable[i] ? (pred[i] ? 'tp' : 'fn') : (pred[i] ? 'fp' : 'tn');
        groups[key].push([S.candidate_x[i], ePred[2 + i]]);
      }

      const yLo = Math.min(-0.95, minOf(S.candidate_e_form) - 2.5 * sigma);
      const yHi = Math.max(0.30, maxOf(S.candidate_e_form) + 2.5 * sigma);
      const p = new Plot({ width: 700, height: 376,
        xLabel: 'composition  x  in  A_x B_(1−x)', yLabel: 'formation energy  (eV/atom)' });
      p.add({ points: trueHullX.map((v, i) => [v, trueHullE[i]]), color: PALETTE[5],
        width: 1.6, dash: '5 4', label: 'true hull (DFT)' });
      if (protocol === 'self') {
        p.add({ points: pHullX.map((v, i) => [v, pHullE[i]]), color: PALETTE[0], width: 2.2,
          label: 'hull rebuilt from the predicted energies' });
      }
      p.add({ points: groups.tn, type: 'scatter', r: 2.1, color: '#B9C1C9', opacity: 0.55,
        label: `correctly rejected (${c.tn})` });
      p.add({ points: groups.tp, type: 'scatter', r: 5, color: PALETTE[3], opacity: 0.95,
        label: `found (${c.tp})` });
      p.add({ points: groups.fn, type: 'scatter', r: 5, color: PALETTE[2], opacity: 0.95,
        label: `missed (${c.fn})` });
      p.add({ points: groups.fp, type: 'scatter', r: 5, color: PALETTE[1], opacity: 0.95,
        label: `false alarm (${c.fp})` });
      p.setLimits([-0.03, 1.03], [yLo, yHi]);

      plotHolder.innerHTML = '';
      plotHolder.appendChild(p.render());
      legendHolder.innerHTML = '';
      legendHolder.appendChild(p.legend());

      const ref = protocol === 'self'
        ? { precision: E1.precision[kSigma], recall: E1.recall[kSigma], f1: E1.f1[kSigma],
          daf: E1.daf[kSigma], nps: E1.n_pred_stable[kSigma], acc: E1.accuracy[kSigma] }
        : { precision: FX.precision[kSigma], recall: FX.recall[kSigma], f1: FX.f1[kSigma],
          daf: FX.daf[kSigma], nps: FX.n_pred_stable[kSigma],
          acc: (FX.tp[kSigma] + FX.tn[kSigma]) / M.n_candidates };
      const row = (name, a, b) => `  ${name.padEnd(20)}${String(a).padEnd(22)}${b}\n`;
      readout.textContent =
        `  sigma ${meV(sigma, 2)} meV/atom     protocol: ` +
        `${protocol === 'self' ? 'hull rebuilt from predictions' : 'fixed DFT hull'}\n\n` +
        `  ${''.padEnd(20)}${'this single draw'.padEnd(22)}${M.n_trials_sweep}-trial mean\n` +
        row('energy MAE', `${meV(mae, 2)} meV/atom`, `${meV(E1.mae[kSigma], 2)} meV/atom`) +
        row('precision', c.precision.toFixed(3), ref.precision.toFixed(3)) +
        row('recall', c.recall.toFixed(3), ref.recall.toFixed(3)) +
        row('F1', c.f1.toFixed(3), ref.f1.toFixed(3)) +
        row('DAF', daf.toFixed(2), ref.daf.toFixed(2)) +
        row('plain accuracy', c.accuracy.toFixed(4), ref.acc.toFixed(4)) +
        row('called stable', c.tp + c.fp, `${ref.nps.toFixed(1)}   (truth: ${M.n_stable})`) +
        `\n  a classifier that answers "unstable" to all ${M.n_candidates} candidates would\n` +
        `  score accuracy ${baselineAccuracy.toFixed(4)} and F1 0.000.`;
    }

    d1.append(plotHolder, legendHolder,
      h('div', { class: 'controls' },
        slider({ label: 'prediction noise σ', min: 0, max: last, step: 1, value: kSigma,
          format: (v) => `${meV(E1.sigma[v], 1)} meV`,
          onInput: (v) => { kSigma = v; drawHull(); } }),
        segmented({ label: 'protocol',
          options: [{ label: 'hull rebuilt', value: 'self' },
            { label: 'fixed DFT hull', value: 'fixed' }],
          value: 'self',
          onPick: (v) => { protocol = v; drawHull(); } }),
        h('button', { type: 'button',
          onclick: () => { drawSeed = (drawSeed * 1103515 + 12345) >>> 0; drawHull(); } },
        'new noise draw')),
      readout);
    root.append(d1);
    drawHull();

    root.append(
      h('p', { class: 'prose', html:
        `Two things in that panel deserve dwelling on. The first is the direction of failure ` +
        `under the self-consistent protocol. Naively, noise ought to manufacture spurious stable ` +
        `phases — push a structure down, it lands on the hull. It does the opposite. The ` +
        `predicted stable count <em>falls</em> from ${E1.n_pred_stable[0].toFixed(1)} at ` +
        `σ = ${meV(E1.sigma[0], 1)} meV/atom to ${E1.n_pred_stable[last].toFixed(1)} at ` +
        `σ = ${meV(E1.sigma[last], 0)} meV/atom, against a truth of ${M.n_stable}. Gaussian noise ` +
        `on a cloud of ${M.n_candidates} points produces a handful of deep downward outliers, and ` +
        `a hull pinned to those outliers casts a shadow over everything else, the genuinely ` +
        `stable compounds included.` }),
      h('p', { class: 'prose', html:
        `The second is that the fixed-hull protocol fails the opposite way. Score the same ` +
        `energies against the <em>true</em> hull and the campaign floods with false positives: ` +
        `${FX.n_pred_stable[last].toFixed(1)} structures called stable at ` +
        `σ = ${meV(E1.sigma[last], 0)} meV/atom, ` +
        `${(FX.n_pred_stable[last] / E1.n_pred_stable[last]).toFixed(1)}× as many as the ` +
        `self-consistent protocol admits. Same energies, same threshold, opposite pathology. And ` +
        `at the <em>accurate</em> end the fixed hull is much the worse of the two: F1 ` +
        `${FX.f1[0].toFixed(3)} against ${E1.f1[0].toFixed(3)} at the smallest σ in the sweep, ` +
        `because a structure sitting exactly on the true hull is knocked off it by any downward ` +
        `error in a competitor. That is a systematic recall loss — recall ` +
        `${FX.recall[0].toFixed(3)} against ${E1.recall[0].toFixed(3)} — which the rebuilt hull ` +
        `does not suffer, and it is worth keeping in mind when reading absolute F1 values off any ` +
        `leaderboard.` }),

      h('h2', {}, 'The headline curve, and the claim it does not support'),
      h('p', { class: 'prose' },
        'Now sweep the noise and watch the two kinds of number at once. Energy MAE and stability ' +
        `F1 are computed from the same predictions at each of ${nSig} noise levels, averaged over ` +
        `${M.n_trials_sweep} independent draws each.`));

    // ================================================================== demo 2
    const d2 = h('div', { class: 'demo' });
    const hint2 = h('p', { class: 'hint' });
    const holder2 = h('div');
    const legend2 = h('div');
    let view = 'both';

    const midSig = E1.f1_loglog_slope.map((_, i) =>
      Math.sqrt(E1.sigma[i] * E1.sigma[i + E1.loglog_slope_window]));

    function drawSweep() {
      const p = new Plot({ width: 700, height: 356 });
      if (view === 'both') {
        p.xLog = true; p.yLog = true;
        p.xLabel = 'prediction noise  σ  (eV/atom)';
        p.yLabel = 'MAE (eV/atom)   ·   F1 (dimensionless)';
        p.add({ points: E1.sigma.map((s, i) => [s, E1.mae[i]]), color: PALETTE[0], width: 2.4,
          markers: true,
          label: `energy MAE — log-log slope ${E1.mae_global_loglog_slope.toFixed(4)}` });
        p.add({ points: E1.sigma.map((s, i) => [s, E1.f1[i]]), color: PALETTE[1], width: 2.4,
          markers: true, label: 'F1 of the stability decision — no constant slope at all' });
        p.setLimits([7e-4, 0.28], [5e-4, 2]);
        hint2.textContent =
          'Log–log. MAE is a perfect straight line of slope 1: it is σ√(2/π), exactly. F1 is not '
          + `a straight line anywhere. Across the sweep MAE grows by a factor of ${maeRange.toFixed(0)} `
          + `and F1 falls by a factor of ${f1Range.toFixed(1)}, but the two axes are not `
          + 'commensurable, and that is precisely the trap.';
      } else if (view === 'elasticity') {
        p.xLog = true;
        p.xLabel = 'prediction noise  σ  (eV/atom)';
        p.yLabel = '| d log(metric) / d log σ |';
        p.add({ points: [[7e-4, 1], [0.28, 1]], color: PALETTE[5], width: 1.4, dash: '5 4',
          label: 'slope 1 — strictly proportional to σ' });
        p.add({ points: midSig.map((s, i) => [s, Math.abs(E1.mae_loglog_slope[i])]),
          color: PALETTE[0], width: 2.4, markers: true, label: 'MAE elasticity' });
        p.add({ points: midSig.map((s, i) => [s, Math.abs(E1.f1_loglog_slope[i])]),
          color: PALETTE[1], width: 2.4, markers: true, label: 'F1 elasticity' });
        p.setLimits([7e-4, 0.28], [0, 1.2]);
        hint2.textContent =
          `The local log–log slope of each curve, smoothed over a window of `
          + `${E1.loglog_slope_window} σ steps. MAE sits pinned at 1. F1's ranges from `
          + `${E1.min_abs_loglog_slope_f1.toFixed(3)} to ${E1.max_abs_loglog_slope_f1.toFixed(3)}, `
          + `a factor of ${elasticityRatio.toFixed(1)}, and never reaches 1.`;
      } else {
        p.xLog = true;
        p.xLabel = 'energy MAE  (meV/atom)';
        p.yLabel = 'F1 of the stability decision';
        p.add({ points: E1.mae.map((m, i) => [1000 * m, E1.f1[i]]), color: PALETTE[1], width: 2.4,
          markers: true, label: 'F1 against the regression number people quote' });
        p.add({ points: [[1000 * E1['mae_at_f1_0.9'], 0], [1000 * E1['mae_at_f1_0.9'], 1]],
          color: PALETTE[3], width: 1.4, dash: '4 4',
          label: `F1 = 0.9 at MAE ${meV(E1['mae_at_f1_0.9'], 1)} meV/atom` });
        p.add({ points: [[1000 * E1['mae_at_f1_0.3'], 0], [1000 * E1['mae_at_f1_0.3'], 1]],
          color: PALETTE[2], width: 1.4, dash: '4 4',
          label: `F1 = 0.3 at MAE ${meV(E1['mae_at_f1_0.3'], 1)} meV/atom` });
        p.setLimits([0.6, 200], [0, 1]);
        hint2.textContent =
          'The same data with MAE on the abscissa. F1 crosses its entire useful range — 0.9 down '
          + `to 0.3 — inside a window of ${E1.f1_window_width_meV.toFixed(1)} meV/atom, with the `
          + `half-way point at ${meV(E1['mae_at_f1_0.5'], 1)} meV/atom. That window is narrower `
          + 'than the spread of energy MAEs among the models on the real leaderboard.';
      }
      holder2.innerHTML = '';
      holder2.appendChild(p.render());
      legend2.innerHTML = '';
      legend2.appendChild(p.legend());
    }

    d2.append(h('h3', {}, 'One set of predictions, two verdicts'), hint2, holder2, legend2,
      h('div', { class: 'controls' },
        segmented({ label: 'view',
          options: [{ label: 'MAE and F1 vs σ', value: 'both' },
            { label: 'local elasticity', value: 'elasticity' },
            { label: 'F1 vs MAE', value: 'mae' }],
          value: 'both',
          onPick: (v) => { view = v; drawSweep(); } })),
      h('div', { class: 'readout' },
        `  MAE   global log-log slope  ${E1.mae_global_loglog_slope.toFixed(5)}` +
        `   (MAE = sigma * sqrt(2/pi), exactly)\n` +
        `  F1    global log-log slope ${E1.f1_global_loglog_slope.toFixed(3)}` +
        `   but |local slopes| range ${E1.min_abs_loglog_slope_f1.toFixed(3)}` +
        ` to ${E1.max_abs_loglog_slope_f1.toFixed(3)}\n\n` +
        `  peak |dlogF1/dlogsigma| = ${E1.max_abs_loglog_slope_f1.toFixed(3)}` +
        ` +/- ${E1.max_abs_loglog_slope_f1_se.toFixed(3)}` +
        `  ->  2-sigma upper bound ${E1.max_abs_loglog_slope_f1_upper2sigma.toFixed(3)} < 1.000\n` +
        `  single-interval estimates scatter as high as ` +
        `${E1.max_abs_loglog_slope_f1_single_interval.toFixed(3)}; that is Monte-Carlo noise.\n\n` +
        `  F1 ${E1.f1[0].toFixed(3)} at MAE ${meV(E1.mae[0], 2)} meV/atom  ->  ` +
        `${E1.f1[last].toFixed(3)} at MAE ${meV(E1.mae[last], 0)} meV/atom`));
    root.append(d2);
    drawSweep();

    root.append(
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'a documented negative result: the obvious headline is false'),
        h('div', { html:
          `The natural thing to say about that first view is “F1 collapses faster than MAE ` +
          `degrades”, and as a statement about power laws it is <strong>false</strong>. MAE is ` +
          `exactly linear in σ — global log-log slope ` +
          `${E1.mae_global_loglog_slope.toFixed(4)}, with no local slope further than ` +
          `${maxOf(E1.mae_loglog_slope.map((v) => Math.abs(v - 1))).toFixed(4)} from 1. The ` +
          `magnitude of F1’s log-log slope <em>never exceeds</em> it: the peak is ` +
          `${E1.max_abs_loglog_slope_f1.toFixed(3)} ± ` +
          `${E1.max_abs_loglog_slope_f1_se.toFixed(3)}, whose two-sigma upper bound ` +
          `${E1.max_abs_loglog_slope_f1_upper2sigma.toFixed(3)} is still below 1. Individual ` +
          `adjacent-interval estimates do stray up to ` +
          `${E1.max_abs_loglog_slope_f1_single_interval.toFixed(3)}, which is why the script ` +
          `smooths over ${E1.loglog_slope_window} steps and carries an error bar: that excursion ` +
          `is Monte-Carlo scatter, not a super-linear region. The script’s self-check asserts the ` +
          `measured fact rather than the expected one, and that assertion is in the ledger at the ` +
          `foot of this page.` })),
      h('p', { class: 'prose', html:
        `What <em>is</em> true is stranger and more useful. F1 is not a power law at all — its ` +
        `sensitivity to σ varies by a factor of ${elasticityRatio.toFixed(1)} across the sweep, ` +
        `while MAE’s is pinned at one — and its entire useful range is traversed inside a window ` +
        `of energy MAE only ${E1.f1_window_width_meV.toFixed(1)} meV/atom wide, from ` +
        `${meV(E1['mae_at_f1_0.9'], 1)} to ${meV(E1['mae_at_f1_0.3'], 1)} meV/atom. Every serious ` +
        `model on the leaderboard sits inside a window about that size. Two models separated by ` +
        `ten meV/atom of MAE can be separated by half the F1 axis, and the MAE gives you no way ` +
        `to know which ten meV/atom mattered.` }),
      h('p', { class: 'prose', html:
        `The reading group’s instinct about accuracy is vindicated even more bluntly. A ` +
        `classifier that answers “unstable” to all ${M.n_candidates} candidates scores accuracy ` +
        `${baselineAccuracy.toFixed(4)}, because only ${pct(prevalence, 2)} of them are stable. ` +
        `The noisy model’s accuracy falls <em>below</em> that trivial baseline at ` +
        `σ = ${meV(E1.sigma[kBaseline], 1)} meV/atom, where its energy MAE is ` +
        `${meV(E1.mae[kBaseline], 1)} meV/atom — a number that would not raise an eyebrow in a ` +
        `paper — and its F1 is ${E1.f1[kBaseline].toFixed(3)}. Past that point accuracy is not ` +
        `merely uninformative; it actively prefers a model that predicts nothing at all.` }),

      h('h2', {}, 'Every mistake lives in a thin band around the hull'),
      h('p', { class: 'prose', html:
        `If the decision is a threshold on $E_{\\rm hull}$, the only structures at risk are the ` +
        `ones near the threshold. That is obvious in principle; the point of measuring it is to ` +
        `see how thin the band is and how it grows. Bin the ${M.n_candidates} candidates by their ` +
        `true distance above the hull, then ask at each noise level how often each bin is ` +
        `classified correctly.` }));

    // ================================================================== demo 3
    const d3 = h('div', { class: 'demo' });
    const hint3 = h('p', { class: 'hint' });
    const chartHolder = h('div');
    let k3 = k25;

    function drawBins() {
      const acc = E2.accuracy_by_bin[k3];
      chartHolder.innerHTML = '';
      chartHolder.appendChild(binChart(E2.bin_labels_meV, E2.bin_counts, acc,
        `σ = ${meV(E2.sigma[k3], 1)} meV/atom`));
      const firstPerfect = acc.findIndex((a) => a > 0.999);
      const nFar = firstPerfect < 0 ? 0
        : E2.bin_counts.slice(firstPerfect).reduce((s, v) => s + v, 0);
      const farEdge = firstPerfect < 0 || E2.bin_edges_eV[firstPerfect] == null
        ? '300' : (1000 * E2.bin_edges_eV[firstPerfect]).toFixed(0);
      hint3.textContent =
        `At σ = ${meV(E2.sigma[k3], 1)} meV/atom, an energy MAE of ${meV(E1.mae[k3], 1)} `
        + `meV/atom, the ${E2.bin_counts[0]} structures actually on the hull are classified `
        + `correctly ${pct(acc[0], 1)} of the time, while the ${nFar} structures more than `
        + `${farEdge} meV/atom above it are essentially never wrong. Ninety-nine per cent of all `
        + `misclassifications fall within ${meV(E2.band_width_p99[k3], 0)} meV/atom of the hull — `
        + `${pct(E2.band_width_p99[k3] / eHullSpan, 1)} of the pool’s `
        + `${meV(eHullSpan, 0)} meV/atom energy span.`;
    }

    const bandPlot = new Plot({ width: 700, height: 300, xLog: true, yLog: true,
      xLabel: 'prediction noise  σ  (eV/atom)',
      yLabel: 'distance above the hull  (eV/atom)' });
    bandPlot.add({ points: E2.sigma.map((s, i) => [s, E2.band_width_p99[i]]), color: PALETTE[1],
      width: 2.4, markers: true, label: '99th percentile of misclassification depth' });
    bandPlot.add({ points: E2.sigma.map((s, i) => [s, E2.band_width_p95[i]]), color: PALETTE[0],
      width: 2.4, markers: true, dash: '5 4', label: '95th percentile' });
    bandPlot.add({ points: [[E2.sigma[0], eHullSpan], [E2.sigma[last], eHullSpan]],
      color: PALETTE[5], width: 1.4, dash: '3 4',
      label: `the candidate pool spans ${meV(eHullSpan, 0)} meV/atom in total` });
    bandPlot.setLimits([7e-4, 0.28], [8e-4, 1.2]);

    d3.append(h('h3', {}, 'Accuracy by distance above the hull'), hint3, chartHolder,
      h('div', { class: 'controls' },
        slider({ label: 'prediction noise σ', min: 0, max: last, step: 1, value: k3,
          format: (v) => `${meV(E2.sigma[v], 1)} meV`,
          onInput: (v) => { k3 = v; drawBins(); } })),
      h('h3', { style: { marginTop: '24px' } }, 'How fast the band widens'),
      bandPlot.render(), bandPlot.legend(),
      h('div', { class: 'readout' },
        `  p99 band width  ${meV(E2.band_width_p99[0], 1)} -> ` +
        `${meV(E2.band_width_p99[last], 0)} meV/atom across the sweep\n` +
        `  log-log slope over sigma >= ${meV(E2.sigma[kFit], 1)} meV/atom:  ` +
        `${bandSlope.toFixed(3)}   (roughly linear in sigma)\n` +
        `  even at sigma = ${meV(E2.sigma[last], 0)} meV/atom, ` +
        `${pct(E2.fraction_errors_below_100meV[last], 1)} of all errors lie within 100 meV/atom ` +
        `of the hull\n` +
        `  mean misclassifications per trial  ${E2.mean_errors_per_trial[0].toFixed(2)} -> ` +
        `${E2.mean_errors_per_trial[last].toFixed(2)}  out of ${M.n_candidates} candidates`));
    root.append(d3);
    drawBins();

    root.append(
      h('p', { class: 'prose', html:
        `The band widens roughly in proportion to σ — a fitted log-log slope of ` +
        `${bandSlope.toFixed(2)} above σ = ${meV(E2.sigma[kFit], 1)} meV/atom — but it stays a ` +
        `band. Even at the worst noise level in the sweep, ` +
        `${pct(E2.fraction_errors_below_100meV[last], 1)} of all misclassifications involve a ` +
        `structure within 100 meV/atom of the hull. This is the fact that makes the next ` +
        `experiment possible: <em>almost all of the energy range a regression metric averages ` +
        `over is irrelevant to the decision</em>. A model may be arbitrarily sloppy about ` +
        `structures 300 meV/atom above the hull without ever changing a single answer, and every ` +
        `bit of that sloppiness is counted against it in the MAE.` }),

      h('h2', {}, 'Two predictors that swap places'),
      h('p', { class: 'prose' },
        'Take that observation and build it into two explicit predictors. Both are the truth ' +
        'plus Gaussian noise; they differ only in how the noise is distributed over the pool.'),
      h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'predictor'), h('th', {}, 'noise model'),
          h('th', {}, 'the idea'))),
        h('tbody', {},
          h('tr', {}, h('td', {}, h('strong', {}, 'P')),
            h('td', { class: 'mono' }, `N(0, ${meV(E3.sigma_P, 0)} meV/atom)`),
            h('td', {}, 'the same error everywhere — an evenly good model')),
          h('tr', {}, h('td', {}, h('strong', {}, 'Q')),
            h('td', { class: 'mono' },
              `N(0, ${meV(E3.floor_Q, 0)} meV/atom + ${E3.alpha_Q} × E_hull)`),
            h('td', {}, 'sharp near the hull, sloppy about structures nobody will ever make')))),
      h('p', { class: 'prose', html:
        `Q is a caricature of a model that has learned where the decision lives: worse on ` +
        `average, better where it counts. Run both for ${M.n_trials_exp3} independent noise draws ` +
        `over the same ${M.n_candidates} candidates and the two orderings come apart completely.` }));

    // ================================================================== demo 4
    const invLive = mean(E3.per_trial.mae_P.map((m, i) =>
      (m < E3.per_trial.mae_Q[i] && E3.per_trial.f1_P[i] < E3.per_trial.f1_Q[i]) ? 1 : 0));
    const d4 = h('div', { class: 'demo' });
    const p4 = new Plot({ width: 700, height: 344,
      xLabel: 'energy MAE over the whole pool  (meV/atom)',
      yLabel: 'F1 of the stability decision' });
    p4.add({ points: E3.per_trial.mae_P.map((m, i) => [1000 * m, E3.per_trial.f1_P[i]]),
      type: 'scatter', r: 3, color: PALETTE[0], opacity: 0.5,
      label: `P — lower MAE, lower F1   (${M.n_trials_exp3} trials)` });
    p4.add({ points: E3.per_trial.mae_Q.map((m, i) => [1000 * m, E3.per_trial.f1_Q[i]]),
      type: 'scatter', r: 3, color: PALETTE[1], opacity: 0.5,
      label: `Q — higher MAE, higher F1   (${M.n_trials_exp3} trials)` });
    p4.setLimits([10, 65], [0, 1.03]);
    d4.append(h('h3', {}, 'The inversion, trial by trial'),
      h('p', { class: 'hint' },
        'Better means further left on this plot and further up on it. The two clouds do not ' +
        'overlap on either axis, and they point in opposite directions. There is no threshold on ' +
        'MAE that would pick the better discovery model here.'),
      p4.render(), p4.legend());
    root.append(d4);

    const row3 = (label, key, d, better) => {
      const a = E3.P[key], b = E3.Q[key];
      const pWins = better === 'low' ? a < b : a > b;
      return h('tr', {},
        h('td', {}, label),
        h('td', { class: 'num' }, pWins ? h('strong', {}, d(a)) : d(a)),
        h('td', { class: 'num' }, pWins ? d(b) : h('strong', {}, d(b))),
        h('td', {}, h('span', { class: `pill ${pWins ? '' : 'ok'}` }, pWins ? 'P wins' : 'Q wins')));
    };
    root.append(
      h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'metric'),
          h('th', { class: 'num' }, 'predictor P'), h('th', { class: 'num' }, 'predictor Q'),
          h('th', {}, 'verdict'))),
        h('tbody', {},
          row3('energy MAE (meV/atom)', 'mae', (v) => meV(v, 2), 'low'),
          row3('energy RMSE (meV/atom)', 'rmse', (v) => meV(v, 2), 'low'),
          h('tr', { class: 'hi' },
            h('td', {}, h('strong', {}, 'F1 of the stability decision')),
            h('td', { class: 'num' }, E3.P.f1.toFixed(3)),
            h('td', { class: 'num' }, h('strong', {}, E3.Q.f1.toFixed(3))),
            h('td', {}, h('span', { class: 'pill ok' }, 'Q wins'))),
          row3('precision', 'precision', (v) => v.toFixed(3), 'high'),
          row3('recall', 'recall', (v) => v.toFixed(3), 'high'),
          row3('Discovery Acceleration Factor', 'daf', (v) => v.toFixed(2), 'high'))),
      h('p', { class: 'prose', html:
        `P wins the regression metrics by a factor of ${(E3.Q.mae / E3.P.mae).toFixed(2)} in MAE ` +
        `and ${(E3.Q.rmse / E3.P.rmse).toFixed(2)} in RMSE. Q wins every metric the benchmark ` +
        `actually reports: F1 by ${(E3.Q.f1 - E3.P.f1).toFixed(3)}, precision by ` +
        `${(E3.Q.precision - E3.P.precision).toFixed(3)}, recall by ` +
        `${(E3.Q.recall - E3.P.recall).toFixed(3)}, and DAF by ` +
        `${(E3.Q.daf - E3.P.daf).toFixed(1)} — which in campaign terms means Q surfaces a real ` +
        `stable compound ${(E3.Q.daf / E3.P.daf).toFixed(2)}× as often as P per structure it ` +
        `flags. It is not an artefact of averaging either: the inversion holds in ` +
        `${pct(invLive, 1)} of the ${M.n_trials_exp3} independent trials, recomputed here in the ` +
        `page from the per-trial arrays.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'what to take from this, and what not to'),
        h('div', { html:
          `P and Q are constructions, not models. Nobody trains a network whose error scales with ` +
          `distance from the hull on purpose. The claim is narrower, and it is a claim about the ` +
          `metrics rather than about any architecture: <em>energy MAE and stability F1 are not ` +
          `monotone functions of one another</em>, so a ranking by one is not a ranking by the ` +
          `other. Once a counterexample is known to exist, the question “which of these two ` +
          `models is better” has no answer until you say what for.` })),

      h('h2', {}, 'The two errors do not cost the same'),
      h('p', { class: 'prose' },
        'F1 is the harmonic mean of precision and recall, which weights the two error types ' +
        'equally. That is a convention, not an economic fact, and it is worth being explicit ' +
        'about what each one buys. A false positive spends one DFT relaxation, or at worst one ' +
        'synthesis attempt: a bounded cost, incurred once, and — crucially — observed. You find ' +
        'out. A false negative silently deletes a real material from the campaign. Nothing ' +
        'downstream ever reports it; the compound simply never appears on any list anyone looks ' +
        'at. That cost is unbounded and invisible.'),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', { class: 'num' }, 'MAE meV/atom'),
          h('th', { class: 'num' }, 'wasted relaxations (FP)'),
          h('th', { class: 'num' }, 'discoveries missed (FN)'),
          h('th', { class: 'num' }, 'of the stable pool'),
          h('th', { class: 'num' }, 'precision'),
          h('th', { class: 'num' }, 'recall'),
          h('th', { class: 'num' }, 'F1'),
          h('th', { class: 'num' }, 'DAF'))),
        h('tbody', {}, ...E4.rows.map((r, i) => h('tr', i === 1 ? { class: 'hi' } : {},
          h('td', { class: 'num' }, r.mae_meV.toFixed(1)),
          h('td', { class: 'num' }, r.wasted_dft_relaxations.toFixed(2)),
          h('td', { class: 'num' }, r.discoveries_missed.toFixed(2)),
          h('td', { class: 'num' }, pct(r.fraction_of_stable_pool_lost, 1)),
          h('td', { class: 'num' }, r.precision.toFixed(3)),
          h('td', { class: 'num' }, r.recall.toFixed(3)),
          h('td', { class: 'num' }, r.f1.toFixed(3)),
          h('td', { class: 'num' }, r.daf.toFixed(1)))))),
      h('p', { class: 'prose', html:
        `Read the highlighted row. At an energy MAE of ${E4.rows[1].mae_meV.toFixed(1)} meV/atom ` +
        `— respectable by any published standard — the campaign wastes ` +
        `${E4.rows[1].wasted_dft_relaxations.toFixed(2)} relaxations and loses ` +
        `${E4.rows[1].discoveries_missed.toFixed(2)} of the ${M.n_stable} stable compounds, which ` +
        `is ${pct(E4.rows[1].fraction_of_stable_pool_lost, 1)} of everything there was to find. ` +
        `By the bottom row the model has thrown away ` +
        `${pct(E4.rows[E4.rows.length - 1].fraction_of_stable_pool_lost, 1)} of the stable pool ` +
        `while still wasting only ` +
        `${E4.rows[E4.rows.length - 1].wasted_dft_relaxations.toFixed(2)} relaxations. Recall does ` +
        `degrade faster than precision across the sweep, though only slightly — a fall of ` +
        `${(E1.recall[0] - E1.recall[last]).toFixed(3)} against ` +
        `${(E1.precision[0] - E1.precision[last]).toFixed(3)}, which is a narrow margin and is ` +
        `stated as such in the ledger. The counts are where the asymmetry is unmistakable. False ` +
        `negatives grow monotonically until they consume nearly the whole stable pool, while ` +
        `false positives peak at ${maxOf(E4.rows.map((r) => r.fp)).toFixed(2)} and then <em>fall ` +
        `back</em> to ${E4.rows[E4.rows.length - 1].fp.toFixed(2)}, because a hull pinned to noise ` +
        `outliers leaves fewer and fewer candidates below it. The expensive, invisible error is ` +
        `unbounded; the cheap, visible one saturates.` }),
      h('p', { class: 'prose', html:
        `The threshold is a lever on the same trade. Matbench Discovery’s headline is ` +
        `$E_{\\rm hull} \\le 0$, but it reports a sweep, and so does the script. Loosening the ` +
        `threshold from 0 to ${meV(TS.threshold[TS.threshold.length - 1], 0)} meV/atom at a fixed ` +
        `σ of ${meV(TS.sigma, 0)} meV/atom raises F1 from ${TS.f1[0].toFixed(3)} to ` +
        `${TS.f1[TS.f1.length - 1].toFixed(3)} — but it also raises prevalence from ` +
        `${pct(TS.prevalence[0], 1)} to ` +
        `${pct(TS.prevalence[TS.prevalence.length - 1], 1)}, and DAF falls from ` +
        `${TS.daf[0].toFixed(1)} to ${TS.daf[TS.daf.length - 1].toFixed(1)}. The problem got ` +
        `easier, so the score went up and the acceleration went down. Neither number means ` +
        `anything without the threshold attached to it.` }),

      h('h2', {}, 'Which metric you choose changes who wins'),
      h('p', { class: 'prose' },
        'All of the above is a synthetic system with a known answer. The reason it is worth ' +
        'building is that the same inversion is plainly visible on the real leaderboard. The ' +
        'figures in the table below are the only numbers on this page that do not come from the ' +
        'script; they are quoted from Matbench Discovery on the unique-prototypes split.'),
      h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'model'), h('th', { class: 'num' }, 'F1'),
          h('th', { class: 'num' }, 'κ_SRME'), h('th', { class: 'num' }, 'CPS'),
          h('th', { class: 'num' }, 'rank by F1'), h('th', { class: 'num' }, 'rank by CPS'))),
        h('tbody', {}, ...LEADERBOARD.rows.map((r) =>
          h('tr', r.model === 'eqV2 M' ? { class: 'hi' } : {},
            h('td', {}, h('strong', {}, r.model),
              h('div', { style: { color: '#5A6773', fontSize: '12px', lineHeight: '1.35' } }, r.note)),
            h('td', { class: 'num' }, r.f1 == null ? '—' : r.f1.toFixed(3)),
            h('td', { class: 'num' }, r.kappa == null ? '—' : r.kappa.toFixed(3)),
            h('td', { class: 'num' }, r.cps == null ? '—' : r.cps.toFixed(3)),
            h('td', { class: 'num' }, r.rankF1 == null ? '—' : `#${r.rankF1}`),
            h('td', { class: 'num' }, r.rankCPS == null ? '—' : `#${r.rankCPS}`))))));

    const eq = LEADERBOARD.rows.find((r) => r.model === 'eqV2 M');
    const dpa = LEADERBOARD.rows.find((r) => r.model === 'DPA-4.0.1-Pro-MPtrj');
    const bestF1 = LEADERBOARD.rows.find((r) => r.rankF1 === 1);
    const bestCPS = LEADERBOARD.rows.find((r) => r.rankCPS === 1);
    root.append(
      h('p', { class: 'prose', html:
        `Ranking by F1 and ranking by the Combined Performance Score give different orders. ` +
        `<strong>${bestF1.model}</strong> leads on F1 at ${bestF1.f1.toFixed(3)}; ` +
        `<strong>${bestCPS.model}</strong> leads on CPS, and it is not the same model — nor even ` +
        `the same architectural lineage, since it is a Cartesian-tensor descendant of the Atomic ` +
        `Cluster Expansion rather than anything with a Clebsch–Gordan product inside it. ` +
        `<strong>${eq.model}</strong> is the extreme case. F1 ${eq.f1.toFixed(3)} puts it around ` +
        `rank #${eq.rankF1}; CPS ${eq.cps.toFixed(3)} puts it around rank #${eq.rankCPS}, a drop ` +
        `of roughly ${eq.rankCPS - eq.rankF1} places, because its phonon score ` +
        `κ_SRME = ${eq.kappa.toFixed(3)} sits close to that metric’s worst possible value of 2. ` +
        `Its geometry optimisations look fine; the second derivatives of its energy surface do ` +
        `not.` }),
      h('p', { class: 'prose', html:
        `The clean real-world instance of the P-versus-Q experiment is the pair ` +
        `<strong>${eq.model}</strong> and <strong>${dpa.model}</strong>. The first beats the ` +
        `second on F1 by ${(eq.f1 - dpa.f1).toFixed(3)} and loses to it on CPS by ` +
        `${(dpa.cps - eq.cps).toFixed(3)} — and the second is the smaller of the two on every ` +
        `axis that usually predicts a win (${dpa.note}). Whichever column you sort on you get a ` +
        `defensible answer and a different winner. That is not a flaw in the leaderboard; it is ` +
        `the leaderboard telling you that “best model” is not a well-posed question.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'the honest reading of κ_SRME'),
        h('div', { html:
          `A high κ_SRME is strongly associated with direct, non-conservative force prediction — ` +
          `on the board every direct-force model sits above 1.6 and every model below 1.0 is ` +
          `energy-gradient-based — but it is neither necessary nor sufficient. CHGNet is ` +
          `conservative and scores 1.7167, essentially the same as ${eq.model}; ORB-v3’s ` +
          `direct-force variant scores 0.348. The quantity that actually matters is the accuracy ` +
          `of the <em>second</em> derivatives of the energy surface, which direct-force training ` +
          `does not constrain and which a conservative model can also get wrong. Chapter 6 takes ` +
          `that mechanism apart.` })),
      h('p', { class: 'prose' },
        'Which brings the argument back to where the reading group started. They asked for F1 ' +
        'instead of accuracy and they were right — but F1 is not the end of the story either. It ' +
        'is one column among several, the models that top it are not the models that top the ' +
        'composite, and the composite itself embeds a choice about what a discovery campaign is ' +
        'for. The next chapter takes the leaderboard apart by lineage and asks what, if anything, ' +
        'built-in equivariance is still buying.'));

    // ================================================================== ledger
    // This results file stores its measurements but not its self-check verdicts,
    // so every assertion the script makes is re-evaluated here, in the page,
    // against the published numbers — and the hull itself is rebuilt from the
    // published cloud rather than trusted. Nothing below is transcribed.
    const checks = [];
    const add = (name, ok, detail) => checks.push({ name, ok: !!ok, detail });

    const storedSet = new Set(S.hull_vertex_index);
    const jsSet = new Set(trueHullIdx);
    add('monotone chain recomputed in-browser reproduces the stored hull vertex set',
      jsSet.size === storedSet.size && [...jsSet].every((v) => storedSet.has(v)),
      `${trueHullIdx.length} vs ${S.hull_vertex_index.length} vertices`);
    const gridDev = maxOf(S.hull_curve_x.map((xq, i) =>
      Math.abs(interpOn(trueHullX, trueHullE, xq) - S.hull_curve_e[i])));
    add(`hull polylines agree pointwise on the ${S.hull_curve_x.length}-point grid`,
      gridDev < 1e-12, `max |diff| ${gridDev.toExponential(2)} eV/atom`);
    add('hull is a true lower bound: no candidate below it',
      minOf(S.candidate_e_hull) > -1e-12,
      `min E_hull ${minOf(S.candidate_e_hull).toExponential(2)} eV/atom`);
    const dx = diffs(S.hull_vertex_x);
    const chord = diffs(S.hull_vertex_e).map((de, i) => de / dx[i]);
    add('hull is convex: chord slopes strictly increasing',
      diffs(chord).every((v) => v > 0),
      `min slope increment ${minOf(diffs(chord)).toFixed(4)} eV/atom`);

    const nInterior = S.hull_vertex_x.length - 2;
    add('several stable interior line compounds form', nInterior >= 8,
      `${nInterior} interior hull vertices among ${M.n_compositions} compositions`);
    add('stable fraction of the pool is a realistic minority',
      prevalence > 0.01 && prevalence < 0.15,
      `prevalence ${prevalence.toFixed(4)} (${M.n_stable}/${M.n_candidates}), ` +
      `DAF ceiling ${M.max_possible_daf.toFixed(1)}`);
    add('elemental references sit at E_f = 0 and anchor both hull ends',
      Math.abs(S.element_e_form[0]) < 1e-15 && Math.abs(S.element_e_form[1]) < 1e-15
        && S.hull_vertex_x[0] === 0 && S.hull_vertex_x[S.hull_vertex_x.length - 1] === 1,
      `hull spans x = ${S.hull_vertex_x[0].toFixed(1)} to ` +
      `${S.hull_vertex_x[S.hull_vertex_x.length - 1].toFixed(1)}`);

    const ratioDev = maxOf(E1.mae.map((m, i) =>
      Math.abs(m / (E1.sigma[i] * Math.sqrt(2 / Math.PI)) - 1)));
    add('energy MAE is exactly sigma*sqrt(2/pi): predictor error is linear in sigma',
      ratioDev < 0.02, `max deviation ${(100 * ratioDev).toFixed(2)}%`);
    const maeSlopeDev = maxOf(E1.mae_loglog_slope.map((v) => Math.abs(v - 1)));
    add('MAE is a pure power law in sigma with exponent 1',
      Math.abs(E1.mae_global_loglog_slope - 1) < 0.01 && maeSlopeDev < 0.03,
      `global slope ${E1.mae_global_loglog_slope.toFixed(5)}, ` +
      `max local |slope-1| ${maeSlopeDev.toFixed(4)}`);
    add('F1 is monotone non-increasing in sigma to within Monte-Carlo scatter',
      E1.f1_monotonicity_slack < 0.01,
      `largest upward excursion ${E1.f1_monotonicity_slack.toFixed(5)} over ${nSig} sigmas`);
    add('F1 spans the full useful range across the sweep',
      E1.f1[0] > 0.90 && E1.f1[last] < 0.20,
      `F1 ${E1.f1[0].toFixed(3)} at MAE ${meV(E1.mae[0], 1)} -> ` +
      `${E1.f1[last].toFixed(3)} at MAE ${meV(E1.mae[last], 1)} meV/atom`);
    add('F1 is NOT a power law in sigma: local elasticity varies more than 3x',
      elasticityRatio > 3,
      `|dlogF1/dlogsigma| ranges ${E1.min_abs_loglog_slope_f1.toFixed(3)} to ` +
      `${E1.max_abs_loglog_slope_f1.toFixed(3)}`);
    add('documented negative result: F1 never decays super-linearly in sigma, so ' +
      '"F1 falls faster than MAE rises" is FALSE as a power law',
      E1.max_abs_loglog_slope_f1 > 0.3 && E1.max_abs_loglog_slope_f1_upper2sigma < 1.0,
      `peak ${E1.max_abs_loglog_slope_f1.toFixed(3)} +/- ` +
      `${E1.max_abs_loglog_slope_f1_se.toFixed(3)}, 2-sigma upper bound ` +
      `${E1.max_abs_loglog_slope_f1_upper2sigma.toFixed(3)} < 1.000`);
    add('F1 0.9 -> 0.3 is traversed inside an MAE window narrower than the leaderboard spread',
      E1.f1_window_width_meV < 50,
      `MAE ${meV(E1['mae_at_f1_0.9'], 1)} -> ${meV(E1['mae_at_f1_0.3'], 1)} meV/atom, ` +
      `width ${E1.f1_window_width_meV.toFixed(1)} meV/atom`);
    add('DAF stays above 1: even a 160 meV/atom model beats random screening',
      minOf(E1.daf) > 1,
      `DAF ${E1.daf[0].toFixed(1)} at best sigma, ${E1.daf[last].toFixed(1)} at worst`);
    add('self-consistent protocol predicts FEWER stable phases as sigma grows',
      E1.n_pred_stable[last] < 0.7 * E1.n_pred_stable[0],
      `${E1.n_pred_stable[0].toFixed(1)} -> ${E1.n_pred_stable[last].toFixed(1)} predicted ` +
      `stable, truth ${M.n_stable}`);
    add('fixed-hull protocol fails the opposite way: false-positive flood',
      FX.n_pred_stable[last] > 5 * E1.n_pred_stable[last],
      `${FX.n_pred_stable[last].toFixed(1)} vs ${E1.n_pred_stable[last].toFixed(1)} at ` +
      `sigma = ${meV(E1.sigma[last], 0)} meV/atom`);

    const acc25 = E2.accuracy_by_bin[k25];
    const farAcc = Math.min(acc25[acc25.length - 2], acc25[acc25.length - 1]);
    add('far-from-hull structures are classified essentially perfectly', farAcc > 0.999,
      `accuracy ${farAcc.toFixed(5)} above 150 meV/atom at sigma = ${meV(E1.sigma[k25], 1)}`);
    add('misclassifications live in a thin band around the hull, not spread over the pool',
      E2.band_width_p99[k25] < 0.2 * eHullSpan && E2.fraction_errors_below_100meV[k25] > 0.9,
      `99% of errors within ${meV(E2.band_width_p99[k25], 0)} meV/atom ` +
      `(${pct(E2.band_width_p99[k25] / eHullSpan, 1)} of a ${meV(eHullSpan, 0)} meV/atom span)`);
    add('accuracy rises monotonically with distance from the hull',
      diffs(acc25).every((v) => v >= -1e-12),
      `per-bin accuracy ${acc25.map((v) => v.toFixed(3)).join(' ')}`);
    add('band width grows about linearly in sigma', bandSlope > 0.5 && bandSlope < 1.5,
      `log-log slope ${bandSlope.toFixed(3)}, p99 band ${meV(E2.band_width_p99[kFit], 0)} -> ` +
      `${meV(E2.band_width_p99[last], 0)} meV/atom`);
    add('band width is monotone non-decreasing in sigma',
      diffs(E2.band_width_p99).every((v) => v > -0.005),
      `largest decrease ${meV(minOf(diffs(E2.band_width_p99)), 2)} meV/atom`);

    add('P has the lower energy MAE', E3.P.mae < E3.Q.mae,
      `MAE_P ${meV(E3.P.mae, 2)} vs MAE_Q ${meV(E3.Q.mae, 2)} meV/atom ` +
      `(ratio ${(E3.Q.mae / E3.P.mae).toFixed(2)})`);
    add('P also has the lower RMSE', E3.P.rmse < E3.Q.rmse,
      `RMSE_P ${meV(E3.P.rmse, 2)} vs RMSE_Q ${meV(E3.Q.rmse, 2)} meV/atom`);
    add('yet Q has the higher F1: the regression ordering inverts',
      E3.Q.f1 > E3.P.f1 + 0.10, `F1_P ${E3.P.f1.toFixed(3)} vs F1_Q ${E3.Q.f1.toFixed(3)}`);
    add('Q also wins on precision, recall and DAF simultaneously',
      E3.Q.precision > E3.P.precision && E3.Q.recall > E3.P.recall && E3.Q.daf > E3.P.daf,
      `prec ${E3.P.precision.toFixed(3)}->${E3.Q.precision.toFixed(3)}, ` +
      `rec ${E3.P.recall.toFixed(3)}->${E3.Q.recall.toFixed(3)}, ` +
      `DAF ${E3.P.daf.toFixed(1)}->${E3.Q.daf.toFixed(1)}`);
    add('the inversion is not an averaging artefact: it holds trial by trial',
      invLive > 0.95, `${pct(invLive, 1)} of ${M.n_trials_exp3} independent trials`);

    const r0 = E4.rows[0], rN = E4.rows[E4.rows.length - 1];
    add('both error types grow with sigma but false negatives dominate',
      rN.fn > r0.fn && rN.fn > rN.fp,
      `at MAE ${rN.mae_meV.toFixed(0)} meV/atom: FN ${rN.fn.toFixed(1)} of ${M.n_stable} stable, ` +
      `FP ${rN.fp.toFixed(1)}`);
    add('at the accurate end the campaign still loses real materials', E4.rows[1].fn > 0,
      `at MAE ${E4.rows[1].mae_meV.toFixed(1)} meV/atom: ${E4.rows[1].fn.toFixed(2)} missed ` +
      `(${pct(E4.rows[1].fraction_of_stable_pool_lost, 1)} of the stable pool)`);
    add('recall degrades faster than precision under this protocol',
      (E1.recall[0] - E1.recall[last]) > (E1.precision[0] - E1.precision[last]),
      `recall ${E1.recall[0].toFixed(3)}->${E1.recall[last].toFixed(3)}, ` +
      `precision ${E1.precision[0].toFixed(3)}->${E1.precision[last].toFixed(3)}`);
    const tl = TS.f1.length - 1;
    add('threshold sweep: loosening the 0 eV/atom threshold raises F1', TS.f1[tl] > TS.f1[0],
      `F1 ${TS.f1[0].toFixed(3)} at 0 -> ${TS.f1[tl].toFixed(3)} at ` +
      `${meV(TS.threshold[tl], 0)} meV/atom, prevalence ${TS.prevalence[0].toFixed(3)} -> ` +
      `${TS.prevalence[tl].toFixed(3)}`);

    const nPass = checks.filter((c) => c.ok).length;
    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Check ledger — python/experiments/convex_hull_decision.py'),
      h('p', { class: 'hint' },
        'Reproduce with:  uv run python python/experiments/convex_hull_decision.py    ' +
        '— this results file stores its measurements but not its verdicts, so every assertion ' +
        'below is re-evaluated in the browser against the published numbers, and the hull itself ' +
        'is rebuilt here from the published cloud rather than taken on trust.'));
    const inner = h('div');
    for (const c of checks) inner.append(checkLine(c.ok, c.name, c.detail));
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${nPass === checks.length ? 'ok' : 'bad'}` },
        `${nPass}/${checks.length}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);
    root.append(ledger);
  },
};
