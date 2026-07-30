import { h, slider, segmented, checkLine, fmt, loadResults, Plot, PALETTE } from '../ui.js';
import { realSH, wignerD, randomRotation, mulberry32 } from '../e3.js';
import { rotationMatrix, matvec, norm } from '../linalg.js';

// A three-atom cluster whose energy has genuine angular structure, so that a
// model which ignores orientation cannot possibly be right.
const CLUSTER = [
  [1.10, 0.15, -0.20],
  [-0.55, 0.95, 0.35],
  [-0.30, -0.85, 0.75],
];

// Ground truth: pair terms plus a three-body angular term. Exactly invariant.
function trueEnergy(pos) {
  let E = 0;
  for (let i = 0; i < pos.length; i++) {
    const r = norm(pos[i]);
    E += Math.exp(-1.4 * (r - 1.0) ** 2);
    for (let j = i + 1; j < pos.length; j++) {
      const rj = norm(pos[j]);
      const c = (pos[i][0] * pos[j][0] + pos[i][1] * pos[j][1] + pos[i][2] * pos[j][2]) / (r * rj);
      E += 0.6 * (c + 1 / 3) ** 2;
    }
  }
  return E;
}

// True forces on the central atom, by exact-enough central differences of the
// analytic energy. (Chapter 7 makes the gradient structure the whole subject.)
function trueForce(pos) {
  const eps = 1e-6;
  const f = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const shift = (s) => pos.map((p) => p.map((v, k) => (k === a ? v + s : v)));
    f[a] = -(trueEnergy(shift(eps)) - trueEnergy(shift(-eps))) / (2 * eps);
  }
  return f;
}

export default {
  id: 'symmetry',
  title: 'What symmetry demands',
  async render(root) {
    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 2'),
      h('h1', {}, 'What symmetry demands'),
      h('p', { class: 'lede' },
        'Rotate a molecule and its energy must not change; rotate a molecule and its forces must ' +
        'rotate with it. Those are two different requirements, and the difference between them is ' +
        'the whole architectural argument.'),

      h('p', { class: 'prose', html:
        'Write $g$ for a rigid motion — a rotation, a reflection, a translation, or any ' +
        'composition of them. A function $f$ from atomic positions to some output is ' +
        'called <em>equivariant</em> when' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$f\\bigl(\\rho_{\\text{in}}(g)\\,x\\bigr) \\;=\\; \\rho_{\\text{out}}(g)\\,f(x)$$'),
      h('p', { class: 'prose', html:
        'where $\\rho_{\\text{in}}$ and $\\rho_{\\text{out}}$ say how the input and the output are ' +
        'each supposed to transform. <em>Invariance</em> is the special case where ' +
        '$\\rho_{\\text{out}}(g)$ is the identity: the output does not move at all. Energy is a ' +
        'scalar, so it is invariant. Force is a vector, so it is equivariant — it must turn with ' +
        'the molecule.' }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'The distinction that matters'),
        h('div', { html:
          'A network built only from invariant quantities can represent the energy. But if every ' +
          '<em>internal</em> feature is also invariant, the network has thrown away orientation ' +
          'at the first layer and has to reconstruct angular relationships indirectly from ' +
          'distances forever after. NequIP’s move was to let internal features be equivariant — ' +
          'vectors and higher tensors that rotate with the molecule — and only collapse to an ' +
          'invariant at the very end. Chapter 3 measures exactly what that buys.' })),

      h('h2', {}, 'The rotation test'),
      h('p', { class: 'prose', html:
        'Below is a cluster of three atoms around a centre, and a ground-truth energy with real ' +
        'angular structure. Drag the rotation and watch three quantities: the true energy, which ' +
        'must not move; the true force, whose components must move but whose <em>length</em> must ' +
        'not; and the residual of the equivariance condition, computed live.' }),
    );

    // --- interactive rotation test ---------------------------------------
    const demo = h('div', { class: 'demo' });
    demo.append(
      h('h3', {}, 'Rotate the cluster'),
      h('p', { class: 'hint' },
        'The projection is onto the xy-plane; the third axis is drawn as circle size. ' +
        'Nothing about the physics changes — that is the point.'));

    const cv = h('canvas', { width: 640, height: 300, style: { width: '100%', maxWidth: '640px' } });
    const readout = h('div', { class: 'readout' });
    let angle = 0, axis = [0.3, 0.4, 0.86];

    function draw() {
      const R = rotationMatrix(axis, angle);
      const pos = CLUSTER.map((p) => matvec(R, p));
      const ctx = cv.getContext('2d');
      const W = cv.width, H = cv.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

      const cx = W / 2, cy = H / 2, S = 88;
      ctx.strokeStyle = '#EDF0F3'; ctx.lineWidth = 1;
      for (let g = -2; g <= 2; g++) {
        ctx.beginPath(); ctx.moveTo(cx + g * S, 20); ctx.lineTo(cx + g * S, H - 20); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(60, cy + g * S); ctx.lineTo(W - 60, cy + g * S); ctx.stroke();
      }

      const F = trueForce(pos);
      // bonds
      ctx.strokeStyle = '#C9CFD6'; ctx.lineWidth = 1.6;
      for (const p of pos) {
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + p[0] * S, cy - p[1] * S); ctx.stroke();
      }
      // force on the central atom
      const fs = 150;
      ctx.strokeStyle = '#B5443C'; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + F[0] * fs, cy - F[1] * fs); ctx.stroke();
      const ang = Math.atan2(-F[1], F[0]);
      ctx.fillStyle = '#B5443C';
      ctx.beginPath();
      ctx.moveTo(cx + F[0] * fs, cy - F[1] * fs);
      ctx.lineTo(cx + F[0] * fs - 9 * Math.cos(ang - 0.4), cy - F[1] * fs - 9 * Math.sin(ang - 0.4));
      ctx.lineTo(cx + F[0] * fs - 9 * Math.cos(ang + 0.4), cy - F[1] * fs - 9 * Math.sin(ang + 0.4));
      ctx.fill();

      pos.forEach((p, i) => {
        const r = 8 + 4 * p[2];
        ctx.beginPath();
        ctx.arc(cx + p[0] * S, cy - p[1] * S, Math.max(3, r), 0, 2 * Math.PI);
        ctx.fillStyle = ['#1F4E79', '#3E7C59', '#6C4A8C'][i];
        ctx.fill();
      });
      ctx.beginPath(); ctx.arc(cx, cy, 7, 0, 2 * Math.PI);
      ctx.fillStyle = '#1B2733'; ctx.fill();

      // Equivariance residuals, measured against the unrotated reference.
      const E0 = trueEnergy(CLUSTER), F0 = trueForce(CLUSTER);
      const E = trueEnergy(pos);
      const RF0 = matvec(R, F0);
      const dF = Math.hypot(F[0] - RF0[0], F[1] - RF0[1], F[2] - RF0[2]);
      readout.innerHTML =
        `rotation angle          ${(angle * 180 / Math.PI).toFixed(1)}°\n` +
        `\n` +
        `E(x)                    ${E0.toFixed(12)}\n` +
        `E(gx)                   ${E.toFixed(12)}\n` +
        `|E(gx) - E(x)|          <span class="${Math.abs(E - E0) < 1e-9 ? 'ok' : 'bad'}">` +
        `${Math.abs(E - E0).toExponential(2)}</span>   <span class="dim">energy is invariant</span>\n` +
        `\n` +
        `F(gx)                   [${F.map((v) => v.toFixed(6).padStart(10)).join(', ')}]\n` +
        `g F(x)                  [${RF0.map((v) => v.toFixed(6).padStart(10)).join(', ')}]\n` +
        `|F(gx) - g F(x)|        <span class="${dF < 1e-6 ? 'ok' : 'bad'}">${dF.toExponential(2)}</span>` +
        `   <span class="dim">force is equivariant</span>\n` +
        `|F| unchanged           ${norm(F).toFixed(12)}  vs  ${norm(F0).toFixed(12)}`;
    }

    demo.append(cv,
      h('div', { class: 'controls' },
        slider({
          label: 'rotation', min: 0, max: 360, step: 1, value: 0,
          format: (v) => `${v}°`,
          onInput: (v) => { angle = v * Math.PI / 180; draw(); },
        }),
        segmented({
          label: 'axis',
          options: [
            { label: 'tilted', value: 'tilt' },
            { label: 'z', value: 'z' },
            { label: 'x', value: 'x' },
          ],
          value: 'tilt',
          onPick: (v) => {
            axis = v === 'z' ? [0, 0, 1] : v === 'x' ? [1, 0, 0] : [0.3, 0.4, 0.86];
            draw();
          },
        })),
      readout);
    root.append(demo);
    draw();

    // --- why this is not free -------------------------------------------
    root.append(
      h('h2', {}, 'Why this is not automatic'),
      h('p', { class: 'prose' },
        'The ground truth above satisfies both conditions exactly, because it was written down in ' +
        'terms of distances and angles. A neural network fed raw Cartesian coordinates satisfies ' +
        'neither. It has three options: learn the symmetry approximately from rotated training ' +
        'examples; destroy the orientation information at the input by using only invariants; or ' +
        'build the transformation law into every layer. Those are the three eras of this field, ' +
        'and chapters 2 and 5 measure what separates them.'),

      h('p', { class: 'prose' },
        'The last option needs a language for “this feature is a vector, that one is a rank-2 ' +
        'tensor, and here is exactly how each turns.” That language is the representation theory ' +
        'of the rotation group, and it is the subject of chapter 4. First, though, the case for ' +
        'the prosecution: what precisely does a network lose by keeping only invariants?'),
    );

    // --- a small live check ----------------------------------------------
    const checks = h('div', { class: 'demo' });
    checks.append(
      h('h3', {}, 'The condition, checked at 200 random rotations'),
      h('p', { class: 'hint' },
        'Not a claim about the figure above — an actual sweep over Haar-random rotations of the ' +
        'group, run when you clicked into this chapter.'));
    const box = h('div');
    const rng = mulberry32(31337);
    let worstE = 0, worstF = 0, worstY = 0;
    const E0 = trueEnergy(CLUSTER), F0 = trueForce(CLUSTER);
    for (let t = 0; t < 200; t++) {
      const R = randomRotation(rng);
      const pos = CLUSTER.map((p) => matvec(R, p));
      worstE = Math.max(worstE, Math.abs(trueEnergy(pos) - E0));
      const RF0 = matvec(R, F0), F = trueForce(pos);
      worstF = Math.max(worstF, Math.hypot(F[0] - RF0[0], F[1] - RF0[1], F[2] - RF0[2]));
      // and the harmonics themselves, the objects chapter 4 is built from
      const D = wignerD(2, R);
      const lhs = realSH(2, matvec(R, CLUSTER[0]));
      const rhs = matvec(D, realSH(2, CLUSTER[0]));
      worstY = Math.max(worstY, Math.max(...lhs.map((v, i) => Math.abs(v - rhs[i]))));
    }
    box.append(
      checkLine(worstE < 1e-12, 'energy invariant: max |E(gx) − E(x)| over 200 rotations',
        worstE.toExponential(2)),
      checkLine(worstF < 1e-7, 'force equivariant: max |F(gx) − gF(x)| over 200 rotations',
        `${worstF.toExponential(2)}  (finite-difference floor ~1e-9)`),
      checkLine(worstY < 1e-12, 'degree-2 harmonics: max |Y₂(gr) − D²(g)Y₂(r)|',
        worstY.toExponential(2)));
    checks.append(box);
    root.append(checks);

    // ---- built-in symmetry versus symmetry learned from augmentation -----
    const A = await loadResults('augmentation_vs_equivariance');
    const P = A.offdist_probe;
    const bi = A.builtin, au = A.augmented;

    root.append(
      h('h2', {}, 'Built in, or learned from examples?'),
      h('p', { class: 'prose' },
        'The third option — show the network rotated copies and let it work the symmetry out — is ' +
        'the one most people reach for first, and it is worth knowing exactly what it gets you. ' +
        'The experiment behind the next two figures fits the same invariant target two ways: once ' +
        'with features that are invariant by construction, and once with an unconstrained model on ' +
        'raw coordinates trained with $K$ random rotations of every example.'),
      h('p', { class: 'prose' },
        'The quantity to watch is the invariance error ' +
        '$\\epsilon(g) = |f(gx) - f(x)| / (1 + |f(x)|)$, measured on held-out configurations under ' +
        'rotations the model never saw.'));

    const cmp = h('div', { class: 'demo' });
    cmp.append(h('h3', {}, 'Invariance error against training-set size'),
      h('p', { class: 'hint' },
        'Lower is better, and the log scale spans about fifteen decades. One augmentation curve ' +
        'per multiplicity K, darker for larger K. Points where the fit had collapsed to predicting ' +
        'the mean are omitted — such a model is perfectly invariant for a reason that has nothing ' +
        'to do with symmetry — which is why the sparser curves begin further right.'));
    const p1 = new Plot({ width: 660, height: 300, xLog: true, yLog: true,
      xLabel: 'training configurations', yLabel: 'mean invariance error ε(g)' });
    p1.add({ points: bi.n_train.map((n, i) => [n, Math.max(bi.inv_err_mean[i], 1e-17)]),
      color: PALETTE[0], width: 2.8, markers: true, label: 'invariant by construction' });
    // au.inv_err_mean is indexed [n_train][K], so one curve per augmentation
    // multiplicity. Points flagged near_trivial are dropped: a fit that has
    // collapsed to predicting the mean is perfectly invariant for a reason that
    // has nothing to do with having learned the symmetry.
    au.K.forEach((K, k) => {
      const pts = au.n_train
        .map((n, i) => (au.near_trivial[i][k] ? null : [n, Math.max(au.inv_err_mean[i][k], 1e-17)]))
        .filter(Boolean);
      if (!pts.length) return;
      p1.add({ points: pts, color: PALETTE[1], width: 1.8, markers: true, dash: '5 4',
        opacity: 0.35 + 0.65 * (k / (au.K.length - 1)),
        label: `augmentation, K = ${K}` });
    });
    cmp.append(p1.render(), p1.legend(),
      h('div', { class: 'readout', html:
        `built-in, worst case over every training size:   ` +
        `${Math.max(...bi.inv_err_max).toExponential(2)}\n` +
        `augmented, best mean invariance ever achieved:   ` +
        `${A.augmented.min_inv_err_nontrivial.toExponential(2)}\n\n` +
        `<span class="dim">${A.augmented.trivial_note}</span>` }));
    root.append(cmp);

    root.append(
      h('p', { class: 'prose', html:
        `Two things are worth pulling out. The built-in model's invariance error is ` +
        `${Math.max(...bi.inv_err_max).toExponential(1)} — that is float rounding, not learning, ` +
        `and no amount of training changes it because there is nothing to learn. The augmented ` +
        `model never gets below ` +
        `${A.augmented.min_inv_err_nontrivial.toExponential(1)}, about ` +
        `${Math.round(Math.log10(A.augmented.min_inv_err_nontrivial / Math.max(...bi.inv_err_max)))} ` +
        `orders of magnitude worse. And the experiment is careful about a trap that is easy to ` +
        `fall into: a model that has given up and predicts a constant is perfectly invariant for a ` +
        `stupid reason, so the honest figure excludes fits that are within 10% of just predicting ` +
        `the mean.` }),

      h('h2', {}, 'The sharper test: rotations you did not train on'),
      h('p', { class: 'prose' },
        'Averaged invariance error understates the problem, because augmentation makes a model ' +
        'invariant on the orbit it was shown rather than on the group. The probe below trains the ' +
        'unconstrained model three ways — with fully general rotations, with rotations about the ' +
        '$z$-axis only, and with small rotations only — then tests all three under general ' +
        'rotations.'));

    const probe = h('div', { class: 'demo' });
    probe.append(h('h3', {}, 'Off-distribution probe'),
      h('p', { class: 'hint' },
        `Unconstrained model, ${P.n_train} training configurations, K = ${P.K} augmentations each. ` +
        'nRMSE of 1.0 means no better than predicting the mean.'));
    const famLabel = { full: 'all rotations', zaxis: 'z-axis only', smallangle: 'small angles only' };
    const rows = P.families.map((tr) => h('tr', { class: tr === 'full' ? '' : 'hi' },
      h('td', {}, famLabel[tr]),
      ...P.families.map((te) => h('td', { class: 'num' },
        `${fmt(P.inv_err_mean[tr][te], 3)}`,
        h('span', { class: 'dim', style: { color: '#8A929B' } }, ` / ${fmt(P.inv_err_max[tr][te], 2)}`))),
      h('td', { class: 'num' }, fmt(P.nrmse_canonical[tr], 3)),
      h('td', { class: 'num' }, fmt(P.nrmse_rotated[tr], 3))));
    probe.append(h('table', {},
      h('thead', {},
        h('tr', {},
          h('th', {}, ''), h('th', { class: 'num', colspan: '3' }, 'ε(g) mean / max, tested under'),
          h('th', { class: 'num', colspan: '2' }, 'nRMSE')),
        h('tr', {},
          h('th', {}, 'trained with'),
          ...P.families.map((te) => h('th', { class: 'num' }, famLabel[te])),
          h('th', { class: 'num' }, 'canonical'), h('th', { class: 'num' }, 'rotated'))),
      h('tbody', {}, rows)));
    root.append(probe);

    root.append(
      h('p', { class: 'prose', html:
        `The first row is the honest baseline: trained on all rotations, the model does about ` +
        `equally badly in both orientations (${fmt(P.nrmse_canonical.full, 3)} against ` +
        `${fmt(P.nrmse_rotated.full, 3)}) — it is at least consistent. The other two rows are the ` +
        `result that matters. Restrict the training rotations and the model looks <em>better</em> ` +
        `in the orientation it was shown — ${fmt(P.nrmse_canonical.zaxis, 3)} for the $z$-axis ` +
        `model, against ${fmt(P.nrmse_canonical.full, 3)} — and then fails completely when the ` +
        `test configuration is rotated somewhere it has never been: ` +
        `${fmt(P.nrmse_rotated.zaxis, 3)}, which is worse than predicting the mean.` }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'What augmentation actually buys'),
        h('div', { html:
          'Invariance on the orbit you sampled, not invariance on the group. That distinction is ' +
          'invisible if your test set is drawn the same way as your training set, and it is exactly ' +
          'the situation a molecular-dynamics trajectory puts you in — the molecule tumbles into ' +
          'orientations nobody chose. Building the transformation law into the architecture makes ' +
          'the question moot. Chapter 4 is how that is done.' })),
      h('p', { class: 'prose' },
        'One caveat in fairness to augmentation: the unconstrained model here is a random-feature ' +
        'ridge regression, not a deep network trained for a long time, and at 2026 data scales a ' +
        'large model with heavy augmentation can get close enough that the difference stops ' +
        'dominating. Chapter 10 returns to this, because one of the models currently in the top five ' +
        'of Matbench Discovery is deliberately not equivariant at all.'));
  },
};
