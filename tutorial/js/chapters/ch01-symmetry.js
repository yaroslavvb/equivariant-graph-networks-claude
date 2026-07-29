import { h, slider, segmented, checkLine, fmt } from '../ui.js';
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
// analytic energy. (Chapter 6 makes the gradient structure the whole subject.)
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
  render(root) {
    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 1'),
      h('h1', {}, 'What symmetry demands'),
      h('p', { class: 'lede' },
        'Rotate a molecule and its energy must not change; rotate a molecule and its forces must ' +
        'rotate with it. Those are two different requirements, and the difference between them is ' +
        'the whole architectural argument.'),

      h('p', { class: 'prose' },
        'Write $g$ for a rigid motion — a rotation, a reflection, a translation, or any ' +
        'composition of them. A function $f$ from atomic positions to some output is ' +
        'called <em>equivariant</em> when'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$f\\bigl(\\rho_{\\text{in}}(g)\\,x\\bigr) \\;=\\; \\rho_{\\text{out}}(g)\\,f(x)$$'),
      h('p', { class: 'prose' },
        'where $\\rho_{\\text{in}}$ and $\\rho_{\\text{out}}$ say how the input and the output are ' +
        'each supposed to transform. <em>Invariance</em> is the special case where ' +
        '$\\rho_{\\text{out}}(g)$ is the identity: the output does not move at all. Energy is a ' +
        'scalar, so it is invariant. Force is a vector, so it is equivariant — it must turn with ' +
        'the molecule.'),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'The distinction that matters'),
        h('div', { html:
          'A network built only from invariant quantities can represent the energy. But if every ' +
          '<em>internal</em> feature is also invariant, the network has thrown away orientation ' +
          'at the first layer and has to reconstruct angular relationships indirectly from ' +
          'distances forever after. NequIP’s move was to let internal features be equivariant — ' +
          'vectors and higher tensors that rotate with the molecule — and only collapse to an ' +
          'invariant at the very end. Chapter 2 measures exactly what that buys.' })),

      h('h2', {}, 'The rotation test'),
      h('p', { class: 'prose' },
        'Below is a cluster of three atoms around a centre, and a ground-truth energy with real ' +
        'angular structure. Drag the rotation and watch three quantities: the true energy, which ' +
        'must not move; the true force, whose components must move but whose <em>length</em> must ' +
        'not; and the residual of the equivariance condition, computed live.'),
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
        'of the rotation group, and it is the subject of chapter 3. First, though, the case for ' +
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
      // and the harmonics themselves, the objects chapter 3 is built from
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
  },
};
