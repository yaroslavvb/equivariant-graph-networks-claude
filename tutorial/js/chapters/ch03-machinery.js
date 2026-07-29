import { h, slider, segmented, checkLine, fmt } from '../ui.js';
import {
  realSH, wignerD, wignerResidual, clebschGordan, tensorProduct, cgAllowed, randomRotation, mulberry32,
} from '../e3.js';
import { rotationMatrix, matvec, matmul, transpose, maxAbs, norm, dot } from '../linalg.js';

const COLORS = ['#1F4E79', '#B5443C', '#C9A227', '#3E7C59', '#6C4A8C', '#5A6773', '#8C5A2B'];

export default {
  id: 'machinery',
  title: 'Building the machinery',
  render(root) {
    root.append(
      h('p', { class: 'eyebrow geo' }, 'Chapter 3'),
      h('h1', {}, 'Building the machinery'),
      h('p', { class: 'lede' },
        'Three objects turn “respect rotations” into something you can put in a network: ' +
        'spherical harmonics, Wigner matrices, and Clebsch–Gordan couplings. None of them is ' +
        'looked up here. Each is derived on this page, then checked against the property that ' +
        'defines it.'),

      h('h2', {}, '1. Irreducible representations: the atoms of rotation'),
      h('p', { class: 'prose' },
        'A rotation acts on a scalar by doing nothing, on a vector by a $3\\times3$ matrix, and on ' +
        'a rank-2 tensor by something bigger. The useful fact is that every way a rotation can ' +
        'act decomposes into blocks that cannot be broken down further. Those blocks are indexed ' +
        'by a degree $\\ell = 0, 1, 2, \\dots$, and the block of degree $\\ell$ is ' +
        '$(2\\ell+1)$-dimensional.'),
      h('p', { class: 'prose', html:
        'So $\\ell = 0$ is a scalar (one number, unchanged). $\\ell = 1$ is a vector (three ' +
        'numbers that rotate). $\\ell = 2$ is five numbers that transform like the traceless part ' +
        'of a symmetric matrix — a quadrupole. Reflections add a second label, the parity ' +
        '$p = \\pm1$: a vector flips sign under inversion and is written <code>1o</code> (odd), ' +
        'while an angular momentum does not and is written <code>1e</code> (even). That pair ' +
        '$(\\ell, p)$ is the complete vocabulary.' }),

      h('h2', {}, '2. Spherical harmonics: turning a direction into features'),
      h('p', { class: 'prose' },
        'The map from a unit vector $\\hat{r}$ to the degree-$\\ell$ block is the real spherical ' +
        'harmonic $Y_\\ell(\\hat{r}) \\in \\mathbb{R}^{2\\ell+1}$. This is how a bond direction ' +
        'enters the network. Drag a direction and watch each degree respond — $\\ell=0$ ignores ' +
        'it entirely, which is precisely the blindness of chapter 2.'),
    );

    // ---- SH explorer ----------------------------------------------------
    const shDemo = h('div', { class: 'demo' });
    shDemo.append(h('h3', {}, 'Spherical harmonics of a direction'),
      h('p', { class: 'hint' }, 'Components are ordered $m = -\\ell \\dots +\\ell$. Bar length is the value.'));
    const shCanvas = h('canvas', { width: 660, height: 230, style: { width: '100%', maxWidth: '660px' } });
    const shRead = h('div', { class: 'readout' });
    let theta = 0.9, phi = 0.6;

    function drawSH() {
      const d = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
      const ctx = shCanvas.getContext('2d');
      const W = shCanvas.width, H = shCanvas.height;
      ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

      let x0 = 24;
      for (let l = 0; l <= 3; l++) {
        const Y = realSH(l, d);
        const bw = 15, gap = 3;
        const groupW = Y.length * (bw + gap);
        const mid = H / 2;
        ctx.strokeStyle = '#DDE3E8'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0 - 6, mid); ctx.lineTo(x0 + groupW, mid); ctx.stroke();
        Y.forEach((v, i) => {
          const hgt = v * 190;
          ctx.fillStyle = COLORS[l];
          ctx.globalAlpha = 0.85;
          ctx.fillRect(x0 + i * (bw + gap), mid - Math.max(hgt, 0), bw, Math.abs(hgt));
          ctx.globalAlpha = 1;
        });
        ctx.fillStyle = '#5A6773';
        ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
        ctx.fillText(`l = ${l}`, x0, H - 8);
        x0 += groupW + 34;
      }
      shRead.innerHTML =
        `direction     [${d.map((v) => v.toFixed(4).padStart(8)).join(', ')}]\n` +
        [0, 1, 2, 3].map((l) => {
          const Y = realSH(l, d);
          return `Y_${l}          [${Y.map((v) => v.toFixed(4).padStart(8)).join(', ')}]` +
                 `   <span class="dim">|Y| = ${norm(Y).toFixed(6)}</span>`;
        }).join('\n') +
        `\n\n<span class="dim">|Y_l| is constant on the sphere — the norm is invariant, the ` +
        `components are not. That is exactly the split chapter 2 exploited.</span>`;
    }
    shDemo.append(shCanvas,
      h('div', { class: 'controls' },
        slider({ label: 'polar θ', min: 0, max: 180, step: 1, value: 52,
          format: (v) => `${v}°`, onInput: (v) => { theta = v * Math.PI / 180; drawSH(); } }),
        slider({ label: 'azimuth φ', min: 0, max: 360, step: 1, value: 34,
          format: (v) => `${v}°`, onInput: (v) => { phi = v * Math.PI / 180; drawSH(); } })),
      shRead);
    root.append(shDemo);
    drawSH();

    // ---- Wigner D -------------------------------------------------------
    root.append(
      h('h2', {}, '3. Wigner matrices: how each degree turns'),
      h('p', { class: 'prose' },
        'Rotating the direction first and taking harmonics second is the same as taking harmonics ' +
        'first and applying some $(2\\ell+1)\\times(2\\ell+1)$ matrix second. That matrix is the ' +
        'Wigner matrix $D^\\ell(R)$, and its existence is the statement that degree $\\ell$ is a ' +
        'closed, self-contained block:'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.1em 0' } },
        '$$Y_\\ell(R\\,\\hat{r}) \\;=\\; D^\\ell(R)\\, Y_\\ell(\\hat{r}).$$'),
      h('p', { class: 'prose', html:
        'Most treatments hand you a closed-form expression involving Euler angles. You do not ' +
        'need one. Read that equation as a linear system: evaluate both sides at a batch of ' +
        'sample directions and solve for $D^\\ell$ by least squares. If a degree-$\\ell$ block ' +
        'really is closed under rotation, an exact solution exists and the residual on ' +
        '<em>held-out</em> directions is float noise. If it were not, there would be no such matrix ' +
        'and the residual would be large. The check is the derivation.' }));

    const wDemo = h('div', { class: 'demo' });
    wDemo.append(h('h3', {}, 'Solve for D, then test it on fresh directions'),
      h('p', { class: 'hint' }, 'The matrix is fitted on 8(2ℓ+1) random directions and tested on 256 it has never seen.'));
    const wCanvas = h('canvas', { width: 300, height: 300, style: { width: '300px', maxWidth: '100%' } });
    const wRead = h('div', { class: 'readout' });
    let wl = 2, wangle = 0.7;

    function drawW() {
      const R = rotationMatrix([0.31, 0.42, 0.85], wangle);
      const D = wignerD(wl, R);
      const n = D.length;
      const ctx = wCanvas.getContext('2d');
      const S = 300 / n;
      ctx.clearRect(0, 0, 300, 300);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const v = D[i][j];
        const t = Math.min(1, Math.abs(v));
        ctx.fillStyle = v >= 0 ? `rgba(31,78,121,${t})` : `rgba(181,68,60,${t})`;
        ctx.fillRect(j * S, i * S, S - 1, S - 1);
      }
      const res = wignerResidual(wl, R);
      const orth = maxAbs(matmul(transpose(D), D).map((r, i) => r.map((v, j) => v - (i === j ? 1 : 0))));
      const R2 = rotationMatrix([0.9, -0.2, 0.3], 1.1);
      const hom = maxAbs(matmul(wignerD(wl, R), wignerD(wl, R2))
        .map((r, i) => r.map((v, j) => v - wignerD(wl, matmul(R, R2))[i][j])));
      wRead.innerHTML =
        `l = ${wl},  D is ${n} x ${n},  rotation angle ${(wangle * 180 / Math.PI).toFixed(0)}°\n\n` +
        `held-out residual  max |Y(Rr) - D Y(r)|   <span class="${res < 1e-10 ? 'ok' : 'bad'}">` +
        `${res.toExponential(2)}</span>\n` +
        `orthogonality      max |DᵀD - I|          <span class="${orth < 1e-10 ? 'ok' : 'bad'}">` +
        `${orth.toExponential(2)}</span>\n` +
        `homomorphism       max |D(R₁R₂) - D(R₁)D(R₂)|  <span class="${hom < 1e-10 ? 'ok' : 'bad'}">` +
        `${hom.toExponential(2)}</span>\n\n` +
        `<span class="dim">Blue positive, red negative. Note the matrix is dense within the block ` +
        `and never mixes one degree into another — that closure is what makes ℓ a good label.</span>`;
    }
    wDemo.append(
      h('div', { class: 'controls' },
        segmented({ label: 'degree', options: [0, 1, 2, 3].map((l) => ({ label: `ℓ=${l}`, value: l })),
          value: 2, onPick: (v) => { wl = v; drawW(); } }),
        slider({ label: 'angle', min: 0, max: 360, step: 1, value: 40,
          format: (v) => `${v}°`, onInput: (v) => { wangle = v * Math.PI / 180; drawW(); } })),
      wCanvas, wRead);
    root.append(wDemo);
    drawW();

    // ---- Clebsch-Gordan -------------------------------------------------
    root.append(
      h('h2', {}, '4. Clebsch–Gordan: the only way to multiply two features'),
      h('p', { class: 'prose', html:
        'A network has to combine features. Adding two degree-$\\ell$ features is fine — they ' +
        'transform the same way. Multiplying is the problem: what is a vector times a vector? ' +
        'The answer is forced. The space of bilinear maps ' +
        '$V_{\\ell_1} \\times V_{\\ell_2} \\to V_{\\ell_3}$ that commute with rotation is ' +
        '<em>at most one-dimensional</em>, and it is nonzero exactly when' }),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1em 0' } },
        '$$|\\ell_1 - \\ell_2| \\;\\le\\; \\ell_3 \\;\\le\\; \\ell_1 + \\ell_2, \\qquad p_1 p_2 = p_3.$$'),
      h('p', { class: 'prose', html:
        'So there is essentially <em>one</em> equivariant product for each allowed output degree, ' +
        'and the network’s only freedom is how much of each to use. That is why NequIP’s learnable ' +
        'weights live in the radial functions and the channel mixing, never in the angular part: ' +
        'the angular part has no free parameters to learn.' }),
      h('div', { class: 'note geo' },
        h('span', { class: 'tag' }, 'How to get the coefficients without a table'),
        h('div', { html:
          'A coupling tensor $C$ is equivariant precisely when it is <em>invariant</em> as an ' +
          'element of $V_{\\ell_1} \\otimes V_{\\ell_2} \\otimes V_{\\ell_3}$, i.e. when ' +
          '$\\bigl(D^{\\ell_1}\\!\\otimes D^{\\ell_2}\\!\\otimes D^{\\ell_3}\\bigr)\\,\\mathrm{vec}(C) ' +
          '= \\mathrm{vec}(C)$ for every rotation. Stack that constraint for a handful of generic ' +
          'rotations — generic rotations generate a dense subgroup, so a finite handful already ' +
          'pins down the exact invariant subspace — and take the smallest singular vector. The ' +
          'smallest singular value being ~0 says a coupling exists; the second-smallest being of ' +
          'order 1 says it is unique.' })),

      h('p', { class: 'prose' },
        'Run it below. Pick two input degrees and see which outputs are reachable, then let the ' +
        'page solve for the coupling and test it. For $\\ell_1 = \\ell_2 = 1$ the three allowed ' +
        'outputs are objects you already know: the dot product, the cross product, and the ' +
        'traceless outer product.'));

    const cgDemo = h('div', { class: 'demo' });
    cgDemo.append(h('h3', {}, 'Derive a coupling, then verify it'),
      h('p', { class: 'hint' }, 'Solved live — nothing below is read from a table of coefficients.'));
    const cgPaths = h('div', { class: 'controls' });
    const cgRead = h('div', { class: 'readout' });
    let cl1 = 1, cl2 = 1, cl3 = 1;

    function pathButtons() {
      cgPaths.innerHTML = '';
      const opts = [];
      for (let l3 = 0; l3 <= 6; l3++) opts.push({ l3, ok: cgAllowed(cl1, cl2, l3) });
      const seg = h('span', { class: 'seg' });
      opts.forEach(({ l3, ok }) => {
        const b = h('button', {
          type: 'button', disabled: !ok, 'aria-pressed': String(ok && l3 === cl3),
          title: ok ? '' : 'forbidden by the triangle rule',
          onclick: () => { cl3 = l3; pathButtons(); runCG(); },
        }, `→ ${l3}`);
        seg.appendChild(b);
      });
      cgPaths.append(h('div', { class: 'ctl' }, h('label', {}, 'output degree'), seg));
    }

    function runCG() {
      if (!cgAllowed(cl1, cl2, cl3)) { cl3 = Math.abs(cl1 - cl2); pathButtons(); }
      const t0 = performance.now();
      const { C, sigmaMin, sigmaSecond } = clebschGordan(cl1, cl2, cl3);
      const ms = performance.now() - t0;

      const rng = mulberry32(909);
      let worst = 0;
      for (let t = 0; t < 8; t++) {
        const R = randomRotation(rng);
        const u = Array.from({ length: 2 * cl1 + 1 }, () => 2 * rng() - 1);
        const w = Array.from({ length: 2 * cl2 + 1 }, () => 2 * rng() - 1);
        const lhs = tensorProduct(C, matvec(wignerD(cl1, R), u), matvec(wignerD(cl2, R), w));
        const rhs = matvec(wignerD(cl3, R), tensorProduct(C, u, w));
        const s = Math.max(norm(rhs), 1e-12);
        worst = Math.max(worst, Math.max(...lhs.map((v, i) => Math.abs(v - rhs[i]) / s)));
      }

      let named = '';
      if (cl1 === 1 && cl2 === 1) {
        const a = [0.3, -1.1, 0.7], b = [-0.4, 0.25, 1.3];
        const got = tensorProduct(C, realSH(1, a), realSH(1, b));
        if (cl3 === 0) {
          const want = dot(a, b) / (norm(a) * norm(b));
          named = `\n<span class="dim">identified: this is the DOT PRODUCT.  ` +
            `C(Y₁(a), Y₁(b)) / (â·b̂) = ${(got[0] / want).toFixed(9)}</span>`;
        } else if (cl3 === 1) {
          const cr = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
          const want = [cr[1], cr[2], cr[0]];
          const ratios = got.map((g, i) => g / want[i]);
          named = `\n<span class="dim">identified: this is the CROSS PRODUCT.  ` +
            `component ratios ${ratios.map((r) => r.toFixed(6)).join(', ')} — all equal, so it ` +
            `agrees up to one global constant.</span>`;
        } else if (cl3 === 2) {
          named = `\n<span class="dim">identified: the traceless symmetric outer product — ` +
            `the quadrupole of the pair.</span>`;
        }
      }

      cgRead.innerHTML =
        `coupling  (${cl1}, ${cl2}) → ${cl3}    tensor shape ${2 * cl1 + 1} × ${2 * cl2 + 1} × ${2 * cl3 + 1}` +
        `   <span class="dim">solved in ${ms.toFixed(0)} ms</span>\n\n` +
        `σ_min   ${sigmaMin.toExponential(3)}   <span class="dim">→ an invariant coupling exists` +
        ` (floor here is ~1.5e-8, see the note below)</span>\n` +
        `σ_2     ${sigmaSecond.toExponential(3)}   <span class="dim">→ order 1, so it is the ONLY one` +
        ` — multiplicity exactly 1</span>\n\n` +
        `equivariance test on 8 random rotations and random inputs:\n` +
        `  max |C(Du, Dv) − D C(u, v)| / |C(u,v)|   <span class="${worst < 1e-8 ? 'ok' : 'bad'}">` +
        `${worst.toExponential(2)}</span>` + named;
    }

    cgDemo.append(
      h('div', { class: 'controls' },
        segmented({ label: 'input ℓ₁', options: [0, 1, 2, 3].map((l) => ({ label: String(l), value: l })),
          value: 1, onPick: (v) => { cl1 = v; pathButtons(); runCG(); } }),
        segmented({ label: 'input ℓ₂', options: [0, 1, 2, 3].map((l) => ({ label: String(l), value: l })),
          value: 1, onPick: (v) => { cl2 = v; pathButtons(); runCG(); } })),
      cgPaths, cgRead);
    root.append(cgDemo);
    pathButtons();
    runCG();

    root.append(
      h('div', { class: 'note' },
        h('span', { class: 'tag' }, 'On that σ_min floor'),
        h('div', { html:
          'This page gets singular values from the eigenvalues of $A^\\top A$, and squaring puts a ' +
          'floor of $\\sqrt{\\varepsilon} \\approx 1.5\\times10^{-8}$ under $\\sigma_{\\min}$ no ' +
          'matter how exact the null space is. <code>python/e3.py</code> runs a true SVD on the ' +
          'same construction and reports ~$3\\times10^{-15}$. The equivariance residual, which is ' +
          'measured directly and lands at ~$10^{-15}$ in both, is the sharp test.' })),

      h('h2', {}, 'What you now have'),
      h('p', { class: 'prose' },
        'A way to turn a bond direction into features of every degree; a way to say exactly how ' +
        'each degree turns; and a complete, forced list of the ways two features may be ' +
        'multiplied. That is the entire toolkit. Chapter 4 assembles it into a layer.'),
    );
  },
};
