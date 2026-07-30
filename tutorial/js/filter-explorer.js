// The convolution filter, taken apart.
//
//      S^(l)_m(r_ij)  =  R(|r_ij|)  ·  Y^(l)_m(rhat_ij)
//                        \_______/     \____________/
//                        learnable       fixed
//
// Everything the network is free to choose sits in R, which sees only a distance
// and so cannot know which way the bond points. Everything that knows about
// direction sits in Y, which has no parameters. That split is the whole reason
// the filter is equivariant, and it is what this explorer makes visible: move
// the radial sliders as violently as you like and the equivariance residual does
// not move off machine precision, because you are not touching the part that
// carries the transformation law.
//
// Used by tutorial chapter 4 and by interactive/filter-explorer.html.

import { h, slider, segmented, PALETTE } from './ui.js';
import { realSH, wignerD, envelope, besselBasis, randomRotation, mulberry32 } from './e3.js';
import { matvec, norm, rotationMatrix, transpose, matmul } from './linalg.js';

const R_CUT = 2.6;
const N_BASIS = 4;
const GRID = 132;          // heatmap resolution, in pixels per side
const EXTENT = R_CUT * 1.06;

/** The learnable radial function: a Bessel expansion times a smooth envelope
 *  that takes the filter, its slope and its curvature to zero at the cutoff. */
function radial(r, coeffs) {
  if (r <= 0 || r >= R_CUT) return 0;
  let s = 0;
  for (let n = 0; n < coeffs.length; n++) s += coeffs[n] * besselBasis(r, n + 1, R_CUT);
  return s * envelope(r, R_CUT);
}

/** The filter component S^(l)_m at a displacement vector. `tilt` is the
 *  deliberately broken variant: it lets the radial factor peek at direction,
 *  which is exactly the thing the constraint forbids. */
function filterValue(l, m, vec, coeffs, tilt) {
  const r = norm(vec);
  if (r <= 0 || r >= R_CUT) return 0;
  let R = radial(r, coeffs);
  if (tilt !== 0) R *= 1 + tilt * (vec[2] / r);   // zhat . rhat — a preferred axis
  return R * realSH(l, vec)[m + l];
}

/** Sum a filter over a set of neighbour displacements, giving the (2l+1)-vector
 *  the convolution would actually produce. */
function filterSum(l, vecs, coeffs, tilt) {
  const out = new Array(2 * l + 1).fill(0);
  for (const v of vecs) {
    for (let m = -l; m <= l; m++) out[m + l] += filterValue(l, m, v, coeffs, tilt);
  }
  return out;
}

function paintPanel(canvas, l, m, coeffs, tilt, basis, scale) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(GRID, GRID);
  const [e1, e2] = basis;
  let localMax = 0;
  const vals = new Float64Array(GRID * GRID);
  for (let j = 0; j < GRID; j++) {
    const v = EXTENT * (1 - (2 * j) / (GRID - 1));
    for (let i = 0; i < GRID; i++) {
      const u = EXTENT * ((2 * i) / (GRID - 1) - 1);
      const vec = [e1[0] * u + e2[0] * v, e1[1] * u + e2[1] * v, e1[2] * u + e2[2] * v];
      const s = filterValue(l, m, vec, coeffs, tilt);
      vals[j * GRID + i] = s;
      localMax = Math.max(localMax, Math.abs(s));
    }
  }
  const denom = scale > 0 ? scale : (localMax || 1);
  for (let k = 0; k < GRID * GRID; k++) {
    const t = Math.max(-1, Math.min(1, vals[k] / denom));
    const a = Math.abs(t);
    // diverging ramp: white at zero, repo blue for positive, repo red for negative
    const [r0, g0, b0] = t >= 0 ? [31, 78, 121] : [181, 68, 60];
    img.data[4 * k] = Math.round(255 + a * (r0 - 255));
    img.data[4 * k + 1] = Math.round(255 + a * (g0 - 255));
    img.data[4 * k + 2] = Math.round(255 + a * (b0 - 255));
    img.data[4 * k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  // cutoff circle, so the envelope's support is visible
  ctx.strokeStyle = 'rgba(90,103,115,0.45)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(GRID / 2, GRID / 2, (R_CUT / EXTENT) * (GRID / 2), 0, 2 * Math.PI);
  ctx.stroke();
  return localMax;
}

export function filterExplorer() {
  const root = h('div');
  let l = 1;
  let coeffs = [1, 0.35, -0.2, 0.1];
  let tilt = 0;
  let azimuth = 30 * Math.PI / 180;   // slice plane spanned by e1(azimuth) and z
  let rotAngle = 0;

  // A fixed neighbourhood for the equivariance test.
  const rng0 = mulberry32(90210);
  const NEIGH = Array.from({ length: 6 }, () => {
    const d = [2 * rng0() - 1, 2 * rng0() - 1, 2 * rng0() - 1];
    const n = norm(d) || 1;
    const target = 0.7 + 1.5 * rng0();
    return d.map((x) => (x / n) * target);
  });

  // ---- panels ------------------------------------------------------------
  const radialSvg = h('svg', { viewBox: '0 0 420 190', width: '420', height: '190',
    style: { width: '100%', maxWidth: '420px', height: 'auto', display: 'block',
      background: '#fff', borderRadius: '6px' } });
  const panelRow = h('div', { style: { display: 'flex', gap: '10px', flexWrap: 'wrap',
    alignItems: 'flex-start', marginTop: '4px' } });
  const wignerBox = h('div');
  const readout = h('div', { class: 'readout' });
  const note = h('div');

  function drawRadial() {
    const W = 420, H = 190, L = 44, Rr = 12, T = 12, B = 34;
    const ns = 'http://www.w3.org/2000/svg';
    const el = (t, a) => { const e = document.createElementNS(ns, t);
      for (const [k, v] of Object.entries(a)) e.setAttribute(k, String(v)); return e; };
    radialSvg.innerHTML = '';
    const pts = [];
    let maxAbs = 1e-9;
    for (let i = 0; i <= 300; i++) {
      const r = (R_CUT * i) / 300;
      const y = radial(r, coeffs);
      pts.push([r, y]);
      maxAbs = Math.max(maxAbs, Math.abs(y));
    }
    const tx = (r) => L + (r / R_CUT) * (W - L - Rr);
    const ty = (y) => (T + H - B) / 2 + (H - T - B) / 2 * (-y / (maxAbs * 1.1)) + (T - 0) * 0;
    const midY = ty(0);
    radialSvg.appendChild(el('line', { x1: L, x2: W - Rr, y1: midY, y2: midY,
      stroke: '#C9CFD6', 'stroke-width': 1 }));
    radialSvg.appendChild(el('line', { x1: L, x2: L, y1: T, y2: H - B,
      stroke: '#C9CFD6', 'stroke-width': 1 }));
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${tx(p[0]).toFixed(1)},${ty(p[1]).toFixed(1)}`).join('');
    radialSvg.appendChild(el('path', { d, fill: 'none', stroke: PALETTE[0], 'stroke-width': 2.4 }));
    // cutoff marker
    radialSvg.appendChild(el('line', { x1: tx(R_CUT), x2: tx(R_CUT), y1: T, y2: H - B,
      stroke: '#B5443C', 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }));
    const lab = el('text', { x: tx(R_CUT) - 4, y: H - B + 14, 'text-anchor': 'end',
      'font-size': 11, fill: '#B5443C', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    lab.textContent = 'r_cut';
    radialSvg.appendChild(lab);
    const xlab = el('text', { x: (L + W - Rr) / 2, y: H - 8, 'text-anchor': 'middle',
      'font-size': 11.5, fill: '#5A6773', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    xlab.textContent = 'bond length';
    radialSvg.appendChild(xlab);
    const ylab = el('text', { x: 13, y: (T + H - B) / 2, 'text-anchor': 'middle',
      'font-size': 11.5, fill: '#5A6773', transform: `rotate(-90 13 ${(T + H - B) / 2})`,
      'font-family': 'ui-sans-serif, system-ui, sans-serif' });
    ylab.textContent = 'R';
    radialSvg.appendChild(ylab);
  }

  function sliceBasis() {
    const Rot = rotationMatrix([0.31, 0.42, 0.85], rotAngle);
    const e1 = matvec(Rot, [Math.cos(azimuth), Math.sin(azimuth), 0]);
    const e2 = matvec(Rot, [0, 0, 1]);
    return [e1, e2];
  }

  function drawPanels() {
    panelRow.innerHTML = '';
    const basis = sliceBasis();
    // one common colour scale across the row, so relative magnitudes are honest
    let globalMax = 0;
    const probe = document.createElement('canvas');
    probe.width = probe.height = GRID;
    for (let m = -l; m <= l; m++) {
      globalMax = Math.max(globalMax, paintPanel(probe, l, m, coeffs, tilt, basis, 0));
    }
    const dead = [];
    for (let m = -l; m <= l; m++) {
      const cv = h('canvas', { width: GRID, height: GRID,
        style: { width: '118px', height: '118px', border: '1px solid #E2E6EA', borderRadius: '5px' } });
      const localMax = paintPanel(cv, l, m, coeffs, tilt, basis, globalMax);
      if (localMax < 1e-12 * Math.max(globalMax, 1e-12)) dead.push(m);
      panelRow.appendChild(h('div', { style: { textAlign: 'center' } }, cv,
        h('div', { class: 'sans', style: { fontSize: '12px', color: '#5A6773', marginTop: '3px' } },
          `m = ${m > 0 ? '+' : ''}${m}`)));
    }

    note.innerHTML = dead.length
      ? `<p class="hint" style="margin:8px 0 0">Components ` +
        `<strong>m = ${dead.map((m) => (m > 0 ? '+' : '') + m).join(', ')}</strong> are blank in ` +
        `this slice. That is real, not a drawing bug: those harmonics are odd in the direction ` +
        `perpendicular to the slice, so they vanish identically on it. Sweep the slice azimuth and ` +
        `they reappear.</p>`
      : '';
  }

  function drawWigner() {
    const Rot = rotationMatrix([0.31, 0.42, 0.85], rotAngle);
    const D = wignerD(l, Rot);
    const n = D.length, cell = 26;
    const cv = h('canvas', { width: n * cell, height: n * cell,
      style: { border: '1px solid #E2E6EA', borderRadius: '5px' } });
    const ctx = cv.getContext('2d');
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = D[i][j], a = Math.min(1, Math.abs(v));
        const [r0, g0, b0] = v >= 0 ? [31, 78, 121] : [181, 68, 60];
        ctx.fillStyle = `rgb(${Math.round(255 + a * (r0 - 255))},` +
          `${Math.round(255 + a * (g0 - 255))},${Math.round(255 + a * (b0 - 255))})`;
        ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
      }
    }
    wignerBox.innerHTML = '';
    wignerBox.appendChild(h('div', { style: { textAlign: 'center' } }, cv,
      h('div', { class: 'sans', style: { fontSize: '12px', color: '#5A6773', marginTop: '3px' } },
        `D^${l}(R), ${n}×${n}`)));
  }

  function updateReadout() {
    const Rot = rotationMatrix([0.31, 0.42, 0.85], rotAngle);
    const before = filterSum(l, NEIGH, coeffs, tilt);
    const after = filterSum(l, NEIGH.map((v) => matvec(Rot, v)), coeffs, tilt);
    const predicted = matvec(wignerD(l, Rot), before);
    const denom = Math.max(norm(predicted), 1e-12);
    let worst = 0;
    for (let i = 0; i < after.length; i++) {
      worst = Math.max(worst, Math.abs(after[i] - predicted[i]) / denom);
    }
    // the radial factor is a scalar of |r| alone, so rotation cannot touch it
    const radiiBefore = NEIGH.map((v) => norm(v));
    const radiiAfter = NEIGH.map((v) => norm(matvec(Rot, v)));
    const dR = Math.max(...radiiBefore.map((r, i) => Math.abs(r - radiiAfter[i])));

    const ok = worst < 1e-9;
    const fmtv = (a) => '[' + a.map((v) => v.toFixed(6).padStart(10)).join(' ') + ']';
    readout.innerHTML =
      `rotation ${(rotAngle * 180 / Math.PI).toFixed(0)}°` +
      `   radial coefficients [${coeffs.map((c) => c.toFixed(2)).join(', ')}]` +
      (tilt !== 0 ? `   <span class="bad">radial tilt ${tilt.toFixed(2)}</span>` : '') + `\n\n` +
      `every |r_ij| under rotation   max change ${dR.toExponential(2)}` +
      `   <span class="dim">the radial factor cannot see the rotation at all</span>\n\n` +
      `sum_j S(r_j)        ${fmtv(before)}\n` +
      `sum_j S(R r_j)      ${fmtv(after)}\n` +
      `D(R) · sum_j S(r_j) ${fmtv(predicted)}\n` +
      `relative residual   <span class="${ok ? 'ok' : 'bad'}">${worst.toExponential(2)}</span>` +
      (ok
        ? `   <span class="dim">equivariant</span>`
        : `   <span class="dim">BROKEN — the radial factor is reading direction</span>`);
  }

  function redraw() { drawRadial(); drawPanels(); drawWigner(); updateReadout(); }

  // ---- controls ----------------------------------------------------------
  const coeffControls = h('div', { class: 'controls' },
    ...coeffs.map((c, n) => slider({
      label: `c${n + 1}`, min: -100, max: 100, step: 1, value: Math.round(c * 100),
      format: (v) => (v / 100).toFixed(2),
      onInput: (v) => { coeffs[n] = v / 100; redraw(); },
    })));

  const topControls = h('div', { class: 'controls' },
    segmented({
      label: 'degree ℓ',
      options: [0, 1, 2, 3].map((v) => ({ label: `ℓ=${v}`, value: v })),
      value: 1,
      onPick: (v) => { l = v; redraw(); },
    }),
    slider({ label: 'rotate', min: 0, max: 360, step: 1, value: 0,
      format: (v) => `${v}°`, onInput: (v) => { rotAngle = v * Math.PI / 180; redraw(); } }),
    slider({ label: 'slice azimuth', min: 0, max: 180, step: 1, value: 30,
      format: (v) => `${v}°`, onInput: (v) => { azimuth = v * Math.PI / 180; redraw(); } }));

  const breakControls = h('div', { class: 'controls' },
    slider({ label: 'let R peek at direction', min: 0, max: 90, step: 1, value: 0,
      format: (v) => (v === 0 ? 'off' : `α = ${(v / 100).toFixed(2)}`),
      onInput: (v) => { tilt = v / 100; redraw(); } }),
    h('button', { type: 'button', onclick: () => { coeffs = [1, 0.35, -0.2, 0.1]; redraw();
      coeffControls.querySelectorAll('input').forEach((inp, n) => {
        inp.value = String(Math.round(coeffs[n] * 100));
        inp.parentElement.querySelector('.val').textContent = coeffs[n].toFixed(2);
      }); } }, 'reset radial'));

  root.append(
    h('div', { class: 'grid2', style: { alignItems: 'start' } },
      h('div', {},
        h('h4', { class: 'sans', style: { fontSize: '13px', letterSpacing: '.02em',
          fontWeight: '700', color: PALETTE[0], margin: '0 0 6px' } },
          'The learnable half:  $R(\\lVert\\vec r_{ij}\\rVert)$'),
        radialSvg, coeffControls,
        h('p', { class: 'hint', style: { margin: '6px 0 0' } },
          'Four Bessel coefficients times a smooth envelope. Drag them anywhere you like — ' +
          'this is the only part of the filter the network fits.')),
      h('div', {},
        h('h4', { class: 'sans', style: { fontSize: '13px', letterSpacing: '.02em',
          fontWeight: '700', color: PALETTE[1], margin: '0 0 6px' } },
          'The fixed half: how a rotation mixes $m$'),
        wignerBox,
        h('p', { class: 'hint', style: { margin: '6px 0 0' } },
          'Rotating does not move each component independently. It mixes them within the degree, ' +
          'by exactly this matrix — which is why the 2ℓ+1 components are one object and not ' +
          '2ℓ+1 separate numbers.'))),
    topControls,
    h('h4', { class: 'sans', style: { fontSize: '13px', letterSpacing: '.02em',
      fontWeight: '700', color: '#5A6773', margin: '16px 0 4px' } },
      'Their product: the filter $S^{(\\ell)}_{m}(\\vec r_{ij})$ on a plane through the origin'),
    panelRow,
    note,
    breakControls,
    readout);

  redraw();
  return root;
}
