// Every interaction path equation 8 allows.
//
// Equation 8 carries six superscripts, (l_o, p_o, l_f, l_i, p_f, p_i), and one
// learnable radial function per combination. Which combinations survive is not a
// design choice — it is fixed by two selection rules:
//
//     triangle   |l_i - l_f|  <=  l_o  <=  l_i + l_f
//     parity     p_o = p_i * p_f,   with  p_f = (-1)^{l_f}  for a Y_{l_f} filter
//
// This enumerates them, so the index soup becomes a countable object.

import { h, slider, segmented, PALETTE } from './ui.js';
import { clebschGordan, cgAllowed, wignerD, tensorProduct, randomRotation, mulberry32 } from './e3.js';
import { matvec, norm } from './linalg.js';

const par = (p) => (p > 0 ? 'e' : 'o');
const irrep = (l, p) => `${l}${par(p)}`;

/** All (l_i,p_i) x (l_f) -> (l_o,p_o) paths permitted at a given l_max.
 *  The filter's parity is not free: a degree-l_f spherical harmonic has parity
 *  (-1)^{l_f}, so p_f is determined by l_f. */
export function enumeratePaths(lmax, parities = 'both') {
  const inputs = [];
  for (let l = 0; l <= lmax; l++) {
    if (parities === 'both') { inputs.push([l, 1]); inputs.push([l, -1]); }
    else inputs.push([l, Math.pow(-1, l)]);      // "natural" parity only, i.e. Y_l
  }
  const paths = [];
  for (const [li, pi] of inputs) {
    for (let lf = 0; lf <= lmax; lf++) {
      const pf = Math.pow(-1, lf);
      for (let lo = Math.abs(li - lf); lo <= Math.min(li + lf, lmax); lo++) {
        const po = pi * pf;
        if (!cgAllowed(li, lf, lo, pi, pf, po)) continue;
        paths.push({ li, pi, lf, pf, lo, po });
      }
    }
  }
  return paths;
}

export function pathExplorer() {
  const root = h('div');
  let lmax = 2;
  let parities = 'both';
  let selected = null;

  const grid = h('div');
  const summary = h('div', { class: 'readout' });
  const detail = h('div', { class: 'readout', style: { marginTop: '10px' } });

  function draw() {
    const paths = enumeratePaths(lmax, parities);
    // group by (l_i,p_i) -> rows, l_f -> columns
    const inputs = [...new Set(paths.map((p) => `${p.li}|${p.pi}`))];
    const filters = [...new Set(paths.map((p) => p.lf))].sort((a, b) => a - b);

    const rows = inputs.map((key) => {
      const [li, pi] = key.split('|').map(Number);
      const cells = filters.map((lf) => {
        const here = paths.filter((p) => p.li === li && p.pi === pi && p.lf === lf);
        if (!here.length) return h('td', { class: 'num', style: { color: '#C9CFD6' } }, '—');
        return h('td', { class: 'num' },
          ...here.map((p, k) => h('button', {
            type: 'button',
            style: { padding: '1px 7px', margin: '1px', fontSize: '12px',
              background: selected && selected.li === p.li && selected.pi === p.pi &&
                selected.lf === p.lf && selected.lo === p.lo ? PALETTE[0] : '#F2F3F6',
              color: selected && selected.li === p.li && selected.pi === p.pi &&
                selected.lf === p.lf && selected.lo === p.lo ? '#fff' : '#1B2733' },
            onclick: () => { selected = p; draw(); },
          }, irrep(p.lo, p.po))));
      });
      return h('tr', {}, h('td', {}, h('code', {}, irrep(li, pi))), ...cells);
    });

    grid.innerHTML = '';
    grid.appendChild(h('table', {},
      h('thead', {},
        h('tr', {}, h('th', {}, 'input  ↓'),
          ...filters.map((lf) => h('th', { class: 'num' }, `filter Y${lf} (${irrep(lf, Math.pow(-1, lf))})`)))),
      h('tbody', {}, rows)));

    // one learnable radial function per path per channel
    summary.innerHTML =
      `l_max = ${lmax},  ${parities === 'both' ? 'both parities carried' : 'natural parity only'}\n` +
      `input irreps            ${inputs.map((k) => { const [l, p] = k.split('|').map(Number); return irrep(l, p); }).join(', ')}\n` +
      `allowed paths           <span class="ok">${paths.length}</span>` +
      `   <span class="dim">each one gets its own learnable radial function R, per channel</span>\n` +
      `growth with l_max       ` +
      [0, 1, 2, 3].map((L) => `${L}:${enumeratePaths(L, parities).length}`).join('   ') +
      `\n\n<span class="dim">Cells show the output irrep. Click one to build its coupling and test it.</span>`;

    if (!selected) {
      detail.innerHTML = '<span class="dim">no path selected</span>';
      return;
    }
    const p = selected;
    const t0 = performance.now();
    const { C, sigmaMin, sigmaSecond } = clebschGordan(p.li, p.lf, p.lo);
    const ms = performance.now() - t0;
    const rng = mulberry32(4242);
    let worst = 0;
    for (let t = 0; t < 6; t++) {
      const R = randomRotation(rng);
      const u = Array.from({ length: 2 * p.li + 1 }, () => 2 * rng() - 1);
      const w = Array.from({ length: 2 * p.lf + 1 }, () => 2 * rng() - 1);
      const lhs = tensorProduct(C, matvec(wignerD(p.li, R), u), matvec(wignerD(p.lf, R), w));
      const rhs = matvec(wignerD(p.lo, R), tensorProduct(C, u, w));
      const s = Math.max(norm(rhs), 1e-12);
      worst = Math.max(worst, Math.max(...lhs.map((v, i) => Math.abs(v - rhs[i]) / s)));
    }
    const nCG = (2 * p.li + 1) * (2 * p.lf + 1) * (2 * p.lo + 1);
    detail.innerHTML =
      `selected path   ${irrep(p.li, p.pi)}  ⊗  Y${p.lf} (${irrep(p.lf, p.pf)})  →  ${irrep(p.lo, p.po)}\n\n` +
      `triangle rule   |${p.li} − ${p.lf}| = ${Math.abs(p.li - p.lf)}  ≤  ${p.lo}  ≤  ${p.li + p.lf}` +
      `   <span class="ok">satisfied</span>\n` +
      `parity rule     p_o = p_i · p_f = (${par(p.pi)})·(${par(p.pf)}) = ${par(p.po)}` +
      `   <span class="ok">satisfied</span>\n\n` +
      `C tensor        ${2 * p.li + 1} × ${2 * p.lf + 1} × ${2 * p.lo + 1} = ${nCG} numbers,` +
      ` <strong>none of them learnable</strong>   <span class="dim">solved in ${ms.toFixed(0)} ms</span>\n` +
      `uniqueness      σ_min ${sigmaMin.toExponential(2)},  σ_2 ${sigmaSecond.toExponential(2)}` +
      `   <span class="dim">one-dimensional, so the coupling is forced</span>\n` +
      `equivariance    max relative deviation <span class="${worst < 1e-8 ? 'ok' : 'bad'}">` +
      `${worst.toExponential(2)}</span>\n\n` +
      `learnable here  the single radial function R^(${p.lf},${p.li},${par(p.pf)},${par(p.pi)})_c(r_ab),` +
      ` one per channel c`;
  }

  root.append(
    h('div', { class: 'controls' },
      slider({ label: 'ℓ_max', min: 0, max: 3, step: 1, value: 2,
        format: String, onInput: (v) => { lmax = v; selected = null; draw(); } }),
      segmented({
        label: 'feature parities',
        options: [{ label: 'both (e and o)', value: 'both' },
          { label: 'natural only', value: 'natural' }],
        value: 'both',
        onPick: (v) => { parities = v; selected = null; draw(); },
      })),
    grid, summary, detail);

  draw();
  return root;
}
