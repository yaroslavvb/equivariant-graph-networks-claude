// Shared UI helpers for tutorial chapters: DOM building, plotting, results loading.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** Load a JSON results file produced by the Python experiments. */
const cache = new Map();
export async function loadResults(name) {
  if (cache.has(name)) return cache.get(name);
  const p = fetch(`../results/${name}.json`).then((r) => {
    if (!r.ok) throw new Error(`results/${name}.json not found (HTTP ${r.status}) — serve the repo root over HTTP`);
    return r.json();
  });
  cache.set(name, p);
  return p;
}

/** A labelled slider. onInput receives the numeric value. */
export function slider({ label, min, max, step = 1, value, format = (v) => v, onInput }) {
  const out = h('span', { class: 'val' }, format(value));
  const input = h('input', {
    type: 'range', min: String(min), max: String(max), step: String(step), value: String(value),
    oninput: (e) => {
      const v = Number(e.target.value);
      out.textContent = format(v);
      onInput(v);
    },
  });
  const wrap = h('div', { class: 'ctl' }, h('label', {}, label), input, out);
  wrap.setValue = (v) => { input.value = String(v); out.textContent = format(v); onInput(v); };
  return wrap;
}

/** A segmented button group. Returns the element; onPick receives the value. */
export function segmented({ label, options, value, onPick }) {
  const seg = h('span', { class: 'seg' });
  const buttons = options.map((o) => {
    const b = h('button', {
      type: 'button',
      'aria-pressed': String(o.value === value),
      onclick: () => {
        buttons.forEach((bb) => bb.setAttribute('aria-pressed', 'false'));
        b.setAttribute('aria-pressed', 'true');
        onPick(o.value);
      },
    }, o.label);
    seg.appendChild(b);
    return b;
  });
  return h('div', { class: 'ctl' }, label ? h('label', {}, label) : null, seg);
}

export function checkLine(pass, name, detail) {
  return h('div', { class: 'checkline' },
    h('span', { class: `mark ${pass ? 'ok' : 'bad'}` }, pass ? '[PASS]' : '[FAIL]'),
    h('span', {}, name),
    detail ? h('span', { class: 'detail' }, detail) : null);
}

export const fmt = (x, d = 3) => {
  if (!Number.isFinite(x)) return String(x);
  if (x !== 0 && (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5)) return x.toExponential(2);
  return x.toFixed(d);
};

// ---------------------------------------------------------------------------
// A very small SVG plotting layer — enough for line plots and scatter, themed
// to match the prose.
// ---------------------------------------------------------------------------

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, String(v));
  return e;
};

export class Plot {
  constructor({ width = 640, height = 340, pad = { l: 62, r: 18, t: 16, b: 46 },
                xLabel = '', yLabel = '', xLog = false, yLog = false } = {}) {
    this.w = width; this.h = height; this.pad = pad;
    this.xLog = xLog; this.yLog = yLog;
    this.svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}`, width: '100%' });
    this.xLabel = xLabel; this.yLabel = yLabel;
    this.series = [];
    this.xlim = null; this.ylim = null;
  }
  setLimits(xlim, ylim) { this.xlim = xlim; this.ylim = ylim; return this; }
  add(series) { this.series.push(series); return this; }
  _tx(v) {
    const [a, b] = this.xlim;
    const f = this.xLog ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a))
                        : (v - a) / (b - a);
    return this.pad.l + f * (this.w - this.pad.l - this.pad.r);
  }
  _ty(v) {
    const [a, b] = this.ylim;
    const f = this.yLog ? (Math.log10(v) - Math.log10(a)) / (Math.log10(b) - Math.log10(a))
                        : (v - a) / (b - a);
    return this.h - this.pad.b - f * (this.h - this.pad.t - this.pad.b);
  }
  _autolim() {
    if (this.xlim && this.ylim) return;
    let xs = [], ys = [];
    for (const s of this.series) for (const p of s.points) { xs.push(p[0]); ys.push(p[1]); }
    const span = (arr, log) => {
      let lo = Math.min(...arr), hi = Math.max(...arr);
      if (log) { lo = Math.max(lo, 1e-300); hi = Math.max(hi, lo * 10); return [lo / 1.6, hi * 1.6]; }
      const m = (hi - lo) * 0.08 || 1;
      return [lo - m, hi + m];
    };
    if (!this.xlim) this.xlim = span(xs, this.xLog);
    if (!this.ylim) this.ylim = span(ys, this.yLog);
  }
  _ticks(lim, log, n = 5) {
    const [a, b] = lim;
    if (log) {
      const out = [];
      for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) {
        const v = Math.pow(10, e);
        if (v >= a && v <= b) out.push(v);
      }
      return out.length >= 2 ? out : [a, b];
    }
    const raw = (b - a) / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + 1e-12; v += step) out.push(v);
    return out;
  }
  render() {
    this._autolim();
    const { svg } = this;
    svg.innerHTML = '';
    const L = this.pad.l, R = this.w - this.pad.r, T = this.pad.t, B = this.h - this.pad.b;

    const fmtTick = (v) => {
      if (v === 0) return '0';
      const a = Math.abs(v);
      if (a < 1e-3 || a >= 1e5) return v.toExponential(0).replace('e+', 'e');
      if (a >= 1000) return String(Math.round(v));
      return String(Number(v.toPrecision(3)));
    };

    for (const t of this._ticks(this.xlim, this.xLog)) {
      const x = this._tx(t);
      svg.appendChild(svgEl('line', { x1: x, x2: x, y1: T, y2: B, stroke: '#EDF0F3', 'stroke-width': 1 }));
      const lab = svgEl('text', { x, y: B + 18, 'text-anchor': 'middle',
        'font-size': 11.5, fill: '#5A6773', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
      lab.textContent = fmtTick(t);
      svg.appendChild(lab);
    }
    for (const t of this._ticks(this.ylim, this.yLog)) {
      const y = this._ty(t);
      svg.appendChild(svgEl('line', { x1: L, x2: R, y1: y, y2: y, stroke: '#EDF0F3', 'stroke-width': 1 }));
      const lab = svgEl('text', { x: L - 8, y: y + 4, 'text-anchor': 'end',
        'font-size': 11.5, fill: '#5A6773', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
      lab.textContent = fmtTick(t);
      svg.appendChild(lab);
    }

    svg.appendChild(svgEl('line', { x1: L, x2: R, y1: B, y2: B, stroke: '#C9CFD6', 'stroke-width': 1.2 }));
    svg.appendChild(svgEl('line', { x1: L, x2: L, y1: T, y2: B, stroke: '#C9CFD6', 'stroke-width': 1.2 }));

    if (this.xLabel) {
      const t = svgEl('text', { x: (L + R) / 2, y: this.h - 8, 'text-anchor': 'middle',
        'font-size': 12.5, fill: '#1B2733', 'font-family': 'ui-sans-serif, system-ui, sans-serif' });
      t.textContent = this.xLabel; svg.appendChild(t);
    }
    if (this.yLabel) {
      const t = svgEl('text', { x: 14, y: (T + B) / 2, 'text-anchor': 'middle',
        'font-size': 12.5, fill: '#1B2733', transform: `rotate(-90 14 ${(T + B) / 2})`,
        'font-family': 'ui-sans-serif, system-ui, sans-serif' });
      t.textContent = this.yLabel; svg.appendChild(t);
    }

    for (const s of this.series) {
      const pts = s.points.filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])
        && (!this.xLog || p[0] > 0) && (!this.yLog || p[1] > 0));
      if (!pts.length) continue;
      if (s.type !== 'scatter') {
        const d = pts.map((p, i) => `${i ? 'L' : 'M'}${this._tx(p[0]).toFixed(2)},${this._ty(p[1]).toFixed(2)}`).join('');
        svg.appendChild(svgEl('path', { d, fill: 'none', stroke: s.color || '#1F4E79',
          'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || null,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: s.opacity ?? 1 }));
      }
      if (s.type === 'scatter' || s.markers) {
        for (const p of pts) {
          svg.appendChild(svgEl('circle', { cx: this._tx(p[0]), cy: this._ty(p[1]),
            r: s.r || 3, fill: s.color || '#1F4E79', opacity: s.opacity ?? 0.85 }));
        }
      }
    }
    return svg;
  }
  legend() {
    const box = h('div', { class: 'controls', style: { gap: '14px', marginTop: '4px' } });
    for (const s of this.series) {
      if (!s.label) continue;
      box.appendChild(h('span', { class: 'ctl', style: { fontSize: '13px' } },
        h('span', { style: { display: 'inline-block', width: '14px', height: '3px',
          borderRadius: '2px', background: s.color || '#1F4E79',
          opacity: String(s.opacity ?? 1) } }),
        h('span', { style: { color: '#5A6773' } }, s.label)));
    }
    return box;
  }
}

export const PALETTE = ['#1F4E79', '#B5443C', '#C9A227', '#3E7C59', '#6C4A8C', '#5A6773'];
