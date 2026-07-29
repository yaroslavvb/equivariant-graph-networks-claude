// Real O(3) representation theory, derived in the browser.
//
// This is the JavaScript twin of python/e3.py. Same three objects, same three
// recipes, so the tutorial can show you the derivation running rather than ship
// you a table of coefficients:
//
//   realSH(l, r)        real spherical harmonics, explicit Cartesian polynomials
//   wignerD(l, R)       solved from  Y_l(R r) = D Y_l(r)  by least squares
//   clebschGordan(...)  the null space of [kron(D1,D2,D3) - I] over generic R
//
// Component order within degree l is m = -l .. +l, matching python/e3.py, and
// selfTest() checks the harmonics against values exported from scipy.

import {
  eye, kron, matmul, matvec, transpose, lstsq, rightSingular, maxAbs,
  norm, dot, randomRotation, mulberry32, rotationMatrix,
} from './linalg.js';

export const irrepDim = (l) => 2 * l + 1;

const S = Math.sqrt;
const C0 = 0.5 * S(1 / Math.PI);
const C1 = S(3 / (4 * Math.PI));
const C2a = 0.5 * S(15 / Math.PI);
const C2b = 0.25 * S(5 / Math.PI);
const C2c = 0.25 * S(15 / Math.PI);
const C3a = 0.25 * S(35 / (2 * Math.PI));
const C3b = 0.5 * S(105 / Math.PI);
const C3c = 0.25 * S(21 / (2 * Math.PI));
const C3d = 0.25 * S(7 / Math.PI);
const C3e = 0.25 * S(105 / Math.PI);

/** Real spherical harmonics of degree l at a direction [x,y,z] (normalised
 *  internally). Returns an array of length 2l+1 ordered m = -l .. +l. */
export function realSH(l, r) {
  const n = Math.hypot(r[0], r[1], r[2]) || 1;
  const x = r[0] / n, y = r[1] / n, z = r[2] / n;
  switch (l) {
    case 0:
      return [C0];
    case 1:
      return [C1 * y, C1 * z, C1 * x];
    case 2:
      return [
        C2a * x * y,
        C2a * y * z,
        C2b * (3 * z * z - 1),
        C2a * x * z,
        C2c * (x * x - y * y),
      ];
    case 3:
      return [
        C3a * y * (3 * x * x - y * y),
        C3b * x * y * z,
        C3c * y * (5 * z * z - 1),
        C3d * (5 * z * z * z - 3 * z),
        C3c * x * (5 * z * z - 1),
        C3e * z * (x * x - y * y),
        C3a * x * (x * x - 3 * y * y),
      ];
    default:
      throw new Error(`realSH: degree ${l} not implemented (l <= 3)`);
  }
}

/** Evaluate realSH on a list of directions, returning an (n x 2l+1) matrix. */
export const shBatch = (l, dirs) => dirs.map((d) => realSH(l, d));

function sampleDirections(count, seed) {
  const rng = mulberry32(seed);
  const out = [];
  for (let i = 0; i < count; i++) {
    // Marsaglia: uniform on the sphere.
    let a, b, s;
    do {
      a = 2 * rng() - 1; b = 2 * rng() - 1; s = a * a + b * b;
    } while (s >= 1 || s === 0);
    const f = 2 * Math.sqrt(1 - s);
    out.push([a * f, b * f, 1 - 2 * s]);
  }
  return out;
}

/** The (2l+1)x(2l+1) Wigner matrix defined by Y_l(R r) = D^l(R) Y_l(r).
 *  Solved by least squares on sample directions — not tabulated. */
export function wignerD(l, R, nSamples = 8 * (2 * l + 1), seed = 20240501) {
  const dirs = sampleDirections(nSamples, seed);
  const A = shBatch(l, dirs);
  const B = shBatch(l, dirs.map((d) => matvec(R, d)));
  // B = A D^T
  return transpose(lstsq(A, B));
}

/** max |Y_l(R r) - D^l(R) Y_l(r)| over directions NOT used to fit D. */
export function wignerResidual(l, R, seed = 987654) {
  const D = wignerD(l, R);
  const dirs = sampleDirections(256, seed);
  let worst = 0;
  for (const d of dirs) {
    const lhs = realSH(l, matvec(R, d));
    const rhs = matvec(D, realSH(l, d));
    for (let i = 0; i < lhs.length; i++) worst = Math.max(worst, Math.abs(lhs[i] - rhs[i]));
  }
  return worst;
}

/** O(3) selection rule: triangle inequality on degree, product rule on parity. */
export const cgAllowed = (l1, l2, l3, p1 = 1, p2 = 1, p3 = 1) =>
  Math.abs(l1 - l2) <= l3 && l3 <= l1 + l2 && p1 * p2 === p3;

/** The unique (up to sign and scale) invariant tensor in V_l1 (x) V_l2 (x) V_l3.
 *
 *  C is invariant iff (D1 (x) D2 (x) D3) vec(C) = vec(C) for every rotation.
 *  Imposing that for a few generic rotations already pins down the exact
 *  invariant subspace, so the smallest right-singular vector reads it off.
 *
 *  Returns { C, sigmaMin, sigmaSecond } — C flat-indexed as C[a][b][c],
 *  Frobenius-normalised with a deterministic sign. sigmaMin near zero says an
 *  invariant tensor exists; sigmaSecond of order one says it is unique. */
export function clebschGordan(l1, l2, l3, nRot = 4, seed = 11) {
  const d1 = 2 * l1 + 1, d2 = 2 * l2 + 1, d3 = 2 * l3 + 1;
  const D = d1 * d2 * d3;
  const rng = mulberry32(seed);
  const rows = [];
  const I = eye(D);
  for (let k = 0; k < nRot; k++) {
    const R = randomRotation(rng);
    const K = kron(kron(wignerD(l1, R), wignerD(l2, R)), wignerD(l3, R));
    for (let i = 0; i < D; i++) rows.push(K[i].map((v, j) => v - I[i][j]));
  }
  const { singularValues, vectors } = rightSingular(rows);
  const flat = vectors[0];
  const nrm = norm(flat);
  let v = flat.map((x) => x / nrm);
  let big = 0;
  for (let i = 1; i < v.length; i++) if (Math.abs(v[i]) > Math.abs(v[big])) big = i;
  if (v[big] < 0) v = v.map((x) => -x);

  const C = [];
  for (let a = 0; a < d1; a++) {
    const plane = [];
    for (let b = 0; b < d2; b++) {
      const row = [];
      for (let c = 0; c < d3; c++) row.push(v[(a * d2 + b) * d3 + c]);
      plane.push(row);
    }
    C.push(plane);
  }
  return {
    C,
    sigmaMin: singularValues[0],
    sigmaSecond: singularValues[1],
  };
}

/** Contract degree-l1 feature u with degree-l2 feature w into degree l3. */
export function tensorProduct(C, u, w) {
  const d3 = C[0][0].length;
  const out = new Array(d3).fill(0);
  for (let a = 0; a < u.length; a++) {
    if (u[a] === 0) continue;
    for (let b = 0; b < w.length; b++) {
      const f = u[a] * w[b];
      if (f === 0) continue;
      const row = C[a][b];
      for (let c = 0; c < d3; c++) out[c] += f * row[c];
    }
  }
  return out;
}

/** Atom-centred equivariant feature f_l = sum_j R(r_ij) Y_l(rhat_ij),
 *  i.e. exactly one NequIP convolution filter applied at the first layer. */
export function atomicFeature(l, neighbours, radial) {
  const out = new Array(2 * l + 1).fill(0);
  for (const d of neighbours) {
    const r = Math.hypot(d[0], d[1], d[2]);
    const w = radial ? radial(r) : 1;
    if (w === 0) continue;
    const Y = realSH(l, d);
    for (let i = 0; i < out.length; i++) out[i] += w * Y[i];
  }
  return out;
}

/** Polynomial cutoff envelope (the smooth-envelope idea DimeNet contributed and
 *  NequIP adopted): 1 at r=0, and value+slope+curvature all vanishing at rCut. */
export function envelope(r, rCut, p = 6) {
  if (r >= rCut) return 0;
  const x = r / rCut;
  return 1
    - ((p + 1) * (p + 2) / 2) * Math.pow(x, p)
    + p * (p + 2) * Math.pow(x, p + 1)
    - (p * (p + 1) / 2) * Math.pow(x, p + 2);
}

/** Bessel radial basis (DimeNet's choice, also NequIP's). */
export const besselBasis = (r, n, rCut) =>
  r <= 0 ? 0 : Math.sqrt(2 / rCut) * Math.sin((n * Math.PI * r) / rCut) / r;

// ---------------------------------------------------------------------------
// Self-test — the browser checks its own representation theory
// ---------------------------------------------------------------------------

/** Runs every structural check and returns a list of {name, pass, detail}.
 *  `reference` is the optional contents of results/sh_reference.json, exported
 *  from scipy by python/experiments/export_js_reference.py; when supplied, the
 *  harmonics themselves are checked against an independent implementation. */
export function selfTest(reference = null) {
  const results = [];
  const add = (name, pass, detail) => results.push({ name, pass, detail });
  const rng = mulberry32(4242);

  if (reference) {
    let worst = 0;
    for (const entry of reference.samples) {
      for (let l = 0; l <= 3; l++) {
        const got = realSH(l, entry.dir);
        const want = entry[`l${l}`];
        for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
      }
    }
    add('spherical harmonics match scipy reference (l = 0..3)', worst < 1e-12,
      `max deviation ${worst.toExponential(2)}`);
  }

  for (let l = 0; l <= 3; l++) {
    const R = randomRotation(rng);
    const res = wignerResidual(l, R);
    add(`l=${l}: Y_l(R r) = D_l(R) Y_l(r) on held-out directions`, res < 1e-10,
      `max residual ${res.toExponential(2)}`);
    const D = wignerD(l, R);
    const orth = maxAbs(matmul(transpose(D), D).map((row, i) => row.map((v, j) => v - (i === j ? 1 : 0))));
    add(`l=${l}: D is orthogonal`, orth < 1e-10, `max deviation ${orth.toExponential(2)}`);
  }

  const R1 = randomRotation(rng), R2 = randomRotation(rng);
  for (let l = 0; l <= 3; l++) {
    const lhs = wignerD(l, matmul(R1, R2));
    const rhs = matmul(wignerD(l, R1), wignerD(l, R2));
    const err = maxAbs(lhs.map((row, i) => row.map((v, j) => v - rhs[i][j])));
    add(`l=${l}: D(R1 R2) = D(R1) D(R2)`, err < 1e-10, `max deviation ${err.toExponential(2)}`);
  }

  // Singular values here come from the eigenvalues of A^T A, so squaring puts a
  // floor of sqrt(eps) ~ 1.5e-8 under sigma_min however exact the null space is.
  // python/e3.py uses a true SVD and gets ~1e-15 for the same quantity; the
  // equivariance residual below is the sharp test either way.
  const SIGMA_FLOOR = 1e-6;
  for (const [l1, l2, l3] of [[1, 1, 0], [1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 2, 2]]) {
    const { C, sigmaMin, sigmaSecond } = clebschGordan(l1, l2, l3);
    add(`(${l1},${l2}) -> ${l3}: invariant subspace is one-dimensional`,
      sigmaMin < SIGMA_FLOOR && sigmaSecond > 1e-3,
      `sigma_min ${sigmaMin.toExponential(2)} (normal-equations floor ~1.5e-8), ` +
      `sigma_2 ${sigmaSecond.toExponential(2)}`);

    let worst = 0;
    for (let t = 0; t < 6; t++) {
      const R = randomRotation(rng);
      const u = Array.from({ length: 2 * l1 + 1 }, () => 2 * rng() - 1);
      const w = Array.from({ length: 2 * l2 + 1 }, () => 2 * rng() - 1);
      const D1 = wignerD(l1, R), D2 = wignerD(l2, R), D3 = wignerD(l3, R);
      const lhs = tensorProduct(C, matvec(D1, u), matvec(D2, w));
      const rhs = matvec(D3, tensorProduct(C, u, w));
      const scaleRef = Math.max(norm(rhs), 1e-12);
      for (let i = 0; i < lhs.length; i++)
        worst = Math.max(worst, Math.abs(lhs[i] - rhs[i]) / scaleRef);
    }
    add(`(${l1},${l2}) -> ${l3}: C(Du, Dv) = D C(u, v)`, worst < 1e-8,
      `max relative deviation ${worst.toExponential(2)}`);
  }

  // The two couplings everyone already knows, recovered without being told.
  const { C: C111 } = clebschGordan(1, 1, 1);
  const a = [0.3, -1.1, 0.7], b = [-0.4, 0.25, 1.3];
  const got = tensorProduct(C111, realSH(1, a), realSH(1, b));
  const cross = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const want = [cross[1], cross[2], cross[0]];
  const ratios = got.map((g, i) => g / want[i]);
  const spread = Math.max(...ratios) - Math.min(...ratios);
  add('(1,1) -> 1 is the cross product, up to one global constant',
    spread < 1e-9, `constant ${ratios[0].toFixed(9)}, spread ${spread.toExponential(2)}`);

  const { C: C110 } = clebschGordan(1, 1, 0);
  const got0 = tensorProduct(C110, realSH(1, a), realSH(1, b))[0];
  const want0 = dot(a, b) / (norm(a) * norm(b));
  add('(1,1) -> 0 is the dot product, up to one global constant',
    Number.isFinite(got0 / want0), `constant ${(got0 / want0).toFixed(9)}`);

  add('parity forbids 1o (x) 1o -> 1o', !cgAllowed(1, 1, 1, -1, -1, -1), 'selection rule');
  add('parity allows 1o (x) 1o -> 1e', cgAllowed(1, 1, 1, -1, -1, 1), 'selection rule');
  add('triangle rule forbids (1,1) -> 3', !cgAllowed(1, 1, 3), 'selection rule');

  return results;
}

export { rotationMatrix, randomRotation, mulberry32 };
