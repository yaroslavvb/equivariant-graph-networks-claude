// Minimal dense linear algebra. No dependencies — everything the tutorial needs
// to derive representation theory in the browser rather than load it from a table.
//
// Matrices are plain arrays of row arrays. Vectors are plain arrays.

export const zeros = (n, m) => Array.from({ length: n }, () => new Array(m).fill(0));
export const eye = (n) => Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

export function matmul(A, B) {
  const n = A.length, k = B.length, m = B[0].length;
  const C = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const Ai = A[i], Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < m; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}

export function transpose(A) {
  const n = A.length, m = A[0].length;
  const T = zeros(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

export function matvec(A, v) {
  return A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
}

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = (a) => Math.sqrt(dot(a, a));
export const scale = (a, c) => a.map((x) => x * c);
export const sub = (a, b) => a.map((x, i) => x - b[i]);

export function maxAbs(A) {
  let m = 0;
  for (const row of A) for (const x of row) m = Math.max(m, Math.abs(x));
  return m;
}

export function kron(A, B) {
  const an = A.length, am = A[0].length, bn = B.length, bm = B[0].length;
  const C = zeros(an * bn, am * bm);
  for (let i = 0; i < an; i++)
    for (let j = 0; j < am; j++) {
      const a = A[i][j];
      if (a === 0) continue;
      for (let p = 0; p < bn; p++)
        for (let q = 0; q < bm; q++) C[i * bn + p][j * bm + q] = a * B[p][q];
    }
  return C;
}

/** Solve the least-squares problem min ||A X - B|| via the normal equations
 *  with a small Tikhonov floor for conditioning. A is (n x k), B is (n x m). */
export function lstsq(A, B, ridge = 0) {
  const At = transpose(A);
  const AtA = matmul(At, A);
  const AtB = matmul(At, B);
  if (ridge > 0) for (let i = 0; i < AtA.length; i++) AtA[i][i] += ridge;
  return solve(AtA, AtB);
}

/** Solve A X = B for square symmetric-positive-definite-ish A by Gaussian
 *  elimination with partial pivoting. */
export function solve(A, B) {
  const n = A.length, m = B[0].length;
  const M = A.map((row, i) => [...row, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    if (Math.abs(d) < 1e-300) continue;
    for (let j = c; j < n + m; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f === 0) continue;
      for (let j = c; j < n + m; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row.slice(n));
}

/** Symmetric eigendecomposition by the cyclic Jacobi method.
 *  Returns { values, vectors } with vectors[k] the k-th eigenvector (row form),
 *  sorted by ascending eigenvalue. Accurate to ~1e-15 for the sizes used here. */
export function jacobiEigen(Ain, sweeps = 60, tol = 1e-14) {
  const n = Ain.length;
  const A = Ain.map((r) => [...r]);
  let V = eye(n);
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (Math.sqrt(off) < tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - sn * akq;
          A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - sn * aqk;
          A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - sn * vkq;
          V[k][q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[a][a] - A[b][b]);
  return {
    values: idx.map((i) => A[i][i]),
    vectors: idx.map((i) => V.map((row) => row[i])),
  };
}

/** Singular values and right-singular vectors of A, via the eigendecomposition
 *  of A^T A. Adequate here because we only ever want the smallest right-singular
 *  vector of a well-scaled constraint matrix. */
export function rightSingular(A) {
  const AtA = matmul(transpose(A), A);
  const { values, vectors } = jacobiEigen(AtA);
  return {
    singularValues: values.map((v) => Math.sqrt(Math.max(v, 0))),
    vectors,
  };
}

/** Ridge regression: returns w minimising ||X w - y||^2 + lambda ||w||^2. */
export function ridgeFit(X, y, lambda) {
  const Xt = transpose(X);
  const A = matmul(Xt, X);
  for (let i = 0; i < A.length; i++) A[i][i] += lambda;
  const b = matvec(Xt, y).map((v) => [v]);
  return solve(A, b).map((r) => r[0]);
}

// ---------------------------------------------------------------------------
// 3-D geometry
// ---------------------------------------------------------------------------

export function rotationMatrix(axis, angle) {
  const n = norm(axis);
  const [ux, uy, uz] = axis.map((v) => v / n);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
    [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
    [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
  ];
}

/** A deterministic pseudo-random generator so every figure is reproducible. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Haar-uniform rotation, via a normalised random quaternion. */
export function randomRotation(rng) {
  let q = [randn(rng), randn(rng), randn(rng), randn(rng)];
  const n = norm(q);
  q = q.map((x) => x / n);
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
