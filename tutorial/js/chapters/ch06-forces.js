import { h, slider, segmented, checkLine, loadResults, Plot, PALETTE, fmt } from '../ui.js';

// ---------------------------------------------------------------------------
// The two fitted force fields, reimplemented here so the page can integrate
// them live.  Everything below is a transcription of
// python/experiments/conservative_vs_direct.py; the coefficient vectors, the
// reference configuration and the values it is checked against all come out of
// the results JSON.
// ---------------------------------------------------------------------------

export const flatten = (rows) => Float64Array.from(rows.flat());

export function makeSystem(D) {
  const N = D.system.n_particles, K = D.basis.n_radial;
  return {
    N, K,
    MU: Float64Array.from(D.basis.mu), SIG: D.basis.sigma, RC: D.basis.r_cut,
    De: D.system.morse_D_e, aM: D.system.morse_a, re: D.system.morse_r_e,
    kTrap: D.system.trap_k,
    // one reusable scratch workspace: this force field is evaluated a few
    // hundred thousand times per interaction, and allocating per call dominates.
    ws: {
      s: new Float64Array(N * K), W: new Float64Array(N * K * 3),
      bp: new Float64Array(N * N * K), ev: new Float64Array(N * N * 3),
      a: new Float64Array(N * K), gg: new Float64Array(N * K),
      F: new Float64Array(3 * N), sE: new Float64Array(N * K),
    },
  };
}

/** Force from a coefficient vector in the shared 79-feature raw basis. */
export function forceFromRaw(S, X, c, out) {
  const N = S.N, K = S.K, RC = S.RC, SIG2 = S.SIG * S.SIG;
  const s = S.ws.s, W = S.ws.W, bp = S.ws.bp, ev = S.ws.ev;
  s.fill(0); W.fill(0); bp.fill(0); ev.fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = X[3 * i] - X[3 * j], dy = X[3 * i + 1] - X[3 * j + 1],
        dz = X[3 * i + 2] - X[3 * j + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!(r < RC)) continue;
      const ex = dx / r, ey = dy / r, ez = dz / r, p = (i * N + j) * 3;
      ev[p] = ex; ev[p + 1] = ey; ev[p + 2] = ez;
      const fc = 0.5 * (1 + Math.cos(Math.PI * r / RC));
      const dfc = -0.5 * Math.PI / RC * Math.sin(Math.PI * r / RC);
      for (let k = 0; k < K; k++) {
        const d = r - S.MU[k];
        const g = Math.exp(-0.5 * (d / S.SIG) * (d / S.SIG));
        s[i * K + k] += g * fc;
        const q = g * (-d / SIG2) * fc + g * dfc;
        bp[(i * N + j) * K + k] = q;
        const w = (i * K + k) * 3;
        W[w] += q * ex; W[w + 1] += q * ey; W[w + 2] += q * ez;
      }
    }
  }
  const cx = c[c.length - 1], offA = K, offB = K + K * K;
  const a = S.ws.a, gg = S.ws.gg;
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < K; k++) {
      let av = c[k], gv = 0;
      for (let m = 0; m < K; m++) {
        av += c[offA + k * K + m] * s[i * K + m];
        gv += c[offB + k * K + m] * s[i * K + m];
      }
      a[i * K + k] = av; gg[i * K + k] = gv;
    }
  }
  const F = out || S.ws.F;
  F.fill(0);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < K; k++) {
      const w = (i * K + k) * 3, av = a[i * K + k];
      F[3 * i] += av * W[w]; F[3 * i + 1] += av * W[w + 1]; F[3 * i + 2] += av * W[w + 2];
    }
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      let t = 0;
      for (let k = 0; k < K; k++) t += bp[(i * N + j) * K + k] * gg[j * K + k];
      const p = (i * N + j) * 3;
      F[3 * i] += t * ev[p]; F[3 * i + 1] += t * ev[p + 1]; F[3 * i + 2] += t * ev[p + 2];
    }
    F[3 * i] += cx * X[3 * i]; F[3 * i + 1] += cx * X[3 * i + 1];
    F[3 * i + 2] += cx * X[3 * i + 2];
  }
  return F;
}

/** The 28 scalar invariants whose linear combination is the learned energy. */
export function energyFeatures(S, X) {
  const N = S.N, K = S.K, RC = S.RC;
  const s = S.ws.sE;
  s.fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = X[3 * i] - X[3 * j], dy = X[3 * i + 1] - X[3 * j + 1],
        dz = X[3 * i + 2] - X[3 * j + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!(r < RC)) continue;
      const fc = 0.5 * (1 + Math.cos(Math.PI * r / RC));
      for (let k = 0; k < K; k++) {
        const d = r - S.MU[k];
        s[i * K + k] += Math.exp(-0.5 * (d / S.SIG) * (d / S.SIG)) * fc;
      }
    }
  }
  const phi = [];
  for (let k = 0; k < K; k++) {
    let t = 0;
    for (let i = 0; i < N; i++) t += s[i * K + k];
    phi.push(0.5 * t);
  }
  for (let k = 0; k < K; k++) {
    for (let m = k; m < K; m++) {
      let t = 0;
      for (let i = 0; i < N; i++) t += s[i * K + k] * s[i * K + m];
      phi.push(t);
    }
  }
  let t2 = 0;
  for (let q = 0; q < 3 * N; q++) t2 += X[q] * X[q];
  phi.push(t2);
  return phi;
}

export const modelEnergy = (S, X, theta) =>
  energyFeatures(S, X).reduce((acc, v, i) => acc + v * theta[i], 0);

/** The exact potential the data came from: Morse pairs plus an isotropic trap. */
export function trueEnergy(S, X) {
  const N = S.N;
  let V = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = X[3 * i] - X[3 * j], dy = X[3 * i + 1] - X[3 * j + 1],
        dz = X[3 * i + 2] - X[3 * j + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const e = Math.exp(-S.aM * (r - S.re));
      V += S.De * ((1 - e) * (1 - e) - 1);
    }
  }
  let t = 0;
  for (let q = 0; q < 3 * N; q++) t += X[q] * X[q];
  return 0.5 * V + 0.5 * S.kTrap * t;
}

export const kinetic = (V) => {
  let s = 0;
  for (let q = 0; q < V.length; q++) s += V[q] * V[q];
  return 0.5 * s;
};

/** Velocity-Verlet at unit mass. Records the true energy, and optionally E_theta. */
export function verlet(S, X0, V0, c, dt, nSteps, nRec, theta) {
  const X = Float64Array.from(X0), V = Float64Array.from(V0);
  let F = forceFromRaw(S, X, c);
  const stride = Math.max(1, Math.floor(nSteps / nRec));
  const t = [0], eTrue = [trueEnergy(S, X) + kinetic(V)];
  const eMod = theta ? [modelEnergy(S, X, theta) + kinetic(V)] : null;
  for (let step = 0; step < nSteps; step++) {
    for (let q = 0; q < X.length; q++) { V[q] += 0.5 * dt * F[q]; X[q] += dt * V[q]; }
    F = forceFromRaw(S, X, c);
    for (let q = 0; q < X.length; q++) V[q] += 0.5 * dt * F[q];
    if ((step + 1) % stride === 0) {
      t.push((step + 1) * dt);
      eTrue.push(trueEnergy(S, X) + kinetic(V));
      if (eMod) eMod.push(modelEnergy(S, X, theta) + kinetic(V));
    }
  }
  return { t, eTrue, eMod };
}

/** Work done going once around the circle X0 + radius*(cos t U + sin t V). */
export function loopWork(S, X0, U, V, radius, c, nq) {
  const n = X0.length, P = new Float64Array(n), T = new Float64Array(n);
  const F = new Float64Array(n);
  let net = 0, gross = 0;
  for (let q = 0; q < nq; q++) {
    const th = 2 * Math.PI * q / nq, cs = Math.cos(th), sn = Math.sin(th);
    for (let z = 0; z < n; z++) {
      P[z] = X0[z] + radius * (cs * U[z] + sn * V[z]);
      T[z] = radius * (-sn * U[z] + cs * V[z]);
    }
    forceFromRaw(S, P, c, F);
    let d = 0;
    for (let z = 0; z < n; z++) d += F[z] * T[z];
    net += d; gross += Math.abs(d);
  }
  return { net: net / nq * 2 * Math.PI, gross: gross / nq * 2 * Math.PI };
}

/** ||antisymmetric part of dF/dX|| / ||dF/dX||, by central differences. */
export function jacobianAsymmetry(S, X, c, hStep) {
  const n = X.length, J = new Float64Array(n * n);
  const Xp = Float64Array.from(X), Fp = new Float64Array(n), Fm = new Float64Array(n);
  for (let b = 0; b < n; b++) {
    Xp[b] = X[b] + hStep; forceFromRaw(S, Xp, c, Fp);
    Xp[b] = X[b] - hStep; forceFromRaw(S, Xp, c, Fm);
    Xp[b] = X[b];
    for (let a = 0; a < n; a++) J[a * n + b] = (Fp[a] - Fm[a]) / (2 * hStep);
  }
  let sa = 0, sj = 0;
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      const asym = 0.5 * (J[a * n + b] - J[b * n + a]);
      sa += asym * asym; sj += J[a * n + b] * J[a * n + b];
    }
  }
  return Math.sqrt(sa) / Math.sqrt(sj);
}

// --- small numerics ---------------------------------------------------------

const mulberry32 = (seed) => function () {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

function normalize(v) {
  let s = 0;
  for (const q of v) s += q * q;
  s = Math.sqrt(s);
  for (let i = 0; i < v.length; i++) v[i] /= s;
  return v;
}

/** A deterministic orthonormal 2-plane in the 3N-dimensional configuration space. */
function loopPlane(seed, n) {
  const rand = mulberry32(seed);
  const gauss = () => {
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(rand(), 1e-12), u2 = rand();
      v[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return v;
  };
  const U = normalize(gauss()), V = gauss();
  let d = 0;
  for (let i = 0; i < n; i++) d += U[i] * V[i];
  for (let i = 0; i < n; i++) V[i] -= d * U[i];
  return { U, V: normalize(V) };
}

function lsqSlope(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  return num / den;
}

const logSlope = (pts, xMax) => {
  const sub = pts.filter((p) => p[0] <= xMax && p[1] > 0);
  return lsqSlope(sub.map((p) => Math.log(p[0])), sub.map((p) => Math.log(p[1])));
};

const maxAbsDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

const e2 = (x, d = 2) => (Number.isFinite(x) ? x.toExponential(d) : String(x));
const pad = (s, n) => String(s).padEnd(n);

// ---------------------------------------------------------------------------

export default {
  id: 'forces',
  title: 'Forces from a gradient',
  async render(root) {
    const D = await loadResults('conservative_vs_direct');
    const M = D.models, JA = D.jacobian_asymmetry, LW = D.loop_work, MD = D.md;
    const SW = D.dt_sweep, TG = D.time_growth, B = D.browser, LB = D.leaderboard_context;
    const S = makeSystem(D);
    const theta = M.energy_coefficients_conservative;
    const coef = {
      conservative: Float64Array.from(B.coef_conservative),
      direct: Float64Array.from(B.coef_direct),
    };
    const Xref = flatten(B.reference_config), Vref = flatten(B.reference_velocity);
    const NAMES = ['conservative', 'direct'];
    const COL = { conservative: PALETTE[0], direct: PALETTE[1] };
    const nIC = SW.n_initial_conditions;

    // --- reproduce the Python reference values, here, in the page ------------
    const ver = {};
    ver.force = Math.max(...NAMES.map((n) =>
      maxAbsDiff(forceFromRaw(S, Xref, coef[n]), flatten(B['reference_force_' + n]))));
    ver.energy = Math.abs(trueEnergy(S, Xref) - B.reference_true_energy);
    const RT = B.reference_trajectory;
    ver.traj = Math.max(...NAMES.map((n) =>
      maxAbsDiff(verlet(S, Xref, Vref, coef[n], RT.dt, RT.n_steps,
        RT.n_steps / RT.record_stride).eTrue, RT['e_total_' + n])));
    {
      // is the conservative model's force really minus the gradient of E_theta?
      const Fa = forceFromRaw(S, Xref, coef.conservative), Xt = Float64Array.from(Xref);
      const hh = 1e-5;
      let worst = 0, scale = 0;
      for (let q = 0; q < Xref.length; q++) {
        Xt[q] = Xref[q] + hh; const ep = modelEnergy(S, Xt, theta);
        Xt[q] = Xref[q] - hh; const em = modelEnergy(S, Xt, theta);
        Xt[q] = Xref[q];
        worst = Math.max(worst, Math.abs(-(ep - em) / (2 * hh) - Fa[q]));
        scale = Math.max(scale, Math.abs(Fa[q]));
      }
      ver.grad = worst / scale;
    }

    const pctBetter = (1 - M.force_rmse_ratio_direct_over_conservative) * 100;
    const relForceErr = M.direct.rmse_test_clean / D.data.force_rms;

    root.append(
      h('p', { class: 'eyebrow dat' }, 'Chapter 6'),
      h('h1', {}, 'Forces from a gradient'),
      h('p', { class: 'lede' },
        `Two force fields, fitted to the same data on the same equivariant features, both ` +
        `exactly rotation-equivariant, agreeing to within ${fmt(pctBetter, 0)}% on the metric ` +
        `every leaderboard reports — and the one that scores better is the worse physics. One of ` +
        `them is the gradient of a scalar and the other is not, and no measurement made at a ` +
        `single configuration can tell which is which.`),

      h('h2', {}, 'Two ways to produce a force'),
      h('p', { class: 'prose' },
        'A machine-learned interatomic potential can emit forces two ways. Either it learns a ' +
        'scalar energy and differentiates it, or it learns the force field itself as a vector ' +
        'output and never mentions an energy at all.'),
      h('div', { class: 'prose', style: { textAlign: 'center', margin: '1.2em 0' } },
        '$$\\text{conservative:}\\;\\; \\mathbf{F}_i = -\\nabla_{\\mathbf{r}_i} E_\\theta(X) ' +
        '\\qquad\\qquad \\text{direct:}\\;\\; \\mathbf{F}_i = \\mathbf{f}^{(i)}_\\theta(X)$$'),
      h('p', { class: 'prose', html:
        'The second is cheaper — no backward pass through the network at every step of a ' +
        'simulation — and it is perfectly compatible with exact $O(3)$ equivariance, which is ' +
        'what the previous five chapters were about. Equivariance and integrability are ' +
        'independent properties of a model, and nothing in the argument for the first settles ' +
        'the second. Strong models on today’s leaderboard take each route.' }),
      h('p', { class: 'prose', html:
        `The experiment behind this chapter builds the smallest system in which the choice can ` +
        `be isolated: ${D.system.n_particles} unit-mass particles in three dimensions ` +
        `interacting through a Morse pair potential ($D_e = ${D.system.morse_D_e}$, ` +
        `$a = ${D.system.morse_a}$, $r_e = ${D.system.morse_r_e}$) inside an isotropic harmonic ` +
        `trap ($k = ${D.system.trap_k}$). The ground-truth forces are the exact analytic ` +
        `gradient. The ${D.data.n_configs} training configurations are snapshots of true-force ` +
        `trajectories, so the fitting distribution is exactly the distribution the models will ` +
        `later be integrated through, and the force labels carry Gaussian noise of ` +
        `${D.data.force_label_noise_sigma}, which is ` +
        `${fmt(100 * D.data.noise_fraction_of_force_rms, 1)}% of the force RMS, standing in for ` +
        `the convergence noise on DFT labels.` }),
      h('p', { class: 'prose', html:
        `Both surrogates are linear in the <em>same</em> set of ${D.basis.n_params_direct} ` +
        `equivariant per-atom vector features, built from a ${D.basis.n_radial}-channel Gaussian ` +
        `radial basis with a cosine cutoff at $r_c = ${D.basis.r_cut}$. The direct model gets ` +
        `all ${D.basis.n_params_direct} coefficients free. The conservative model is the ` +
        `${D.basis.n_params_conservative}-coefficient scalar energy whose exact analytic ` +
        `gradient is a <em>tied</em> combination of those same features. Its function class is a ` +
        `linear subspace of the direct model’s, so the only thing the direct model gives up is ` +
        `the constraint.` }),

      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'here, integrability is one linear condition'),
        h('div', { html:
          'The learned energy contains a term $\\sum_{k \\le m} t_{km} \\sum_i s_{ik} s_{im}$, ' +
          'where $s_{ik}$ is atom $i$’s radial environment descriptor. Differentiating it ' +
          'produces two blocks of vector features: one weighted by atom $i$’s own environment, ' +
          'one by the back-reaction of its neighbour $j$. Being a gradient forces those two ' +
          'blocks to share a coefficient and to be symmetric in $(k,m)$. That is the entire ' +
          'difference between the two models here. Nothing about rotation, nothing about ' +
          'permutation, nothing about capacity in the usual sense — one tying condition on ' +
          `${D.basis.n_params_direct} numbers.` })),

      h('h2', {}, 'Matched on the metric everyone reports'));

    // --- demo 1: matched accuracy -------------------------------------------
    const accDemo = h('div', { class: 'demo' });
    accDemo.append(
      h('h3', {}, 'Held-out force error, and the ridge sweep behind it'),
      h('p', { class: 'hint' },
        'Both models were fitted by ridge regression over the same ' +
        `${M.lambda_grid.length}-point regularisation grid, with the penalty chosen on a ` +
        'held-out validation split. The validation curves are scored against the noisy labels, ' +
        `which is why they sit near the noise floor of ${D.data.force_label_noise_sigma}; the ` +
        'table reports error against the clean forces.'),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, ''), h('th', { class: 'num' }, 'conservative'),
          h('th', { class: 'num' }, 'direct'), h('th', {}, ''))),
        h('tbody', {},
          h('tr', {},
            h('td', {}, 'free coefficients'),
            h('td', { class: 'num' }, D.basis.n_params_conservative),
            h('td', { class: 'num' }, D.basis.n_params_direct),
            h('td', {}, h('span', { class: 'pill' }, 'same feature basis'))),
          h('tr', {},
            h('td', {}, 'ridge penalty chosen on validation'),
            h('td', { class: 'num' }, e2(M.conservative.lam, 1)),
            h('td', { class: 'num' }, e2(M.direct.lam, 1)),
            h('td', {}, '')),
          h('tr', { class: 'hi' },
            h('td', {}, 'held-out force RMSE, against clean forces'),
            h('td', { class: 'num' }, fmt(M.conservative.rmse_test_clean, 5)),
            h('td', { class: 'num' }, fmt(M.direct.rmse_test_clean, 5)),
            h('td', {}, h('span', { class: 'pill bad' },
              `direct is ${fmt(pctBetter, 1)}% better`))),
          h('tr', {},
            h('td', {}, 'the same, as a fraction of the force RMS'),
            h('td', { class: 'num' },
              `${fmt(100 * M.conservative.rmse_test_clean / D.data.force_rms, 2)}%`),
            h('td', { class: 'num' }, `${fmt(100 * relForceErr, 2)}%`),
            h('td', {}, '')),
          h('tr', {},
            h('td', {}, 'worst deviation from O(3) equivariance'),
            h('td', { class: 'num' }, e2(D.equivariance.max_relative_deviation_conservative, 1)),
            h('td', { class: 'num' }, e2(D.equivariance.max_relative_deviation_direct, 1)),
            h('td', {}, h('span', { class: 'pill ok' }, 'both exact'))))));
    {
      const p = new Plot({ width: 660, height: 300, xLog: true,
        xLabel: 'ridge penalty λ', yLabel: 'validation force RMSE (noisy labels)' });
      NAMES.forEach((n) => p.add({
        points: M.lambda_grid.map((l, i) => [l, M['val_rmse_curve_' + n][i]]),
        color: COL[n], width: 2.2, label: `${n}   (best λ = ${e2(M[n].lam, 1)})` }));
      p.setLimits([M.lambda_grid[0] / 1.6, M.lambda_grid[M.lambda_grid.length - 1] * 1.6],
        [0.03, 0.15]);
      accDemo.append(p.render(), p.legend());
    }
    root.append(accDemo);

    root.append(
      h('p', { class: 'prose', html:
        `Read that table honestly. The direct model is <em>better</em> on the number that gets ` +
        `reported — ${fmt(pctBetter, 1)}% lower held-out force RMSE — and it is better across ` +
        `the whole regularisation sweep, not at one lucky penalty. This is not a flaw in the ` +
        `setup. It has ${D.basis.n_params_direct} parameters against ` +
        `${D.basis.n_params_conservative} and its function class strictly contains the other’s, ` +
        `so on a metric evaluated one configuration at a time it ought to win. The pathology is ` +
        `reproduced here deliberately: if force MAE were the whole story, you would ship the ` +
        `direct model.` }),

      h('h2', {}, 'The first thing that differs: the force-constant matrix'),
      h('p', { class: 'prose', html:
        'If $\\mathbf{F} = -\\nabla E$ then $\\partial F_a / \\partial x_b = -\\partial^2 E / ' +
        '\\partial x_a \\partial x_b$, and equality of mixed partials makes that matrix ' +
        'symmetric. This is not a stylistic preference but a theorem about any vector field ' +
        'that is a gradient. In three dimensions the same statement reads $\\nabla \\times ' +
        '\\mathbf{F} = 0$; in the $3N$ dimensions of a configuration space, the antisymmetric ' +
        'part of the Jacobian is what the curl becomes.' }),
      h('p', { class: 'prose', html:
        'That matrix has a name in this field. $\\Phi_{ab} = -\\partial F_a / \\partial x_b$ is ' +
        'the force-constant matrix, and its eigenvalues are squared phonon frequencies. A model ' +
        'whose Jacobian is not symmetric does not have a well-defined one. The panel below ' +
        'measures how far each model is from having one.' }));

    // --- demo 2: Jacobian asymmetry -----------------------------------------
    const jacDemo = h('div', { class: 'demo' });
    jacDemo.append(
      h('h3', {}, 'Antisymmetric part of the Jacobian, configuration by configuration'),
      h('p', { class: 'hint' },
        `Each point is ‖asym(J)‖ / ‖J‖ at one of the ${JA.n_configs} held-out configurations, ` +
        `by central differences with step ${e2(JA.finite_difference_h, 0)}. The lower band is ` +
        `the finite-difference floor: it is what “exactly symmetric” looks like once you ` +
        `measure it numerically.`));
    {
      const p = new Plot({ width: 660, height: 300, yLog: true,
        xLabel: 'held-out configuration', yLabel: '‖asym(J)‖ / ‖J‖' });
      NAMES.forEach((n) => p.add({
        points: JA['relative_' + n].map((v, i) => [i + 1, v]),
        color: COL[n], type: 'scatter', r: 3.2, opacity: 0.85,
        label: `${n}   (rms ${e2(JA['rms_relative_' + n], 2)})` }));
      p.setLimits([0, JA.n_configs + 1], [1e-9, 1e-1]);
      const live = {};
      NAMES.forEach((n) => {
        live[n] = jacobianAsymmetry(S, Xref, coef[n], JA.finite_difference_h);
      });
      const agree = (n) => Math.abs(live[n] - B['reference_jacobian_relative_asymmetry_' + n])
        / B['reference_jacobian_relative_asymmetry_' + n];
      jacDemo.append(p.render(), p.legend(), h('div', { class: 'readout' },
        'recomputed in your browser at the reference configuration, from the coefficient ' +
        'vectors in the results file:\n\n' +
        `  ${pad('', 14)}${pad('browser', 16)}${pad('python', 16)}agree to\n` +
        NAMES.map((n) => `  ${pad(n, 14)}${pad(e2(live[n], 6), 16)}` +
          `${pad(e2(B['reference_jacobian_relative_asymmetry_' + n], 6), 16)}` +
          `${e2(agree(n), 1)} relative`).join('\n') +
        `\n\n  the conservative row agrees only to ${e2(agree('conservative'), 0)} because both ` +
        `sides of it are\n  differencing noise; the direct row, which is a real number, ` +
        `agrees to ${e2(agree('direct'), 0)}.\n  the forces themselves reproduce python to ` +
        `${e2(ver.force, 2)} absolute, on forces of order 1.`));
    }
    root.append(jacDemo);

    root.append(
      h('p', { class: 'prose', html:
        `The conservative model sits at ${e2(JA.rms_relative_conservative, 2)}, which is not ` +
        `small — it is zero, measured with a finite-difference stencil. The direct model sits at ` +
        `${e2(JA.rms_relative_direct, 2)}, a factor ${e2(JA.ratio_direct_over_conservative, 1)} ` +
        `higher. The per-atom curl, the part of the asymmetry living inside a single atom’s own ` +
        `three coordinates, tells the same story: ${e2(JA.rms_atomic_curl_direct, 2)} against ` +
        `${e2(JA.rms_atomic_curl_conservative, 2)}.` }),
      h('p', { class: 'prose', html:
        `The number worth pausing on is not the ratio but the size of the direct model’s ` +
        `asymmetry on its own terms. At ${e2(JA.rms_relative_direct, 2)} it is the same order as ` +
        `that model’s own relative force error, ${e2(relForceErr, 2)}. Nothing in the training ` +
        `objective pushed it down, because nothing in the training objective could see it: a ` +
        `force-matching loss is evaluated one configuration at a time, and the symmetry of the ` +
        `Jacobian is a statement about how the force <em>changes</em> from one configuration to ` +
        `the next.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'what a lattice-dynamics code does with this'),
        h('div', { html:
          `Phonon codes symmetrise the force-constant matrix before diagonalising it, because ` +
          `an antisymmetric part has no physical meaning. For the direct model that silently ` +
          `discards ${fmt(100 * JA.rms_relative_direct, 2)}% of the matrix norm and then ` +
          `reports frequencies as though nothing had happened. The model has not been ` +
          `corrected; the evidence that it was inconsistent has been.` })),

      h('h2', {}, 'The second thing: work around a closed loop'),
      h('p', { class: 'prose', html:
        'The integral form of the same statement is more physical. If $\\mathbf{F} = -\\nabla E$ ' +
        'then the work done along a path depends only on its endpoints, so around any ' +
        '<em>closed</em> path the net work is exactly zero. If it is not zero you have a machine ' +
        'that manufactures energy by going in circles, and molecular dynamics is nothing but ' +
        'going in circles.' }),
      h('p', { class: 'prose', html:
        'The loops used here are circles in the full $3N$-dimensional configuration space: fix a ' +
        'configuration $X_0$ and an orthonormal pair of directions $U, V$, and traverse ' +
        '$X(t) = X_0 + \\rho\\,(\\cos t\\, U + \\sin t\\, V)$. The integrand is smooth and ' +
        '$2\\pi$-periodic, so a uniform grid is spectrally accurate and quadrature error falls ' +
        'far below anything reported. Net work is quoted against the <em>gross</em> work, ' +
        '$\\oint |\\mathbf{F} \\cdot d\\mathbf{r}|$, so the figure is a fraction of the work ' +
        'actually done rather than an absolute scale.' }));

    // --- demo 3: loop work ---------------------------------------------------
    const loopDemo = h('div', { class: 'demo' });
    const planes = [
      { label: 'reference plane', U: flatten(B.reference_loop.U), V: flatten(B.reference_loop.V) },
      ...[11, 22, 33].map((s, i) => Object.assign({ label: `plane ${i + 2}` },
        loopPlane(s, 3 * S.N))),
    ];
    const radii = [];
    for (let i = 0; i <= 24; i++) radii.push(0.015 * Math.pow(40, i / 24));
    const loopHolder = h('div'), loopOut = h('div', { class: 'readout' });
    let planeIdx = 0;
    const drawLoops = () => {
      const pl = planes[planeIdx];
      const t0 = performance.now();
      const curves = {};
      NAMES.forEach((n) => {
        curves[n] = radii.map((r) =>
          [r, Math.abs(loopWork(S, Xref, pl.U, pl.V, r, coef[n], 512).net)]);
      });
      const ms = performance.now() - t0;
      const expo = logSlope(curves.direct, 0.06);
      const p = new Plot({ width: 660, height: 330, xLog: true, yLog: true,
        xLabel: 'loop radius ρ  (configuration-space units)',
        yLabel: '| net work around the loop |' });
      NAMES.forEach((n) => p.add({
        points: curves[n].map((pt) => [pt[0], Math.max(pt[1], 1e-18)]),
        color: COL[n], width: 2.2, markers: true, label: `${n} — this plane, computed live` }));
      const anchor = curves.direct[0];
      p.add({ points: radii.map((r) => [r, anchor[1] * (r / anchor[0]) ** 2]),
        color: PALETTE[5], width: 1.4, dash: '5 4', label: 'slope 2 — Stokes: work ∝ area' });
      NAMES.forEach((n) => p.add({
        points: LW.radii.flatMap((r, i) =>
          LW['net_work_' + n][i].map((v) => [r, Math.max(Math.abs(v), 1e-18)])),
        color: COL[n], type: 'scatter', r: 2.6, opacity: 0.4,
        label: `${n} — the ${LW.radii.length * LW.n_loops_per_radius} loops in the results file` }));
      p.setLimits([0.012, 0.75], [1e-18, 1e0]);
      loopHolder.innerHTML = '';
      loopHolder.append(p.render(), p.legend());
      const chk = NAMES.map((n) => [n, loopWork(S, Xref, planes[0].U, planes[0].V,
        B.reference_loop.radius, coef[n], B.reference_loop.n_quadrature)]);
      loopOut.textContent =
        `${pl.label}:  small-loop exponent fitted live over ρ ≤ 0.06 is ${expo.toFixed(3)}, ` +
        `against the 2 that Stokes’ theorem predicts\n` +
        `${(2 * radii.length * 512).toLocaleString()} force evaluations in ${ms.toFixed(0)} ms\n\n` +
        `reference loop, ρ = ${B.reference_loop.radius} at ` +
        `${B.reference_loop.n_quadrature} quadrature points — browser against python:\n` +
        chk.map((row) => {
          const n = row[0], w = row[1];
          return `  ${pad(n, 14)}${pad('net   ' + e2(w.net, 6), 24)}python ` +
            `${e2(B.reference_loop['net_work_' + n], 6)}\n` +
            `  ${pad('', 14)}${pad('gross ' + e2(w.gross, 6), 24)}python ` +
            `${e2(B.reference_loop['gross_work_' + n], 6)}`;
        }).join('\n');
    };
    loopDemo.append(
      h('h3', {}, 'Net work around a closed loop, swept over loop size'),
      h('p', { class: 'hint' },
        `Computed live in this page: ${radii.length} radii × 2 models × 512 quadrature points, ` +
        `from the coefficient vectors in the results file. The first plane is the one the Python ` +
        `script exported, so the browser can be checked against it; the others are freshly drawn ` +
        `orthonormal 2-planes in the ${3 * S.N}-dimensional configuration space.`),
      h('div', { class: 'controls' }, segmented({
        label: 'loop plane',
        options: planes.map((p, i) => ({ label: p.label, value: i })),
        value: 0,
        onPick: (v) => {
          planeIdx = v;
          loopOut.textContent = 'integrating 25 loops…';
          setTimeout(drawLoops, 20);
        },
      })), loopHolder, loopOut);
    root.append(loopDemo);
    drawLoops();

    root.append(
      h('p', { class: 'prose', html:
        `Across the ${LW.radii.length * LW.n_loops_per_radius} loops in the results file the ` +
        `conservative model’s worst relative closing error is ` +
        `${e2(LW.max_relative_conservative, 2)}, which is double-precision roundoff on an ` +
        `integral of order one. It is not approximately path-independent; it is path-independent, ` +
        `and the residual is the arithmetic. The direct model’s median is ` +
        `${e2(LW.median_relative_direct, 2)} of the gross work and its worst is ` +
        `${e2(LW.max_relative_direct, 2)}, a factor ${e2(LW.median_relative_ratio, 1)} apart.` }),
      h('p', { class: 'prose', html:
        'The slope of the live curve says more than the ratio does. For a small loop, Stokes’ ' +
        'theorem makes the closing error the curl times the enclosed area, so it must grow as ' +
        '$\\rho^2$ — and the exponent fitted in the panel comes out at that value on every plane ' +
        'you can select. The two measurements, Jacobian asymmetry and loop work, are not ' +
        'independent evidence; they are the differential and integral versions of one fact. ' +
        'Where a curve dips sharply the net work has passed through zero and changed sign, which ' +
        'is a preview of something the dynamics will insist on: a field with curl does not pump ' +
        'energy in one direction.' }),

      h('h2', {}, 'Integrate, and the difference becomes physical'),
      h('p', { class: 'prose', html:
        'Velocity-Verlet is symplectic. Applied to a gradient field it does not conserve the ' +
        'energy exactly, but it does conserve a nearby <em>shadow</em> Hamiltonian to ' +
        '$\\mathcal{O}(\\Delta t^2)$, and the true energy therefore oscillates inside a bounded ' +
        'envelope for arbitrarily long times. That guarantee is a theorem about gradient fields. ' +
        'Applied to a field that is not one it says nothing whatsoever, because there is no ' +
        'Hamiltonian to shadow.' }),
      h('p', { class: 'prose', html:
        'The panel below runs the two learned fields from an identical initial condition and ' +
        'plots the drift of the <em>true</em> total energy: the exact Morse-plus-trap energy of ' +
        'whatever configuration the model has driven the system to, plus its kinetic energy. ' +
        'That is the quantity a practitioner cares about, and neither model was trained on it.' }));

    // --- demo 4: interactive MD ----------------------------------------------
    const mdDemo = h('div', { class: 'demo' });
    const mdHolder = h('div'), mdOut = h('div', { class: 'readout' });
    const DTS = SW.dt.slice();
    let dtIdx = Math.max(0, DTS.indexOf(MD.dt));
    let totalT = SW.total_time, icIdx = 0, pending = null;
    const runMD = () => {
      const dt = DTS[dtIdx], nSteps = Math.round(totalT / dt);
      const X0 = flatten(B.md_ic_X[icIdx]), V0 = flatten(B.md_ic_V[icIdx]);
      const t0 = performance.now();
      const runs = {};
      NAMES.forEach((n) => {
        runs[n] = verlet(S, X0, V0, coef[n], dt, nSteps, 400,
          n === 'conservative' ? theta : null);
      });
      const ms = performance.now() - t0;
      const p = new Plot({ width: 660, height: 330,
        xLabel: 'time', yLabel: 'E_true(t) − E_true(0)' });
      const stat = {};
      NAMES.forEach((n) => {
        const r = runs[n], dev = r.eTrue.map((e) => e - r.eTrue[0]);
        stat[n] = { slope: Math.abs(lsqSlope(r.t, dev)),
          maxdev: Math.max(...dev.map(Math.abs)) };
        p.add({ points: r.t.map((t, i) => [t, dev[i]]), color: COL[n], width: 2,
          label: `${n}:  |slope| ${e2(stat[n].slope, 2)},  max|ΔE| ${e2(stat[n].maxdev, 2)}` });
      });
      const eMod = runs.conservative.eMod;
      const modExc = Math.max(...eMod.map((e) => Math.abs(e - eMod[0])));
      mdHolder.innerHTML = '';
      mdHolder.append(p.render(), p.legend());
      mdOut.textContent =
        `Δt = ${dt}    T = ${totalT}    ${nSteps.toLocaleString()} steps × 2 models = ` +
        `${(2 * (nSteps + 1)).toLocaleString()} force evaluations in ${ms.toFixed(0)} ms\n\n` +
        `  ${pad('drift-rate ratio on this one trajectory', 52)}×` +
        `${(stat.direct.slope / stat.conservative.slope).toFixed(1)}\n` +
        `  ${pad(`python median over ${nIC} ICs at this Δt, T = ${SW.total_time}`, 52)}×` +
        `${SW.separation_at_each_dt[dtIdx].toFixed(1)}\n` +
        `  ${pad(`python median over ${nIC} ICs at Δt = ${MD.dt}, T = ${MD.total_time}`, 52)}×` +
        `${MD.drift_ratio_direct_over_conservative.toFixed(1)}\n` +
        `  (a single trajectory scatters widely, and the ratio grows with run length — ` +
        `both points are made below)\n\n` +
        `  over this run the conservative model's own invariant E_θ + KE moves by ` +
        `${e2(modExc, 2)},\n` +
        `  against a true-energy excursion of ${e2(stat.conservative.maxdev, 2)} — ` +
        `the integrator is not what is failing\n\n` +
        `browser against python, checked when this chapter loaded:\n` +
        `  ${pad('forces at the reference configuration', 48)}${e2(ver.force, 2)} absolute\n` +
        `  ${pad('true energy at the reference configuration', 48)}` +
        `${ver.energy === 0 ? 'exact, to the last bit' : e2(ver.energy, 2) + ' absolute'}\n` +
        `  ${pad('E_true(t) over the exported T = ' + (RT.dt * RT.n_steps) + ' trajectory', 48)}` +
        `${e2(ver.traj, 2)} absolute\n` +
        `  ${pad('−∇E_θ against the analytic conservative force', 48)}${e2(ver.grad, 2)} relative`;
    };
    const schedule = () => {
      if (pending) clearTimeout(pending);
      mdOut.textContent = `integrating ${(2 * Math.round(totalT / DTS[dtIdx])).toLocaleString()}` +
        ` steps…`;
      pending = setTimeout(() => { pending = null; runMD(); }, 60);
    };
    mdDemo.append(
      h('h3', {}, 'Velocity-Verlet, run in your browser, from the same initial condition'),
      h('p', { class: 'hint' },
        `Both trajectories start from one of the ${B.md_ic_X.length} initial conditions the ` +
        `Python script exported. Shrinking the timestep repairs an integrator error and does ` +
        `nothing at all to an error in the field itself; that is the whole diagnostic, and you ` +
        `can drive it here.`),
      h('div', { class: 'controls' },
        slider({ label: 'timestep Δt', min: 0, max: DTS.length - 1, step: 1, value: dtIdx,
          format: (v) => String(DTS[v]), onInput: (v) => { dtIdx = v; schedule(); } }),
        segmented({ label: 'run length',
          options: [25, 50, 100, 200].map((t) => ({ label: String(t), value: t })),
          value: totalT, onPick: (v) => { totalT = v; schedule(); } }),
        segmented({ label: 'initial condition',
          options: B.md_ic_X.map((_, i) => ({ label: String(i + 1), value: i })),
          value: 0, onPick: (v) => { icIdx = v; schedule(); } })),
      mdHolder, mdOut);
    root.append(mdDemo);
    runMD();

    root.append(
      h('p', { class: 'prose', html:
        `A single trajectory is an anecdote, so the script runs ${nIC} of them per model from ` +
        `identical initial conditions and reports medians. Over $T = ${MD.total_time}$ at ` +
        `$\\Delta t = ${MD.dt}$ the median absolute drift rate is ` +
        `${e2(MD.median_abs_slope_conservative, 3)} per unit time for the conservative model and ` +
        `${e2(MD.median_abs_slope_direct, 3)} for the direct one, a factor ` +
        `${MD.drift_ratio_direct_over_conservative.toFixed(0)}. In physical units the direct ` +
        `model’s total energy moves by ${fmt(100 * MD.drift_frac_of_E0_total_direct, 3)}% of ` +
        `$|E_0|$ over the run — ${fmt(100 * MD.drift_frac_of_KE0_total_direct, 2)}% of the ` +
        `kinetic energy it started with — while the conservative model moves by ` +
        `${e2(100 * MD.drift_frac_of_E0_total_conservative, 1)}%.` }),
      h('p', { class: 'prose', html:
        `A drift rate on its own could be a fitting artefact, so the script separates secular ` +
        `motion from bounded oscillation: the straight-line change over the run divided by the ` +
        `RMS scatter about that line. The conservative model scores ` +
        `${MD.median_secular_ratio_conservative.toFixed(3)}, meaning the change is small ` +
        `compared with the wobble, which is what “oscillates but does not drift” amounts to ` +
        `quantitatively. The direct model scores ${MD.median_secular_ratio_direct.toFixed(2)}. ` +
        `And the comparison survives pairing: on ${nIC} of ${nIC} shared initial conditions the ` +
        `direct model’s secular energy change is the larger one, by a median factor ` +
        `${MD.paired_secular_ratio_median.toFixed(0)} and, in the single best case for it, still ` +
        `${MD.paired_secular_ratio_min.toFixed(0)}.` }));

    {
      const fig = h('figure', {});
      const p = new Plot({ width: 700, height: 320,
        xLabel: 'time', yLabel: 'median E_true(t) − E_true(0)' });
      NAMES.forEach((n) => p.add({
        points: MD.t.map((t, i) => [t, MD['dE_true_' + n + '_median'][i]]),
        color: COL[n], width: 2, label: n }));
      p.setLimits([0, MD.total_time], [-0.006, 0.021]);
      const q90 = Math.max(...MD.dE_true_direct_q90);
      fig.append(p.render(), p.legend(), h('figcaption', {},
        `Medians from the results file rather than from your browser: ${nIC} trajectories per ` +
        `model, ${MD.n_steps.toLocaleString()} steps each. The lower trace is the flat one. The ` +
        `median also hides the tail — the 90th percentile of the direct model reaches ` +
        `${fmt(q90, 2)}, which is ${fmt(q90 / MD.KE0, 1)} times the kinetic energy the system ` +
        `started with.`));
      root.append(fig);
    }

    root.append(
      h('h2', {}, 'Two scalings that identify the mechanism'),
      h('p', { class: 'prose', html:
        'Drift alone does not prove non-conservativity, because a bad integrator drifts too. The ' +
        'experiments that settle it are scalings: what happens as the timestep shrinks, and what ' +
        'happens as the run gets longer. An integrator error vanishes as $\\Delta t \\to 0$; an ' +
        'error in the vector field does not. A bounded error stops growing; an accumulating one ' +
        'does not.' }));

    const scaleDemo = h('div', { class: 'demo' });
    scaleDemo.append(h('h3', {}, 'Timestep sweep'),
      h('p', { class: 'hint' },
        `Each point is ${nIC} trajectories per model run to T = ${SW.total_time}, at four ` +
        `timesteps spanning a factor ${SW.dt_ratio_coarse_over_fine}.`));
    {
      const grid = h('div', { class: 'grid2' });
      const pA = new Plot({ width: 470, height: 300, xLog: true, yLog: true,
        pad: { l: 66, r: 14, t: 14, b: 46 },
        xLabel: 'timestep Δt', yLabel: 'median |dE_true/dt|' });
      NAMES.forEach((n) => pA.add({
        points: SW.dt.map((d, i) => [d, SW['median_abs_slope_' + n][i]]),
        color: COL[n], width: 2.2, markers: true, label: n }));
      pA.setLimits([9e-4, 1.4e-2], [1e-6, 1e-3]);
      const pB = new Plot({ width: 470, height: 300, xLog: true, yLog: true,
        pad: { l: 66, r: 14, t: 14, b: 46 },
        xLabel: 'timestep Δt', yLabel: 'relative excursion of E_θ + KE' });
      pB.add({ points: SW.dt.map((d, i) => [d, SW.model_energy_rel_excursion_conservative[i]]),
        color: COL.conservative, width: 2.2, markers: true,
        label: 'the conservative model’s own invariant' });
      const a0 = SW.dt[0], b0 = SW.model_energy_rel_excursion_conservative[0];
      pB.add({ points: SW.dt.map((d) => [d, b0 * (d / a0) ** 2]),
        color: PALETTE[5], width: 1.4, dash: '5 4', label: 'slope 2 — pure Verlet error' });
      pB.setLimits([9e-4, 1.4e-2], [5e-9, 1e-6]);
      const cA = h('div'), cB = h('div');
      cA.append(pA.render(), pA.legend());
      cB.append(pB.render(), pB.legend());
      grid.append(cA, cB);
      scaleDemo.append(grid, h('div', { class: 'readout' },
        `left:   neither model's true-energy drift rate improves as Δt shrinks. the direct ` +
        `model's varies by only\n` +
        `        ×${SW.direct_slope_spread.toFixed(2)} across the whole ` +
        `×${SW.dt_ratio_coarse_over_fine} range in Δt, and the conservative model's max|ΔE| by ` +
        `only ×${SW.conservative_true_maxdev_spread.toFixed(2)}.\n` +
        `        both are model error, not integrator error, and the separation holds at every ` +
        `timestep tested:\n` +
        `        ${SW.separation_at_each_dt.map((v) => '×' + v.toFixed(0)).join(',  ')}\n\n` +
        `right:  the one quantity that does follow the integrator. the conservative model's own ` +
        `invariant E_θ + KE\n` +
        `        degrades by ×${SW.model_energy_excursion_growth_coarse_over_fine.toFixed(1)} ` +
        `across the sweep, where Δt² predicts ` +
        `×${(SW.dt_ratio_coarse_over_fine ** 2).toFixed(0)}. that is textbook velocity-Verlet, ` +
        `and it\n` +
        `        vanishes as Δt → 0. the direct model has no such invariant to plot.`));
    }
    root.append(scaleDemo);

    root.append(
      h('p', { class: 'prose', html:
        `The second scaling is length of run. Fitting a straight line to a <em>bounded</em> ` +
        `error over a window of length $T$ gives an apparent rate that falls like $1/T$, while a ` +
        `genuinely accumulating error gives a rate that does not depend on $T$ at all. Going ` +
        `from $T = ${TG.short_time}$ to $T = ${TG.long_time}$, a factor ` +
        `${TG.time_ratio.toFixed(0)}, the conservative model’s median max$|\\Delta E|$ grows by ` +
        `×${TG.maxdev_growth_factor_conservative.toFixed(2)} where linear accumulation would ` +
        `give ×${TG.time_ratio.toFixed(0)}, and its apparent rate falls by ` +
        `×${TG.rate_ratio_long_over_short_conservative.toFixed(3)} against the ` +
        `×${(1 / TG.time_ratio).toFixed(2)} that $1/T$ predicts. The direct model’s rate changes ` +
        `by ×${TG.rate_ratio_long_over_short_direct.toFixed(3)}. It is the same rate, and it ` +
        `will still be that rate at $T = 10^6$.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' },
          `the factor ${MD.drift_ratio_direct_over_conservative.toFixed(0)} is not a constant ` +
          `of nature`),
        h('div', { html:
          `Because the conservative model’s apparent rate is a bounded error divided by $T$, ` +
          `the headline drift ratio grows roughly linearly with how long you run. Quoted at ` +
          `$T = ${MD.total_time}$ it is ${MD.drift_ratio_direct_over_conservative.toFixed(0)}; ` +
          `at $T = ${SW.total_time}$ the same comparison at the same timestep gives ` +
          `${SW.separation_at_each_dt[SW.dt.indexOf(MD.dt)].toFixed(0)}. Any number of this ` +
          `kind is a statement about a run length, and the sweep above exists so that the ` +
          `dependence is visible instead of hidden.` })),

      h('h2', {}, 'What this experiment does not show'),
      h('p', { class: 'prose', html:
        `The tidy story would be that a non-conservative field heats the system, like friction ` +
        `with the wrong sign. That is not what happens, and the script records the failure ` +
        `rather than tuning it away. Of ${nIC} trajectories, ` +
        `${MD.n_trajectories_gaining_energy_direct} gained energy and ` +
        `${nIC - MD.n_trajectories_gaining_energy_direct} lost it, a binomial ` +
        `$p = ${MD.sign_bias_binomial_p_direct.toFixed(3)}$ against a fair coin. A field with ` +
        `curl pumps energy in or out depending on which way the trajectory happens to circulate ` +
        `through the region where the curl lives. What is systematic is the ` +
        `<em>magnitude</em>, which is why the claim that survives checking is the paired one.` }),
      h('p', { class: 'prose', html:
        `Three further caveats, all recorded in the results file. ` +
        `${fmt(100 * MD.frac_runaway_direct, 0)}% of direct-model trajectories are destroyed ` +
        `outright, with $|\\Delta E|$ exceeding a full Morse well depth, against ` +
        `${fmt(100 * MD.frac_runaway_conservative, 0)}% of conservative ones; because that ` +
        `minority would dominate any mean, every MD aggregate here is a median. The ` +
        `conservative model’s true energy is not constant either — it wanders by about ` +
        `${fmt(MD.median_maxdev_conservative, 4)}, because $E_\\theta$ is a fit and not the true ` +
        `potential — and the claim is only that the wandering stays bounded while the other does ` +
        `not. And the headline that the conservative model conserves its own invariant to ` +
        `${e2(MD.model_energy_rel_excursion_conservative, 2)} is a <em>relative</em> figure ` +
        `against a learned energy carrying a large arbitrary additive constant, since only its ` +
        `gradient was ever fitted; the live panel above reports the same excursion in absolute ` +
        `units, which is the honest version of it.` }),
      h('p', { class: 'prose' },
        'Finally the scope. This is a linear model on a fixed feature basis, six particles, one ' +
        'potential. It isolates the mechanism cleanly precisely because it is small enough that ' +
        '“conservative” and “direct” differ by exactly one linear constraint and nothing else. ' +
        'It does not measure how large the effect is for a deep network on a real dataset, and ' +
        'it cannot. That is an empirical question about particular models, and the next section ' +
        'is about how badly it is possible to get the answer wrong.'),

      h('h2', {}, 'Does this explain the leaderboard?'),
      h('p', { class: 'prose', html:
        `There is a widely repeated version of this chapter’s argument that runs: Matbench ` +
        `Discovery’s thermal-conductivity metric $\\kappa_{\\mathrm{SRME}}$ punishes ` +
        `direct-force models catastrophically, therefore non-conservative forces are the cause. ` +
        `The first half is well evidenced. The second half is false, and the numbers that refute ` +
        `it sit in the same table.` }),
      h('p', { class: 'prose', html:
        `The association is real, and it is strong. ${LB.association} On this metric 0 is ` +
        `perfect and ${LB.worst_possible_kappa_srme.toFixed(1)} is the worst attainable value, ` +
        `so a score of 1.77 is not “poor” in the way a large MAE is poor; it is closer to having ` +
        `no useful signal at all.` }));

    const lbDemo = h('div', { class: 'demo' });
    const rows = LB.entries;
    const grad = rows.filter((r) => r.forces === 'gradient');
    const dirRows = rows.filter((r) => r.forces === 'direct');
    const bestDirect = dirRows.reduce((m, r) => (r.kappa_srme < m.kappa_srme ? r : m), dirRows[0]);
    const worstGrad = grad.reduce((m, r) => (r.kappa_srme > m.kappa_srme ? r : m), grad[0]);
    const eqv2 = rows.reduce((m, r) => (r.kappa_srme > m.kappa_srme ? r : m), rows[0]);
    lbDemo.append(
      h('h3', {}, 'Five published entries that decide the question'),
      h('p', { class: 'hint' },
        'These are the only numbers in this chapter that were transcribed rather than measured. ' +
        'They are kept in the results file with their provenance attached so that the sourcing ' +
        'travels with them.'),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, 'model'), h('th', {}, 'forces'),
          h('th', { class: 'num' }, 'F1 (hull)'),
          h('th', { class: 'num' }, 'κ_SRME'), h('th', {}, 'source'))),
        h('tbody', {}, ...rows.map((r) => h('tr',
          { class: (r === bestDirect || r === worstGrad) ? 'hi' : null },
          h('td', {}, r.model),
          h('td', {}, h('span', { class: `pill ${r.forces === 'gradient' ? 'ok' : 'bad'}` },
            r.forces === 'gradient' ? 'F = −∇E' : 'direct')),
          h('td', { class: 'num' }, r.f1 == null ? '—' : r.f1.toFixed(3)),
          h('td', { class: 'num' }, r.kappa_srme.toFixed(4)),
          h('td', {}, r.source))))),
      h('div', { class: 'readout' },
        `${pad('worst conservative entry here', 36)}${pad(worstGrad.model, 24)}κ_SRME ` +
        `${worstGrad.kappa_srme.toFixed(4)}\n` +
        `${pad('the direct model it is set against', 36)}${pad(eqv2.model, 24)}κ_SRME ` +
        `${eqv2.kappa_srme.toFixed(4)}\n` +
        `${pad('', 36)}a difference of ` +
        `${(100 * (eqv2.kappa_srme - worstGrad.kappa_srme) / worstGrad.kappa_srme).toFixed(1)}%, ` +
        `with both inside ` +
        `${(100 * (1 - Math.min(eqv2.kappa_srme, worstGrad.kappa_srme)
          / LB.worst_possible_kappa_srme)).toFixed(0)}% of the worst possible ` +
        `${LB.worst_possible_kappa_srme.toFixed(1)}\n\n` +
        `${pad('best direct entry here', 36)}${pad(bestDirect.model, 24)}κ_SRME ` +
        `${bestDirect.kappa_srme.toFixed(3)}\n` +
        `${pad('', 36)}lower than ` +
        `${grad.filter((r) => r.kappa_srme > bestDirect.kappa_srme).length} of ${grad.length} ` +
        `conservative entries in this table, at F1 ${bestDirect.f1.toFixed(3)}`));
    root.append(lbDemo);

    root.append(
      h('p', { class: 'prose', html:
        `Take those two rows in turn. ${worstGrad.model} predicts forces as an energy gradient — ` +
        `it is conservative by construction — and scores ${worstGrad.kappa_srme.toFixed(4)}, ` +
        `essentially the same as the direct-force ${eqv2.model} at ` +
        `${eqv2.kappa_srme.toFixed(4)}, whose unique-prototype F1 of ${eqv2.f1.toFixed(3)} is ` +
        `excellent. So non-conservativeness is not <em>necessary</em> for a catastrophic phonon ` +
        `score. And ${bestDirect.model} predicts forces directly, with no energy gradient ` +
        `anywhere, and scores ${bestDirect.kappa_srme.toFixed(3)}. So it is not ` +
        `<em>sufficient</em> either. The remaining conservative models in the table, at ` +
        `${grad.filter((r) => r !== worstGrad).map((r) => r.kappa_srme.toFixed(4)).join(' and ')}, ` +
        `sit in between and make the same point more mildly.` }),
      h('p', { class: 'prose', html:
        `What survives is a mechanism rather than a law. ${LB.mechanism} That is exactly the ` +
        `quantity this chapter measured, and the link is worth stating precisely: the Jacobian ` +
        `$\\partial F_a / \\partial x_b$ whose asymmetry was plotted above <em>is</em> the ` +
        `force-constant matrix, up to a sign. A direct-force model is never penalised for ` +
        `getting it wrong during training and is not even obliged to produce a symmetric one. ` +
        `A conservative model produces a symmetric one automatically — and that matrix can still ` +
        `be numerically wrong, in which case its phonons are wrong too. Being a gradient buys ` +
        `consistency, not accuracy.` }),
      h('div', { class: 'note dat' },
        h('span', { class: 'tag' }, 'what the toy licenses, and what it does not'),
        h('div', { html:
          `It licenses this much: a force field that is not a gradient has nonzero curl, does ` +
          `net work around closed loops, and accumulates energy error without bound under a ` +
          `symplectic integrator, and none of that is visible in a force MAE. Every step of ` +
          `that chain is measured above and none of it depends on the model being small. It ` +
          `does <em>not</em> license reading a model’s $\\kappa_{\\mathrm{SRME}}$ off its ` +
          `training target. On the current leaderboard that inference fails in both directions. ` +
          `The defensible summary is that conservative forces are one way of keeping an energy ` +
          `surface’s derivatives consistent with each other, not a guarantee that any of them ` +
          `are right.` })));

    // --- check ledger --------------------------------------------------------
    const ledger = h('div', { class: 'demo' });
    ledger.append(h('h3', {}, 'Check ledger — python/experiments/conservative_vs_direct.py'),
      h('p', { class: 'hint' },
        'Reproduce with:  uv run python python/experiments/conservative_vs_direct.py'));
    const inner = h('div');
    for (const c of D.checks) {
      inner.append(checkLine(c.pass !== undefined ? c.pass : c.passed, c.name, c.detail));
    }
    inner.append(h('div', { class: 'checkline', style: { marginTop: '10px', fontWeight: '700' } },
      h('span', { class: `mark ${D.meta.n_pass === D.meta.n_total ? 'ok' : 'bad'}` },
        `${D.meta.n_pass}/${D.meta.n_total}`),
      h('span', {}, 'PASS')));
    ledger.append(inner);
    root.append(ledger);
  },
};
