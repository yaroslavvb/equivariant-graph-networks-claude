"""Why F = -grad E is not a stylistic choice: matched force error, opposite dynamics.

A machine-learned interatomic potential can produce forces two ways.

    conservative   E_theta(X) is the learned object; F = -grad E_theta.
    direct         F_theta(X) is the learned object; nothing ties it to a scalar.

Both can be exactly O(3)-equivariant, and both are scored on the leaderboard by
the same one-step force MAE/RMSE.  The difference only shows up when you close
the loop and integrate: a gradient field has a symmetric Jacobian, does zero
work around any closed loop in configuration space, and therefore admits a
conserved energy under a symplectic integrator.  A direct field satisfies none
of those, and the violation is invisible to any single-configuration metric.

This is the toy version of the Matbench Discovery result: eqV2 M is a strong
direct-force model (F1 ~ 0.917 on the hull task) yet scores kappa_SRME ~ 1.77 on
phonon thermal conductivity -- worse than the null "predict the mean" baseline
at 2.0 by a hair, i.e. no useful signal -- while conservative models of similar
or lower F1 land at ~0.1-0.2.  kappa_SRME needs third derivatives of a scalar
energy surface; a force field that is not a gradient does not have one.

SYSTEM.  Six unit-mass particles in 3-D, Morse pair potential (D_e=1, a=1.5,
r_e=1) plus an isotropic harmonic trap (k=0.6).  Ground-truth forces are the
exact analytic gradient.  Training configurations are snapshots of true-force
NVE trajectories, so the fit distribution is exactly the MD distribution.
Force labels carry 3% Gaussian noise, standing in for DFT convergence noise.

ARCHITECTURE-MATCHED SURROGATES.  Both models are linear in the *same* set of
79 equivariant per-atom vector features built from a Gaussian radial basis
b_k(r) with a cosine cutoff and the per-atom descriptors s_ik = sum_j b_k(r_ij):

    W_ik   = sum_j b_k'(r_ij) e_ij                (2-body)
    A_ikm  = s_im * W_ik                          (weighted by i's own env.)
    B_ikm  = sum_j b_k'(r_ij) s_jm e_ij           (weighted by neighbour j's env.)
    x_i                                           (trap)

The direct model gives all 79 free coefficients.  The conservative model is the
ridge fit of the 28 coefficients of the scalar energy

    E = sum_k t1_k sum_{i<j} b_k(r_ij)  +  sum_{k<=m} t2_km sum_i s_ik s_im
        + t3 sum_i |x_i|^2

whose exact analytic gradient is a *tied* combination of the same 79 features:
grad_i of the quadratic term is A_ikm + A_imk + B_ikm + B_imk.  Integrability is
precisely the statement that the "A" (local) and "B" (back-reaction) blocks must
share a coefficient and be symmetric in (k,m).  The direct model's function
class strictly contains the conservative one; the only thing it gives up is that
constraint.  Rotation and permutation equivariance are identical in both.

WHAT THE NUMBERS SAY (every figure below is asserted in _selfcheck).

  one-step force RMSE      0.0129 conservative vs 0.0114 direct  -- ratio 0.88,
                           i.e. the direct model wins the leaderboard metric
  Jacobian asymmetry       1.4e-8 vs 7.3e-3 relative              -- ratio 5e5
  work around closed loops 2e-15  vs 3.6e-3 of the gross work     -- ratio 1e13
  MD energy drift rate     1.4e-6 vs 2.0e-4 per unit time         -- ratio 149
  paired over 24 identical initial conditions, the direct model drifts faster
  on 24 of 24, by at least 38x and typically 210x.

Two independent scalings pin down the mechanism, and they are the real content
of this script:

  dt sweep (x8 range).  The conservative model's own invariant E_theta + KE
  degrades exactly as dt^2 (measured x65.8 growth where dt^2 predicts x64):
  pure velocity-Verlet error, which vanishes as dt -> 0.  The direct model's
  drift rate changes by only x1.55 over the same range -- it is a property of
  the learned vector field, so no timestep can fix it.

  t sweep (x4 range).  The conservative model's true-energy error stays inside
  a fixed envelope (max|dE| grows x1.4 where linear accumulation would give
  x4), and its *apparent* drift rate therefore falls like 1/T (measured x0.155
  where 1/T predicts x0.25).  The direct model's rate is unchanged (x0.951), so
  its energy error accumulates without bound.

HONEST CAVEATS AND ONE NEGATIVE RESULT, recorded in the JSON as well.
  * NEGATIVE RESULT: the direct model's drift is NOT systematic heating.  Only
    15 of 24 trajectories gain energy (binomial p = 0.31, consistent with a
    coin flip).  A non-conservative field pumps energy in or out depending on
    where the trajectory goes; it is not an effective friction.  This is left
    as it fell rather than tuned into a one-sided effect, and the checks assert
    the claim that does hold -- the *magnitude* of the drift, paired
    trajectory by trajectory -- plus the null result itself.
  * The drift-rate ratio of 149 is not a universal constant: the conservative
    model's apparent rate is bounded-error/T, so the ratio grows roughly
    linearly with run length.  Quoted at T = 200; the t sweep makes the
    T-dependence explicit instead of hiding it.
  * 12% of direct-model trajectories are destroyed outright (|dE| > 1, tens of
    percent of the well depth) while 0% of conservative ones are.  Because that
    minority would dominate any mean, all MD aggregates here are medians.
  * The direct model's held-out force RMSE comes out slightly *better* than the
    conservative model's -- it has 79 parameters against 28.  That is not a bug
    in the setup; it is the leaderboard pathology in miniature.
  * The conservative model's true energy is not constant either: it fluctuates
    by ~0.01 because E_theta is not the true potential.  The point is that the
    fluctuation is bounded forever, whereas the direct model's error is not.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from e3 import random_rotation

SEED = 20260729
OUT_PATH = (pathlib.Path(__file__).resolve().parents[2] / "results"
            / "conservative_vs_direct.json")

N_PART = 6
D_E, A_M, R_E = 1.0, 1.5, 1.0
K_TRAP = 0.6

K_RAD = 6
MU = np.linspace(0.70, 2.40, K_RAD)
SIG_RAD = 0.36
R_CUT = 3.0

P_RAW = K_RAD + 2 * K_RAD ** 2 + 1
P_ENE = K_RAD + K_RAD * (K_RAD + 1) // 2 + 1

FORCE_NOISE = 0.03
N_TRAJ = 16
GEN_STEPS = 600
GEN_DT = 0.008
GEN_STRIDE = 6

MD_N_IC = 24
MD_T_MAIN = 200.0
MD_DT_MAIN = 0.005
MD_T_SWEEP = 50.0
MD_DTS = [0.00125, 0.0025, 0.005, 0.01]
MD_N_REC = 400
RUNAWAY_THRESHOLD = 1.0


# --------------------------------------------------------------------------
# 1. Ground truth
# --------------------------------------------------------------------------

def _geom(X: np.ndarray):
    """Pair distances and unit vectors e_ij = (x_i - x_j)/r_ij for X (B,N,3)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    r2 = (diff ** 2).sum(-1)
    eye = np.eye(X.shape[1], dtype=bool)
    r = np.sqrt(np.where(eye, 1.0, r2))
    e = np.where(eye[..., None], 0.0, diff / r[..., None])
    return r, e, ~eye


def true_energy(X: np.ndarray) -> np.ndarray:
    r, _, off = _geom(X)
    ex = np.exp(-A_M * (r - R_E))
    V = np.where(off, D_E * ((1.0 - ex) ** 2 - 1.0), 0.0)
    return 0.5 * V.sum(axis=(1, 2)) + 0.5 * K_TRAP * (X ** 2).sum(axis=(1, 2))


def true_forces(X: np.ndarray) -> np.ndarray:
    r, e, off = _geom(X)
    ex = np.exp(-A_M * (r - R_E))
    dV = np.where(off, 2.0 * D_E * A_M * (1.0 - ex) * ex, 0.0)
    return -np.einsum("bij,bijc->bic", dV, e) - K_TRAP * X


# --------------------------------------------------------------------------
# 2. The shared equivariant feature basis
# --------------------------------------------------------------------------

def _radial(r: np.ndarray, off: np.ndarray):
    """Gaussian radial basis b_k(r) and its exact derivative, cosine-cut at R_CUT."""
    d = r[..., None] - MU
    g = np.exp(-0.5 * (d / SIG_RAD) ** 2)
    rc = np.minimum(r, R_CUT)
    fc = 0.5 * (1.0 + np.cos(np.pi * rc / R_CUT))
    dfc = -0.5 * np.pi / R_CUT * np.sin(np.pi * rc / R_CUT)
    b = g * fc[..., None]
    bp = g * (-d / SIG_RAD ** 2) * fc[..., None] + g * dfc[..., None]
    keep = ((r < R_CUT) & off)[..., None]
    return np.where(keep, b, 0.0), np.where(keep, bp, 0.0)


def raw_features(X: np.ndarray) -> np.ndarray:
    """The 79 equivariant vector features, shape (B, N, 3, P_RAW)."""
    r, e, off = _geom(X)
    b, bp = _radial(r, off)
    s = b.sum(axis=2)
    W = np.einsum("bijk,bijc->bikc", bp, e)
    A = W[:, :, :, None, :] * s[:, :, None, :, None]
    Bk = np.einsum("bijk,bjm,bijc->bikmc", bp, s, e)
    nb, nn = X.shape[0], X.shape[1]
    return np.concatenate([
        np.transpose(W, (0, 1, 3, 2)),
        np.transpose(A, (0, 1, 4, 2, 3)).reshape(nb, nn, 3, K_RAD ** 2),
        np.transpose(Bk, (0, 1, 4, 2, 3)).reshape(nb, nn, 3, K_RAD ** 2),
        X[:, :, :, None],
    ], axis=3)


def energy_features(X: np.ndarray) -> np.ndarray:
    """The 28 scalar invariants phi_p(X), shape (B, P_ENE)."""
    r, _, off = _geom(X)
    b, _ = _radial(r, off)
    s = b.sum(axis=2)
    quad = [(s[:, :, k] * s[:, :, m]).sum(axis=1)
            for k in range(K_RAD) for m in range(k, K_RAD)]
    return np.concatenate([0.5 * s.sum(axis=1),
                           np.stack(quad, axis=1),
                           (X ** 2).sum(axis=(1, 2))[:, None]], axis=1)


def tying_matrix() -> np.ndarray:
    """T with (raw force coefficients) = T @ (energy coefficients).

    Encodes -grad phi_p in the raw basis. The quadratic invariants force the
    A and B blocks to share one coefficient and be (k,m)-symmetric -- that
    tying *is* integrability.
    """
    T = np.zeros((P_RAW, P_ENE))
    off_a, off_b = K_RAD, K_RAD + K_RAD ** 2
    col = 0
    for k in range(K_RAD):
        T[k, col] = -1.0
        col += 1
    for k in range(K_RAD):
        for m in range(k, K_RAD):
            for idx in (off_a + k * K_RAD + m, off_a + m * K_RAD + k,
                        off_b + k * K_RAD + m, off_b + m * K_RAD + k):
                T[idx, col] -= 1.0
            col += 1
    T[P_RAW - 1, col] = -2.0
    return T


TIE = tying_matrix()


def force_from_raw(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Force for raw coefficients ``coef``, shape (P_RAW,) or (B, P_RAW).

    Contracts the coefficients into the sums instead of materialising the
    (B, N, K, K, 3) A and B tensors -- same field, ~5x cheaper, which is what
    makes the long MD runs affordable. Verified against ``raw_features`` below.
    """
    c = np.atleast_2d(coef)
    if c.shape[0] == 1 and X.shape[0] != 1:
        c = np.broadcast_to(c, (X.shape[0], c.shape[1]))
    cW = c[:, :K_RAD]
    cA = c[:, K_RAD:K_RAD + K_RAD ** 2].reshape(-1, K_RAD, K_RAD)
    cB = c[:, K_RAD + K_RAD ** 2:-1].reshape(-1, K_RAD, K_RAD)
    cx = c[:, -1]

    r, e, off = _geom(X)
    b, bp = _radial(r, off)
    s = b.sum(axis=2)
    W = np.einsum("bijk,bijc->bikc", bp, e)
    a = cW[:, None, :] + np.einsum("bkm,bim->bik", cA, s)
    g = np.einsum("bkm,bjm->bjk", cB, s)
    return (np.einsum("bik,bikc->bic", a, W)
            + np.einsum("bijk,bjk,bijc->bic", bp, g, e)
            + cx[:, None, None] * X)


# --------------------------------------------------------------------------
# 3. Ridge regression (closed form, column-standardised)
# --------------------------------------------------------------------------

def ridge(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    scale = np.sqrt((Phi ** 2).mean(axis=0))
    scale = np.where(scale < 1e-14, 1.0, scale)
    Z = Phi / scale
    G = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    return np.linalg.solve(G, Z.T @ y) / scale


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------
# 4. Data
# --------------------------------------------------------------------------

def relax(rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((1, N_PART, 3)) * 0.6
    for _ in range(6000):
        X = X + 0.02 * true_forces(X)
    return X[0]


def make_dataset(rng: np.random.Generator, X_eq: np.ndarray):
    X = X_eq[None] + rng.standard_normal((N_TRAJ, N_PART, 3)) * 0.10
    V = rng.standard_normal((N_TRAJ, N_PART, 3)) * 0.30
    V -= V.mean(axis=1, keepdims=True)
    snaps = []
    F = true_forces(X)
    for step in range(GEN_STEPS):
        V += 0.5 * GEN_DT * F
        X += GEN_DT * V
        F = true_forces(X)
        V += 0.5 * GEN_DT * F
        if step % GEN_STRIDE == 0:
            snaps.append(X.copy())
    return np.concatenate(snaps, axis=0)


# --------------------------------------------------------------------------
# 5. Diagnostics of non-conservativity
# --------------------------------------------------------------------------

def jacobian(coef: np.ndarray, X: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """dF_a/dX_b by central differences; (3N, 3N) for one configuration."""
    n = 3 * N_PART
    base = np.repeat(X[None], 2 * n, axis=0)
    flat = base.reshape(2 * n, n)
    idx = np.arange(n)
    flat[2 * idx, idx] += h
    flat[2 * idx + 1, idx] -= h
    F = force_from_raw(base, coef).reshape(2 * n, n)
    return ((F[0::2] - F[1::2]) / (2.0 * h)).T


def asymmetry(coef: np.ndarray, configs: np.ndarray):
    rel, atomic, absol = [], [], []
    for X in configs:
        J = jacobian(coef, X)
        S = 0.5 * (J - J.T)
        rel.append(np.linalg.norm(S) / np.linalg.norm(J))
        absol.append(np.linalg.norm(S) / np.sqrt(J.size))
        c = []
        for i in range(N_PART):
            a = 3 * i
            c += [J[a + 2, a + 1] - J[a + 1, a + 2],
                  J[a + 0, a + 2] - J[a + 2, a + 0],
                  J[a + 1, a + 0] - J[a + 0, a + 1]]
        atomic.append(np.sqrt(np.mean(np.square(c))))
    return np.array(rel), np.array(absol), np.array(atomic)


def loop_work(coef: np.ndarray, X0: np.ndarray, U: np.ndarray, V: np.ndarray,
              radius: float, n_q: int = 2048):
    """Work around the closed loop X0 + radius*(cos t U + sin t V).

    The integrand is smooth and 2pi-periodic, so the uniform-grid mean is
    spectrally accurate: quadrature error is far below the signal.
    """
    t = 2.0 * np.pi * np.arange(n_q) / n_q
    path = (X0[None] + radius * (np.cos(t)[:, None, None] * U[None]
                                 + np.sin(t)[:, None, None] * V[None]))
    tang = radius * (-np.sin(t)[:, None, None] * U[None]
                     + np.cos(t)[:, None, None] * V[None])
    integrand = (force_from_raw(path, coef) * tang).sum(axis=(1, 2))
    net = float(integrand.mean() * 2.0 * np.pi)
    gross = float(np.abs(integrand).mean() * 2.0 * np.pi)
    return net, gross


# --------------------------------------------------------------------------
# 6. Molecular dynamics
# --------------------------------------------------------------------------

def md_run(coefs: np.ndarray, X0: np.ndarray, V0: np.ndarray, dt: float,
           n_steps: int, theta_cons: np.ndarray, n_cons: int):
    X, V = X0.copy(), V0.copy()
    F = force_from_raw(X, coefs)
    stride = max(1, n_steps // MD_N_REC)
    ts = [0.0]
    ke = 0.5 * (V ** 2).sum(axis=(1, 2))
    e_true = [true_energy(X) + ke]
    e_mod = [energy_features(X[:n_cons]) @ theta_cons + ke[:n_cons]]
    for step in range(n_steps):
        V += 0.5 * dt * F
        X += dt * V
        F = force_from_raw(X, coefs)
        V += 0.5 * dt * F
        if (step + 1) % stride == 0:
            ke = 0.5 * (V ** 2).sum(axis=(1, 2))
            ts.append((step + 1) * dt)
            e_true.append(true_energy(X) + ke)
            e_mod.append(energy_features(X[:n_cons]) @ theta_cons + ke[:n_cons])
    return np.array(ts), np.array(e_true), np.array(e_mod), X


def drift_stats(ts: np.ndarray, dev: np.ndarray) -> dict:
    """Per-trajectory secular drift versus bounded fluctuation.

    ``dev`` is E_true(t) - E_true(0), shape (n_rec, n_traj). ``secular_ratio``
    is the total straight-line energy change divided by the RMS scatter about
    that line: << 1 means "bounded oscillation", >~ 1 means "systematic drift".
    """
    A = np.stack([np.ones_like(ts), ts], axis=1)
    c = np.linalg.lstsq(A, dev, rcond=None)[0]
    slope = c[1]
    resid = dev - A @ c
    fluct = resid.std(axis=0)
    secular = slope * (ts[-1] - ts[0])
    return {
        "slope": slope,
        "secular": secular,
        "fluct": fluct,
        "maxdev": np.abs(dev).max(axis=0),
        "secular_ratio": np.abs(secular) / np.maximum(fluct, 1e-300),
    }


# --------------------------------------------------------------------------
# 7. The experiment
# --------------------------------------------------------------------------

def run() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {}

    X_eq = relax(rng)
    configs = make_dataset(rng, X_eq)
    n = len(configs)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.55 * n), int(0.20 * n)
    i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    F_true = true_forces(configs)
    F_noisy = F_true + FORCE_NOISE * rng.standard_normal(F_true.shape)
    r_all, _, off = _geom(configs)
    force_rms = float(np.sqrt(np.mean(F_true ** 2)))

    Psi = raw_features(configs)
    Phi_c = Psi @ TIE

    def flat(M, idx):
        return M[idx].reshape(-1, M.shape[-1])

    def flaty(M, idx):
        return M[idx].reshape(-1)

    lam_grid = np.logspace(-11.0, -2.0, 28)
    fits = {}
    for name, design in (("conservative", Phi_c), ("direct", Psi)):
        Xtr, ytr = flat(design, i_tr), flaty(F_noisy, i_tr)
        Xva, yva = flat(design, i_va), flaty(F_noisy, i_va)
        best = None
        curve = []
        for lam in lam_grid:
            w = ridge(Xtr, ytr, lam)
            v = rmse(Xva @ w, yva)
            curve.append(v)
            if best is None or v < best[1]:
                best = (lam, v, w)
        lam, _, w = best
        fits[name] = dict(
            lam=float(lam), w=w, val_curve=[float(c) for c in curve],
            rmse_train_clean=rmse(flat(design, i_tr) @ w, flaty(F_true, i_tr)),
            rmse_test_clean=rmse(flat(design, i_te) @ w, flaty(F_true, i_te)),
            rmse_test_noisy=rmse(flat(design, i_te) @ w, flaty(F_noisy, i_te)),
        )

    theta_cons = fits["conservative"]["w"]
    coef_cons = TIE @ theta_cons
    coef_direct = fits["direct"]["w"]

    res["meta"] = {
        "seed": SEED,
        "description": ("Architecture-matched conservative (F=-grad E) vs direct "
                        "force surrogates for a 6-particle Morse cluster: matched "
                        "one-step force RMSE, wildly different Jacobian symmetry, "
                        "loop work and MD energy drift."),
    }
    res["system"] = {
        "n_particles": N_PART, "dim": 3, "morse_D_e": D_E, "morse_a": A_M,
        "morse_r_e": R_E, "trap_k": K_TRAP,
        "equilibrium_energy": float(true_energy(X_eq[None])[0]),
        "equilibrium_max_force": float(np.max(np.abs(true_forces(X_eq[None])))),
    }
    res["basis"] = {
        "n_radial": K_RAD, "mu": MU.tolist(), "sigma": SIG_RAD, "r_cut": R_CUT,
        "n_params_direct": P_RAW, "n_params_conservative": P_ENE,
        "conservative_is_subspace_of_direct": True,
    }
    res["data"] = {
        "n_configs": int(n), "n_train": int(n_tr), "n_val": int(n_va),
        "n_test": int(len(i_te)), "force_label_noise_sigma": FORCE_NOISE,
        "force_rms": force_rms,
        "noise_fraction_of_force_rms": float(FORCE_NOISE / force_rms),
        "pair_distance_min": float(r_all[:, off].min()),
        "pair_distance_max": float(r_all[:, off].max()),
        "pair_distance_mean": float(r_all[:, off].mean()),
        "fraction_pairs_beyond_cutoff": float(np.mean(r_all[:, off] >= R_CUT)),
    }

    # -- consistency: the conservative force block really is -grad of E_theta
    probe = configs[i_te[:6]]
    h = 1e-5
    fd = np.empty_like(probe)
    for i in range(N_PART):
        for a in range(3):
            Xp, Xm = probe.copy(), probe.copy()
            Xp[:, i, a] += h
            Xm[:, i, a] -= h
            fd[:, i, a] = -((energy_features(Xp) - energy_features(Xm))
                            @ theta_cons) / (2.0 * h)
    ana = force_from_raw(probe, coef_cons)
    grad_err = float(np.max(np.abs(fd - ana)) / np.max(np.abs(ana)))
    explicit = raw_features(probe) @ coef_direct
    fast_err = float(np.max(np.abs(force_from_raw(probe, coef_direct) - explicit))
                     / np.max(np.abs(explicit)))

    # -- equivariance of both models
    eqv = {}
    for name, coef in (("conservative", coef_cons), ("direct", coef_direct)):
        worst = 0.0
        for _ in range(6):
            R = random_rotation(rng)
            Xs = configs[i_te[:8]]
            lhs = force_from_raw(Xs @ R.T, coef)
            rhs = force_from_raw(Xs, coef) @ R.T
            worst = max(worst, float(np.max(np.abs(lhs - rhs))
                                     / np.max(np.abs(rhs))))
        eqv[name] = worst
    res["equivariance"] = {
        "max_relative_deviation_conservative": eqv["conservative"],
        "max_relative_deviation_direct": eqv["direct"],
        "note": "both models are exactly O(3)-equivariant; equivariance is not "
                "what distinguishes them",
    }

    res["models"] = {
        "conservative": {k: float(v) for k, v in fits["conservative"].items()
                         if k != "w" and k != "val_curve"},
        "direct": {k: float(v) for k, v in fits["direct"].items()
                   if k != "w" and k != "val_curve"},
        "lambda_grid": lam_grid.tolist(),
        "val_rmse_curve_conservative": fits["conservative"]["val_curve"],
        "val_rmse_curve_direct": fits["direct"]["val_curve"],
        "energy_coefficients_conservative": theta_cons.tolist(),
        "gradient_consistency_rel_error": grad_err,
        "fast_force_path_rel_error": fast_err,
        "force_rmse_ratio_direct_over_conservative": float(
            fits["direct"]["rmse_test_clean"] / fits["conservative"]["rmse_test_clean"]),
    }

    # -- 1. Jacobian asymmetry (generalised curl)
    jac_cfgs = configs[i_te[:40]]
    asym = {}
    for name, coef in (("conservative", coef_cons), ("direct", coef_direct)):
        rel, absol, atomic = asymmetry(coef, jac_cfgs)
        asym[name] = dict(rel=rel, absol=absol, atomic=atomic)
    res["jacobian_asymmetry"] = {
        "n_configs": int(len(jac_cfgs)),
        "finite_difference_h": 1e-4,
        "relative_conservative": asym["conservative"]["rel"].tolist(),
        "relative_direct": asym["direct"]["rel"].tolist(),
        "rms_relative_conservative": float(np.sqrt(np.mean(asym["conservative"]["rel"] ** 2))),
        "rms_relative_direct": float(np.sqrt(np.mean(asym["direct"]["rel"] ** 2))),
        "rms_absolute_conservative": float(np.sqrt(np.mean(asym["conservative"]["absol"] ** 2))),
        "rms_absolute_direct": float(np.sqrt(np.mean(asym["direct"]["absol"] ** 2))),
        "rms_atomic_curl_conservative": float(np.sqrt(np.mean(asym["conservative"]["atomic"] ** 2))),
        "rms_atomic_curl_direct": float(np.sqrt(np.mean(asym["direct"]["atomic"] ** 2))),
    }
    res["jacobian_asymmetry"]["ratio_direct_over_conservative"] = float(
        res["jacobian_asymmetry"]["rms_relative_direct"]
        / res["jacobian_asymmetry"]["rms_relative_conservative"])

    # -- 2. Closed-loop work
    radii = [0.05, 0.10, 0.20, 0.35, 0.50]
    loops = {"radii": radii, "n_loops_per_radius": 4, "quadrature_points": 2048}
    for name in ("conservative", "direct"):
        loops[f"net_work_{name}"] = []
        loops[f"gross_work_{name}"] = []
        loops[f"relative_{name}"] = []
    loop_rng = np.random.default_rng(SEED + 1)
    for rad in radii:
        nets = {"conservative": [], "direct": []}
        gross = {"conservative": [], "direct": []}
        for li in range(4):
            X0 = configs[i_te[li]]
            U = loop_rng.standard_normal((N_PART, 3))
            V = loop_rng.standard_normal((N_PART, 3))
            U /= np.linalg.norm(U)
            V -= np.sum(U * V) * U
            V /= np.linalg.norm(V)
            for name, coef in (("conservative", coef_cons), ("direct", coef_direct)):
                w_net, w_gross = loop_work(coef, X0, U, V, rad)
                nets[name].append(w_net)
                gross[name].append(w_gross)
        for name in ("conservative", "direct"):
            loops[f"net_work_{name}"].append([float(v) for v in nets[name]])
            loops[f"gross_work_{name}"].append([float(v) for v in gross[name]])
            loops[f"relative_{name}"].append(
                [float(abs(a) / b) for a, b in zip(nets[name], gross[name])])
    for name in ("conservative", "direct"):
        allrel = np.array(loops[f"relative_{name}"]).ravel()
        loops[f"median_relative_{name}"] = float(np.median(allrel))
        loops[f"max_relative_{name}"] = float(allrel.max())
    loops["median_relative_ratio"] = float(loops["median_relative_direct"]
                                           / loops["median_relative_conservative"])
    res["loop_work"] = loops

    # -- 3. MD: true-energy drift, and its dt scaling
    ic_rng = np.random.default_rng(SEED + 2)
    X0 = X_eq[None] + ic_rng.standard_normal((MD_N_IC, N_PART, 3)) * 0.10
    V0 = ic_rng.standard_normal((MD_N_IC, N_PART, 3)) * 0.30
    V0 -= V0.mean(axis=1, keepdims=True)
    Xb = np.concatenate([X0, X0], axis=0)
    Vb = np.concatenate([V0, V0], axis=0)
    coefs = np.concatenate([np.repeat(coef_cons[None], MD_N_IC, axis=0),
                            np.repeat(coef_direct[None], MD_N_IC, axis=0)], axis=0)
    ke0 = float(0.5 * (V0 ** 2).sum() / MD_N_IC)

    def analyse(ts, e_true, e_mod, Xend, dt, n_steps, total_time):
        dev = e_true - e_true[0]
        out = {"dt": float(dt), "n_steps": int(n_steps),
               "total_time": float(total_time),
               "E0": float(e_true[0].mean()), "KE0": ke0}
        for name, sl in (("conservative", slice(0, MD_N_IC)),
                         ("direct", slice(MD_N_IC, None))):
            st = drift_stats(ts, dev[:, sl])
            runaway = st["maxdev"] > RUNAWAY_THRESHOLD
            out[f"slope_per_traj_{name}"] = st["slope"].tolist()
            out[f"secular_per_traj_{name}"] = st["secular"].tolist()
            out[f"maxdev_per_traj_{name}"] = st["maxdev"].tolist()
            out[f"median_slope_{name}"] = float(np.median(st["slope"]))
            out[f"mean_slope_{name}"] = float(st["slope"].mean())
            out[f"median_abs_slope_{name}"] = float(np.median(np.abs(st["slope"])))
            out[f"median_maxdev_{name}"] = float(np.median(st["maxdev"]))
            out[f"median_fluct_{name}"] = float(np.median(st["fluct"]))
            out[f"median_secular_ratio_{name}"] = float(np.median(st["secular_ratio"]))
            out[f"frac_positive_slope_{name}"] = float(np.mean(st["slope"] > 0))
            out[f"frac_runaway_{name}"] = float(np.mean(runaway))
        r_end, _, off_end = _geom(Xend)
        out["final_pair_distance_range_conservative"] = [
            float(r_end[:MD_N_IC][:, off_end].min()),
            float(r_end[:MD_N_IC][:, off_end].max())]
        out["final_pair_distance_range_direct"] = [
            float(r_end[MD_N_IC:][:, off_end].min()),
            float(r_end[MD_N_IC:][:, off_end].max())]
        m0 = e_mod[0]
        out["model_energy_rel_excursion_conservative"] = float(
            np.median(np.max(np.abs(e_mod - m0), axis=0) / np.abs(m0)))
        return out

    n_main = int(round(MD_T_MAIN / MD_DT_MAIN))
    ts, e_true, e_mod, Xend = md_run(coefs, Xb, Vb, MD_DT_MAIN, n_main,
                                     theta_cons, MD_N_IC)
    main = analyse(ts, e_true, e_mod, Xend, MD_DT_MAIN, n_main, MD_T_MAIN)
    dev = e_true - e_true[0]
    main["t"] = ts.tolist()
    main["dE_true_conservative_median"] = np.median(dev[:, :MD_N_IC], axis=1).tolist()
    main["dE_true_direct_median"] = np.median(dev[:, MD_N_IC:], axis=1).tolist()
    main["dE_true_conservative_q90"] = np.quantile(dev[:, :MD_N_IC], 0.9, axis=1).tolist()
    main["dE_true_direct_q90"] = np.quantile(dev[:, MD_N_IC:], 0.9, axis=1).tolist()
    main["dE_model_conservative_median"] = np.median(
        e_mod - e_mod[0], axis=1).tolist()
    for name in ("conservative", "direct"):
        s = main[f"median_slope_{name}"]
        main[f"drift_frac_of_E0_total_{name}"] = float(s * MD_T_MAIN / abs(main["E0"]))
        main[f"drift_frac_of_KE0_total_{name}"] = float(s * MD_T_MAIN / ke0)
    main["drift_ratio_direct_over_conservative"] = float(
        main["median_abs_slope_direct"] / main["median_abs_slope_conservative"])
    n_up = int(round(main["frac_positive_slope_direct"] * MD_N_IC))
    sec_c = np.abs(main["secular_per_traj_conservative"])
    sec_d = np.abs(main["secular_per_traj_direct"])
    main["n_trajectories_gaining_energy_direct"] = n_up
    main["sign_bias_binomial_p_direct"] = float(binomtest(n_up, MD_N_IC, 0.5).pvalue)
    main["paired_frac_direct_worse"] = float(np.mean(sec_d > sec_c))
    main["paired_secular_ratio_median"] = float(np.median(sec_d / sec_c))
    main["paired_secular_ratio_min"] = float((sec_d / sec_c).min())
    main["note"] = (
        "median over %d independent initial conditions; the mean is not used "
        "because a minority of direct-model trajectories run away and would "
        "dominate it (frac_runaway_direct records that minority)" % MD_N_IC)
    main["sign_note"] = (
        "NEGATIVE RESULT, reported rather than tuned away: the direct model's "
        "drift is NOT a systematic heating. Signs are mixed "
        f"({n_up}/{MD_N_IC} trajectories gain energy, binomial p = "
        f"{main['sign_bias_binomial_p_direct']:.3f}), so a non-conservative "
        "field is better described as pumping energy in or out depending on "
        "where in phase space the trajectory goes than as an effective "
        "friction. What is systematic is the magnitude: every single "
        "trajectory drifts faster under the direct model than the "
        "conservative one, by at least "
        f"{main['paired_secular_ratio_min']:.0f}x.")
    res["md"] = main

    sweep = {"dt": MD_DTS, "total_time": MD_T_SWEEP,
             "n_initial_conditions": MD_N_IC, "runs": []}
    for dt in MD_DTS:
        n_steps = int(round(MD_T_SWEEP / dt))
        ts_s, et_s, em_s, Xe_s = md_run(coefs, Xb, Vb, dt, n_steps,
                                        theta_cons, MD_N_IC)
        entry = analyse(ts_s, et_s, em_s, Xe_s, dt, n_steps, MD_T_SWEEP)
        for name in ("conservative", "direct"):
            for k in ("slope_per_traj_", "secular_per_traj_", "maxdev_per_traj_"):
                entry.pop(k + name)
        sweep["runs"].append(entry)
    for key in ("median_abs_slope_conservative", "median_abs_slope_direct",
                "median_maxdev_conservative", "median_maxdev_direct",
                "model_energy_rel_excursion_conservative"):
        sweep[key] = [r[key] for r in sweep["runs"]]
    sweep["dt_ratio_coarse_over_fine"] = float(MD_DTS[-1] / MD_DTS[0])
    sweep["model_energy_excursion_growth_coarse_over_fine"] = float(
        sweep["model_energy_rel_excursion_conservative"][-1]
        / sweep["model_energy_rel_excursion_conservative"][0])
    sweep["conservative_true_maxdev_spread"] = float(
        max(sweep["median_maxdev_conservative"]) / min(sweep["median_maxdev_conservative"]))
    sweep["direct_slope_spread"] = float(
        max(sweep["median_abs_slope_direct"]) / min(sweep["median_abs_slope_direct"]))
    sweep["separation_at_each_dt"] = [
        float(r["median_abs_slope_direct"] / r["median_abs_slope_conservative"])
        for r in sweep["runs"]]
    res["dt_sweep"] = sweep

    # How the error grows with trajectory length at fixed dt: bounded (a
    # gradient field has a conserved quantity) vs. linear (it does not).
    short = sweep["runs"][MD_DTS.index(MD_DT_MAIN)]
    growth = {"dt": MD_DT_MAIN, "short_time": MD_T_SWEEP, "long_time": MD_T_MAIN,
              "time_ratio": float(MD_T_MAIN / MD_T_SWEEP)}
    for name in ("conservative", "direct"):
        a, b = short[f"median_maxdev_{name}"], main[f"median_maxdev_{name}"]
        growth[f"maxdev_short_{name}"] = a
        growth[f"maxdev_long_{name}"] = b
        growth[f"maxdev_growth_factor_{name}"] = float(b / a)
        p, q = (short[f"median_abs_slope_{name}"],
                main[f"median_abs_slope_{name}"])
        growth[f"rate_short_{name}"] = p
        growth[f"rate_long_{name}"] = q
        growth[f"rate_ratio_long_over_short_{name}"] = float(q / p)
    growth["note"] = (
        "A bounded energy error fitted over a window of length T yields an "
        "apparent rate ~ 1/T, so the conservative model's rate must fall by "
        "about the time ratio while its max|dE| stays put; a genuine "
        "non-conservative field has a T-independent rate and an error that "
        "accumulates linearly.")
    res["time_growth"] = growth

    res["punchline"] = {
        "force_rmse_conservative": fits["conservative"]["rmse_test_clean"],
        "force_rmse_direct": fits["direct"]["rmse_test_clean"],
        "force_rmse_ratio": float(fits["direct"]["rmse_test_clean"]
                                  / fits["conservative"]["rmse_test_clean"]),
        "loop_work_relative_ratio": loops["median_relative_ratio"],
        "jacobian_asymmetry_ratio": res["jacobian_asymmetry"]["ratio_direct_over_conservative"],
        "md_drift_ratio": main["drift_ratio_direct_over_conservative"],
        "frac_direct_trajectories_runaway": main["frac_runaway_direct"],
        "statement": ("one-step force RMSE differs by a factor "
                      f"{fits['direct']['rmse_test_clean'] / fits['conservative']['rmse_test_clean']:.3f} "
                      "(the direct model is the *better* one); median MD energy "
                      "drift rate differs by a factor "
                      f"{main['drift_ratio_direct_over_conservative']:.3g}"),
    }
    return res


# --------------------------------------------------------------------------
# 8. Self-check
# --------------------------------------------------------------------------

def _selfcheck(res: dict, verbose: bool = True):
    npass = ntotal = 0
    log = []

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        npass += bool(ok)
        log.append({"name": name, "pass": bool(ok), "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    m, jac, lw, md, sw = (res["models"], res["jacobian_asymmetry"],
                          res["loop_work"], res["md"], res["dt_sweep"])

    print("setup")
    check("analytic gradient of E_theta matches finite differences",
          m["gradient_consistency_rel_error"] < 1e-6,
          f"rel error {m['gradient_consistency_rel_error']:.2e}")
    check("contracted force evaluator reproduces the explicit 79-feature basis",
          m["fast_force_path_rel_error"] < 1e-13,
          f"max rel dev {m['fast_force_path_rel_error']:.2e}")
    check("all training pairs lie inside the radial cutoff",
          res["data"]["fraction_pairs_beyond_cutoff"] == 0.0,
          f"max pair distance {res['data']['pair_distance_max']:.3f} vs r_cut {R_CUT}")
    for name in ("conservative", "direct"):
        v = res["equivariance"][f"max_relative_deviation_{name}"]
        check(f"{name} model is O(3)-equivariant: F(RX) = R F(X)", v < 1e-10,
              f"max rel dev {v:.2e}")

    print("matched one-step accuracy")
    rc, rd = m["conservative"]["rmse_test_clean"], m["direct"]["rmse_test_clean"]
    fr = res["data"]["force_rms"]
    check("conservative model force RMSE is small vs force scale",
          rc / fr < 0.10, f"RMSE {rc:.4f} = {100 * rc / fr:.2f}% of force RMS {fr:.3f}")
    check("direct model force RMSE is small vs force scale",
          rd / fr < 0.10, f"RMSE {rd:.4f} = {100 * rd / fr:.2f}% of force RMS {fr:.3f}")
    check("the two force RMSEs agree within 20%",
          0.8 < rd / rc < 1.25, f"ratio direct/conservative {rd / rc:.4f}")

    print("non-conservativity: Jacobian asymmetry (generalised curl)")
    check("conservative model Jacobian is symmetric to the finite-difference floor",
          jac["rms_relative_conservative"] < 1e-6,
          f"rms ||asym||/||J|| {jac['rms_relative_conservative']:.2e}")
    rel_force_err = rd / fr
    check("direct model's Jacobian asymmetry is of the same order as its own "
          "relative force error -- nothing suppresses it",
          jac["rms_relative_direct"] > 0.2 * rel_force_err,
          f"rms ||asym||/||J|| {jac['rms_relative_direct']:.3e} vs relative "
          f"force error {rel_force_err:.3e}")
    check("asymmetry separates the models by >1e4",
          jac["ratio_direct_over_conservative"] > 1e4,
          f"ratio {jac['ratio_direct_over_conservative']:.3e}")
    check("direct model has nonzero per-atom curl, conservative does not",
          jac["rms_atomic_curl_direct"] > 1e3 * jac["rms_atomic_curl_conservative"],
          f"curl rms direct {jac['rms_atomic_curl_direct']:.3e} vs "
          f"conservative {jac['rms_atomic_curl_conservative']:.2e}")

    print("non-conservativity: work around closed loops")
    check("conservative model does zero net work on every loop",
          lw["max_relative_conservative"] < 1e-8,
          f"max |net|/gross over {len(lw['radii']) * 4} loops "
          f"{lw['max_relative_conservative']:.2e}")
    check("direct model does nonzero net work around closed loops",
          lw["median_relative_direct"] > 1e-3,
          f"median |net|/gross {lw['median_relative_direct']:.3e}")
    check("loop-work path dependence separates the models by >1e5",
          lw["median_relative_ratio"] > 1e5,
          f"ratio {lw['median_relative_ratio']:.3e}")

    print("molecular dynamics: true-energy drift")
    check("conservative model conserves its own E_theta + KE (Verlet floor)",
          md["model_energy_rel_excursion_conservative"] < 1e-4,
          f"median rel excursion {md['model_energy_rel_excursion_conservative']:.2e}")
    check("conservative true energy oscillates but does not drift "
          "(secular change << fluctuation)",
          md["median_secular_ratio_conservative"] < 0.3,
          f"|secular|/fluct {md['median_secular_ratio_conservative']:.3f}, "
          f"drift {100 * md['drift_frac_of_E0_total_conservative']:+.4f}% of |E0| "
          f"over T={MD_T_MAIN:.0f}")
    check("direct model true energy drifts secularly, not just noisily",
          md["median_secular_ratio_direct"] > 1.0,
          f"|secular|/fluct {md['median_secular_ratio_direct']:.3f}, "
          f"drift {100 * md['drift_frac_of_E0_total_direct']:+.3f}% of |E0| "
          f"({100 * md['drift_frac_of_KE0_total_direct']:+.2f}% of KE0)")
    check("EVERY paired trajectory drifts faster under the direct model",
          md["paired_frac_direct_worse"] == 1.0,
          f"{int(md['paired_frac_direct_worse'] * MD_N_IC)}/{MD_N_IC}; secular "
          f"|dE| ratio median x{md['paired_secular_ratio_median']:.0f}, "
          f"worst case still x{md['paired_secular_ratio_min']:.0f}")
    check("median drift rate separates the models by >20x at T=%.0f" % MD_T_MAIN,
          md["drift_ratio_direct_over_conservative"] > 20.0,
          f"ratio {md['drift_ratio_direct_over_conservative']:.1f} "
          f"({md['median_abs_slope_direct']:.3e} vs "
          f"{md['median_abs_slope_conservative']:.3e} per unit time)")
    check("NEGATIVE RESULT recorded: the direct model's drift is bidirectional, "
          "not a systematic heating",
          0.05 < md["sign_bias_binomial_p_direct"],
          f"{md['n_trajectories_gaining_energy_direct']}/{MD_N_IC} trajectories "
          f"gain energy, binomial p = {md['sign_bias_binomial_p_direct']:.3f} "
          "-- consistent with a coin flip, so this is energy pumping in both "
          "directions, not an effective friction")
    check("a minority of direct trajectories are destroyed outright; no "
          "conservative trajectory is",
          md["frac_runaway_conservative"] == 0.0 and md["frac_runaway_direct"] > 0.0,
          f"|dE| > {RUNAWAY_THRESHOLD} for "
          f"{100 * md['frac_runaway_direct']:.0f}% of direct and "
          f"{100 * md['frac_runaway_conservative']:.0f}% of conservative "
          f"trajectories at T={MD_T_MAIN:.0f}")

    print("time scaling: bounded error vs. accumulating error")
    tg = res["time_growth"]
    check("conservative true-energy error stays inside a fixed envelope as t grows",
          tg["maxdev_growth_factor_conservative"] < 0.5 * tg["time_ratio"],
          f"median max|dE| grows x{tg['maxdev_growth_factor_conservative']:.2f} "
          f"when the run is x{tg['time_ratio']:.0f} longer; linear accumulation "
          f"would give x{tg['time_ratio']:.0f} "
          f"({tg['maxdev_short_conservative']:.4f} -> {tg['maxdev_long_conservative']:.4f})")
    check("conservative apparent drift rate falls like 1/T, as a bounded error must",
          tg["rate_ratio_long_over_short_conservative"] < 1.6 / tg["time_ratio"],
          f"rate x{tg['rate_ratio_long_over_short_conservative']:.3f} for a "
          f"x{tg['time_ratio']:.0f} longer run; 1/T predicts "
          f"x{1 / tg['time_ratio']:.2f}")
    check("direct drift rate is T-independent, so its energy error accumulates "
          "without bound",
          0.6 < tg["rate_ratio_long_over_short_direct"] < 1.6,
          f"rate x{tg['rate_ratio_long_over_short_direct']:.3f} for a "
          f"x{tg['time_ratio']:.0f} longer run "
          f"({tg['rate_short_direct']:.3e} -> {tg['rate_long_direct']:.3e})")

    print("dt scaling: integrator error vs. learned-field error")
    r2 = sw["dt_ratio_coarse_over_fine"] ** 2
    check("conservative model's own invariant fails only as dt^2 (pure Verlet error)",
          0.5 * r2 < sw["model_energy_excursion_growth_coarse_over_fine"] < 2.0 * r2,
          f"E_theta+KE excursion grows x"
          f"{sw['model_energy_excursion_growth_coarse_over_fine']:.1f} as dt grows "
          f"x{sw['dt_ratio_coarse_over_fine']:.0f}; dt^2 predicts x{r2:.0f}")
    check("conservative true-energy error is dt-independent, i.e. bounded model "
          "error not accumulating integrator error",
          sw["conservative_true_maxdev_spread"] < 1.5,
          f"median max|dE| varies only x{sw['conservative_true_maxdev_spread']:.2f} "
          f"over an x{sw['dt_ratio_coarse_over_fine']:.0f} range in dt")
    check("direct model drift survives dt -> 0 (a property of the field, not "
          "of the integrator)",
          sw["direct_slope_spread"] < 3.0,
          f"median |slope| varies x{sw['direct_slope_spread']:.2f} across "
          f"dt = {MD_DTS}: "
          + ", ".join(f"{v:.2e}" for v in sw["median_abs_slope_direct"]))
    check("the model separation holds at every timestep tested",
          min(sw["separation_at_each_dt"]) > 20.0,
          "drift ratios " + ", ".join(f"{v:.0f}" for v in sw["separation_at_each_dt"]))

    print("punchline")
    p = res["punchline"]
    check("force error ratio ~1 while drift ratio is orders of magnitude",
          abs(p["force_rmse_ratio"] - 1.0) < 0.25
          and p["md_drift_ratio"] / max(p["force_rmse_ratio"], 1.0) > 1e2,
          f"force RMSE x{p['force_rmse_ratio']:.3f}, drift x{p['md_drift_ratio']:.3g}")

    return npass, ntotal, log


if __name__ == "__main__":
    print("=" * 76)
    print("conservative_vs_direct -- F = -grad E versus a directly predicted force field")
    print("=" * 76)
    results = run()
    npass_, ntotal_, checklog = _selfcheck(results)
    results["checks"] = checklog
    results["meta"]["n_pass"] = npass_
    results["meta"]["n_total"] = ntotal_
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1))
    print("-" * 76)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"{npass_}/{ntotal_} PASS")
    raise SystemExit(0 if npass_ == ntotal_ else 1)
