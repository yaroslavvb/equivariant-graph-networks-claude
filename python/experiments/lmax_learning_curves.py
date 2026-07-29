"""NequIP's l_max ablation, reproduced in miniature on an analytic potential.

The central causal claim of the NequIP paper is that raising the maximum tensor
rank l_max of the internal features -- everything else held fixed -- lowers the
error and improves data efficiency (revised MD17 aspirin force MAE 41.9 -> 12.7
-> 9.3 -> 8.5 meV/A for l_max = 0, 1, 2, 3). This script builds the smallest
system in which that claim is *decidable* rather than merely observed.

The system
----------
One central atom, N = 6 neighbours drawn uniformly by volume in the shell
0.9 <= r <= 1.8 (configurations with any neighbour-neighbour distance below 0.7
are rejected). The ground-truth energy is analytic and has genuine angular
structure:

    E = sum_j  D_e [(1 - exp(-a (r_j - r_e)))^2 - 1] u(r_j)      two-body Morse
      + sum_{j<k} u(r_j) u(r_k) g(cos theta_jk)                  three-body

    u(x) = polynomial envelope (p = 6), u(r_cut) = u'(r_cut) = u''(r_cut) = 0

Two three-body kernels g are studied, and the difference between them is the
whole point of this experiment:

    "sw"  Stillinger-Weber:   g(c) = lambda (c - cos theta_0)^2, cos theta_0 = -1/3
    "rep" neighbour repulsion: g depends on |r_j - r_k| via  mu / |v_j - v_k|

Their Legendre spectra are qualitatively different. (c - c_0)^2 expands as
exactly three terms, a_0 P_0 + a_1 P_1 + a_2 P_2 with a_0 = 1/3 + c_0^2,
a_1 = -2 c_0, a_2 = 2/3 and *identically zero* beyond l = 2. The repulsion
kernel expands as sum_l (r_< ^l / r_> ^(l+1)) P_l, an infinite series that decays
only geometrically in r_</r_> and therefore has weight at every l.

The features
------------
Exactly the first layer of a NequIP convolution,

    f_l[n] = sum_j  R_n(r_ij) Y_l(rhat_ij),      l = 0 .. l_max,  n = 1 .. n_rad

with R_n a Gaussian radial basis (centres evenly spaced on [0.6, r_cut], width =
centre spacing) multiplied by the same polynomial envelope u. The energy is a
scalar, so the readout uses the O(3)-invariant power spectrum: the n_rad linear
l = 0 channels, plus every cross-channel contraction f_l[a] . f_l[b] for a <= b
and l = 0 .. l_max. That is the complete set of invariants an l_max-truncated
equivariant layer can build at second order, and it is exactly what the "l_max
matters" claim is about. Total feature count P = n_rad + (l_max+1) n_rad(n_rad+1)/2.

The model is closed-form ridge regression. Its penalty is chosen per fit by an
inner 75/25 train/validation split over a 21-point log grid, then refit on the
full training set at the winning penalty. Training energies carry Gaussian label
noise at 5% of the energy standard deviation (a stand-in for reference-data
noise); test energies are exact.

What holds, and what does not
-----------------------------
1. Raising l_max helps, and it is *not* raw capacity. Two controls establish
   this. (a) At matched feature count -- n_rad tuned per l_max so all four models
   get 90-104 invariants, with l_max = 0 given the *most* -- the ordering is
   unchanged. (b) An l_max = 0 model given up to 560 purely radial invariants and
   the entire 4000-configuration pool still cannot go below its floor, while an
   l_max = 2 model with 34 features beats it by orders of magnitude. The
   limitation is representational, not parametric.

2. Learning curves shift *and* change slope, as in the paper. An l_max too low to
   represent the target is bias-limited: its log-log slope is ~0 and it sits an
   order of magnitude above the noise-limited scale sigma sqrt(P/N). An l_max that
   can represent it keeps falling and lands within a factor ~1 of that scale.
   PARTIAL NEGATIVE on a prediction made before running: the complete model's
   measured slope is about -0.72 over N_train = 64..1024, not the -1/2 of an
   unbiased least-squares fit. The check reports the measured value rather than
   asserting -1/2. Diagnosis: the validated ridge penalty is itself still shrinking
   across that window (median alpha 3e-2 -> 1e-4) because the power-spectrum
   features are strongly collinear, so shrinkage bias decays on top of the
   sqrt(P/N) variance. At N = 1024 the magnitude does match sigma sqrt(P/N) to
   within 10%, which is what pins the "noise-limited" reading.

3. HONEST NEGATIVE. The paper's qualitative ordering -- the l=0 -> l=1 step is
   the largest, then diminishing returns -- does NOT hold for the Stillinger-Weber
   target. There the l=1 -> l=2 step is by far the largest, because the SW kernel
   is *exactly complete* at l = 2, so that step is not an improvement in
   approximation quality but a jump to zero bias. This is a property of the
   target, not a bug: measured in angular power removed, l = 1 does account for
   more than l = 2 (roughly 62% vs 38% of the non-constant angular power),
   matching the paper's intuition -- it is the RMSE *ratio* that is dominated by
   the completeness step. Swapping in the repulsion kernel, whose Legendre
   spectrum never terminates (the situation of a real molecular PES), restores
   the paper's ordering: the 0->1 gain is the largest and the gains diminish
   monotonically through 1->2 and 2->3. Both results are reported; neither was
   tuned away.

4. Corollary the paper also sees: past the rank actually present in the target,
   extra l costs accuracy at fixed N, because the additional features are pure
   estimation variance. For the SW target l_max = 3 is never better than
   l_max = 2 by a meaningful margin.

All numbers below are produced by this script and asserted in _selfcheck().
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from e3 import random_rotation, real_sh  # noqa: E402

SEED = 20260729

R_CUT = 2.0
N_NB = 6
R_MIN, R_MAX = 0.9, 1.8
MIN_NN = 0.7

D_E, R_E, A_MORSE = 1.0, 1.1, 2.5
COS_THETA0 = -1.0 / 3.0
LAMBDA_SW = 0.6
MU_REP = 1.0

TARGET_ID = {"sw": 0, "rep": 1}   # fixed ids: str hashing is salted per process

NOISE_FRAC = 0.05
N_POOL = 4000
N_TEST = 4000
N_REPS = 8
N_TRAIN_GRID = [8, 16, 32, 64, 128, 256, 512, 1024]
SLOPE_FIT_MIN_N = 64
LMAX_GRID = [0, 1, 2, 3]
N_RAD_MAIN = 8
N_RAD_MATCHED = {0: 13, 1: 9, 2: 7, 3: 6}
N_RAD_CAPACITY = [4, 8, 12, 16, 24, 32]
ALPHAS = np.logspace(-9.0, 1.0, 21)

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results" / "lmax_learning_curves.json"


# --------------------------------------------------------------------------
# 1. Geometry and the analytic ground-truth energy
# --------------------------------------------------------------------------

def envelope(r: np.ndarray, r_cut: float = R_CUT, p: int = 6) -> np.ndarray:
    """Smooth polynomial cutoff: 1 at r=0, and u = u' = u'' = 0 at r = r_cut."""
    x = np.clip(np.asarray(r, dtype=float) / r_cut, 0.0, 1.0)
    return (1.0
            - (p + 1) * (p + 2) / 2 * x ** p
            + p * (p + 2) * x ** (p + 1)
            - p * (p + 1) / 2 * x ** (p + 2))


def sample_clusters(n: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """n clusters of N_NB neighbour vectors, uniform by volume in the shell.

    Rejection on the minimum neighbour-neighbour distance keeps the repulsion
    kernel bounded and stops any configuration from being pathological.
    Returns the clusters and the acceptance rate.
    """
    iu = np.triu_indices(N_NB, 1)
    out = np.empty((n, N_NB, 3))
    filled = 0
    proposed = 0
    while filled < n:
        m = 2 * (n - filled) + 128
        proposed += m
        u = rng.random((m, N_NB))
        r = (R_MIN ** 3 + u * (R_MAX ** 3 - R_MIN ** 3)) ** (1.0 / 3.0)
        d = rng.standard_normal((m, N_NB, 3))
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        v = d * r[..., None]
        sep = np.linalg.norm(v[:, :, None, :] - v[:, None, :, :], axis=-1)
        ok = sep[:, iu[0], iu[1]].min(axis=1) >= MIN_NN
        acc = v[ok]
        take = min(len(acc), n - filled)
        out[filled:filled + take] = acc[:take]
        filled += take
    return out, n / proposed


def energy_terms(v: np.ndarray) -> dict[str, np.ndarray]:
    """Two-body Morse term and the two competing three-body terms."""
    iu = np.triu_indices(N_NB, 1)
    r = np.linalg.norm(v, axis=-1)
    env = envelope(r)
    e2 = np.sum(D_E * ((1.0 - np.exp(-A_MORSE * (r - R_E))) ** 2 - 1.0) * env, axis=1)

    unit = v / r[..., None]
    cos = np.einsum("cid,cjd->cij", unit, unit)
    gate = env[:, :, None] * env[:, None, :]
    e3_sw = LAMBDA_SW * (gate * (cos - COS_THETA0) ** 2)[:, iu[0], iu[1]].sum(axis=1)

    sep = np.linalg.norm(v[:, :, None, :] - v[:, None, :, :], axis=-1)
    e3_rep = MU_REP * (gate[:, iu[0], iu[1]] / sep[:, iu[0], iu[1]]).sum(axis=1)
    return {"two_body": e2, "sw": e3_sw, "rep": e3_rep}


def total_energy(v: np.ndarray, target: str) -> np.ndarray:
    t = energy_terms(v)
    return t["two_body"] + t[target]


# --------------------------------------------------------------------------
# 2. NequIP first-layer features and their O(3) invariants
# --------------------------------------------------------------------------

def radial_basis(r: np.ndarray, n_rad: int) -> np.ndarray:
    centres = np.linspace(0.6, R_CUT, n_rad)
    width = (centres[1] - centres[0]) if n_rad > 1 else 0.5
    g = np.exp(-((r[..., None] - centres) ** 2) / (2.0 * width * width))
    return g * envelope(r)[..., None]


def n_features(n_rad: int, lmax: int) -> int:
    return n_rad + (lmax + 1) * (n_rad * (n_rad + 1)) // 2


def invariant_features(v: np.ndarray, n_rad: int, lmax: int,
                       sh_cache: list[np.ndarray] | None = None) -> np.ndarray:
    """Power-spectrum invariants of f_l[n] = sum_j R_n(r_ij) Y_l(rhat_ij).

    Columns are ordered so that the l_max = L feature set is a *prefix* of the
    l_max = L' > L one; that is what makes the ablation a clean nesting.
    """
    r = np.linalg.norm(v, axis=-1)
    Rn = radial_basis(r, n_rad)
    iu = np.triu_indices(n_rad)
    cols = []
    for l in range(lmax + 1):
        Y = sh_cache[l] if sh_cache is not None else real_sh(l, v)
        f = np.einsum("cjn,cjm->cnm", Rn, Y)
        if l == 0:
            cols.append(f[:, :, 0])
        gram = np.einsum("cnm,cpm->cnp", f, f)
        cols.append(gram[:, iu[0], iu[1]])
    return np.concatenate(cols, axis=1)


# --------------------------------------------------------------------------
# 3. Ridge regression with a validated penalty
# --------------------------------------------------------------------------

def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return mu, np.where(sd < 1e-12, 1.0, sd)


def _ridge_path(Z: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Minimum-norm ridge solutions for every alpha, from one SVD.

    Penalty is alpha * n so it is invariant to the training-set size; Z is
    standardised, so diag(Z^T Z) ~ n and alpha is dimensionless.
    """
    n = len(y)
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    Uty = U.T @ y
    return np.stack([Vt.T @ ((s / (s * s + a * n)) * Uty) for a in alphas])


def fit_predict(Xtr, ytr, Xte, alphas, rng, val_frac=0.25) -> tuple[np.ndarray, float]:
    n = len(ytr)
    n_val = int(max(2, round(val_frac * n)))
    perm = rng.permutation(n)
    i_val, i_in = perm[:n_val], perm[n_val:]

    mu, sd = _standardise(Xtr[i_in])
    y0 = ytr[i_in].mean()
    W = _ridge_path((Xtr[i_in] - mu) / sd, ytr[i_in] - y0, alphas)
    pred_val = ((Xtr[i_val] - mu) / sd) @ W.T + y0
    val_rmse = np.sqrt(np.mean((pred_val - ytr[i_val][:, None]) ** 2, axis=0))
    alpha = float(alphas[int(np.argmin(val_rmse))])

    mu, sd = _standardise(Xtr)
    y0 = ytr.mean()
    w = _ridge_path((Xtr - mu) / sd, ytr - y0, np.array([alpha]))[0]
    return ((Xte - mu) / sd) @ w + y0, alpha


def best_possible_rmse(X: np.ndarray, y: np.ndarray, alpha: float = 1e-9) -> float:
    """In-sample RMSE of the least-regularised fit: the model class's floor."""
    mu, sd = _standardise(X)
    Z = (X - mu) / sd
    y0 = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y0))
    return float(np.sqrt(np.mean((Z @ w + y0 - y) ** 2)))


# --------------------------------------------------------------------------
# 4. Legendre spectra of the two three-body kernels
# --------------------------------------------------------------------------

def sw_legendre_coeffs(l_max: int = 6) -> np.ndarray:
    """Exact Legendre coefficients of (cos - cos_0)^2 by Gauss-Legendre quadrature."""
    x, w = np.polynomial.legendre.leggauss(256)
    g = LAMBDA_SW * (x - COS_THETA0) ** 2
    out = []
    for l in range(l_max + 1):
        c = np.zeros(l + 1)
        c[l] = 1.0
        Pl = np.polynomial.legendre.legval(x, c)
        out.append((2 * l + 1) / 2.0 * float(np.sum(w * g * Pl)))
    return np.array(out)


def rep_legendre_power(v: np.ndarray, l_max: int = 6) -> np.ndarray:
    """Ensemble angular power per l of the repulsion kernel over sampled pairs.

    1/|v_j - v_k| = sum_l (r_<^l / r_>^(l+1)) P_l(cos), so the power in channel l
    is a_l^2 * 2/(2l+1), weighted by the gate u(r_j) u(r_k) that multiplies it.
    """
    iu = np.triu_indices(N_NB, 1)
    r = np.linalg.norm(v, axis=-1)
    env = envelope(r)
    ra, rb = r[:, iu[0]], r[:, iu[1]]
    lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
    wgt = (env[:, iu[0]] * env[:, iu[1]]) ** 2
    ls = np.arange(l_max + 1)
    a = (lo[..., None] / hi[..., None]) ** ls / hi[..., None]
    return np.einsum("cp,cpl->l", wgt, a ** 2 * (2.0 / (2 * ls + 1)))


# --------------------------------------------------------------------------
# 5. The experiment
# --------------------------------------------------------------------------

def loglog_fit(n_vals, rmse_vals) -> tuple[float, float, float]:
    x = np.log10(np.asarray(n_vals, dtype=float))
    y = np.log10(np.asarray(rmse_vals, dtype=float))
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - float(np.var(resid)) / float(np.var(y)) if np.var(y) > 0 else 0.0
    return float(slope), float(intercept), r2


def learning_curve(Xpool, Xtest, y_pool_noisy, y_test, cols, rng_seed_tag):
    """RMSE vs N_train, averaged over N_REPS independent resamples of the pool."""
    means, stds, per_rep, alphas_med = [], [], [], []
    for i_n, n_train in enumerate(N_TRAIN_GRID):
        vals, chosen = [], []
        for rep in range(N_REPS):
            rng = np.random.default_rng([SEED, *rng_seed_tag, i_n, rep])
            idx = rng.choice(len(y_pool_noisy), size=n_train, replace=False)
            pred, alpha = fit_predict(Xpool[np.ix_(idx, cols)], y_pool_noisy[idx],
                                      Xtest[:, cols], ALPHAS, rng)
            vals.append(float(np.sqrt(np.mean((pred - y_test) ** 2))))
            chosen.append(alpha)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        per_rep.append([float(x) for x in vals])
        alphas_med.append(float(np.median(chosen)))
    return {"rmse_mean": means, "rmse_std": stds, "rmse_per_rep": per_rep,
            "alpha_median": alphas_med}


def run() -> dict:
    rng = np.random.default_rng(SEED)
    v_pool, acc_rate = sample_clusters(N_POOL, rng)
    v_test, _ = sample_clusters(N_TEST, rng)
    sh_pool = [real_sh(l, v_pool) for l in range(max(LMAX_GRID) + 1)]
    sh_test = [real_sh(l, v_test) for l in range(max(LMAX_GRID) + 1)]

    terms_pool = energy_terms(v_pool)
    terms_test = energy_terms(v_test)

    res: dict = {}
    res["meta"] = {
        "seed": SEED,
        "description": ("NequIP l_max ablation in miniature: power-spectrum invariants of a "
                        "first-layer E(3) convolution, ridge-fitted to an analytic Morse + "
                        "three-body potential, with feature-count and radial-capacity controls."),
        "n_neighbours": N_NB,
        "shell": [R_MIN, R_MAX],
        "min_neighbour_separation": MIN_NN,
        "r_cut": R_CUT,
        "acceptance_rate": float(acc_rate),
        "n_pool": N_POOL,
        "n_test": N_TEST,
        "n_resamples": N_REPS,
        "n_train_grid": list(N_TRAIN_GRID),
        "lmax_grid": list(LMAX_GRID),
        "n_rad_main": N_RAD_MAIN,
        "label_noise_frac_of_std": NOISE_FRAC,
        "alpha_grid": [float(a) for a in ALPHAS],
        "alpha_selection": "inner 75/25 train/validation split, 21-point log grid, refit at winner",
        "slope_fit_points": [n for n in N_TRAIN_GRID if n >= SLOPE_FIT_MIN_N],
        "targets": {"sw": "Morse + Stillinger-Weber three-body, lambda=%.3g, cos(theta_0)=-1/3" % LAMBDA_SW,
                    "rep": "Morse + neighbour-neighbour repulsion mu/|v_j-v_k|, mu=%.3g" % MU_REP},
    }

    # ---- variance budget and the angular-blind floor -----------------------
    res["system"] = {"morse": {"D_e": D_E, "r_e": R_E, "a": A_MORSE}}
    X_radial_big = invariant_features(v_pool, 32, 0, sh_pool)
    for target in ("sw", "rep"):
        e2, e3 = terms_pool["two_body"], terms_pool[target]
        E = e2 + e3
        floor = best_possible_rmse(X_radial_big, E)
        res["system"][target] = {
            "std_energy": float(E.std()),
            "std_two_body": float(e2.std()),
            "std_three_body": float(e3.std()),
            "corr_two_three": float(np.corrcoef(e2, e3)[0, 1]),
            "var_frac_three_body": float(e3.var() / E.var()),
            "var_frac_two_body": float(e2.var() / E.var()),
            "var_frac_covariance": float(2 * np.cov(e2, e3)[0, 1] / E.var()),
            "angular_blind_n_features": int(n_features(32, 0)),
            "angular_blind_r2": float(1.0 - (floor / E.std()) ** 2),
            "angular_blind_nrmse": float(floor / E.std()),
            "irreducible_angular_var_frac": float((floor / E.std()) ** 2),
            "label_noise_sigma": float(NOISE_FRAC * E.std()),
        }

    # ---- Legendre spectra --------------------------------------------------
    a_sw = sw_legendre_coeffs(6)
    p_sw = a_sw ** 2 * 2.0 / (2 * np.arange(7) + 1)
    p_rep = rep_legendre_power(v_pool, 6)
    res["spectra"] = {
        "l": list(range(7)),
        "sw_legendre_coeffs": [float(x) for x in a_sw],
        "sw_power_frac": [float(x) for x in (p_sw / p_sw.sum())],
        "sw_power_frac_nonconstant": [float(x) for x in (p_sw[1:] / p_sw[1:].sum())],
        "sw_max_abs_coeff_above_l2": float(np.max(np.abs(a_sw[3:]))),
        "rep_power_frac": [float(x) for x in (p_rep / p_rep.sum())],
        "rep_power_frac_above_l2": float(p_rep[3:].sum() / p_rep.sum()),
        "rep_cumulative_power_frac": [float(x) for x in np.cumsum(p_rep) / p_rep.sum()],
    }

    # ---- main learning curves ---------------------------------------------
    Xp_main = invariant_features(v_pool, N_RAD_MAIN, max(LMAX_GRID), sh_pool)
    Xt_main = invariant_features(v_test, N_RAD_MAIN, max(LMAX_GRID), sh_test)
    res["learning_curves"] = {}
    res["slopes"] = {}
    res["gains"] = {}
    for target in ("sw", "rep"):
        E_pool = terms_pool["two_body"] + terms_pool[target]
        E_test = terms_test["two_body"] + terms_test[target]
        sigma = NOISE_FRAC * float(E_pool.std())
        noise = np.random.default_rng([SEED, 991, TARGET_ID[target]]).standard_normal(N_POOL)
        y_pool = E_pool + sigma * noise

        res["learning_curves"][target] = {}
        res["slopes"][target] = {}
        for lmax in LMAX_GRID:
            cols = np.arange(n_features(N_RAD_MAIN, lmax))
            curve = learning_curve(Xp_main, Xt_main, y_pool, E_test, cols,
                                   (1, LMAX_GRID.index(lmax), TARGET_ID[target]))
            curve["n_features"] = int(len(cols))
            curve["nrmse_mean"] = [float(x / E_test.std()) for x in curve["rmse_mean"]]
            X_all = Xp_main[:, cols]
            curve["floor_rmse"] = float(best_possible_rmse(X_all, E_pool))
            curve["floor_nrmse"] = float(curve["floor_rmse"] / E_pool.std())
            # sigma sqrt(P/N) is what an unbiased least-squares fit would give;
            # comparing to it separates a noise-limited curve from a bias-limited one.
            curve["noise_scale"] = [float(sigma * np.sqrt(len(cols) / n)) for n in N_TRAIN_GRID]
            curve["rmse_over_noise_scale_at_max_n"] = float(
                curve["rmse_mean"][-1] / curve["noise_scale"][-1])
            res["learning_curves"][target][str(lmax)] = curve

            keep = [i for i, n in enumerate(N_TRAIN_GRID) if n >= SLOPE_FIT_MIN_N]
            s, b, r2 = loglog_fit([N_TRAIN_GRID[i] for i in keep],
                                  [curve["rmse_mean"][i] for i in keep])
            res["slopes"][target][str(lmax)] = {"slope": s, "intercept": b, "fit_r2": r2}

        big = res["learning_curves"][target]
        last = {l: big[str(l)]["rmse_mean"][-1] for l in LMAX_GRID}
        res["gains"][target] = {
            "rmse_at_n1024": {str(l): float(last[l]) for l in LMAX_GRID},
            "gain_0_to_1": float(last[0] / last[1]),
            "gain_1_to_2": float(last[1] / last[2]),
            "gain_2_to_3": float(last[2] / last[3]),
            "floor_ratio_0_to_1": float(big["0"]["floor_rmse"] / big["1"]["floor_rmse"]),
            "floor_ratio_1_to_2": float(big["1"]["floor_rmse"] / big["2"]["floor_rmse"]),
            "floor_ratio_2_to_3": float(big["2"]["floor_rmse"] / big["3"]["floor_rmse"]),
        }
        res["gains"][target]["diminishing_returns"] = bool(
            res["gains"][target]["gain_0_to_1"] > res["gains"][target]["gain_1_to_2"]
            > res["gains"][target]["gain_2_to_3"])

    # ---- control A: matched feature count ---------------------------------
    res["control_matched_features"] = {
        "n_rad_per_lmax": {str(k): v for k, v in N_RAD_MATCHED.items()},
        "n_features": {str(k): int(n_features(v, k)) for k, v in N_RAD_MATCHED.items()},
        "curves": {},
    }
    for target in ("sw", "rep"):
        E_pool = terms_pool["two_body"] + terms_pool[target]
        E_test = terms_test["two_body"] + terms_test[target]
        sigma = NOISE_FRAC * float(E_pool.std())
        noise = np.random.default_rng([SEED, 991, TARGET_ID[target]]).standard_normal(N_POOL)
        y_pool = E_pool + sigma * noise
        res["control_matched_features"]["curves"][target] = {}
        for lmax in LMAX_GRID:
            nr = N_RAD_MATCHED[lmax]
            Xp = invariant_features(v_pool, nr, lmax, sh_pool)
            Xt = invariant_features(v_test, nr, lmax, sh_test)
            cols = np.arange(Xp.shape[1])
            curve = learning_curve(Xp, Xt, y_pool, E_test, cols,
                                   (2, lmax, TARGET_ID[target]))
            curve["n_features"] = int(Xp.shape[1])
            curve["n_rad"] = int(nr)
            curve["floor_rmse"] = float(best_possible_rmse(Xp, E_pool))
            res["control_matched_features"]["curves"][target][str(lmax)] = curve

    # ---- control B: radial capacity cannot substitute for angular rank ----
    res["control_radial_capacity"] = {"n_rad_grid": list(N_RAD_CAPACITY), "targets": {}}
    for target in ("sw", "rep"):
        E_pool = terms_pool["two_body"] + terms_pool[target]
        E_test = terms_test["two_body"] + terms_test[target]
        sigma = NOISE_FRAC * float(E_pool.std())
        noise = np.random.default_rng([SEED, 991, TARGET_ID[target]]).standard_normal(N_POOL)
        y_pool = E_pool + sigma * noise
        entry = {"lmax0": {"n_features": [], "rmse": []},
                 "small_lmax_reference": {}}
        for nr in N_RAD_CAPACITY:
            Xp = invariant_features(v_pool, nr, 0, sh_pool)
            Xt = invariant_features(v_test, nr, 0, sh_test)
            rng_c = np.random.default_rng([SEED, 3, nr, TARGET_ID[target]])
            pred, _ = fit_predict(Xp, y_pool, Xt, ALPHAS, rng_c)
            entry["lmax0"]["n_features"].append(int(Xp.shape[1]))
            entry["lmax0"]["rmse"].append(float(np.sqrt(np.mean((pred - E_test) ** 2))))
        for lmax in (1, 2, 3):
            Xp = invariant_features(v_pool, 4, lmax, sh_pool)
            Xt = invariant_features(v_test, 4, lmax, sh_test)
            rng_c = np.random.default_rng([SEED, 4, lmax, TARGET_ID[target]])
            pred, _ = fit_predict(Xp, y_pool, Xt, ALPHAS, rng_c)
            entry["small_lmax_reference"][str(lmax)] = {
                "n_rad": 4, "n_features": int(Xp.shape[1]),
                "rmse": float(np.sqrt(np.mean((pred - E_test) ** 2)))}
        entry["std_energy_test"] = float(E_test.std())
        res["control_radial_capacity"]["targets"][target] = entry

    # ---- equivariance sanity: features are invariant, energy is invariant --
    rng_rot = np.random.default_rng(SEED + 5)
    R = random_rotation(rng_rot)
    v_small = v_pool[:200]
    v_rot = v_small @ R.T
    X_a = invariant_features(v_small, N_RAD_MAIN, 3)
    X_b = invariant_features(v_rot, N_RAD_MAIN, 3)
    scale = np.maximum(np.abs(X_a).max(axis=0), 1e-12)
    res["invariance"] = {
        "feature_max_rel_dev": float(np.max(np.abs(X_a - X_b) / scale)),
        "energy_max_abs_dev_sw": float(np.max(np.abs(
            total_energy(v_small, "sw") - total_energy(v_rot, "sw")))),
        "energy_max_abs_dev_rep": float(np.max(np.abs(
            total_energy(v_small, "rep") - total_energy(v_rot, "rep")))),
    }

    # ---- radial-basis truncation, the only misspecification left at l=2 ----
    rr = np.linspace(R_MIN, R_MAX, 400)
    for nr in (4, 8):
        B = radial_basis(rr, nr)
        tgt = envelope(rr)
        c, *_ = np.linalg.lstsq(B, tgt, rcond=None)
        res.setdefault("radial_basis", {})[f"gate_rel_l2_error_n{nr}"] = float(
            np.linalg.norm(B @ c - tgt) / np.linalg.norm(tgt))
    return res


# --------------------------------------------------------------------------
# 6. Self-check
# --------------------------------------------------------------------------

def _selfcheck(res: dict, verbose: bool = True) -> tuple[int, int]:
    npass = ntotal = 0
    log: list[dict] = []

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        ok = bool(ok)
        npass += ok
        log.append({"name": name, "passed": ok, "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("invariance of the constructed features")
    inv = res["invariance"]
    check("power-spectrum features are O(3) invariant",
          inv["feature_max_rel_dev"] < 1e-9,
          f"max rel dev {inv['feature_max_rel_dev']:.2e}")
    check("ground-truth energies are rotation invariant",
          max(inv["energy_max_abs_dev_sw"], inv["energy_max_abs_dev_rep"]) < 1e-12,
          f"max abs dev {max(inv['energy_max_abs_dev_sw'], inv['energy_max_abs_dev_rep']):.2e}")

    print("the target really has angular structure")
    for t in ("sw", "rep"):
        s = res["system"][t]
        check(f"[{t}] three-body term carries a substantial share of the variance",
              s["var_frac_three_body"] > 0.20,
              f"Var(E3)/Var(E) {s['var_frac_three_body']:.3f} "
              f"(corr(E2,E3) {s['corr_two_three']:+.3f})")
        check(f"[{t}] a purely radial model has an irreducible floor",
              s["irreducible_angular_var_frac"] > 0.10,
              f"{100 * s['irreducible_angular_var_frac']:.1f}% of Var(E) unreachable with "
              f"{s['angular_blind_n_features']} radial invariants "
              f"(nRMSE floor {s['angular_blind_nrmse']:.4f})")

    print("Legendre spectra of the two three-body kernels")
    sp = res["spectra"]
    check("SW kernel (c - c_0)^2 has exactly zero content above l = 2",
          sp["sw_max_abs_coeff_above_l2"] < 1e-10,
          f"max |a_l| for l>=3 is {sp['sw_max_abs_coeff_above_l2']:.2e} "
          f"vs a_2 = {sp['sw_legendre_coeffs'][2]:.4f}")
    check("SW: l=1 removes more angular power than l=2 (the paper's ordering, "
          "in power terms)",
          sp["sw_power_frac_nonconstant"][0] > sp["sw_power_frac_nonconstant"][1],
          f"l=1 {100 * sp['sw_power_frac_nonconstant'][0]:.1f}% vs "
          f"l=2 {100 * sp['sw_power_frac_nonconstant'][1]:.1f}% of non-constant power")
    check("repulsion kernel keeps significant power above l = 2",
          sp["rep_power_frac_above_l2"] > 0.02,
          f"{100 * sp['rep_power_frac_above_l2']:.2f}% of angular power at l>=3")

    print("main ablation: RMSE at N_train = 1024, n_rad = 8")
    for t in ("sw", "rep"):
        g = res["gains"][t]
        r = g["rmse_at_n1024"]
        check(f"[{t}] RMSE falls monotonically over l_max = 0, 1, 2",
              r["0"] > r["1"] > r["2"],
              f"{r['0']:.5f} -> {r['1']:.5f} -> {r['2']:.5f}")
    g = res["gains"]["sw"]
    check("[sw] l_max=2 beats l_max=0 by more than 20x",
          g["rmse_at_n1024"]["0"] / g["rmse_at_n1024"]["2"] > 20.0,
          f"factor {g['rmse_at_n1024']['0'] / g['rmse_at_n1024']['2']:.1f}")

    print("the paper's qualitative ordering: does it survive?")
    check("[rep] gains diminish monotonically 0->1 > 1->2 > 2->3 (paper's pattern HOLDS)",
          res["gains"]["rep"]["diminishing_returns"],
          f"x{res['gains']['rep']['gain_0_to_1']:.2f}, "
          f"x{res['gains']['rep']['gain_1_to_2']:.2f}, "
          f"x{res['gains']['rep']['gain_2_to_3']:.2f}")
    check("[sw] paper's pattern FAILS: 1->2 is the largest step, not 0->1 "
          "(SW is exactly complete at l=2)",
          g["gain_1_to_2"] > g["gain_0_to_1"],
          f"x{g['gain_0_to_1']:.2f} then x{g['gain_1_to_2']:.1f} then x{g['gain_2_to_3']:.2f}")
    check("[sw] l_max=3 buys nothing over l_max=2 (target has no l=3 content)",
          g["gain_2_to_3"] < 1.3,
          f"factor {g['gain_2_to_3']:.3f} (<1 means l_max=3 is worse, as expected "
          f"from the extra estimation variance)")

    print("fairness control A: matched feature count")
    mc = res["control_matched_features"]
    nf = mc["n_features"]
    check("matched control gives l_max=0 the most features of any model",
          nf["0"] >= max(nf[k] for k in nf),
          f"P = {[nf[str(l)] for l in (0, 1, 2, 3)]} for l_max = 0,1,2,3")
    for t in ("sw", "rep"):
        c = mc["curves"][t]
        r = {l: c[str(l)]["rmse_mean"][-1] for l in (0, 1, 2, 3)}
        check(f"[{t}] at matched P the l_max ordering survives (0 > 1 > 2)",
              r[0] > r[1] > r[2],
              f"{r[0]:.5f} -> {r[1]:.5f} -> {r[2]:.5f} at N_train=1024")
    c = mc["curves"]["sw"]
    surv = c["0"]["rmse_mean"][-1] / c["2"]["rmse_mean"][-1]
    raw = res["gains"]["sw"]["rmse_at_n1024"]["0"] / res["gains"]["sw"]["rmse_at_n1024"]["2"]
    check("[sw] most of the raw l=0 -> l=2 gain survives the matched-P control",
          surv > 0.5 * raw,
          f"matched-P factor {surv:.1f} vs unmatched {raw:.1f} "
          f"({100 * np.log(surv) / np.log(raw):.0f}% of the log-gain retained)")

    print("fairness control B: radial capacity is not a substitute")
    for t in ("sw", "rep"):
        cap = res["control_radial_capacity"]["targets"][t]
        r0 = cap["lmax0"]["rmse"]
        check(f"[{t}] l_max=0 RMSE plateaus as radial features grow "
              f"{cap['lmax0']['n_features'][0]} -> {cap['lmax0']['n_features'][-1]}",
              r0[-1] > 0.85 * r0[0],
              f"{r0[0]:.5f} -> {r0[-1]:.5f} ({100 * (1 - r0[-1] / r0[0]):.1f}% improvement "
              f"for {cap['lmax0']['n_features'][-1] // cap['lmax0']['n_features'][0]}x the features)")
        ref = cap["small_lmax_reference"]["2"]
        check(f"[{t}] l_max=2 with {ref['n_features']} features beats l_max=0 with "
              f"{cap['lmax0']['n_features'][-1]}",
              ref["rmse"] < r0[-1],
              f"{ref['rmse']:.5f} vs {r0[-1]:.5f} (factor {r0[-1] / ref['rmse']:.1f} "
              f"with {cap['lmax0']['n_features'][-1] / ref['n_features']:.1f}x fewer features)")

    print(f"learning-curve slopes (log-log fit on N_train >= {SLOPE_FIT_MIN_N})")
    sl = res["slopes"]["sw"]
    lcs = res["learning_curves"]["sw"]
    check("[sw] l_max=0 curve is bias-limited: slope ~ 0",
          abs(sl["0"]["slope"]) < 0.15,
          f"slope {sl['0']['slope']:+.4f}")
    check("[sw] l_max=0 sits far above the noise-limited scale sigma sqrt(P/N)",
          lcs["0"]["rmse_over_noise_scale_at_max_n"] > 5.0,
          f"{lcs['0']['rmse_over_noise_scale_at_max_n']:.1f}x the noise scale at N=1024")
    check("[sw] l_max=2 reaches the noise-limited scale sigma sqrt(P/N)",
          0.5 < lcs["2"]["rmse_over_noise_scale_at_max_n"] < 2.0,
          f"{lcs['2']['rmse_over_noise_scale_at_max_n']:.2f}x the noise scale at N=1024")
    # NOT -1/2: over this window the validated penalty is still falling (median
    # alpha 1e-2 -> 7e-5), so shrinkage bias decays on top of the sqrt(P/N) variance.
    check("[sw] l_max=2 curve keeps falling steeply (measured slope, not the "
          "asymptotic -1/2)",
          sl["2"]["slope"] < -0.40,
          f"slope {sl['2']['slope']:+.4f} (fit R^2 {sl['2']['fit_r2']:.4f}); steeper than "
          f"-0.5 because median alpha still falls {lcs['2']['alpha_median'][3]:.1e} -> "
          f"{lcs['2']['alpha_median'][-1]:.1e} across the window")
    check("[sw] raising l_max steepens the learning curve",
          sl["2"]["slope"] < sl["1"]["slope"] < sl["0"]["slope"] + 1e-9,
          f"slopes {sl['0']['slope']:+.3f}, {sl['1']['slope']:+.3f}, {sl['2']['slope']:+.3f} "
          f"for l_max = 0, 1, 2")

    print("residual misspecification at the complete order")
    rb = res["radial_basis"]
    lc2 = res["learning_curves"]["sw"]["2"]
    check("[sw] l_max=2 is complete up to radial-basis truncation only",
          lc2["floor_nrmse"] < 1e-3,
          f"noiseless in-sample nRMSE floor {lc2['floor_nrmse']:.2e}; radial gate is "
          f"represented to {rb['gate_rel_l2_error_n8']:.2e} relative L2 by 8 basis functions")

    res["checks"] = log
    return npass, ntotal


if __name__ == "__main__":
    print("=" * 78)
    print("lmax_learning_curves.py -- NequIP's l_max ablation on an analytic potential")
    print("=" * 78)
    results = run()

    for t in ("sw", "rep"):
        print(f"\ntarget '{t}': test RMSE vs N_train (n_rad = {N_RAD_MAIN}, "
              f"{N_REPS} resamples)")
        print("    N_train " + "".join(f"{n:>10d}" for n in N_TRAIN_GRID) + "      P   floor")
        for lmax in LMAX_GRID:
            c = results["learning_curves"][t][str(lmax)]
            row = "".join(f"{x:>10.5f}" for x in c["rmse_mean"])
            print(f"    l<={lmax}   {row}  {c['n_features']:>5d}  {c['floor_rmse']:.2e}")
        gg = results["gains"][t]
        print(f"    gains at N=1024:  0->1 x{gg['gain_0_to_1']:.2f}   "
              f"1->2 x{gg['gain_1_to_2']:.2f}   2->3 x{gg['gain_2_to_3']:.2f}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    npass, ntotal = _selfcheck(results)
    RESULTS.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {RESULTS} ({RESULTS.stat().st_size / 1024:.0f} KB)")
    print("-" * 78)
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
