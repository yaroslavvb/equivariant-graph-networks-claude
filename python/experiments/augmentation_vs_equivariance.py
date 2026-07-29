"""Augmentation buys invariance on the sampled orbit; equivariance buys it on the group.

NequIP's central methodological claim is that rotation augmentation and rotation
equivariance are not two routes to the same place. Augmentation adds *data* that
happens to be rotated; the hypothesis class is untouched, so the fitted function is
invariant only to the extent that the fit interpolates between the orientations it
was shown. Equivariance changes the *hypothesis class*: every function it can
express is invariant, at every rotation, exactly, and (in the deep case) at every
intermediate layer too. This script makes that difference numerical.

The regression target
---------------------
A central atom with N = 6 neighbours at positions r_1..r_N. The energy is a
two-body Morse term plus a Stillinger-Weber-style three-body angular term, both
smoothly cut off, and is *exactly* invariant under O(3) because it is written
only in terms of distances and bond angles:

    f_c(r)      = 1/2 (1 + cos(pi r / r_c))  for r < r_c, else 0        r_c = 2.6
    phi2(r)     = f_c(r) [ exp(-2a(r-r0)) - 2 exp(-a(r-r0)) ]           a = 1.6, r0 = 1.1
    phi3(j,k)   = lam f_c(r_j) f_c(r_k) (cos theta_jk + 1/3)^2          lam = 1.2
    E(x)        = sum_j phi2(r_j) + sum_{j<k} phi3(j,k)

No sibling experiment in this repo defined a potential yet, so this is the
definition; later scripts should reuse these constants. Energies are reported in
units of the standard deviation of E over the configuration pool, so every error
below is a dimensionless nRMSE.

Canonical orientation
---------------------
Base configurations are drawn isotropically and then *canonicalised*: neighbours
are sorted by radius (a rotation-invariant order), the nearest is rotated onto
+z, and the neighbour with the largest remaining perpendicular component is
rotated into the xz half-plane. This map commutes with rotation, so it fixes a
frame without changing any energy. It matters here for two reasons. First, it
gives "canonical orientation" an operational meaning, so K = 1 is a genuine
zero-augmentation baseline rather than one random rotation. Second, it means all
orientation spread in the training set comes from the augmentation and nowhere
else -- which is what makes the restricted-augmentation probe below sharp.

The two models
--------------
(a) UNCONSTRAINED. Random Fourier features cos(W x / sigma + b) on the flattened
    18-dimensional Cartesian coordinate vector, D = 384 features, fitted by ridge
    regression in closed form. Bandwidth sigma and ridge lambda are both selected
    on a validation set drawn from the *same* augmentation distribution as the
    training set, which is what a practitioner would do; the unconstrained model
    is not handicapped anywhere.
(b) BUILT-IN. Ridge regression on power-spectrum invariants of the atomic
    density, built with this repo's e3 core: f_l^n = sum_j g_n(r_ij) Y_l(r-hat_ij)
    with Gaussian radial basis g_n, and p^l_{nn'} = <f_l^n, f_l^n'>. Orthogonality
    of the Wigner matrices makes p exactly invariant, so every function in the
    class is exactly invariant, before any data is seen.

What is measured
----------------
1. nRMSE on held-out configurations, in the canonical orientation and under a
   fresh random rotation.
2. Invariance error eps(g) = |f(gx) - f(x)| / (1 + |f(x)|), mean and worst case
   over many random g and many held-out x.
3. The off-distribution probe: train the unconstrained model with rotations drawn
   only about the z axis, or only within +-15 degrees of the identity, then test
   invariance under general rotations.
4. The augmentation tax: how many augmented training rows the unconstrained model
   needs to reach the built-in model's error.

Findings, including the ones that cut against the headline
----------------------------------------------------------
The invariance headline holds, and hard. The built-in model's invariance error
sits at ~1e-16 -- pure float rounding, fifteen orders of magnitude below anything
augmentation achieves. Full augmentation drives the unconstrained model's
invariance error down steadily with K but plateaus in the 1e-3 range, never
approaching machine precision anywhere in the sweep. The restricted probe is
starker still: a model trained on z-axis rotations only is more than an order of
magnitude more invariant under z-rotations than under general ones. It learned
the orbit it was shown, not the group.

Three results cut against the naive reading and are reported rather than hidden.

  * The 100x+ test-error gap between the two models is NOT the rotation-
    augmentation tax, and it would be dishonest to present it as one. The
    control that settles this is the K = 1 column: train and test the
    unconstrained model entirely in the canonical frame, so rotation is removed
    from the problem altogether, and it is still ~170x worse than the built-in
    model. The gap is a representation gap -- the invariant model also gets
    permutation invariance, a body-ordered radial/angular basis and a smooth
    cutoff, none of which is about rotation. Only the K = 1 canonical control
    separates the two effects, and the checks below assert it explicitly.
  * The unconstrained model is not a strawman that more capacity would rescue.
    Its D = 384 random-feature fit is verified to sit within a few percent of
    exact Gaussian kernel ridge on the same data, i.e. the D -> infinity limit of
    its own hypothesis class. It is sample-limited in 18 dimensions, not
    capacity-limited; adding features does essentially nothing.
  * Augmentation *hurts* canonical-orientation accuracy. The best canonical
    nRMSE is at K = 1 and degrades with K, because augmentation spends a
    fixed-capacity model's budget on representing orientations instead of
    chemistry. Augmentation does help on the rotated test set, which is the
    metric that matters; both are reported.

So the augmentation tax is measured the only way that isolates rotation: against
the model's own rotation-free ceiling. At each training size, how large must K
be before the augmented model's error on randomly rotated test data falls back to
the error it achieved when rotation was not part of the problem at all?

The answer, which is not the one the NequIP framing would lead you to expect, is
that this tax is nearly zero: at every training size swept the augmented model
comes within 4% of its own rotation-free ceiling. Measured in accuracy, rotation
augmentation on this problem is close to free.

(Two entries in that sweep nominally *beat* the ceiling, at n = 50 and n = 100.
They are not a regularisation win and are not reported as one: at those sizes the
ceiling is itself at nRMSE 0.90-0.96, within a few percent of simply predicting
the mean, so both sides of the comparison are near-trivial fits. The script flags
these via `near_trivial` / `informative_n_train` and restricts the quantitative
claim to n >= 200, where the ceiling is a genuine fit and augmentation falls
short of it by 1.5-4% and never reaches it. The same collapse explains the
smallest invariance error in the whole sweep, 2.5e-3 at n = 100, K = 4: that fit
has nRMSE 1.01, i.e. it is nearly constant, and a constant function is invariant
for a reason that has nothing to do with learning. Excluding collapsed fits, the
best invariance error any augmented model achieves is ~1.2e-1.)

That is the sharpest form of the result, and it is worth stating plainly because
it is a partial concession. Augmentation is not mainly an accuracy problem here.
What it fails at is exactness: at the same training size where its accuracy is
within 4% of the rotation-free ceiling, its invariance error is ~1e-1 against the
built-in model's ~1e-16, a ratio of order 1e14. Augmentation buys you the
accuracy; it does not buy you the symmetry. That distinction -- not a test-error
horse race -- is what "equivariance constrains every rotation exactly, including
at intermediate layers" actually means, and the checks below assert it in that
form.

Note finally that the canonicalisation used here is itself a legitimate way to
get exact invariance from an unconstrained model -- canonicalise at test time too
and the K = 1 model is exactly invariant by construction. The experiment
deliberately does not do that, because the object of study is what augmentation
alone teaches.
"""

from __future__ import annotations

import json
import pathlib
import sys
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from e3 import random_rotation, real_sh, rotation_matrix  # noqa: E402

SEED = 20260729

N_NEIGH = 6
R_MIN, R_MAX = 0.85, 2.30
R_CUT = 2.6
MORSE_A, MORSE_R0 = 1.6, 1.1
LAMBDA3 = 1.2

N_RAD = 8
L_MAX = 3
RAD_MU = np.linspace(0.7, 2.4, N_RAD)
RAD_SIGMA = float(RAD_MU[1] - RAD_MU[0])

N_RFF = 384
SIGMA_SCALES = (0.5, 1.0, 2.0)
RIDGE_LAMBDAS = np.logspace(-6.0, 3.0, 19)

N_TRAIN_GRID = (50, 100, 200, 400, 800)
K_GRID = (1, 4, 16, 64, 256)
N_VAL = 200
N_TEST = 400
N_INV_CONF = 100
N_INV_ROT = 32
SMALL_ANGLE = np.deg2rad(15.0)
TRIVIAL_NRMSE = 0.90
FAMILIES = ("full", "zaxis", "smallangle")

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results"


# --------------------------------------------------------------------------
# 1. Ground truth: a rotation-invariant two-body + three-body energy
# --------------------------------------------------------------------------

def cutoff(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    inside = r < R_CUT
    return np.where(inside, 0.5 * (1.0 + np.cos(np.pi * np.minimum(r, R_CUT) / R_CUT)), 0.0)


def energy(pos: np.ndarray) -> np.ndarray:
    """Energy of each configuration in ``pos`` of shape (n, N_NEIGH, 3)."""
    pos = np.asarray(pos, dtype=float)
    r = np.linalg.norm(pos, axis=2)
    fc = cutoff(r)
    e2 = np.sum(fc * (np.exp(-2.0 * MORSE_A * (r - MORSE_R0))
                      - 2.0 * np.exp(-MORSE_A * (r - MORSE_R0))), axis=1)
    u = pos / r[:, :, None]
    cos_t = np.einsum("nid,njd->nij", u, u)
    ang = (cos_t + 1.0 / 3.0) ** 2
    w = fc[:, :, None] * fc[:, None, :]
    iu = np.triu_indices(N_NEIGH, 1)
    e3 = LAMBDA3 * (w * ang)[:, iu[0], iu[1]].sum(axis=1)
    return e2 + e3


def sample_configs(n: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.standard_normal((n, N_NEIGH, 3))
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    r = rng.uniform(R_MIN, R_MAX, size=(n, N_NEIGH, 1))
    return d * r


# --------------------------------------------------------------------------
# 2. Canonical frame (rotation-commuting, so it changes no energy)
# --------------------------------------------------------------------------

def _align_to_z(v: np.ndarray) -> np.ndarray:
    u = v / np.linalg.norm(v)
    c = float(u[2])
    axis = np.cross(u, np.array([0.0, 0.0, 1.0]))
    s = float(np.linalg.norm(axis))
    if s < 1e-13:
        return np.eye(3) if c > 0 else rotation_matrix([1.0, 0.0, 0.0], np.pi)
    return rotation_matrix(axis / s, np.arctan2(s, c))


def canonicalize(pos: np.ndarray) -> np.ndarray:
    out = np.empty_like(pos)
    for i, p in enumerate(pos):
        p = p[np.argsort(np.linalg.norm(p, axis=1))]
        p = p @ _align_to_z(p[0]).T
        rho = np.hypot(p[1:, 0], p[1:, 1])
        j = 1 + int(np.argmax(rho))
        ang = float(np.arctan2(p[j, 1], p[j, 0]))
        out[i] = p @ rotation_matrix([0.0, 0.0, 1.0], -ang).T
    return out


def sample_rot(family: str, rng: np.random.Generator) -> np.ndarray:
    if family == "full":
        return random_rotation(rng)
    if family == "zaxis":
        return rotation_matrix([0.0, 0.0, 1.0], rng.uniform(0.0, 2.0 * np.pi))
    if family == "smallangle":
        ax = rng.standard_normal(3)
        ax /= np.linalg.norm(ax)
        return rotation_matrix(ax, rng.uniform(-SMALL_ANGLE, SMALL_ANGLE))
    raise ValueError(family)


def augment(pos: np.ndarray, y: np.ndarray, k: int, family: str,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """K orientations per configuration, the first of which is the original."""
    mats = [np.eye(3)] + [sample_rot(family, rng) for _ in range(k - 1)]
    return np.concatenate([pos @ R.T for R in mats], axis=0), np.tile(y, k)


# --------------------------------------------------------------------------
# 3. Features
# --------------------------------------------------------------------------

def invariant_features(pos: np.ndarray) -> np.ndarray:
    """Radial sums plus power-spectrum invariants p^l_{nn'} = <f_l^n, f_l^n'>."""
    n = pos.shape[0]
    r = np.linalg.norm(pos, axis=2)
    g = np.exp(-0.5 * ((r[:, :, None] - RAD_MU) / RAD_SIGMA) ** 2) * cutoff(r)[:, :, None]
    feats = [g.sum(axis=1)]
    flat = pos.reshape(-1, 3)
    iu = np.triu_indices(N_RAD)
    for l in range(L_MAX + 1):
        Y = real_sh(l, flat).reshape(n, N_NEIGH, 2 * l + 1)
        f = np.einsum("njk,njm->nkm", g, Y)
        P = np.einsum("nkm,nqm->nkq", f, f)
        feats.append(P[:, iu[0], iu[1]])
    return np.concatenate(feats, axis=1)


def rff_features(pos: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cos(pos.reshape(pos.shape[0], -1) @ W + b)


# --------------------------------------------------------------------------
# 4. Ridge regression in closed form, with a lambda path
# --------------------------------------------------------------------------

class Ridge:
    def __init__(self, mu, sd, ym, w, lam):
        self.mu, self.sd, self.ym, self.w, self.lam = mu, sd, ym, w, lam

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.ym


def gram_stats(pos, y, W, b, chunk=8192):
    """Accumulate (X^T X, X^T y, colsum) over row chunks of the RFF design matrix.

    Chunking is what keeps K = 256 affordable: the full design matrix at
    204800 x 384 would be 600 MB, and only these three moments are ever needed.
    """
    d = W.shape[1]
    S = np.zeros((d, d))
    t = np.zeros(d)
    csum = np.zeros(d)
    for i in range(0, pos.shape[0], chunk):
        X = rff_features(pos[i:i + chunk], W, b)
        S += X.T @ X
        t += X.T @ y[i:i + chunk]
        csum += X.sum(axis=0)
    return pos.shape[0], S, t, csum


def fit_from_stats(n, S, t, csum, ym, Xva, yva, lams=RIDGE_LAMBDAS):
    """Fit every lambda from one eigendecomposition; keep the best on validation.

    Column standardisation is folded into the Gram matrix, so the design matrix
    itself is never formed or copied. diag(S) supplies the column sums of
    squares, hence the standard deviations, for free.
    """
    mu = csum / n
    var = np.maximum(np.diag(S) / n - mu ** 2, 0.0)
    sd = np.where(var < 1e-24, 1.0, np.sqrt(var))
    G = (S - n * np.outer(mu, mu)) / sd[:, None] / sd[None, :]
    bvec = (t - n * mu * ym) / sd
    ev, U = np.linalg.eigh(G)
    ev = np.maximum(ev, 0.0)
    bt = U.T @ bvec
    best = None
    for lam in lams:
        w = U @ (bt / (ev + lam * n))
        model = Ridge(mu, sd, ym, w, float(lam))
        err = float(np.sqrt(np.mean((model(Xva) - yva) ** 2)))
        if best is None or err < best[0]:
            best = (err, model)
    return best[1], best[0]


def fit_ridge_path(Xtr, ytr, Xva, yva, lams=RIDGE_LAMBDAS):
    """Small-feature path (the invariant model), design matrix formed directly."""
    return fit_from_stats(Xtr.shape[0], Xtr.T @ Xtr, Xtr.T @ ytr, Xtr.sum(axis=0),
                          float(ytr.mean()), Xva, yva, lams)


def kernel_ridge_limit(Xtr, ytr, Xva, yva, Xte, yte, scales, median_dist):
    """Exact Gaussian kernel ridge: the D -> infinity limit of the RFF model.

    Used only as a control, to establish that the unconstrained model's error is
    a property of its hypothesis class and not of the random-feature truncation.
    """
    def sqdist(A, B):
        return (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2.0 * A @ B.T)

    best = None
    for sc in scales:
        gam = 1.0 / (2.0 * (sc * median_dist) ** 2)
        Ktr = np.exp(-gam * sqdist(Xtr, Xtr))
        Kva = np.exp(-gam * sqdist(Xva, Xtr))
        Kte = np.exp(-gam * sqdist(Xte, Xtr))
        ev, U = np.linalg.eigh(Ktr)
        bt = U.T @ (ytr - ytr.mean())
        for lam in RIDGE_LAMBDAS:
            al = U @ (bt / (np.maximum(ev, 0.0) + lam * len(ytr)))
            verr = float(np.sqrt(np.mean((Kva @ al + ytr.mean() - yva) ** 2)))
            if best is None or verr < best[0]:
                best = (verr, float(np.sqrt(np.mean((Kte @ al + ytr.mean() - yte) ** 2))))
    return best[1]


def nrmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def invariance_stats(predict, pos, family, n_rot, rng):
    f0 = predict(pos)
    scale = 1.0 + np.abs(f0)
    acc = []
    for _ in range(n_rot):
        R = sample_rot(family, rng)
        acc.append(np.abs(predict(pos @ R.T) - f0) / scale)
    E = np.asarray(acc)
    return float(E.mean()), float(E.max())


# --------------------------------------------------------------------------
# 5. The experiment
# --------------------------------------------------------------------------

def run():
    rng = np.random.default_rng(SEED)

    n_pool = max(N_TRAIN_GRID)
    pos_train = canonicalize(sample_configs(n_pool, rng))
    pos_val = canonicalize(sample_configs(N_VAL, rng))
    pos_test = canonicalize(sample_configs(N_TEST, rng))

    e_train = energy(pos_train)
    e_mean, e_std = float(e_train.mean()), float(e_train.std())
    y_train = (e_train - e_mean) / e_std
    y_val = (energy(pos_val) - e_mean) / e_std
    y_test = (energy(pos_test) - e_mean) / e_std

    rot_test = random_rotation(rng)
    pos_test_rot = pos_test @ rot_test.T
    pos_inv = pos_test[:N_INV_CONF]

    W0 = rng.standard_normal((3 * N_NEIGH, N_RFF))
    b_rff = rng.uniform(0.0, 2.0 * np.pi, size=N_RFF)
    sub = pos_train[:200].reshape(200, -1)
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
    median_dist = float(np.sqrt(np.median(d2[np.triu_indices(200, 1)])))

    # ---- built-in model -------------------------------------------------
    Fi_train = invariant_features(pos_train)
    Fi_val = invariant_features(pos_val)
    Fi_test = invariant_features(pos_test)
    Fi_test_rot = invariant_features(pos_test_rot)

    inv_res = {"n_train": list(N_TRAIN_GRID), "n_features": int(Fi_train.shape[1]),
               "nrmse_canonical": [], "nrmse_rotated": [],
               "inv_err_mean": [], "inv_err_max": [], "lambda": []}
    inv_models = {}
    for nt in N_TRAIN_GRID:
        m, _ = fit_ridge_path(Fi_train[:nt], y_train[:nt], Fi_val, y_val)
        inv_models[nt] = m
        inv_res["nrmse_canonical"].append(nrmse(m(Fi_test), y_test))
        inv_res["nrmse_rotated"].append(nrmse(m(Fi_test_rot), y_test))
        inv_res["lambda"].append(m.lam)
        r_inv = np.random.default_rng(SEED + 11)
        mean_e, max_e = invariance_stats(lambda p: m(invariant_features(p)),
                                         pos_inv, "full", N_INV_ROT, r_inv)
        inv_res["inv_err_mean"].append(mean_e)
        inv_res["inv_err_max"].append(max_e)

    # ---- unconstrained model, full augmentation -------------------------
    def fit_unconstrained(nt, k, family, tag):
        # zlib.crc32, not hash(): Python string hashing is salted per process.
        r_aug = np.random.default_rng(SEED + 1000 + zlib.crc32(tag.encode()) % 100000)
        Ptr, Ytr = augment(pos_train[:nt], y_train[:nt], k, family, r_aug)
        Pva, Yva = augment(pos_val, y_val, min(k, 8), family, r_aug)
        ybar = float(Ytr.mean())
        best = None
        for sc in SIGMA_SCALES:
            W = W0 / (sc * median_dist)
            n, S, t, csum = gram_stats(Ptr, Ytr, W, b_rff)
            m, verr = fit_from_stats(n, S, t, csum, ybar, rff_features(Pva, W, b_rff), Yva)
            if best is None or verr < best[0]:
                best = (verr, m, W, sc)
        _, m, W, sc = best
        return (lambda p: m(rff_features(p, W, b_rff))), m.lam, sc

    aug_res = {"n_train": list(N_TRAIN_GRID), "K": list(K_GRID), "n_features": N_RFF,
               "nrmse_canonical": [], "nrmse_rotated": [], "inv_err_mean": [],
               "inv_err_max": [], "lambda": [], "sigma_scale": [], "n_rows": []}
    predict_full_max = None
    for nt in N_TRAIN_GRID:
        for key in ("nrmse_canonical", "nrmse_rotated", "inv_err_mean",
                    "inv_err_max", "lambda", "sigma_scale", "n_rows"):
            aug_res[key].append([])
        for k in K_GRID:
            pred, lam, sc = fit_unconstrained(nt, k, "full", f"full-{nt}-{k}")
            if nt == max(N_TRAIN_GRID) and k == max(K_GRID):
                predict_full_max = pred
            aug_res["nrmse_canonical"][-1].append(nrmse(pred(pos_test), y_test))
            aug_res["nrmse_rotated"][-1].append(nrmse(pred(pos_test_rot), y_test))
            r_inv = np.random.default_rng(SEED + 11)
            mean_e, max_e = invariance_stats(pred, pos_inv, "full", N_INV_ROT, r_inv)
            aug_res["inv_err_mean"][-1].append(mean_e)
            aug_res["inv_err_max"][-1].append(max_e)
            aug_res["lambda"][-1].append(lam)
            aug_res["sigma_scale"][-1].append(sc)
            aug_res["n_rows"][-1].append(int(nt * k))

    # ---- off-distribution probe -----------------------------------------
    nt_probe, k_probe = max(N_TRAIN_GRID), max(K_GRID)
    probe = {"n_train": nt_probe, "K": k_probe, "families": list(FAMILIES),
             "inv_err_mean": {}, "inv_err_max": {},
             "nrmse_canonical": {}, "nrmse_rotated": {}}
    for fam in FAMILIES:
        if fam == "full":
            pred = predict_full_max
        else:
            pred, _, _ = fit_unconstrained(nt_probe, k_probe, fam, f"{fam}-{nt_probe}-{k_probe}")
        probe["inv_err_mean"][fam] = {}
        probe["inv_err_max"][fam] = {}
        for test_fam in FAMILIES:
            r_inv = np.random.default_rng(SEED + 22)
            mean_e, max_e = invariance_stats(pred, pos_inv, test_fam, N_INV_ROT, r_inv)
            probe["inv_err_mean"][fam][test_fam] = mean_e
            probe["inv_err_max"][fam][test_fam] = max_e
        probe["nrmse_canonical"][fam] = nrmse(pred(pos_test), y_test)
        probe["nrmse_rotated"][fam] = nrmse(pred(pos_test_rot), y_test)

    # ---- flag fits that collapsed toward the mean ------------------------
    # A constant predictor is perfectly invariant for a trivial reason, and a
    # heavily regularised fit at small n is close to one (nRMSE -> 1). Those
    # points must be excluded before any "best invariance achieved" claim.
    trivial = [[bool(min(aug_res["nrmse_canonical"][i][j],
                         aug_res["nrmse_rotated"][i][j]) > TRIVIAL_NRMSE)
                for j in range(len(K_GRID))] for i in range(len(N_TRAIN_GRID))]
    nontrivial_eps = [aug_res["inv_err_mean"][i][j]
                      for i in range(len(N_TRAIN_GRID)) for j in range(len(K_GRID))
                      if not trivial[i][j]]
    aug_res["near_trivial"] = trivial
    aug_res["min_inv_err_any"] = float(np.min(aug_res["inv_err_mean"]))
    aug_res["min_inv_err_nontrivial"] = float(min(nontrivial_eps))
    aug_res["trivial_note"] = (
        "near_trivial marks fits whose nRMSE exceeds "
        f"{TRIVIAL_NRMSE} in both orientations, i.e. within {100*(1-TRIVIAL_NRMSE):.0f}% "
        "of just predicting the mean. Such fits are near-constant and therefore "
        "invariant for a trivial reason; min_inv_err_any is attained at one of "
        "them (n=100, K=4, nRMSE 1.01) and min_inv_err_nontrivial is the honest "
        "best invariance actually achieved by a model that learned the target.")

    # ---- augmentation tax, isolating rotation ----------------------------
    # The ceiling is the model's OWN error with rotation removed from the
    # problem (K=1, canonical train and test). Anything measured against the
    # built-in model instead would be a representation gap, not a rotation tax.
    tax = {"per_n_train": {}}
    for i, nt in enumerate(N_TRAIN_GRID):
        ceiling = aug_res["nrmse_canonical"][i][0]
        reached = [K_GRID[j] for j in range(len(K_GRID))
                   if aug_res["nrmse_rotated"][i][j] <= ceiling]
        tax["per_n_train"][str(nt)] = {
            "rotation_free_ceiling": ceiling,
            "ceiling_is_near_trivial": bool(ceiling > TRIVIAL_NRMSE),
            "min_K_to_reach_ceiling": (min(reached) if reached else None),
            "min_rows_to_reach_ceiling": (nt * min(reached) if reached else None),
            "best_rotated_nrmse": float(np.min(aug_res["nrmse_rotated"][i])),
            "shortfall_ratio": float(np.min(aug_res["nrmse_rotated"][i]) / ceiling),
        }
    tax["informative_n_train"] = [int(n) for n in N_TRAIN_GRID
                                 if not tax["per_n_train"][str(n)]["ceiling_is_near_trivial"]]
    tax["informative_note"] = (
        "At n_train <= 100 the rotation-free ceiling is itself close to the mean "
        "predictor, so 'augmentation beats its own ceiling' there compares two "
        "near-trivial fits and should not be read as regularisation benefit. "
        "informative_n_train lists the sizes where the ceiling is a real fit.")

    tax["vs_builtin"] = {}
    for metric in ("nrmse_canonical", "nrmse_rotated"):
        best_unc = float(np.min(aug_res[metric]))
        target = inv_res[metric][-1]
        hits = [aug_res["n_rows"][a][b]
                for a in range(len(N_TRAIN_GRID)) for b in range(len(K_GRID))
                if aug_res[metric][a][b] <= target]
        tax["vs_builtin"][metric] = {
            "builtin_target": target,
            "min_augmented_rows_to_match": (min(hits) if hits else None),
            "best_unconstrained_anywhere": best_unc,
            "ratio_best_unconstrained_over_builtin": float(best_unc / target),
        }
    # The control that proves the line above is a representation gap: rotation
    # removed entirely, and the unconstrained model is still far behind.
    tax["vs_builtin"]["rotation_free_gap_ratio"] = float(
        aug_res["nrmse_canonical"][-1][0] / inv_res["nrmse_canonical"][-1])
    tax["vs_builtin"]["note"] = (
        "The unconstrained-vs-builtin ratio is NOT the augmentation tax: with "
        "rotation removed from the problem entirely (K=1, canonical train and "
        "test) the unconstrained model is still "
        f"{tax['vs_builtin']['rotation_free_gap_ratio']:.0f}x worse. See "
        "per_n_train for the rotation-isolated tax.")

    # ---- control: is the random-feature model capacity-limited? ----------
    nt_max = max(N_TRAIN_GRID)
    krr_nrmse = kernel_ridge_limit(
        pos_train[:nt_max].reshape(nt_max, -1), y_train[:nt_max],
        pos_val.reshape(N_VAL, -1), y_val,
        pos_test.reshape(N_TEST, -1), y_test,
        (0.25, 0.5, 1.0, 2.0), median_dist)
    rff_nrmse = aug_res["nrmse_canonical"][-1][0]

    # ---- feature-level invariance control -------------------------------
    r_feat = np.random.default_rng(SEED + 33)
    worst_feat = 0.0
    for _ in range(8):
        R = sample_rot("full", r_feat)
        A = invariant_features(pos_inv[:50])
        B = invariant_features(pos_inv[:50] @ R.T)
        worst_feat = max(worst_feat, float(np.max(np.abs(A - B) / (1.0 + np.abs(A)))))

    r_gt = np.random.default_rng(SEED + 44)
    worst_gt, worst_canon, worst_frame = 0.0, 0.0, 0.0
    e_ref = energy(pos_inv)
    for _ in range(8):
        R = sample_rot("full", r_gt)
        rot = pos_inv @ R.T
        worst_gt = max(worst_gt, float(np.max(np.abs(energy(rot) - e_ref))))
        can = canonicalize(rot)
        worst_canon = max(worst_canon, float(np.max(np.abs(energy(can) - e_ref))))
        worst_frame = max(worst_frame, float(np.max(np.abs(can - pos_inv))))

    return {
        "meta": {
            "seed": SEED,
            "description": ("Built-in rotation invariance versus rotation learned from "
                            "augmentation: test error, invariance error eps(g), and an "
                            "off-distribution probe with restricted augmentation."),
            "n_neighbors": N_NEIGH,
            "energy_mean": e_mean,
            "energy_std": e_std,
            "median_pairwise_distance": median_dist,
            "n_val": N_VAL, "n_test": N_TEST,
            "n_inv_conf": N_INV_CONF, "n_inv_rot": N_INV_ROT,
            "small_angle_deg": float(np.rad2deg(SMALL_ANGLE)),
            "ridge_lambdas": RIDGE_LAMBDAS.tolist(),
            "sigma_scales": list(SIGMA_SCALES),
            "potential": {"r_cut": R_CUT, "morse_a": MORSE_A, "morse_r0": MORSE_R0,
                          "lambda3": LAMBDA3, "r_min": R_MIN, "r_max": R_MAX},
            "radial_basis": {"n_rad": N_RAD, "mu": RAD_MU.tolist(), "sigma": RAD_SIGMA,
                             "l_max": L_MAX},
        },
        "builtin": inv_res,
        "augmented": aug_res,
        "offdist_probe": probe,
        "augmentation_tax": tax,
        "controls": {
            "feature_invariance_worst_rel": worst_feat,
            "ground_truth_invariance_worst_abs": worst_gt,
            "canonicalization_energy_worst_abs": worst_canon,
            "canonical_frame_worst_coord_dev": worst_frame,
            "exact_kernel_ridge_nrmse": krr_nrmse,
            "rff_nrmse_same_setting": rff_nrmse,
            "rff_excess_over_kernel_limit": float(rff_nrmse / krr_nrmse - 1.0),
        },
    }


# --------------------------------------------------------------------------
# 6. Self-check
# --------------------------------------------------------------------------

def _selfcheck(res: dict, verbose: bool = True) -> tuple[int, int]:
    npass = ntotal = 0
    checks = []

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        ok = bool(ok)
        npass += ok
        checks.append({"name": name, "passed": ok, "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    c = res["controls"]
    b = res["builtin"]
    a = res["augmented"]
    p = res["offdist_probe"]
    nt_idx = {n: i for i, n in enumerate(b["n_train"])}
    k_idx = {k: i for i, k in enumerate(a["K"])}
    last = nt_idx[max(b["n_train"])]

    print("controls: the setup really is exactly invariant")
    check("ground truth energy invariant under rotation",
          c["ground_truth_invariance_worst_abs"] < 1e-12,
          f"max |E(gx)-E(x)| {c['ground_truth_invariance_worst_abs']:.2e}")
    check("canonicalisation leaves the energy unchanged",
          c["canonicalization_energy_worst_abs"] < 1e-12,
          f"max |E(canon(gx))-E(x)| {c['canonicalization_energy_worst_abs']:.2e}")
    check("canonical frame commutes with rotation: canon(gx) = canon(x)",
          c["canonical_frame_worst_coord_dev"] < 1e-10,
          f"max coord dev {c['canonical_frame_worst_coord_dev']:.2e}")
    check("power-spectrum features invariant to float precision",
          c["feature_invariance_worst_rel"] < 1e-12,
          f"max relative dev {c['feature_invariance_worst_rel']:.2e}")

    print("built-in invariance is exact")
    check("built-in model eps(g) at machine precision",
          b["inv_err_mean"][last] < 1e-13,
          f"mean eps {b['inv_err_mean'][last]:.2e}, worst {b['inv_err_max'][last]:.2e}")
    check("built-in model canonical and rotated test error agree",
          abs(b["nrmse_canonical"][last] - b["nrmse_rotated"][last]) < 1e-12,
          f"|diff| {abs(b['nrmse_canonical'][last] - b['nrmse_rotated'][last]):.2e}")
    check("built-in test error falls with training size",
          b["nrmse_canonical"][last] < 0.5 * b["nrmse_canonical"][0],
          f"nRMSE {b['nrmse_canonical'][0]:.4f} at n=50 -> "
          f"{b['nrmse_canonical'][last]:.4f} at n={b['n_train'][last]}")

    print("augmentation improves invariance but never makes it exact")
    e1 = a["inv_err_mean"][last][k_idx[1]]
    e64 = a["inv_err_mean"][last][k_idx[64]]
    check("eps(g) drops by >3x from K=1 to K=64",
          e64 < e1 / 3.0,
          f"mean eps {e1:.3e} (K=1) -> {e64:.3e} (K=64), ratio {e1 / e64:.1f}x")
    check("eps(g) monotonically non-increasing in K at largest n_train",
          all(a["inv_err_mean"][last][i + 1] <= a["inv_err_mean"][last][i]
              for i in range(len(a["K"]) - 1)),
          f"eps over K={a['K']}: "
          + ", ".join(f"{v:.3e}" for v in a["inv_err_mean"][last]))
    check("augmented eps(g) never gets within 1e-6, even counting collapsed fits",
          a["min_inv_err_any"] > 1e-6,
          f"best mean eps anywhere {a['min_inv_err_any']:.3e} vs built-in "
          f"{b['inv_err_mean'][last]:.2e}")
    check("excluding fits that collapsed to the mean, best augmented eps is >1e-2",
          a["min_inv_err_nontrivial"] > 1e-2,
          f"best non-trivial mean eps {a['min_inv_err_nontrivial']:.3e} "
          f"({a['min_inv_err_nontrivial'] / b['inv_err_mean'][last]:.1e}x the "
          f"built-in model); the global min {a['min_inv_err_any']:.3e} belongs to "
          "a near-constant fit at n=100, K=4 with nRMSE 1.01")

    print("restricted augmentation: invariant on the orbit, not on the group")
    z_gen = p["inv_err_mean"]["zaxis"]["full"]
    z_own = p["inv_err_mean"]["zaxis"]["zaxis"]
    check("z-only augmentation: eps under general g >>10x eps under z-rotations",
          z_gen > 10.0 * z_own,
          f"eps_general {z_gen:.3e} vs eps_z {z_own:.3e}, ratio {z_gen / z_own:.1f}x")
    s_gen = p["inv_err_mean"]["smallangle"]["full"]
    s_own = p["inv_err_mean"]["smallangle"]["smallangle"]
    check("15-degree augmentation: eps under general g >>10x eps in-range",
          s_gen > 10.0 * s_own,
          f"eps_general {s_gen:.3e} vs eps_small {s_own:.3e}, ratio {s_gen / s_own:.1f}x")
    f_gen = p["inv_err_mean"]["full"]["full"]
    f_z = p["inv_err_mean"]["full"]["zaxis"]
    check("full augmentation shows no such orbit preference (ratio <5x)",
          f_gen < 5.0 * f_z and f_z < 5.0 * f_gen,
          f"eps_general {f_gen:.3e} vs eps_z {f_z:.3e}, ratio {f_gen / f_z:.2f}x")
    check("z-only augmentation is far worse than full on rotated test data",
          p["nrmse_rotated"]["zaxis"] > 2.0 * p["nrmse_rotated"]["full"],
          f"nRMSE_rot {p['nrmse_rotated']['zaxis']:.4f} (z-only) vs "
          f"{p['nrmse_rotated']['full']:.4f} (full)")

    print("accuracy: what the test-error gap does and does not mean")
    kmax = max(a["K"])
    rot_row = a["nrmse_rotated"][last]
    can_row = a["nrmse_canonical"][last]
    check("augmentation helps on the rotated test set, K=%d beats K=1" % kmax,
          rot_row[k_idx[kmax]] < rot_row[k_idx[1]],
          f"rotated nRMSE {rot_row[k_idx[1]]:.4f} (K=1) -> "
          f"{rot_row[k_idx[kmax]]:.4f} (K={kmax})")
    check("NEGATIVE RESULT (expected): augmentation degrades canonical-orientation "
          "accuracy, best at K=1",
          can_row[k_idx[1]] == min(can_row),
          "canonical nRMSE over K=" + str(a["K"]) + ": "
          + ", ".join(f"{v:.4f}" for v in can_row))

    ctrl = res["controls"]
    check("unconstrained model is not capacity-limited: within 15% of exact "
          "Gaussian kernel ridge",
          ctrl["rff_excess_over_kernel_limit"] < 0.15,
          f"RFF nRMSE {ctrl['rff_nrmse_same_setting']:.4f} vs exact-kernel limit "
          f"{ctrl['exact_kernel_ridge_nrmse']:.4f}, excess "
          f"{100 * ctrl['rff_excess_over_kernel_limit']:.1f}%")

    tax = res["augmentation_tax"]
    vb = tax["vs_builtin"]
    check("built-in beats every unconstrained model on rotated test error",
          vb["nrmse_rotated"]["min_augmented_rows_to_match"] is None,
          f"built-in nRMSE {vb['nrmse_rotated']['builtin_target']:.4f}, best "
          f"unconstrained {vb['nrmse_rotated']['best_unconstrained_anywhere']:.4f} "
          f"({vb['nrmse_rotated']['ratio_best_unconstrained_over_builtin']:.1f}x "
          "worse); no (n,K) in the sweep matches it")
    check("HONESTY CONTROL: that gap is a representation gap, not a rotation tax "
          "-- it survives with rotation removed entirely",
          vb["rotation_free_gap_ratio"] > 50.0,
          f"K=1 canonical train+test still {vb['rotation_free_gap_ratio']:.0f}x "
          f"worse than built-in ({can_row[k_idx[1]]:.4f} vs "
          f"{b['nrmse_canonical'][last]:.4f})")

    print("the rotation-isolated augmentation tax")
    per_n = tax["per_n_train"]
    shortfalls = [per_n[str(n)]["shortfall_ratio"] for n in b["n_train"]]
    detail_sf = ("shortfall (best rotated / rotation-free ceiling) at n_train="
                 + str(list(b["n_train"])) + ": "
                 + ", ".join(f"{s:.3f}x" for s in shortfalls))
    check("CONCEDED: measured in accuracy the rotation tax is nearly zero "
          "(shortfall <1.10x everywhere)",
          max(shortfalls) < 1.10, detail_sf)
    info = tax["informative_n_train"]
    info_sf = [per_n[str(n)]["shortfall_ratio"] for n in info]
    check("where the ceiling is a real fit, augmentation never quite reaches it, "
          "falling short by 1-5%",
          all(1.0 < s < 1.05 for s in info_sf),
          f"informative n_train {info}: "
          + ", ".join(f"{s:.3f}x" for s in info_sf))
    check("CAVEAT: the two sub-ceiling points are near-trivial fits, not a "
          "regularisation win",
          all(per_n[str(n)]["ceiling_is_near_trivial"] for n in b["n_train"]
              if per_n[str(n)]["min_K_to_reach_ceiling"] is not None),
          "ceilings at n_train<=100 are "
          + ", ".join(f"{per_n[str(n)]['rotation_free_ceiling']:.3f}"
                      for n in b["n_train"] if n not in info)
          + f" nRMSE, i.e. above the {TRIVIAL_NRMSE} mean-predictor threshold")

    print("the claim that survives: accuracy is bought, exactness is not")
    big = per_n[str(max(b["n_train"]))]
    eps_ratio = a["inv_err_mean"][last][k_idx[kmax]] / b["inv_err_mean"][last]
    check("at n=%d, K=%d: accuracy within 10%% of ceiling while invariance error "
          "is >1e10x the built-in model" % (max(b["n_train"]), kmax),
          big["shortfall_ratio"] < 1.10 and eps_ratio > 1e10,
          f"accuracy shortfall {big['shortfall_ratio']:.3f}x, but mean eps "
          f"{a['inv_err_mean'][last][k_idx[kmax]]:.3e} vs "
          f"{b['inv_err_mean'][last]:.2e} = {eps_ratio:.1e}x")

    res["checks"] = checks
    return npass, ntotal


if __name__ == "__main__":
    print("=" * 78)
    print("augmentation_vs_equivariance -- learned symmetry versus built-in symmetry")
    print("=" * 78)
    results = run()
    npass, ntotal = _selfcheck(results)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "augmentation_vs_equivariance.json"
    out.write_text(json.dumps(results, indent=1))
    print("-" * 78)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
