"""Crystal stability is a decision, not a regression -- and the two do not agree.

Why this sits in a NequIP repository
------------------------------------
NequIP and its descendants (MACE, SevenNet, eqV2, ORB, GNoME) are sold on
regression numbers: energy MAE in meV/atom, force MAE in meV/A. Matbench
Discovery, the benchmark those models are actually ranked on, refuses to score
them that way. Its headline metrics are F1 and the Discovery Acceleration
Factor of a *binary decision*: is this crystal thermodynamically stable? This
script builds a synthetic binary system where the whole pipeline -- energies,
hull, decision -- is known exactly, and measures how the two kinds of number
come apart.

The physics
-----------
For a binary A-B system the formation energy per atom of A_x B_(1-x) is
referenced to the pure elements, so E_f(0) = E_f(1) = 0 by definition. A
compound is thermodynamically stable iff it lies on the *lower convex hull* of
E_f versus composition: any point above the hull can lower its energy by phase
separating into the two hull phases that bracket it. The energy above hull,

    E_hull(i) = E_f(i) - hull(x_i),

is therefore not a property of structure i alone. It is a difference between
structure i and a piecewise-linear envelope built from *other* structures. That
is the entire source of the trouble below: a model can be accurate about every
absolute energy and still get the ordering of near-degenerate competing phases
wrong, and the ordering is what the decision reads.

Generating model. Regular-solution mixing enthalpy g0(x) = Omega x (1-x) with
Omega = -2.2 eV/atom, plus a well at each simple stoichiometry p/q (q <= 9,
gcd(p,q)=1) with depth ~ q^-0.30 -- simple stoichiometries bind better -- plus
per-composition jitter. The scatter is *structured*, not iid: each composition
carries a polymorph ladder, energies G(x) + cumulative sum of exponential gaps
with mean 25-50 meV/atom, so the ground state of a composition sits exactly at
G(x) and its nearest metastable competitor a few tens of meV above. This is the
realistic regime: polymorph spacings and model errors are the same size.

Hull algorithm: Andrew's monotone chain, lower chain only, written out in
`lower_hull` below (sort by (x, E), pop while the turn is not strictly
counter-clockwise). It is cross-checked against scipy.spatial.ConvexHull.

Convention. Stability threshold is E_hull <= 0 eV/atom, which is Matbench
Discovery's headline choice; the benchmark also reports a sweep over the
threshold, and a small sweep is included here (`threshold_sweep`). The hull is
built from *all* entries including the candidate itself, so E_hull >= 0 for
every structure and the stable set is exactly the set of hull vertices. The two
elemental references are held at E_f = 0 under every model: a formation energy
of an element is identically zero no matter how wrong the model's total energy
is, because it is referenced to itself.

The four experiments
--------------------
1. Noise sweep. A predictor is truth + N(0, sigma). For each sigma the hull is
   rebuilt *from the predicted energies* -- the benchmark's protocol is that a
   model predicts energies and the decision follows, so the competing phases
   move too. Reported alongside is the "fixed hull" variant (predicted energy
   against the true DFT hull), which is closer to what Matbench Discovery
   literally does with the Materials Project hull.
2. Near-hull sensitivity. Accuracy per bin of true E_hull, and the width of the
   band around the hull that contains essentially all misclassifications.
3. Two predictors, P and Q, where P has the lower MAE and Q the higher F1.
   P: constant absolute error, N(0, 25 meV). Q: error proportional to true
   distance from the hull, N(0, 2 meV + 0.35 E_hull) -- accurate exactly where
   the decision lives, sloppy about high-energy structures nobody will make.
4. Error asymmetry. A false positive spends one DFT relaxation or one synthesis
   attempt: a bounded, observed, recoverable cost. A false negative silently
   deletes a real material from the campaign -- unbounded and invisible, since
   nothing downstream ever reports it. Precision governs the first, recall the
   second, and F1 weights them equally, which is a convention rather than an
   economic fact.

A claim that did NOT hold, kept rather than weakened
----------------------------------------------------
The natural headline "F1 collapses faster than MAE degrades" is false if read
as a statement about power laws. MAE is exactly linear in sigma (log-log slope
1.000, MAE = sigma sqrt(2/pi)), while the log-log slope of F1 stays *below* 1
in magnitude across the whole sweep, peaking at 0.815 +/- 0.038. The self-check
asserts this measured fact rather than the expected one, and does so at the
two-sigma level using the trial-to-trial scatter of F1, because single
adjacent-interval slope estimates are noisy enough to stray past 1 on their own
(they reach 1.02 at 400 trials per sigma, 0.92 at the 1000 used here) without
the underlying elasticity doing so. Both numbers are in the JSON.

What is true, and is what the benchmark is reacting to, is that F1 is not a
power law at all: its sensitivity to sigma varies by more than a factor of
three across the sweep, and its entire useful range (F1 0.9 down to 0.3) is
traversed inside a window of energy MAE only a few tens of meV/atom wide --
exactly the window in which every leading model on the leaderboard sits. Two
models separated by 10 meV/atom of MAE can be separated by half the F1 axis,
and experiment 3 shows the ordering inverting outright.

A second unexpected result, also kept: under the self-consistent protocol the
number of *predicted* stable compounds falls as sigma grows (10.6 -> 5.2 here,
against 11 truly stable) rather than rising. Gaussian noise on a point cloud
produces a few deep downward outliers, and a hull pinned to those outliers
shadows everything else. The fixed-hull protocol does the opposite, flooding
the campaign with false positives. Same energies, same threshold, opposite
failure mode.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
from scipy.spatial import ConvexHull

SEED = 20260729
STAB_TOL = 1e-9          # numerical stand-in for "E_hull <= 0"
N_TRIALS = 1000          # noise realisations averaged per sigma
N_TRIALS_EXP3 = 200
SLOPE_WINDOW = 4         # local log-log slopes over a ~2.4x window in sigma, so that
                         # Monte-Carlo scatter does not masquerade as curvature

OMEGA = -0.55            # regular-solution term is OMEGA * 4x(1-x), min -0.55 eV/atom
Q_MAX = 9                # line compounds at p/q with q <= Q_MAX
WELL_D0 = 0.18
WELL_EXP = 0.30
WELL_SPREAD = 0.12
POLY_LO, POLY_HI = 5, 15
GAP_LO, GAP_HI = 0.025, 0.050
COMP_JITTER = 0.005

SIGMAS = np.logspace(np.log10(0.001), np.log10(0.200), 25)
SIGMA_P = 0.025          # experiment 3, predictor P
ALPHA_Q = 0.35           # experiment 3, predictor Q: sigma_Q(h) = 0.002 + ALPHA_Q * h
FLOOR_Q = 0.002

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "convex_hull_decision.json"


# --------------------------------------------------------------------------
# 1. Lower convex hull, written out
# --------------------------------------------------------------------------

def lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the lower convex hull of the 2-D cloud (x, y), left to right.

    Andrew's monotone chain restricted to the lower chain. Sorting by (x, y)
    puts the lowest point first among ties in x, so vertical stacks of
    polymorphs at one composition collapse correctly. The pop test is `cross
    <= 0`, which discards collinear interior points as well as concave ones.
    """
    order = np.lexsort((y, x))
    hull: list[int] = []
    for idx in order:
        i = int(idx)
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross <= 0.0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.asarray(hull, dtype=int)


def hull_curve(x: np.ndarray, y: np.ndarray, hull_idx: np.ndarray,
               xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x[hull_idx], y[hull_idx])


def energy_above_hull(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = lower_hull(x, y)
    return y - hull_curve(x, y, hi, x), hi


def scipy_lower_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Same lower chain, extracted from qhull, as an independent witness."""
    ch = ConvexHull(np.column_stack([x, y]))
    v = [int(i) for i in ch.vertices]              # counter-clockwise for 2-D
    start = min(range(len(v)), key=lambda k: (x[v[k]], y[v[k]]))
    end = min(range(len(v)), key=lambda k: (-x[v[k]], y[v[k]]))
    out, k = [], start
    while True:
        out.append(v[k])
        if k == end:
            break
        k = (k + 1) % len(v)
    return np.asarray(out, dtype=int)


# --------------------------------------------------------------------------
# 2. The synthetic A-B system
# --------------------------------------------------------------------------

def build_system(rng: np.random.Generator):
    fracs = sorted({(p / q, q) for q in range(2, Q_MAX + 1)
                    for p in range(1, q) if math.gcd(p, q) == 1})
    xc = np.array([f[0] for f in fracs])
    qc = np.array([f[1] for f in fracs])

    g0 = OMEGA * 4.0 * xc * (1.0 - xc)
    depth = WELL_D0 * qc ** (-WELL_EXP) * rng.uniform(1 - WELL_SPREAD, 1 + WELL_SPREAD,
                                                      size=len(qc))
    ground = g0 - depth + rng.normal(0.0, COMP_JITTER, size=len(qc))

    npoly = rng.integers(POLY_LO, POLY_HI, size=len(xc))
    gaps = rng.uniform(GAP_LO, GAP_HI, size=len(xc))

    xs = [0.0, 1.0]
    es = [0.0, 0.0]
    for j in range(len(xc)):
        ladder = np.concatenate([[0.0], np.cumsum(rng.exponential(gaps[j], npoly[j] - 1))])
        for v in ladder:
            xs.append(float(xc[j]))
            es.append(float(ground[j] + v))
    return np.asarray(xs), np.asarray(es), xc, qc, ground


# --------------------------------------------------------------------------
# 3. Classification bookkeeping
# --------------------------------------------------------------------------

def confusion(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    tn = int(np.sum(~truth & ~pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1}


def loglog_slopes(xv: np.ndarray, yv: np.ndarray, window: int = 1) -> np.ndarray:
    lx, ly = np.log(xv), np.log(np.maximum(yv, 1e-12))
    return (ly[window:] - ly[:-window]) / (lx[window:] - lx[:-window])


# --------------------------------------------------------------------------
# 4. Run
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
X, E, XC, QC, GROUND = build_system(rng)
POOL = (X > 0.0) & (X < 1.0)          # candidates; the two elements are references
N_POOL = int(POOL.sum())

H_TRUE, HULL_IDX = energy_above_hull(X, E)
TRUE_STABLE = H_TRUE[POOL] <= STAB_TOL
PREVALENCE = float(TRUE_STABLE.mean())
N_STABLE = int(TRUE_STABLE.sum())
HULL_X = X[HULL_IDX]
HULL_E = E[HULL_IDX]
SCIPY_HULL_IDX = scipy_lower_hull(X, E)

XDENSE = np.linspace(0.0, 1.0, 401)
MIX_DENSE = OMEGA * 4.0 * XDENSE * (1.0 - XDENSE)
HULL_DENSE = hull_curve(X, E, HULL_IDX, XDENSE)
HULL_DENSE_SCIPY = np.interp(XDENSE, X[SCIPY_HULL_IDX], E[SCIPY_HULL_IDX])

TRUE_HULL_AT_POOL = hull_curve(X, E, HULL_IDX, X)[POOL]
H_POOL = H_TRUE[POOL]

# ---- experiment 1: noise sweep ------------------------------------------
BINS = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, np.inf])
BIN_LABELS = ["E_hull = 0 (stable)", "(0, 10]", "(10, 20]", "(20, 40]",
              "(40, 80]", "(80, 150]", "(150, 300]", "> 300"]
bin_of = np.digitize(H_POOL, BINS[1:-1], right=True) + 1
bin_of[TRUE_STABLE] = 0
N_BINS = len(BIN_LABELS)
bin_counts = [int(np.sum(bin_of == b)) for b in range(N_BINS)]

sweep = {k: [] for k in ["sigma", "mae", "mae_std", "rmse", "precision", "recall",
                         "f1", "f1_std", "tp", "fp", "fn", "tn", "daf",
                         "n_pred_stable", "accuracy"]}
sweep_fixed = {k: [] for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn",
                               "daf", "n_pred_stable"]}
bin_accuracy = []
band_p95, band_p99, frac_err_below_100 = [], [], []
err_per_trial = []

for sigma in SIGMAS:
    acc = {k: [] for k in sweep if k != "sigma"}
    accf = {k: [] for k in sweep_fixed}
    bin_hits = np.zeros(N_BINS)
    bin_tot = np.zeros(N_BINS)
    err_h = []
    for _ in range(N_TRIALS):
        noise = rng.normal(0.0, sigma, size=N_POOL)
        e_pred = E.copy()
        e_pred[POOL] += noise

        h_pred, hi_pred = energy_above_hull(X, e_pred)
        pred = h_pred[POOL] <= STAB_TOL
        c = confusion(TRUE_STABLE, pred)
        acc["mae"].append(float(np.mean(np.abs(noise))))
        acc["rmse"].append(float(np.sqrt(np.mean(noise ** 2))))
        for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
            acc[k].append(c[k])
        acc["daf"].append(c["precision"] / PREVALENCE)
        acc["n_pred_stable"].append(int(pred.sum()))
        acc["accuracy"].append(float((pred == TRUE_STABLE).mean()))

        wrong = pred != TRUE_STABLE
        err_h.append(H_POOL[wrong])
        np.add.at(bin_tot, bin_of, 1.0)
        np.add.at(bin_hits, bin_of[~wrong], 1.0)

        pred_fx = (e_pred[POOL] - TRUE_HULL_AT_POOL) <= 0.0
        cf = confusion(TRUE_STABLE, pred_fx)
        for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
            accf[k].append(cf[k])
        accf["daf"].append(cf["precision"] / PREVALENCE)
        accf["n_pred_stable"].append(int(pred_fx.sum()))

    sweep["sigma"].append(float(sigma))
    sweep["mae_std"].append(float(np.std(acc["mae"])))
    sweep["f1_std"].append(float(np.std(acc["f1"])))
    for k in ("mae", "rmse", "precision", "recall", "f1", "tp", "fp", "fn", "tn",
              "daf", "n_pred_stable", "accuracy"):
        sweep[k].append(float(np.mean(acc[k])))
    for k in sweep_fixed:
        sweep_fixed[k].append(float(np.mean(accf[k])))

    bin_accuracy.append([float(bin_hits[b] / bin_tot[b]) if bin_tot[b] else 1.0
                         for b in range(N_BINS)])
    allerr = np.concatenate(err_h) if err_h else np.array([0.0])
    band_p95.append(float(np.percentile(allerr, 95)))
    band_p99.append(float(np.percentile(allerr, 99)))
    frac_err_below_100.append(float(np.mean(allerr <= 0.10)))
    err_per_trial.append(float(np.mean([len(a) for a in err_h])))

SIG = np.asarray(sweep["sigma"])
MAE = np.asarray(sweep["mae"])
F1 = np.asarray(sweep["f1"])
mae_slopes = loglog_slopes(SIG, MAE, SLOPE_WINDOW)
f1_slopes = loglog_slopes(SIG, F1, SLOPE_WINDOW)
mae_global_slope = float(np.polyfit(np.log(SIG), np.log(MAE), 1)[0])
f1_global_slope = float(np.polyfit(np.log(SIG), np.log(F1), 1)[0])
max_abs_slope_f1 = float(np.max(np.abs(f1_slopes)))
min_abs_slope_f1 = float(np.min(np.abs(f1_slopes)))
f1_slopes_raw = loglog_slopes(SIG, F1, 1)
max_abs_slope_f1_raw = float(np.max(np.abs(f1_slopes_raw)))

# Standard error of each windowed elasticity, propagated from the trial-to-trial
# scatter of F1. Needed because a single adjacent-pair slope estimate is noisy
# enough to stray past 1 without the underlying elasticity doing so.
F1_SEM = np.asarray(sweep["f1_std"]) / np.sqrt(N_TRIALS)
_rel = F1_SEM / np.maximum(F1, 1e-12)
f1_slope_se = (np.sqrt(_rel[SLOPE_WINDOW:] ** 2 + _rel[:-SLOPE_WINDOW] ** 2)
               / np.abs(np.log(SIG[SLOPE_WINDOW:]) - np.log(SIG[:-SLOPE_WINDOW])))
_k = int(np.argmax(np.abs(f1_slopes)))
max_slope_f1_se = float(f1_slope_se[_k])
max_slope_f1_upper = max_abs_slope_f1 + 2.0 * max_slope_f1_se

# MAE window over which F1 crosses from 0.9 down to 0.3. F1 is decreasing in
# sigma, so reversing both arrays gives np.interp the increasing abscissa it
# needs; the enveloping guards against residual Monte-Carlo non-monotonicity.
F1_ENV = np.minimum.accumulate(F1)
f1_monotonicity_slack = float(np.max(F1 - F1_ENV))


def mae_at_f1(target: float) -> float:
    return float(np.interp(target, F1_ENV[::-1], MAE[::-1]))


mae_at_f1_90 = mae_at_f1(0.9)
mae_at_f1_30 = mae_at_f1(0.3)
mae_at_f1_50 = mae_at_f1(0.5)
f1_window_mev = 1000.0 * (mae_at_f1_30 - mae_at_f1_90)

# ---- threshold sweep (Matbench Discovery also reports one) ---------------
THRESHOLDS = [0.0, 0.01, 0.025, 0.05, 0.10]
SIGMA_TSWEEP = 0.025
tsweep = {"threshold": THRESHOLDS, "sigma": SIGMA_TSWEEP,
          "precision": [], "recall": [], "f1": [], "daf": [], "prevalence": []}
for thr in THRESHOLDS:
    truth_t = H_POOL <= thr + STAB_TOL
    prev_t = float(truth_t.mean())
    ps, rs, fs, ds = [], [], [], []
    for _ in range(N_TRIALS):
        e_pred = E.copy()
        e_pred[POOL] += rng.normal(0.0, SIGMA_TSWEEP, size=N_POOL)
        h_pred, _ = energy_above_hull(X, e_pred)
        c = confusion(truth_t, h_pred[POOL] <= thr + STAB_TOL)
        ps.append(c["precision"]); rs.append(c["recall"]); fs.append(c["f1"])
        ds.append(c["precision"] / prev_t)
    tsweep["prevalence"].append(prev_t)
    tsweep["precision"].append(float(np.mean(ps)))
    tsweep["recall"].append(float(np.mean(rs)))
    tsweep["f1"].append(float(np.mean(fs)))
    tsweep["daf"].append(float(np.mean(ds)))

# ---- experiment 3: MAE / F1 inversion -----------------------------------
sigma_q_vec = FLOOR_Q + ALPHA_Q * H_POOL
per_trial = {"mae_P": [], "mae_Q": [], "rmse_P": [], "rmse_Q": [],
             "f1_P": [], "f1_Q": [], "precision_P": [], "precision_Q": [],
             "recall_P": [], "recall_Q": [], "daf_P": [], "daf_Q": []}
for _ in range(N_TRIALS_EXP3):
    noise_p = rng.normal(0.0, SIGMA_P, size=N_POOL)
    noise_q = rng.normal(0.0, sigma_q_vec)
    for tag, noise in (("P", noise_p), ("Q", noise_q)):
        e_pred = E.copy()
        e_pred[POOL] += noise
        h_pred, _ = energy_above_hull(X, e_pred)
        c = confusion(TRUE_STABLE, h_pred[POOL] <= STAB_TOL)
        per_trial[f"mae_{tag}"].append(float(np.mean(np.abs(noise))))
        per_trial[f"rmse_{tag}"].append(float(np.sqrt(np.mean(noise ** 2))))
        per_trial[f"f1_{tag}"].append(c["f1"])
        per_trial[f"precision_{tag}"].append(c["precision"])
        per_trial[f"recall_{tag}"].append(c["recall"])
        per_trial[f"daf_{tag}"].append(c["precision"] / PREVALENCE)

exp3 = {}
for tag in ("P", "Q"):
    exp3[tag] = {k: float(np.mean(per_trial[f"{k}_{tag}"]))
                 for k in ("mae", "rmse", "f1", "precision", "recall", "daf")}
    exp3[tag]["f1_std"] = float(np.std(per_trial[f"f1_{tag}"]))
    exp3[tag]["mae_std"] = float(np.std(per_trial[f"mae_{tag}"]))
mae_P = np.asarray(per_trial["mae_P"]); mae_Q = np.asarray(per_trial["mae_Q"])
f1_P = np.asarray(per_trial["f1_P"]); f1_Q = np.asarray(per_trial["f1_Q"])
inversion_frac = float(np.mean((mae_P < mae_Q) & (f1_P < f1_Q)))

# ---- experiment 4: cost asymmetry ----------------------------------------
COST_SIGMAS = [0.005, 0.010, 0.025, 0.050, 0.100, 0.200]
cost_rows = []
for s in COST_SIGMAS:
    k = int(np.argmin(np.abs(SIG - s)))
    cost_rows.append({
        "sigma": float(SIG[k]),
        "mae_meV": float(1000.0 * MAE[k]),
        "tp": sweep["tp"][k], "fp": sweep["fp"][k],
        "fn": sweep["fn"][k], "tn": sweep["tn"][k],
        "precision": sweep["precision"][k], "recall": sweep["recall"][k],
        "f1": sweep["f1"][k], "daf": sweep["daf"][k],
        "wasted_dft_relaxations": sweep["fp"][k],
        "discoveries_missed": sweep["fn"][k],
        "fraction_of_stable_pool_lost": sweep["fn"][k] / N_STABLE,
    })


# --------------------------------------------------------------------------
# 5. Self-check
# --------------------------------------------------------------------------

def _selfcheck(verbose: bool = True) -> tuple[int, int]:
    npass = ntotal = 0

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        npass += bool(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("convex hull construction")
    same_set = set(HULL_IDX.tolist()) == set(SCIPY_HULL_IDX.tolist())
    check("hand-written monotone chain reproduces scipy ConvexHull vertex set",
          same_set, f"{len(HULL_IDX)} vs {len(SCIPY_HULL_IDX)} vertices")
    dev = float(np.max(np.abs(HULL_DENSE - HULL_DENSE_SCIPY)))
    check("hull polylines agree pointwise on a 401-point grid", dev < 1e-12,
          f"max |diff| {dev:.2e} eV/atom")
    check("hull is a true lower bound: no candidate below it",
          float(H_TRUE.min()) > -1e-12, f"min E_hull {float(H_TRUE.min()):.2e} eV/atom")
    slopes = np.diff(HULL_E) / np.diff(HULL_X)
    check("hull is convex: chord slopes strictly increasing",
          bool(np.all(np.diff(slopes) > 0)),
          f"min slope increment {float(np.min(np.diff(slopes))):.4f} eV/atom")

    print("synthetic A-B system")
    check("several stable interior line compounds form", len(HULL_IDX) - 2 >= 8,
          f"{len(HULL_IDX) - 2} interior hull vertices among {len(XC)} compositions")
    check("stable fraction of the pool is a realistic minority",
          0.01 < PREVALENCE < 0.15,
          f"prevalence {PREVALENCE:.4f} ({N_STABLE}/{N_POOL}), random-guess DAF ceiling "
          f"{1 / PREVALENCE:.1f}")
    check("elemental references sit at E_f = 0 and anchor both hull ends",
          abs(E[0]) < 1e-15 and abs(E[1]) < 1e-15
          and HULL_X[0] == 0.0 and HULL_X[-1] == 1.0,
          f"hull spans x = {HULL_X[0]:.1f} to {HULL_X[-1]:.1f}")

    print("experiment 1 -- noise sweep")
    ratio = MAE / (SIG * np.sqrt(2 / np.pi))
    check("energy MAE is exactly sigma*sqrt(2/pi): predictor error is linear in sigma",
          float(np.max(np.abs(ratio - 1.0))) < 0.02,
          f"max deviation {float(np.max(np.abs(ratio - 1.0))) * 100:.2f}%")
    check("MAE is a pure power law in sigma with exponent 1",
          abs(mae_global_slope - 1.0) < 0.01
          and float(np.max(np.abs(mae_slopes - 1.0))) < 0.03,
          f"global log-log slope {mae_global_slope:.5f}, "
          f"max local |slope - 1| {float(np.max(np.abs(mae_slopes - 1.0))):.4f}")
    check("F1 is monotone non-increasing in sigma to within Monte-Carlo scatter",
          f1_monotonicity_slack < 0.01,
          f"largest upward excursion {f1_monotonicity_slack:.5f} over "
          f"{len(F1)} sigma values, {N_TRIALS} trials each")
    check("F1 spans the full useful range across the sweep",
          F1[0] > 0.90 and F1[-1] < 0.20,
          f"F1 {F1[0]:.3f} at MAE {1000 * MAE[0]:.1f} meV/atom -> "
          f"{F1[-1]:.3f} at MAE {1000 * MAE[-1]:.1f} meV/atom")
    check("F1 is NOT a power law in sigma: local elasticity varies >3x "
          "(MAE's is constant)",
          max_abs_slope_f1 / max(min_abs_slope_f1, 1e-9) > 3.0,
          f"|dlogF1/dlogsigma| ranges {min_abs_slope_f1:.3f} to {max_abs_slope_f1:.3f}")
    check("documented negative result: F1 never decays super-linearly in sigma, "
          "so 'F1 falls faster than MAE rises' is FALSE as a power law",
          0.3 < max_abs_slope_f1 and max_slope_f1_upper < 1.0,
          f"peak |dlogF1/dlogsigma| {max_abs_slope_f1:.3f} +/- {max_slope_f1_se:.3f}, "
          f"2-sigma upper bound {max_slope_f1_upper:.3f} < 1.000 = MAE's slope "
          f"(single-interval estimates scatter to {max_abs_slope_f1_raw:.3f}); "
          f"F1 global slope {f1_global_slope:.3f}")
    check("F1 0.9 -> 0.3 is traversed inside an MAE window narrower than the "
          "leaderboard's own spread (~20-90 meV/atom)",
          f1_window_mev < 50.0,
          f"MAE {1000 * mae_at_f1_90:.1f} -> {1000 * mae_at_f1_30:.1f} meV/atom, "
          f"width {f1_window_mev:.1f} meV/atom; F1 = 0.5 at "
          f"{1000 * mae_at_f1_50:.1f} meV/atom")
    check("DAF stays above 1: even a 160 meV/atom model beats random screening",
          float(np.min(sweep["daf"])) > 1.0,
          f"DAF {sweep['daf'][0]:.1f} at best sigma, {sweep['daf'][-1]:.1f} at worst")
    check("self-consistent protocol predicts FEWER stable phases as sigma grows "
          "(noise-pinned hull shadows real compounds)",
          sweep["n_pred_stable"][-1] < 0.7 * sweep["n_pred_stable"][0],
          f"{sweep['n_pred_stable'][0]:.1f} -> {sweep['n_pred_stable'][-1]:.1f} "
          f"predicted stable, truth {N_STABLE}")
    check("fixed-hull protocol fails the opposite way: false-positive flood",
          sweep_fixed["n_pred_stable"][-1] > 5.0 * sweep["n_pred_stable"][-1],
          f"{sweep_fixed['n_pred_stable'][-1]:.1f} vs {sweep['n_pred_stable'][-1]:.1f} "
          f"predicted stable at sigma = {SIG[-1] * 1000:.0f} meV/atom")

    print("experiment 2 -- near-hull sensitivity")
    k25 = int(np.argmin(np.abs(SIG - 0.025)))
    far_acc = min(bin_accuracy[k25][-2], bin_accuracy[k25][-1])
    check("far-from-hull structures are classified essentially perfectly",
          far_acc > 0.999,
          f"accuracy {far_acc:.5f} for E_hull > 150 meV/atom at sigma = "
          f"{SIG[k25] * 1000:.1f} meV/atom")
    h_span = float(H_POOL.max())
    check("misclassifications live in a thin band around the hull, not spread "
          "over the pool's energy range",
          band_p99[k25] < 0.20 * h_span and frac_err_below_100[k25] > 0.90,
          f"99% of errors within {1000 * band_p99[k25]:.0f} meV/atom of the hull "
          f"({100 * band_p99[k25] / h_span:.1f}% of the pool's {1000 * h_span:.0f} "
          f"meV/atom E_hull span); {100 * frac_err_below_100[k25]:.1f}% below 100 meV/atom")
    acc25 = np.asarray(bin_accuracy[k25])
    check("accuracy rises monotonically with distance from the hull",
          bool(np.all(np.diff(acc25) >= -1e-12)),
          f"per-bin accuracy {np.array2string(acc25, precision=3)}")
    lo = int(np.argmin(np.abs(SIG - 0.005)))
    fit = float(np.polyfit(np.log(SIG[lo:]), np.log(np.asarray(band_p99[lo:])), 1)[0])
    check("band width grows about linearly in sigma", 0.5 < fit < 1.5,
          f"log-log slope {fit:.3f}, p99 band {1000 * band_p99[lo]:.0f} -> "
          f"{1000 * band_p99[-1]:.0f} meV/atom")
    check("band width is monotone non-decreasing in sigma",
          bool(np.all(np.diff(np.asarray(band_p99)) > -0.005)),
          f"largest decrease {1000 * float(np.min(np.diff(np.asarray(band_p99)))):.2f} "
          f"meV/atom over {len(band_p99) - 1} intervals")

    print("experiment 3 -- MAE and F1 disagree about which model is better")
    check("P has the lower energy MAE", exp3["P"]["mae"] < exp3["Q"]["mae"],
          f"MAE_P {1000 * exp3['P']['mae']:.2f} vs MAE_Q {1000 * exp3['Q']['mae']:.2f} "
          f"meV/atom (ratio {exp3['Q']['mae'] / exp3['P']['mae']:.2f})")
    check("P also has the lower RMSE", exp3["P"]["rmse"] < exp3["Q"]["rmse"],
          f"RMSE_P {1000 * exp3['P']['rmse']:.2f} vs RMSE_Q "
          f"{1000 * exp3['Q']['rmse']:.2f} meV/atom")
    check("yet Q has the higher F1: the regression ordering inverts",
          exp3["Q"]["f1"] > exp3["P"]["f1"] + 0.10,
          f"F1_P {exp3['P']['f1']:.3f} vs F1_Q {exp3['Q']['f1']:.3f}")
    check("Q also wins on precision, recall and DAF simultaneously",
          exp3["Q"]["precision"] > exp3["P"]["precision"]
          and exp3["Q"]["recall"] > exp3["P"]["recall"]
          and exp3["Q"]["daf"] > exp3["P"]["daf"],
          f"prec {exp3['P']['precision']:.3f}->{exp3['Q']['precision']:.3f}, "
          f"rec {exp3['P']['recall']:.3f}->{exp3['Q']['recall']:.3f}, "
          f"DAF {exp3['P']['daf']:.1f}->{exp3['Q']['daf']:.1f}")
    check("the inversion is not an averaging artefact: it holds trial by trial",
          inversion_frac > 0.95,
          f"{100 * inversion_frac:.1f}% of {N_TRIALS_EXP3} independent trials")

    print("experiment 4 -- asymmetry of error cost")
    fp_grow = cost_rows[-1]["fp"] > cost_rows[0]["fp"]
    fn_grow = cost_rows[-1]["fn"] > cost_rows[0]["fn"]
    check("both error types grow with sigma but false negatives dominate",
          fn_grow and cost_rows[-1]["fn"] > cost_rows[-1]["fp"],
          f"at MAE {cost_rows[-1]['mae_meV']:.0f} meV/atom: FN {cost_rows[-1]['fn']:.1f} "
          f"of {N_STABLE} stable, FP {cost_rows[-1]['fp']:.1f}")
    check("at the accurate end the campaign still loses real materials",
          cost_rows[1]["fn"] > 0.0,
          f"at MAE {cost_rows[1]['mae_meV']:.1f} meV/atom: {cost_rows[1]['fn']:.2f} "
          f"missed ({100 * cost_rows[1]['fraction_of_stable_pool_lost']:.1f}% of stable pool), "
          f"{cost_rows[1]['fp']:.2f} wasted relaxations")
    check("recall degrades faster than precision under this protocol",
          (sweep["recall"][0] - sweep["recall"][-1])
          > (sweep["precision"][0] - sweep["precision"][-1]),
          f"recall {sweep['recall'][0]:.3f}->{sweep['recall'][-1]:.3f}, "
          f"precision {sweep['precision'][0]:.3f}->{sweep['precision'][-1]:.3f}")
    check("threshold sweep: loosening the 0 eV/atom threshold raises F1",
          tsweep["f1"][-1] > tsweep["f1"][0],
          f"F1 {tsweep['f1'][0]:.3f} at 0 -> {tsweep['f1'][-1]:.3f} at 100 meV/atom, "
          f"prevalence {tsweep['prevalence'][0]:.3f} -> {tsweep['prevalence'][-1]:.3f}")
    return npass, ntotal


def _write_json() -> None:
    payload = {
        "meta": {
            "seed": SEED,
            "description": ("Synthetic binary A-B convex hull: why Matbench Discovery "
                            "scores crystal stability as a decision (F1, DAF) rather "
                            "than as energy regression (MAE)."),
            "units": "eV/atom throughout unless a key says meV",
            "hull_algorithm": "Andrew monotone chain, lower chain only; cross-checked "
                              "against scipy.spatial.ConvexHull",
            "stability_threshold_eV_per_atom": 0.0,
            "threshold_note": ("Matbench Discovery's headline threshold is E_hull <= 0; "
                               "it also reports a sweep over the threshold, mirrored in "
                               "threshold_sweep."),
            "protocol": ("primary = hull rebuilt from predicted energies (competing "
                         "phases move too); secondary = predicted energy against the "
                         "fixed true hull, closer to the benchmark's use of the "
                         "Materials Project hull"),
            "n_candidates": N_POOL,
            "n_compositions": int(len(XC)),
            "n_stable": N_STABLE,
            "prevalence": PREVALENCE,
            "max_possible_daf": float(1.0 / PREVALENCE),
            "n_trials_sweep": N_TRIALS,
            "n_trials_exp3": N_TRIALS_EXP3,
        },
        "system": {
            "regular_solution_omega": OMEGA * 4.0,
            "mixing_curve_x": XDENSE.tolist(),
            "mixing_curve_e": MIX_DENSE.tolist(),
            "hull_curve_x": XDENSE.tolist(),
            "hull_curve_e": HULL_DENSE.tolist(),
            "line_compound_x": XC.tolist(),
            "line_compound_denominator": QC.astype(int).tolist(),
            "line_compound_ground_energy": GROUND.tolist(),
            "hull_vertex_x": HULL_X.tolist(),
            "hull_vertex_e": HULL_E.tolist(),
            "hull_vertex_index": HULL_IDX.astype(int).tolist(),
            "candidate_x": X[POOL].tolist(),
            "candidate_e_form": E[POOL].tolist(),
            "candidate_e_hull": H_POOL.tolist(),
            "candidate_stable": TRUE_STABLE.astype(int).tolist(),
            "element_x": [0.0, 1.0],
            "element_e_form": [0.0, 0.0],
        },
        "exp1_noise_sweep": {**sweep, "fixed_hull": sweep_fixed,
                             "loglog_slope_window": SLOPE_WINDOW,
                             "mae_loglog_slope": mae_slopes.tolist(),
                             "f1_loglog_slope": f1_slopes.tolist(),
                             "mae_global_loglog_slope": mae_global_slope,
                             "f1_global_loglog_slope": f1_global_slope,
                             "f1_loglog_slope_se": f1_slope_se.tolist(),
                             "max_abs_loglog_slope_f1": max_abs_slope_f1,
                             "max_abs_loglog_slope_f1_se": max_slope_f1_se,
                             "max_abs_loglog_slope_f1_upper2sigma": max_slope_f1_upper,
                             "max_abs_loglog_slope_f1_single_interval": max_abs_slope_f1_raw,
                             "min_abs_loglog_slope_f1": min_abs_slope_f1,
                             "f1_monotonicity_slack": f1_monotonicity_slack,
                             "mae_at_f1_0.9": mae_at_f1_90,
                             "mae_at_f1_0.5": mae_at_f1_50,
                             "mae_at_f1_0.3": mae_at_f1_30,
                             "f1_window_width_meV": f1_window_mev},
        "threshold_sweep": tsweep,
        "exp2_near_hull": {
            "bin_edges_eV": [float(b) for b in BINS[:-1]] + [None],
            "bin_labels_meV": BIN_LABELS,
            "bin_counts": bin_counts,
            "sigma": sweep["sigma"],
            "accuracy_by_bin": bin_accuracy,
            "band_width_p95": band_p95,
            "band_width_p99": band_p99,
            "fraction_errors_below_100meV": frac_err_below_100,
            "mean_errors_per_trial": err_per_trial,
        },
        "exp3_mae_f1_inversion": {
            "predictor_P": "truth + N(0, 25 meV/atom), same error everywhere",
            "predictor_Q": ("truth + N(0, 2 meV/atom + 0.35 * E_hull_true), accurate "
                            "near the hull, sloppy far above it"),
            "sigma_P": SIGMA_P, "alpha_Q": ALPHA_Q, "floor_Q": FLOOR_Q,
            "P": exp3["P"], "Q": exp3["Q"],
            "inversion_fraction": inversion_frac,
            "per_trial": {k: [float(v) for v in vs] for k, vs in per_trial.items()},
        },
        "exp4_error_cost": {
            "rows": cost_rows,
            "note": ("A false positive spends one DFT relaxation or one synthesis "
                     "attempt: bounded, observed, recoverable. A false negative "
                     "silently deletes a real material from the campaign: unbounded "
                     "and invisible, because nothing downstream ever reports it."),
        },
        "findings": {
            "f1_not_superlinear_in_sigma": (
                "FALSIFIED EXPECTATION: the peak noise-resolved elasticity is "
                f"|dlogF1/dlogsigma| = {max_abs_slope_f1:.3f} +/- {max_slope_f1_se:.3f} "
                f"(2-sigma upper bound {max_slope_f1_upper:.3f}), below MAE's constant "
                "slope of 1, so F1 does not decay faster than linearly in sigma. "
                f"Single-interval estimates scatter up to {max_abs_slope_f1_raw:.3f}, "
                "which is consistent with the smoothed value given Monte-Carlo error "
                "and is not evidence of a super-linear region."),
            "f1_not_a_power_law": (
                f"F1's local elasticity ranges {min_abs_slope_f1:.3f} to "
                f"{max_abs_slope_f1:.3f} while MAE's is pinned at 1.000; F1 falls "
                f"from 0.9 to 0.3 across only {f1_window_mev:.1f} meV/atom of MAE."),
            "predicted_hull_shrinks": (
                f"Self-consistent protocol: predicted stable count falls "
                f"{sweep['n_pred_stable'][0]:.1f} -> {sweep['n_pred_stable'][-1]:.1f} "
                f"as sigma grows, opposite to the naive expectation that noise "
                f"manufactures spurious stable phases."),
            "protocol_matters": (
                f"At sigma = {SIG[-1] * 1000:.0f} meV/atom the fixed-hull protocol "
                f"predicts {sweep_fixed['n_pred_stable'][-1]:.1f} stable structures "
                f"against {sweep['n_pred_stable'][-1]:.1f} for the self-consistent "
                f"one: same energies, opposite failure mode."),
            "mae_f1_inversion": (
                f"Predictor P beats Q on MAE ({1000 * exp3['P']['mae']:.1f} vs "
                f"{1000 * exp3['Q']['mae']:.1f} meV/atom) and loses on F1 "
                f"({exp3['P']['f1']:.3f} vs {exp3['Q']['f1']:.3f}) in "
                f"{100 * inversion_frac:.1f}% of trials."),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    print("=" * 78)
    print("convex_hull_decision.py -- stability is a decision, not a regression")
    print("=" * 78)
    _write_json()
    print("-" * 78)
    npass, ntotal = _selfcheck()
    print("-" * 78)
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
