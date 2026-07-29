"""Receptive field vs. locality: what message passing buys and what it costs.

NequIP is a message-passing network: after L interaction layers an atom's
prediction depends on everything within L hops of it in the cutoff graph.
Allegro is the same representation theory with L forced to 1 -- every edge is
processed independently, nothing is ever passed between atoms. The whole
argument between them is a trade this script makes numerical, on a crystal
where every quantity is exactly computable.

The lattice is rock salt: a simple cubic lattice of alternating +-1 ions with
nearest-neighbour distance d = 1. It is used twice -- as the graph whose hop
neighbourhoods we enumerate, and as the charge distribution whose lattice sums
we truncate.

What is measured
----------------
1. RECEPTIVE FIELD GROWTH.  For cutoff r_cut the L-hop neighbourhood of an atom
   has effective radius R_eff(L) = max distance from the centre.  The textbook
   statement "the receptive field grows like L * r_cut" is an *upper bound*
   (triangle inequality) and is attained only when r_cut happens to be itself a
   realizable lattice distance -- as at r_cut = 3.0 below, where c/r_cut is
   exactly 1.  What actually holds, exactly, is

        R_eff(L) = L * c(r_cut),   c(r_cut) = longest lattice vector <= r_cut,

   because a lattice vector can be chained collinearly and nothing longer can.
   The measured shortfall c/r_cut runs from 0.909 to 1.000 over the six cutoffs
   tested, and it is not monotone in r_cut -- it drops back to 0.909 at
   r_cut = 2.2, where the next shell (sqrt 5 = 2.236) has not yet been reached.
   So "reach = L * r_cut" overstates the receptive field by up to 9% here, and
   the error is a sawtooth in r_cut, not a smooth correction.  Reported as a
   deviation, not smoothed away.

2. COST OF DEPTH.  The number of atoms in the receptive field is what a
   domain-decomposed MD code must own as ghost atoms -- exactly the
   communication Allegro was designed to delete.  For r_cut in (1, sqrt 2) the
   graph is the 6-neighbour cubic graph and the L-hop count is the octahedral
   number (2L+1)(2L^2+2L+3)/3, a cubic polynomial in L with leading coefficient
   4/3 = (volume of the unit l1-ball) x (number density).  For r_cut in
   (sqrt 3, 2) the graph is the 26-neighbour graph and the count is exactly
   (2L+1)^3.  Both are verified as integer identities.  Honest caveat: at the
   depths anyone actually uses (L <= 6) the asymptotic cubic law is *not* yet
   reached -- the measured log-log exponent is 2.37 at L = 2->4 and only
   crawls to 2.80 by L = 6->12.  Surface terms still dominate; the count at
   L = 5 is 1.39x its own leading cubic term.

3. DOES LOCALITY SUFFICE?  A model with cutoff r_cut can only ever see the
   pair sum inside r_cut.  Truncating the true lattice sum at r_cut gives the
   error any strictly local model inherits before it has made a single fitting
   mistake.  Three pair potentials on the same geometry:

        exp(-kappa r)/r   short-ranged: tail ~ 4 pi rho e^{-kappa r_c}(r_c/kappa
                          + 1/kappa^2).  Fitted decay rate -0.99898 against
                          kappa = 1, prefactor 12.28 against 4 pi rho = 12.57.
                          Locality is fine.
        1/r^6             dispersion: tail ~ (4 pi rho/3) r_c^{-3}.  Fitted
                          power -3.0118, prefactor 4.238 against 4.189.  A slow
                          but convergent power law -- locality is a controlled
                          approximation whose error you can quote.
        1/r  (alternating charges)  Coulomb: the spherically truncated Madelung
                          series does not converge at all.  At r_c = 12 d the
                          error is still 117% of the exact energy, and over
                          r_c in [10, 20] the truncated sum swings across a
                          range 14.6x larger than the answer itself.  The
                          effective decay exponent is -0.19: the error envelope
                          grows.  No cutoff fixes this; it is the quantitative
                          form of the question "does locality suffice?" and the
                          answer, for charged systems, is no.

   Honest caveat on the two convergent cases: the ratio tail/prediction is not
   flat.  It oscillates about 1 with up to 9.4% (exp) and 6.1% (1/r^6) scatter,
   because the lattice adds whole coordination shells while the continuum
   formula is smooth.  The mean ratio is 1 to within 1%, and for 1/r^6 the
   scatter falls from 6.1% at r_cut in [4,10) to 2.0% beyond -- so it is
   discreteness, not a wrong law.  The checks therefore test the fitted decay
   *rate* (R^2 > 0.9998 in both cases), not pointwise agreement.

   The exact Coulomb reference is an Ewald sum written here from scratch; it
   reproduces the NaCl Madelung constant 1.747564594633182 to 0.0 at double
   precision, and is independent of the splitting parameter alpha to 7e-16,
   which is what licenses calling the truncation error an error.

4. DEPTH vs CUTOFF AT FIXED REACH.  Four (L, r_cut) pairs reaching R_eff ~ 5 d.
   Going from deep-and-narrow (L=5, r_cut=1.1) to shallow-and-wide (L=1,
   r_cut=5.0) multiplies the edge evaluations per atom by 17.1 (30 -> 514) and
   the ghost-atom halo by 2.23 (230 -> 514), while dividing the number of
   communication rounds by 5.  Deep and narrow is enormously cheaper in
   arithmetic and enormously more expensive in communication.  Held instead at
   a fixed physical cutoff (r_cut = 2.55 d), going from Allegro's L = 1 to
   NequIP's L = 4 buys 4.0x the reach for 41.9x the halo -- the cubic law of
   part 2 seen from the message-passing side.  That is why NequIP is deep, and
   why Allegro -- which cannot be deep -- has to buy expressivity per edge
   instead of reach.

5. A representation-theory aside using the repo's e3 core: every hop shell here
   is octahedrally symmetric, so sum_j Y_l(r_hat_j) over a shell vanishes
   identically for l = 1, 2, 3 (measured max norm 6.4e-15) and is first nonzero
   at l = 4 (min norm 1.94 over the five shells).  A model that only ever forms
   shell sums is blind to the anisotropy of this environment until it carries
   l = 4 features -- which is why the features live on edges, not on shells.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from e3 import real_sh, random_rotation  # noqa: E402

SEED = 3
NAME = "receptive_field"
RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results" / f"{NAME}.json"

D_NN = 1.0          # nearest-neighbour distance, sets the unit of length
N_CELLS = 12        # periodic box is N_CELLS x N_CELLS x N_CELLS sites
R_SUM_MAX = 60.0    # radius of the explicit lattice-sum enumeration
MADELUNG_REF = 1.7475645946331822


# --------------------------------------------------------------------------
# 1. Integer-lattice machinery
# --------------------------------------------------------------------------

_BASE = 1024
_SHIFT = 512


def _encode(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return ((v[..., 0] + _SHIFT) * _BASE + (v[..., 1] + _SHIFT)) * _BASE + (v[..., 2] + _SHIFT)


def _decode(c: np.ndarray) -> np.ndarray:
    z = c % _BASE - _SHIFT
    y = (c // _BASE) % _BASE - _SHIFT
    x = (c // (_BASE * _BASE)) % _BASE - _SHIFT
    return np.stack([x, y, z], axis=-1)


def _in_sorted(q: np.ndarray, pool: np.ndarray) -> np.ndarray:
    if pool.size == 0:
        return np.zeros(q.shape, dtype=bool)
    i = np.clip(np.searchsorted(pool, q), 0, pool.size - 1)
    return pool[i] == q


def lattice_offsets(r_cut: float) -> np.ndarray:
    """All nonzero integer vectors with |n| <= r_cut: the edges of one atom."""
    R = int(np.floor(r_cut + 1e-12))
    n = np.arange(-R, R + 1)
    G = np.array(np.meshgrid(n, n, n, indexing="ij")).reshape(3, -1).T
    r = np.linalg.norm(G, axis=1)
    keep = (r > 0) & (r <= r_cut + 1e-12)
    G = G[keep]
    return G[np.argsort(np.linalg.norm(G, axis=1), kind="stable")]


def hop_shells(r_cut: float, l_max: int) -> list[np.ndarray]:
    """Shell L = atoms first reachable in exactly L hops, on the infinite lattice."""
    off = lattice_offsets(r_cut)
    frontier = np.zeros((1, 3), dtype=np.int64)
    seen = _encode(frontier)
    shells = [frontier]
    for _ in range(l_max):
        cand = (frontier[:, None, :] + off[None, :, :]).reshape(-1, 3)
        codes = np.unique(_encode(cand))
        fresh = codes[~_in_sorted(codes, seen)]
        frontier = _decode(fresh)
        seen = np.union1d(seen, fresh)
        shells.append(frontier)
    return shells


def cumulative(shells: list[np.ndarray]) -> tuple[list[int], list[float]]:
    sizes, radii, acc = [], [], []
    for sh in shells:
        acc.append(sh)
        pts = np.concatenate(acc)
        sizes.append(int(len(pts)))
        radii.append(float(np.max(np.linalg.norm(pts.astype(float), axis=1))))
    return sizes, radii


# --------------------------------------------------------------------------
# 2. Periodic crystal and its minimum-image neighbour graph
# --------------------------------------------------------------------------

def build_crystal(n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(n_cells)
    G = np.array(np.meshgrid(n, n, n, indexing="ij")).reshape(3, -1).T
    charge = np.where(G.sum(axis=1) % 2 == 0, 1.0, -1.0)
    return G.astype(float) * D_NN, charge


def neighbours_minimum_image(pos: np.ndarray, box: float, r_cut: float) -> list[np.ndarray]:
    """Brute-force minimum-image neighbour list. Valid only for r_cut < box/2."""
    assert r_cut < box / 2, "minimum image convention requires r_cut < box/2"
    out = []
    for i in range(len(pos)):
        d = pos - pos[i]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        idx = np.flatnonzero((r > 1e-12) & (r <= r_cut + 1e-12))
        out.append(idx)
    return out


def hop_sets_pbc(nbrs: list[np.ndarray], centre: int, l_max: int) -> list[np.ndarray]:
    n = len(nbrs)
    seen = np.zeros(n, dtype=bool)
    seen[centre] = True
    frontier = np.array([centre])
    shells = [frontier]
    for _ in range(l_max):
        if frontier.size == 0:
            shells.append(frontier)
            continue
        cand = np.unique(np.concatenate([nbrs[j] for j in frontier]))
        fresh = cand[~seen[cand]]
        seen[fresh] = True
        frontier = fresh
        shells.append(fresh)
    return shells


# --------------------------------------------------------------------------
# 3. Ewald sum -- the exact Coulomb reference
# --------------------------------------------------------------------------

def madelung_ewald(alpha: float = 2.0, n_img: int = 6, k_max: int = 12) -> float:
    """Madelung constant of rock salt by Ewald summation, d = 1.

    Cell: cube of side 2 holding the 8 sites {0,1}^3 with charge (-1)^(i+j+k).
    Energy per ion is -M/2 (the 1/2 is the double-counting factor), so the
    constant read off is M = -2 E_cell / 8.  alpha must be irrelevant to the
    answer -- that independence is the check that the split is done right.
    """
    a = 2.0 * D_NN
    pos = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float) * D_NN
    q = np.array([(-1.0) ** int(round(p.sum() / D_NN)) for p in pos])
    V = a ** 3

    ns = np.arange(-n_img, n_img + 1)
    N = np.array(np.meshgrid(ns, ns, ns, indexing="ij")).reshape(3, -1).T * a
    e_real = 0.0
    for i in range(len(pos)):
        d = pos[i] - pos[:, None, :] + N[None, :, :]
        r = np.linalg.norm(d, axis=-1)
        ok = r > 1e-12
        rr = np.where(ok, r, 1.0)
        e_real += 0.5 * np.sum(np.where(ok, (q[i] * q[:, None]) * erfc(alpha * rr) / rr, 0.0))

    ks = np.arange(-k_max, k_max + 1)
    K = np.array(np.meshgrid(ks, ks, ks, indexing="ij")).reshape(3, -1).T * (2 * np.pi / a)
    k2 = np.sum(K ** 2, axis=1)
    K, k2 = K[k2 > 1e-12], k2[k2 > 1e-12]
    S = np.exp(1j * (K @ pos.T)) @ q
    e_recip = (2 * np.pi / V) * np.sum(np.exp(-k2 / (4 * alpha ** 2)) / k2 * np.abs(S) ** 2)

    e_self = -(alpha / np.sqrt(np.pi)) * np.sum(q ** 2)
    return float(-2.0 * (e_real + e_recip + e_self) / len(pos))


# --------------------------------------------------------------------------
# 4. Truncated lattice sums
# --------------------------------------------------------------------------

def sorted_lattice(r_max: float) -> tuple[np.ndarray, np.ndarray]:
    R = int(np.ceil(r_max))
    n = np.arange(-R, R + 1)
    G = np.array(np.meshgrid(n, n, n, indexing="ij")).reshape(3, -1).T
    r = np.linalg.norm(G, axis=1)
    keep = (r > 0) & (r <= r_max)
    G, r = G[keep], r[keep]
    o = np.argsort(r, kind="stable")
    G, r = G[o], r[o]
    sign = np.where(G.sum(axis=1) % 2 == 0, 1.0, -1.0)
    return r * D_NN, sign


def truncated(vals: np.ndarray, r: np.ndarray, grid: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(vals)])
    return c[np.searchsorted(r, grid, side="right")]


def loglin_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares y = a + b x. Returns (b, a, R^2)."""
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return float(b), float(a), float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))


def local_exponent(tail: np.ndarray, grid: np.ndarray, lo: float, hi: float) -> float:
    """Effective power-law exponent p in tail ~ r_cut^-p, from two grid points."""
    i, j = int(np.argmin(np.abs(grid - lo))), int(np.argmin(np.abs(grid - hi)))
    return float(-np.log(tail[j] / tail[i]) / np.log(grid[j] / grid[i]))


# --------------------------------------------------------------------------
# 5. The computation
# --------------------------------------------------------------------------

def compute() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {}
    box = N_CELLS * D_NN

    # ---- periodic crystal, minimum-image graph, hop shells -----------------
    pos, charge = build_crystal(N_CELLS)
    centre = int(np.flatnonzero(np.all(np.isclose(pos, (N_CELLS // 2) * D_NN), axis=1))[0])
    r_cut_ref = 1.1 * D_NN
    nbrs = neighbours_minimum_image(pos, box, r_cut_ref)
    coord = np.array([len(x) for x in nbrs])

    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs_tree = set(map(tuple, np.sort(np.array(sorted(tree.query_pairs(r_cut_ref + 1e-12))), axis=1)))
    pairs_bf = set()
    for i, js in enumerate(nbrs):
        for j in js:
            pairs_bf.add((min(i, int(j)), max(i, int(j))))

    l_max_box = 5
    shells_pbc = hop_sets_pbc(nbrs, centre, l_max_box)
    n_pbc = np.cumsum([len(s) for s in shells_pbc]).tolist()

    hop_index = -np.ones(len(pos), dtype=int)
    for L, s in enumerate(shells_pbc):
        hop_index[s] = L

    shells_inf = hop_shells(r_cut_ref, l_max_box)
    n_inf, r_eff = cumulative(shells_inf)
    octahedral = [(2 * L + 1) * (2 * L * L + 2 * L + 3) // 3 for L in range(l_max_box + 1)]

    # min-image displacement equals the true one while R_eff < box/2, so this
    # is a real cross-check of the periodic bookkeeping, not a tautology
    r_eff_pbc = []
    for L in range(l_max_box + 1):
        idx = np.concatenate(shells_pbc[: L + 1])
        d = pos[idx] - pos[centre]
        d -= box * np.round(d / box)
        r_eff_pbc.append(float(np.max(np.linalg.norm(d, axis=1))))

    res["lattice"] = {
        "structure": "rock salt (simple cubic, alternating +-1)",
        "spacing_d": float(D_NN),
        "n_cells": int(N_CELLS),
        "box": float(box),
        "n_atoms": int(len(pos)),
        "centre_index": centre,
        "coords_int": np.round(pos / D_NN).astype(int).tolist(),
        "charge": charge.astype(int).tolist(),
        "hop_index_rcut_1p1": hop_index.tolist(),
    }
    res["receptive_field"] = {
        "r_cut": float(r_cut_ref),
        "L": list(range(l_max_box + 1)),
        "n_atoms_pbc": [int(x) for x in n_pbc],
        "n_atoms_infinite": [int(x) for x in n_inf],
        "closed_form_octahedral": [int(x) for x in octahedral],
        "R_eff": [float(x) for x in r_eff],
        "R_eff_pbc": [float(x) for x in r_eff_pbc],
        "L_times_r_cut": [float(L * r_cut_ref) for L in range(l_max_box + 1)],
        "ratio_R_eff_over_L_r_cut": [float(r_eff[L] / (L * r_cut_ref)) for L in range(1, l_max_box + 1)],
        "coordination_min": int(coord.min()),
        "coordination_max": int(coord.max()),
        "kdtree_pairs": int(len(pairs_tree)),
        "brute_force_pairs": int(len(pairs_bf)),
        "pair_set_mismatches": int(len(pairs_tree ^ pairs_bf)),
    }

    # ---- finite-size failure: when the cell can no longer host the field ---
    deep = 12
    shells_pbc_deep = hop_sets_pbc(nbrs, centre, deep)
    n_pbc_deep = np.cumsum([len(s) for s in shells_pbc_deep]).tolist()
    n_inf_deep, _ = cumulative(hop_shells(r_cut_ref, deep))
    first_bad = next((L for L in range(deep + 1) if n_pbc_deep[L] != n_inf_deep[L]), None)
    res["finite_size"] = {
        "L": list(range(deep + 1)),
        "n_atoms_pbc": [int(x) for x in n_pbc_deep],
        "n_atoms_infinite": [int(x) for x in n_inf_deep],
        "first_L_where_cell_too_small": int(first_bad) if first_bad is not None else None,
        "note": "beyond this depth the periodic image of an atom re-enters its own "
                "receptive field: the simulation cell is smaller than the model.",
    }

    # ---- growth constant across cutoffs -----------------------------------
    sweep = []
    for r_cut, lm in [(1.1, 5), (1.5, 5), (1.8, 5), (2.2, 4), (2.5, 4), (3.0, 4)]:
        off = lattice_offsets(r_cut)
        c = float(np.max(np.linalg.norm(off.astype(float), axis=1)))
        sizes, radii = cumulative(hop_shells(r_cut, lm))
        slopes = [radii[L + 1] - radii[L] for L in range(lm)]
        sweep.append({
            "r_cut": float(r_cut),
            "k_neighbours": int(len(off)),
            "longest_edge_c": c,
            "c_over_r_cut": float(c / r_cut),
            "L": list(range(lm + 1)),
            "n_atoms": [int(x) for x in sizes],
            "R_eff": [float(x) for x in radii],
            "increment_max_dev_from_c": float(np.max(np.abs(np.array(slopes) - c))),
            "R_eff_minus_L_c_max": float(np.max([abs(radii[L] - L * c) for L in range(lm + 1)])),
            "R_eff_over_L_r_cut_at_Lmax": float(radii[lm] / (lm * r_cut)),
        })
    res["growth_sweep"] = sweep

    # ---- cubic cost scaling ------------------------------------------------
    deep_sizes, _ = cumulative(hop_shells(1.1, 12))
    Ls = np.arange(6)
    coef = np.polyfit(Ls, np.array(deep_sizes[:6], dtype=float), 3)
    cubic_resid = float(np.max(np.abs(np.polyval(coef, Ls) - np.array(deep_sizes[:6], dtype=float))))
    cube_sizes, cube_radii = cumulative(hop_shells(1.8, 6))
    exps = [(int(a), int(2 * a), float(np.log(deep_sizes[2 * a] / deep_sizes[a]) / np.log(2.0)))
            for a in (2, 3, 4, 5, 6)]
    res["cubic_scaling"] = {
        "r_cut": 1.1,
        "L": list(range(13)),
        "n_atoms": [int(x) for x in deep_sizes],
        "cubic_fit_coeffs_L0_to_5": [float(x) for x in coef],
        "cubic_fit_max_residual": cubic_resid,
        "leading_coefficient": float(coef[0]),
        "leading_coefficient_theory": 4.0 / 3.0,
        "leading_term_only_at_L5": float(coef[0] * 125.0),
        "actual_over_leading_term_at_L5": float(deep_sizes[5] / (coef[0] * 125.0)),
        "loglog_exponent_L_to_2L": [[a, b, e] for a, b, e in exps],
        "cube_case": {
            "r_cut": 1.8,
            "k_neighbours": int(len(lattice_offsets(1.8))),
            "L": list(range(7)),
            "n_atoms": [int(x) for x in cube_sizes],
            "closed_form_2Lplus1_cubed": [int((2 * L + 1) ** 3) for L in range(7)],
            "R_eff": [float(x) for x in cube_radii],
        },
    }

    # ---- truncation of lattice sums ---------------------------------------
    r, sign = sorted_lattice(R_SUM_MAX)
    grid = np.round(np.arange(1.0, 20.0 + 1e-9, 0.25), 6)
    kappa = 1.0
    rho = 1.0 / D_NN ** 3

    u_coul = truncated(sign / r, r, grid)
    m_ewald = madelung_ewald()
    u_coul_exact = -m_ewald
    rel_coul = np.abs(u_coul - u_coul_exact) / abs(u_coul_exact)

    u_yuk = truncated(np.exp(-kappa * r) / r, r, grid)
    u_yuk_inf = float(np.sum(np.exp(-kappa * r) / r))
    tail_yuk = u_yuk_inf - u_yuk

    def yuk_tail_pred(x):
        return 4 * np.pi * rho * np.exp(-kappa * x) * (x / kappa + 1.0 / kappa ** 2)

    pred_yuk = yuk_tail_pred(grid) - yuk_tail_pred(R_SUM_MAX)

    u_vdw = truncated(1.0 / r ** 6, r, grid)
    u_vdw_win = float(np.sum(1.0 / r ** 6))
    tail_vdw = u_vdw_win - u_vdw
    pred_vdw = (4 * np.pi * rho / 3.0) * (grid ** -3 - R_SUM_MAX ** -3)
    u_vdw_inf = u_vdw_win + (4 * np.pi * rho / 3.0) * R_SUM_MAX ** -3

    u_pos = truncated(1.0 / r, r, grid)
    pred_pos = 2 * np.pi * rho * grid ** 2

    win = (grid >= 4.0)
    near, far = (grid >= 4.0) & (grid < 10.0), (grid >= 10.0)
    sel = (grid >= 10.0) & (grid <= 20.0)
    i12 = int(np.argmin(np.abs(grid - 12.0)))
    i6 = int(np.argmin(np.abs(grid - 6.0)))
    rel_yuk = np.abs(tail_yuk) / abs(u_yuk_inf)
    rel_vdw = np.abs(u_vdw - u_vdw_inf) / abs(u_vdw_inf)

    # The pointwise ratio tail/pred oscillates a few percent about 1 because the
    # lattice adds whole shells while the continuum law is smooth. The rate, not
    # the pointwise value, is the claim -- so fit it.
    slope_y, inter_y, r2_y = loglin_fit(
        grid[win], np.log(tail_yuk[win]) - np.log(grid[win] / kappa + 1.0 / kappa ** 2))
    slope_v, inter_v, r2_v = loglin_fit(np.log(grid[win]), np.log(tail_vdw[win]))
    p_yuk = local_exponent(tail_yuk, grid, 12.0, 18.0)
    p_vdw = local_exponent(tail_vdw, grid, 12.0, 18.0)
    env = np.abs(u_coul - u_coul_exact)
    e_lo = float(np.max(env[(grid >= 10.0) & (grid <= 14.0)]))
    e_hi = float(np.max(env[(grid >= 16.0) & (grid <= 20.0)]))
    p_coul = float(-np.log(e_hi / e_lo) / np.log(18.0 / 12.0))

    res["truncation"] = {
        "r_cut": grid.tolist(),
        "kappa": float(kappa),
        "r_sum_max": float(R_SUM_MAX),
        "coulomb": {
            "U_trunc": u_coul.tolist(),
            "U_exact_ewald": float(u_coul_exact),
            "madelung_ewald": float(m_ewald),
            "madelung_reference": MADELUNG_REF,
            "madelung_abs_error": float(abs(m_ewald - MADELUNG_REF)),
            "alpha_spread": float(np.ptp([madelung_ewald(a) for a in (1.0, 1.5, 2.0, 2.5, 3.0)])),
            "rel_err": rel_coul.tolist(),
            "rel_err_at_6d": float(rel_coul[i6]),
            "rel_err_at_12d": float(rel_coul[i12]),
            "U_ptp_over_rc_10_to_20": float(np.ptp(u_coul[sel])),
            "n_sign_changes_of_error": int(np.sum(np.diff(np.sign(u_coul - u_coul_exact)) != 0)),
            "error_envelope_10_to_14": e_lo,
            "error_envelope_16_to_20": e_hi,
            "effective_decay_exponent": p_coul,
        },
        "yukawa": {
            "U_trunc": u_yuk.tolist(),
            "U_exact": u_yuk_inf,
            "tail": tail_yuk.tolist(),
            "tail_continuum_pred": pred_yuk.tolist(),
            "tail_over_pred_mean": float(np.mean(tail_yuk[win] / pred_yuk[win])),
            "tail_over_pred_max_dev": float(np.max(np.abs(tail_yuk[win] / pred_yuk[win] - 1.0))),
            "fitted_decay_rate": slope_y,
            "fitted_prefactor": float(np.exp(inter_y)),
            "prefactor_theory": float(4 * np.pi * rho),
            "fit_r2": r2_y,
            "effective_decay_exponent": p_yuk,
            "rel_err": rel_yuk.tolist(),
            "rel_err_at_6d": float(rel_yuk[i6]),
            "rel_err_at_12d": float(rel_yuk[i12]),
        },
        "vdw_r6": {
            "U_trunc": u_vdw.tolist(),
            "U_exact": float(u_vdw_inf),
            "tail": tail_vdw.tolist(),
            "tail_continuum_pred": pred_vdw.tolist(),
            "tail_over_pred_mean": float(np.mean(tail_vdw[win] / pred_vdw[win])),
            "tail_over_pred_max_dev": float(np.max(np.abs(tail_vdw[win] / pred_vdw[win] - 1.0))),
            "tail_over_pred_max_dev_near": float(np.max(np.abs(tail_vdw[near] / pred_vdw[near] - 1.0))),
            "tail_over_pred_max_dev_far": float(np.max(np.abs(tail_vdw[far] / pred_vdw[far] - 1.0))),
            "fitted_power": slope_v,
            "fitted_prefactor": float(np.exp(inter_v)),
            "prefactor_theory": float(4 * np.pi * rho / 3.0),
            "fit_r2": r2_v,
            "effective_decay_exponent": p_vdw,
            "rel_err": rel_vdw.tolist(),
            "rel_err_at_6d": float(rel_vdw[i6]),
            "rel_err_at_12d": float(rel_vdw[i12]),
        },
        "coulomb_same_sign": {
            "U_trunc": u_pos.tolist(),
            "continuum_pred_2pi_rho_rc2": pred_pos.tolist(),
            "ratio_max_dev_rc_ge_6": float(np.max(np.abs((u_pos / pred_pos)[grid >= 6.0] - 1.0))),
            "note": "diverges as r_cut^2: an uncompensated 1/r lattice has no local energy at all.",
        },
        "locality_ratio_at_12d": float(rel_coul[i12] / rel_yuk[i12]),
    }

    # ---- depth vs cutoff at fixed reach -----------------------------------
    trade = []
    for L, r_cut in [(5, 1.1), (3, 1.75), (2, 2.55), (1, 5.0)]:
        off = lattice_offsets(r_cut)
        sizes, radii = cumulative(hop_shells(r_cut, L))
        trade.append({
            "L": int(L),
            "r_cut": float(r_cut),
            "R_eff": float(radii[L]),
            "k_edges_per_atom": int(len(off)),
            "halo_atoms": int(sizes[L] - 1),
            "edge_evals_per_atom": int(L * len(off)),
            "comm_rounds": int(L),
        })
    reach = np.array([t["R_eff"] for t in trade])
    res["depth_vs_cutoff"] = {
        "target_R_eff": 5.0,
        "configs": trade,
        "R_eff_max_rel_spread": float(np.max(np.abs(reach / 5.0 - 1.0))),
        "edge_eval_ratio_shallow_over_deep": float(trade[-1]["edge_evals_per_atom"] / trade[0]["edge_evals_per_atom"]),
        "halo_ratio_shallow_over_deep": float(trade[-1]["halo_atoms"] / trade[0]["halo_atoms"]),
        "comm_ratio_deep_over_shallow": float(trade[0]["comm_rounds"] / trade[-1]["comm_rounds"]),
    }

    # NequIP-style depth against an Allegro-style single layer, same cutoff
    r_phys = 2.55
    off_phys = lattice_offsets(r_phys)
    sz4, rad4 = cumulative(hop_shells(r_phys, 4))
    res["nequip_vs_allegro"] = {
        "r_cut": float(r_phys),
        "k_edges_per_atom": int(len(off_phys)),
        "nequip_L": 4,
        "nequip_R_eff": float(rad4[4]),
        "nequip_halo_atoms": int(sz4[4] - 1),
        "allegro_R_eff": float(rad4[1]),
        "allegro_halo_atoms": int(sz4[1] - 1),
        "halo_ratio": float((sz4[4] - 1) / (sz4[1] - 1)),
        "reach_ratio": float(rad4[4] / rad4[1]),
    }

    # ---- shell symmetry: what a shell sum can and cannot see ---------------
    shell_sums = []
    R = random_rotation(rng)
    for L, sh in enumerate(shells_inf):
        if L == 0:
            continue
        v = sh.astype(float)
        entry = {"L": int(L), "n": int(len(v))}
        for l in (1, 2, 3, 4):
            s = np.sum(real_sh(l, v), axis=0)
            entry[f"norm_l{l}"] = float(np.linalg.norm(s))
        s4 = np.sum(real_sh(4, v), axis=0)
        s4r = np.sum(real_sh(4, v @ R.T), axis=0)
        entry["l4_norm_rotation_dev"] = float(abs(np.linalg.norm(s4r) - np.linalg.norm(s4)))
        shell_sums.append(entry)
    res["shell_harmonics"] = {
        "r_cut": float(r_cut_ref),
        "shells": shell_sums,
        "max_norm_l123": float(max(max(e[f"norm_l{l}"] for l in (1, 2, 3)) for e in shell_sums)),
        "min_norm_l4": float(min(e["norm_l4"] for e in shell_sums)),
        "max_l4_rotation_dev": float(max(e["l4_norm_rotation_dev"] for e in shell_sums)),
    }
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

    rf = res["receptive_field"]
    print("periodic graph construction")
    check("minimum-image pair set identical to periodic KD-tree",
          rf["pair_set_mismatches"] == 0,
          f"{rf['pair_set_mismatches']} mismatches over {rf['brute_force_pairs']} pairs")
    check("perfect crystal: every atom has coordination 6 at r_cut = 1.1 d",
          rf["coordination_min"] == 6 and rf["coordination_max"] == 6,
          f"min {rf['coordination_min']}, max {rf['coordination_max']}")

    print("receptive field growth")
    check("periodic L-hop counts equal infinite-lattice counts for L <= 5",
          rf["n_atoms_pbc"] == rf["n_atoms_infinite"],
          f"pbc {rf['n_atoms_pbc']} vs inf {rf['n_atoms_infinite']}")
    check("L-hop count equals the octahedral number (2L+1)(2L^2+2L+3)/3",
          rf["n_atoms_infinite"] == rf["closed_form_octahedral"],
          f"counts {rf['n_atoms_infinite']}")
    dev_pbc = max(abs(a - b) for a, b in zip(rf["R_eff"], rf["R_eff_pbc"]))
    check("min-image radius from centre matches unwrapped radius (R_eff < box/2)",
          dev_pbc < 1e-12, f"max dev {dev_pbc:.2e}")

    worst_bound = max(rf["ratio_R_eff_over_L_r_cut"])
    check("R_eff(L) <= L * r_cut, the triangle-inequality bound",
          worst_bound <= 1.0 + 1e-12, f"max R_eff/(L r_cut) {worst_bound:.6f}")

    print("growth constant across cutoffs")
    for s in res["growth_sweep"]:
        check(f"r_cut={s['r_cut']}: R_eff(L) = L * c exactly, c = longest edge <= r_cut",
              s["R_eff_minus_L_c_max"] < 1e-12,
              f"c {s['longest_edge_c']:.6f}, max |R_eff - L c| {s['R_eff_minus_L_c_max']:.2e}, "
              f"c/r_cut {s['c_over_r_cut']:.4f}")
    ratios = [s["c_over_r_cut"] for s in res["growth_sweep"]]
    check("the L*r_cut estimate is an overestimate on a lattice (c/r_cut < 1 for some cutoff)",
          min(ratios) < 0.95, f"c/r_cut ranges {min(ratios):.4f} to {max(ratios):.4f}")

    print("cost of depth")
    cs = res["cubic_scaling"]
    check("N(L) is exactly cubic in L for the 6-neighbour graph",
          cs["cubic_fit_max_residual"] < 1e-6, f"max residual {cs['cubic_fit_max_residual']:.2e}")
    check("leading coefficient = 4/3 = (unit l1-ball volume) x (number density)",
          abs(cs["leading_coefficient"] - 4.0 / 3.0) < 1e-9,
          f"fitted {cs['leading_coefficient']:.12f}")
    cc = cs["cube_case"]
    check("26-neighbour graph: N(L) = (2L+1)^3 exactly, the clean cubic law",
          cc["n_atoms"] == cc["closed_form_2Lplus1_cubed"], f"counts {cc['n_atoms']}")
    exps = [e[2] for e in cs["loglog_exponent_L_to_2L"]]
    check("log-log exponent rises monotonically toward 3 but is not there by L = 12",
          all(exps[i] < exps[i + 1] for i in range(len(exps) - 1)) and 2.7 < exps[-1] < 3.0,
          f"exponents {[round(e, 3) for e in exps]}")
    check("at usable depth the asymptotic cubic term understates the real count",
          cs["actual_over_leading_term_at_L5"] > 1.2,
          f"N(5)/(4/3 * 5^3) = {cs['actual_over_leading_term_at_L5']:.4f}")

    fs = res["finite_size"]
    check("periodic cell stops hosting the receptive field at a finite depth",
          fs["first_L_where_cell_too_small"] == 6,
          f"first L with wrap-around: {fs['first_L_where_cell_too_small']} "
          f"(pbc {fs['n_atoms_pbc'][6]} vs infinite {fs['n_atoms_infinite'][6]})")

    print("exact Coulomb reference")
    co = res["truncation"]["coulomb"]
    check("Ewald reproduces the NaCl Madelung constant 1.747564594633182",
          co["madelung_abs_error"] < 1e-10, f"abs error {co['madelung_abs_error']:.2e}")
    check("Madelung value independent of the Ewald splitting parameter alpha",
          co["alpha_spread"] < 1e-12, f"spread over alpha in [1,3] {co['alpha_spread']:.2e}")

    print("does locality suffice")
    yu = res["truncation"]["yukawa"]
    vd = res["truncation"]["vdw_r6"]
    ss = res["truncation"]["coulomb_same_sign"]
    check("short-range exp(-r)/r: fitted tail decay rate equals kappa = 1",
          abs(yu["fitted_decay_rate"] + res["truncation"]["kappa"]) < 0.02 and yu["fit_r2"] > 0.999,
          f"rate {yu['fitted_decay_rate']:.5f} vs -1, R^2 {yu['fit_r2']:.6f}")
    check("short-range exp(-r)/r: fitted tail prefactor equals 4 pi rho",
          abs(yu["fitted_prefactor"] / yu["prefactor_theory"] - 1.0) < 0.03,
          f"fitted {yu['fitted_prefactor']:.4f} vs 4 pi rho = {yu['prefactor_theory']:.4f}")
    check("dispersion 1/r^6: fitted tail power equals 3",
          abs(vd["fitted_power"] + 3.0) < 0.03 and vd["fit_r2"] > 0.999,
          f"power {vd['fitted_power']:.5f} vs -3, R^2 {vd['fit_r2']:.6f}")
    check("dispersion 1/r^6: fitted tail prefactor equals 4 pi rho / 3",
          abs(vd["fitted_prefactor"] / vd["prefactor_theory"] - 1.0) < 0.03,
          f"fitted {vd['fitted_prefactor']:.4f} vs 4 pi rho/3 = {vd['prefactor_theory']:.4f}")
    check("continuum laws are unbiased: mean tail/pred = 1 (residual scatter is lattice shells)",
          abs(yu["tail_over_pred_mean"] - 1.0) < 0.02 and abs(vd["tail_over_pred_mean"] - 1.0) < 0.02,
          f"mean {yu['tail_over_pred_mean']:.4f} (yukawa, scatter {yu['tail_over_pred_max_dev']:.3f}), "
          f"{vd['tail_over_pred_mean']:.4f} (1/r^6, scatter {vd['tail_over_pred_max_dev']:.3f})")
    check("that scatter is discreteness: it shrinks as the cutoff grows (1/r^6)",
          vd["tail_over_pred_max_dev_far"] < 0.5 * vd["tail_over_pred_max_dev_near"],
          f"{vd['tail_over_pred_max_dev_near']:.4f} at r_cut in [4,10) -> "
          f"{vd['tail_over_pred_max_dev_far']:.4f} at r_cut >= 10")
    check("three regimes separate: exponent p in tail ~ r_cut^-p is >10, ~3, <=0",
          yu["effective_decay_exponent"] > 10.0
          and abs(vd["effective_decay_exponent"] - 3.0) < 0.15
          and co["effective_decay_exponent"] <= 0.0,
          f"exp(-r)/r {yu['effective_decay_exponent']:.2f}, "
          f"1/r^6 {vd['effective_decay_exponent']:.3f}, "
          f"coulomb {co['effective_decay_exponent']:.3f}")
    check("uncompensated 1/r: truncated sum grows as 2 pi rho r_cut^2, no limit exists",
          ss["ratio_max_dev_rc_ge_6"] < 0.03,
          f"max |U/(2 pi rho r_cut^2) - 1| over r_cut >= 6: {ss['ratio_max_dev_rc_ge_6']:.4f}")
    check("Coulomb: spherical truncation does NOT converge -- error still > 50% at r_cut = 12 d",
          co["rel_err_at_12d"] > 0.5, f"relative error {co['rel_err_at_12d']:.4f}")
    check("Coulomb: truncated sum swings wider than the answer itself over r_cut in [10,20]",
          co["U_ptp_over_rc_10_to_20"] > abs(co["U_exact_ewald"]),
          f"peak-to-peak {co['U_ptp_over_rc_10_to_20']:.4f} vs |U_exact| "
          f"{abs(co['U_exact_ewald']):.4f}")
    check("at r_cut = 12 d the Coulomb error exceeds the short-range error by >1e3",
          res["truncation"]["locality_ratio_at_12d"] > 1e3,
          f"ratio {res['truncation']['locality_ratio_at_12d']:.3e} "
          f"(coulomb {co['rel_err_at_12d']:.3e}, yukawa {yu['rel_err_at_12d']:.3e})")

    print("depth versus cutoff at fixed reach")
    dv = res["depth_vs_cutoff"]
    check("the four configurations really do reach the same radius (within 5%)",
          dv["R_eff_max_rel_spread"] < 0.05,
          f"max relative spread {dv['R_eff_max_rel_spread']:.4f}, "
          f"R_eff {[round(c['R_eff'], 3) for c in dv['configs']]}")
    ev = [c["edge_evals_per_atom"] for c in dv["configs"]]
    ha = [c["halo_atoms"] for c in dv["configs"]]
    check("at fixed reach, shallow-and-wide costs strictly more edge evaluations",
          all(ev[i] < ev[i + 1] for i in range(len(ev) - 1)) and dv["edge_eval_ratio_shallow_over_deep"] > 10,
          f"edge evals {ev}, shallow/deep = {dv['edge_eval_ratio_shallow_over_deep']:.1f}x")
    check("at fixed reach, shallow-and-wide also needs a strictly larger halo",
          all(ha[i] < ha[i + 1] for i in range(len(ha) - 1)),
          f"halo atoms {ha}, shallow/deep = {dv['halo_ratio_shallow_over_deep']:.2f}x")
    check("what depth buys back is communication rounds",
          dv["comm_ratio_deep_over_shallow"] == 5.0,
          f"{dv['configs'][0]['comm_rounds']} rounds vs {dv['configs'][-1]['comm_rounds']}")
    na = res["nequip_vs_allegro"]
    check("NequIP L=4 vs Allegro L=1 at the same cutoff: halo grows far faster than reach",
          na["halo_ratio"] > 4 * na["reach_ratio"],
          f"halo {na['allegro_halo_atoms']} -> {na['nequip_halo_atoms']} "
          f"({na['halo_ratio']:.1f}x) for {na['reach_ratio']:.1f}x reach")

    print("what a shell sum can see")
    sh = res["shell_harmonics"]
    check("octahedral shells: sum_j Y_l vanishes for l = 1, 2, 3",
          sh["max_norm_l123"] < 1e-10, f"max norm {sh['max_norm_l123']:.2e}")
    check("first nonzero shell invariant is l = 4",
          sh["min_norm_l4"] > 0.1, f"min |sum Y_4| over shells {sh['min_norm_l4']:.4f}")
    check("|sum Y_4| is rotation invariant (it is a genuine O(3) feature)",
          sh["max_l4_rotation_dev"] < 1e-9, f"max dev under a random rotation {sh['max_l4_rotation_dev']:.2e}")

    res["checks"] = log
    return npass, ntotal


if __name__ == "__main__":
    print("=" * 78)
    print("receptive_field.py -- message passing grows reach; locality is an approximation")
    print("=" * 78)
    results = compute()
    npass, ntotal = _selfcheck(results)
    results["meta"] = {
        "seed": SEED,
        "name": NAME,
        "description": "Receptive-field growth of message passing on a rock-salt crystal, the "
                       "cubic cost of depth, and the truncation error of short- versus "
                       "long-range pair interactions -- the NequIP/Allegro locality trade.",
        "n_pass": int(npass),
        "n_total": int(ntotal),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1))
    print("-" * 78)
    print(f"wrote {RESULTS}  ({RESULTS.stat().st_size / 1024:.1f} KB)")
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
