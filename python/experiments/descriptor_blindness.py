"""What invariant descriptors throw away -- the exact motivation for l>=1 features.

NequIP's design choice is to carry rotation-*equivariant* features of degree
l = 0, 1, 2, ... between layers instead of collapsing every message down to
rotation-*invariant* scalars. The usual defence is expressive power. This
script makes that defence concrete and provable on three levels, with no
fitting and no appeal to authority: each level exhibits two atomic
environments that a descriptor family literally cannot tell apart, and then
measures exactly where the blindness stops.

(A) TWO-BODY BLINDNESS -- trivial. A centre with three neighbours all at
    r = 1: an equilateral triangle in the xy-plane versus the same three
    neighbours pushed together (angular gaps 30/30/300 deg). The multiset of
    centre-neighbour distances is bitwise identical, so *every* pair
    descriptor -- Behler-Parrinello G2, any radial-basis message, any
    Lennard-Jones-style sum -- returns bitwise identical bytes. A
    Stillinger-Weber three-body term differs by a factor of ~43. Two-body is
    not enough, which nobody disputes.

(B) THREE-BODY BLINDNESS -- the real result, and it is exact. Take the
    Z-related pair from combinatorics on the cyclic group Z_12, known in
    music theory as the all-interval tetrachords:

        A = {0, 1, 4, 6}        B = {0, 1, 3, 7}

    Both have interval vector <1,1,1,1,1,1>: every cyclic interval class
    1..6 occurs exactly once. Place four neighbours on the unit circle at
    angles 2*pi*k/12. Then

      * all centre-neighbour distances are 1 (so radial descriptors die);
      * the sorted multiset of all six neighbour-neighbour distances agrees
        to 2e-16 -- this is *homometry*, and since
        |r_j - r_k|^2 = 2 - 2 cos(theta_jik), the multiset of bond angles
        agrees too, so every three-body descriptor dies as well;
      * yet the two environments are genuinely NOT congruent: minimising
        RMSD over all 4! neighbour permutations, a fine in-plane rotation
        scan and the reflection -- confirmed independently by an exact 2-D
        closed form and by 3-D Kabsch -- bottoms out at (sqrt(3)-1)/2 =
        0.3660, nowhere near zero.

    The blindness is not a coincidence of this example. By the spherical
    harmonic addition theorem the SOAP/power-spectrum invariant is

        p_l = ||sum_j Y_l(r_hat_j)||^2 = (2l+1)/(4pi) sum_{j,k} P_l(cos theta_jk),

    a function of the bond-angle multiset *alone*. Homometry therefore forces
    p_l^A = p_l^B at EVERY l, which is exactly what is measured here: the
    largest discrepancy over l = 0..6 is 3.1e-15, i.e. zero. Collapsing to
    invariants at each layer, as an invariant-message network does, destroys
    the distinction permanently.

    The equivariant features do not. Reported honestly, because the naive
    answer is misleading:

      * l = 0: identical (both are 4 * Y_00).
      * l = 1: the features DIFFER in the lab frame, ||f_1^A - f_1^B|| =
        0.3577. But that difference is a pure rotation -- an explicit R*
        built from the two dipole directions satisfies
        ||D^1(R*) f_1^A - f_1^B|| = 1e-16. So l <= 1 alone does NOT separate
        the two *shapes*; the lab-frame difference is orientation, not
        content.
      * l <= 2 DOES separate, and this is certified rather than searched.
        The (1,1)->2 Clebsch-Gordan bispectrum b_112 is an exact rotation
        invariant of (f_1, f_2), and b_112^A - b_112^B = -0.2857. Trilinearity
        of the CG contraction turns that gap into a rigorous lower bound
        min_R sqrt(||D^1 f_1^A - f_1^B||^2 + ||D^2 f_2^A - f_2^B||^2) >= 0.121,
        with no search involved. A 20000-rotation SO(3) search plus local
        refinement independently reports 0.666, consistent with the bound.

    So the smallest degree at which an equivariant representation separates
    these environments is l = 2, and it must be used *jointly* with l = 1
    -- the invariant doing the work, b_112, couples two vectors into a rank-2
    tensor, exactly the operation a scalar-only message layer cannot perform:
    neither the l <= 1 stack nor the invariant power spectrum at any l can do
    it. That is precisely NequIP's argument -- keep orientation alive between
    layers so that CG products of different l can build the higher-body
    invariants (the bispectrum) that the power spectrum lacks.

(C) Every coordinate, feature and curve below is written to
    results/descriptor_blindness.json so a page can redraw it.

Claims that did NOT come out as one might guess, kept rather than softened:
  * The invariant power spectrum fails at every l tested -- it is not a
    matter of going to higher l. Predicted by the addition theorem, verified
    to 1e-14 here.
  * "l = 1 separates them" is false as a statement about shape, even though
    the lab-frame l = 1 features differ. Both statements are reported.
  * All parity-odd (pseudo-scalar) bispectrum entries vanish for both
    environments, to 7.4e-16: the configurations are planar, hence invariant
    under reflection through their own plane, so chirality-sensing invariants
    have nothing to see here.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
from scipy.special import eval_legendre

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from e3 import real_sh, wigner_D, random_rotation, rotation_matrix, clebsch_gordan  # noqa: E402

SEED = 20260729
LMAX = 6
N_RING = 12
Z_A = (0, 1, 4, 6)
Z_B = (0, 1, 3, 7)
SW_LAMBDA = 21.0            # Stillinger-Weber silicon three-body strength
SW_COS_THETA0 = -1.0 / 3.0  # tetrahedral reference angle, 109.47 deg
N_SO3_COARSE = 20000

RESULTS = pathlib.Path(__file__).resolve().parents[2] / "results" / "descriptor_blindness.json"


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def ring(ks, n: int = N_RING) -> np.ndarray:
    """Unit-circle positions in the xy-plane at angles 2 pi k / n."""
    a = 2.0 * np.pi * np.asarray(ks, dtype=float) / n
    return np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)


def sorted_pair_distances(c: np.ndarray) -> np.ndarray:
    return np.sort(np.array([np.linalg.norm(c[i] - c[j])
                             for i, j in itertools.combinations(range(len(c)), 2)]))


def sorted_bond_cosines(c: np.ndarray) -> np.ndarray:
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    return np.sort(np.array([u[i] @ u[j]
                             for i, j in itertools.combinations(range(len(c)), 2)]))


def radial_descriptor(c: np.ndarray, mu: np.ndarray, eta: float) -> np.ndarray:
    """Behler-Parrinello G2 with a Gaussian radial basis: a pure two-body sum."""
    r = np.sort(np.linalg.norm(c, axis=1))
    return np.exp(-eta * (r[:, None] - mu[None, :]) ** 2).sum(axis=0)


def sw_three_body(c: np.ndarray, lam: float = SW_LAMBDA,
                  cos0: float = SW_COS_THETA0) -> float:
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    return float(sum(lam * (u[j] @ u[k] - cos0) ** 2
                     for j, k in itertools.combinations(range(len(c)), 2)))


def interval_vector(S, n: int = N_RING) -> list[int]:
    """Multiplicity of each cyclic interval class 1..n/2 in the set S."""
    v = [0] * (n // 2)
    for i, j in itertools.combinations(range(len(S)), 2):
        d = (S[i] - S[j]) % n
        v[min(d, n - d) - 1] += 1
    return v


def dihedral_orbit(S, n: int = N_RING) -> set:
    """All transpositions and inversions of S on Z_n -- the exact congruences."""
    out = set()
    for t in range(n):
        out.add(tuple(sorted((s + t) % n for s in S)))
        out.add(tuple(sorted((t - s) % n for s in S)))
    return out


# --------------------------------------------------------------------------
# congruence: is there any isometry fixing the centre that maps A onto B?
# --------------------------------------------------------------------------

def rmsd_inplane_closed_form(P: np.ndarray, Q: np.ndarray):
    """Exact min RMSD over 2-D rotations x reflection x permutations.

    All points sit at radius 1, so sum_j |R c_j - q_j|^2 = 2n - 2 Re(e^{i a} Z)
    with Z = sum_j exp(i(gamma_j - beta_j)); the best angle gives |Z|.
    """
    n = len(P)
    ap = np.arctan2(P[:, 1], P[:, 0])
    aq = np.arctan2(Q[:, 1], Q[:, 0])
    best, arg = np.inf, None
    for perm in itertools.permutations(range(n)):
        for s in (1, -1):
            Z = np.sum(np.exp(1j * (s * ap - aq[list(perm)])))
            val = np.sqrt(max(0.0, 2.0 - 2.0 * abs(Z) / n))
            if val < best:
                best, arg = val, (perm, s, float(abs(Z)))
    return float(best), arg


def rmsd_inplane_scan(P: np.ndarray, Q: np.ndarray, n_ang: int = 3601):
    """Brute force: every permutation x reflection x a fine in-plane rotation grid."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    perms = list(itertools.permutations(range(len(P))))
    curve = np.empty(n_ang)
    best = np.inf
    for i, a in enumerate(angles):
        R = rotation_matrix([0.0, 0.0, 1.0], a)
        here = np.inf
        for s in (1, -1):
            Ps = P.copy()
            Ps[:, 1] *= s
            Pr = Ps @ R.T
            for perm in perms:
                v = np.sqrt(np.mean(np.sum((Pr - Q[list(perm)]) ** 2, axis=1)))
                here = min(here, v)
        curve[i] = here
        best = min(best, here)
    return float(best), angles, curve


def rmsd_kabsch_o3(P: np.ndarray, Q: np.ndarray) -> float:
    """Exact min RMSD over full O(3) (centre pinned at the origin) x permutations."""
    best = np.inf
    for perm in itertools.permutations(range(len(P))):
        Qp = Q[list(perm)]
        U, _, Vt = np.linalg.svd(P.T @ Qp)
        for d in (1.0, -1.0):
            R = (U @ np.diag([1.0, 1.0, d]) @ Vt).T
            best = min(best, np.sqrt(np.mean(np.sum((P @ R.T - Qp) ** 2, axis=1))))
    return float(best)


# --------------------------------------------------------------------------
# equivariant features
# --------------------------------------------------------------------------

def features(c: np.ndarray, lmax: int = LMAX) -> list[np.ndarray]:
    """Atom-centred equivariant features f_l = sum_j Y_l(r_hat_ij)."""
    return [real_sh(l, c).sum(axis=0) for l in range(lmax + 1)]


def feature_errors(Rs: np.ndarray, P: np.ndarray, fQ: list[np.ndarray]) -> np.ndarray:
    """Per-degree squared mismatch ||f_l(R P) - f_l(Q)||^2 for a batch of rotations."""
    pts = np.einsum("nij,kj->nki", Rs, P)
    out = np.empty((len(Rs), len(fQ)))
    for l, g in enumerate(fQ):
        d = real_sh(l, pts).sum(axis=1) - g
        out[:, l] = np.sum(d * d, axis=1)
    return out


def haar_batch(rng: np.random.Generator, n: int) -> np.ndarray:
    A = rng.standard_normal((n, 3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.einsum("nii->ni", R))[:, None, :]
    Q[np.linalg.det(Q) < 0, :, 0] *= -1.0
    return Q


def perturb(Rs: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    w = rng.standard_normal((len(Rs), 3)) * sigma
    th = np.linalg.norm(w, axis=1)
    ax = w / np.where(th == 0.0, 1.0, th)[:, None]
    K = np.zeros((len(Rs), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    P = (np.eye(3) + np.sin(th).reshape(-1, 1, 1) * K
         + (1.0 - np.cos(th)).reshape(-1, 1, 1) * (K @ K))
    return np.einsum("nij,njk->nik", P, Rs)


def so3_search(P: np.ndarray, fQ: list[np.ndarray], rng: np.random.Generator):
    """Upper bound on min_R sqrt(sum_{l<=L} ||D^l(R) f_l(P) - f_l(Q)||^2), per L.

    Only an upper bound -- a search can never certify a minimum. The rigorous
    positivity statement comes from the bispectrum bound instead.
    """
    Rs = haar_batch(rng, N_SO3_COARSE)
    cum = np.cumsum(feature_errors(Rs, P, fQ), axis=1)
    out = []
    for L in range(len(fQ)):
        seeds = Rs[np.argsort(cum[:, L])[:12]]
        best = float(np.sqrt(cum[:, L].min()))
        sigma = 0.4
        for _ in range(16):
            cand = np.concatenate([seeds] + [perturb(seeds, rng, sigma) for _ in range(24)], 0)
            c2 = np.cumsum(feature_errors(cand, P, fQ), axis=1)[:, L]
            order = np.argsort(c2)[:12]
            seeds = cand[order]
            best = min(best, float(np.sqrt(c2[order[0]])))
            sigma *= 0.55
        out.append(best)
    return out


def align_l1(fA1: np.ndarray, fB1: np.ndarray) -> np.ndarray:
    """The rotation carrying f_1^A onto f_1^B. real_sh l=1 is ordered (y, z, x)."""
    a = fA1[[2, 0, 1]]
    b = fB1[[2, 0, 1]]
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    return rotation_matrix(axis, float(np.arctan2(np.linalg.norm(axis), a @ b)))


# --------------------------------------------------------------------------
# computation
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

# ---- (A) two-body blindness -------------------------------------------------
tri = ring([0, 4, 8])          # equilateral, gaps 120/120/120
clu = ring([0, 1, 2])          # clustered,   gaps  30/30/300
r_tri = np.sort(np.linalg.norm(tri, axis=1))
r_clu = np.sort(np.linalg.norm(clu, axis=1))
mu_grid = np.linspace(0.5, 1.5, 24)
G2_ETA = 20.0
g_tri = radial_descriptor(tri, mu_grid, G2_ETA)
g_clu = radial_descriptor(clu, mu_grid, G2_ETA)
radial_bitwise = bool(r_tri.tobytes() == r_clu.tobytes())
g2_bitwise = bool(g_tri.tobytes() == g_clu.tobytes())
a_radius_dev = float(max(np.max(np.abs(r_tri - 1.0)), np.max(np.abs(r_clu - 1.0))))
e_tri, e_clu = sw_three_body(tri), sw_three_body(clu)
p1_tri = float(np.dot(features(tri, 1)[1], features(tri, 1)[1]))
p1_clu = float(np.dot(features(clu, 1)[1], features(clu, 1)[1]))

# ---- (B) three-body blindness ----------------------------------------------
cA, cB = ring(Z_A), ring(Z_B)
rc_A = np.sort(np.linalg.norm(cA, axis=1))
rc_B = np.sort(np.linalg.norm(cB, axis=1))
d_A, d_B = sorted_pair_distances(cA), sorted_pair_distances(cB)
cos_A, cos_B = sorted_bond_cosines(cA), sorted_bond_cosines(cB)
homometry_dev = float(np.max(np.abs(d_A - d_B)))
angle_dev = float(np.max(np.abs(cos_A - cos_B)))
iv_A, iv_B = interval_vector(Z_A), interval_vector(Z_B)
same_orbit = bool(tuple(sorted(Z_B)) in dihedral_orbit(Z_A))

rmsd_cf, rmsd_cf_arg = rmsd_inplane_closed_form(cA, cB)
rmsd_scan, scan_angles, scan_curve = rmsd_inplane_scan(cA, cB)
rmsd_kabsch = rmsd_kabsch_o3(cA, cB)
rmsd_analytic = float((np.sqrt(3.0) - 1.0) / 2.0)

fA, fB = features(cA), features(cB)
lab_diff = [float(np.linalg.norm(fA[l] - fB[l])) for l in range(LMAX + 1)]
pA = [float(fA[l] @ fA[l]) for l in range(LMAX + 1)]
pB = [float(fB[l] @ fB[l]) for l in range(LMAX + 1)]
power_dev = float(max(abs(a - b) for a, b in zip(pA, pB)))

# Addition theorem: p_l is a function of the bond-angle multiset alone.
def power_from_angles(c: np.ndarray, l: int) -> float:
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    return float((2 * l + 1) / (4 * np.pi) * np.sum(eval_legendre(l, np.clip(u @ u.T, -1, 1))))


addition_dev = float(max(max(abs(power_from_angles(cA, l) - pA[l]),
                             abs(power_from_angles(cB, l) - pB[l]))
                         for l in range(LMAX + 1)))

# Equivariance of the feature map itself, against e3's Wigner matrices.
equi_dev = 0.0
for _ in range(5):
    R = random_rotation(rng)
    fr = features(cA @ R.T)
    for l in range(LMAX + 1):
        equi_dev = max(equi_dev, float(np.max(np.abs(fr[l] - wigner_D(l, R) @ fA[l]))))

# l <= 1 does not separate: build the aligning rotation explicitly.
R_star = align_l1(fA[1], fB[1])
l1_residual = float(np.linalg.norm(wigner_D(1, R_star) @ fA[1] - fB[1]))
l1_residual_direct = float(np.linalg.norm(real_sh(1, cA @ R_star.T).sum(axis=0) - fB[1]))
l2_at_Rstar = float(np.linalg.norm(wigner_D(2, R_star) @ fA[2] - fB[2]))

# l <= 2 does separate: certified via the (1,1)->2 bispectrum.
C112 = clebsch_gordan(1, 1, 2)
c112_fro = float(np.linalg.norm(C112))
b112_A = float(np.einsum("abc,a,b,c->", C112, fA[1], fA[1], fA[2]))
b112_B = float(np.einsum("abc,a,b,c->", C112, fB[1], fB[1], fB[2]))
n1 = float(max(np.linalg.norm(fA[1]), np.linalg.norm(fB[1])))
n2 = float(max(np.linalg.norm(fA[2]), np.linalg.norm(fB[2])))
lip_K = float(c112_fro * np.sqrt((2 * n1 * n2) ** 2 + (n1 ** 2) ** 2))
certified_lb = abs(b112_A - b112_B) / lip_K

search_delta = so3_search(cA, fB, np.random.default_rng(SEED + 1))

# In-plane slice of the same landscape, for plotting.
ip_angles = np.linspace(0.0, 2.0 * np.pi, 2001, endpoint=False)
Rz = np.zeros((len(ip_angles), 3, 3))
Rz[:, 0, 0], Rz[:, 0, 1] = np.cos(ip_angles), -np.sin(ip_angles)
Rz[:, 1, 0], Rz[:, 1, 1] = np.sin(ip_angles), np.cos(ip_angles)
Rz[:, 2, 2] = 1.0
ip_cum = np.cumsum(feature_errors(Rz, cA, fB), axis=1)
ip_delta2 = np.sqrt(ip_cum[:, 2])
ip_delta6 = np.sqrt(ip_cum[:, LMAX])

# Full bispectrum table up to l = 4.
LB = 4
bispectrum = []
for l1 in range(LB + 1):
    for l2 in range(l1, LB + 1):
        for l3 in range(abs(l1 - l2), min(l1 + l2, LB) + 1):
            C = clebsch_gordan(l1, l2, l3)
            ba = float(np.einsum("abc,a,b,c->", C, fA[l1], fA[l2], fA[l3]))
            bb = float(np.einsum("abc,a,b,c->", C, fB[l1], fB[l2], fB[l3]))
            bispectrum.append({"l1": l1, "l2": l2, "l3": l3,
                               "parity": int((-1) ** (l1 + l2 + l3)),
                               "b_A": ba, "b_B": bb, "abs_diff": abs(ba - bb)})
bispectrum.sort(key=lambda d: -d["abs_diff"])
bis_even = [d for d in bispectrum if d["parity"] > 0]
bis_odd = [d for d in bispectrum if d["parity"] < 0]
bis_even_max = max(d["abs_diff"] for d in bis_even)
bis_odd_max = max(max(abs(d["b_A"]), abs(d["b_B"])) for d in bis_odd)


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

CHECKS: list[dict] = []


def _selfcheck(verbose: bool = True) -> tuple[int, int]:
    npass = ntotal = 0

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        ok = bool(ok)
        npass += ok
        CHECKS.append({"name": name, "passed": ok, "detail": detail})
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("(A) two-body blindness: equilateral triangle vs clustered triple")
    check("both configs have every neighbour at r = 1",
          a_radius_dev < 1e-15, f"max |r - 1| {a_radius_dev:.2e}")
    check("sorted centre-neighbour distance multisets are bitwise identical",
          radial_bitwise, f"identical bytes {radial_bitwise}")
    check("24-channel Gaussian two-body descriptor is bitwise identical",
          g2_bitwise, f"max |dG2| {np.max(np.abs(g_tri - g_clu)):.2e}, identical bytes {g2_bitwise}")
    # An angular term must see what the radial term cannot: demand an order of magnitude.
    check("Stillinger-Weber three-body energy differs by > 10x",
          e_clu / e_tri > 10.0 and abs(e_clu - e_tri) > 10.0,
          f"E_tri {e_tri:.6f}, E_clu {e_clu:.6f}, ratio {e_clu / e_tri:.2f}")
    check("l=1 power separates them (0 for the triangle by symmetry)",
          p1_clu - p1_tri > 0.1 and p1_tri < 1e-20,
          f"p1_tri {p1_tri:.3e}, p1_clu {p1_clu:.6f}")

    print("(B) three-body blindness: the Z-related tetrachords on Z_12")
    check("A and B are both all-interval tetrachords <1,1,1,1,1,1>",
          iv_A == [1] * 6 and iv_B == [1] * 6, f"iv(A) {iv_A}, iv(B) {iv_B}")
    check("all centre-neighbour distances equal 1 in both",
          max(np.max(np.abs(rc_A - 1.0)), np.max(np.abs(rc_B - 1.0))) < 1e-15,
          f"max |r - 1| {max(np.max(np.abs(rc_A - 1.0)), np.max(np.abs(rc_B - 1.0))):.2e}")
    # 2 ulp of a distance of order 2 is 4.4e-16; 1e-14 is a generous but non-vacuous bound.
    check("sorted neighbour-neighbour distance multisets agree (homometry)",
          homometry_dev < 1e-14, f"max |dd| {homometry_dev:.2e}")
    check("sorted bond-angle-cosine multisets agree (so every 3-body term is blind)",
          angle_dev < 1e-14, f"max |dcos| {angle_dev:.2e}")
    check("A and B are NOT in the same dihedral orbit of Z_12 (exact, 24 symmetries)",
          not same_orbit, f"B in orbit(A): {same_orbit}")
    # A congruent pair would give RMSD ~ 1e-16; 0.3 on unit-radius points is decisive.
    check("brute-force min RMSD over 24 perms x 3601 angles x reflection is >> 0",
          rmsd_scan > 0.3, f"min RMSD {rmsd_scan:.12f}")
    check("the fine scan reproduces the exact 2-D closed form",
          abs(rmsd_scan - rmsd_cf) < 1e-6,
          f"scan {rmsd_scan:.12f}, closed form {rmsd_cf:.12f}, |d| {abs(rmsd_scan - rmsd_cf):.2e}")
    check("full O(3) Kabsch over perms finds nothing better (no out-of-plane escape)",
          abs(rmsd_kabsch - rmsd_cf) < 1e-9,
          f"Kabsch {rmsd_kabsch:.12f}, |d| {abs(rmsd_kabsch - rmsd_cf):.2e}")
    check("min RMSD equals the closed form (sqrt(3)-1)/2",
          abs(rmsd_cf - rmsd_analytic) < 1e-12,
          f"measured {rmsd_cf:.15f}, (sqrt(3)-1)/2 {rmsd_analytic:.15f}")

    print("equivariant features vs invariant descriptors")
    check("feature map is equivariant: f_l(R X) = D^l(R) f_l(X), l = 0..6",
          equi_dev < 1e-10, f"max dev over 5 rotations {equi_dev:.2e}")
    check("l = 0 features are identical",
          lab_diff[0] < 1e-14, f"||df_0|| {lab_diff[0]:.2e}")
    check("l = 1 is the lowest degree whose lab-frame features differ",
          lab_diff[0] < 1e-14 and lab_diff[1] > 0.1,
          f"||df_1|| {lab_diff[1]:.6f}")
    check("invariant power spectrum p_l is BLIND at every l = 0..6",
          power_dev < 1e-13, f"max_l |p_l^A - p_l^B| {power_dev:.2e}")
    check("addition theorem p_l = (2l+1)/4pi sum_jk P_l(cos theta_jk) holds",
          addition_dev < 1e-12, f"max dev {addition_dev:.2e}")
    check("l <= 1 does NOT separate: an explicit rotation aligns f_1 exactly",
          l1_residual < 1e-12 and l1_residual_direct < 1e-12,
          f"||D^1(R*) f_1^A - f_1^B|| {l1_residual:.2e} (direct {l1_residual_direct:.2e})")
    check("that same R* leaves l = 2 badly mismatched",
          l2_at_Rstar > 0.1, f"||D^2(R*) f_2^A - f_2^B|| {l2_at_Rstar:.6f}")
    check("l <= 2 DOES separate: certified lower bound on min_R mismatch is > 0",
          certified_lb > 1e-3,
          f"|db_112| {abs(b112_A - b112_B):.6f} / K {lip_K:.6f} = {certified_lb:.6f}")
    check("SO(3) search agrees with the certificate and finds no near-alignment",
          search_delta[2] > certified_lb and search_delta[2] > 0.5,
          f"search min (l<=2) {search_delta[2]:.6f} vs bound {certified_lb:.6f}")
    check("CG bispectrum separates A and B where the power spectrum cannot",
          bis_even_max > 0.05,
          f"max |db| over {len(bis_even)} even-parity triples {bis_even_max:.6f} "
          f"at (l1,l2,l3) = ({bispectrum[0]['l1']},{bispectrum[0]['l2']},{bispectrum[0]['l3']})")
    check("parity-odd bispectrum entries vanish for both (planar = achiral)",
          bis_odd_max < 1e-12,
          f"max |b_odd| {bis_odd_max:.2e} over {len(bis_odd)} triples")

    return npass, ntotal


def _write_json(npass: int, ntotal: int) -> None:
    out = {
        "meta": {
            "seed": SEED,
            "description": ("Invariant descriptors are provably blind: homometric "
                            "(Z-related) neighbour sets share every two- and three-body "
                            "quantity and the whole power spectrum, but differ at l=2 "
                            "equivariantly."),
            "lmax": LMAX,
            "n_so3_coarse": N_SO3_COARSE,
            "n_pass": npass,
            "n_total": ntotal,
        },
        "part_a_two_body_blindness": {
            "coords_equilateral": tri.tolist(),
            "coords_clustered": clu.tolist(),
            "angles_deg_equilateral": [0.0, 120.0, 240.0],
            "angles_deg_clustered": [0.0, 30.0, 60.0],
            "angular_gaps_deg_equilateral": [120.0, 120.0, 120.0],
            "angular_gaps_deg_clustered": [30.0, 30.0, 300.0],
            "sorted_radii_equilateral": r_tri.tolist(),
            "sorted_radii_clustered": r_clu.tolist(),
            "radii_bitwise_identical": radial_bitwise,
            "g2_mu_grid": mu_grid.tolist(),
            "g2_eta": G2_ETA,
            "g2_equilateral": g_tri.tolist(),
            "g2_clustered": g_clu.tolist(),
            "g2_bitwise_identical": g2_bitwise,
            "g2_max_abs_diff": float(np.max(np.abs(g_tri - g_clu))),
            "sw_lambda": SW_LAMBDA,
            "sw_cos_theta0": SW_COS_THETA0,
            "sw_energy_equilateral": e_tri,
            "sw_energy_clustered": e_clu,
            "sw_energy_ratio": float(e_clu / e_tri),
            "power_l1_equilateral": p1_tri,
            "power_l1_clustered": p1_clu,
        },
        "part_b_three_body_blindness": {
            "z12_set_A": list(Z_A),
            "z12_set_B": list(Z_B),
            "n_ring": N_RING,
            "coords_A": cA.tolist(),
            "coords_B": cB.tolist(),
            "angles_deg_A": [360.0 * k / N_RING for k in Z_A],
            "angles_deg_B": [360.0 * k / N_RING for k in Z_B],
            "interval_vector_A": iv_A,
            "interval_vector_B": iv_B,
            "sorted_centre_distances_A": rc_A.tolist(),
            "sorted_centre_distances_B": rc_B.tolist(),
            "sorted_pair_distances_A": d_A.tolist(),
            "sorted_pair_distances_B": d_B.tolist(),
            "homometry_max_abs_diff": homometry_dev,
            "sorted_bond_cosines_A": cos_A.tolist(),
            "sorted_bond_cosines_B": cos_B.tolist(),
            "bond_cosine_max_abs_diff": angle_dev,
            "same_dihedral_orbit": same_orbit,
            "rmsd_min_scan": rmsd_scan,
            "rmsd_min_closed_form": rmsd_cf,
            "rmsd_min_kabsch_o3": rmsd_kabsch,
            "rmsd_analytic_sqrt3_minus_1_over_2": rmsd_analytic,
            "rmsd_best_permutation": list(rmsd_cf_arg[0]),
            "rmsd_best_reflection": int(rmsd_cf_arg[1]),
            "rmsd_best_correlation_abs_Z": rmsd_cf_arg[2],
            "rmsd_scan_angles_rad": scan_angles[::5].tolist(),
            "rmsd_scan_curve": scan_curve[::5].tolist(),
        },
        "features": {
            "l_values": list(range(LMAX + 1)),
            "f_l_A": [fA[l].tolist() for l in range(LMAX + 1)],
            "f_l_B": [fB[l].tolist() for l in range(LMAX + 1)],
            "lab_frame_feature_diff_norm": lab_diff,
            "power_spectrum_A": pA,
            "power_spectrum_B": pB,
            "power_spectrum_abs_diff": [abs(a - b) for a, b in zip(pA, pB)],
            "power_spectrum_max_abs_diff": power_dev,
            "addition_theorem_max_dev": addition_dev,
            "equivariance_max_dev": equi_dev,
        },
        "rotation_quotient": {
            "R_star_aligning_l1": R_star.tolist(),
            "l1_residual_at_R_star": l1_residual,
            "l1_residual_direct_at_R_star": l1_residual_direct,
            "l2_residual_at_R_star": l2_at_Rstar,
            "search_min_delta_by_Lmax": search_delta,
            "search_is_upper_bound_only": True,
            "certified_lower_bound_L2": certified_lb,
            "certificate_b112_A": b112_A,
            "certificate_b112_B": b112_B,
            "certificate_lipschitz_K": lip_K,
            "certificate_cg_frobenius_norm": c112_fro,
            "inplane_angles_rad": ip_angles[::4].tolist(),
            "inplane_delta_L2": ip_delta2[::4].tolist(),
            "inplane_delta_L6": ip_delta6[::4].tolist(),
        },
        "bispectrum": {
            "entries": bispectrum,
            "max_abs_diff_even_parity": bis_even_max,
            "max_abs_value_odd_parity": bis_odd_max,
        },
        "checks": CHECKS,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    print("=" * 76)
    print("descriptor_blindness -- what invariant descriptors cannot see")
    print("=" * 76)
    npass, ntotal = _selfcheck()
    _write_json(npass, ntotal)
    print("-" * 76)
    print(f"wrote {RESULTS}  ({RESULTS.stat().st_size / 1024:.1f} KB)")
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
