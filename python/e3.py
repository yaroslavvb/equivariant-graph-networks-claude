"""Real O(3) representation theory, built from scratch and verified numerically.

Nothing here is looked up from a table. The three objects NequIP needs --

    real spherical harmonics   Y_l : S^2 -> R^(2l+1)
    Wigner D matrices          D^l(R), the (2l+1)-dim irrep of SO(3)
    Clebsch-Gordan tensors     C^(l1 l2 l3), the unique equivariant bilinear map

-- are each *constructed* by a short numerical recipe and then checked against
the property that defines them. The recipes are the point: they are what the
tutorial's chapter 3 walks through.

    Y_l         : real combinations of scipy's complex sph_harm_y
    D^l(R)      : solve Y_l(R r) = D^l(R) Y_l(r) in the least-squares sense
                  over a batch of sample directions -- exact up to float noise
    C^(l1l2l3)  : the null space of  [kron(D1,D2,D3) - I]  stacked over a few
                  generic rotations. One-dimensional, so an SVD reads it off at
                  machine precision, and sigma_2/sigma_1 certifies the dimension.

Conventions
-----------
Component order within degree l is m = -l, ..., 0, ..., +l.
Parity of the degree-l spherical harmonic is (-1)^l, so an O(3) irrep is
written (l, p) with p in {+1, -1}: `1o` is a vector, `1e` a pseudovector.
"""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y

__all__ = [
    "real_sh",
    "wigner_D",
    "random_rotation",
    "rotation_matrix",
    "clebsch_gordan",
    "cg_allowed",
    "tensor_product",
    "irrep_dim",
]


# --------------------------------------------------------------------------
# 1. Real spherical harmonics
# --------------------------------------------------------------------------

def real_sh(l: int, xyz: np.ndarray) -> np.ndarray:
    """Real spherical harmonics of degree ``l`` evaluated at directions ``xyz``.

    ``xyz`` has shape (..., 3) and need not be normalised. Returns shape
    (..., 2l+1) with components ordered m = -l ... +l, orthonormal on the
    sphere (integral of Y_lm Y_l'm' over S^2 equals delta).

    Built from scipy's complex harmonics by the standard real transform:

        m > 0:  sqrt(2) (-1)^m Re Y_l^m
        m = 0:  Y_l^0
        m < 0:  sqrt(2) (-1)^m Im Y_l^|m|
    """
    xyz = np.asarray(xyz, dtype=float)
    r = np.linalg.norm(xyz, axis=-1, keepdims=True)
    unit = xyz / np.where(r == 0.0, 1.0, r)
    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]

    theta = np.arccos(np.clip(z, -1.0, 1.0))   # polar angle from +z
    phi = np.arctan2(y, x)                     # azimuth in the xy-plane

    out = np.empty(unit.shape[:-1] + (2 * l + 1,), dtype=float)
    for m in range(-l, l + 1):
        am = abs(m)
        Y = sph_harm_y(l, am, theta, phi)      # scipy >= 1.15 signature
        if m > 0:
            val = np.sqrt(2.0) * (-1.0) ** m * Y.real
        elif m == 0:
            val = Y.real
        else:
            val = np.sqrt(2.0) * (-1.0) ** m * Y.imag
        out[..., m + l] = val
    return out


def irrep_dim(l: int) -> int:
    return 2 * l + 1


# --------------------------------------------------------------------------
# 2. Rotations and Wigner D matrices
# --------------------------------------------------------------------------

def rotation_matrix(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation by ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform rotation via QR of a Gaussian matrix, det fixed to +1."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def wigner_D(l: int, R: np.ndarray, n_samples: int | None = None,
             seed: int = 0) -> np.ndarray:
    """The (2l+1)x(2l+1) real Wigner matrix defined by Y_l(R r) = D Y_l(r).

    Solved, not tabulated: evaluate both sides on a batch of sample directions
    and least-squares the linear map. With more samples than 2l+1 the system is
    overdetermined, so a small residual is itself the check that a degree-l
    subspace really is closed under rotation.
    """
    R = np.asarray(R, dtype=float)
    n = n_samples if n_samples is not None else 8 * (2 * l + 1)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    A = real_sh(l, dirs)                 # (n, 2l+1)  features before rotating
    B = real_sh(l, dirs @ R.T)           # (n, 2l+1)  features after rotating
    # B = A D^T  =>  D^T = pinv(A) B
    DT, *_ = np.linalg.lstsq(A, B, rcond=None)
    return DT.T


def wigner_D_residual(l: int, R: np.ndarray, seed: int = 12345) -> float:
    """max |Y_l(R r) - D^l(R) Y_l(r)| on fresh directions not used in the fit."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((512, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    D = wigner_D(l, R)
    lhs = real_sh(l, dirs @ np.asarray(R, dtype=float).T)
    rhs = real_sh(l, dirs) @ D.T
    return float(np.max(np.abs(lhs - rhs)))


# --------------------------------------------------------------------------
# 3. Clebsch-Gordan tensors by null space
# --------------------------------------------------------------------------

def cg_allowed(l1: int, l2: int, l3: int,
               p1: int = 1, p2: int = 1, p3: int = 1) -> bool:
    """O(3) selection rule: triangle inequality on l, product rule on parity."""
    return (abs(l1 - l2) <= l3 <= l1 + l2) and (p1 * p2 == p3)


def clebsch_gordan(l1: int, l2: int, l3: int, n_rot: int = 6,
                   seed: int = 0, return_diagnostics: bool = False):
    """The unique (up to scale) invariant tensor in V_l1 (x) V_l2 (x) V_l3.

    C is invariant iff  (D1 (x) D2 (x) D3) vec(C) = vec(C)  for every rotation.
    Impose that for a handful of *generic* rotations -- generic rotations
    generate a dense subgroup, so the null space of the stacked constraint is
    already the exact invariant subspace -- and read it off with an SVD.

    Returns an array of shape (2l1+1, 2l2+1, 2l3+1) with Frobenius norm 1 and a
    deterministic sign (largest-magnitude entry made positive).

    With ``return_diagnostics``, also returns ``(sigma_min, sigma_second)``.
    sigma_min ~ 1e-15 says an invariant tensor exists; sigma_second of order 1
    says it is unique, i.e. the multiplicity really is one.
    """
    d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_rot):
        R = random_rotation(rng)
        K = np.kron(np.kron(wigner_D(l1, R), wigner_D(l2, R)), wigner_D(l3, R))
        blocks.append(K - np.eye(d1 * d2 * d3))
    M = np.vstack(blocks)

    # Smallest right-singular vector spans the invariant line.
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    C = Vt[-1].reshape(d1, d2, d3)

    flat = C.reshape(-1)
    if flat[np.argmax(np.abs(flat))] < 0:
        C = -C
    C = C / np.linalg.norm(C)

    if return_diagnostics:
        return C, (float(S[-1]), float(S[-2]))
    return C


def tensor_product(C: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Contract features u (degree l1) and v (degree l2) into degree l3."""
    return np.einsum("abc,a,b->c", C, u, v)


# --------------------------------------------------------------------------
# 4. Self-check
# --------------------------------------------------------------------------

def _selfcheck(verbose: bool = True) -> tuple[int, int]:
    """Verify every construction above. Returns (n_pass, n_total)."""
    rng = np.random.default_rng(7)
    npass = ntotal = 0

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        npass += bool(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("real spherical harmonics")
    dirs = rng.standard_normal((4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for l in range(4):
        Y = real_sh(l, dirs)
        # Monte-Carlo orthonormality: <Y_lm Y_lm'> over the sphere = delta/(4 pi)
        G = (Y.T @ Y) / len(dirs) * (4 * np.pi)
        err = np.max(np.abs(G - np.eye(2 * l + 1)))
        check(f"l={l} orthonormal on S^2 (MC, 4000 dirs)", err < 0.15, f"max dev {err:.3f}")

    # l=1 really is the vector representation, in (y, z, x) order.
    v = rng.standard_normal((6, 3))
    Y1 = real_sh(1, v)
    unit = v / np.linalg.norm(v, axis=1, keepdims=True)
    expect = np.sqrt(3 / (4 * np.pi)) * unit[:, [1, 2, 0]]
    check("l=1 equals sqrt(3/4pi) * (y, z, x)", np.allclose(Y1, expect, atol=1e-12),
          f"max dev {np.max(np.abs(Y1 - expect)):.2e}")

    print("Wigner D matrices")
    for l in range(4):
        R = random_rotation(rng)
        res = wigner_D_residual(l, R)
        check(f"l={l} Y_l(R r) = D Y_l(r) on held-out directions", res < 1e-10,
              f"max residual {res:.2e}")
        D = wigner_D(l, R)
        orth = np.max(np.abs(D.T @ D - np.eye(2 * l + 1)))
        check(f"l={l} D orthogonal", orth < 1e-10, f"max dev {orth:.2e}")

    R1, R2 = random_rotation(rng), random_rotation(rng)
    for l in range(4):
        lhs = wigner_D(l, R1 @ R2)
        rhs = wigner_D(l, R1) @ wigner_D(l, R2)
        err = np.max(np.abs(lhs - rhs))
        check(f"l={l} homomorphism D(R1 R2) = D(R1) D(R2)", err < 1e-10, f"max dev {err:.2e}")

    print("Clebsch-Gordan tensors")
    for (l1, l2, l3) in [(1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 2, 2), (3, 1, 2)]:
        C, (s_min, s_2) = clebsch_gordan(l1, l2, l3, return_diagnostics=True)
        check(f"({l1},{l2})->{l3} invariant subspace is 1-D",
              s_min < 1e-9 and s_2 > 1e-3, f"sigma_min {s_min:.2e}, sigma_2 {s_2:.2e}")

        # The defining equivariance property, on fresh random data and rotations.
        worst = 0.0
        for _ in range(8):
            R = random_rotation(rng)
            u = rng.standard_normal(2 * l1 + 1)
            w = rng.standard_normal(2 * l2 + 1)
            D1, D2, D3 = wigner_D(l1, R), wigner_D(l2, R), wigner_D(l3, R)
            rotated_in = tensor_product(C, D1 @ u, D2 @ w)
            rotated_out = D3 @ tensor_product(C, u, w)
            scale = max(np.linalg.norm(rotated_out), 1e-12)
            worst = max(worst, np.max(np.abs(rotated_in - rotated_out)) / scale)
        check(f"({l1},{l2})->{l3} equivariant: C(Du, Dv) = D C(u, v)", worst < 1e-9,
              f"max relative dev {worst:.2e}")

    # The l=1 x l=1 -> l=1 coupling is the cross product, up to scale.
    C111 = clebsch_gordan(1, 1, 1)
    a, b = rng.standard_normal(3), rng.standard_normal(3)
    ya, yb = real_sh(1, a), real_sh(1, b)
    got = tensor_product(C111, ya, yb)                    # in (y, z, x) order
    want = np.cross(a, b)[[1, 2, 0]]
    ratio = got / want
    check("(1,1)->1 is the cross product up to one global scale",
          np.allclose(ratio, ratio[0], rtol=1e-9),
          f"ratios {np.array2string(ratio, precision=6)}")

    # The l=1 x l=1 -> l=0 coupling is the dot product, up to scale. Test that
    # one single constant reproduces it across many independent vector pairs.
    C110 = clebsch_gordan(1, 1, 0)
    ratios = []
    for _ in range(20):
        p, q = rng.standard_normal(3), rng.standard_normal(3)
        got0 = tensor_product(C110, real_sh(1, p), real_sh(1, q))[0]
        want0 = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))
        ratios.append(got0 / want0)
    ratios = np.array(ratios)
    check("(1,1)->0 is the dot product up to one global scale",
          np.allclose(ratios, ratios[0], rtol=1e-9),
          f"constant {ratios[0]:.9f}, spread {np.ptp(ratios):.2e}")

    # Parity selection rule: 1o (x) 1o -> 1o is forbidden, 1o (x) 1o -> 1e allowed.
    check("parity rule forbids 1o (x) 1o -> 1o", not cg_allowed(1, 1, 1, -1, -1, -1))
    check("parity rule allows 1o (x) 1o -> 1e", cg_allowed(1, 1, 1, -1, -1, +1))
    check("triangle rule forbids (1,1) -> 3", not cg_allowed(1, 1, 3))

    return npass, ntotal


if __name__ == "__main__":
    print("=" * 68)
    print("e3.py self-check -- O(3) representation theory built from scratch")
    print("=" * 68)
    npass, ntotal = _selfcheck()
    print("-" * 68)
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
