"""Export reference spherical-harmonic values so the browser can check itself.

python/e3.py builds real spherical harmonics out of scipy's complex ones.
tutorial/js/e3.js re-implements them as explicit Cartesian polynomials, because
the browser has no scipy. Those are two genuinely independent implementations,
so pinning one against the other is a real test rather than a restatement.

This script writes results/sh_reference.json; tutorial/js/e3.js selfTest() loads
it and reports the largest disagreement.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from e3 import real_sh  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SEED = 20260729
N_DIRS = 96
LMAX = 3


def build() -> dict:
    rng = np.random.default_rng(SEED)
    dirs = rng.standard_normal((N_DIRS, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # A few exactly-placed directions too, so axis conventions are pinned down.
    special = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
    ])
    special /= np.linalg.norm(special, axis=1, keepdims=True)
    dirs = np.vstack([special, dirs])

    samples = []
    for d in dirs:
        entry = {"dir": [float(v) for v in d]}
        for l in range(LMAX + 1):
            entry[f"l{l}"] = [float(v) for v in real_sh(l, d)]
        samples.append(entry)

    return {
        "meta": {
            "seed": SEED,
            "lmax": LMAX,
            "n_directions": len(dirs),
            "description": "Real spherical harmonics from scipy, m ordered -l..+l, "
                           "for cross-checking the browser's Cartesian polynomials.",
        },
        "samples": samples,
    }


def _selfcheck(data: dict) -> tuple[int, int]:
    npass = ntotal = 0

    def check(name, ok, detail=""):
        nonlocal npass, ntotal
        ntotal += 1
        npass += bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    n = len(data["samples"])
    check("exported at least 100 directions", n >= 100, f"{n} directions")

    dims_ok = all(len(s[f"l{l}"]) == 2 * l + 1 for s in data["samples"] for l in range(LMAX + 1))
    check("every entry has 2l+1 components for l = 0..3", dims_ok)

    unit = max(abs(float(np.linalg.norm(s["dir"])) - 1.0) for s in data["samples"])
    check("directions are unit vectors", unit < 1e-12, f"max deviation {unit:.2e}")

    # l=0 is constant; a browser that got normalisation wrong would fail here.
    l0 = np.array([s["l0"][0] for s in data["samples"]])
    check("l=0 is the constant 1/(2 sqrt(pi))", np.allclose(l0, 0.5 / np.sqrt(np.pi), atol=1e-14),
          f"value {l0[0]:.12f}")

    # +z must light up exactly one l=1 component (the m=0 one, index 1).
    z = np.array(data["samples"][2]["l1"])
    check("l=1 at +z is (0, c, 0), fixing the (y, z, x) component order",
          abs(z[0]) < 1e-14 and abs(z[2]) < 1e-14 and z[1] > 0,
          f"[{z[0]:.2e}, {z[1]:.6f}, {z[2]:.2e}]")

    finite = all(np.isfinite(s[f"l{l}"]).all() for s in data["samples"] for l in range(LMAX + 1))
    check("all values finite", finite)

    return npass, ntotal


if __name__ == "__main__":
    print("=" * 68)
    print("export_js_reference.py -- scipy harmonics for the browser to check against")
    print("=" * 68)
    data = build()
    npass, ntotal = _selfcheck(data)
    out = ROOT / "results" / "sh_reference.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"  wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
    print("-" * 68)
    print(f"{npass}/{ntotal} PASS")
    raise SystemExit(0 if npass == ntotal else 1)
