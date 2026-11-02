"""Minimal CertiSpec example on synthetic patient scores.

Run:  python examples/quickstart.py
"""
import numpy as np
from certispec import certify

rng = np.random.default_rng(0)
s0 = rng.normal(0.0, 1.0, 150)   # calibration scores of negative patients
s1 = rng.normal(3.0, 1.0, 150)   # calibration scores of positive patients

alphas = [0.05, 0.10]            # maximum tolerated risk (1 - NPV or 1 - PPV)
pis = [0.10, 0.50]               # target prevalences
res = certify(s0, s1, "os", pis, alphas, rng)

for (alpha, pi), r in res.items():
    neg = f"S <= {r['l_neg']:.2f}" if np.isfinite(r["l_neg"]) else "not certified"
    pos = f"S >  {r['l_pos']:.2f}" if np.isfinite(r["l_pos"]) else "not certified"
    print(f"alpha={alpha:.2f} pi={pi:.2f} | negative: {neg:16s} | positive: {pos:16s} | "
          f"decided: {r['frac']:.2f}")
