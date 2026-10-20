"""Triangular lattice with a straight 'ridge' cut: crossing edges get length 1 + Delta.
Uniform measure prediction for crossing edges as Delta -> inf: kappa -> 1/2 (c_x/d_x + c_y/d_y) = 1/3 (c = 2, d = 6).
Also decomposes gravity-weighted response into 'weights only' and 'lengths only'."""
import numpy as np

import core

nx_, ny_ = 16, 12
pts = []
for j in range(ny_):
    for i in range(nx_):
        pts.append((i + 0.5 * (j % 2), j * np.sqrt(3) / 2))
pts = np.array(pts)
n = len(pts)
E = []
for a in range(n):
    for b in range(a + 1, n):
        if abs(np.linalg.norm(pts[a] - pts[b]) - 1) < 1e-9:
            E.append((a, b))
E = np.array(E)
xcut = 7.75
cross = (pts[E[:, 0], 0] - xcut) * (pts[E[:, 1], 0] - xcut) < 0
deg = np.bincount(E.ravel(), minlength=n)
interior = (deg[E[:, 0]] == 6) & (deg[E[:, 1]] == 6)
# 'near' edges: not crossing, exactly one endpoint touches a crossing edge
cn = np.zeros(n, bool)
cn[E[cross].ravel()] = True
near_perp = ~cross & (cn[E[:, 0]] ^ cn[E[:, 1]]) & interior
near_par = ~cross & cn[E[:, 0]] & cn[E[:, 1]] & interior
sel_c = cross & interior
P = np.ones(n)
tau = 1.0
print("crossing interior edges", sel_c.sum(), "near-perp", near_perp.sum(), "near-parallel", near_par.sum())
print("Delta | uniform: cross  near_perp  near_par | gravity(tau=1): cross near_perp | grav weights-only cross | grav lengths-only cross")
for Dl in (0, 0.25, 0.5, 1, 2, 4, 8, 32, 128):
    L = np.where(cross, 1 + Dl, 1.0)
    D = core.closure(n, E, L)
    D0 = core.closure(n, E, np.ones(len(E)))
    ku, _ = core.orc(n, E, D)
    wg = P[E[:, 0]] * P[E[:, 1]] * np.exp(-D[E[:, 0], E[:, 1]] / tau)
    wg0 = P[E[:, 0]] * P[E[:, 1]] * np.exp(-D0[E[:, 0], E[:, 1]] / tau)
    kg, _ = core.orc(n, E, D, wg)
    kw_only, _ = core.orc(n, E, D0, wg)   # weights from barrier, lengths flat
    kl_only, _ = core.orc(n, E, D, wg0)   # lengths from barrier, weights flat
    print(f"{Dl:6g} | {ku[sel_c].mean():+.4f} {ku[near_perp].mean():+.4f} {ku[near_par].mean():+.4f} | "
          f"{kg[sel_c].mean():+.4f} {kg[near_perp].mean():+.4f} | {kw_only[sel_c].mean():+.4f} | {kl_only[sel_c].mean():+.4f}")
