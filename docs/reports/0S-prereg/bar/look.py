import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kw

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
t0 = time.time()
W = kw.make_world(seed)
print(f"world {seed}: {time.time()-t0:.1f}s  n={W['n']}  edges={len(W['E'])}  island links={np.sum(W['etype']=='island')}")
deg = W["deg"]
print("degree: mean %.2f  min %d  max %d  hist %s" % (deg.mean(), deg.min(), deg.max(), np.bincount(deg)))
area = np.bincount(W["lab"][W["lab"] >= 0], minlength=W["n"])
print("cell area km2: min %d  p10 %d  median %d  p90 %d  max %d" % tuple(np.percentile(area, [0, 10, 50, 90, 100])))
print("coastal cells %d  border cells %d" % (W["coast"].sum(), W["border"].sum()))
E = W["E"]
pts = W["pts"]
cr = np.array([kw.seg_cross_poly(pts[u], pts[v], W["ridge"].P) > 0 for u, v in E])
crm = np.array([any(kw.seg_cross_poly(pts[u], pts[v], m.P) > 0 for m in W["minors"]) for u, v in E])
adj = W["etype"] == "adj"
print("crossing main ridge: %d of %d adj edges (%.1f%%); minor ridges %d; union %d (%.1f%%)" % (
    cr[adj].sum(), adj.sum(), 100 * cr[adj].mean(), crm[adj].sum(), (cr | crm)[adj].sum(), 100 * (cr | crm)[adj].mean()))
z = W["zr"]
print("DEM raster: land max %.0f  p50 %.0f  p90 %.0f" % (z[W["land"]].max(), np.median(z[W["land"]]), np.percentile(z[W["land"]], 90)))
print("seed elevations: p50 %.0f p90 %.0f max %.0f" % tuple(np.percentile(W["z_fn"](pts[:, 0], pts[:, 1]), [50, 90, 100])))

fig, ax = plt.subplots(1, 2, figsize=(16, 10))
ext = [kw.X0, kw.X1, kw.Y0, kw.Y1]
zz = np.where(W["land"], z, np.nan)
ax[0].imshow(zz, origin="lower", extent=ext, cmap="terrain", vmin=-200, vmax=1300)
ax[0].plot(W["ridge"].P[:, 0], W["ridge"].P[:, 1], "r-", lw=1)
for m in W["minors"]:
    ax[0].plot(m.P[:, 0], m.P[:, 1], "m--", lw=1)
ax[0].plot(pts[:, 0], pts[:, 1], "k.", ms=3)
ax[0].set_title("DEM + ridge crest (red) + minor ridges (magenta)")
lab = W["lab"].astype(float)
lab[lab < 0] = np.nan
rng = np.random.default_rng(1)
perm = rng.permutation(W["n"])
ax[1].imshow(np.where(np.isnan(lab), np.nan, perm[np.nan_to_num(lab).astype(int)]), origin="lower", extent=ext, cmap="tab20", interpolation="nearest")
for (u, v), t, c in zip(E, W["etype"], cr):
    col = "r" if c else ("b" if t == "island" else "k")
    ax[1].plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], "-", color=col, lw=0.6 if col == "k" else 1.5)
ax[1].plot(pts[W["coast"], 0], pts[W["coast"], 1], "co", ms=3)
ax[1].plot(pts[W["border"], 0], pts[W["border"], 1], "ys", ms=4)
ax[1].set_title("cells, G_adj (red = crosses ridge, blue = island links), cyan = coastal, yellow = border")
for a in ax:
    a.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"look_{seed}.png", dpi=70)
print("saved")
