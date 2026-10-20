"""Primary-test inference variants: naive one-sided Spearman p, effect-size floors, delete-one-block jackknife (16 blocks)."""
import glob
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.cluster.vq import kmeans2

B = 16
T_CRIT = stats.t.ppf(0.95, B - 1)


def blocks(R):
    pts = R["pts"]
    rng = np.random.default_rng(0)
    cent, lab = kmeans2(pts, B, seed=rng, minit="++")
    E = R["E"]
    mid = 0.5 * (pts[E[:, 0]] + pts[E[:, 1]])
    d = ((mid[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


def jk(y, z, blk):
    rho = stats.spearmanr(y, z)[0]
    ub = np.unique(blk)
    vals = np.array([stats.spearmanr(y[blk != b], z[blk != b])[0] for b in ub])
    nb = len(ub)
    se = np.sqrt((nb - 1) / nb * np.sum((vals - vals.mean()) ** 2))
    return rho, se


def per_map(args):
    fn, scen = args
    R = dict(np.load(fn, allow_pickle=True))
    a = R["adj"]
    blk = blocks(R)[a]
    out = {}
    for model, c in scen:
        key = f"ku_{model}_{c:g}"
        if key not in R:
            continue
        dk = (R[key] - R["kgeo_unif"])[a]
        for ix in ("ms", "er", "flag"):
            z = R[ix][a].astype(float)
            rho, pg = stats.spearmanr(dk, z, alternative="greater")
            r2, se = jk(dk, z, blk)
            out[(model, c, ix)] = (rho, pg, se)
    return out


def summarize(label, rows, model, c):
    print(f"{label:34s}", end="")
    for ix in ("ms", "er", "flag"):
        v = [r[(model, c, ix)] for r in rows if (model, c, ix) in r]
        rho = np.array([x[0] for x in v])
        p = np.array([x[1] for x in v])
        se = np.array([x[2] for x in v])
        naive = np.mean(p < 0.05)
        f08 = np.mean((p < 0.05) & (rho >= 0.08))
        f10 = np.mean((p < 0.05) & (rho >= 0.10))
        jkr = np.mean(rho / se > T_CRIT)
        naive_se = 1 / np.sqrt(len(rho) and 660)
        print(f" | {ix}: naive {naive:.2f} floor.08 {f08:.2f} floor.10 {f10:.2f} jk16 {jkr:.2f} (jk SE med {np.median(se):.3f}, rho med {np.median(rho):+.3f})", end="")
    print()


if __name__ == "__main__":
    scen_n1 = [("S", 0.0), ("S", 0.01), ("S", 0.02), ("S", 0.05), ("R", 0.0), ("R", 0.01), ("R", 0.02), ("R", 0.05)]
    scen_n0 = [("S", 0.0), ("S", 0.02), ("S", 0.05)]
    f1 = sorted(glob.glob("res/m*.npz"))
    f0 = sorted(glob.glob("res_n0/m*.npz"))
    with Pool(10) as p:
        r1 = p.map(per_map, [(f, scen_n1) for f in f1])
        r0 = p.map(per_map, [(f, scen_n0) for f in f0])
    print(f"t crit (one-sided .05, df={B-1}) = {T_CRIT:.3f}; naive SE ~ 1/sqrt(n) = {1/np.sqrt(660):.3f}")
    summarize("N0 pure null (S)", r0, "S", 0.0)
    summarize("N0 + travel c=0.02 (S)", r0, "S", 0.02)
    summarize("N0 + travel c=0.05 (S)", r0, "S", 0.05)
    for model in ("S", "R"):
        for c in (0.0, 0.01, 0.02, 0.05):
            summarize(f"N1 {'null' if c == 0 else 'c=%g' % c} ({model})", r1, model, c)
