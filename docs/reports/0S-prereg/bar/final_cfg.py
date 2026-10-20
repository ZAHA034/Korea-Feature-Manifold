"""Recommended configuration on all maps: s = mean slope on DEM low-passed at 2*250 m, sampled every 250 m;
h = excess relief on 90 m DEM sampled every 100 m.  Outcome kappa_ex (uniform; reference = Euclid x median min/km, links keep d_sym).
Inference: naive one-sided Spearman p and delete-one-block jackknife (16 blocks) t-test; both < 0.05."""
import glob
import sys
from multiprocessing import Pool

import numpy as np
from scipy import stats

import eval_jk as J


def work(args):
    fn, pop_elev = args
    import core
    import kw
    R = dict(np.load(fn, allow_pickle=True))
    seed = int(fn[-8:-4])
    W = kw.make_world(seed, pop_elev=pop_elev)
    a = R["adj"]
    E = R["E"]
    assert np.array_equal(W["E"], E)
    _, ca250 = core.barrier_indices(W, ds=0.25, lam_cut=0.5, edges=E[a])
    er90, _ = core.barrier_indices(W, ds=0.1, lam_cut=0.18, edges=E[a])
    s = ca250 / R["len_e"][a]
    h = er90
    n = len(R["deg"])
    blk = J.blocks(R)[a]
    out = {}
    for model in ("S", "R"):
        for c in (0.0, 0.01, 0.02, 0.05):
            key = f"L_{model}_{c:g}"
            if key not in R:
                continue
            L = R[key]
            v = np.median(L[a] / R["len_e"][a])
            Lg = np.where(a, v * R["len_e"], L)
            Dg = core.closure(n, E, Lg)
            kex = (R[f"ku_{model}_{c:g}"] - core.orc(n, E, Dg)[0])[a]
            for nm, z in (("s", s), ("h", h)):
                rho, pg = stats.spearmanr(kex, z, alternative="greater")
                _, se = J.jk(kex, z, blk)
                pj = stats.t.sf(rho / se, J.B - 1)
                pl = stats.spearmanr(kex, z, alternative="less")[1]
                out[(model, c, nm)] = (rho, pg, pj, pl)
            if c == 0:
                for cv in ("degterm", "bnd_any"):
                    out[(model, "cov", cv)] = stats.spearmanr(kex, R[cv][a])[0]
                out[(model, "cov", "loglen")] = stats.spearmanr(kex, np.log(R["len_e"][a]))[0]
    out["s_len"] = stats.spearmanr(s, R["len_e"][a])[0]
    out["h_len"] = stats.spearmanr(h, R["len_e"][a])[0]
    return out


if __name__ == "__main__":
    f1 = sorted(glob.glob("res/m*.npz"))
    f0 = sorted(glob.glob("res_n0/m*.npz"))
    with Pool(10) as p:
        r1 = p.map(work, [(f, 400.0) for f in f1])
        r0 = p.map(work, [(f, None) for f in f0])
    print("rho(s, length) median %.3f ; rho(h, length) median %.3f" % (np.median([r["s_len"] for r in r1]), np.median([r["h_len"] for r in r1])))
    for model in ("S", "R"):
        print(f"{model} null |rho(kex, cov)|: " + " ".join(f"{cv} {np.median([r[(model, 'cov', cv)] for r in r1]):+.3f}" for cv in ("degterm", "bnd_any", "loglen")))
    for lab, rows, model, c in (("N0 null", r0, "S", 0.0), ("N0 c=.02", r0, "S", 0.02), ("N0 c=.05", r0, "S", 0.05),
                                ("N1 null S", r1, "S", 0.0), ("N1 null R", r1, "R", 0.0),
                                ("N1 c=.01 S", r1, "S", 0.01), ("N1 c=.01 R", r1, "R", 0.01),
                                ("N1 c=.02 S", r1, "S", 0.02), ("N1 c=.02 R", r1, "R", 0.02),
                                ("N1 c=.05 S", r1, "S", 0.05), ("N1 c=.05 R", r1, "R", 0.05)):
        line = f"{lab:11s}"
        for nm in ("s", "h"):
            v = np.array([r[(model, c, nm)] for r in rows if (model, c, nm) in r])
            rho, pg, pj, pl = v.T
            line += (f" | {nm}: naive {np.mean(pg < .05):.2f} jk {np.mean(pj < .05):.2f} both {np.mean((pg < .05) & (pj < .05)):.2f} "
                     f"wrong-sign {np.mean(pl < .05):.2f} rho med {np.median(rho):+.3f} [p5 {np.percentile(rho, 5):+.3f}, p95 {np.percentile(rho, 95):+.3f}]")
        print(line)
