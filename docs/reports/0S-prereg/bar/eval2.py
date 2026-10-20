"""Compare null scenarios N0 (pure) vs N1 (population avoids terrain) and channels; rho distributions for the primary test."""
import glob
import json
import sys

import numpy as np
from scipy import stats

IDX = ("ms", "er", "ph", "ca", "flag", "flag_all")


def resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def topmask(v, k=20):
    m = np.zeros(len(v), bool)
    m[np.argsort(v, kind="stable")[:k]] = True
    return m


def evaluate(d, model, c):
    files = sorted(glob.glob(f"{d}/m*.npz"))
    acc = {}
    for f in files:
        R = dict(np.load(f, allow_pickle=True))
        a = R["adj"]
        fl = R["flag"][a].astype(bool)
        deg_cov = np.c_[R["degterm"][a], R["bnd_any"][a]]
        len_cov = np.c_[deg_cov, np.log(R["len_e"][a])]
        for meas, key in (("unif", "ku"), ("grav", "kg")):
            k = R[f"{key}_{model}_{c:g}"][a]
            dk = k - R[f"kgeo_{meas}"][a]
            ys = {"raw": k, "D": resid(k, deg_cov), "D2": resid(k, len_cov), "dk": dk}
            for yn, y in ys.items():
                t = topmask(y)
                x = int((t & fl).sum())
                acc.setdefault((meas, yn, "A5"), []).append(x >= 5)
                acc.setdefault((meas, yn, "hyper"), []).append(stats.hypergeom.sf(x - 1, len(fl), int(fl.sum()), 20) < 0.05)
                # most POSITIVE 20 for dk
                tp = topmask(-y)
                xp = int((tp & fl).sum())
                acc.setdefault((meas, yn, "hyper_pos"), []).append(stats.hypergeom.sf(xp - 1, len(fl), int(fl.sum()), 20) < 0.05)
                acc.setdefault((meas, yn, "x_pos"), []).append(xp)
                for ix in IDX:
                    z = R[ix][a].astype(float)
                    ok = np.isfinite(z)
                    rho, pl = stats.spearmanr(y[ok], z[ok], alternative="less")
                    pg = stats.spearmanr(y[ok], z[ok], alternative="greater")[1]
                    acc.setdefault((meas, yn, ix, "neg"), []).append(pl < 0.05)
                    acc.setdefault((meas, yn, ix, "pos"), []).append(pg < 0.05)
                    acc.setdefault((meas, yn, ix, "rho"), []).append(rho)
                    acc.setdefault((meas, yn, ix, "pos_t10"), []).append((pg < 0.05) and (rho >= 0.10))
    return acc, len(files)


if __name__ == "__main__":
    out = {}
    for d, model, c, lab in (("res_n0", "S", 0.0, "N0 pure null"), ("res", "S", 0.0, "N1 demographic only"),
                             ("res", "R", 0.0, "N1 demographic only (R)"),
                             ("res_n0", "S", 0.02, "N0 + travel 0.02"), ("res_n0", "S", 0.05, "N0 + travel 0.05"),
                             ("res", "S", 0.01, "N1 + travel 0.01"), ("res", "S", 0.02, "N1 + travel 0.02"),
                             ("res", "R", 0.01, "N1 + travel 0.01 (R)"), ("res", "R", 0.02, "N1 + travel 0.02 (R)")):
        acc, nm = evaluate(d, model, c)
        print(f"\n### {lab}  [{d}, model {model}, c={c}]  maps={nm}")
        for meas in ("unif", "grav"):
            for yn in ("raw", "D", "D2", "dk"):
                s = (f"{meas:4s} {yn:3s}: A5 {np.mean(acc[(meas, yn, 'A5')]):.2f} hyperNeg {np.mean(acc[(meas, yn, 'hyper')]):.2f} "
                     f"hyperPos {np.mean(acc[(meas, yn, 'hyper_pos')]):.2f} (xPos {np.mean(acc[(meas, yn, 'x_pos')]):.2f}) | ")
                s += " ".join(f"{ix}: neg {np.mean(acc[(meas, yn, ix, 'neg')]):.2f} pos {np.mean(acc[(meas, yn, ix, 'pos')]):.2f} "
                              f"rho {np.mean(acc[(meas, yn, ix, 'rho')]):+.3f}[{np.percentile(acc[(meas, yn, ix, 'rho')], 5):+.3f},{np.percentile(acc[(meas, yn, ix, 'rho')], 95):+.3f}]"
                              f" pos&rho>=.1 {np.mean(acc[(meas, yn, ix, 'pos_t10')]):.2f};" for ix in IDX)
                print(s)
        out[lab] = {"|".join(k): float(np.mean(v)) for k, v in acc.items()}
    json.dump(out, open("eval2.json", "w"), indent=0)
