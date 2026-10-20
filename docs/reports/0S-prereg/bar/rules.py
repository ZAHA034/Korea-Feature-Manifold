import glob
import json

import numpy as np
from scipy import stats

C_LIST = [0.0, 0.01, 0.02, 0.05, 0.1]
files = sorted(glob.glob("res/m*.npz"))
M = [dict(np.load(f, allow_pickle=True)) for f in files]
TOP = 20
IDX = ("er", "ca", "ph", "ms", "flag", "flag_all")


def resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def hyp(top, fl):
    x = int((top & fl).sum())
    return stats.hypergeom.sf(x - 1, len(fl), int(fl.sum()), int(top.sum())), x


def topmask(v, k=TOP):
    m = np.zeros(len(v), bool)
    m[np.argsort(v, kind="stable")[:k]] = True
    return m


def one_map(R, model, key, c):
    a = R["adj"]
    k = R[f"{key}_{model}_{c:g}"][a]
    kg = R[f"kgeo_{'unif' if key == 'ku' else 'grav'}"][a]
    deg_cov = np.c_[R["degterm"][a], R["bnd_any"][a]]
    len_cov = np.c_[deg_cov, np.log(R["len_e"][a])]
    res1 = resid(k, deg_cov)
    res2 = resid(k, len_cov)
    dk = k - kg
    out = {}
    t0, t1, t2 = topmask(k), topmask(res1), topmask(res2)
    out["overlap_D"] = int((t0 & t1).sum())
    out["overlap_D2"] = int((t0 & t2).sum())
    for nm in ("flag", "flag_all"):
        fl = R[nm][a].astype(bool)
        p, x = hyp(t0, fl)
        out[f"A_{nm}"] = x >= 5
        out[f"B_{nm}"] = p < 0.05
        out[f"x_{nm}"] = x
        pE, xE = hyp(t1, fl)
        out[f"E_{nm}"] = pE < 0.05
        pE2, xE2 = hyp(t2, fl)
        out[f"E2_{nm}"] = pE2 < 0.05
        pG, xG = hyp(topmask(dk), fl)
        out[f"Etop_dk_{nm}"] = pG < 0.05
    for nm in IDX:
        z = R[nm][a].astype(float)
        ok = np.isfinite(z)
        for tag, y in (("C", k), ("D", res1), ("D2", res2), ("G", dk)):
            rho, pl = stats.spearmanr(y[ok], z[ok], alternative="less")
            pg = stats.spearmanr(y[ok], z[ok], alternative="greater")[1]
            out[f"{tag}_{nm}"] = pl < 0.05
            out[f"{tag}pos_{nm}"] = pg < 0.05
            out[f"{tag}rho_{nm}"] = rho
    return out


table = {}
for model in ("S", "R"):
    for meas, key in (("unif", "ku"), ("grav", "kg")):
        for c in C_LIST:
            rows = [one_map(R, model, key, c) for R in M]
            agg = {}
            for kk in rows[0]:
                v = np.array([r[kk] for r in rows], float)
                agg[kk] = float(v.mean())
            table[f"{model}_{meas}_{c:g}"] = agg

json.dump({"n_maps": len(M), "table": table}, open("rules.json", "w"), indent=1)

cols = ["A_flag", "B_flag", "A_flag_all", "B_flag_all", "E_flag", "E2_flag", "Etop_dk_flag",
        "C_er", "C_ca", "C_ph", "C_ms", "C_flag", "D_er", "D_ca", "D_ph", "D_ms", "D_flag",
        "D2_er", "D2_ca", "D2_ph", "D2_ms", "D2_flag", "G_er", "G_ph", "G_flag"]
print(f"maps={len(M)}; rejection rate (c=0 -> false positive rate; c>0 -> power); one-sided alpha=0.05 (H1: more negative)")
for model in ("S", "R"):
    for meas in ("unif", "grav"):
        print(f"\n--- model {model}, measure {meas}")
        print("c      " + " ".join(f"{c_:>8s}" for c_ in cols))
        for c in C_LIST:
            t = table[f"{model}_{meas}_{c:g}"]
            print(f"{c:<6g} " + " ".join(f"{t[c_]:8.2f}" for c_ in cols))
        print("positive-direction rejection (rho>0, p<.05): " + "  ".join(
            f"c={c:g}: C_er {table[f'{model}_{meas}_{c:g}']['Cpos_er']:.2f} D2_er {table[f'{model}_{meas}_{c:g}']['D2pos_er']:.2f} "
            f"D2_flag {table[f'{model}_{meas}_{c:g}']['D2pos_flag']:.2f} G_flag {table[f'{model}_{meas}_{c:g}']['Gpos_flag']:.2f}" for c in C_LIST))
        print("mean rho: " + "  ".join(
            f"c={c:g}: C_er {table[f'{model}_{meas}_{c:g}']['Crho_er']:+.3f} D_er {table[f'{model}_{meas}_{c:g}']['Drho_er']:+.3f} "
            f"D2_er {table[f'{model}_{meas}_{c:g}']['D2rho_er']:+.3f} D2_flag {table[f'{model}_{meas}_{c:g}']['D2rho_flag']:+.3f} "
            f"D2_ph {table[f'{model}_{meas}_{c:g}']['D2rho_ph']:+.3f}" for c in C_LIST))
        print("top20 overlap after control (D: deg+bnd, D2: +log len): " + "  ".join(
            f"c={c:g}: {table[f'{model}_{meas}_{c:g}']['overlap_D']:.1f}/{table[f'{model}_{meas}_{c:g}']['overlap_D2']:.1f}" for c in C_LIST)
            + " | mean x_flag in top20: " + " ".join(f"{table[f'{model}_{meas}_{c:g}']['x_flag']:.2f}" for c in C_LIST))
