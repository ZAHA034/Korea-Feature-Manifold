import glob

import numpy as np
from scipy import stats

files = sorted(glob.glob("res/m*.npz"))
M = [dict(np.load(f, allow_pickle=True)) for f in files]
IDX = ("er", "ca", "ph", "ms")


def auc(score, lab):
    r = stats.rankdata(score)
    n1 = lab.sum()
    n0 = len(lab) - n1
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


agg = {}
for R in M:
    a = R["adj"]
    L = R["len_e"][a]
    fl = R["flag"][a].astype(bool)
    for nm in IDX:
        z = R[nm][a]
        ok = np.isfinite(z)
        agg.setdefault(f"{nm}_len_all", []).append(stats.spearmanr(z[ok], L[ok])[0])
        nc = ok & ~fl
        agg.setdefault(f"{nm}_len_noncross", []).append(stats.spearmanr(z[nc], L[nc])[0])
        agg.setdefault(f"{nm}_auc_flag", []).append(auc(z[ok], fl[ok]))
        for model in ("S", "R"):
            ex = R[f"L_{model}_0.05"][a] / R[f"L_{model}_0"][a] - 1
            agg.setdefault(f"{nm}_excess_{model}", []).append(stats.spearmanr(z[ok], ex[ok])[0])
            exa = R[f"L_{model}_0.05"][a] - R[f"L_{model}_0"][a]
            agg.setdefault(f"{nm}_excessabs_{model}", []).append(stats.spearmanr(z[ok], exa[ok])[0])
    agg.setdefault("er_ca", []).append(stats.spearmanr(R["er"][a], R["ca"][a])[0])
    ok = np.isfinite(R["ph"][a])
    agg.setdefault("er_ph", []).append(stats.spearmanr(R["er"][a][ok], R["ph"][a][ok])[0])
    agg.setdefault("ph_nan", []).append(np.mean(~ok))
    agg.setdefault("ph_zero_frac", []).append(np.mean(R["ph"][a][ok] <= 0))
    agg.setdefault("er_neg_frac", []).append(np.mean(R["er"][a] <= 0))
for k, v in agg.items():
    print(f"{k:22s} median {np.median(v):+.3f}  [p10 {np.percentile(v, 10):+.3f}, p90 {np.percentile(v, 90):+.3f}]")
