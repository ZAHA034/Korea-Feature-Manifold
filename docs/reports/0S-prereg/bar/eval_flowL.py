"""Flow-weighted ORC, implementable reference: same observed weights, lengths swapped to geometric (median-speed scaled).
dk_L = kappa(d_sym, w_obs) - kappa(d_geo, w_obs).  Also kappa_unif reference built the same way (median-speed scaling)."""
import glob
from multiprocessing import Pool

import numpy as np
from scipy import stats


def per_map(fn):
    import core
    R = dict(np.load(fn, allow_pickle=True))
    E = R["E"]
    n = len(R["deg"])
    a = R["adj"]
    P = R["P"]
    out = {}
    for model in ("S", "R"):
        for c in (0.0, 0.01, 0.02, 0.05):
            tag = f"{model}_{c:g}"
            L = R[f"L_{tag}"]
            D = core.closure(n, E, L)
            w_obs = P[E[:, 0]] * P[E[:, 1]] * np.exp(-D[E[:, 0], E[:, 1]] / 30.0)
            v = np.median(L[a] / R["len_e"][a])  # median minutes per km on adjacency edges
            Lg = np.where(a, v * R["len_e"], L)  # island links keep their d_sym
            Dg = core.closure(n, E, Lg)
            kref_g = core.orc(n, E, Dg, w_obs)[0]
            kref_u = core.orc(n, E, Dg)[0]
            dkL = (R[f"kg_{tag}"] - kref_g)[a]
            dkU = (R[f"ku_{tag}"] - kref_u)[a]
            fl = R["flag"][a].astype(bool)
            out[(tag, "dkL_cross_mean")] = dkL[fl].mean() - dkL[~fl].mean()
            out[(tag, "dkU_cross_mean")] = dkU[fl].mean() - dkU[~fl].mean()
            for nm, y in (("flowL", dkL), ("unifV", dkU)):
                for ix in ("ms", "er", "flag"):
                    z = R[ix][a].astype(float)
                    rho, pg = stats.spearmanr(y, z, alternative="greater")
                    pl = stats.spearmanr(y, z, alternative="less")[1]
                    out[(tag, nm, ix, "pos")] = pg < 0.05
                    out[(tag, nm, ix, "neg")] = pl < 0.05
                    out[(tag, nm, ix, "rho")] = rho
                if c == 0:
                    for cv in ("degterm", "bnd_any", "len_e"):
                        out[(tag, nm, cv, "rho0")] = stats.spearmanr(y, R[cv][a])[0]
    return out


if __name__ == "__main__":
    files = sorted(glob.glob("res/m*.npz"))
    with Pool(10) as p:
        rows = p.map(per_map, files)
    keys = rows[0].keys()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for model in ("S", "R"):
        for nm in ("flowL", "unifV"):
            print(f"\n{model} {nm}: null rho with degterm {agg[(f'{model}_0', nm, 'degterm', 'rho0')]:+.3f} bnd {agg[(f'{model}_0', nm, 'bnd_any', 'rho0')]:+.3f} len {agg[(f'{model}_0', nm, 'len_e', 'rho0')]:+.3f}")
            for c in (0.0, 0.01, 0.02, 0.05):
                tag = f"{model}_{c:g}"
                print(f"  c={c:<5g} " + " ".join(f"{ix}: pos {agg[(tag, nm, ix, 'pos')]:.2f} neg {agg[(tag, nm, ix, 'neg')]:.2f} rho {agg[(tag, nm, ix, 'rho')]:+.3f};" for ix in ("ms", "er", "flag"))
                      + f"  mean dk(cross) - dk(other): {agg[(tag, 'dkL_cross_mean' if nm == 'flowL' else 'dkU_cross_mean')]:+.4f}")
