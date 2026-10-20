"""Reference-curvature variants implementable on real data + tail composition after covariate control.

ref_geo  : lengths = 1.3*Euclid, weights = gravity on 1.3*Euclid      (both lengths and weights geometric)
ref_wgeo : lengths = observed travel time, weights = gravity on 1.3*Euclid  (only weights geometric)
uniform  : ref = uniform ORC with Euclidean lengths
tau_ref in {15, 30, 60} (true tau = 30) to probe misspecification.
"""
import glob
import json
from multiprocessing import Pool

import numpy as np
from scipy import stats

C_LIST = [0.0, 0.01, 0.02, 0.05, 0.1]
IDX = ("er", "ca", "ph", "ms", "flag", "flag_all")
TAUS = (15.0, 30.0, 60.0)


def resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def topmask(v, k=20):
    m = np.zeros(len(v), bool)
    m[np.argsort(v, kind="stable")[:k]] = True
    return m


def per_map(fn):
    import core
    R = dict(np.load(fn, allow_pickle=True))
    E = R["E"]
    n = len(R["deg"])
    a = R["adj"]
    P = R["P"]
    Lg = np.where(a, 1.3 * R["len_e"], R["L_S_0"])
    Dg = core.closure(n, E, Lg)
    refs = {}
    for tau in TAUS:
        wg = P[E[:, 0]] * P[E[:, 1]] * np.exp(-Dg[E[:, 0], E[:, 1]] / tau)
        refs[("geo", tau)] = core.orc(n, E, Dg, wg)[0]
    out = {}
    coast = R["coast_any"][a] > 0
    deg_cov = np.c_[R["degterm"][a], R["bnd_any"][a]]
    len_cov = np.c_[deg_cov, np.log(R["len_e"][a])]
    for model in ("S", "R"):
        for c in C_LIST:
            tag = f"{model}_{c:g}"
            L = R[f"L_{tag}"]
            D = core.closure(n, E, L)
            kg = R[f"kg_{tag}"]
            ku = R[f"ku_{tag}"]
            dks = {}
            for tau in TAUS:
                dks[f"grav_geo_t{tau:g}"] = kg - refs[("geo", tau)]
                wg = P[E[:, 0]] * P[E[:, 1]] * np.exp(-Dg[E[:, 0], E[:, 1]] / tau)
                dks[f"grav_wgeo_t{tau:g}"] = kg - core.orc(n, E, D, wg)[0]
            dks["unif_geo"] = ku - R["kgeo_unif"]
            for nm, dk in dks.items():
                y = dk[a]
                for ix in IDX:
                    z = R[ix][a].astype(float)
                    ok = np.isfinite(z)
                    rl = stats.spearmanr(y[ok], z[ok], alternative="less")
                    rg = stats.spearmanr(y[ok], z[ok], alternative="greater")
                    out[(tag, nm, ix, "neg")] = rl[1] < 0.05
                    out[(tag, nm, ix, "pos")] = rg[1] < 0.05
                    out[(tag, nm, ix, "rho")] = rl[0]
                fl = R["flag"][a].astype(bool)
                t = topmask(y)
                x = int((t & fl).sum())
                out[(tag, nm, "hyper", "neg")] = stats.hypergeom.sf(x - 1, len(fl), int(fl.sum()), 20) < 0.05
                out[(tag, nm, "x", "val")] = x
                out[(tag, nm, "coast_top", "val")] = coast[t].mean()
                if c == 0:
                    for cv in ("degterm", "bnd_any", "len_e", "coast_any"):
                        out[(tag, nm, f"rho_{cv}", "val")] = stats.spearmanr(y, R[cv][a])[0]
            for meas, k in (("unif", ku), ("grav", kg)):
                y = k[a]
                out[(tag, meas, "coast_top_raw", "val")] = coast[topmask(y)].mean()
                out[(tag, meas, "coast_top_D", "val")] = coast[topmask(resid(y, deg_cov))].mean()
                out[(tag, meas, "coast_top_D2", "val")] = coast[topmask(resid(y, len_cov))].mean()
                out[(tag, meas, "degterm_top_D", "val")] = R["degterm"][a][topmask(resid(y, deg_cov))].mean()
    return out


if __name__ == "__main__":
    files = sorted(glob.glob("res/m*.npz"))
    with Pool(10) as p:
        rows = p.map(per_map, files)
    keys = rows[0].keys()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    json.dump({"|".join(map(str, k)): v for k, v in agg.items()}, open("rules2.json", "w"), indent=0)
    print("maps", len(rows))
    names = [f"grav_geo_t{t:g}" for t in TAUS] + [f"grav_wgeo_t{t:g}" for t in TAUS] + ["unif_geo"]
    for model in ("S", "R"):
        print(f"\n=== model {model}: rejection rates (neg = H1 rho<0; pos = H1 rho>0); columns c = {C_LIST}")
        for nm in names:
            for ix in ("flag", "ph", "er", "ca", "ms", "flag_all"):
                for d in ("neg", "pos"):
                    vals = [agg[(f"{model}_{c:g}", nm, ix, d)] for c in C_LIST]
                    print(f"{nm:16s} {ix:8s} {d}: " + " ".join(f"{v:.2f}" for v in vals))
            vals = [agg[(f"{model}_{c:g}", nm, "hyper", "neg")] for c in C_LIST]
            xs = [agg[(f"{model}_{c:g}", nm, "x", "val")] for c in C_LIST]
            print(f"{nm:16s} top20-hyper  : " + " ".join(f"{v:.2f}" for v in vals) + "   mean x: " + " ".join(f"{v:.2f}" for v in xs))
            print(f"{nm:16s} null rho with degterm {agg[(f'{model}_0', nm, 'rho_degterm', 'val')]:+.3f} bnd {agg[(f'{model}_0', nm, 'rho_bnd_any', 'val')]:+.3f} "
                  f"len_e {agg[(f'{model}_0', nm, 'rho_len_e', 'val')]:+.3f} coast {agg[(f'{model}_0', nm, 'rho_coast_any', 'val')]:+.3f} | coastal share in top20 (null) {agg[(f'{model}_0', nm, 'coast_top', 'val')]:.2f}")
        for meas in ("unif", "grav"):
            print(f"{model} {meas}: coastal share in top20 raw/D/D2 (null): {agg[(f'{model}_0', meas, 'coast_top_raw', 'val')]:.2f} / "
                  f"{agg[(f'{model}_0', meas, 'coast_top_D', 'val')]:.2f} / {agg[(f'{model}_0', meas, 'coast_top_D2', 'val')]:.2f}; "
                  f"mean degterm in residual top20 (D) {agg[(f'{model}_0', meas, 'degterm_top_D', 'val')]:.3f}")
