"""Sensitivity of per-edge barrier indices to DEM resolution (low-pass at 2*res) and segment sampling interval,
and of the primary test (Spearman(dk_unif, index) > 0) to that choice."""
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy import stats

COMBOS = [(0.03, 0.025), (0.03, 0.05), (0.03, 0.1), (0.03, 0.25), (0.09, 0.05), (0.09, 0.1), (0.09, 0.25), (0.09, 0.5),
          (0.25, 0.1), (0.25, 0.25), (0.25, 0.5), (0.5, 0.25), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]


def work(seed):
    import core
    import kw
    W = kw.make_world(seed)
    R = dict(np.load(f"res/m{seed:04d}.npz", allow_pickle=True))
    a = R["adj"]
    E = W["E"][a]
    assert np.array_equal(E, R["E"][a])
    out = {"seed": seed}
    for res, ds in COMBOS:
        er, ca = core.barrier_indices(W, ds=ds, lam_cut=2 * res, edges=E)
        out[f"{res}_{ds}"] = (er, ca)
    return out


if __name__ == "__main__":
    seeds = list(range(int(sys.argv[1]), int(sys.argv[2])))
    with Pool(10) as p:
        rows = p.map(work, seeds)
    ref = "0.03_0.025"
    summ = {}
    for res, ds in COMBOS:
        key = f"{res}_{ds}"
        d = {}
        for r in rows:
            R = dict(np.load(f"res/m{r['seed']:04d}.npz", allow_pickle=True))
            a = R["adj"]
            L = R["len_e"][a]
            fl = R["flag"][a].astype(bool)
            er0, ca0 = r[ref]
            er, ca = r[key]
            ms0, ms = ca0 / L, ca / L
            d.setdefault("rho_er", []).append(stats.spearmanr(er, er0)[0])
            d.setdefault("rho_ca", []).append(stats.spearmanr(ca, ca0)[0])
            d.setdefault("rho_ms", []).append(stats.spearmanr(ms, ms0)[0])
            big = er0 > 100
            d.setdefault("relerr_er", []).append(np.median(np.abs(er[big] - er0[big]) / er0[big]))
            d.setdefault("ratio_ca", []).append(np.median(ca / ca0))
            d.setdefault("er_cross", []).append(np.median(er[fl]))
            d.setdefault("ca_cross", []).append(np.median(ca[fl]))
            d.setdefault("ms_cross_km", []).append(np.median(ms[fl]))
            d.setdefault("ms_other_km", []).append(np.median(ms[~fl]))
            d.setdefault("rho_ca_len", []).append(stats.spearmanr(ca, L)[0])
            d.setdefault("rho_er_len", []).append(stats.spearmanr(er, L)[0])
            d.setdefault("rho_ms_len", []).append(stats.spearmanr(ms, L)[0])
            for model in ("S", "R"):
                for c in (0.0, 0.01, 0.02):
                    dk = (R[f"ku_{model}_{c:g}"] - R["kgeo_unif"])[a]
                    for nm, z in (("ms", ms), ("er", er)):
                        rho, pg = stats.spearmanr(dk, z, alternative="greater")
                        d.setdefault(f"rej_{nm}_{model}_{c:g}", []).append(pg < 0.05)
                        d.setdefault(f"rho_{nm}_{model}_{c:g}", []).append(rho)
        summ[key] = {k: float(np.mean(v)) if k.startswith("rej") else float(np.median(v)) for k, v in d.items()}
        s = summ[key]
        print(f"DEM {res*1000:5.0f} m step {ds*1000:5.0f} m | er: rho_ref {s['rho_er']:.3f} relerr {s['relerr_er']:.3f} cross {s['er_cross']:.0f} m | "
              f"ca: rho_ref {s['rho_ca']:.3f} ratio {s['ratio_ca']:.2f} cross {s['ca_cross']:.0f} m | ms: rho_ref {s['rho_ms']:.3f} "
              f"cross {s['ms_cross_km']:.1f} other {s['ms_other_km']:.1f} m/km | len-rho ca {s['rho_ca_len']:+.2f} er {s['rho_er_len']:+.2f} ms {s['rho_ms_len']:+.2f} | "
              f"primary(ms) rej S {s['rej_ms_S_0']:.2f}/{s['rej_ms_S_0.01']:.2f}/{s['rej_ms_S_0.02']:.2f} R {s['rej_ms_R_0']:.2f}/{s['rej_ms_R_0.01']:.2f}/{s['rej_ms_R_0.02']:.2f} "
              f"| er rej S {s['rej_er_S_0']:.2f}/{s['rej_er_S_0.01']:.2f}/{s['rej_er_S_0.02']:.2f} R {s['rej_er_R_0']:.2f}/{s['rej_er_R_0.01']:.2f}/{s['rej_er_R_0.02']:.2f}", flush=True)
    json.dump(summ, open("resstudy.json", "w"), indent=1)
