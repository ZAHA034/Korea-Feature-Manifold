"""Full experiment: per map, static features + ORC under models S/R, strengths C_LIST, measures unif/grav."""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

C_LIST = [0.0, 0.01, 0.02, 0.05, 0.1]
OUT = "res"


def work(seed):
    import core
    import kw
    fn = os.path.join(OUT, f"m{seed:04d}.npz")
    if os.path.exists(fn):
        return seed, 0.0
    t0 = time.time()
    W = kw.make_world(seed)
    F = core.static_features(W)
    E, n = W["E"], W["n"]
    rec = {k: np.asarray(v) for k, v in F.items()}
    rec["E"] = E
    rec["etype"] = W["etype"]
    rec["deg"] = W["deg"]
    rec["kind"] = W["kind"]
    rec["pts"] = W["pts"]
    rec["P"] = W["P"]
    rec["coast"] = W["coast"]
    rec["border"] = W["border"]
    # geometric (flat, no-detour) reference curvature
    Lg = np.where(F["adj"], 1.3 * F["len_e"], np.array([W["link_time"].get((int(u), int(v)), np.nan) for u, v in E]))
    Dg = core.closure(n, E, Lg)
    rec["kgeo_unif"], _ = core.orc(n, E, Dg)
    rec["kgeo_grav"], _ = core.orc(n, E, Dg, core.gravity_w(W, E, Dg))
    for model in ("S", "R"):
        D0 = None
        for c in C_LIST:
            L = core.lengths_S(W, F, c) if model == "S" else core.lengths_R(W, F, c)
            D = core.closure(n, E, L)
            if c == 0:
                D0 = D
            tag = f"{model}_{c:g}"
            rec[f"L_{tag}"] = L
            rec[f"d_{tag}"] = D[E[:, 0], E[:, 1]]
            ku, w1u = core.orc(n, E, D)
            wg = core.gravity_w(W, E, D)
            kg, w1g = core.orc(n, E, D, wg)
            rec[f"ku_{tag}"], rec[f"kg_{tag}"] = ku, kg
            rec[f"w1u_{tag}"], rec[f"w1g_{tag}"] = w1u, w1g
            if c > 0:
                # decomposition for gravity: weights from barrier times with null lengths, and the reverse
                rec[f"kgW_{tag}"], _ = core.orc(n, E, D0, wg)
                rec[f"kgL_{tag}"], _ = core.orc(n, E, D, core.gravity_w(W, E, D0))
    np.savez_compressed(fn, **rec)
    return seed, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(int(sys.argv[1]), int(sys.argv[2])))
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    t0 = time.time()
    with Pool(nproc) as pool:
        for s, dt in pool.imap_unordered(work, seeds):
            print(f"map {s} done in {dt:.1f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    print("all done", time.time() - t0)
