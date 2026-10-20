"""Pure null: population independent of terrain (pop_elev=None). Model S, c in {0, 0.02, 0.05}."""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

C_LIST = [0.0, 0.02, 0.05]
OUT = "res_n0"


def work(seed):
    import core
    import kw
    fn = os.path.join(OUT, f"m{seed:04d}.npz")
    if os.path.exists(fn):
        return seed, 0.0
    t0 = time.time()
    W = kw.make_world(seed, pop_elev=None)
    F = core.static_features(W)
    E, n = W["E"], W["n"]
    rec = {k: np.asarray(v) for k, v in F.items()}
    rec.update(E=E, etype=W["etype"], deg=W["deg"], kind=W["kind"], pts=W["pts"], P=W["P"])
    Lg = np.where(F["adj"], 1.3 * F["len_e"], np.array([W["link_time"].get((int(u), int(v)), np.nan) for u, v in E]))
    Dg = core.closure(n, E, Lg)
    rec["kgeo_unif"], _ = core.orc(n, E, Dg)
    rec["kgeo_grav"], _ = core.orc(n, E, Dg, core.gravity_w(W, E, Dg))
    for c in C_LIST:
        L = core.lengths_S(W, F, c)
        D = core.closure(n, E, L)
        tag = f"S_{c:g}"
        rec[f"L_{tag}"] = L
        rec[f"ku_{tag}"], _ = core.orc(n, E, D)
        rec[f"kg_{tag}"], _ = core.orc(n, E, D, core.gravity_w(W, E, D))
    np.savez_compressed(fn, **rec)
    return seed, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    seeds = list(range(int(sys.argv[1]), int(sys.argv[2])))
    with Pool(10) as pool:
        for s, dt in pool.imap_unordered(work, seeds):
            print(f"map {s} {dt:.1f}s", flush=True)
    print("all done")
