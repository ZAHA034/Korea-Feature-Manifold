import glob
from multiprocessing import Pool

import numpy as np
from scipy import stats

import eval_jk as J


def run(files, scen):
    with Pool(10) as p:
        return p.map(J.per_map, [(f, scen) for f in files])


if __name__ == "__main__":
    r1 = run(sorted(glob.glob("res/m*.npz")), [("S", 0.0), ("S", 0.01), ("S", 0.02), ("R", 0.0), ("R", 0.01), ("R", 0.02)])
    r0 = run(sorted(glob.glob("res_n0/m*.npz")), [("S", 0.0), ("S", 0.02)])
    for lab, rows, model, c in (("N0 null S", r0, "S", 0.0), ("N0 c=.02 S", r0, "S", 0.02), ("N1 null S", r1, "S", 0.0), ("N1 null R", r1, "R", 0.0),
                                ("N1 c=.01 S", r1, "S", 0.01), ("N1 c=.01 R", r1, "R", 0.01), ("N1 c=.02 S", r1, "S", 0.02), ("N1 c=.02 R", r1, "R", 0.02)):
        s = f"{lab:12s}"
        for ix in ("ms", "er", "flag"):
            v = [r[(model, c, ix)] for r in rows]
            rho = np.array([x[0] for x in v]); p = np.array([x[1] for x in v]); se = np.array([x[2] for x in v])
            pj = stats.t.sf(rho / se, J.B - 1)
            s += f" | {ix}: naive {np.mean(p < .05):.2f} jk {np.mean(pj < .05):.2f} both {np.mean((p < .05) & (pj < .05)):.2f}"
        print(s)
