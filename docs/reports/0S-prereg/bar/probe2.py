import sys

import numpy as np
from scipy import stats

import core
import kw

seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3]
for seed in seeds:
    W = kw.make_world(seed)
    F = core.static_features(W)
    E, n = W["E"], W["n"]
    adj, fl = F["adj"], F["flag"]
    cross_nodes = np.unique(E[fl].ravel())
    near = adj & ~fl & (np.isin(E[:, 0], cross_nodes) | np.isin(E[:, 1], cross_nodes))
    far = adj & ~fl & ~near
    isl_inc = np.isin(E[:, 0], np.flatnonzero(W["kind"] != "main")) | np.isin(E[:, 1], np.flatnonzero(W["kind"] != "main"))
    print(f"== seed {seed}: adj {adj.sum()} crossing {fl.sum()} near {near.sum()} far {far.sum()}")
    for model in ("S",):
        base = {}
        for c in (0.0, 0.02, 0.05, 0.1):
            L = core.lengths_S(W, F, c) if model == "S" else core.lengths_R(W, F, c)
            D = core.closure(n, E, L)
            for meas in ("unif", "grav"):
                w = None if meas == "unif" else core.gravity_w(W, E, D)
                kap, _ = core.orc(n, E, D, w)
                if c == 0:
                    base[meas] = kap
                dk = kap - base[meas]
                k = kap[adj]
                o = np.flatnonzero(adj)[np.argsort(k)[:20]]
                t20 = np.zeros(len(E), bool)
                t20[o] = True
                sp = lambda a, b: stats.spearmanr(a, b)[0]  # noqa: E731
                print(f" {model} c={c:<5}{meas}: dk cross {dk[fl].mean():+.3f} near {dk[near].mean():+.3f} far {dk[far].mean():+.3f} | "
                      f"top20: cross {t20[fl].sum()} near {t20[near].sum()} coast {int(F['coast_any'][t20].sum())} "
                      f"bord {int(F['border_any'][t20].sum())} islinc {int(isl_inc[t20].sum())} "
                      f"degterm {F['degterm'][t20].mean():.3f} (pop {F['degterm'][adj].mean():.3f}) "
                      f"len_e {np.median(F['len_e'][t20]):.1f} (pop {np.median(F['len_e'][adj]):.1f}) kap20max {k[np.argsort(k)[19]]:+.3f} | "
                      f"rho deg {sp(k, F['degterm'][adj]):+.2f} coast {sp(k, F['coast_any'][adj]):+.2f} bord {sp(k, F['border_any'][adj]):+.2f} "
                      f"len_e {sp(k, F['len_e'][adj]):+.2f} len_t {sp(k, L[adj]):+.2f} contact {sp(k, F['contact'][adj]):+.2f} "
                      f"island links k={np.round(kap[~adj], 2)}")
