import sys
import time

import numpy as np
from scipy import stats

import core
import kw

print("fixtures:", {k: round(v, 6) if isinstance(v, float) else v for k, v in core.fixture_check().items()})
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
t0 = time.time()
W = kw.make_world(seed)
t1 = time.time()
F = core.static_features(W)
t2 = time.time()
print(f"world {t1-t0:.1f}s features {t2-t1:.1f}s")
adj = F["adj"]
fl = F["flag"]
base = F["len_e"] * W["rho"]
print("base time (min) adj: median %.1f  crossing median %.1f" % (np.median(base[adj]), np.median(base[fl])))
print("ca_true (m): non-crossing median %.0f p90 %.0f | crossing median %.0f p10 %.0f p90 %.0f" % (
    np.median(F["ca_true"][adj & ~fl]), np.percentile(F["ca_true"][adj & ~fl], 90),
    np.median(F["ca_true"][fl]), np.percentile(F["ca_true"][fl], 10), np.percentile(F["ca_true"][fl], 90)))
print("er (m): non-crossing median %.0f p90 %.0f | crossing median %.0f" % (
    np.median(F["er"][adj & ~fl]), np.percentile(F["er"][adj & ~fl], 90), np.median(F["er"][fl])))
print("ph (m): non-crossing median %.0f p90 %.0f | crossing median %.0f" % (
    np.nanmedian(F["ph"][adj & ~fl]), np.nanpercentile(F["ph"][adj & ~fl], 90), np.nanmedian(F["ph"][fl])))
med_pen_per_c = np.median(F["ca_true"][fl] - np.median(F["ca_true"][adj & ~fl]))
print("median excess ascent of crossing edges (m): %.0f" % med_pen_per_c)
for model in ("S", "R"):
    for c in (0.0, 0.02, 0.05, 0.1, 0.2):
        t3 = time.time()
        L = core.lengths_S(W, F, c) if model == "S" else core.lengths_R(W, F, c)
        D = core.closure(W["n"], W["E"], L)
        for meas in ("unif", "grav"):
            w = None if meas == "unif" else core.gravity_w(W, W["E"], D)
            kap, w1 = core.orc(W["n"], W["E"], D, w)
            k = kap[adj]
            o = np.argsort(k)[:20]
            dxy = D[W["E"][:, 0], W["E"][:, 1]]
            print(f"{model} c={c:<5} {meas}: L_cross/L_other={np.median(dxy[fl])/np.median(dxy[adj & ~fl]):.2f} "
                  f"kappa mean all={k.mean():+.3f} cross={kap[fl].mean():+.3f} (median {np.median(kap[fl]):+.3f}) "
                  f"top20 flag={fl[adj][o].sum()} flag_all={F['flag_all'][adj][o].sum()} "
                  f"rho(k,degterm)={stats.spearmanr(k, F['degterm'][adj])[0]:+.3f} rho(k,er)={stats.spearmanr(k, F['er'][adj])[0]:+.3f} "
                  f"rho(k,flag)={stats.spearmanr(k, fl[adj])[0]:+.3f}  [{time.time()-t3:.1f}s]")
