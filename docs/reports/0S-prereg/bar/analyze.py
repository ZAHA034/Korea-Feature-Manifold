import glob
import json
import sys

import numpy as np
from scipy import stats

C_LIST = [0.0, 0.01, 0.02, 0.05, 0.1]
files = sorted(glob.glob("res/m*.npz"))
M = [dict(np.load(f, allow_pickle=True)) for f in files]
print("maps:", len(M))
out = {"n_maps": len(M)}


def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return stats.spearmanr(a[ok], b[ok])[0]


def summ(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "frac_pos": float(np.mean(v > 0)), "mean": float(v.mean())}


# ---------------------------------------------------------------- map stats
ms = {"n_adj": [], "n_cross": [], "n_cross_all": [], "base_rate": [], "base_rate_all": [], "deg_mean": [], "coast_edge_frac": []}
for R in M:
    a = R["adj"]
    ms["n_adj"].append(a.sum())
    ms["n_cross"].append(R["flag"].sum())
    ms["n_cross_all"].append(R["flag_all"].sum())
    ms["base_rate"].append(R["flag"].sum() / a.sum())
    ms["base_rate_all"].append(R["flag_all"].sum() / a.sum())
    ms["deg_mean"].append(R["deg"].mean())
    ms["coast_edge_frac"].append(R["coast_any"][a].mean())
out["map_stats"] = {k: summ(v) for k, v in ms.items()}
print(json.dumps(out["map_stats"], indent=0)[:1500])

# ---------------------------------------------------------------- task 2: null correlations
null = {}
for model in ("S", "R"):
    for meas, key in (("unif", "ku"), ("grav", "kg")):
        rows = {k: [] for k in ("degterm", "coast_any", "border_any", "bnd_any", "len_t", "len_e", "contact")}
        tail = {"degterm_top": [], "degterm_pop": [], "coast_top": [], "coast_pop": [], "coast_enrich_p": [], "border_top": [],
                "lowdeg_top": [], "lowdeg_pop": []}
        isl = {"ferry": [], "bridge": [], "ferry_pct": [], "bridge_pct": []}
        for R in M:
            a = R["adj"]
            k = R[f"{key}_{model}_0"]
            ka = k[a]
            rows["degterm"].append(sp(ka, R["degterm"][a]))
            rows["coast_any"].append(sp(ka, R["coast_any"][a]))
            rows["border_any"].append(sp(ka, R["border_any"][a]))
            rows["bnd_any"].append(sp(ka, R["bnd_any"][a]))
            rows["len_t"].append(sp(ka, R[f"d_{model}_0"][a]))
            rows["len_e"].append(sp(ka, R["len_e"][a]))
            rows["contact"].append(sp(ka, R["contact"][a]))
            o = np.argsort(ka, kind="stable")[:20]
            tail["degterm_top"].append(R["degterm"][a][o].mean())
            tail["degterm_pop"].append(R["degterm"][a].mean())
            ca = R["coast_any"][a] > 0
            x = int(ca[o].sum())
            tail["coast_top"].append(x / 20)
            tail["coast_pop"].append(ca.mean())
            tail["coast_enrich_p"].append(stats.hypergeom.sf(x - 1, len(ca), int(ca.sum()), 20))
            tail["border_top"].append(R["border_any"][a][o].mean())
            E = R["E"]
            mind = np.minimum(R["deg"][E[:, 0]], R["deg"][E[:, 1]])[a]
            tail["lowdeg_top"].append(np.mean(mind[o] <= 3))
            tail["lowdeg_pop"].append(np.mean(mind <= 3))
            # island links: percentile among all edges (0 = most negative)
            et = R["etype"]
            kind = R["kind"]
            for j in np.flatnonzero(et == "island"):
                u, v = E[j]
                typ = "bridge" if "near" in (kind[u], kind[v]) else "ferry"
                isl[typ].append(k[j])
                isl[typ + "_pct"].append(float(np.mean(k < k[j])))
        null[f"{model}_{meas}"] = {
            "spearman": {kk: summ(v) for kk, v in rows.items()},
            "tail": {kk: float(np.mean(v)) for kk, v in tail.items()},
            "coast_enrich_sig_rate": float(np.mean(np.array(tail["coast_enrich_p"]) < 0.05)),
            "island": {kk: summ(v) for kk, v in isl.items()},
        }
out["null"] = null
print("\n=== TASK 2: null (c=0) Spearman rho(kappa, covariate) over adjacency edges: median [p10, p90] (share of maps > 0)")
for cfg, d in null.items():
    s = d["spearman"]
    print(cfg, " ".join(f"{kk}={v['median']:+.3f}[{v['p10']:+.2f},{v['p90']:+.2f}]({v['frac_pos']:.2f})" for kk, v in s.items()))
    t = d["tail"]
    print("    top20 mean degterm %.3f vs pop %.3f | coastal share %.2f vs pop %.2f (enrichment p<.05 in %.0f%% of maps) | border %.2f | min-deg<=3 %.2f vs %.2f" % (
        t["degterm_top"], t["degterm_pop"], t["coast_top"], t["coast_pop"], 100 * d["coast_enrich_sig_rate"], t["border_top"], t["lowdeg_top"], t["lowdeg_pop"]))
    i = d["island"]
    print("    island links kappa: ferry median %+.3f [%+.2f,%+.2f] pct %.2f | bridge median %+.3f [%+.2f,%+.2f] pct %.2f" % (
        i["ferry"]["median"], i["ferry"]["p10"], i["ferry"]["p90"], i["ferry_pct"]["median"],
        i["bridge"]["median"], i["bridge"]["p10"], i["bridge"]["p90"], i["bridge_pct"]["median"]))

# ---------------------------------------------------------------- task 3: barrier response
resp = {}
print("\n=== TASK 3: response to barrier strength c (min per m mean ascent). mean over maps of per-map means")
for model in ("S", "R"):
    for meas, key in (("unif", "ku"), ("grav", "kg")):
        for c in C_LIST:
            tag = f"{model}_{c:g}"
            r = {k: [] for k in ("dk_cross", "dk_near", "dk_far", "k_cross", "k_all", "top20_cross", "top20_near", "dd_rel_cross", "dw1_rel_cross",
                                 "pen_rel_cross", "pen_rel_other", "dkW_cross", "dkL_cross", "rho_flag", "rho_er")}
            for R in M:
                a, fl, E = R["adj"], R["flag"], R["E"]
                cn = np.zeros(len(R["deg"]), bool)
                cn[E[fl].ravel()] = True
                near = a & ~fl & (cn[E[:, 0]] | cn[E[:, 1]])
                far = a & ~fl & ~near
                k0 = R[f"{key}_{model}_0"]
                k = R[f"{key}_{tag}"]
                dk = k - k0
                r["dk_cross"].append(dk[fl].mean())
                r["dk_near"].append(dk[near].mean())
                r["dk_far"].append(dk[far].mean())
                r["k_cross"].append(k[fl].mean())
                r["k_all"].append(k[a].mean())
                ka = k[a]
                o = np.argsort(ka, kind="stable")[:20]
                r["top20_cross"].append(fl[a][o].sum())
                r["top20_near"].append(near[a][o].sum())
                d0, d1 = R[f"d_{model}_0"], R[f"d_{tag}"]
                w0, w1 = R[f"w1{key[1]}_{model}_0"], R[f"w1{key[1]}_{tag}"]
                r["dd_rel_cross"].append(np.mean(d1[fl] / d0[fl] - 1))
                r["dw1_rel_cross"].append(np.mean(w1[fl] / w0[fl] - 1))
                L0, L1 = R[f"L_{model}_0"], R[f"L_{tag}"]
                r["pen_rel_cross"].append(np.median(L1[fl] / L0[fl] - 1))
                r["pen_rel_other"].append(np.median(L1[a & ~fl] / L0[a & ~fl] - 1))
                if key == "kg" and c > 0:
                    r["dkW_cross"].append((R[f"kgW_{tag}"] - k0)[fl].mean())
                    r["dkL_cross"].append((R[f"kgL_{tag}"] - k0)[fl].mean())
                r["rho_flag"].append(sp(ka, fl[a].astype(float)))
                r["rho_er"].append(sp(ka, R["er"][a]))
            resp[f"{model}_{meas}_{c:g}"] = {kk: (float(np.mean(v)) if len(v) else None) for kk, v in r.items()}
            q = resp[f"{model}_{meas}_{c:g}"]
            extra = (f" | grav decomposition dk_cross weights-only {q['dkW_cross']:+.3f} lengths-only {q['dkL_cross']:+.3f}" if q["dkW_cross"] is not None else "")
            print(f"{model} {meas} c={c:<5g} travel-time excess cross {100*q['pen_rel_cross']:5.1f}% other {100*q['pen_rel_other']:4.1f}% | "
                  f"dk cross {q['dk_cross']:+.3f} near {q['dk_near']:+.3f} far {q['dk_far']:+.3f} | k_cross {q['k_cross']:+.3f} (all {q['k_all']:+.3f}) | "
                  f"d +{100*q['dd_rel_cross']:.0f}% W1 +{100*q['dw1_rel_cross']:.0f}% | top20 cross {q['top20_cross']:.2f} near {q['top20_near']:.2f} | "
                  f"rho(k,flag) {q['rho_flag']:+.3f} rho(k,er) {q['rho_er']:+.3f}{extra}")
out["response"] = resp
json.dump(out, open("analysis_part1.json", "w"), indent=1)
