"""Protocol A analysis: Adam (plain 1e-6/200 rule, tmax 50k) + bounded L-BFGS-B polish."""
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def ok(pol):
    return "CONVERGENCE" in pol["msg"]


def summarize(tag, H, E):
    H = sorted(H, key=lambda r: r["seed"])
    E = sorted(E, key=lambda r: r["seed"])
    hl = np.array([r["pA_stop"]["loss"] for r in H])
    hm = np.array([r["pA_stop"]["mrd"] for r in H])
    el = np.array([r["pA_stop"]["loss"] for r in E])
    em = np.array([r["pA_stop"]["mrd"] for r in E])
    hb, eb = int(np.argmin(hl)), int(np.argmin(el))
    d2 = np.array([abs(r["pA_cont"]["mrd"] - r["pA_stop"]["mrd"]) for r in H])
    d2l = np.array([abs(r["pA_cont"]["loss"] - r["pA_stop"]["loss"]) / r["pA_stop"]["loss"] for r in H])
    adam_gap = np.array([r["stop"]["loss"] / r["pA_stop"]["loss"] - 1 for r in H])
    nit = np.array([r["pA_stop"]["nit"] for r in H])
    ks = np.array([r["stop"]["k"] for r in H])
    conv = np.array([r["converged"] for r in H])
    pconv = np.array([ok(r["pA_stop"]) for r in H])
    psec = np.array([r["pA_stop"]["seconds"] for r in H])
    print(f"{tag:22s} H polished best {hm[hb]:.5f} med {np.median(hm):.5f} | E best {em[eb]:.5f} med {np.median(em):.5f} | "
          f"ratio best {hm[hb] / em[eb]:.4f} med {np.median(hm) / np.median(em):.4f} stress {hl[hb] / el[eb]:.4f} | "
          f"Adam rule fired {conv.sum()}/{len(H)} k med {int(np.median(ks))} max {ks.max()} | polish conv {pconv.sum()}/{len(H)} "
          f"nit med {int(np.median(nit))} max {nit.max()} sec max {psec.max():.1f} | Adam-stop stress above polished: max {adam_gap.max():.1e} | "
          f"polish(stop) vs polish(2x Adam): max|dmrd| {d2.max():.1e} max rel dstress {d2l.max():.1e} | "
          f"at_cap(best) {H[hb]['pA_stop']['at_cap']} maxrad {max(r['pA_stop']['max_radius'] for r in H):.3f}")
    return H, E


A50 = load("A_tree50.jsonl")
if A50:
    E = [r for r in A50 if r["job"]["space"] == "euclid"]
    for R in [3, 4, 5, 6]:
        summarize(f"tree R{R} (50 seeds)", [r for r in A50 if r["job"].get("rmax") == R and r["job"]["space"] == "poincare"], E)
    # thresholds on polished values, all C(50,5) subsets
    seeds = sorted(r["seed"] for r in E)
    Ed = {r["seed"]: r for r in E}
    em = np.array([Ed[s]["pA_stop"]["mrd"] for s in seeds])
    el = np.array([Ed[s]["pA_stop"]["loss"] for s in seeds])
    C5 = np.array(list(combinations(range(len(seeds)), 5)), dtype=np.int32)
    ie = C5[np.arange(len(C5)), np.argmin(el[C5], axis=1)]
    eb = em[ie]
    emed = np.median(em[C5], axis=1)
    guard = np.abs(eb - 0.1506) <= 0.003
    for R, ref in [(3, 0.0664), (4, 0.0465), (5, 0.0339), (6, 0.0260)]:
        Hd = {r["seed"]: r for r in A50 if r["job"].get("rmax") == R and r["job"]["space"] == "poincare"}
        hm = np.array([Hd[s]["pA_stop"]["mrd"] for s in seeds])
        hl = np.array([Hd[s]["pA_stop"]["loss"] for s in seeds])
        hc = np.array([ok(Hd[s]["pA_stop"]) for s in seeds])
        ih = C5[np.arange(len(C5)), np.argmin(hl[C5], axis=1)]
        hb = hm[ih]
        rb = hb / eb
        rm = np.median(hm[C5], axis=1) / emed
        c4 = (np.abs(hb - ref) <= 0.001) & hc[ih]
        rule = (rb <= 0.3) & (rm <= 0.8) & guard & c4
        old = (rb <= 0.5) & (rm <= 0.8) & guard
        vals, cnt = np.unique(np.round(hm, 4), return_counts=True)
        print(f"   R{R}: basins {dict(zip(vals.tolist(), cnt.tolist()))}; best-ratio range {rb.min():.4f}-{rb.max():.4f}; "
              f"median-ratio max {rm.max():.4f}; pass(new rule: best<=0.3, med<=0.8, guard, H={ref}+-0.001 & polish conv) "
              f"{rule.mean():.6f} (fail {int((~rule).sum())}); pass(old 0.5/0.8/guard) {old.mean():.6f}; "
              f"all-5 H<E {(hm[C5] < em[C5]).all(axis=1).mean():.4f}")
        if R == 5:
            print("   R5 seeds 0-4 polished H", [round(Hd[s]["pA_stop"]["mrd"], 6) for s in range(5)],
                  "E", [round(Ed[s]["pA_stop"]["mrd"], 6) for s in range(5)],
                  "Adam k", [Hd[s]["stop"]["k"] for s in range(5)], "polish nit", [Hd[s]["pA_stop"]["nit"] for s in range(5)])
            ix = np.arange(5)
            hbs = hm[ix][np.argmin(hl[ix])]
            ebs = em[ix][np.argmin(el[ix])]
            print(f"   seeds 0-4: best ratio {hbs / ebs:.4f} median ratio {np.median(hm[ix]) / np.median(em[ix]):.4f}")
for ds in ["grid_road", "grid_road_hw", "grid_l1"]:
    G = load(f"A_grid_{ds}.jsonl")
    if G:
        summarize(f"{ds} R5", [r for r in G if r["job"]["space"] == "poincare"], [r for r in G if r["job"]["space"] == "euclid"])
T7 = load("A_tree7.jsonl")
if T7:
    E7 = [r for r in T7 if r["job"]["space"] == "euclid"]
    for R in [3, 5]:
        H = [r for r in T7 if r["job"].get("rmax") == R and r["job"]["space"] == "poincare"]
        if H and E7:
            summarize(f"tree7 R{R}", H, E7)
