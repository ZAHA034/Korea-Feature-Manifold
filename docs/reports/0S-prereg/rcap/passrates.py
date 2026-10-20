"""Seed-subset pass rates for tree-fixture thresholds under the capped, converged protocol."""
import json
import sys

import numpy as np

path = sys.argv[1]
R = [json.loads(line) for line in open(path)]
by = {}
for r in R:
    j = r["job"]
    tag = "E" if j["space"] == "euclid" else f"R{j.get('rmax')}"
    by.setdefault(tag, {})[r["seed"]] = r
seeds = sorted(by["E"])
S = len(seeds)
E_m = np.array([by["E"][s]["stop"]["mrd"] for s in seeds])
E_l = np.array([by["E"][s]["stop"]["loss"] for s in seeds])
print(f"E: basins {dict(zip(*np.unique(np.round(E_m, 4), return_counts=True)))}")
rng = np.random.default_rng(0)
draws = [rng.choice(S, 5, replace=False) for _ in range(20000)]
groups = [np.arange(5 * g, 5 * g + 5) for g in range(S // 5)]
for tag in sorted(k for k in by if k != "E"):
    H_m = np.array([by[tag][s]["stop"]["mrd"] for s in seeds])
    H_l = np.array([by[tag][s]["stop"]["loss"] for s in seeds])
    conv = np.array([by[tag][s]["converged"] for s in seeds])
    ks = np.array([by[tag][s]["stop"]["k"] for s in seeds])
    d2 = np.array([abs(by[tag][s]["cont"]["mrd"] - by[tag][s]["stop"]["mrd"]) for s in seeds])
    vals, cnt = np.unique(np.round(H_m, 4), return_counts=True)
    print(f"\n{tag}: converged {conv.sum()}/{S}; k_stop (converged) median {int(np.median(ks[conv]))} "
          f"p90 {int(np.quantile(ks[conv], 0.9))} max {ks[conv].max()}; |2x-stop| converged max {d2[conv].max():.1e} "
          f"(global-basin runs max {d2[conv & (H_m < np.min(H_m) + 1e-3)].max():.1e}); non-converged seeds "
          f"{[seeds[i] for i in np.where(~conv)[0]]} mrd {[round(x, 4) for x in H_m[~conv]]}")
    print("   basins (mrd rounded 4dp x count): " + ", ".join(f"{a:.4f}x{c}" for a, c in zip(vals, cnt)))

    def stats(ix):
        hb = H_m[ix][np.argmin(H_l[ix])]
        eb = E_m[ix][np.argmin(E_l[ix])]
        return hb / eb, np.median(H_m[ix]) / np.median(E_m[ix]), eb, H_m[ix].min() / E_m[ix].min()

    st = np.array([stats(ix) for ix in draws])
    print(f"   5-subsets (20000 draws): best ratio quantiles min {st[:, 0].min():.3f} p50 {np.median(st[:, 0]):.3f} "
          f"max {st[:, 0].max():.3f}; median ratio p50 {np.median(st[:, 1]):.3f} p99 {np.quantile(st[:, 1], 0.99):.3f} "
          f"p99.9 {np.quantile(st[:, 1], 0.999):.3f} max {st[:, 1].max():.3f}")
    guard = np.abs(st[:, 2] - 0.1506) <= 0.003
    for bt, mt in [(0.5, 0.8), (0.4, 0.6), (0.35, 0.5), (0.3, 0.5), (0.3, 0.4), (0.25, 0.3)]:
        ok = (st[:, 0] <= bt) & (st[:, 1] <= mt) & guard
        print(f"   pass(best<={bt}, median<={mt}, guard 0.1506+-0.003) = {ok.mean():.4f}")
    allp = np.mean([(H_m[ix] < E_m[ix]).all() for ix in draws])
    print(f"   all 5 paired H<E: {allp:.4f}; guard alone {guard.mean():.4f}")
    g = np.array([stats(ix) for ix in groups])
    print("   disjoint groups (best ratio, median ratio): " + " ".join(f"({a:.3f},{b:.3f})" for a, b, *_ in g))
    ix = np.arange(5)
    a, b, eb, _ = stats(ix)
    print(f"   seeds 0-4: best ratio {a:.4f} median ratio {b:.4f} E best {eb:.4f} | H 0-4 {[round(x, 4) for x in H_m[:5]]} "
          f"E 0-4 {[round(x, 4) for x in E_m[:5]]}")
    ix = np.arange(10)
    a, b, eb, _ = stats(ix)
    print(f"   seeds 0-9: best ratio {a:.4f} median ratio {b:.4f} | H best {H_m[ix][np.argmin(H_l[ix])]:.5f} "
          f"median {np.median(H_m[ix]):.5f}; E best {eb:.5f} median {np.median(E_m[ix]):.5f}")
