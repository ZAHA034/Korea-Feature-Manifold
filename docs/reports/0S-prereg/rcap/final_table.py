"""Print every number cited in the final answer from the result files (halving protocol K=8 unless noted)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, name)
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def best(rows):
    return min(rows, key=lambda r: r["stop"]["loss"])


def group(rows, space, **kw):
    out = []
    for r in rows:
        j = r["job"]
        if j["space"] != space:
            continue
        if all(j.get(k) == v for k, v in kw.items()):
            out.append(r)
    return out


def line(tag, H, E):
    hb, eb = best(H), best(E)
    hm = np.median([r["stop"]["mrd"] for r in H])
    em = np.median([r["stop"]["mrd"] for r in E])
    conv = sum(r["converged"] for r in H)
    ks = [r["stop"]["k"] for r in H if r["converged"]]
    d2 = [abs(r["cont"]["mrd"] - r["stop"]["mrd"]) for r in H if r["converged"]]
    print(f"{tag:28s} H best {hb['stop']['mrd']:.5f} (conv {hb['converged']}) med {hm:.5f} | E best {eb['stop']['mrd']:.5f} med {em:.5f} | "
          f"ratio best {hb['stop']['mrd'] / eb['stop']['mrd']:.4f} med {hm / em:.4f} stress {hb['stop']['loss'] / eb['stop']['loss']:.4f} | "
          f"conv {conv}/{len(H)} k med {int(np.median(ks)) if ks else '-'} max {max(ks) if ks else '-'} | "
          f"max|2x| {max(d2) if d2 else float('nan'):.1e} | at_cap(best) {hb['stop'].get('at_cap')} maxrad(all) {max(r['stop'].get('max_radius', 0) for r in H):.3f} s {hb['stop']['s']:.4f}")


tm = load("h8_tree_main.jsonl")
E = group(tm, "euclid")
for R in [2, 3, 4, 5, 6, 8, None]:
    line(f"tree R{R}", group(tm, "poincare", rmax=R), E)
for ds in ["grid_road", "grid_road_hw", "grid_l1"]:
    g = load(f"h8_grid_{ds}.jsonl")
    for R in [2, 3, 5, 8]:
        line(f"{ds} R{R}", group(g, "poincare", rmax=R), group(g, "euclid"))
    gf = load(f"h8_gridfixed_{ds}.jsonl")
    for DH in [4, 6, 10, 16]:
        H = group(gf, "poincare", DH=DH)
        if H:
            line(f"{ds} fixed DH{DH}", H, group(g, "euclid"))
tf = load("h8_tree_fixed.jsonl") or load("tree_fixed.jsonl")
for DH in [4, 6, 10, 16]:
    H = group(tf, "poincare", DH=DH)
    if H:
        n_glob = sum(r["stop"]["mrd"] < 0.0345 for r in H) if DH == 10 else None
        line(f"tree fixed DH{DH}" + (f" (global {n_glob}/10)" if n_glob is not None else ""), H, E)
t7 = load("h8_tree7.jsonl")
if t7:
    E7 = group(t7, "euclid")
    if E7:
        for R in [3, 5, 8]:
            H = group(t7, "poincare", rmax=R)
            if H:
                line(f"tree7 R{R}", H, E7)
