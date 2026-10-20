"""Exact pass rates over all C(50,5) seed subsets (vectorized). usage: python passexact.py h8_tree50.jsonl"""
import json
import sys
from itertools import combinations

import numpy as np

R = [json.loads(line) for line in open(sys.argv[1])]
by = {}
for r in R:
    j = r["job"]
    tag = "E" if j["space"] == "euclid" else f"R{j['rmax']}"
    by.setdefault(tag, {})[r["seed"]] = r
seeds = sorted(by["E"])
S = len(seeds)
C5 = np.array(list(combinations(range(S), 5)), dtype=np.int32)
E_m = np.array([by["E"][s]["stop"]["mrd"] for s in seeds])
E_l = np.array([by["E"][s]["stop"]["loss"] for s in seeds])
ie = C5[np.arange(len(C5)), np.argmin(E_l[C5], axis=1)]
eb = E_m[ie]
emed = np.median(E_m[C5], axis=1)
guard = np.abs(eb - 0.1506) <= 0.003
print(f"subsets {len(C5)}; E guard pass {guard.mean():.6f}")
REF = {"R3": 0.0664, "R4": 0.0465, "R5": 0.0339, "R6": 0.0260}
for tag in ["R3", "R4", "R5", "R6"]:
    H_m = np.array([by[tag][s]["stop"]["mrd"] for s in seeds])
    H_l = np.array([by[tag][s]["stop"]["loss"] for s in seeds])
    H_c = np.array([by[tag][s]["converged"] for s in seeds])
    ih = C5[np.arange(len(C5)), np.argmin(H_l[C5], axis=1)]
    hb = H_m[ih]
    rb = hb / eb
    rm = np.median(H_m[C5], axis=1) / emed
    c4 = (np.abs(hb - REF[tag]) <= 0.001) & H_c[ih]
    print(f"{tag}: best ratio min {rb.min():.4f} max {rb.max():.4f}; median ratio max {rm.max():.4f}; "
          f"P(best restart converged & H=ref+-0.001) {c4.mean():.6f}")
    for bt, mt, u4 in [(0.5, 0.8, False), (0.3, 0.8, False), (0.3, 0.8, True), (0.5, 0.8, True), (0.3, 0.6, True)]:
        ok = (rb <= bt) & (rm <= mt) & guard & (c4 if u4 else True)
        print(f"   best<={bt} median<={mt} guardE{' + H=' + str(REF[tag]) + '+-0.001&conv' if u4 else ''}: {ok.mean():.6f} "
              f"(fail {int((~ok).sum())} of {len(C5)})")
    allHE = (H_m[C5] < E_m[C5]).all(axis=1)
    print(f"   all 5 paired H<E: {allHE.mean():.4f}")
