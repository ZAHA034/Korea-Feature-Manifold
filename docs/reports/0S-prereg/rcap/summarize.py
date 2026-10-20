"""Summaries of driver output. usage: python summarize.py <file.jsonl> [--seeds a-b] [--detail]"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path)]


def key(r):
    j = r["job"]
    kk = f"K{j.get('K', 0)}"
    if j["space"] == "euclid":
        return (j["ds"], "E" + kk, j.get("lr", 0.05), j.get("tol", 1e-6), j.get("window", 200))
    tag = f"DH{j['DH']}" if j.get("DH") is not None else f"R{j.get('rmax')}"
    return (j["ds"], tag + kk, j.get("lr", 0.05), j.get("tol", 1e-6), j.get("window", 200))


def main():
    path = sys.argv[1]
    detail = "--detail" in sys.argv
    seeds = None
    if "--seeds" in sys.argv:
        a, b = sys.argv[sys.argv.index("--seeds") + 1].split("-")
        seeds = set(range(int(a), int(b) + 1))
    R = load(path)
    groups = defaultdict(list)
    for r in R:
        if seeds is not None and r["seed"] not in seeds:
            continue
        groups[key(r)].append(r)
    Ekeys = {}
    for k in groups:
        if k[1].startswith("E"):
            Ekeys[(k[0], k[2])] = k
    for k in sorted(groups, key=lambda x: (x[0], x[2], x[3], x[4], x[1])):
        rs = sorted(groups[k], key=lambda r: r["seed"])
        stop = np.array([r["stop"]["mrd"] for r in rs])
        loss = np.array([r["stop"]["loss"] for r in rs])
        cont = np.array([r["cont"]["mrd"] for r in rs])
        ks = np.array([r["stop"]["k"] for r in rs])
        conv = np.array([r["converged"] for r in rs])
        ib = int(np.argmin(loss))
        line = (f"{k[0]:13s} {k[1]:6s} lr{k[2]:<5} tol{k[3]:.0e} w{k[4]:<4d} n={len(rs):2d} conv {conv.sum():2d}/{len(rs)} "
                f"k_stop med {int(np.median(ks)):6d} max {ks.max():6d} | mrd best(minloss) {stop[ib]:.4f} "
                f"min {stop.min():.4f} med {np.median(stop):.4f} | 2x: best {cont[ib]:.4f} med {np.median(cont):.4f} "
                f"max|d| {np.max(np.abs(cont - stop)):.1e}")
        if not k[1].startswith("E"):
            atcap = np.array([r["stop"].get("at_cap", 0) for r in rs])
            mr = np.array([r["stop"].get("max_radius", 0) for r in rs])
            n = 31 if k[0] == "tree" else 225
            line += f" | at_cap(best) {atcap[ib]}/{n} med {np.median(atcap):.0f} maxrad {mr.max():.2f}"
            ek = Ekeys.get((k[0], k[2]))
            if ek is None:
                ek = Ekeys.get((k[0], 0.05))
            if ek is not None:
                es = sorted(groups[ek], key=lambda r: r["seed"])
                e_stop = np.array([r["stop"]["mrd"] for r in es])
                e_loss = np.array([r["stop"]["loss"] for r in es])
                eb = e_stop[int(np.argmin(e_loss))]
                line += f" | ratio best {stop[ib] / eb:.3f} med {np.median(stop) / np.median(e_stop):.3f} stress-ratio(best) {loss[ib] / e_loss.min():.3f}"
        if "polish" in rs[0]:
            pm = np.array([r["polish"]["mrd"] for r in rs])
            pl = np.array([r["polish"]["loss"] for r in rs])
            line += (f" | polish: best {pm[int(np.argmin(pl))]:.4f} med {np.median(pm):.4f} "
                     f"max(loss_adam/loss_pol-1) {np.max(loss / pl - 1):.1e} max|dmrd| {np.max(np.abs(pm - stop)):.1e}")
        print(line)
        if detail:
            for r in rs:
                s = r["stop"]
                extra = f" polish {r['polish']['mrd']:.5f} ({r['polish']['loss']:.6f}, nit {r['polish']['nit']})" if "polish" in r else ""
                print(f"    seed {r['seed']:2d} k_stop {s['k']:6d} conv {int(r['converged'])} loss {s['loss']:.6f} "
                      f"mrd {s['mrd']:.5f} -> 2x {r['cont']['mrd']:.5f} (k {r['cont']['k']}) s {s['s']:.4f} "
                      f"maxrad {s.get('max_radius', 0):.3f} at_cap {s.get('at_cap', '-')} nonmono {r.get('nonmono_to_stop')} "
                      f"{r['seconds']:.1f}s" + extra)


if __name__ == "__main__":
    main()
