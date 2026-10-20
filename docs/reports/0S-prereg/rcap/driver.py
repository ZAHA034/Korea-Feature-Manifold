"""Parallel job runner. usage: python driver.py <plan> <out.jsonl> [workers]"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capemb as C  # noqa: E402

_CACHE = {}


def problem(ds):
    if ds not in _CACHE:
        _CACHE[ds] = C.Problem(C.DATASETS[ds]())
    return _CACHE[ds]


def work(job):
    P = problem(job["ds"])
    s_fixed = None
    if job.get("DH") is not None:
        s_fixed = P.diam / job["DH"]
    r = C.run(P, job["space"], job["seed"], rmax=job.get("rmax"), lr=job.get("lr", 0.05),
              tol=job.get("tol", 1e-6), window=job.get("window", 200), tmax=job.get("tmax", 50000),
              cont=job.get("cont", 2.0), s_fixed=s_fixed, keepX=bool(job.get("polish") or job.get("polishA")),
              halvings=job.get("K", 0))
    if job.get("polish") and "X_stop" in r:
        t0 = time.perf_counter()
        if job["space"] == "poincare":
            pol, _ = C.polish_poincare_polar(P, r["X_stop"], job.get("rmax"), s_fixed)
        else:
            pol, _ = C.polish_euclid(P, r["X_stop"])
        pol["seconds"] = time.perf_counter() - t0
        r["polish"] = pol
    if job.get("polishA") and "X_stop" in r:
        for tag in ["stop", "cont"]:
            if job["space"] == "poincare":
                pol = C.polish_rho_w(P, r["X_" + tag], job["rmax"], ftol=job.get("pftol", 1e-15))
            else:
                t0 = time.perf_counter()
                pol, _ = C.polish_euclid(P, r["X_" + tag])
                pol["seconds"] = time.perf_counter() - t0
            r["pA_" + tag] = pol
    r.pop("X_stop", None)
    r.pop("X_cont", None)
    r["job"] = job
    r["ckpt"] = {str(k): v for k, v in r["ckpt"].items()}
    return r


def plan(name):
    J = []
    if name == "tree_main":
        for seed in range(10):
            for rmax in [2, 3, 5, 8, None]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, polish=True))
            J.append(dict(ds="tree", space="euclid", seed=seed, polish=True))
    elif name == "tree_lr":
        for seed in range(10):
            for rmax in [3, 5, 8]:
                for lr in [0.02, 0.1]:
                    J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, lr=lr))
            for lr in [0.02, 0.1]:
                J.append(dict(ds="tree", space="euclid", seed=seed, lr=lr))
    elif name == "tree_tol":
        for seed in range(10):
            for rmax in [3, 5, 8]:
                for tol, win in [(1e-5, 200), (1e-7, 200), (1e-6, 500)]:
                    J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, tol=tol, window=win))
    elif name == "tree_fixed":
        for seed in range(10):
            for DH in [4, 6, 10, 16]:
                J.append(dict(ds="tree", space="poincare", seed=seed, DH=DH, rmax=None))
    elif name == "tree50":
        for seed in range(50):
            for rmax in [2, 3, 4, 5, 6, 8]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax))
            J.append(dict(ds="tree", space="euclid", seed=seed))
    elif name.startswith("grid:"):
        ds = name.split(":")[1]
        for seed in range(10):
            for rmax in [2, 3, 5, 8]:
                J.append(dict(ds=ds, space="poincare", seed=seed, rmax=rmax))
            J.append(dict(ds=ds, space="euclid", seed=seed))
    elif name.startswith("gridfixed:"):
        ds = name.split(":")[1]
        for seed in range(10):
            for DH in [4, 6, 10, 16]:
                J.append(dict(ds=ds, space="poincare", seed=seed, DH=DH, rmax=None))
    elif name.startswith("griduncap:"):
        ds = name.split(":")[1]
        for seed in range(10):
            J.append(dict(ds=ds, space="poincare", seed=seed, rmax=None))
    # ---- protocol with plateau lr halving (K = 8) for both spaces
    elif name == "h8_tree_main":
        for seed in range(10):
            for rmax in [2, 3, 4, 5, 6, 8, None]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, K=8, polish=True))
            J.append(dict(ds="tree", space="euclid", seed=seed, K=8, polish=True))
    elif name == "h8_tree50":
        for seed in range(50):
            for rmax in [3, 4, 5, 6]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, K=8))
            J.append(dict(ds="tree", space="euclid", seed=seed, K=8))
    elif name == "h8_tree_sens":
        for seed in range(10):
            for lr in [0.02, 0.1]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=5, K=8, lr=lr))
                J.append(dict(ds="tree", space="euclid", seed=seed, K=8, lr=lr))
            for tol, win in [(1e-5, 200), (1e-7, 200), (1e-6, 500)]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=5, K=8, tol=tol, window=win))
    elif name.startswith("h8_grid:"):
        ds = name.split(":")[1]
        for seed in range(10):
            for rmax in [2, 3, 5, 8]:
                J.append(dict(ds=ds, space="poincare", seed=seed, rmax=rmax, K=8, polish=(rmax == 5)))
            J.append(dict(ds=ds, space="euclid", seed=seed, K=8, polish=True))
    elif name.startswith("h8_gridfixed:"):
        ds = name.split(":")[1]
        for seed in range(5):
            for DH in [4, 6, 10, 16]:
                J.append(dict(ds=ds, space="poincare", seed=seed, DH=DH, rmax=None, K=8))
    elif name == "h8_tree7":
        for seed in range(5):
            for rmax in [3, 5, 8]:
                J.append(dict(ds="tree7", space="poincare", seed=seed, rmax=rmax, K=8))
            J.append(dict(ds="tree7", space="euclid", seed=seed, K=8))
    elif name == "h8_tree_fixed":
        for seed in range(10):
            for DH in [4, 6, 10, 16]:
                J.append(dict(ds="tree", space="poincare", seed=seed, DH=DH, rmax=None, K=8))
    # ---- protocol A: Adam (plain 1e-6/200 rule, tmax 50k) + bounded L-BFGS-B polish
    elif name == "A_tree50":
        for seed in range(50):
            for rmax in [3, 4, 5, 6]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, polishA=True))
            J.append(dict(ds="tree", space="euclid", seed=seed, polishA=True))
    elif name == "A2_tree50":
        for seed in range(50):
            for rmax in [5, 6]:
                J.append(dict(ds="tree", space="poincare", seed=seed, rmax=rmax, polishA=True, pftol=1e-12))
    elif name.startswith("A_grid:"):
        ds = name.split(":")[1]
        for seed in range(10):
            J.append(dict(ds=ds, space="poincare", seed=seed, rmax=5, polishA=True))
            J.append(dict(ds=ds, space="euclid", seed=seed, polishA=True))
    elif name == "A_tree7":
        for seed in range(5):
            for rmax in [3, 5]:
                J.append(dict(ds="tree7", space="poincare", seed=seed, rmax=rmax, polishA=True))
            J.append(dict(ds="tree7", space="euclid", seed=seed, polishA=True))
    elif name == "h8_grid_euc":
        for seed in range(3):
            J.append(dict(ds="grid_euc", space="poincare", seed=seed, rmax=5, K=8, cont=1.0))
            J.append(dict(ds="grid_euc", space="euclid", seed=seed, K=8, cont=1.0))
    else:
        raise ValueError(name)
    # n = 225 poincare jobs first (longest)
    J.sort(key=lambda j: (j["ds"] == "tree", j["space"] == "euclid"))
    return J


def main():
    name, out = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    jobs = plan(name)
    # longest first: n=225 poincare jobs before euclid
    t0 = time.time()
    with mp.Pool(workers) as pool, open(out, "w") as f:
        for i, r in enumerate(pool.imap_unordered(work, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % max(1, len(jobs) // 20) == 0 or i + 1 == len(jobs):
                print(f"{name}: {i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
