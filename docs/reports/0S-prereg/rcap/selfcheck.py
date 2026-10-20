"""Self checks: gradients, distance vs geoopt, numpy optimizer == geoopt RiemannianAdam (+cap), fixed-T replication."""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import geoopt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capemb as C  # noqa: E402

torch.set_num_threads(1)
res = {}

# 1. gradient checks (central differences)
rng = np.random.default_rng(5)
for name in ["tree", "grid_road_hw"]:
    D = C.DATASETS[name]()
    if name != "tree":
        idx = rng.choice(len(D), 25, replace=False)
        D = D[np.ix_(idx, idx)]
    P = C.Problem(D)
    for space, lg in [("poincare", C.poincare_lg), ("euclid", C.euclid_lg)]:
        for sf in [None, 0.37]:
            X = rng.uniform(-0.6, 0.6, size=(P.n, 2)) * (1.0 if space == "poincare" else 5.0)
            L, G, s, p = lg(X, P, sf)
            h = 1e-6
            num = np.zeros_like(X)
            for i in range(P.n):
                for k in range(2):
                    Xp = X.copy(); Xp[i, k] += h
                    Xm = X.copy(); Xm[i, k] -= h
                    num[i, k] = (lg(Xp, P, sf)[0] - lg(Xm, P, sf)[0]) / (2 * h)
            res[f"gradcheck_{name}_{space}_{'fixed' if sf else 'profiled'}"] = float(np.abs(num - G).max() / np.abs(G).max())

# 2. distance vs geoopt
X = rng.uniform(-0.7, 0.7, size=(40, 2))
X = C.project_eps(X)
ball = geoopt.PoincareBall(c=1.0)
Xt = torch.tensor(X, dtype=torch.float64)
I, J = np.triu_indices(40, 1)
pg = ball.dist(Xt[I], Xt[J]).numpy()
pn = C.poincare_dist(X)[0][I, J]
res["dist_vs_geoopt_maxrelerr"] = float(np.max(np.abs(pg - pn) / pg))
res["d0x_closed_form_err"] = float(abs(C.poincare_dist(np.array([[0.0, 0.0], [0.3, 0.4]]))[0][0, 1] - 2 * np.arctanh(0.5)))
# cap: d_P(0, cap(x)) == Rmax
xs = rng.uniform(-1, 1, size=(10, 2)); xs = C.project_eps(xs * 0.999 / np.linalg.norm(xs, axis=1, keepdims=True))
for R in [2, 3, 5, 8]:
    res[f"cap_radius_err_R{R}"] = float(np.max(np.abs(C.radius(C.cap(xs, R)) - R)))


# 3. numpy optimizer vs geoopt RiemannianAdam with the same cap (applied after each opt.step())
def geoopt_run(D, seed, rmax, lr, T, dtype=torch.float64, ck=(200, 1000, 3000)):
    P = C.Problem(D)
    X0 = np.random.default_rng(seed).uniform(-0.1, 0.1, size=(P.n, 2))
    It, Jt = torch.tensor(P.iu[0]), torch.tensor(P.iu[1])
    d = torch.tensor(P.D[P.iu], dtype=dtype)
    Xp = geoopt.ManifoldParameter(torch.tensor(X0, dtype=dtype), manifold=ball)
    opt = geoopt.optim.RiemannianAdam([Xp], lr=lr)
    c = float(np.tanh(rmax / 2)) if rmax is not None else None
    out = {}
    for t in range(1, T + 1):
        opt.zero_grad()
        p = ball.dist(Xp[It], Xp[Jt])
        q = p / d
        s = (q.sum() / (q * q).sum()).detach()
        loss = (((s * p - d) / d) ** 2).sum()
        loss.backward()
        opt.step()
        if c is not None:
            with torch.no_grad():
                nr = Xp.norm(dim=-1, keepdim=True)
                Xp.copy_(torch.where(nr > c, Xp / nr * c, Xp))
        if t in ck:
            with torch.no_grad():
                pp = ball.dist(Xp[It], Xp[Jt]).double().numpy()
                qq = pp / P.D[P.iu]
                ss = qq.sum() / (qq * qq).sum()
                out[t] = {"mrd": float(np.mean(np.abs(ss * qq - 1))), "X": Xp.detach().double().numpy().copy()}
    return out


Dt = C.binary_tree(4)
Pt = C.Problem(Dt)
eq = {}
t0 = time.time()
for rmax in [None, 3, 5]:
    for seed in [0, 1]:
        g = geoopt_run(Dt, seed, rmax, 0.05, 3000)
        r = C.run(Pt, "poincare", seed, rmax=rmax, lr=0.05, tol=-1, tmax=3000, cont=1.0, ckpts=(200, 1000, 3000))
        for T in (200, 1000, 3000):
            eq[f"R{rmax}_s{seed}_T{T}"] = {"geoopt": g[T]["mrd"], "numpy": r["ckpt"][T]["mrd"],
                                          "absdiff": abs(g[T]["mrd"] - r["ckpt"][T]["mrd"])}
res["numpy_vs_geoopt_capped"] = eq
res["numpy_vs_geoopt_max_absdiff"] = max(v["absdiff"] for v in eq.values())
print("geoopt equivalence done", time.time() - t0, flush=True)

# 4. fixed-T replication of the earlier protocol (uncapped, T=2000, lr 0.05, U(+-0.1), seeds 0-9)
rep = {"P": [], "E": []}
for seed in range(10):
    rP = C.run(Pt, "poincare", seed, rmax=None, lr=0.05, tol=-1, tmax=2000, cont=1.0, ckpts=(2000,))
    rE = C.run(Pt, "euclid", seed, lr=0.05, tol=-1, tmax=2000, cont=1.0, ckpts=(2000,))
    rep["P"].append(rP["ckpt"][2000]["mrd"])
    rep["E"].append(rE["ckpt"][2000]["mrd"])
res["fixedT2000_P_med"] = float(np.median(rep["P"]))
res["fixedT2000_P_min"] = float(np.min(rep["P"]))
res["fixedT2000_P_max"] = float(np.max(rep["P"]))
res["fixedT2000_E_med"] = float(np.median(rep["E"]))
res["fixedT2000_E_min"] = float(np.min(rep["E"]))
res["fixedT2000_P"] = [round(x, 4) for x in rep["P"]]
res["fixedT2000_E"] = [round(x, 4) for x in rep["E"]]
# timing
for name in ["tree", "grid_road"]:
    P = C.Problem(C.DATASETS[name]())
    X = np.random.default_rng(0).uniform(-0.1, 0.1, size=(P.n, 2))
    t0 = time.perf_counter()
    for _ in range(200):
        C.poincare_lg(X, P)
    res[f"ms_per_eval_{name}"] = (time.perf_counter() - t0) / 200 * 1e3
print(json.dumps({k: v for k, v in res.items() if k != "numpy_vs_geoopt_capped"}, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfcheck.json"), "w"), indent=1)
