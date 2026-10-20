"""Does a final L-BFGS-B polish (rho_i in [0, Rmax] box, free direction) make the converged value crisp?
1) geoopt grid_road seed 0 R5 (premature stop) -> polish
2) tree7 R5 seeds 0, 2 (numpy; seed 2 crawls past 50k) -> polish at 50k, compare with Adam at 100k
3) general-d polish check: tree depth 4, dim 5, R5"""
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capemb as C  # noqa: E402


def polish_rho_w(P, X, rmax, maxiter=20000, ftol=1e-15, gtol=1e-11):
    """General-dimension bounded polish: x_i = tanh(rho_i/2) * w_i/|w_i|, rho_i in [0, rmax], w_i free."""
    n, dim = X.shape
    nr = np.linalg.norm(X, axis=1)
    rho0 = np.minimum(2 * np.arctanh(np.minimum(nr, 1 - 1e-12)), rmax)
    w0 = X / np.maximum(nr, 1e-12)[:, None]

    def unpack(z):
        rho = z[:n]
        w = z[n:].reshape(n, dim)
        wn = np.maximum(np.linalg.norm(w, axis=1), 1e-300)
        u = w / wn[:, None]
        a = np.tanh(rho / 2)
        return rho, w, wn, u, a

    def f(z):
        rho, w, wn, u, a = unpack(z)
        Xc = a[:, None] * u
        L, G, s, p = C.poincare_lg(Xc, P)
        g_rho = (G * u).sum(1) * 0.5 * (1 - a * a)
        Gu = G * a[:, None]                       # dL/du
        g_w = (Gu - (Gu * u).sum(1, keepdims=True) * u) / wn[:, None]
        return L, np.concatenate([g_rho, g_w.ravel()])

    z0 = np.concatenate([rho0, w0.ravel()])
    bounds = [(0.0, rmax)] * n + [(None, None)] * (n * dim)
    t0 = time.perf_counter()
    res = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": ftol, "gtol": gtol, "maxcor": 30})
    rho, w, wn, u, a = unpack(res.x)
    Xc = a[:, None] * u
    L, G, s, p = C.poincare_lg(Xc, P)
    return {"loss": float(L), "mrd": C.mrd(p, P, s), "nit": int(res.nit), "msg": str(res.message),
            "at_cap": int((rho >= rmax - 1e-9).sum()), "max_radius": float(rho.max()),
            "seconds": time.perf_counter() - t0}


out = {}
# 1) geoopt grid_road seed 0, R5, same protocol as geo_protocol.fit (float64), keep X
import torch  # noqa: E402
import geoopt  # noqa: E402

torch.set_num_threads(1)
ball = geoopt.PoincareBall(c=1.0)
P = C.Problem(C.DATASETS["grid_road"]())
X0 = np.random.default_rng(0).uniform(-0.1, 0.1, size=(P.n, 2))
It, Jt = torch.tensor(P.iu[0]), torch.tensor(P.iu[1])
d = torch.tensor(P.D[P.iu], dtype=torch.float64)
Xp = geoopt.ManifoldParameter(torch.tensor(X0, dtype=torch.float64), manifold=ball)
opt = geoopt.optim.RiemannianAdam([Xp], lr=0.05)
c = float(np.tanh(2.5))
best, besth, k_change, nh, k = np.inf, [], 0, 0, 0
while True:
    opt.zero_grad()
    p = ball.dist(Xp[It], Xp[Jt])
    q = p / d
    s = (q.sum() / (q * q).sum()).detach()
    loss = (((s * p - d) / d) ** 2).sum()
    L = float(loss)
    best = min(best, L)
    besth.append(best)
    plateau = k - k_change >= 200 and (besth[k - 200] - best) <= 1e-6 * best
    if plateau and nh < 8:
        nh += 1
        k_change = k
        for g in opt.param_groups:
            g["lr"] *= 0.5
        plateau = False
    if plateau or k >= 50000:
        break
    loss.backward()
    opt.step()
    k += 1
    with torch.no_grad():
        nr = Xp.norm(dim=-1, keepdim=True)
        Xp.copy_(torch.where(nr > c, Xp / nr * c, Xp))
Xg = Xp.detach().numpy().copy()
Lg = C.poincare_lg(Xg, P)[0]
LE = 29.758672705023127
pol = polish_rho_w(P, Xg, 5.0)
out["geoopt_grid_road"] = {"k_stop": k, "loss_stop": Lg, "loss_stop_over_E": Lg / LE, "polish": pol,
                           "polish_over_E": pol["loss"] / LE}
print("geoopt grid_road:", out["geoopt_grid_road"], flush=True)

# 2) tree7 R5 seeds 0 and 2: Adam (numpy, K=8, tmax 50k) then polish
P7 = C.Problem(C.DATASETS["tree7"]())
for seed in [2, 0]:
    r = C.run(P7, "poincare", seed, rmax=5, halvings=8, keepX=True, cont=1.0)
    pol = polish_rho_w(P7, r["X_stop"], 5.0)
    out[f"tree7_R5_seed{seed}"] = {"k_stop": r["stop"]["k"], "converged": r["converged"],
                                   "loss_stop": r["stop"]["loss"], "mrd_stop": r["stop"]["mrd"], "polish": pol}
    print(f"tree7 R5 seed {seed}:", out[f"tree7_R5_seed{seed}"], flush=True)

# 3) general-d polish on tree depth 4, dim 5, R5
Pt = C.Problem(C.binary_tree(4))
r = C.run(Pt, "poincare", 1, rmax=5, halvings=8, keepX=True, cont=1.0, dim=5)
pol = polish_rho_w(Pt, r["X_stop"], 5.0)
out["tree_dim5_R5_seed1"] = {"k_stop": r["stop"]["k"], "loss_stop": r["stop"]["loss"], "mrd_stop": r["stop"]["mrd"],
                             "polish": pol}
print("tree dim5:", out["tree_dim5_R5_seed1"], flush=True)
# 2D consistency of the general polish vs polar polish (tree R5 seed 1)
r = C.run(Pt, "poincare", 1, rmax=5, halvings=8, keepX=True, cont=1.0)
pol_g = polish_rho_w(Pt, r["X_stop"], 5.0)
pol_p, _ = C.polish_poincare_polar(Pt, r["X_stop"], 5.0)
out["tree_2d_general_vs_polar"] = {"general": pol_g, "polar": pol_p}
print("tree 2d general vs polar:", pol_g["loss"], pol_p["loss"], pol_g["mrd"], pol_p["mrd"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "polish_checks.json"), "w"), indent=1,
          default=float)
