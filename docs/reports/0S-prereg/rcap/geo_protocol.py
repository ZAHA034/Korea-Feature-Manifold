"""Full proposed protocol in geoopt (RiemannianAdam lr 0.05, cap, closed-form s, plateau halving K=8,
stop rule 1e-6 / 200, tmax 50000), tree fixture, float64 and float32. usage: python geo_protocol.py out.json"""
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import geoopt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capemb as C  # noqa: E402

torch.set_num_threads(1)
ball = geoopt.PoincareBall(c=1.0)


def fit(D, space, seed, rmax, dtype=torch.float64, lr=0.05, tol=1e-6, window=200, tmax=50000, K=8):
    t0 = time.perf_counter()
    P = C.Problem(D)
    X0 = np.random.default_rng(seed).uniform(-0.1, 0.1, size=(P.n, 2))
    It, Jt = torch.tensor(P.iu[0]), torch.tensor(P.iu[1])
    d = torch.tensor(P.D[P.iu], dtype=dtype)
    if space == "poincare":
        X = geoopt.ManifoldParameter(torch.tensor(X0, dtype=dtype), manifold=ball)
        dist = lambda X: ball.dist(X[It], X[Jt])  # noqa: E731
    else:
        X = torch.nn.Parameter(torch.tensor(X0, dtype=dtype))
        dist = lambda X: (X[It] - X[Jt]).norm(dim=-1)  # noqa: E731
    opt = geoopt.optim.RiemannianAdam([X], lr=lr)
    c = float(np.tanh(rmax / 2)) if (rmax is not None and space == "poincare") else None
    best, besth, k_change, nh, k = np.inf, [], 0, 0, 0
    conv = False
    while True:
        opt.zero_grad()
        p = dist(X)
        q = p / d
        s = (q.sum() / (q * q).sum()).detach()
        loss = (((s * p - d) / d) ** 2).sum()
        L = float(loss)
        best = min(best, L)
        besth.append(best)
        plateau = k - k_change >= window and (besth[k - window] - best) <= tol * best
        if plateau and nh < K:
            nh += 1
            k_change = k
            for g in opt.param_groups:
                g["lr"] *= 0.5
            plateau = False
        if plateau or k >= tmax:
            conv = bool(plateau)
            break
        loss.backward()
        opt.step()
        k += 1
        if c is not None:
            with torch.no_grad():
                nr = X.norm(dim=-1, keepdim=True)
                X.copy_(torch.where(nr > c, X / nr * c, X))
    with torch.no_grad():
        Xd = X.detach().double().numpy()
    # evaluate in float64 numpy at the final float32/64 coordinates
    p = C.poincare_dist(Xd)[0] if space == "poincare" else C.euclid_dist(Xd)
    s = C.s_star(p, P)
    out = {"space": space, "seed": seed, "rmax": rmax, "dtype": str(dtype).split(".")[-1], "k_stop": k,
           "converged": conv, "mrd": C.mrd(p, P, s), "s": s, "loss_internal": L,
           "seconds": time.perf_counter() - t0, "finite": bool(np.isfinite(p).all())}
    if space == "poincare":
        rad = C.radius(Xd)
        out["max_radius"] = float(rad.max())
        lim = min(rmax if rmax is not None else 99, 2 * np.arctanh(1 - (4e-3 if dtype == torch.float32 else 1e-5)))
        out["at_cap"] = int((rad >= lim - 1e-4).sum())
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "grid":
        D = C.DATASETS["grid_road"]()
        res = [fit(D, "euclid", 0, None), fit(D, "poincare", 0, 5)]
        for r in res:
            print(r, flush=True)
        json.dump(res, open(sys.argv[1], "w"), indent=1)
        return
    D = C.binary_tree(4)
    res = []
    for dtype in [torch.float64, torch.float32]:
        for seed in range(5):
            res.append(fit(D, "euclid", seed, None, dtype))
            print(res[-1], flush=True)
            for rmax in ([5] if dtype == torch.float64 else [5, 8]):
                res.append(fit(D, "poincare", seed, rmax, dtype))
                print(res[-1], flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
