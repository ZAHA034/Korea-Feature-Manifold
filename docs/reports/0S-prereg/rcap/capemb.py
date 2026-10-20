"""Relative-stress embeddings with a hard hyperbolic radius cap (Poincare ball, curvature -1).

Objective (ADR 0010 3): L = sum_{i<j} ((s * p_ij - d_ij) / d_ij)^2, s = closed form
    s* = sum(p/d) / sum((p/d)^2) every step (envelope theorem -> exact gradient of profiled L),
or a fixed s (fixed-scale alternative).
Judgment metric: mrd = mean_{i<j} |s * p_ij - d_ij| / d_ij.

Poincare optimizer = numpy mirror of geoopt 0.5.1 RiemannianAdam on PoincareBall(c=1):
    rgrad = egrad / lambda^2, m <- b1 m + (1-b1) rgrad, v <- b2 v + (1-b2) lambda^2 |rgrad|^2 (per point),
    y = project_eps(x - lr * (m/bc1) / (sqrt(v/bc2) + eps))   (retraction x+u, eps = 1e-5 in float64),
    m <- transp(x, y, m), then the hard cap y <- y * min(1, tanh(Rmax/2)/|y|)  (d_P(0, y) <= Rmax).
Euclid optimizer = Adam (geoopt RiemannianAdam on a Euclidean parameter), no cap.
Stopping: best-so-far objective B_k; converged at k if k >= W and (B_{k-W} - B_k) <= tol * B_k.
After the stop the same trajectory continues (stopping disabled) to cont * k_stop for the budget check.
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

B1, B2, AEPS = 0.9, 0.999, 1e-8
PEPS64 = 1e-5  # geoopt float64 projection eps -> max radius 2 artanh(1 - 1e-5) = 12.21


# ---------------------------------------------------------------- data
def binary_tree(depth=4):
    n = 2 ** (depth + 1) - 1
    A = csr_matrix((np.ones(n - 1), (list(range(1, n)), [(i - 1) // 2 for i in range(1, n)])), shape=(n, n))
    return shortest_path(A, unweighted=True, directed=False)


def grid_coords(side=15):
    ii, jj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return np.stack([ii.ravel(), jj.ravel()], 1).astype(float)


def grid_euclid(side=15):
    G = grid_coords(side)
    return np.linalg.norm(G[:, None] - G[None], axis=-1)


def grid_l1(side=15):
    G = grid_coords(side)
    return np.abs(G[:, None] - G[None]).sum(-1)


def floyd_warshall(W):
    D = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(D, 0.0)
    for k in range(D.shape[0]):
        np.minimum(D, D[:, [k]] + D[[k], :], out=D)
    return D


# copied verbatim from scratchpad/flowdist/scripts/kfmflow.py (earlier synthetic world, "15 x 15 grid")
def grid_world(n_side=15, spacing=20.0, jitter=4.0, detour=1.25, vmin=60.0, vmax=90.0, seed=0):
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    grid = np.stack([ii.ravel(), jj.ravel()], 1)
    X = grid * spacing + rng.uniform(-jitter, jitter, size=grid.shape)
    n = len(X)
    A = np.full((n, n), np.inf)
    np.fill_diagonal(A, 0.0)
    for r in range(n_side):
        for c in range(n_side):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < n_side and 0 <= c2 < n_side:
                    a, b = r * n_side + c, r2 * n_side + c2
                    L = np.linalg.norm(X[a] - X[b]) * detour
                    t = 60.0 * L / rng.uniform(vmin, vmax)
                    A[a, b] = A[b, a] = t
    return X, grid, A


def hierarchy(grid, n_side=15):
    idx = lambda r, c: r * n_side + c  # noqa: E731
    mid = n_side // 2
    capital = idx(mid, mid)
    reg_rc = [(3, 3), (3, 11), (11, 3), (11, 11)]
    regional = [idx(*rc) for rc in reg_rc]
    sub = [idx(r + dr, c + dc) for (r, c) in reg_rc for dr in (-2, 2) for dc in (-2, 2)]
    return capital, regional, sub


def add_highways(A, X, capital, regional, sub, v=100.0, detour=1.1):
    A2 = np.array(A, copy=True)
    pairs = [(capital, r) for r in regional] + [(regional[k // 4], s) for k, s in enumerate(sub)]
    for a, b in pairs:
        t = 60.0 * np.linalg.norm(X[a] - X[b]) * detour / v
        A2[a, b] = A2[b, a] = min(A2[a, b], t)
    return A2


def grid_road(seed=0, highways=False):
    X, grid, A = grid_world(seed=seed)
    if highways:
        A = add_highways(A, X, *hierarchy(grid))
    return floyd_warshall(A)


DATASETS = {
    "tree": lambda: binary_tree(4),
    "tree7": lambda: binary_tree(7),
    "grid_euc": lambda: grid_euclid(15),
    "grid_l1": lambda: grid_l1(15),
    "grid_road": lambda: grid_road(0, False),
    "grid_road_hw": lambda: grid_road(0, True),
}


class Problem:
    def __init__(self, D):
        D = np.asarray(D, dtype=np.float64)
        self.D = D
        self.n = n = len(D)
        self.off = ~np.eye(n, dtype=bool)
        self.Dsafe = np.where(self.off, D, 1.0)
        self.invD = np.where(self.off, 1.0 / self.Dsafe, 0.0)
        self.iu = np.triu_indices(n, 1)
        self.diam = float(D.max())


# ---------------------------------------------------------------- loss + gradient (dense)
def poincare_dist(X):
    sq = (X * X).sum(1)
    al = 1.0 - sq
    diff = X[:, None, :] - X[None, :, :]
    dl = (diff * diff).sum(-1)
    u = 2.0 * dl / (al[:, None] * al[None, :])
    np.fill_diagonal(u, 0.0)
    return np.log1p(u + np.sqrt(u * (u + 2.0))), u, dl, al


def poincare_lg(X, P, s_fixed=None):
    p, u, dl, al = poincare_dist(X)
    q = p * P.invD
    s = q.sum() / (q * q).sum() if s_fixed is None else s_fixed
    r = s * q - 1.0
    np.fill_diagonal(r, 0.0)
    L = 0.5 * (r * r).sum()
    us = np.where(P.off, np.maximum(u, 1e-30), 1.0)
    dpdu = 1.0 / np.sqrt(us * (us + 2.0))
    c = (2.0 * s) * r * P.invD * dpdu * (4.0 / (al[:, None] * al[None, :]))
    np.fill_diagonal(c, 0.0)
    G = X * c.sum(1)[:, None] - c @ X + X * ((c * dl).sum(1) / al)[:, None]
    return L, G, s, p


def euclid_dist(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def euclid_lg(X, P, s_fixed=None):
    p = euclid_dist(X)
    q = p * P.invD
    s = q.sum() / (q * q).sum() if s_fixed is None else s_fixed
    r = s * q - 1.0
    np.fill_diagonal(r, 0.0)
    L = 0.5 * (r * r).sum()
    c = (2.0 * s) * r * P.invD / np.where(P.off, np.maximum(p, 1e-300), 1.0)
    np.fill_diagonal(c, 0.0)
    G = X * c.sum(1)[:, None] - c @ X
    return L, G, s, p


def mrd(p, P, s):
    q = p[P.iu] / P.D[P.iu]
    return float(np.mean(np.abs(s * q - 1.0)))


def s_star(p, P):
    q = p[P.iu] / P.D[P.iu]
    return float(q.sum() / (q * q).sum())


# ---------------------------------------------------------------- Poincare ball ops (c = 1)
def lam(X):
    return 2.0 / (1.0 - (X * X).sum(-1, keepdims=True))


def project_eps(X, eps=PEPS64):
    nr = np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-15)
    maxn = 1.0 - eps
    return np.where(nr > maxn, X / nr * maxn, X)


def cap(X, rmax):
    if rmax is None:
        return X
    c = np.tanh(rmax / 2.0)
    nr = np.linalg.norm(X, axis=-1, keepdims=True)
    return np.where(nr > c, X / np.maximum(nr, 1e-300) * c, X)


def mobius_add(x, y):
    xy = (x * y).sum(-1, keepdims=True)
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    return ((1 + 2 * xy + y2) * x + (1 - x2) * y) / (1 + 2 * xy + x2 * y2)


def gyration(u, v, w):
    u2 = (u * u).sum(-1, keepdims=True)
    v2 = (v * v).sum(-1, keepdims=True)
    uv = (u * v).sum(-1, keepdims=True)
    uw = (u * w).sum(-1, keepdims=True)
    vw = (v * w).sum(-1, keepdims=True)
    a = -uw * v2 + vw + 2 * uv * vw
    b = -vw * u2 - uw
    d = 1 + 2 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / np.maximum(d, 1e-15)


def transp(x, y, v):
    return gyration(y, -x, v) * lam(x) / lam(y)


def radius(X):
    nr = np.linalg.norm(X, axis=1)
    return 2.0 * np.arctanh(np.minimum(nr, 1 - 1e-16))


# ---------------------------------------------------------------- optimizers
def _snap(space, X, P, L, s, p, k, rmax, s_fixed):
    rec = {"k": int(k), "loss": float(L), "s": float(s), "mrd": mrd(p, P, s)}
    if s_fixed is not None:
        rec["mrd_sstar"] = mrd(p, P, s_star(p, P))
    if space == "poincare":
        rad = radius(X)
        rec["max_radius"] = float(rad.max())
        rec["mean_radius"] = float(rad.mean())
        lim = rmax if rmax is not None else 2.0 * np.arctanh(1.0 - PEPS64)
        rec["at_cap"] = int((rad >= lim - 1e-9).sum())
    return rec


def run(P, space, seed, rmax=None, lr=0.05, tol=1e-6, window=200, tmax=50000, cont=2.0,
        s_fixed=None, init=0.1, ckpts=(1000, 2000, 5000, 10000, 20000, 50000), keepX=False, X0=None,
        halvings=0, dim=2):
    """One optimizer run. Returns dict with 'stop' (first time the rule fires, or tmax) and 'cont'
    (same trajectory continued to cont * k_stop with stopping disabled, at the final lr).
    halvings = K > 0: on a plateau (same test as the stopping rule, measured since the last lr change)
    halve lr; the stop fires on the first plateau after K halvings (lr_final = lr * 2^-K)."""
    t0 = time.perf_counter()
    lr0 = lr
    n_half = 0
    k_change = 0
    lr_hist = []
    rng = np.random.default_rng(seed)
    X = rng.uniform(-init, init, size=(P.n, dim)) if X0 is None else np.array(X0, dtype=float)
    lg = poincare_lg if space == "poincare" else euclid_lg
    if space == "poincare":
        X = cap(project_eps(X), rmax)
        v = np.zeros((P.n, 1))
    else:
        v = np.zeros_like(X)
    m = np.zeros_like(X)
    besth = []
    best = np.inf
    out = {"space": space, "seed": seed, "rmax": rmax, "lr": lr, "tol": tol, "window": window,
           "tmax": tmax, "s_fixed": s_fixed, "ckpt": {}}
    k_stop, k_end = None, None
    k = 0
    nonmono = 0
    prevL = np.inf
    while True:
        L, G, s, p = lg(X, P, s_fixed)
        if not np.isfinite(L):
            out["error"] = f"non-finite loss at k={k}"
            break
        if L > prevL:
            nonmono += 1
        prevL = L
        best = min(best, L)
        besth.append(best)
        if k in ckpts:
            out["ckpt"][k] = _snap(space, X, P, L, s, p, k, rmax, s_fixed)
        if k_stop is None:
            plateau = k - k_change >= window and (besth[k - window] - best) <= tol * best
            if plateau and n_half < halvings:
                lr *= 0.5
                n_half += 1
                k_change = k
                lr_hist.append(int(k))
                plateau = False
            conv = plateau
            if conv or k >= tmax:
                out["lr_changes"] = lr_hist
                out["lr_final"] = lr
                k_stop = k
                out["converged"] = bool(conv)
                out["stop"] = _snap(space, X, P, L, s, p, k, rmax, s_fixed)
                out["nonmono_to_stop"] = nonmono
                if keepX:
                    out["X_stop"] = X.copy()
                k_end = int(np.ceil(cont * max(k_stop, 1)))
        if k_stop is not None and k >= k_end:
            out["cont"] = _snap(space, X, P, L, s, p, k, rmax, s_fixed)
            if keepX:
                out["X_cont"] = X.copy()
            break
        # ---- one optimizer step
        k += 1
        bc1, bc2 = 1 - B1 ** k, 1 - B2 ** k
        if space == "poincare":
            lx = lam(X)
            rg = G / lx ** 2
            m = B1 * m + (1 - B1) * rg
            v = B2 * v + (1 - B2) * (lx ** 2 * (rg * rg).sum(1, keepdims=True))
            Y = project_eps(X - lr * (m / bc1) / (np.sqrt(v / bc2) + AEPS))
            m = transp(X, Y, m)
            X = cap(Y, rmax)
        else:
            m = B1 * m + (1 - B1) * G
            v = B2 * v + (1 - B2) * G * G
            X = X - lr * (m / bc1) / (np.sqrt(v / bc2) + AEPS)
    out["seconds"] = time.perf_counter() - t0
    if "stop" in out and "cont" in out:
        out["d_mrd_2x"] = out["cont"]["mrd"] - out["stop"]["mrd"]
        out["rel_d_mrd_2x"] = out["d_mrd_2x"] / out["stop"]["mrd"]
    return out


# ---------------------------------------------------------------- L-BFGS-B polish (verification only)
def polish_poincare_polar(P, X, rmax, s_fixed=None, maxiter=50000):
    """Constrained optimum check: x_i = tanh(rho_i/2) (cos th_i, sin th_i), rho_i in [0, rmax]."""
    nr = np.linalg.norm(X, axis=1)
    rho = 2 * np.arctanh(np.minimum(nr, 1 - 1e-12))
    th = np.arctan2(X[:, 1], X[:, 0])
    n = P.n
    hi = rmax if rmax is not None else 2.0 * np.arctanh(1.0 - PEPS64)

    def f(z):
        rh, t = z[:n], z[n:]
        a = np.tanh(rh / 2)
        cs, sn = np.cos(t), np.sin(t)
        Xc = np.stack([a * cs, a * sn], 1)
        L, G, s, p = poincare_lg(Xc, P, s_fixed)
        da = 0.5 * (1 - a * a)
        g_rho = (G[:, 0] * cs + G[:, 1] * sn) * da
        g_th = (-G[:, 0] * sn + G[:, 1] * cs) * a
        return L, np.concatenate([g_rho, g_th])

    z0 = np.concatenate([np.minimum(rho, hi), th])
    bounds = [(0.0, hi)] * n + [(None, None)] * n
    res = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-15, "gtol": 1e-11,
                            "maxcor": 30})
    rh, t = res.x[:n], res.x[n:]
    a = np.tanh(rh / 2)
    Xc = np.stack([a * np.cos(t), a * np.sin(t)], 1)
    L, G, s, p = poincare_lg(Xc, P, s_fixed)
    return {"loss": float(L), "mrd": mrd(p, P, s), "s": float(s), "nit": int(res.nit),
            "msg": str(res.message), "max_radius": float(rh.max()),
            "at_cap": int((rh >= hi - 1e-9).sum())}, Xc


def polish_euclid(P, X, maxiter=50000):
    n, dim = X.shape

    def f(z):
        L, G, s, p = euclid_lg(z.reshape(n, dim), P)
        return L, G.ravel()

    res = minimize(f, X.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-15, "gtol": 1e-11,
                            "maxcor": 30})
    L, G, s, p = euclid_lg(res.x.reshape(n, dim), P)
    return {"loss": float(L), "mrd": mrd(p, P, s), "s": float(s), "nit": int(res.nit),
            "msg": str(res.message)}, res.x.reshape(n, dim)


# ---------------------------------------------------------------- general-d bounded polish (protocol A)
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
        L, G, s, p = poincare_lg(Xc, P)
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
    L, G, s, p = poincare_lg(Xc, P)
    return {"loss": float(L), "mrd": mrd(p, P, s), "s": float(s), "nit": int(res.nit), "msg": str(res.message),
            "at_cap": int((rho >= rmax - 1e-9).sum()), "max_radius": float(rho.max()),
            "seconds": time.perf_counter() - t0}

