"""ORC, travel-time models, barrier indices and decision rules for the 0S barrier check."""
import numpy as np
import ot
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path, minimum_spanning_tree, breadth_first_order

import kw

V_KMPM = 1.0  # 60 km/h in km/min


# ----------------------------------------------------------------------------- ORC
def closure(n, E, length):
    A = csr_matrix((np.r_[length, length], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    return shortest_path(A, directed=False, method="D")


def orc(n, E, D, w=None, alpha=0.5):
    """kappa(x,y) = 1 - W1(m_x, m_y)/D[x,y];  m_x = alpha*delta_x + (1-alpha)*w-weighted neighbours."""
    nb = [[] for _ in range(n)]
    wt = [[] for _ in range(n)]
    for k, (u, v) in enumerate(E):
        ww = 1.0 if w is None else w[k]
        nb[u].append(v)
        wt[u].append(ww)
        nb[v].append(u)
        wt[v].append(ww)
    sup, mass = [], []
    for x in range(n):
        ws = np.asarray(wt[x], float)
        sup.append(np.r_[x, nb[x]].astype(int))
        mass.append(np.r_[alpha, (1 - alpha) * ws / ws.sum()])
    kap = np.empty(len(E))
    w1s = np.empty(len(E))
    for k, (x, y) in enumerate(E):
        C = D[np.ix_(sup[x], sup[y])]
        w1 = ot.emd2(mass[x], mass[y], C)
        w1s[k] = w1
        kap[k] = 1.0 - w1 / D[x, y]
    return kap, w1s


def fixture_check():
    import itertools
    out = {}
    # barbell
    E = [(a, b) for a, b in itertools.combinations(range(5), 2)] + [(a + 5, b + 5) for a, b in itertools.combinations(range(5), 2)] + [(0, 5)]
    E = np.array(E)
    D = closure(10, E, np.ones(len(E)))
    k, _ = orc(10, E, D)
    out["barbell_bridge"] = float(k[-1])
    out["barbell_min_unique"] = bool(np.sum(np.isclose(k, k.min())) == 1)
    E = np.array([(i, (i + 1) % 5) for i in range(5)])
    k, _ = orc(5, E, closure(5, E, np.ones(5)))
    out["C5"] = float(k[0])
    E = np.array([(i, (i + 1) % 8) for i in range(8)])
    k, _ = orc(8, E, closure(8, E, np.ones(8)))
    out["C8"] = float(np.abs(k).max())
    E = np.array(list(itertools.combinations(range(5), 2)))
    k, _ = orc(5, E, closure(5, E, np.ones(len(E))))
    out["K5"] = float(k[0])
    # long bridge: kappa -> 0.2 - 0.8/L
    for L in (1, 2, 4, 8, 32):
        E = np.array([(a, b) for a, b in itertools.combinations(range(5), 2)] + [(a + 5, b + 5) for a, b in itertools.combinations(range(5), 2)] + [(0, 5)])
        lens = np.ones(len(E))
        lens[-1] = L
        k, _ = orc(10, E, closure(10, E, lens))
        out[f"barbell_bridge_L{L}"] = float(k[-1])
    return out


# ----------------------------------------------------------------------------- static edge features
def seg_profile(W, p, q, ds, lam_cut):
    L = float(np.linalg.norm(q - p))
    m = max(2, int(np.ceil(L / ds)) + 1)
    t = np.linspace(0, 1, m)
    xs = p[0] + t * (q[0] - p[0])
    ys = p[1] + t * (q[1] - p[1])
    return W["z_fn"](xs, ys, lam_cut=lam_cut), L


def barrier_indices(W, ds=0.1, lam_cut=0.06, edges=None):
    """Per-edge (excess relief, mean cumulative ascent both directions) along the straight origin-origin segment."""
    E = W["E"] if edges is None else edges
    pts = W["pts"]
    er = np.full(len(E), np.nan)
    ca = np.full(len(E), np.nan)
    for k, (u, v) in enumerate(E):
        z, L = seg_profile(W, pts[u], pts[v], ds, lam_cut)
        er[k] = z.max() - max(z[0], z[-1])
        ca[k] = 0.5 * np.abs(np.diff(z)).sum()
    return er, ca


def pass_height(W, u, v):
    """Minimax elevation of a path between the two origin pixels inside the union of the two cells (1 km raster)."""
    lab, zr = W["lab"], W["zr"]
    m = (lab == u) | (lab == v)
    ii, jj = np.nonzero(m)
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    sub = m[i0:i1, j0:j1]
    zs = zr[i0:i1, j0:j1]
    pid = -np.ones(sub.shape, int)
    q = np.argwhere(sub)
    pid[q[:, 0], q[:, 1]] = np.arange(len(q))
    rows, cols, ws = [], [], []
    for di, dj in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a, b = kw._pairs(pid, di, dj)
        za, zb = kw._pairs(np.where(sub, zs, 0.0), di, dj)
        mm = (a >= 0) & (b >= 0)
        rows.append(a[mm])
        cols.append(b[mm])
        ws.append(np.maximum(za[mm], zb[mm]) + 1.0)  # +1 keeps weights > 0
    A = csr_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(len(q), len(q)))
    T = minimum_spanning_tree(A)
    T = T + T.T
    pu = W["pts"][u]
    pv = W["pts"][v]
    su = pid[int((pu[1] - kw.Y0) / kw.RES) - i0, int((pu[0] - kw.X0) / kw.RES) - j0]
    sv = pid[int((pv[1] - kw.Y0) / kw.RES) - i0, int((pv[0] - kw.X0) / kw.RES) - j0]
    if su < 0 or sv < 0:
        return np.nan
    order, pred = breadth_first_order(T, su, directed=False, return_predecessors=True)
    if pred[sv] < 0 and sv != su:
        return np.nan  # union not connected (should not happen for adjacent cells)
    Tc = T.tocsr()
    h = max(zs[q[su, 0], q[su, 1]], zs[q[sv, 0], q[sv, 1]])
    x = sv
    hmax = 0.0
    while x != su:
        p = pred[x]
        hmax = max(hmax, Tc[p, x] - 1.0)
        x = p
    return max(hmax, h) - h


def static_features(W):
    E, et, pts, n = W["E"], W["etype"], W["pts"], W["n"]
    adj = et == "adj"
    F = {"adj": adj}
    deg = W["deg"]
    F["degterm"] = 1.0 / deg[E[:, 0]] + 1.0 / deg[E[:, 1]]
    coast, border = W["coast"], W["border"]
    F["coast_any"] = (coast[E[:, 0]] | coast[E[:, 1]]).astype(float)
    F["border_any"] = (border[E[:, 0]] | border[E[:, 1]]).astype(float)
    F["bnd_any"] = np.maximum(F["coast_any"], F["border_any"])
    F["bnd_cnt"] = (coast | border)[E[:, 0]].astype(float) + (coast | border)[E[:, 1]].astype(float)
    F["len_e"] = np.linalg.norm(pts[E[:, 0]] - pts[E[:, 1]], axis=1)
    F["contact"] = np.array([W["adj_contact"].get((int(u), int(v)), 0) for u, v in E], float)
    F["island"] = (~adj).astype(float)
    F["flag"] = np.array([kw.seg_cross_poly(pts[u], pts[v], W["ridge"].P) > 0 for u, v in E]) & adj
    F["flag_all"] = F["flag"] | (np.array([any(kw.seg_cross_poly(pts[u], pts[v], m.P) > 0 for m in W["minors"]) for u, v in E]) & adj)
    # observed indices (fine DEM ~ 30-60 m, 100 m sampling)
    er, ca = barrier_indices(W, ds=0.1, lam_cut=0.06)
    F["er"], F["ca"] = np.where(adj, er, np.nan), np.where(adj, ca, np.nan)
    F["ms"] = F["ca"] / F["len_e"]
    F["ph"] = np.array([pass_height(W, u, v) if a else np.nan for (u, v), a in zip(E, adj)])
    # "true" climbing driver for model S: smooth DEM (>= 2 km wavelengths), 0.5 km sampling
    er_t, ca_t = barrier_indices(W, ds=0.5, lam_cut=2.0)
    F["ca_true"] = np.where(adj, ca_t, 0.0)
    # distance of edge midpoint to main crest; endpoint exposure
    mid = 0.5 * (pts[E[:, 0]] + pts[E[:, 1]])
    F["dist_crest"] = W["ridge"].tree.query(mid)[0]
    return F


# ----------------------------------------------------------------------------- travel time models
def lengths_S(W, F, c):
    """Segment model: time = L*rho/v + c * mean ascent (both directions) on the smooth DEM."""
    E = W["E"]
    t = np.empty(len(E))
    for k, (u, v) in enumerate(E):
        if F["adj"][k]:
            t[k] = F["len_e"][k] * W["rho"][k] / V_KMPM + c * F["ca_true"][k]
        else:
            t[k] = W["link_time"][(int(u), int(v))]
    return t


def raster_graph(W, c):
    lp, pid, zr = W["lp"], W["pid"], W["zr"]
    rho = W["rho_field"]
    rows, cols, ws = [], [], []
    for di, dj, s in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))]:
        a, b = kw._pairs(pid, di, dj)
        za, zb = kw._pairs(zr, di, dj)
        ra, rb = kw._pairs(rho, di, dj)
        m = (a >= 0) & (b >= 0)
        base = s * kw.RES * 0.5 * (ra[m] + rb[m]) / V_KMPM
        up = np.maximum(zb[m] - za[m], 0.0)
        dn = np.maximum(za[m] - zb[m], 0.0)
        rows += [a[m], b[m]]
        cols += [b[m], a[m]]
        ws += [base + c * up, base + c * dn]
    N = len(lp)
    return csr_matrix((np.concatenate(ws), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))


def lengths_R(W, F, c):
    """Raster cost-distance model: anisotropic least-cost travel time on land; d_sym of adjacent pairs."""
    G = raster_graph(W, c)
    sp = W["seed_pix"]
    n = W["n"]
    Dd = np.full((n, n), np.inf)
    for s in range(0, n, 25):
        d = dijkstra(G, directed=True, indices=sp[s:s + 25])
        Dd[s:s + 25] = d[:, sp]
    Ds = 0.5 * (Dd + Dd.T)
    E = W["E"]
    t = np.empty(len(E))
    for k, (u, v) in enumerate(E):
        t[k] = Ds[u, v] if F["adj"][k] else W["link_time"][(int(u), int(v))]
    if not np.isfinite(t).all():
        raise RuntimeError("non-finite adjacency length in model R")
    return t


def gravity_w(W, E, D, tau=30.0):
    P = W["P"]
    return P[E[:, 0]] * P[E[:, 1]] * np.exp(-D[E[:, 0], E[:, 1]] / tau)


# ----------------------------------------------------------------------------- decision rules
def hyper_p(top_mask, flag):
    N = len(flag)
    K = int(flag.sum())
    n = int(top_mask.sum())
    x = int((top_mask & flag).sum())
    return float(stats.hypergeom.sf(x - 1, N, K, n)), x, K, N


def ols_resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta, beta


def rules(F, kap, top=20, covs=("degterm", "bnd_any"), idx_list=("er", "ca", "ph", "ms", "flag", "flag_all")):
    pop = F["adj"]
    k = kap[pop]
    flag = F["flag"][pop]
    flag_all = F["flag_all"][pop]
    o = np.argsort(k, kind="stable")
    topm = np.zeros(len(k), bool)
    topm[o[:top]] = True
    res = {}
    for nm, fl in (("flag", flag), ("flag_all", flag_all)):
        p, x, K, N = hyper_p(topm, fl)
        res[f"A_{nm}"] = x >= 5
        res[f"B_{nm}"] = p < 0.05
        res[f"x_{nm}"] = x
        res[f"K_{nm}"] = K
        res[f"pB_{nm}"] = p
    res["N"] = int(pop.sum())
    X = np.column_stack([F[c][pop] for c in covs])
    r, beta = ols_resid(k, X)
    res["beta"] = beta.tolist()
    o2 = np.argsort(r, kind="stable")
    topr = np.zeros(len(k), bool)
    topr[o2[:top]] = True
    res["overlap"] = int((topm & topr).sum())
    for nm, fl in (("flag", flag), ("flag_all", flag_all)):
        p, x, K, N = hyper_p(topr, fl)
        res[f"E_{nm}"] = p < 0.05
        res[f"xE_{nm}"] = x
    for nm in idx_list:
        z = F[nm][pop].astype(float)
        ok = np.isfinite(z)
        rho_c, p_c = stats.spearmanr(k[ok], z[ok], alternative="less")
        rho_d, p_d = stats.spearmanr(r[ok], z[ok], alternative="less")
        res[f"rhoC_{nm}"], res[f"pC_{nm}"] = float(rho_c), float(p_c)
        res[f"rhoD_{nm}"], res[f"pD_{nm}"] = float(rho_d), float(p_d)
    return res
