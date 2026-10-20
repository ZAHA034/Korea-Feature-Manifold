"""Synthetic Korea-like region maps for the 0S barrier/curvature check.

Coordinates in km, y = north.  Raster resolution 1 km.  Everything is generated from one integer seed.
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path, minimum_spanning_tree, breadth_first_order
from scipy.spatial import cKDTree

X0, X1, Y0, Y1 = 0.0, 440.0, -80.0, 470.0
RES = 1.0
NX, NY = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
XC = X0 + (np.arange(NX) + 0.5) * RES
YC = Y0 + (np.arange(NY) + 0.5) * RES
GX, GY = np.meshgrid(XC, YC)  # [iy, ix]


def y_north(x):
    return 425.0 + 0.10 * x


def x_east(y):
    return 285.0 + 0.05 * (450.0 - y)


def gnoise(rng, sigma_px, shape=(NY, NX)):
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma_px, mode="reflect")
    return f / f.std()


class Waves:
    """Sum of random plane waves; evaluable at arbitrary points; low-pass by wavelength cut."""

    def __init__(self, rng, n, lam_min, lam_max, expo, rms):
        lam = np.exp(rng.uniform(np.log(lam_min), np.log(lam_max), n))
        th = rng.uniform(0, 2 * np.pi, n)
        k = 2 * np.pi / lam
        self.kx, self.ky = k * np.cos(th), k * np.sin(th)
        self.phi = rng.uniform(0, 2 * np.pi, n)
        a = lam ** expo
        a *= rms / np.sqrt(np.sum(a ** 2) / 2)
        self.a, self.lam = a, lam

    def __call__(self, x, y, lam_cut=0.0, chunk=200_000):
        sel = self.lam >= lam_cut
        kx, ky, ph, a = self.kx[sel], self.ky[sel], self.phi[sel], self.a[sel]
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        out = np.empty_like(x)
        for s in range(0, len(x), chunk):
            xs, ys = x[s:s + chunk, None], y[s:s + chunk, None]
            out[s:s + chunk] = np.cos(xs * kx + ys * ky + ph) @ a
        return out


def catmull(points, step=0.5):
    """Dense polyline through control points (centripetal-ish uniform Catmull-Rom)."""
    P = np.asarray(points, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(seg)
    out.append(P[-2][None])
    Q = np.vstack(out)
    # resample to uniform arclength
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    su = np.arange(0, s[-1], step)
    return np.c_[np.interp(su, s, Q[:, 0]), np.interp(su, s, Q[:, 1])], su


class Ridge:
    def __init__(self, ctrl, H0, width, rng, n_pass=3, pass_depth=0.55, taper=(0.0, 25.0)):
        self.P, self.s = catmull(ctrl)
        self.tree = cKDTree(self.P)
        L = self.s[-1]
        self.L = L
        # height profile along arclength: H0 * (0.85 + 0.15 noise) with passes
        ss = np.linspace(0, L, 400)
        nz = ndimage.gaussian_filter1d(rng.standard_normal(400), 6)
        nz /= nz.std() + 1e-12
        prof = 0.85 + 0.15 * nz
        self.pass_s = np.sort(rng.uniform(0.12 * L, 0.88 * L, n_pass)) if n_pass else np.array([])
        for ps in self.pass_s:
            prof *= 1 - pass_depth * np.exp(-0.5 * ((ss - ps) / 5.0) ** 2)
        t0, t1 = taper
        if t0 > 0:
            prof *= np.clip(ss / t0, 0, 1)
        if t1 > 0:
            prof *= np.clip((L - ss) / t1, 0, 1)
        self.ss, self.prof = ss, prof
        self.H0, self.w = H0, width

    def height_at_s(self, s):
        return self.H0 * np.interp(s, self.ss, self.prof)

    def __call__(self, x, y):
        pts = np.c_[np.ravel(x), np.ravel(y)]
        d, i = self.tree.query(pts)
        return self.height_at_s(self.s[i]) * np.exp(-0.5 * (d / self.w) ** 2)


def seg_cross_poly(p, q, P):
    """Does segment p-q cross polyline P (n x 2)?  Vectorised over polyline pieces."""
    a, b = P[:-1], P[1:]
    d1 = q - p
    d2 = b - a
    den = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((a[:, 0] - p[0]) * d2[:, 1] - (a[:, 1] - p[1]) * d2[:, 0]) / den
        u = ((a[:, 0] - p[0]) * d1[1] - (a[:, 1] - p[1]) * d1[0]) / den
    ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return int(ok.sum())


def poisson_disc(rng, cand_xy, lam, target, iters=18):
    """Greedy variable-radius dart throwing; radius r = k / sqrt(lam); tune k to hit target count."""
    order = rng.permutation(len(cand_xy))
    cxy, cl = cand_xy[order], lam[order]

    def run(k):
        r = k / np.sqrt(cl)
        cell = 2.0
        grid = {}
        acc = []
        rmax = r.max()
        for i in range(len(cxy)):
            x, y = cxy[i]
            gx, gy = int(x // cell), int(y // cell)
            rr = r[i]
            span = int(np.ceil(rr / cell)) + 1
            ok = True
            for ix in range(gx - span, gx + span + 1):
                for iy in range(gy - span, gy + span + 1):
                    for j in grid.get((ix, iy), ()):
                        xj, yj, rj = acc[j]
                        if (x - xj) ** 2 + (y - yj) ** 2 < (0.5 * (rr + rj)) ** 2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault((gx, gy), []).append(len(acc))
                acc.append((x, y, rr))
        return np.array([(a[0], a[1]) for a in acc])

    lo, hi = 1.0, 200.0
    best = None
    for _ in range(iters):
        k = np.sqrt(lo * hi)
        pts = run(k)
        if best is None or abs(len(pts) - target) < abs(len(best) - target):
            best = pts
        if len(pts) > target:
            lo = k
        elif len(pts) < target:
            hi = k
        else:
            break
    if len(best) > target:  # drop random extras
        best = best[rng.choice(len(best), target, replace=False)]
    return best


def make_world(seed, n_regions=250, minor_ridges=3, pop_elev=400.0):
    rng = np.random.default_rng(seed)
    W = {"seed": seed}
    # ---------------- land mask
    x, y = GX, GY
    xw = 40 + 18 * np.sin(2 * np.pi * y / 250 + 1.0 + rng.uniform(-0.5, 0.5))
    ys = 45 + 10 * np.sin(2 * np.pi * x / 150 + rng.uniform(0, 6.28))
    base = (x > xw) & (x < x_east(y)) & (y > ys) & (y < y_north(x))
    sdf = ndimage.distance_transform_edt(base) - ndimage.distance_transform_edt(~base)
    amp = 2.5 + 11.5 * np.clip((x_east(y) - x - 25) / 40, 0, 1)
    nse = 0.6 * gnoise(rng, 6) + 0.4 * gnoise(rng, 2.5)
    f = sdf + 1.25 * amp * nse
    foreign = y >= y_north(x)
    land0 = (f > 0) & ~foreign
    lab, n = ndimage.label(land0)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main = lab == (1 + int(np.argmax(sizes)))
    main = ndimage.binary_fill_holes(main | foreign) & ~foreign
    lab, n = ndimage.label(main)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main = lab == (1 + int(np.argmax(sizes)))
    # islands
    isl = np.zeros_like(main)
    jeju_c = np.array([120 + rng.uniform(-15, 15), -35.0])
    isl_j = ((x - jeju_c[0]) / 35) ** 2 + ((y - jeju_c[1]) / 15) ** 2 < 1
    ull_c = np.array([x_east(330) + 130, 330 + rng.uniform(-20, 20)])
    isl_u = (x - ull_c[0]) ** 2 + (y - ull_c[1]) ** 2 < 5.5 ** 2
    row = int((330 - Y0) / RES)
    xs_main = np.flatnonzero(main[row])
    gw = XC[xs_main.min()] if len(xs_main) else 40.0
    near_c = np.array([gw - 15.0, 330.0])
    isl_n = ((x - near_c[0]) / 6) ** 2 + ((y - near_c[1]) / 9) ** 2 < 1
    # keep a sea gap of >= 3 km between near-shore island and mainland
    dist_main = ndimage.distance_transform_edt(~main)
    isl_n &= dist_main > 3
    isl = isl_j | isl_u | isl_n
    land = main | isl
    W.update(land=land, main=main, foreign=foreign, isl_j=isl_j, isl_u=isl_u, isl_n=isl_n)
    dist_sea = ndimage.distance_transform_edt(land | foreign) * RES  # km to nearest sea pixel
    # ---------------- DEM (analytic)
    hills = Waves(rng, 300, 6.0, 140.0, 0.5, 1.0)
    rough = Waves(rng, 400, 0.06, 6.0, 0.5, 1.0)
    W["hills"], W["rough"] = hills, rough
    # main ridge: Korea-like (parallel to east coast, then SW)
    def jit(v):
        return v + rng.uniform(-8, 8, 2)
    xe = lambda yy: x_east(yy)  # noqa: E731
    ctrl = [jit([xe(445) - 38, 452]), jit([xe(380) - 34, 380]), jit([xe(300) - 36, 300]),
            jit([xe(240) - 50, 235]), jit([200, 195]), jit([160, 160]), jit([125, 130])]
    ridge = Ridge(ctrl, H0=1000.0, width=7.0, rng=rng, n_pass=3, taper=(0.0, 30.0))
    minors = []
    for _ in range(minor_ridges):  # low ridges (watershed divides that are not real barriers)
        for _try in range(50):
            p0 = np.array([rng.uniform(60, 280), rng.uniform(70, 420)])
            ang = rng.uniform(0, np.pi)
            Lm = rng.uniform(60, 120)
            p1 = p0 + Lm * np.array([np.cos(ang), np.sin(ang)])
            mid = 0.5 * (p0 + p1) + rng.uniform(-15, 15, 2)
            pts = np.array([p0, mid, p1])
            ii = np.clip(((pts[:, 1] - Y0) / RES).astype(int), 0, NY - 1)
            jj = np.clip(((pts[:, 0] - X0) / RES).astype(int), 0, NX - 1)
            if main[ii, jj].all():
                break
        minors.append(Ridge(pts, H0=rng.uniform(250, 400), width=5.0, rng=rng, n_pass=1, taper=(15.0, 15.0)))
    W["ridge"], W["minors"] = ridge, minors
    bg_amp = lambda xx: 50 + 110 * np.clip(np.asarray(xx) / 300, 0, 1)  # noqa: E731

    def z_fn(px, py, lam_cut=0.0, dsea=None):
        px = np.asarray(px, float).ravel()
        py = np.asarray(py, float).ravel()
        zb = 110 + bg_amp(px) * hills(px, py, lam_cut)
        zr = 18 * rough(px, py, lam_cut) if lam_cut < 6.0 else 0.0
        zz = zb + zr + ridge(px, py)
        for m in minors:
            zz = zz + m(px, py)
        if dsea is None:
            # coastal ramp via raster distance to sea (nearest pixel)
            ii = np.clip(((py - Y0) / RES).astype(int), 0, NY - 1)
            jj = np.clip(((px - X0) / RES).astype(int), 0, NX - 1)
            dsea = dist_sea[ii, jj]
            onland = land[ii, jj]
        else:
            onland = np.ones_like(px, bool)
        ramp = np.clip(dsea / 12.0, 0, 1)
        zz = np.maximum(zz * (0.15 + 0.85 * ramp), 1.0)
        return np.where(onland, zz, 0.0)

    W["z_fn"] = z_fn
    # raster DEM at 1 km (modes with wavelength >= 2 km)
    zr_ = np.zeros((NY, NX))
    zr_[land] = z_fn(GX[land], GY[land], lam_cut=2.0)
    W["zr"] = zr_
    # ---------------- population intensity and seeds
    metros = [((110, 360), 18, 60), ((285, 80), 12, 35), ((240, 150), 10, 20), ((90, 110), 8, 15), ((150, 230), 8, 15)]
    lam = np.ones((NY, NX))
    for (cx, cy), s, A in metros:
        cx, cy = cx + rng.uniform(-12, 12), cy + rng.uniform(-12, 12)
        lam += A * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / s ** 2)
    if pop_elev:
        lam *= np.exp(-zr_ / pop_elev)
    lam[~main] = 0
    W["lam"] = lam
    idx = np.flatnonzero(main.ravel())
    p = lam.ravel()[idx]
    p = p / p.sum()
    pick = rng.choice(idx, size=30000, replace=True, p=p)
    cxy = np.c_[GX.ravel()[pick], GY.ravel()[pick]] + rng.uniform(-0.5, 0.5, (30000, 2)) * RES
    cl = lam.ravel()[pick]
    n_isl = 4
    mains = poisson_disc(rng, cxy, cl, n_regions - n_isl)
    isl_pts = np.array([jeju_c + [-15, 0], jeju_c + [15, 0], ull_c, near_c])
    # snap island seeds onto island pixels
    for k, m in enumerate([isl_j, isl_j, isl_u, isl_n]):
        ii = int((isl_pts[k, 1] - Y0) / RES)
        jj = int((isl_pts[k, 0] - X0) / RES)
        if not m[ii, jj]:
            cand = np.argwhere(m)
            d = (cand[:, 0] - ii) ** 2 + (cand[:, 1] - jj) ** 2
            ii, jj = cand[np.argmin(d)]
            isl_pts[k] = [XC[jj], YC[ii]]
    pts = np.vstack([mains, isl_pts])
    n = len(pts)
    kind = np.array(["main"] * (n - 4) + ["jeju", "jeju", "ulleung", "near"])
    W.update(pts=pts, kind=kind, n=n)
    # ---------------- geodesic Voronoi on land raster (8-neighbour)
    pid = -np.ones((NY, NX), int)
    lp = np.argwhere(land)
    pid[lp[:, 0], lp[:, 1]] = np.arange(len(lp))
    rows, cols, wts = _neigh_edges(pid)
    G8 = csr_matrix((wts, (rows, cols)), shape=(len(lp), len(lp)))
    seed_pix = pid[((pts[:, 1] - Y0) / RES).astype(int), ((pts[:, 0] - X0) / RES).astype(int)]
    assert (seed_pix >= 0).all()
    # unique pixels per seed
    assert len(np.unique(seed_pix)) == n, "two seeds share a pixel"
    dist, pred, src = dijkstra(G8, directed=False, indices=seed_pix, min_only=True, return_predecessors=True)
    lab = -np.ones((NY, NX), int)
    seed_of_pix = -np.ones(len(lp), int)
    m_ = src >= 0
    inv = {int(sp): k for k, sp in enumerate(seed_pix)}
    seed_of_pix[m_] = np.array([inv[int(s)] for s in src[m_]])
    lab[lp[:, 0], lp[:, 1]] = seed_of_pix
    W.update(lab=lab, pid=pid, lp=lp, G8=G8, seed_pix=seed_pix)
    # ---------------- adjacency by shared boundary (4-neighbour contacts)
    cnt = {}
    for A_, B_ in [(lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])]:
        m2 = (A_ >= 0) & (B_ >= 0) & (A_ != B_)
        a, b = A_[m2], B_[m2]
        lo_, hi_ = np.minimum(a, b), np.maximum(a, b)
        key = lo_ * n + hi_
        u, c = np.unique(key, return_counts=True)
        for kk, cc in zip(u, c):
            cnt[int(kk)] = cnt.get(int(kk), 0) + int(cc)
    adj = sorted((k // n, k % n) for k in cnt)
    W["adj_contact"] = {e: cnt[e[0] * n + e[1]] for e in adj}
    # coastal / border flags per cell
    sea = ~land & ~foreign
    coast = np.zeros(n, bool)
    border = np.zeros(n, bool)
    for nb in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shs = np.roll(np.roll(sea, nb[0], 0), nb[1], 1)
        shf = np.roll(np.roll(foreign, nb[0], 0), nb[1], 1)
        # lab at p, neighbour at p - nb
        c_ = lab[shs & (lab >= 0)]
        coast[np.unique(c_)] = True
        b_ = lab[shf & (lab >= 0)]
        border[np.unique(b_)] = True
    W.update(coast=coast, border=border)
    # population per cell
    lamc = np.where(land, lam, 0)
    P = np.bincount(lab[land], weights=lamc[land], minlength=n)
    # islands get small populations (Jeju ~ 1.3% , Ulleung tiny)
    tot = P.sum()
    P[kind == "jeju"] = 0.006 * tot
    P[kind == "ulleung"] = 0.0002 * tot
    P[kind == "near"] = 0.0013 * tot
    W["P"] = P / P.sum() * 5.0e7
    # ---------------- island links (ferry / bridge), mainland coastal seeds only
    mainc = np.flatnonzero((kind == "main") & coast)
    links = []
    def nearest_main(pt, k):
        d = np.linalg.norm(pts[mainc] - pt, axis=1)
        return mainc[np.argsort(d)[:k]], np.sort(d)[:k]
    jj = np.flatnonzero(kind == "jeju")
    # link the northern-most Jeju cell (both identical y; take the one closer to mainland) to 2 ports
    jn = jj[np.argmin([np.min(np.linalg.norm(pts[mainc] - pts[j], axis=1)) for j in jj])]
    for t, d in zip(*nearest_main(pts[jn], 2)):
        links.append((int(jn), int(t), "ferry", d / 0.6 + 60.0))
    uu = int(np.flatnonzero(kind == "ulleung")[0])
    for t, d in zip(*nearest_main(pts[uu], 2)):
        links.append((uu, int(t), "ferry", d / 0.6 + 60.0))
    nn = int(np.flatnonzero(kind == "near")[0])
    for t, d in zip(*nearest_main(pts[nn], 1)):
        links.append((nn, int(t), "bridge", 1.3 * d / 1.0))
    W["links"] = links
    # ---------------- edge list
    E = [(u, v) for (u, v) in adj]
    etype = ["adj"] * len(E)
    for (u, v, md, t) in links:
        e = (min(u, v), max(u, v))
        if e not in W["adj_contact"]:
            E.append(e)
            etype.append("island")
    W["E"] = np.array(E)
    W["etype"] = np.array(etype)
    W["link_time"] = {(min(u, v), max(u, v)): t for (u, v, md, t) in links}
    deg = np.bincount(W["E"].ravel(), minlength=n)
    W["deg"] = deg
    # per-edge static detour noise for model S
    W["rho"] = 1.3 * np.exp(0.12 * rng.standard_normal(len(E)))
    # smooth detour field for model R
    W["rho_field"] = 1.25 * np.exp(0.10 * gnoise(rng, 10))
    return W


def _pairs(pid, di, dj):
    """Arrays (a, b) of pixel ids for all pixel pairs p, p + (di, dj); di >= 0."""
    ny, nx = pid.shape
    if dj >= 0:
        a = pid[0:ny - di, 0:nx - dj]
        b = pid[di:ny, dj:nx]
    else:
        a = pid[0:ny - di, -dj:nx]
        b = pid[di:ny, 0:nx + dj]
    return a, b


def _neigh_edges(pid):
    rows, cols, wts = [], [], []
    for di, dj, w in [(0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))]:
        a, b = _pairs(pid, di, dj)
        m = (a >= 0) & (b >= 0)
        rows.append(a[m])
        cols.append(b[m])
        wts.append(np.full(m.sum(), w * RES))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
