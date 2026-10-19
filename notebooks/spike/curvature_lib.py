"""KFM 0S-3 곡률 판정 계산 코드 (ADR 0013 1·2·5·6항, PLAN 6.1a).

0S 탐색 코드라서 kfm 패키지·runs·스키마 규칙 밖이다(PLAN 6.1a). 실행 환경은 C:\\dev\\kfm-spike.
순수 함수만 둔다(입력을 제자리에서 바꾸지 않는다). 노트북 02_curvature.ipynb가 이 모듈을 불러 쓴다.

구성
  1. 그래프 기본, ORC(POT ot.emd2, 게으른 측도 alpha), 합성 fixture
  2. kappa_geo (경계 공유 엣지만 직선 거리 x median(d_sym / 직선 거리)로 바꾼 길이)
  3. 통계: 단측 Spearman, 우회율 편 Spearman, 시도 jackknife 최대 p, 유효성 점검
  4. PLAN 6.1a 5분류, 옆 엣지, 초기하, ADR 0013 5항 판정 함수
  5. 입력 준비 보조: GPX 연결, DEM 프로파일, 연륙·연도교 후보 탐지
  6. 한 번에 도는 분석(analyze_curvature)과 지도, 합성 장난감 나라(파이프라인 배선 점검용)
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

try:  # POT는 kfm-spike 환경에만 있다
    import ot
except ImportError:  # pragma: no cover
    ot = None

W_FLOOR = 1e-6  # 0 가중치 하한(공통 규칙: scipy csr는 0을 엣지로 보지 않는다)
ALPHA_DEFAULT = 0.5
P_THRESHOLD = 0.05  # ADR 0013 1항: 단순 p와 jackknife p가 모두 0.05 미만이면 통과
VALIDITY_THRESHOLD = 0.1  # ADR 0013 2항: |rho| < 0.1 (효과 크기 기준)
DATA_PROBLEM_MAJORITY = 11  # ADR 0013 5항: 상위 20개 중 11개 이상이면 과반
CLASSES = ["데이터·설정 문제", "자연 장벽", "시도 경계", "망 단절", "기타"]  # PLAN 6.1a 판정 순서


# ===================================================================== 1. 그래프, ORC
def _edges(u, v):
    return np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)


def edge_key(u, v, n):
    """무방향 엣지의 정수 키(작은 끝 * n + 큰 끝)."""
    u, v = _edges(u, v)
    return np.minimum(u, v) * n + np.maximum(u, v)


def validate_graph(n, u, v, length):
    """자기 루프, 중복 엣지, 비양수 길이, 고립 노드를 막는다. (u, v, length, deg)를 돌려준다."""
    u, v = _edges(u, v)
    length = np.asarray(length, dtype=float)
    if not (len(u) == len(v) == len(length)):
        raise ValueError("u, v, length 길이가 다르다")
    if (u == v).any():
        raise ValueError("자기 루프가 있다")
    if ((u < 0) | (v < 0) | (u >= n) | (v >= n)).any():
        raise ValueError("노드 번호가 범위 밖이다")
    if len(np.unique(edge_key(u, v, n))) != len(u):
        raise ValueError("중복 엣지가 있다")
    if not np.isfinite(length).all() or (length <= 0).any():
        raise ValueError("엣지 길이는 유한한 양수여야 한다")
    deg = np.bincount(np.r_[u, v], minlength=n)
    if (deg == 0).any():
        raise ValueError(f"고립 노드가 있다: {np.flatnonzero(deg == 0).tolist()}")
    return u, v, length, deg


def shortest_paths(n, u, v, length):
    """엣지 길이 위 최단경로 거리 행렬(무방향). 도달 불가는 inf."""
    u, v = _edges(u, v)
    w = np.maximum(np.asarray(length, dtype=float), W_FLOOR)
    a = csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    return dijkstra(a, directed=True)


def n_components(n, u, v):
    u, v = _edges(u, v)
    a = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    return connected_components(a, directed=False)[0]


def orc(n, u, v, length, alpha=ALPHA_DEFAULT, weight=None, dist=None, return_w1=False):
    """엣지별 Ollivier-Ricci 곡률 kappa(x, y) = 1 - W1(mu_x, mu_y) / d(x, y).

    - mu_x: x에 alpha, 이웃 z에 (1 - alpha) * w_xz / sum_z w_xz (weight=None이면 균등 1/deg).
    - W1의 바닥 거리와 분모 d(x, y)는 엣지 길이 위 최단경로 거리다(엣지 길이 자체가 아니다).
    - W1은 POT ot.emd2(정확한 선형계획 해)로 구한다.
    weight는 엣지별 무방향 가중치(u, v 순서와 같은 배열)다.
    """
    u, v, length, _ = validate_graph(n, u, v, length)
    if dist is None:
        dist = shortest_paths(n, u, v, length)
    if not np.isfinite(dist).all():
        raise ValueError("그래프가 연결되어 있지 않다")
    m = len(u)
    w = np.ones(m) if weight is None else np.asarray(weight, dtype=float)
    if w.shape != (m,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("이웃 가중치는 엣지 수와 같은 길이의 양수 배열이어야 한다")
    nbr = [[] for _ in range(n)]
    for e in range(m):
        nbr[u[e]].append((int(v[e]), float(w[e])))
        nbr[v[e]].append((int(u[e]), float(w[e])))
    tot = [sum(wz for _, wz in lst) for lst in nbr]

    def measure(x):
        d = {x: alpha}
        for z, wz in nbr[x]:
            d[z] = (1.0 - alpha) * wz / tot[x]
        return d

    kappa = np.empty(m)
    w1 = np.empty(m)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UserWarning, module=r"ot(\..*)?")  # POT 경고(반복 한도 등)는 오류로 취급
        for e in range(m):
            x, y = int(u[e]), int(v[e])
            mx, my = measure(x), measure(y)
            supp = sorted(set(mx) | set(my))
            a = np.array([mx.get(s, 0.0) for s in supp])
            b = np.array([my.get(s, 0.0) for s in supp])
            cost = dist[np.ix_(supp, supp)]
            w1[e] = ot.emd2(a, b, cost)
            kappa[e] = 1.0 - w1[e] / dist[x, y]
    return (kappa, w1) if return_w1 else kappa


def w1_lp(a, b, cost):
    """검산용: 수송 선형계획(scipy HiGHS)으로 구한 W1. ot.emd2와 독립이다."""
    from scipy.optimize import linprog

    a, b, cost = np.asarray(a, float), np.asarray(b, float), np.asarray(cost, float)
    na, nb = len(a), len(b)
    a_eq = np.zeros((na + nb, na * nb))
    for i in range(na):
        a_eq[i, i * nb:(i + 1) * nb] = 1.0
    for j in range(nb):
        a_eq[na + j, j::nb] = 1.0
    res = linprog(cost.ravel(), A_eq=a_eq, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun)


def orc_lp(n, u, v, length, alpha=ALPHA_DEFAULT):
    """검산용 ORC: orc()와 같은 정의, W1만 선형계획으로 푼다(균등 이웃 분포)."""
    u, v, length, _ = validate_graph(n, u, v, length)
    dist = shortest_paths(n, u, v, length)
    nbr = [[] for _ in range(n)]
    for e in range(len(u)):
        nbr[u[e]].append(int(v[e]))
        nbr[v[e]].append(int(u[e]))
    out = np.empty(len(u))
    for e in range(len(u)):
        x, y = int(u[e]), int(v[e])
        supp = sorted({x, y, *nbr[x], *nbr[y]})
        a = np.array([alpha if s == x else ((1 - alpha) / len(nbr[x]) if s in nbr[x] else 0.0) for s in supp])
        b = np.array([alpha if s == y else ((1 - alpha) / len(nbr[y]) if s in nbr[y] else 0.0) for s in supp])
        out[e] = 1.0 - w1_lp(a, b, dist[np.ix_(supp, supp)]) / dist[x, y]
    return out


# --------------------------------------------------------------------- 합성 fixture (PLAN 7.1)
def graph_complete(k):
    u, v = np.triu_indices(k, 1)
    return k, u, v, np.ones(len(u))


def graph_cycle(k):
    u = np.arange(k)
    return k, u, (u + 1) % k, np.ones(k)


def graph_barbell(bridge_length=1.0):
    """K5 두 개(a0..a4 = 0..4, b0..b4 = 5..9) + 다리 엣지 a0-b0(마지막 엣지). 클릭 엣지 길이 1."""
    u, v = [], []
    for base in (0, 5):
        for i in range(5):
            for j in range(i + 1, 5):
                u.append(base + i)
                v.append(base + j)
    u.append(0)
    v.append(5)
    length = np.ones(len(u))
    length[-1] = bridge_length
    return 10, np.array(u), np.array(v), length


# ===================================================================== 2. kappa_geo
def geo_lengths(d_sym, straight, is_boundary):
    """경계 공유 엣지는 직선 거리 x median(d_sym / 직선 거리), 나머지(연륙·연도교, 섬 링크)는 d_sym.

    median은 경계 공유 엣지 전체에서 구한다. (길이 배열, median)을 돌려준다.
    """
    d_sym, straight = np.asarray(d_sym, float), np.asarray(straight, float)
    is_boundary = np.asarray(is_boundary, bool)
    med = float(np.median(d_sym[is_boundary] / straight[is_boundary]))
    return np.where(is_boundary, straight * med, d_sym), med


def curvature_set(n, u, v, d_sym, straight, is_boundary, alpha=ALPHA_DEFAULT, weight=None):
    """kappa(d_sym 길이), kappa_geo, kappa_ex = kappa - kappa_geo와 부속 거리를 한 번에 낸다."""
    len_geo, med = geo_lengths(d_sym, straight, is_boundary)
    dist_d = shortest_paths(n, u, v, d_sym)
    dist_g = shortest_paths(n, u, v, len_geo)
    kappa = orc(n, u, v, d_sym, alpha, weight, dist_d)
    kappa_geo = orc(n, u, v, len_geo, alpha, weight, dist_g)
    uu, vv = _edges(u, v)
    return {"kappa": kappa, "kappa_geo": kappa_geo, "kappa_ex": kappa - kappa_geo, "median_ratio": med, "len_geo": len_geo,
            "d_graph_dsym": dist_d[uu, vv], "d_graph_geo": dist_g[uu, vv]}


def shorter_detour(length, d_graph, rel_tol=1e-9):
    """최단경로가 직접 엣지보다 짧은 엣지(우회가 더 짧음) 지시자."""
    length, d_graph = np.asarray(length, float), np.asarray(d_graph, float)
    return d_graph < length * (1.0 - rel_tol)


# ===================================================================== 3. 통계
def spearman_greater(x, y):
    """단측 Spearman(H1: rho > 0). (rho, p)."""
    r = stats.spearmanr(x, y, alternative="greater")
    return float(r.statistic), float(r.pvalue)


def spearman_two_sided(x, y):
    return float(stats.spearmanr(x, y).statistic)


def _ols_resid(a, b):
    design = np.column_stack([np.ones(len(b)), b])
    beta = np.linalg.lstsq(design, a, rcond=None)[0]
    return a - design @ beta


def partial_spearman_greater(x, y, z):
    """z(우회율 r)를 통제한 편 Spearman, 단측. (rho, p).

    세 변수의 순위에서 x와 y를 각각 z에 OLS 회귀한 잔차의 Pearson 상관이 rho이고,
    t = rho * sqrt((n - 3) / (1 - rho^2)), 자유도 n - 3의 t 분포로 단측 p를 구한다(ADR 0013 1항).
    """
    rx, ry, rz = (stats.rankdata(np.asarray(a, float)) for a in (x, y, z))
    n = len(rx)
    ex, ey = _ols_resid(rx, rz), _ols_resid(ry, rz)
    if np.allclose(ex, ex[0]) or np.allclose(ey, ey[0]):
        return float("nan"), float("nan")
    rho = float(np.corrcoef(ex, ey)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0 if rho > 0 else 1.0
    t = rho * math.sqrt((n - 3) / (1.0 - rho * rho))
    return rho, float(stats.t.sf(t, n - 3))


def jackknife_max_p(test, su, sv, sido_ids, *arrays):
    """시도 jackknife: 시도마다 끝점 하나라도 그 시도인 엣지를 빼고 다시 구한 단측 p의 최댓값.

    test(*arrays) -> (rho, p). 계산되지 않은 p(NaN)는 보수적으로 1로 본다.
    (최댓값, 최댓값을 만든 시도, {시도: p}, {시도: 남은 엣지 수})를 돌려준다.
    """
    su, sv = np.asarray(su), np.asarray(sv)
    ps, ns = {}, {}
    for s in sido_ids:
        keep = (su != s) & (sv != s)
        _, p = test(*[np.asarray(a)[keep] for a in arrays])
        ps[s] = 1.0 if not np.isfinite(p) else float(p)
        ns[s] = int(keep.sum())
    worst = max(ps, key=lambda k: ps[k])
    return ps[worst], worst, ps, ns


def run_battery(kex, x, r, su, sv, sido_ids, alpha=P_THRESHOLD):
    """ADR 0013 1항 두 검정. x는 지형 지표(경사 s, h_e, 산맥 선 교차 지시변수 b_e 등).

    main: 단측 Spearman rho(kex, x), partial: 우회율 r을 통제한 편 Spearman. 각각 단순 p와 jackknife p,
    둘 다 alpha 미만이면 passed다.
    """
    kex, x, r = (np.asarray(a, float) for a in (kex, x, r))
    out = {}
    for name, fn, arrays in (("main", spearman_greater, (kex, x)), ("partial", partial_spearman_greater, (kex, x, r))):
        rho, p = fn(*arrays)
        pj, worst, _, _ = jackknife_max_p(fn, su, sv, sido_ids, *arrays)
        passed = bool(np.isfinite(p) and p < alpha and pj < alpha)
        out[name] = {"rho": rho, "p": p, "p_jack": pj, "worst_sido": worst, "n": int(len(kex)), "passed": passed}
    return out


def validity_checks(kex, inv_deg_sum, coastal, log_straight, threshold=VALIDITY_THRESHOLD):
    """ADR 0013 2항: kex와 (1/deg_u + 1/deg_v), 해안·접경 지시변수, log 직선 거리의 Spearman |rho|가 모두 0.1 미만."""
    names = ("1/deg_u+1/deg_v", "해안·접경 지시변수", "log 직선 거리")
    rho = {nm: spearman_two_sided(kex, var) for nm, var in zip(names, (inv_deg_sum, coastal, log_straight))}
    ok = bool(all(np.isfinite(r) and abs(r) < threshold for r in rho.values()))
    return {"rho": rho, "ok": ok, "threshold": threshold}


# ===================================================================== 4. 분류, 판정
def top_indices(values, k=20):
    """값이 큰 순서 k개의 위치(동점이면 앞선 엣지 먼저)."""
    return np.argsort(-np.asarray(values, float), kind="stable")[:k]


def island_sensitive(kex_base, kex_scenarios, top_idx, k=20, tol=1e-9):
    """상위 k개 중 섬 링크 시간 x0.5나 x2에서 자기 kappa_ex가 기준 설정의 k위 값보다 작아지는 엣지 지시자(전체 길이 배열).

    k위 엣지 자신이 부동소수점 잡음(tol 이내)으로 '작아진' 것으로 잡히지 않게 tol만큼은 봐준다.
    """
    kex_base = np.asarray(kex_base, float)
    thr = np.sort(kex_base)[::-1][k - 1]
    flag = np.zeros(len(kex_base), bool)
    for scen in kex_scenarios:
        flag[top_idx] |= np.asarray(scen, float)[top_idx] < thr - tol
    return flag, float(thr)


def classify_edges(data_problem, natural_barrier, cross_sido, r_top10):
    """PLAN 6.1a 5분류. 데이터·설정 문제 -> 자연 장벽 -> 시도 경계 -> 망 단절 -> 기타 순으로 처음 해당하는 하나."""
    conds = [np.asarray(c, bool) for c in (data_problem, natural_barrier, cross_sido, r_top10)]
    return np.select(conds, CLASSES[:4], default=CLASSES[4])


def side_edges(u, v, barrier):
    """barrier 엣지와 끝점을 공유하는, barrier가 아닌 엣지의 지시자(옆 엣지)."""
    u, v, barrier = np.asarray(u), np.asarray(v), np.asarray(barrier, bool)
    nodes = set(u[barrier]) | set(v[barrier])
    touch = np.array([(a in nodes) or (b in nodes) for a, b in zip(u, v)])
    return touch & ~barrier


def hypergeom_top(values, top_idx, frac=0.10):
    """values 상위 frac를 'N 중 K개'로 보고 상위 k개 중 겹치는 수의 단측 초기하 p(기저율 frac)."""
    values = np.asarray(values, float)
    is_top = values >= np.quantile(values, 1.0 - frac)
    n_all, k_all, k = len(values), int(is_top.sum()), len(top_idx)
    hit = int(is_top[top_idx].sum())
    return {"hit": hit, "k": k, "N": n_all, "K": k_all, "p": float(stats.hypergeom.sf(hit - 1, n_all, k_all, k))}


def decide_0s(validity_ok, n_data_problem_top20, main, partial, island_pass, od_main=None, stage1=None, alpha=P_THRESHOLD):
    """ADR 0013 5항 0S 곡률 판정. 조정 -> 계속 -> 전제 재검토 -> 부분 대응 순서로 처음 해당하는 하나.

    입력
      validity_ok           2항 유효성 점검 통과 여부(bool)
      n_data_problem_top20  상위 20개 중 데이터·설정 문제 수(int)
      main, partial         {'p': 단순 p, 'p_jack': jackknife p}  (1항 두 검정, 균등 이웃 분포)
      island_pass           {'base': (주 통과, 편 통과), 'x0.5': (...), 'x2': (...)}  (섬 링크 민감도)
      od_main               OD 가중 주 검정 {'p', 'p_jack'}; 계산하지 못했으면 None
      stage1                1단계 점검 {'rho', 'p'}(분/km와 경사의 단측 Spearman); 없으면 None
    판정에 필요한 수치가 빠지면 '미완'이다.
    {'label', 'detail', 'missing'}를 돌려준다.
    """

    def ok(t):
        return t is not None and all(t.get(k) is not None and np.isfinite(t[k]) for k in ("p", "p_jack"))

    def passed(t):
        return bool(t["p"] < alpha and t["p_jack"] < alpha)

    missing = []
    if validity_ok is None:
        missing.append("2항 유효성 점검")
    if n_data_problem_top20 is None:
        missing.append("상위 20개 데이터·설정 문제 수")
    if not ok(main):
        missing.append("1항 주 검정")
    if not ok(partial):
        missing.append("1항 편상관 검정")
    keys = ("base", "x0.5", "x2")
    if (island_pass is None or any(island_pass.get(k) is None or any(b is None for b in island_pass[k]) for k in keys)):
        missing.append("섬 링크 민감도")
    if missing:
        return {"label": "미완", "detail": "판정에 필요한 수치가 없다: " + ", ".join(missing), "missing": missing}

    main_pass, partial_pass = passed(main), passed(partial)
    flips = [(k, nm) for k in ("x0.5", "x2") for i, nm in enumerate(("주 검정", "편상관 검정"))
             if bool(island_pass[k][i]) != bool(island_pass["base"][i])]
    reasons = []
    if not validity_ok:
        reasons.append("유효성 점검 실패")
    if n_data_problem_top20 >= DATA_PROBLEM_MAJORITY:
        reasons.append(f"상위 20개 중 데이터·설정 문제 {n_data_problem_top20}개(과반)")
    if flips:
        reasons.append("섬 링크 민감도에서 통과 여부 뒤집힘: " + ", ".join(f"{k} {nm}" for k, nm in flips))
    if reasons:
        return {"label": "조정", "detail": "; ".join(reasons), "missing": []}

    if main_pass and partial_pass:
        return {"label": "계속", "detail": "주 검정과 편상관 검정이 단순 p와 jackknife p 모두 통과", "missing": []}

    if not main_pass:
        if od_main is not None and ok(od_main):
            od_pass = passed(od_main)
            if not od_pass:
                if stage1 is None or not all(stage1.get(k) is not None and np.isfinite(stage1[k]) for k in ("rho", "p")):
                    return {"label": "미완", "detail": "판정에 필요한 수치가 없다: 1단계 점검", "missing": ["1단계 점검"]}
                if stage1["rho"] > 0 and stage1["p"] < alpha:
                    return {"label": "전제 재검토", "detail": "주 검정 실패, OD 가중 주 검정 실패, 1단계 점검 상관이 양수(단측 p < 0.05): "
                            "통행시간은 지형을 반영하는데 곡률은 반영하지 않는다", "missing": []}
                return {"label": "부분 대응", "detail": "부분 대응: 1단계 점검이 양수가 아니어서 통행시간에 지형 신호가 없음", "missing": []}
            return {"label": "부분 대응", "detail": "부분 대응: OD 가중에서만 통과", "missing": []}
        return {"label": "부분 대응", "detail": "부분 대응: OD 가중을 계산하지 못함", "missing": []}

    return {"label": "부분 대응", "detail": "부분 대응: 곡률의 지형 신호는 통행시간 신호와 같음", "missing": []}


# ===================================================================== 5. 입력 준비 보조
def connect_courses(courses):
    """GPX 코스를 잇는다(ADR 0013 6항).

    courses: [(정렬 키, 이름, xy 배열 (n, 2))]. 정렬 키 순(산권 순서, 산권 안 코스 번호)으로 잇고,
    각 코스는 앞 코스 끝점에 가까운 끝부터 잇는다. 첫 코스는 다음 코스에 더 가까운 쪽이 끝이 되게 놓는다.
    틈은 직선으로 잇는다(좌표가 이어 붙을 뿐이다).
    {'line', 'gaps', 'order', 'line_len_m', 'course_len_sum_m', 'gap_total_m'}를 돌려준다.
    """
    cs = sorted(courses, key=lambda c: c[0])
    if not cs:
        raise ValueError("코스가 없다")
    parts, order, gaps = [], [], []

    def d(a, b):
        return float(np.hypot(*(np.asarray(a) - np.asarray(b))))

    first = np.asarray(cs[0][2], float)
    if len(cs) > 1:
        nxt = np.asarray(cs[1][2], float)
        to_nxt = lambda p: min(d(p, nxt[0]), d(p, nxt[-1]))  # noqa: E731
        if to_nxt(first[0]) < to_nxt(first[-1]):
            first = first[::-1]
            order.append((cs[0][1], True))
        else:
            order.append((cs[0][1], False))
    else:
        order.append((cs[0][1], False))
    parts.append(first)
    for key, name, xy in cs[1:]:
        xy = np.asarray(xy, float)
        end = parts[-1][-1]
        rev = d(end, xy[-1]) < d(end, xy[0])
        if rev:
            xy = xy[::-1]
        gaps.append({"after": order[-1][0], "before": name, "length_m": d(end, xy[0])})
        order.append((name, bool(rev)))
        parts.append(xy)
    line = np.vstack(parts)

    def plen(a):
        return float(np.hypot(*np.diff(a, axis=0).T).sum()) if len(a) > 1 else 0.0

    return {"line": line, "gaps": gaps, "order": order, "line_len_m": plen(line),
            "course_len_sum_m": float(sum(plen(np.asarray(c[2], float)) for c in cs)), "gap_total_m": float(sum(g["length_m"] for g in gaps))}


def segment_points(x0, y0, x1, y1, max_step):
    """두 점 사이를 간격이 max_step 이하가 되게 고르게 나눈 표본(양 끝 포함). (xs, ys, 누적 거리, 전체 길이)."""
    length = float(np.hypot(x1 - x0, y1 - y0))
    k = max(1, int(math.ceil(length / max_step)))
    t = np.linspace(0.0, 1.0, k + 1)
    return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, length * t, length


def fill_linear(pos, z):
    """NaN(바다·자료 없음)을 누적 거리 기준 선형 보간한다. 양 끝이 NaN이면 가까운 유효값으로 채운다. 유효값이 없으면 그대로."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    if not ok.any():
        return z.copy()
    return np.interp(pos, pos[ok], z[ok])


def slope_profile(pos, z, length_m):
    """s = sum|dz| / 2 / 길이(km) (m/km), 바다·자료 없음 표본 비율(보간 전), 표본 수."""
    z = np.asarray(z, float)
    sea = float(np.mean(~np.isfinite(z)))
    zi = fill_linear(pos, z)
    if length_m <= 0 or not np.isfinite(zi).all():
        return float("nan"), sea, len(z)
    return float(np.abs(np.diff(zi)).sum() / 2.0 / (length_m / 1000.0)), sea, len(z)


def excess_height(pos, z):
    """h_e = 선분 위 최고 고도 - 두 끝점 고도 중 큰 값(보간한 표본 기준, 항상 0 이상)."""
    zi = fill_linear(pos, np.asarray(z, float))
    if not np.isfinite(zi).all():
        return float("nan")
    return float(zi.max() - max(zi[0], zi[-1]))


def detect_bridge_candidates(way_ids, node_ids, labels, adj_pairs, xy=None, land=None):
    """연륙·연도교 후보 탐지(ADR 0013 1항)와 바다 횡단 판정.

    way_ids, node_ids, labels: (way, 노드 순서)로 정렬된 같은 길이의 배열. labels는 노드가 속한 시군구 번호이고
    다각형 밖(바다 위) 노드는 -1이다. adj_pairs: 경계를 공유하는 (작은 번호, 큰 번호) 쌍의 집합.
    xy: (행 수, 2) 노드 좌표(EPSG:5179), land: 지역 다각형 합집합(shapely 지오메트리). 둘 다 주면 바다 횡단을 판정한다.

    규칙 A('within_way'): 한 way 안에서 -1 노드를 건너뛰고 이웃한 두 라벨이 서로 다른 시군구이고 경계를 공유하지
      않으면 후보다. (지시 원문의 규칙)
    규칙 B('chain'): -1 노드끼리 이어진 덩어리(way 경계를 넘어 노드를 공유하는 도로 조각들)에 라벨 두 개 이상이
      닿아 있으면 그 라벨 쌍 중 경계를 공유하지 않는 쌍이 후보다. 바다 위 노드만 있는 다리·터널 way가
      육지 쪽 way와 별개로 매핑된 경우(거가대로, 원산안면대교 등)를 잡는다.
    바다 횡단(evidence의 sea 열): 두 육지 라벨 사이에 합집합 밖 노드가 하나 이상 있으면 'nodes', 그렇지 않고 라벨이 붙은
      이웃 노드 사이 선분이 합집합 밖을 지나면 'segment', 둘 다 아니면 'none'(바다를 지나지 않는 육지 터널 등).
      규칙 B 후보는 정의상 합집합 밖 노드를 지나므로 'nodes'다. xy나 land를 주지 않으면 모두 'unchecked'다.
      이 함수는 후보를 거르지 않는다(sea == 'none'인 후보를 부르는 쪽에서 뺀다).
    (evidence DataFrame[lo, hi, rule, evidence, n_nodes, sea], evidence_ways DataFrame[evidence, way_id])를 돌려준다.
    """
    import shapely

    way_ids, node_ids, labels = (np.asarray(a) for a in (way_ids, node_ids, labels))
    labels = labels.astype(np.int64)
    check = xy is not None and land is not None
    if check:
        xy = np.asarray(xy, float)
    # --- 규칙 A
    lab_pos = np.flatnonzero(labels >= 0)
    vw, vl = way_ids[lab_pos], labels[lab_pos]
    chg = (vw[:-1] == vw[1:]) & (vl[:-1] != vl[1:])
    ia, ib = lab_pos[:-1][chg], lab_pos[1:][chg]
    rows_a = []
    for k in range(len(ia)):
        lo, hi = sorted((int(labels[ia[k]]), int(labels[ib[k]])))
        if (lo, hi) in adj_pairs:
            continue
        sea = "unchecked"
        if check:
            between = xy[ia[k] + 1:ib[k]]
            if len(between) and (~shapely.intersects_xy(land, between[:, 0], between[:, 1])).any():
                sea = "nodes"
            elif shapely.covers(land, shapely.linestrings([xy[ia[k]], xy[ib[k]]])):
                sea = "none"
            else:
                sea = "segment"
        rows_a.append((lo, hi, "within_way", f"a{k}", int(ib[k] - ia[k] - 1), sea, int(way_ids[ia[k]])))
    ev_a = pd.DataFrame(rows_a, columns=["lo", "hi", "rule", "evidence", "n_nodes", "sea", "way_id"])
    ways_a = ev_a[["evidence", "way_id"]].copy()
    # --- 규칙 B
    uniq, inv = np.unique(node_ids, return_inverse=True)
    lab_u = np.full(len(uniq), -1, dtype=np.int64)
    lab_u[inv] = labels
    same = way_ids[:-1] == way_ids[1:]
    eu, ev_ = inv[:-1][same], inv[1:][same]
    la, lb = lab_u[eu], lab_u[ev_]
    both_sea = (la < 0) & (lb < 0)
    g = coo_matrix((np.ones(int(both_sea.sum())), (eu[both_sea], ev_[both_sea])), shape=(len(uniq), len(uniq)))
    _, comp = connected_components(g, directed=False)
    comp = np.where(lab_u < 0, comp, -1)
    m1, m2 = (la >= 0) & (lb < 0), (la < 0) & (lb >= 0)
    att = pd.DataFrame({"comp": np.r_[comp[ev_[m1]], comp[eu[m2]]], "label": np.r_[la[m1], lb[m2]]}).drop_duplicates()
    rows_b = []
    for c, grp in att.groupby("comp"):
        labs = sorted(grp["label"].tolist())
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                if (labs[i], labs[j]) not in adj_pairs:
                    rows_b.append((labs[i], labs[j], "chain", f"c{c}", int((comp == c).sum()), "nodes" if check else "unchecked"))
    ev_b = pd.DataFrame(rows_b, columns=["lo", "hi", "rule", "evidence", "n_nodes", "sea"])
    ev_ids_b = set(ev_b["evidence"])
    if len(ev_b):
        cw = pd.DataFrame({"comp": comp[inv], "way_id": way_ids})
        cw = cw[cw["comp"] >= 0].drop_duplicates()
        cw["evidence"] = "c" + cw["comp"].astype(str)
        ways_b = cw[cw["evidence"].isin(ev_ids_b)][["evidence", "way_id"]]
    else:
        ways_b = pd.DataFrame(columns=["evidence", "way_id"])
    evidence = pd.concat([ev_a.drop(columns="way_id"), ev_b], ignore_index=True)
    ways = pd.concat([ways_a, ways_b], ignore_index=True)
    return evidence, ways


EDGE_CSV_DTYPE = {"u_code": str, "v_code": str, "sido_u": str, "sido_v": str, "bridge_rule": str}
EDGE_REQUIRED = ["u_idx", "v_idx", "kind", "d_sym", "straight_km", "r", "inv_deg_sum", "coastal_edge", "cross_sido", "sido_u", "sido_v", "s", "sea_frac", "h_e",
                 "b_baekdu", "b_sens", "conn_gt5km", "node_match_error"]


def read_edge_csv(path):
    """g_adj_edges.csv / g_adj249_edges.csv를 분석 함수가 기대하는 자료형으로 읽는다."""
    return pd.read_csv(path, encoding="utf-8-sig", dtype=EDGE_CSV_DTYPE)


def check_edge_table(E, n, scenarios=True, od=False):
    """analyze_curvature가 받는 엣지 표의 계약을 점검한다(열, 자료형, 결측, 범위, 중복). 어긋나면 ValueError."""
    problems = []
    need = EDGE_REQUIRED + (["d_sym_x0.5", "d_sym_x2"] if scenarios else []) + (["w_od"] if od else [])
    miss = [c for c in need if c not in E.columns]
    if miss:
        raise ValueError(f"열이 없다: {miss}")
    u, v = E["u_idx"].to_numpy(), E["v_idx"].to_numpy()
    if not (u < v).all() or u.min() < 0 or v.max() >= n:
        problems.append("u_idx < v_idx 또는 노드 번호 범위가 맞지 않는다")
    if len(np.unique(edge_key(u, v, n))) != len(E):
        problems.append("중복 엣지")
    if not set(E["kind"]) <= {"boundary", "bridge", "island"}:
        problems.append(f"알 수 없는 kind {set(E['kind'])}")
    for c in ["d_sym", "straight_km", "r", "inv_deg_sum"] + (["d_sym_x0.5", "d_sym_x2"] if scenarios else []) + (["w_od"] if od else []):
        x = E[c].to_numpy(float)
        if not (np.isfinite(x).all() and (x > 0).all()):
            problems.append(f"{c}: 유한한 양수가 아닌 값이 있다")
    bnd = E["kind"] == "boundary"
    for c in ("s", "sea_frac", "h_e", "coastal_edge"):
        if not np.isfinite(E.loc[bnd, c].to_numpy(float)).all():
            problems.append(f"{c}: 경계 공유 엣지에 결측이 있다")
    for c in ("cross_sido", "b_baekdu", "b_sens", "conn_gt5km", "node_match_error"):
        if E[c].dtype != bool:
            problems.append(f"{c}: bool이 아니다({E[c].dtype})")
    if not all(isinstance(x, str) for x in E["sido_u"].tolist() + E["sido_v"].tolist()):
        problems.append("sido_u, sido_v가 문자열이 아니다")
    if (E["sido_u"] != E["sido_v"]).to_numpy().tolist() != E["cross_sido"].to_numpy().tolist():
        problems.append("cross_sido가 sido_u != sido_v와 다르다")
    if n_components(n, u, v) != 1:
        problems.append("그래프가 연결되어 있지 않다")
    if problems:
        raise ValueError("엣지 표 계약 위반: " + "; ".join(problems))
    return {"edges": len(E), "boundary": int(bnd.sum()), "nodes": n}


# ===================================================================== 6. 통합 분석, 지도, 장난감 나라
def analyze_curvature(E, n, sido_ids, scenario_cols=("d_sym_x0.5", "d_sym_x2"), alpha=ALPHA_DEFAULT, weight=None):
    """엣지 표 E(g_adj_edges.csv와 같은 열)에서 3절의 핵심 계산을 한 번에 한다.

    필요한 열: u_idx, v_idx, kind('boundary' | 'bridge' | 'island'), d_sym, straight_km, r(=d_sym/직선 km),
      inv_deg_sum, coastal_edge, cross_sido, sido_u, sido_v, s, sea_frac, h_e, b_baekdu, (선택) b_sens,
      conn_gt5km, node_match_error, scenario_cols(섬 링크 시간 x0.5, x2 재계산 d_sym).
    weight: 엣지별 이웃 가중치(OD 가중). None이면 균등.
    입력 E는 바꾸지 않는다. 결과 dict를 돌려준다.
    """
    u, v = E["u_idx"].to_numpy(), E["v_idx"].to_numpy()
    bnd = (E["kind"] == "boundary").to_numpy()
    straight = E["straight_km"].to_numpy(float)
    scen_names = {"base": "d_sym"}
    for c in scenario_cols:
        if c in E.columns:
            scen_names[c.replace("d_sym_", "")] = c
    cur = {k: curvature_set(n, u, v, E[c].to_numpy(float), straight, bnd, alpha, weight) for k, c in scen_names.items()}
    res = {"cur": cur, "scenarios": list(scen_names)}
    sub = E[bnd]
    kex = cur["base"]["kappa_ex"][bnd]
    su, sv = sub["sido_u"].to_numpy(), sub["sido_v"].to_numpy()
    r = sub["r"].to_numpy(float)

    def battery(kx, xcol, mask=None):
        m = np.ones(len(sub), bool) if mask is None else mask
        return run_battery(kx[m], sub[xcol].to_numpy(float)[m], r[m], su[m], sv[m], sido_ids)

    res["battery"] = battery(kex, "s")
    res["island_pass"] = {"base": (res["battery"]["main"]["passed"], res["battery"]["partial"]["passed"])}
    res["island_battery"] = {}
    for k in scen_names:
        if k == "base":
            continue
        b = battery(cur[k]["kappa_ex"][bnd], "s")
        res["island_battery"][k] = b
        res["island_pass"][k] = (b["main"]["passed"], b["partial"]["passed"])
    res["validity"] = validity_checks(kex, sub["inv_deg_sum"].to_numpy(float), sub["coastal_edge"].to_numpy(float), np.log(sub["straight_km"].to_numpy(float)))
    kap = cur["base"]["kappa"][bnd]
    res["structure"] = {
        "kappa": validity_checks(kap, sub["inv_deg_sum"].to_numpy(float), sub["coastal_edge"].to_numpy(float), np.log(sub["straight_km"].to_numpy(float)))["rho"],
        "kappa_ex": res["validity"]["rho"]}
    res["stage1"] = dict(zip(("rho", "p"), spearman_greater(r, sub["s"].to_numpy(float))))
    res["rho_kex_r"] = spearman_two_sided(kex, r)
    # 보조 검정
    res["sea_excluded"] = battery(kex, "s", (sub["sea_frac"].to_numpy(float) <= 0.5))
    res["n_sea_over_half"] = int((sub["sea_frac"] > 0.5).sum())
    res["h_e"] = battery(kex, "h_e")
    res["barrier_baekdu"] = battery(kex, "b_baekdu")
    if "b_sens" in sub.columns and sub["b_sens"].notna().all():
        res["barrier_sens"] = battery(kex, "b_sens")
    # 상위 20개와 5분류
    top = top_indices(kex, 20)
    flag_isl, thr20 = island_sensitive(kex, [cur[k]["kappa_ex"][bnd] for k in scen_names if k != "base"], top)
    data_problem = sub["conn_gt5km"].to_numpy(bool) | sub["node_match_error"].to_numpy(bool) | flag_isl
    rtop = r >= np.quantile(r, 0.9)
    labels = classify_edges(data_problem, sub["b_baekdu"].to_numpy(bool), sub["cross_sido"].to_numpy(bool), rtop)
    res["top20"] = {"idx_in_boundary": top, "edge_pos": np.flatnonzero(bnd)[top], "label": labels[top], "kappa_ex": kex[top],
                    "island_sensitive": flag_isl[top], "threshold_20th": thr20,
                    "n_data_problem": int(data_problem[top].sum()), "n_barrier": int((labels[top] == "자연 장벽").sum()),
                    "n_each": {c: int((labels[top] == c).sum()) for c in CLASSES},
                    "coastal_ratio": float(sub["coastal_edge"].to_numpy(float)[top].mean()),
                    "coastal_ratio_all": float(sub["coastal_edge"].to_numpy(float).mean()),
                    "h_e_hyper": hypergeom_top(sub["h_e"].to_numpy(float), top)}
    res["bottom20_kappa_pos"] = np.flatnonzero(bnd)[top_indices(-kap, 20)]
    bar = sub["b_baekdu"].to_numpy(bool)
    side = side_edges(sub["u_idx"].to_numpy(), sub["v_idx"].to_numpy(), bar)
    res["side_edges"] = {"n_barrier": int(bar.sum()), "n_side": int(side.sum()),
                         "mean_kex_barrier": float(kex[bar].mean()) if bar.any() else float("nan"),
                         "mean_kex_side": float(kex[side].mean()) if side.any() else float("nan"),
                         "mean_kex_other": float(kex[~bar & ~side].mean())}
    nb = ~bnd
    res["non_boundary_kappa"] = pd.DataFrame({"kind": E["kind"].to_numpy()[nb], "u_name": E["u_name"].to_numpy()[nb], "v_name": E["v_name"].to_numpy()[nb],
                                              "kappa": cur["base"]["kappa"][nb], "kappa_ex": cur["base"]["kappa_ex"][nb]}) if "u_name" in E.columns else None
    # 판정
    res["decision_inputs"] = {"validity_ok": res["validity"]["ok"], "n_data_problem_top20": res["top20"]["n_data_problem"],
                              "main": {"p": res["battery"]["main"]["p"], "p_jack": res["battery"]["main"]["p_jack"]},
                              "partial": {"p": res["battery"]["partial"]["p"], "p_jack": res["battery"]["partial"]["p_jack"]},
                              "island_pass": res["island_pass"], "stage1": res["stage1"]}
    return res


def plot_edge_map(ax, boundaries, xy, u, v, values, kind=None, highlight=None, line=None, vmax=None, cmap="RdBu", title="", label="", annotate=None, lw=1.6):
    """엣지를 값으로 칠한 지도. boundaries: 배경 경계(GeoSeries/GeoDataFrame 또는 None), xy: (n, 2) 출발점, u/v: 엣지 끝점(노드 번호).

    kind가 'bridge'/'island'인 엣지는 파선/점선, highlight 위치의 엣지는 검은 테두리로 굵게 그린다.
    값이 작을수록 붉고 클수록 푸르다(0 중심 발산 색). annotate = [(엣지 위치, 글자)].
    """
    import matplotlib as mpl
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    values = np.asarray(values, float)
    vmax = float(np.nanmax(np.abs(values))) if vmax is None else vmax
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cm = mpl.colormaps[cmap]
    u, v = np.asarray(u), np.asarray(v)
    if boundaries is not None:
        boundaries.boundary.plot(ax=ax, linewidth=0.25, color="0.75")
    if line is not None:
        ax.plot(line[:, 0], line[:, 1], color="tab:green", linewidth=1.4, alpha=0.85, zorder=3)
    kind = np.asarray(kind) if kind is not None else np.array(["boundary"] * len(u))
    hl = np.zeros(len(u), bool)
    if highlight is not None:
        hl[np.asarray(highlight, int)] = True
    for kd, ls in (("boundary", "-"), ("bridge", "--"), ("island", ":")):
        sel = (kind == kd) & ~hl
        if sel.any():
            segs = [[xy[a], xy[b]] for a, b in zip(u[sel], v[sel])]
            ax.add_collection(LineCollection(segs, colors=cm(norm(values[sel])), linewidths=lw, linestyles=ls, zorder=4))
    if hl.any():
        segs = [[xy[a], xy[b]] for a, b in zip(u[hl], v[hl])]
        ax.add_collection(LineCollection(segs, colors="k", linewidths=lw * 3.2, zorder=5))
        ax.add_collection(LineCollection(segs, colors=cm(norm(values[hl])), linewidths=lw * 1.7, zorder=6))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="0.35", zorder=7)
    if annotate is not None:
        for pos, text in annotate:
            a, b = xy[u[pos]], xy[v[pos]]
            ax.annotate(text, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), fontsize=7, zorder=8, ha="center", va="center", bbox={"boxstyle": "round,pad=0.1", "fc": "w", "ec": "none", "alpha": 0.7})
    handles = [Line2D([0], [0], color="0.4", lw=lw, ls="-", label="경계 공유"), Line2D([0], [0], color="0.4", lw=lw, ls="--", label="연륙·연도교"),
               Line2D([0], [0], color="0.4", lw=lw, ls=":", label="섬 링크")]
    if line is not None:
        handles.append(Line2D([0], [0], color="tab:green", lw=1.4, label="백두대간 선"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax.figure.colorbar(sm, ax=ax, shrink=0.55, label=label)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.autoscale_view()


def make_toy_country(seed=0, rows=6, cols=6, cell_km=20.0):
    """파이프라인 배선 점검용 장난감 나라(실데이터가 아니다). 격자 시군구, 산맥 선 하나, 다리 하나, 섬 링크 하나.

    g_adj_edges와 같은 열을 가진 엣지 표 E, 노드 수 n, 출발점 xy(m), 산맥 선(m), 시도 번호를 돌려준다.
    산맥을 건너는 엣지는 우회율과 경사가 크다(심어 둔 신호). 장난감 값으로 판정의 의미를 읽지 않는다.
    """
    rng = np.random.default_rng(seed)
    n = rows * cols
    cell = cell_km * 1000.0
    ij = np.array([(i, j) for i in range(rows) for j in range(cols)])
    xy = np.c_[ij[:, 1] * cell, ij[:, 0] * cell] + rng.normal(0, cell * 0.08, (n, 2))
    sido = (ij[:, 0] // 2) * 3 + (ij[:, 1] // 2)  # 3 x 3 = 9개 시도 블록
    ridge_x = 2.5 * cell
    pairs, kinds = [], []
    idx = lambda i, j: i * cols + j  # noqa: E731
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                pairs.append((idx(i, j), idx(i, j + 1)))
                kinds.append("boundary")
            if i + 1 < rows:
                pairs.append((idx(i, j), idx(i + 1, j)))
                kinds.append("boundary")
            if i + 1 < rows and j + 1 < cols and (i + j) % 2 == 0:
                pairs.append((idx(i, j), idx(i + 1, j + 1)))
                kinds.append("boundary")
    pairs.append((idx(0, 0), idx(2, 2)))
    kinds.append("bridge")
    pairs.append((idx(0, cols - 1), idx(rows - 1, 0)))
    kinds.append("island")
    u = np.array([min(a, b) for a, b in pairs])
    v = np.array([max(a, b) for a, b in pairs])
    straight = np.hypot(*(xy[u] - xy[v]).T) / 1000.0
    cross_ridge = ((xy[u, 0] - ridge_x) * (xy[v, 0] - ridge_x) < 0)
    r = 1.4 + rng.normal(0, 0.08, len(u)) + 0.9 * cross_ridge
    is_island = np.array(kinds) == "island"
    d_sym = np.where(is_island, 150.0, straight * r)
    deg = np.bincount(np.r_[u, v], minlength=n)
    E = pd.DataFrame({
        "u_idx": u, "v_idx": v, "kind": kinds, "d_sym": d_sym, "d_sym_x0.5": np.where(is_island, 75.0, d_sym), "d_sym_x2": np.where(is_island, 300.0, d_sym),
        "straight_km": straight, "r": d_sym / straight, "inv_deg_sum": 1.0 / deg[u] + 1.0 / deg[v],
        "coastal_edge": ((ij[u, 0] % (rows - 1) == 0) | (ij[v, 0] % (rows - 1) == 0) | (ij[u, 1] % (cols - 1) == 0) | (ij[v, 1] % (cols - 1) == 0)).astype(int),
        "cross_sido": sido[u] != sido[v], "sido_u": sido[u], "sido_v": sido[v],
        "s": 20 + rng.gamma(2.0, 6.0, len(u)) + 40 * cross_ridge, "sea_frac": np.where(is_island, 0.9, rng.uniform(0, 0.3, len(u))),
        "h_e": rng.gamma(2.0, 50.0, len(u)) + 400 * cross_ridge, "b_baekdu": cross_ridge, "b_sens": cross_ridge | (rng.uniform(size=len(u)) < 0.05),
        "conn_gt5km": False, "node_match_error": rng.uniform(size=len(u)) < 0.03,
        "u_name": [f"R{a}" for a in u], "v_name": [f"R{b}" for b in v]})
    line = np.array([[ridge_x, -cell], [ridge_x, rows * cell]])
    return {"n": n, "E": E, "xy": xy, "line": line, "sido": sido, "sido_ids": sorted(set(sido.tolist()))}
