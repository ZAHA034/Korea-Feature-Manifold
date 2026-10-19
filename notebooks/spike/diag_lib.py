"""KFM 0S 사후 탐색 진단(A–F) 계산 코드.

탐색적(사전 판정과 무관, ADR 0013 7항). 0S 판정(조정, 쌍곡 이점 없음, 중력모형 이상의 위계 없음)을 바꾸지 않는다.
이 모듈의 결과는 "왜 없었는가"를 진단하는 용도이며 판정 규칙의 기준값이 아니다.

embed_lib.py(고정된 판정 코드)는 수정하지 않고 import해서 쓴다. 순수 함수만 둔다(입력 배열을 제자리에서 바꾸지 않는다).
실행 환경은 C:\\dev\\kfm-spike. 노트북 04_diagnostics.ipynb가 이 모듈을 불러 쓴다.

구성
  0. 공통: 입력 불러오기(MD5 확인), 표시 문구
  A. 밀도를 맞춘 대조군: 같은 지지 집합 대조군(A1), 엣지 수를 맞춘 재표본 대조군(A2), 대조군 밀도 곡선(A3, 보조)
  B. 밀도별 δ̄ 곡선: ℓ_sym 상위 p만 남기고 폐포
  C. 차원: 차원 D에서 2단계 최적화(embed_lib를 cfg["dim"]으로 재사용)
  D. 인구이동 OD(KOSIS 국내인구이동 2025): 읽기, 229개 단위, d_time 229, 준비
  E. 반경별 지역 수 증가(공 성장): 다항식형과 지수형 적합
  F. 가설별 판정 규칙(탐색적)
  P. 그림
"""

from __future__ import annotations

import embed_lib as el  # numpy, torch보다 먼저: 스레드 환경변수와 PYTORCH_JIT=0을 정한다

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

EXPLORATORY = "탐색적(사전 판정과 무관, ADR 0013 7항)"
OSM_NOTICE = "© OpenStreetMap contributors (ODbL 1.0)"
NORMS, SUBSETS = el.NORMS, el.SUBSETS
COND = [(n, s) for n in NORMS for s in SUBSETS]  # 판정 조건 4개의 순서
NORM_KO = {"main": "주", "alt": "대안"}


def cond_key(norm, subset):
    return f"{norm}|{subset}"


# ===================================================================== 0. 공통
def md5_file(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(cwd):
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip() if r.returncode == 0 else None


S4_NPY = (
    "d_time_247", "t_dir_247", "od_247", "d_flow_main_247", "d_flow_alt_247", "gravity_expected_247",
    "controls_d_main_247", "controls_d_alt_247", "controls_T_247", "d_time_250", "t_dir_250",
)


def load_s4_inputs(s4_dir, freeze_path):
    """S4가 고정한 입력 행렬(C:\\dev\\kfm-data\\spike\\s4)을 불러온다. s4_freeze.json의 MD5와 다르면 중단한다."""
    s4 = Path(s4_dir)
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    md5s = freeze["outputs_md5"]
    out, checked = {}, {}
    for k in S4_NPY:
        fn = f"{k}.npy"
        got = md5_file(s4 / fn)
        if got != md5s[fn]:
            raise RuntimeError(f"{fn}: s4_freeze.json의 MD5와 다르다")
        checked[fn] = got
        out[k] = np.load(s4 / fn, allow_pickle=False)
    for fn in ("units_247.csv", "units_250.csv"):
        got = md5_file(s4 / fn)
        if got != md5s[fn]:
            raise RuntimeError(f"{fn}: s4_freeze.json의 MD5와 다르다")
        checked[fn] = got
    out["units247"] = pd.read_csv(s4 / "units_247.csv", encoding="utf-8-sig", dtype={"unit_code": str})
    out["units250"] = pd.read_csv(s4 / "units_250.csv", encoding="utf-8-sig", dtype={"unit_code": str})
    out["island247"] = out["units247"]["island_connected"].to_numpy(dtype=bool)
    out["island250"] = out["units250"]["island_connected"].to_numpy(dtype=bool)
    out["pop250"] = out["units250"]["pop_2025_12"].to_numpy(dtype=np.float64)
    out["md5"] = checked
    return out


def stats_of(values):
    v = np.asarray(values, dtype=float)
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def json_clean(obj):
    """numpy 값을 기본형으로 바꾸고 NaN, inf는 null로 바꾼다(엄격한 JSON)."""

    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rec(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    return rec(el.to_jsonable(obj))


def group_centroids(x, y, groups, weights, k):
    """그룹별 가중 평균 좌표(인구 가중 중심)."""
    w = np.bincount(groups, weights=weights, minlength=k)
    return np.bincount(groups, weights=np.asarray(x) * weights, minlength=k) / w, np.bincount(groups, weights=np.asarray(y) * weights, minlength=k) / w


# ===================================================================== A. 밀도를 맞춘 대조군
def n_edges(T):
    """양방향 엣지(T_ij > 0이고 T_ji > 0인 쌍) 수."""
    return int(el.flow_edge_mask(T).sum() // 2)


def delta_bars(d_main, d_alt, island, cfg):
    """한 쌍의 d_flow(주·대안)의 δ̄. {(정규화, 부분집합): 값}. 섬 제외는 섬 연결 지역의 부분행렬(폐포 다시 하지 않음)."""
    r = el.control_delta_bars({"main": [d_main], "alt": [d_alt]}, island, cfg)
    return {k: float(v[0]) for k, v in r.items()}


def same_support_control(T_exp, T_obs):
    """A1: 중력모형 기댓값 T̂를 재표본 없이 관측의 양방향 엣지 집합에만 두고 나머지는 0으로 한다."""
    edge = el.flow_edge_mask(T_obs)
    Tc = np.where(edge, np.asarray(T_exp, dtype=np.float64), 0.0)
    np.fill_diagonal(Tc, 0.0)
    return Tc


def resample_cfg(base_cfg, frac):
    """재표본 설정: 유효 표본 수 = 존 간 합 x frac, 확대 배율 = 1/frac(총량 보존. ℓ은 배율에 불변)."""
    frac = float(frac)
    return {**base_cfg, "resample_frac": frac, "resample_scale": 1.0 / frac}


def mean_edges(T_exp, T_obs, frac, seeds, base_cfg):
    """재표본 대조군(관측 0 패턴 적용)의 평균 양방향 엣지 수와 시드별 값."""
    cfg = resample_cfg(base_cfg, frac)
    e = [n_edges(el.resample_control(T_exp, T_obs, s, cfg)[0]) for s in seeds]
    return float(np.mean(e)), e


def match_sample_multiplier(T_exp, T_obs, base_cfg, tol=0.01, seeds=tuple(range(20)), k_max=2.0**24, rel_prec=1e-3):
    """A2: 유효 표본 배율 k(0S의 frac 0.2에 곱하는 값)를 이분법으로 정해, 관측 0 패턴을 적용한 재표본 대조군의 평균 양방향 엣지 수가
    관측 엣지 수의 +-tol 안에 들게 한다. 엣지 수는 관측 지지 집합 크기(관측 엣지 수)에 점근적으로만 접근하므로(그 위로 갈 수 없다)
    허용 대역 [target(1 - tol), target]의 가운데(target(1 - tol/2))를 목표로 삼는다. 평균은 고정한 시드 집합 위에서 구한다(결정적).
    목표 이상이 되는 가장 작은 k를 기하 이분법으로 찾는다."""
    target = n_edges(T_obs)
    band = (target * (1.0 - tol), target * (1.0 + tol))
    goal = target * (1.0 - tol / 2.0)
    f0 = float(base_cfg["resample_frac"])
    hist = []

    def E(k):
        e, _ = mean_edges(T_exp, T_obs, f0 * k, seeds, base_cfg)
        hist.append({"k": float(k), "frac": f0 * k, "mean_edges": e})
        return e

    k_lo, e_lo = 1.0, E(1.0)
    if e_lo >= goal:
        return {"ok": band[0] <= e_lo <= band[1], "k": 1.0, "frac": f0, "mean_edges": e_lo, "target": target, "band": band, "goal": goal, "history": hist}
    k_hi = 2.0
    while E(k_hi) < goal:
        k_lo, k_hi = k_hi, k_hi * 2.0
        if k_hi > k_max:
            return {"ok": False, "k": None, "frac": None, "mean_edges": hist[-1]["mean_edges"], "target": target, "band": band, "goal": goal,
                    "history": hist, "reason": f"k <= {k_max:g}에서 목표 엣지 수에 닿지 못했다"}
    while k_hi / k_lo - 1.0 > rel_prec:
        k_mid = math.sqrt(k_lo * k_hi)
        if E(k_mid) < goal:
            k_lo = k_mid
        else:
            k_hi = k_mid
    e_star, _ = mean_edges(T_exp, T_obs, f0 * k_hi, seeds, base_cfg)
    return {"ok": band[0] <= e_star <= band[1], "k": float(k_hi), "frac": f0 * k_hi, "mean_edges": e_star, "target": target, "band": band,
            "goal": goal, "history": hist}


def make_matched_controls(T_exp, T_obs, frac, base_cfg, n_wanted=20):
    """엣지 수를 맞춘 재표본 대조군 n_wanted개(시드 0, 1, ...순, 연결 성분이 둘 이상이면 건너뜀). embed_lib.make_controls 그대로."""
    return el.make_controls(T_exp, T_obs, n_wanted=n_wanted, cfg=resample_cfg(base_cfg, frac))


def control_density_sweep(T_exp, T_obs, ks, seeds, island, base_cfg):
    """A3(보조): 유효 표본 배율 k마다 재표본 대조군(시드 seeds)의 평균 엣지 수와 δ̄(전체·섬 제외, 주·대안) 평균.
    대조군 밀도가 δ̄를 움직이는지 직접 본다. 연결 성분이 둘 이상인 시드는 건너뛰고 개수를 적는다."""
    f0 = float(base_cfg["resample_frac"])
    rows = []
    for k in ks:
        cfg = resample_cfg(base_cfg, f0 * k)
        vals = {c: [] for c in COND}
        edges, skipped = [], 0
        for s in seeds:
            Tc, n_eff = el.resample_control(T_exp, T_obs, s, cfg)
            try:
                fd = el.flow_distance(Tc, kind="estimated", role=None)
            except el.DisconnectedFlowError:
                skipped += 1
                continue
            edges.append(int(fd["edge"].sum() // 2))
            b = delta_bars(fd["d_main"], fd["d_alt"], island, cfg)
            for c in COND:
                vals[c].append(b[c])
        row = {"k": float(k), "frac": f0 * k, "n_eff": int(n_eff), "n_controls": len(edges), "skipped": skipped, "mean_edges": float(np.mean(edges))}
        for c in COND:
            v = np.asarray(vals[c])
            row[f"{cond_key(*c)}_mean"] = float(v.mean())
            row[f"{cond_key(*c)}_min"] = float(v.min())
            row[f"{cond_key(*c)}_max"] = float(v.max())
        rows.append(row)
    return pd.DataFrame(rows)


def bars_long(name, bars, obs, n_edges_mean=None, n_edges_obs=None):
    """한 대조군 집합({(정규화, 부분집합): 배열 또는 값})의 δ̄ 요약(긴 형식). obs = 관측 δ̄ {(정규화, 부분집합): 값}."""
    rows = []
    for c in COND:
        v = np.atleast_1d(np.asarray(bars[c], dtype=float))
        rows.append(
            {
                "대상": name,
                "정규화": NORM_KO[c[0]],
                "부분집합": c[1],
                "n": int(len(v)),
                "엣지(평균)": n_edges_mean,
                "δ̄ 평균": float(v.mean()),
                "δ̄ 최솟값": float(v.min()),
                "δ̄ 최댓값": float(v.max()),
                "관측 δ̄": float(obs[c]),
                "관측 - 평균": float(obs[c] - v.mean()),
                "관측 이하인 대조군 수": int((v <= obs[c]).sum()),
            }
        )
    return rows


def _fmt_stats(v):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} [{v.min():.4f}-{v.max():.4f}]"


def comparison_table(groups, obs):
    """한 표로 비교: 행 = 대상, 열 = 4개 조건(정규화 x 부분집합)의 δ̄(대조군이 여럿이면 평균 [최솟값-최댓값]).
    groups = [(이름, 평균 엣지 수, {(정규화, 부분집합): 값 또는 배열})], obs = 관측 δ̄ {(정규화, 부분집합): 값}.
    마지막 열은 관측 이하인 대조군 수(4조건 순서: 주-전체, 주-섬 제외, 대안-전체, 대안-섬 제외)."""
    rows = []
    for name, e_mean, bars in groups:
        row = {"대상": name, "엣지(평균)": round(float(e_mean), 1), "n": int(np.atleast_1d(bars[COND[0]]).size)}
        for c in COND:
            row[f"δ̄ {NORM_KO[c[0]]}-{c[1]}"] = _fmt_stats(bars[c])
        row["관측 이하 대조군 수(4조건)"] = "-" if name.startswith("관측") else "/".join(str(int((np.atleast_1d(bars[c]) <= obs[c]).sum())) for c in COND)
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================== B. 밀도별 δ̄ 곡선
def density_curve(raw_ell, ps=tuple(round(0.1 * i, 1) for i in range(1, 11)), tie_rtol=el.CFG["tie_rtol"]):
    """폐포 전 ℓ(엣지 없는 쌍은 inf)에서 양방향 쌍 중 ℓ이 작은(상대 흐름이 강한) 상위 p만 남기고 최단경로 폐포를 한 뒤 δ̄를 구한다.
    연결이 끊기면 최대 연결 성분만 쓴다(n을 적는다). 각 p에서 남기는 쌍 수는 반올림(p x 전체 쌍 수)."""
    raw = np.asarray(raw_ell, dtype=np.float64)
    n = raw.shape[0]
    iu, ju = np.triu_indices(n, 1)
    w = raw[iu, ju]
    fin = np.isfinite(w)
    iu, ju, w = iu[fin], ju[fin], w[fin]
    order = np.argsort(w, kind="stable")
    iu, ju, w = iu[order], ju[order], w[order]
    m = len(w)
    rows = []
    for p in ps:
        k = min(m, max(1, int(math.floor(p * m + 0.5))))
        W = np.full((n, n), np.inf)
        W[iu[:k], ju[:k]] = w[:k]
        W[ju[:k], iu[:k]] = w[:k]
        edge = np.isfinite(W)
        nc, lab = connected_components(csr_matrix(edge), directed=False)
        sizes = np.bincount(lab)
        main = int(np.argmax(sizes))
        idx = np.flatnonzero(lab == main)
        row = {"p": float(p), "남긴 쌍": int(k), "ℓ 상한": float(w[k - 1]), "연결 성분": int(nc), "고립 지역": int((sizes == 1).sum()), "n(최대 성분)": int(len(idx))}
        if len(idx) >= 4:
            ec = edge[np.ix_(idx, idx)]
            D = el.metric_closure(W[np.ix_(idx, idx)], ec)
            st = el.four_point_stats(D, tie_rtol)
            diam = float(D.max())
            row.update(
                {
                    "엣지(최대 성분)": int(ec.sum() // 2),
                    "δ̄": st["delta_bar"],
                    "δ": st["delta"],
                    "diam": diam,
                    "δ_rel": (2.0 * st["delta"] / diam) if diam > 0 else None,
                    "동률 네 점": st["n_ties"],
                }
            )
        else:
            row.update({"엣지(최대 성분)": int(edge[np.ix_(idx, idx)].sum() // 2), "δ̄": None, "δ": None, "diam": None, "δ_rel": None, "동률 네 점": None})
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================== C. 차원
def embed_dims(problems, runner, base_cfg, max_coordinators=None):
    """problems = {이름: (D 행렬, 차원)}. 입력마다 cfg["dim"]만 바꿔 embed_lib.embed_input(2단계 최적화, 재시작 구성, 재실행 규칙 그대로)을 부른다.
    입력마다 조정 스레드 하나가 공용 runner에 재시작을 보낸다. 돌려주는 값: (결과 {이름: embed_input 결과}, 설정 {이름: cfg})."""
    cfgs = {nm: el.check_cfg({**base_cfg, "dim": int(dim)}) for nm, (_, dim) in problems.items()}
    with ThreadPoolExecutor(max_workers=max_coordinators or len(problems)) as ex:
        futs = {nm: ex.submit(el.embed_input, problems[nm][0], runner, cfgs[nm], nm) for nm in problems}
        results = {nm: futs[nm].result() for nm in problems}
    return results, cfgs


def embedding_rows(results, cfgs, extra=None):
    """embed_lib.embedding_table에 차원(dim) 열을 붙인 표. 왜곡 지표(평균 상대 왜곡, 최대 반지름, 상한 점 수)를 함께 둔다(CLAUDE.md)."""
    t = el.embedding_table(results)
    t.insert(1, "dim", [cfgs[nm]["dim"] for nm in t["입력"]])
    if extra:
        for k, fn in extra.items():
            t[k] = [fn(nm) for nm in t["입력"]]
    return t


def ratio_any_convergence(result):
    """보조값: 마무리(L-BFGS-B) 수렴 여부와 관계없이 상대 stress가 가장 작은 쌍곡·유클리드 재시작의 평균 상대 왜곡 비.
    0S 규칙의 비(수렴한 재시작만 쓴다)를 낼 수 없을 때(쌍곡 재시작이 모두 maxiter에 닿음 등)를 위한 것이며 판정 규칙의 값이 아니다."""
    hyp = [r for r in result["restarts"] if r["space"] == "poincare"]
    euc = [r for r in result["restarts"] if r["space"] == "euclid"]
    bh, be = min(hyp, key=lambda r: r["stress"]), min(euc, key=lambda r: r["stress"])
    return {
        "ratio": bh["distortion"] / be["distortion"],
        "stress_ratio": bh["stress"] / be["stress"],
        "hyp_distortion": bh["distortion"],
        "euc_distortion": be["distortion"],
        "hyp_converged": bool(bh["finish_converged"]),
        "euc_converged": bool(be["finish_converged"]),
        "hyp_max_radius": bh.get("max_radius"),
    }


def load_d2_summary(s4_real_dir, name):
    """0S(D = 2) 임베딩 요약(s4_real/summary_<이름>.json)에서 표의 한 행을 만든다(재계산 없음)."""
    s = json.loads((Path(s4_real_dir) / f"summary_{name}.json").read_text(encoding="utf-8"))
    bh, be = s["hyp_best"], s["euc_best"]
    return {
        "입력": name,
        "dim": 2,
        "기록": s["record"],
        "쌍곡 이점": s["advantage"],
        "평균 상대 왜곡 쌍곡": None if bh is None else bh["distortion"],
        "평균 상대 왜곡 유클리드": None if be is None else be["distortion"],
        "왜곡 비": s["ratio"],
        "stress 비(판정 재시작)": s["stress_ratio"],
        "최대 반지름": None if bh is None else bh.get("max_radius"),
        "상한에 닿은 점 수": None if bh is None else bh.get("n_at_cap"),
        "출처": "0S(s4_real)",
    }


# ===================================================================== D. 인구이동 OD
SIDO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원", "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전특", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주",
}


def read_kosis_migration(xlsx_path):
    """KOSIS 국내인구이동 시군구 전출입자수(전출지 x 전입지) 연간 xlsx의 '계' 열을 읽는다.
    행은 (전출지, 전입지) 전체 격자(전국 1 + 시도 17 + 시군구 229 = 247 단위)이며 전출지 순서로 묶여 있다. 돌려주는 값: (이름 목록, M[전출, 전입])."""
    df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl", usecols=[0, 1, 2])
    df.columns = ["o", "d", "total"]
    o = df["o"].astype(str).str.strip().to_numpy()
    d = df["d"].astype(str).str.strip().to_numpy()
    tot = pd.to_numeric(df["total"], errors="raise").to_numpy(dtype=np.float64)
    n = int(round(math.sqrt(len(df))))
    if n * n != len(df):
        raise ValueError(f"행 수 {len(df)}가 정사각 격자가 아니다")
    O, Dm = o.reshape(n, n), d.reshape(n, n)
    if not all((O[i] == O[i, 0]).all() for i in range(n)) or not all((Dm[i] == Dm[0]).all() for i in range(n)):
        raise ValueError("전출지 x 전입지 격자 순서가 아니다")
    labels = [str(x) for x in Dm[0]]
    if [str(O[i, 0]) for i in range(n)] != labels:
        raise ValueError("전출지 순서와 전입지 순서가 다르다")
    return labels, tot.reshape(n, n)


def split_migration_units(labels):
    """행 이름에서 시군구 행을 가려낸다: 전국(첫 행)과 시도 행(뒤에 같은 이름 또는 '<이름> ...'이 오는 한 단어 행)을 뺀다.
    세종은 시도 행과 시군구 행이 둘 다 '세종'이다(뒤의 것이 시군구)."""
    n = len(labels)
    is_sgg = np.ones(n, dtype=bool)
    is_sgg[0] = False
    for i in range(1, n):
        lab = labels[i]
        if " " not in lab and i + 1 < n and (labels[i + 1] == lab or labels[i + 1].startswith(lab + " ")):
            is_sgg[i] = False
    return np.flatnonzero(is_sgg)


def migration_label_of_unit(name250):
    """250개 단위 이름('경기도 수원시 장안구')의 인구이동 단위 이름('경기 수원시')."""
    parts = name250.split()
    s = SIDO_SHORT[parts[0]]
    return s if len(parts) == 1 else f"{s} {parts[1]}"


def migration_groups(names250, sgg_labels):
    """250개 단위 -> 229개 인구이동 단위(일반구는 상위 시)의 인덱스. 모든 229 단위에 하위 지역이 있어야 한다."""
    pos = {}
    for i, lab in enumerate(sgg_labels):
        if lab in pos:
            raise ValueError(f"시군구 이름이 겹친다: {lab}")
        pos[lab] = i
    g = np.array([pos[migration_label_of_unit(nm)] for nm in names250], dtype=int)
    used = np.bincount(g, minlength=len(sgg_labels))
    if not (used > 0).all():
        raise ValueError(f"하위 지역이 없는 인구이동 단위: {[sgg_labels[i] for i in np.flatnonzero(used == 0)]}")
    return g


def prepare_migration(inp, xlsx_path, total_expected=6_117_784):
    """인구이동 OD 229개 단위 준비. d_time 229 = 250개 행렬의 하위 지역 쌍 인구 가중 평균(embed_lib.merge_weighted).
    T_dir_229(방향 통행시간)도 같은 방식이다. 전국 이동자 합(존 내 이동 포함)이 공표값과 같은지 확인한다."""
    labels, M = read_kosis_migration(xlsx_path)
    sgg = split_migration_units(labels)
    if len(sgg) != 229:
        raise ValueError(f"시군구 행 {len(sgg)}개(229개여야 한다)")
    sgg_labels = [labels[i] for i in sgg]
    T = M[np.ix_(sgg, sgg)].copy()
    checks = {
        "national_total_row": float(M[0, 0]),
        "sum_sgg_cells": float(T.sum()),
        "within_sgg_diag_sum": float(np.trace(T)),
        "between_sgg_sum": float(T.sum() - np.trace(T)),
        "dest_totals_match": bool(np.array_equal(M[0, sgg], T.sum(axis=0))),
        "origin_totals_match": bool(np.array_equal(M[sgg, 0], T.sum(axis=1))),
    }
    if checks["national_total_row"] != total_expected or checks["sum_sgg_cells"] != total_expected:
        raise ValueError(f"전국 이동자 합이 {total_expected:,}명과 다르다: {checks}")
    if not (checks["dest_totals_match"] and checks["origin_totals_match"]):
        raise ValueError(f"전국 행·열 합계가 시군구 셀 합과 다르다: {checks}")
    names250 = inp["units250"]["name"].astype(str).tolist()
    grp = migration_groups(names250, sgg_labels)
    pop = inp["pop250"]
    members = np.bincount(grp, minlength=229)
    D_time = el.merge_weighted(inp["d_time_250"], grp, pop)
    T_dir = el.merge_weighted(inp["t_dir_250"], grp, pop)
    D_time = (D_time + D_time.T) / 2.0
    isl = np.bincount(grp, weights=inp["island250"].astype(float), minlength=229) > 0
    off = ~np.eye(229, dtype=bool)
    if not ((D_time[off] > 0).all() and (T_dir[off] > 0).all() and np.isfinite(D_time).all() and np.isfinite(T_dir).all()):
        raise ValueError("d_time 229에 양수가 아닌 값이 있다")
    return {
        "labels247": labels, "sgg_idx": sgg, "labels229": sgg_labels, "T": T, "groups250": grp, "members": members,
        "merged_units": int((members > 1).sum()), "merged_subunits": int(members[members > 1].sum()),
        "d_time": D_time, "t_dir": T_dir, "island": isl, "checks": checks,
    }


# ===================================================================== E. 반경별 지역 수 증가
def ball_mean_counts(D, r_grid):
    """B(r) = 노드별 반경 r 안 노드 수(자기 자신 포함)의 평균."""
    D = np.asarray(D, dtype=np.float64)
    r = np.asarray(r_grid, dtype=np.float64)
    cnt = (D[:, :, None] <= r[None, None, :]).sum(axis=1)  # (n, len(r)): 노드마다 반경 r 안 노드 수(대각 0 포함 = 자기 자신)
    return cnt.mean(axis=0)


def _linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    sse = float(((y - pred) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return float(coef[0]), float(coef[1]), (1.0 - sse / sst) if sst > 0 else float("nan")


def growth_fit(D, n_r=100, q=(0.10, 0.90), grid="r", r2_margin=0.002):
    """거리 10–90분위 구간에서 log B ~ log r(다항식형)와 log B ~ r(지수형)의 R²를 비교한다.
    grid = 'r'이면 r을 구간에서 같은 간격으로, 'q'이면 쌍 거리의 분위수를 같은 간격으로 잡는다.
    slope_pow = 유효 차원(다항식형 기울기), rate_exp = 지수형 성장률(r 한 단위당 log B의 증가), rate_exp_diam = rate_exp x 지름.
    prefer: R² 차이가 r2_margin보다 크면 그쪽, 아니면 '구분 불가'."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    lo, hi = np.quantile(off, q)
    if grid == "r":
        r = np.linspace(lo, hi, n_r)
    elif grid == "q":
        r = np.quantile(off, np.linspace(q[0], q[1], n_r))
    else:
        raise ValueError("grid는 'r' 또는 'q'")
    B = ball_mean_counts(D, r)
    s_pow, b_pow, r2_pow = _linfit(np.log(r), np.log(B))
    s_exp, b_exp, r2_exp = _linfit(r, np.log(B))
    d = r2_pow - r2_exp
    prefer = "다항식형" if d > r2_margin else ("지수형" if d < -r2_margin else "구분 불가")
    quality = "양호" if max(r2_pow, r2_exp) >= 0.98 else "불량(둘 다 R² < 0.98)"
    return {
        "n": int(n), "r_lo": float(lo), "r_hi": float(hi), "diam": float(D.max()), "grid": grid, "n_r": int(n_r),
        "B_lo": float(B[0]), "B_hi": float(B[-1]),
        "slope_pow": s_pow, "r2_pow": r2_pow, "rate_exp": s_exp, "rate_exp_diam": s_exp * float(D.max()), "r2_exp": r2_exp,
        "r2_diff": float(d), "prefer": prefer, "quality": quality, "r": r.tolist(), "B": B.tolist(),
        "fit_pow": {"slope": s_pow, "intercept": b_pow}, "fit_exp": {"slope": s_exp, "intercept": b_exp},
    }


def geo_distance_km(x_m, y_m):
    """대표점 좌표(EPSG:5179, m)의 직선 거리 행렬(km). 참고용 평면 기준선."""
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) / 1000.0


# ===================================================================== F. 가설별 판정 규칙(탐색적)
HYPOTHESES = {
    1: "대조군 밀도 차이가 위계 판정을 좌우했다",
    2: "통근이 국지적이라 격자형 거리다",
    3: "2차원이 너무 좁다",
    4: "전국 규모(인구이동)에서는 위계나 쌍곡성이 보인다",
    5: "d_time은 평면형 성장이다",
}


def _f(x):
    return float("nan") if x is None else float(x)


def _cnt(x):
    v = _f(x)
    return "n/a" if not math.isfinite(v) else f"{v:.0f}"


def hypothesis_verdicts(R):
    """R(분석 결과 dict)에서 가설별 지지 여부를 규칙으로 정한다. 규칙은 사전 고정 규칙이 아니라 탐색적이며 각 항목의 "규칙"에 적는다.
    판정 문구: 지지 / 부분 지지 / 기각. R의 구조는 노트북 04_diagnostics.ipynb가 만든다."""
    out = []
    A, C, Dm, E = R["A"], R["C"], R["D"], R["E"]
    # (1) 대조군 밀도
    obs = A["observed"]["main|전체"]
    c0 = A["ctrl_0S"]["main|전체"]["mean"]
    a1 = A["A1"]["bars"]["main|전체"]
    a2 = A["A2"]["bars"]["main|전체"]["mean"]
    gap0, gap_a1, gap_a2 = obs - c0, obs - a1, obs - a2
    flips = bool(A["hierarchy_A1"]["hierarchy"] and A["hierarchy_A2"]["hierarchy"])
    red = 1.0 - max(gap_a1, gap_a2) / gap0 if gap0 > 0 else float("nan")
    v1 = "지지" if flips else ("부분 지지" if red >= 0.5 else "기각")
    sw = A["sweep"]
    out.append({
        "no": 1, "가설": HYPOTHESES[1], "판정": v1,
        "근거": (f"δ̄(주, 전체): 관측 {obs:.3f}, 0S 대조군 {c0:.3f}(엣지 {A['ctrl_0S']['mean_edges']:.0f}), A1 {a1:.3f}(엣지 {A['A1']['edges']}), "
                 f"A2 {a2:.3f}(엣지 {A['A2']['mean_edges']:.0f}, 표본 배율 x{A['A2']['k']:.0f}). 관측-대조군 간격 {gap0:.3f} -> A1 {gap_a1:.3f}, A2 {gap_a2:.3f}(축소 {red * 100:.0f}%). "
                 f"위계 판정 뒤집힘: A1 {A['hierarchy_A1']['hierarchy']}, A2 {A['hierarchy_A2']['hierarchy']}. 대조군 엣지 {sw[0]['mean_edges']:.0f} -> {sw[-1]['mean_edges']:.0f}에서 δ̄ {sw[0]['main|전체_mean']:.3f} -> {sw[-1]['main|전체_mean']:.3f}"),
        "규칙": "A1·A2 모두에서 4개 조건 위계 통과(판정 뒤집힘)면 지지, 뒤집히지 않고 관측-대조군 간격이 0S의 절반 이하로 줄면 부분 지지, 그 밖은 기각",
    })
    # (2) 통근이 국지적 -> 격자형
    g, gc, geo = E["d_flow_commute"], E["ctrl_commute_0S"], E["geo_247"]
    v2 = "지지" if (g["prefer"] == "다항식형" and 1.5 <= g["slope_pow"] <= 3.0) else ("기각" if g["prefer"] == "지수형" else "부분 지지")
    out.append({
        "no": 2, "가설": HYPOTHESES[2], "판정": v2,
        "근거": (f"통근 d_flow 공 성장: 다항식형 R² {g['r2_pow']:.3f}(유효 차원 {g['slope_pow']:.2f}) 대 지수형 R² {g['r2_exp']:.3f} -> {g['prefer']}. "
                 f"0S 중력 대조군 20개 평균은 R² {gc['r2_pow']:.3f}/{gc['r2_exp']:.3f}, 차원 {gc['slope_pow']:.2f}(같은 지지 집합 A1 {E['A1_commute']['slope_pow']:.2f}); 지리 직선거리 기준선 차원 {geo['slope_pow']:.2f}. "
                 f"통행량 가중 평균 통행시간 {R['meta']['mean_time_obs_min']:.1f}분. δ̄(주, 전체)는 관측 {obs:.3f}, 0S 중력 대조군 {c0:.3f}로 관측이 더 크다(덜 트리형)"),
        "규칙": "통근 d_flow의 다항식형 R²가 지수형보다 0.002 넘게 크고 유효 차원이 1.5–3이면 지지, 지수형이 더 맞으면 기각, 그 밖은 부분 지지",
    })
    # (3) 2차원이 너무 좁다
    names = ("d_time_247", "d_flow_main")
    def ratio_of(row):  # 0S 규칙의 비가 없으면(수렴한 쌍곡 재시작 없음) 수렴 무관 값을 쓰고 근거에 *를 붙인다
        v = _f(row.get("왜곡 비"))
        return (v, "") if math.isfinite(v) else (_f(row.get("왜곡 비(수렴 무관)")), "*")

    r2 = {nm: _f(C["D2"][nm]) for nm in names}
    rr_full = {(nm, D): ratio_of(C["rows"][f"{nm}_D{D}"]) for nm in names for D in (5, 10)}
    rr = {k: v[0] for k, v in rr_full.items()}
    gains = [r2[nm] - rr[(nm, D)] for (nm, D) in rr]
    if any(v <= 0.8 for v in rr.values()):
        v3 = "지지"
    elif any(g_ > 0.05 for g_ in gains):
        v3 = "부분 지지"
    else:
        v3 = "기각"
    ctrl5 = [ratio_of(C["rows"][f"ctrl_main_{r}_D5"])[0] for r in (0, 1)]
    out.append({
        "no": 3, "가설": HYPOTHESES[3], "판정": v3,
        "근거": ("왜곡 비(쌍곡/유클리드) D=2 -> 5 -> 10: "
                 + ", ".join(f"{nm} {r2[nm]:.3f} -> {rr[(nm, 5)]:.3f}{rr_full[(nm, 5)][1]} -> {rr[(nm, 10)]:.3f}{rr_full[(nm, 10)][1]}" for nm in names)
                 + f". 대조군 D=5: {ctrl5[0]:.3f}, {ctrl5[1]:.3f}. (*는 수렴한 쌍곡 재시작이 없어 수렴 무관 최소 stress 재시작으로 구한 보조값)"),
        "규칙": "D = 5 또는 10에서 d_time 또는 d_flow의 왜곡 비가 0.8 이하면 지지, 비가 2차원보다 0.05 넘게 내려가면 부분 지지, 그 밖은 기각",
    })
    # (4) 전국 규모(인구이동)
    hier = bool(Dm["hierarchy"]["hierarchy"])
    adv = any(Dm["advantage"][f"D{D}"]["advantage"] == el.ADV_YES for D in (2, 5))
    v4 = "지지" if (hier and adv) else ("부분 지지" if (hier or adv) else "기각")
    rows = Dm["rows"]
    out.append({
        "no": 4, "가설": HYPOTHESES[4], "판정": v4,
        "근거": (f"인구이동 δ̄(주, 전체) {Dm['observed']['main|전체']:.3f} 대 대조군 20개 {Dm['controls']['main|전체']['min']:.3f}–{Dm['controls']['main|전체']['max']:.3f}"
                 f"(조건 통과 {sum(1 for r_ in Dm['hierarchy']['table'] if r_['통과'])}/4, 위계 {hier}); 같은 지지 집합 대조군 {Dm['A1']['bars']['main|전체']:.3f}. "
                 f"왜곡 비 D=2 {_f(rows['mig_obs_D2']['왜곡 비']):.3f}(대조군 {_f(rows['mig_ctrl_main_0_D2']['왜곡 비']):.3f}, {_f(rows['mig_ctrl_main_1_D2']['왜곡 비']):.3f}), "
                 f"D=5 {_f(rows['mig_obs_D5']['왜곡 비']):.3f}(대조군 {_f(rows['mig_ctrl_main_0_D5']['왜곡 비']):.3f}, {_f(rows['mig_ctrl_main_1_D5']['왜곡 비']):.3f}). "
                 f"R_max 상한에 닿은 점(D=2, 229개 중) 관측 {_cnt(rows['mig_obs_D2']['상한에 닿은 점 수'])}, 대조군 {_cnt(rows['mig_ctrl_main_0_D2']['상한에 닿은 점 수'])}, {_cnt(rows['mig_ctrl_main_1_D2']['상한에 닿은 점 수'])}개. "
                 f"d_time 229 δ̄ {Dm['d_time_229_delta_bar']:.3f}"),
        "규칙": "위계(4조건 모두 대조군 20개보다 δ̄ 작음)와 쌍곡 이점(D = 2 또는 5에서 비 <= 0.8이고 대조군 2개 비보다 작음)이 둘 다면 지지, 하나면 부분 지지, 없으면 기각",
    })
    # (5) d_time 평면형 성장(기준선 = 같은 지역 대표점의 지리 직선거리)
    g, geo = E["d_time"], E["geo_247"]
    if g["prefer"] == "다항식형" and abs(g["slope_pow"] - geo["slope_pow"]) <= 0.5:
        v5 = "지지"
    elif g["prefer"] == "지수형":
        v5 = "기각"
    else:
        v5 = "부분 지지"
    out.append({
        "no": 5, "가설": HYPOTHESES[5], "판정": v5,
        "근거": (f"d_time 공 성장: 다항식형 R² {g['r2_pow']:.3f}(유효 차원 {g['slope_pow']:.2f}) 대 지수형 R² {g['r2_exp']:.3f} -> {g['prefer']}. "
                 f"지리 직선거리 기준선: R² {geo['r2_pow']:.3f}/{geo['r2_exp']:.3f}, 차원 {geo['slope_pow']:.2f}. "
                 f"10–90분위 구간의 기울기는 q90/q10 거리비({g['r_hi'] / g['r_lo']:.2f})로 정해지는 면이 커서 2보다 작다(국토가 길쭉하고 경계 효과가 있다)"),
        "규칙": "다항식형 R²가 지수형보다 0.002 넘게 크고 유효 차원이 지리 직선거리 기준선(같은 대표점)의 +-0.5 안이면 지지, 지수형이 더 맞으면 기각, 그 밖은 부분 지지",
    })
    return out


# ===================================================================== P. 그림
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]  # 고정 순서: 파랑, 주황, 청록, 보라, 분홍


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
            "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
            "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.grid": True, "axes.axisbelow": True, "lines.linewidth": 1.8, "lines.markersize": 6.5, "legend.frameon": False,
        }
    )
    return plt


def _footer(fig, extra=None):
    fig.text(0.012, 0.008, EXPLORATORY + (" | " + extra if extra else ""), ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.text(0.988, 0.008, OSM_NOTICE, ha="right", va="bottom", fontsize=7.5, color=MUTED)


def plot_A(sweep, obs, obs_edges, a1, a2, path):
    """A3 곡선: 재표본 대조군의 평균 엣지 수 대 δ̄(주·대안, 전체). 관측은 가로선, 관측 엣지 수는 세로 점선. A1·A2는 점."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, norm in zip(axes, NORMS):
        key = cond_key(norm, "전체")
        x = sweep["mean_edges"].to_numpy()
        ax.fill_between(x, sweep[f"{key}_min"], sweep[f"{key}_max"], color=SERIES[1], alpha=0.2, lw=0)
        ax.plot(x, sweep[f"{key}_mean"], "-o", color=SERIES[1])
        ax.axhline(obs[(norm, "전체")], color=SERIES[0], lw=1.8)
        ax.axvline(obs_edges, color=MUTED, lw=1.0, ls=":")
        ax.plot([obs_edges], [a1[(norm, "전체")]], "D", color=SERIES[2], ms=8, mec=SURFACE, mew=1.5)
        ax.plot([a2["mean_edges"]], [a2["bars"][key]["mean"]], "s", color=SERIES[3], ms=8, mec=SURFACE, mew=1.5)
        ax.annotate(f"관측 {obs[(norm, '전체')]:.3f}", (x[0], obs[(norm, "전체")]), textcoords="offset points", xytext=(2, 5), fontsize=9, color=INK2)
        ax.annotate(f"0S(k=1) {sweep[f'{key}_mean'].iloc[0]:.3f}", (x[0], sweep[f"{key}_mean"].iloc[0]), textcoords="offset points", xytext=(2, -14), fontsize=8.5, color=INK2)
        ax.annotate(f"A1 {a1[(norm, '전체')]:.3f}", (obs_edges, a1[(norm, "전체")]), textcoords="offset points", xytext=(-8, 10), fontsize=8.5, color=INK2, ha="right")
        ax.annotate(f"A2 {a2['bars'][key]['mean']:.3f}", (a2["mean_edges"], a2["bars"][key]["mean"]), textcoords="offset points", xytext=(-8, -16), fontsize=8.5, color=INK2, ha="right")
        ax.set_title(f"{NORM_KO[norm]} 정규화 (전체 247개)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("대조군의 평균 양방향 엣지 수 (점선 = 관측 6,815)")
        ax.set_ylim(0.25, 0.31)
    axes[0].set_ylabel("δ̄ (네 점 평균, 작을수록 트리형)")
    handles = [
        Line2D([0], [0], color=SERIES[0], lw=1.8, label="관측"),
        Line2D([0], [0], color=SERIES[1], lw=1.8, marker="o", label="재표본 대조군(표본 배율 k를 키움)"),
        Patch(facecolor=SERIES[1], alpha=0.2, label="시드 최소-최대"),
        Line2D([0], [0], color=SERIES[2], lw=0, marker="D", ms=8, label="A1 같은 지지 집합(재표본 없음)"),
        Line2D([0], [0], color=SERIES[3], lw=0, marker="s", ms=8, label=f"A2 엣지 맞춤(k = {a2['k']:.0f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle("A. 대조군 엣지 수를 관측에 맞춰도 δ̄는 관측에 가까워지지 않는다", x=0.012, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    _footer(fig, "통근·통학 2020, 247개 단위")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_B(curves, path):
    """B: 밀도(p)별 δ̄ 곡선과 최대 성분 크기. curves = {이름: DataFrame(density_curve)}."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    marks = {"관측": "o", "A1 대조군(같은 지지 집합)": "D"}
    for (name, df), c in zip(curves.items(), SERIES):
        axes[0].plot(df["p"], df["δ̄"], "-" + marks.get(name, "o"), color=c, label=name)
        axes[1].plot(df["p"], df["n(최대 성분)"], "-" + marks.get(name, "o"), color=c, label=name)
        last = df.iloc[-1]
        axes[0].annotate(f"{last['δ̄']:.3f}", (last["p"], last["δ̄"]), textcoords="offset points", xytext=(-6, 9), fontsize=8.5, color=INK2, ha="right")
    axes[0].set_xlabel("남긴 비율 p (양방향 쌍 중 ℓ_sym이 작은 순)")
    axes[0].set_ylabel("δ̄ (최대 연결 성분)")
    axes[0].set_title("밀도별 δ̄", fontsize=10.5, color=INK, loc="left")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[1].set_xlabel("남긴 비율 p")
    axes[1].set_ylabel("최대 연결 성분의 지역 수 n")
    axes[1].set_title("최대 성분 크기 (전체보다 작으면 연결이 끊김)", fontsize=10.5, color=INK, loc="left")
    from matplotlib.ticker import MaxNLocator

    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    n_all = max(int(df["n(최대 성분)"].max()) for df in curves.values())
    axes[1].set_ylim(min(int(df["n(최대 성분)"].min()) for df in curves.values()) - 2, n_all + 2)
    for ax in axes:
        ax.set_xticks([round(0.1 * i, 1) for i in range(1, 11)])
    fig.suptitle("B. 상대 흐름이 강한 쌍부터 남겨 밀도를 낮출 때 δ̄ (통근·통학 d_flow 주 정규화)", x=0.012, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _footer(fig, "통근·통학 2020, 247개 단위")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_C(rows, path):
    """C: 차원별 왜곡 비(쌍곡/유클리드)와 평균 상대 왜곡. rows = [{입력, dim, 왜곡 비, 평균 상대 왜곡 쌍곡, 평균 상대 왜곡 유클리드, (선택) ... 수렴 무관}].
    0S 규칙의 비가 없는 점(수렴한 쌍곡 재시작 없음)은 수렴 무관 값으로 대신하고 속이 빈 마름모로 표시한다."""
    plt = _plt()
    df = pd.DataFrame(rows)
    for c in ("왜곡 비", "평균 상대 왜곡 쌍곡", "평균 상대 왜곡 유클리드", "왜곡 비(수렴 무관)", "평균 상대 왜곡 쌍곡(수렴 무관)", "평균 상대 왜곡 유클리드(수렴 무관)"):
        df[c] = pd.to_numeric(df[c] if c in df else np.nan, errors="coerce")
    df["fb"] = df["왜곡 비"].isna() & df["왜곡 비(수렴 무관)"].notna()
    df["ratio_p"] = df["왜곡 비"].where(~df["fb"], df["왜곡 비(수렴 무관)"])
    df["hyp_p"] = df["평균 상대 왜곡 쌍곡"].where(~df["fb"], df["평균 상대 왜곡 쌍곡(수렴 무관)"])
    df["euc_p"] = df["평균 상대 왜곡 유클리드"].where(~df["fb"], df["평균 상대 왜곡 유클리드(수렴 무관)"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    spec = [("d_time_247", "d_time (247개)", SERIES[0], "o", "-"), ("d_flow_main", "d_flow 주 (통근, 247개)", SERIES[1], "s", "-"),
            ("ctrl_main_0", "중력 대조군 0", SERIES[2], "^", "--"), ("ctrl_main_1", "중력 대조군 1", SERIES[2], "v", ":")]
    any_fb = bool(df["fb"].any())
    for nm, lab, c, mk, ls in spec:
        d = df[df["입력"] == nm].sort_values("dim")
        axes[0].plot(d["dim"], d["ratio_p"], ls, marker=mk, color=c, label=lab)
        if nm in ("d_time_247", "d_flow_main"):
            axes[1].plot(d["dim"], d["euc_p"], "--", marker=mk, color=c, label=f"{lab} 유클리드")
            axes[1].plot(d["dim"], d["hyp_p"], "-", marker=mk, color=c, mfc=SURFACE, label=f"{lab} 쌍곡")
        for _, r in d[d["fb"]].iterrows():  # 수렴한 쌍곡 재시작이 없어 수렴 무관 값을 쓴 점
            axes[0].plot([r["dim"]], [r["ratio_p"]], "D", color=c, mfc=SURFACE, ms=9, mew=1.6)
            axes[1].plot([r["dim"]], [r["hyp_p"]], "D", color=c, mfc=SURFACE, ms=9, mew=1.6)
    axes[0].axhline(0.8, color=INK2, lw=1.0, ls=":")
    axes[0].annotate("쌍곡 이점 기준 0.8", (10, 0.8), textcoords="offset points", xytext=(0, -12), fontsize=8, color=INK2, ha="right")
    axes[0].axhline(1.0, color=GRID, lw=1.2)
    rv = df["ratio_p"]
    axes[0].set_ylim(min(0.6, float(rv.min()) - 0.03) if rv.notna().any() else 0.6, max(1.08, float(rv.max()) + 0.03) if rv.notna().any() else 1.08)
    axes[0].set_xticks([2, 5, 10])
    axes[0].set_xlabel("임베딩 차원 D")
    axes[0].set_ylabel("평균 상대 왜곡 비 (쌍곡 / 유클리드)")
    axes[0].set_title("왜곡 비 (D=2는 0S 결과)", fontsize=10.5, color=INK, loc="left")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].set_xticks([2, 5, 10])
    axes[1].set_xlabel("임베딩 차원 D")
    axes[1].set_ylabel("평균 상대 왜곡 mean|s·p - d|/d")
    axes[1].set_title("왜곡 (실선 = 쌍곡, 점선 = 유클리드)", fontsize=10.5, color=INK, loc="left")
    axes[1].legend(loc="upper right", fontsize=8, ncol=1)
    if any_fb:
        fig.text(0.012, 0.045, "속이 빈 마름모 = 마무리가 수렴한 쌍곡 재시작이 없어 수렴 무관(상대 stress 최소 재시작) 값으로 대신한 점", fontsize=8, color=INK2, ha="left")
    fig.suptitle("C. 차원 D = 2, 5, 10의 쌍곡·유클리드 왜곡 (R_max 5, 같은 재시작 구성)", x=0.012, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _footer(fig, "통근·통학 2020, 247개 단위")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_E(curves, path):
    """E: 공 성장. curves = [(이름, growth_fit 결과, 색, 참고 기준선 growth_fit 또는 None)]. 위: log-log(다항식형), 아래: log B 대 r(지수형)."""
    plt = _plt()
    k = len(curves)
    fig, axes = plt.subplots(2, k, figsize=(4.3 * k + 0.6, 7.6))
    for j, (name, g, c, ref) in enumerate(curves):
        r, B = np.asarray(g["r"]), np.asarray(g["B"])
        for row in (0, 1):
            ax = axes[row, j]
            ax.plot(r, B, "o", color=c, ms=3.8, label="관측값")
            if row == 0:
                ax.set_xscale("log")
                ax.set_yscale("log")
                fit = np.exp(g["fit_pow"]["intercept"]) * r ** g["fit_pow"]["slope"]
                ax.plot(r, fit, "-", color=INK, lw=1.4, label=f"다항식형 R² {g['r2_pow']:.3f}, 차원 {g['slope_pow']:.2f}")
                ax.set_xlabel("반경 r (로그)")
            else:
                ax.set_yscale("log")
                fit = np.exp(g["fit_exp"]["intercept"] + g["fit_exp"]["slope"] * r)
                ax.plot(r, fit, "--", color=INK, lw=1.4, label=f"지수형 R² {g['r2_exp']:.3f}, 성장률 x 지름 {g['rate_exp_diam']:.1f}")
                ax.set_xlabel("반경 r")
            if ref is not None:
                rr, BB = np.asarray(ref["r"]), np.asarray(ref["B"])
                sc = np.exp(np.log(B).mean() - np.log(BB).mean())  # 모양만 비교하려고 B 평균을 맞춘다(축척은 의미 없음)
                ax.plot(rr / rr.mean() * r.mean(), BB * sc, ":", color=MUTED, lw=1.4, label="지리 직선거리(모양 비교)")
            ax.set_ylabel("B(r) = 반경 r 안 지역 수 평균" if j == 0 else "")
            ax.legend(loc="upper left", fontsize=7.8)
            if row == 0:
                ax.set_title(f"{name}\n구간 r = {g['r_lo']:.3g}–{g['r_hi']:.3g} (거리 10–90분위), n = {g['n']}", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("E. 반경별 지역 수 증가: 다항식형(위) 대 지수형(아래)", x=0.012, y=0.995, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    _footer(fig, "d_time 247, 통근 d_flow 247, 인구이동 d_flow 229")
    fig.savefig(path, dpi=170)
    plt.close(fig)
