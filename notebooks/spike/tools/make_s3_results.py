"""S3 2부 결과 정리: 실행된 노트북의 3절 셀을 그대로 다시 실행해(결정적) 객체를 얻고, 핵심 수치를 s3_results.json 등으로 남긴다.

노트북, curvature_lib.py, 설정, 입력 파일은 바꾸지 않는다(읽기 전용). 새 파일만 만든다.
"""
import base64
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
NB_DIR = Path(r"C:/Users/mycoms/OneDrive/Desktop/Korea Feature Manifold/notebooks/spike")
os.chdir(NB_DIR)
sys.path.insert(0, str(NB_DIR))

import nbformat
import numpy as np
import pandas as pd
from scipy import stats

nb = nbformat.read(NB_DIR / "02_curvature.ipynb", as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]


def find(prefix):
    hits = [c for c in code_cells if c.source.lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def saved_stdout(cell):
    return "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream" and o.get("name") == "stdout")


G = {"__name__": "__main__"}


def run(cell, label):
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(cell.source, f"<{label}>", "exec"), G)
    return buf.getvalue(), time.time() - t0


# 설정 셀(경로, 함수, 원자료 MD5 점검, GPX 연결)은 그대로 실행한다. RUN_REAL_CURVATURE는 노트북 값(True)을 쓴다.
c1 = find("# 02_curvature 설정 셀")
_, t_cfg = run(c1, "config")
assert G["RUN_REAL_CURVATURE"] is True
LABELS = ["# 3-1a.", "# 3-1b.", "# 3-2.", "# 3-3.", "# 3-4.", "# 3-6.", "# 3-7."]
t_sec3 = 0.0
same = {}
for lab in LABELS:
    cell = find(lab)
    out, dt = run(cell, lab)
    t_sec3 += dt
    saved = saved_stdout(cell)
    same[lab] = out.strip() == saved.strip()
print("셀 출력이 노트북 저장 출력과 같은가:", same)
assert all(same.values()), same
print(f"3절 계산 시간(셀 합) {t_sec3:.1f}초, 설정 셀 {t_cfg:.1f}초")

import curvature_lib as cl  # noqa: E402

INP, E, RES, RES249, E249, DEC = G["INP"], G["E"], G["RES"], G["RES249"], G["E249"], G["DEC"]
SIDOS = G["SIDOS"]
bnd = (E["kind"] == "boundary").to_numpy()
sub = E[bnd]

# ------------------------------------------------------------------ 무결성 점검(결과를 바꾸지 않는 독립 재계산)
integ = {}
# (1) ot.emd2와 선형계획(HiGHS)으로 구한 κ가 실제 그래프 전체에서 같은가(균등 이웃 분포, d_sym 길이와 geo 길이)
u, v = E["u_idx"].to_numpy(), E["v_idx"].to_numpy()
t0 = time.time()
k_lp = cl.orc_lp(INP["n"], u, v, E["d_sym"].to_numpy(float))
k_lp_geo = cl.orc_lp(INP["n"], u, v, RES["cur"]["base"]["len_geo"])
integ["emd2_vs_linprog_kappa_max_abs_diff"] = float(np.abs(k_lp - RES["cur"]["base"]["kappa"]).max())
integ["emd2_vs_linprog_kappa_geo_max_abs_diff"] = float(np.abs(k_lp_geo - RES["cur"]["base"]["kappa_geo"]).max())
integ["linprog_check_seconds"] = round(time.time() - t0, 1)


# (2) OD 가중 ORC를 선형계획으로 독립 계산(249개 단위 그래프, d_sym 길이)
def orc_lp_weighted(n, u, v, length, w, alpha=0.5):
    dist = cl.shortest_paths(n, u, v, length)
    nbr = [[] for _ in range(n)]
    for e in range(len(u)):
        nbr[u[e]].append((int(v[e]), float(w[e])))
        nbr[v[e]].append((int(u[e]), float(w[e])))
    tot = [sum(x for _, x in l) for l in nbr]

    def meas(x, supp):
        d = {x: alpha}
        for z, wz in nbr[x]:
            d[z] = (1 - alpha) * wz / tot[x]
        return np.array([d.get(s, 0.0) for s in supp])

    out = np.empty(len(u))
    for e in range(len(u)):
        x, y = int(u[e]), int(v[e])
        supp = sorted({x, y} | {z for z, _ in nbr[x]} | {z for z, _ in nbr[y]})
        out[e] = 1 - cl.w1_lp(meas(x, supp), meas(y, supp), dist[np.ix_(supp, supp)]) / dist[x, y]
    return out


u9, v9 = E249["u_idx"].to_numpy(), E249["v_idx"].to_numpy()
w9 = E249["w_od"].to_numpy(float)
k9_lp = orc_lp_weighted(INP["n249"], u9, v9, E249["d_sym"].to_numpy(float), w9)
integ["emd2_vs_linprog_od_weighted_kappa_max_abs_diff"] = float(np.abs(k9_lp - RES249["cur"]["base"]["kappa"]).max())

# (3) 편 Spearman을 독립 공식으로(1차 편상관 공식 + 베타 함수 p)
from scipy import special  # noqa: E402

kex = RES["cur"]["base"]["kappa_ex"][bnd]
s_ = sub["s"].to_numpy(float)
r_ = sub["r"].to_numpy(float)
rxy, rxz, ryz = (stats.spearmanr(a, b).statistic for a, b in ((kex, s_), (kex, r_), (s_, r_)))
rho_f = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
n_b = len(kex)
t_f = rho_f * np.sqrt((n_b - 3) / (1 - rho_f ** 2))
p_f = 0.5 * special.betainc((n_b - 3) / 2, 0.5, (n_b - 3) / (n_b - 3 + t_f ** 2)) if t_f > 0 else 1 - 0.5 * special.betainc((n_b - 3) / 2, 0.5, (n_b - 3) / (n_b - 3 + t_f ** 2))
integ["partial_spearman_vs_formula_abs_diff"] = {"rho": float(abs(RES["battery"]["partial"]["rho"] - rho_f)), "p": float(abs(RES["battery"]["partial"]["p"] - p_f))}
# (4) 주 검정 단순 p를 scipy로 직접
integ["main_test_vs_scipy_direct_abs_diff"] = {"rho": float(abs(RES["battery"]["main"]["rho"] - stats.spearmanr(kex, s_, alternative="greater").statistic)),
                                               "p": float(abs(RES["battery"]["main"]["p"] - stats.spearmanr(kex, s_, alternative="greater").pvalue))}
# (5) κ_ex = κ - κ_geo, 대칭성 등 기본 점검
c0 = RES["cur"]["base"]
integ["kex_equals_kappa_minus_kappa_geo_max_abs"] = float(np.abs(c0["kappa_ex"] - (c0["kappa"] - c0["kappa_geo"])).max())
integ["finite_all_kappa"] = bool(all(np.isfinite(RES["cur"][k][m]).all() for k in RES["cur"] for m in ("kappa", "kappa_geo", "kappa_ex")))
integ["kappa_le_1_all"] = bool(all((RES["cur"][k]["kappa"] <= 1 + 1e-9).all() for k in RES["cur"]))
print("무결성 점검:", json.dumps(integ, ensure_ascii=False))

# ------------------------------------------------------------------ 시도별 jackknife p (두 검정, 17개)
su, sv = sub["sido_u"].to_numpy(), sub["sido_v"].to_numpy()
jk = {}
for name, fn, arrays in (("main", cl.spearman_greater, (kex, s_)), ("partial", cl.partial_spearman_greater, (kex, s_, r_))):
    pj, worst, ps, ns = cl.jackknife_max_p(fn, su, sv, SIDOS, *arrays)
    assert abs(pj - RES["battery"][name]["p_jack"]) < 1e-15 and worst == RES["battery"][name]["worst_sido"]
    jk[name] = {"max_p": pj, "worst_sido": worst, "p_by_sido": {k: float(x) for k, x in ps.items()}, "n_left_by_sido": ns}
sido_names = dict(zip(pd.read_parquet(r"C:/dev/kfm-data/spike/regions.parquet", columns=["sido_code", "sido_name"]).drop_duplicates()["sido_code"],
                      pd.read_parquet(r"C:/dev/kfm-data/spike/regions.parquet", columns=["sido_code", "sido_name"]).drop_duplicates()["sido_name"]))


# ------------------------------------------------------------------ JSON
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dist_summary(a):
    a = np.asarray(a, float)
    q = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return dict(zip(["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"], q.tolist()), mean=float(a.mean()), std=float(a.std(ddof=1)), n=int(len(a)))


def battery_dict(b):
    return {k: {kk: vv for kk, vv in d.items()} for k, d in b.items()}


t20 = RES["top20"]
pos20 = t20["edge_pos"]
top_tbl = pd.DataFrame({"rank": np.arange(1, 21), "edge_id": E["edge_id"].to_numpy()[pos20], "u_code": E["u_code"].to_numpy()[pos20], "v_code": E["v_code"].to_numpy()[pos20],
                        "u_name": E["u_name"].to_numpy()[pos20], "v_name": E["v_name"].to_numpy()[pos20], "kappa_ex": t20["kappa_ex"], "kappa": c0["kappa"][pos20],
                        "kappa_geo": c0["kappa_geo"][pos20], "s_m_per_km": E["s"].to_numpy()[pos20], "r_min_per_km": E["r"].to_numpy()[pos20], "h_e_m": E["h_e"].to_numpy()[pos20],
                        "sea_frac": E["sea_frac"].to_numpy()[pos20], "b_baekdu": E["b_baekdu"].to_numpy()[pos20], "cross_sido": E["cross_sido"].to_numpy()[pos20],
                        "island_sensitive": t20["island_sensitive"], "class": t20["label"]})
posb = RES["bottom20_kappa_pos"]
bot_tbl = pd.DataFrame({"rank": np.arange(1, 21), "edge_id": E["edge_id"].to_numpy()[posb], "u_code": E["u_code"].to_numpy()[posb], "v_code": E["v_code"].to_numpy()[posb],
                        "u_name": E["u_name"].to_numpy()[posb], "v_name": E["v_name"].to_numpy()[posb], "kappa": c0["kappa"][posb], "kappa_ex": c0["kappa_ex"][posb],
                        "kappa_geo": c0["kappa_geo"][posb]})

result = {
    "session": "S3 2부 (0S-3 실데이터 곡률, ADR 0013 1·2·5·6항)",
    "run": {"frozen_commit": "b2c7ca2", "notebook": "notebooks/spike/02_curvature.ipynb", "RUN_REAL_CURVATURE": True, "alpha": INP["freeze"]["config"]["alpha"],
            "nbconvert_wall_seconds": 107.973, "section3_compute_seconds_rerun": round(t_sec3, 1),
            "inputs": {"G_adj": {"nodes": INP["n"], "edges": len(E), "boundary_test_edges": int(bnd.sum())}, "G_adj_249": {"nodes": INP["n249"], "edges": len(E249)}},
            "note": "이 파일은 실행된 노트북의 3절 셀을 그대로 다시 실행해(결정적) 만들었고, 노트북 저장 출력과 셀별로 같음을 확인했다."},
    "decision": {"label": DEC["label"], "detail": DEC["detail"], "inputs": clean(RES["decision_inputs"]), "od_main_input": {"p": RES249["battery"]["main"]["p"], "p_jack": RES249["battery"]["main"]["p_jack"]}},
    "main_test": clean(RES["battery"]["main"]),
    "partial_test": clean(RES["battery"]["partial"]),
    "jackknife_by_sido": {"sido_names": sido_names, **clean(jk)},
    "validity": {"rho": clean(RES["validity"]["rho"]), "ok": RES["validity"]["ok"], "threshold": RES["validity"]["threshold"],
                 "kappa_vs_inv_deg_rho": RES["structure"]["kappa"]["1/deg_u+1/deg_v"]},
    "island_sensitivity": {"pass_by_setting": {k: list(v) for k, v in RES["island_pass"].items()},
                           "base": clean(RES["battery"]), **{k: clean(v) for k, v in RES["island_battery"].items()},
                           "flipped_settings": [k for k in RES["island_battery"] if tuple(RES["island_pass"][k]) != tuple(RES["island_pass"]["base"])]},
    "stage1_check": clean(RES["stage1"]),
    "rho_kappa_ex_vs_min_per_km": RES["rho_kex_r"],
    "od_weighted_249": {"main_test": clean(RES249["battery"]["main"]), "partial_test": clean(RES249["battery"]["partial"]), "validity": clean(RES249["validity"]),
                        "stage1_check": clean(RES249["stage1"]), "n_test_edges": int((E249["kind"] == "boundary").sum()),
                        "caveat": "통근·통학 흐름은 시도 경계에서 줄어 시도 경계 효과와 겹친다(시도 경계 엣지 가중치 중앙값 1,810명 vs 시도 안 4,278명)."},
    "top20_kappa_ex": {"class_counts": t20["n_each"], "n_data_problem": t20["n_data_problem"], "n_natural_barrier": t20["n_barrier"], "threshold_20th_kappa_ex": t20["threshold_20th"],
                       "coastal_ratio": t20["coastal_ratio"], "coastal_ratio_all_test_edges": t20["coastal_ratio_all"], "h_e_top10pct_hypergeom": clean(t20["h_e_hyper"]),
                       "edges": clean(top_tbl.to_dict("records"))},
    "bottom20_kappa": clean(bot_tbl.to_dict("records")),
    "auxiliary": {"sea_over_half_excluded": {"n_excluded": RES["n_sea_over_half"], **clean(battery_dict(RES["sea_excluded"]))},
                  "h_e_test": clean(battery_dict(RES["h_e"])), "barrier_baekdu_test": clean(battery_dict(RES["barrier_baekdu"])),
                  "barrier_sensitivity_line_test": clean(battery_dict(RES["barrier_sens"])), "side_edges": clean(RES["side_edges"]),
                  "structure_rho": clean(RES["structure"]),
                  "non_boundary_edge_kappa": clean(RES["non_boundary_kappa"].to_dict("records"))},
    "distribution": {sc: {m: dist_summary(RES["cur"][sc][m][bnd]) for m in ("kappa", "kappa_geo", "kappa_ex")} for sc in RES["scenarios"]},
    "median_ratio_d_sym_over_straight": c0["median_ratio"],
    "integrity_checks": integ,
}
result = clean(result)
(NB_DIR / "s3_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("저장: s3_results.json", (NB_DIR / "s3_results.json").stat().st_size, "bytes")

# ------------------------------------------------------------------ 표와 그림 파일
OUTD = NB_DIR / "s3_outputs"
(OUTD / "figures").mkdir(parents=True, exist_ok=True)
top_tbl.to_csv(OUTD / "top20_kappa_ex.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
bot_tbl.to_csv(OUTD / "bottom20_kappa.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
cols = ["edge_id", "u_code", "v_code", "u_name", "v_name", "kind"]
ed = E[cols].copy()
for sc in RES["scenarios"]:
    for m in ("kappa", "kappa_geo", "kappa_ex"):
        ed[f"{m}_{sc}"] = RES["cur"][sc][m]
ed.to_csv(OUTD / "edge_curvatures_g_adj.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
e9 = E249[cols].copy()
e9["w_od"] = E249["w_od"].to_numpy()
for m in ("kappa", "kappa_geo", "kappa_ex"):
    e9[f"{m}_od"] = RES249["cur"]["base"][m]
e9.to_csv(OUTD / "edge_curvatures_g_adj249_od.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
# 그림: 노트북 3-5 셀의 출력 그림 3장(κ, κ_ex, κ 하위 20개)을 그대로 꺼낸다
c35 = find("# 3-5.")
pngs = [o["data"]["image/png"] for o in c35.outputs if o.get("output_type") in ("display_data", "execute_result") and "image/png" in o.get("data", {})]
assert len(pngs) == 3, len(pngs)
for name, b64 in zip(("map_kappa.png", "map_kappa_ex_top20.png", "map_kappa_bottom20.png"), pngs):
    (OUTD / "figures" / name).write_bytes(base64.b64decode(b64))
print("저장:", sorted(str(p.relative_to(NB_DIR)) for p in OUTD.rglob("*") if p.is_file()))
