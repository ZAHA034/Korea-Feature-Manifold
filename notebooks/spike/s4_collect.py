# -*- coding: utf-8 -*-
"""S4 2부 후처리: 03_embedding.ipynb 3절이 남긴 산출물(C:\\dev\\kfm-data\\spike\\s4\\s4_real\\)에서
핵심 수치(s4_results.json)와 그림 4장(s4_outputs/)을 만든다.

임베딩·δ를 다시 계산하지 않는다. 저장된 결과를 읽어 모으고, 좌표의 닫힌 해 척도 s*만 저장된 좌표와 입력 거리에서 다시 구한다.
실행: python s4_collect.py [--wall-seconds N]   (kfm-spike 환경)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

S4 = Path(r"C:\dev\kfm-data\spike\s4")
REAL = S4 / "s4_real"
NB_DIR = Path(__file__).resolve().parent
RESULTS = NB_DIR / "s4_results.json"
FIGDIR = NB_DIR / "s4_outputs"

INPUTS = ["d_time_250", "d_time_247", "d_flow_main", "d_flow_alt"]
CTRL_NAMES = {"main": [f"ctrl_main_{r}" for r in range(5)], "alt": [f"ctrl_alt_{r}" for r in range(5)]}
NORM_LABEL = {"main": "주 정규화 ℓ_sym", "alt": "대안 정규화 ℓ′"}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def short_name(name: str) -> str:
    """'서울특별시 강남구' -> '서울 강남구', '경기도 수원시' -> '수원시', '경상북도 포항시 남구' -> '포항시 남구'."""
    toks = name.split()
    first = toks[0]
    if first.endswith(("특별시", "광역시", "특별자치시")):
        return first.replace("특별자치시", "").replace("특별시", "").replace("광역시", "") + " " + " ".join(toks[1:]) if len(toks) > 1 else first[:2]
    return " ".join(toks[1:]) if len(toks) > 1 else name


def load_units():
    u247 = pd.read_csv(S4 / "units_247.csv", encoding="utf-8-sig", dtype={"unit_code": str})
    u249 = pd.read_csv(S4 / "units_249.csv", encoding="utf-8-sig", dtype={"unit_code": str})
    u250 = pd.read_csv(S4 / "units_250.csv", encoding="utf-8-sig", dtype={"unit_code": str})
    pop247 = u249.set_index("unit_code").loc[u247["unit_code"], "pop_2025_12"].to_numpy(float)
    u247 = u247.assign(pop=pop247)
    return u250, u249, u247


def read_summary(name):
    return json.loads((REAL / f"summary_{name}.json").read_text(encoding="utf-8"))


def best_brief(s, key):
    b = s.get(key)
    if b is None:
        return None
    keep = ("init", "seed", "m", "attempt", "stress", "distortion", "scale", "finish_converged", "adam_converged", "n_iter_adam", "n_iter_finish", "n_at_cap", "max_radius", "max_rel_err", "finish_message")
    return {k: b.get(k) for k in keep if k in b}


def worst_summary(name, tag, labels):
    path = REAL / f"worst20_{name}_{tag}.csv"
    if not path.exists():
        return None
    w = pd.read_csv(path)
    ab = w["abs_rel_err"].to_numpy()
    ids = np.concatenate([w["i"].to_numpy(), w["j"].to_numpy()])
    counts = pd.Series(ids).value_counts()
    top_units = [{"unit": labels[int(i)], "n_pairs": int(c)} for i, c in counts.head(3).items()]
    isl = [int(i) for i in ids if labels[int(i)].endswith("울릉군")]
    return {
        "max_abs_rel_err": float(ab.max()), "median_abs_rel_err": float(np.median(ab)), "min_abs_rel_err": float(ab.min()),
        "n_underestimated": int((w["rel_err"] < 0).sum()), "n_overestimated": int((w["rel_err"] > 0).sum()),
        "n_pairs_with_ulleung": int(len(isl)), "most_frequent_units": top_units,
        "top5": [{"i": labels[int(r.i)], "j": labels[int(r.j)], "d_true": float(r.d_true), "d_emb": float(r.d_emb), "rel_err": float(r.rel_err)} for r in w.head(5).itertuples()],
    }


def build_results(wall_seconds=None):
    u250, u249, u247 = load_units()
    labels = {250: [str(x) for x in u250["name"]], 247: [str(x) for x in u247["name"]]}
    delta = pd.read_csv(REAL / "delta_table.csv", encoding="utf-8-sig")
    hier = pd.read_csv(REAL / "hierarchy_table.csv", encoding="utf-8-sig")
    emb_tab = pd.read_csv(REAL / "embedding_table.csv", encoding="utf-8-sig")
    judge = json.loads((REAL / "judge.json").read_text(encoding="utf-8"))
    runs = json.loads((REAL / "runs_record.json").read_text(encoding="utf-8"))
    ctrl = np.load(REAL / "control_delta_bars.npz")
    res = {"created_from": str(REAL), "git_head_at_run": runs["code_version"]["git_head"], "config_hash": runs["config_hash"], "embed_lib_md5": runs["code_version"]["embed_lib.py"]}

    # 임베딩 입력별 요약
    emb = {}
    for name in INPUTS + CTRL_NAMES["main"] + CTRL_NAMES["alt"]:
        s = read_summary(name)
        rt = pd.read_csv(REAL / f"restarts_{name}.csv", encoding="utf-8-sig")
        row = emb_tab[emb_tab["입력"] == name].iloc[0]
        emb[name] = {
            "n": int(row["n"]), "record": s["record"], "advantage": s["advantage"], "reason": s["reason"],
            "distortion_hyp": None if s["hyp_best"] is None else s["hyp_best"]["distortion"],
            "distortion_euc": None if s["euc_best"] is None else s["euc_best"]["distortion"],
            "ratio": s["ratio"], "stress_ratio_judged": s["stress_ratio"], "stress_ratio_all_min": s["stress_ratio_all"],
            "hyp_min_stress_all": s["hyp_min_stress_all"], "euc_min_stress_all": s["euc_min_stress_all"],
            "opt_failure": s["opt_failure"], "flat_limit": s["flat_limit"],
            "n_converged_hyp": s["n_converged_hyp"], "n_restarts_hyp": s["n_restarts_hyp"], "n_converged_euc": s["n_converged_euc"], "n_restarts_euc": s["n_restarts_euc"],
            "judged_hyp": best_brief(s, "hyp_best"), "judged_euc": best_brief(s, "euc_best"),
            "attempts": int(row["시도 수"]), "retry_occurred": bool(rt["attempt"].max() > 0),
            "restart_seconds_sum": float(row["초(재시작 합)"]),
            "all_hyp_stress": rt.loc[rt["space"] == "poincare", "stress"].round(6).tolist(),
            "all_euc_stress": rt.loc[rt["space"] == "euclid", "stress"].round(6).tolist(),
            "hyp_finish_converged_flags": rt.loc[rt["space"] == "poincare", "finish_converged"].tolist(),
            "euc_finish_converged_flags": rt.loc[rt["space"] == "euclid", "finish_converged"].tolist(),
        }
    res["embedding"] = emb

    # 판정
    adv = {nm: {k: emb[nm][k] for k in ("record", "advantage", "ratio", "reason", "opt_failure", "flat_limit")} for nm in ("d_time_250", "d_time_247")}
    for norm, nm in (("main", "d_flow_main"), ("alt", "d_flow_alt")):
        j = judge["flow_advantage"].get(norm, {})
        ratios = [emb[c]["ratio"] for c in CTRL_NAMES[norm]]
        adv[nm] = {"own_record": emb[nm]["record"], "own_advantage": emb[nm]["advantage"], "own_ratio": emb[nm]["ratio"], "vs_controls": j.get("advantage"), "reason": j.get("reason"),
                   "control_ratios": ratios, "control_records": [emb[c]["record"] for c in CTRL_NAMES[norm]],
                   "control_min_ratio": None if any(r is None for r in ratios) else float(min(ratios))}
    res["hyperbolic_advantage"] = adv
    hier_rows = hier.to_dict("records")
    res["hierarchy"] = {"overall": bool(hier["통과"].all()), "conditions": hier_rows}

    # δ 표와 대조군 δ̄
    res["delta"] = delta[["입력", "부분집합", "n", "diam", "delta", "delta_rel", "delta_bar", "n_ties", "delta_diff", "delta_quad", "n_quads"]].to_dict("records")
    cd = {}
    for key in ctrl.files:
        norm, subset = key.split("|")
        arr = ctrl[key]
        name = "d_flow_main" if norm == "main" else "d_flow_alt"
        obs = float(delta[(delta["입력"] == name) & (delta["부분집합"] == subset)]["delta_bar"].iloc[0])
        cd[key] = {"n": int(len(arr)), "min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
                   "observed": obs, "n_controls_below_observed": int((arr < obs).sum()), "n_controls_le_observed": int((arr <= obs).sum()),
                   "rank_of_observed_among_21": int(1 + (arr < obs).sum()), "observed_minus_ctrl_median": obs - float(np.median(arr)),
                   "observed_z": (obs - float(arr.mean())) / float(arr.std(ddof=1)), "all_20": arr.round(6).tolist()}
    res["control_delta_bar"] = cd

    # 최악 20쌍
    ws = {}
    for nm in INPUTS:
        n = emb[nm]["n"]
        ws[nm] = {"poincare": worst_summary(nm, "hyp", labels[n]), "euclid": worst_summary(nm, "euc", labels[n])}
    res["worst20"] = ws

    # 실행
    res["run"] = {"embedding_seconds": runs.get("embedding_seconds"), "wall_seconds_nbconvert": wall_seconds, "n_workers": runs.get("n_workers"),
                  "attempts_used": runs.get("attempts_used"),
                  "restart_seconds_sum_all_inputs": float(sum(v["restart_seconds_sum"] for v in emb.values()))}
    res["files"] = {"results_json": str(RESULTS), "figures_dir": str(FIGDIR), "outputs_dir": str(REAL), "notebook": str(NB_DIR / "03_embedding.ipynb")}
    return res


# ------------------------------------------------------------------------------------------------ 그림
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_250_700 = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # 파란색 순차 램프 250 -> 700 (palette.md)
ACCENT = "#e34948"  # 강조(섬 연결 지역 테두리 아님, 범례용 표식에는 쓰지 않는다)


def pair_scale(D, X, space):
    """닫힌 해 척도 s* = sum(p/d) / sum((p/d)^2)."""
    i, j = np.triu_indices(len(D), 1)
    diff2 = ((X[i] - X[j]) ** 2).sum(1)
    if space == "poincare":
        a = 1.0 - (X**2).sum(1)
        p = np.arccosh(1.0 + 2.0 * diff2 / (a[i] * a[j]))
    else:
        p = np.sqrt(diff2)
    q = p / D[i, j]
    return float(q.sum() / (q * q).sum())


def pca_rotate(X, about_origin=False):
    """주축이 x축이 되게 돌린다. about_origin이면 원점 둘레 회전만(푸앵카레 원판의 등장 변환, 이동 없음), 아니면 무게중심을 원점으로 옮긴 뒤 회전."""
    Xc = X if about_origin else X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt.T


def place_labels(fig, ax, xy, texts, prio_idx, all_xy, fontsize=8.5):
    """겹치지 않게 직접 라벨을 놓는다(후보 위치를 차례로 시도). 놓지 못한 라벨은 건너뛴다."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(rend)
    pts = ax.transData.transform(all_xy)
    placed = []
    out = []
    cands = []
    for r in (9, 15, 22, 32, 44):
        for ang in (35, 145, -35, -145, 0, 180, 90, -90, 70, 110, -70, -110):
            cands.append((r, np.deg2rad(ang)))
    for k in prio_idx:
        p = ax.transData.transform(xy[k])
        done = False
        for r, ang in cands:
            c = p + r * np.array([np.cos(ang), np.sin(ang)])
            ha = "left" if np.cos(ang) > 0.3 else ("right" if np.cos(ang) < -0.3 else "center")
            va = "bottom" if np.sin(ang) > 0.3 else ("top" if np.sin(ang) < -0.3 else "center")
            inv = ax.transData.inverted().transform(c)
            t = ax.text(inv[0], inv[1], texts[k], fontsize=fontsize, color=INK, ha=ha, va=va, zorder=6)
            bb = t.get_window_extent(rend).expanded(1.06, 1.15)
            ok = ax_bb.x0 + 2 <= bb.x0 and bb.x1 <= ax_bb.x1 - 2 and ax_bb.y0 + 2 <= bb.y0 and bb.y1 <= ax_bb.y1 - 2
            if ok:
                for q in placed:
                    if not (bb.x1 < q.x0 or bb.x0 > q.x1 or bb.y1 < q.y0 or bb.y0 > q.y1):
                        ok = False
                        break
            if ok:
                inside = ((pts[:, 0] > bb.x0 - 3) & (pts[:, 0] < bb.x1 + 3) & (pts[:, 1] > bb.y0 - 3) & (pts[:, 1] < bb.y1 + 3))
                inside[k] = False
                if inside.any():
                    ok = False
            if ok:
                placed.append(bb)
                if r >= 22:
                    ax.plot([xy[k][0], inv[0]], [xy[k][1], inv[1]], color=MUTED, lw=0.5, zorder=4)
                out.append(k)
                done = True
                break
            t.remove()
        if not done:
            continue
    return out


def draw_embedding(name, space, X, D, units, row, out_path, label_idx):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_250_700)
    pop = units["pop"].to_numpy(float)
    norm = Normalize(vmin=np.log10(8e3), vmax=np.log10(1.3e6))
    isl = units["island_connected"].to_numpy(bool)
    scale = pair_scale(D, X, space)
    if space == "poincare":
        P = pca_rotate(X, about_origin=True)
        assert np.linalg.norm(P, axis=1).max() <= np.tanh(2.5) + 1e-9
    else:
        P = pca_rotate(X) * scale  # 입력 거리 단위로 읽는다
    fig = plt.figure(figsize=(7.4, 8.9), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes([0.07, 0.205, 0.88, 0.675], facecolor=SURFACE)
    ax.set_aspect("equal")
    if space == "poincare":
        lim = 1.04
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        th = np.linspace(0, 2 * np.pi, 400)
        for r in (1, 2, 3, 4):
            nr = np.tanh(r / 2)
            ax.plot(nr * np.cos(th), nr * np.sin(th), color=GRID, lw=0.6, zorder=1)
            if r <= 3:
                ax.text(0.0, nr + 0.006, f"r = {r}", fontsize=6.5, color=MUTED, ha="center", va="bottom", zorder=2)
        ax.plot(np.tanh(2.5) * np.cos(th), np.tanh(2.5) * np.sin(th), color=AXIS, lw=0.8, zorder=1)
        ax.text(0.0, -np.tanh(2.5) - 0.006, "r = 5 (상한)", fontsize=6.5, color=MUTED, ha="center", va="top", zorder=2)
        ax.plot(np.cos(th), np.sin(th), color=AXIS, lw=0.8, zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.text(0.07, 0.172, "원 = 원점에서의 쌍곡 거리 r (r = 4는 선만). 좌표는 최적화 결과 그대로이고 원점 둘레로만 돌렸다(이동 없음).", fontsize=8, color=INK2, ha="left", va="center")
    else:
        span = np.abs(P).max() * 1.12
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(AXIS)
            sp.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        unit = "분" if name.startswith("d_time") else "ℓ 단위"
        fig.text(0.07, 0.172, f"축 = 입력 거리 단위({unit}), 무게중심을 원점으로 옮기고 주축을 x축에 맞췄다.", fontsize=8, color=INK2, ha="left", va="center")
    order = np.argsort(pop)  # 인구가 큰 지역이 위에 오도록 작은 것부터 그린다
    reg = order[~isl[order]]
    ax.scatter(P[reg, 0], P[reg, 1], s=34, c=np.log10(pop[reg]), cmap=cmap, norm=norm, edgecolors=SURFACE, linewidths=0.8, zorder=3)
    if isl.any():
        ax.scatter(P[isl, 0], P[isl, 1], s=60, marker="D", c=np.log10(pop[isl]), cmap=cmap, norm=norm, edgecolors=INK, linewidths=1.0, zorder=3)
    texts = [short_name(n) for n in units["name"]]
    placed = place_labels(fig, ax, P, texts, label_idx, P)
    # 제목과 부제
    sp_name = "푸앵카레 원판 (곡률 -1)" if space == "poincare" else "유클리드 평면"
    nm_h = {"d_time_247": "d_time (247개 단위, 통행시간 d_sym)", "d_flow_main": "d_flow 주 정규화 ℓ_sym (247개 단위)"}[name]
    fig.text(0.07, 0.972, f"{nm_h}: {sp_name}", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    dh, de, rt = row["평균 상대 왜곡 쌍곡"], row["평균 상대 왜곡 유클리드"], row["왜곡 비"]
    sub1 = f"평균 상대 왜곡 쌍곡 {dh:.4f}, 유클리드 {de:.4f}, 비 {rt:.4f} (판정 재시작 기준)" if pd.notna(dh) and pd.notna(de) else "판정 재시작 없음"
    cap = row["상한에 닿은 점 수"]
    sub2 = f"기록: {row['기록']} · 쌍곡 상한(R_max = 5)에 닿은 점 {int(cap)}개, 최대 반지름 {row['최대 반지름']:.3f}" if space == "poincare" else f"기록: {row['기록']} · 점 {len(units)}개"
    fig.text(0.07, 0.943, sub1, fontsize=9.5, color=INK2, ha="left", va="top")
    fig.text(0.07, 0.918, sub2, fontsize=9.5, color=INK2, ha="left", va="top")
    # 색 눈금(순차 램프)과 범례
    cax = fig.add_axes([0.12, 0.092, 0.50, 0.014])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [np.log10(v) for v in (1e4, 3e4, 1e5, 3e5, 1e6)]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["1만", "3만", "10만", "30만", "100만"])
    cb.outline.set_visible(False)
    cax.tick_params(colors=INK2, labelsize=8, length=2)
    fig.text(0.12, 0.122, "점 색 = 지역 인구 (주민등록 2025-12, 로그 눈금)", fontsize=8.5, color=INK2, ha="left", va="center")
    handles = [Line2D([], [], marker="o", ls="", markersize=6.5, markerfacecolor=BLUE_250_700[4], markeredgecolor=SURFACE, label="시군구"),
               Line2D([], [], marker="D", ls="", markersize=7, markerfacecolor=BLUE_250_700[4], markeredgecolor=INK, label="섬 연결 지역(울릉군)")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, 0.055), frameon=False, fontsize=8.5, labelcolor=INK2, handletextpad=0.4)
    fig.text(0.07, 0.022, f"라벨: 인구 상위 10곳, 울릉군, 중심에서 먼 곳(겹치면 생략). 입력 {name}.", fontsize=7.5, color=MUTED, ha="left", va="center")
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {"scale": scale, "labels_placed": [texts[k] for k in placed]}


def choose_labels(units, panels):
    """입력 하나의 두 그림에 같은 라벨 집합을 쓴다(우선순위순): 인구 상위 10곳, 울릉군, 각 패널에서 무게중심(원판은 원점)에서 먼 4곳."""
    pop = units["pop"].to_numpy(float)
    names = units["name"].tolist()
    out = [int(i) for i in np.argsort(-pop)[:10]]
    out += [i for i, n in enumerate(names) if n.endswith("울릉군")]
    for P, center in panels:
        r = np.linalg.norm(P - center, axis=1)
        out += [int(i) for i in np.argsort(-r)[:4]]
    seen, res = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            res.append(i)
    return res


def make_figures():
    FIGDIR.mkdir(exist_ok=True)
    _, _, u247 = load_units()
    emb_tab = pd.read_csv(REAL / "embedding_table.csv", encoding="utf-8-sig")
    info = {}
    for name, dfile in (("d_time_247", "d_time_247.npy"), ("d_flow_main", "d_flow_main_247.npy")):
        D = np.load(S4 / dfile)
        npz = np.load(REAL / f"coords_{name}.npz")
        row = emb_tab[emb_tab["입력"] == name].iloc[0]
        Xh = npz["X_hyp"] if "X_hyp" in npz.files else None
        Xe = npz["X_euc"] if "X_euc" in npz.files else None
        panels = []
        if Xh is not None:
            panels.append((pca_rotate(Xh, about_origin=True), np.zeros(2)))
        if Xe is not None:
            Pe = pca_rotate(Xe)
            panels.append((Pe, np.zeros(2)))
        labels = choose_labels(u247, panels)
        for space, X in (("poincare", Xh), ("euclid", Xe)):
            if X is None:
                info[f"{name}_{space}"] = "판정 재시작 없음: 그림 없음"
                continue
            path = FIGDIR / f"{name}_{'poincare' if space == 'poincare' else 'euclid'}.png"
            info[f"{name}_{space}"] = {"path": str(path), **draw_embedding(name, space, X, D, u247, row, path, labels)}
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-seconds", type=float, default=None)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    res = build_results(a.wall_seconds)
    if not a.no_figures:
        res["figures"] = make_figures()
    RESULTS.write_text(json.dumps(jsonable(res), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
