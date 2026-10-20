"""KFM 전력망 탐색 실험 계산 코드 (ADR 0014 제안, PLAN 6.8 일부를 앞당김).

탐색 코드라서 kfm 패키지·runs·스키마 규칙 밖이다(PLAN 6.1a 0S 선례). 실행 환경은 C:\\dev\\kfm-spike.
순수 함수만 둔다(입력을 제자리에서 바꾸지 않는다). 노트북 05_power_grid.ipynb가 이 모듈을 불러 쓴다.
이 모듈은 파일을 쓰지 않는다. 결과(표·그림·그래프)는 노트북이 저장소 밖 C:\\dev\\kfm-data\\power\\에만 둔다.

구성
  1. 태그 해석: 전압, 회선 수, 출력(MW), 전압 등급
  2. OSM 읽기(DuckDB ST_ReadOSM)와 기하
  3. 자료 목록
  4. 송전 그래프 조립: 구간 분할, 변전소 부착, 끝점 병합
  5. 그래프 지표: 연결 성분, 브리지(다중 엣지 안전), 유효 저항, 레버리지
  6. 시군구 귀속과 시도 집계
  7. 합성 fixture 검산
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely import STRtree

CRS_WORK = "EPSG:5179"  # 계산·저장 좌표계(CLAUDE.md 규약)
CRS_WGS = "EPSG:4326"
_TO_WORK = pyproj.Transformer.from_crs(CRS_WGS, CRS_WORK, always_xy=True)

# ADR 0014 설정. 바꾸면 결과가 달라지므로 노트북이 설정 해시를 run 기록에 남긴다
CONFIG_DEFAULT = {
    "attach_radius_m": 300.0,  # 선로 끝점을 변전소에 붙이는 거리(과제 지정값)
    "snap_m": 0.0,  # 변전소에 붙지 않은 끝점끼리 합치는 거리. 0이면 끈다(민감도에서만 켠다)
    "site_merge_m": 0.0,  # 변전소 도형을 한 부지로 합치는 거리. 0이면 겹치거나 맞닿은 것만 합친다
    "grades": ("765", "345", "154"),  # 송전 그래프에 넣는 전압 등급(154 kV 이상)
    "x_ohm_per_km": 0.35,  # 리액턴스 대용값: 가공선 일반값. 전압 등급과 무관하게 같은 값을 쓴다(대용)
    "s_base_mva": 100.0,  # 공통 기준 용량. x_pu = x_ohm_per_km * L_km * s_base / V_kV^2
    "min_length_km": 0.05,  # 구간 길이 하한(0에 가까운 구간이 전도도를 폭주시키지 않도록)
    "split_in_sites": True,  # 변전소 도형 안의 내부 꼭짓점에서도 선로를 나눈다(False면 끝점·공유 꼭짓점에서만 나눈다, 민감도)
    "include_hvdc": False,  # True면 HVDC(frequency=0) 구간도 구조 연결로 넣는다(민감도)
    "expand_circuits": False,  # True면 회선 수만큼 병렬 엣지로 펼친다(회선 단위 N-1, 민감도)
    "nearest_region_max_km": 5.0,  # 시군구 다각형 밖 노드를 가장 가까운 시군구에 붙이는 한도
}

GRADE_ORDER = ["765", "345", "154", "50-100", "<50", "HVDC", "unknown"]
INTERNAL_LINE_TYPES = ("bay", "busbar")  # power=line·cable 중 변전소 내부 모선·베이(송전 선로가 아니다)


# ===================================================================== 1. 태그 해석
_VOLT_RE = re.compile(r"^\s*(?:±|\+/-)?\s*([0-9]+(?:\.[0-9]+)?)\s*(kv|v)?\s*$", re.IGNORECASE)
_INT_RE = re.compile(r"^\s*([0-9]+)\s*$")
_OUT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(w|kw|mw|gw|tw)\s*$", re.IGNORECASE)
_OUT_UNIT_MW = {"w": 1e-6, "kw": 1e-3, "mw": 1.0, "gw": 1e3, "tw": 1e6}


def _is_missing(x) -> bool:
    return x is None or (not isinstance(x, str) and bool(pd.isna(x)))


def parse_voltage_kv(tag) -> list[float]:
    """voltage 태그를 kV 목록으로 읽는다. ';'로 나눈 값을 모두 읽고, 해석 못 하는 값은 건너뛴다.

    단위가 없으면 OSM 규약대로 V로 보되, 1000 미만이면 kV로 본다(예: "154" -> 154 kV)."""
    if _is_missing(tag):
        return []
    vals = []
    for part in str(tag).split(";"):
        m = _VOLT_RE.match(part.replace(",", ""))
        if m is None:
            continue
        x, unit = float(m.group(1)), (m.group(2) or "").lower()
        if unit == "kv":
            vals.append(x)
        elif unit == "v":
            vals.append(x / 1000.0)
        else:
            vals.append(x / 1000.0 if x >= 1000 else x)
    return vals


def voltage_grade(kv: float, hvdc: bool = False) -> str:
    """최대 전압(kV) -> 등급. 765·345·154는 오기(354, 380, 145 등)를 구간으로 흡수한다."""
    if hvdc:
        return "HVDC"
    if kv is None or not np.isfinite(kv):
        return "unknown"
    if kv >= 600:
        return "765"
    if kv >= 300:
        return "345"
    if kv >= 100:
        return "154"
    if kv >= 50:
        return "50-100"
    return "<50"


def parse_circuits(circuits, cables) -> float:
    """회선 수. circuits 태그가 정수면 그 값, 아니면 cables가 3의 배수일 때 cables / 3. 모르면 NaN."""
    for tag, div in ((circuits, 1), (cables, 3)):
        if _is_missing(tag):
            continue
        m = _INT_RE.match(str(tag))
        if m is None:
            continue
        k = int(m.group(1))
        if k > 0 and k % div == 0:
            return float(k // div)
    return float("nan")


def parse_output_mw(tag) -> float:
    """plant:output:electricity -> MW. 단위가 붙은 숫자만 읽는다. "yes"·단위 없음·해석 불가는 NaN."""
    if _is_missing(tag):
        return float("nan")
    tot, ok = 0.0, False
    for part in str(tag).split(";"):
        m = _OUT_RE.match(part.replace(",", ""))
        if m is None:
            return float("nan")
        tot += float(m.group(1)) * _OUT_UNIT_MW[m.group(2).lower()]
        ok = True
    return tot if ok else float("nan")


# ===================================================================== 2. OSM 읽기와 기하
def connect_osm() -> duckdb.DuckDBPyConnection:
    """DuckDB 연결과 spatial 확장(ST_ReadOSM). 확장은 처음 한 번만 내려받는다."""
    con = duckdb.connect()
    try:
        con.execute("LOAD spatial")
    except duckdb.Error:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
    return con


def _lonlat_to_work(lon, lat):
    x, y = _TO_WORK.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def _fetch_node_xy(con, pbf: str, node_ids) -> dict:
    """OSM 노드 id -> (x, y) (EPSG:5179, m)."""
    ids = pd.DataFrame({"node_id": np.unique(np.asarray(node_ids, dtype="int64"))})
    con.register("_nid", ids)
    try:
        df = con.execute(
            f"SELECT id AS node_id, lon, lat FROM ST_ReadOSM('{pbf}') "
            "WHERE kind = 'node' AND id IN (SELECT node_id FROM _nid)"
        ).df()
    finally:
        con.unregister("_nid")
    x, y = _lonlat_to_work(df["lon"].to_numpy(), df["lat"].to_numpy())
    return dict(zip(df["node_id"].tolist(), zip(x.tolist(), y.tolist(), strict=True), strict=True))


def _way_xy(refs, node_xy: dict):
    """way의 노드 id 목록 -> (좌표 배열 (n, 2), 노드 id 배열). 좌표가 없는 노드(추출 경계 밖)는 건너뛴다."""
    ok = [r for r in np.asarray(refs).tolist() if r in node_xy]
    return np.array([node_xy[r] for r in ok], float).reshape(-1, 2), np.array(ok, dtype="int64")


def polyline_length_km(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.hypot(*np.diff(xy, axis=0).T).sum() / 1000.0)


def read_tag_table(con, p: str, power_value: str, tag_map: dict) -> pd.DataFrame:
    """power=<값> 객체의 kind, osm_id와 지정한 태그. tag_map = {열 이름: OSM 태그 키}."""
    cols = ", ".join(f"tags['{k}'] AS {c}" for c, k in tag_map.items())
    return con.execute(
        f"SELECT kind, id AS osm_id, {cols} FROM ST_ReadOSM('{p}') "
        f"WHERE tags['power'] = '{power_value}' ORDER BY kind, id"
    ).df()


def _ring_or_line(xy: np.ndarray):
    """닫힌 way는 다각형, 아니면 선(변전소 경계가 닫히지 않은 드문 경우)."""
    if len(xy) >= 4 and np.allclose(xy[0], xy[-1]):
        poly = shapely.Polygon(xy)
        return poly if poly.is_valid else shapely.make_valid(poly)
    if len(xy) >= 2:
        return shapely.LineString(xy)
    return shapely.Point(xy[0]) if len(xy) == 1 else shapely.Point()


def _multipolygon_from_ways(parts: list):
    """관계의 outer way들을 다각형으로 묶는다. 닫히지 않으면 선들의 합으로 둔다."""
    segs = [shapely.LineString(xy) for xy in parts if len(xy) >= 2]
    if not segs:
        return shapely.Point()
    polys = list(shapely.polygonize(segs).geoms)
    return shapely.unary_union(polys) if polys else shapely.unary_union(segs)


def read_power_osm(con, pbf) -> dict:
    """PBF에서 송전 관련 객체를 읽는다. PBF 한 번 읽기가 몇 초 안에 끝난다.

    반환 dict
      lines      DataFrame     power=line·minor_line·cable 방향 (way_id, power, 태그, refs, xy, length_km)
      subs       GeoDataFrame  power=substation 노드·방향·관계 (EPSG:5179 도형)
      plants     DataFrame     power=plant (kind, osm_id, output, source)
      generators DataFrame     power=generator (kind, osm_id, output, source)
      converters DataFrame     power=converter (HVDC 변환소 확인용)
    """
    p = Path(pbf).as_posix()
    lines = con.execute(f"""
        SELECT id AS way_id, tags['power'] AS power, tags['voltage'] AS voltage, tags['circuits'] AS circuits,
               tags['cables'] AS cables, tags['frequency'] AS frequency, tags['location'] AS location,
               tags['line'] AS line_type, tags['name'] AS name, refs
        FROM ST_ReadOSM('{p}') WHERE kind = 'way' AND tags['power'] IN ('line', 'minor_line', 'cable')
        ORDER BY id""").df()
    sraw = con.execute(f"""
        SELECT kind, id AS osm_id, tags['voltage'] AS voltage, tags['substation'] AS substation,
               tags['location'] AS location, tags['frequency'] AS frequency, tags['name'] AS name,
               lon, lat, refs, ref_types, ref_roles
        FROM ST_ReadOSM('{p}') WHERE tags['power'] = 'substation' ORDER BY kind, id""").df()
    # 관계 구성원 중 outer way (ref_types 1 = way)
    member_ids = []
    for kind, refs, rtypes, roles in zip(
        sraw["kind"], sraw["refs"], sraw["ref_types"], sraw["ref_roles"], strict=True
    ):
        if kind == "relation":
            for r, t, role in zip(refs, rtypes, roles, strict=True):
                if int(t) == 1 and role in ("outer", ""):
                    member_ids.append(int(r))
    member_ways = pd.DataFrame({"way_id": pd.Series(dtype="int64"), "refs": pd.Series(dtype="object")})
    if member_ids:
        con.register("_mid", pd.DataFrame({"way_id": member_ids}))
        try:
            member_ways = con.execute(
                f"SELECT id AS way_id, refs FROM ST_ReadOSM('{p}') "
                "WHERE kind = 'way' AND id IN (SELECT way_id FROM _mid)"
            ).df()
        finally:
            con.unregister("_mid")
    # 좌표는 한 번에 읽는다
    pieces = [np.asarray(r, dtype="int64") for r in lines["refs"]]
    pieces += [np.asarray(r, dtype="int64") for r in sraw.loc[sraw["kind"] == "way", "refs"]]
    pieces += [np.asarray(r, dtype="int64") for r in member_ways["refs"]]
    node_xy = _fetch_node_xy(con, p, np.concatenate(pieces) if pieces else np.zeros(0, "int64"))

    # --- 선로
    xys, ids2, lens = [], [], []
    for refs in lines["refs"]:
        xy, ids = _way_xy(refs, node_xy)
        xys.append(xy)
        ids2.append(ids)
        lens.append(polyline_length_km(xy))
    lines = lines.assign(refs=ids2, xy=xys, length_km=lens)

    # --- 변전소 도형
    way_geom = {int(w): _way_xy(r, node_xy)[0] for w, r in zip(member_ways["way_id"], member_ways["refs"], strict=True)}
    geoms = []
    for kind, lon, lat, refs, rtypes in zip(
        sraw["kind"], sraw["lon"], sraw["lat"], sraw["refs"], sraw["ref_types"], strict=True
    ):
        if kind == "node":
            x, y = _lonlat_to_work([lon], [lat])
            geoms.append(shapely.Point(x[0], y[0]))
        elif kind == "way":
            geoms.append(_ring_or_line(_way_xy(refs, node_xy)[0]))
        else:
            parts = [way_geom[int(r)] for r, t in zip(refs, rtypes, strict=True) if int(t) == 1 and int(r) in way_geom]
            geoms.append(_multipolygon_from_ways(parts))
    subs = gpd.GeoDataFrame(
        sraw[["kind", "osm_id", "voltage", "substation", "location", "frequency", "name"]].reset_index(drop=True),
        geometry=gpd.GeoSeries(geoms, crs=CRS_WORK),
    )
    return {
        "lines": lines,
        "subs": subs,
        "plants": read_tag_table(con, p, "plant", {"output": "plant:output:electricity", "source": "plant:source"}),
        "generators": read_tag_table(
            con, p, "generator", {"output": "generator:output:electricity", "source": "generator:source"}
        ),
        "converters": con.execute(
            f"SELECT kind, id AS osm_id, tags['voltage'] AS voltage, tags['frequency'] AS frequency, "
            f"tags['converter'] AS conv FROM ST_ReadOSM('{p}') WHERE tags['power'] = 'converter'"
        ).df(),
    }


def classify_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """선로 표에 전압(kV 최댓값·값 수)·등급·회선 수·내부 모선 여부·HVDC 여부 열을 더한다(입력은 그대로 둔다)."""
    vals = [parse_voltage_kv(t) for t in lines["voltage"]]
    kv = np.array([max(v) if v else np.nan for v in vals], float)
    hvdc = np.array([(not _is_missing(f)) and str(f).strip() == "0" for f in lines["frequency"]], bool)
    return lines.assign(
        kv=kv,
        n_voltage_values=[len(v) for v in vals],
        hvdc=hvdc,
        internal=lines["line_type"].isin(INTERNAL_LINE_TYPES).to_numpy(),
        grade=[voltage_grade(k, h) for k, h in zip(kv, hvdc, strict=True)],
        circuits_n=[parse_circuits(c, k) for c, k in zip(lines["circuits"], lines["cables"], strict=True)],
        underground=(lines["location"] == "underground").to_numpy(),
    )


def classify_subs(subs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """변전소 표에 전압(kV 최댓값)·등급·변환소 여부 열을 더한다."""
    vals = [parse_voltage_kv(t) for t in subs["voltage"]]
    kv = np.array([max(v) if v else np.nan for v in vals], float)
    hvdc = np.array([(not _is_missing(f)) and str(f).strip() == "0" for f in subs["frequency"]], bool) | (
        subs["substation"] == "converter"
    ).to_numpy()
    return subs.assign(kv=kv, hvdc=hvdc, grade=[voltage_grade(k, h) for k, h in zip(kv, hvdc, strict=True)])


# ===================================================================== 3. 자료 목록
def inventory_lines(lines: pd.DataFrame) -> dict:
    """선로 자료 목록(classify_lines를 거친 표). 변전소 내부 모선·베이(line=bay·busbar)는 따로 센다."""
    real = lines.loc[~lines["internal"]]
    by_grade = (
        real.groupby(["power", "grade"])
        .agg(
            n_ways=("way_id", "size"),
            length_km=("length_km", "sum"),
            circuits_tag_fill=("circuits", lambda s: float(s.notna().mean())),
            circuits_known_fill=("circuits_n", lambda s: float(s.notna().mean())),
            underground_share=("underground", "mean"),
        )
        .reset_index()
    )
    by_grade["grade"] = pd.Categorical(by_grade["grade"], GRADE_ORDER, ordered=True)
    by_grade = by_grade.sort_values(["power", "grade"]).reset_index(drop=True)
    by_grade["grade"] = by_grade["grade"].astype(str)

    t = real.assign(
        has_voltage=real["voltage"].notna(), parsed=real["n_voltage_values"] > 0, multi=real["n_voltage_values"] > 1
    )
    agg = dict(
        n_ways=("way_id", "size"),
        length_km=("length_km", "sum"),
        voltage_tag_fill=("has_voltage", "mean"),
        voltage_parsed_fill=("parsed", "mean"),
        multi_valued_share=("multi", "mean"),
        circuits_tag_fill=("circuits", lambda s: float(s.notna().mean())),
        circuits_known_fill=("circuits_n", lambda s: float(s.notna().mean())),
    )
    fill = t.groupby("power").agg(**agg).reset_index()
    tx = t.loc[t["grade"].isin(["765", "345", "154"]) & t["power"].isin(["line", "cable"])]
    fill_tx = tx.assign(power="line+cable (765/345/154)").groupby("power").agg(**agg).reset_index()
    fill = pd.concat([fill, fill_tx], ignore_index=True)

    raw = (
        real.assign(voltage=real["voltage"].fillna("(missing)"))
        .groupby(["power", "voltage", "grade"])
        .agg(n_ways=("way_id", "size"), length_km=("length_km", "sum"))
        .reset_index()
        .sort_values(["power", "n_ways"], ascending=[True, False])
        .reset_index(drop=True)
    )
    internal = (
        lines.loc[lines["internal"]]
        .groupby(["power", "line_type"])
        .agg(n_ways=("way_id", "size"), length_km=("length_km", "sum"))
        .reset_index()
    )
    return {"by_grade": by_grade, "fill": fill, "raw_voltage": raw, "internal": internal}


def inventory_subs(subs: gpd.GeoDataFrame) -> dict:
    """변전소 자료 목록(classify_subs를 거친 표)."""
    t = pd.DataFrame(subs.drop(columns="geometry")).assign(
        has_voltage=subs["voltage"].notna().to_numpy(), has_type=subs["substation"].notna().to_numpy()
    )
    by_kind = (
        t.groupby("kind")
        .agg(n=("osm_id", "size"), voltage_tag_fill=("has_voltage", "mean"), type_tag_fill=("has_type", "mean"))
        .reset_index()
    )
    by_grade = t.groupby("grade").size().reindex(GRADE_ORDER, fill_value=0).rename("n").reset_index()
    hv = t.loc[t["grade"].isin(["765", "345", "154"])]
    by_type_hv = (
        hv.assign(substation=hv["substation"].fillna("(missing)")).groupby("substation").size().rename("n").reset_index()
    )
    return {"by_kind": by_kind, "by_grade": by_grade, "by_type_hv": by_type_hv.sort_values("n", ascending=False)}


def inventory_units(df: pd.DataFrame) -> dict:
    """발전소(power=plant)·발전기(power=generator) 목록. df는 kind, osm_id, output, source 열을 갖는다.

    출력 채움률은 두 가지로 센다. 태그가 있는 비율과, 단위가 붙은 숫자로 읽히는 비율("yes"는 숫자가 아니다)."""
    t = df.assign(
        out_mw=[parse_output_mw(x) for x in df["output"]],
        has_output_tag=df["output"].notna().to_numpy(),
        output_yes=(df["output"] == "yes").to_numpy(),
        has_source=df["source"].notna().to_numpy(),
        source_main=df["source"].fillna("(missing)").str.split(";").str[0],
    )
    t["output_numeric"] = t["out_mw"].notna()
    agg = dict(
        n=("osm_id", "size"),
        output_tag_fill=("has_output_tag", "mean"),
        output_yes_share=("output_yes", "mean"),
        output_numeric_fill=("output_numeric", "mean"),
        source_tag_fill=("has_source", "mean"),
    )
    by_kind = t.groupby("kind").agg(**agg).reset_index()
    total = t.assign(kind="all").groupby("kind").agg(**agg).reset_index()
    by_source = (
        t.groupby("source_main")
        .agg(**agg, output_mw_numeric_sum=("out_mw", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    return {"by_kind": pd.concat([by_kind, total], ignore_index=True), "by_source": by_source}


# ===================================================================== 4. 송전 그래프 조립
def select_graph_lines(lines: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """그래프에 넣을 선로: 내부 모선·베이와 minor_line 제외, line·cable 중 지정한 전압 등급(+ 선택적으로 HVDC)."""
    grades = set(cfg["grades"]) | ({"HVDC"} if cfg["include_hvdc"] else set())
    m = (~lines["internal"]) & lines["power"].isin(["line", "cable"]) & lines["grade"].isin(grades)
    return lines.loc[m].reset_index(drop=True)


def _node_use_counts(lines: pd.DataFrame) -> Counter:
    cnt: Counter = Counter()
    for refs in lines["refs"]:
        cnt.update(set(np.asarray(refs).tolist()))
    return cnt


def interior_vertices_in_sites(lines: pd.DataFrame, site_geoms) -> set:
    """끝점도 아니고 다른 선로와 공유하지도 않지만 변전소 도형 안에 있는 내부 꼭짓점의 OSM 노드 id.

    선로가 변전소를 통과하며 이어지도록 그려진 경우다. 이런 꼭짓점에서 구간을 나눈다."""
    if len(site_geoms) == 0:
        return set()
    cnt = _node_use_counts(lines)
    ids, pts = [], []
    for refs, xy in zip(lines["refs"], lines["xy"], strict=True):
        for i in range(1, len(refs) - 1):
            r = int(refs[i])
            if cnt[r] < 2:
                ids.append(r)
                pts.append(xy[i])
    if not ids:
        return set()
    ii, _ = STRtree(np.asarray(site_geoms)).query(shapely.points(np.array(pts)), predicate="intersects")
    return {ids[i] for i in np.unique(ii)}


def split_lines_at_junctions(lines: pd.DataFrame, force_nodes=()) -> pd.DataFrame:
    """선로 way를 접합 꼭짓점에서 구간으로 나눈다.

    접합 꼭짓점 = way 양 끝 + 둘 이상의 선로 way가 공유하는 내부 꼭짓점 + force_nodes.
    반환 열: way_id, seg_no, u_osm, v_osm, ux, uy, vx, vy, length_km, coords"""
    cnt = _node_use_counts(lines)
    force = {int(x) for x in force_nodes}
    rec = []
    for wid, refs, xy in zip(lines["way_id"], lines["refs"], lines["xy"], strict=True):
        n = len(refs)
        if n < 2:
            continue
        is_j = np.zeros(n, bool)
        is_j[0] = is_j[-1] = True
        for i in range(1, n - 1):
            r = int(refs[i])
            if cnt[r] >= 2 or r in force:
                is_j[i] = True
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        idx = np.flatnonzero(is_j)
        for s, (a, b) in enumerate(zip(idx[:-1], idx[1:], strict=True)):
            rec.append(
                (int(wid), s, int(refs[a]), int(refs[b]), xy[a, 0], xy[a, 1], xy[b, 0], xy[b, 1],
                 (cum[b] - cum[a]) / 1000.0, xy[a : b + 1])
            )
    cols = ["way_id", "seg_no", "u_osm", "v_osm", "ux", "uy", "vx", "vy", "length_km", "coords"]
    return pd.DataFrame(rec, columns=cols)


def attach_to_sites(points, site_geoms, radius_m: float, priority):
    """점 -> 반경 안의 변전소. priority가 낮은 후보를 먼저, 같으면 가까운 쪽. 없으면 -1.

    points, site_geoms는 shapely 도형 배열이다. 반환: (변전소 인덱스, 거리)"""
    n = len(points)
    site = np.full(n, -1, dtype=int)
    dist = np.full(n, np.nan)
    if n == 0 or len(site_geoms) == 0:
        return site, dist
    priority = np.asarray(priority)
    ii, jj = STRtree(site_geoms).query(points, predicate="dwithin", distance=radius_m)
    if len(ii) == 0:
        return site, dist
    d = shapely.distance(points[ii], site_geoms[jj])
    order = np.lexsort((d, priority[jj], ii))
    ii, jj, d = ii[order], jj[order], d[order]
    first = np.r_[True, ii[1:] != ii[:-1]]
    site[ii[first]] = jj[first]
    dist[ii[first]] = d[first]
    return site, dist


def cluster_geometries(geoms, dist_m: float) -> np.ndarray:
    """거리가 dist_m 이하인 도형끼리 같은 묶음으로. dist_m = 0이면 겹치거나 맞닿은 도형만 묶는다."""
    n = len(geoms)
    if n == 0:
        return np.zeros(0, dtype=int)
    geoms = np.asarray(geoms)
    ii, jj = STRtree(geoms).query(geoms, predicate="dwithin", distance=max(float(dist_m), 1e-6))
    keep = ii < jj
    adj = coo_matrix((np.ones(int(keep.sum())), (ii[keep], jj[keep])), shape=(n, n))
    return connected_components(adj, directed=False)[1]


def build_graph(lines: pd.DataFrame, subs: gpd.GeoDataFrame, cfg: dict | None = None) -> dict:
    """송전 그래프(노드 = 변전소 부지와 선로 끝점, 엣지 = 선로 구간)를 만든다. 입력은 바꾸지 않는다.

    규칙 (ADR 0014)
      - 선로: 내부 모선·베이와 minor_line을 빼고, 전압 등급이 cfg["grades"]인 line·cable.
      - 구간: way 양 끝과 선로끼리 공유한 내부 꼭짓점, 변전소 안에 있는 내부 꼭짓점에서 나눈다.
      - 부착: 구간 끝점이 attach_radius_m 안의 변전소(전압 100 kV 이상, 또는 전압 미상)에 있으면 그 변전소에 붙인다.
        전압이 확인된 변전소를 먼저, 같은 범주에서는 가까운 쪽을 고른다.
      - 변전소 도형이 site_merge_m 이하이면 한 부지(한 노드)로 합친다. 붙지 않은 끝점은 snap_m 이하끼리 합칠 수 있다.
      - 양 끝이 같은 노드인 구간(자기 루프)은 뺀다. 엣지가 없는 노드도 뺀다.
    반환: dict(nodes, edges, diag)"""
    cfg = {**CONFIG_DEFAULT, **(cfg or {})}
    gl = select_graph_lines(lines, cfg)
    kv = subs["kv"].to_numpy(float)
    elig = subs.loc[(kv >= 100.0) | np.isnan(kv)].reset_index(drop=True)
    geoms = elig.geometry.to_numpy()
    prio = np.isnan(elig["kv"].to_numpy(float)).astype(int)  # 0: 100 kV 이상 확인, 1: 전압 미상
    force = interior_vertices_in_sites(gl, geoms[prio == 0]) if cfg["split_in_sites"] else set()
    segs = split_lines_at_junctions(gl, force)

    # 끝점(터미널): OSM 노드 id 단위
    xy_of: dict = {}
    for a, b, x1, y1, x2, y2 in zip(
        segs["u_osm"], segs["v_osm"], segs["ux"], segs["uy"], segs["vx"], segs["vy"], strict=True
    ):
        xy_of[a] = (x1, y1)
        xy_of[b] = (x2, y2)
    t_ids = np.array(sorted(xy_of), dtype="int64")
    tx = np.array([xy_of[i] for i in t_ids], float).reshape(-1, 2)
    site_of, _ = attach_to_sites(shapely.points(tx), geoms, cfg["attach_radius_m"], prio)
    lab = cluster_geometries(geoms, cfg["site_merge_m"])
    cl_of_term = np.where(site_of >= 0, lab[np.maximum(site_of, 0)], -1) if len(lab) else np.full(len(t_ids), -1)

    # 붙지 않은 끝점은 각자 노드, snap_m > 0이면 가까운 끝점끼리 합친다
    free = np.flatnonzero(cl_of_term < 0)
    group = np.arange(len(free))
    if cfg["snap_m"] > 0 and len(free) > 1:
        pairs = cKDTree(tx[free]).query_pairs(cfg["snap_m"], output_type="ndarray")
        if len(pairs):
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(free), len(free)))
            group = connected_components(adj, directed=False)[1]
    used_cl = np.unique(cl_of_term[cl_of_term >= 0])
    n_site = len(used_cl)
    node_of_term = np.full(len(t_ids), -1, dtype=int)
    has_site = cl_of_term >= 0
    node_of_term[has_site] = np.searchsorted(used_cl, cl_of_term[has_site])
    node_of_term[free] = n_site + group
    n_raw = n_site + (int(group.max()) + 1 if len(free) else 0)

    pos = {int(t): i for i, t in enumerate(t_ids)}
    u = node_of_term[[pos[int(a)] for a in segs["u_osm"]]]
    v = node_of_term[[pos[int(b)] for b in segs["v_osm"]]]
    edges = segs.drop(columns=["u_osm", "v_osm", "ux", "uy", "vx", "vy"]).assign(u=u, v=v)
    attrs = gl[["way_id", "power", "kv", "grade", "circuits_n", "hvdc", "underground"]]
    edges = edges.merge(attrs, on="way_id", how="left")
    n_loop = int((edges["u"] == edges["v"]).sum())
    edges = edges.loc[edges["u"] != edges["v"]].reset_index(drop=True)

    # 노드 속성
    node_x = np.zeros(n_raw)
    node_y = np.zeros(n_raw)
    sub_kv = np.full(n_raw, np.nan)
    n_members = np.zeros(n_raw, dtype=int)
    osm_ref = np.full(n_raw, "", dtype=object)
    kinds = elig["kind"].to_numpy() if "kind" in elig else np.full(len(elig), "site")
    oids = elig["osm_id"].to_numpy() if "osm_id" in elig else np.arange(len(elig))
    for k, c in enumerate(used_cl):
        members = np.flatnonzero(lab == c)
        osm_ref[k] = f"{kinds[members[0]]}/{oids[members[0]]}"
        cen = shapely.centroid(geoms[members])
        node_x[k] = float(np.mean(shapely.get_x(cen)))
        node_y[k] = float(np.mean(shapely.get_y(cen)))
        mk = elig["kv"].to_numpy(float)[members]
        sub_kv[k] = np.nanmax(mk) if np.isfinite(mk).any() else np.nan
        n_members[k] = len(members)
    fr_x = np.bincount(group, weights=tx[free, 0], minlength=len(set(group.tolist())) if len(free) else 0)
    fr_y = np.bincount(group, weights=tx[free, 1], minlength=len(fr_x))
    fr_n = np.bincount(group, minlength=len(fr_x))
    if len(free):
        node_x[n_site:] = fr_x / fr_n
        node_y[n_site:] = fr_y / fr_n
        n_members[n_site:] = fr_n
        first = np.unique(group, return_index=True)[1]
        osm_ref[n_site:] = [f"node/{t_ids[free][i]}" for i in first]
    deg_all = np.bincount(np.r_[edges["u"], edges["v"]].astype(int), minlength=n_raw)
    keep = deg_all > 0
    remap = np.cumsum(keep) - 1
    nodes = pd.DataFrame(
        {
            "x": node_x, "y": node_y, "is_site": np.arange(n_raw) < n_site, "site_kv": sub_kv,
            "n_members": n_members, "osm_ref": osm_ref, "deg": deg_all,
        }
    ).loc[keep].reset_index(drop=True)
    nodes["node_type"] = np.where(nodes["is_site"], "substation", np.where(nodes["deg"] >= 2, "junction", "end"))
    edges = edges.assign(u=remap[edges["u"].to_numpy()], v=remap[edges["v"].to_numpy()])
    edges.insert(0, "edge_id", np.arange(len(edges)))

    # 리액턴스 대용값(p.u., 공통 기준 용량): x = x'·L·S_base / V^2. 전압 등급과 무관하게 x'를 같게 둔다(대용)
    length_eff = np.maximum(edges["length_km"].to_numpy(float), cfg["min_length_km"])
    edges["x_pu"] = cfg["x_ohm_per_km"] * length_eff * cfg["s_base_mva"] / edges["kv"].to_numpy(float) ** 2

    diag = {
        "n_lines_selected": int(len(gl)),
        "n_segments": int(len(segs)),
        "n_force_junction_nodes": int(len(force)),
        "n_terminals": int(len(t_ids)),
        "n_terminals_attached": int(has_site.sum()),
        "n_self_loops_dropped": n_loop,
        "n_sites_eligible": int(len(elig)),
        "n_sites_eligible_known_kv": int((prio == 0).sum()),
        "n_site_clusters_used": int(n_site),
        "n_site_clusters_known_kv": int(len(np.unique(lab[prio == 0]))) if (prio == 0).any() else 0,
        "n_site_nodes_kept": int(nodes["is_site"].sum()),
        "n_site_nodes_kept_known_kv": int((nodes["is_site"] & nodes["site_kv"].notna()).sum()),
        "n_site_nodes_unknown_kv": int((nodes["is_site"] & nodes["site_kv"].isna()).sum()),
        "n_nodes_dropped_no_edge": int((~keep).sum()),
    }
    return {"nodes": nodes, "edges": edges, "diag": diag}


# ===================================================================== 5. 그래프 지표
def _adjacency(n: int, u, v):
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    return coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))


def graph_components(n: int, u, v) -> tuple[int, np.ndarray]:
    """연결 성분 수와 노드별 성분 번호."""
    if n == 0:
        return 0, np.zeros(0, dtype=int)
    k, lab = connected_components(_adjacency(n, u, v), directed=False)
    return int(k), lab


def find_bridges(n: int, u, v) -> tuple[np.ndarray, np.ndarray]:
    """브리지(끊으면 연결 성분이 나뉘는 엣지)를 찾는다. 평행 엣지는 서로 대체 경로이므로 브리지가 아니다.

    반복형 DFS 저점(low-link) 알고리즘이다. 반환: (브리지 여부 [m], 작은 쪽 노드 수 [m]).
    작은 쪽 노드 수는 브리지를 끊을 때 갈라지는 두 부분 중 작은 쪽의 노드 수이고, 브리지가 아니면 0이다."""
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    m = len(u)
    is_bridge = np.zeros(m, dtype=bool)
    child_size = np.zeros(m, dtype=int)
    if m == 0:
        return is_bridge, child_size
    src = np.concatenate([u, v])
    dst = np.concatenate([v, u])
    eid = np.concatenate([np.arange(m), np.arange(m)])
    order = np.argsort(src, kind="stable")
    dst, eid = dst[order], eid[order]
    ptr = np.zeros(n + 1, dtype=int)
    np.add.at(ptr, src[order] + 1, 1)
    ptr = np.cumsum(ptr)
    it = ptr[:-1].copy()
    disc = np.full(n, -1, dtype=int)
    low = np.zeros(n, dtype=int)
    size = np.ones(n, dtype=int)
    parent = np.full(n, -1, dtype=int)
    parent_edge = np.full(n, -1, dtype=int)
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        stack = [root]
        while stack:
            x = stack[-1]
            if it[x] < ptr[x + 1]:
                k = it[x]
                it[x] += 1
                y, e = int(dst[k]), int(eid[k])
                if e == parent_edge[x]:
                    continue  # 들어온 엣지로 되돌아가지 않는다(평행 엣지는 다른 id라서 건너뛰지 않는다)
                if disc[y] == -1:
                    parent[y], parent_edge[y] = x, e
                    disc[y] = low[y] = t
                    t += 1
                    stack.append(y)
                else:
                    low[x] = min(low[x], disc[y])
            else:
                stack.pop()
                if stack:
                    p = int(parent[x])
                    low[p] = min(low[p], low[x])
                    size[p] += size[x]
                    if low[x] > disc[p]:
                        is_bridge[parent_edge[x]] = True
                        child_size[parent_edge[x]] = size[x]
    _, lab = graph_components(n, u, v)
    comp_size = np.bincount(lab, minlength=int(lab.max()) + 1)
    small = np.where(is_bridge, np.minimum(child_size, comp_size[lab[u]] - child_size), 0)
    return is_bridge, small


def bridges_circuit_level(n: int, u, v, circuits_n):
    """회선 단위 N-1: 회선 수가 k인 구간을 k개 평행 엣지로 펼친 뒤 브리지를 찾는다(회선 수를 모르면 1).

    반환은 원래 엣지 순서의 (브리지 여부, 작은 쪽 노드 수)다. 회선이 둘 이상이면 그 구간은 브리지가 아니다."""
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    c = np.asarray(circuits_n, dtype=float)
    k = np.where(np.isfinite(c) & (c >= 1), c, 1).astype(int)
    rep = np.repeat(np.arange(len(u)), k)
    br, small = find_bridges(n, u[rep], v[rep])
    first = np.r_[0, np.cumsum(k)[:-1]] if len(k) else np.zeros(0, dtype=int)
    return br[first], small[first]


def effective_resistance(n: int, u, v, c) -> dict:
    """연결 성분마다 L^+ (L = 전도도 c로 가중한 라플라시안)로 유효 저항을 구한다. 저항은 1/c의 단위다.

    반환 dict
      r_edge[m]     엣지 양 끝 사이 유효 저항 R_eff(u, v)
      leverage[m]   c_e · R_eff(e). 1이면 브리지, 합은 n - (성분 수)  (Foster 정리)
      rbar[n]       같은 성분의 다른 노드까지 유효 저항의 평균(성분이 노드 1개면 NaN)
      kirchhoff[k]  성분별 Σ_{i<j} R_ij
      comp[n], comp_size[k]
    L^+ = (L + J/N)^-1 - J/N (J = 1행렬, N = 성분 노드 수)로 구한다. 자기 루프는 무시한다."""
    u = np.asarray(u, dtype=int)
    v = np.asarray(v, dtype=int)
    c = np.asarray(c, dtype=float)
    k, lab = graph_components(n, u, v)
    comp_size = np.bincount(lab, minlength=k) if n else np.zeros(0, dtype=int)
    r_edge = np.zeros(len(u))
    rbar = np.full(n, np.nan)
    kirch = np.zeros(k)
    elab = lab[u] if len(u) else np.zeros(0, dtype=int)
    for comp in range(k):
        idx = np.flatnonzero(lab == comp)
        nn = len(idx)
        if nn < 2:
            continue
        pos = np.full(n, -1, dtype=int)
        pos[idx] = np.arange(nn)
        em = np.flatnonzero((elab == comp) & (u != v))
        lu, lv, lc = pos[u[em]], pos[v[em]], c[em]
        lap = np.zeros((nn, nn))
        np.add.at(lap, (lu, lu), lc)
        np.add.at(lap, (lv, lv), lc)
        np.add.at(lap, (lu, lv), -lc)
        np.add.at(lap, (lv, lu), -lc)
        lp = np.linalg.inv(lap + 1.0 / nn) - 1.0 / nn
        dg = np.diag(lp)
        r_edge[em] = dg[lu] + dg[lv] - 2.0 * lp[lu, lv]
        tr = float(dg.sum())
        rbar[idx] = (nn * dg + tr) / (nn - 1)  # 행합이 0이므로 Σ_j R_ij = N·Lp_ii + tr(Lp)
        kirch[comp] = nn * tr
    leverage = c * r_edge
    leverage[u == v] = 0.0
    return {"r_edge": r_edge, "leverage": leverage, "rbar": rbar, "kirchhoff": kirch, "comp": lab, "comp_size": comp_size}


def analyze_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """연결 성분, 브리지(구간 단위와 회선 단위), 유효 저항을 계산해 노드·엣지 표에 열을 더하고 요약 수치를 만든다.

    요약 수치는 모두 집계값이다(개별 선로·변전소를 가리키지 않는다)."""
    n, m = len(nodes), len(edges)
    u = edges["u"].to_numpy(int)
    v = edges["v"].to_numpy(int)
    length = edges["length_km"].to_numpy(float)
    k, lab = graph_components(n, u, v)
    sizes = np.bincount(lab, minlength=k)
    lc = int(np.argmax(sizes))
    br, small = find_bridges(n, u, v)
    br_c, small_c = bridges_circuit_level(n, u, v, edges["circuits_n"].to_numpy(float))
    deg = np.bincount(np.r_[u, v], minlength=n)
    pend = (deg[u] == 1) | (deg[v] == 1)
    nonpend_br = br & ~pend & (small >= 2)
    er = effective_resistance(n, u, v, 1.0 / edges["x_pu"].to_numpy(float))

    def touch(flag):
        return np.bincount(np.r_[u[flag], v[flag]], minlength=n) > 0

    lev_sum = np.bincount(np.r_[u, v], weights=np.r_[er["leverage"], er["leverage"]], minlength=n)
    nodes2 = nodes.assign(
        comp=lab,
        comp_size=sizes[lab],
        in_lc=lab == lc,
        touches_bridge=touch(br),
        touches_bridge_nonpendant=touch(nonpend_br),
        touches_bridge_circuit=touch(br_c),
        rbar=er["rbar"],
        lev_mean=np.divide(lev_sum, deg, out=np.full(n, np.nan), where=deg > 0),
    )
    edges2 = edges.assign(
        comp=lab[u], in_lc=lab[u] == lc, is_bridge=br, bridge_small_side=small, bridge_nonpendant=nonpend_br,
        is_bridge_circuit=br_c, pendant=pend, r_edge=er["r_edge"], leverage=er["leverage"],
    )
    in_lc_n = lab == lc
    in_lc_e = lab[u] == lc
    srt = np.sort(sizes)[::-1]
    summary = {
        "n_nodes": n,
        "n_nodes_substation": int(nodes["is_site"].sum()),
        "n_nodes_junction": int((nodes["node_type"] == "junction").sum()),
        "n_nodes_end": int((nodes["node_type"] == "end").sum()),
        "n_edges": m,
        "length_km_total": float(length.sum()),
        "n_components": k,
        "comp_sizes_top5": [int(x) for x in srt[:5]],
        "n_components_size_le3": int((sizes <= 3).sum()),
        "lc_nodes": int(in_lc_n.sum()),
        "lc_node_share": float(in_lc_n.mean()),
        "lc_edges": int(in_lc_e.sum()),
        "lc_length_share": float(length[in_lc_e].sum() / length.sum()),
        "n_bridges": int(br.sum()),
        "bridge_share": float(br.mean()),
        "bridge_length_share": float(length[br].sum() / length.sum()),
        "n_bridges_lc": int((br & in_lc_e).sum()),
        "bridge_share_lc": float((br & in_lc_e).sum() / max(int(in_lc_e.sum()), 1)),
        "n_pendant_edges": int(pend.sum()),
        "n_bridges_nonpendant": int(nonpend_br.sum()),
        "n_bridges_circuit_level": int(br_c.sum()),
        "bridge_share_circuit_level": float(br_c.mean()),
        "nodes_touching_bridge_share": float(touch(br).mean()),
        "nodes_touching_bridge_share_lc": float(touch(br)[in_lc_n].mean()),
        "nodes_touching_nonpendant_bridge_share_lc": float(touch(nonpend_br)[in_lc_n].mean()),
        "bridge_small_side_median": float(np.median(small[br])) if br.any() else float("nan"),
        "bridge_small_side_p90": float(np.percentile(small[br], 90)) if br.any() else float("nan"),
        "rbar_mean_lc": float(np.nanmean(er["rbar"][in_lc_n])),
        "rbar_median_lc": float(np.nanmedian(er["rbar"][in_lc_n])),
        "kirchhoff_lc": float(er["kirchhoff"][lc]),
        "leverage_mean": float(er["leverage"].mean()),
        "foster_residual": float(abs(er["leverage"].sum() - (n - k))),
    }
    return {"nodes": nodes2, "edges": edges2, "summary": summary}


def subgraph_edge_components(nodes: pd.DataFrame, edges: pd.DataFrame, mask) -> dict:
    """엣지 부분집합이 만드는 부분 그래프의 연결 성분. 성분 번호는 길이가 큰 순서(0이 가장 큼)다.

    반환 dict: edge_comp[m](마스크 밖은 -1), comp_length_km[k](내림차순), node_ids(부분 그래프의 노드), node_comp"""
    mask = np.asarray(mask, dtype=bool)
    sub = edges.loc[mask]
    edge_comp = np.full(len(edges), -1, dtype=int)
    if sub.empty:
        return {"edge_comp": edge_comp, "comp_length_km": np.zeros(0), "node_ids": np.zeros(0, dtype=int), "node_comp": np.zeros(0, dtype=int)}
    node_ids = np.unique(np.r_[sub["u"], sub["v"]])
    pos = np.full(len(nodes), -1, dtype=int)
    pos[node_ids] = np.arange(len(node_ids))
    k, lab = graph_components(len(node_ids), pos[sub["u"].to_numpy()], pos[sub["v"].to_numpy()])
    raw_len = np.bincount(lab[pos[sub["u"].to_numpy()]], weights=sub["length_km"].to_numpy(float), minlength=k)
    rank = np.empty(k, dtype=int)
    rank[np.argsort(-raw_len, kind="stable")] = np.arange(k)
    edge_comp[np.flatnonzero(mask)] = rank[lab[pos[sub["u"].to_numpy()]]]
    return {"edge_comp": edge_comp, "comp_length_km": np.sort(raw_len)[::-1], "node_ids": node_ids, "node_comp": rank[lab]}


def subgraph_summary(nodes: pd.DataFrame, edges: pd.DataFrame, mask) -> dict:
    """엣지 부분집합(예: 765 kV)의 집계: 전압 등급별 골격이 하나로 이어지는지 본다.

    반환 집계: 엣지 수, 길이, 성분 수, 가장 큰 성분의 길이 비율, 성분별 길이(내림차순, km), 가장 큰 두 성분 사이의 최단 노드 간 거리(km),
    다른 엣지와 노드를 공유하는 노드 비율, 이 부분집합 길이 중 전체 그래프의 최대 성분에 든 비율."""
    mask = np.asarray(mask, dtype=bool)
    sub = edges.loc[mask]
    if sub.empty:
        return {"n_edges": 0}
    comp = subgraph_edge_components(nodes, edges, mask)
    ids = comp["node_ids"]
    others = edges.loc[~mask]
    other_nodes = np.unique(np.r_[others["u"], others["v"]]) if len(others) else np.zeros(0, dtype=int)
    cl = comp["comp_length_km"]
    gap = float("nan")
    if len(cl) >= 2:
        xy = nodes[["x", "y"]].to_numpy(float)[ids]
        a, b = xy[comp["node_comp"] == 0], xy[comp["node_comp"] == 1]
        gap = float(cKDTree(b).query(a)[0].min() / 1000.0)
    return {
        "n_edges": int(len(sub)),
        "n_nodes": int(len(ids)),
        "length_km": float(sub["length_km"].sum()),
        "n_components": int(len(cl)),
        "lc_length_share": float(cl[0] / cl.sum()),
        "comp_length_km_desc": [round(float(x), 1) for x in cl],
        "gap_km_between_top2": gap,
        "share_nodes_shared_with_other_edges": float(np.isin(ids, other_nodes).mean()),
        "share_length_in_graph_lc": float(sub.loc[sub["in_lc"], "length_km"].sum() / sub["length_km"].sum())
        if "in_lc" in sub
        else float("nan"),
    }


# ===================================================================== 6. 시군구 귀속과 시도 집계
def assign_regions(xy: np.ndarray, regions: gpd.GeoDataFrame, max_nearest_km: float) -> pd.DataFrame:
    """점을 시군구에 귀속한다(located_in). 다각형 안이면 'within', 밖이면 한도 안의 가장 가까운 시군구 'nearest', 없으면 'none'.

    regions는 EPSG:5179의 idx, name, unit_code, sido_code, sido_name 열을 갖는다."""
    cols = ["idx", "name", "unit_code", "sido_code", "sido_name"]
    pts = gpd.GeoDataFrame(
        {"pid": np.arange(len(xy))}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=regions.crs
    )
    reg = regions[[*cols, "geometry"]]
    j = gpd.sjoin(pts, reg, how="left", predicate="within")
    j = j.loc[~j.index.duplicated(keep="first")].sort_index()
    out = pd.DataFrame(
        {
            "region_idx": j["idx"].astype("Int64"),
            "sigungu": j["name"].astype(object),
            "unit_code": j["unit_code"].astype(object),
            "sido_code": j["sido_code"].astype(object),
            "sido_name": j["sido_name"].astype(object),
            "located_in_method": np.where(j["idx"].notna(), "within", "none"),
            "dist_m": np.where(j["idx"].notna(), 0.0, np.nan),
        }
    )
    miss = out["region_idx"].isna().to_numpy()
    if miss.any():
        jn = gpd.sjoin_nearest(
            pts.loc[miss], reg, how="left", max_distance=max_nearest_km * 1000.0, distance_col="d"
        )
        jn = jn.loc[~jn.index.duplicated(keep="first")]
        jn = jn.loc[jn["idx"].notna()]
        out.loc[jn.index, "region_idx"] = jn["idx"].astype("Int64")
        out.loc[jn.index, "sigungu"] = jn["name"].astype(object)
        out.loc[jn.index, "unit_code"] = jn["unit_code"].astype(object)
        out.loc[jn.index, "sido_code"] = jn["sido_code"].astype(object)
        out.loc[jn.index, "sido_name"] = jn["sido_name"].astype(object)
        out.loc[jn.index, "located_in_method"] = "nearest"
        out.loc[jn.index, "dist_m"] = jn["d"]
    return out


def aggregate_by_sido(nodes: pd.DataFrame) -> pd.DataFrame:
    """시도별 집계. nodes는 analyze_graph의 노드 표에 sido_name 열을 더한 것이다.

    열: 노드 수, 변전소 노드 수, 최대 성분 노드 수, 브리지에 닿는 노드 비율(전체, 최대 성분, 최대 성분에서 말단이 아닌 브리지만, 회선 단위),
    최대 성분 노드의 평균 유효 저항(rbar)과 평균 레버리지, 자기 성분 기준 평균 rbar. 마지막 행은 전국이다."""
    t = nodes.assign(sido_name=nodes["sido_name"].fillna("(미귀속)"))

    def one(g: pd.DataFrame) -> pd.Series:
        lc = g.loc[g["in_lc"]]
        return pd.Series(
            {
                "n_nodes": len(g),
                "n_substation_nodes": int(g["is_site"].sum()),
                "n_nodes_lc": len(lc),
                "lc_share": len(lc) / len(g),
                "share_touch_bridge_all": float(g["touches_bridge"].mean()),
                "share_touch_bridge_lc": float(lc["touches_bridge"].mean()) if len(lc) else np.nan,
                "share_touch_nonpendant_bridge_lc": float(lc["touches_bridge_nonpendant"].mean()) if len(lc) else np.nan,
                "share_touch_bridge_circuit_lc": float(lc["touches_bridge_circuit"].mean()) if len(lc) else np.nan,
                "rbar_mean_lc": float(lc["rbar"].mean()) if len(lc) else np.nan,
                "rbar_mean_own_comp": float(g["rbar"].mean()),  # 자기 성분 안에서 잰 값(최대 성분 밖의 제주 등은 이 열만 값이 있다)
                "lev_mean_lc": float(lc["lev_mean"].mean()) if len(lc) else np.nan,
            }
        )

    by = pd.DataFrame({name: one(g) for name, g in t.groupby("sido_name")}).T
    by.index.name = "sido_name"
    by = by.sort_values("n_nodes", ascending=False)
    tot = one(t).to_frame("전국").T
    tot.index.name = "sido_name"
    out = pd.concat([by, tot]).reset_index()
    for c in ("n_nodes", "n_substation_nodes", "n_nodes_lc"):
        out[c] = out[c].astype(int)
    return out


# ===================================================================== 5b. 직렬 노드 합침 (민감도용 단순화)
def contract_series_nodes(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """차수 2인 비변전소 노드를 직렬로 합친다. 입력은 바꾸지 않는다.

    OSM이 한 선로를 여러 way로 나눠 그리면 통과 노드와 짧은 엣지가 생긴다. 합치면 길이와 리액턴스 대용값은 더해지고,
    회선 수는 작은 쪽을 따른다(하나라도 모르면 모름). 두 엣지가 같은 이웃으로 이어진 경우는 합치지 않는다.
    반환 dict(nodes, edges, n_contracted)"""
    n, m = len(nodes), len(edges)
    is_site = nodes["is_site"].to_numpy(bool)
    u = edges["u"].to_numpy(int).tolist()
    v = edges["v"].to_numpy(int).tolist()
    length = edges["length_km"].to_numpy(float).tolist()
    xp = edges["x_pu"].to_numpy(float).tolist()
    circ = edges["circuits_n"].to_numpy(float).tolist()
    attr = edges[["way_id", "power", "kv", "grade", "hvdc", "underground"]].to_dict("records")
    inc = [set() for _ in range(n)]
    for e in range(m):
        inc[u[e]].add(e)
        inc[v[e]].add(e)
    alive = [True] * m
    removed = np.zeros(n, dtype=bool)
    for x in range(n):
        if is_site[x] or len(inc[x]) != 2:
            continue
        e1, e2 = sorted(inc[x])
        a = u[e1] if v[e1] == x else v[e1]
        b = u[e2] if v[e2] == x else v[e2]
        if a == b:
            continue
        ne = len(u)
        u.append(a)
        v.append(b)
        length.append(length[e1] + length[e2])
        xp.append(xp[e1] + xp[e2])
        circ.append(float("nan") if (np.isnan(circ[e1]) or np.isnan(circ[e2])) else min(circ[e1], circ[e2]))
        rec = dict(attr[e1] if length[e1] >= length[e2] else attr[e2])
        rec["hvdc"] = bool(attr[e1]["hvdc"] or attr[e2]["hvdc"])
        attr.append(rec)
        alive.append(True)
        alive[e1] = alive[e2] = False
        inc[a].discard(e1)
        inc[b].discard(e2)
        inc[a].add(ne)
        inc[b].add(ne)
        inc[x].clear()
        removed[x] = True
    keep_e = [e for e in range(len(u)) if alive[e]]
    remap = np.cumsum(~removed) - 1
    edges_c = pd.concat(
        [
            pd.DataFrame([attr[e] for e in keep_e]),
            pd.DataFrame(
                {
                    "u": remap[[u[e] for e in keep_e]],
                    "v": remap[[v[e] for e in keep_e]],
                    "length_km": [length[e] for e in keep_e],
                    "x_pu": [xp[e] for e in keep_e],
                    "circuits_n": [circ[e] for e in keep_e],
                }
            ),
        ],
        axis=1,
    )
    edges_c.insert(0, "edge_id", np.arange(len(edges_c)))
    nodes_c = nodes.loc[~removed].reset_index(drop=True)
    deg = np.bincount(np.r_[edges_c["u"], edges_c["v"]].astype(int), minlength=len(nodes_c))
    nodes_c = nodes_c.assign(deg=deg)
    nodes_c["node_type"] = np.where(nodes_c["is_site"], "substation", np.where(deg >= 2, "junction", "end"))
    return {"nodes": nodes_c, "edges": edges_c, "n_contracted": int(removed.sum())}


# ===================================================================== 7. 합성 fixture 검산
def graph_path(n: int):
    u = np.arange(n - 1)
    return n, u, u + 1


def graph_cycle(n: int):
    u = np.arange(n)
    return n, u, (u + 1) % n


def graph_complete(n: int):
    iu, ju = np.triu_indices(n, 1)
    return n, iu, ju


def graph_barbell():
    """K5 두 개(0-4, 5-9)와 다리 엣지 (0, 5). 다리는 마지막 엣지다."""
    _, u1, v1 = graph_complete(5)
    return 10, np.r_[u1, u1 + 5, 0], np.r_[v1, v1 + 5, 5]


def graph_tree(depth: int):
    n = 2 ** (depth + 1) - 1
    v = np.arange(1, n)
    return n, (v - 1) // 2, v


def graph_two_components():
    """K3 {0,1,2}와 경로 {3,4,5}."""
    return 6, np.array([0, 1, 0, 3, 4]), np.array([1, 2, 2, 4, 5])


def _brute_bridges(n, u, v):
    k0 = graph_components(n, u, v)[0]
    out = np.zeros(len(u), dtype=bool)
    for e in range(len(u)):
        keep = np.arange(len(u)) != e
        out[e] = graph_components(n, u[keep], v[keep])[0] > k0
    return out


def _random_multigraph(rng):
    n = int(rng.integers(4, 13))
    m = int(rng.integers(0, 2 * n))
    u, v = rng.integers(0, n, m), rng.integers(0, n, m)
    ok = u != v
    return n, u[ok], v[ok]


def _laplacian_pinv(n, u, v, c):
    lap = np.zeros((n, n))
    np.add.at(lap, (u, u), c)
    np.add.at(lap, (v, v), c)
    np.add.at(lap, (u, v), -c)
    np.add.at(lap, (v, u), -c)
    return np.linalg.pinv(lap, hermitian=True)


def _toy_line(way_id, refs, coords, kv, power="line", internal=False, hvdc=False, circuits=np.nan):
    xy = np.array(coords, float)
    return {
        "way_id": way_id, "power": power, "kv": kv, "grade": voltage_grade(kv, hvdc), "internal": internal,
        "hvdc": hvdc, "circuits_n": circuits, "underground": False, "refs": np.array(refs, dtype="int64"),
        "xy": xy, "length_km": polyline_length_km(xy),
    }


def synthetic_map():
    """build_graph 검산용 장난감 지도(좌표 m). 기대값은 run_synthetic_checks에 적어 두었다."""
    box = shapely.box
    subs = gpd.GeoDataFrame(
        {"kv": [345.0, 345.0, 154.0, np.nan, 154.0, 22.9, 154.0]},
        geometry=[
            box(0, 0, 100, 100), box(10000, 0, 10100, 100), box(5000, 8000, 5100, 8100), box(20000, 0, 20100, 100),
            shapely.Point(30000, 0), shapely.Point(40000, 0), box(60000, 0, 60100, 100),
        ],
        crs=CRS_WORK,
    )
    rows = [
        _toy_line(1, [1, 2, 3], [(350, 50), (5000, 50), (9800, 50)], 345.0),  # A - T - B (T는 L2와 공유한 접합점)
        _toy_line(2, [2, 4], [(5000, 50), (5050, 7900)], 154.0),  # T - C
        _toy_line(3, [5, 6], [(10200, 50), (19900, 50)], 154.0),  # B - D(전압 미상 변전소)
        _toy_line(4, [7, 8], [(10200, 60), (19900, 60)], 154.0),  # B - D 평행
        _toy_line(5, [9, 10], [(20350, 50), (30100, 0)], 154.0),  # D - E
        _toy_line(6, [12, 13], [(30200, 0), (39800, 0)], 154.0),  # E - 끝(근처 F는 22.9 kV라 대상이 아니다)
        _toy_line(7, [14, 15], [(10, 10), (90, 90)], 345.0, internal=True),  # 변전소 내부 베이(제외)
        _toy_line(8, [16, 17], [(150, 150), (150, -150)], 345.0),  # A 안의 짧은 구간(자기 루프, 제외)
        _toy_line(9, [19, 20, 21], [(55000, 50), (60050, 50), (65000, 50)], 154.0),  # G를 통과(내부 꼭짓점이 G 안)
        _toy_line(10, [22, 23], [(70000, 0), (75000, 0)], 154.0),  # P0 - P1
        _toy_line(11, [24, 25], [(75060, 0), (80000, 0)], 154.0),  # Q0 - Q1 (P1과 60 m 떨어짐)
        _toy_line(12, [26, 27], [(90000, 0), (95000, 0)], 250.0, power="cable", hvdc=True),  # HVDC 케이블
        _toy_line(13, [28, 29], [(100000, 0), (101000, 0)], 22.9, power="minor_line"),  # 배전선(제외)
        _toy_line(14, [30, 31], [(110000, 0), (111000, 0)], np.nan),  # 전압 미상(제외)
    ]
    return pd.DataFrame(rows), subs


def run_synthetic_checks(seed: int = 0, n_random: int = 300) -> pd.DataFrame:
    """합성 그래프와 장난감 지도로 이 모듈의 함수를 검산한다. 하나라도 어긋나면 AssertionError로 멈춘다.

    반환: 검산 항목, 값, 기대, 결과를 담은 표(항목마다 한 줄)."""
    rows = []

    def chk(name, got, expect, tol=1e-9):
        err = float(np.max(np.abs(np.asarray(got, float) - np.asarray(expect, float)))) if np.size(got) else 0.0
        assert err <= tol, f"{name}: 최대 오차 {err:.3e} (허용 {tol:g})"
        rows.append({"항목": name, "오차": err, "허용": tol, "결과": "OK"})

    def eq(name, got, expect):
        assert got == expect, f"{name}: {got!r} != {expect!r}"
        rows.append({"항목": name, "오차": 0.0, "허용": 0.0, "결과": "OK"})

    # --- 태그 해석
    eq("voltage 154000", parse_voltage_kv("154000"), [154.0])
    eq("voltage 다중 값", parse_voltage_kv("345000;154000"), [345.0, 154.0])
    eq("voltage 단위 없는 kV", parse_voltage_kv("154"), [154.0])
    eq("voltage 22900", parse_voltage_kv("22900"), [22.9])
    eq("voltage 'kV' 단위", parse_voltage_kv("154 kV"), [154.0])
    eq("voltage 해석 불가·결측", (parse_voltage_kv("abc"), parse_voltage_kv(None)), ([], []))
    eq("등급 765·354·380·145·66·22.9", [voltage_grade(x) for x in (765, 354, 380, 145, 66, 22.9)],
       ["765", "345", "345", "154", "50-100", "<50"])
    eq("등급 결측·HVDC", (voltage_grade(float("nan")), voltage_grade(250.0, hvdc=True)), ("unknown", "HVDC"))
    eq("회선 수", [parse_circuits("2", None), parse_circuits(None, "6"), parse_circuits("2", "12")], [2.0, 2.0, 2.0])
    eq("회선 수 모름", [np.isnan(parse_circuits(None, "4")), np.isnan(parse_circuits("1/3", None))], [True, True])
    chk("출력 MW 변환", [parse_output_mw("1.2 GW"), parse_output_mw("500 kW"), parse_output_mw("2 MW;3 MW")],
        [1200.0, 0.5, 5.0])
    eq("출력 'yes'·단위 없음은 NaN", [np.isnan(parse_output_mw("yes")), np.isnan(parse_output_mw("12"))], [True, True])

    # --- 연결 성분과 브리지
    k, lab = graph_components(*graph_two_components())
    eq("두 성분 fixture", (k, sorted(np.bincount(lab).tolist())), (2, [3, 3]))
    n, u, v = graph_path(6)
    br, small = find_bridges(n, u, v)
    eq("경로 P6: 엣지 5개 모두 브리지", br.tolist(), [True] * 5)
    eq("경로 P6: 작은 쪽 노드 수", small.tolist(), [1, 2, 3, 2, 1])
    eq("사이클 C6: 브리지 없음", int(find_bridges(*graph_cycle(6))[0].sum()), 0)
    eq("완전그래프 K5: 브리지 없음", int(find_bridges(*graph_complete(5))[0].sum()), 0)
    n, u, v = graph_barbell()
    br, small = find_bridges(n, u, v)
    eq("barbell: 다리 하나만 브리지, 작은 쪽 5", (br.tolist().index(True), int(br.sum()), int(small[-1])), (20, 1, 5))
    n, u, v = graph_tree(4)
    eq("포화 이진 트리(31노드): 엣지 30개 모두 브리지", int(find_bridges(n, u, v)[0].sum()), 30)
    br, _ = find_bridges(4, np.array([0, 1, 1, 2]), np.array([1, 2, 2, 3]))
    eq("평행 엣지는 브리지가 아니다", br.tolist(), [True, False, False, True])
    brc, _ = bridges_circuit_level(4, np.array([0, 1, 2]), np.array([1, 2, 3]), np.array([1.0, 2.0, np.nan]))
    eq("회선 단위: 2회선 구간은 브리지가 아니고 모르면 1회선", brc.tolist(), [True, False, True])
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_random):
        n, u, v = _random_multigraph(rng)
        bad += int((find_bridges(n, u, v)[0] != _brute_bridges(n, u, v)).any())
    eq(f"무작위 다중 그래프 {n_random}개: 브리지가 엣지 하나씩 지워 본 결과와 일치", bad, 0)

    # --- 유효 저항 (전도도 c, 저항 = 1/c의 단위)
    n, u, v = graph_path(7)
    er = effective_resistance(n, u, v, np.ones(6))
    chk("경로: 엣지 R = 1, 레버리지 = 1", np.r_[er["r_edge"], er["leverage"]], np.ones(12))
    for nn in (5, 8):
        n, u, v = graph_cycle(nn)
        er = effective_resistance(n, u, v, np.ones(nn))
        chk(f"사이클 C{nn}: 엣지 R = (n-1)/n", er["r_edge"], np.full(nn, (nn - 1) / nn))
        dm = np.arange(1, nn)
        chk(f"사이클 C{nn}: rbar = 평균 d(n-d)/n", er["rbar"], np.full(nn, np.mean(dm * (nn - dm) / nn)))
    for nn in (3, 6, 10):
        n, u, v = graph_complete(nn)
        er = effective_resistance(n, u, v, np.ones(len(u)))
        chk(f"완전그래프 K{nn}: R = 2/n", np.r_[er["r_edge"], er["rbar"]], np.full(len(u) + nn, 2.0 / nn))
    n, u, v = graph_barbell()
    er = effective_resistance(n, u, v, np.ones(len(u)))
    chk("barbell: 다리 R = 1·레버리지 1, 클릭 엣지 R = 0.4", [er["r_edge"][-1], er["leverage"][-1], er["r_edge"][0]],
        [1.0, 1.0, 0.4])
    er = effective_resistance(3, np.array([0, 1]), np.array([1, 2]), np.array([2.0, 4.0]))
    chk("직렬 저항: 1/2, 1/4", er["r_edge"], [0.5, 0.25])
    er = effective_resistance(2, np.array([0, 0]), np.array([1, 1]), np.array([1.0, 3.0]))
    chk("병렬 전도도 1+3: R = 1/4, 레버리지 1/4·3/4", np.r_[er["r_edge"], er["leverage"]], [0.25, 0.25, 0.25, 0.75])
    er = effective_resistance(*graph_two_components(), np.ones(5))
    eq("성분별 계산: 노드 3개 성분의 K3 R = 2/3", (round(er["r_edge"][0], 12), er["comp_size"].tolist()), (round(2 / 3, 12), [3, 3]))
    worst = {"foster": 0.0, "lev_bridge": 0.0, "r_edge": 0.0, "rbar": 0.0}
    for _ in range(n_random):
        n, u, v = _random_multigraph(rng)
        if len(u) == 0:
            continue
        c = rng.uniform(0.1, 5.0, len(u))
        er = effective_resistance(n, u, v, c)
        kk = graph_components(n, u, v)[0]
        worst["foster"] = max(worst["foster"], abs(er["leverage"].sum() - (n - kk)))
        br, _ = find_bridges(n, u, v)
        worst["lev_bridge"] = max(worst["lev_bridge"], float(np.max(np.abs((er["leverage"] > 1 - 1e-9) ^ br))))
        lp = _laplacian_pinv(n, u, v, c)
        ref = np.diag(lp)[u] + np.diag(lp)[v] - 2 * lp[u, v]
        worst["r_edge"] = max(worst["r_edge"], float(np.max(np.abs(ref - er["r_edge"]))))
        lab = er["comp"]
        full = np.diag(lp)[:, None] + np.diag(lp)[None, :] - 2 * lp
        for i in range(n):
            same = np.flatnonzero((lab == lab[i]) & (np.arange(n) != i))
            if len(same):
                worst["rbar"] = max(worst["rbar"], abs(full[i, same].mean() - er["rbar"][i]))
    chk(f"무작위 가중 그래프 {n_random}개: Foster 정리 Σ레버리지 = n - 성분 수", worst["foster"], 0.0, 1e-8)
    chk("무작위 가중 그래프: 레버리지 = 1 <=> 브리지", worst["lev_bridge"], 0.0, 0.0)
    chk("무작위 가중 그래프: 엣지 R이 유사역행렬 직접 계산과 일치", worst["r_edge"], 0.0, 1e-8)
    chk("무작위 가중 그래프: rbar이 R 행렬 평균과 일치", worst["rbar"], 0.0, 1e-8)

    # --- 도형 보조 함수
    boxes = np.array([shapely.box(0, 0, 100, 100), shapely.box(100, 0, 200, 100), shapely.box(250, 0, 350, 100)])
    eq("변전소 도형 묶기: 맞닿으면 합치고 50 m 떨어지면 안 합친다",
       [len(set(cluster_geometries(boxes, d).tolist())) for d in (0.0, 40.0, 60.0)], [2, 2, 1])
    site, dist = attach_to_sites(
        shapely.points([[0.0, 0.0]]), np.array([shapely.Point(10, 0), shapely.Point(200, 0)]), 300.0, np.array([1, 0])
    )
    eq("부착 우선순위: 전압 확인된 변전소가 먼저, 아니면 가까운 쪽", (int(site[0]), round(float(dist[0]), 6)), (1, 200.0))
    site, _ = attach_to_sites(shapely.points([[0.0, 0.0]]), np.array([shapely.Point(400, 0)]), 300.0, np.array([0]))
    eq("부착 반경 밖이면 -1", int(site[0]), -1)

    # --- 장난감 지도로 build_graph 검산
    ln, sb = synthetic_map()
    g = build_graph(ln, sb)
    a = analyze_graph(g["nodes"], g["edges"])
    s = a["summary"]
    eq("장난감 지도: 노드 14(변전소 6 + 접합 1 + 끝 7)", (s["n_nodes"], s["n_nodes_substation"], s["n_nodes_junction"], s["n_nodes_end"]), (14, 6, 1, 7))
    eq("장난감 지도: 엣지 11, 자기 루프 1개 제외", (s["n_edges"], g["diag"]["n_self_loops_dropped"]), (11, 1))
    eq("장난감 지도: 연결 성분 4, 최대 성분 7", (s["n_components"], s["comp_sizes_top5"][:2]), (4, [7, 3]))
    eq("장난감 지도: 브리지 9(말단 가지 포함), 말단이 아닌 브리지 2", (s["n_bridges"], s["n_bridges_nonpendant"]), (9, 2))
    eq("장난감 지도: 변전소 내부 통과 꼭짓점 1개에서 구간 분할", g["diag"]["n_force_junction_nodes"], 1)
    x_expect = 0.35 * 9.7 * 100.0 / 154.0**2
    e_bd = a["edges"].loc[(a["edges"]["way_id"] == 3)]
    chk("리액턴스 대용값 x = 0.35·L·100/V² (154 kV, 9.7 km)", e_bd["x_pu"].to_numpy(), [x_expect], 1e-12)
    chk("리액턴스 대용값: 같은 길이에서 (154/345)² 배", 0.35 * 100 * 100 / 345.0**2 / (0.35 * 100 * 100 / 154.0**2),
        (154.0 / 345.0) ** 2, 1e-12)
    chk("장난감 지도: Foster 정리", s["foster_residual"], 0.0, 1e-9)
    g2 = build_graph(ln, sb, {"snap_m": 100.0})
    s2 = analyze_graph(g2["nodes"], g2["edges"])["summary"]
    eq("snap 100 m: 60 m 떨어진 두 끝점이 합쳐져 노드 13, 성분 3", (s2["n_nodes"], s2["n_components"]), (13, 3))
    g3 = build_graph(ln, sb, {"include_hvdc": True})
    s3 = analyze_graph(g3["nodes"], g3["edges"])["summary"]
    eq("HVDC 포함: 엣지 +1, 노드 +2, 성분 +1", (s3["n_edges"], s3["n_nodes"], s3["n_components"]), (12, 16, 5))
    g5 = build_graph(ln, sb, {"split_in_sites": False})
    s5 = analyze_graph(g5["nodes"], g5["edges"])["summary"]
    eq("변전소 안 분할 끔: 변전소 통과 선로가 한 엣지로 남아 노드 13, 엣지 10, 성분 4", (s5["n_nodes"], s5["n_edges"], s5["n_components"]), (13, 10, 4))
    g4 = build_graph(ln, sb, {"attach_radius_m": 100.0})
    assert g4["diag"]["n_terminals_attached"] < g["diag"]["n_terminals_attached"], "반경을 줄이면 부착이 줄어야 한다"
    rows.append({"항목": "부착 반경 100 m: 300 m보다 부착 끝점이 적다", "오차": 0.0, "허용": 0.0, "결과": "OK"})
    ln0 = ln.copy(deep=True)
    build_graph(ln, sb)
    eq("입력 불변: build_graph가 선로 표를 바꾸지 않는다", ln.drop(columns=["refs", "xy"]).equals(ln0.drop(columns=["refs", "xy"])), True)

    # --- 직렬 합침
    nd = pd.DataFrame({"x": 0.0, "y": 0.0, "is_site": [True, False, False, True, False], "site_kv": np.nan, "n_members": 1,
                       "deg": [1, 2, 3, 1, 1], "node_type": ""})
    ed = pd.DataFrame({"edge_id": range(4), "u": [0, 1, 2, 2], "v": [1, 2, 3, 4], "length_km": [10.0, 20.0, 5.0, 1.0],
                       "x_pu": [0.1, 0.2, 0.05, 0.01], "circuits_n": [2.0, 1.0, np.nan, 2.0], "way_id": 1, "power": "line",
                       "kv": 154.0, "grade": "154", "hvdc": False, "underground": False})
    ct = contract_series_nodes(nd, ed)
    ec = ct["edges"]
    e02 = ec.loc[((ec["u"] == 0) & (ec["v"] == 1)) | ((ec["u"] == 1) & (ec["v"] == 0))]
    eq("직렬 합침: 노드 5 -> 4, 엣지 4 -> 3", (len(ct["nodes"]), len(ec), ct["n_contracted"]), (4, 3, 1))
    chk("직렬 합침: 길이 30, x 0.3, 회선 수 min(2, 1) = 1", [e02["length_km"].iloc[0], e02["x_pu"].iloc[0], e02["circuits_n"].iloc[0]],
        [30.0, 0.3, 1.0])
    return pd.DataFrame(rows)


# ===================================================================== 8. 데이터 완결성 진단 (집계만)
def _eligible_sites(subs: gpd.GeoDataFrame):
    kv = subs["kv"].to_numpy(float)
    elig = subs.loc[(kv >= 100.0) | np.isnan(kv)].reset_index(drop=True)
    return elig, elig.geometry.to_numpy()


def diagnose_unknown_voltage(lines: pd.DataFrame, subs: gpd.GeoDataFrame, cfg: dict | None = None) -> dict:
    """전압 태그가 없는 line·cable(내부 모선·베이 제외)과 송전 그래프의 관계. 닿는다 = 그래프 선로와 노드를 공유하거나
    끝점이 변전소 부착 반경 안에 있다. 집계만 돌려준다."""
    cfg = {**CONFIG_DEFAULT, **(cfg or {})}
    gl = select_graph_lines(lines, cfg)
    graph_nodes = set()
    for refs in gl["refs"]:
        graph_nodes.update(np.asarray(refs).tolist())
    unk = lines.loc[(~lines["internal"]) & lines["power"].isin(["line", "cable"]) & (lines["grade"] == "unknown")]
    _, geoms = _eligible_sites(subs)
    tree = STRtree(geoms) if len(geoms) else None
    touch = []
    for refs, xy in zip(unk["refs"], unk["xy"], strict=True):
        shared = any(int(r) in graph_nodes for r in refs)
        near = False
        if tree is not None and len(xy) >= 2:
            near = len(tree.query(shapely.points(xy[[0, -1]]), predicate="dwithin", distance=cfg["attach_radius_m"])[0]) > 0
        touch.append(shared or near)
    touch = np.array(touch, dtype=bool)
    return {
        "n_ways": int(len(unk)),
        "length_km": float(unk["length_km"].sum()),
        "n_touch_graph": int(touch.sum()),
        "length_km_touch_graph": float(unk.loc[touch, "length_km"].sum()) if len(unk) else 0.0,
    }


def diagnose_mixed_grade_nodes(lines: pd.DataFrame, subs: gpd.GeoDataFrame, cfg: dict | None = None) -> dict:
    """전압 등급이 다른 선로가 한 OSM 노드를 공유하는 곳(같은 철탑 공유나 연결 오류일 수 있다)의 집계."""
    cfg = {**CONFIG_DEFAULT, **(cfg or {})}
    gl = select_graph_lines(lines, cfg)
    role: dict = {}
    pos: dict = {}
    for refs, xy, gr in zip(gl["refs"], gl["xy"], gl["grade"], strict=True):
        n = len(refs)
        for i, r in enumerate(np.asarray(refs).tolist()):
            role.setdefault(r, []).append((gr, i in (0, n - 1)))
            pos[r] = xy[i]
    mixed = [r for r, v in role.items() if len({g for g, _ in v}) > 1]
    out = {"n_mixed_nodes": len(mixed), "n_interior_in_all": 0, "n_endpoint_in_all": 0, "n_endpoint_in_some": 0, "n_within_site": 0}
    for r in mixed:
        ends = [e for _, e in role[r]]
        out["n_interior_in_all"] += int(not any(ends))
        out["n_endpoint_in_all"] += int(all(ends))
        out["n_endpoint_in_some"] += int(any(ends) and not all(ends))
    _, geoms = _eligible_sites(subs)
    if mixed and len(geoms):
        ii, _ = STRtree(geoms).query(shapely.points(np.array([pos[r] for r in mixed])), predicate="dwithin", distance=cfg["attach_radius_m"])
        out["n_within_site"] = int(len(np.unique(ii)))
    return out
