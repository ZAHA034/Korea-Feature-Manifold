# 자료 출처와 라이선스 고지

이 저장소의 코드에는 아직 라이선스를 정하지 않았다. 아래 자료로 만든 표·그림·노트북 출력은 각 자료의 조건을 따른다.

## OpenStreetMap (ODbL 1.0)

- 도로망: © OpenStreetMap contributors. Geofabrik 한국 추출본 `south-korea-261017.osm.pbf`를 썼고, 조건은 [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)이다.
- OSM으로 만든 파생 데이터베이스(통행시간과 그로부터 계산한 값)가 들어 있는 다음 파일은 ODbL 1.0으로 공개한다.
  - `notebooks/spike/`의 `g_adj_edges.csv`, `g_adj249_edges.csv`, `bridges.csv`, `node_matching.csv`, `island_links.csv`(끝점 좌표), `s3_outputs/*.csv`, `s3_results.json`, `s4_results.json`, `diag_results.json`, `diag_outputs/*.csv`
  - `notebooks/spike/`의 노트북 출력(`01_network`, `02_curvature`, `03_embedding`, `04_diagnostics`)
- 지도와 그림(ODbL의 Produced Work): `notebooks/spike/s3_outputs/figures/*.png`, `notebooks/spike/s4_outputs/d_time_*.png`, `notebooks/spike/diag_outputs/*.png`, 노트북 안의 그림. 그림과 노트북 첫 셀에 출처를 적었다.
- 만든 방법(ODbL 4.6): `notebooks/spike/`의 노트북과 `curvature_lib.py`, `embed_lib.py`, `diag_lib.py`가 원자료에서 위 파일을 만드는 전 과정이다. 통행시간 행렬 원본은 저장소 밖에 있고, 같은 코드로 다시 만들 수 있다.

## 그 밖의 자료

- 행정동 경계: [vuski/admdongkor](https://github.com/vuski/admdongkor) ver20251231(CC BY 4.0). 원자료는 통계청 SGIS(공공누리 제1유형, 출처표시)다.
- 주민등록 인구 2025-12: 행정안전부 주민등록 인구통계([jumin.mois.go.kr](https://jumin.mois.go.kr))
- 통근·통학 OD 2020: 국가데이터처 인구주택총조사(KOSIS `DT_1PA2021`). `notebooks/spike/s4_outputs/d_flow_*.png`는 이 자료로 만들었다.
- 인구이동 OD 2025: 국가데이터처 국내인구이동통계, KOSIS 온라인간행물 「시군구 전출입지별 이동자수」 2025년(연간)(공공누리 제1유형, 출처표시). `notebooks/spike/04_diagnostics.ipynb`의 D절과 `diag_results.json`·`diag_outputs/`의 인구이동 항목은 이 자료로 만들었다.
- 고도: Copernicus DEM GLO-90. "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved." DOI [10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65). 전체 조건은 ESA 사용자 라이선스를 따른다.
- 백두대간 종주길 GPX: 한국등산트레킹지원센터, 공공데이터포털 15098180(이용허락범위 제한 없음)
- 섬 링크 소요 시간: 한국공항공사, 한일고속, 대저페리의 공시 시간표(2026-10-19 조회)
