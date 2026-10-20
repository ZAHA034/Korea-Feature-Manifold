# 자료 출처와 라이선스 고지

이 저장소의 코드에는 아직 라이선스를 정하지 않았다. 아래 자료로 만든 표·그림·노트북 출력은 각 자료의 조건을 따른다.

## OpenStreetMap (ODbL 1.0)

- 도로망: © OpenStreetMap contributors. Geofabrik 한국 추출본 `south-korea-261017.osm.pbf`를 썼고, 조건은 [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)이다.
- OSM으로 만든 파생 데이터베이스(통행시간과 그로부터 계산한 값)가 들어 있는 다음 파일은 ODbL 1.0으로 공개한다.
  - `notebooks/spike/`의 `g_adj_edges.csv`, `g_adj249_edges.csv`, `bridges.csv`, `node_matching.csv`, `island_links.csv`(끝점 좌표), `s3_outputs/*.csv`, `s3_results.json`, `s4_results.json`, `diag_results.json`, `diag_outputs/*.csv`, `s3_freeze.json`(속도 계수, OSM 정맥 관계 길이, 연륙·연도교 집계), `s4_freeze.json`(OSM 통행시간으로 맞춘 중력모형 β)
  - 열 단위 출처: `g_adj249_edges.csv`의 `T_uv`·`T_vu`·`flow_sum`·`flow_zero`·`w_od`와 `s3_outputs/edge_curvatures_g_adj249_od.csv`의 `w_od`는 KOSIS 통근·통학 OD(이용 조건 미확인)에서 나온 값이라 OSM 파생이 아니다. `shared_m`과 해안 지시변수는 admdongkor(CC BY 4.0), `s`·`h_e`는 Copernicus DEM에서 나온 값이다. 이 0S 파일들은 PLAN 6.1a의 분리 규칙(인구·경계·OD는 OSM 자료와 섞지 않은 별도 테이블)을 지키지 못했고, 증거 보존을 위해 고치지 않는다. 공개할 때 이 파일들을 위 ODbL 목록에 그대로 둘지는 사용자가 정한다(공개 전 점검).
  - `notebooks/spike/`의 노트북 출력(`01_network`, `02_curvature`, `03_embedding`, `04_diagnostics`). `00_tree_baseline.ipynb`는 합성 트리만 쓰므로 OSM과 무관하다.
- 지도와 그림(ODbL의 Produced Work): `notebooks/spike/s3_outputs/figures/*.png`, `notebooks/spike/s4_outputs/d_time_*.png`, `notebooks/spike/diag_outputs/*.png`, 노트북 안의 그림. 그림과 노트북 첫 셀에 출처를 적었다.
- 만든 방법(ODbL 4.6): 통행시간 행렬(파생 DB)은 `notebooks/spike/01_network.ipynb`가, 섬 링크 ×0.5·×2 행렬은 `02_curvature.ipynb` 2-5가 원자료에서 만든다. 곡률·임베딩·진단 값은 `02_curvature.ipynb`·`03_embedding.ipynb`·`04_diagnostics.ipynb`와 `curvature_lib.py`·`embed_lib.py`·`diag_lib.py`가 계산한다. `s4_results.json`과 `s4_outputs/*.png`는 `s4_collect.py`가 03 노트북의 저장된 결과에서 모은다. `s3_results.json`과 `s3_outputs/`의 표·그림은 `notebooks/spike/tools/make_s3_results.py`가 02 노트북 3절 셀(3-5 제외)을 다시 실행해 썼고, 그림 3장은 3-5 셀의 저장 출력에서 꺼냈다. 그림 7장의 출처 문구는 2026-10-19 22:08(d588aab)에 `notebooks/spike/tools/attrib_odbl_20261019.py`로 이미지 위에 덧썼다. 실행 순서와 입출력은 `notebooks/spike/README.md`에, 실행 환경은 `notebooks/spike/env/`에 있다. 통행시간 행렬 원본은 저장소 밖에 있다. 같은 코드로 다시 만들 수 있게 두었다(원자료부터 다시 만들어 비교한 적은 아직 없다). 파일 메타데이터의 생성 시각이 달라지므로 비교는 행렬 내용 해시로 한다(`notebooks/spike/README.md`의 재현 수준).
- 전력망 탐색(`notebooks/spike/power/`, ADR 0014): OSM `power` 태그를 처리하는 코드와 출력을 지운 노트북만 있다. 결과 파일(표·그림·그래프)은 저장소에 넣지 않으며, 보안 검토 전에는 공개하지 않는다. 문서에 집계를 적을 수 있는지는 ADR 0014 열린 점이다.

## 그 밖의 자료

- 행정동 경계: [vuski/admdongkor](https://github.com/vuski/admdongkor) ver20251231(CC BY 4.0). 원자료는 통계청 SGIS(공공누리 제1유형, 출처표시)다.
- 주민등록 인구 2025-12: 행정안전부 주민등록 인구통계([jumin.mois.go.kr](https://jumin.mois.go.kr))
- 통근·통학 OD 2020: 국가데이터처 인구주택총조사(KOSIS `DT_1PA2021`, 이용 조건 미확인). `notebooks/spike/s4_outputs/d_flow_*.png`는 이 자료로 만들었다.
- 인구이동 OD 2025: 국가데이터처 국내인구이동통계, KOSIS 온라인간행물 「시군구 전출입지별 이동자수」 2025년(연간)(공공누리 제1유형, 출처표시). `notebooks/spike/04_diagnostics.ipynb`의 D절과 `diag_results.json`·`diag_outputs/`의 인구이동 항목은 이 자료로 만들었다.
- 고도: Copernicus DEM GLO-90. 저장소에는 가공한 값(경사 `s`, 초과 고도 `h_e` 등)만 있으므로 다음 고지를 단다. "produced using Copernicus WorldDEM-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved". DOI [10.5270/ESA-c5d3d65](https://doi.org/10.5270/ESA-c5d3d65). 전체 조건은 ESA 사용자 라이선스를 따른다. 문구 출처: [Copernicus Data Space Ecosystem의 COP-DEM 설명](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)의 User Obligations 절(위 DOI가 이 페이지로 연결된다), 2026-10-20 확인. spacedata.copernicus.eu의 라이선스 페이지는 이 PC에서 열리지 않아 대조하지 못했고, Product Handbook(v5.0)에는 이 문구가 없다.
- 백두대간 종주길 GPX: 한국등산트레킹지원센터, 공공데이터포털 15098180(이용허락범위 제한 없음)
- 섬 링크 소요 시간: 한국공항공사, 한일고속, 대저페리의 공시 시간표(2026-10-19 조회)

## 공개 전 점검 (저장소를 공개로 바꾸기 전)

원격 저장소는 지금 비공개로 보인다(익명 API 404, 2026-10-20 확인). 공개하기 전에 다음을 정하거나 확인한다. 0S 증거물과 바이트 사본은 고치지 않고 이 목록에만 올린다.

- 코드 라이선스와 문서 이용 조건: 아직 정하지 않았다(사용자 결정).
- Copernicus DEM 크레딧 문구: ESA 사용자 라이선스 전문과 대조하지 못했다.
- 통근·통학 OD(KOSIS `DT_1PA2021`)의 이용 조건: 확인하지 않았다.
- KOSIS·admdongkor·DEM 값이 섞인 0S 파일(`g_adj249_edges.csv` 등)을 위 ODbL 목록에 그대로 둘지(사용자 결정).
- 전력망 집계(`docs/IMPLEMENTATION.md` 진행 현황)를 저장소 문서에 적어도 되는지와 보안 검토: ADR 0014 열린 점의 안 A/B(사용자 결정), PLAN 6.8.
- 사용자 이름이 든 절대 경로: `notebooks/spike/s4_results.json` 7곳, `notebooks/spike/04_diagnostics.ipynb` 출력 1곳, `notebooks/spike/tools/`의 `make_s3_results.py` 15행·`attrib_odbl_20261019.py` 7행·`README.md` 23행, `docs/reports/0S-prereg/results_full.md` 41·125·148행, `docs/IMPLEMENTATION.md`의 작업 폴더 항목.
