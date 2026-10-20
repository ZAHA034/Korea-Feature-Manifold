# KFM 구현 지시서 (0S 탐색 실험 + 0단계 골격)

> 새 세션은 아래 "진행 현황"에서 다음 세션 블록을 확인하고, 그 블록에 해당하는 PLAN 절과 ADR만 읽고 시작한다. 세션이 끝나면 이 표를 갱신해 같은 커밋에 넣는다.
> "진행 현황", "지시서와 달라진 점", "작업 방식"은 2026-10-18에 덧붙인 것이고, 그 아래 "공통 규칙"부터가 사용자 지시서 원문이다.

## 진행 현황

| 세션 | 상태 | 커밋 | 비고 |
|---|---|---|---|
| S0 | 완료(변형) | e51f8f3 | C:\dev\kfm로 옮기지 않고 OneDrive 폴더에서 계속한다(아래 "지시서와 달라진 점") |
| P1 | 완료 | a73668f | 테스트가 0개면 pytest가 종료 코드 5로 실패하므로 import 스모크 테스트 1개를 두었다. ruff는 `docs/`를 제외한다 |
| S1 | 완료(변형) | 태그 `0S-start` | ADR 0010 2·5·6·7항과 0013 채택(2026-10-19). 합성 사전 검증(566a1c6)을 반영해 0013의 곡률 검정과 반지름 상한을 바꿨다(아래 "지시서와 달라진 점") |
| S1b | 완료 | aa718a8 | 판정 1–3 통과. 쌍곡 최솟값 0.03394, 시드 0–4 비율 0.2253(사전 검증과 일치). 0S 실행 환경 `C:\dev\kfm-spike`를 여기서 만들었다 |
| S2 | 완료 | e527fc1·4f646e5·bc4cab9 | 250×250 행렬(속도 계수 c = 1.0527, 네이버 지도 새벽 출발 6쌍, 보정 후 오차 −4.6~+6.3%), 도달 불가 0. 섬 링크는 공시 시간표 소요 + 도착 권장 시간(항공 60분, 배 30분)이며, 바꾸려면 S3 고정 커밋 전에 다시 실행한다 |
| S3 | 완료 | 29cc115·b2c7ca2(고정)·48fab3c | 판정 **조정**(유효성 점검 실패: κ_ex와 차수 항 0.227, log 직선 거리 0.207). 주 검정은 통과, 편상관은 실패(κ_ex와 우회율 ρ 0.82, 구성상 관계) |
| S4 | 완료 | c901dae(고정)·aaa416f | 쌍곡 이점 없음(왜곡 비 `d_time` 0.979, `d_flow` 0.913 > 0.8), 중력모형 이상의 위계 없음(δ̄ 관측 0.301 > 대조군 20개 전부). `d_flow`는 247개 단위(제주시·서귀포시 제외) |
| S5 | 완료 | a2055c2(태그 `0S-end`) | 0S 판단 **조정**(곡률), 쌍곡 이점·위계 없음. 보고서 `docs/reports/0S-spike.md`(ADR 0009·0010 문구 제안 포함). 0S-end 뒤 d588aab(2026-10-19 22:08)에서 그림 7장에 출처 띠를, 노트북 01·02·03 첫 셀에 출처 고지를 더했다(계산 무관, 보고서 "검증 방법" 참조) |
| P2 | 초안 완료(채택 대기) | 710eaf1 | ADR 0001·0002·0003·0005·0006 초안, 0004 수정·0012 점검, 0009·0010 0S 반영 개정안. 사용자 채택 결정 뒤 DESIGN.md v0.2 |
| 전력망 탐색 | 완료(탐색) | c8de6c2 | ADR 0014(제안). OSM 154 kV 이상 1,683노드·1,925엣지, 최대 성분 88.9%. 결과는 저장소 밖 `C:\dev\kfm-data\power\`에만 있고, 보안 검토 전에는 공개하지 않는다 |
| 검토 반영(R1–R5) | 완료 | (이 커밋) | 0S 보고서 정정·보강(날짜 표시), ADR 0001·0002·0004·0005·0006·0009·0010·0012·0014 수정(모두 제안), CI(`.github/workflows/ci.yml`)·저장소 불변 테스트·`notebooks/spike/env/`·`tools/`·`docs/reports/0S-prereg/` 사본, `NOTICE.md`의 공개 전 점검. 사용자 결정 대기 항목은 각 ADR 열린 점과 `NOTICE.md`에 있다 |
| P3–P7 | 대기 | — | |

## 지시서와 달라진 점 (2026-10-18)

- **작업 폴더**: 저장소는 OneDrive 폴더(`C:\Users\mycoms\OneDrive\Desktop\Korea Feature Manifold`)에 그대로 둔다. 무거운 것만 밖에 둔다.
  - 가상환경: `C:\dev\kfm-venv`. Claude Code는 `.claude/settings.local.json`, VSCode 통합 터미널은 `.vscode/settings.json`으로 `UV_PROJECT_ENVIRONMENT`가 자동 지정된다(둘 다 git 제외). 저장소 안에 `.venv`가 생기면 잘못된 것이다.
  - 자료: `C:\dev\kfm-data`. 아래 원문의 `C:\dev\kfm\data\…`는 `C:\dev\kfm-data\…`로 읽는다(예: `C:\dev\kfm-data\raw\`, `C:\dev\kfm-data\spike\travel_time.parquet`).
  - 0S 실행 환경은 원문대로 `C:\dev\kfm-spike`.
- **S0**: clone 단계는 하지 않았다. 나머지 작업(ADR 0013 추가, 변경 이력 두 행, CLAUDE.md 현재 단계)은 원문대로 했다.
- **S1 (2026-10-19)**: 합성 사전 검증(`docs/reports/0S-prereg-checks.md`)에 따라 ADR 0013을 기준 차분 곡률 검정과 R_max 5로 바꿔 채택했다. 아래 세션 블록 원문은 고치지 않고 다음처럼 읽는다.
  - S1b: R_max 5와 2단계 최적화(ADR 0013 4항)로 계산한다. 판정 문턱은 그대로다.
  - S2: 90 m DEM을 함께 받는다(ADR 0013 3항). 환경 단계(`C:\dev\kfm-spike`)는 S1b에서 했다. 이 세션의 `UV_PROJECT_ENVIRONMENT`는 본 프로젝트용이므로 kfm-spike에서 uv를 쓸 때는 지운다.
  - S3: 실데이터 곡률 전에 ADR 0013 7항의 고정 커밋을 한다. 4번 판정 계산은 ADR 0013 1·2항(κ_ex와 경사의 순위상관과 우회율 편상관, 시도 jackknife 최대 p, 유효성 점검)이며 초기하 검정과 κ̃는 판정에 쓰지 않는다. 섬 링크 시간 ×0.5·×2 재계산과 6항 보조 보고(GPX 연결 확인 포함)를 더한다. 5번 지도 두 장은 κ와 κ_ex다.
  - S4: 2번 최적화는 ADR 0013 4항(R_max 5, Adam 뒤 L-BFGS-B 마무리, 재시작 구성, 실패 규칙)을 따른다. S1b에서 확인한 구현 주의: geoopt `PoincareBall`의 곡률 텐서가 float32라 `torch.set_default_dtype(torch.float64)`를 먼저 둔다. ADR이 정하지 않은 L-BFGS-B 설정은 maxfun 50,000(기본 15,000은 maxiter보다 작다), gtol 1e-10(ftol이 실제 종료 기준이 되게. S1b는 기본 1e-5였고 차이는 5e-9 이내)으로 두고 실데이터 실행 전 설정 셀에 커밋한다. nbconvert 실행에는 `--ExecutePreprocessor.timeout=3600`을 붙인다. (2026-10-20 보강: 03 머리말은 `--ExecutePreprocessor.timeout=36000`이다. 03은 10개 프로세스로 nbconvert 734초가 걸렸고 재시작 계산 시간의 합은 5,331초라, 코어가 적은 기기에서는 한 셀이 3600초를 넘을 수 있다.)
  - S5: 판정은 ADR 0013 5항의 조정 / 계속 / 전제 재검토 / 부분 대응 또는 미완이다.
- **S2 (2026-10-19)**: 자료는 모두 로그인 없이 Claude Code가 받았다. 주민등록 인구는 jumin.mois.go.kr CSV에서, 통근·통학 OD는 사전 검증 세션이 받은 KOSIS 원본에서 가져왔다. 출처와 MD5는 `C:\dev\kfm-data\raw\manifest.csv`에 있다(사본: `notebooks/spike/env/raw_manifest.csv`. 2026-10-20 보강). 원문에 없던 처리는 다음과 같다(ADR 0013 7항의 고정 커밋 대상).
  - 출발점: 주민등록 출장소 인구(63곳, 231,907명)를 상위 읍면동에 얹는다. 인구 가중 중심이 자기 시군구 밖이면 인구가 가장 많은 행정동의 대표점을 쓴다(옹진→영흥면, 신안→압해읍, 통영→광도면, 완주→봉동읍).
  - 연결선 시간은 직선 거리 / 30 km/h다. 섬 링크는 3개(김포–제주 항공, 완도–제주 배, 포항–울릉 배)이고, 대기 시간은 공항·터미널 도착 권장 시간(항공 60분, 배 30분, 2026-10-19 검색 근거)이고, 배차 대기와 결항은 넣지 않는다.
  - 검증 쌍 청사 좌표에는 `amenity=townhall` 외의 태그도 썼다(대전 중구청은 정류장 좌표로 근사).

## 작업 방식 (2026-10-18 합의)

- 대량 구현은 Sonnet 서브에이전트(`model: "sonnet"`)가 맡는다. 메인 루프(Opus)는 작업 지시 작성, `git diff`와 테스트 결과 검토, 방향 제시, 커밋과 보고만 한다. 수정은 같은 에이전트에게 SendMessage로 이어서 시킨다.
- 구현 에이전트는 세션 블록당 1개다. 서로 독립된 작업일 때만 2개까지 쓴다. 지시에는 범위, 건드릴 파일, 완료 기준, 금지 사항, 읽을 PLAN·ADR 절을 적고, 결과는 요약(바꾼 파일, 테스트 결과, 질문)만 돌려받는다.
- 다중 에이전트 검증은 결정에 근거가 꼭 필요할 때만 한다.
- 사용자 결정 사항(판정 값 확인, 로그인이 필요한 자료, ADR 채택)은 사용자에게 넘긴다.

---

## 공통 규칙 (모든 세션 시작 시 적용)

- 기준 문서: docs/PLAN.md v0.3(동결), docs/DESIGN.md v0.1, CLAUDE.md, docs/decisions/.
  세션 시작 시 이 세션에 해당하는 PLAN 절과 ADR만 읽는다.
- PLAN 본문은 고치지 않는다. 계획과 다르게 해야 하면 작업을 멈추고
  ADR 초안(상태: 제안)을 써서 사용자에게 보여준다.
- DESIGN.md는 ADR이 '채택'된 뒤에만 고친다.
- 한 세션은 아래 세션 블록 하나만 수행한다. 범위를 넘는 작업은 하지 않고
  "다음 세션 후보"로 적는다.
- 자료가 없거나 내려받기에 사람의 조작(로그인, 신청서)이 필요하면
  추측으로 대체하지 말고 멈춰서 필요한 파일과 받을 곳을 알려준다.
- data/와 .venv는 git에 넣지 않는다. 원천 파일은 data/raw/에 넣고 덮어쓰지 않는다.
- Windows 주의: 경로는 짧게(C:\dev\...), 좌표 변환은 pyproj always_xy=True,
  scipy csr_matrix는 중복 엣지를 더하므로 먼저 최솟값으로 줄이고
  0 가중치에는 하한 1e-6분을 둔다.
- 커밋: 세션 끝에 한 번 이상. 메시지 형식 "<세션ID>: <요약>"
  (예: "S2: 0S 도로망 그래프와 통행시간 행렬").
- 세션 종료 보고(채팅으로): 한 일, 산출 파일, 완료 기준 충족 여부,
  막힌 점, 다음 세션 후보. 5–10줄.

---

## 트랙 A: 0S 탐색 실험 (10/19 시작, 10/25 보고서 마감)

### S0. 환경 이전과 문서 정리 (10/18–19, 0S 시작 전)

[사용자] 현재 OneDrive 폴더에서 Claude Code를 닫고,
  `git clone https://github.com/ZAHA034/Korea-Feature-Manifold C:\dev\kfm`
  으로 새로 받은 뒤 그 폴더에서 Claude Code를 연다.
  OneDrive 쪽 폴더는 확인 후 보관만 하고 더 쓰지 않는다.

Claude Code 작업:
1. 사용자 환경 변수 안내: UV_CACHE_DIR=C:\dev\.uvcache (설정 방법만 알려준다).
2. docs/decisions/0013-0s-판정-기준-보강.md 를 사용자가 준 초안 그대로
   추가한다(상태: 제안).
3. docs/PLAN.md 변경 이력에 두 행만 추가한다(본문 변경 아님, 동결 규칙상 허용):
   - 0.3 | 2026-10-18 | 동결, 변경 규칙, 0S 시간 상한
   - ADR 0013 | 제안 | 0S 판정 기준 보강
4. CLAUDE.md "현재 단계"를 다음으로 바꾼다:
   "0단계 + 0S 탐색 실험. 0S 판정 규칙(ADR 0010 2·5·6·7항, 0013)
    채택 전에는 실데이터로 곡률·임베딩·δ를 계산하지 않는다."
5. 커밋.

완료 기준: C:\dev\kfm에서 git status가 깨끗하고, 원격과 동기화됨.

### S1. 판정 규칙 채택 (10/19 오전, 0S 1일차 시작)

Claude Code 작업:
1. ADR 0010의 2·5·6·7항과 ADR 0013을 읽고, 사용자가 확인할 값만
   표 하나로 정리해 보여준다:
   - 초기하 검정 유의수준(0.05), 단측 여부
   - 차수 통제 회귀식, 섬링크 지시변수 포함 여부
   - 쌍곡 반지름 상한 R_max(10), 수렴 기준(200회 상대 변화 < 1e-6, 최대 20,000회)
   - 왜곡 비 기준(0.8), d_flow 정규화와 대안 정규화
   - 중력 대조군 횟수(δ̄ 20회, 왜곡 5회), 유효 표본 수 불명 시 여유폭(0.02)
   - 엣지 분류 규칙(PLAN 6.1a), 산맥 선 출처
[사용자] 값을 확인하거나 수정해 답한다.
2. 답을 반영해 두 ADR의 상태를 '채택'으로, 날짜·시각을 기록한다.
3. 커밋 후 태그 `0S-start` 를 단다. 이 태그 시각이 0S 1일차 기준이다.
4. docs/reports/0S-spike.md 뼈대를 만든다(시작일, 채택한 판정 규칙 요약,
   빈 결과 절).

완료 기준: 태그 존재, 두 ADR 상태 '채택', 이 커밋 전에는
실데이터 곡률·임베딩 코드가 저장소에 없다.

### S1b. tree fixture 기준값 재계산 (S1 직후, 합성 데이터만)

ADR 0013 4항에 따라 반지름 상한·수렴 기준을 적용한 tree 판정 규약을 다시 계산한다.
1. notebooks/spike/00_tree_baseline.ipynb: 깊이 4 포화 이진 트리,
   PLAN 7.1의 목적함수·최적화 설정 + R_max·수렴 기준.
2. 시드 0–9로 원판·유클리드 왜곡, 시드 0–4의 최솟값·중앙값 비율을 낸다.
3. 판정 1–3(0.5, 0.8, 유클리드 0.1506 ± 0.003)이 여전히 통과하는지 확인하고
   새 기준값을 ADR 0013 "열린 점" 아래 결과로 적는다(판정 문턱은 바꾸지 않는다).
4. 통과하지 못하면 멈추고 사용자에게 보고한다(R_max 조정은 ADR 개정 사항).

### S2. 0S-1·0S-2: 자료와 통행시간 행렬 (10/19–20)

[사용자] 미리 받아 C:\dev\kfm\data\raw\ 아래에 둘 것:
  - 주민등록 인구 2025-12 행정동별 (jumin.mois.go.kr)
  - 통근·통학 OD 2020 시군구 (KOSIS DT_1PA2021) — 0S-3 가중용, 없으면 균등만
  - (선택) Copernicus DEM GLO-90 한국 범위 — ADR 0013 3항용
Claude Code가 직접 받는 것: Geofabrik south-korea PBF, vuski/admdongkor 경계.

환경: C:\dev\kfm-spike 에 별도 uv 프로젝트(Python 3.11).
  패키지: duckdb, geopandas, shapely, pyproj, scipy, numpy, pandas, pyarrow,
  POT, torch(CPU), geoopt, matplotlib, jupyter. 설치 결과를 기록한다.

노트북 notebooks/spike/01_network.ipynb (저장소 안, 실행 환경은 kfm-spike):
1. 첫 셀: 자료 파일명·날짜·MD5, 패키지 버전, 가정값(속도 표, 섬 링크).
2. 경계: admdongkor 2025 행정동 → 시군구 합치기. 단위는 부천 합친 250개.
   2025-12 인구로 인구 가중 중심 계산.
3. 도로망: PLAN 6.1a "도로망 파이프라인" 그대로(필터, 단순화, 방향,
   등급별 속도, 중복 엣지 최솟값, 0 가중치 하한, 강연결 최대 성분).
4. 출발점 연결: 인구 가중 중심 → 최근접 노드. 연결선 거리 > 5 km면 목록으로 남긴다
   (엣지 분류의 '데이터·설정 문제' 판정에 쓰인다).
5. 섬 링크: notebooks/spike/island_links.csv 를 만든다
   (제주, 울릉, 옹진 백령 등. 소요 + 대기 시간, 출처 메모).
   [사용자] 값 확인.
6. 통행시간 행렬: 25개씩 묶어 dijkstra. 비대칭 원본과 d_sym 저장
   (C:\dev\kfm\data\spike\travel_time.parquet).
7. 속도 계수: 검증 도시 쌍 5쌍 이상으로 전역 계수 하나만 맞춘다.
   [사용자] 지도 서비스 참고값(혼잡 적은 시간대, 조회 일시 기록) 제공.
8. 진단: 도달 불가 쌍 0개, 대각 0, 비대칭 지수 분포.

완료 기준: 250×250 행렬, 도달 불가 0, 검증 쌍 오차 표.
금지: 이 세션에서 곡률·임베딩 계산 금지(S1 태그 이후라도 세션 범위 밖).

### S3. 0S-3: G_adj와 곡률 (10/21–22)

노트북 notebooks/spike/02_curvature.ipynb:
1. G_adj: 경계 공유 쌍 + 연륙·연도교 링크 + 섬 링크.
   연륙·연도교는 notebooks/spike/bridges.csv 로 손으로 정리(거제, 남해, 진도 등).
   고립 노드가 없어야 한다.
2. ORC: POT ot.emd2 직접 구현, 엣지 길이 d_sym, α = 0.5.
   이웃 분포: 균등(기본), 통근·통학 OD 가중(있으면).
   먼저 합성 그래프 barbell·C8·C5·K5로 해석값(−0.6, 0, 0.25, 0.625)을
   1e-9 이내로 재현한 뒤 실데이터에 적용한다.
3. 엣지 분류: PLAN 6.1a 규칙 순서대로. 산맥 선은 S1에서 채택한 출처를 쓴다.
4. 판정 계산(ADR 0013):
   - p0, 상위 20개 중 자연 장벽 k, 기대값, 단측 초기하 p값
   - κ와 (1/deg_u + 1/deg_v)의 스피어만 상관
   - 잔차 곡률 κ̃로 같은 검정 반복
   - (DEM 있으면) h_e와 κ̃의 스피어만 상관, 순열 1,000회
5. 지도: 곡률 색칠 지도(κ, κ̃ 두 장), 상위 20개 엣지 표시.

완료 기준: 위 수치 전부와 지도 두 장. 판정 문턱은 바꾸지 않는다.

### S4. 0S-4: 임베딩과 δ (10/23–24)

노트북 notebooks/spike/03_embedding.ipynb:
1. δ 진단: d_time에 대해 δ, δ_rel, δ̄ (PLAN 3-5 정의, 시군구는 정확값).
   전체와 섬 연결 지역 제외 부분집합 둘 다.
2. 임베딩: d_time, 2차원 원판(geoopt)과 2차원 유클리드, 같은 상대 stress,
   척도 s 닫힌 해, R_max·수렴 기준(ADR 0013).
   재시작: 유클리드 해 축소 초기값 2개 + U(±0.1) 3개 이상.
   쌍곡 최솟값 > 유클리드 × (1 + 1e-3)이면 최적화 실패로 보고 최대 2회 재실행.
3. 보고: 평균 상대 왜곡, 비(쌍곡/유클리드), 수렴 여부, 반복 수,
   상한에 닿은 점 수, 최악 20쌍.
4. (선택) 통근·통학 OD가 있으면 d_flow(PLAN 3-3a, 249개 단위)와
   중력 대조군(3-3b)을 함께 계산. 시간이 부족하면 "미완"으로 둔다.

### S5. 0S-5 보고서 (10/25, 마감일)

docs/reports/0S-spike.md 작성. PLAN 6.1a "보고서 항목" 전부와:
- 판정: 계속 / 조정 / 전제 재검토 중 하나, 근거 수치
- ADR 0009·0010에 줄 시사점(구체적 문구 제안)
- 미완 항목 목록(기한 연장 없음)
커밋 후 태그 `0S-end`.

---

## 트랙 B: 0단계 골격 (0S와 병행, 세션을 번갈아 진행)

### P1. 프로젝트 설정 (PLAN 0-3)
- pyproject.toml: uv, src 레이아웃, 패키지 kfm, Python 3.11 유지
  (3.12 상향은 0S 종료 후 결정, 이유를 README에 한 줄).
- 0단계 의존성만: pydantic, pyyaml, numpy, pandas, pyarrow, geopandas,
  shapely, pyproj, duckdb, scipy, networkx. 개발: pytest, hypothesis, ruff.
- ruff·pytest 설정, `@pytest.mark.data` 마커 등록.
- README.md: 목적 3줄, 설치, 테스트 명령, 문서 위치.
완료: uv sync, uv run pytest(0개 통과), uv run ruff check . 성공.

### P2. ADR 작성 (PLAN 0-2)
- 0001, 0002, 0003, 0005, 0006을 PLAN 11.3 반 페이지 양식으로 작성.
  다루는 항목은 PLAN 4.2 표 그대로. 0001에는 PLAN 동결 규칙,
  0S 규칙 예외, 사실 표의 configs/reference 이전을 넣는다.
- 0004, 0012(0단계 부분: 1–3항, 4항의 설정 키·값 목록)는 기존 초안 점검만.
[사용자] 검토 후 채택 여부 결정.
- 채택된 것만 DESIGN.md(3장, 4장, 9.2, 11.2, 13)와 CLAUDE.md 규약에 반영,
  DESIGN.md 버전 0.2.
완료: 7개 ADR 상태 확정, DESIGN.md v0.2.

### P3. core (PLAN 0-4, 5.4)
- core/config.py: DESIGN 9.1 YAML 예시를 그대로 읽는 pydantic 모델,
  layers.<layer>.flow_model 값 검증(v1은 choice만 허용).
- core/hashing.py: config_hash(기본값 채운 키 정렬 JSON의 SHA-256),
  파일·테이블 해시.
- core/runs.py: runs 기록(PLAN 5.4 전 항목, code_dirty, env_lock_hash).
- core/seed.py, core/io.py(GeoParquet은 geopandas로만 쓰기, DuckDB 읽기 전용),
  core/world.py(읽기 전용 World 스냅샷).
- 테스트: YAML 공백·주석·키 순서가 달라도 같은 해시, 잘못된 flow_model 거부,
  dirty 상태 감지.

### P4. schema (PLAN 0-5, 5.3)
- schema/tables.py: 5.3 (가) 1–3층 전체와 runs, (나) 4층 파생 테이블.
  P2에서 채택된 ADR 기준. 컬럼 명세는 pydantic, pyarrow 스키마는 여기서 생성.
- schema/validate.py: 열 단위 벡터 검증(필수 컬럼, 타입, CRS = EPSG:5179,
  키 유일성, 값 범위, layer 값 목록).
- 임베딩 저장 API: embeddings·embedding_distortion·embedding_worst_pairs가
  모두 있을 때만 쓰는 함수(없으면 예외).
- 테스트: 테이블마다 정상·실패 사례 하나 이상.

### P5. 합성 fixture (PLAN 0-6, 7.1)
- tests/fixtures/: grid10, asym_triangle, two_components,
  crosswalk_split_merge, od_crosswalk, barbell, cycle(C8, C5), complete,
  tree, c4, square4, gravity_grid.
- 0단계 사용 fixture(grid10 등 5개)의 기대값 테스트는 지금 통과시킨다.
  3단계용 fixture는 생성만 하고 테스트는 xfail(reason="3단계")로 둔다.

### P6. 파이프라인 골격과 E2E (PLAN 0-7, 0-8)
- pipeline.py: 단계 함수 스텁, 빈 입력이어도 스키마를 만족하는 빈 테이블 반환.
- cli.py: `kfm run configs/baseline.yaml` (ingest, scenario, report는 자리만).
- configs/baseline.yaml, configs/scenarios/S-0001.yaml(DESIGN 9.1 예시),
  configs/presets/density/default.yaml, configs/reference/README.md.
- 테스트 test_pipeline_e2e.py: grid10으로 전체 실행 → 모든 출력 테이블 존재·
  스키마 통과, runs 1행, 재실행 시 config_hash 동일.

### P7. 0단계 마감
- PLAN 6.1 완료 기준 점검표를 docs/reports/M0.md에 기록.
- CLAUDE.md "현재 단계" 갱신(0S 판정에 따라 1단계 또는 계획 재검토).
- 커밋 후 태그 `M0`.

---

## 일정 (권장)

| 날짜 | 트랙 A (0S) | 트랙 B (0단계) |
|---|---|---|
| 10/18 밤 | S0 | — |
| 10/19 | S1, S1b, S2 시작 | P1 |
| 10/20 | S2 | P2(초안) |
| 10/21 | S3 | P3 |
| 10/22 | S3 | P4 |
| 10/23 | S4 | P5 |
| 10/24 | S4 | P6 |
| 10/25 | S5 마감 | — |
| 10/26~ | 판정에 따라 | P2 채택 반영, P7 |

0S는 10/25에 결과가 어디까지 왔든 마감한다. 트랙 B가 밀리는 것은 괜찮지만
트랙 A의 마감은 늦추지 않는다.
