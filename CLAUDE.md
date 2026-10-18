# CLAUDE.md

## 프로젝트
Korea Feature Manifold (KFM): 한국의 지형·인구·연결망을 비유클리드 기하(통행시간 준거리, 이산 곡률, 푸앵카레 임베딩) 위에서 다루는 국토 분석 baseline 프레임워크.

전체 설계는 `docs/DESIGN.md`에 있다. 작업 전 관련 절을 반드시 읽는다.

## 명령어
- 환경: `uv sync`
- 테스트: `uv run pytest`
- 린트/포맷: `uv run ruff check . && uv run ruff format .`

## 규약
- 좌표계: 저장·계산은 EPSG:5179, 웹 표시만 EPSG:4326
- 단위: 거리 km, 시간 min, 통행량 trips/day, 고도 m
- 지역 ID: `region_id`는 프레임워크 내부 고정 ID. 원 데이터 코드는 `source_code`에만 둔다.
- 저장 형식: GeoParquet, 질의는 DuckDB
- 거리 행렬은 비대칭을 보존한다. 대칭화가 필요하면 별도 컬럼/함수로 만든다.

## 규칙
- `data/raw/`는 불변. 수정·덮어쓰기 금지. `data/`는 git에 올리지 않는다.
- 계산 함수는 순수 함수로 작성하고, 입력 데이터를 제자리에서 바꾸지 않는다.
- 새 지표·기하 함수는 `tests/fixtures`의 합성 그래프로 먼저 테스트한다.
- 모든 실행은 `runs` 기록(config_hash, data_snapshot, code_version, seed)을 남긴다.
- 임베딩 산출물에는 왜곡 지표(DESIGN.md 6.6)를 반드시 함께 저장한다.
- 밀도 적응 배분에는 항상 최소 하한 `n_min`을 적용한다(DESIGN.md 7.3).
- 설계 변경 시 `docs/decisions/`에 결정 기록을 추가하고 DESIGN.md를 갱신한다.

## 현재 단계
0단계: 골격 + 0S 탐색 실험 (docs/PLAN.md 6.1·6.1a절, DESIGN.md 12장 참고)
- 0S(`notebooks/spike/`)는 버리는 탐색 코드로, `runs` 기록·스키마·fixture 선행 테스트 규칙의 예외다(PLAN 6.1a).
