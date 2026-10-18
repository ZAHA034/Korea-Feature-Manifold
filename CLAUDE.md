# CLAUDE.md

## 프로젝트
Korea Feature Manifold (KFM): 한국의 지형·인구·연결망을 비유클리드 기하(통행시간 준거리, 이산 곡률, 푸앵카레 임베딩) 위에서 다루는 국토 분석 baseline 프레임워크.

전체 설계는 `docs/DESIGN.md`에 있다. 작업 전 관련 절을 반드시 읽는다.

## 명령어
- 환경: `uv sync`
- 테스트: `uv run pytest`
- 린트/포맷: `uv run ruff check . && uv run ruff format .`
- 가상환경은 OneDrive 밖 `C:\dev\kfm-venv`에 있다. Claude Code는 `.claude/settings.local.json`, VSCode 터미널은 `.vscode/settings.json`으로 자동 지정된다(둘 다 git 제외). 저장소 안에 `.venv`가 생기면 잘못된 것이다.

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
- `docs/PLAN.md`는 v0.3에서 동결했다. 계획 변경은 `docs/decisions/`의 ADR로만 하고 PLAN 본문은 고치지 않는다.

## 현재 단계
0단계 + 0S 탐색 실험. 0S 판정 규칙(ADR 0010 2·5·6·7항, 0013) 채택 전에는 실데이터로 곡률·임베딩·δ를 계산하지 않는다.
구현은 `docs/IMPLEMENTATION.md`의 세션 블록 단위로 진행한다. 맨 위 "진행 현황" 표에서 다음 세션을 확인하고, 세션이 끝나면 표를 갱신한다.
