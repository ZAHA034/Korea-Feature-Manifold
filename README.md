# Korea Feature Manifold (KFM)

한국의 지형·인구·연결망을 하나의 데이터 구조와 기하 구조로 묶은 국토 분석 기준선(baseline) 프레임워크다.
통행시간 준거리, 이산 곡률, 푸앵카레 임베딩으로 지역 간 연결 구조를 분석하고, 모든 분석을 기준선 대비 변화량으로 보고한다.
현재 0단계(골격)와 0S 탐색 실험을 진행 중이다.

## 설치

Python 3.11과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```powershell
uv sync
```

OneDrive 같은 동기화 폴더에서 작업하면 가상환경을 밖에 둔다(`$env:UV_PROJECT_ENVIRONMENT = "C:\dev\kfm-venv"`). VSCode 통합 터미널에서는 `.vscode/settings.json`으로 자동 적용된다.

Python은 3.11로 고정했다. 0S 도구가 3.11에서 검증되었으므로, 최신 scipy·numpy·networkx·pyproj가 요구하는 3.12로 올릴지는 0S가 끝난 뒤 정한다.

## 테스트와 린트

```powershell
uv run pytest            # 실데이터 테스트(@pytest.mark.data)는 기본 실행에서 제외된다
uv run pytest -m data    # 실데이터 테스트만 실행
uv run ruff check . && uv run ruff format .
```

## 문서

- 설계: [docs/DESIGN.md](docs/DESIGN.md)
- 실행 계획(v0.3 동결): [docs/PLAN.md](docs/PLAN.md)
- 설계 결정 기록(ADR): [docs/decisions/](docs/decisions/)
- 작업 규약: [CLAUDE.md](CLAUDE.md)
