# Korea Feature Manifold (KFM)

한국의 지형·인구·연결망을 하나의 데이터 구조와 기하 구조로 묶은 국토 분석 기준선(baseline) 프레임워크다.
통행시간 준거리, 이산 곡률, 푸앵카레 임베딩으로 지역 간 연결 구조를 분석하고, 모든 분석을 기준선 대비 변화량으로 보고한다.
현재 0단계(골격)를 진행 중이다. 0S 탐색 실험은 2026-10-19에 끝났다([보고서](docs/reports/0S-spike.md)).

## 설치

Python 3.11과 [uv](https://docs.astral.sh/uv/) 0.12.19 이상이 필요하다.

OneDrive 같은 동기화 폴더에서 작업하면 `uv sync` **전에** 가상환경을 밖으로 지정한다. 지정하지 않으면 저장소 안에 `.venv`가 생긴다(생겼으면 지우고 다시 한다).

```powershell
$env:UV_PROJECT_ENVIRONMENT = "C:\dev\kfm-venv"   # 현재 터미널에만 적용된다
uv sync
```

`.vscode/settings.json`과 `.claude/settings.local.json`은 git에서 제외되어 새로 받은 저장소에는 없다. 이 기기와 같게 쓰려면 직접 만든다.

```jsonc
// .vscode/settings.json
{
  "terminal.integrated.env.windows": { "UV_PROJECT_ENVIRONMENT": "C:/dev/kfm-venv" },
  "python.defaultInterpreterPath": "C:/dev/kfm-venv/Scripts/python.exe"
}
// .claude/settings.local.json
{ "env": { "UV_PROJECT_ENVIRONMENT": "C:/dev/kfm-venv", "UV_CACHE_DIR": "C:/dev/.uvcache" } }
```

Python은 3.11로 고정했다. 0S 도구가 3.11에서 검증되었으므로, 최신 scipy·numpy·networkx·pyproj가 요구하는 3.12로 올릴지는 0S가 끝난 뒤 정한다.

## 테스트와 린트

```powershell
uv run pytest            # 실데이터 테스트(@pytest.mark.data)는 기본 실행에서 제외된다
uv run pytest -m data    # 실데이터 테스트만 실행(아직 없어서 종료 코드 5가 난다)
uv run ruff check .
uv run ruff format .     # 파일을 고친다. 검사만 하려면 uv run ruff format --check .
```

Windows PowerShell 5.1은 `&&`를 지원하지 않는다. 이어서 실행하려면 `uv run ruff check .; if ($?) { uv run ruff format . }`를 쓴다.

## 문서

- 설계: [docs/DESIGN.md](docs/DESIGN.md)
- 실행 계획(v0.3 동결): [docs/PLAN.md](docs/PLAN.md)
- 구현 지시서와 진행 현황: [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)
- 설계 결정 기록(ADR): [docs/decisions/](docs/decisions/)
- 작업 규약: [CLAUDE.md](CLAUDE.md)
- 0S 탐색 실험 보고서: [docs/reports/0S-spike.md](docs/reports/0S-spike.md)
- 0S 합성 사전 검증: [docs/reports/0S-prereg-checks.md](docs/reports/0S-prereg-checks.md)
- 0S 노트북 실행 안내: [notebooks/spike/README.md](notebooks/spike/README.md)
- 자료 출처와 라이선스 고지(OpenStreetMap ODbL 포함): [NOTICE.md](NOTICE.md)
