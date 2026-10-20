# 0S 실행 환경과 원자료 목록

0S 노트북(`00`–`05`), `s4_collect.py`, `tools/make_s3_results.py`를 돌린 환경 `C:\dev\kfm-spike`의 정의와, 원자료 목록의 사본이다. 자료 파일은 저장소에 넣지 않는다. 이 폴더에서 uv를 실행하지 않는다(OneDrive 안에 `.venv`가 생긴다).

## 환경: `pyproject.toml`, `uv.lock`

- `C:\dev\kfm-spike`의 두 파일을 바이트 그대로 복사했다. 둘 다 LF다.
  - `pyproject.toml`: 755 B, SHA-256 `3ec011e4dc864785c9564f35fabe5c3cc2cd967ae7b078ebb16f2816d9bbf110`
  - `uv.lock`: 119,512 B, SHA-256 `d8e9a9d1e1ccedab07fe95b63f12ca0a3d3964c6b8e0fc9dd3a6be5fea7719d1`, MD5 `57ac9f9f3ea44d7dc2cc38ee88365f41`
- 원본은 2026-10-19 20:29 KST 판이다. S4에서 numba를 더한 직후의 lock이고, 03·04·05와 `s4_collect.py`가 이 lock으로 돌았다.
- 00·01·02와 `tools/make_s3_results.py`는 numba·llvmlite를 더하기 전 lock으로 돌았다. 그 lock은 남아 있지 않다. 다만 01 첫 셀이 찍은 15개 패키지 버전은 모두 이 lock과 같다(numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, pyarrow 26.0.0, geopandas 1.2.0, shapely 2.2.0, pyproj 3.7.2, rasterio 1.4.4, duckdb 1.5.6, POT 0.9.7.post1, torch 2.14.1, geoopt 0.5.1 등). torch는 00·03·04가 `2.14.1+cpu`로 찍는다.
- 기록 환경은 Windows(빌드 10.0.26200), CPython 3.11.9, uv 0.12.19다. lock의 `[[package]]` 항목은 134개(프로젝트 자신 1개와 전이 의존성 포함)이고 설치되는 패키지는 131개다. `environments = ["sys_platform == 'win32'"]`라서 Windows 전용이다.
- 환경을 바꾸면 이 사본도 같은 커밋에서 갱신한다.

### 다시 만들기

VSCode·Claude Code 터미널은 `UV_PROJECT_ENVIRONMENT=C:\dev\kfm-venv`를 자동으로 지정한다(CLAUDE.md). 이 변수가 있으면 `uv sync --project C:\dev\kfm-spike`가 spike lock을 본 프로젝트 가상환경에 동기화해 버리므로 먼저 지운다. 경로에 `kfm-spike`가 들어가야 한다(노트북이 `assert "kfm-spike" in sys.executable`로 확인한다).

```powershell
# 저장소 루트에서 실행한다
Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force C:\dev\kfm-spike | Out-Null
Copy-Item notebooks\spike\env\pyproject.toml, notebooks\spike\env\uv.lock C:\dev\kfm-spike\
uv sync --project C:\dev\kfm-spike --locked --python 3.11.9
```

2026-10-20에 다른 폴더에서 같은 방식으로 만들어 보니 lock이 바뀌지 않았고, 설치된 131개 패키지가 `C:\dev\kfm-spike\.venv`와 같았다.

### DuckDB spatial 확장 (처음 한 번)

01·02는 `LOAD spatial`만 부르고 설치하지 않는다. 확장은 가상환경이 아니라 사용자 프로필(`~\.duckdb\extensions\v1.5.6\windows_amd64\`)에 설치되어 `uv.lock`으로 고정되지 않는다. 새 기기에서는 먼저 설치한다.

```powershell
Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
uv run --locked --project C:\dev\kfm-spike python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL spatial'); c.execute('LOAD spatial'); print(duckdb.__version__, c.execute('FROM duckdb_extensions() SELECT extension_name, extension_version, installed_from WHERE extension_name = ?', ['spatial']).fetchall())"
```

0S를 돌릴 때의 빌드는 `04270fe`(DuckDB v1.5.6, windows_amd64, extensions.duckdb.org, 2026-10-18 설치)다. 출력이 `1.5.6 [('spatial', '04270fe', 'core')]`와 다르면 이 문서에 적고, 결과를 비교할 때 고려한다.

## 원자료 목록: `raw_manifest.csv`

- `C:\dev\kfm-data\raw\manifest.csv`의 2026-10-19 22:37 KST 사본이다. 41,324 B, MD5 `d32828bc91cb6e4659730357f1dde8e4`, UTF-8 BOM, CRLF이고 머리행 아래 55행이다. 줄바꿈이 바뀌면 MD5가 달라지므로 `.gitattributes`에서 이 파일을 `-text`로 둔다.
- 열은 `path`, `source_url`, `retrieved_at`, `md5`, `note`다. `note`에 받은 절차가 있다.
- 다시 돌릴 때는 이 파일을 `C:\dev\kfm-data\raw\manifest.csv`로 복사하고 각 파일을 `raw\<path>`에 둔다. 01은 manifest의 모든 행을 MD5로 확인하고 하나라도 다르거나 없으면 멈춘다. 그래서 01이 읽지 않는 섬 시간표 응답 8개, GPX, 인구이동 xlsx 등도 같은 바이트로 있어야 한다.
- 01을 돌렸을 때 manifest는 53행이었다. 그 뒤 S3에서 GPX, 사후 분석에서 인구이동 행을 더해 55행이다. 지금 01을 돌리면 manifest 표 출력이 55행으로 찍히고 계산은 같다.
- 노트북별로 읽는 자료: 01은 행정동 경계, 주민등록 2개, KOSIS 통근·통학 OD, KOSIS 코드 연계표, PBF, DEM 35개. 02는 PBF, KOSIS OD, GPX, DEM. 03은 KOSIS OD. 04는 인구이동 xlsx. 05는 PBF.

### 다시 받을 수 있는가

| 자료 | 받는 방법 | 같은 바이트인가 |
|---|---|---|
| 행정동 경계 | `source_url`이 커밋 `d5d5a821…`로 고정된 주소다 | 같다 |
| DEM 타일 35개 | `copernicus-dem-90m` 공개 S3 버킷. ETag가 MD5와 같다. 시도한 48개 중 13개는 404다(`download_log.tsv`) | 같다 |
| OSM PBF | `latest` 주소는 매일 바뀐다. 날짜 파일 `south-korea-261017.osm.pbf`(MD5 `355b718187224e6f85c7fd5b911c108f`)가 필요하다 | 그 파일이 남아 있어야 한다 |
| 주민등록 인구 | `note`의 POST 파라미터(세션 쿠키와 Referer `statMonth.do` 필요) | 확인하지 않았다 |
| KOSIS DT_1PA2021 | `note`의 대량통계 다운로드 번호 253513124 | 확인하지 않았다 |
| KOSIS 코드 연계표, 인구이동 | `source_url`에 간행물 위치만 있다. 정확한 다운로드 URL은 기록되지 않았다 | 받는 경로를 새로 적어야 한다 |
| 백두대간 GPX | 공공데이터포털 15098180에서 `note`의 절차로 받는다 | 확인하지 않았다 |
| 섬 시간표 8개 | 조회일 2026-10-19의 응답이다(파일명의 20261020은 조회한 운항일이다). JSON 3개는 세션 쿠키가 필요한 POST 응답이고 HTML 5개는 그날의 페이지 사본이다 | 다시 받을 수 없다. `island_links.csv` 값의 근거일 뿐 코드가 읽지 않는다 |
