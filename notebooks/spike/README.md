# 0S 탐색 코드와 결과

0S 탐색 실험(2026-10-19)의 노트북, 라이브러리, 사전 고정 파일, 결과물이다. 판정과 해석은 [`docs/reports/0S-spike.md`](../../docs/reports/0S-spike.md)에 있다. `kfm` 패키지와 별개인 탐색 코드라서 `runs`·스키마·테스트 규칙 밖이다(0S는 PLAN 6.1a·ADR 0001 5항, `power/`는 ADR 0014 5항).

0S 증거물은 고치지 않는다. 범위는 두 묶음이다. 하나는 태그 `0S-end`의 `notebooks/spike/` 파일 27개이고, 그림 7장과 노트북 01·02·03에는 커밋 `d588aab`의 출처 표기가 더해져 있다. 다른 하나는 사후 분석으로 `c6969b4`(0S-end 뒤)에 더한 17개(`04_diagnostics.ipynb`, `diag_lib.py`, `diag_results.json`, `diag_outputs/`의 그림 4장·표 10개)다. `tests/test_repo_invariants.py`가 앞의 27개는 `d588aab`와, 뒤의 17개는 `c6969b4`와 같은지 확인하고, `s3_outputs/`·`s4_outputs/`·`diag_outputs/`의 파일 집합이 `c6969b4`와 같은지(새 파일이나 빠진 파일이 없는지)도 확인한다. `power/`와 이 README, `env/`·`tools/`는 0S-end 뒤에 더했고 증거물이 아니다.

| 폴더·파일 | 내용 |
|---|---|
| `env/` | 실행 환경(`pyproject.toml`, `uv.lock`)과 원자료 목록 사본. 환경 만들기와 원자료 받기는 `env/README.md` |
| `tools/` | `s3_results.json`과 그림의 출처 띠를 만든 일회용 스크립트 |
| `power/` | 전력망 탐색(ADR 0014). 코드 두 파일만 두고 결과는 저장소 밖 `C:\dev\kfm-data\power\`에 쓴다. `05_power_grid.ipynb` 머리말 셀(셀 1)의 "(PLAN 6.1a)"는 실행 당시(`c8de6c2`) 문구이고, 지금 근거는 ADR 0014 5항이다 |

## 재현 수준

- **A. 합성만.** `00_tree_baseline.ipynb` 전체. `03_embedding.ipynb`는 설정 셀과 1절(`# 1절 요약` 셀까지)이 자료를 읽지 않으므로 `C:\dev\kfm-data\spike` 폴더만 있으면 Jupyter에서 차례로 돌릴 수 있다. `02_curvature.ipynb`의 1절은 설정 셀이 원자료 MD5와 GPX를 먼저 읽으므로 원자료가 있어야 한다. 2026-10-20에 00을 아래 명령으로 다시 돌렸더니 저장 출력과 다른 것은 실행 날짜와 소요 시간(110.8 s)뿐이었다. 03의 위 범위도 같은 날 다시 돌려 합성 검산 43개가 통과했다(저장 출력 87 s, 이번 92 s).
- **B. 고정 재생.** 소유자의 원자료 55개와 `C:\dev\kfm-data\spike` 중간 산출물이 있어야 한다. 02·03을 그대로 돌리면 2절이 `검증: …(기존 파일과 같다)`를 찍고, 04는 s4 행렬의 MD5를 `s4_freeze.json`과 맞춰 본다. 3절 수치는 저장된 출력과 같아야 한다. 02의 3절은 같은 기기에서 확인했고(`tools/make_s3_results.py`), 다른 기기에서 비교해 본 적은 없다. 01은 중간 산출물을 새로 쓰므로 B에서 돌리지 않는다.
- **C. 원자료부터.**
  - `travel_time.parquet`(01 11-3)와 `travel_time_island_x0.5`·`x2.parquet`(02 2-5)는 실행 시각을 parquet 메타데이터(`created`)에 넣는다(각각 2026-10-19T17:54:47, 19:20:34). 그래서 원자료에서 다시 만들면 파일 MD5가 `s3_freeze.json`의 값(`6a9e34cb…`, `7c751dfd…`, `8c3c4ca4…`)과 반드시 달라지고, 02(2-10, 3-1a)와 03(2-0)이 고정 확인에서 멈춘다. 고정 확인으로서는 맞는 동작이다.
  - 다시 만든 결과가 같은지는 내용으로 비교한다. 01 11-3 출력의 행렬 해시(`T 1f370fd917c042395d61a572e0e4e564`, `d_sym 2643537bfd50c5c71dcbf277c6d4faf2`)와 같은지 보고, 다르면 값의 최대 절대 차이를 본다(02 2-5가 쓰는 허용 오차는 1e-9분이다).
  - 02·03을 이어서 돌리려면 임시 clone의 임시 브랜치에서 고정 파일(`g_adj_edges.csv`, `g_adj249_edges.csv`, `bridges.csv`, `node_matching.csv`, `s3_freeze.json`, `s4_freeze.json`)을 지우고 다시 만든 뒤 커밋해야 한다(03 3-1이 커밋된 상태를 요구한다). 그다음 원래 커밋과 `git diff`로 비교한다. 예상되는 차이는 parquet MD5 줄뿐이다. 이 절차는 아직 실행해 보지 않았다.

사전 검증(합성) 스크립트는 [`docs/reports/0S-prereg/`](../../docs/reports/0S-prereg/README.md)에 있고, 중간 결과가 없어 사본만으로는 다시 돌릴 수 없다.

## 준비

- Windows 전용이다. `C:\dev\kfm-data\{raw,spike,power}` 경로가 코드에 고정돼 있다. 자료를 다른 곳에 두면 `New-Item -ItemType Junction -Path C:\dev\kfm-data -Target <실제 폴더>`로 잇는다.
- 환경은 `env/README.md`대로 만든다. 경로에 `kfm-spike`가 있어야 노트북의 assert를 통과한다. `UV_PROJECT_ENVIRONMENT`는 먼저 지운다(VSCode·Claude Code 터미널이 이 변수를 `C:\dev\kfm-venv`로 지정해 두므로, 지우지 않으면 spike lock이 본 프로젝트 가상환경에 동기화된다).
- 원자료는 `env/README.md`의 목록(`raw_manifest.csv`)대로 `C:\dev\kfm-data\raw\`에 둔다.
- DuckDB spatial 확장을 한 번 설치한다(`env/README.md`). 01·02는 설치하지 않고 불러오기만 한다.
- `git`이 PATH에 있어야 한다(03·04·05가 부른다). 01·02·04(`diag_lib.py`)·05와 `s4_collect.py`는 matplotlib 글꼴로 Malgun Gothic을 쓴다.
- 작업 폴더는 노트북이 있는 폴더다. `Path.cwd()`로 `*_lib.py`를 찾는다(05는 `power/`).

## 순서

| 순서 | 입력 | 출력 | 기록된 시간 |
|---|---|---|---|
| 00 tree (S1b) | 합성 자료만 | 노트북 출력만 | 108 s |
| 01 (S2) | 원자료(행정동 경계, 주민등록 2개, KOSIS OD·코드 연계표, PBF, DEM)와 `island_links.csv` | `C:\dev\kfm-data\spike\`의 `regions`·`origins`·`origin_nodes`·`road_nodes`·`road_edges`·`travel_time`(.parquet)과 `connectors_over5km.csv` | 약 1분 |
| 02 (S3) | 원자료(PBF, KOSIS OD, GPX, DEM)와 위 중간 산출물 | 저장소의 `g_adj_edges.csv`·`g_adj249_edges.csv`·`bridges.csv`·`node_matching.csv`·`s3_freeze.json`(이미 있으면 바이트 비교만)과 spike의 `travel_time_island_x0.5`·`x2`·`baekdudaegan_line`(.parquet, 이미 있으면 값 비교만) | 108 s |
| 2b `tools/make_s3_results.py` | 02의 저장 출력, 원자료, spike | `s3_results.json`, `s3_outputs/`. 경로가 원래 저장소로 고정돼 있어 임시 clone에서 돌려도 증거물을 덮어쓴다(아래 "덮어쓰는 것") | 7 s |
| 03 (S4) | KOSIS OD, 02의 고정 파일, spike | `C:\dev\kfm-data\spike\s4\`(행렬, `s4_real\`, `restart_cache\`)와 `s4_freeze.json`(바이트 비교만) | 729 s (`N_WORKERS` 10, 재시작 계산 시간 합 5,331 s) |
| 3b `s4_collect.py --wall-seconds 734` | `s4\s4_real\` | `s4_results.json`, `s4_outputs/*.png` | |
| 04 (사후 분석) | `s4\`, 인구이동 xlsx | `diag_results.json`, `diag_outputs/`, `spike\diag\` | 98 s (재시작 캐시가 있을 때). 캐시 없는 첫 계산은 약 2,269 s이고 그중 C·D2 임베딩이 2,178 s |
| 05 (전력망) | PBF, `spike\regions.parquet` | `C:\dev\kfm-data\power\`에만 쓴다 | |

## 명령

노트북 하나를 확인용으로 돌리는 명령이다. `00_tree_baseline.ipynb`를 01–04로 바꿔 같은 방식으로 돌린다.

```powershell
# 저장소 루트에서 시작한다
Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
cd notebooks\spike
uv run --locked --project C:\dev\kfm-spike jupyter nbconvert --to notebook --execute --output-dir C:\dev\kfm-data\rerun --ExecutePreprocessor.timeout=36000 00_tree_baseline.ipynb
```

- `--inplace`는 쓰지 않는다. 고정 노트북의 저장 출력이 덮인다. 02·03은 그래도 저장소의 고정 CSV·JSON과 바이트를 비교한다.
- 타임아웃은 02 머리말이 `3600`, 03 머리말이 `36000`이다. 코어가 적은 기기에서는 03·04의 한 셀이 3,600 s를 넘을 수 있고, 03의 `N_WORKERS = 10`은 노트북에 들어 있어 바꿀 수 없으므로 모두 `36000`을 쓴다.
- 04는 `KFM_DIAG_WORKERS`로 프로세스 수를 조정한다(기본 10). `KFM_DIAG_SMOKE=1`은 배선 점검 모드로, 반복 수를 줄이고 산출물을 `KFM_DIAG_SCRATCH`가 가리키는 임시 폴더에 둔다.
- 05는 `cd notebooks\spike\power` 뒤 `--output-dir C:\dev\kfm-data\power --output 05_power_grid.executed.ipynb 05_power_grid.ipynb`로 돌린다. 결과는 저장소 밖에만 둔다.
- 02·03 머리말의 `RUN_… = False`는 고정 커밋 때의 상태다. 커밋된 실행본은 설정 셀이 `True`이고 3절 출력이 있다.

### 덮어쓰는 것

확인용 실행은 임시 clone에서 하고 `git diff --stat`으로 원래 커밋과 비교한다(`tools/`의 두 스크립트는 예외다. 아래). 소유자의 `C:\dev\kfm-data`가 있는 기기에서는 그 폴더를 복사하고, 원본의 이름을 바꾼 뒤, 복사본을 `C:\dev\kfm-data`로 이어(Junction) 돌린다. 아래는 다시 돌리면 덮어쓴다.

- 01: `C:\dev\kfm-data\spike\`의 parquet 6개와 `connectors_over5km.csv`. `travel_time.parquet`의 MD5가 달라져 02·03의 고정 확인이 멈춘다.
- 03: `s4\s4_real\`(`runs_record.json` 포함). `s4_collect.py`가 이것을 읽는다.
- 04: 저장소의 `diag_results.json`·`diag_outputs/`와 `spike\diag\`.
- `s4_collect.py`: 저장소의 `s4_results.json`·`s4_outputs/`. `tools/make_s3_results.py`: `s3_results.json`·`s3_outputs/`. 띠가 있는 그림을 다시 얻으려면 이어서 `tools/attrib_odbl_20261019.py`를 돌린다(`tools/README.md`).
- `tools/`의 두 스크립트는 원래 저장소 경로(`NB_DIR`·`BASE`)가 고정돼 있어 임시 clone에서 돌려도 원래 저장소의 증거물을 덮어쓴다. `make_s3_results.py`는 `s3_results.json`(시간 필드)과 `s3_outputs/`를 쓰고 그림 3장을 띠 없는 02 저장 출력으로 되돌린다. `attrib_odbl_20261019.py`는 그림 7장에 띠를 한 번 더 겹친다. `tools/README.md`대로 `NB_DIR`·`BASE`를 clone 경로로 바꾼 사본으로만 돌린다.

## 설정·시드 기록 위치

- 00: 첫 셀의 `CFG`(시드 0–9, 판정 시드 0–4).
- 01: 첫 코드 셀의 `ASSUMPTIONS`, 9절의 `MAP_TIME_MIN`, `travel_time.parquet` 메타데이터.
- 02: `CONFIG`는 `s3_freeze.json`의 `config`다. 합성 검산(1절)만 고정 시드의 난수를 쓰고 실데이터 계산에는 난수가 없다.
- 03: `s4_freeze.json`의 `config`(`seed_rules`와 `config_hash` `c2ae3329a5afc3d944bb26ba125c1ad4` 포함). 저장소 밖 `runs_record.json`의 내용은 `s4_freeze.json`(`config`, `inputs_md5`, `outputs_md5`)과 `s4_results.json`(`git_head_at_run`, `embed_lib_md5`, `run`)에 들어 있다.
- 04: `diag_results.json`의 `meta`.
- 05: `C:\dev\kfm-data\power\power_run_meta.json`(저장소 밖).

## 고정 MD5와 줄바꿈

`s3_freeze.json`·`s4_freeze.json`의 MD5는 바이트 기준이다. 고정 파일 가운데 `island_links.csv`만 고정 시점의 작업 트리가 CRLF였고, 기록값(`e8db8abad2580da7161f894488ae8ae8`)은 CRLF 바이트로 구했다. 저장소 blob은 LF(MD5 `0a04b04c8cf3fed67f2f956aec33e6b4`)이므로 `.gitattributes`가 이 파일만 CRLF로 푼다. 새 clone에서 `git ls-files --eol notebooks/spike/island_links.csv`가 `w/crlf`이고 MD5가 기록값과 같아야 02(2-10, 3-1a)와 03(2-0)의 고정 확인을 통과한다. `tests/test_repo_invariants.py`가 이를 확인한다.

04가 `diag_results.json`에 적은 `diag_lib.py`의 MD5(`44c0dffea3c8f95745a8184545f63936`)도 CRLF 작업 트리의 값이다. 저장소 파일(LF)은 `778fcbb5bc76e17889e6f9782b30b4b4`이고 내용은 같다. 04는 이 값을 확인하지 않는다.

05가 저장소 밖 `power_run_meta.json`에 적은 `notebook_sha256`(`8fdac1e3…`)과 `power_lib_sha256`(`513744a0…`)도 CRLF 작업 트리의 값이다. 저장소 파일(LF)의 SHA-256은 각각 `9489c070…`, `903075fc…`이고 내용은 같다(노트북을 CRLF로 바꾸면 기록값이 나온다. 2026-10-20 확인). 그래서 커밋된 05 노트북(`c8de6c2`)은 실행 때 쓴 것과 내용이 같다. 05는 이 값을 확인하지 않는다.

## 이력 (KST)

| 커밋 | 시각 | 내용 |
|---|---|---|
| `207704a` (태그 `0S-start`) | 10-19 10:08 | S1 판정 규칙 채택 |
| `aa718a8` | 11:02 | S1b: 00 |
| `e527fc1`, `4f646e5`, `bc4cab9` | 13:43–17:55 | S2: 01 |
| `29cc115` | 18:17 | ADR 0013 개정 |
| `b2c7ca2` | 19:40 | S3 고정 |
| `48fab3c` | 19:49 | S3 실행. 02의 셀 소스는 `RUN_REAL_CURVATURE` 한 줄만 바뀌었다 |
| `c901dae` | 21:29 | S4 고정 |
| `aaa416f` | 21:49 | S4 실행. 03의 셀 소스는 `RUN_REAL` 한 줄만 바뀌었다. `s4_collect.py`를 더했다 |
| `a2055c2` (태그 `0S-end`) | 21:58 | 0S 보고서 마감 |
| `d588aab` | 22:08 | 계산과 무관. 01·02·03 맨 앞에 출처 고지 셀을 더해 셀 번호가 하나씩 밀렸고, 그림 7장 하단에 출처 띠를 더했다(`tools/attrib_odbl_20261019.py`) |
| `c6969b4` | 23:55 | 04 사후 분석 |
| `c8de6c2` | 10-20 17:49 | 05 전력망(ADR 0014) |

0S-end판 그림은 Git Bash에서 `git show 0S-end:notebooks/spike/s3_outputs/figures/map_kappa.png > map_kappa_0S-end.png`로 꺼낸다. PowerShell 5.1의 `>`는 PNG를 깨뜨린다.
