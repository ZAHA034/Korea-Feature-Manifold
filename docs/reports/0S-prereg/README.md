# 0S 사전 검증 스크립트와 전체 결과

[`0S-prereg-checks.md`](../0S-prereg-checks.md)(0S 판정 규칙의 합성 자료 점검, 2026-10-18)의 근거 스크립트와 전체 결과다. ADR 0013 채택(기준 차분 곡률 검정, R_max 5)의 근거였고, `notebooks/spike/00_tree_baseline.ipynb`가 맞춰 보는 기댓값(쌍곡 0.033939, 비율 0.2253)도 여기서 나왔다.

- `C:\dev\kfm-data\0S-prereg\`(git 밖)의 2026-10-18 실행본을 2026-10-20에 옮겼다. `bar/`는 장벽과 곡률(보고서 1절), `rcap/`는 쌍곡 반지름 상한(2절), `results_full.md`는 전체 결과다. 합성 자료만 쓰고, 실데이터를 읽거나 절대 경로를 쓰는 코드는 없다.
- `results_full.md` 41·125·148행의 설명 글에는 사용자 이름이 든 임시 폴더 절대 경로가 있다(위 "절대 경로를 쓰는 코드는 없다"는 코드만 말한다). 원문 그대로 두고, 공개 전 점검 목록(`NOTICE.md`)에 올렸다.
- 줄바꿈은 저장소 규칙(LF)에 맞췄다. 원본에서 CRLF였던 7개 파일(`results_full.md`, `rcap/capemb.py`, `rcap/driver.py`, `rcap/*.json` 4개)의 줄바꿈만 바꿨다.
- **사본만으로는 분석을 다시 돌릴 수 없다.** 분석 스크립트(`bar/analyze.py`, `bar/eval*.py`, `rcap/analyzeA.py` 등)가 읽는 중간 결과(`res/m*.npz`, `res_n0/m*.npz`, `*.jsonl`)가 이 폴더에 없다. 중간 결과를 만드는 `bar/run.py`, `bar/run_n0.py`, `rcap/driver.py`는 있지만, 실제로 쓴 인자(시드 범위, plan 이름)와 실행 환경은 기록되지 않았다. `results_full.md` 148행에는 환경이 임시 폴더 옆 `fixtures71\.venv-torch`라는 것과, 원본 raw 결과가 당시 세션의 임시 폴더에 있었다는 것만 적혀 있다.
- 결과를 확인할 때는 `results_full.md`와 `0S-prereg-checks.md`를 읽는다. 스크립트는 계산 방법의 기록이다.
