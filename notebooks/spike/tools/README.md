# 0S 결과물 후처리 도구

0S 결과물을 만든 일회용 스크립트 두 개다. 만든 세션의 임시 폴더에만 있던 것을 2026-10-20에 바이트 그대로 옮겼다. 고치지 않는다.

| 파일 | 만든 것 | 실행 | MD5 (원본과 같다) |
|---|---|---|---|
| `make_s3_results.py` | `s3_results.json`, `s3_outputs/`의 CSV 4개와 그림 3장 (커밋 `48fab3c`) | 2026-10-19 19:46, 7.3 s | `440658709e50e32d29bf0975f247b130` |
| `attrib_odbl_20261019.py` | 그림 7장 하단의 출처 띠와 01·02·03 노트북 맨 앞의 출처 고지 셀 (커밋 `d588aab`) | 2026-10-19 22:07, 한 번 | `9ad4cb6a3d050a07d5efa1e4beed17a0` |

## make_s3_results.py

- `02_curvature.ipynb`의 설정 셀과 3-1a–3-7 셀(3-5 제외)을 다시 실행해 결과 객체를 얻는다. 셀 출력 7개가 노트북 저장 출력과 같지 않으면 멈춘다.
- 이 객체로 `s3_results.json`(`run.note`에 이 방식을 적었다)과 `s3_outputs/*.csv` 4개를 쓰고, 그림 3장은 3-5 셀에 저장된 PNG 출력을 꺼내 쓴다.
- `C:\dev\kfm-spike` 환경에서 돌리고, 원자료(`C:\dev\kfm-data\raw`)와 `C:\dev\kfm-data\spike`가 있어야 한다.

## attrib_odbl_20261019.py

- 그림 7장(`s3_outputs/figures/` 3장, `s4_outputs/` 4장) 하단에 출처 띠를 그리고(Pillow, `C:\Windows\Fonts\arial.ttf`), 01·02·03 노트북 맨 앞에 출처 고지 셀을 넣는다.
- 한 번만 돌린다. 노트북은 고지 셀이 이미 있으면 건너뛰지만 그림은 건너뛰지 않아, 다시 돌리면 띠가 겹친다.

## 다시 돌릴 때

두 스크립트는 저장소의 원래 위치(`C:/Users/mycoms/OneDrive/Desktop/Korea Feature Manifold/notebooks/spike`)를 `NB_DIR`(`make_s3_results.py`)·`BASE`(`attrib_odbl_20261019.py`)에 고정해 두고, 그 위치의 증거 파일을 덮어쓴다. 그대로 돌리지 않는다. 임시 clone에서 이 상수만 clone 경로로 바꾼 사본을 `C:\dev\kfm-spike` 환경으로 돌리고, `git diff`로 원래 커밋과 비교한다.
