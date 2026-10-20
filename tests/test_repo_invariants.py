"""저장소 불변 조건. 0S 증거물과 전력망 탐색의 경계를 커밋 전에 기계로 확인한다.

근거: ADR 0013 7항(사전 고정), ADR 0014 4항(전력망 결과는 저장소 밖).
git 이력과 태그를 쓰는 검사는 얕은 clone에서 건너뛴다. CI는 fetch-depth: 0으로 받고,
CI(환경 변수 CI)에서 git 명령이 실패하면 건너뛰지 않고 실패한다.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / "notebooks" / "spike"
POWER_DIR = "notebooks/spike/power"
POWER_ALLOWED = {f"{POWER_DIR}/power_lib.py", f"{POWER_DIR}/05_power_grid.ipynb"}
# 0S-end 뒤 그림·노트북에 출처 표기만 더한 커밋이다. 태그 0S-end 시점의 notebooks/spike/ 파일은
# 이 커밋 뒤로 바뀌면 안 된다. 기준 커밋을 바꿔야 하면(출처 표기를 또 더할 때 등)
# 이 값을 고치고 그 이유를 0S 보고서에 적는다.
EVIDENCE_BASE = "d588aab538c4c76ae3e4ea9388de0bc4c759f739"
# 0S-end 뒤 사후 탐색 분석(04_diagnostics.ipynb, diag_lib.py, diag_results.json, diag_outputs/)을
# 더한 커밋이다. 이 커밋에서 notebooks/spike/에 새로 들어온 파일도 이 커밋 뒤로 바뀌면 안 된다.
# 기준 커밋을 바꿔야 하면 EVIDENCE_BASE처럼 이 값을 고치고 그 이유를 0S 보고서에 적는다.
POSTHOC_BASE = "c6969b4bc6f8fd3b3af1c3549c8e51a316db788a"
# 증거 폴더: 파일을 더하거나 빼지 않는다(POSTHOC_BASE의 파일 집합과 같아야 한다)
EVIDENCE_DIRS = (
    "notebooks/spike/s3_outputs",
    "notebooks/spike/s4_outputs",
    "notebooks/spike/diag_outputs",
)
# 고정 기록(s3_freeze.json, s4_freeze.json)의 MD5를 저장소 파일로 다시 확인해야 하는 파일
FROZEN_IN_REPO = {
    "island_links.csv",
    "g_adj_edges.csv",
    "g_adj249_edges.csv",
    "bridges.csv",
    "node_matching.csv",
    "curvature_lib.py",
    "embed_lib.py",
    "s3_freeze.json",
}


def _git(*args: str) -> str:
    if shutil.which("git") is None:
        pytest.skip("git이 없다")
    res = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if res.returncode != 0:
        msg = f"git {args[0]} 실패(얕은 clone이면 태그와 이력이 없다): {res.stderr.strip()}"
        if os.environ.get("CI"):
            pytest.fail(msg)
        pytest.skip(msg)
    return res.stdout


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def test_power_notebook_has_no_outputs():
    nb = json.loads((ROOT / POWER_DIR / "05_power_grid.ipynb").read_text(encoding="utf-8"))
    dirty = [
        i
        for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count") is not None)
    ]
    assert dirty == [], f"출력이 남은 셀 {dirty}: 출력을 지우고 커밋한다(ADR 0014 4항)"


def test_power_dir_tracks_code_only():
    tracked = set(_git("ls-files", "--", POWER_DIR).splitlines())
    extra = sorted(tracked - POWER_ALLOWED)
    assert extra == [], f"전력망 결과는 저장소 밖에만 둔다(ADR 0014 4항): {extra}"


def test_data_dir_not_tracked():
    assert _git("ls-files", "--", "data").strip() == ""


def test_0s_evidence_unchanged():
    files = _git("ls-tree", "-r", "--name-only", "0S-end", "--", "notebooks/spike").splitlines()
    assert files, "태그 0S-end에 notebooks/spike/ 파일이 없다"
    changed = _git("diff", "--name-only", EVIDENCE_BASE, "--", *files).splitlines()
    assert changed == [], f"0S 증거물이 바뀌었다(설명은 문서로만 보탠다): {changed}"


def test_0s_posthoc_evidence_unchanged():
    """사후 분석 커밋이 더한 파일(04, diag_lib.py, diag_results.json, diag_outputs/)."""
    files = _git(
        "diff",
        "--name-only",
        "--diff-filter=A",
        f"{POSTHOC_BASE}^",
        POSTHOC_BASE,
        "--",
        "notebooks/spike",
    ).splitlines()
    assert files, "사후 분석 커밋에서 notebooks/spike/에 더한 파일이 없다"
    changed = _git("diff", "--name-only", POSTHOC_BASE, "--", *files).splitlines()
    assert changed == [], f"사후 분석 증거물이 바뀌었다(설명은 문서로만 보탠다): {changed}"


def test_evidence_dirs_keep_their_file_set():
    """증거 폴더에 새 파일(추적·미추적)이 생기거나 파일이 빠지지 않았다."""
    problems: list[str] = []
    for d in EVIDENCE_DIRS:
        want = set(_git("ls-tree", "-r", "--name-only", POSTHOC_BASE, "--", d).splitlines())
        assert want, f"{POSTHOC_BASE[:7]}에 {d}/ 파일이 없다"
        have = set(
            _git("ls-files", "--cached", "--others", "--exclude-standard", "--", d).splitlines()
        )
        problems += [f"새 파일 {p}" for p in sorted(have - want)]
        problems += [f"빠진 파일 {p}" for p in sorted(want - have)]
    assert problems == [], f"증거 폴더의 파일 집합이 바뀌었다: {problems}"


def test_frozen_files_match_recorded_md5():
    """고정 MD5는 바이트 기준이다(island_links.csv는 CRLF 값이라 .gitattributes로 CRLF로 푼다)."""
    checked: set[str] = set()
    wrong: list[str] = []
    for freeze in ("s3_freeze.json", "s4_freeze.json"):
        record = json.loads((SPIKE / freeze).read_text(encoding="utf-8"))
        for section in ("inputs_md5", "outputs_md5"):
            for name, want in record[section].items():
                path = SPIKE / name
                if isinstance(want, str) and path.is_file():
                    checked.add(name)
                    if _md5(path) != want:
                        wrong.append(f"{freeze} {section} {name}")
    assert wrong == [], f"고정 MD5와 다른 파일(줄바꿈을 확인한다, .gitattributes): {wrong}"
    assert checked >= FROZEN_IN_REPO, f"확인하지 못한 고정 파일: {sorted(FROZEN_IN_REPO - checked)}"
