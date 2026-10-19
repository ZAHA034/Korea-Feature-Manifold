"""KFM 0S-4 임베딩·δ·d_flow·중력 대조군 판정 계산 코드 (ADR 0010 2·5·6·7항, ADR 0013 4항, PLAN 3-3~3-5).

0S 탐색 코드라서 kfm 패키지·runs·스키마 규칙 밖이다(PLAN 6.1a). 실행 환경은 C:\\dev\\kfm-spike.
순수 함수를 둔다(입력 배열을 제자리에서 바꾸지 않는다). 노트북 03_embedding.ipynb가 이 모듈을 불러 쓴다.

구성
  1. 임베딩 최적화: 상대 stress(닫힌 해 척도), 2단계 최적화(RiemannianAdam -> L-BFGS-B), 재시작 규약, 재실행 규칙
  2. 요약과 판정: 쌍곡 이점, 평탄 극한, 최적화 실패, 보류, d_flow와 대조군 비교, 중력모형 이상의 위계
  3. δ 진단: 서로 다른 네 점 정확 계산(numba), 기준점 (max, min) 행렬곱
  4. d_flow: ell_sym, ell', 최단경로 폐포, 추정 OD 거부
  5. 중력 대조군: 이중 제약 지수 모형 적합(Furness), 다항 재표본
  6. 입력 준비 보조: 249개 단위 변환, 섬 연결 지역, OD 제외 지역
  7. 합성 fixture
  8. 3절 드라이버(실데이터 실행은 노트북의 RUN_REAL 플래그가 켜질 때만)

결과 재현성: PYTORCH_JIT=0, float64, 프로세스당 torch 1스레드. 재시작 결과는 (D, 공간, 초기값, 설정)으로만 정해지며 프로세스 이력이나 병렬 배치에 의존하지 않는다.

재시작과 재실행 (ADR 0013 4항 + 2026-10-19 보강. 노트북 설정 셀과 s4_freeze.json에 같은 문구를 남긴다)
  - 첫 시도: 유클리드 5회(고전 MDS 1 + U(±0.1) 시드 0..3) + 쌍곡 5회(유클리드 해를 줄인 초기값 2 + U(±0.1) 시드 0..2).
  - 실패(쌍곡 최소 stress > 유클리드 최소 stress x (1 + 1e-3))이면 쌍곡의 U(±0.1) 재시작 3개만 U 시드에 +5를 더해 다시 돌리고,
    그래도 실패이면 +10으로 한 번 더 돌린다(최대 2회). 유클리드 결과와 줄인 초기값 2개는 다시 돌리지 않고 그대로 둔다.
  - 쌍곡은 모든 시도의 재시작을 합쳐 판정한다. 실패 판정은 시도마다 그때까지 합친 쌍곡 최솟값으로 한다.
    두 번 재실행해도 실패이면 기록은 "최적화 실패", 쌍곡 이점은 "보류"다.
  - 실패와 평탄 극한의 구분은 마무리 수렴 여부와 관계없는 최소 stress로 하고(ADR 0013: 평탄 극한 기록은 수렴을 요구하지 않는다),
    쌍곡 이점 판정은 합친 재시작 중 마무리가 수렴한 것의 최소 stress 재시작으로 한다.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"  # numpy, torch를 불러오기 전에 고정한다(자식 프로세스는 이 환경변수를 물려받는다)
# geoopt가 쓰는 torch.jit.script의 프로파일링 실행기는 호출 순서(프로세스 이력)에 따라 부동소수점 마지막 자리를 바꾸고,
# Adam 수천 회를 거치며 반복 수와 마무리 값이 1e-7 규모로 달라진다(같은 시드의 첫 재시작과 둘째 재시작이 다름).
# JIT를 꺼서 재시작 결과가 시드와 입력으로만 정해지게 한다. torch를 불러오기 전에 정해야 한다.
os.environ["PYTORCH_JIT"] = "0"

import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

# geoopt가 쓰는 torch.jit.script에 대한 torch의 사용 중단 경고만 거른다(동작에 영향 없음)
warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)

import geoopt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import brentq, minimize  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

try:  # numba는 kfm-spike 환경에만 있다
    import numba
    from numba import njit, prange
except ImportError:  # pragma: no cover
    numba = None

if torch.jit._state._enabled.enabled:  # pragma: no cover
    raise RuntimeError("torch를 embed_lib보다 먼저 불러와 JIT가 켜져 있다. torch를 불러오기 전에 PYTORCH_JIT=0을 정해야 한다")
torch.set_num_threads(1)
# geoopt 0.5.1의 PoincareBall은 곡률 텐서를 기본 dtype(float32)으로 만든다. 그러면 float64 좌표에서도
# 경계 근처 거리에 1e-6 규모의 상대오차가 생기므로(S1b 사전 점검) 기본 dtype을 float64로 올린다.
torch.set_default_dtype(torch.float64)
BALL = geoopt.PoincareBall(c=1.0)  # 곡률 -1

# ----------------------------------------------------------------------------- 기록 문구
ADV_YES = "쌍곡 이점 있음"
ADV_NO = "쌍곡 이점 없음"
ADV_FLAT = "쌍곡 이점 없음(평탄 극한)"
ADV_HOLD = "보류"
REC_FAIL = "최적화 실패"
NORMS = ("main", "alt")  # 주 정규화(ell_sym), 대안 정규화(ell')
SUBSETS = ("전체", "섬 제외")

CFG = {
    "dim": 2,
    "init_halfwidth": 0.1,  # U(±0.1)
    "adam_lr": 0.05,
    "adam_window": 200,  # L*_t가 200회 동안 상대 개선 1e-6 이하이면 멈춘다
    "adam_tol": 1e-6,
    "adam_max_iter": 50_000,
    "r_max": 5.0,  # 쌍곡 반지름 상한(곡률 -1). 매 단계 뒤 노름 <= tanh(r_max / 2)로 투영
    "fin_ftol": 1e-12,
    "fin_gtol": 1e-10,  # ADR 미지정. ftol이 실제 종료 기준이 되게 한다(S1b 메모)
    "fin_maxiter": 20_000,
    "fin_maxfun": 50_000,  # ADR 미지정. scipy 기본값 15000이 maxiter 20000보다 작아 maxiter가 닿지 못한다
    "mds_max_norm": 0.1 * math.sqrt(2),
    "m1": None,  # 줄인 유클리드 해의 최대 노름. 노트북 설정 셀에서 정한다(ADR 0013 4항)
    "m2": None,
    "n_euc_u": 4,  # 유클리드: 고전 MDS 1 + U 4
    "n_hyp_u": 3,  # 쌍곡: 줄인 초기값 2 + U 3
    "retry_offsets": (5, 10),  # 최적화 실패 시 쌍곡 U 재시작 3개의 시드에 더하는 값(유클리드·줄인 초기값은 다시 돌리지 않음)
    "flat_rtol": 1e-3,  # 쌍곡 최솟값 > 유클리드 최솟값 x (1 + 1e-3)이면 실패
    "ratio_max": 0.8,  # 쌍곡 이점: 평균 상대 왜곡 비 <= 0.8
    "n_worst": 20,
    "tie_rtol": 1e-12,  # 세 쌍 합이 같다고 보는 상대 오차(S1 - S3 <= tie_rtol x S1이면 0으로 둔다)
    "resample_frac": 0.2,
    "resample_scale": 5,
}
RESAMPLE_STREAM = 20261019  # 재표본 난수열의 식별자. 임베딩 재시작은 default_rng(seed)를 쓰므로 서로 다른 난수열이다


def check_cfg(cfg):
    """m1, m2가 정해졌고 0.1*sqrt(2) <= m1 < m2 < tanh(r_max / 2)인지 확인한다."""
    cap = math.tanh(cfg["r_max"] / 2)
    if cfg.get("m1") is None or cfg.get("m2") is None:
        raise ValueError("cfg['m1'], cfg['m2']를 노트북 설정 셀에서 정해야 한다(ADR 0013 4항)")
    if not (cfg["mds_max_norm"] - 1e-15 <= cfg["m1"] < cfg["m2"] < cap):
        raise ValueError(f"0.1*sqrt(2) <= m1 < m2 < tanh(r_max/2) = {cap:.6f} 이어야 한다: m1={cfg['m1']}, m2={cfg['m2']}")
    return cfg


def config_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ===================================================================== 1. 임베딩 최적화
class Problem:
    """목표 거리 행렬 D(대칭, 대각 0, 나머지 양수)의 상삼각 쌍 i<j에 대한 상대 stress 문제."""

    def __init__(self, D):
        D = np.asarray(D, dtype=np.float64)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D는 정사각 행렬이어야 한다")
        n = D.shape[0]
        i, j = np.triu_indices(n, 1)
        d = D[i, j]
        if not np.isfinite(d).all() or d.min() <= 0:
            raise ValueError("D의 서로 다른 쌍은 모두 양의 유한값이어야 한다(상대 stress는 d로 나눈다)")
        if np.abs(D - D.T).max() > 1e-9 * d.max():
            raise ValueError("D가 대칭이 아니다. 대칭화는 호출하는 쪽에서 별도로 한다")
        self.n = n
        self.i, self.j = torch.as_tensor(i), torch.as_tensor(j)
        self.d = torch.as_tensor(d, dtype=torch.float64)

    def dist(self, X, space):
        if space == "poincare":
            return BALL.dist(X[self.i], X[self.j])
        return (X[self.i] - X[self.j]).norm(dim=-1)

    def stress(self, X, space):
        """(상대 stress, s*). s*는 계산 그래프에서 떼어 낸다(포락선 정리로 기울기는 정확하다)."""
        p = self.dist(X, space)
        q = p / self.d
        s = (q.sum() / (q * q).sum()).detach()
        return (((s * p - self.d) / self.d) ** 2).sum(), s

    def metrics(self, X, space, r_max=None):
        """상대 stress, 닫힌 해 척도, 평균 상대 왜곡 mean|s*p - d|/d, 최대 상대 오차. 쌍곡이면 최대 반지름과 상한 점 수."""
        X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        with torch.no_grad():
            p = self.dist(X, space)
            q = p / self.d
            s = q.sum() / (q * q).sum()
            rel = (s * p - self.d).abs() / self.d
            out = {
                "stress": float((rel**2).sum()),
                "scale": float(s),
                "distortion": float(rel.mean()),
                "max_rel_err": float(rel.max()),
            }
        if space == "poincare":
            r = hyp_radius(X.numpy())
            out["max_radius"] = float(r.max())
            if r_max is not None:
                out["n_at_cap"] = int((r >= r_max - 1e-9).sum())
        return out


def hyp_radius(X):
    """원점 기준 쌍곡 반지름 2 artanh(|x|) (곡률 -1)."""
    norm = np.linalg.norm(np.asarray(X), axis=1)
    return 2.0 * np.arctanh(np.minimum(norm, 1.0 - 1e-16))


def np_distortion(D, X, space):
    """torch, geoopt와 별개인 numpy 구현(교차 점검용). 원판 거리는 arccosh 공식. (평균 상대 왜곡, stress)를 돌려준다."""
    D = np.asarray(D, dtype=np.float64)
    i, j = np.triu_indices(len(D), 1)
    diff2 = ((X[i] - X[j]) ** 2).sum(1)
    if space == "poincare":
        alpha = 1.0 - (X**2).sum(1)
        p = np.arccosh(1.0 + 2.0 * diff2 / (alpha[i] * alpha[j]))
    else:
        p = np.sqrt(diff2)
    d = D[i, j]
    q = p / d
    s = q.sum() / (q * q).sum()
    rel = np.abs(s * p - d) / d
    return float(rel.mean()), float((rel**2).sum())


def init_uniform(n, seed, cfg=CFG):
    """U(±h)^dim 초기값. 같은 시드는 두 공간에서 같다."""
    h = cfg["init_halfwidth"]
    return np.random.default_rng(seed).uniform(-h, h, size=(n, cfg["dim"]))


def classical_mds(D, dim=2):
    """Torgerson 고전 MDS. 양의 고유값 dim개가 없으면 오류."""
    D = np.asarray(D, dtype=np.float64)
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(w)[::-1][:dim]
    if np.any(w[order] <= 0):
        raise ValueError(f"고전 MDS의 상위 {dim}개 고유값이 양수가 아니다: {w[order]}")
    return V[:, order] * np.sqrt(w[order])


def shrink_to_max_norm(X, m):
    """무게중심을 원점으로 옮긴 뒤 최대 노름이 m이 되게 균등 축소(또는 확대)한다."""
    Xc = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0)
    return Xc * (m / np.linalg.norm(Xc, axis=1).max())


def init_mds(D, cfg=CFG):
    """고전 MDS를 최대 노름 0.1*sqrt(2)로 줄인 초기값."""
    return shrink_to_max_norm(classical_mds(D, cfg["dim"]), cfg["mds_max_norm"])


def run_adam(prob, space, X0, cfg=CFG):
    """1단계: geoopt RiemannianAdam. 정지 시점의 좌표(마지막 값)를 돌려준다.
    t회까지의 최소 stress L*_t가 (L*_{t-200} - L*_t) / L*_{t-200} <= 1e-6이면 멈추고, 최대 50,000회. 쌍곡은 매 단계 뒤 반지름 상한을 투영한다."""
    t0 = time.perf_counter()
    X0 = torch.tensor(np.asarray(X0), dtype=torch.float64)
    X = geoopt.ManifoldParameter(X0, manifold=BALL) if space == "poincare" else torch.nn.Parameter(X0)
    opt = geoopt.optim.RiemannianAdam([X], lr=cfg["adam_lr"])
    cap = math.tanh(cfg["r_max"] / 2) if space == "poincare" else None
    window, tol = cfg["adam_window"], cfg["adam_tol"]
    best, hist, t = math.inf, [], 0
    while True:
        opt.zero_grad()
        loss, _ = prob.stress(X, space)
        best = min(best, loss.item())
        hist.append(best)  # hist[t] = L*_t, t회 단계 뒤까지의 최소 stress
        if t >= window and hist[t - window] - best <= tol * hist[t - window]:
            converged = True
            break
        if t >= cfg["adam_max_iter"]:
            converged = False
            break
        loss.backward()
        opt.step()
        t += 1
        if cap is not None:  # 매 단계 뒤 반지름 상한 투영
            with torch.no_grad():
                norm = X.norm(dim=-1, keepdim=True)
                X.copy_(torch.where(norm > cap, X * (cap / norm), X))
    return {"X": X.detach().numpy().copy(), "n_iter": t, "converged": converged, "seconds": time.perf_counter() - t0}


def make_objective(prob, space):
    """L-BFGS-B용 (unpack, fun). 쌍곡 z = [r (n개), w (n x dim)]에서 x = tanh(r/2)·w/‖w‖. 유클리드 z = 좌표를 편 것.
    fun(z)는 (stress, 기울기)를 돌려준다(기울기는 torch 자동미분)."""
    n = prob.n

    def unpack(z):
        if space == "poincare":
            r, w = z[:n], z[n:].reshape(n, -1)
            return torch.tanh(r / 2).unsqueeze(1) * w / w.norm(dim=-1, keepdim=True)
        return z.reshape(n, -1)

    def fun(z):
        zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        loss, _ = prob.stress(unpack(zt), space)
        loss.backward()
        return loss.item(), zt.grad.numpy().copy()

    return unpack, fun


def finish(prob, space, X_end, cfg=CFG, options=None):
    """2단계: Adam 끝점에서 scipy L-BFGS-B 마무리. 쌍곡은 r in [0, r_max] 상자 제약."""
    if options is None:
        options = {
            "ftol": cfg["fin_ftol"],
            "gtol": cfg["fin_gtol"],
            "maxiter": cfg["fin_maxiter"],
            "maxfun": cfg["fin_maxfun"],
        }
    n, r_max = prob.n, cfg["r_max"]
    unpack, fun = make_objective(prob, space)
    if space == "poincare":
        norm = np.linalg.norm(X_end, axis=1)
        r0 = np.minimum(2.0 * np.arctanh(np.minimum(norm, 1.0 - 1e-16)), r_max)
        w0 = np.where(norm[:, None] > 1e-12, X_end / np.maximum(norm, 1e-12)[:, None], np.eye(X_end.shape[1])[0])
        z0 = np.concatenate([r0, w0.ravel()])
        bounds = [(0.0, r_max)] * n + [(None, None)] * w0.size
    else:
        z0, bounds = X_end.ravel().copy(), None
    t0 = time.perf_counter()
    res = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
    X_fin = unpack(torch.tensor(res.x, dtype=torch.float64)).numpy().copy()
    return {
        "X": X_fin,
        "loss": float(res.fun),
        "converged": bool(res.success),  # L-BFGS-B status 0: ftol 또는 gtol 조건으로 끝남
        "message": str(res.message),
        "n_iter": int(res.nit),
        "n_fev": int(res.nfev),
        "seconds": time.perf_counter() - t0,
    }


def optimize_from_init(D, space, X0, cfg=CFG):
    """초기값 X0에서 2단계 최적화를 한 번 한다. 재시작 한 번의 기록(dict, 좌표 X 포함)을 돌려준다."""
    prob = Problem(D)
    a = run_adam(prob, space, X0, cfg)
    f = finish(prob, space, a["X"], cfg)
    m = prob.metrics(f["X"], space, cfg["r_max"])
    rec = {
        "space": space,
        "stress": m["stress"],
        "distortion": m["distortion"],
        "scale": m["scale"],
        "max_rel_err": m["max_rel_err"],
        "finish_converged": f["converged"],
        "finish_message": f["message"],
        "adam_converged": a["converged"],
        "n_iter_adam": a["n_iter"],
        "n_iter_finish": f["n_iter"],
        "n_iter": a["n_iter"] + f["n_iter"],
        "n_fev_finish": f["n_fev"],
        "seconds_adam": a["seconds"],
        "seconds_finish": f["seconds"],
        "seconds": a["seconds"] + f["seconds"],
        "X": f["X"],
        "distortion_adam": prob.metrics(a["X"], space, cfg["r_max"])["distortion"],
    }
    if space == "poincare":
        rec["max_radius"] = m["max_radius"]
        rec["n_at_cap"] = m["n_at_cap"]
    return rec


_CODE_HASH = hashlib.md5(Path(__file__).read_bytes()).hexdigest()  # 이 파일이 바뀌면 재시작 캐시를 쓰지 않는다


def _task_key(task):
    h = hashlib.md5()
    h.update(_CODE_HASH.encode("utf-8"))
    h.update(np.ascontiguousarray(task["D"], dtype=np.float64).tobytes())
    h.update(task["space"].encode("utf-8"))
    h.update(np.ascontiguousarray(task["X0"], dtype=np.float64).tobytes())
    h.update(config_hash(task["cfg"]).encode("utf-8"))
    return h.hexdigest()


def run_restart(task):
    """프로세스 풀에서 부르는 모듈 최상위 함수. task = {'D', 'space', 'X0', 'cfg', 'meta', 'cache_dir'(선택)}.
    결과는 task와 시드로만 정해진다. cache_dir이 있으면 (D, 공간, 초기값, 설정)이 같은 재시작 결과를 디스크에 두고 다시 쓴다(중간에 멈춘 실행을 이어 가기 위함)."""
    torch.set_num_threads(1)
    cache_dir = task.get("cache_dir")
    path = None
    if cache_dir:
        path = Path(cache_dir) / f"{_task_key(task)}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                rec = pickle.load(f)
            rec.update(task["meta"])
            rec["from_cache"] = True
            return rec
    rec = optimize_from_init(task["D"], task["space"], task["X0"], task["cfg"])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, path)
        try:  # 진행 확인용 로그(여러 프로세스가 덧붙인다)
            with open(Path(cache_dir) / "progress.log", "a", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%H:%M:%S')} pid={os.getpid()} {task.get('label', '')} {rec['space']} stress={rec['stress']:.6g} "
                    f"dist={rec['distortion']:.6f} conv={rec['finish_converged']} it={rec['n_iter_adam']}+{rec['n_iter_finish']} sec={rec['seconds']:.0f}\n"
                )
        except OSError:
            pass
    rec.update(task["meta"])
    rec["from_cache"] = False
    return rec


class SerialRunner:
    """재시작을 차례로 돌린다(1 프로세스)."""

    def __init__(self, cache_dir=None):
        self.cache_dir = None if cache_dir is None else str(cache_dir)

    def run(self, tasks):
        return [run_restart({**t, "cache_dir": self.cache_dir}) for t in tasks]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PoolRunner:
    """재시작을 여러 프로세스로 병렬 실행한다(각 프로세스 1스레드, spawn). run은 여러 스레드에서 불러도 된다."""

    def __init__(self, max_workers, cache_dir=None):
        self.max_workers = max_workers
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.ex = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))

    def run(self, tasks):
        futs = [self.ex.submit(run_restart, {**t, "cache_dir": self.cache_dir}) for t in tasks]
        return [f.result() for f in futs]

    def close(self):
        self.ex.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _x0_key(space, X0):
    return (space, hashlib.md5(np.ascontiguousarray(X0, dtype=np.float64).tobytes()).hexdigest())


def _run_specs(D, specs, attempt, runner, cfg, label=""):
    """specs의 재시작을 한꺼번에 runner로 돌려 기록(좌표 X 포함) 목록을 돌려준다. (공간, 초기값)이 같은 재시작은 한 번만 돌린다."""
    tasks, index, keys = [], {}, []
    for s in specs:
        k = _x0_key(s["space"], s["X0"])
        keys.append(k)
        if k not in index:
            index[k] = len(tasks)
            tasks.append({"D": D, "space": s["space"], "X0": s["X0"], "cfg": cfg, "meta": {}, "label": f"{label}/a{attempt}/{s['init']}"})
    results = runner.run(tasks)
    out = []
    for s, k in zip(specs, keys):
        rec = dict(results[index[k]])
        rec.update({key: s.get(key) for key in ("init", "seed", "m", "source")})
        rec["attempt"] = attempt
        out.append(rec)
    return out


def run_first_attempt(D, runner, cfg, label=""):
    """첫 시도: 유클리드 5회(고전 MDS 1 + U 시드 0..3) + 쌍곡 5회(줄인 초기값 2 + U 시드 0..2).
    쌍곡 U 재시작은 유클리드와 독립이라 함께 보내고, 줄인 초기값 2개는 유클리드가 끝난 뒤 그중 최소 stress 해로 만들어 보낸다."""
    n = len(D)
    wave1 = [{"space": "euclid", "init": "mds", "seed": None, "X0": init_mds(D, cfg)}]
    wave1 += [{"space": "euclid", "init": "U", "seed": s, "X0": init_uniform(n, s, cfg)} for s in range(cfg["n_euc_u"])]
    wave1 += [{"space": "poincare", "init": "U", "seed": s, "X0": init_uniform(n, s, cfg)} for s in range(cfg["n_hyp_u"])]
    recs1 = _run_specs(D, wave1, 0, runner, cfg, label)
    euc = [r for r in recs1 if r["space"] == "euclid"]
    best_euc = min(euc, key=lambda r: r["stress"])  # 수렴 여부와 관계없이 상대 stress가 가장 작은 유클리드 재시작
    src = f"{best_euc['init']}" + ("" if best_euc["seed"] is None else f"_{best_euc['seed']}")
    wave2 = [
        {"space": "poincare", "init": f"shrunk_{tag}", "seed": None, "m": m, "source": src, "X0": shrink_to_max_norm(best_euc["X"], m)}
        for tag, m in (("m1", cfg["m1"]), ("m2", cfg["m2"]))
    ]
    recs2 = _run_specs(D, wave2, 0, runner, cfg, label)
    return euc + [r for r in recs1 if r["space"] == "poincare"] + recs2


def run_hyp_retry(D, offset, attempt, runner, cfg, label=""):
    """재실행: 쌍곡의 U(±0.1) 재시작 3개만 시드 + offset(5, 10)으로 다시 돌린다. 유클리드 결과와 줄인 초기값 2개는 다시 돌리지 않는다."""
    n = len(D)
    specs = [{"space": "poincare", "init": "U", "seed": s + offset, "X0": init_uniform(n, s + offset, cfg)} for s in range(cfg["n_hyp_u"])]
    return _run_specs(D, specs, attempt, runner, cfg, label)


def _brief(rec):
    return None if rec is None else {k: v for k, v in rec.items() if k != "X"}


def summarize_restarts(restarts, cfg=CFG):
    """그 시도까지 합친 재시작 기록(유클리드 5 + 쌍곡 5 + 재실행분)에서 최솟값, 판정 재시작, 비, 기록 문구를 만든다.
    실패와 평탄 극한은 수렴과 관계없는 최소 stress로, 쌍곡 이점의 비는 마무리가 수렴한 재시작 중 최소 stress로 구한다."""
    hyp = [r for r in restarts if r["space"] == "poincare"]
    euc = [r for r in restarts if r["space"] == "euclid"]
    if not hyp or not euc:
        raise ValueError("두 공간의 재시작이 모두 필요하다")
    hmin, emin = min(r["stress"] for r in hyp), min(r["stress"] for r in euc)
    opt_failure = hmin > emin * (1.0 + cfg["flat_rtol"])
    flat_limit = (not opt_failure) and hmin > emin
    hconv = [r for r in hyp if r["finish_converged"]]
    econv = [r for r in euc if r["finish_converged"]]
    bh = min(hconv, key=lambda r: r["stress"]) if hconv else None
    be = min(econv, key=lambda r: r["stress"]) if econv else None
    ratio = (bh["distortion"] / be["distortion"]) if (bh and be) else None
    stress_ratio = (bh["stress"] / be["stress"]) if (bh and be) else None
    if opt_failure:
        record, advantage, reason = REC_FAIL, ADV_HOLD, f"쌍곡 최소 stress {hmin:.6g} > 유클리드 {emin:.6g} x (1 + {cfg['flat_rtol']:g})"
    elif flat_limit:
        record, advantage, reason = ADV_FLAT, ADV_FLAT, f"쌍곡 최소 stress가 유클리드의 1 ~ 1 + {cfg['flat_rtol']:g}배 안에 있다"
    elif bh is None or be is None:
        record, advantage = ADV_HOLD, ADV_HOLD
        reason = "마무리가 수렴한 재시작이 없는 공간이 있다: " + ", ".join(
            name for name, x in (("쌍곡", bh), ("유클리드", be)) if x is None
        )
    else:
        advantage = ADV_YES if ratio <= cfg["ratio_max"] else ADV_NO
        record, reason = advantage, f"평균 상대 왜곡 비 {ratio:.4f} (문턱 {cfg['ratio_max']})"
    return {
        "hyp_min_stress_all": hmin,
        "euc_min_stress_all": emin,
        "stress_ratio_all": hmin / emin,
        "opt_failure": opt_failure,
        "flat_limit": flat_limit,
        "n_converged_hyp": len(hconv),
        "n_converged_euc": len(econv),
        "n_restarts_hyp": len(hyp),
        "n_restarts_euc": len(euc),
        "hyp_best": _brief(bh),
        "euc_best": _brief(be),
        "ratio": ratio,
        "stress_ratio": stress_ratio,
        "record": record,
        "advantage": advantage,
        "reason": reason,
    }


summarize_attempt = summarize_restarts  # 이전 이름


def embed_input(D, runner=None, cfg=None, name=""):
    """한 입력 거리 행렬의 2차원 쌍곡·유클리드 임베딩(재시작, 재실행 규칙 포함).
    첫 시도 뒤 쌍곡 최소 stress가 유클리드 최소 stress x (1 + 1e-3)보다 크면 쌍곡 U 재시작 3개만 시드 +5, +10으로 최대 2회 더 돌린다.
    쌍곡은 모든 시도의 재시작을 합쳐 판정한다. attempts[k]["restarts"]는 그 시도에서 새로 돌린 것, attempts[k]["summary"]는 그때까지 합친 것,
    결과의 "restarts"와 "summary"는 합친 전체다."""
    cfg = check_cfg(cfg if cfg is not None else CFG)
    runner = runner if runner is not None else SerialRunner()
    D = np.ascontiguousarray(D, dtype=np.float64)
    first = run_first_attempt(D, runner, cfg, name)
    pooled = list(first)
    attempts = [{"attempt": 0, "offset": 0, "restarts": first, "summary": summarize_restarts(pooled, cfg)}]
    for a, offset in enumerate(cfg["retry_offsets"], start=1):
        if not attempts[-1]["summary"]["opt_failure"]:
            break
        new = run_hyp_retry(D, offset, a, runner, cfg, name)
        pooled = pooled + new
        attempts.append({"attempt": a, "offset": offset, "restarts": new, "summary": summarize_restarts(pooled, cfg)})
    final = attempts[-1]
    return {
        "name": name,
        "n": len(D),
        "attempts": attempts,
        "final_attempt": final["attempt"],
        "restarts": pooled,
        "summary": final["summary"],
        "config_hash": config_hash(cfg),
    }


def embed_many(problems, runner, cfg=None, max_coordinators=None):
    """여러 입력을 동시에 임베딩한다. 입력마다 조정 스레드 하나가 공용 runner에 재시작을 보낸다. problems = {이름: D}."""
    cfg = check_cfg(cfg if cfg is not None else CFG)
    names = list(problems)
    workers = max_coordinators or len(names)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {nm: ex.submit(embed_input, problems[nm], runner, cfg, nm) for nm in names}
        return {nm: futs[nm].result() for nm in names}


def best_coords(result, space):
    """판정에 쓴 재시작(마무리 수렴 중 최소 stress)의 좌표. 없으면 None."""
    key = "hyp_best" if space == "poincare" else "euc_best"
    brief = result["summary"][key]
    if brief is None:
        return None
    for r in result["restarts"]:
        if (r["space"], r["init"], r.get("seed"), r.get("m")) == (space, brief["init"], brief.get("seed"), brief.get("m")) and r["stress"] == brief["stress"]:
            return r["X"]
    return None


def worst_pairs(D, X, space, k=20, labels=None):
    """최악 상대 왜곡 k쌍. 컬럼: i, j, d_true, d_emb(척도 곱한 임베딩 거리), rel_err(부호 포함), abs_rel_err."""
    prob = Problem(D)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    with torch.no_grad():
        p = prob.dist(Xt, space)
        q = p / prob.d
        s = q.sum() / (q * q).sum()
        e = ((s * p - prob.d) / prob.d).numpy()
    order = np.argsort(-np.abs(e), kind="stable")[:k]
    df = pd.DataFrame(
        {
            "i": prob.i.numpy()[order],
            "j": prob.j.numpy()[order],
            "d_true": prob.d.numpy()[order],
            "d_emb": (s * p).numpy()[order],
            "rel_err": e[order],
            "abs_rel_err": np.abs(e[order]),
        }
    )
    if labels is not None:
        df.insert(2, "name_i", [labels[i] for i in df["i"]])
        df.insert(3, "name_j", [labels[j] for j in df["j"]])
    return df


# ===================================================================== 2. 요약과 판정
def judge_flow_advantage(flow_summary, control_summaries, cfg=CFG):
    """d_flow의 "쌍곡 이점" (ADR 0010 7항, 0013 4항). 비 <= 0.8이고 대조군 5회 비의 최솟값보다도 작아야 한다.
    대조군 중 하나라도 판정에 쓸 재시작이 없으면(최적화 실패 또는 수렴한 재시작이 없는 공간) "보류"다."""
    unusable = [i for i, c in enumerate(control_summaries) if c["record"] in (REC_FAIL, ADV_HOLD) or c["ratio"] is None]
    if unusable:
        return {"advantage": ADV_HOLD, "reason": f"판정 재시작이 없는 대조군: {unusable}", "ratio": flow_summary["ratio"], "control_min_ratio": None}
    cmin = min(c["ratio"] for c in control_summaries)
    adv, ratio = flow_summary["advantage"], flow_summary["ratio"]
    if adv == ADV_HOLD:
        return {"advantage": ADV_HOLD, "reason": flow_summary["reason"], "ratio": ratio, "control_min_ratio": cmin}
    if adv in (ADV_NO, ADV_FLAT):
        return {"advantage": adv, "reason": flow_summary["reason"], "ratio": ratio, "control_min_ratio": cmin}
    if ratio < cmin:
        return {"advantage": ADV_YES, "reason": f"비 {ratio:.4f} <= {cfg['ratio_max']}이고 대조군 비의 최솟값 {cmin:.4f}보다 작다", "ratio": ratio, "control_min_ratio": cmin}
    return {"advantage": ADV_NO, "reason": f"비 {ratio:.4f} >= 대조군 비의 최솟값 {cmin:.4f}", "ratio": ratio, "control_min_ratio": cmin}


def judge_hierarchy(flow_db, control_db, n_required=20):
    """"중력모형 이상의 위계" (ADR 0010 7항). flow_db[(정규화, 부분집합)] = d_flow의 δ̄, control_db[...] = 대조군 20개의 δ̄.
    주·대안 정규화 각각, 전체·섬 제외 각각에서 δ̄(d_flow)가 대조군 20개보다 모두 작아야 한다(4개 조건 모두)."""
    rows = []
    for norm in NORMS:
        for subset in SUBSETS:
            c = np.asarray(control_db[(norm, subset)], dtype=float)
            if len(c) != n_required or not np.isfinite(c).all():
                raise ValueError(f"대조군 δ̄가 {n_required}개의 유한값이어야 한다: {norm}/{subset} {len(c)}개")
            f = float(flow_db[(norm, subset)])
            rows.append(
                {
                    "정규화": norm,
                    "부분집합": subset,
                    "d_flow": f,
                    "대조군 최솟값": float(c.min()),
                    "대조군 중앙값": float(np.median(c)),
                    "대조군 최댓값": float(c.max()),
                    "d_flow 이하인 대조군 수": int((c <= f).sum()),
                    "여유(최솟값 - d_flow)": float(c.min() - f),
                    "통과": bool(f < c.min()),
                }
            )
    table = pd.DataFrame(rows)
    return {"table": table, "hierarchy": bool(table["통과"].all())}


# ===================================================================== 3. δ 진단
def triangle_violation(D):
    """max(D_ik - D_ij - D_jk, 0). 삼각 부등식이 성립하면 0."""
    D = np.asarray(D, dtype=np.float64)
    worst = 0.0
    for j in range(len(D)):
        worst = max(worst, float((D - D[:, [j]] - D[[j], :]).max()))
    return max(worst, 0.0)


if numba is not None:

    @njit(parallel=True, cache=True)
    def _quad_stats_nb(D, tie_rtol):
        """모든 서로 다른 네 점 i<j<k<l에서 (합계 (S1-S2)/(S1-S3), 동률 수, (S1-S2)/2의 최댓값)을 i별로 모은다."""
        n = D.shape[0]
        part_sum = np.zeros(n)
        part_tie = np.zeros(n)
        part_max = np.zeros(n)
        for i in prange(n):
            s_acc = 0.0
            ties = 0.0
            mx = 0.0
            for j in range(i + 1, n):
                dij = D[i, j]
                for k in range(j + 1, n):
                    dik = D[i, k]
                    djk = D[j, k]
                    for l in range(k + 1, n):
                        s1 = dij + D[k, l]
                        s2 = dik + D[j, l]
                        s3 = D[i, l] + djk
                        if s1 >= s2:
                            a = s1
                            b = s2
                        else:
                            a = s2
                            b = s1
                        if s3 >= a:
                            hi = s3
                            mid = a
                            lo = b
                        elif s3 >= b:
                            hi = a
                            mid = s3
                            lo = b
                        else:
                            hi = a
                            mid = b
                            lo = s3
                        half = 0.5 * (hi - mid)
                        if half > mx:
                            mx = half
                        den = hi - lo
                        if den <= tie_rtol * hi:
                            ties += 1.0
                        else:
                            s_acc += (hi - mid) / den
            part_sum[i] = s_acc
            part_tie[i] = ties
            part_max[i] = mx
        return part_sum, part_tie, part_max

    @njit(parallel=True, cache=True)
    def _delta_ref_nb(D):
        """기준점 w마다 Gromov 곱 A_xy = (x|y)_w의 (max, min) 행렬곱 M으로 max(M - A)를 구하고 w에 대해 최댓값을 취한다."""
        n = D.shape[0]
        best = np.zeros(n)
        for w in prange(n):
            A = np.empty((n, n))
            for x in range(n):
                for y in range(n):
                    A[x, y] = 0.5 * (D[w, x] + D[w, y] - D[x, y])
            m = 0.0
            for x in range(n):
                for z in range(n):
                    mx = -np.inf
                    for y in range(n):
                        v = A[x, y] if A[x, y] < A[y, z] else A[y, z]
                        if v > mx:
                            mx = v
                    diff = mx - A[x, z]
                    if diff > m:
                        m = diff
            best[w] = m
        return best.max()


def _check_dist(D):
    D = np.ascontiguousarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 4:
        raise ValueError("D는 4x4 이상의 정사각 행렬이어야 한다")
    if not np.isfinite(D).all():
        raise ValueError("D에 유한하지 않은 값이 있다")
    if np.abs(D - D.T).max() > 1e-9 * max(1.0, np.abs(D).max()):
        raise ValueError("D가 대칭이 아니다")
    return D


def four_point_stats(D, tie_rtol=CFG["tie_rtol"]):
    """서로 다른 네 점 모든 조합(정확값). 세 쌍 합 S1 >= S2 >= S3에서
    delta = max (S1 - S2)/2, delta_bar = mean (S1 - S2)/(S1 - S3) (세 합이 같으면 0), n_quads, n_ties.
    삼각 부등식 없이도 쓸 수 있다(폐포 전 ell_sym 점검용)."""
    if numba is None:
        raise ImportError("numba가 필요하다(kfm-spike 환경)")
    D = _check_dist(D)
    n = D.shape[0]
    ps, pt, pm = _quad_stats_nb(D, tie_rtol)
    n_quads = math.comb(n, 4)
    return {
        "delta": float(pm.max()),
        "delta_bar": float(ps.sum() / n_quads),
        "n_quads": n_quads,
        "n_ties": int(round(pt.sum())),
    }


def delta_refpoint(D, check=True, tol=1e-9):
    """기준점별 (max, min) 행렬곱으로 구한 δ (정확값). 퇴화한 네 점을 포함하므로 삼각 부등식이 성립하는 거리에서만 쓴다."""
    if numba is None:
        raise ImportError("numba가 필요하다(kfm-spike 환경)")
    D = _check_dist(D)
    if check:
        v = triangle_violation(D)
        if v > tol * max(1.0, D.max()):
            raise ValueError(f"삼각 부등식 위반 {v:.3g}: 기준점 방식을 쓸 수 없다")
    return float(_delta_ref_nb(D))


def delta_report(D, cfg=CFG, cross_check=True):
    """한 거리 행렬의 δ(기준점 방식), diam, δ_rel = 2δ/diam, δ̄. cross_check이면 네 점 전체 계산의 δ와 맞는지도 낸다."""
    D = _check_dist(D)
    t0 = time.perf_counter()
    st = four_point_stats(D, cfg["tie_rtol"])
    delta = delta_refpoint(D)
    diam = float(D.max())
    out = {
        "n": D.shape[0],
        "diam": diam,
        "delta": delta,
        "delta_rel": 2.0 * delta / diam,
        "delta_bar": st["delta_bar"],
        "n_quads": st["n_quads"],
        "n_ties": st["n_ties"],
        "delta_quad": st["delta"],
        "seconds": time.perf_counter() - t0,
    }
    if cross_check:
        out["delta_diff"] = abs(delta - st["delta"])
    return out


def subset_matrix(D, keep):
    """전체 행렬의 부분행렬(폐포는 다시 하지 않는다)."""
    keep = np.asarray(keep)
    if keep.dtype == bool:
        keep = np.flatnonzero(keep)
    return np.asarray(D)[np.ix_(keep, keep)]


# ===================================================================== 4. d_flow
class EstimatedODError(ValueError):
    """관측 모드에서 추정 OD(중력모형 등)를 받았다."""


class DisconnectedFlowError(ValueError):
    """통행량이 있는 쌍으로 만든 그래프의 연결 성분이 둘 이상이다."""


def od_margins(T):
    """존 내 통행(대각)을 뺀 OD, 발생 O_i, 도착 D_j, V_i = O_i + D_i."""
    T0 = np.array(T, dtype=np.float64, copy=True)
    np.fill_diagonal(T0, 0.0)
    if (T0 < 0).any() or not np.isfinite(T0).all():
        raise ValueError("OD에 음수 또는 유한하지 않은 값이 있다")
    O, Dm = T0.sum(axis=1), T0.sum(axis=0)
    return T0, O, Dm, O + Dm


def flow_edge_mask(T):
    """한 방향이라도 0이면 엣지가 없다: T_ij > 0이고 T_ji > 0인 쌍(대각 제외)."""
    T0, *_ = od_margins(T)
    return (T0 > 0) & (T0.T > 0)


def flow_components(T):
    """엣지 마스크의 연결 성분 (개수, 라벨)."""
    return connected_components(csr_matrix(flow_edge_mask(T)), directed=False)


def metric_closure(W, edge):
    """최단경로 폐포(Floyd-Warshall). W는 엣지 길이, edge는 엣지 마스크(0 길이도 엣지로 본다)."""
    n = len(W)
    d = np.where(edge, W, np.inf)
    np.fill_diagonal(d, 0.0)
    for k in range(n):
        np.minimum(d, d[:, [k]] + d[[k], :], out=d)
    return d


def flow_distance(T, *, kind, role):
    """통행량 거리 (ADR 0010 2항, PLAN 3-3a). 주 정규화 ell_sym과 대안 정규화 ell', 각각의 최단경로 폐포.

    kind: "observed" | "estimated". role: "observed"(관측 모드) | None(대조군 모드).
    관측 모드에서는 추정 OD를 거부한다(EstimatedODError). 대조군 모드는 같은 함수를 role=None으로 부른다.
    연결 성분이 둘 이상이면 DisconnectedFlowError. OD에서 빠진 지역은 호출 전에 뺀다(flow_exclusion).
    돌려주는 값: d_main, d_alt(폐포), raw_main, raw_alt(폐포 전, 엣지 없는 쌍은 inf), edge(마스크), O, D, V.
    """
    if kind not in ("observed", "estimated"):
        raise ValueError("kind는 'observed' 또는 'estimated'")
    if role not in ("observed", None):
        raise ValueError("role은 'observed' 또는 None(대조군)")
    if role == "observed" and kind == "estimated":
        raise EstimatedODError("추정 OD(중력모형 등)는 관측 입력으로 받지 않는다. 대조군은 role=None으로 부른다(ADR 0010 2항)")
    T0, O, Dm, V = od_margins(T)
    edge = (T0 > 0) & (T0.T > 0)
    n_comp, labels = connected_components(csr_matrix(edge), directed=False)
    if n_comp > 1:
        sizes = np.bincount(labels).tolist()
        raise DisconnectedFlowError(f"연결 성분 {n_comp}개(크기 {sizes}). 빠진 지역은 미리 빼야 한다")
    with np.errstate(divide="ignore", invalid="ignore"):
        ell_dir = -np.log(T0 / np.sqrt(np.outer(O, Dm)))
        ell_main = np.where(edge, (ell_dir + ell_dir.T) / 2.0, np.inf)
        ell_alt = np.where(edge, -np.log((T0 + T0.T) / np.sqrt(np.outer(V, V))), np.inf)
    for nm, e in (("ell_sym", ell_main), ("ell'", ell_alt)):
        lo = float(e[edge].min())
        if lo < -1e-9:
            raise AssertionError(f"{nm} < 0 ({lo:.3g}): T_ij <= min(O_i, D_j) 가정이 깨졌다")
    ell_main = np.where(edge, np.maximum(ell_main, 0.0), np.inf)
    ell_alt = np.where(edge, np.maximum(ell_alt, 0.0), np.inf)
    return {
        "d_main": metric_closure(ell_main, edge),
        "d_alt": metric_closure(ell_alt, edge),
        "raw_main": ell_main,
        "raw_alt": ell_alt,
        "edge": edge,
        "O": O,
        "D": Dm,
        "V": V,
    }


def od_unlinked_units(T):
    """존 간 통행이 전혀 없는 지역(O_i + D_i = 0): "OD에서 빠진 지역"."""
    _, _, _, V = od_margins(T)
    return np.flatnonzero(V == 0)


def flow_exclusion(T, island_mask):
    """d_flow 계산 전에 뺄 지역 (ADR 0010 2항: OD 자료에서 빠진 지역(도서지역 제외 등)을 미리 빼고 목록을 기록한다).
    (1) 존 간 통행이 전혀 없는 지역, (2) 쌍방향 통행 그래프에서 가장 큰 연결 성분에 들지 않는 지역 중 섬 연결 지역(island_mask)을 뺀다.
    남은 지역이 섬 연결 지역 밖의 이유로 여전히 둘 이상의 성분이면 DisconnectedFlowError(중단)."""
    n = len(T)
    island_mask = np.asarray(island_mask, dtype=bool)
    unlinked = od_unlinked_units(T)
    keep = np.ones(n, dtype=bool)
    keep[unlinked] = False
    idx = np.flatnonzero(keep)
    T_sub = np.asarray(T)[np.ix_(idx, idx)]
    n_comp, labels = flow_components(T_sub)
    dropped_isl = np.array([], dtype=int)
    if n_comp > 1:
        main = np.argmax(np.bincount(labels))
        out_idx = idx[labels != main]
        if not island_mask[out_idx].all():
            bad = out_idx[~island_mask[out_idx]]
            raise DisconnectedFlowError(f"섬 연결 지역이 아닌 지역이 본토 성분과 이어지지 않는다: {bad.tolist()}")
        dropped_isl = out_idx
        keep[out_idx] = False
    return {"keep": keep, "unlinked": unlinked, "island_dropped": dropped_isl}


# ===================================================================== 5. 중력 대조군
def furness(logK, logO, logD, la=None, lb=None, tol=1e-13, max_iter=200_000):
    """로그 영역 Furness(이중 비례 보정). T = exp(la_i + lb_j + logK_ij)의 행합 O, 열합 D. logK의 대각은 -inf."""
    n = len(logO)
    la = np.zeros(n) if la is None else la.copy()
    lb = np.zeros(n) if lb is None else lb.copy()
    for it in range(1, max_iter + 1):
        la_new = logO - logsumexp(logK + lb[None, :], axis=1)
        lb_new = logD - logsumexp(logK + la_new[:, None], axis=0)
        step = max(np.abs(la_new - la).max(), np.abs(lb_new - lb).max())
        la, lb = la_new, lb_new
        if step < tol:
            break
    else:
        raise RuntimeError(f"Furness가 {max_iter}회 안에 수렴하지 않았다(마지막 변화 {step:.3g})")
    return la, lb, it


def fit_gravity(T_obs, t_dir, tol=1e-13):
    """이중 제약 지수 중력모형 T_ij = a_i b_j exp(-beta t_ij) (i != j)의 beta를 적합한다 (ADR 0010 6항).
    발생·도착 합은 존 내 통행을 뺀 관측 T의 O_i, D_j. beta는 모형의 평균 통행시간이 존 내 통행을 뺀 관측 T로
    가중한 방향 통행시간 t_dir(i -> j) 평균과 같아지게 한다. 균형은 Furness. 돌려주는 값: beta, T_exp(기댓값), 진단."""
    T0, O, Dm, _ = od_margins(T_obs)
    t = np.asarray(t_dir, dtype=np.float64)
    n = len(t)
    off = ~np.eye(n, dtype=bool)
    if (O <= 0).any() or (Dm <= 0).any():
        raise ValueError("발생 또는 도착 합이 0인 지역이 있다: 먼저 빼야 한다")
    m_obs = float((T0[off] * t[off]).sum() / T0[off].sum())
    logO, logD = np.log(O), np.log(Dm)
    state = {"la": None, "lb": None}

    def solve(beta):
        logK = np.where(off, -beta * t, -np.inf)
        la, lb, it = furness(logK, logO, logD, state["la"], state["lb"], tol)
        state["la"], state["lb"] = la, lb
        logT = la[:, None] + lb[None, :] + logK
        T = np.exp(logT)
        return float((T[off] * t[off]).sum() / T[off].sum()), la, lb, it

    f = lambda b: solve(b)[0] - m_obs  # noqa: E731
    lo, hi = 0.0, 1e-3
    if f(lo) <= 0:
        raise ValueError("beta = 0에서도 모형의 평균 통행시간이 관측 이하다: 지수 감쇠로 맞출 수 없다")
    while f(hi) > 0:
        lo, hi = hi, hi * 2.0
        if hi > 1e3:
            raise RuntimeError("beta 구간을 찾지 못했다")
    beta = brentq(f, lo, hi, xtol=1e-15, rtol=1e-13, maxiter=500)
    state["la"] = state["lb"] = None  # 마지막은 차가운 시작으로 다시 푼다(호출 순서에 의존하지 않게)
    m_fit, la, lb, it = solve(beta)
    logK = np.where(off, -beta * t, -np.inf)
    T_exp = np.exp(la[:, None] + lb[None, :] + logK)
    return {
        "beta": float(beta),
        "T_exp": T_exp,
        "log_a": la,
        "log_b": lb,
        "mean_time_obs": m_obs,
        "mean_time_fit": m_fit,
        "furness_iter": it,
        "row_sum_max_rel_err": float(np.abs(T_exp.sum(axis=1) / O - 1).max()),
        "col_sum_max_rel_err": float(np.abs(T_exp.sum(axis=0) / Dm - 1).max()),
    }


def resample_control(T_exp, T_obs, seed, cfg=CFG):
    """대조군 OD 하나: 존 간 합 x 0.2(반올림)를 총량으로 존 간 셀을 다항 재표본하고, x5 확대한 뒤 관측의 0 패턴을 적용한다."""
    T0, _, _, _ = od_margins(T_obs)
    n = len(T0)
    off = ~np.eye(n, dtype=bool)
    n_eff = int(math.floor(cfg["resample_frac"] * T0.sum() + 0.5))
    p = T_exp[off] / T_exp[off].sum()
    rng = np.random.default_rng([RESAMPLE_STREAM, seed])
    counts = rng.multinomial(n_eff, p)
    Tc = np.zeros((n, n))
    Tc[off] = counts * cfg["resample_scale"]
    Tc[T0 == 0] = 0.0
    return Tc, n_eff


def make_controls(T_exp, T_obs, n_wanted=20, max_seed=500, cfg=CFG):
    """시드 0, 1, 2, ...순으로 대조군을 만들고, 연결 성분이 둘 이상인 시드는 건너뛰어 연결된 n_wanted개를 모은다.
    건너뛴 시드를 기록한다. 대조군은 d_flow 함수를 대조군 모드(role=None)로 불러 같은 0 처리와 폐포를 거친다."""
    out, skipped = [], []
    for seed in range(max_seed):
        Tc, n_eff = resample_control(T_exp, T_obs, seed, cfg)
        try:
            fd = flow_distance(Tc, kind="estimated", role=None)
        except DisconnectedFlowError as e:
            skipped.append({"seed": seed, "reason": str(e)})
            continue
        out.append({"seed": seed, "T": Tc, "d_main": fd["d_main"], "d_alt": fd["d_alt"], "n_eff": n_eff, "n_edges": int(fd["edge"].sum() // 2)})
        if len(out) == n_wanted:
            return out, skipped
    raise RuntimeError(f"시드 {max_seed}개 안에서 연결된 대조군 {n_wanted}개를 모으지 못했다({len(out)}개)")


# ===================================================================== 6. 입력 준비 보조
def merge_weighted(M, groups, weights, zero_diag=True):
    """행렬 M을 그룹별 가중 평균으로 합친다: (W^T M W) / (P P^T), W[i, g(i)] = weights[i]. 합치지 않은 단위는 원래 값이다."""
    groups, w = np.asarray(groups), np.asarray(weights, dtype=np.float64)
    k = groups.max() + 1
    W = np.zeros((len(groups), k))
    W[np.arange(len(groups)), groups] = w
    P = W.sum(axis=0)
    out = (W.T @ np.asarray(M, dtype=np.float64) @ W) / np.outer(P, P)
    if zero_diag:
        np.fill_diagonal(out, 0.0)
    return out


def merge_sum(T, groups, zero_diag=True):
    """OD 행렬을 그룹별로 더한다: G^T T G. zero_diag이면 합친 지역 안의 통행을 존 내 통행으로 보고 뺀다."""
    groups = np.asarray(groups)
    k = groups.max() + 1
    G = np.zeros((len(groups), k))
    G[np.arange(len(groups)), groups] = 1.0
    out = G.T @ np.asarray(T, dtype=np.float64) @ G
    if zero_diag:
        np.fill_diagonal(out, 0.0)
    return out


def island_connected_mask(n, u, v, kind):
    """G_adj에서 섬 링크 엣지(kind == 'island')를 모두 뺐을 때 가장 큰 연결 성분(본토)에 들지 않는 지역 (ADR 0010 5항).
    연륙·연도교 엣지는 빼지 않는다."""
    u, v, kind = np.asarray(u), np.asarray(v), np.asarray(kind)
    keep = kind != "island"
    A = csr_matrix((np.ones(keep.sum()), (u[keep], v[keep])), shape=(n, n))
    n_comp, labels = connected_components(A, directed=False)
    main = np.argmax(np.bincount(labels, minlength=n_comp))
    return labels != main


# ===================================================================== 7. 합성 fixture
def tree_hop_distances(depth):
    """깊이 depth 포화 이진 트리의 hop 거리 행렬. 노드 i(> 0)의 부모는 (i - 1) // 2."""
    from scipy.sparse.csgraph import shortest_path

    n = 2 ** (depth + 1) - 1
    child = np.arange(1, n)
    adjacency = csr_matrix((np.ones(n - 1), (child, (child - 1) // 2)), shape=(n, n))
    return shortest_path(adjacency, unweighted=True, directed=False)


def c4_distances():
    """4-순환 그래프의 hop 거리."""
    return np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]], dtype=float)


def square4_distances():
    """정사각형 네 꼭짓점의 유클리드 거리."""
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def random_metric(n, seed, dim=3):
    """무작위 점의 유클리드 거리(삼각 부등식 성립)."""
    pts = np.random.default_rng(seed).normal(size=(n, dim))
    return np.linalg.norm(pts[:, None] - pts[None], axis=-1)


def gravity_grid(k=7, spacing=10.0, beta=0.08, seed=0):
    """k x k 격자 지역 + 지수 중력 OD. 거리 = 격자 중심 사이 유클리드 거리(km)를 시간(분)으로 읽는다.
    T_ij = a_i b_j exp(-beta d_ij) (i != j), a_i, b_j는 로그정규 무작위. 돌려주는 값: d_sym, T, beta."""
    ix, iy = np.meshgrid(np.arange(k), np.arange(k))
    pts = np.c_[ix.ravel(), iy.ravel()].astype(float) * spacing
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0.0, 1.0, len(pts)))
    b = np.exp(rng.normal(0.0, 1.0, len(pts)))
    T = 1000.0 * a[:, None] * b[None, :] * np.exp(-beta * d)
    np.fill_diagonal(T, 0.0)
    return {"d_sym": d, "T": T, "beta": beta, "pts": pts}


def make_toy_data(k=5, seed=0, n_ctrl=6, beta_true=0.05):
    """3절 배선 점검용 합성 자료(실데이터 아님). k x k 격자 지역, 방향 통행시간(비대칭 잡음 + 지역별 오프셋 + 섬 두 곳의 긴 연결),
    관측 OD(지수 중력 기댓값의 포아송 잡음, 무작위 0). 구조는 노트북 2절이 만드는 실데이터 입력(data)과 같다."""
    rng = np.random.default_rng(seed)
    n = k * k
    ix, iy = np.meshgrid(np.arange(k), np.arange(k))
    pts = np.c_[ix.ravel(), iy.ravel()].astype(float) * 10.0
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    island = np.zeros(n, dtype=bool)
    island[-2:] = True
    g = rng.uniform(0.0, 3.0, n)
    t0 = d * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, (n, n))) + g[:, None] + g[None, :]
    t0[island[:, None] ^ island[None, :]] += 40.0  # 섬 연결 지역과 본토 사이의 긴 연결
    np.fill_diagonal(t0, 0.0)
    t_dir = metric_closure(t0, ~np.eye(n, dtype=bool))  # 방향 삼각 부등식을 만족시킨다(비대칭은 남긴다)
    d_sym = (t_dir + t_dir.T) / 2.0
    a, b = np.exp(rng.normal(0, 0.8, n)), np.exp(rng.normal(0, 0.8, n))
    lam = 4000.0 * a[:, None] * b[None, :] * np.exp(-beta_true * t_dir)
    np.fill_diagonal(lam, 0.0)
    T = rng.poisson(lam).astype(float)
    T[rng.random((n, n)) < 0.15] = 0.0
    for i in np.flatnonzero(island):  # 연결이 끊기지 않게 섬에서 가까운 두 지역과 쌍방향 소량 통행
        near = np.argsort(d_sym[i] + np.where(island, 1e9, 0.0))[:2]
        T[i, near] = np.maximum(T[i, near], 3.0)
        T[near, i] = np.maximum(T[near, i], 3.0)
    np.fill_diagonal(T, 0.0)
    fd = flow_distance(T, kind="observed", role="observed")
    fit = fit_gravity(T, t_dir)
    controls, skipped = make_controls(fit["T_exp"], T, n_wanted=n_ctrl)
    return {
        "delta_inputs": {
            "d_time_toy": {"D": d_sym, "island": island},
            "d_flow_main": {"D": fd["d_main"], "island": island},
            "d_flow_alt": {"D": fd["d_alt"], "island": island},
        },
        "flow_names": {"main": "d_flow_main", "alt": "d_flow_alt"},
        "flow_island": island,
        "controls": {"main": [c["d_main"] for c in controls], "alt": [c["d_alt"] for c in controls]},
        "embed_inputs": {"d_time_toy": d_sym, "d_flow_main": fd["d_main"], "d_flow_alt": fd["d_alt"]},
        "meta": {"beta_fit": fit["beta"], "beta_true": beta_true, "skipped_seeds": [x["seed"] for x in skipped], "T": T, "t_dir": t_dir},
    }


def bench_iteration_cost(n=250, n_adam=200, n_eval=100, seed=0, cfg=CFG):
    """합성 점(평면 위 n개, 무작위)에서 Adam 한 단계와 마무리 함수·기울기 한 번의 시간(초). 실데이터가 아니다."""
    pts = np.random.default_rng(seed).uniform(0.0, 500.0, (n, 2))
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    prob = Problem(D)
    X0 = init_uniform(n, seed, cfg)
    out = {"n": n, "n_pairs": n * (n - 1) // 2}
    for space in ("poincare", "euclid"):
        fast = {**cfg, "adam_max_iter": n_adam, "adam_window": 10**9}
        a = run_adam(prob, space, X0, fast)
        _, fun = make_objective(prob, space)
        z0 = np.concatenate([np.full(n, 1.0), X0.ravel()]) if space == "poincare" else X0.ravel()
        t0 = time.perf_counter()
        for _ in range(n_eval):
            fun(z0)
        out[space] = {"sec_per_adam_iter": a["seconds"] / n_adam, "sec_per_eval": (time.perf_counter() - t0) / n_eval}
    return out


# ===================================================================== 8. 3절 드라이버
def delta_table(inputs, cfg=CFG):
    """inputs = {이름: {"D": 행렬, "island": 섬 연결 지역 마스크}}. 전체와 섬 연결 지역 제외 부분집합의 δ, δ_rel, δ̄ 표."""
    rows = []
    for name, spec in inputs.items():
        D = np.asarray(spec["D"])
        island = np.asarray(spec["island"], dtype=bool)
        for subset, keep in (("전체", np.ones(len(D), dtype=bool)), ("섬 제외", ~island)):
            rep = delta_report(subset_matrix(D, keep), cfg)
            rows.append({"입력": name, "부분집합": subset, **rep})
    return pd.DataFrame(rows)


def control_delta_bars(controls, island, cfg=CFG):
    """controls = {"main": [행렬...], "alt": [행렬...]}. 반환 {(정규화, 부분집합): δ̄ 배열}."""
    island = np.asarray(island, dtype=bool)
    out = {}
    for norm in NORMS:
        for subset, keep in (("전체", np.ones(len(island), dtype=bool)), ("섬 제외", ~island)):
            out[(norm, subset)] = np.array(
                [four_point_stats(subset_matrix(D, keep), cfg["tie_rtol"])["delta_bar"] for D in controls[norm]]
            )
    return out


def embedding_table(results):
    """임베딩 결과 {이름: embed_input 결과}의 S4 보고표(판정 재시작 기준)."""
    rows = []
    for name, r in results.items():
        s = r["summary"]
        bh, be = s["hyp_best"], s["euc_best"]
        rows.append(
            {
                "입력": name,
                "n": r["n"],
                "기록": s["record"],
                "쌍곡 이점": s["advantage"],
                "평균 상대 왜곡 쌍곡": None if bh is None else bh["distortion"],
                "평균 상대 왜곡 유클리드": None if be is None else be["distortion"],
                "왜곡 비": s["ratio"],
                "stress 비(판정 재시작)": s["stress_ratio"],
                "stress 비(전체 최솟값)": s["stress_ratio_all"],
                "수렴 쌍곡/유클리드": f"{s['n_converged_hyp']}/{s['n_restarts_hyp']}, {s['n_converged_euc']}/{s['n_restarts_euc']}",
                "마무리 수렴(판정 재시작) 쌍곡": None if bh is None else bh["finish_converged"],
                "마무리 수렴(판정 재시작) 유클리드": None if be is None else be["finish_converged"],
                "반복 수 쌍곡(Adam+마무리)": None if bh is None else (bh["n_iter_adam"], bh["n_iter_finish"]),
                "반복 수 유클리드(Adam+마무리)": None if be is None else (be["n_iter_adam"], be["n_iter_finish"]),
                "상한에 닿은 점 수": None if bh is None else bh.get("n_at_cap"),
                "최대 반지름": None if bh is None else bh.get("max_radius"),
                "시도 수": r["final_attempt"] + 1,
                "초(재시작 합)": float(sum(x["seconds"] for a in r["attempts"] for x in a["restarts"])),
            }
        )
    return pd.DataFrame(rows)


def restart_table(result):
    """한 입력의 재시작별 표(최종 시도)."""
    cols = ["attempt", "space", "init", "seed", "m", "stress", "distortion", "finish_converged", "adam_converged", "n_iter_adam", "n_iter_finish", "n_at_cap", "max_radius", "seconds"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in result["restarts"]])


def judge_all(delta_df, ctrl_db, embed_results, flow_names, ctrl_names, cfg=CFG, n_required=20):
    """ADR 0010 7항 판정. flow_names = {"main": d_flow 주 정규화 입력 이름, "alt": ...},
    ctrl_names = {"main": [대조군 임베딩 이름...], "alt": [...]}.
    중력모형 이상의 위계(4개 조건)와, 정규화마다 d_flow의 쌍곡 이점(대조군 5개 비와 비교)을 돌려준다."""
    flow_db = {}
    for norm in NORMS:
        for subset in SUBSETS:
            sel = delta_df[(delta_df["입력"] == flow_names[norm]) & (delta_df["부분집합"] == subset)]
            flow_db[(norm, subset)] = float(sel["delta_bar"].iloc[0])
    hier = judge_hierarchy(flow_db, ctrl_db, n_required)
    adv = {}
    for norm in NORMS:
        if flow_names[norm] in embed_results and ctrl_names[norm] and all(c in embed_results for c in ctrl_names[norm]):
            adv[norm] = judge_flow_advantage(
                embed_results[flow_names[norm]]["summary"], [embed_results[c]["summary"] for c in ctrl_names[norm]], cfg
            )
    return {"hierarchy": hier, "flow_advantage": adv}


def to_jsonable(obj):
    """numpy 값을 JSON으로 쓸 수 있게 바꾼다."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_embeddings(results, out_dir, labels=None, cfg=CFG, D_by_name=None):
    """임베딩 산출물(좌표, 재시작표, 요약 JSON, 최악 20쌍)을 out_dir에 저장한다. 왜곡 지표를 반드시 함께 저장한다(CLAUDE.md).
    labels = {n: 지역 이름 목록}(입력 크기 n으로 고른다) 또는 목록 하나."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        restart_table(r).to_csv(out_dir / f"restarts_{name}.csv", index=False, encoding="utf-8-sig")
        (out_dir / f"summary_{name}.json").write_text(json.dumps(to_jsonable(r["summary"]), ensure_ascii=False, indent=1), encoding="utf-8")
        arrays = {}
        for space, tag in (("poincare", "hyp"), ("euclid", "euc")):
            X = best_coords(r, space)
            if X is not None:
                arrays[f"X_{tag}"] = X
                if D_by_name is not None:
                    lab = labels.get(len(D_by_name[name])) if isinstance(labels, dict) else labels
                    worst_pairs(D_by_name[name], X, space, cfg["n_worst"], lab).to_csv(
                        out_dir / f"worst20_{name}_{tag}.csv", index=False, encoding="utf-8-sig"
                    )
        if arrays:
            np.savez(out_dir / f"coords_{name}.npz", **arrays)


def embedding_problems(data, n_ctrl_embed=5, embed_alt=True):
    """임베딩할 입력 {이름: D}와 대조군 이름 {정규화: [이름...]}. 대조군은 연결된 대조군의 앞 n_ctrl_embed개(시드 순)."""
    problems = dict(data["embed_inputs"])
    ctrl_names = {"main": [], "alt": []}
    for nm in NORMS:
        if nm == "alt" and not embed_alt:
            continue
        for r in range(n_ctrl_embed):
            key = f"ctrl_{nm}_{r}"
            problems[key] = data["controls"][nm][r]
            ctrl_names[nm].append(key)
    return problems, ctrl_names


def run_section3(data, runner, cfg, out_dir, n_ctrl_delta=20, n_ctrl_embed=5, embed_alt=True, labels=None):
    """노트북 3절 전체. data 구조는 make_toy_data(합성)와 노트북 2절(실데이터)이 같다.
      data["delta_inputs"]  : {이름: {"D", "island"}}  (d_time 250·249·247, d_flow 주·대안)
      data["flow_names"]    : {"main": 이름, "alt": 이름}
      data["flow_island"]   : d_flow 단위의 섬 연결 지역 마스크
      data["controls"]      : {"main": [행렬...], "alt": [행렬...]} (연결된 대조군, 시드 순)
      data["embed_inputs"]  : {이름: D}  (임베딩할 d_time·d_flow)
    """
    cfg = check_cfg(cfg)
    t0 = time.perf_counter()
    res = {}
    res["delta"] = delta_table(data["delta_inputs"], cfg)
    ctrl = {nm: data["controls"][nm][:n_ctrl_delta] for nm in NORMS}
    res["ctrl_db"] = control_delta_bars(ctrl, data["flow_island"], cfg)
    problems, ctrl_names = embedding_problems(data, n_ctrl_embed, embed_alt)
    res["embed"] = embed_many(problems, runner, cfg)
    res["embed_table"] = embedding_table(res["embed"])
    res["judge"] = judge_all(res["delta"], res["ctrl_db"], res["embed"], data["flow_names"], ctrl_names, cfg, n_required=n_ctrl_delta)
    if out_dir is not None:
        save_embeddings(res["embed"], out_dir, labels=labels, cfg=cfg, D_by_name=problems)
        res["delta"].to_csv(Path(out_dir) / "delta_table.csv", index=False, encoding="utf-8-sig")
        res["embed_table"].to_csv(Path(out_dir) / "embedding_table.csv", index=False, encoding="utf-8-sig")
        res["judge"]["hierarchy"]["table"].to_csv(Path(out_dir) / "hierarchy_table.csv", index=False, encoding="utf-8-sig")
        np.savez(Path(out_dir) / "control_delta_bars.npz", **{f"{a}|{b}": v for (a, b), v in res["ctrl_db"].items()})
    res["seconds"] = time.perf_counter() - t0
    return res
