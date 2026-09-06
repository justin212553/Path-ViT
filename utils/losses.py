"""
Cox Proportional Hazards 생존분석 loss.
"""
import numpy as np
import pandas as pd
import torch


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Breslow 근사 Cox partial negative log-likelihood.

    risk가 클수록 위험(사망 가능성)이 높다고 가정하고, 각 사망 이벤트에 대해 그 시점까지
    생존해 있던 환자들(위험집합, risk set) 중 실제로 더 높은 risk를 예측했는지를 점수화한다.
    배치 전체가 하나의 위험집합 후보 모집단이 되므로, risk/time/event는 반드시 같은 배치
    (여러 환자)에서 함께 계산되어야 한다 — 환자 1명 단위로는 loss를 정의할 수 없다.

    Args:
        risk:  (B,) 예측 log-risk score
        time:  (B,) OS_time
        event: (B,) OS_event (1=사망, 0=censored)
    Returns:
        scalar loss. 배치 내 event가 하나도 없으면 gradient가 0인 스칼라를 반환한다.
    """
    risk, time, event = risk.float(), time.float(), event.float()

    # time 내림차순 정렬 시, i번째 원소의 위험집합 {j: time_j >= time_i} 은 정확히 앞쪽 0..i 구간이 된다
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]

    log_risk_set = torch.logcumsumexp(risk, dim=0)
    n_events = event.sum()
    if n_events == 0:
        return risk.sum() * 0.0
    return -((risk - log_risk_set) * event).sum() / n_events


def nll_surv_loss(
    h: torch.Tensor, y: torch.Tensor, event: torch.Tensor,
    alpha: float = 0.0, eps: float = 1e-7,
) -> torch.Tensor:
    """
    PORPOISE(Chen et al. 2022, Cancer Cell)의 discretized-time negative log-likelihood
    survival loss(Zadeh & Schmid 2020) — porpoise/utils/loss_func.py::nll_loss를 그대로
    포팅했다(수식 동일). 원본은 censorship을 1=censored로 쓰지만, 이 프로젝트는 cox_ph_loss와
    똑같이 event를 1=사망으로 쓰므로 인자 극성을 그쪽에 맞추고 함수 내부에서만 변환한다 —
    train.py 호출부에서 cox_ph_loss와 event 인자를 바꿔 낄 일이 없게 하려는 것.

    Args:
        h:     (B, n_bins) 시간-구간별 raw hazard logit(모델 risk_head 출력, sigmoid 이전).
        y:     (B,) 정답 시간-구간 index(0..n_bins-1). fit_survival_bins/digitize_survival_time으로 계산.
        event: (B,) OS_event, 1=사망 0=censored(cox_ph_loss와 동일 극성).
    """
    y = y.view(-1, 1).long()
    c = (1 - event).view(-1, 1).float()  # PORPOISE censorship 극성(1=censored)으로 변환
    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)
    # S(-1)=1(모든 환자가 -inf~0 구간엔 생존)로 패딩 — S_padded[:,0]=S(-1), S_padded[:,i+1]=S(i).
    S_padded = torch.cat([torch.ones_like(c), S], dim=1)
    s_prev = torch.gather(S_padded, 1, y).clamp(min=eps)      # S(y-1)
    h_this = torch.gather(hazards, 1, y).clamp(min=eps)       # h(y)
    s_this = torch.gather(S_padded, 1, y + 1).clamp(min=eps)  # S(y)
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -c * torch.log(s_this)
    loss = (1 - alpha) * (censored_loss + uncensored_loss) + alpha * uncensored_loss
    return loss.mean()


def hazard_to_risk(h: torch.Tensor) -> torch.Tensor:
    """discretized hazard logit(B, n_bins) -> 기존 스칼라 risk 파이프라인(C-index, checkpoint
    선택, cox_add 등) 호환용 스칼라 변환. PORPOISE 공식 관례(porpoise/utils/core_utils.py):
    risk = -sum(survival curve). survival 총합이 클수록(오래 생존 예측) risk는 낮아지도록 부호를
    뒤집는다 — 그래야 "risk가 클수록 위험"인 이 프로젝트의 C-index 계산 관례와 맞는다."""
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=-1)
    return -survival.sum(dim=-1)


def fit_survival_bins(times: np.ndarray, events: np.ndarray, n_bins: int = 4, eps: float = 1e-6) -> np.ndarray:
    """PORPOISE 관례(porpoise/porpoise_datasets/dataset_survival.py) 그대로 — 사망(event=1)한
    환자의 OS_time만으로 quantile 경계를 잡고(censored 환자는 경계 계산에서 제외), 양끝을 살짝
    넓혀 전체 환자(censored 포함)가 반드시 어느 한 구간에 들어가게 한다.

    **PORPOISE 원본과의 유일한 의도적 차이(leakage 방지)**: 원본 코드는 이 경계를 train+val
    전체(fold로 나누기 전 전체 코호트)에서 한 번만 계산해 재사용한다. 이 프로젝트는 RNA 유전자
    선정에서 "split 경계를 넘어간 전체-코호트 정보"가 실제 leakage로 확인된 전례가 있어
    (literature_1500_intersection, findings_backlog.md) — 반드시 그 fold의 **train split
    시점의 시간(time)만**으로 경계를 계산해서 넘겨야 한다(train.py 호출부 책임)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    uncensored = times[events == 1]
    # duplicates="drop": 표본이 작아 quantile 경계가 겹치면(동일 사망 시각 다수) 자동으로
    # 구간 수가 n_bins보다 줄어든다 — pd.qcut 표준 동작, 별도 처리 불필요.
    _, bin_edges = pd.qcut(uncensored, q=n_bins, retbins=True, duplicates="drop")
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_edges[0] = min(times.min(), bin_edges[0]) - eps
    bin_edges[-1] = max(times.max(), bin_edges[-1]) + eps
    return bin_edges


def digitize_survival_time(times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """fit_survival_bins의 경계로 시간을 0..n_bins-1 정수 구간 index로 변환 — PORPOISE 원본의
    pd.cut(..., right=False, include_lowest=True)와 동일 동작."""
    times = np.asarray(times, dtype=float)
    labels = pd.cut(times, bins=bin_edges, labels=False, right=False, include_lowest=True)
    return np.asarray(labels, dtype=np.int64)
