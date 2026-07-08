"""
Cox Proportional Hazards 생존분석 loss.
"""
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
