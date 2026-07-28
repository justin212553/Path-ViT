"""
SAM(Sharpness-Aware Minimization, Foret et al. 2020) — 현재 지점의 loss만 낮추는 대신, 그
지점 주변(반경 rho) 최악점 기준 gradient로 업데이트해 flat minimum을 명시적으로 찾는다.

2026-07-27: 이 프로젝트 규모(코호트 91~152명, 82만 파라미터)에서는 train loss를 거의 0으로
만드는 파라미터 조합이 무수히 많고(under-determined 최적화), 그중 어디로 수렴하느냐가 순전히
초기화에 좌우된다는 게 확인됐다(seed168 vs seed42 초기화 교체만으로 external c-index +0.04~0.05).
"여러 local minimum 중 뭘 고르느냐"가 문제라면, flat minimum을 직접 찾도록 학습 방식 자체를
바꾸는 SAM이 dropout/subsampling 같은 정규화보다 더 직접적인 대응일 수 있다는 가설의 구현.

base optimizer(AdamW 등)를 감싸는 래퍼. 사용법(2-pass):
    loss = compute_loss(); loss.backward()
    sam.first_step()                    # 근방 최악점으로 이동
    loss2 = compute_loss(); loss2.backward()  # 같은 배치를 최악점에서 재평가
    sam.second_step()                   # 원위치 복귀 + 실제 base optimizer step
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_kwargs):
        if rho < 0:
            raise ValueError(f"rho는 0 이상이어야 합니다: {rho}")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self, skip_update: bool = False):
        """원위치(perturbation 이전)로 복귀한 뒤, skip_update=False면 base optimizer로 실제
        파라미터 업데이트를 적용한다. skip_update=True(2차 forward가 non-finite였을 때)면
        복귀만 하고 업데이트는 건너뛴다."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        if not skip_update:
            self.base_optimizer.step()

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(2).to(device)
                for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
            2,
        )

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
