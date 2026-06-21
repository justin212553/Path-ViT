"""
Ray Tune 기반 하이퍼파라미터 탐색 드라이버 스크립트
탐색 대상: lr, weight_decay, warmup_epochs, dropout, (embed_dim, num_heads) 조합, num_transformer_layers
스케줄러:  ASHA — 성능이 낮은 trial을 조기 종료해 탐색 효율을 높임
지표:      val_auc_roc (maximize)

trainable 본체는 tune_trainable.py에 있다 (이 스크립트는 __main__으로 실행되므로
여기에 train_fn을 직접 정의하면 cloudpickle이 by-value 직렬화를 시도하다
torch.backends.cudnn 관련 객체에서 실패한다 — tune_trainable.py 상단 docstring 참조).

사용 예:
    python tune.py
(탐색 규모는 파일 하단의 NUM_SAMPLES / TUNE_EPOCHS / GPUS_PER_TRIAL / CPUS_PER_TRIAL 상수로 조절)
"""
from pathlib import Path

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from config import Config
from tune_trainable import EMBED_HEAD_CHOICES, train_fn

SEARCH_SPACE = {
    "lr":                     tune.loguniform(1e-6, 1e-5),
    "weight_decay":           tune.loguniform(1e-5, 1e-4),
    "warmup_epochs":          tune.choice([1, 2, 3, 4, 5]),
    "dropout":                tune.uniform(0.0, 0.4),
    "embed_head":             tune.choice(EMBED_HEAD_CHOICES),
    "num_transformer_layers": tune.choice([2, 4, 6, 8]),
}

NUM_SAMPLES    = 20   # 탐색할 trial(하이퍼파라미터 조합) 수
TUNE_EPOCHS    = 8    # trial당 학습 epoch 수 (본 학습보다 짧게)
GPUS_PER_TRIAL = 1.0
CPUS_PER_TRIAL = 4.0


def main():
    base_cfg = Config()

    asha = ASHAScheduler(
        metric="val_auc_roc",
        mode="max",
        max_t=TUNE_EPOCHS,
        grace_period=2,
        reduction_factor=2,
    )

    trainable = tune.with_resources(
        tune.with_parameters(train_fn, base_cfg=base_cfg, tune_epochs=TUNE_EPOCHS),
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )

    tuner = tune.Tuner(
        trainable,
        param_space=SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            scheduler=asha,
            num_samples=NUM_SAMPLES,
        ),
        run_config=tune.RunConfig(
            name="path_vit_raytune",
            storage_path=str(Path(__file__).parent / "ray_results"),
        ),
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_auc_roc", mode="max")
    embed_dim, num_heads = best.config["embed_head"]
    print("\n=== Best trial ===")
    print(f"  val_auc_roc             : {best.metrics['val_auc_roc']:.4f}")
    print(f"  lr                      : {best.config['lr']:.3e}")
    print(f"  weight_decay            : {best.config['weight_decay']:.3e}")
    print(f"  warmup_epochs           : {best.config['warmup_epochs']}")
    print(f"  dropout                 : {best.config['dropout']:.3f}")
    print(f"  embed_dim               : {embed_dim}")
    print(f"  num_heads               : {num_heads}")
    print(f"  num_transformer_layers  : {best.config['num_transformer_layers']}")


if __name__ == "__main__":
    main()
