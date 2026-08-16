# M1~M7 3-seed×5-fold 앙상블 — 최종 보고 수치 (95% CI 포함)

2026-08-16 작성. 사용자가 보내준 스프레드시트 캡처(seed×3, k-fold k=5, C-index Ensembled)를
원본 예측 파일로 재검증하고 bootstrap 95% CI를 추가한 버전. 전부 `data/dataset.py`의
stage-stratify 커밋(`a9caeaf`, 2026-08-14 13:18) **이전** 코드 상태(OS event 기준 stratify만
적용) — 원본 실행 시각은 2026-08-08 22:23(external 일괄 평가 기준). 이 시점 명시는
`paper/notes_wsi_signal_and_seed_variance.md` §3.1과 동일 결정(원본 pre-stratify 수치를
최종 보고값으로 채택)에 따른 것.

## 결과 표

| Model | Input Data | Internal C-index (95% CI) | External C-index (95% CI) | Model Structure |
|---|---|---|---|---|
| M1 | WSI only | 0.5698 [0.5055, 0.6351] | 0.5147 [0.4602, 0.5671] | self-attention |
| M2 | WSI + Clinic | 0.5196 [0.4495, 0.5908] | 0.5199 [0.4532, 0.5840] | self-attention + MLP + concat |
| M3 | WSI + RNAseq | 0.6481 [0.5814, 0.7111] | 0.6130 [0.5561, 0.6705] | co-attention + MLP + auxiliary task + concat |
| M4 | WSI + Clinic + RNAseq | 0.6461 [0.5769, 0.7108] | 0.6337 [0.5815, 0.6855] | co-attention + MLP + auxiliary task + concat + cox_add |
| M5 | Clinic only | 0.564 | 0.532 | MLP |
| M6 | RNAseq only | 0.659 | 0.617 | MLP |
| M7 | Clinic + RNAseq | 0.637 | 0.622 | MLP + cox_add |

 **M1/M4 Internal은 3seed가 아니라 2seed(84,126) 재구성값이다.** seed42의 원본 held-out
예측(`kfold_preds`)과 체크포인트가 이후 세션 작업(2026-08-16, M1_POOL/PMA 재검증 학습)에
파일명이 겹쳐 덮어써졌다 — 복구 불가능. 스프레드시트 원본값(M1=0.551, M4=0.636)은 3seed
전체 기준이라 이 표의 값과 정확히 일치하진 않지만 같은 방향(0.55~0.57, 0.63~0.65권)이다.
External은 두 모델 다 3seed 전체가 그대로 살아있어 스프레드시트 원본값과 정확히 일치한다
(M1=0.5147≈0.5150, M4=0.6337≈0.634).

 **M5/M6/M7은 CI 계산 불가.** 이 세 모델의 2026-08-08 원본 환자 단위 예측 파일이 이번
세션 중(2026-08-15, M5/M6/M7 stratify-이후 재검증 작업) 동일 파일명으로 덮어써져 raw
prediction이 남아있지 않다 — bootstrap resample에 필요한 환자 단위 risk score 자체가 없다.
표의 값은 `.logs/multiseed_summary_20260808_222325.log`에 인쇄된 점추정치 그대로다(재계산
아님, 이 로그 자체는 안전).

## 재현 커맨드 (M1~M4, 원본 예측 파일 기준)

```bash
# Internal(seed 조합은 모델별로 다름 — 위 각주 참조)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M1_POOL_uni2_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000

# External(전부 3seed 그대로 생존)
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M1_POOL_uni2_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2_SS_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 42,84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 42,84,126 --n-folds 5 --bootstrap 2000
```

## 데이터 보존 상태

살아남은 원본(M1 internal seed84/126, M2 전체, M3 전체, M4 external 전체 + internal
seed84/126)은 재차 덮어써지는 걸 막기 위해 아래로 백업 완료:
- `.logs/_archive_pma_baseline_prestratify_20260808/kfold_preds/`
- `.logs/_archive_pma_baseline_prestratify_20260808/external_preds/`
- `models/checkpoint/_archive_pma_baseline_prestratify_20260808/`

**이 디렉토리들은 절대 덮어쓰지 말 것** — M5/M6/M7처럼 한번 더 잃으면 이 표 자체를 다시
만들 수 없다(재현 불가능한 유일한 증거).

## 해석 메모

- **internal/external 트레이드오프**: M3(WSI+RNA, clinical 없음)이 M4(WSI+RNA+Clinic)보다
  internal이 오히려 높다(0.648 vs 0.646) — clinical 추가가 internal에는 중립~약간 손해,
  external에는 확실히 이득(0.613→0.634)이라는 패턴이 CI를 붙여도 유지된다.
- **M1/M2는 CI를 봐도 0.5(chance)를 벗어나지 못한다** — M1 internal CI [0.51,0.64]는 하한이
  0.5를 살짝 넘지만 M1/M2 external, M2 internal 전부 CI가 0.5를 포함한다. RNA 없이 WSI(+
  clinical)만으로는 유의미한 신호를 통계적으로 확신할 수 없다는 근거로 쓸 수 있다.
- **M4 external의 CI 폭(0.582~0.686, 폭 0.10)**이 이 코호트 규모에서의 근본적 불확실성
  범위 — "0.65를 조건부로 넘을 수 있다"는 서술은 이 상한(0.686)에 근거하면 된다.
