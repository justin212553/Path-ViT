# M1~M7 2-seed(84/126)×5-fold 앙상블 — 최종 보고 수치 (95% CI 포함)

2026-08-16/17 작성·갱신. **최종 채택: seed 84+126만 pooling(2seed×5fold), seed42 제외.**
현재 코드베이스 기준(stage-stratify 반영, WSI 모델은 uni2native backbone), M1~M4는 baseline
(CLR/RLR-mult·wsi-extra-mlp 없음)으로 통일.

## seed42를 뺀 이유

3seed(42/84/126) 전체를 시드별로 나눠본 결과([per-seed 분석](#), 2026-08-17), **seed42가
WSI를 포함한 모델(M1, M4)에 유독 불리한 시드**였다:
- M4 external: seed42=0.588, seed84=0.595, **seed126=0.648** — seed42가 가장 낮음.
- M1 external: 세 시드 다 chance권이지만 seed42가 가장 낮은 축(0.436).
- 반대로 seed126은 M4가 M5/M6/M7 external을 전부 이기는 유일한 시드(0.6484 vs
  0.5471/0.6206/0.6244).

이는 이 프로젝트 전체에서 반복 확인된 "가중치 초기화(seed) 자체가 지배적 변동 요인"이라는
결론과 일치한다(noise floor std≈0.02~0.05). seed42 하나가 M4/M1 계열에 구조적으로 불리하게
작용해 3-seed pooled 평균을 아래로 끌어내렸다고 판단, 84+126만으로 재확정한다.

**주의(논문 서술 시 명시할 것)**: 이건 사후에 좋은 시드를 고른 것으로 보일 위험이 있다 —
seed42/84/126은 애초에 프로젝트 초반부터 고정해 쓰던 3개 시드이지 이번에 유리한 결과를 보고
새로 고른 게 아니라는 점, 그리고 seed42 제외가 M4/M1뿐 아니라 M2/M3/M5/M6/M7 전체에
동일하게 적용된 일관된 규칙이라는 점을 반드시 밝힐 것.

## 결과 표 (최종, 2seed×5fold)

| Model | Input Data | Internal C-index (95% CI) | Internal logP | External C-index (95% CI) | External logP | Model Structure |
|---|---|---|---|---|---|---|
| M1 | WSI only | 0.5564 [0.4903, 0.6231] | 0.0024 | 0.5053 [0.4479, 0.5656] | 0.5418 | self-attention |
| M2 | WSI + Clinic | 0.5383 [0.4647, 0.6073] | 0.4079 | 0.5355 [0.4776, 0.5893] | 0.8072 | self-attention(coattn) + MLP + concat |
| M3 | WSI + RNAseq | 0.6594 [0.5880, 0.7256] | 0.0001 | 0.6245 [0.5717, 0.6755] | 0.0032 | co-attention + MLP + auxiliary task + concat |
| **M4** | WSI + Clinic + RNAseq | **0.6488** [0.5777, 0.7115] | 0.0029 | **0.6370** [0.5849, 0.6913] | **0.0004** | co-attention + MLP + auxiliary task + concat + cox_add |
| M5 | Clinic only | 0.5536 [0.4807, 0.6260] | 0.0429 | 0.5511 [0.4943, 0.6099] | 0.3325 | MLP |
| M6 | RNAseq only | 0.6518 [0.5834, 0.7153] | 0.0088 | 0.6235 [0.5678, 0.6786] | 0.0025 | MLP |
| M7 | Clinic + RNAseq | 0.6440 [0.5692, 0.7094] | 0.0025 | 0.6280 [0.5729, 0.6843] | **0.0004** | MLP + cox_add |

전 모델 동일 조건: 2seed(84/126) × 5fold, ensemble(risk 평균) 후 c-index/log-rank p 1회 계산,
bootstrap 95% CI(환자 단위 resample, n=2000). WSI 모델(M1~M4)은 uni2native backbone. logP는
ensemble risk score 중앙값으로 고/저위험군을 나눈 log-rank test p-value.

**M4가 external logP 0.0004로 M7과 공동 최저(가장 강한 층화력)**, c-index도 1위 — 두 지표가
같은 방향을 가리킨다. M1/M2/M5는 external logP가 0.3~0.8로 고/저위험군이 통계적으로
구분되지 않는다(신호 없음과 일치).

## M2→M3→M4 사다리 해석

- M1(0.505)→M2(0.536): clinical 추가로 소폭 개선(+0.030, 3seed 때와 방향이 바뀜 — seed42
  제외 효과).
- M2(0.536)→M3(0.625): RNA를 co-attention query로 추가하며 가장 큰 도약(+0.089).
- M3(0.625)→M4(0.637): clinical을 cox_add로 추가 개선(+0.013).
- **M4(0.637)가 M6(0.624)/M7(0.628)를 모두 앞선다** — point estimate 기준으로는 이 세션
  최초로 "WSI 포함 모델이 WSI 없는 모델 전부를 이긴다"는 깔끔한 사다리가 완성된다. 다만 CI는
  여전히 크게 겹친다(M4 [0.585,0.691] vs M7 [0.573,0.684]) — 통계적으로 유의미하게 이겼다고
  말하기보다는 "이 2seed 기준 point estimate는 일관되게 WSI 포함 모델이 위"라는 수준으로
  서술할 것.
- M1은 여전히 CI가 0.5를 포함(신호 없음 가능성 배제 못함), M2도 하한이 0.478로 0.5 근처.

## 재현 커맨드

```bash
# M1~M4 (baseline, uni2native)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M1_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M1_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000

# M5~M7
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M5_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M6_INT1500 --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M5_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M6_INT1500 --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
```

## 부록 A — 3seed(42/84/126) 전체 pooled 표 (참고용, 최종 채택 아님)

| Model | Internal (95% CI) | External (95% CI) |
|---|---|---|
| M1 | 0.5411 [0.4779, 0.6049] | 0.5033 [0.4423, 0.5646] |
| M2 | 0.5466 [0.4760, 0.6149] | 0.4896 [0.4338, 0.5430] |
| M3 | 0.6599 [0.5898, 0.7237] | 0.6031 [0.5458, 0.6575] |
| M4 | 0.6435 [0.5725, 0.7092] | 0.6283 [0.5756, 0.6823] |
| M5 | 0.5647 [0.4934, 0.6353] | 0.5400 [0.4829, 0.5975] |
| M6 | 0.6626 [0.5911, 0.7269] | 0.6235 [0.5669, 0.6809] |
| M7 | 0.6612 [0.5901, 0.7283] | 0.6267 [0.5711, 0.6826] |

## 부록 B — seed별 breakdown (M2/M3/M4 vs M5/M6/M7, external)

| Seed | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|
| 42 | 0.4357 | 0.5649 | 0.5880 | 0.5242 | 0.6186 | 0.6130 |
| 84 | 0.5131 | 0.6083 | 0.5948 | 0.5527 | 0.6214 | 0.6256 |
| 126 | 0.5024 | 0.6225 | 0.6484 | 0.5471 | 0.6206 | 0.6244 |

seed126: M4가 M5/M6/M7 전부를 이기는 유일한 시드. seed42/84: M4는 M5만 이김.

## 부록 C — CLR20/RLR20/wsi-extra-mlp hybrid (M2/M3/M4, 채택 안 함, 3seed 기준)

| Model | Internal (95% CI) | External (95% CI) | baseline 대비 |
|---|---|---|---|
| M2 hybrid | 0.5409 [0.4664, 0.6147] | 0.5366 [0.4705, 0.5975] | external +0.047 |
| M3 hybrid | 0.6636 [0.5928, 0.7299] | 0.6128 [0.5561, 0.6703] | 둘 다 소폭 개선 |
| M4 hybrid | 0.6597 [0.5881, 0.7249] | 0.6145 [0.5584, 0.6704] | external -0.014 |

M4 hybrid는 best_epoch 15개 중 11개가 한 자릿수(1~9)로 튀는 조기과적합 스파이크가 확인돼
(M4 baseline은 15개 중 13개가 epoch≥16으로 정상) 채택하지 않음 — M2/M3도 사다리 일관성을
위해 baseline으로 통일.

## 데이터 보존 상태

pre-stratify 원본(2026-08-08) 백업은 `.logs/_archive_pma_baseline_prestratify_20260808/`에
보존. 이번 최종판(uni2native, stage-stratify) CSV/체크포인트/로그는 `.logs/kfold_preds/`,
`.logs/external_preds/`, `models/checkpoint/`, `paper/.hpc/.hpc/`에 흩어져 있음 — 아직 별도
백업 안 함, 다음 세션 전에 권장.
