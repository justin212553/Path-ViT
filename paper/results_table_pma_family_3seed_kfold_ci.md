# M1~M7 2-seed(84/126)×5-fold 앙상블 — 최종 보고 수치 (95% CI 포함)

2026-08-16~21 작성·갱신. **최종 채택: seed 84+126만 pooling(2seed×5fold), seed42 제외.**
현재 코드베이스 기준(stage-stratify 반영, WSI 모델은 uni2native backbone), M1~M4는 baseline
(CLR/RLR-mult·wsi-extra-mlp 없음)으로 통일.

## ⚠️ 2026-08-21 — clinical cox_add ClinicalEncoder 실험 원복 완료 (M4/M7 확정, M2만 재학습 대기)

2026-08-20에 "RNA cox_add와 대칭 맞추자"며 clinical cox_add를 raw feature 직결에서
ClinicalEncoder(MLP) 경유로 바꿨는데, M7 기준 2×2 ablation(`--legacy-rna-encoder`/
`--legacy-clinical-coxadd`, 각 2seed×5fold)으로 원인을 분리해보니 internal -0.027 하락의
거의 전부(-0.025)가 이 clinical 변경 때문이었고 RNA 인코더 교체(RNAEncoderExtend→RNAEncoder)는
거의 무해(-0.006)했다. **결론: RNA 인코더 교체는 유지, clinical cox_add는 raw feature 직결로
원복.** `models/vit_pma.py`(M4)·`models/vit_m2_pool.py`(M2)·`models/clinical_rna_only.py`(M7)를
모두 원복했고, ablation용 `--legacy-rna-encoder`/`--legacy-clinical-coxadd` 플래그는 결론이
났으므로 삭제(raw feature 직결이 cox_add의 유일한/영구 구현).

추가로 M2는 이번 기회에 pooling_mode도 `coattn`(clinical이 4개 pooling 관점의 co-attention
query)에서 `selfattn`(clinical이 pooling에 전혀 관여하지 않음)으로 단순화했다 — M2는 baseline/
floor 모델이라 M3/M4를 이길 일이 없으므로, 성능보다 구조 단순성을 택함(사용자 결정).

**코드 원복 후 각 모델 상태:**
- **M4**: `vit_pma.py` 코드를 원복하면 원래(2026-08-17에 생성된, ClinicalEncoder 변경이 있기
  전) 체크포인트/CSV와 정확히 같은 레시피가 된다 — 실제로 기존 pred CSV(2026-08-17 생성,
  ClinicalEncoder 변경보다 3일 앞섬)를 재풀링해 internal=0.6488/external=0.6370이 그대로
  재현됨을 확인. **재학습 불필요, 기존 수치가 곧 최종 수치.**
- **M7**: RNA 인코더 교체+clinical raw 원복 조합은 2×2 ablation의 "LEGCLIN" 콤보
  (`legacy_clinical_coxadd=True`, RNA는 기본값이라 이미 신버전)와 코드상 완전히 동일 —
  이미 2seed×5fold 체크포인트가 존재해 **재학습 없이** `--eval-external-ckpt` 스윕으로
  external만 새로 계산하고, internal CSV는 LEGCLIN 태그 파일을 정식 태그(`M7_INT1500_STG_R_COX_ADD`)
  이름으로 복사해 재풀링. **재학습 불필요.**
- **M2**: `selfattn`+`cox_add`(raw) 조합은 이전에 한 번도 정식으로 학습된 적이 없는 새 조합
  (기존엔 항상 `coattn`+`concat` 또는 `coattn`+`cox_add`-ClinicalEncoder였음). **HPC 재학습
  필요** — `sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_margin_3seed_kfold_array_hpc.sh`
  (2026-08-21 갱신, `--pooling-mode selfattn` 추가, 태그
  `M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN`).

로컬 스모크테스트(uni2 backbone, 1epoch, fold0)로 M2/M7 두 경로 모두 에러 없이 끝까지
동작함을 확인(1-epoch 결과 자체는 폐기, 코드 검증용).

## 결과 표 (최종, 2seed×5fold)

| Model | Input Data | Internal C-index (95% CI) | Internal logP | External C-index (95% CI) | External logP | Model Structure | 상태 |
|---|---|---|---|---|---|---|---|
| M1 | WSI only | 0.5564 [0.4903, 0.6231] | 0.0024 | 0.5053 [0.4479, 0.5656] | 0.5418 | self-attention | 확정 |
| M2 | WSI + Clinic | — | — | — | — | self-attention + cox_add(raw feature 직결) | **HPC 재학습 대기**(selfattn+cox_add 신규 조합) |
| M3 | WSI + RNAseq | 0.6594 [0.5880, 0.7256] | 0.0001 | 0.6245 [0.5717, 0.6755] | 0.0032 | co-attention + MLP + auxiliary task + concat | 확정 |
| M4 | WSI + Clinic + RNAseq | **0.6488** [0.5777, 0.7115] | 0.0029 | **0.6370** [0.5849, 0.6913] | 0.0004 | co-attention + MLP + auxiliary task + concat + cox_add(raw feature 직결) | 확정(재학습 불필요, 기존 체크포인트 유효) |
| M5 | Clinic only | 0.5536 [0.4807, 0.6260] | 0.0429 | 0.5511 [0.4937, 0.6083] | 0.3035 | MLP | 확정 |
| M6 | RNAseq only | 0.6518 [0.5834, 0.7153] | 0.0088 | 0.6146 [0.5585, 0.6721] | 0.0002 | MLP | 확정 |
| M7 | Clinic + RNAseq | **0.6552** [0.5820, 0.7204] | 0.0012 | **0.6221** [0.5661, 0.6782] | 0.0007 | MLP + cox_add(raw feature 직결, RNAEncoder) | 확정(재학습 불필요, ablation 체크포인트 재사용) |

전 모델 동일 조건: 2seed(84/126) × 5fold, ensemble(risk 평균) 후 c-index/log-rank p 1회 계산,
bootstrap 95% CI(환자 단위 resample, n=2000). WSI 모델(M1~M4)은 uni2native backbone. logP는
ensemble risk score 중앙값으로 고/저위험군을 나눈 log-rank test p-value.

**M2만 HPC 재학습 대기 — 완료 후 이 표의 M2 셀을 채우고 아래 사다리 해석도 완성할 것.**

## M2→M3→M4 사다리 해석 (M2 재학습 전까지 잠정)

M1(0.5053) → M3(0.6245, WSI+RNA) → M4(0.6370, WSI+Clinic+RNA) — external 기준 WSI 단독보다
RNA 추가가 크게 기여하고(+0.119), 거기에 clinical(margin+staging)까지 더하면 추가로 소폭
개선(+0.013). Internal도 동일 경향(M1 0.5564 → M3 0.6594 → M4 0.6488, M3→M4는 근소 하락이지만
external은 개선 — **internal/external gap이 M3(0.6594→0.6245, gap 0.035)보다 M4(0.6488→0.6370,
gap 0.012)에서 더 작다**는 점이 M4/PMA의 핵심 novelty 후보). M2(WSI+Clinic만, RNA 없음)가
채워지면 M1→M2→M3→M4 전체 사다리에서 "clinical 단독 기여분(M1→M2)"과 "clinical이 RNA와
결합했을 때의 기여분(M3→M4)"을 분리해서 볼 수 있다.

참고: M6(0.6146) < M7(0.6221) < M4(0.6370) — clinical+RNA(M7)가 RNA 단독(M6)보다 낫고,
WSI까지 더한 M4가 가장 좋다 — 세 모달리티(WSI/Clinic/RNA)가 external에서 단조적으로 누적
기여한다는 일관된 그림.

## 재현 커맨드

```bash
# M1, M3 (baseline, uni2native, 변경 없음 — 그대로 유효)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M1_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M1_POOL_uni2native_SS_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_NOCLINICAL_DISP --seeds 84,126 --n-folds 5 --bootstrap 2000

# M4 (재학습 불필요 — 2026-08-17 체크포인트/CSV가 코드 원복 후 레시피와 정확히 일치, 그대로 유효)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model PMA_uni2native_INT1500_SS_AUX_STG_R_DISP_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000

# M2 — HPC 재학습 필요(코드 재동기화 후)
#   sbatch sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_margin_3seed_kfold_array_hpc.sh
#   완료 후 eval-external-ckpt 스윕(scripts/final_eval_external_ckpt_sweep.py, CONFIGS에 selfattn
#   버전 추가 필요)까지 반드시 실행(일반 --fold 경로가 external CSV를 안 남기므로)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000

# M5, M6 (internal/external 모두 정상 저장됨, 변경 없음)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M5_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M6_INT1500 --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M5_STG_R --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M6_INT1500 --seeds 84,126 --n-folds 5 --bootstrap 2000

# M7 (재학습 불필요 — ablation "LEGCLIN" 체크포인트를 정식 태그로 복사 + eval-external-ckpt 스윕으로 external만 새로 계산)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M7_INT1500_STG_R_COX_ADD --seeds 84,126 --n-folds 5 --bootstrap 2000
```

## 부록 A — 3seed(42/84/126) 전체 pooled 표 (참고용, 최종 채택 아님, 옛 external 버그 포함 가능)

| Model | Internal (95% CI) | External (95% CI) |
|---|---|---|
| M1 | 0.5411 [0.4779, 0.6049] | 0.5033 [0.4423, 0.5646] |
| M2 | 0.5466 [0.4760, 0.6149] | *(폐기 — coattn+concat 구버전, 지금은 selfattn+cox_add로 레시피 자체가 다름)* |
| M3 | 0.6599 [0.5898, 0.7237] | 0.6031 [0.5458, 0.6575] |
| M4 | 0.6435 [0.5725, 0.7092] | *(3seed 기준 참고치, 최종 채택은 2seed 위 본표)* |
| M5 | 0.5647 [0.4934, 0.6353] | *(폐기 — external 버그 값)* |
| M6 | 0.6626 [0.5911, 0.7269] | *(폐기 — external 버그 값)* |
| M7 | *(폐기 — RNAEncoderExtend 구버전)* | *(폐기 — external 버그 값)* |

## 부록 B — seed별 breakdown (M4/M7 신버전 이전 자료, 참고용)

| Seed | M2(구버전 coattn) | M3 | M4(2026-08-17, 최종과 동일 코드) | M5 | M6 | M7(RNAEncoderExtend 구버전) |
|---|---|---|---|---|---|---|
| 42 | 0.4357 | 0.5649 | 0.5880 | 0.5242 | 0.6186 | 0.6130 |
| 84 | 0.5131 | 0.6083 | 0.5948 | 0.5527 | 0.6214 | 0.6256 |
| 126 | 0.5024 | 0.6225 | 0.6484 | 0.5471 | 0.6206 | 0.6244 |

## 부록 C — CLR20/RLR20/wsi-extra-mlp hybrid (M2/M3/M4, 채택 안 함, 3seed 기준, 구버전)

| Model | Internal (95% CI) | External (95% CI) | baseline 대비 |
|---|---|---|---|
| M2 hybrid | 0.5409 [0.4664, 0.6147] | 0.5366 [0.4705, 0.5975] | external +0.047 |
| M3 hybrid | 0.6636 [0.5928, 0.7299] | 0.6128 [0.5561, 0.6703] | 둘 다 소폭 개선 |
| M4 hybrid | 0.6597 [0.5881, 0.7249] | 0.6145 [0.5584, 0.6704] | external -0.014 |

M4 hybrid는 best_epoch 15개 중 11개가 한 자릿수(1~9)로 튀는 조기과적합 스파이크가 확인돼
(M4 baseline은 15개 중 13개가 epoch≥16으로 정상) 채택하지 않음 — M2/M3도 사다리 일관성을
위해 baseline으로 통일.

## 부록 D — M7 clinical cox_add ablation 원본 데이터 (2026-08-20, 결론의 근거)

2×2 ablation(각 조합 2seed×5fold, internal만 — 이 시점엔 external CSV 저장 안 함):

| 조합 | RNA 인코더 | clinical cox_add | Internal (95% CI) |
|---|---|---|---|
| 신버전(원복 전) | RNAEncoder | ClinicalEncoder 경유 | 0.6341 [0.559, 0.704] |
| LEGRNA만 | RNAEncoderExtend | ClinicalEncoder 경유 | (RNA만 legacy, 상세는 세션 로그 참조) |
| **LEGCLIN만(=현재 최종 채택)** | **RNAEncoder** | **raw feature 직결** | **0.6552 [0.582, 0.720]** |
| 둘 다 구버전 | RNAEncoderExtend | raw feature 직결 | 0.6612 [0.590, 0.728] (2026-08-08 원본, pre-stratify) |

RNA 인코더 교체 단독 효과(신버전 vs LEGRNA만): -0.006(거의 무해). clinical 구조 변경 단독
효과(신버전 vs LEGCLIN만): -0.025(하락의 대부분). → clinical은 raw feature 직결 유지, RNA는
신버전(RNAEncoder) 채택 — 위 본표의 M7 최종 수치(0.6552/0.6221)가 이 "LEGCLIN만" 콤보를
그대로 정식 태그로 승격한 것.

## 부록 E — M6 vs M7 순서 회복 확인

한때(2026-08-20, ClinicalEncoder 경유 버전) M7(internal 0.6341, external 0.6100)이 M6(internal
0.6518, external 0.6146)에 internal/external 둘 다 뒤지는 역전이 발생했었다 — 사용자가 원래
기대하던 M6<M7 순서와 반대. clinical cox_add를 raw feature 직결로 원복한 뒤(본표 최종
수치) M7(0.6552/0.6221)이 M6(0.6518/0.6146)을 internal/external 둘 다에서 다시 앞서 원래
기대했던 순서가 회복됨 — clinical 신호가 약해 MLP(ClinicalEncoder)를 거치면 오히려 과적합만
늘어난다는 이번 ablation의 결론과 일관된 결과.

## 데이터 보존 상태

pre-stratify 원본(2026-08-08) 백업은 `.logs/_archive_pma_baseline_prestratify_20260808/`에
보존. M4는 2026-08-17 체크포인트/CSV가 곧 최종본(재학습 없음). M7은 2026-08-20 ablation
"LEGCLIN" 체크포인트를 정식 태그로 복사한 것이 최종본 — 원본 LEGCLIN/LEGRNA 접미사 파일도
ablation 근거 자료로 당분간 보존(`.logs/kfold_preds/`, `models/checkpoint/`에 `_LEGCLIN`/
`_LEGRNA` 접미사로 남아있음). **M2 HPC 재학습 완료 후에는 이 파일을 한 번 더 정리(부록 D/E
정리, 사다리 해석 완성)할 것.**
