# M1~M7 2-seed(84/126)×5-fold 앙상블 — 최종 보고 수치 (95% CI 포함)

2026-08-16~21 작성·갱신. **최종 채택: seed 84+126만 pooling(2seed×5fold), seed42 제외.**
현재 코드베이스 기준(stage-stratify 반영, WSI 모델은 uni2native backbone), M1~M4는 baseline
(CLR/RLR-mult·wsi-extra-mlp 없음)으로 통일.

## ⚠️ 2026-08-21 — clinical cox_add ClinicalEncoder 실험 원복 완료, M1~M7 전부 확정

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
  (기존엔 항상 `coattn`+`concat` 또는 `coattn`+`cox_add`-ClinicalEncoder였음). HPC 재학습
  완료(`sbatch/final_m1m4_3seed_kfold_hpc/m2_pool_margin_3seed_kfold_array_hpc.sh`,
  2026-08-21 갱신, `--pooling-mode selfattn` 추가) — `free-gpu` 파티션 선점(preemption)으로
  seed126/fold4가 두 번 잘려서 세 번째 재제출로 완료, eval-external-ckpt 스윕까지 마쳐
  internal/external 둘 다 확보.

로컬 스모크테스트(uni2 backbone, 1epoch, fold0)로 M2/M7 두 경로 모두 에러 없이 끝까지
동작함을 확인(1-epoch 결과 자체는 폐기, 코드 검증용).

## ⚠️ 2026-08-21(추가) — M5도 raw_linear(MLP 없음)로 통일

M2/M4/M7의 clinical cox_add가 전부 raw feature 직결로 원복된 마당에 M5(Clinic only)만
`ClinicalEncoder`(MLP)를 쓸 architectural 근거가 없다는 지적(사용자) — clinical 브랜치는 이제
전 모델에서 예외 없이 raw feature 직결, 학습되는 nonlinear 인코더는 RNA/WSI 브랜치에만 쓴다는
일관된 원칙. `models/clinical_only.py`에 `raw_linear` 옵션 추가(2seed×5fold 실측 결과 MLP
버전과 오차범위 내: internal -0.006, external -0.019 — 성능상 손해가 거의 없어 통일 쪽을 채택).
M5는 어차피 참고용 baseline이라, 이 결과는 "clinical 신호는 단독으로는 약하고(M5, external
logP=0.18로 비유의) RNA와 결합해야(M7) 비로소 유의미해진다"는 서사로 정직하게 보고한다 — 아래
"M5 단독 vs 결합" 절 참고.

## 결과 표 (최종, 2seed×5fold)

| Model | Input Data | Internal C-index (95% CI) | Internal logP | External C-index (95% CI) | External logP | Model Structure | 상태 |
|---|---|---|---|---|---|---|---|
| M1 | WSI only | 0.5564 [0.4903, 0.6231] | 0.0024 | 0.5053 [0.4479, 0.5656] | 0.5418 | self-attention | 확정 |
| M2 | WSI + Clinic | 0.5587 [0.4857, 0.6282] | 0.0482 | **0.5001** [0.4404, 0.5594] | 0.6604 | self-attention + cox_add(raw feature 직결) | 확정 |
| M3 | WSI + RNAseq | 0.6594 [0.5880, 0.7256] | 0.0001 | 0.6245 [0.5717, 0.6755] | 0.0032 | co-attention + MLP + auxiliary task + concat | 확정 |
| M4 | WSI + Clinic + RNAseq | **0.6488** [0.5777, 0.7115] | 0.0029 | **0.6370** [0.5849, 0.6913] | 0.0004 | co-attention + MLP + auxiliary task + concat + cox_add(raw feature 직결) | 확정(재학습 불필요, 기존 체크포인트 유효) |
| M5 | Clinic only | 0.5475 [0.4802, 0.6139] | 0.3795 | 0.5324 [0.4737, 0.5902] | 0.1751 | raw feature 직결(Cox linear, MLP 없음) | 확정 |
| M6 | RNAseq only | 0.6518 [0.5834, 0.7153] | 0.0088 | 0.6146 [0.5585, 0.6721] | 0.0002 | MLP | 확정 |
| M7 | Clinic + RNAseq | **0.6552** [0.5820, 0.7204] | 0.0012 | **0.6221** [0.5661, 0.6782] | 0.0007 | MLP + cox_add(raw feature 직결, RNAEncoder) | 확정(재학습 불필요, ablation 체크포인트 재사용) |

전 모델 동일 조건: 2seed(84/126) × 5fold, ensemble(risk 평균) 후 c-index/log-rank p 1회 계산,
bootstrap 95% CI(환자 단위 resample, n=2000). WSI 모델(M1~M4)은 uni2native backbone. logP는
ensemble risk score 중앙값으로 고/저위험군을 나눈 log-rank test p-value.

**전 모델(M1~M7) 확정 — 재학습 대기 항목 없음.**

## M1→M2→M3→M4 사다리 해석 (2026-08-21 paired bootstrap 반영 후 수정)

⚠️ **아래 문단은 점추정치만 보고 쓴 최초 해석이다 — "부록 G: paired bootstrap delta" 검정 결과
Clinical/WSI 추가 효과는 통계적으로 유의하지 않은 것으로 나왔다. 논문에는 반드시 부록 G 이후의
수정된 해석을 따를 것.**

M1(0.5053) → M3(0.6245, WSI+RNA) → M4(0.6370, WSI+Clinic+RNA) — external 기준 WSI 단독보다
RNA 추가가 크게 기여하고(+0.119), 거기에 clinical(margin+staging)까지 더하면 추가로 소폭
개선(+0.013). Internal도 동일 경향(M1 0.5564 → M3 0.6594 → M4 0.6488, M3→M4는 근소 하락이지만
external은 개선 — internal/external gap이 M3(0.6594→0.6245, gap 0.035)보다 M4(0.6488→0.6370,
gap 0.012)에서 더 작다는 점이 M4/PMA의 핵심 novelty 후보).

**M2(WSI+Clinic, RNA 없음)는 external이 0.5001로 정확히 chance 수준** — M1(0.5053) 대비 개선이
전혀 없다. fold별로 보면 seed84(5개 fold 전부 0.51~0.55)와 seed126(5개 fold 전부 0.46~0.49)이
정반대 방향으로 갈려 평균이 우연히 상쇄된 게 아니라, 이 표본 크기에서 "WSI+clinical" 조합
자체가 진짜 신호를 못 낸다는 뜻으로 읽힌다.

참고(점추정치만): M6(0.6146) < M7(0.6221) < M4(0.6370) — clinical+RNA(M7)가 RNA 단독(M6)보다
낫고, WSI까지 더한 M4가 가장 좋다는 순서 자체는 맞지만, 아래 paired 검정에서 이 차이들은 유의
수준에 못 미친다.

### 수정된 해석 (부록 G 검정 결과 반영, 최종)

RNA 추가 3쌍(M1→M3, M5→M7, M2→M4) **전부 internal/external 둘 다 통계적으로 유의**(p<0.05,
95% CI가 0을 포함 안 함) — **이 코호트 크기에서 통계적으로 방어 가능한 성능 동력은 RNA뿐이다.**
Clinical 추가 3쌍(M1→M2, M6→M7, M3→M4)과 WSI 추가 3쌍(M5→M2, M6→M3, M7→M4)은 internal/
external 6쌍 전부 비유의 — 점추정치는 대부분 양의 방향(예: M7→M4 external +0.0149)이지만
95% CI가 전부 0을 포함해 우연과 통계적으로 구분되지 않는다. 따라서 "WSI/Clinic/RNA 세
모달리티가 단조적으로 누적 기여한다"는 서사는 **점추정치 수준의 관찰**이지 통계적으로
검증된 주장이 아니다 — 논문에서는 "RNA가 지배적 기여 인자이고, clinical/WSI의 추가 기여는
이 코호트 크기에서 통계적으로 유의하지 않다"로 톤을 낮춰 정직하게 서술할 것. M4의
internal/external gap이 M3보다 작다는 관찰(위 문단)은 c-index 자체의 우열과는 별개 지표라 이
결론의 영향을 받지 않는다.

## M5 단독 vs 결합 — clinical 신호는 약하고 RNA와 결합해야 유의미해진다

M5(clinical 단독, external 0.5324, logP=0.1751, 비유의) vs M7(clinical+RNA, external 0.6221,
logP=0.0007, 강하게 유의) — clinical 정보(age/sex+margin+staging) 자체는 단독으로는 chance
수준에 가깝고 통계적으로 유의하지 않지만, RNA와 결합하면 RNA 단독(M6, 0.6146)보다도 낫다.
즉 **clinical은 그 자체로 예후 예측력이 있다기보다, 강한 신호(RNA)의 risk 추정에 미세 보정을
더하는 보조적 역할**이라는 해석이 데이터와 일관된다 — 실제로 M2(WSI+Clinic)도 M1(WSI only,
0.5053) 대비 external 개선이 정확히 0(0.5001)이라 같은 패턴이 WSI 쪽에서도 그대로 재현됐다
(위 "M1→M2→M3→M4 사다리 해석" 절 참고). 이 해석은 "clinical cox_add가 raw feature 직결일 때
가장 좋다"(M2/M4/M5/M7 공통 결론)와도 부합 — 신호가 약한 변수는 학습되는 인코더(MLP)로
표현력을 늘리기보다 고전적 Cox 공변량처럼 단순하게 쓰는 편이 과적합을 피하고 낫다. 부록 G의
paired bootstrap 검정에서도 M5→M7(+RNA)은 internal/external 둘 다 유의(p=0.025/0.019)한 반면
M1→M2, M6→M7(+Clinic)은 둘 다 비유의로 확인돼 이 해석이 통계적으로도 뒷받침된다.

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

# M2 (재학습 완료 — HPC free-gpu 파티션 선점으로 seed126/fold4가 두 번 잘려서 세 번째 재제출로 완료)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M2_POOL_uni2native_SS_STG_R_DISP_COX_ADD_SELFATTN --seeds 84,126 --n-folds 5 --bootstrap 2000

# M5 (raw_linear, MLP 없음 — 2026-08-21 최종 채택), M6 (변경 없음)
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M5_STG_R_RAWLIN --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_kfold_preds.py --dataset tcga --model M6_INT1500 --seeds 84,126 --n-folds 5 --bootstrap 2000
python scripts/pool_multiseed_external_preds.py --dataset cptac --model M5_STG_R_RAWLIN --seeds 84,126 --n-folds 5 --bootstrap 2000
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

## 부록 F — M5 raw_linear vs MLP 비교 (2026-08-21, 채택 근거)

| | Internal (95% CI) | Internal logP | External (95% CI) | External logP |
|---|---|---|---|---|
| M5(MLP=ClinicalEncoder, 폐기) | 0.5536 [0.481, 0.626] | 0.0429 | 0.5511 [0.494, 0.608] | 0.3035 |
| **M5(raw_linear, 최종 채택)** | **0.5475** [0.480, 0.614] | 0.3795 | **0.5324** [0.474, 0.590] | 0.1751 |
| Δ(raw−MLP) | -0.0061 | | -0.0187 | |

두 버전 다 95% CI가 넓게 겹쳐 통계적으로 유의한 차이는 아니다 — 성능 손해가 사실상 없는
상태에서, M2/M4/M7과의 아키텍처 일관성(clinical은 항상 raw feature 직결)을 우선해 raw_linear를
채택. MLP 버전 수치는 참고용으로만 보존.

## 부록 G — Paired bootstrap on delta (모달리티 추가 효과 9쌍, 2026-08-21)

외부 피드백(paired bootstrap on delta 제안) 반영. 기존 bootstrap CI는 두 모델을 각자 따로
resample해서 CI가 겹치는지만 봤는데, 이건 같은 환자에 대한 두 모델의 예측이 짝지어져 있다는
정보를 안 쓰는 보수적인 방법이다. 대신 **매 bootstrap 회차마다 같은 환자 집합을 resample**해서
그 안에서 두 모델의 C-index를 같이 계산하고 **delta(=B−A)**를 2000번 기록 → delta 분포의
95% CI와 양측 p-value(`2 * min(P(delta≤0), P(delta≥0))`)를 본다. 짝지어진 데이터라 분산이
줄어 독립 비교보다 검정력이 높다. `scripts/paired_bootstrap_delta.py`(재사용 가능한 CLI),
`scripts/run_ladder_paired_bootstrap.py`(아래 9쌍 일괄 실행), `scripts/snapshot_final_preds.py`
(예측 CSV를 `paper/final_preds_snapshot/`으로 복사)로 구현 — **재학습/재추론 없이 이미 저장된
예측 CSV만 사용**.

9쌍 = 3가지 모달리티 추가 효과(+RNA, +Clinic, +WSI) x 각 3개 baseline 조합(2x2x2 factorial의
모든 인접 간선):

### Internal (tcga, N=152)

| 효과 | A | B | C(A) | C(B) | delta(B-A) | 95% CI | p | 판정 |
|---|---|---|---|---|---|---|---|---|
| +RNA | M1 | M3 | 0.5116† | 0.6594 | +0.1478 | [+0.0561, +0.2389] | 0.0000 | **유의** |
| +RNA | M5 | M7 | 0.5475 | 0.6552 | +0.1077 | [+0.0154, +0.1963] | 0.0250 | **유의** |
| +RNA | M2 | M4 | 0.5587 | 0.6488 | +0.0901 | [+0.0007, +0.1780] | 0.0480 | **유의** |
| +Clinic | M1 | M2 | 0.5116† | 0.5587 | +0.0471 | [-0.0354, +0.1280] | 0.2550 | 비유의 |
| +Clinic | M6 | M7 | 0.6518 | 0.6552 | +0.0033 | [-0.0326, +0.0377] | 0.8570 | 비유의 |
| +Clinic | M3 | M4 | 0.6594 | 0.6488 | -0.0105 | [-0.0418, +0.0219] | 0.5020 | 비유의 |
| +WSI | M5 | M2 | 0.5475 | 0.5587 | +0.0112 | [-0.0810, +0.0972] | 0.7990 | 비유의 |
| +WSI | M6 | M3 | 0.6518 | 0.6594 | +0.0075 | [-0.0350, +0.0534] | 0.7480 | 비유의 |
| +WSI | M7 | M4 | 0.6552 | 0.6488 | -0.0063 | [-0.0381, +0.0302] | 0.7240 | 비유의 |

### External (cptac, N=144)

| 효과 | A | B | C(A) | C(B) | delta(B-A) | 95% CI | p | 판정 |
|---|---|---|---|---|---|---|---|---|
| +RNA | M1 | M3 | 0.5053 | 0.6245 | +0.1192 | [+0.0468, +0.1933] | 0.0010 | **유의** |
| +RNA | M5 | M7 | 0.5324 | 0.6221 | +0.0897 | [+0.0140, +0.1689] | 0.0190 | **유의** |
| +RNA | M2 | M4 | 0.5001 | 0.6370 | +0.1369 | [+0.0639, +0.2110] | 0.0000 | **유의** |
| +Clinic | M1 | M2 | 0.5053 | 0.5001 | -0.0052 | [-0.0750, +0.0648] | 0.8890 | 비유의 |
| +Clinic | M6 | M7 | 0.6146 | 0.6221 | +0.0075 | [-0.0067, +0.0212] | 0.3020 | 비유의 |
| +Clinic | M3 | M4 | 0.6245 | 0.6370 | +0.0125 | [-0.0195, +0.0437] | 0.4320 | 비유의 |
| +WSI | M5 | M2 | 0.5324 | 0.5001 | -0.0323 | [-0.1195, +0.0557] | 0.4910 | 비유의 |
| +WSI | M6 | M3 | 0.6146 | 0.6245 | +0.0099 | [-0.0221, +0.0432] | 0.5520 | 비유의 |
| +WSI | M7 | M4 | 0.6221 | 0.6370 | +0.0149 | [-0.0182, +0.0502] | 0.4180 | 비유의 |

† M1이 낀 두 쌍(internal)의 C(M1)=0.5116은 본표의 M1 단독 수치(0.5564)와 다르다 — M3가
RNA-seq 보유 환자만 쓸 수 있어 M1보다 코호트가 33명 작고(185→152), paired 비교는 반드시
교집합(152명)에서만 계산되기 때문. M1 자체의 성능이 바뀐 게 아니라 비교 대상 코호트가
좁혀진 것 — 이 좁혀진 152명 부분집합에서는 M1이 0.5116으로 살짝 낮게 나온다는 뜻.

**결론: 9쌍 중 +RNA 3쌍(internal/external 6/6)만 유의, +Clinic·+WSI 6쌍(internal/external
12/12)은 전부 비유의.** RNA가 이 비교 프레임워크에서 유일하게 통계적으로 방어 가능한 성능
동력이다 — clinical/WSI의 추가 기여는 점추정치상 방향은 대체로 맞지만 이 코호트 크기(내부
152명, 외부 144명)에서 우연과 통계적으로 구분되지 않는다.

**재현**: `python scripts/snapshot_final_preds.py`로 예측 CSV 스냅샷 생성 후
`python scripts/run_ladder_paired_bootstrap.py` 실행하면 위 두 표가 그대로 재현된다. 개별
쌍만 다시 보려면 `python scripts/paired_bootstrap_delta.py --split {internal|external} ...`.

## 부록 H — M4 컴포넌트 ablation: RNA 보조과제(aux)와 attention dispersion (2026-08-30)

M4/PMA에 붙어있는 두 부가 장치(RNA 예측 보조과제 `--rna-aux-weight`, attention 공간 분산 특징
`--attn-dispersion`)가 실제로 성능에 기여하는지 각각 하나씩만 끄고 재학습해서(2seed×5fold,
나머지 레시피는 원본 M4와 동일) paired bootstrap으로 검정했다. 원본 M4(둘 다 있음, 본표
기준선) 대비:

| 효과 | Internal delta | Internal 95% CI | Internal p | External delta | External 95% CI | External p |
|---|---|---|---|---|---|---|
| +RNA aux (noaux→원본) | +0.0046 | [-0.0030, +0.0130] | 0.2770 | -0.0001 | [-0.0037, +0.0035] | 1.0000 | 비유의 |
| +attn-dispersion (nodisp→원본) | +0.0041 | [-0.0100, +0.0189] | 0.5720 | +0.0013 | [-0.0186, +0.0213] | 0.8930 | 비유의 |

각 ablation의 독립 점추정치(참고용, 위 delta의 baseline):

| 모델 | Internal (95% CI) | External (95% CI) |
|---|---|---|
| M4 원본(AUX+DISP 둘 다) | 0.6488 [0.5777, 0.7115] | 0.6370 [0.5849, 0.6913] |
| M4-noaux(DISP만) | 0.6442 [0.5728, 0.7081] | 0.6371 [0.5853, 0.6905] |
| M4-nodisp(AUX만) | 0.6448 [0.5746, 0.7090] | 0.6357 [0.5808, 0.6939] |

**결론: 둘 다 internal/external 전부 비유의(4/4) — 세 모델의 점추정치가 사실상 오차범위 안에서
동일하다.** RNA aux/attn-dispersion 둘 다 방향은 대체로 미세하게 양(있는 쪽이 근소 우세)이지만
delta가 거의 0에 붙어있다(특히 external의 RNA aux는 delta=-0.0001로 사실상 무효과). 앞선
부록 G의 +Clinic/+WSI 비유의 결과와 같은 패턴 — M4/PMA의 실제 성능은 "WSI+RNA 기본 구조 +
clinical cox_add"에서 나오고, aux task/dispersion 같은 부가 컴포넌트는 이 코호트 크기에서
유의한 추가 기여를 확인하지 못했다. 논문에는 "정교화를 시도했으나 유의한 개선은 확인되지
않았다"로 정직하게 서술할 것.

**재현**: `sbatch/final_m1m4_3seed_kfold_hpc/m4_pma_noaux_kfold_hpc.sh`,
`m4_pma_nodisp_kfold_hpc.sh`(학습) → `eval_external_ckpt_sweep_hpc.sh`(external, `M4_noaux`/
`M4_nodisp` 등록됨) → `pool_multiseed_kfold_preds.py`/`pool_multiseed_external_preds.py`
(태그 `PMA_uni2native_INT1500_SS_STG_R_DISP_COX_ADD`=noaux,
`PMA_uni2native_INT1500_SS_AUX_STG_R_COX_ADD`=nodisp) → `paired_bootstrap_delta.py`.
internal CSV가 실수로 삭제됐을 때는 `train.py --eval-internal-ckpt`(2026-08-30 추가,
`--eval-external-ckpt`와 동일 관례)로 재학습 없이 복구했다 — 원본과 byte 단위로 diff 없이
동일하게 복구됨을 로컬 스모크테스트로 확인.

## 데이터 보존 상태

pre-stratify 원본(2026-08-08) 백업은 `.logs/_archive_pma_baseline_prestratify_20260808/`에
보존. M4는 2026-08-17 체크포인트/CSV가 곧 최종본(재학습 없음). M7은 2026-08-20 ablation
"LEGCLIN" 체크포인트를 정식 태그로 복사한 것이 최종본 — 원본 LEGCLIN/LEGRNA 접미사 파일도
ablation 근거 자료로 당분간 보존(`.logs/kfold_preds/`, `models/checkpoint/`에 `_LEGCLIN`/
`_LEGRNA` 접미사로 남아있음). M2는 2026-08-21 HPC 재학습 완료(체크포인트는 HPC에만 있고 로컬엔
CSV만 동기화됨 — 재현하려면 HPC에서 다시 eval-external-ckpt 필요). 결과표를 만든 예측 CSV
140개(7모델 x seed{84,126} x 5fold x internal/external)는 `paper/final_preds_snapshot/`에
별도 스냅샷으로 보존(`scripts/snapshot_final_preds.py`) — `.logs/`가 온갖 ablation으로
어지러워도 이 폴더만 보면 최종표 재현 가능. **M1~M7 전 모델 확정, paired bootstrap(부록 G)까지
완료 — 결과 표는 이제 완료 상태.**
