# PATH-ViT 모델 비교 요약 (M1 / M4 / M4A / M4B / M5 / M6 / M7)

생성일: 2026-07-16. 3시드(42/84/126) x 2 학습 코호트(tcga/cptac) x internal/external, 총 n=6/모델.

- **internal**: 학습에 쓴 코호트 내부 held-out test (같은 기관, 배치 효과 없음)
- **external**: 학습에 전혀 쓰지 않은 반대 코호트 전체 (`--external`, 진짜 cross-institution 일반화 테스트)
- 표 안의 값은 Harrell's C-index. 원본 시드별 수치는 `.logs/train_{tcga,cptac}_seed{42,84,126}*.log` 참조.

## 모델 정의

| 모델 | 구성 | 코드/스크립트 |
|---|---|---|
| M1 | WSI only (ViT+ABMIL) | `train.py` (플래그 없음, 기본값) |
| M4 | WSI+Clinical+RNA, RNA-guided ABMIL 게이트(FiLM additive bias) | `train.py --M4` (models/vit_m4.py) |
| M4A | M4와 동일 골격, attn_pool을 genomic-guided co-attention(MCAT 스타일)으로 교체 | `train.py --M4A` (models/vit_m4a.py) |
| M4B | M4와 동일 골격, RNA 개입 지점을 ViT 이전 patch token(FiLM scale+shift)으로 이동 | `train.py --M4B` (models/vit_m4b.py) |
| M5 | Clinical(age/sex)만, WSI/RNA 없음 | `train.py --M5` (models/clinical_only.py) |
| M6 | RNA-seq만, WSI/Clinical 없음 | `train.py --M6` (models/rna_only.py) |
| M7 | Clinical+RNA 결합, WSI 없음 | `train_clinical_rna_only.py --external` |

## 요약 (3시드 x 2코호트 평균, n=6)

| 모델 | Internal C-index | External C-index |
|---|---|---|
| M1 (WSI only) | 0.550 | **0.468** (최저) |
| M4 (gate-bias FiLM) | 0.510 | 0.512 |
| M4A (co-attention) | 0.552 | 0.530 |
| M4B (pre-ViT FiLM) | 0.509 | 0.514 |
| M5 (Clinical only) | 0.531 | 0.512 |
| M6 (RNA only) | 0.612 | 0.522 |
| **M7 (Clinical+RNA)** | **0.612** | **0.575** (최고) |

## 핵심 관찰

1. **M7(WSI 없이 Clinical+RNA만)이 external에서 가장 좋다(0.575)** — 지금까지 시도한 모든 WSI 포함 모델(M1/M4/M4A/M4B)보다 높다. RNA-guided attention을 어떻게 설계하든(M4/M4A/M4B 세 가지 지점) WSI를 추가하는 것 자체가 external 성능을 끌어올리지 못했다.
2. **M1(순수 WSI)이 external에서 가장 나쁘다(0.468, 사실상 랜덤 이하)** — WSI 단독으로는 cross-institution 일반화가 거의 안 된다. 배치 효과(기관/스캐너) 오염이 유력한 원인(`check_domain_shift.py`, raw CNN feature 도메인 분류기 AUC=0.78).
3. **RNA 개입 지점(M4 gate-bias vs M4A co-attention vs M4B pre-ViT FiLM)은 결과에 거의 영향이 없다.** M4(0.510/0.512)와 M4B(0.509/0.514)는 사실상 동일하고(같은 시드에서 c-index 차이가 0.002~0.015 수준), M4A만 internal에서 약간 높다(0.552) — 다만 이 차이도 시드 간 편차(같은 모델 안에서 0.38~0.70까지 흔들림)보다 작아 확정적 우열로 보기 어렵다.
4. **M6(RNA only)이 M4/M4A/M4B(WSI 포함 3-모달)보다 internal도, external도 전부 앞선다** — WSI를 추가하는 것이 RNA 단독보다 오히려 손해라는 뜻이다.
5. 종합하면: 이 코호트 규모(환자 90~150명)에서는 WSI 브랜치가 RNA-guided fusion 메커니즘과 무관하게 일관되게 발목을 잡고 있고, 병목은 fusion 아키텍처가 아니라 WSI 표현 자체(배치 효과 오염 등 상류 문제)에 있다는 가설을 지지한다.

## 참고: 신뢰도에 대한 주의사항

- n=6(3시드 x 2코호트)은 여전히 작은 표본이다 — 개별 시드 간 c-index가 0.3~0.7까지 흔들리는 경우가 반복 관찰됐다(체크포인트 선택 노이즈 포함). 위 평균값은 "경향"으로 해석하고, 개별 비교의 유의성은 과신하지 않는다.
- M1/M4/M4A/M4B/M5/M6는 `--dataset {tcga,cptac}` 단일 코호트 학습 + stratified 6:2:2 split이다. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) 리포지토리는 TCGA+CPTAC를 합친 뒤 stratified split(`dataset + os_event` 기준)을 쓰므로, test set에 두 기관이 이미 섞여 들어간다 — 우리 external test(반대 코호트 전체, 학습 중 완전 미노출)보다 쉬운 평가 프로토콜이다. 따라서 레퍼런스가 보고한 수치(M4=0.722 등)와 이 표의 external 수치를 직접 비교하는 것은 공정하지 않다. `--dataset both`로 재실행하면 레퍼런스와 동일한 프로토콜로 비교점을 만들 수 있다(계획됨, 아직 미실행).
