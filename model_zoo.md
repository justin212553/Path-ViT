# PATH-ViT 모델 계보 정리 (M1 → PMA_EX)

2026-07-16 하루 동안 만든 모든 모델 변형을 시간 순으로 정리한다. 각 모델의 정확한 메커니즘,
왜 만들었는지, 결과가 어땠는지를 기록해 나중에 "이 모델이 정확히 뭘 했더라"를 다시 찾지 않아도
되게 한다. "왜"에 대한 더 자세한 논의는 `findings_backlog.md` 참조.

공통 골격: 패치 이미지 → CNN(frozen, ResNet50/Lunit SwAV 또는 UNI) → ViT self-attention
(Nystromformer 근사, 패치끼리 공간 문맥 교환) → **여기서부터 모델마다 갈라진다** → risk_head
(LayerNorm→Linear→scalar) → Cox partial likelihood loss로 학습.

---

## 세대 0 — 기준선 (WSI 단독)

### M1 (`models/vit_m1.py::ViT_M1`, `train.py` 기본값)
- ViT를 지난 패치 토큰들을 **ABMIL**(Ilse et al. 2018, gated attention pooling)로 단일 벡터(D,)로 압축.
  `gate = tanh(Wv·token) * sigmoid(Wu·token)` → softmax → 가중합.
- risk_head(D→1). 순수 병리 단독 모델, 모든 멀티모달 비교의 기준선.

### M1_AvgPool (`models/vit_m1_avgpool.py`, `--avgpool`)
- M1과 동일하되 ABMIL 대신 학습 파라미터 없는 단순 평균 풀링.
- **목적**: "학습되는 attention 가중치가 정말 도움이 되나?"를 테스트한 ablation.
- **결과**: ABMIL과 유의미한 차이 없음(초기 조사에서 결론, 재조사 안 함).

### LateFusionViT (`models/patch_vit_fusion.py`, `--fusion`)
- M1의 WSI 임베딩(Path A) + Cluster Histogram(Path B, K-means 군집 중심에 대한 타일 소속 히스토그램) 결합.
- RNA/Clinical과는 무관한 별도 축(형태학적 군집 기반 fusion) — 이후 RNA 중심 탐구로 초점이 옮겨가면서 더 발전시키지 않음.

### M2 (`models/vit_m2.py::ViT_M2`, `--M2`)
- M1의 WSI 임베딩 + `ClinicalEncoder`(age/sex MLP) → `[z_wsi, z_clinical]` (2D) concat → risk_head.
- Clinical 정보 추가가 WSI 단독보다 나은지 확인하는 첫 멀티모달 스텝.

---

## 세대 1 — M4: WSI+Clinical+RNA, RNA 결합 방식의 세 번의 진화

M4라는 이름 아래 실제로는 세 가지 다른 메커니즘을 거쳤다. **RNA를 어디서/어떻게 WSI 표현에
개입시키는가**가 이 프로젝트 탐구의 핵심 축이었기 때문.

### M4 v1 — Concat Late Fusion (최초 버전)
- M1의 WSI 임베딩 + ClinicalEncoder + RNAEncoder를 **그냥 이어붙이기만** 함: `[z_wsi, z_clinical, z_rna]` (3D) → risk_head.
- RNA가 WSI의 patch attention이나 pooling에 전혀 개입하지 않는, 가장 단순한 3-모달 결합.
- 노이즈 문제(시드마다 성능이 크게 흔들림)의 근본 원인을 찾던 초기 단계의 기준점.

### M4 v2 — Post-hoc Sigmoid 게이트 ("아핀변환 추가")
- WSI 임베딩은 M1과 동일하게(RNA 개입 없이) 먼저 완성.
- `gate = sigmoid(Linear(z_rna))`; `z_wsi_gated = z_wsi * gate`.
- `[z_wsi, z_wsi_gated, z_clinical, z_rna]` (4D) concat → risk_head.
- 레퍼런스 리포지토리(Leeyoungsup/pancreatic_cancer_pathology)의 초기 설계를 참고해 이식.
- **한계**: pooling이 이미 끝난 뒤에야 RNA가 개입해 "RNA subtype에 따라 어떤 패치를 볼지"는 학습 불가.

### M4 v3 — ABMIL 게이트 내부 FiLM bias (현재 커밋되지 않은 최신 버전)
- RNA가 ABMIL의 게이트 **pre-activation 자체**에 additive bias로 개입:
  `v = attn_v(token) + context_v(z_rna)`, `u = attn_u(token) + context_u(z_rna)`,
  `gate = tanh(v) * sigmoid(u)`.
- Patch attention score 자체가 RNA로 조건화됨 — v2보다 훨씬 이른 지점에서 개입.
- `combine_with_clinical_rna()`: `[z_wsi(이미 RNA-informed), z_clinical, z_rna]` (3D) → risk_head.
- **현재 `--M4`가 가리키는 버전.** `models/vit_m4.py`.

**세 버전 다 --dataset both 3시드 기준 c-index 0.51~0.55 범위로 사실상 동일** — RNA 개입 지점을
아무리 바꿔도 결과가 안 바뀐다는 게 이후 M4A/M4B/PM4/PMA 탐구의 출발점이 됐다(v1/v2는 재측정 안 함, v3부터 both 프로토콜로 정식 비교).

---

## 세대 2 — RNA 개입 지점 사다리 (M4A, M4B)

M4(v3)와 동일한 골격에서 RNA가 개입하는 **지점**만 바꾼 통제 비교.

### M4A (`models/vit_m4a.py::ViT_M4A`, `--M4A`)
- `CoAttentionPooling`: z_rna를 query, 패치 토큰 N개를 key/value로 하는 multi-head cross-attention.
- "이 RNA subtype과 가장 유사한 패치가 무엇인가"를 명시적으로 학습 — MCAT(Chen et al. 2021) 스타일.
- `[z_wsi, z_clinical, z_rna]` (3D) → risk_head.

### M4B (`models/vit_m4b.py::ViT_M4B`, `--M4B`)
- RNA가 **ViT 통과 전** 패치 토큰 자체에 FiLM(scale+shift)으로 개입.
- `attn_pool`은 일반 ABMIL로 복귀(RNA가 이미 토큰에 스며들어 이중 개입 방지).
- 지금까지 시도한 것 중 가장 이른 지점에서 개입하는 버전.
- **다성분 pooling과 궁합이 안 좋음**(원본/조절본 분리가 안 됨 — 별도 논의로 결론, 이후 세대에서 제외).

**both 프로토콜 3시드 결과(internal C-index)**: M4=0.539, M4A=0.549, M4B=0.536 — **개입 지점은 결과에 거의 영향 없음**. 레퍼런스 M4(0.722)와는 여전히 큰 격차. → "문제는 개입 지점이 아니라 WSI 표현 자체가 단일 벡터로 너무 압축돼 있다"는 결론으로 이어짐.

---

## 세대 2.5 — WSI-free 기준선/구색 모델

레퍼런스가 아니라 우리 자체 필요(과적합·노이즈 원인 분리, 논문용 완전성)로 만든 대조군.

| 모델 | 파일 | 구성 |
|---|---|---|
| M5 (ClinicalOnly) | `models/clinical_only.py` | age/sex만, WSI/RNA 없음 |
| M6 (RNAOnly) | `models/rna_only.py` | RNA-seq만, WSI/Clinical 없음 |
| M6X (RNAOnlyExtend) | `models/rna_only_extend.py` | M6와 동일 유전자 입력, 인코더만 레퍼런스 사양(64→256차원, dropout 0.25)으로 확장 |
| M7 (ClinicalRNAOnly) | `models/clinical_rna_only.py` | Clinical+RNA 결합, WSI 없음. 원래 `train_clinical_rna_only.py`(lr=1e-3 독립 스크립트)였다가 `train_light.py`로 통합 |

**핵심 발견**: M7(external C-index 0.575)이 이 시점까지 시도한 모든 WSI 포함 모델(M1/M4/M4A/M4B, external 0.47~0.53)보다 나았다 — "WSI를 추가하는 것 자체가 RNA+Clinical 단독보다 손해"라는, 이 프로젝트에서 가장 오래 유지된 결론.

---

## 세대 3 — 다성분(Multi-Component) Pooling: 돌파구

M1/M4/M4A/M4B가 `--dataset both`(레퍼런스와 동일한 combined+stratified 프로토콜)에서도 레퍼런스
M4(0.722)에 크게 못 미친 원인 분석 결과 — **M1은 레퍼런스와 거의 일치하는데 M4류만 못 미쳤다.**
즉 병목은 병리 인코더가 아니라 "WSI를 ABMIL로 단일 벡터에 압축한 뒤 그 위에 RNA를 개입시키는" fusion 설계.

### 공용 모듈: `models/multi_component_pooling.py::MultiComponentPooling`
- ABMIL처럼 N개 패치 토큰을 벡터 1개로 뭉개지 않고, **4개 관점을 병렬로 유지**:
  `mean`, `std`, `attention-weighted`(기존 ABMIL 재사용), `top-10%-mean`.
  → `components: (4, D)`.
- 레퍼런스의 Morphology Burden Pooling(mean/std/risk-weighted/top10%/top25%/risk 분포 통계 6그룹)을
  4개로 단순화해 이식한 버전 — "ABMIL이냐 CLAM이냐"가 핵심이 아니라 "단일 벡터냐 다성분이냐"가 핵심이라는 재진단에서 나옴.

### PM4 (`models/vit_pm4.py::ViT_PM4`, `--PM4`)
- `components (4,D)` → flatten → `H_i (4D,)`.
- RNA는 pooling 이후 **post-hoc sigmoid 게이트**: `gate=sigmoid(Linear(z_rna))`, `H_i_gated = H_i*gate`.
- `[H_i, H_i_gated, z_clinical, z_rna]` (10D) → risk_head. 레퍼런스 M3/M4 설계를 다성분 버전으로 그대로 이식.

### PMA (`models/vit_pma.py::ViT_PMA`, `--PMA`)
- `components (4,D)`는 flatten하지 않고 유지.
- `CoAttentionPooling`(M4A와 같은 모듈 재사용)으로 z_rna가 **4개 관점 중 어느 것이 중요한지** query로 선택/가중합 → `z_wsi (D,)`.
- `[z_wsi, z_clinical, z_rna]` (3D) → risk_head.

**both 프로토콜 3시드 결과(internal C-index)**: PM4=0.553, **PMA=0.583**(지금까지 최고) — 다성분 pooling 자체는 방향이 맞았지만 개선폭은 아직 작음.

---

## 세대 4 — 유전자 재선정 + PMA_EX: 첫 유의미한 도약

### 유전자 재선정 파이프라인 (`data/select_rnaseq_genes.py`)
- 기존 RNA 입력(339개, Bailey 2016+Moffitt 2015 PDAC **subtype 분류**용)을 레퍼런스 방식으로 교체.
- 문헌 큐레이션 PDAC 유전자(8개 카테고리, 163개, driver/DNA repair/subtype/EMT/stromal/immune/proliferation/hypoxia) +
  train split(both 기준) 내부에서 TCGA/CPTAC 각각 독립적인 **univariate Cox score test** +
  **Stouffer meta-analysis**(`meta_z = sum(z)/sqrt(2)`)로 전체 유전자 순위화.
- 문헌 유전자를 자체 Cox 순위로 먼저 배치하고, 남는 자리는 전체 유전자 Cox 순위로 채움 → 상위 1000/1500/2000개 저장.
- `data/dataset.py::literature_guided_gene_ids(top_n)`로 로드, `train.py --rna-genes literature_{1000,1500,2000}`
  (wandb에 `_EX` 접미사 자동 부착).
- **생존 예측에 직접 최적화된 기준**이라는 점이 기존 339개(분류 목적)와의 핵심 차이.

### PMA_EX = PMA + literature_1500
- 아키텍처는 PMA와 동일, RNA 입력만 339개(subtype) → 1500개(literature-guided) 유전자로 교체.

**both 프로토콜 3시드 결과(internal C-index)**: **0.656** (범위 0.604~0.733) — PMA(0.583) 대비 뚜렷한 개선,
seed126 하나는 0.733/HR 3.27/p=0.0003으로 레퍼런스(0.722/3.32/0.00064)와 거의 일치.
**구조(다성분 pooling+co-attention) + 유전자셋(Cox+Stouffer 재선정)을 함께 바꾸자 처음으로 레퍼런스에 근접** —
개별 요소만으로는(PM4/PMA 단독, M6X 단독) 전부 미미했던 것과 대조적.

---

## 전체 요약 표 (`--dataset both`, 3시드 평균, internal C-index)

| 세대 | 모델 | C-index | HR | log-rank p |
|---|---|---|---|---|
| 0 | M1 | 0.546 | 1.41 | 0.35 |
| 1 | M4 (v3, 현재) | 0.539 | 1.45 | 0.37 |
| 2 | M4A (co-attention, 패치) | 0.549 | 1.42 | 0.32 |
| 2 | M4B (pre-ViT FiLM) | 0.536 | 1.26 | 0.55 |
| 3 | PM4 (다성분+게이트) | 0.553 | 1.24 | 0.54 |
| 3 | PMA (다성분+co-attention) | 0.583 | 1.65 | 0.32 |
| 4 | **PMA_EX (PMA+literature_1500)** | **0.656** | **2.15** | **0.10** |
| — | 레퍼런스 M4 | 0.722 | 3.32 | 0.00064 |

(M1/M4/M4A/M4B/PM4/PMA/PMA_EX는 전부 `--dataset both`, M5/M6/M6X/M7은 `--dataset {tcga,cptac} --external`
프로토콜로 측정 — 두 프로토콜 난이도가 다르다는 점은 findings_backlog.md 0번 항목 참조.)

## 다음에 볼 것
- `findings_backlog.md` — 우선순위와 "왜"에 대한 상세 논의.
- `results_summary_M1-M7.md` — external 프로토콜 기준 M1~M7 상세 비교(이 파일 이전에 작성, both 프로토콜 도입 전).
