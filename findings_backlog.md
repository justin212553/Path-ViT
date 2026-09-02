# PATH-ViT 발견 사항 및 우선순위 백로그

## 🔴 최상위 발견(2026-09-02) — RNA gene selection에 실제 leakage 확인(internal +0.10 부풀림), clinical branch starvation은 leak과 무관, pooled c-index 계산방식 자체도 재검토 필요

**상태: 2seed×5fold 전체 검증으로 확인됨. 다음 결정 대기 중(어떤 gene panel/계산방식을 논문 기준으로 쓸지) — 사용자 판단 보류.**

발단은 HDP_Pretrain_Cluster 체크포인트 진단(`scripts/diagnose_hdp_checkpoint_weights.py`)에서
`clinical_linear`(margin+staging, 실제 신호가 있다고 이미 검증된 항)의 설명분산비율이 **0.00%**로
나온 것 — RNA branch(`risk_head`)가 99%+를 차지했다. 사용자 질문: "RNA가 너무 강한 신호여서
그런 걸까... 1500개 크로스 리스트를 만든 것이 어쩌면 data leakage가 있어서 당연히 그럴 수 있지만."

### 1) Gene selection leakage — 실재하며, 정량화됨

`literature_guided_gene_ids_intersection`(`data/dataset.py:202-215`, M4/M4A/PM4/PMA/M6/M6X/M7/HDP
전부가 공유하는 1500유전자 패널)은 TCGA-only/CPTAC-only Cox 순위를 각자 코호트 자기 라벨로
독립 계산해 교집합을 쓴다 — "반대 코호트 라벨은 안 본다"는 점만 leakage-free라고 문서화돼 있었다.
그런데 이 순위 계산에 쓰는 "train"이 paper-spec의 실제 5-fold CV가 아니라 **완전히 별개의 고정
단일 6:2:2 split**(`data/select_rnaseq_genes.py:150`, `cfg.data.seed` 기본값=42, fold 인자 자체가
없음)이었다. 실측 overlap(`data/dataset.py::_kfold_case_split`/`_stratified_case_split` 직접
호출로 계산):

| | overlap |
|---|---|
| TCGA internal, seed84 pooled 5fold test | 91/152 = **59.9%** |
| TCGA internal, seed126 pooled 5fold test | 91/152 = **59.9%** |
| TCGA internal, seed42(gene selection과 같은 시드로 k-fold 해도) | 91/152 = **59.9%** |
| CPTAC external(전체 144명) | 87/144 = **60.4%** |

즉 "held-out"이라 부르는 환자의 약 60%가 gene selection 시점에 이미 자기 생존 라벨을 한 번
썼다 — univariate selection bias의 전형적 형태. 세 시드(42/84/126) 전부 겹침 비율이 거의 동일해
"seed42가 WSI에 불리해서 제외했다"는 기존 결정과는 무관하다는 것도 확인.

### 2) 대안: `pathway8`(문헌 큐레이션, OS 라벨 미사용) — dataset 레벨엔 이미 있었는데 M5/M6/M6X/M7/HDP
### 계열(`train_light.py`)엔 배선이 안 돼 있었다

`data/dataset.py::pathway_category_gene_ids()`/`WSISurvivalDataset(rna_pathway_categories=...)`는
이미 `train.py`(M4 계열)에만 연결돼 있었다(`--rna-genes pathway8`, 8개 생물학적 카테고리 평균
z-score, Cox test 전혀 안 씀 → 구조적으로 leakage 불가능). `train_light.py`에 동일 옵션을 이식
(`--rna-genes pathway8`, 2026-09-02 커밋).

**주의**: 기존 M4계열(WSI 포함)에서의 pathway8은 이미 실패 기록이 있다(위 2026-08-31 항목 21-22행
참조, external C 0.49~0.52) — 이번 결과는 **M7(WSI 없음, Clinical+RNA만)**에서의 pathway8이라
직접 비교 대상이 아니다. WSI가 있는 구조에서 8차원으로 압축한 RNA가 co-attention query로 쓰였을 때
실패한 것과, WSI 없이 cox_add로 단순히 더해질 때의 결과는 별개다.

### 3) M7으로 leak(`literature_1500_intersection`) vs no-leak(`pathway8`) 전체 2seed×5fold 비교
(`--combine-mode cox_add --clinical-margin --clinical-staging`, `clinical-lr-mult` 기본값=1,
로컬에서 전부 실행 — M7은 WSI 없어 uni2native 불필요, 10 combo 전부 40초 내외/run)

| | Internal(pooled ensemble) | External(10-run ensemble) |
|---|---|---|
| **literature_1500(leak 있음)** | 0.6552 [0.582, 0.720] | 0.6221 [0.566, 0.678] |
| **pathway8(leak 없음)** | 0.5566 [0.479, 0.633] | 0.6039 [0.550, 0.657] |
| 차이(leak − no-leak) | **+0.0986** | **+0.0182** |

**leak이 전혀 닿지 않는 external에서는 두 버전 차이가 CI 안에 완전히 묻힌다(0.018).** internal에서
보이는 leak 버전의 "강함"(+0.099) 중 대부분이 실제 예측력이 아니라 gene selection이 자기가 나중에
평가할 환자의 라벨을 미리 본 데서 온 착시라는 뜻 — leak-free 버전의 진짜 external 실력(0.6039)은
leak 버전의 external(0.6221)과 사실상 같다.

### 4) Clinical branch starvation은 leak과 무관 — 별개의, 진짜인 문제

`scripts/diagnose_m7_branch_contrib.py`(HDP 체크포인트 진단의 WSI-불필요 경량판)로 leak/no-leak
양쪽에서 `clinical_linear` 설명분산비율을 봤더니 **둘 다 0.00~0.01%**로 동일했다 — clinical이
죽는 건 gene panel의 leak 때문이 아니라, RNA branch(자체 nonlinear encoder 보유)가 zero-init
linear뿐인 clinical과 gradient 경쟁에서 원래 유리한 구조적 문제(`train.py`의 다른 모델들에서 이미
`--clinical-lr-mult`로 대응했던 것과 같은 종류)다. `train_light.py`엔 이 옵션이 없었어서
`--clinical-lr-mult`/`--rna-lr-mult`(`_branch_param_groups`)를 이식(2026-09-02 커밋).

**mult sweep(seed84/fold0, pathway8 기준) — clinical을 강제로 살리는 건 성공하지만 성능은 안 좋아진다**:

| mult | clinical 설명분산 | internal test(fold0) | external |
|---|---|---|---|
| 1(기본) | 0.01% | 0.6097 | 0.6450 |
| 2 | 0.04% | 0.6059 | 0.6458 |
| 5 | 0.25% | 0.5948 | 0.6474 |
| 10 | 0.97% | 0.5948 | 0.6492 |
| 20 | 3.56% | 0.5836 | 0.6531 |
| 50 | 16.37% | 0.5874 | 0.6557 |
| 100 | 37.82% | 0.5725 | 0.6521 |

fold0 하나만 보면 "mult 올릴수록 external이 단조증가"하는 것처럼 보였다 — **그런데 mult=50을
2seed×5fold 전체로 검증하니 이 추세가 사라졌다**: internal pooled=**0.5544**(더 나빠짐),
external 10-run ensemble=**0.6362**(fold0의 0.6557보다 낮고, mult=1 fold0의 0.6450 대비도 개선
없음, 10개 run 자체가 0.584~0.656로 크게 흩어짐 std=0.027). **결론: branch-competition은 진짜
문제이고 고칠 수도 있지만, 고쳐도 이 코호트 규모에서 순증분은 안 보인다** — clinical-lr-mult는
기본값(1)이 가장 안전.

### 5) Pooled c-index 계산방식 자체에 대한 의문 — 아직 결론 안 남

pathway8 internal fold별 c-index(seed×fold 10개): 0.4660~0.6834로 폭이 0.22 — fold당
N=29~31/event=16~17뿐이라 통계적 검정력 한계로 자연스러운 변동. **그런데 fold별 단순 평균
(0.580)이 pool_multiseed_kfold_preds.py가 계산하는 "pooled"(out-of-fold 예측을 전부 이어붙여
한 번에 concordance 계산, 0.5566)보다 뚜렷이 높다.** 같은 효과가 leak 버전에도 있다
(fold평균 0.6709 vs pooled 0.6446~0.6552) — **pooling 자체가 fold마다 독립적으로 초기화·
학습된 모델들의 risk score를 서로 다른 잣대인 채로 이어붙여 비교하는 구조**라, cross-fold
환자쌍 비교가 신호를 갉아먹는 것으로 보인다(두 버전에 공통, leak과 무관한 별도 아티팩트).

fold평균 기준으로 다시 보면: pathway8 internal(0.580) vs external(0.604) — 차이 0.024(노이즈
안). literature_1500 internal(0.671) vs external(0.622) — 차이 +0.049(반대 방향, leak 지문
그대로 유지). **즉 "pooled 숫자가 낮아 보인다"는 인상의 상당 부분은 계산 방식 아티팩트이지만,
leak 버전만 internal>external로 뒤집힌다는 핵심 신호는 이 아티팩트를 걷어내고도 남는다.**

**미해결(사용자 보류, 2026-09-02)**: `pool_multiseed_kfold_preds.py`의 pooled/ensemble
c-index(이 프로젝트 전체의 표준 "internal" 보고 지표)가 "이 모델의 진짜 성능"을 나타내는
지표로 타당한지 자체가 불확실해졌다 — fold 간 모델 보정(calibration) 불일치를 명시적으로
다루지 않는 한, 특히 신호가 약한(pathway8류) 모델에서 실제보다 비관적인 숫자를 낼 수 있다.
대안(예: fold별 c-index 평균+표준오차로 보고, 또는 fold 간 risk score를 재보정한 뒤 pooling)을
검토할지는 다음 세션 결정 사항.

### 다음 결정 필요 사항
- 논문/이후 실험에서 gene panel을 `pathway8`(leak 없음, RNA 8차원)로 완전히 바꿀지, fold-aware
  nested Cox selection(폴드마다 독립적으로 1500개 재계산, 작업량 큼)을 새로 만들지.
- pooled c-index 계산방식을 그대로 쓸지, fold평균 기반 등 대안으로 바꿀지.
- 이미 보고된 이 프로젝트의 모든 internal 헤드라인 숫자(M4/M4A/PM4/PMA/M6/M6X/M7, HDP 계열 전부
  `literature_1500_intersection` 사용)가 이번 발견의 영향권 안에 있다는 점 — 논문에 실린 숫자
  재검토 필요.

관련 코드: `scripts/diagnose_hdp_checkpoint_weights.py`, `scripts/diagnose_m7_branch_contrib.py`,
`scripts/diagnose_hdp_feature_signal.py`(선행 진단, raw feature 자체의 생존 상관 없음 확인),
`train_light.py --rna-genes pathway8`/`--clinical-lr-mult`/`--rna-lr-mult`(전부 2026-09-02 신규).

### 후속(2026-09-02 밤) — "TCGA 전체 train + 고정epoch + external만" 프로토콜로 재검증

Pooled/ensembled internal k-fold c-index의 신뢰성 자체가 의심스러워진 것(위 5번 항목)에 대한
사용자 대안 제안: "차라리 TCGA를 통째로 train set으로 쓴 다음 눈 가리고 30에폭 쓰고 external만
보고하는 게 검정력이 높지 않을까" — `train_light.py --full-train`(k-fold/6:2:2 split 없이 코호트
전체를 train, 고정 epoch, val 없어 early-stop 불가)로 구현, 시드 5개(42/84/126/168/210) 반복해
external 평균±bootstrap CI로 비교. 버그 2개 수정(`scripts/run_fulltrain_leak_sweep_local.sh`
커밋 참조 — `--full-train`이 `test_metrics` 미정의로 Slack 알림 단계에서 매번 크래시하던 문제,
checkpoint 자체가 저장 안 돼 `--eval-external-ckpt` 후속 조회가 불가능하던 문제).

**결과(M7, `--epochs 30`, external=CPTAC 전체 144명, 5시드 앙상블)**:

| 설정 | 시드 평균±std | 앙상블 c_index | bootstrap 95% CI |
|---|---|---|---|
| literature_1500(leak) | 0.6142±0.0066 | **0.6154** | [0.555, 0.671] |
| pathway8(no-leak), mult=1 | 0.5737±0.0075 | 0.5729 | [0.513, 0.634] |
| pathway8 + mult=5 | 0.5863±0.0069 | 0.5867 | [0.528, 0.643] |
| pathway8 + mult=10 | 0.5864±0.0075 | **0.5912** | [0.533, 0.647] |
| pathway8 + mult=20 | 0.5839±0.0082 | 0.5882 | [0.530, 0.644] |
| pathway8 + mult=50 | 0.5789±0.0106 | 0.5825 | [0.525, 0.639] |

**leak 효과가 이 프로토콜에서는 external에도 더 뚜렷이 남는다** — leak/no-leak external 격차가
+0.0425(0.6154−0.5729)로, 어제 k-fold 기반 비교(2seed×5fold pooled, +0.018)의 2배 이상. CI는
여전히 크게 겹치지만(점추정치 격차는 뚜렷) 어제 "leak은 internal에만 있고 external엔 거의
안 닿는다"는 결론을 다소 후퇴시킨다. **가능한 원인(미분리)**: full-train은 fold당(~90~122명)
보다 훨씬 큰 train set(152명 전체)을 쓰므로, RNA encoder(1500차원 입력, 파라미터 많음)가
데이터量 자체의 이득을 leak과 무관하게 더 크게 볼 수 있다 — "leak이 진짜 신호를 일부 담고
있어서"인지 "더 큰 encoder가 더 큰 train set에서 원래 유리해서"인지 이 실험만으론 분리 안 됨
(encoder 크기를 맞춘 대조군이 있어야 분리 가능, 다음 세션 후보 작업).

**clinical-lr-mult는 이 프로토콜에서 훨씬 깨끗한 신호를 보인다** — mult=1(0.5729)→5(0.5867)→
**10(0.5912, 최고점)**→20(0.5882)→50(0.5825)로 매끈한 산 모양(어제 k-fold 기반의 "fold0에서만
보이던 가짜 추세"와 달리 시드별 std 0.007~0.011로 작고 방향이 일관됨). mult=10 근방이
branch-competition 해소에 실제로 도움이 된다는 쪽에 무게가 실리나, mult=1의 bootstrap CI 안에
mult=10의 점추정치가 들어가 통계적으로 확정은 아님.

**다음 결정 필요(추가)**: leak vs "단순 train set 크기 효과" 분리 실험(encoder 폭을 pathway8/
literature_1500 양쪽에 맞춘 대조군), clinical-lr-mult=10을 표준값으로 채택할지, 이 full-train
프로토콜을 앞으로의 모든 아키텍처 비교의 표준으로 삼을지.

관련 코드(신규): `scripts/pool_fulltrain_external_preds.py`, `scripts/run_fulltrain_leak_sweep_local.sh`.

### 후속2(2026-09-02) — PMA(WSI+RNA+Clinical)로 확장, paired bootstrap으로 WSI 추가 유의성 정식 검정

M7(RNA+Clinical만) 대비 "WSI를 추가하면 external에서 통계적으로 유의한 순증분이 있는가"를
같은 full-train+external+5시드 프로토콜로 PMA(M4, `train.py --PMA --backbone uni2`)까지 확장.
`train.py --full-train`도 동일한 문제(checkpoint 미저장, external CSV 미저장)가 있어 같은 방식
으로 수정. `scripts/paired_bootstrap_delta_fulltrain.py`(fold 없는 버전, 기존
`paired_bootstrap_delta.py`의 핵심 함수 재사용)로 M7 vs PMA paired bootstrap 수행.

| 설정 | 시드 평균±std | 앙상블 | bootstrap CI |
|---|---|---|---|
| PMA lit1500 baseline | 0.6148±0.0203 | 0.6283 | [0.572, 0.684] |
| PMA lit1500+CLR10 | 0.5288±0.0522 | 0.5730 | [0.515, 0.627] |
| PMA pathway8 baseline | 0.5545±0.0499 | 0.5939 | [0.540, 0.649] |
| PMA pathway8+CLR10 | 0.5015±0.0693 | 0.4924 | [0.432, 0.552] |

**Paired bootstrap(M7=A, PMA=B, delta=B−A)**:

| 비교 | C(A) | C(B) | delta | 95% CI | p | 결과 |
|---|---|---|---|---|---|---|
| lit1500, mult=1 | 0.6154 | 0.6283 | +0.0129 | [-0.019, +0.043] | 0.419 | 비유의 |
| pathway8, mult=1 | 0.5729 | 0.5939 | +0.0209 | [-0.044, +0.084] | 0.521 | 비유의 |
| pathway8, mult=10 | 0.5912 | 0.4924 | **-0.0988** | [-0.188, -0.008] | 0.031 | **유의(음수)** |

**정상 조건(mult=1) 둘 다 WSI 추가가 점추정치로는 +0.01~0.02지만 유의하지 않다** — 이 프로젝트
전체의 반복된 "이 코호트 규모에서 WSI가 순증분을 못 준다"는 결론이 paired bootstrap 정식 검정
으로 재확인됨. **mult=10 조합의 유의한 음수는 WSI의 해로움이 아니라 별개 문제** —
clinical-lr-mult=10이 M7에서는 도움됐지만(어제 결과) PMA(WSI branch까지 3파전 경쟁)에서는
학습 자체를 불안정하게 만든다(std 0.05~0.07, 일부 시드 chance 이하 0.40~0.46) — WSI가 없는
M7은 이 불안정성에 안 걸려 멀쩡했을 뿐. **결론: clinical-lr-mult는 branch 개수가 많은 모델
(WSI 포함)에는 그대로 이식하면 위험하다 — 안전하게 쓰려면 더 작은 값이나 WSI-aware 튜닝 필요.**

HDP_Pretrain_Cluster(pathway8, WSI+cluster)는 uni2native가 로컬에 없어(`uni2`와
`uni2native`는 서로 다른 파이프라인 산출물 — 전자만 로컬에 있음, 사용자 확인 요청으로 재확인)
HPC 전용(`sbatch/hdp_pretrain_cluster_fulltrain_pw8_sweep_hpc.sh`), 아직 미완료.

관련 코드(신규): `scripts/paired_bootstrap_delta_fulltrain.py`, `scripts/run_pma_leak_sweep_local.sh`,
`train.py --full-train` external CSV 저장 버그 수정, `train_hdp_pretrain_cluster.py --rna-genes
pathway8`/`--full-train`/`--clinical-lr-mult`/`--rna-lr-mult`.

### 후속3(2026-09-02) — HDP_Pretrain_Cluster(HPC) 완료, 이 세션 WSI 시도 중 가장 유의에 근접

`sbatch/hdp_pretrain_cluster_fulltrain_pw8_sweep_hpc.sh`(pathway8 baseline + clinical-lr-mult=10,
5시드) HPC 실행 완료(1개 시드는 free-gpu partition preemption으로 한 번 끊겼다가 재제출로 완료 —
로그는 preemption 시점에서 멈춰 있지만 CSV는 144명 전부 정상 생성돼 있음을 직접 검증). M7/PMA는
로컬, HDP는 HPC라 예측 CSV가 서로 다른 머신에 있었던 것도 확인 — `paper/hdp/`로 다운로드한 CSV를
`.logs/external_preds/`에 합쳐서 로컬에서 paired bootstrap 수행.

| | 시드 평균±std | 앙상블 | bootstrap CI |
|---|---|---|---|
| M7 baseline | 0.5737±0.0075 | 0.5729 | [0.513, 0.634] |
| **HDP_Pretrain_Cluster baseline** | **0.6000±0.0078** | **0.6064** | [0.553, 0.661] |
| M7 +CLR10 | 0.5864±0.0075 | 0.5912 | [0.533, 0.647] |
| **HDP_Pretrain_Cluster +CLR10** | **0.6093±0.0059** | **0.6140** | [0.561, 0.667] |

**Paired bootstrap(M7=A, HDP=B, delta=B−A)**: baseline delta=+0.0335, CI=[-0.011,+0.077], **p=0.132**
(비유의) / CLR10 delta=+0.0227, CI=[-0.028,+0.068], p=0.362(비유의).

**여전히 유의하지 않지만, 이 세션에서 시도한 모든 WSI 통합 방식(MCAT/PORPOISE/M4A/PMA 전부
p>0.4) 중 p값이 가장 낮다(0.132).** 시드 간 표준편차도 0.006~0.008로 PMA(0.05~0.07)보다 한
자릿수 더 안정적 — attention을 아예 안 쓰고 mean-pooling+cox_add 선형항만 쓴 설계 선택
(`models/hdp_cluster.py`, "이 세션 내내 attention/MIL이 반복 실패한 패턴을 다시 밟지 않기
위한 의도적 선택")이 실제로 학습 안정성 면에서 확실히 우위였다는 근거. clinical-lr-mult=10은
점추정치는 살짝 올리지만(0.606→0.614) paired bootstrap p값은 오히려 baseline보다 나쁨(0.132→
0.362) — 여기서는 득도 실도 없는 중립적 개입으로 보임.

**PanNuke pretrain head 해상도 보정 발견(같은 날, 별도 시도)**: `scripts/train_hdp_pretrain_head.py`가
PanNuke(40x/0.25um) 이미지를 그대로 UNI2-h에 먹였던 게 uni2native(20x/0.5um) 대비 배율
불일치였다는 기존 한계를 실제로 고쳐봄 — 학습 이미지를 절반 크기로 다운샘플했다 다시 업샘플해
겉보기 배율만 uni2native와 맞춘 뒤(`_simulate_uni2native_resolution`) 재학습한 결과
**val_corr 0.60 → 0.78로 개선**(로컬, `--legacy-resolution` 플래그로 기존 동작도 재현 가능,
결과는 별도 경로 `data/hdp_pretrain_tumor_content_head_resmatch.pt`에 저장해 기존 head와
분리). 아직 이 새 head를 코호트 전체에 적용(`scripts/apply_hdp_pretrain_head.py`)해 실제
HDP_Pretrain_Cluster 성능이 오르는지까지는 미검증 — 다음 세션 후보 작업.

관련 코드(신규): `scripts/train_hdp_pretrain_head.py::_simulate_uni2native_resolution`
(`--legacy-resolution` 플래그로 기존 동작 보존).

---

## 🔴 최상위 발견(2026-08-31) — MCAT 스타일 multi-pathway co-attention(Phase 1)도 실패, gradient는 정상 도달 확인 → "query 개수 부족"이 원인이 아니었다

**상태: 확인됨(seed84 단일 파일럿 + 체크포인트 진단 + 30epoch gradient/loss 추적, 3가지 독립 증거) — PORPOISE(Phase 3)로 진행 결정.**

BRCA 스케일 검증(위 2026-07-22 항목)이 "표본만 늘리면 WSI가 기여한다"는 반례를 보여줬지만, PAAD
표본을 늘릴 방법이 없는 상황에서 "외부 SOTA 문헌은 이 문제를 어떻게 푸는가"를 조사(`WebSearch`,
2026-08-31)한 결과, 이 프로젝트가 겪는 현상이 문헌에 이미 "modality imbalance / inter-modality
capability gap"으로 이름 붙은 알려진 문제라는 걸 확인했다. MCAT(Chen et al. 2021)/SurvPath(Jaume
et al.)는 RNA를 6~50개의 "pathway 토큰"으로 나눠 **여러 개의 query**가 병렬로 patch 전체에
co-attention한다 — 이 프로젝트의 기존 `--M4A`(`models/vit_m4a.py::CoAttentionPooling`)는 RNA
전체를 **단일 벡터 1개**로 압축해 query 1개로만 co-attention했다(스스로 "pathway별로 안 쪼갠
단순화 버전"이라고 문서화돼 있었음). "query 1개 vs key 4개(PMA 기준)/patch 전체(M4A 기준)라는
저용량 구조 자체가 attention entropy 붕괴(기존 2026-07-21 2차 발견 (B))의 원인 아니냐"는 가설로,
사용자가 세운 3단계 계획의 **Phase 1**로 이 구조를 직접 이식해 검증했다.

- **구현**(`models/gene_group_encoder.py`, `models/vit_mcat.py`, `train.py --MCAT`): RNA를
  PDAC 기능별 8개 유전자 카테고리(`data/select_rnaseq_genes.py::PDAC_LITERATURE_GENE_SETS`, 163개
  유전자)로 나눠, 카테고리마다 **학습되는 선형결합**(LayerNorm+Linear, hidden layer 없음)으로
  pathway 토큰 1개씩(총 8개)을 만든다. 예전 `--rna-genes pathway8`(카테고리를 unsigned mean으로
  뭉갬, external C 0.49~0.52로 실패, 아래 "완료" 섹션 10번) 실패 원인을 구조적으로 피하도록,
  유전자별 z-score를 뭉개지 않고 그대로 넣어 부호/크기가 다른 학습 가능한 가중치로 합친다. 8개
  토큰이 `MultiQueryCoAttentionPooling`(M4A의 `CoAttentionPooling`을 다중 query로 확장)을 통해
  동시에 patch 전체에 cross-attention한다.
- **파일럿 결과(seed84, fold0, uni2, cox_add+staging+margin+attn-dispersion+rna-aux, M4A/PMA와
  동일 최종 레시피)**: internal C=**0.4610**(chance 이하), HR=**0.712**(<1, 방향 반전),
  log-rank p=0.485(무의미). 같은 조건 기존 파일럿 대비 뚜렷이 낮음:

  | 모델 | internal C(seed84, fold0) |
  |---|---|
  | PMA(baseline) | 0.68 |
  | PM4 | 0.66 |
  | M4A(query 1개) | 0.61 |
  | **MCAT(query 8개)** | **0.46** |

- **진단 1 — 체크포인트(epoch=17, 30-epoch 학습 완료본) 직접 조사**(`train.py
  --eval-internal-ckpt`에 추가한 MCAT 전용 진단 블록, `hasattr(model, "gene_group_encoder")`로
  게이팅): pathway 토큰은 환자간 cosine similarity **0.0039**(0에 가까움, collapse 아님), per-dim
  std **0.5718**(정상) — GeneGroupEncoder는 의도대로 환자별로 다른 8개 토큰을 만들고 있다. 그런데도
  co-attention entropy는 **0.9998**(0~1 정규화, 1이면 완전 uniform) — 2026-07-21 2차 발견 (B)의
  "4-view co-attention이 환자 무관하게 uniform으로 수렴" 패턴이 query를 1개→8개로 늘려도 **똑같이
  재발**했다. **"query가 부족해서 붕괴한다"는 Phase 1의 핵심 가설이 직접 반박됐다.**
- **진단 2 — 학습 도중 gradient/loss 추적**(`scripts/diagnose_mcat_gradients.py`,
  `scripts/diagnose_wsi_gradients.py`를 MCAT용으로 재구성, seed84/uni2/30epoch, 로컬 재현): Cox
  loss는 2.12(epoch1)→노이즈 속에서 완만히 1.9~2.0대로 하락(최저 1.87 @epoch26, 깔끔한 수렴
  아니라 거의 평평 + 노이즈). `attn_pool`(cnn/vit와 분리 측정) gradient norm은 **30 epoch 내내
  0.79~0.86 사이로 거의 일정**(risk_head 대비 ~0.27배) — 시간이 갈수록 죽어가는 패턴이 전혀 없고,
  gene_group_encoder(0.18배)·clinical_encoder(0.36배)와 비슷한 자릿수라 극단적으로 굶주려 있지도
  않다. 2026-07-21 2차 발견 (C)에서 PMA의 WSI 브랜치가 RNA 대비 gradient가 **100~250배** 작았던
  것과는 확연히 다른 패턴 — **gradient는 attn_pool까지 정상적으로, 꾸준히 도달한다.**
- **종합 해석**: 진단 1(query 8개로 늘려도 attention entropy 붕괴 재발)과 진단 2(gradient는
  정상 도달)를 합치면, "query 개수가 부족해서/gradient가 안 와서 attention이 못 배운다"는 두
  가설이 모두 기각된다. gradient가 attn_pool에 꾸준히 오는데도 그 방향으로 움직여도 Cox loss가
  거의 안 줄어드니(→진단 2의 loss가 노이즈 속에서만 완만히 변함) attention이 뾰족해질 유인 자체가
  없다는 뜻 — **"이 WSI feature 공간 자체에 patch 단위로 구별해 Cox loss를 낮출 신호가 거의
  없다"는 2026-07-21 2차 발견의 결론을, 아키텍처를 바꿔가며(4-view co-attention → 8-pathway
  multi-query co-attention) 또 한 번 독립적으로 재확인한 것에 가깝다.** fusion 구조(query 개수,
  pooling 방식)를 바꾸는 접근으로는 이 병목을 못 뚫는다는 근거가 하나 더 쌓였다.
- **다음 방향(사용자 결정)**: attention이 "중요한 patch를 찾아내는" 데 의존하는 구조(M4A/PMA/MCAT
  전부 이 계열)는 전부 같은 벽에 부딪히므로, attention에 의존하지 않는 PORPOISE(Chen et al.
  2022, bilinear/Kronecker product pooling으로 WSI×RNA pairwise 상호작용을 직접 포착)로 전환.
  Self-Enhancement Learning(원래 Phase 2, "약한 브랜치에 보조 loss로 gradient를 더 밀어주는"
  방식)은 진단 2가 "gradient 크기 자체는 이미 정상"임을 보여줘 전제가 약해져 보류.
- **관련 파일**: `models/gene_group_encoder.py`, `models/vit_mcat.py`, `train.py`(`--MCAT`,
  `--eval-internal-ckpt` MCAT 진단 블록), `scripts/diagnose_mcat_gradients.py`,
  `sbatch/mcat_uni2_coxadd_stg_r_pilot_seed84_fold0_hpc.sh`,
  `sbatch/mcat_diagnose_pilot_seed84_fold0_hpc.sh`.

---

## 🔴 최상위 발견(2026-07-22) — TCGA-BRCA로 스케일 검증: 표본이 커지면 WSI가 순증분 기여를 한다

**상태: 확인됨(seed 42 단일 결과) — 다른 시드/더 긴 학습으로 재현성 확인 필요.**

지금까지(1차/2차 최상위 발견 포함) TCGA-PAAD(152 case)에서 WSI 브랜치가 ablation-neutral,
attention uniform 붕괴, RNA 대비 gradient 1/4 등 온갖 진단으로도 "WSI가 기여를 못 한다"는 결론이
계속 나왔다 — architecture를 바꿔도(Nystrom↔full-attention, spatial embed 유무, backbone
ResNet50↔UNI 100배 확장) 전부 null. **표본 자체가 152명으로 너무 작아서 WSI 브랜치(수십만 파라미터)를
학습시킬 신호가 부족했던 것 아니냐는 가설을 표본을 7배(TCGA-BRCA, 1058 case) 늘려 직접 검증했다.**

- **데이터**: TCGA-BRCA, WSI(HuggingFace `Dearcat/CPathPatchFeature`의 UNI backbone 사전추출
  feature) + RNA-seq(GDC STAR-Counts, PAAD와 동일하게 log2(FPKM-UQ+1) 후 코호트 내부 z-score,
  `scripts/extract_brca_rna.py`) + clinical(age, GDC API 직접 조회, `scripts/extract_brca_labels.py`)
  세 데이터 모두 존재하는 case 1058명(`scripts/brca_common.py`). RNA 유전자셋은 PDAC 전용
  literature_1500(Bailey/Moffitt subtype 큐레이션)이 유방암엔 안 맞아 대신 **train split 내부
  고분산 상위 1500개**로 대체(`scripts/select_brca_rna_genes.py`) — val/test 라벨은 유전자 선정에
  전혀 안 씀(literature_1500과 동일 원칙).
- **모델**: PMA_EX_SS_AUX(ViT_PMA, 지금까지 가장 나은 M4 레시피)를 아키텍처/하이퍼파라미터 그대로
  재현 — Nystromformer 포함(embed_dim=64, num_heads=2, num_landmarks=128), patch_keep_frac=0.8,
  rna_aux_weight=1.0, lr=1e-5/wd=1e-1/epochs=30. `train.py` 실제 코드(`_patient_risk`/
  `train_one_epoch`/`evaluate`/`_build_scheduler`)와 `models/vit_pma.py::ViT_PMA`를 그대로 import해
  재사용(`scripts/train_brca_m4.py`) — 로직 드리프트 방지. 대조군 M7(`models/clinical_rna_only.py`
  ::ClinicalRNAOnly, WSI 없음)도 `train_light.py` 실제 코드를 그대로 import(`scripts/
  train_brca_m7.py`)해 **같은 case/split/유전자셋**으로 비교.
- **결과(BRCA internal, seed=42, 동일 환경)**:

  | 모델 | test_c_index | test_HR | logrank_p | best val epoch |
  |---|---|---|---|---|
  | M7(Clinical+RNA, WSI 없음) | 0.6620 | 1.503 | 0.2751 | 4 (epoch 24 early stop, patience=20) |
  | **M4(ViT_PMA, WSI+RNA+Clinical)** | **0.7155** | 1.885 | 0.0845 | 30 (끝까지 상승, plateau 안 됨) |

  **M4가 M7보다 +0.0535(test c-index) 높다.** PAAD에서는 한 번도 못 봤던 방향의 결과 — 같은
  아키텍처(PMA_EX_SS_AUX)를 표본만 7배 늘려 재현했더니 WSI가 확실한 순증분 기여를 한다. M7은
  lr=1e-3 특유의 빠른 과적합으로 epoch 4에서 정점(train_c_index 0.97+로 치솟음)을 찍고 조기
  종료된 반면, M4는 val_c_index가 30 epoch 끝까지 단조 상승 중이었고 아직 plateau에 도달하지
  않았다 — "덜 학습된" 상태에서도 이겼다는 뜻이라, epoch을 늘리면 격차가 더 벌어질 가능성이 있다.
- **해석**: 지금까지의 "WSI ablation-neutral" 계열 negative result들은 아키텍처 결함이 아니라
  **TCGA-PAAD 152명이라는 표본 규모 자체가 병목이었다는 가설을 강하게 지지한다.** UNI backbone
  100배 확장(파라미터 용량)으로도 못 풀렸던 문제가 표본 7배 확장(학습 신호량)으로는 풀렸다는 점이
  대조적 — "이미 종료된 경로" 절의 "backbone 품질/용량보다 표본 규모·censored 라벨의 낮은 신호
  대 잡음비 자체가 병목"이라는 기존 결론과 정확히 부합한다.
- **주의(재현성 확인 전)**: 아직 **seed 42 단일 결과**다. 다음 확인 필요: (1) 다른 시드(84/126)로
  같은 방향이 재현되는지, (2) M4 val이 plateau 안 된 채 30 epoch에서 끊겼으므로 epoch을 늘렸을 때
  더 좋아지는지(혹은 오히려 과적합으로 꺾이는지). (3) `--patch-keep-frac 0.8`이 매 학습 step마다
  `torch.randperm`으로 패치를 서브샘플하는데, 이게 Nystromformer의 순서-의존적 landmark 그룹핑과
  결합해 공간 정보를 원치 않게 뒤섞었을 가능성 — 이 조합이 이번 긍정적 결과에 실제로 기여했는지,
  아니면 오히려 그것과 무관하게(혹은 그것을 극복하고) 나온 결과인지는 별도 ablation으로 검증
  필요(진행 예정, PAAD에서 이미 개별적으로는 `--shuffle-patches` 단독(null)·`--full-attention`
  단독(3시드 재현 안 됨, 노이즈) 둘 다 null이었지만 **"patch_keep_frac<1.0으로 인한 매 epoch
  암묵적 재셔플 + Nystrom"의 조합 자체**는 별도로 검증된 적이 없다).
- **후속 검증(2026-07-22, 같은 날) — `patch_keep_frac<1.0`의 암묵적 셔플 우려는 기각**: 위 "주의"
  항목 (3)을 직접 검증 — 같은 seed=42로 `--patch-keep-frac 1.0`(매 step 셔플 없음, 순수 baseline)
  버전을 추가로 돌려 기존 0.8(암묵적 셔플) 버전과 비교했다.

  | patch_keep_frac | test_c_index | best_val_c_index | best epoch |
  |---|---|---|---|
  | 0.8(기존 레시피, 암묵적 셔플) | 0.7155 | 0.6683 | 30(끝까지 상승, plateau 안 됨) |
  | 1.0(셔플 없음) | 0.7175 | 0.6794 | 22(여기서 진짜로 꺾임, plateau 확인됨) |

  차이는 +0.002로 노이즈 수준 — PAAD에서 나온 `--shuffle-patches` 단독 null, `--full-attention`
  단독 null(3시드 재현 안 됨)과 같은 결론이다. **`patch_keep_frac<1.0`의 암묵적 셔플이 Nystrom
  landmark 그룹핑을 망가뜨려 이번 BRCA 긍정 결과를 훼손했을 것이라는 우려는 기각된다** — 위 M4
  결과는 이 조합의 부작용이 아니라 실재하는 신호다. 부수 관찰: 1.0(셔플 없음)은 val이 epoch 22에서
  명확히 꺾여 진짜 최적점을 보여주는 반면, 0.8(패치 드롭아웃)은 30 epoch 끝까지 계속 상승만 했다 —
  패치 드롭아웃이 의도대로 정규화 역할을 해 수렴을 늦추고 있다는 뜻으로 보인다. 성능 차이가
  미미하므로 기존 레시피(patch_keep_frac=0.8)를 그대로 유지한다.
- **후속 검증(2026-07-22, 같은 날) — `--no-spatial-embed`를 BRCA(진짜 WSI 신호가 있는 환경)에서
  재검증, PAAD의 null이 재확인됨**: PAAD에서 `--no-spatial-embed`가 null이었던 건 WSI 브랜치
  전체가 무의미했던 환경이라 "공간 임베딩만 특별히 무의미하다"는 증거로 보기 약했다(뭘 빼도 다
  null이었을 수 있음) — BRCA는 WSI가 실제로 M7 대비 +0.0535 기여하는 것을 확인한 환경이라, 여기서
  `--no-spatial-embed`가 null인지가 훨씬 신뢰도 높은 테스트다. `scripts/train_brca_m4.py`에
  `--no-spatial-embed`(`cfg.model.use_spatial_embed=False`) 플래그를 추가해 같은 seed=42,
  patch_keep_frac=0.8(표준 레시피)로 재검증:

  | | test_c_index | best_val_c_index | best epoch |
  |---|---|---|---|
  | 공간 임베딩 있음(기존) | 0.7155 | 0.6683 | 30 |
  | 공간 임베딩 없음 | 0.7093 | 0.6762 | 27 |

  test는 -0.006, val은 +0.008로 방향이 엇갈리는 노이즈 수준 차이 — **WSI가 진짜 신호를 내는
  환경에서도 좌표 기반 SpatialPositionEmbedding은 여전히 기여가 없다.** PAAD 때의 null이 "표본
  부족으로 뭘 빼도 null"이 아니라 진짜 null이었다는 게 더 신뢰도 높게 확인된 셈 — MultiComponentPooling
  (mean/std/attention/top-k 4관점) + co-attention 조합이 이미 필요한 정보를 통계적 관점만으로
  충분히 뽑아내고 있고, 명시적 좌표 정보는 부가가치가 없다는 쪽으로 결론이 굳어진다.
- **후속 시도(2026-07-22, 같은 날) — BRCA WSI trunk pretrain → PAAD 전이, 3시드 평균 negative
  result**: WSI trunk(proj+Nystromformer)만 BRCA 전체 1058명으로 RNA-예측 보조과제(HE2RNA류,
  raw top1500 유전자 그대로, 뭉뚱그리지 않음 — `--rna-genes pathway8`의 부호 없는 평균 실패
  전례를 피함)로 pretrain한 뒤(`scripts/pretrain_brca_wsi_trunk.py`), `train.py
  --pretrained-wsi-trunk`로 PAAD(PMA_EX_SS_AUX, `--backbone uni`, literature_1500)에 이식해
  from-scratch baseline과 3시드(42/84/126) 비교:

  | seed | baseline internal | pretrained internal | baseline external | pretrained external |
  |---|---|---|---|---|
  | 42  | 0.6288 | 0.6524 (+0.024) | 0.5993 | 0.6011 (+0.002) |
  | 84  | 0.6585 | 0.5976 (-0.061) | 0.6139 | 0.5952 (-0.019) |
  | 126 | 0.6680 | 0.6598 (-0.008) | 0.6084 | 0.6203 (+0.012) |
  | 평균 | **0.6518** | **0.6366 (-0.015)** | **0.6072** | **0.6055 (-0.002)** |

  seed42 단독으로는 방향이 긍정적으로 보였으나(internal +0.024), 3시드 평균은 오히려 근소하게
  더 나쁘고(internal -0.015, external -0.002) seed84는 명확히 악화(-0.061) — 이 프로젝트에서
  반복돼온 "단일 시드 결과는 재현 안 됨" 패턴과 같다. **결론: BRCA(raw top1500, PDAC과 유전자
  겹침 12.3%뿐) trunk pretrain은 PAAD에 재현 가능한 이득을 주지 못한다** — negative transfer
  우려가 완전히 틀리진 않았을 가능성. `--pretrained-wsi-trunk` 인프라(train.py)는 남겨두되(다른
  pretrain 소스로 재시도 가능성 대비), 지금 우선순위 갱신에는 반영 안 함. 후속으로 PDAC과 유전자
  프로그램이 더 겹치는 암종(COAD/STAD/ESCA/CHOL 등 소화기 선암 계열)을 pretrain 소스로 스크리닝
  중(`scripts/screen_gene_overlap.py`, WSI 없이 RNA만 가볍게 비교) — 결과 나옴, 아래 참조.
  - **유전자 겹침 스크리닝 결과(2026-07-22) — BRCA를 이기는 후보 없음**: 소화기 선암 계열
    (COAD/STAD/ESCA/CHOL, 각 최대 200 case RNA-seq만, WSI 없이)로 top1500 고분산 유전자를
    뽑아 PAAD literature_1500과 겹침 비교.

    | 암종 | n | overlap/1500 | Jaccard |
    |---|---|---|---|
    | BRCA(기존, 비교 기준) | 1058 | 185 (12.3%) | 0.0657 |
    | STAD(위암) | 200 | 182 (12.1%) | 0.0646 |
    | CHOL(담관암) | 35 | 178 (11.9%) | 0.0631 |
    | COAD(대장암) | 196 | 162 (10.8%) | 0.0571 |
    | ESCA(식도암) | 184 | 151 (10.1%) | 0.0530 |

    "소화기 선암이 조직학적으로 PDAC과 더 가까우니 유전자 프로그램도 더 겹칠 것"이라는
    가설이 이 지표로는 기각됨 — 전부 BRCA와 비슷하거나 낮다. top1500이 "고분산" 기준으로
    뽑히는 이상 어느 암종에서든 증식/면역/기질 같은 범용 프로그램이 대부분을 차지하고,
    PAAD의 literature_1500은 그와 별개로 Bailey/Moffitt subtype 유전자(GATA6/HNF1A/PDX1 등
    췌장 발생학적 전사인자, 꼭 고분산은 아님)가 섞여 있어 조직학적 유사성이 이 비교 방식에는
    잘 반영되지 않는 것으로 보인다. **결론: 위 negative result(BRCA pretrain 3시드 평균 무효)와
    합쳐, WSI trunk를 다른 암종에서 RNA-aux로 pretrain해 PAAD로 전이하는 경로는 현재로선
    닫는다.** WSI를 추가로 다운로드하지 않고 RNA만으로 먼저 스크리닝한 덕에 비용 손실 없이
    빠르게 접을 수 있었음.
  - **"다른 task"(pretrain 신호 강화) 재시도 — 3가지 변형 전부 negative, 경로 최종 종료
    (2026-07-22)**: 유전자 겹침 스크리닝이 BRCA를 이길 대안을 못 찾은 뒤, "pretrain 신호 자체가
    약해서(top1500 기준 분산 설명력 5.7%뿐) 전이가 안 된 것 아니냐"는 가설로 두 가지를 추가
    시도했다 — (1) BRCA/PAAD 공유 185개 유전자만으로 타깃을 좁힘(범용 프로그램에 집중, 뭉뚱그리지
    않고 개별 유전자 그대로 유지), (2) lr을 1e-5→1e-4로 올려 pretrain 자체의 수렴을 강화. 둘 다
    실제로 pretrain 품질은 확연히 좋아졌다(val MSE 분산 설명력: top1500 5.7% → overlap185/lr1e-5
    7.5% → overlap185/lr1e-4 **24.6%**, 약 3배). PAAD 3시드(42/84/126, `--backbone uni`,
    PMA_EX_SS_AUX) 최종 비교:

    | 구성 | internal 평균 | external 평균 |
    |---|---|---|
    | baseline(from-scratch) | 0.6518 | 0.6072 |
    | top1500, lr=1e-5 | 0.6366 (-0.015) | 0.6055 (-0.002) |
    | overlap185, lr=1e-5 | 0.6432 (-0.009) | 0.6115 (+0.004) |
    | overlap185, lr=1e-4 | 0.6360 (-0.016) | 0.6119 (+0.005) |

    pretrain 신호가 3배 이상 강해졌는데도 다운스트림 결과는 사실상 그대로다(오히려 internal은
    가장 좋은 pretrain에서 가장 나쁨) — **pretrain 품질과 PAAD 전이 성능이 무관하다는 게 세
    변형에 걸쳐 일관되게 확인됐다.** 병목이 "pretrain이 약해서"가 아니라는 뜻 — PAAD 자체의
    fine-tune 표본(91명, --external 기준)이 너무 작아 어떤 초기화 차이도 그 학습 노이즈에
    묻혀버리는 쪽이 훨씬 유력한 설명이다. **결론: WSI trunk를 다른 코호트에서 RNA-aux로
    pretrain해 PAAD로 전이하는 경로는(유전자셋/lr 등 하이퍼파라미터를 어떻게 바꿔도) 재현 가능한
    이득이 없다 — 이 접근 자체를 최종 종료한다.** 표본 자체를 늘리는 것(추가 코호트 확보,
    실시간 augmentation 등) 외에는 이 병목을 우회할 방법을 못 찾았다.
  - **부수 발견(코드 버그) — checkpoint 파일명에 seed/`--pretrained-wsi-trunk` 미반영**: 이 비교를
    병렬(baseline+pretrained 동시 실행)로 처음 돌렸을 때 두 run이 같은 checkpoint 파일을 공유해
    최종 test/external 지표가 완전히 동일하게 나오는 사고가 있었다(경합 상태로 서로 덮어씀) —
    `tag`에 `_PRETRAINED`/`_seed{N}`를 추가해 수정(`train.py`). 기존 스윕 스크립트(`*.ps1`)는 전부
    시드를 순차 실행해왔어서 과거 결과엔 영향 없음 — 이번에 병렬 실행을 처음 시도하면서 드러난
    문제.
- **관련 파일**: `scripts/brca_common.py`(공유 case/split/RNA 로더), `scripts/select_brca_rna_genes.py`,
  `scripts/train_brca_m4.py`, `scripts/train_brca_m7.py`, `scripts/extract_brca_labels.py`,
  `scripts/extract_brca_rna.py`, `scripts/prepare_brca_data.py`, `scripts/_download_brca_hf.py`.
  wandb: 그룹 `BRCA_PMA_TOP1500_SS_AUX_seed42_0722` / `BRCA_M7_TOP1500_seed42_0722` (프로젝트
  `Path-ViT`). **주의**: 이 세션에서 `python`(base conda env)로 실행한 초기 시도는 wandb가
  설치 안 돼 있어 기록 안 됨 — 반드시 `PathViT-ray` conda env로 실행해야 wandb가 기록된다(base
  env엔 wandb 미설치, requirements.txt엔 있지만 base에 반영 안 됨).

---

## 🔴 최상위 발견(2026-07-21) — RNA-seq 전처리 버그, 프로젝트 전체에 영향 가능성

**상태: 확인됨(통제실험으로 검증), 메인 파이프라인 수정 및 전면 재검증 필요.**

`data/extract_rna_clinical.py::_read_tpm()`가 원본 GDC STAR-counts TSV에서 **`tpm_unstranded`
컬럼을 로그 변환 없이 그대로 z-score**해왔다. 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)는
`fpkm_uq_unstranded` 컬럼에 `log2(x+1)`을 적용한 뒤 z-score한다 — 이 프로젝트가 처음부터 이 차이를
"매핑 로직은 같다"고 여러 번 확인했지만, **실제 값 컬럼/변환 자체를 대조한 적이 없었다**.

- **왜곡도 비교**(TCGA 샘플 1건, protein-coding 유전자): TPM 원본 skew=**32.4**(상위 발현 유전자
  TPM 20,163까지 극단적 outlier) vs log2(FPKM-UQ+1) skew=**0.82**(정규분포에 훨씬 가까움) — 원본
  파이프라인은 z-score 정규화의 전제(대략 정규분포) 자체가 심하게 깨진 입력을 모델에 줘왔다.
- **`scripts/reference_repro_m7.py --rna-source fpkm_uq_log2`로 검증**(레퍼런스 M7 원본 코드 +
  우리 case pool, RNA 값만 교체):

  | 프로토콜 | 기존(TPM, 로그 없음) | 수정(log2 FPKM-UQ) |
  |---|---|---|
  | pooled(internal, 3 split seed 평균) | 0.598 | **0.664**(split_seed=126: **0.702**, 레퍼런스 헤드라인 0.701 거의 일치) |
  | external(tcga→cptac, 3시드 평균) | 0.494(무의미) | **0.624**(뚜렷한 신호, 이 프로젝트의 WSI+RNA 최고 기록과 맞먹음) |

  RNA 전처리를 레퍼런스와 맞춘 것만으로 external이 동전 던지기 수준에서 이 프로젝트 최고 기록급
  으로 뛰었다 — 지금까지 데이터/모델/프로토콜 통제실험 중 가장 강력한 신호.
- **의의**: RNA를 쓰는 모든 모델(M4/M4A/PM4/PMA/M6/M6X/M7)이 이 프로젝트 시작부터 이 왜곡된
  입력으로 학습돼왔다. 오늘 확인한 "dropout 0.3 아니면 붕괴", "슬라이드 구성 조금만 바꿔도 붕괴"
  같은 극단적 민감성도 이 조건 나쁜 입력이 근본 원인 중 하나였을 가능성이 있다 — **지금까지의
  negative result들이 RNA 전처리를 고치면 다르게 나올 수 있다는 뜻이라, 재검증이 필요하다.**
- **다음 액션(완료)**: `data/extract_rna_clinical.py`를 `fpkm_uq_unstranded`+`log2(x+1)`로 고쳐
  `data/rna_{tcga,cptac}.csv`·유전자 재선정까지 재생성(기존 파일은 `data/_backup_pre_fpkmuq_fix/`에
  백업). 대표 모델 재검증 결과:

  | 모델 | External C(tcga→cptac, 3시드) 기존 | RNA 수정 후 |
  |---|---|---|
  | M7_EX | 0.631/0.634/0.629(평균 0.631) | 0.624/0.632/0.601(평균 0.619) |
  | M4A_EX_SS_AUX | 0.581/0.609/0.611(평균 0.600) | 0.568/0.645/0.588(평균 0.600) |
  | PMA_EX_SS_AUX | 0.611/0.630/0.599(평균 0.614) | 0.629/0.637/0.564(평균 0.610) |
  | PM4_EX_SS_AUX | (기존 both만 0.593, external 미검증) | 0.547/0.613/0.577(평균 0.579, 0/3 유의) |

  **레퍼런스 코드 통제실험(0.49→0.62)만큼 우리 기존 모델엔 극적 효과가 없었다** — M7/M4A/PMA
  전부 평균이 사실상 그대로(±0.01 수준). RNA 수정의 극적 효과는 레퍼런스의 전체 셋업(risk head
  구조 + lr=5e-5+스케줄러+100epoch 레시피)과 결합됐을 때만 나타나고, RNA 값만 바꿔 끼우는 것만
  으로는 우리 기존 설계에 큰 영향이 없다는 뜻으로 보인다. **다만 RNA 버그 자체는 명백한 결함이라
  수정을 유지한다** — 성능과 무관하게 옳은 수정.
- **파생 실험(RNA 수정 이후, 전부 PMA_EX_SS_AUX 기준 external 3시드, 2026-07-21)**:
  - **tile fusion에 GELU+Dropout(0.4) 추가**(`models/cnn_encoder.py::CNNEncoder.proj`, 레퍼런스
    tile_fusion과 동일 — 기존엔 Linear+LayerNorm뿐): 0.633/0.638/0.563(평균 **0.611**, 기존
    0.610과 사실상 동일) — 깨끗한 null result, 유지.
  - **RNA/Clinical 인코더를 레퍼런스 비율(RNA=256, Clinical=16)로 교체**(tile-fusion 위에 추가
    적용, `CoAttentionPooling`에 `context_dim` 파라미터 신규 지원해 RNA/WSI 폭이 달라도 동작하게
    함): 0.619/0.610/0.574(평균 **0.601**, 1/3만 유의) — tile-fusion 단독(0.611)보다 소폭 하락,
    개선 없음.
  - **RNA=128/Clinical=16만 축소, WSI(embed_dim)는 64로 그대로 유지**(위 RNA=256 시도는 WSI
    폭까지 같이 커진 게 원인이었는지 분리하기 위한 재시도, `models/vit_pma.py`
    `rna_dim`/`clinical_dim` 파라미터, `train.py --rna-dim --clinical-dim`, `--PMA` 전용):
    0.611/0.608/0.614(평균 **0.611**, 3/3 유의, HR 1.46~1.63) — 베이스라인(0.610)과 사실상 동일한
    null result. WSI 폭을 그대로 둬도 결과가 바뀌지 않아, RNA=256 시도의 negative가 "WSI 차원까지
    같이 키운 부작용"이었다는 가설은 기각된다 — 인코더 폭 비율 자체가 이 아키텍처엔 영향이 없다는
    쪽이 맞다. (참고: 3시드 값이 0.608~0.614로 매우 좁게 모여, 베이스라인의 0.564~0.637보다 폭이
    좁긴 했으나 표본이 3개뿐이라 이 자체를 변동성 감소 효과로 단정하지는 않는다.)
  - **M7_EX risk head에 Dropout(0.4)만 추가**(은닉층 없이, 레퍼런스 M4 사양 — 이전 시도는 은닉층
    까지 있는 M7 사양이었음): seed42 external C=0.622로 수치만 보면 나쁘지 않았지만, **학습 곡선이
    명백히 퇴화**(train_c_index 30 epoch 만에 0.9865, val은 0.50대 고정, lifelines가 "collinearity
    or complete separation" 경고 반복) — 결과 확정 전 사용자 판단으로 배치 중단. RNA를 고쳐도
    risk head 직전 Dropout은 여전히 해롭다는 기존 결론 유지, 코드 원복.
- **결론**: risk head/tile-fusion/인코더 비율 쪽 개입은 RNA 수정 이후에도 전부 null 또는 negative
  (인코더 비율은 WSI 폭 고정 여부와 무관하게 두 버전 다 null) — 이 축의 탐색은 여기서 마무리.
  남은 유력한 다음 방향은 (a) 레퍼런스 학습 레시피(lr/스케줄러/epochs) 전체를 WSI 모델에 적용,
  (b) 패치 단위 augmentation 도입 — 둘 다 이번엔 보류(사용자 판단).

- **레퍼런스 M4(WSI 포함) 통제실험(2026-07-21) — 이 프로젝트에서 가장 결정적인 확인**:
  M7에 이어 레퍼런스의 실제 M4(`PathologyRNASeqClinicalMIL`, MorphologyBurdenPooling+RNA-gated
  sigmoid 게이트)를 그대로 가져와(`scripts/reference_repro_m4.py`) 우리 데이터로 검증했다.
  단순화한 부분(사용자 승인): backbone은 레퍼런스의 UNI2-h(1536dim) 대신 우리 캐시된 ResNet50
  Lunit SwAV feature(2048dim)를 그대로 사용(backbone 자체는 이미 검증된 대로 큰 영향 없음),
  공간 임베딩(coord_dim=6)은 제외(우리 좌표 포맷이 달라 이 축은 우리 고유 novelty 영역으로
  남겨둠). 슬라이드는 대표 1장/환자(`one_slide_per_case=True`, 레퍼런스와 동일), 학습 레시피는
  M4_Train.ipynb 그대로(`lr=5e-5, weight_decay=1e-3, epochs=50, patience=15, batch=16,
  ReduceLROnPlateau`).

  | | External(tcga→cptac, 3시드) | Pooled(3 split seed) |
  |---|---|---|
  | **레퍼런스 M4 원본 아키텍처 + 우리 데이터** | 0.618/0.627/0.593 → 평균 **0.613** | 0.619/0.664/0.674 → 평균 **0.652** |
  | (참고) 레퍼런스 M7 통제실험(WSI 없음) | 평균 0.624 | 평균 0.664 |
  | (참고) 레퍼런스 공식 M4 헤드라인 | — | 0.722 |
  | (참고) 우리 PMA_EX_SS_AUX(RNA 수정판, external) | 평균 0.610 | — |

  **레퍼런스의 진짜 WSI 융합 아키텍처를 그대로 써도, WSI가 전혀 없는 M7 통제실험보다 external도
  pooled도 더 낫지 않다**(0.613<0.624, 0.652<0.664) — 오히려 둘 다 근소하게 낮다. 지금까지
  우리 자체 아키텍처(M4A/PMA)로 반복 확인해온 "이 데이터·코호트 규모에서는 WSI가 fusion 설계와
  무관하게 순증분 기여를 못 한다"는 결론(1번 항목 (f))을, **레퍼런스 자신의 원본 코드로도 동일하게
  재확인**했다 — 우리 WSI 브랜치 설계가 문제였다는 가설은 이제 근거가 매우 약해졌고, WSI 자체가
  (적어도 이 코호트 규모·전처리 조건에서는) 이 태스크에 유의미하게 기여하지 못한다는 쪽이 훨씬
  설득력 있는 결론이 됐다.
  - **주의(단순화의 한계)**: backbone을 UNI2-h→ResNet50으로, 공간 임베딩을 켬→끔으로 바꿨기
    때문에 레퍼런스 헤드라인(0.722)과의 잔여 격차(0.652 vs 0.722)가 이 단순화 때문인지, 데이터
    자체의 잔여 차이 때문인지는 이 실험만으로 완전히 분리되지 않는다. 다만 "WSI가 M7 대비
    추가 기여를 못 한다"는 핵심 결론은 이 단순화와 무관하게 안정적으로 재현됐다(같은 데이터·같은
    레시피 안에서 M4 vs M7 비교이므로).
  - **누락됐던 "진짜 internal" 보강(2026-07-21, seed42)**: 위 표의 "External" 프로토콜은
    train(tcga)/val(tcga, early stopping 전용)/test(cptac, external) 구조라, val은 held-out
    test가 아니라 모델 선택용이었다 — 우리 자체 모델이 보고해온 "Internal C"(학습에 전혀 안 쓰인
    순수 held-out)에 대응하는 값이 레퍼런스 재현 스크립트엔 없었다. tcga "test" split을 추가해
    `internal_test_c_index(tcga)`로 별도 평가(`reference_repro_m7.py`/`reference_repro_m4.py`에
    `internal_test` 인자 추가):

    | | internal_test(tcga held-out) | external(cptac) | pooled_test(split_seed=42) |
    |---|---|---|---|
    | 레퍼런스 M7(WSI 없음) | **0.6094** | 0.6282 | **0.6211** |
    | 레퍼런스 M4(WSI 포함) | **0.5923** | 0.6182 | **0.6186** |

    **레퍼런스 자신의 코드·데이터로 봐도 internal(tcga held-out)에서조차 M4가 M7보다 낮다**
    (0.5923<0.6094) — pooled(0.6186<0.6211)에서도 근소하게 같은 방향. external뿐 아니라
    internal·pooled 세 프로토콜 전부에서 일관되게 WSI가 순증분 기여를 못 한다는 뜻으로,
    "external에서만 유독 WSI가 해롭다"가 아니라 **평가 방식과 무관하게 레퍼런스 자신의 WSI
    아키텍처가 이 데이터에서 기여하지 못한다**는 훨씬 강한 결론을 뒷받침한다.

- **시드 앙상블(재학습 없이, 2026-07-21)**: PMA_EX_SS_AUX(RNA 수정판, 균일 embed_dim) 체크포인트를
  3시드(42/84/126) 새로 학습해 seed-tagged로 보존한 뒤(`scripts/_pma_ex_ss_aux_ensemble_ckpt_ext.ps1`),
  external(cptac) risk score를 환자 단위로 단순 평균(`scripts/ensemble_eval.py`).

  | | seed42 | seed84 | seed126 | 평균/앙상블 |
  |---|---|---|---|---|
  | 개별 시드 | 0.6289 | 0.6368 | 0.5638 | (평균 0.6098) |
  | 앙상블(risk score 평균) | — | — | — | **0.6165**, HR 1.789 [1.232, 2.599], p=0.0019 |

  앙상블이 단순 3시드 평균(0.6098)보다는 소폭 높지만(+0.007), 최고 단일 시드(seed84, 0.6368)보다는
  낮다 — 최악의 시드(seed126, 0.5638)를 뽑을 위험을 완만하게 낮춰주는 정도이지, 분산을 상쇄할
  만큼 강한 효과는 아니다.

## 🔴 최상위 발견(2026-07-21, 2차) — WSI 브랜치가 학습 신호를 거의 못 받아 attention이 uniform으로 붕괴

**상태: 확인됨(3가지 독립 진단으로 교차검증), "어떤 fusion 구조를 쓸까"보다 근본적인 문제.**

지금까지 risk head/tile-fusion/인코더 비율/앙상블 등 아키텍처 축을 다 시도해도 external에서
WSI+RNA+Clinical이 RNA+Clinical만 쓰는 M7_EX을 못 넘는 패턴이 계속 재현됐다. "더 나은 fusion을
찾자"는 시도를 멈추고, 이미 학습된 PMA_EX_SS_AUX 체크포인트(3시드, 재학습 없음)가 **실제로
WSI를 어떻게 쓰고 있는지** 직접 진단했다.

- **(A) WSI ablation**(`scripts/diagnose_wsi_reliance.py`) — risk_head 입력 `[z_wsi, z_clinical,
  z_rna]` 중 z_wsi만 0으로 치환하거나 환자 간 무작위로 셔플(patient-specific 신호 제거, risk_head
  재실행만 하므로 재학습 불필요):

  | | baseline | z_wsi=0(zero-ablation) | z_wsi 셔플(perm-ablation) |
  |---|---|---|---|
  | internal(tcga held-out, 3시드 평균) | 0.6767 | **0.6967**(+0.020) | 0.6898(+0.013) |
  | external(cptac, 3시드 평균) | 0.6105 | **0.6147**(+0.004) | 0.6134(+0.003) |

  **WSI 신호를 통째로 지우거나 섞어도 internal/external 둘 다 성능이 떨어지지 않는다** — 오히려
  seed84/126의 internal에서는 +0.02~+0.04로 뚜렷이 좋아졌다. "external에서만 WSI가 해롭다"가
  아니라 **internal이든 external이든 학습된 모델이 z_wsi를 사실상 안 쓰고 있다**는 뜻이다.

- **(B) attention 분포**(같은 스크립트) — RNA-guided co-attention(4개 통계적 관점 mean/std/
  attn/topk 중 선택)과 패치 단위 ABMIL attention 둘 다, 3시드·internal/external 전부
  **엔트로피 0.999~1.000/1.0(완전 균등)에 수렴** — RNA가 "이 관점이 중요하다"고 고르는 것도,
  패치 attention이 특정 패치에 집중하는 것도 전혀 없이 거의 완전 평균 풀링과 동일하게 붕괴돼 있다.

- **(C) 브랜치별 gradient norm**(`scripts/diagnose_wsi_gradients.py`, seed42, PMA_EX_SS_AUX
  레시피 그대로 재현, 30 epoch 전체 추적) — Cox 배치(16명) backward 직후·optimizer.step() 이전에
  브랜치별 파라미터 gradient L2 norm 기록:

  | 브랜치 | epoch 1 | 마지막 5epoch 평균 | risk_head 대비 |
  |---|---|---|---|
  | WSI(cnn+vit+attn_pool+component_coattn) | 2.59 | 1.98 | 0.67배 |
  | RNA encoder | 10.76 | 8.55 | **2.84배** |
  | Clinical encoder | 0.98 | 0.88 | 0.26배 |

  **RNA 인코더는 학습 1epoch부터 30epoch까지 일관되게 WSI 브랜치보다 약 4배 큰 gradient norm을
  받는다** — 일시적 초기화 효과가 아니라 학습 내내 유지되는 패턴.

- **종합 해석**: 세 진단이 서로 다른 각도에서 같은 결론을 가리킨다 — **이 표본 규모(TCGA
  학습 91명)에서 Cox loss가 RNA/Clinical이라는 "쉬운 지름길"을 곧바로 찾고, WSI 브랜치에는
  학습 내내 약한 신호만 흘려보낸다. 그 결과 co-attention과 패치 attention이 둘 다 아무것도
  구분 못 하는 균등분포로 수렴해, 학습이 끝난 모델에서 z_wsi를 지워도 아무 일도 안 일어난다.**
  지금까지 "fusion 지점을 바꾸자/차원 비율을 바꾸자/risk head를 바꾸자" 축이 전부 null이었던
  이유를 한 번에 설명한다 — 문제는 fusion *방식*이 아니라 WSI 브랜치가 이 생존 라벨(censoring
  있는 소표본)로부터 애초에 학습 신호를 거의 못 받는다는 더 근본적인 지점이다.
- **`--rna-gate-only` 검증 — 반대 방향으로 확인됨: RNA 지름길은 "방해물"이 아니라 오히려
  모델을 붙잡아주던 규제(regularizer)였다** (`models/vit_pma.py rna_gate_only`, z_rna를
  component_coattn의 query로만 남기고 risk_head 직결 concat에서는 제거, PMA_EX_SS_AUX 기준
  external, seed42 학습 중 명백한 과적합 붕괴 확인돼 즉시 중단·나머지 시드 취소):

  | epoch | train_c_index | val_c_index | val_HR |
  |---|---|---|---|
  | 1 | 0.412 | 0.491 | 1.076 |
  | 15 | 0.657 | 0.386 | 0.612 |
  | 30 | 0.683 | 0.386 | 0.826 |

  train은 30epoch 내내 꾸준히 상승(0.41→0.68)하는데 val은 처음부터 계속 하락(0.49→0.39, 랜덤
  이하)하고 **HR이 1 미만으로 방향 자체가 뒤집힘**(고위험군이 오히려 저위험) — internal
  test_c_index=0.4421, external_c_index=0.4995(둘 다 동전 던지기 수준). M7_EX risk head
  dropout-only 실험 때 봤던 것과 같은 명백한 퇴화 패턴이라 사용자 판단으로 즉시 중단.
  - **해석(가설 기각, 반대 결론)**: 애초 가설(RNA 직결 경로가 WSI 브랜치의 gradient를 굶기는
    "지름길"이니 없애면 WSI가 제대로 배울 것)과 반대로, **z_rna 직결 concat이 오히려 risk_head를
    저용량이면서 실제 예측력 있는 입력에 붙들어매는 안정화 장치였다.** 그걸 제거하니 risk_head
    입력이 [z_wsi(64), z_clinical(64)]뿐인데, z_wsi는 여전히 21만 파라미터짜리 고용량
    WSI+co-attention 경로가 자유롭게 만들어내는 값이라, 91명짜리 학습셋을 그대로 외워버리는
    쪽으로 붕괴했다. WSI 브랜치의 문제는 "gradient가 부족하다"가 아니라, **오히려 규제 없이
    풀어주면 이 표본 규모에서 감당 못 할 만큼 쉽게 과적합한다는 쪽에 더 가깝다** — (A)/(B)/(C)
    진단(gradient norm이 작다)과 이 결과(규제를 풀면 바로 과적합)가 동시에 성립하려면, WSI
    브랜치가 "약하게 배우는 게 아니라 애초에 이 표본에서 일반화 가능한 신호를 배울 수 없고,
    지금은 RNA 직결 경로가 그걸 억눌러 과적합을 막아주고 있었다"는 쪽이 가장 앞뒤가 맞는다.
  - `models/vit_pma.py`/`train.py`의 `rna_gate_only`/`--rna-gate-only`는 기본값 False로 기존
    동작에 영향 없는 인프라로만 남겨두고, 이 경로는 재시도하지 않는다.

- **Clinical 브랜치도 같은 방식으로 ablation — WSI와 마찬가지로 기여도가 사실상 0, RNA만 압도적**
  (`scripts/diagnose_wsi_reliance.py`를 세 브랜치(wsi/clinical/rna) 전부 ablation하도록 확장,
  PMA_EX_SS_AUX 3시드 평균, perm-ablation 기준):

  | 브랜치 | internal(baseline 대비) | external(baseline 대비) |
  |---|---|---|
  | WSI | +0.017 | +0.004 |
  | **Clinical** | **+0.005** | **-0.002** |
  | RNA | **-0.189** | **-0.122** |

  **Clinical을 지워도 WSI를 지웠을 때와 마찬가지로 성능이 거의 안 변한다** — WSI/Clinical 둘 다
  0 근처(사실상 노이즈 수준)에 몰려 있어 "어느 쪽이 더 쓸모없다"를 통계적으로 가릴 수 없다.
  반면 **RNA를 지우면 external C-index가 0.6105→0.4884(동전 던지기 수준)로 붕괴** — 이 모델은
  사실상 RNA 단독으로 거의 모든 예측력을 내고 있고, WSI·Clinical은 risk_head에 concat만 될 뿐
  실질적으로 "타고 있는" 상태다. `--rna-gate-only`(위 항목)가 붕괴한 이유와 정확히 들어맞는다 —
  RNA를 빼면 남는 [z_wsi, z_clinical]에는 애초에 감당할 만한 신호가 거의 없다.

- **`--shuffle-patches` — 나이스트롬 landmark 그룹핑 고정 순서 문제 검증, null result(2026-07-22)**:
  `list_patch_paths()`가 항상 좌표순 정렬된 고정 순서를 반환하는데, `nystrom_attention` 패키지의
  landmark 계산이 `rearrange('... (n l) d -> ... n d')`로 **시퀀스 순서대로 연속된 패치를
  그룹핑해 평균**내는 방식이라, 지금까지 매 epoch 똑같은 landmark 그룹핑이 반복돼왔다(패치
  keep_frac<1.0일 땐 `torch.randperm`의 부수효과로 이미 순서가 섞여왔지만, frac=1.0이면 이
  효과가 없었음). `train.py --shuffle-patches`(frac과 독립적으로 학습 forward마다 순서만
  재정렬)로 순수 셔플 효과를 격리해 검증(seed42, external, precomputed):

  | | internal test C | internal test AUC | external C | external AUC |
  |---|---|---|---|---|
  | PMA_EX_AUX(baseline, 셔플 없음) | 0.6266 | 0.6956 | 0.6270 | 0.6800 |
  | PMA_EX_SS_AUX(frac=1.0, shuffle-patches) | 0.6266 | 0.6859 | 0.6280 | 0.6818 |

  전부 노이즈 범위 안(±0.002 수준) — **패치 순서를 섞어도 좋아지지도 나빠지지도 않는다.** 걱정했던
  "나이스트롬 근사 품질 저하"도, 기대했던 "정규화로 인한 개선"도 이 아키텍처·이 데이터 규모에서는
  안 보인다.

- **`--full-attention` — 나이스트롬 근사 자체를 표준 O(N²) attention으로 교체, 3시드로 확인하니
  null(2026-07-22, seed42 단일 결과에서 정정됨)**: 셔플이 무효과였던 것과 별개로, 슬라이드당
  패치 수 실측(TCGA train 기준 평균 131/중앙값 67/최대 544)이 `num_landmarks=128`보다도 작은
  경우가 절반 이상이라 — 근사가 오히려 패딩 토큰을 landmark에 섞어 넣는 역효과를 냈을 수 있다는
  의심. 이 규모면 O(N²)도 GPU에 전혀 부담 없어(544²≈30만) `nn.MultiheadAttention`으로 교체하는
  옵션(`models/vit_encoder.py _FullSelfAttention`, `cfg.model.use_nystrom`)을 추가해
  PMA_EX_SS_AUX(external, 3시드)로 검증:

  | seed | Nystrom internal→external | Full attention internal→external |
  |---|---|---|
  | 42 | 0.6309 → 0.6289 | 0.6567(+0.026) → 0.6011(-0.028) |
  | 84 | 0.7033 → 0.6374 | 0.6667(-0.037) → 0.6253(-0.012) |
  | 126 | 0.6224 → 0.5639 | 0.5934(-0.029) → 0.5886(+0.025) |
  | **평균** | **0.6522 → 0.6101** | **0.6389(-0.013) → 0.6050(-0.005)** |

  **seed42에서 봤던 "internal 개선/external 악화" 트레이드오프는 재현되지 않았다** — 시드마다
  방향이 제각각(42는 internal↑external↓, 84는 둘 다 ↓, 126은 둘 다 ↓인데 external만 미세하게
  ↑)이고, 3시드 평균으로는 internal·external 둘 다 소폭 하락(노이즈 수준)에 그친다. **"나이스트롬의
  거친 근사가 cross-institution 일반화에 도움을 준다"는 가설은 근거가 약해졌다** — seed42 1개
  결과에 낚인 것에 가깝다. 이 프로젝트 전체에서 반복돼온 극심한 시드 편차가 여기서도 재현된
  사례로 정리한다.

- **`--no-spatial-embed` — 좌표 임베딩 자체를 빼도 null(2026-07-22)**: attention이 이미 uniform으로
  붕괴해 있고(diagnose_wsi_reliance.py) 나이스트롬/full-attention 둘 다 뚜렷한 신호가 없었던
  상황에서, "애초에 좌표 임베딩이 최종 예측에 기여하긴 하는가"를 직접 검증(`SpatialPositionEmbedding`을
  아예 빼고 `patch_tokens`만 사용, `cfg.model.use_spatial_embed`). PMA_EX_SS_AUX(seed42, external,
  Nystrom):

  | | internal test C | external C |
  |---|---|---|
  | 좌표 임베딩 있음(기존) | 0.6309 | 0.6289 |
  | **좌표 임베딩 제거** | **0.6245**(-0.006) | **0.6279**(-0.001) |

  둘 다 노이즈 범위 안 — **좌표 임베딩을 통째로 빼도 아무 변화가 없다.** attention 붕괴·WSI
  ablation 무효과와 정확히 같은 결. "나이스트롬의 뭉갬 때문에 좌표 정보가 못 쓰이는 것 아니냐"는
  가능성까지 배제하기 위해 full-attention(근사 없음)에서도 좌표 임베딩을 빼서 재검증(seed42, external):

  | | 좌표 임베딩 있음 | 좌표 임베딩 없음 |
  |---|---|---|
  | Nystrom | internal 0.6309 / external 0.6289 | internal 0.6245 / external 0.6279 |
  | Full attention | internal 0.6567 / external 0.6011 | internal 0.6481(-0.009) / external 0.5872(-0.014) |

  Full attention(근사 없이 정밀)에서도 좌표 임베딩을 빼니 소폭 하락(-0.009/-0.014)에 그친다 —
  나이스트롬이든 정밀한 full attention이든 결과가 대동소이하다.

  **종합 결론**: 좌표 기반 `SpatialPositionEmbedding`이 나이스트롬 근사 여부와 무관하게 이
  아키텍처·이 데이터 규모에서 사실상 기여를 못 하고 있다. attention(co-attention·패치 ABMIL
  둘 다)이 이미 uniform으로 붕괴해 있는 상태(diagnose_wsi_reliance.py)라, 공간 정보를 아무리
  정밀하게 넣어줘도 그걸 "써먹을" 메커니즘 자체가 죽어있었던 것으로 보인다 — WSI 브랜치
  전체(ablation 무효과)와 정확히 같은 결의 결론이다. 공간 컨텍스트 블록은 이 프로젝트의 자체
  novelty로 명시돼 있었지만, 지금 형태로는 그 novelty가 실질적 성능 기여로 이어지지 않는다는
  뜻이라 추가 기록해둔다.

---

생성일: 2026-07-16 / 최종 수정: 2026-07-16. 지금까지 조사에서 확인된 문제/가설을 우선순위 순으로 정리. "왜 지금 이 순위인지"를 각 항목에 명시해, 나중에 순서를 바꿀 때 근거를 다시 따라갈 수 있게 한다.

---

## 완료

### 0. 평가 프로토콜 난이도 차이 검증 (`--dataset both`)
**상태: M1/M4/M4A/M4B 전부 완료.**

| | Internal C-index | HR | log-rank p |
|---|---|---|---|
| M1 | 0.546 | 1.41 | 0.35 |
| M4 (gate-bias) | 0.539 | 1.45 | 0.37 |
| M4A (co-attention) | 0.549 | 1.42 | 0.32 |
| M4B (pre-ViT FiLM) | 0.536 | 1.26 | 0.55 |
| 레퍼런스 M4 | 0.722 | 3.32 | 0.00064 |

- **확인된 사실**: M1은 레퍼런스(UNI 기반)와 both 프로토콜에서 거의 일치(HR 1.41 vs 1.44, p 0.35 vs 0.30). 반면 **M4/M4A/M4B는 모두 레퍼런스(C-index 0.722, HR 3.32, p 0.00064)에 한참 못 미침**, 심지어 M1보다도 딱히 낫지 않다 — 넷 다 0.54 내외에 몰려 있다.
- **판단 결과**: M4A가 M4보다 0.01 높지만(0.549 vs 0.539) 시드 간 편차(0.50~0.59) 안에 묻히는 수준이라 "새로운 RNA-이미지 상호작용을 발견했다"고 부를 근거는 없음. M4B는 오히려 M4보다 낮음. **RNA 개입 지점(post-hoc/게이트-bias/co-attention/pre-ViT FiLM)은 결과에 유의미한 영향이 없다는 결론이 재확인됨.**
- **해석**: 병리 인코더/MIL 집계 자체는 문제가 아니다(M1이 이미 맞아떨어지므로). 격차는 clinical+RNA를 WSI와 **결합(fusion)하는 방식**에서 발생한다. 유력 원인: 우리 WSI 표현이 ABMIL로 만든 단일 벡터라 RNA가 조절할 "손잡이"가 너무 적다(아래 1번 항목) — 개입 *지점*을 바꿔봐도 안 되니, 개입 *대상*(표현 자체의 풍부함)을 바꿔야 한다는 쪽으로 결론.

### 1. ABMIL 단일 벡터 압축 → 다성분(multi-component) pooling + co-attention (PMA) + 레퍼런스식 유전자 재선정
**상태(2026-07-17 최종 업데이트): 유전자셋 교체(2번 항목)가 압도적 기여 요인임을 재확인했고, 한발 더 나아가 (f)에서 — external(진짜 cross-institution) 기준으로는 WSI를 아예 안 쓰는 M7_EX가 WSI를 쓰는 모든 모델(PMA_EX 포함)을 능가함을 확인. "다성분 pooling+co-attention"뿐 아니라 "WSI 자체"의 기여도까지 재검토가 필요한 상황.**

| | Internal C-index | HR | log-rank p |
|---|---|---|---|
| M4A (기존, 단일 벡터+co-attention) | 0.549 | 1.42 | 0.32 |
| PM4 (다성분 4개 + post-hoc 게이트) | 0.553 | 1.24 | 0.54 |
| PMA (다성분 4개 + co-attention) | 0.583 | 1.65 | 0.32 |
| **PMA_EX (PMA + literature_1500 유전자)** | **0.656** (seed 범위 0.604~0.733) | **2.15** | **0.10** |
| 레퍼런스 M4 | 0.722 | 3.32 | 0.00064 |

- **핵심 발견**: 다성분 pooling(표현력 확보) + co-attention(RNA가 능동적으로 관점 선택) + 레퍼런스식 유전자 재선정(생존 예측에 최적화된 Cox+Stouffer 1500개, 아래 2번 항목) — **세 가지를 함께 적용하니 처음으로 레퍼런스에 근접**. 개별 요소만으로는(PM4/PMA 단독, M6X 단독) 전부 미미했는데 함께 쌓으니 확실히 다른 그림. seed126은 c-index 0.733/HR 3.27/p 0.0003으로 레퍼런스(0.722/3.32/0.00064)와 거의 정확히 일치.
- **주의**: seed42(0.604)/seed84(0.630)는 seed126(0.733)보다 수수함 — 평균이 진짜 실력인지 seed126이 특히 잘 맞은 표본인지 추가 시드로 확인 필요. 그래도 최소값(0.604)조차 이전 최고 기록(PMA-subtype 0.583)보다 높다는 점은 고무적.
- **문제의 정확한 위치(다성분 pooling 자체)**: "ABMIL이냐 CLAM이냐"가 핵심이 아니었다 — CLAM(Lu et al. 2021)도 내부적으로 동일한 gated attention pooling을 쓰고 최종 표현은 여전히 압축 벡터 1개다. 레퍼런스가 쓰는 Morphology Burden Pooling처럼 **여러 통계적 관점을 압축 없이 병렬로 유지**하는 게 핵심이었다.
- **노벨티**: ViT self-attention(Nystromformer) 공간 컨텍스트 블록(레퍼런스에는 없음) + RNA 개입 지점 체계적 비교(M4/M4A/M4B/PM4/PMA 사다리, 레퍼런스는 게이트 하나만 고정) + 멀티시드/internal-external/both 프로토콜 rigor — 레퍼런스와 겹치는 다성분 pooling·유전자 재선정 인프라 위에 이 세 축을 얹은 조합이 차별점.
- **(c) external 검증 결과 — 재현 안 됨**: PMA(subtype 유전자)를 `--external`(3시드×tcga/cptac)로 재검증한 결과:

  | | Internal | External |
  |---|---|---|
  | M1 | 0.550 | 0.468 |
  | M4 | 0.510 | 0.512 |
  | M4A | 0.552 | 0.530 |
  | M4B | 0.509 | 0.514 |
  | **PMA** | 0.528 | **0.528** |
  | M7(WSI 없음) | — | 0.575 |

  PMA의 external(0.528)이 M4A(0.530)와 사실상 동률이고, M4/M4B와 같은 좁은 범위(0.51~0.53)에 몰림 — both에서 보인 우위(0.583, 최고 기록)가 **진짜 cross-institution 일반화에서는 사라짐**. M7(0.575)도 여전히 못 넘음. 다성분 pooling+co-attention *만으로는* 진짜 일반화가 개선되지 않는다는 뜻.

- **(d) PMA_EX(literature_1500) external 검증 — 재현됨, 지금까지 최고 성적**: (c)에서 빠졌던 PMA_EX를 동일하게 `--external`(3시드×tcga/cptac)로 검증. (M7만 `train_light.py`가 time-dependent AUC를 아예 계산하지 않아 AUC 열이 없음.)

  | | Internal C | Internal HR | Internal p | Internal AUC | External C | External HR | External p | External AUC |
  |---|---|---|---|---|---|---|---|---|
  | M1 | 0.550 | 1.106 | 0.629 | 0.518 | 0.468 | 0.803 | 0.387 | 0.464 |
  | M4 | 0.510 | 1.398 | 0.425 | 0.510 | 0.512 | 1.031 | 0.252 | 0.535 |
  | M4A | 0.552 | 1.422 | 0.368 | 0.531 | 0.529 | 1.117 | 0.483 | 0.545 |
  | M4B | 0.509 | 1.481 | 0.402 | 0.509 | 0.514 | 1.089 | 0.187 | 0.538 |
  | M7(WSI 없음) | 0.612 | 2.134 | 0.109 | — | 0.575 | 1.453 | 0.197 | — |
  | PMA (subtype) | 0.528 | 1.355 | 0.422 | 0.567 | 0.528 | 1.148 | 0.341 | 0.563 |
  | **PMA_EX (literature_1500)** | **0.613** | **1.615** | 0.407 | **0.608** | **0.603** | **1.781** | 0.150 | **0.610** |

  PMA_EX는 internal(0.613)·external(0.603) 둘 다 이전 최고였던 M7(0.612/0.575)을 넘어섰고, external HR(1.781)·external AUC(0.610)도 전체 모델 중 최고(M4A 0.545, PMA 0.563 대비 뚜렷한 격차). external p 평균(0.150)은 유의하지 않지만, 시드별로 보면 6개 중 5개가 p<0.01(0.0, 0.0011, 0.0024, 0.0069, 0.0044)이고 단 하나(cptac seed84, c=0.512/AUC=0.508/p=0.883)가 평균을 끌어올림 — **아웃라이어 하나를 빼면 사실상 일관되게 유의미한 신호**.

  **PMA_EX 시드별 상세**:

  | 코호트 | seed | internal C | internal AUC | external C | external HR | external p | external AUC |
  |---|---|---|---|---|---|---|---|
  | tcga | 42 | 0.569 | 0.583 | 0.625 | 2.325 | 0.000 | 0.647 |
  | tcga | 84 | 0.642 | 0.491 | 0.632 | 1.840 | 0.001 | 0.657 |
  | tcga | 126 | 0.635 | 0.570 | 0.598 | 1.765 | 0.002 | 0.590 |
  | cptac | 42 | 0.569 | 0.550 | 0.628 | 1.839 | 0.007 | 0.624 |
  | cptac | 84 | 0.601 | 0.697 | 0.512 | 1.033 | 0.883 | 0.508 |
  | cptac | 126 | 0.662 | 0.758 | 0.621 | 1.882 | 0.004 | 0.631 |
  → PMA(subtype)와 PMA_EX(literature_1500)의 대비로, 1번 항목(다성분 pooling 아키텍처) 자체는 external 일반화에 크게 기여하지 않았고, **2번 항목(레퍼런스식 유전자 재선정)이 진짜 기여 요인이었다**는 게 명확해짐 — "both에서만 좋고 external엔 재현 안 되는 착시"라는 이전 결론은 PMA(subtype)에 한정된 얘기였다.

- **(e) literature_1500을 M4/M4A/PM4에도 적용 — 아키텍처 기여도가 애초 생각보다 작음이 확인됨** (`--dataset both`, 3시드, 2026-07-17 새벽 배치):

  | | Internal C | Internal HR | Internal p | Internal AUC | (subtype 시절 both C) |
  |---|---|---|---|---|---|
  | M4_EX | 0.628 | 1.914 | 0.123 | 0.646 | 0.539 |
  | M4A_EX | **0.644** | 2.023 | 0.097 | 0.651 | 0.549 |
  | PM4_EX | 0.611 | 1.745 | 0.200 | 0.635 | 0.553 |
  | M6_EX (WSI 없음) | 0.619 | 2.190 | 0.095 | 0.647 | — |
  | M7_EX (WSI 없음) | 0.621 | 2.091 | 0.252 | 0.669 | — |
  | PMA_EX (다성분+co-attention) | 0.656 | 2.15 | 0.10 | — | 0.583 |

  **세 모델(M4/M4A/PM4) 전부 유전자셋만 바꿨는데 subtype 대비 +0.06~+0.09 c-index가 일제히 뛰었다** — PMA_EX가 처음 보여준 도약(0.583→0.656, +0.073)과 거의 같은 폭이 가장 단순한 M4(플레인 FiLM 게이트-bias, 다성분 pooling 없음)에서도 그대로 재현됨(0.539→0.628, +0.089). 심지어 **M4A_EX(0.644, 다성분 pooling 없이 co-attention만)가 PM4_EX(0.611, 다성분 pooling+post-hoc 게이트)보다 높고, PMA_EX(0.656)에 거의 근접** — 다성분 pooling을 추가한 게 오히려 약간의 손해(M4_EX 0.628 > PM4_EX 0.611)로 보이는 정황도 있음.
  - **결론**: 애초 "다성분 pooling이 핵심 병목을 풀었다"는 가설(0번 항목 참조)은 **부분적으로만 맞다** — 진짜 압도적 기여 요인은 유전자셋(2번 항목)이었고, RNA 개입 지점 중에서는 co-attention(M4A류)이 post-hoc 게이트(M4류)나 다성분+게이트(PM4)보다 근소 우위를 보이는 정도. PMA_EX가 여전히 both 프로토콜 최고 기록(0.656)이긴 하지만, M4A_EX(0.644)와의 격차(+0.012)는 M4A_EX가 M4_EX(0.628)를 이기는 격차(+0.016)와 비슷한 수준이라 "다성분 pooling의 추가 기여"라고 부르기엔 근거가 약함.
  - **WSI 유무 비교**: M6_EX/M7_EX(WSI 완전히 없음)도 0.619/0.621로, M4_EX(0.628)·PM4_EX(0.611)와 사실상 같은 범위 — **문헌기반 유전자셋을 쓰면 WSI를 아예 빼도 거의 동일한 both-프로토콜 성능이 나온다**는, 이전보다 더 불편한 관찰. WSI가 유의미하게 앞서는 건 M4A_EX(0.644)와 PMA_EX(0.656)뿐이라, "co-attention 방식으로 WSI와 RNA를 결합하는 것" 자체는 여전히 근소하게 의미 있어 보이지만 격차가 크지 않음.
  - **주의**: 전부 `--dataset both`(internal)만 검증됨 — PMA(subtype)가 both에서 좋았다가 external에서 무너진 전례가 있으므로, 이 표의 순위(M4A_EX/PM4_EX 등)가 external에서도 유지되는지는 아직 모른다. 다음 액션 (a) 참조.

- **(f) M4_EX/M4A_EX/PM4_EX/M6_EX/M7_EX external 검증 — WSI가 오히려 안 좋을 수 있다는 증거, 지금까지 가장 강력한 신호** (`--external`, 2코호트×3시드=6, 2026-07-17):

  | | Internal C | Internal HR | Internal p | Internal AUC | External C | External HR | External p | External AUC |
  |---|---|---|---|---|---|---|---|---|
  | M4_EX (WSI+게이트) | 0.609 | 1.609 | 0.530 | 0.569 | 0.604 | 1.716 | 0.103 | 0.618 |
  | M4A_EX (WSI+co-attn) | 0.620 | 2.376 | 0.227 | 0.600 | 0.611 | 1.677 | 0.074 | 0.620 |
  | PM4_EX (WSI+다성분+게이트) | 0.606 | 1.475 | 0.452 | 0.574 | 0.593 | 1.651 | 0.125 | 0.606 |
  | **M6_EX (RNA만, WSI 없음)** | 0.637 | 1.897 | 0.269 | 0.662 | **0.627** | 1.914 | **0.005** | 0.657 |
  | **M7_EX (RNA+Clinical, WSI 없음)** | 0.627 | 1.785 | 0.373 | 0.640 | **0.634** | 1.975 | **0.0025** | 0.670 |
  | (참고) PMA_EX(다성분+co-attn) | 0.613 | 1.615 | 0.407 | 0.608 | 0.603 | 1.781 | 0.150 | 0.610 |

  **WSI를 아예 안 쓰는 M6_EX/M7_EX가 external C-index·HR·AUC 전부에서 WSI를 쓰는 모든 모델(M4_EX/M4A_EX/PM4_EX/PMA_EX)을 능가한다.** 특히 M7_EX는 6시드 전부 p<0.01(0.0000, 0.0019, 0.0087, 0.0015, 0.0022, 0.0010)이고 M6_EX도 5/6이 p<0.05 — **이 프로젝트 전체를 통틀어 가장 일관되고 통계적으로 강력한 결과**다. 반면 WSI를 쓰는 모델들은 external p가 전부 0.07~0.15로 어느 하나도 유의하지 않다.
  - **both 프로토콜과 완전히 다른 그림**: both에서는 M4A_EX(0.644)·PMA_EX(0.656)가 M6_EX/M7_EX(0.619/0.621)를 앞섰는데, external에서는 정반대로 뒤집힌다. PMA(subtype)의 both→external 붕괴가 재현된 것과 같은 패턴이 이번엔 "WSI 유무" 축에서도 나타난 것 — **both 프로토콜 순위는 이 프로젝트에서 반복적으로 신뢰할 수 없는 신호였다는 게 세 번째로 확인됨**(1번 항목 (c), 그리고 이번 (f)).
  - **결론(중요, 서사 재조정 필요)**: 지금까지 "PMA_EX가 최고 기록"이라고 불러온 건 both 프로토콜 기준이었다. **진짜 cross-institution 일반화 기준으로는 WSI+RNA+Clinical 융합 모델 중 어느 것도 RNA+Clinical만 쓰는 M7_EX를 못 넘는다.** WSI(ViT/Nystromformer 공간 컨텍스트, 다성분 pooling, co-attention 전부 포함)가 이 데이터셋·이 fusion 설계 하에서는 external 일반화에 순증분 기여를 못 하고 있다는 것이 지금까지 나온 증거 중 가장 강력하다.
  - **해석 후보**: (i) WSI 표현 자체가 여전히 노이즈가 많아 external에서 오히려 방해가 됨(재타일링으로 해소될 수도 — 진행 중인 Task 3/4와 직접 연결), (ii) 애초에 이 코호트 규모(수백 명)에서 WSI 파라미터(62만~)를 추가로 학습하기엔 표본이 작아 RNA 단독보다 과적합에 취약함, (iii) 정말로 이 fusion 설계(late concat/gate/co-attention 전부)가 WSI 신호를 제대로 못 끌어냄. 지금 증거만으로는 셋을 구분 못 함 — 재타일링(Task 3/4) 결과가 (i)를 검증하는 첫 단서가 될 것.
- **남은 다음 액션**: (a) [완료 → (f) 참조] (b) PMA_EX 추가 시드로 cptac seed84 아웃라이어 확인 — 우선순위 하향(어차피 PMA_EX가 최종 승자가 아닐 수 있음). (c) **재타일링(Task 3/4) 결과가 WSI 모델의 external 성능을 실제로 끌어올리는지가 이제 가장 중요한 다음 검증** — 만약 재타일링 후에도 M4A_EX/PMA_EX류가 M7_EX를 못 넘으면, "이 프로젝트에서 WSI가 유의미한 추가 기여를 못 한다"는 결론을 훨씬 강하게 내릴 수 있게 됨. (1000/2000개 유전자 비교는 6번 항목으로 이동.)

### 2. RNA 브랜치 유전자 선정 기준 (레퍼런스 방법론 이식)
**상태: 파이프라인 구축 + 검증 완료.** `data/select_rnaseq_genes.py` — 문헌 큐레이션 PDAC 유전자(8개 카테고리, 163개, `PDAC_LITERATURE_GENE_SETS`) + train split(both 기준) 내부 TCGA/CPTAC 각각 독립적인 univariate Cox score test + Stouffer meta-analysis(단순 결합, `meta_z = sum(z)/sqrt(2)`)로 순위 산정, 상위 1000/1500/2000개 저장. `data/dataset.py::literature_guided_gene_ids(top_n)`로 로드, `train.py --rna-genes literature_{1000,1500,2000}`로 사용(wandb에 `_EX` 접미사 자동 부착).

- **확인된 사실**: 인코더 폭만 넓힌 M6X는 M6 대비 미미한 변화(internal -0.02, external +0.02)였지만, **유전자셋 자체를 literature_1500으로 바꾸자 PMA 기준 internal이 0.583→0.656으로 크게 뛰었다**(1번 항목 참조) — "인코더 폭보다 유전자 선정 기준(어떤 유전자를 보는가)이 핵심"이었다는 가설이 사실상 확인됨.
- **다음 액션**: M6/M6X 자체도 이 유전자셋으로 재검증(1번 항목 참조). (1000/2000개 버전과의 비교는 6번 항목으로 이동.)

### 8. Nystromformer FFN 서브레이어 맛보기 ablation (M4A_FF / M2_FF)
**상태: 완료, 둘 다 소득 없음.** attention(패치 간 정보 혼합)과 FFN(패치 단위 비선형 다듬기)이 독립적 역할이라는
점에 착안해 두 변형을 `--external`(2코호트×3시드)로 검증(2026-07-17):

- **M4A_FF**(M4A_EX에서 FFN 서브레이어 제거): external C=0.610, HR=1.599, p=0.100, AUC=0.616 — **M4A_EX(C=0.611,
  HR=1.677, p=0.074, AUC=0.620)와 사실상 동일**. FFN이 전체 파라미터의 5%뿐이라(RNA 인코더가 65%) 빼봐야
  체감 효과가 없는 깨끗한 null result. FFN 제거로 과적합이 줄어들 거라는 기대는 틀렸음이 확인됨.
- **M2_FF**(M2+RNA를 FFN 직전 FiLM으로만 주입, mean pooling, 최종 결합엔 RNA 미노출): external C=0.506~0.554,
  p=0.17~0.86 — **사실상 무작위 수준**. 3/6 seed만 완료한 시점에서 프로세스 종료(cptac 3개 안 돌림) —
  나머지도 비슷할 걸로 판단, 추가 시드 의미 없음. RNA를 최종 concat에서 완전히 빼고 FFN-FiLM+mean pooling
  만으로 신호를 전달하려는 설계는, 학습된 instance 선택(ABMIL/attention)이 없는 mean pooling과 RNA의
  간접적(가산 bias) 개입이 겹쳐 신호가 거의 다 씻겨나간 것으로 보임.
- **결론**: RNA 개입 지점을 이 방향(ViT 블록 내부, FFN 전후)으로 더 파는 건 낮은 우선순위. 재타일링(Task 3/4)
  결과를 기다리는 게 맞다.

---

## 최우선 (다음 시도)

### 7. 패치 단위 서브샘플링(PatchDropout) — 과적합 완화
**상태: 완료, 단독으로는 소득 없음(9번 항목과 결합하니 효과 있음).** `train.py --patch-keep-frac`로 구현
(학습 시에만 슬라이드 패치를 비율만큼 랜덤 서브셋, val/test/external은 항상 전체 패치 사용).

- **배경**: PMA/PMA_EX 로그에서 반복적으로 train_c_index(0.75~0.86)와 val_c_index(0.6~0.73) 사이 큰 격차가 관찰됨 — 패치 단위 augmentation이 전혀 없는 게 원인 후보(ViT 문헌의 PatchDropout, Liu et al. 2022과 같은 개념).
- **결과** (`keep_frac=0.8`, `--external`, 2026-07-17): M4A_EX_SS external C=0.610(M4A_EX 0.611과 동일), PMA_EX_SS external
  C=0.599(PMA_EX 0.603과 동일, p도 0.140으로 비슷). **단독으로는 FFN 제거(8번 항목)처럼 깨끗한 null result.**
- **다음 액션**: 단독으로는 안 통했지만, RNA 보조과제(9번 항목)와 결합하니 PMA에서 유의미한 효과가 나옴 — 9번 참조.

### 9. RNA 예측 보조과제(auxiliary task) — WSI 인코더 학습 신호 보강
**상태: 완료, PMA에서 처음으로 실질적 개선 확인.** `models/rna_predictor.py::RNAPredictionHead` +
`train.py --rna-aux-weight`로 구현.

- **배경**: 지금까지(1~8번 항목) 전부 "WSI와 RNA를 추론 시점에 어떻게 결합할까"였는데, 전부 같은 좁은 밴드(external
  C 0.59~0.62)에 몰려 있었다. 진짜 병목은 결합 방식이 아니라 **WSI 브랜치가 생존 라벨(환자당 1개, censoring으로
  더 약함)만으로 62만 파라미터를 학습하다 과적합하는 것**이라는 진단(model_zoo.md) — HE2RNA(Schmauch et al. 2020)
  방식처럼 RNA 발현(환자당 1500차원, 훨씬 촘촘한 신호)을 보조 라벨로 써서 WSI 인코더를 정규화.
- **설계**: ViT 직후 RNA-free mean-pooled 표현(`meanpool_embed`, attn_pool의 RNA 개입과 별개 경로)에서 RNA를
  예측하는 작은 MLP 헤드를 붙이고, `cox_loss + rna_aux_weight(=1.0) * MSE(rna_pred, rna_true)`로 학습. 순환
  논리(이미 RNA로 물든 표현에서 RNA를 다시 맞히는 것) 방지를 위해 RNA가 전혀 개입 안 한 지점에서만 예측.
- **결과**(`patch-keep-frac=0.8` + `rna-aux-weight=1.0` 결합, `--external`, 2026-07-17):

  | | External C | HR | p | AUC |
  |---|---|---|---|---|
  | M4A_EX(기존) | 0.611 | 1.677 | 0.074 | 0.620 |
  | M4A_EX_SS_AUX | 0.612 | 1.749 | **0.048** | 0.622 |
  | PMA_EX(기존) | 0.603 | 1.781 | 0.150 | 0.610 |
  | **PMA_EX_SS_AUX** | **0.619** | 1.841 | **0.0043** | 0.623 |

  PMA_EX_SS_AUX가 지금까지 WSI 포함 모델 중 가장 안정적 — **6시드 전부 p<0.011**(0.0030/0.0003/0.0008/0.0107/
  0.0039/0.0070), M6_EX/M7_EX급 일관성. C-index 자체(+0.016)보다 **분산이 확 줄어든 게 핵심** — 과적합성
  불안정이 완화됐다는 가설(7번 항목 "예상 파급효과")과 부합. M4A는 C는 거의 그대로지만 p가 유의성 문턱(0.05)을
  넘었다.
- **주의**: 그래도 M7_EX(external C=0.634, p=0.0025)는 아직 못 넘음 — "격차를 처음으로 눈에 띄게 좁혔다" 정도.

- **(a) AUX-only(patch dropout 없이 RNA aux만) 검증 — 예상과 정반대, RNA aux는 "단독으론 오히려 해롭다"**
  (`--rna-aux-weight 1.0`만, `patch-keep-frac`은 기본 1.0=꺼짐, `--external`, 2코호트×3시드, 2026-07-18):

  | | External C (avg of 6 seeds) | 유의(p<0.05) |
  |---|---|---|
  | M4A_EX(기존, 아무 규제 없음) | 0.611 | — |
  | M4A_EX_SS(패치 드롭아웃만, 7번 항목) | 0.610 | — |
  | **M4A_EX_AUX(RNA aux만, SS 없음)** | **0.505** | 1/6 |
  | M4A_EX_SS_AUX(둘 다) | 0.612 | 6/6(단, p<0.05 기준 낮춰 재확인 시 다름, 원 표는 p<0.011) |
  | PMA_EX(기존) | 0.603 | — |
  | PMA_EX_SS(패치 드롭아웃만) | 0.599 | — |
  | **PMA_EX_AUX(RNA aux만, SS 없음)** | **0.496** | 0/6 |
  | PMA_EX_SS_AUX(둘 다) | 0.619 | 6/6 |

  처음엔 "패치 드롭아웃은 단독으론 null이었으니(7번 항목), SS_AUX의 개선은 RNA aux가 주된 기여"라고 추정했는데
  **완전히 틀렸다** — RNA aux를 패치 드롭아웃 없이 단독으로 켜면 오히려 거의 동전 던지기 수준(0.50 근처)까지
  떨어진다. 11번 항목에서 stage aux(AUX2)를 단독으로 켰을 때와 **정확히 같은 패턴**(0.49~0.51대 붕괴)이다.
  즉 RNA aux는 "그 자체로 좋은 정규화"가 아니라 **패치 드롭아웃과 결합했을 때만 도움이 되는, 그 둘의 상호작용
  효과**였다. 패치 드롭아웃 단독으로는 null(전혀 도움도 해도 안 됨)이었지만, 보조 loss(RNA 예측이든 stage
  예측이든)를 켜는 순간 패치 드롭아웃이 "그 보조 loss가 WSI 표현을 과하게 지배하지 못하게 막아주는 안전장치"
  역할을 하는 것으로 보인다 — 안전장치 없이 보조 loss 하나만 얹으면 그 loss가 원래 생존 신호와 무관한 방향으로
  표현을 끌고 가 버리는 것(둘 다 근거는 추정, 직접 검증 안 됨).
  **다만 이 안전장치도 무한하진 않다** — 11번 항목에서 SS+AUX(이미 안전한 조합)에 stage aux(AUX2)를 더
  얹었을 때는 patch dropout이 켜져 있었음에도 다시 0.49~0.53대로 무너졌다. 즉 patch dropout은 보조 loss
  "하나"는 감당하지만 "둘"은 감당 못 하는 것으로 보인다.
- **결론**: 지금 채택할 조합은 여전히 **SS_AUX(패치 드롭아웃 0.8 + RNA aux 1.0 동시 사용)뿐**이며, 이 둘은
  개별로는 무의미하거나 해롭고 반드시 함께 써야 한다는 게 명확해졌다. (b) M7_EX와의 격차가 재타일링으로
  마저 좁혀지는지 확인 — Task 3/4와 결합 실험 후보(다음 액션 유지).

### 11. T-stage/grade 보조과제(AUX2) + Clinical 병기 주입(STG) — 둘 다 독립적으로 악화, negative result
**상태: 완료, negative result (원인 진단 1차 정정됨).** `models/stage_predictor.py::StagePredictionHead`
(`--stage-aux-weight`, 9번 항목 RNA aux와 동일 설계 — meanpool_embed에서 T-stage/grade 예측, 그래디언트만
WSI 인코더로, N/M-stage는 원발암 슬라이드만으론 판단 근거 없어 제외) + `models/clinical_encoder.py::
ClinicalEncoder(use_staging=True)`(`--clinical-staging`, T/N/M/grade를 age/sex 옆에 순서형 z-score로 추가)를
M4A_EX_SS_AUX/PMA_EX_SS_AUX 베이스에 단독/결합 3가지 조합으로 얹어 `--external`(2코호트×3시드) 검증.

- **배경**: WSI로 병기(특히 grade/T-stage)를 예측하는 게 원칙적으로 불가능한 일이 아니라는 논의(원발암 WSI
  MIL로 grade 예측은 문헌에서도 흔한 타깃) 끝에, RNA aux(9번 항목)와 같은 논리로 "생존 라벨보다 촘촘한 보조
  신호"를 병기에서도 얻을 수 있는지 시험. 별개로, 우리 ClinicalEncoder가 age/sex만 쓰고 있었다는 점(레퍼런스와
  internal 격차의 유력 후보)도 병기 주입으로 직접 보완.
- **결과** (`patch-keep-frac=0.8` + `rna-aux-weight=1.0`은 항상 켜둔 채, 3가지 조합 비교):

  | | External C (avg of 6 seeds) |
  |---|---|
  | M4A_EX_SS_AUX(9번 항목, 기준) | **0.612** |
  | M4A_EX_SS_AUX_AUX2 (stage aux만) | 0.505 |
  | M4A_EX_SS_AUX_STG (진짜 병기 주입만) | 0.532 |
  | M4A_EX_SS_AUX_AUX2_STG (둘 다) | 0.533 |
  | PMA_EX_SS_AUX(9번 항목, 기준) | **0.619** |
  | PMA_EX_SS_AUX_AUX2 (stage aux만) | 0.494 |
  | PMA_EX_SS_AUX_STG (진짜 병기 주입만) | 0.527 |
  | PMA_EX_SS_AUX_AUX2_STG (둘 다) | 0.527 |

  **STG(진짜 병기를 clinical에 직접 주입) 단독으로도 AUX2와 거의 동일한 수준(0.53 근처)까지 떨어진다** —
  처음엔 "AUX2가 주범, STG가 그걸 일부 상쇄"라고 진단했으나, STG-only 결과가 AUX2_STG(둘 다)와 사실상
  같은 수치(0.532 vs 0.533, 0.527 vs 0.527)로 나오면서 그 진단은 틀렸다. 오히려 **STG 자체가 이미 대부분의
  하락을 일으키고, AUX2를 더해도 별로 더 나빠지지 않는다(AUX2 단독이 STG 단독보다 살짝 더 나쁘지만 같은
  구간)** — 두 개입이 서로 다른 원인으로 독립적으로 해를 끼치는 게 아니라, 같은 근본 원인(아래) 위에서
  둘 다 바닥을 치고 있는 것으로 보인다.
- **원인 재추정**: multi-task interference(RNA aux vs stage aux 경쟁) 가설은 STG-only도 거의 같이
  나쁘다는 사실과 안 맞는다 - STG는 loss를 추가하지 않고 입력 feature만 늘린 것이므로 "그래디언트 경쟁"
  자체가 없다. 더 설득력 있는 설명은 **표본 대비 용량 과다(overfitting)** 다 - ClinicalEncoder 입력이
  2차원(age/sex)에서 10차원(age/sex + T/N/M/grade 각각 z-score+known-flag)으로 5배 늘었는데, `--external`
  학습 표본은 코호트당 90~95명뿐이다. 게다가 이 프로젝트 전체에서 반복 확인된 패턴(RNA 브랜치가 파라미터의
  65%를 차지하며 과적합 주범이었던 것, pathway8이 정보를 오염시켜 실패한 것 등)과 같은 방향 - **표본이 이렇게
  작을 때는 새 정보를 "그냥 concat"하는 방식 자체가 정규화 없이는 거의 항상 해가 됐다**. RNA aux(9번 항목)가
  유일하게 성공한 이유도 "표현력을 늘려서"가 아니라 "그래디언트 정규화로 과적합을 줄여서"였다는 가설과 일치.
  추가로 TCGA의 M-stage는 절반 이상(78/152)이 MX(판정불가)라 known-flag 자체의 분산이 낮고, cross-institution
  external 프로토콜에서는 코호트 간 병기 분포/판정 관행 차이(batch effect)에 더 취약할 수 있다는 점도 배경
  요인으로 남겨둔다(미검증).
- **결론**: 이번 세션에서 나온 "Clinical에 병기를 직접 주입하자"(사용자가 원래 선호했던 방향)와 "WSI가
  병기를 예측하도록 보조과제로 규제하자" 둘 다, 현재의 단순 concat 설계로는 **오히려 성능을 떨어뜨린다** -
  external 기준으론 채택하지 않는다. `_STG`/`_AUX2` 플래그/코드는 남겨두되(향후 정규화를 곁들인 재시도
  가능성 대비), 지금 우선순위 갱신에는 반영 안 함.

### 10. Pathway8 집계(RNA 8개 카테고리 평균) — 실패
**상태: 완료, 명확한 negative result.** `--rna-genes pathway8`(163개 문헌 큐레이션 유전자를 8개 생물학적
카테고리 평균 z-score로 집계, SurvPath 방식 모방)를 `--external`(2코호트×3시드)로 검증.

- **결과**: M4A_PW8 external C=0.521(HR 1.173, p=0.543), PMA_PW8 external C=0.493(HR 0.999, p=0.496) —
  **6시드 전부 무의미, PMA_PW8은 사실상 동전 던지기 이하**. literature_1500(C≈0.60~0.61)보다 뚜렷하게 나쁨.
- **원인 추정**: 카테고리 내부에 생존과 반대 방향으로 작용하는 유전자가 섞여 있을 수 있음(예: immune_inflammation
  카테고리 안에 CD8A(침윤 T세포, 보통 좋은 예후)와 FOXP3(억제성 Treg, 보통 나쁜 예후)가 같이 있음) — 단순
  평균은 이런 반대 부호 신호를 서로 상쇄시켜 정보를 오히려 지워버린다. literature_1500은 Cox 순위로 유전자를
  개별 선택해 이 문제가 없었다.
- **결론**: "생물학적 도메인 지식으로 압축"이라는 아이디어 자체보다, **부호(방향성) 고려 없는 단순 평균 집계**가
  문제였을 가능성이 높다. 재시도한다면 카테고리 내부에서 Cox 방향에 따라 부호를 맞춰 평균(또는 가중합)내는 식으로
  개선해야 함 — 지금 우선순위는 아님.

### 12. 모델 용량 증대(embed_dim 64→256, num_heads 2→4, num_transformer_layers 1→2) — 실패, 과적합 재확인
**상태: 완료, 명확한 negative result.** "WSI 브랜치가 과적합이 아니라 오히려 과압축(under-capacity)돼서
신호를 못 살렸을 수도 있다"는 반대 가설을 직접 검증 — `config.py::ModelConfig`를 표준 관례 크기
(embed_dim=256, num_heads=4→head_dim=64, num_transformer_layers=2, TransMIL 스펙)로 일시적으로
키워 PMA_EX_SS_AUX를 재검증(주의: embed_dim은 WSI뿐 아니라 Clinical/RNA 인코더·light 모델까지
전부 공유하는 값이라 WSI 단독 실험은 아님).

- **결과**(tcga→cptac 방향, seed42, PMA 파라미터 82.7만→297만로 3.6배 증가): `train_c_index`가
  0.55→**0.87**까지 계속 올라가는데 `val_c_index`는 epoch 12(0.537)에서 정체된 뒤 그대로 —
  **교과서적인 과적합 곡선**. External C=0.4715(동전 던지기 이하) — 같은 방향 기존 결과(0.625~0.632대)
  대비 완전히 붕괴. 나머지 2시드는 이 시점에서 사용자 판단으로 중단(추가 확인 불필요할 만큼 명확함).
- **결론**: "과압축" 가설은 기각. 기존 진단("코호트 규모 대비 WSI 브랜치가 과적합에 취약하다")이
  오히려 더 강하게 재확인됐다. `config.py`는 원래 값(embed_dim=64/num_heads=2/num_transformer_layers=1)
  으로 즉시 롤백함 — 이 값이 임의로 정한 게 아니라 실증적으로 뒷받침되는 선택이라는 근거가 하나 더 쌓인 셈.

### 13. M7을 레퍼런스 사양(RNA 인코더 정규화+256차원, Clinical 16차원, lr=5e-5/wd=1e-3)에 맞춤 — 실패
**상태: 완료, negative result — 12번 항목과 같은 패턴이 RNA 브랜치에서도 재현됨.**
GitHub에서 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) 소스코드를 직접 확인해 M7을
`scripts/models/tabular_survival.py::RNASeqEmbedding`(LayerNorm+Dropout 입력 정규화 →
Linear(1500→256) → GELU → Dropout → Linear(256→256) → LayerNorm+GELU 출력)과 동일하게
재구현(`models/rna_encoder_extend.py`), Clinical은 레퍼런스와 동일하게 16차원 그대로 두고
(RNA에 맞춰 확장하지 않음 — 레퍼런스도 [rna=256, clinical=16] 비대칭), 학습 레시피도
레퍼런스 M7(`lr=5e-5, weight_decay=1e-3`)에 맞췄다.

- **레퍼런스 코드 직접 clone해서 우리 데이터 통제 실험(2026-07-21) — 결정적 결과, 데이터 자체가
  미묘하게 다르다는 게 최종 결론**: "데이터 문제냐 모델 문제냐"를 가르기 위해 `reference_repo/`에
  레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)를 그대로 git clone하고, 우리 재구현 코드는
  전혀 안 쓴 채 레퍼런스의 실제 `ClinicalRNASeqSurvivalModel`/`cox_ph_loss`/`harrell_c_index`를
  직접 import해서(`scripts/reference_repro_m7.py`) 우리 데이터(RNA literature_1500, Clinical
  age/sex, OS_time/event, 지금까지와 동일한 case 조인 로직)를 그대로 태웠다:

  | 프로토콜 | 결과(3시드/3 split seed 평균) |
  |---|---|
  | 우리 external(tcga→cptac), 레퍼런스 코드+우리 데이터 | 0.532/0.464/0.486 → **0.494** |
  | 레퍼런스 pooled(TCGA+CPTAC 무작위 분할, split_seed=42/84/126), 레퍼런스 코드+우리 데이터 | 0.587/0.566/0.641 → **0.598** |
  | 레퍼런스 공식 헤드라인(M7) | **0.701** |

  두 가지가 동시에 확인됨:
  1. **external 프로토콜에서 레퍼런스의 진짜 코드도 우리 재구현판(0.509)과 사실상 동일하게
     붕괴(0.494)** — 지금까지 "우리 재구현이 틀렸을 수도"라는 의심이 완전히 해소됨. 재구현
     버그가 아니라 진짜 이 아키텍처+이 데이터 조합의 특성이다.
  2. **레퍼런스 자신의 pooled 분할 방식을 정확히 재현해도(같은 split 코드, `stratify=dataset+
     event`, `test_size=0.2/0.25`, `random_state` 고정) 0.598 — 레퍼런스 헤드라인(0.701)과 여전히
     ~0.10 격차가 남는다.** "pooled vs 진짜 external"이라는 평가 프로토콜 차이는 격차의 일부
     (0.494→0.598)만 설명하고, 전부를 설명하지 못한다.
  - **결론**: 모델(레퍼런스 원본 그대로)도, 평가 프로토콜(레퍼런스 원본 그대로)도 통제했는데
    남는 격차가 있다는 건, **남은 유일한 설명은 데이터 자체의 미묘한 차이**다 — RNA-seq
    전처리/유전자 값 자체, OS 라벨 구성, 코호트를 이루는 실제 case 집합, 혹은 clinical 변수
    구성 중 어딘가가 방법론 서술은 같아 보여도 실제 수치가 다를 가능성이 높다. 지금까지
    문서/코드 비교로는 전처리 로직 자체가 일치함을 반복 확인했지만(RNA z-score, 유전자 선정,
    24개월 horizon 기준 등), **로직이 아니라 로직을 통과한 결과값 자체를 직접 대조하는 것**이
    다음으로 필요한 검증 — 이제 "무엇을 더 시도할까"가 아니라 "레퍼런스가 실제로 쓴 데이터
    산출물(가능하다면)과 우리 산출물을 케이스 단위로 직접 대조"가 남은 유일하게 결정적인
    다음 단계다.

- **부수적으로 확인된 사실 — 레퍼런스 헤드라인 숫자(M4=0.722, M7=0.701)는 코드로 직접 확인한 결과
  진짜 external이 아니다**: `M4_Train.ipynb`의 split 생성 코드가 TCGA+CPTAC을 하나로 합친
  `slide_df`에서 (dataset, event) 조합으로 stratify한 `train_test_split`을 쓴다 — 우리
  `--dataset both`(`_stratified_case_split`)와 동일한 방식. 두 기관이 train/valid/test 전부에
  섞여 들어간다. 지금까지 정황 추론이었던 걸 코드로 확정.
- **Clinical이 "더 풍부하다"는 가설도 기각**: `M4_Train.ipynb` 주석에 "Clinical 입력에서는
  pathologic_stage/T/N/M/tumor_grade를 사용하지 않습니다"라고 명시돼 있고, 실제
  `CLINICAL_FEATURE_COLUMNS = ["age_years_z", "sex_male", "sex_female"]` — age+sex(원-핫)뿐,
  우리와 정보량 동일.
- **결과** (both/external 각 3시드):

  | | 기존 M7_EX | 레퍼런스 사양 M7 |
  |---|---|---|
  | both(3시드 평균) | 0.621 | **0.596** |
  | external(tcga→cptac, 3시드 평균) | ~0.6대(유의미) | **0.509**(3/3 무의미, p=0.83/0.36/0.55) |

  둘 다 기존보다 나쁘다. 학습 곡선(external seed42)을 보면 `train_c_index`가 30 epoch 만에
  **0.97**까지 오르고 `val_c_index`는 0.60대에서 정체 — lr을 낮췄는데도(1e-3→5e-5) 명확한
  과적합. RNA 인코더가 256차원(파라미터 증가)으로 커지면서, 12번 항목에서 WSI 용량을 키웠을
  때와 **동일한 실패 패턴이 RNA 브랜치에서도 재현**됐다.
- **원인 추정**: 레퍼런스는 이 lr(5e-5)을 `epochs=100 + early stopping(patience=20)` 조합으로
  썼는데, 우리는 lr/weight_decay만 가져오고 `epochs=30 고정(early stopping 없음)`은 그대로
  뒀다 — 레시피를 일부만 이식해 내부 정합성이 깨졌을 가능성. 다만 train_c_index가 이미 0.97까지
  오른 걸 보면 "epoch이 부족해서 덜 배운 것"이 아니라 "더 큰 용량이 우리 표본에서 더 빨리
  과적합한 것"이 더 맞는 설명으로 보인다.
- **결론**: RNA 인코더 확장(rna_encoder.py → rna_encoder_extend.py)은 롤백하지 않고 코드는
  남겨두되(M7 자체는 이 negative result를 반영해 기존 RNAEncoder로 되돌릴지 다음에 결정),
  **"WSI도 128/RNA 256/Clinical 16 비율로 키우자"는 다음 계획은 보류를 강력히 권고** — WSI
  없이 RNA만 키운 M7조차 이렇게 무너졌으니, 여기에 WSI(128)까지 얹은 더 무거운 fusion
  모델은 같은 문제가 커질 뿐일 가능성이 높다.
- **risk head 재검증(2026-07-21, 후속) — 역시 negative result, 같은 패턴 재확인**: RNA
  인코더/차원/레시피는 그대로 두고(기존 성공 조합인 M7_EX, `--dataset` 기본 lr=1e-3/epochs=30,
  `--match-reference-cohort` 미사용), risk_head만 레퍼런스 사양(`LayerNorm→Dropout(0.4)→
  Linear(272→128)→GELU→Dropout(0.4)→Linear(128→1)`)으로 교체해 `--external`(2코호트×3시드)
  재검증(`models/clinical_rna_only.py`, `scripts/_m7_riskhead_ext.ps1`):

  | | External C | p<0.05 |
  |---|---|---|
  | 기존 M7_EX(선형 risk head) | 0.634 | 6/6 |
  | **M7_EX_riskhead(128차원+Dropout0.4×2)** | **0.533**(seed별: 0.5642/0.4588/0.5807/0.5543/0.5045/0.5381) | 2/6(cptac126 0.019, cptac84 0.031, tcga126은 0.054로 경계) |

  risk head를 레퍼런스처럼 깊게(128차원 은닉층) 만들면 Dropout 0.4를 두 겹 넣어도 기존 선형
  risk head보다 뚜렷하게 나빠진다 — 12번(WSI 용량)·13번 본문(RNA 인코더 폭) 항목과 정확히
  같은 패턴이 risk head 레벨에서도 재현됨. "레퍼런스처럼 깊게 만들되 Dropout으로 규제하면
  다를 것"이라는 가설은 기각 — Dropout 0.4가 이 표본 규모(코호트당 학습 90~100명 수준)에서
  128차원 은닉층이 만드는 과적합 경향을 상쇄하기에 충분하지 않은 것으로 보인다.
  **결론**: risk head는 기존 선형(LayerNorm→Linear) 형태를 유지한다 — 이 프로젝트에서
  "구조를 레퍼런스에 맞춰 깊게/넓게 만드는" 시도는 RNA 인코더, Clinical 입력, WSI 용량,
  이제 risk head까지 예외 없이 전부 실패했다.

- **M4/PMA risk head 대조 및 Dropout-only 재검증(2026-07-21) — 레퍼런스 M4≠M7 확인, 또 negative
  result**: 레퍼런스 M4(`m4_pathology_rnaseq_clinical_mil.py::classifier`)를 직접 확인해보니 M7과
  risk head 구조가 다르다는 게 드러났다 — M7은 `LayerNorm→Dropout(0.4)→Linear(→128)→GELU→
  Dropout(0.4)→Linear(128→1)`(은닉층 있음, 위에서 이미 negative result 확인)인데, **M4는
  `LayerNorm→Dropout(0.4)→Linear(→1)`로 은닉층 없이 Dropout만 있는 더 얕은 구조**다. 우리
  M4/M4A/PMA(전부 `LayerNorm→Linear(→1)`, Dropout 전혀 없음)에 레퍼런스 M4처럼 Dropout(0.4)만
  추가(파라미터 증가 없는 순수 규제, `models/vit_m4.py`·`models/vit_pma.py`)해 PMA_EX_SS_AUX
  기준으로 `--external`(tcga→cptac, 3시드) 재검증:

  | | External C(tcga→cptac, 3시드) |
  |---|---|
  | 기존 PMA_EX_SS_AUX | 0.611 / 0.630 / 0.599 (평균 0.614, 3/3 p<0.003) |
  | **PMA_EX_SS_AUX + risk head Dropout(0.4)만 추가** | **0.507 / 0.514 / 0.462 (평균 0.494, 3/3 무의미)** |

  파라미터를 하나도 늘리지 않은 순수 규제 추가인데도 거의 동전 던지기 수준까지 붕괴 — "은닉층
  때문에 실패했다"는 이전 진단(13번 항목 본문)이 불완전했음을 보여준다. **Dropout 자체가, 위치와
  무관하게 이 risk head에서 해롭다**는 게 더 정확한 결론으로 보인다 — 가설: Cox loss는 배치(risk
  set) 내 risk score들의 상대적 순서로 손실을 계산하는데, 최종 스칼라 출력 바로 직전에 Dropout
  노이즈를 주입하면 일반 분류/회귀 헤드보다 그 순서 자체가 더 민감하게 흔들릴 수 있다(미검증
  추정). `models/vit_m4.py`(M4/M4A 공유)·`models/vit_pma.py` 모두 Dropout 없는 원래 선형 형태로
  롤백함. **결론**: risk head 개입(은닉층 추가든 Dropout만 추가든)은 이 프로젝트에서 지금까지
  예외 없이 실패 — risk head는 현재의 단순 `LayerNorm→Linear` 형태를 유지한다.

- **body dropout(cfg.model.dropout) 스윕(2026-07-21) — 0.3이 날카로운 최적점, 멀티시드로 재확인됨**:
  risk head 자체의 Dropout(위 항목)과 별개로, ViT/Nystromformer·ABMIL·RNA/Clinical 인코더 전체가
  공유하는 기존 dropout rate(기본 0.3)가 이 프로젝트에서 한 번도 스윕된 적이 없었다는 점에 착안해
  `train.py --dropout`(신규 CLI 오버라이드)로 PMA_EX_SS_AUX 기준 external(tcga→cptac) 재검증:

  | dropout | seed42 | seed84 | seed126 | 평균 | 유의(p<0.05) |
  |---|---|---|---|---|---|
  | 0.2 | 0.503 | 0.520 | 0.455 | 0.493 | 0/3 |
  | **0.3(기본값)** | 0.611 | 0.630 | 0.599 | **0.614** | 3/3 |
  | 0.4 | 0.512 | 0.522 | 0.458 | 0.497 | 0/3 |

  처음 단일 시드(42)에서 0.1/0.2/0.4/0.5가 전부 붕괴하는(0.3만 튀는) 부자연스러운 패턴이 나와
  "seed 편차 아티팩트 아니냐"는 의심이 있었으나, 0.2/0.4를 seed 84/126까지 확장해도 **똑같이
  일관되게 붕괴** — 우연이 아니라 재현되는 패턴으로 확인됨. 0.3은 이 아키텍처(WSI+RNA+Clinical,
  82.7만 파라미터, 코호트당 학습 표본 90~100명)에서 매우 좁고 날카로운 최적점이라는 뜻 —
  0.1 차이만으로도 유의미한 신호가 완전히 사라진다. 이 프로젝트 전체를 관통하는 "표본 대비
  용량이 극도로 민감하다"는 서사(9/11/12/13번 항목)에 규제 강도 자체도 포함된다는 걸 보여주는
  사례.
  - **다음 액션(미실행)**: 지금 0.3이 우리 아키텍처 설계 당시 임의로 고른 값이 아니라 결과적으로
    최적점에 해당한다는 게 사후적으로 확인된 셈 — 건드리지 않고 유지한다. 다만 이렇게 날카로운
    최적점이라는 건 과적합-과소적합 경계가 매우 좁다는 신호이기도 해서, 향후 아키텍처를 조금이라도
    바꿀 때마다(risk head, 인코더 폭 등) 이 dropout 값도 함께 재튜닝이 필요할 수 있다는 점을
    염두에 둘 것.

- **코호트 크기 재검증(2026-07-19, 후속)**: "우리 CPTAC(159) vs 레퍼런스(140)"는 처음엔 큰
  격차로 보였으나, 기준을 맞춰 다시 비교하니 착시였음이 확인됨 — 우리 159는 "RNA+Clinical+OS만
  있음(WSI 무관)" 집합이고 레퍼런스 140은 "WSI+Clinical+RNA 셋 다 있음" 집합이라 애초에 다른
  모집단을 비교하고 있었다. 같은 기준(WSI까지 요구)으로 맞추면 TCGA 152(우리)/160(레퍼런스,
  8명 차), CPTAC 144(우리)/140(레퍼런스, 4명 차)로 격차가 훨씬 작고 방향도 코호트마다 다름 —
  근본적인 데이터 오류의 증거는 아님. 즉 M7 재현 실패의 원인은 여전히 "레퍼런스 헤드라인 숫자의
  검증 가능성(단일 시드 42 고정, 코드상 멀티시드 재현성 확인 불가)" 쪽에 남아있고, 코호트
  크기 자체는 용의선상에서 사실상 배제됨.

### 14. 데이터 전처리/코호트 구성 + WSI 처리 파이프라인 + risk head 전면 재점검
**상태: 착수 전 — 다음 시도로 등록.** 13번 항목에서 "코호트 크기" 자체는 큰 문제가 아님을
확인했지만, 이 프로젝트 전체를 관통하는 "레퍼런스와 설계가 사실상 같은데 성적이 안 나온다"는
의문은 아직 해소되지 않았다. 사용자 지시("데이터 쪽으론 더 파봐야겠어. 무슨 코호트를 어떻게
preprocess 했는지, WSI도 그렇고 데이터 프로세싱 과정과 risk head 쪽을 파봐야겠어", 2026-07-19)에
따라 다음 세 갈래를 점검한다:

- **코호트 구성/전처리**: `data/extract_os_labels.py`, `data/extract_rna_clinical.py` 등 케이스
  선정·조인 로직을 레퍼런스의 `M4_Train.ipynb`/`data_preprocessing.ipynb`와 한 줄 단위로 재대조.
  13번 항목에서 이미 OS 라벨/RNA 전처리/유전자 선정은 대체로 일치함을 확인했으나, 아직 안 본
  지점(예: CPTAC WSI "series type" 필터 — Method.md가 언급하는 "HE tumor series 우선 사용"을
  우리가 적용하고 있는지, `data/preprocess_cptac.py`/`slide_index_task*.csv` 확인 필요)이 남아있음.
- **WSI 처리 파이프라인**: 타일링(4번 항목에서 이미 재검증·기각됨)뿐 아니라, feature 추출
  backbone 자체의 전처리(정규화 상수, color jitter 유무), 타일 필터링 기준(조직 비율 threshold
  등)이 레퍼런스와 어떻게 다른지 재점검.
- **risk head**: 지금까지 fusion 이후 최종 risk head(Cox 예측 MLP) 자체의 구조·정규화는
  깊게 파고든 적이 없음 — 레퍼런스의 `ClinicalRNASeqSurvivalModel` 최종 헤드(LayerNorm→Dropout
  0.4→Linear→GELU→Dropout 0.4→Linear)와 우리 risk head 설계를 직접 대조.
- **다음 액션**: 위 세 갈래를 순서대로 점검하고, 발견 사항을 이 항목에 이어서 기록한다.

- **risk head 대조(2026-07-21) — negative result**: 위 13번 항목에 이미 기록. 선형(LayerNorm→
  Linear) → 레퍼런스식 128차원 은닉층+GELU+Dropout0.4×2로 교체했으나 external C 0.634→0.533로
  악화, 원복함.

- **WSI feature 추출 전처리 대조(2026-07-21) — 일치, 차이 없음**: 레퍼런스 `scripts/
  preprocess_wsi_tiles.py`를 GitHub에서 직접 확인 — `tissue_threshold=0.15`, `tile_size=1024`,
  `target_mpp=1.0`, ImageNet 정규화(mean/std 표준값) 전부 우리 `config.py`/`data/preprocess.py`
  (`TISSUE_RATIO_THRESH=0.15`, `TILE_SIZE=1024`, `TARGET_MPP=1.0`, `data/patch_utils.py::
  PATCH_TRANSFORM`)와 정확히 일치. 이 갈래는 차이 없음 — 4번 항목(재타일링 실험)에서 이미
  다뤄진 해상도 이슈와는 별개로, 원 파이프라인의 타일링/정규화 파라미터 자체는 레퍼런스와
  처음부터 동일했다는 게 재확인됨.

- **WSI 슬라이드 선정(환자당 몇 장을 쓰는가) 대조(2026-07-21) — 큰 차이 발견, 유력한 새 단서**:
  `data_verification.ipynb`(cell 11, 13)와 `scripts/preprocess_wsi_tiles.py::tile_cptac_case()`
  코드로 직접 확인:
  - **TCGA**: 레퍼런스는 `tcga_paad_matched_patient_table_dx_one_per_patient.csv` — 처음부터
    "환자당 diagnostic WSI 1개"로 사전 선별된 테이블을 사용(전체 WSI 테이블 `..._dx_all_files.csv`도
    따로 존재하지만 모델에는 안 씀).
  - **CPTAC**: `SeriesDescription`에 `"tumor"`(대소문자 무관)가 포함된 series만 후보로 추리고,
    그중 `series_size_MB`가 가장 큰 series 하나만 선택(`tumor_series.sort_values("series_size_MB",
    ascending=False).iloc[0]`) — case당 정확히 1개 슬라이드.
  - **우리**: TCGA/CPTAC 둘 다 `data/preprocess.py::_build_slide_list()`가 `ROOT.glob("*.svs")`로
    디렉터리 안의 svs 파일을 series/tumor 여부 구분 없이 전부 가져온다. 실측(`slide_index_task*.csv`
    기준): **TCGA 평균 2.52장/case(최대 9장), CPTAC 평균 3.22장/case(최대 8장)** — 이 여러 장을
    WSISurvivalDataset이 환자 단위로 그대로 풀링한다(`models/vit_m1.py` 등 다수 모델 docstring에
    이미 명시된 기존 설계).
  - **의의**: 지금까지(12/13번 항목) 실패한 "레퍼런스처럼 용량/깊이를 늘리자"는 시도들과는 성격이
    다르다 — 이건 모델 용량이 아니라 **입력 데이터 큐레이션**의 차이다. 우리는 tumor 여부·대표성이
    검증 안 된 슬라이드(같은 환자의 여러 조직 블록, 개중엔 non-tumor/저품질 스캔이 섞여 있을 가능성)를
    구분 없이 섞어 쓰고 있고, 레퍼런스는 "tumor" 라벨이 붙은 가장 큰(=가장 많은 조직을 담은) 슬라이드
    하나만 엄선해서 쓴다. 지금까지 WSI가 external에서 순증분 기여를 못 한다는 결론(1번 항목 (f))의
    새로운 유력한 설명 후보 — 노이즈가 많은 다중 슬라이드 풀링이 신호를 희석시켰을 가능성.
  - **GDC biospecimen API로 CPTAC tumor/normal 실측 라벨 확보(2026-07-21) — CPTAC도 진짜 tumor
    필터가 가능해짐**: TCIA의 CPTAC-PDA SVS에는 레퍼런스가 쓴 DICOM `SeriesDescription` 태그가
    없지만(위 항목 참조), GDC(api.gdc.cancer.gov) biospecimen 계층(`cases -> samples -> portions ->
    slides`, project `CPTAC-3`)을 직접 조회하면 slide 단위 `sample_type`(Primary Tumor / Solid
    Tissue Normal)과 병리 QC 지표(`percent_tumor_nuclei`, `percent_necrosis`)까지 나온다. CPTAC-3/
    Pancreas 170개 case 전체를 조회해 `data/cptac_gdc_slide_sample_type.csv`에 저장:
    - 우리가 다운로드한 CPTAC svs 567장 중 **295장(52%)이 GDC biospecimen 기록과 매칭**되고,
      그중 **215장(73%)이 Primary Tumor, 80장(27%)이 Solid Tissue Normal**(정상 조직) — 지금까지
      tumor/normal 구분 없이 case당 슬라이드를 전부 써온 기존 방식이 **정상 조직 슬라이드까지
      섞어 학습에 썼다는 걸 실측으로 확인**. 나머지 272장(48%)은 GDC에 slide 단위로 등록 안 됨
      (TCIA Pathology Portal에는 있지만 GDC biospecimen에는 formal entity로 안 올라간 경우로 추정,
      tumor 여부 미상으로 남김).
    - case 단위로 보면 **178개 WSI-보유 케이스 중 148개(83%)가 GDC로 확인된 Primary Tumor
      슬라이드를 최소 1장 보유** — 이 148개는 진짜 tumor 슬라이드를 대표로 선택할 수 있다.
    - `data/dataset.py::_select_representative_slide()`를 이 매핑을 쓰도록 업그레이드(순수
      "조직량 최대" 프록시 대신) — CPTAC은 Solid Tissue Normal로 확인된 슬라이드는 항상 후보에서
      제외하고, Primary Tumor로 확인된 슬라이드가 있으면 그중에서만 고른다(GDC 미등록 슬라이드만
      있는 나머지 case는 "정상이 아님" 조건만 적용한 채 조직량 최대로 폴백). 검증 결과 대표 슬라이드
      148건이 전부 Primary Tumor로 확정 선택됐고, Solid Tissue Normal이 대표로 뽑힌 경우는 0건.
  - **M4A_EX_SS_AUX_1SLIDE 결과(2026-07-21) — negative result**: `--one-slide-per-case`(TCGA:
    DX 우선, CPTAC: GDC 확인 tumor 우선)를 M4A_EX_SS_AUX에 적용해 `--external`(tcga→cptac 단일
    방향, 3시드)로 재검증(이 세션의 표준 관례대로 external은 tcga train → cptac test 방향만 — 원래
    스크립트가 실수로 양방향을 돌고 있어서 cptac→tcga 방향은 도중에 중단하고 제외함):

    | | External C(tcga→cptac, 3시드) |
    |---|---|
    | 기존 M4A_EX_SS_AUX | 0.581 / 0.609 / 0.611 (평균 0.600) |
    | **M4A_EX_SS_AUX_1SLIDE** | **0.491 / 0.507 / 0.549 (평균 0.516)** |

    세 시드 전부 악화됐고(개별 비교 0.581→0.491, 0.609→0.507, 0.611→0.549), 2/3 시드는 p=0.91/0.99로
    사실상 무의미한 수준까지 떨어졌다. **정상 조직 슬라이드를 제거하고 대표 슬라이드 1장만 쓰는
    큐레이션이 오히려 손해였다** — 가능한 해석: (i) 여러 슬라이드를 풀링하는 게 실제로는 노이즈
    상쇄/표본 증강 역할을 해왔을 수 있다(슬라이드 하나의 국소적 편차를 여러 장 평균이 완충), (ii)
    대표 슬라이드 선택 기준(TCGA: DX+최대 조직량, CPTAC: GDC tumor 확인+최대 조직량)이 실제 예후
    신호와 무관한 슬라이드를 고를 수도 있다(조직량이 많다고 예후에 유리한 형태가 담겨 있다는
    보장은 없음), (iii) 슬라이드 수 자체가 patient-level 표본 크기의 일부처럼 작동해(각 슬라이드가
    ABMIL/co-attention을 한 번씩 더 통과) 정보량이 줄어든 것일 수 있다. 다음으로 PMA_EX_SS_AUX_
    1SLIDE(같은 실험을 external 최고 기록 모델로 재확인)까지 부정적이면, 이 접근 자체를 기각.
  - **사용자 가설(2026-07-21) — M4A의 실패는 구조 특이적일 수 있다는 가설, PMA 결과로 기각됨**:
    M4A(co-attention)는 여러 슬라이드/패치 전체를 놓고 "어디에 주목할지"를 고르는 MIL을 전제로
    설계돼 슬라이드를 1장으로 줄이면 attention 모집단이 줄어 오히려 과적합에 더 취약해질 수
    있다는 논리 — 반면 PMA는 multi-component pooling(mean/std/attn/top-10%)의 결과를 그대로 최종
    표현으로 쓰는 구조라 이 문제에서 상대적으로 자유로울 거라 예상했다. **결과는 정반대**:

    | | External C(tcga→cptac, 3시드) |
    |---|---|
    | 기존 PMA_EX_SS_AUX | 0.611 / 0.630 / 0.599 (평균 0.614, 3/3 p<0.003) |
    | **PMA_EX_SS_AUX_1SLIDE** | **0.490 / 0.506 / 0.469 (평균 0.488, 3/3 유의하지 않음)** |
    | (참고) M4A_EX_SS_AUX_1SLIDE | 평균 0.516 |

    PMA가 M4A보다 오히려 더 크게 무너졌다(0.614→0.488, -0.126 vs M4A의 -0.084) — "이중 co-attention
    과적합" 가설은 기각. 두 서로 다른 구조(co-attention 단일 벡터 vs multi-component pooling)가
    똑같이 나쁜 방향으로 무너졌다는 건, 원인이 **모델 구조가 아니라 데이터 큐레이션 개입 자체**에
    있다는 쪽에 더 힘을 싣는다 — (i) 여러 슬라이드 풀링이 실제로는 노이즈 상쇄/표본 증강 역할을
    해왔을 가능성, (ii) "대표 슬라이드 선정 기준"(TCGA: DX+최대 조직량, CPTAC: GDC tumor 확인+최대
    조직량) 자체가 예후 신호와 무관했을 가능성.

  - **완화 절충안(`--exclude-normal-slides`) 결과(2026-07-21) — 이것도 negative, "약간의 변화도
    치명적"이라는 패턴 재확인**: 대표 1장으로 줄이는 대신, 확인된 정상 조직 슬라이드만 제외하고
    케이스당 나머지는 그대로 두는 훨씬 덜 급진적인 옵션(`data/dataset.py::_exclude_normal_slides`,
    `train.py --exclude-normal-slides`)을 구현 — TCGA 평균 슬라이드/case 2.52→2.28(단 44장만
    제외), CPTAC 3.22→2.76. PMA_EX_SS_AUX 기준 `--external`(tcga→cptac, 3시드) 재검증:

    | | External C(tcga→cptac, 3시드) |
    |---|---|
    | 기존 PMA_EX_SS_AUX | 0.611 / 0.630 / 0.599 (평균 0.614, 3/3 p<0.003) |
    | **PMA_EX_SS_AUX_NONORMAL** | **0.507 / 0.516 / 0.458 (평균 0.494, 3/3 무의미)** |

    TCGA 학습 세트는 겨우 44장(2.52→2.28)만 줄었는데도 1장으로 축소한 실험(평균 0.516)과 거의
    동일한 수준으로 붕괴했다 — 개입의 급진성과 붕괴 정도가 비례하지 않는다. "슬라이드 구성을
    조금이라도 건드리면 이 모델이 극도로 취약하게 반응한다"는 게 1SLIDE·NONORMAL 두 실험에서
    공통으로 확인됨 — SS_AUX의 두 정규화 장치(patch dropout 0.8, RNA aux 1.0)가 *이 특정* 슬라이드
    구성에 맞춰 미세 조정된 상태였고, 그 전제 중 하나만 바뀌어도 전체 균형이 깨지는 것으로 보인다
    (가설, 직접 검증 안 됨 — patch-keep-frac/rna-aux-weight 재튜닝과 결합하면 다를 수 있음).

  - **최종 결론**: 슬라이드 구성 관련 개입(1SLIDE, NONORMAL) 둘 다 명확한 negative result —
    이 방향은 기각하고 기존(슬라이드 전부 사용) 방식을 유지한다. 14번 항목의 "WSI 슬라이드 선정"
    갈래는 여기서 마무리.

---

### 15. M7(RNA+Clinical)이 M6(RNA만)를 못 넘는 문제 — clinical 브랜치 내부를 세 방향으로 고쳐봤지만 전부 negative result

`train_light.py --M7`(RNA=64/Clinical=4, `--dataset tcga --external`, seed42 고정, 단일 시드)로
레퍼런스(tabular_survival.py::ClinicalEmbedding)와 우리 `ClinicalEncoder`(models/clinical_encoder.py)의
구조 차이 두 가지를 직접 이식해 검증했다:

- **clinical 브랜치 내부 dropout(0.25) 추가**: 레퍼런스엔 있고 우리에겐 없던 것. external C
  0.6385→0.6179로 악화(AUC 0.6750→0.6430도 동반 악화). HR/p는 개선(1.570→1.746, 0.0157→0.0031)됐지만
  주 지표인 c-index 기준 negative.
- **sex를 스칼라 이진 인덱스(0/1) 대신 one-hot(2차원, 레퍼런스 clinical_dim=3 방식)으로 인코딩**:
  external C 0.6385→0.6370, AUC 0.6750→0.6756 — 사실상 무차이(이미 확인된 seed 간 자연 변동폭
  ±0.04~0.05 이내). 이론적으로는 one-hot이 유효 파라미터 하나(스칼라 가중치)를 비식별 파라미터
  둘(w_male/w_female)로 표현해 초기화 분산·Adam 최적화 경로가 달라질 수 있다는 가설을 세웠지만,
  단일 seed 점추정으로는 차이가 드러나지 않았다(여러 seed로 분산 자체를 비교해야 진짜 검증됨,
  아직 안 함).

같은 조건(seed42, tcga→cptac)에서 M5(ClinicalOnly, age/sex만)와 M6(RNAOnly)도 재검증했다:

| 모델 | external C | external HR | external p | external AUC |
|---|---|---|---|---|
| M5 (ClinicalOnly, age/sex만) | 0.5369 | 1.058 [0.733, 1.527] | 0.7540(무의미) | 0.5412 |
| **M6 (RNAOnly)** | **0.6468** | **2.477 [1.687, 3.637]** | **0.0000** | **0.7074** |
| M7_EX (RNA=64/Clin=4, 기존) | 0.6385 | 1.570 [1.086, 2.270] | 0.0157 | 0.6750 |
| M7_EX + clinical-dropout 0.25 | 0.6179 | 1.746 [1.201, 2.539] | 0.0031 | 0.6430 |
| M7_EX + sex one-hot | 0.6370 | 1.808 [1.245, 2.625] | 0.0016 | 0.6756 |

**M5(age/sex만)는 external에서 사실상 무의미(p=0.754, HR≈1)** — 이전 세션에서 "M5가 전 모델을
이긴다"고 봤던 관찰은 아마 `--dataset both`(같은-코호트) 프로토콜에서 나온 것으로 보이며, 지금
프로토콜(seed42/`--external`, 진짜 cross-institution)에서는 재현되지 않는다. **반면 M6(RNA만)는
c-index·HR·p·AUC 전 지표에서 M7_EX 세 변형 전부를 앞선다.**

**결론(가설)**: clinical(age/sex) 정보 자체가 이 코호트에서 external 기준으로 통계적으로 무의미한
수준(M5: p=0.754)인데, 이를 RNA와 concat해 하나의 작은 risk_head(`LayerNorm→Linear(rna_dim+
clinical_dim→1)`)로 공동 학습시키면, 그 risk_head는 훈련 세트(~90명)에서 clinical 가중치를
0이 아닌 값으로 피팅하게 된다 — 신호가 없으니 이 가중치는 사실상 훈련 세트 노이즈에 대한
과적합이고, held-out(특히 external)에서는 오히려 risk 순위를 흐트러뜨리는 역할을 한다(c-index는
순위 기반 지표라 이런 노이즈에 특히 민감).

**후속 검증(2026-07-29) — clinical의 자유도를 이론적 최솟값까지 줄여도 소용없음**:
concat이 문제라면 "clinical이 개입할 수 있는 파라미터 수를 극단적으로 줄이면 해결될 것"이라는
가설로 `models/clinical_rna_only.py::ClinicalRNAOnly`에 `combine_mode` 옵션 두 개를 추가해
같은 조건(RNA=64, seed42, tcga→cptac)으로 검증했다:

- **film**: clinical(age_z, sex)에서 스칼라 γ,β(항등 근처 초기화, 파라미터 6개)만 만들어
  `z_rna' = γ·z_rna + β`로 RNA 임베딩 전체를 균일 스케일/이동만 시킴.
- **cox_add**: 고전적 Cox 가산항처럼 `risk = risk_head(z_rna) + β_age·age_z + β_sex·sex_idx`로
  최종 risk 스칼라에 직접 더함(파라미터 딱 2개, 세 방식 중 최소).

| combine_mode | 파라미터 수(clinical 쪽) | external C | external HR | external p | external AUC |
|---|---|---|---|---|---|
| concat(기존) | ~수백(MLP 포함) | **0.6385** | 1.570 | 0.0157 | 0.6750 |
| concat + sex one-hot | ~수백 | 0.6370 | 1.808 | 0.0016 | 0.6756 |
| **cox_add** | **2** | 0.6304 | 1.933 | 0.0004 | 0.6864 |
| concat + dropout 0.25 | ~수백 | 0.6228→0.6179* | — | — | — |
| **film** | **6** | 0.6228 | 1.693 | 0.0049 | 0.6744 |
| M6(RNA만, clinical 없음) | 0 | **0.6468** | **2.477** | **0.0000** | **0.7074** |

(*dropout 행은 위 표와 별개 실험, 참고용으로만 병기)

**가설 기각, 그것도 역방향으로**: 자유도를 줄일수록 오히려 나빠졌다(concat 0.6385 > cox_add
0.6304 > film 0.6228) — "파라미터를 줄이면 노이즈 과적합이 줄어 M6에 가까워질 것"이라는 예측과
정반대다. 그럴듯한 이유: concat은 clinical의 가중치가 RNA의 64개 차원과 함께 하나의
LayerNorm+Linear로 희석되며 반영되는 반면, film은 γ·β가 z_rna **전체를 곱셈으로 직접 왜곡**하고
(노이즈가 있으면 RNA의 64차원 전부에 비례해서 퍼짐), cox_add는 단 2개 파라미터가 최종 risk
스칼라에 **감쇠 없이 직결**된다 — 파라미터 수는 적어도 그 각각이 손실에 미치는 "직접성"이 커서,
오히려 더 적은 데이터로 더 공격적으로 피팅됐을 수 있다(가설, 추가 검증 안 됨).

**추가 검증(2026-07-29, 3차) — risk_head 자체를 레퍼런스처럼 깊게(히든 레이어 추가), 단 비율은
축소**: 13번 항목에서 레퍼런스 `classifier`(LayerNorm→Dropout(0.4)→Linear→GELU→Dropout(0.4)→
Linear, hidden=128) 사양을 절대 폭 그대로(rna=256/clinical=16) 이식했을 땐 붕괴(0.634→0.533)
했었는데, 그건 폭 자체가 이 프로젝트 다른 모델과 안 맞았을 가능성이 있어 지금 우리 폭(rna=64/
clinical=4)에 맞춰 hidden도 같은 비율로 4배 축소(128/4=32)하고 dropout도 0.4→0.3으로 낮춰
재검증(`--risk-hidden-dim 32 --risk-dropout 0.3`):

| risk_head 구조 | external C | external HR | external p | external AUC |
|---|---|---|---|---|
| 레퍼런스 원본 비율(hidden=128, rna=256/clin=16, dropout=0.4) | 0.5330(13번 항목) | — | — | — |
| **hidden=32(4배 축소), dropout=0.3, rna=64/clin=4** | **0.6340** | 2.016 [1.390, 2.925] | 0.0002 | 0.6701 |
| 히든 없음(기존 concat, rna=64/clin=4) | 0.6385 | 1.570 | 0.0157 | 0.6750 |
| M6(RNA만) | **0.6468** | 2.477 | 0.0000 | 0.7074 |

비율을 맞춰 축소하니 레퍼런스 원본 비율(0.533)보다는 훨씬 나아졌지만(+0.101), 여전히 히든 없는
기존 concat(0.6385)보다 낮고 M6(0.6468)는 당연히 못 넘었다 — risk_head를 깊게 만드는 것도
순증분 이득이 없다는 뜻.

**최종 결론**: dropout 유무, sex 인코딩 방식, clinical 차원(16/4), 결합 방식(concat/film/
cox_add, 파라미터 수백↔2), risk_head 깊이(히든 없음 vs 32) — clinical·결합부에서 시도해볼 수
있는 거의 모든 축을 바꿔봤지만 **단 하나도 M6(RNA 단독)를 넘지 못했다.** 이 정도로 다방면에서
일관되게 실패하면 "clinical을 넣는 방법이 잘못됐다"보다 **"이 코호트에서 clinical(age/sex)은
어떤 형태로 넣어도 RNA 대비 순증분 가치가 없다"**는 결론이 더 타당하다 — M6를 최종 baseline으로
채택하고 이 실험 갈래는 종료.

### 3. 학습 하이퍼파라미터 검증 (light + WSI 모델 모두)
**상태: 본격 스윕은 보류 — 아키텍처가 확정되지 않은 지금 시점엔 하지 않기로 결정(2026-07-17). 아키텍처 최종
확정 후 다른 하이퍼파라미터와 함께 Ray Tune 등으로 한 번에 다루는 게 맞다는 판단.** 다만 baseline(M7_EX)을
lr=1e-5(WSI 모델과 동일)로 재검증해 "더 단단하게" 만들 수 있는지 스모크 테스트는 해봄 — 아래 참조.

- **WSI-free 모델(M5/M6/M6X/M7)**: `train_light.py`(`LightTrainConfig`, lr=1e-3)가 아직 제대로 검증된 값은
  아님 — 스모크 테스트에서 lr=1e-3이 M6를 train_c_index 0.99까지 과적합시키는 것도 확인했다.
- **`--lr` 오버라이드 추가 + M7_EX를 lr=1e-5로 재검증 시도 — 실패, lr=1e-3 유지가 정답이었음**
  (`train_light.py --lr`, 2026-07-17): M7_EX(literature_1500, `--external`, cptac seed42)를 WSI 모델과
  동일한 lr=1e-5로 30 epoch 돌려본 결과, **train_c_index가 0.501→0.503으로 30 epoch 내내 거의 그대로였다**
  (사실상 학습이 안 됨) — external C-index도 0.442로, 기존 lr=1e-3 baseline(0.634)보다 훨씬 나쁨. light
  모델은 파라미터가 훨씬 적은 작은 MLP라 warmup+cosine decay가 고정된 30 epoch 안에서는 lr을 100배 낮추면
  수렴 자체가 안 된다 — WSI 모델의 lr을 그대로 가져다 쓴 전제가 틀렸음을 확인. **결론: 지금 갖고 있는
  M7_EX/M6_EX(lr=1e-3) 수치가 여전히 가장 신뢰할 수 있는 baseline이며, 이걸 건드리지 않는다.** lr=1e-3이
  M6를 과적합시킨다는 관찰과 lr=1e-5가 아예 학습을 못 시킨다는 관찰을 종합하면, 진짜 최적값은 그 사이
  어딘가(예: 3e-4)일 가능성이 높다 — 나중에 본격 스윕 시 탐색 범위를 좁히는 참고 자료로 남겨둔다.
- **WSI 포함 모델(M1/M4/.../PM4/PMA)**: `config.py::TrainConfig.lr=1e-5`도 재검토 대상. Backbone은 어차피 얼려있어 ViT/pooling/risk_head는 light 모델과 마찬가지로 처음부터 학습되는데, 왜 1e-5로 이례적으로 낮게 잡았는지는 불명. 다만 ViT self-attention(Nystromformer) 블록이 껴 있어 light 모델보다 LR에 민감할 수 있다(gradient clipping·warmup이 이미 있어 어느 정도 안전판은 있음).
- **다음 액션(보류)**: 아키텍처가 확정된 뒤(재타일링 결과 반영 후), M1(WSI 계열 대표)·M6(light 계열 대표)
  하나씩으로 다른 하이퍼파라미터(weight_decay, epochs 등)와 함께 Ray Tune 스윕. 지금은 진행하지 않음.

---

## 오늘 밤에 돌릴 것

### 4. WSI 타일 해상도/물리 스케일 미스매치
**상태: 완료, 가설 기각(negative result) — 재타일링이 오히려 더 악화시켰다.**

- **사실관계**: 기존 타일은 1024×1024px @ 1.0 MPP, 리사이즈 없이 backbone 투입. Lunit SwAV는 512×512px @
  0.5MPP(20배율)/0.25MPP(40배율) 사전학습 — 픽셀당 해상도 2~4배, 타일당 물리 면적 16배 차이. UNI도 이미 확인된
  대로 ~0.5MPP 학습 분포에서 4배 이상 어긋남.
- **가설**: ResNet50/UNI 두 backbone 모두에 걸리는 문제라, "어떤 인코더냐"보다 "타일링 컨벤션 자체"가 external
  일반화 실패(1번 항목 (f))의 근본 원인일 수 있다 — 재타일링(512px@0.5MPP, backbone 사전학습 스펙에 정확히
  맞춤)으로 해소되는지 검증.
- **실행**: TCGA/CPTAC 전체를 512px@0.5MPP로 재타일링 + feature 재추출(`data/patches_{tcga,cptac}_512`,
  2026-07-18~19) — 466/567 슬라이드 전부 성공(TCGA 466/466, CPTAC 564/567 + 기존에 알려진 손상 파일 3개
  스킵). 이 데이터로 지금까지 external 최고 기록인 M4A_EX_SS_AUX/PMA_EX_SS_AUX를 3시드×tcga/cptac 양방향
  (24 runs)으로 재검증.
- **결과 — 가설과 정반대, 재타일링이 오히려 큰 폭으로 악화시켰다**:

  | | External C (기존 타일링) | External C (재타일링 512px@0.5MPP) |
  |---|---|---|
  | M4A_EX_SS_AUX | 0.612 | **0.514** |
  | PMA_EX_SS_AUX | 0.619 | **0.490** |

  둘 다 9번/11번 항목에서 봤던 "보조 loss 단독 붕괴"와 같은 구간(0.49~0.52)까지 떨어졌다 — 유의미한 시드도
  M4A 1/6, PMA 0/6뿐. Backbone 사전학습 해상도에 정확히 맞췄는데도(픽셀당 해상도·타일당 물리 면적 모두 일치)
  성능이 개선은커녕 거의 동전 던지기 수준으로 후퇴했다.
- **원인 추정**: (i) 슬라이드당 타일 수가 평균 1804개(기존 대비 수 배)로 폭증하면서, `--patch-keep-frac 0.8`·
  `Nystromformer(num_landmarks=128)` 등 기존에 검증된 하이퍼파라미터가 이 표본 밀도에 더는 맞지 않을 가능성
  (Nystrom 근사가 이제 거의 모든 슬라이드에서 발동 — 기존엔 16.6%뿐이었음). (ii) 타일 하나당 물리적 시야
  (256μm×256μm)가 기존(1024μm×1024μm)보다 16배 좁아져, 예후와 관련된 더 큰 스케일의 조직 구조/성장 패턴
  정보가 오히려 타일 단위에서 유실됐을 가능성 — "backbone 사전학습 해상도에 맞추는 것"과 "예후 예측에 필요한
  형태학적 스케일"이 서로 다른 요구일 수 있다는 뜻. 어느 쪽이 주된 원인인지는 미검증.
- **결론**: "타일링 해상도 미스매치가 WSI external 실패의 근본 원인"이라는 가설은 기각됐다. 재타일링 인프라
  자체(`data/patches_{tcga,cptac}_512`)는 남겨두되(향후 patch-keep-frac/landmark 재튜닝 등과 결합해 재시도할
  여지는 있음), 지금 우선순위 갱신에는 반영하지 않는다 — 1번 항목 (f)의 "WSI가 이 fusion 설계로는 external에
  순증분 기여를 못 한다"는 결론이 재타일링으로 해소되지 않았다는 점만 확정.

---

## 나중에 (논문 작성 시 — 지금은 돌파구 탐색 단계라 우선순위 아님)

### 6. 정당화·해석용 ablation (성능 개선 목적 아님)
지금 필요한 건 "뭐가 성능을 올리는가"를 찾는 것이지 "이미 잘 나온 설계를 왜 이렇게 했는지 정당화"하는 게 아니다. 아래 둘은 논문/보고서 작성 단계에서 방법론 정당화용으로 돌리면 되고, 지금 시점에는 결과에 영향 없다.

- **1000/2000 유전자 버전과 literature_1500 비교**: 1500이 최적 개수인지 확인(1번/2번 항목에서 이동). 이미 1500이 상당히 잘 나온 터라 상한 폭이 크지 않을 것으로 예상 — "확인" 성격.
- **mean/std/attn/top-10% 개별 ablation**: `MultiComponentPooling`의 4개 관점을 레퍼런스 Morphology Burden Pooling에서 그대로 이식했을 뿐, 우리가 하나씩 빼보고 개별 기여도를 검증한 적은 없다. 성능이 이미 잘 나오는 상황이라 지금 우선순위는 낮음.

---

## 미정 (현재 해결책 없음)

### 5. 도메인/배치 효과 오염의 근본 원인
raw CNN feature로 TCGA/CPTAC를 구분하는 도메인 분류기 AUC=0.78(강한 배치 효과). Stain normalization(Macenko)으로 시도했으나 AUC 0.803으로 오히려 소폭 악화 — 이 경로는 종료.

- 색상 보정으로 안 풀린다는 건, 원인이 스캐너 해상도/PSF, JPEG 압축, 조직 절편 두께, 디지털화 파이프라인 차이 등 더 근본적인 곳에 있을 가능성을 시사하지만, 이건 SOTA 모델들도 만성적으로 겪는 문제다 — 지금 수준에서 깔끔한 해결책이 없다. **재타일링(4번 항목)이 스캐너별 해상도 차이를 부수적으로 완화할 가능성은 있으니, 그 결과가 나오면 도메인 AUC를 다시 재보는 정도로만 연결**하고, 별도의 적극적인 해결 시도는 지금 보류한다.
- **실시간 tile augmentation(ColorJitter+GaussianBlur+Flip)도 동일하게 안 풀림(2026-07-22)**: 레퍼런스
  M4 pooled에서 augmentation이 성능을 끌어올린 것(0.674→0.711, 아래 "레퍼런스 M4 실시간 augmentation"
  항목)이 "도메인 격차를 줄여서"인지 확인하려고, `utils/extract_features_augmented.py`로 TCGA+CPTAC
  둘 다 augmented feature(단일 실현, seed=42)를 뽑고 `check_domain_shift.py --use-augmented`로
  재검사했다:

  | feature | 도메인 분류기 AUC |
  |---|---|
  | 원본(features.pt) | 0.8082 |
  | augmented(features_aug.pt) | **0.8255**(오히려 악화) |

  Stain normalization과 완전히 같은 패턴 — **색상/블러 계열 개입(정규화든 augmentation이든)으로는
  도메인 분리가 전혀 안 줄고 오히려 근소하게 늘어난다.** 이 도메인 신호가 색상 공간이 아니라
  스캐너 해상도/PSF/압축 등 더 근본적인 곳에 있다는 가설을 다시 한번 뒷받침한다. **결론: 레퍼런스
  M4의 augmentation 효과(pooled C 상승)는 도메인 격차 축소가 아니라 일반적인 정규화(과적합 완화)
  효과일 가능성이 높다** — 이 가설은 external(진짜 cross-institution) 프로토콜에서 같은 개선폭이
  재현되는지로 추가 검증이 필요하다(진행 중).

---

## 이미 종료된 경로 (참고용, 재시도 불필요)

- **Stain normalization**: AUC 0.78→0.803, 개선 없음.
- **UNI backbone 단순 교체(리사이즈만, 재타일링 없이)**: 224/512 리사이즈 둘 다 ResNet50 대비 이득 없음(오히려 근소 열세). 4번 항목의 진짜 해결(재타일링) 없이는 재시도 의미 없음.
  - **재확인(2026-07-22, RNA 수정·WSI 브랜치 진단 이후)**: frozen ResNet50(Lunit SwAV) backbone이
    2,350만 파라미터인데 우리가 학습해온 WSI 브랜치 전체(proj+ViT/Nystromformer+pooling+
    co-attention)는 21.9만 파라미터뿐 — backbone이 107배 더 크다. "지금까지 건드린 건 꼬리뿐,
    실제 feature를 만드는 몸통은 한 번도 안 바꿔봤다"는 문제의식으로, 약 3억 파라미터급인 UNI
    (ViT-L/16)로 PMA_EX_SS_AUX(seed42, external)를 다시 검증:

    | | internal test C | external C |
    |---|---|---|
    | ResNet50(Lunit SwAV, 기존) | 0.6309 | 0.6289 |
    | **UNI(ViT-L/16)** | 0.6459(+0.015) | **0.6179**(-0.011) |

    둘 다 노이즈 범위 안 — **backbone을 100배 이상 키워도 external은 오히려 근소하게 나빠진다.**
    RNA 버그 수정 이전에 나온 "UNI 단순 교체, 이득 없음" 결론이 RNA 수정과 오늘의 WSI 브랜치
    진단(gradient 기아, attention 붕괴, 좌표 임베딩 무효과)을 다 거친 지금도 그대로 재확인됐다.
    **backbone 품질/용량보다 표본 규모(91명)·censored survival 라벨의 낮은 신호 대 잡음비
    자체가 병목이라는 쪽에 계속 힘이 실린다** — backbone을 아무리 키워도 91명분 학습 신호로는
    그 용량을 못 살린다는 뜻으로 보인다.
- **ABMIL vs AvgPool, 모델 capacity 축소**: 유의미한 차이 없음(이미 초기 조사에서 결론).
- **train_clinical_rna_only.py**: `train_light.py`로 대체돼 삭제됨.
