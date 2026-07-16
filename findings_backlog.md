# PATH-ViT 발견 사항 및 우선순위 백로그

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

---

## 최우선순위

### 1. ABMIL 단일 벡터 압축 → 다성분(multi-component) pooling으로 교체
**우선순위: 최우선** — M4 both 결과가 레퍼런스에 크게 못 미친 근본 원인으로 지목됨 (0번 항목 참조).

- **문제의 정확한 위치**: "ABMIL이냐 CLAM이냐"가 핵심이 아니다 — CLAM(Lu et al. 2021)도 내부적으로 동일한 gated attention pooling을 쓰고, 최종 표현은 여전히 압축된 벡터 1개다(instance-level clustering loss로 attention을 추가 지도할 뿐). 레퍼런스가 실제로 쓰는 건 CLAM이 아니라 자체 설계한 Morphology Burden Pooling — **mean, std, risk-weighted pooling, top10%/top25% high-risk-tile 평균, risk 분포 통계(mean/std/max/quartile/top5%/top10%)까지 여러 벡터를 concat**해서 정보를 압축하지 않는다. RNA 게이트가 이 풍부한 표현의 각 성분을 따로 조절할 수 있다는 게 핵심 차이.
- **액션**: `AttentionPooling`(vit_m1.py)을 다성분 pooling 모듈로 교체 — 최소 mean/std/attention-weighted/top-k 정도부터 시작해 레퍼런스 수준(6개 성분)까지 단계적으로 확장 검토.
- **노벨티 확보**: 다성분 pooling 자체는 레퍼런스와 겹치는 인프라 교정이라, 그 위에 이 프로젝트만의 차별점을 최대한 얹는다:
  - **ViT self-attention(Nystromformer) 공간 컨텍스트 블록** — 레퍼런스는 좌표 임베딩만 더하고 패치 간 self-attention 층이 없음(각 타일 독립 임베딩 후 곧장 pooling). 우리는 pooling 전에 패치들이 서로 주목하는 self-attention이 있다 — "공간 문맥을 주고받은 뒤 다성분 요약"이라는 조합이 차별점.
  - **RNA 개입 지점 체계적 비교(M4/M4A/M4B)** — 레퍼런스는 sigmoid 게이트 하나만 고정. 우리는 post-hoc/게이트-bias/co-attention/pre-ViT FiLM을 통제 비교한 사다리를 이미 구축함(0번 항목에서 M4A/M4B가 이기면 이 축의 의의가 더 커짐).
  - **평가 rigor**: 멀티시드 + internal/external 이중 검증 + 프로토콜 난이도 자체의 별도 검증(0번 항목). 레퍼런스 쪽 보고서도 스스로 "반복 split/CV, bootstrap CI 부재"를 gap으로 지적했던 부분.
  - 그 외 추가로 올릴 수 있는 노벨티 축이 있으면 이 항목 진행하면서 계속 탐색.

---

## 중간 우선순위

### 2. RNA 브랜치 유전자 선정 기준 (레퍼런스 참고)
현재 339개(Bailey 2016 + Moffitt 2015 subtype 분류용) 유전자 사용. 레퍼런스는 1000~2000개를 문헌 큐레이션 9개 카테고리 + train split 내부 univariate Cox score test + Stouffer meta-analysis로 선정(생존 예측에 직접 최적화).

- **확인된 사실**: 인코더 폭만 레퍼런스 사양(64→256차원, dropout 0.25)으로 넓힌 M6X는 M6 대비 internal -0.02, external +0.02 — 방향은 긍정적이나 크지 않음. "개수"보다 "선정 기준"이 핵심일 가능성.
- **다음 액션**: train split 내부 Cox score test + Stouffer meta-analysis 파이프라인 구축(`scripts/select_rnaseq_gene_features.py` 참고), literature-curated seed gene 목록 정리.

### 3. 학습 하이퍼파라미터 검증 (light + WSI 모델 모두)
- **WSI-free 모델(M5/M6/M6X/M7)**: `train_light.py`(`LightTrainConfig`, lr=1e-3)가 아직 검증된 값이 아님 — 스모크 테스트에서 lr=1e-3이 M6를 train_c_index 0.99까지 과적합시키는 것도 확인했다. lr 스윕(1e-5/1e-4/1e-3/3e-3 등) 필요.
- **WSI 포함 모델(M1/M4/.../PM4/PMA)**: `config.py::TrainConfig.lr=1e-5`도 재검토 대상. Backbone은 어차피 얼려있어 ViT/pooling/risk_head는 light 모델과 마찬가지로 처음부터 학습되는데, 왜 1e-5로 이례적으로 낮게 잡았는지는 불명. 다만 ViT self-attention(Nystromformer) 블록이 껴 있어 light 모델보다 LR에 민감할 수 있다(gradient clipping·warmup이 이미 있어 어느 정도 안전판은 있음). **액션**: M1(가장 가볍고 기준값 확보됨)으로 1e-5/3e-5/1e-4 단일 시드 민감도 비교부터 — 지금 진행 중인 both 비교 시리즈(0번, PM4/PMA)가 전부 lr=1e-5로 통일돼 있으니, 그 비교들이 마무리된 뒤에 진행해 lr 변경이 기존 비교를 오염시키지 않게 한다.

---

## 오늘 밤에 돌릴 것

### 4. WSI 타일 해상도/물리 스케일 미스매치
**중요도와 무관하게 "시간이 오래 걸린다"는 이유만으로 야간 배치로 돌린다.**

- **사실관계**: 우리 타일은 1024×1024px @ 1.0 MPP, 리사이즈 없이 backbone 투입. Lunit SwAV는 512×512px @ 0.5MPP(20배율)/0.25MPP(40배율) 사전학습 — 픽셀당 해상도 2~4배, 타일당 물리 면적 16배 차이. UNI는 이미 확인된 대로 ~0.5MPP 학습 분포에서 4배 이상 어긋남.
- **왜 중요한가**: ResNet50/UNI 두 backbone 모두에 걸리는 문제라, "어떤 인코더냐"보다 "우리 타일링 컨벤션 자체"가 근본 병목일 가능성.
- **액션**: 원본 WSI에서 물리적으로 더 작은 FOV로 재타일링(예: 256px@0.5MPP, 타일 수 ~64배 증가) — 오래 걸리는 배치 작업이라 야간에 실행.

---

## 미정 (현재 해결책 없음)

### 5. 도메인/배치 효과 오염의 근본 원인
raw CNN feature로 TCGA/CPTAC를 구분하는 도메인 분류기 AUC=0.78(강한 배치 효과). Stain normalization(Macenko)으로 시도했으나 AUC 0.803으로 오히려 소폭 악화 — 이 경로는 종료.

- 색상 보정으로 안 풀린다는 건, 원인이 스캐너 해상도/PSF, JPEG 압축, 조직 절편 두께, 디지털화 파이프라인 차이 등 더 근본적인 곳에 있을 가능성을 시사하지만, 이건 SOTA 모델들도 만성적으로 겪는 문제다 — 지금 수준에서 깔끔한 해결책이 없다. **재타일링(4번 항목)이 스캐너별 해상도 차이를 부수적으로 완화할 가능성은 있으니, 그 결과가 나오면 도메인 AUC를 다시 재보는 정도로만 연결**하고, 별도의 적극적인 해결 시도는 지금 보류한다.

---

## 이미 종료된 경로 (참고용, 재시도 불필요)

- **Stain normalization**: AUC 0.78→0.803, 개선 없음.
- **UNI backbone 단순 교체(리사이즈만, 재타일링 없이)**: 224/512 리사이즈 둘 다 ResNet50 대비 이득 없음(오히려 근소 열세). 4번 항목의 진짜 해결(재타일링) 없이는 재시도 의미 없음.
- **ABMIL vs AvgPool, 모델 capacity 축소**: 유의미한 차이 없음(이미 초기 조사에서 결론).
- **train_clinical_rna_only.py**: `train_light.py`로 대체돼 삭제됨.
