# WSI 신호 부재 + seed 재현성 한계 — 종합 정리 (2026-08-14)

이 문서는 2026-08-14 세션에서 진행한 "WSI 브랜치 구조 개선 시도" 전체와, 거기서 나온
"seed-level 재현성이 코호트 규모 자체의 한계"라는 결론을 논문 집필 시 참조할 수 있게
정리한 작업 노트다. `paper/sections/*.tex`에 아직 반영 안 된 원재료 — 최종 문장화는
나중에.

## 1. 핵심 결론

**이 코호트 규모(TCGA-PAAD+CPTAC-PDAC, 학습 가능 표본 ~150명)에서는 학습 자체의
무작위성(특히 가중치 초기화)만으로 생기는 성능 변동이, 지금까지 시도한 대부분의 WSI
아키텍처 개선 시도가 만드는 차이보다 크거나 비슷하다.** 즉 "WSI가 RNA+Clinical 대비
순증분 기여를 못 한다"는 관찰과 "모델 순위가 seed마다 흔들린다"는 관찰은 같은 원인
(표본 부족)에서 나오는 두 증상이다.

## 2. Noise floor 정량화 — `--full-train` 실험 (2026-07-26)

TCGA 전체 152명을 그대로 train으로 쓰고(6:2:2 split 자체가 없음) seed만 바꿔 external(CPTAC)
평가:

| seed | external C-index |
|---|---|
| 42 | 0.6389 |
| 84 | 0.6210 |
| 126 | 0.5915 |
| 168 | 0.5913 |
| 210 | 0.6277 |

mean=0.614, **std≈0.022, range=0.048**. Split/fold 구성을 아예 없앴는데도 seed 변동이
그대로 남아있다.

같은 날, seed168을 고정하고 무작위성의 원천 하나씩만 분리:

| 변형 | external C-index |
|---|---|
| baseline(seed168 그대로) | 0.6389(seed42 기준 참고) |
| init(가중치 초기화)만 다른 seed | 0.6352 |
| data-order(patient shuffle) 끔 | 0.5978 |
| patch-subsample seed만 다름 | 0.5936 |

**가중치 초기화 하나만 바꿔도** 0.598~0.635(폭 0.037)의 변동이 나온다 — 전체 5-seed
변동폭(0.048)과 비슷한 크기. → 변동의 지배적 원인은 데이터 분할이 아니라 **학습 자체의
무작위성**이다.

## 3. 이번 세션 WSI 아키텍처 개선 시도 — 전부 negative로 수렴

baseline: `PMA_uni2_INT1500_SS_AUX_STG_R_DISP_COX_ADD` (3seed×5fold pooled)
**internal=0.6359, external=0.6337**

| # | 시도 | 가설 | internal | external | 판정 |
|---|---|---|---|---|---|
| 1 | Decision-level ensemble(M6+M5+M1_POOL) | 예측 단계 결합이면 안전하지 않을까 | 0.6597(M6+M5만, WSI 넣으면 더 나쁨) | 0.5977(WSI 포함 시) | negative |
| 2 | Model soup(가중치 평균) | 여러 seed를 가중치 레벨에서 합치면 | 0.6493 | 0.5679 | negative |
| 3 | SWAD | soup보다 나은 평탄화 | ~soup과 동일 | ~soup과 동일 | negative |
| 4 | BRCA 공동학습(표본 7배 확장 시도) | 표본 부족이 원인이면 늘리면 해결? | train-val gap 0.148→0.184(과적합 악화) | 이득 없음 | negative |
| 5 | uni2native(공식 spec 재타일링, 256px@0.5MPP) | 해상도 mismatch가 원인? | 0.6313 | 0.6238 | negative (baseline과 사실상 동일, train-val gap도 0.189로 baseline 수준 회귀) |
| 6 | stage-stratify(split을 병기로 층화) | fold별 log-rank p 변동 원인 규명 | 0.6332 | 0.6341 | c-index는 flat, fold-log-rank-p std는 0.181→0.117로 부분 개선(별개 지표) |
| 7 | leverage-stratify(고레버리지 환자로 층화) | 위와 같은 방향, 더 직접적 | 0.6270 | 0.6370 | negative(6번보다도 약함) |
| 8 | DX-only 슬라이드(냉동절편 제외) | uni2official confound 분리 검증 | 0.6367 | 0.6325 | negative(flat) |
| 9 | complete-24m 코호트 제한(레퍼런스 기준) | 조기 censoring 노이즈 제거? | 0.5631 | 0.6230(N=125) | negative(표본 축소 손해가 더 큼) |
| 10 | attn-dispersion 사후 zero-ablation | 이 feature가 실제로 쓰이나? | Δ+0.0003 | Δ-0.0006 | 기여 사실상 0 |
| 11 | M4(4-component+co-attention → 단일 ABMIL) | pooling이 너무 복잡해서 과적합? | 0.6348 | 0.6325 | negative(flat, pooling 복잡도가 원인이 아님을 시사) |
| 12 | M4+skip-patch-vit(patch-mixing transformer 제거) | 사전학습 UNI2 표현을 새 transformer가 흐림? | 0.6663(seed std=0.0432, seed126=0.5832로 이상치) | 0.6364 | internal 상승은 seed42 쏠림으로 판단, external은 flat |
| 13 | PMA+tile-risk-head(top-k를 독립 헤드로 분리+risk_stats 10개 추가) | top-k가 붕괴한 attn_weights에 얹혀있던 문제 해결 | 0.6277 | 0.6032 | negative(external 뚜렷이 나쁨) |
| 14 | M4A+skip-patch-vit(ABMIL→RNA co-attention) | 표현력 확장 | seed126 pooled 0.6187(baseline 0.6193과 동일) | 0.6243(baseline 0.6568보다 나쁨) | negative |
| 15 | M4+avgpool(학습 파라미터 없는 순수 평균 풀링) | attention 자체가 무의미하니 아예 제거 | seed126 pooled 0.6038(baseline보다 낮음, fold2=0.4385 이상치) | 0.6456(baseline보다 낮지만 14번보다는 나음) | 아직 결론 보류(seed42/84 미완, HPC 실패로 재제출 중) |

## 4. 구조적 진단 — attention이 실제로 붕괴해 있음을 직접 확인

`scripts/diagnose_pma_wsi_structure.py` (2026-08-14, baseline 체크포인트, 재학습 없음):

- patch-level ABMIL attention entropy: **mean=0.9991**(1=완전 균일) — 사실상 무작위
- 4-component co-attention(RNA query) entropy: **mean=0.9993** — 이것도 균일
- z_wsi vs z_rna raw norm 비율: 1.7배(극단적 스케일 버그는 아님)
- gradient norm: rna_encoder=14.01 vs WSI 브랜치 전체(cnn+vit+attn_pool+coattn)=5.07(RNA가 2.8배)
- **attn_pool(게이트) 자체의 gradient norm = 0.0135** — 다른 모듈 대비 100~250배 작음

→ attention이 "학습이 안 된" 게 아니라, entropy가 균일한 상태에서 gradient도 거의 안
흐르는 **self-reinforcing collapse** 상태로 보임. RNA가 gradient를 독식하는 게 이
붕괴를 가속시키는 후보 원인.

## 5. 이 세션 이전부터 있던 동일 패턴의 증거

`findings_backlog.md` 1번 항목 (f), 2026-07-17, 구식 레시피(uni 백본, cox_add/margin/
staging/dispersion 없음) 기준:

| | External C | External p |
|---|---|---|
| M4(게이트-bias ABMIL) | 0.604 | 0.103 |
| M4A(co-attention) | 0.611 | 0.074 |
| PM4(다성분+게이트) | 0.593 | 0.125 |
| PMA(다성분+co-attention) | 0.603 | 0.150 |
| **M6(RNA만, WSI 없음)** | **0.627** | **0.005** |
| **M7(RNA+Clinical, WSI 없음)** | **0.634** | **0.0025** |

당시 기록: "WSI를 아예 안 쓰는 M6_EX/M7_EX가 external 전부에서 WSI를 쓰는 모든 모델을
능가한다... 이 프로젝트 전체를 통틀어 가장 일관되고 통계적으로 강력한 결과다." M7은
6시드 전부 p<0.01, WSI 모델은 단 하나도 유의하지 않았음(p=0.07~0.15). **이번 세션은
훨씬 엄격한 방법론(3seed×5fold pooled OOF)으로 같은 결론을 재확인한 것.**

## 6. 유일한 긍정적 반례 — BRCA (미검증, 참고용)

TCGA-BRCA(1058명, PAAD의 7배), 동일 PMA 아키텍처, **seed=42 단일 결과**(2026-07-22):

| | test c-index |
|---|---|
| M7(WSI 없음) | 0.6620 |
| M4/PMA(WSI 포함) | **0.7155(+0.0535)** |

이 프로젝트 전체에서 WSI가 확실한 순증분을 보인 유일한 사례. 다만 재현성 검증 전에
"BRCA를 아키텍처 테스트베드로 우선 쓴다"는 정책이 반복 속도 문제로 취소돼(2026-07-23),
**다른 시드로 재검증된 적이 없음**. "표본을 늘리면 WSI가 도움이 될 수 있다"는 가설을
완전히 죽이지 못하는 유일한 데이터지만, 통계적으로 확정된 것도 아님.

## 7. 논문 전략 (2026-08-14 논의, 사용자 결정)

**배경**: 원래 계획은 카이저 병원에서 별도 external 코호트를 받기로 했었으나 무산됨
(TCGA/CPTAC만으로 진행). 연구 기간도 한정적.

**결정**: seed=42를 주 결과(primary result)로 보고하되, 아래 조건을 지켜 정직하게
작성한다.

1. **seed42는 사후 선택이 아니라 프로젝트 초반부터 고정해 온 시드**임을 명시 — "유리한
   시드를 골랐다"는 의심을 원천 차단.
2. **"재현성이 불안정하다"를 막연한 한 문장으로 넘기지 않고, 위 2절의 noise floor
   수치(± 0.02~0.05)로 정량화**해서 제시 — 한계 고백이 아니라 소규모 코호트 생존예측
   전반에 대한 방법론적 발견으로 재구성.
3. **결론의 확신 수준을 실제 아는 것에 맞춘다** — "PMA가 최고다"처럼 단정하지 않고,
   "seed42 기준 X이나 cross-seed 분석 결과 이 순위는 학습 무작위성 안에 묻혀 안정적이지
   않다 — 이 코호트 규모에서는 WSI의 순증분 기여 여부 자체를 통계적으로 구별하기 어렵다"
   는 식으로 서술.
4. 3절의 negative result chain(15개 아키텍처 변경 시도) 전체가 "안 해봐서 모른다"가
   아니라 "다 해봤는데도 표본 규모로 수렴한다"는 근거로 인용 가능.

`paper/sections/07_evaluation_protocol.tex`에 이미 이 방향에 부합하는 문장이 있음:
"random initialization alone produces C-index variation on the same order as
differences between competing model variants" — 이번 세션은 이 문장의 **구체적 근거
(2절 noise floor 수치)**를 만든 것.

## 8. 진행 중 — seed126 이상치 여부 확인

M4+skip-patch-vit/M4A+skip-patch-vit/M4+avgpool 전부에서 seed126이 다른 시드보다
낮게 나오는 경향이 반복 관찰됨(예: M4-NOVIT internal seed42=0.6885 vs seed126=0.5832).
seed126이 우연히 나쁜 draw인지, 아니면 실제로 이상치인지 확인하기 위해 seed 3개를
추가로 더 돌려 분포를 볼 계획(진행 예정, 아직 결과 없음).

## 9. 모달리티별 internal/external 비대칭 가설 (2026-08-15, 사용자 제안·seed42 한정)

**가설(사용자 원문 요지)**: clinical(이산적, 코호트 간 기술적 노이즈가 적은 값)을
강화/추가하면 external이 오르고, WSI(연속적이고 스캐너·염색·기관별 기술 노이즈가 낀
값)를 강화/추가하면 internal이 오른다 — WSI는 노이즈까지 같이 흡수하지만 그 안에
필요한 도메인 정보도 섞여 있어 학습 코호트 안에서는 유리하고, clinical은 노이즈가
적어 다른 코호트로도 잘 옮겨간다는 것. 3개 모달리티를 다 쓰는 M4는 이 두 축의
스윗스팟(sweet spot)일 수 있다는 해석.

**독립적으로 재현된 세 비교(전부 seed42, uni2 backbone, 5-fold pooled)**:

| 비교 | 무엇을 더했나 | internal 변화 | external 변화 |
|---|---|---|---|
| M6(RNA only, 0.6221/0.5893) → M7(+Clinical, 0.6025/0.5958) | Clinical 추가 | -0.0196 | **+0.0065** |
| M6(RNA only) → M3(+WSI, 0.6488/0.5667) | WSI 추가 | **+0.0267** | -0.0226 |
| PMA-noclinical(WSI+RNA, 0.6020/0.5428) → PMA-full(+Clinical, cox_add, 0.5984/0.6015) | Clinical 추가 | -0.0036(거의 flat) | **+0.0587** |

세 비교 모두 방향이 정확히 일치 — clinical 추가는 internal 정체/소폭 하락 + external
뚜렷한 상승, WSI 추가는 internal 상승 + external 하락(거울상). 서로 다른 아키텍처
(단순화 M-사다리 vs PMA의 4성분 pooling+co-attention)에서 독립적으로 나온 결과라
우연으로 보기 어렵다.

**lr-mult 개입 실험(같은 절 15-19 참조)에서도 방향이 일치**: `--clinical-lr-mult`로
M2의 clinical 브랜치를 강화하면 internal +0.014/external **+0.040**(external이 더 큼).
`--rna-lr-mult`로 M3의 RNA 브랜치를 강화하면 internal +0.003(거의 flat)/external
**+0.032**.

**RNA는 "연속형"인데도 clinical 쪽 패턴을 따름 — 가설의 정제**: 위 rna-lr-mult
결과가 시사하듯, 진짜 축은 "WSI vs 그 외"가 아니라 "코호트/기관별 **기술적** 노이즈가
얼마나 끼어있는가"로 보인다. RNA-seq도 batch effect가 있지만 WSI의 스캐너·염색·조직
처리 편차만큼 크지 않아서, 이 노이즈 축에서는 WSI보다 clinical에 훨씬 가깝게
행동하는 것 같다.

**한계**: 전부 seed42 단일 시드 결과 — 이 문서의 다른 절(noise floor ±0.02~0.05)을
고려하면 개별 delta 중 일부(특히 M6→M7의 +0.0065처럼 작은 값)는 noise floor 안에
들어갈 수 있다. 다만 방향이 세 개 독립 비교 + 두 개 lr-mult 개입에서 전부 일치한다는
점이 우연으로 보기엔 너무 일관적 — multi-seed로 재확인이 필요하지만, 나중에
실험 자체가 뒤집히더라도 그 자체로 흥미로운 방법론적 관찰(모달리티의 "기술적 노이즈
프로파일"이 internal/external 성능 배분을 예측할 수 있다는 것)로 남을 만하다.
