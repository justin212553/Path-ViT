# 오늘 밤 배치 결과 보고 (2026-07-17 새벽)

사용자 지시: 4개 임무를 순서대로, 전부 `--dataset both`(internal, 노이즈 최소화 + validation set 확보) 프로토콜로,
GPU 병목 방지를 위해 완전 직렬 실행. 진행 로그: `.logs/train_both_seed*_*_EX_both.log`,
`.logs/train_light_both_seed*_*_EX_both.log`, `.logs/retile_cptac_512.log`.

---

## Task 1: literature_1500 → M4/M4A/PM4 적용

### 목적
`findings_backlog.md` 1번 항목의 핵심 미해결 질문 — PMA_EX(다성분 pooling+co-attention+literature_1500)의
both-프로토콜 도약(0.583→0.656)이 **아키텍처(다성분 pooling+co-attention) 때문인지, 유전자셋 교체만으로도
재현되는지**를 분리 검증. subtype(339유전자) 기반이던 M4/M4A/PM4에 유전자셋만 literature_1500(1500유전자,
Cox+Stouffer 재선정)으로 바꿔서 동일 아키텍처로 재학습.

### 실행
```
python train.py --dataset both --seed {42,84,126} --M4  --rna-genes literature_1500 --group-ts 0717::0106
python train.py --dataset both --seed {42,84,126} --M4A --rna-genes literature_1500 --group-ts 0717::0106
python train.py --dataset both --seed {42,84,126} --PM4 --rna-genes literature_1500 --group-ts 0717::0106
```
9 run, 01:06:52 ~ 01:59:51 (약 53분). Train/Val/Test = 178/60/58 patients(both, 6:2:2 stratified).

### 결과

| 모델 | Internal C-index | HR | log-rank p | AUC mean | subtype 시절 C-index |
|---|---|---|---|---|---|
| **M4_EX** | 0.628 | 1.914 | 0.123 | 0.646 | 0.539 |
| **M4A_EX** | **0.644** | 2.023 | 0.097 | 0.651 | 0.549 |
| **PM4_EX** | 0.611 | 1.745 | 0.200 | 0.635 | 0.553 |
| (참고) PMA_EX | 0.656 | 2.15 | 0.10 | — | 0.583 |

시드별 상세:

| 모델 | seed42 C | seed84 C | seed126 C |
|---|---|---|---|
| M4_EX | 0.568 | 0.615 | 0.701 |
| M4A_EX | 0.586 | 0.647 | 0.699 |
| PM4_EX | 0.572 | 0.627 | 0.635 |

### 핵심 발견 — 노벨티 서술 재조정 필요

**세 모델 전부 유전자셋만 바꿨는데 subtype 대비 +0.06~+0.09 c-index가 일제히 뛰었다.** PMA_EX가 처음
보여준 도약(+0.073)과 거의 같은 폭이, 다성분 pooling이 전혀 없는 가장 단순한 M4에서도 그대로 재현됨
(+0.089). 심지어 **M4A_EX(0.644, co-attention만)가 PM4_EX(0.611, 다성분 pooling+게이트)보다 높고
PMA_EX(0.656)에 거의 근접**한다 — 다성분 pooling을 추가한 게 오히려 근소한 손해로 보이는 정황도 있다
(M4_EX 0.628 > PM4_EX 0.611).

**결론**: "다성분 pooling이 핵심 병목을 풀었다"는 기존 가설은 **부분적으로만 맞다**. 압도적 기여 요인은
유전자셋(Task 2/findings_backlog 2번 항목)이고, RNA 개입 지점 중에서는 co-attention(M4A류)이 post-hoc
게이트(M4/PM4류)보다 근소 우위를 보이는 정도다. PMA_EX가 여전히 both 최고 기록이지만 M4A_EX와의 격차
(+0.012)는 M4A_EX가 M4_EX를 이기는 격차(+0.016)와 비슷한 수준이라, "다성분 pooling의 순증분 기여"라고
부르기엔 근거가 약하다.

**주의**: 전부 `--dataset both`만 검증됨. PMA(subtype)가 both에서 최고 기록을 냈다가 external에서 완전히
무너진 전례가 있으므로(findings_backlog 1번 항목 (c)), 이 순위가 external에서도 유지되는지는 **아직 모른다**
— 다음 우선순위로 반드시 검증 필요.

---

## Task 2: M6/M7을 literature_1500으로 재검증 (lr=1e-3)

### 목적
WSI가 전혀 없는 모델(RNA+Clinical만)도 유전자셋 교체로 얼마나 좋아지는지 확인 — Task 1 결과와 함께 보면
"WSI가 실제로 얼마나 기여하는지"를 가늠하는 대조군.

### 실행
`train_light.py`에 `--rna-genes` 옵션이 원래 없어서(subtype 고정) 이번에 추가함(train.py와 동일한
literature_{1000,1500,2000} 선택 + `_EX` 접미사 관례). lr은 `LightTrainConfig` 기본값이 이미 1e-3이라
별도 변경 불필요(지시사항과 일치 확인).
```
python train_light.py --dataset both --seed {42,84,126} --M6 --rna-genes literature_1500 --group-ts 0717::0106
python train_light.py --dataset both --seed {42,84,126} --M7 --rna-genes literature_1500 --group-ts 0717::0106
```
6 run, 01:59:51 ~ 02:09:15 (약 9분 — WSI 처리가 없어 훨씬 빠름).

### 결과

| 모델 | Internal C-index | HR | log-rank p | AUC mean |
|---|---|---|---|---|
| M6_EX (RNA만) | 0.619 | 2.190 | 0.095 | 0.647 |
| M7_EX (RNA+Clinical) | 0.621 | 2.091 | 0.252 | 0.669 |

### 핵심 발견

M6_EX/M7_EX(WSI 완전히 없음)가 0.619/0.621로, **WSI를 쓰는 M4_EX(0.628)·PM4_EX(0.611)와 사실상 같은
범위**다. 문헌기반 유전자셋을 쓰면 WSI를 아예 빼도 both-프로토콜 성능이 거의 동일하다는, Task 1보다도
더 불편한 관찰. WSI가 유의미하게 앞서는 건 M4A_EX(0.644)와 PMA_EX(0.656)뿐 — "co-attention 방식으로
WSI와 RNA를 결합하는 것" 자체는 여전히 근소하게 의미 있어 보이지만 격차가 크지 않다.

(단, 이전 subtype 시절엔 M7이 external에서 오히려 강세(0.575, WSI 모델들보다 높음)였던 전례가 있어 —
both에서 WSI 유무 차이가 작다고 해서 external에서도 그렇다고 단정할 수 없음. → 실제로 아래 Task 3.5에서
external 검증해보니 이 우려가 그대로 맞았다.)

---

## Task 3: CPTAC 재타일링 — 완료 (우여곡절 있었음)

**스펙**: 512px @ 0.5MPP — Lunit SwAV 사전학습 스펙(512px, 20배율/0.5MPP)에 정확히 맞춤(백로그 4번 항목의
"256px" 예시보다 타일 수 증가가 완만하면서도 사전학습 해상도와 정확히 일치하는 쪽을 선택). 기존 데이터
(`data/patches_cptac`, 1024px@1.0MPP)는 그대로 두고 `data/patches_cptac_512`에 새로 저장.

**겪은 문제 3가지(전부 수정 완료)**:
1. `data/preprocess.py`가 멀티프로세싱 워커에 `--target-mpp`/`--tile-size` 등을 전달하는 방식이
   Windows(spawn)에서 조용히 무시되는 구조 → `Pool(initializer=...)` 패턴으로 수정, 재현 테스트로 확인.
2. 567장 중 3장(`C3L-03513/14/15-21`)이 실제로는 조직 스캔이 아니라 손상된 이미지(2880×2048px,
   native_mpp가 2,540,000이라는 비정상 값 — 정상은 0.05~5.0)였음. 이 값을 그대로 grid 계산에 쓰면 타일
   1픽셀 단위로 수백만 셀이 생겨 사실상 무한루프처럼 멈춤 → `_extract_slide()`에 native_mpp sanity
   check(0.05~5.0 범위 밖이면 즉시 ValueError, 기존 실패 처리 경로로 스킵) 추가.
3. 위 수정에서 넣은 에러 메시지에 Windows 콘솔(cp949)이 못 읽는 문자(—)가 있어서 worker 프로세스가
   또 죽는 2차 사고 → ASCII 메시지로 교체.
4. (부수적으로) 세션/VSCode가 두 번 정도 예기치 않게 재시작되면서 백그라운드 작업이 같이 죽는 일이 반복돼,
   `.done` 마커 기반 재개(resume) 방식으로 매번 복구.

**최종 결과**: 567장 중 **564장 정상 완료, 3장은 손상 파일로 정상 제외**(`--tiles-only`라 아직 GPU
feature 추출은 안 함, Task 4에서 진행).

---

## Task 3.5: M4_EX/M4A_EX/PM4_EX/M6_EX/M7_EX external 검증 — 완료, 핵심 결과

**목적**: Task 1/2가 `--dataset both`로만 검증됐는데, 이 프로젝트에서 both 프로토콜 순위는 반복적으로
external에서 뒤집힌 전례가 있어(PMA subtype→EX 때 한 번) 신뢰할 수 없다는 게 이미 확인된 상태였음.
CPTAC 재타일링이 끝나는 대로 이 5개 모델을 전부 `--external`(tcga↔cptac 양방향×3시드=6)로 재검증.

**실행 중 문제**: 세션이 재시작되면서 학습 작업이 두 번 죽음(1/30만 끝난 채로, 그다음 2/30 도중에 또)
→ `Start-Process`로 완전히 분리된(detached) 프로세스로 재시작, `.logs` 파일 존재 여부로 이미 끝난 run은
건너뛰도록 재개 스크립트(`scripts/_ext_validation_resume.ps1`) 작성해서 결국 30개 다 완료.

### 결과 — WSI 안 쓰는 모델이 최고 기록

| 모델 | Internal C | External C | External HR | External p | External AUC |
|---|---|---|---|---|---|
| M4_EX (WSI+게이트) | 0.609 | 0.604 | 1.716 | 0.103 | 0.618 |
| M4A_EX (WSI+co-attn) | 0.620 | 0.611 | 1.677 | 0.074 | 0.620 |
| PM4_EX (WSI+다성분+게이트) | 0.606 | 0.593 | 1.651 | 0.125 | 0.606 |
| **M6_EX (RNA만)** | 0.637 | **0.627** | 1.914 | **0.005** | 0.657 |
| **M7_EX (RNA+Clinical)** | 0.627 | **0.634** | 1.975 | **0.0025** | 0.670 |
| (참고) PMA_EX | 0.613 | 0.603 | 1.781 | 0.150 | 0.610 |

**WSI를 아예 안 쓰는 M6_EX/M7_EX가 external C-index·HR·AUC 전부에서 WSI를 쓰는 모든 모델을 능가한다.**
M7_EX는 6시드 전부 p<0.01 — 이 프로젝트 전체에서 가장 통계적으로 강력하고 일관된 결과다. WSI를 쓰는
모델은 하나도 p<0.05를 못 넘는다. both에서는 정반대(WSI 모델이 앞섬)였는데 external에서 뒤집힘 — both
프로토콜 순위가 신뢰할 수 없다는 게 이번이 세 번째 확인.

**의미**: 지금까지 "PMA_EX가 최고 기록"이라던 서술은 both 프로토콜 한정이었다. 진짜 cross-institution
기준으로는 **WSI+RNA+Clinical 융합 모델 중 어느 것도 RNA+Clinical만 쓰는 M7_EX를 못 넘는다** — 이 데이터셋·
이 fusion 설계 하에서 WSI가 순증분 기여를 못 한다는 것이 지금까지 나온 증거 중 가장 강력하다. 상세는
findings_backlog.md 1번 항목 (f) 참조.

---

## Task 4: 재타일링 기반 PM4_EX — TCGA 대기 중

**블로커**: TCGA 원본 WSI가 로컬에 없었음. 사용자가 HPC3에서 수동으로 복사 중인데, 처음엔 `data/`가 아니라
프로젝트 루트의 `tcga_paad_wsi/`(gdc-client 기본 UUID-서브디렉토리 구조, `.svs.partial`→완료 시 `.svs`로
rename)로 잘못된 경로에 받아지고 있는 게 확인됨 — 다운로드 다 끝나면 `data/tcga_paad_wsi`로 옮기고
`data/flatten_tcga_paad_wsi.py`로 평탄화할 예정. 아직 다운로드 진행 중.

**Task 3.5 결과로 인해 Task 4의 의미가 달라짐**: 원래는 "재타일링하면 PMA_EX가 더 좋아지는지" 확인이
목적이었는데, 이제는 **"재타일링해도 WSI 모델이 M7_EX(RNA+Clinical 단독)를 못 넘으면, WSI가 이 문제에서
근본적으로 기여를 못 한다는 결론이 훨씬 강해진다"**는, 더 중요한 검증으로 격상됨.

**계획**: TCGA 파일 확보 확인 → 동일 스펙(512px@0.5MPP)으로 재타일링(CPTAC와 같은 native_mpp sanity
check 적용) → 두 코호트 feature 재추출(GPU) → PM4_EX(및 가능하면 M4A_EX도)를 새 패치 경로로 재학습
(`train.py --patches-root-tcga data/patches_tcga_512 --patches-root-cptac data/patches_cptac_512`,
이번에 옵션 추가해둠) → **--external로 검증해서 M7_EX(external 0.634)를 넘는지가 관건**.

---

## 다음 우선순위 (상의 필요)

1. **Task 3.5 결과의 해석과 서술 재조정** — "다성분 pooling+co-attention이 핵심 노벨티"에서 "유전자
   재선정이 핵심"으로 한 번 물러났는데, 이번엔 "WSI 자체가 이 설계로는 external에서 기여를 못 한다"는
   더 큰 재조정이 필요할 수 있음. 논문/보고서 프레이밍을 어떻게 할지 논의 필요.
2. Task 4 계속 진행(TCGA 대기, 자동으로 이어감) — 재타일링이 이 결론을 뒤집을 수 있는 마지막 카드.
3. WSI가 external에서 손해를 보는 이유 후보 3가지(findings_backlog 1번 항목 (f) 해석 후보 참조) 중
   어느 쪽이 맞는지 — 재타일링 결과가 첫 단서가 될 것.
