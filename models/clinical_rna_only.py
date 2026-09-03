"""
ClinicalRNAOnly — M7, Clinical(age/sex)+RNA-seq 결합, WSI 없음. train_light.py --M7.

2026-07-19: RNA 인코더를 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology)
scripts/models/tabular_survival.py 사양(RNAEncoderExtend, G->256->256, LayerNorm+Dropout
입출력 정규화)으로 교체 — GitHub 원문(모델 구조·RNA-seq 전처리·gene selection·split 프로토콜
전부)을 직접 확인해 이식했다. dim만 64로 낮춰본 ablation도 해봤지만 개선이 없어서(오히려
악화, findings_backlog.md 13번 항목) 원래 레퍼런스 폭(256)으로 되돌리고, 대신 학습 레시피
(epochs=100+early stopping patience=20, --dataset both)까지 전부 맞춰서 재검증한다.
Clinical은 레퍼런스와 동일하게 RNA(256)에 맞추지 않고 16차원 그대로 둔다(레퍼런스도
[rnaseq_embed_dim=256, clinical_embed_dim=16]로 비대칭). age/sex 정보량 자체는 기존과
동일(레퍼런스 clinical_dim=3=age+sex_onehot도 age/sex뿐, stage/grade 미사용 - GitHub
M4_Train.ipynb 주석으로 확인됨).

2026-07-21: risk_head를 레퍼런스(tabular_survival.py::ClinicalRNASeqSurvivalModel) 사양
(LayerNorm→Dropout(0.4)→Linear(272→128)→GELU→Dropout(0.4)→Linear(128→1))으로 교체해
M7_EX(기본 레시피)로 `--external` 재검증했으나 negative result(external C 0.634→0.533,
findings_backlog.md 13번 항목 참조) — 원래의 단순 선형(LayerNorm→Linear) 형태로 되돌렸다.

2026-07-21(후속, RNA 전처리 버그 수정 이후 재검증): 은닉층 없이 Dropout(0.4)만 추가한
최소 버전(레퍼런스 M4 사양)을 RNA 수정 후 M7_EX로 재검증했으나 또 negative — train_c_index가
30 epoch 만에 0.9865까지 치솟는데 val_c_index는 0.50대에 붙박이(교과서적 과적합), lifelines가
"collinearity or complete separation" 경고를 반복 — 학습 곡선 자체가 불안정/퇴화한 것으로
판단해 즉시 원복(사용자 판단, 결과 확정 전 중단). RNA를 고쳐도 risk head 직전 Dropout은
여전히 해롭다는 결론 유지 — 원래의 단순 선형(LayerNorm→Linear) 형태로 되돌렸다.

2026-08-05: 기본 폭을 RNA=256/Clinical=16(위 2026-07-19 결정)에서 RNA=64/Clinical=64로
바꿨다 — leakage-free 재검증 과정에서 M6(cfg.embed_dim=64 고정)와 공정 비교하려면 매번
--rna-dim 64 --clinical-dim 64를 명시해야 했는데, 이걸 두 번이나 빠뜨려 256/16으로 잘못
비교한 채 보고하는 사고가 반복됐다(사용자 지적으로 발견). 위 07-19 결정(256/16이 더
나았다는 관찰)은 leaky gene selection 시절 결과라 지금 기준으로 재검증되지 않았고, 앞으로
이 비율을 다시 바꿀 일은 거의 없을 것이므로 --rna-dim/--clinical-dim 인자 없이도 안전한
기본값(64/64, 균일)으로 코드 자체를 고정한다. 필요하면 여전히 --rna-dim/--clinical-dim으로
덮어쓸 수 있다.

2026-08-20~21: RNA 인코더를 RNAEncoderExtend(레퍼런스 이식)에서 RNAEncoder(M3/M4/M6와 동일,
hidden=256)로 교체 — M3/M4/M6와 다른 구조를 썼던 게 부당 비교였다는 지적(사용자)에 따른 것.
동시에 clinical cox_add/film을 raw feature 직결에서 ClinicalEncoder(MLP) 경유로 바꿔 RNA
cox_add(rna_linear가 RNAEncoder 출력을 받음)와 대칭을 맞췄으나, 두 변경을 분리한 2x2
ablation(--legacy-rna-encoder/--legacy-clinical-coxadd, 각 2seed×5fold) 결과 internal
c-index 하락(-0.027)의 거의 전부(-0.025)가 clinical의 ClinicalEncoder 경유 때문으로 확인됨
(RNA 인코더 교체는 -0.006으로 거의 무해). 결론: RNA 인코더 교체는 유지하고, clinical
cox_add는 raw feature 직결 방식으로 원복(clinical 신호가 약해 MLP를 거치면 오히려 과적합만
늘어난다는 뜻 — RNA는 신호가 강해 인코더가 도움되지만 clinical은 반대). ablation용
--legacy-rna-encoder/--legacy-clinical-coxadd 플래그는 결론이 났으므로 삭제하고, raw feature
직결을 cox_add의 유일한/영구 구현으로 고정한다.
"""
import torch
import torch.nn as nn

from .clinical_encoder import ClinicalEncoder, STAGE_FIELDS, _STAGE_BUFFER_NAMES, MUTATION_FIELDS, _MUTATION_BUFFER_NAMES
from .rna_encoder import RNAEncoder
from config import ModelConfig

RNA_EMBED_DIM = 64
CLINICAL_EMBED_DIM = 64


class ClinicalRNAOnly(nn.Module):
    """
    combine_mode(2026-07-29, train_light.py --combine-mode)로 clinical과 RNA를 합치는 방식을
    선택한다 — concat이 M6(RNA만)를 못 넘는 게 "clinical 브랜치 내부 설계"가 아니라 "신호 없는
    변수(M5 external p=0.754)를 신호 있는 RNA와 하나의 선형 결합기에서 공동 학습시키는 구조 자체"
    때문이라는 가설(findings_backlog.md 15번 항목) 검증용 — clinical이 개입할 수 있는 자유도를
    극단적으로 줄인 대안 두 가지를 추가했다.

    - "concat"(기본, 기존 동작): clinical_encoder(MLP)→z_c(clinical_dim)를 z_rna와 이어붙여
      risk_head가 처리. clinical 쪽 유효 파라미터가 가장 많다.
    - "film": clinical(age_z, sex)에서 스칼라 γ,β(각 2개 파라미터, 총 4개)만 만들어
      z_rna' = γ·z_rna + β로 RNA 임베딩 전체를 균일하게 스케일/이동만 시킨다. γ→1, β→0
      항등(identity) 근처로 초기화해 학습 시작 시점엔 사실상 M6와 동일하게 동작하고,
      clinical이 실제로 유용해야만 gradient가 항등에서 밀어낸다.
    - "cox_add": clinical을 임베딩하지 않고 고전적 Cox 비례위험모형처럼 최종 risk 스칼라에
      직접 선형항으로 더한다: risk = risk_head(z_rna) + β_age·age_z + β_sex·sex_idx
      (+ β_margin·margin_z, --clinical-margin일 때). age/sex 둘뿐일 때 파라미터 2개로 세 방식
      중 자유도가 가장 작다.

    2026-08-05: film/cox_add에도 margin(--clinical-margin, use_age_sex=False로 age/sex 제외
    가능) 지원을 추가했다 — M7이 concat 방식으로는 M6(RNA만)의 internal을 못 넘는 걸 보고,
    margin의 "약하지만 진짜인"(단독 Cox HR≈1.6, 양쪽 코호트 유의) 신호가 RNA와 같은 선형층에서
    직접 경쟁하는 구조 자체가 문제일 수 있다는 가설 검증용 — cox_add는 margin을 고전적 Cox
    공변량처럼 최종 스칼라에만 더해 RNA 임베딩 표현 자체를 전혀 건드리지 않는다.
    """

    def __init__(self, cfg: ModelConfig, age_mean: float, age_std: float, rna_input_dim: int,
                 clinical_dim: int | None = None, rna_dim: int | None = None,
                 clinical_dropout: float = 0.0, sex_onehot: bool = False,
                 combine_mode: str = "concat",
                 risk_hidden_dim: int | None = None, risk_dropout: float = 0.0,
                 use_margin: bool = False, margin_stats: tuple[float, float] | None = None,
                 use_age_sex: bool = True,
                 use_staging: bool = False, stage_stats: dict[str, tuple[float, float]] | None = None,
                 use_mutation: bool = False, mutation_stats: dict[str, tuple[float, float]] | None = None):
        super().__init__()
        if combine_mode not in ("concat", "film", "cox_add"):
            raise ValueError(f"알 수 없는 combine_mode: {combine_mode}")
        if use_margin and margin_stats is None:
            raise ValueError("use_margin=True면 margin_stats가 필요합니다.")
        if use_staging and stage_stats is None:
            raise ValueError("use_staging=True면 stage_stats가 필요합니다.")
        if use_mutation and mutation_stats is None:
            raise ValueError("use_mutation=True면 mutation_stats가 필요합니다.")
        self.combine_mode = combine_mode
        self.use_margin = use_margin
        self.use_age_sex = use_age_sex
        self.use_staging = use_staging
        self.use_mutation = use_mutation
        # 2026-07-28: clinical_dim을 키워서(레퍼런스 비대칭 16 대신 RNA와 같은 폭) "정보를 더
        # 주면(표현력을 늘리면) 오히려 나빠지는지" 검증하기 위한 ablation(train_light.py
        # --clinical-dim/--rna-dim). 기본값(None)이면 기존 동작(모듈 상수 256/16) 그대로 보존.
        # rna_dim도 같이 줄이면(예: 64/4) 레퍼런스의 RNA:Clinical=16:1 비율을 유지한 채 이
        # 프로젝트의 다른 모델들과 같은 절대 폭(64)으로 맞춰볼 수 있다.
        clinical_dim = clinical_dim or CLINICAL_EMBED_DIM
        rna_dim = rna_dim or RNA_EMBED_DIM
        # 레퍼런스 이식(RNAEncoderExtend)을 버리고 M3/M4/M6와 동일한 RNAEncoder(hidden_dim=256
        # 기본값, dropout=0.3 기본값)로 통일 — "레퍼런스는 어디까지나 레퍼런스, 본 연구의 M1~M7
        # 비교는 전부 같은 설정으로"(사용자 지시). rna_dim(출력 차원, 64)은 그대로 유지 —
        # cfg.embed_dim과 이미 같은 값이라 여기선 hidden_dim/dropout/인코더 구조만 바뀐다.
        self.rna_encoder = RNAEncoder(rna_input_dim, rna_dim, dropout=0.3)

        if combine_mode == "concat":
            # 2026-07-29: 레퍼런스 clinical 브랜치엔 Dropout(0.25)이 있는데 우리는 없었다는 차이를
            # 검증하는 ablation(train_light.py --clinical-dropout). 기본 0.0=기존 동작 보존.
            self.clinical_encoder = ClinicalEncoder(clinical_dim, age_mean, age_std, dropout=clinical_dropout,
                                                     sex_onehot=sex_onehot, use_margin=use_margin,
                                                     margin_stats=margin_stats, use_age_sex=use_age_sex,
                                                     use_staging=use_staging, stage_stats=stage_stats,
                                                     use_mutation=use_mutation, mutation_stats=mutation_stats)
            fused_dim = rna_dim + clinical_dim
            if risk_hidden_dim is None:
                self.risk_head = nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Linear(fused_dim, 1),
                )
            else:
                # 2026-07-29: 레퍼런스(tabular_survival.py::ClinicalRNASeqSurvivalModel.classifier)
                # 사양(LayerNorm→Dropout→Linear→GELU→Dropout→Linear, hidden=128)을 13번 항목에서
                # 그대로 이식했을 땐 negative(0.634→0.533)였는데, 그건 레퍼런스 원래 절대 폭
                # (rna=256/clinical=16/hidden=128)까지 다 같이 썼을 때였다 — 지금 우리 폭(rna=64/
                # clinical=4)에 맞춰 hidden도 같은 비율로 4배 축소(128/4=32)하고 dropout도
                # 0.4 대신 0.3(cfg.model.dropout 기본값과 동일 관례)으로 낮춰서 재검증
                # (train_light.py --risk-hidden-dim/--risk-dropout).
                self.risk_head = nn.Sequential(
                    nn.LayerNorm(fused_dim),
                    nn.Dropout(risk_dropout),
                    nn.Linear(fused_dim, risk_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(risk_dropout),
                    nn.Linear(risk_hidden_dim, 1),
                )
        else:
            self.risk_head = nn.Sequential(
                nn.LayerNorm(rna_dim),
                nn.Linear(rna_dim, 1),
            )
            if combine_mode == "cox_add":
                # raw z-score feature를 clinical_linear(bias 없음, zero-init)에 직접 넣는다 —
                # ClinicalEncoder(MLP) 경유는 ablation에서 internal -0.025로 확인되어 채택하지 않음
                # (models/vit_pma.py::ViT_PMA cox_add 분기와 동일 관례).
                self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
                self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))
                if use_margin:
                    m_mean, m_std = margin_stats
                    self.register_buffer("margin_mean", torch.tensor(m_mean, dtype=torch.float32))
                    self.register_buffer("margin_std", torch.tensor(m_std, dtype=torch.float32))
                if use_staging:
                    for field in STAGE_FIELDS:
                        mean, std = stage_stats[field]
                        short = _STAGE_BUFFER_NAMES[field]
                        self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                        self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
                if use_mutation:
                    for field in MUTATION_FIELDS:
                        mean, std = mutation_stats[field]
                        short = _MUTATION_BUFFER_NAMES[field]
                        self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                        self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
                raw_dim = ((2 if use_age_sex else 0) + (2 if use_margin else 0)
                           + (2 * len(STAGE_FIELDS) if use_staging else 0)
                           + (2 * len(MUTATION_FIELDS) if use_mutation else 0))
                self.clinical_linear = nn.Linear(raw_dim, 1, bias=False)
                nn.init.zeros_(self.clinical_linear.weight)
            else:  # film — cox_add ablation과 무관, ClinicalEncoder 경유 유지
                self.clinical_encoder = ClinicalEncoder(
                    clinical_dim, age_mean, age_std, use_margin=use_margin, margin_stats=margin_stats,
                    use_staging=use_staging, stage_stats=stage_stats, use_age_sex=use_age_sex,
                    use_mutation=use_mutation, mutation_stats=mutation_stats,
                )
                self.film_gamma = nn.Linear(clinical_dim, 1)
                self.film_beta = nn.Linear(clinical_dim, 1)
                nn.init.zeros_(self.film_gamma.weight)
                nn.init.constant_(self.film_gamma.bias, 1.0)  # γ≈1(항등) 근처에서 시작
                nn.init.zeros_(self.film_beta.weight)
                nn.init.zeros_(self.film_beta.bias)            # β≈0(항등) 근처에서 시작

    def _clinical_embed(self, age_years: torch.Tensor, sex_idx: torch.Tensor,
                         margin_ord: torch.Tensor | None = None,
                         stage_ord: dict[str, torch.Tensor] | None = None,
                         mutation_ord: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """film/cox_add 전용. cox_add는 raw z-score feature를 (1, raw_dim)으로 직접 반환하고
        (models/vit_pma.py::ViT_PMA._clinical_embed와 동일 관례), film은 ClinicalEncoder(MLP)
        임베딩 (1, D)을 반환한다."""
        if self.combine_mode == "cox_add":
            feats = []
            if self.use_age_sex:
                age_z = (age_years.float() - self.age_mean) / self.age_std
                feats += [age_z, sex_idx.float()]
            if self.use_margin:
                ordv = margin_ord.float()
                known = (ordv >= 0).float()
                z = torch.where(ordv >= 0, (ordv - self.margin_mean) / self.margin_std, torch.zeros_like(ordv))
                feats += [z, known]
            if self.use_staging:
                for field in STAGE_FIELDS:
                    short = _STAGE_BUFFER_NAMES[field]
                    ordv = stage_ord[field].float()
                    known = (ordv >= 0).float()
                    mean = getattr(self, f"{short}_mean")
                    std = getattr(self, f"{short}_std")
                    z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                    feats += [z, known]
            if self.use_mutation:
                for field in MUTATION_FIELDS:
                    short = _MUTATION_BUFFER_NAMES[field]
                    ordv = mutation_ord[field].float()
                    known = (ordv >= 0).float()
                    mean = getattr(self, f"{short}_mean")
                    std = getattr(self, f"{short}_std")
                    z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                    feats += [z, known]
            return torch.stack(feats, dim=-1).unsqueeze(0)  # (1, raw_dim)
        clinical_kwargs = {}
        if margin_ord is not None:
            clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
        if stage_ord is not None:
            clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
        if mutation_ord is not None:
            clinical_kwargs["mutation_ord"] = {k: v.unsqueeze(0) for k, v in mutation_ord.items()}
        return self.clinical_encoder(age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs)  # (1, D)

    def forward(self, age_years: torch.Tensor, sex_idx: torch.Tensor, rna: torch.Tensor,
                margin_ord: torch.Tensor | None = None,
                stage_ord: dict[str, torch.Tensor] | None = None,
                mutation_ord: dict[str, torch.Tensor] | None = None,
                return_components: bool = False):
        """
        Args:
            age_years:  () — 환자 나이(연 단위) 스칼라 텐서
            sex_idx:    () — encode_sex() 인덱스 스칼라 텐서 (0=male, 1=female)
            rna:        (G,) — 코호트 내부 z-score 정규화된 유전자 발현 벡터
            margin_ord: self.use_margin=True(--clinical-margin)일 때만 필요. () 스칼라 long —
                        encode_margin_value() 규약. combine_mode 무관하게 지원.
            stage_ord:  self.use_staging=True(--clinical-staging)일 때만 필요. {field: () 스칼라
                        long} — encode_stage_value() 규약. combine_mode 무관하게 지원.
            return_components: cox_add 전용. True면 risk 대신 {"rna":.., "clin":..} 항별 dict
                                반환(2026-09-02, scripts/diagnose_m7_branch_contrib.py용 —
                                gene-selection leakage 유무에 따라 clinical/RNA 기여도가 어떻게
                                바뀌는지 비교하기 위해 추가). concat/film에서 True로 주면
                                NotImplementedError.
        Returns:
            risk: (1,) (return_components=False) 또는 cox_add 항별 dict(각 (1,))
        """
        z_r = self.rna_encoder(rna.unsqueeze(0)).squeeze(0)  # (D,)

        if return_components and self.combine_mode != "cox_add":
            raise NotImplementedError("return_components는 combine_mode='cox_add'에서만 지원합니다.")

        if self.combine_mode == "concat":
            clinical_kwargs = {}
            if margin_ord is not None:
                clinical_kwargs["margin_ord"] = margin_ord.unsqueeze(0)
            if stage_ord is not None:
                clinical_kwargs["stage_ord"] = {k: v.unsqueeze(0) for k, v in stage_ord.items()}
            if mutation_ord is not None:
                clinical_kwargs["mutation_ord"] = {k: v.unsqueeze(0) for k, v in mutation_ord.items()}
            z_c = self.clinical_encoder(
                age_years.unsqueeze(0), sex_idx.unsqueeze(0), **clinical_kwargs
            ).squeeze(0)  # (D,)
            fused = torch.cat([z_c, z_r], dim=-1)                                                  # (2D,)
            return self.risk_head(fused.unsqueeze(0)).view(1)

        clin_embed = self._clinical_embed(age_years, sex_idx, margin_ord, stage_ord=stage_ord,
                                           mutation_ord=mutation_ord)  # (1, D)
        if self.combine_mode == "film":
            gamma = self.film_gamma(clin_embed).view(1)  # (1,)
            beta = self.film_beta(clin_embed).view(1)    # (1,)
            z_r_mod = gamma * z_r + beta                # (D,) — 전 차원에 균일 스케일/이동
            return self.risk_head(z_r_mod.unsqueeze(0)).view(1)

        # cox_add
        risk_rna = self.risk_head(z_r.unsqueeze(0)).view(1)
        risk_clin = self.clinical_linear(clin_embed).view(1)
        if return_components:
            return {"rna": risk_rna, "clin": risk_clin}
        return risk_rna + risk_clin
