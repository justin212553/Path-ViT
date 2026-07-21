"""
레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M7 코드를 그대로 가져와, 우리 데이터를 넣어
학습/평가한다 — "데이터 문제냐 모델 문제냐 평가 프로토콜 문제냐"를 가르기 위한 통제 실험
(findings_backlog.md 13번 항목 연장선). reference_repo/(git clone)의 실제 nn.Module
(ClinicalRNASeqSurvivalModel)과 cox_ph_loss/harrell_c_index를 직접 import해서 쓴다 — 우리 쪽
재구현 코드(models/clinical_rna_only.py 등)는 전혀 안 쓴다.

--protocol external(기본): 우리 세션 표준 관례대로 TCGA train -> CPTAC test 단일 방향 진짜 external.
    train/val case 목록은 WSISurvivalDataset(--dataset tcga)의 기존 6:2:2 split과 동일하게 맞춰서
    지금까지 우리가 돌린 모든 M7_EX/PMA_EX 실험과 같은 환자 집합으로 비교 가능하게 한다.
--protocol pooled: 레퍼런스가 실제로 헤드라인 성적을 낸 방식 그대로 재현 — TCGA+CPTAC를 하나로 합친
    뒤 sklearn train_test_split(stratify=dataset+event, test_size=0.2/0.25, random_state=42)로
    무작위 분할(M4_Train.ipynb 코드 그대로). age_mean/std도 레퍼런스와 동일하게 pooled train만으로
    계산한다. 이 프로토콜에서 헤드라인급(~0.70) 수치가 나온다면, 격차의 정체가 데이터나 모델이
    아니라 "pooled random split vs 진짜 external" 평가 프로토콜 차이임이 확정된다.

사용법:
    python -m scripts.reference_repro_m7 --protocol external --seed 42
    python -m scripts.reference_repro_m7 --protocol pooled --split-seed 42
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
_REF_ROOT = _ROOT / "reference_repo"

sys.path.insert(0, str(_ROOT))
from config import Config
from data.dataset import WSISurvivalDataset, CLINICAL_PATHS, OS_LABEL_PATHS, RNA_PATHS, literature_guided_gene_ids

sys.path.insert(0, str(_REF_ROOT))
from scripts.models.discrete_survival import cox_ph_loss, harrell_c_index  # noqa: E402  (reference_repo)
from scripts.models.tabular_survival import (  # noqa: E402  (reference_repo)
    TabularSurvivalConfig, ClinicalRNASeqSurvivalModel, build_optimizer,
)

_FPKM_UQ_CACHE_DIR = _ROOT / "scripts" / "_rna_fpkm_uq_cache"


def _build_rna_fpkm_uq_log2_table(dataset: str) -> pd.DataFrame:
    """data/extract_rna_clinical.py와 동일한 case 선정(Primary Tumor 필터, QC 플래그 제외)을
    그대로 재사용하되, tpm_unstranded 대신 레퍼런스가 실제로 쓴 log2(fpkm_uq_unstranded+1)을
    값으로 쓴다 — data/rna_{tcga,cptac}.csv(원본 파이프라인, TPM·로그 변환 없음)와의 차이를
    검증하기 위한 통제용(findings_backlog.md, RNA 전처리 재검토, 2026-07-21).
    z-score는 유전자별(컬럼별) 독립 연산이라 이후 특정 gene_ids로 서브셋해도 결과는 동일하다.
    원본 TSV 재파싱이 느려(TCGA 185 + CPTAC 247 파일) 결과를 캐시한다."""
    cache_path = _FPKM_UQ_CACHE_DIR / f"rna_{dataset}_fpkm_uq_log2_z.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path).set_index("case_id")

    from data.extract_rna_clinical import (
        RNA_ROOTS, CLINIC_ROOTS, NA_VALUES, _list_rna_files, _load_or_query_file_case_map,
        _load_qc_flagged_cases,
    )

    rna_root = RNA_ROOTS[dataset]
    files = _list_rna_files(rna_root)
    file_map = _load_or_query_file_case_map(dataset, rna_root, files["file_id"].tolist())
    merged = files.merge(file_map, on="file_id", how="inner")
    merged = merged[merged["sample_type"] == "Primary Tumor"]

    clinical = pd.read_csv(CLINIC_ROOTS[dataset], sep="\t", na_values=NA_VALUES)
    flagged = _load_qc_flagged_cases(rna_root, clinical)
    merged = merged[~merged["case_id"].isin(flagged)]

    def _read_fpkm_uq_log2(tsv_path) -> pd.Series:
        df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
        df = df[df["gene_type"] == "protein_coding"]
        vals = df.set_index("gene_id")["fpkm_uq_unstranded"].astype(float)
        return np.log2(vals + 1.0)

    print(f"[{dataset}] fpkm_uq_log2 원본 TSV {merged['case_id'].nunique()} case 재파싱 중 (캐시 없음, 시간 걸림)...")
    gene_series_by_case = {}
    for case_id, group in merged.groupby("case_id"):
        series_list = [_read_fpkm_uq_log2(p) for p in group["tsv_path"]]
        gene_series_by_case[case_id] = pd.concat(series_list, axis=1).mean(axis=1)

    rna_df = pd.DataFrame(gene_series_by_case).T
    rna_df.index.name = "case_id"
    z = (rna_df - rna_df.mean()) / rna_df.std(ddof=0).replace(0, 1.0)

    _FPKM_UQ_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    z.reset_index().to_csv(cache_path, index=False)
    return z


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _case_ids_for_split(cfg, dataset: str, split: str, rna_gene_ids: list[str]) -> list[str]:
    """WSISurvivalDataset의 기존 join/split 로직을 그대로 태워 case_id 목록만 뽑는다."""
    ds = WSISurvivalDataset(
        cfg, dataset=dataset, split=split,
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    return list(ds.cases)


def _pooled_case_pool(cfg, rna_gene_ids: list[str]) -> pd.DataFrame:
    """RNA+Clinical+OS+WSI를 전부 갖춘 TCGA+CPTAC 전체 후보 case pool을
    (dataset, case_id, OS_event) 테이블로 반환한다(레퍼런스 pooled split의 대상 모집단)."""
    ds = WSISurvivalDataset(
        cfg, dataset="both", split="all",
        with_clinical=True, with_rna=True, rna_gene_ids=rna_gene_ids,
    )
    case_df = ds.items.groupby("case_id").agg(dataset=("dataset", "first"), OS_event=("OS_event", "first"))
    return case_df.reset_index()


def _build_tensors(rows: pd.DataFrame, rna_gene_ids: list[str], age_mean: float, age_std: float,
                    rna_source: str = "tpm") -> dict[str, torch.Tensor]:
    """rows: columns [dataset, case_id] (여러 코호트가 섞여 있어도 됨, pooled 프로토콜용).
    rna_source="tpm"(기본): 원본 파이프라인(data/rna_{tcga,cptac}.csv, TPM 원본값 z-score).
    rna_source="fpkm_uq_log2": 레퍼런스와 동일하게 log2(fpkm_uq_unstranded+1) z-score
    (_build_rna_fpkm_uq_log2_table 참조, 검증용)."""
    rna_cache, clinical_cache, os_cache = {}, {}, {}
    for name in rows["dataset"].unique():
        rna_cache[name] = (
            _build_rna_fpkm_uq_log2_table(name) if rna_source == "fpkm_uq_log2"
            else pd.read_csv(RNA_PATHS[name]).set_index("case_id")
        )
        clinical_cache[name] = pd.read_csv(CLINICAL_PATHS[name]).set_index("case_id")
        os_cache[name] = pd.read_csv(OS_LABEL_PATHS[name]).set_index("case_id")

    rnaseq_list, age_list, sex_list, os_time_list, os_event_list = [], [], [], [], []
    for name, case_id in zip(rows["dataset"], rows["case_id"]):
        rnaseq_list.append(rna_cache[name].loc[case_id, rna_gene_ids].to_numpy(dtype="float32"))
        age_list.append(float(clinical_cache[name].loc[case_id, "age_years"]))
        sex_list.append(clinical_cache[name].loc[case_id, "sex"])
        os_time_list.append(float(os_cache[name].loc[case_id, "OS_time"]))
        os_event_list.append(float(os_cache[name].loc[case_id, "OS_event"]))

    age_z = (np.array(age_list, dtype="float32") - age_mean) / age_std
    sex_male = (np.array(sex_list) == "male").astype("float32")
    sex_female = (np.array(sex_list) == "female").astype("float32")

    return {
        "rnaseq_features": torch.tensor(np.stack(rnaseq_list)),
        "clinical_features": torch.tensor(np.stack([age_z, sex_male, sex_female], axis=1)),
        "os_time": torch.tensor(np.array(os_time_list, dtype="float32")),
        "os_event": torch.tensor(np.array(os_event_list, dtype="float32")),
    }


@torch.no_grad()
def _evaluate(model, batch: dict, device) -> float:
    model.eval()
    logits = model(batch["rnaseq_features"].to(device), batch["clinical_features"].to(device))["logits"]
    risks = logits.reshape(-1).cpu().numpy()
    return harrell_c_index(batch["os_time"].numpy(), batch["os_event"].numpy().astype(int), risks)


def _train_and_eval(train, val, test, rna_dim, args, device) -> tuple[float, float]:
    config = TabularSurvivalConfig(
        rnaseq_dim=rna_dim, clinical_dim=3,
        rnaseq_hidden_dim=256, rnaseq_embed_dim=256, clinical_embed_dim=16,
        fusion_hidden_dim=128, n_outputs=1,
        dropout=0.40, rnaseq_dropout=0.25, clinical_dropout=0.25,
    )
    model = ClinicalRNASeqSurvivalModel(config).to(device)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    # M7_Train.ipynb cell 3: ReduceLROnPlateau(mode="max", factor=0.5, patience=5, min_lr=1e-6) —
    # 처음 재현 때 빠뜨렸던 부분(2026-07-21 발견).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    n_train = train["rnaseq_features"].shape[0]
    best_val_c, best_state, epochs_since_improve = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            rnaseq = train["rnaseq_features"][idx].to(device)
            clinical = train["clinical_features"][idx].to(device)
            os_time = train["os_time"][idx].to(device)
            os_event = train["os_event"][idx].to(device)

            logits = model(rnaseq, clinical)["logits"]
            loss = cox_ph_loss(logits.reshape(-1), os_time, os_event)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_c = _evaluate(model, val, device)
        score = val_c if not np.isnan(val_c) else -1.0
        scheduler.step(score)
        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch:3d} | lr={lr_now:.2e} | loss={total_loss / max(n_batches, 1):.4f} | val_c_index={val_c:.4f}")

        if score > best_val_c:
            best_val_c, best_state, epochs_since_improve = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_c_index={best_val_c:.4f})")
                break

    model.load_state_dict(best_state)
    test_c = _evaluate(model, test, device)
    return best_val_c, test_c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=str, default="external", choices=["external", "pooled"])
    parser.add_argument("--seed", type=int, default=42, help="--protocol external: train/val split seed(cfg.data.seed)")
    parser.add_argument("--split-seed", type=int, default=42, help="--protocol pooled: sklearn train_test_split random_state(레퍼런스는 항상 42 고정)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rna-source", type=str, default="tpm", choices=["tpm", "fpkm_uq_log2"],
                         help="tpm(기본): 원본 파이프라인(TPM 원본값 z-score). fpkm_uq_log2: "
                              "레퍼런스와 동일하게 log2(fpkm_uq_unstranded+1) z-score(검증용).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    rna_gene_ids = literature_guided_gene_ids(1500)

    if args.protocol == "external":
        set_seed(args.seed)
        cfg.data.seed = args.seed
        train_ids = _case_ids_for_split(cfg.data, "tcga", "train", rna_gene_ids)
        val_ids   = _case_ids_for_split(cfg.data, "tcga", "val",   rna_gene_ids)
        test_ids  = _case_ids_for_split(cfg.data, "cptac", "all",  rna_gene_ids)
        print(f"[external] train(tcga)={len(train_ids)}  val(tcga)={len(val_ids)}  test(cptac)={len(test_ids)}")

        age_mean = pd.read_csv(CLINICAL_PATHS["tcga"])["age_years"].astype(float).mean()
        age_std  = pd.read_csv(CLINICAL_PATHS["tcga"])["age_years"].astype(float).std(ddof=0)

        train = _build_tensors(pd.DataFrame({"dataset": "tcga", "case_id": train_ids}), rna_gene_ids, age_mean, age_std, args.rna_source)
        val   = _build_tensors(pd.DataFrame({"dataset": "tcga", "case_id": val_ids}),   rna_gene_ids, age_mean, age_std, args.rna_source)
        test  = _build_tensors(pd.DataFrame({"dataset": "cptac", "case_id": test_ids}), rna_gene_ids, age_mean, age_std, args.rna_source)
        eval_label = "external_test_c_index(cptac)"

    else:  # pooled — 레퍼런스 M4_Train.ipynb의 split 로직 그대로 재현
        set_seed(args.seed)
        pool = _pooled_case_pool(cfg.data, rna_gene_ids)
        pool["stratify_group"] = pool["dataset"].astype(str) + "_event" + pool["OS_event"].astype(str)
        print(f"[pooled] candidate pool={len(pool)}  by dataset={pool['dataset'].value_counts().to_dict()}")

        train_valid_df, test_df = train_test_split(
            pool, test_size=0.2, random_state=args.split_seed, stratify=pool["stratify_group"])
        train_df, valid_df = train_test_split(
            train_valid_df, test_size=0.25, random_state=args.split_seed, stratify=train_valid_df["stratify_group"])
        print(f"[pooled] train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
        print("split x dataset (train/valid/test):")
        for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            print(f"  {name}: {df['dataset'].value_counts().to_dict()}")

        # age_years이 pool에 없으므로 clinical CSV에서 train_df case_id로 조회
        age_lookup = {}
        for name in pool["dataset"].unique():
            age_lookup[name] = pd.read_csv(CLINICAL_PATHS[name]).set_index("case_id")["age_years"]
        train_ages = [float(age_lookup[d].loc[c]) for d, c in zip(train_df["dataset"], train_df["case_id"])]
        age_mean = float(np.mean(train_ages))
        age_std = float(np.std(train_ages, ddof=0))
        age_std = age_std if age_std > 0 else 1.0
        print(f"[pooled] train age_mean={age_mean:.2f} age_std={age_std:.2f}")

        train = _build_tensors(train_df[["dataset", "case_id"]], rna_gene_ids, age_mean, age_std, args.rna_source)
        val   = _build_tensors(valid_df[["dataset", "case_id"]], rna_gene_ids, age_mean, age_std, args.rna_source)
        test  = _build_tensors(test_df[["dataset", "case_id"]],  rna_gene_ids, age_mean, age_std, args.rna_source)
        eval_label = "pooled_test_c_index"

    best_val_c, test_c = _train_and_eval(train, val, test, len(rna_gene_ids), args, device)
    print(f"\n=== RESULT (protocol={args.protocol}, seed={args.seed}, split_seed={args.split_seed}) ===")
    print(f"best_val_c_index={best_val_c:.4f}")
    print(f"{eval_label}={test_c:.4f}")


if __name__ == "__main__":
    main()
