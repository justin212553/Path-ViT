"""
scripts/train_hdp_multi_head.py로 학습한 6채널(neoplastic/inflammatory/connective/epithelial/
cellularity/nucleus_density) head를 우리 코호트(TCGA-PAAD/CPTAC-PDA)의 uni2native feature에
적용해, 각 채널의 환자 단위 평균 스칼라 하나만으로 생존 판별력(c-index)이 있는지 본다 —
scripts/diagnose_hdp_feature_signal.py와 동일한 관례(신경망 없이, 그 스칼라 자체를 risk score로
취급). 세포밀도/면역세포침윤/스트로마 같은 새 축 중 tumor content(기존, ~0.50-0.55)보다 나은
게 있는지 가장 싸게 먼저 거르는 단계 — 여기서 뭔가 나오면 그 채널만 골라 grid t-SNE(scripts/
viz_tumor_content_grid_tsne.py 패턴)까지 확장한다.

사용법:
    python -m scripts.diagnose_hdp_multi_head_signal --dataset cptac
    python -m scripts.diagnose_hdp_multi_head_signal --dataset tcga
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config
from data.dataset import WSISurvivalDataset, pdac_consistency_gene_ids
from train import _identity_collate
from utils.metrics import compute_survival_metrics
from scripts.train_hdp_multi_head import MultiHead, TARGET_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="uni2native")
    parser.add_argument("--head-path", type=str, default="data/hdp_multi_head.pt")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.head_path, map_location=device, weights_only=False)
    head = MultiHead(in_dim=ckpt["in_dim"], n_targets=len(ckpt["target_names"]), hidden_dim=ckpt["hidden_dim"]).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    head.requires_grad_(False)
    target_names = ckpt["target_names"]
    print(f"head 로드: {args.head_path} (best_val_loss={ckpt['best_val_loss']:.4f}, targets={target_names})")

    cfg = Config()
    rna_gene_ids = pdac_consistency_gene_ids(1500)
    ds = WSISurvivalDataset(
        cfg.data, dataset=args.dataset, split="all",
        with_clinical=True, with_staging=True, with_margin=True, with_mutation=True,
        with_rna=True, with_cnv=True, rna_gene_ids=rna_gene_ids, feature_backbone=args.backbone,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_identity_collate, num_workers=0)
    print(f"{args.dataset} N={len(ds)}, backbone={args.backbone}")

    rows = []
    with torch.no_grad():
        for patient_slides in loader:
            if not patient_slides:
                continue
            case_id = patient_slides[0]["case_id"]
            os_time = float(patient_slides[0]["OS_time"].item())
            os_event = int(patient_slides[0]["OS_event"].item())
            feats = torch.cat([s["features"] for s in patient_slides], dim=0).float()
            scores = np.empty((feats.shape[0], len(target_names)), dtype=np.float64)
            for i in range(0, feats.shape[0], args.batch_size):
                chunk = feats[i:i + args.batch_size].to(device)
                scores[i:i + args.batch_size] = head(chunk).cpu().numpy()
            row = {"case_id": case_id, "OS_time": os_time, "OS_event": os_event, "n_patches": feats.shape[0]}
            for j, name in enumerate(target_names):
                row[f"mean_{name}"] = scores[:, j].mean()
                row[f"std_{name}"] = scores[:, j].std()
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(f"data/hdp_multi_head_features_{args.dataset}.csv")
    df.to_csv(out_path, index=False)
    print(f"환자별 채널 요약 저장: {out_path}\n")

    times = df["OS_time"].to_numpy()
    events = df["OS_event"].to_numpy()
    print(f"=== {args.dataset}: 6채널 x (mean/std) 단독 판별력 (신경망 없음, N={len(df)}) ===")
    for name in target_names:
        for stat in ("mean", "std"):
            col = df[f"{stat}_{name}"].to_numpy()
            m_pos = compute_survival_metrics(col, times, events)
            m_neg = compute_survival_metrics(-col, times, events)
            c_pos, c_neg = m_pos["c_index"], m_neg["c_index"]
            best = m_pos if c_pos >= c_neg else m_neg
            best_c = max(c_pos, c_neg)
            direction = "+" if c_pos >= c_neg else "-"
            flag = "  <-- 참고할만함" if best_c > 0.58 else ""
            print(f"  {stat}_{name:20s} c_index={best_c:.4f} (raw={c_pos:.4f}/inv={c_neg:.4f}, 방향={direction}) "
                  f"HR={best['hr']:.3f} log_rank_p={best['log_rank_p']:.4f}{flag}")


if __name__ == "__main__":
    main()
