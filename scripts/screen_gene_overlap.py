"""
후보 암종의 RNA-seq만 가볍게 받아서(WSI 없이) TCGA-PAAD literature_1500과 고분산 유전자
겹침 비율을 스크리닝한다 — "PDAC과 유전자 프로그램이 더 비슷한 암종을 BRCA 대신 pretrain
소스로 쓰면 낫지 않을까"라는 가설을 WSI 다운로드(비싸고 느림) 전에 싸게 검증하는 단계.

BRCA(scripts/extract_brca_rna.py + select_brca_rna_genes.py)로 이미 확인한 baseline:
  overlap 185/1500(12.3%), Jaccard 0.066 — 대부분 증식/면역/기질/상피 같은 범용 암 프로그램.
후보(소화기 선암 계열, PDAC과 조직학적으로 더 가까움): TCGA-COAD/STAD/ESCA/CHOL.

스크리닝이므로 case를 전부 받지 않고 --max-files로 표본을 제한한다(변동성 순위 추정엔
수백 개면 충분 — 정식으로 pretrain까지 갈 후보만 나중에 전체 다운로드).

출력:
    data/gene_overlap_screen/<project>/rna_raw_log2.csv
    data/gene_overlap_screen/<project>/overlap_summary.json

사용법:
    python -m scripts.screen_gene_overlap --project TCGA-COAD --max-files 200
    python -m scripts.screen_gene_overlap --project TCGA-STAD --project TCGA-ESCA --project TCGA-CHOL
"""
import argparse
import io
import json
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_DATA_API = "https://api.gdc.cancer.gov/data"
BATCH_SIZE = 100

PAAD_LITERATURE_1500_PATH = Path("data/rna_gene_selection/selected_genes_top_1500.csv")
OUT_ROOT = Path("data/gene_overlap_screen")


def _query_files(project_id: str, max_files: int | None) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
            {"op": "=", "content": {"field": "data_category", "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
        ],
    }
    fields = "file_id,file_name,cases.submitter_id,cases.samples.sample_type"
    params = {"filters": json.dumps(filters), "fields": fields, "size": "3000", "format": "json"}
    url = GDC_FILES_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    hits = data["data"]["hits"]
    primary = [h for h in hits if (h.get("cases") or [{}])[0].get("samples", [{}])[0].get("sample_type") == "Primary Tumor"]
    print(f"[{project_id}] STAR-Counts 파일 {len(hits)}개, Primary Tumor {len(primary)}개")
    if max_files:
        primary = primary[:max_files]
        print(f"[{project_id}] 스크리닝용으로 {len(primary)}개만 사용")
    return primary


def _download_batch(raw_root: Path, file_ids: list[str]) -> None:
    req = urllib.request.Request(
        GDC_DATA_API, data=json.dumps({"ids": file_ids}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(raw_root)


def _download_all(raw_root: Path, files: list[dict]) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    remaining = [h for h in files if not any(raw_root.glob(f"{h['file_id']}/*.tsv"))]
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        _download_batch(raw_root, [h["file_id"] for h in batch])
        print(f"  {min(i + BATCH_SIZE, len(remaining))}/{len(remaining)} 다운로드 완료")


def _read_fpkm_uq_log2(tsv_path: Path) -> pd.Series:
    """scripts/extract_brca_rna.py::_read_fpkm_uq_log2와 동일 로직."""
    df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
    df = df[df["gene_type"] == "protein_coding"]
    fpkm_uq = df.set_index("gene_id")["fpkm_uq_unstranded"].astype(float)
    return np.log2(fpkm_uq + 1.0)


def screen_project(project_id: str, max_files: int, top_n: int) -> dict:
    short = project_id.replace("TCGA-", "").lower()
    out_dir = OUT_ROOT / short
    raw_root = out_dir / "raw"

    files = _query_files(project_id, max_files)
    _download_all(raw_root, files)

    case_series = {}
    for h in files:
        case_id = h["cases"][0]["submitter_id"]
        matches = list((raw_root / h["file_id"]).glob("*.tsv"))
        if not matches:
            continue
        case_series[case_id] = _read_fpkm_uq_log2(matches[0])

    rna_df = pd.DataFrame(case_series).T
    rna_df.index.name = "case_id"
    out_dir.mkdir(parents=True, exist_ok=True)
    rna_df.reset_index().to_csv(out_dir / "rna_raw_log2.csv", index=False)
    print(f"[{project_id}] raw log2 case x gene 저장: {rna_df.shape}")

    variance = rna_df.var(axis=0, ddof=0).sort_values(ascending=False)
    top_genes = set(variance.head(top_n).index)

    paad_genes = set(pd.read_csv(PAAD_LITERATURE_1500_PATH)["gene_id"])
    overlap = top_genes & paad_genes
    jaccard = len(overlap) / len(top_genes | paad_genes)

    summary = {
        "project": project_id,
        "n_cases": int(rna_df.shape[0]),
        "n_genes_total": int(rna_df.shape[1]),
        "top_n": top_n,
        "overlap_with_paad_literature_1500": len(overlap),
        "overlap_frac": len(overlap) / top_n,
        "jaccard": jaccard,
    }
    with open(out_dir / "overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{project_id}] 결과: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", required=True,
                         help="GDC project id, 여러 번 지정 가능 (예: --project TCGA-COAD --project TCGA-STAD)")
    parser.add_argument("--max-files", type=int, default=200,
                         help="스크리닝용 표본 상한 (기본 200 — 분산 순위 추정엔 충분, 전체 다운로드 아님)")
    parser.add_argument("--top-n", type=int, default=1500)
    args = parser.parse_args()

    results = [screen_project(p, args.max_files, args.top_n) for p in args.project]

    print("\n=== 스크리닝 결과 요약 (BRCA baseline: overlap 185/1500=12.3%, Jaccard 0.066) ===")
    for r in sorted(results, key=lambda x: -x["overlap_frac"]):
        print(f"  {r['project']:12s} n={r['n_cases']:4d}  overlap={r['overlap_with_paad_literature_1500']:4d}/{r['top_n']} "
              f"({r['overlap_frac']*100:.1f}%)  jaccard={r['jaccard']:.4f}")


if __name__ == "__main__":
    main()
