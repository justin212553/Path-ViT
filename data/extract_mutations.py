"""
KRAS/TP53/SMAD4/CDKN2A(PDAC 4대 driver gene) 체성 변이 유무를 GDC의 open-access "Masked
Somatic Mutation"(ensemble MAF, 환자당 1파일) 데이터에서 뽑아 clinical CSV에 붙일 수 있는
형태로 저장한다.

[배경, 2026-09-02] WSI 쪽(HDP/PMA 계열) 신호가 신경망 구조를 아무리 바꿔도 안 나온다는 게
diagnose_hdp_checkpoint_weights.py/concat 아키텍처 실험까지 다 확인된 뒤, "그럼 WSI 말고 예후와
상관 있는 다른 정보가 있나" 질문에서 나온 후보. mutation은 margin/stage와 성격이 같은
저차원 이산값이라, 새 branch를 만들지 않고 STAGE_FIELDS/margin과 동일한 (값, known_flag)
관례로 clinical raw feature에 그대로 얹는다(models/clinical_encoder.py 확장은 다음 단계).

[방법]
  1. GDC REST API로 TCGA-PAAD/CPTAC-3(Pancreas) 각각의 "Masked Somatic Mutation" 파일 목록을
     case_id와 함께 조회한다(1케이스=1 ensemble MAF, 여러 caller를 GDC가 이미 병합·마스킹).
  2. 파일을 하나씩 gzip 스트리밍으로 받아(디스크에 안 남김) Hugo_Symbol이 4대 유전자 중
     하나이고 Variant_Classification이 침묵변이(Silent)가 아닌 행이 있으면 그 환자의 그
     유전자를 변이 양성(1)으로 표시한다. 파일에 아예 그 유전자 행이 없으면 wildtype(0).
  3. case_id별로 4개 유전자 binary flag를 만들어 저장.

사용법:
    python -m data.extract_mutations --dataset tcga
    python -m data.extract_mutations --dataset cptac
    python -m data.extract_mutations --dataset both
"""
import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_DATA_API = "https://api.gdc.cancer.gov/data"
GENES = ["KRAS", "TP53", "SMAD4", "CDKN2A"]
OUT_PATHS = {
    "tcga": _ROOT / "data" / "mutations_tcga.csv",
    "cptac": _ROOT / "data" / "mutations_cptac.csv",
}


def _list_files(project_id: str, primary_site: str | None = None) -> list[dict]:
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Masked Somatic Mutation"]}},
    ]
    if primary_site:
        content.append({"op": "in", "content": {"field": "cases.primary_site", "value": [primary_site]}})
    filters = {"op": "and", "content": content}
    files, frm, size = [], 0, 100
    while True:
        params = {
            "filters": json.dumps(filters), "size": str(size), "from": str(frm), "format": "JSON",
            "fields": "file_id,cases.submitter_id",
        }
        r = requests.get(GDC_FILES_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        for h in hits:
            for c in h.get("cases", []):
                files.append({"file_id": h["file_id"], "case_id": c["submitter_id"]})
        frm += size
        if frm >= data["pagination"]["total"]:
            break
    return files


def _mutated_genes_in_file(file_id: str) -> set[str]:
    r = requests.get(f"{GDC_DATA_API}/{file_id}", timeout=60)
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    mutated = set()
    header = None
    for line in io.StringIO(text):
        if line.startswith("#"):
            continue
        if header is None:
            header = line.rstrip("\n").split("\t")
            gene_idx = header.index("Hugo_Symbol")
            vc_idx = header.index("Variant_Classification")
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(gene_idx, vc_idx):
            continue
        gene, vc = parts[gene_idx], parts[vc_idx]
        if gene in GENES and vc != "Silent":
            mutated.add(gene)
    return mutated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    args = parser.parse_args()

    for ds in args.datasets.split(","):
        if ds == "tcga":
            files = _list_files("TCGA-PAAD")
        else:
            files = _list_files("CPTAC-3", primary_site="Pancreas")
        print(f"=== {ds}: {len(files)}개 MAF 파일 ===")

        rows = []
        for i, f in enumerate(files):
            try:
                mutated = _mutated_genes_in_file(f["file_id"])
            except Exception as e:
                print(f"  [경고] {f['case_id']}({f['file_id']}) 다운로드/파싱 실패: {e}")
                continue
            row = {"case_id": f["case_id"]}
            for g in GENES:
                row[f"{g}_mut"] = int(g in mutated)
            rows.append(row)
            print(f"  {i+1}/{len(files)} {f['case_id']}: {sorted(mutated) or '(none)'}", end="\r")
        print()

        df = pd.DataFrame(rows).drop_duplicates(subset="case_id")
        out_path = OUT_PATHS[ds]
        df.to_csv(out_path, index=False)
        print(f"  {len(df)}명 -> {out_path}")
        for g in GENES:
            rate = df[f"{g}_mut"].mean()
            print(f"    {g}: 변이율 {rate:.1%} ({int(df[f'{g}_mut'].sum())}/{len(df)})")


if __name__ == "__main__":
    main()
