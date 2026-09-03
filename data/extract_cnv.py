"""
pathway8(PDAC_LITERATURE_GENE_SETS, 163개 유전자) 범위의 카피수 변이(CNV, Copy Number
Variation)를 GDC의 "Gene Level Copy Number" 데이터에서 뽑아 저장한다.

[배경, 2026-09-03] 공식 PORPOISE 저장소(sota/PORPOISE) 코드를 직접 열어 확인한 결과, 그쪽
유전체 입력은 RNA-seq(사실상 선별 없이 거의 전체 유전자, README의 "MAD 상위 2000개"는 실제
배포 데이터와 다름 — 직접 확인)+CNV(~2500유전자)+소수 driver 유전자 mutation 조합이었다.
저희는 RNA-seq/mutation은 이미 있지만 CNV는 한 번도 써본 적이 없는 완전히 새로운 모달리티라,
PAAD에서 WSI+유전체 융합이 유의했던 원 논문 결과를 재현하려면 이게 빠진 조각일 가능성이 있다
(사용자 지시: "우선 pathway8에서 먼저 진행해보지" — 전체 유전자가 아니라 이미 검증된 163개
문헌 큐레이션 유전자 범위로 한정).

[CNV란] 유전자가 세포 안에 몇 카피 있는지(정상=2, 증폭되면 3+, 결실되면 0~1) — RNA 발현량과는
다른, DNA 구조 차원의 독립적인 신호(예: 췌장암 4대 driver 중 SMAD4는 point mutation보다
결실로 없어지는 경우가 흔함).

[GDC 데이터 선택]
  - data_type="Gene Level Copy Number": segment(염색체 구간)를 유전자에 이미 매핑해둔 산출물이라
    직접 genomic interval overlap을 계산할 필요가 없다.
  - analysis.workflow_type="AscatNGS": TCGA-PAAD/CPTAC-3(Pancreas) 둘 다 공통으로 존재하는
    유일한 파이프라인(ASCAT2/ASCAT3/ABSOLUTE LiftOver는 CPTAC에 없음, 직접 GDC API로 확인) —
    두 코호트 간 CNV 산출 알고리즘이 다르면 그 자체가 배치 효과가 되므로 반드시 통일해야 한다.
  - 파일 컬럼: gene_id, gene_name, chromosome, start, end, copy_number, min_copy_number,
    max_copy_number — gene_id가 common_genes.csv와 정확히 같은 GENCODE 버전 표기(직접 대조
    확인, 예: KRAS=ENSG00000133703.13 양쪽 동일)라 별도 매핑 없이 바로 사용 가능.

출력:
    data/cnv_tcga.csv, data/cnv_cptac.csv
        case_id, <pathway8 163개 gene_id 컬럼(정수 copy_number, 결측=호출 안 됨은 NaN)>

사용법:
    python -m data.extract_cnv --datasets tcga,cptac
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import pathway_flat_gene_ids

GDC_FILES_API = "https://api.gdc.cancer.gov/files"
GDC_DATA_API = "https://api.gdc.cancer.gov/data"
WORKFLOW_TYPE = "AscatNGS"
OUT_PATHS = {
    "tcga": _ROOT / "data" / "cnv_tcga.csv",
    "cptac": _ROOT / "data" / "cnv_cptac.csv",
}


def _list_files(project_id: str, primary_site: str | None = None) -> list[dict]:
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Level Copy Number"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": [WORKFLOW_TYPE]}},
    ]
    if primary_site:
        content.append({"op": "in", "content": {"field": "cases.primary_site", "value": [primary_site]}})
    filters = {"op": "and", "content": content}
    files, frm, size = [], 0, 100
    while True:
        params = {
            "filters": json.dumps(filters), "size": str(size), "from": str(frm), "format": "JSON",
            "fields": "file_id,cases.submitter_id,cases.samples.sample_type",
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


def _gene_cnv_in_file(file_id: str, target_genes: set[str]) -> dict[str, int]:
    r = requests.get(f"{GDC_DATA_API}/{file_id}", timeout=60)
    r.raise_for_status()
    text = r.content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    header = lines[0].split("\t")
    gid_idx = header.index("gene_id")
    cn_idx = header.index("copy_number")
    out = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(gid_idx, cn_idx):
            continue
        gid = parts[gid_idx]
        if gid not in target_genes:
            continue
        cn = parts[cn_idx]
        if cn:
            out[gid] = int(cn)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=str, default="tcga,cptac")
    args = parser.parse_args()

    target_genes = set(pathway_flat_gene_ids())
    print(f"대상 유전자(pathway8): {len(target_genes)}개")

    for ds in args.datasets.split(","):
        if ds == "tcga":
            files = _list_files("TCGA-PAAD")
        else:
            files = _list_files("CPTAC-3", primary_site="Pancreas")
        # case당 파일 1개만 남기기(같은 case가 여러 sample/aliquot으로 중복될 수 있음) — 첫 파일 유지
        seen_cases, dedup_files = set(), []
        for f in files:
            if f["case_id"] in seen_cases:
                continue
            seen_cases.add(f["case_id"])
            dedup_files.append(f)
        print(f"=== {ds}: {len(files)}개 파일 -> {len(dedup_files)}명(중복 case 제거) ===")

        rows = []
        for i, f in enumerate(dedup_files):
            try:
                gene_cnv = _gene_cnv_in_file(f["file_id"], target_genes)
            except Exception as e:
                print(f"  [경고] {f['case_id']}({f['file_id']}) 다운로드/파싱 실패: {e}")
                continue
            row = {"case_id": f["case_id"], **gene_cnv}
            rows.append(row)
            print(f"  {i+1}/{len(dedup_files)} {f['case_id']}: {len(gene_cnv)}/{len(target_genes)} 유전자 호출됨", end="\r")
        print()

        df = pd.DataFrame(rows).drop_duplicates(subset="case_id")
        # 순서 고정: case_id 먼저, 그 다음 target_genes 정렬 순서(없는 유전자 컬럼은 전부 NaN으로 채움)
        for g in sorted(target_genes):
            if g not in df.columns:
                df[g] = pd.NA
        df = df[["case_id"] + sorted(target_genes)]
        out_path = OUT_PATHS[ds]
        df.to_csv(out_path, index=False)
        n_called = df[sorted(target_genes)].notna().sum().sum()
        n_total = len(df) * len(target_genes)
        print(f"  {len(df)}명 -> {out_path}  (유전자 호출률 {n_called}/{n_total} = {n_called/n_total:.1%})")


if __name__ == "__main__":
    main()
