"""
2026-10-11: PORPOISE 공식 입력판(시스템 수준 비교)의 CPTAC external 입력에서 근사였던 CNV·mutation을
cBioPortal 실측값으로 교체한다.

배경: 2026-09-14 prepare_porpoise_cptac_official_geneset.py는 CPTAC에 공식 입력 형식의 데이터가 없다고 보고
CNV 82개를 전부 0(neutral), mutation 14개 중 10개를 0으로 근사했다. cBioPortal `pancreas_cptac_gdc`
(CPTAC GDC, 2025)에 TCGA 공식 입력(`paad_tcga_gistic`)과 같은 형식의 GISTIC 이산 CNV(-2~2)와 mutation(MAF)이
있다는 것을 확인 — 우리 CPTAC 136명 전원 CNV 보유, 135명 시퀀싱(1명 미시퀀싱 -> mutation 0).
- CNV: 82개 중 79개 실측. 없는 3개(CCAT1, LINC00977, PCAT2)는 lncRNA라 GISTIC 표에 없음 -> 0.
  옛 symbol 5개는 현재 이름으로 조회(FAM49B=CYRIB, LRRC6=DNAAF11, TMEM241=SLC35D4, TMEM75=LINC02912,
  TRMT12=TYW2). 환자당 샘플이 2개인 2명은 sample ID 정렬 첫 번째.
- mutation: cBioPortal MAF(silent 제외)에 해당 유전자 변이가 1개라도 있으면 1. 공식 TCGA 입력의 "RAS"는
  KRAS와 166명 전원 동일한 값이라 RAS = KRAS. 우리 GDC MAF 추출(data/mutations_cptac.csv)과
  KRAS/TP53/CDKN2A 98.5%, SMAD4 100% 일치.

사용법:
  1) (로컬, 인터넷 필요) python scripts/porpoise_cptac_cbio_cnv_mut.py fetch
       -> data/cptac_porpoise_official_cnv_mut_cbio.csv
  2) (HPC, repo 루트) python scripts/porpoise_cptac_cbio_cnv_mut.py patch
       -> porpoise/datasets_csv_mutsig/cptac_paad_external_cbio.csv.zip
          (기존 근사 CSV cptac_paad_external_clean.csv.zip의 _cnv/_mut 열만 교체, RNA 열과 행은 그대로)
  3) sbatch sbatch/eval_porpoise_official_n110_cbio_external_cptac_5seed_hpc.sh  (재학습 없이 평가만)
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "data" / "cptac_porpoise_official_cnv_mut_cbio.csv"
OFFICIAL_TCGA = _ROOT / "porpoise" / "datasets_csv_mutsig" / "tcga_paad_all_clean.csv.zip"
APPROX_CPTAC = _ROOT / "porpoise" / "datasets_csv_mutsig" / "cptac_paad_external_clean.csv.zip"
PATCHED_CPTAC = _ROOT / "porpoise" / "datasets_csv_mutsig" / "cptac_paad_external_cbio.csv.zip"
API = "https://www.cbioportal.org/api"
STUDY = "pancreas_cptac_gdc"
ALIAS_ENTREZ = {"FAM49B": 51571, "LRRC6": 23639, "TMEM241": 85019, "TMEM75": 641384, "TRMT12": 55039}


def official_genes():
    cols = pd.read_csv(OFFICIAL_TCGA, nrows=0).columns
    return [c[:-4] for c in cols if c.endswith("_cnv")], [c[:-4] for c in cols if c.endswith("_mut")]


def fetch():
    import requests
    cnv, mut = official_genes()
    syms = sorted(set(cnv + mut) - {"RAS"})
    genes = requests.post(f"{API}/genes/fetch", params={"geneIdType": "HUGO_GENE_SYMBOL"}, json=syms).json()
    s2e = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in genes}
    s2e.update(ALIAS_ENTREZ)
    cohort = sorted(c for c in (_ROOT / "data" / "cohort_n110.txt").read_text().split() if not c.startswith("TCGA"))

    e2cnv = {s2e[s]: s for s in cnv if s in s2e}
    cna = requests.post(f"{API}/molecular-profiles/{STUDY}_cna/molecular-data/fetch", params={"projection": "SUMMARY"},
                        json={"sampleListId": f"{STUDY}_cna", "entrezGeneIds": list(e2cnv)}).json()
    cna = pd.DataFrame([{"sample": d["sampleId"], "patient": d["patientId"], "gene": e2cnv[d["entrezGeneId"]],
                         "v": d["value"]} for d in cna])
    first = cna.groupby("patient")["sample"].min()
    cna = cna[cna["sample"] == cna["patient"].map(first)]
    cnv_wide = cna.pivot_table(index="patient", columns="gene", values="v", aggfunc="first")

    e2mut = {s2e[s]: s for s in mut if s in s2e}
    mu = requests.post(f"{API}/molecular-profiles/{STUDY}_mutations/mutations/fetch", params={"projection": "SUMMARY"},
                       json={"sampleListId": f"{STUDY}_sequenced", "entrezGeneIds": list(e2mut)}).json()
    mutated = {(d["patientId"], e2mut[d["entrezGeneId"]]) for d in mu}
    seq = {s.rsplit("-", 1)[0] for s in requests.get(f"{API}/sample-lists/{STUDY}_sequenced/sample-ids").json()}

    rows = []
    for p in cohort:
        r = {"case_id": p}
        for g in cnv:
            v = cnv_wide.loc[p, g] if (p in cnv_wide.index and g in cnv_wide.columns) else float("nan")
            r[f"{g}_cnv"] = 0 if pd.isna(v) else int(v)
        for g in mut:
            src = "KRAS" if g == "RAS" else g
            r[f"{g}_mut"] = int((p, src) in mutated)
        r["cbio_cnv_measured"] = int(p in cnv_wide.index)
        r["cbio_sequenced"] = int(p in seq)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    missing_cnv = sorted(set(cnv) - set(cnv_wide.columns))
    print(f"저장: {OUT} ({len(out)}명) — CNV 실측 {out.cbio_cnv_measured.sum()}명, 시퀀싱 {out.cbio_sequenced.sum()}명, "
          f"CNV 유전자 {len(cnv) - len(missing_cnv)}/{len(cnv)} (0으로 채운 것: {missing_cnv})")


def patch():
    src = pd.read_csv(APPROX_CPTAC)
    new = pd.read_csv(OUT).set_index("case_id")
    cols = [c for c in src.columns if c.endswith("_cnv") or c.endswith("_mut")]
    missing_cols = [c for c in cols if c not in new.columns]
    if missing_cols:
        raise ValueError(f"교체 표에 없는 열: {missing_cols}")
    hit = src["case_id"].isin(new.index)
    src.loc[hit, cols] = new.loc[src.loc[hit, "case_id"], cols].to_numpy()
    print(f"근사 CSV {src.case_id.nunique()}명 중 {src.loc[hit, 'case_id'].nunique()}명의 _cnv/_mut {len(cols)}열 교체 "
          f"(나머지 {src.loc[~hit, 'case_id'].nunique()}명은 우리 코호트 밖이라 근사값 유지, 분석에서는 쓰이지 않음)")
    tmp = PATCHED_CPTAC.with_suffix("")
    src.to_csv(tmp, index=False)
    with zipfile.ZipFile(PATCHED_CPTAC, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp, arcname=tmp.name)
    tmp.unlink()
    print(f"저장: {PATCHED_CPTAC}")


if __name__ == "__main__":
    {"fetch": fetch, "patch": patch}[sys.argv[1]]()
