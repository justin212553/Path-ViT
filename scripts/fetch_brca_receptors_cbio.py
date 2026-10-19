"""
2026-10-18(D3, SciRep Q5a): TCGA-BRCA ER/PR/HER2 수용체 상태를 받아 정리한다.

출처: GDC BCR Biotab 임상 파일 nationwidechildrens.org_clinical_patient_brca.txt(open access, 1,097명).
처음에는 cBioPortal brca_tcga(Firehose Legacy)를 썼으나 우리 코호트 1,036명 중 45명이 아예 없어서
GDC 원본으로 바꿨다(cBioPortal Legacy판도 이 파일에서 만들어진 것). 파일은 없으면 GDC API로 받는다.
파일 이름은 처음 만든 이름(fetch_brca_receptors_cbio.py)을 그대로 둔다.

판정 규칙(2026-10-11 TODO Step 6에서 정한 것):
  ER, PR: er_status_by_ihc / pr_status_by_ihc 가 Positive -> 1, Negative -> 0, 그 외(Indeterminate,
          [Not Evaluated], [Not Available] 등) -> 결측
  HER2:   IHC 3+ 또는 FISH Positive -> 1
          FISH Negative -> 0, 또는 IHC 0/1+ 이고 FISH가 Positive가 아님 -> 0
          IHC 2+ 이고 결정적인 FISH 결과 없음 -> 결측
          IHC·FISH 모두 없으면 결측
  IHC 점수는 her2_ihc_score(0~3)를 우선, 없으면 her2_status_by_ihc(Positive/Negative/Equivocal) 문구로 대신한다.

출력:
  data/brca_receptors_raw.csv  원값(환자별, 쓰는 열만)
  data/brca_receptors.csv      case_id, er, pr, her2 (1/0/빈칸)
사용법: python -m scripts.fetch_brca_receptors_cbio
"""
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BIOTAB = ROOT / "data" / "gdc_biotab_brca" / "nationwidechildrens.org_clinical_patient_brca.txt"
BIOTAB_FILE_ID = "8162d394-8b64-4da2-9f5b-d164c54b9608"  # GDC files API로 찾은 TCGA-BRCA BCR Biotab patient 파일
COLS = ["er_status_by_ihc", "pr_status_by_ihc", "her2_status_by_ihc", "her2_ihc_score", "her2_fish_status"]


def _s(v):
    return v.strip() if isinstance(v, str) else ""


def ihc_score(score, ihc_text):
    s = _s(score).rstrip("+")
    if s in ("0", "1", "2", "3"):
        return "0" if s == "0" else s + "+"
    return {"positive": "3+", "negative": "0", "equivocal": "2+"}.get(_s(ihc_text).lower())


def her2_call(score, ihc_text, fish):
    ihc = ihc_score(score, ihc_text)
    f = _s(fish).lower()
    if ihc == "3+" or f == "positive":
        return 1
    if f == "negative":
        return 0
    if ihc in ("0", "1+"):
        return 0
    return None  # 2+ without a decisive FISH, or nothing at all


def binary(v):
    return {"positive": 1, "negative": 0}.get(_s(v).lower())


def main():
    if not BIOTAB.exists():
        BIOTAB.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://api.gdc.cancer.gov/data/{BIOTAB_FILE_ID}", BIOTAB)
    # Biotab은 머리글 3줄(열 이름, 긴 이름, CDE id) — 첫 줄만 열 이름으로 쓰고 다음 두 줄은 버린다
    df = pd.read_csv(BIOTAB, sep="\t", dtype=str).iloc[2:]
    df = df.rename(columns={"bcr_patient_barcode": "case_id"})[["case_id"] + COLS].reset_index(drop=True)
    df.to_csv(ROOT / "data" / "brca_receptors_raw.csv", index=False)

    out = pd.DataFrame({"case_id": df["case_id"]})
    out["er"] = df["er_status_by_ihc"].map(binary)
    out["pr"] = df["pr_status_by_ihc"].map(binary)
    out["her2"] = [her2_call(s, t, f) for s, t, f in zip(df["her2_ihc_score"], df["her2_status_by_ihc"], df["her2_fish_status"])]
    out.to_csv(ROOT / "data" / "brca_receptors.csv", index=False)
    print(f"GDC BCR Biotab: 환자 {len(out)}명 -> data/brca_receptors.csv")
    for c in COLS:
        print(f"  {c}: {df[c].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
