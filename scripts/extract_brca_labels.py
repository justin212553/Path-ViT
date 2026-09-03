"""
TCGA-BRCA case-level OS(overall survival) + clinical(age/sex) 레이블을 GDC REST API에서 직접
추출한다. `data/extract_os_labels.py`(TCGA-PAAD)와 동일한 OS_time/OS_event 도출 규칙을
그대로 재현하되, BRCA는 로컬에 미리 받아둔 clinical.tsv가 없어 GDC 케이스 API로 직접 조회한다.

    OS_time  = vital_status Dead  -> demographic.days_to_death
               vital_status Alive -> max(follow_ups[].days_to_follow_up)  (BRCA는 diagnoses.
                                      days_to_last_follow_up이 항상 결측 — follow_ups 엔티티 사용)
    OS_event = Dead -> 1, Alive -> 0
    vital_status가 Alive/Dead가 아니거나 위 time이 결측인 case는 제외.

배경: "WSI 브랜치가 표본만 늘리면 공간 신호를 학습하는가"를 검증하기 위해 TCGA-PAAD(152명)보다
훨씬 큰 TCGA-BRCA(1098 case) 코호트를 보조로 준비하는 작업(findings_backlog.md 관련 논의 참조).
WSI feature는 별도로 `scripts/_download_brca_hf.py`가 HF Dataset(Dearcat/CPathPatchFeature)에서
UNI backbone(1024dim) feature + 패치 좌표를 받아온다.

[sex] GDC BRCA case의 demographic.gender가 현재 거의 전부 결측으로 조회된다(확인됨) — BRCA는
역학적으로 99% 이상 여성이므로 전부 "female"로 채운다(이 실험의 핵심은 WSI 단독 신호 검증이라
Clinical 분기의 sex 변수 결측은 결과에 미치는 영향이 미미하다고 판단).

출력:
    data/brca_clinical.csv   case_id, dataset, age_years, sex, OS_time, OS_event

사용법:
    python -m scripts.extract_brca_labels
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

GDC_CASES_API = "https://api.gdc.cancer.gov/cases"
OUT_PATH = Path("data/brca_clinical.csv")


def _fetch_cases() -> list[dict]:
    """2026-09-02: 처음엔 diagnoses.days_to_last_follow_up과 T/N/M을 한 쿼리에 같이 넣으면
    GDC가 버그를 낸다고 오판했었다(별도 쿼리로 나눠도 동일하게 깨짐 — 재확인 결과 원인이 아니었음).
    실제 원인은 TCGA-BRCA는 diagnoses.days_to_last_follow_up 필드 자체가 애초에 거의 항상
    null이라는 것(직접 GDC _mapping 조회+실측으로 확인: Alive 945명 중 1명만 채워짐). BRCA는
    follow-up 기록이 diagnoses가 아니라 별도 follow_ups 엔티티(1:N)에 들어있고, 거기 있는
    follow_ups.days_to_follow_up은 Alive 945명 중 944명이 채워져 있다. 그래서 한 쿼리로
    합쳐도 문제없이 다 받아오고, case당 follow_ups는 여러 번의 방문 기록이라 가장 최근(=최댓값)을
    OS_time으로 쓴다."""
    filters = {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-BRCA"}}
    fields = ",".join([
        "submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.age_at_index",
        "follow_ups.days_to_follow_up",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
    ])
    params = {"filters": json.dumps(filters), "fields": fields, "size": "2000", "format": "json"}
    url = GDC_CASES_API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data = json.load(resp)
    total = data["data"]["pagination"]["total"]
    hits = data["data"]["hits"]
    assert len(hits) == total, f"페이지네이션 필요: total={total}, 받은 건수={len(hits)}"
    return hits


def main():
    hits = _fetch_cases()
    print(f"GDC TCGA-BRCA case 수: {len(hits)}")

    records = []
    for h in hits:
        demo = h.get("demographic", {}) or {}
        diagnoses = h.get("diagnoses", [{}]) or [{}]
        follow_ups = h.get("follow_ups", []) or []
        vital_status = demo.get("vital_status")
        days_to_death = demo.get("days_to_death")
        followup_vals = [f.get("days_to_follow_up") for f in follow_ups if f.get("days_to_follow_up") is not None]
        days_to_last_follow_up = max(followup_vals) if followup_vals else None
        age = demo.get("age_at_index")

        if vital_status == "Dead":
            os_time, os_event = days_to_death, 1
        elif vital_status == "Alive":
            os_time, os_event = days_to_last_follow_up, 0
        else:
            os_time, os_event = None, None

        records.append({
            "case_id": h["submitter_id"],
            "dataset": "tcga_brca",
            "age_years": age,
            "sex": "female",  # 근거: 모듈 docstring 참조
            # 2026-09-02: T/N/M 추가 — GDC 필드 채움율 실측 T=1097/1098, N=1095/1098, M=1095/1098로
            # 매우 양호. tumor_grade는 유방암에서 이 표준 필드에 항상 None(다른 채점 체계라
            # GDC diagnoses.tumor_grade로 안 들어옴) — STAGE_FIELDS(models/clinical_encoder.py)가
            # 4필드를 기대하므로 자리만 채워둔다(항상 known_flag=0으로 안전하게 무시됨).
            # residual_disease(margin)는 실측 결과 BRCA 전체 0/1098 채워짐 — 아예 뺀다(못 씀).
            "ajcc_t": diagnoses[0].get("ajcc_pathologic_t"),
            "ajcc_n": diagnoses[0].get("ajcc_pathologic_n"),
            "ajcc_m": diagnoses[0].get("ajcc_pathologic_m"),
            "tumor_grade": None,
            "OS_time": os_time,
            "OS_event": os_event,
        })

    df = pd.DataFrame(records)
    n_total = len(df)
    df = df.dropna(subset=["OS_time", "age_years"]).reset_index(drop=True)
    df["OS_event"] = df["OS_event"].astype(int)
    n_dropped = n_total - len(df)
    print(f"OS_time/age 결측으로 {n_dropped}/{n_total} case 제외 -> 최종 {len(df)} case")
    print(f"event(Dead)={int(df['OS_event'].sum())}  censored(Alive)={len(df) - int(df['OS_event'].sum())}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
