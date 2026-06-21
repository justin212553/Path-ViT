"""
CAMELYON17 eval WSI → 패치 사전 추출 스크립트 (lesion annotation 있는 노드만)

사용법:
    python -m data.preprocess_eval

입력:
    data/wsi_eval/patient_XXX/patient_XXX_node_Y.tif
    data/lesion_annotations/patient_XXX_node_Y.xml  ← 이 파일이 있는 노드만 처리

출력:
    data/patches/
        patient_XXX_node_Y/
            r0000_c0045.png
            ...
        patch_index.csv    # slide_id, filename, row, col, patch_label (0/1)
"""
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import openslide

# ── 설정 ──────────────────────────────────────────────────────────────────────
EVAL_DIR         = Path("./data/wsi_eval")
ANNO_DIR         = Path("./data/lesion_annotations")
OUT_DIR          = Path("./data/patches")
PATCH_SIZE        = 256
PATCH_LEVEL       = 0
TISSUE_THRESHOLD  = 0.5
NUM_WORKERS       = 16
# ─────────────────────────────────────────────────────────────────────────────


def _load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def send_slack(message: str):
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        data = json.dumps({"text": message}).encode()
        req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Slack] 알림 전송 실패: {e}")


def _load_annotations(xml_path: Path) -> list:
    polygons = []
    root_el = ET.parse(str(xml_path)).getroot()
    for annotation in root_el.findall(".//Annotation"):
        coords = sorted(
            annotation.findall(".//Coordinate"),
            key=lambda c: int(c.get("Order", 0)),
        )
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in coords]
        if pts:
            polygons.append(np.array(pts, dtype=np.float64))
    return polygons


def _patch_overlap_fraction(polygons: list, x: int, y: int) -> float:
    from matplotlib.path import Path as MplPath
    n = 8
    xs = np.linspace(x + 0.5, x + PATCH_SIZE - 0.5, n)
    ys = np.linspace(y + 0.5, y + PATCH_SIZE - 0.5, n)
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(points), dtype=bool)
    for poly in polygons:
        inside |= MplPath(poly).contains_points(points)
    return float(inside.mean())


def _extract_slide(slide_path: Path, xml_path: Path):
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.level_dimensions[PATCH_LEVEL]
    polygons = _load_annotations(xml_path)

    patches, coords, labels = [], [], []
    for y in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
        for x in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
            patch = slide.read_region(
                (x, y), PATCH_LEVEL, (PATCH_SIZE, PATCH_SIZE)
            ).convert("RGB")

            sat = np.array(patch.convert("HSV"))[:, :, 1].mean() / 255.0
            if sat < TISSUE_THRESHOLD:
                continue

            patches.append(patch)
            coords.append([y // PATCH_SIZE, x // PATCH_SIZE])
            ratio = _patch_overlap_fraction(polygons, x, y)
            labels.append(1 if ratio > 0 else 0)

    slide.close()
    return patches, coords, labels


def _save_patches(slide_dir: Path, patches, coords, labels):
    slide_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for patch, (row, col), label in zip(patches, coords, labels):
        fname = f"r{row:04d}_c{col:04d}.png"
        patch.save(slide_dir / fname, format="PNG")
        records.append({
            "filename":    str(slide_dir / fname),
            "row":         row,
            "col":         col,
            "patch_label": label,
        })
    return records


def _build_slide_list() -> list:
    """lesion annotation XML이 존재하는 wsi_eval 노드만 반환."""
    slides = []
    for patient_dir in sorted(EVAL_DIR.glob("patient_*")):
        for tif in sorted(patient_dir.glob("patient_*_node_*.tif")):
            xml_path = ANNO_DIR / f"{tif.stem}.xml"
            if not xml_path.exists():
                continue
            slides.append({
                "slide_id":   tif.stem,
                "slide_path": tif,
                "xml_path":   xml_path,
            })
    return slides


def _process_slide(info):
    slide_dir = OUT_DIR / info["slide_id"]

    try:
        polygons = _load_annotations(info["xml_path"])

        if slide_dir.exists():
            # 패치 파일은 재사용, 라벨만 XML에서 재계산
            patch_records = []
            for png in sorted(slide_dir.glob("*.png")):
                parts = png.stem.split("_")
                row = int(parts[0][1:])
                col = int(parts[1][1:])
                x, y = col * PATCH_SIZE, row * PATCH_SIZE
                ratio = _patch_overlap_fraction(polygons, x, y)
                patch_records.append({
                    "slide_id":    info["slide_id"],
                    "filename":    str(png),
                    "row":         row,
                    "col":         col,
                    "patch_label": 1 if ratio > 0 else 0,
                })
            return info["slide_id"], patch_records, True  # patches reused

        patches, coords, labels = _extract_slide(
            info["slide_path"], info["xml_path"]
        )
        recs = _save_patches(slide_dir, patches, coords, labels)
        for r in recs:
            r["slide_id"] = info["slide_id"]
        return info["slide_id"], recs, False
    except Exception as exc:
        print(f"ERROR {info['slide_id']}: {exc}")
        return None, [], False


def main():
    _load_env()
    start_time = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slides = _build_slide_list()
    if not slides:
        print(f"처리할 슬라이드 없음 — {EVAL_DIR} 와 {ANNO_DIR} 경로를 확인하세요.")
        send_slack(":warning: *CAMELYON17 eval 전처리* — 처리할 슬라이드 없음")
        return

    print(f"처리 대상: {len(slides)}개 슬라이드 (annotation 있는 노드만)")

    patch_records = []

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(_process_slide, slides)
        if use_tqdm:
            results = tqdm(results, total=len(slides), desc="Extracting eval patches", unit="slide")

        for slide_id, patch_recs, skipped in results:
            if slide_id is None:
                continue
            if skipped:
                print(f"reuse patches, recompute labels: {slide_id}")
            patch_records.extend(patch_recs)

    pd.DataFrame(patch_records).to_csv(OUT_DIR / "patch_index.csv", index=False)
    print(f"완료: {len(patch_records)}개 패치 → {OUT_DIR}")

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    send_slack(
        f":white_check_mark: *CAMELYON17 eval 전처리 완료*\n"
        f"> 슬라이드: {len(slides)}개 | 패치: {len(patch_records)}개\n"
        f"> 소요 시간: {h}h {m}m {s}s"
    )


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except Exception as e:
        _load_env()
        send_slack(f":x: *CAMELYON17 eval 전처리 에러*\n```{type(e).__name__}: {e}```")
        raise
