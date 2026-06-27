"""
CAMELYON17 WSI → 패치 사전 추출 스크립트

사용법:
    python -m data.preprocess

출력 구조:
    <OUT_DIR>/
        patches/
            patient_000_node_0/
                r0000_c0045.jpg
                ...
        slide_index.csv    # slide_id, label, center_id
        patch_index.csv    # slide_id, filename, row, col, patch_label
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import openslide
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT              = Path("./data/")
OUT_DIR           = Path("./data/patches")
PATCH_SIZE        = 256
PATCH_LEVEL       = 0
TISSUE_THRESHOLD  = 0.5
OVERLAP_THRESHOLD = 0.5
NUM_WORKERS       = 8
BLOCK_SIZE        = 4096  # PATCH_SIZE의 배수. read_region 호출 횟수를 줄이기 위한 일괄 읽기 단위
JPEG_QUALITY      = 90
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_TO_LABEL = {
    "pN0": 0, "pN0(i+)": 0,
    "pN1mi": 1, "pN1": 1, "pN2": 1,
}


def _load_stage_map(root: Path) -> dict:
    df = pd.read_csv(root / "stage_labels.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    rows = df[df["patient"].str.endswith(".zip")].copy()
    rows["patient"] = rows["patient"].str.replace(".zip", "", regex=False)
    return dict(zip(rows["patient"], rows["stage"]))


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


def _extract_slide(slide_path: Path, xml_path):
    """
    BLOCK_SIZE 단위로 read_region을 호출해 호출 횟수를 줄이고, 블록 내 모든
    패치의 tissue saturation을 한 번의 reshape+mean으로 벡터화해서 계산한다.

    Returns:
        patches:  List[PIL.Image]
        coords:   List[[row, col]]
        labels:   List[int]   (-1 = annotation 없음)
    """
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.level_dimensions[PATCH_LEVEL]
    polygons = _load_annotations(xml_path) if xml_path is not None else []

    patches, coords, labels = [], [], []
    for by in range(0, h - PATCH_SIZE + 1, BLOCK_SIZE):
        bh = min(BLOCK_SIZE, h - by)
        bh -= bh % PATCH_SIZE
        if bh <= 0:
            continue

        for bx in range(0, w - PATCH_SIZE + 1, BLOCK_SIZE):
            bw = min(BLOCK_SIZE, w - bx)
            bw -= bw % PATCH_SIZE
            if bw <= 0:
                continue

            block = slide.read_region((bx, by), PATCH_LEVEL, (bw, bh)).convert("RGB")
            block_arr = np.array(block)
            sat = np.array(block.convert("HSV"))[:, :, 1].astype(np.float32) / 255.0

            n_rows, n_cols = bh // PATCH_SIZE, bw // PATCH_SIZE
            sat_means = sat.reshape(n_rows, PATCH_SIZE, n_cols, PATCH_SIZE).mean(axis=(1, 3))

            for pr in range(n_rows):
                for pc in range(n_cols):
                    if sat_means[pr, pc] < TISSUE_THRESHOLD:
                        continue

                    y, x = by + pr * PATCH_SIZE, bx + pc * PATCH_SIZE
                    patch_arr = block_arr[
                        pr * PATCH_SIZE:(pr + 1) * PATCH_SIZE,
                        pc * PATCH_SIZE:(pc + 1) * PATCH_SIZE,
                    ]
                    patches.append(Image.fromarray(patch_arr))
                    coords.append([y // PATCH_SIZE, x // PATCH_SIZE])
                    if polygons:
                        ratio = _patch_overlap_fraction(polygons, x, y)
                        labels.append(1 if ratio >= OVERLAP_THRESHOLD else 0)
                    else:
                        labels.append(-1)

    slide.close()

    return patches, coords, labels


def _save_patches(slide_dir: Path, patches, coords, labels):
    """패치를 JPEG로 저장하고 patch_index 레코드 목록을 반환."""
    if not patches:
        return []
    slide_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for patch, (row, col), label in zip(patches, coords, labels):
        fname = f"r{row:04d}_c{col:04d}.jpg"
        patch.save(slide_dir / fname, format="JPEG", quality=JPEG_QUALITY)
        records.append({
            "filename":    str(slide_dir / fname),
            "row":         row,
            "col":         col,
            "patch_label": label,
        })
    return records


def _build_slide_list(stage_map: dict) -> list:
    slides = []
    for patient_dir in sorted((ROOT / "wsi_train").glob("patient_*")):
        if not patient_dir.is_dir():
            continue
        patient_num = int(patient_dir.name.split("_")[1])
        center_id = patient_num // 20
        for tif in sorted(patient_dir.glob("patient_*_node_*.tif")):
            patient_id = "_".join(tif.stem.split("_")[:2])
            label = _STAGE_TO_LABEL.get(stage_map.get(patient_id), -1)
            if label < 0:
                continue
            xml_path = ROOT / "lesion_annotations" / f"{tif.stem}.xml"
            slides.append({
                "slide_id":   tif.stem,
                "slide_path": tif,
                "xml_path":   xml_path if xml_path.exists() else None,
                "center_id":  center_id,
                "label":      label,
            })
    return slides


def _process_slide(info):
    """
    단일 슬라이드를 처리하는 워커 함수.
    Returns (slide_record | None, patch_records)
    """
    slide_dir = OUT_DIR / info["slide_id"]

    existing = (
        sorted(slide_dir.glob("*.jpg")) + sorted(slide_dir.glob("*.png"))
        if slide_dir.exists() else []
    )
    if existing:
        patch_records = []
        for img_path in existing:
            parts = img_path.stem.split("_")
            row = int(parts[0][1:])
            col = int(parts[1][1:])
            patch_records.append({
                "slide_id":    info["slide_id"],
                "filename":    str(img_path),
                "row":         row,
                "col":         col,
                "patch_label": -1,
            })
        slide_record = {
            "slide_id":  info["slide_id"],
            "label":     info["label"],
            "center_id": info["center_id"],
        }
        return slide_record, patch_records, True  # skipped=True

    try:
        patches, coords, labels = _extract_slide(
            info["slide_path"], info["xml_path"]
        )
        recs = _save_patches(slide_dir, patches, coords, labels)
        if not recs:
            print(f"WARN no patches extracted: {info['slide_id']}")
        for r in recs:
            r["slide_id"] = info["slide_id"]
        slide_record = {
            "slide_id":  info["slide_id"],
            "label":     info["label"],
            "center_id": info["center_id"],
        }
        return slide_record, recs, False
    except Exception as exc:
        print(f"ERROR {info['slide_id']}: {exc}")
        return None, [], False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_map = _load_stage_map(ROOT)
    slides    = _build_slide_list(stage_map)

    done = 0

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with mp.Pool(processes=NUM_WORKERS) as pool:
        results = pool.imap_unordered(_process_slide, slides)
        if use_tqdm:
            results = tqdm(results, total=len(slides), desc="Extracting patches", unit="slide")

        for slide_record, patch_recs, skipped in results:
            if slide_record is None:
                continue
            if skipped:
                print(f"skip (exists): {slide_record['slide_id']}")
            done += 1

    print(f"완료: {done}개 슬라이드 → {OUT_DIR}")


if __name__ == "__main__":
    mp.freeze_support()  # Windows 멀티프로세싱 필수
    main()
