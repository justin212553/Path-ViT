"""
CAMELYON17 WSI → 패치 사전 추출 스크립트

사용법:
    python -m data.preprocess

출력 구조:
    <OUT_DIR>/
        patches/
            patient_000_node_0/
                r0000_c0045.png
                ...
        slide_index.csv    # slide_id, label, center_id
        patch_index.csv    # slide_id, filename, row, col, patch_label
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import openslide

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT              = Path("data/")
OUT_DIR           = Path("data/patches")
PATCH_SIZE        = 256
PATCH_LEVEL       = 0
TISSUE_THRESHOLD  = 0.5
OVERLAP_THRESHOLD = 0.5
MAX_PATCHES       = 2000
SEED              = 42
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


def _extract_slide(slide_path: Path, xml_path, rng: np.random.Generator):
    """
    Returns:
        patches:  List[PIL.Image]
        coords:   List[[row, col]]
        labels:   List[int]   (-1 = annotation 없음)
    """
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.level_dimensions[PATCH_LEVEL]
    polygons = _load_annotations(xml_path) if xml_path is not None else []

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
            if polygons:
                ratio = _patch_overlap_fraction(polygons, x, y)
                labels.append(1 if ratio >= OVERLAP_THRESHOLD else 0)
            else:
                labels.append(-1)

    slide.close()

    if len(patches) > MAX_PATCHES:
        idx = rng.choice(len(patches), MAX_PATCHES, replace=False)
        patches = [patches[i] for i in idx]
        coords  = [coords[i]  for i in idx]
        labels  = [labels[i]  for i in idx]

    return patches, coords, labels


def _save_patches(slide_dir: Path, patches, coords, labels):
    """패치를 PNG로 저장하고 patch_index 레코드 목록을 반환."""
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


def _build_slide_list(stage_map: dict) -> list:
    slides = []
    for patient_dir in sorted((ROOT / "wsi_train").glob("patient_*")):
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng       = np.random.default_rng(SEED)
    stage_map = _load_stage_map(ROOT)
    slides    = _build_slide_list(stage_map)

    try:
        from tqdm import tqdm
        iterator = tqdm(slides, desc="Extracting patches", unit="slide")
    except ImportError:
        iterator = slides

    slide_records = []
    patch_records = []

    for info in iterator:
        slide_dir = OUT_DIR / info["slide_id"]

        if slide_dir.exists():
            print(f"skip (exists): {info['slide_id']}")
            # 기존 patch 레코드 복원
            for jpg in sorted(slide_dir.glob("*.png")):
                parts = jpg.stem.split("_")
                row = int(parts[0][1:])
                col = int(parts[1][1:])
                patch_records.append({
                    "slide_id":    info["slide_id"],
                    "filename":    str(jpg),
                    "row":         row,
                    "col":         col,
                    "patch_label": -1,  # skip 시 label 미복원
                })
        else:
            try:
                patches, coords, labels = _extract_slide(
                    info["slide_path"], info["xml_path"], rng
                )
                recs = _save_patches(slide_dir, patches, coords, labels)
                for r in recs:
                    r["slide_id"] = info["slide_id"]
                patch_records.extend(recs)
            except Exception as exc:
                print(f"ERROR {info['slide_id']}: {exc}")
                continue

        slide_records.append({
            "slide_id":  info["slide_id"],
            "label":     info["label"],
            "center_id": info["center_id"],
        })

    pd.DataFrame(slide_records).to_csv(OUT_DIR / "slide_index.csv", index=False)
    pd.DataFrame(patch_records).to_csv(OUT_DIR / "patch_index.csv", index=False)
    print(f"완료: {len(slide_records)}개 슬라이드, {len(patch_records)}개 패치 → {OUT_DIR}")


if __name__ == "__main__":
    main()
