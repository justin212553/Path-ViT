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
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import openslide
from PIL import Image
from shapely.geometry import box as shapely_box, Polygon as ShapelyPolygon
from shapely.ops import unary_union

from utils import send_slack

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT              = Path("./data/")
OUT_DIR           = Path("./data/patches")
PATCH_SIZE        = 256
PATCH_LEVEL       = 0
BG_PIXEL_THRESH   = 220   # RGB 평균이 이 값 이상인 픽셀 → 빈 유리(배경)
BG_MAX_FRACTION   = 0.9   # 패치 내 배경 픽셀 비율이 이 값 이상이면 제거
OVERLAP_THRESHOLD = 0.5
NUM_WORKERS       = 8
NUM_IO_THREADS    = 4
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


def _load_annotations(xml_path: Path):
    polygons = []
    root_el = ET.parse(str(xml_path)).getroot()
    for annotation in root_el.findall(".//Annotation"):
        coords = sorted(
            annotation.findall(".//Coordinate"),
            key=lambda c: int(c.get("Order", 0)),
        )
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in coords]
        if len(pts) >= 3:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append(poly)
    return polygons


def _patch_overlap_fraction(annotation_union, x: int, y: int) -> float:
    patch_box = shapely_box(x, y, x + PATCH_SIZE, y + PATCH_SIZE)
    return annotation_union.intersection(patch_box).area / (PATCH_SIZE * PATCH_SIZE)


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

    annotation_union = None
    if xml_path is not None:
        polys = _load_annotations(xml_path)
        if polys:
            annotation_union = unary_union(polys)

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

            n_rows, n_cols = bh // PATCH_SIZE, bw // PATCH_SIZE
            # 배경 감지: RGB 평균 > BG_PIXEL_THRESH인 픽셀(흰 유리)의 패치 내 비율
            gray = block_arr.mean(axis=2)  # (H, W)
            bg_fracs = (gray > BG_PIXEL_THRESH).reshape(
                n_rows, PATCH_SIZE, n_cols, PATCH_SIZE
            ).mean(axis=(1, 3))            # (n_rows, n_cols)

            for pr in range(n_rows):
                for pc in range(n_cols):
                    if bg_fracs[pr, pc] >= BG_MAX_FRACTION:
                        continue

                    y, x = by + pr * PATCH_SIZE, bx + pc * PATCH_SIZE
                    patch_arr = block_arr[
                        pr * PATCH_SIZE:(pr + 1) * PATCH_SIZE,
                        pc * PATCH_SIZE:(pc + 1) * PATCH_SIZE,
                    ]
                    patches.append(Image.fromarray(patch_arr))
                    coords.append([y // PATCH_SIZE, x // PATCH_SIZE])
                    if annotation_union is not None:
                        ratio = _patch_overlap_fraction(annotation_union, x, y)
                        labels.append(1 if ratio >= OVERLAP_THRESHOLD else 0)
                    else:
                        labels.append(-1)

    slide.close()

    return patches, coords, labels


def _write_jpeg(args):
    path, patch = args
    patch.save(path, format="JPEG", quality=JPEG_QUALITY)


def _save_patches(slide_dir: Path, patches, coords, labels):
    if not patches:
        return []
    slide_dir.mkdir(parents=True, exist_ok=True)

    records = []
    save_args = []
    for patch, (row, col), label in zip(patches, coords, labels):
        fpath = slide_dir / f"r{row:04d}_c{col:04d}.jpg"
        save_args.append((fpath, patch))
        records.append({
            "filename":    str(fpath),
            "row":         row,
            "col":         col,
            "patch_label": label,
        })

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as ex:
        list(ex.map(_write_jpeg, save_args))

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
    global NUM_WORKERS, NUM_IO_THREADS

    import argparse
    parser = argparse.ArgumentParser(description="CAMELYON17 patch extractor")
    parser.add_argument("--task-id",    type=int, default=0,              help="0-indexed shard index")
    parser.add_argument("--num-tasks",  type=int, default=1,              help="total number of shards")
    parser.add_argument("--workers",    type=int, default=NUM_WORKERS,    help="mp.Pool process count")
    parser.add_argument("--io-threads", type=int, default=NUM_IO_THREADS, help="JPEG save threads per worker")
    args = parser.parse_args()

    NUM_WORKERS    = args.workers
    NUM_IO_THREADS = args.io_threads

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    tag    = f"job `{job_id}` task `{args.task_id}/{args.num_tasks}`"
    start  = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_map = _load_stage_map(ROOT)
    slides    = _build_slide_list(stage_map)
    slides    = slides[args.task_id :: args.num_tasks]

    print(f"[task {args.task_id}/{args.num_tasks}] {len(slides)} slides")

    done = 0

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    try:
        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.imap_unordered(_process_slide, slides)
            if use_tqdm:
                results = tqdm(results, total=len(slides), desc=f"task {args.task_id}", unit="slide")

            for slide_record, patch_recs, skipped in results:
                if slide_record is None:
                    continue
                if skipped:
                    print(f"skip (exists): {slide_record['slide_id']}")
                done += 1

        elapsed = int(time.time() - start)
        print(f"[task {args.task_id}] 완료: {done}개 슬라이드 → {OUT_DIR}")
        send_slack(f":white_check_mark: *preprocess 완료* {tag} — {done}슬라이드 — {elapsed//60}m{elapsed%60}s")

    except Exception as exc:
        elapsed = int(time.time() - start)
        send_slack(f":x: *preprocess 실패* {tag} — {exc} — {elapsed//60}m{elapsed%60}s")
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
