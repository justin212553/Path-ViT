"""
CPTAC-PDA WSI(.svs) → tile 사전 추출 스크립트

Method.md 3절(전처리) 스펙을 그대로 구현한다:
    - target resolution 1.0 MPP 에서 1024 x 1024 tile 로 분할
    - tissue ratio threshold 0.15 미만인 배경 tile 은 제외
    - slide 당 최대 512 tile 사용 / train=random, val·test=deterministic 샘플링은
      전처리 단계가 아니라 학습 시점(dataloader)의 관심사이므로 여기서는 조건을 만족하는
      tile 을 전부 저장해 두고, 학습 코드에서 사용할 수 있도록 sample_tile_paths() 헬퍼만 제공한다.

사용법:
    python -m data.preprocess_cptac

출력 구조:
    <OUT_DIR>/
        tiles/
            C3L-00017-21/
                r0012_c0034.jpg
                ...
        slide_index.csv   # slide_id, case_id, native_mpp, level_used, n_tiles_kept, ...
        tile_index.csv    # slide_id, case_id, filename, row, col, tissue_ratio
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageFilter

from utils import send_slack

# ── 설정 ──────────────────────────────────────────────────────────────────────
ROOT               = Path("./data/cptac_pda_wsi")
OUT_DIR            = Path("./data/patches_cptac")
TARGET_MPP         = 1.0
TILE_SIZE          = 1024
TISSUE_RATIO_THRESH = 0.15   # tile 내 tissue 픽셀 비율이 이 값 미만이면 배경으로 간주해 제외
VALUE_FLOOR         = 10     # HSV V(밝기)가 이 값 이하면 슬라이드 밖 padding(검정)으로 간주해 배경 처리
SAT_MEDIAN_FILTER   = 7      # thumbnail saturation 채널 노이즈 제거용 median filter 크기(홀수)
MAX_TILES_PER_SLIDE = 512    # 학습 시 slide 당 최대 사용 tile 수 (sample_tile_paths 참고)
THUMB_MAX_SIDE      = 2048   # 배경 grid cell을 미리 걸러내기 위한 thumbnail 최대 변 길이
THUMB_TISSUE_MARGIN = 0.02   # thumbnail 기준 이 비율보다 tissue가 적은 cell은 실제 read_region 생략
NUM_WORKERS         = 8
NUM_IO_THREADS       = 4
JPEG_QUALITY         = 90
DONE_MARKER          = ".done"
# ─────────────────────────────────────────────────────────────────────────────


def _get_native_mpp(slide: openslide.OpenSlide) -> float:
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    if mpp_x is None or mpp_y is None:
        # aperio SVS는 보통 openslide.mpp-x/y로 정규화되지만, 혹시 없으면 raw aperio 태그로 폴백
        mpp_x = mpp_x or slide.properties.get("aperio.MPP")
        mpp_y = mpp_y or slide.properties.get("aperio.MPP")
    if mpp_x is None or mpp_y is None:
        raise ValueError("MPP metadata를 찾을 수 없음 (openslide.mpp-x/y, aperio.MPP 모두 없음)")
    return (float(mpp_x) + float(mpp_y)) / 2.0


def _otsu_threshold(gray: np.ndarray) -> float:
    """0-255 정수 배열에 대한 Otsu 임계값(between-class variance 최대화 지점)."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    sum_all = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0.0
    best_thresh, best_var = 0, -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var, best_thresh = between_var, t
    return float(best_thresh)


def _tissue_ratio(tile_img: Image.Image, sat_thresh: float) -> float:
    """HSV 기준 tissue 픽셀 비율. H&E 염색 조직은 saturation이 높고, 빈 유리(흰 배경)는
    saturation이 낮다. VALUE_FLOOR로 슬라이드 밖 검정 padding까지 함께 배경 처리한다."""
    hsv = np.array(tile_img.convert("HSV"))
    sat, val = hsv[..., 1], hsv[..., 2]
    tissue_mask = (sat > sat_thresh) & (val > VALUE_FLOOR)
    return float(tissue_mask.mean())


def _build_tissue_thumb_mask(slide: openslide.OpenSlide):
    """저해상도 thumbnail의 HSV saturation 채널에 Otsu를 적용해 slide 전역 threshold를 구하고,
    이를 그대로 tissue mask(및 개별 tile ratio 계산)에 재사용한다 — tile마다 Otsu를 따로 돌리면
    tissue가 거의 없는 tile에서 bimodal 가정이 깨져 임계값이 불안정해지기 때문.

    Returns: tissue_mask, slide→thumbnail 좌표 스케일, slide 전역 saturation threshold
    """
    w, h = slide.dimensions
    scale = THUMB_MAX_SIDE / max(w, h)
    thumb = slide.get_thumbnail((max(1, int(w * scale)), max(1, int(h * scale))))
    hsv = np.array(thumb.convert("RGB").convert("HSV"))
    sat, val = hsv[..., 1], hsv[..., 2]

    sat_smooth = np.array(Image.fromarray(sat).filter(ImageFilter.MedianFilter(SAT_MEDIAN_FILTER)))
    sat_thresh = _otsu_threshold(sat_smooth)

    tissue_mask = (sat_smooth > sat_thresh) & (val > VALUE_FLOOR)
    return tissue_mask, scale, sat_thresh


def _grid_cell_has_tissue(tissue_mask, scale, x0: int, y0: int, size_level0: int) -> bool:
    """level-0 좌표 기준 grid cell이 thumbnail tissue mask 상에서 tissue를 포함하는지 빠르게 확인."""
    th, tw = tissue_mask.shape
    tx0 = max(0, min(tw - 1, int(x0 * scale)))
    ty0 = max(0, min(th - 1, int(y0 * scale)))
    tx1 = max(tx0 + 1, min(tw, int((x0 + size_level0) * scale)))
    ty1 = max(ty0 + 1, min(th, int((y0 + size_level0) * scale)))
    cell = tissue_mask[ty0:ty1, tx0:tx1]
    if cell.size == 0:
        return False
    return float(cell.mean()) >= THUMB_TISSUE_MARGIN


def _extract_slide(slide_path: Path):
    """
    단일 SVS를 1.0 MPP / 1024x1024 tile grid로 분할하고 tissue tile만 반환.

    Returns:
        tiles:  List[PIL.Image]   (TILE_SIZE, TILE_SIZE)
        grid:   List[(row, col)]  tile grid 인덱스
        ratios: List[float]       tissue_ratio
        meta:   dict              slide 메타데이터
    """
    slide = openslide.OpenSlide(str(slide_path))
    native_mpp = _get_native_mpp(slide)
    downsample_target = TARGET_MPP / native_mpp

    level = slide.get_best_level_for_downsample(downsample_target)
    level_downsample = slide.level_downsamples[level]

    # level-0 좌표 기준, TILE_SIZE*TARGET_MPP(micron) 물리 영역에 대응하는 픽셀 수
    read_size_level0 = max(1, round(TILE_SIZE * downsample_target))
    read_size_level = max(1, round(read_size_level0 / level_downsample))

    w0, h0 = slide.dimensions
    tissue_mask, thumb_scale, sat_thresh = _build_tissue_thumb_mask(slide)

    tiles, grid, ratios = [], [], []
    n_total = 0
    row = 0
    for y0 in range(0, h0, read_size_level0):
        col = 0
        for x0 in range(0, w0, read_size_level0):
            n_total += 1
            if _grid_cell_has_tissue(tissue_mask, thumb_scale, x0, y0, read_size_level0):
                region = slide.read_region((int(x0), int(y0)), level, (read_size_level, read_size_level))
                region = region.convert("RGB")
                if region.size != (TILE_SIZE, TILE_SIZE):
                    region = region.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)

                ratio = _tissue_ratio(region, sat_thresh)
                if ratio >= TISSUE_RATIO_THRESH:
                    tiles.append(region)
                    grid.append((row, col))
                    ratios.append(ratio)
            col += 1
        row += 1

    slide.close()

    meta = {
        "native_mpp":       native_mpp,
        "level_used":       level,
        "level_downsample": level_downsample,
        "read_size_level0": read_size_level0,
        "sat_thresh":       sat_thresh,
        "n_tiles_total":    n_total,
        "n_tiles_kept":     len(tiles),
    }
    return tiles, grid, ratios, meta


def _write_jpeg(args):
    path, tile = args
    tile.save(path, format="JPEG", quality=JPEG_QUALITY)


def _save_tiles(slide_dir: Path, tiles, grid, ratios, slide_id, case_id):
    if not tiles:
        return []
    slide_dir.mkdir(parents=True, exist_ok=True)

    records, save_args = [], []
    for tile, (row, col), ratio in zip(tiles, grid, ratios):
        fpath = slide_dir / f"r{row:04d}_c{col:04d}.jpg"
        save_args.append((fpath, tile))
        records.append({
            "slide_id":     slide_id,
            "case_id":      case_id,
            "filename":     str(fpath),
            "row":          row,
            "col":          col,
            "tissue_ratio": ratio,
        })

    with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as ex:
        list(ex.map(_write_jpeg, save_args))

    return records


def _case_id_from_slide_id(slide_id: str) -> str:
    # "C3L-00017-21" → "C3L-00017" (마지막 "-NN" 조각은 slide/block 번호)
    return slide_id.rsplit("-", 1)[0]


def _build_slide_list() -> list:
    slides = []
    for svs_path in sorted(ROOT.glob("*.svs")):
        slide_id = svs_path.stem
        slides.append({
            "slide_id":  slide_id,
            "case_id":   _case_id_from_slide_id(slide_id),
            "svs_path":  svs_path,
        })
    return slides


def _process_slide(info):
    """
    단일 슬라이드를 처리하는 워커 함수.
    Returns (slide_record | None, tile_records, skipped)
    """
    slide_dir = OUT_DIR / "tiles" / info["slide_id"]
    marker = slide_dir / DONE_MARKER

    if marker.exists():
        tile_records = []
        for img_path in sorted(slide_dir.glob("*.jpg")):
            parts = img_path.stem.split("_")
            row, col = int(parts[0][1:]), int(parts[1][1:])
            tile_records.append({
                "slide_id":     info["slide_id"],
                "case_id":      info["case_id"],
                "filename":     str(img_path),
                "row":          row,
                "col":          col,
                "tissue_ratio": np.nan,
            })
        slide_record = {
            "slide_id":  info["slide_id"],
            "case_id":   info["case_id"],
            "svs_path":  str(info["svs_path"]),
            "status":    "ok",
            "n_tiles_kept": len(tile_records),
        }
        return slide_record, tile_records, True  # skipped=True

    try:
        tiles, grid, ratios, meta = _extract_slide(info["svs_path"])
        recs = _save_tiles(slide_dir, tiles, grid, ratios, info["slide_id"], info["case_id"])
        if not recs:
            print(f"WARN no tissue tiles extracted: {info['slide_id']}")
        slide_dir.mkdir(parents=True, exist_ok=True)
        marker.touch()

        slide_record = {
            "slide_id":  info["slide_id"],
            "case_id":   info["case_id"],
            "svs_path":  str(info["svs_path"]),
            "status":    "ok",
            **meta,
        }
        return slide_record, recs, False
    except Exception as exc:
        print(f"ERROR {info['slide_id']}: {exc}")
        slide_record = {
            "slide_id":  info["slide_id"],
            "case_id":   info["case_id"],
            "svs_path":  str(info["svs_path"]),
            "status":    "failed",
            "error":     str(exc),
        }
        return slide_record, [], False


def sample_tile_paths(tile_paths: list, split: str, max_tiles: int = MAX_TILES_PER_SLIDE, seed: int = 42) -> list:
    """Method.md 3절: '학습 입력 시 slide 당 최대 512개 tile 사용, 학습 중 random sampling,
    검증/평가에서는 deterministic sampling' 규칙을 구현한 학습 시점 헬퍼.

    train: 매 호출마다(=매 epoch) numpy 전역 RNG로 무작위 서브샘플.
    val/test: slide_id 기반 고정 seed로 한 번만 정해지는 결정론적 서브샘플.
    """
    if len(tile_paths) <= max_tiles:
        return list(tile_paths)
    if split == "train":
        idx = np.random.choice(len(tile_paths), size=max_tiles, replace=False)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(tile_paths), size=max_tiles, replace=False)
        idx.sort()
    return [tile_paths[i] for i in idx]


def main():
    global NUM_WORKERS, NUM_IO_THREADS, ROOT, OUT_DIR

    parser = argparse.ArgumentParser(description="CPTAC-PDA SVS → tile 사전 추출")
    parser.add_argument("--input-dir",  type=str, default=str(ROOT),    help="SVS 파일이 있는 디렉터리")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DIR), help="tile/인덱스 출력 디렉터리")
    parser.add_argument("--task-id",    type=int, default=0,              help="0-indexed shard index")
    parser.add_argument("--num-tasks",  type=int, default=1,              help="total number of shards")
    parser.add_argument("--workers",    type=int, default=NUM_WORKERS,    help="mp.Pool process count")
    parser.add_argument("--io-threads", type=int, default=NUM_IO_THREADS, help="JPEG save threads per worker")
    args = parser.parse_args()

    ROOT           = Path(args.input_dir)
    OUT_DIR        = Path(args.output_dir)
    NUM_WORKERS    = args.workers
    NUM_IO_THREADS = args.io_threads

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    tag    = f"job `{job_id}` task `{args.task_id}/{args.num_tasks}`"
    start  = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slides = _build_slide_list()
    slides = slides[args.task_id :: args.num_tasks]

    print(f"[task {args.task_id}/{args.num_tasks}] {len(slides)} slides")

    done = 0
    slide_records, tile_records = [], []

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

            for slide_record, recs, skipped in results:
                if slide_record is None:
                    continue
                slide_records.append(slide_record)
                tile_records.extend(recs)
                if skipped:
                    print(f"skip (exists): {slide_record['slide_id']}")
                done += 1

        slide_index_path = OUT_DIR / f"slide_index_task{args.task_id}.csv"
        tile_index_path  = OUT_DIR / f"tile_index_task{args.task_id}.csv"
        pd.DataFrame(slide_records).to_csv(slide_index_path, index=False)
        pd.DataFrame(tile_records).to_csv(tile_index_path, index=False)

        elapsed = int(time.time() - start)
        print(f"[task {args.task_id}] 완료: {done}개 슬라이드 → {OUT_DIR}")
        send_slack(f":white_check_mark: *CPTAC preprocess 완료* {tag} — {done}슬라이드 — {elapsed//60}m{elapsed%60}s")

    except Exception as exc:
        elapsed = int(time.time() - start)
        send_slack(f":x: *CPTAC preprocess 실패* {tag} — {exc} — {elapsed//60}m{elapsed%60}s")
        raise


if __name__ == "__main__":
    mp.freeze_support()
    main()
