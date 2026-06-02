"""
Classification Heatmap 시각화
- 패치별 미세전이 의심 점수를 WSI 위에 오버레이
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def render_heatmap(
    heatmap_scores: np.ndarray,
    coords: np.ndarray,
    thumbnail: Optional[np.ndarray] = None,
    patch_size: int = 256,
    alpha: float = 0.5,
    colormap: str = "RdYlGn_r",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    wsi_dims: Optional[Tuple[int, int]] = None,
):
    """
    패치 좌표와 점수를 그리드 히트맵으로 렌더링.

    Args:
        heatmap_scores: (N_patches,) - 각 패치의 미세전이 의심도 [0, 1]
        coords:         (N_patches, 2) - (row, col) 그리드 좌표
        thumbnail:      WSI 썸네일 이미지 (optional, 배경용)
        patch_size:     그리드 셀 크기 (원본 WSI 픽셀)
        alpha:          히트맵 투명도
        colormap:       matplotlib colormap 이름
        save_path:      저장 경로 (None이면 plt.show())
        title:          플롯 제목
        wsi_dims:       원본 WSI 해상도 (width, height) — thumbnail과 함께 쓸 때
                        좌표를 썸네일 해상도에 맞게 스케일링하는 데 사용
    """
    max_row = int(coords[:, 0].max()) + 1
    max_col = int(coords[:, 1].max()) + 1
    grid = np.zeros((max_row, max_col))

    for score, (r, c) in zip(heatmap_scores, coords):
        grid[int(r), int(c)] = score

    fig, ax = plt.subplots(figsize=(max(6, max_col / 4), max(5, max_row / 4)))

    # 좌표 스케일링: 썸네일 해상도에 맞춰 extent 계산
    if thumbnail is not None and wsi_dims is not None:
        thumb_h, thumb_w = thumbnail.shape[:2]
        wsi_w, wsi_h = wsi_dims
        scale_x = thumb_w / wsi_w
        scale_y = thumb_h / wsi_h
        cell_w = patch_size * scale_x
        cell_h = patch_size * scale_y
    else:
        cell_w = cell_h = patch_size

    extent = [0, max_col * cell_w, max_row * cell_h, 0]

    if thumbnail is not None:
        thumb_extent = [0, thumbnail.shape[1], thumbnail.shape[0], 0] if wsi_dims is None else extent
        ax.imshow(thumbnail, extent=thumb_extent, aspect="auto")

    cmap = cm.get_cmap(colormap)
    hm = ax.imshow(
        grid,
        cmap=cmap,
        alpha=alpha,
        vmin=0,
        vmax=1,
        extent=extent,
    )
    plt.colorbar(hm, ax=ax, label="Micro-metastasis Score")
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def render_dual_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_labels: np.ndarray,
    thumbnail: Optional[np.ndarray] = None,
    title: str = "",
    alpha: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    듀얼 오버레이 시각화.

    레이어 구성:
      - 배경  : thumbnail 또는 회색 캔버스
      - L1    : 모델 softmax 히트맵 (Jet, 패치별 alpha = score × alpha)
      - L2    : GT 종양 경계 윤곽선 (초록색 Contour Line)

    Args:
        scores:       (N,) 패치별 softmax 양성 확률 [0, 1]
        coords:       (N, 2) (row, col) 그리드 좌표
        patch_labels: (N,) GT 라벨 (0=정상, 1=종양)
        thumbnail:    WSI 저해상도 썸네일 (optional)
        title:        플롯 제목
        alpha:        히트맵 최대 불투명도
        save_path:    저장 경로 (None이면 plt.show())
    """
    max_row = int(coords[:, 0].max()) + 1
    max_col = int(coords[:, 1].max()) + 1

    score_grid = np.zeros((max_row, max_col), dtype=np.float32)
    gt_grid    = np.zeros((max_row, max_col), dtype=np.float32)
    for score, label, (r, c) in zip(scores, patch_labels, coords):
        score_grid[int(r), int(c)] = float(score)
        gt_grid[int(r), int(c)]    = float(label)

    fig, ax = plt.subplots(figsize=(max(8, max_col / 4), max(6, max_row / 4)))

    # 배경
    if thumbnail is not None:
        ax.imshow(thumbnail, aspect="auto", extent=[0, max_col, max_row, 0])
    else:
        bg = np.full((max_row, max_col, 3), 0.85, dtype=np.float32)
        ax.imshow(bg, extent=[0, max_col, max_row, 0], interpolation="nearest")

    # Layer 1: Jet 히트맵 (score 낮을수록 투명, 높을수록 빨간색·불투명)
    rgba = cm.get_cmap("jet")(score_grid).astype(np.float32)  # (H, W, 4)
    rgba[..., 3] = score_grid * alpha
    ax.imshow(rgba, extent=[0, max_col, max_row, 0], interpolation="bilinear")

    # Layer 2: GT 종양 경계 윤곽선
    if gt_grid.max() > 0:
        xs = np.arange(max_col) + 0.5
        ys = np.arange(max_row) + 0.5
        ax.contour(xs, ys, gt_grid, levels=[0.5], colors=["lime"], linewidths=[1.5])

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Cancer Probability", fraction=0.046, pad=0.04)

    from matplotlib.lines import Line2D
    ax.legend(
        handles=[Line2D([0], [0], color="lime", linewidth=1.5, label="GT Tumor Boundary")],
        loc="upper right",
        fontsize=8,
    )

    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def save_dual_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_labels: np.ndarray,
    slide_id: str,
    thumbnail: Optional[np.ndarray] = None,
    out_dir: str = "heatmaps",
):
    """슬라이드 한 장의 듀얼 오버레이를 파일로 저장."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n_tumor = int(patch_labels.sum())
    n_total = len(patch_labels)
    acc     = float(((scores >= 0.5).astype(int) == patch_labels).mean())
    title   = f"{slide_id}  GT={n_tumor}/{n_total} tumor  acc={acc:.3f}"

    render_dual_overlay(
        scores=scores,
        coords=coords,
        patch_labels=patch_labels,
        thumbnail=thumbnail,
        title=title,
        save_path=str(Path(out_dir) / f"{slide_id}.png"),
    )


def save_heatmap(
    heatmap: np.ndarray,
    coords: np.ndarray,
    slide_id: str,
    label: int,
    score: float,
    out_dir: str = "heatmaps",
    wsi_dims: Optional[Tuple[int, int]] = None,
):
    """
    슬라이드 한 장의 heatmap을 파일로 저장.

    Args:
        heatmap:  (N_patches,) — 패치별 전이 의심도 [0,1]
        coords:   (N_patches, 2) — (row, col) 그리드 좌표
        slide_id: 슬라이드 ID (파일명 사용)
        label:    실제 레이블 (0=N0/음성, 1=N1+/양성)
        score:    슬라이드 레벨 예측 점수
        out_dir:  저장 디렉토리
        wsi_dims: 원본 WSI 해상도 (width, height) — 썸네일 좌표 스케일링용
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    title = f"{slide_id}  GT={'N1+' if label else 'N0'}  score={score:.3f}"
    render_heatmap(
        heatmap,
        coords,
        title=title,
        wsi_dims=wsi_dims,
        save_path=str(Path(out_dir) / f"{slide_id}.png"),
    )


# ---------------------------------------------------------------------------
# 인터랙티브 뷰어 익스포트
# ---------------------------------------------------------------------------

def export_qupath_geojson(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size: int = 256,
    threshold: float = 0.5,
    save_path: str = "annotations.geojson",
    wsi_dims: Optional[Tuple[int, int]] = None,
) -> None:
    """
    패치 예측 결과를 QuPath 호환 GeoJSON으로 익스포트.

    QuPath에서 Extensions > Import Annotations 또는
    File > Import Object File 로 불러올 수 있습니다.

    Args:
        scores:     (N,) 패치별 softmax 양성 확률 [0, 1]
        coords:     (N, 2) (row, col) 그리드 좌표 (원본 WSI 기준)
        patch_size: 원본 WSI에서의 패치 크기 (픽셀)
        threshold:  이 값 이상이면 Tumor, 미만이면 Normal 로 분류
        save_path:  출력 GeoJSON 파일 경로
        wsi_dims:   원본 WSI 해상도 (width, height) — 현재는 메타데이터 기록용
    """
    features = []
    for score, (r, c) in zip(scores, coords):
        r, c = int(r), int(c)
        x0 = c * patch_size
        y0 = r * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        is_tumor = bool(score >= threshold)
        cls_name = "Tumor" if is_tumor else "Normal"
        # QuPath colorRGB: Tumor=빨강(#FF0000), Normal=파랑(#0000FF)
        color_rgb = -65536 if is_tumor else -16776961

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
                ]],
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": cls_name,
                    "colorRGB": color_rgb,
                },
                "measurements": [
                    {"name": "Metastasis Score", "value": float(score)},
                ],
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    if wsi_dims is not None:
        geojson["metadata"] = {"wsi_width": wsi_dims[0], "wsi_height": wsi_dims[1],
                               "patch_size": patch_size}

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(geojson, f, indent=2)


def export_asap_xml(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size: int = 256,
    threshold: float = 0.5,
    save_path: str = "annotations.xml",
) -> None:
    """
    패치 예측 결과를 ASAP 호환 XML로 익스포트.

    ASAP에서 File > Open Annotation 으로 불러올 수 있습니다.

    Args:
        scores:     (N,) 패치별 softmax 양성 확률 [0, 1]
        coords:     (N, 2) (row, col) 그리드 좌표 (원본 WSI 기준)
        patch_size: 원본 WSI에서의 패치 크기 (픽셀)
        threshold:  이 값 이상이면 Tumor, 미만이면 Normal 로 분류
        save_path:  출력 XML 파일 경로
    """
    root = ET.Element("ASAP_Annotations")
    annotations_el = ET.SubElement(root, "Annotations")

    for idx, (score, (r, c)) in enumerate(zip(scores, coords)):
        r, c = int(r), int(c)
        x0, y0 = float(c * patch_size), float(r * patch_size)
        x1, y1 = x0 + patch_size, y0 + patch_size

        is_tumor = bool(score >= threshold)
        group = "Tumor" if is_tumor else "Normal"
        color = "#FF0000" if is_tumor else "#73D216"

        ann = ET.SubElement(
            annotations_el, "Annotation",
            Name=f"Patch_{idx}",
            PartOfGroup=group,
            Color=color,
        )
        ann.set("Score", f"{score:.6f}")
        coords_el = ET.SubElement(ann, "Coordinates")
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for order, (x, y) in enumerate(corners):
            ET.SubElement(coords_el, "Coordinate",
                          Order=str(order), X=f"{x:.1f}", Y=f"{y:.1f}")

    groups_el = ET.SubElement(root, "AnnotationGroups")
    for name, color in [("Tumor", "#FF0000"), ("Normal", "#73D216")]:
        ET.SubElement(groups_el, "Group",
                      Name=name, PartOfGroup="None", Color=color)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(save_path, encoding="utf-8", xml_declaration=True)
