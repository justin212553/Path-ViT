"""
Classification Heatmap 시각화
- 패치별 미세전이 의심 점수를 WSI 위에 오버레이
"""
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
