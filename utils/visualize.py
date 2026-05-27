"""
Classification Heatmap 시각화
- 패치별 미세전이 의심 점수를 WSI 위에 오버레이
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional


def render_heatmap(
    heatmap_scores: np.ndarray,
    coords: np.ndarray,
    thumbnail: Optional[np.ndarray] = None,
    patch_size: int = 256,
    alpha: float = 0.5,
    colormap: str = "RdYlGn_r",
    save_path: Optional[str] = None,
):
    """
    패치 좌표와 점수를 그리드 히트맵으로 렌더링.

    Args:
        heatmap_scores: (N_patches,) - 각 패치의 미세전이 의심도 [0, 1]
        coords:         (N_patches, 2) - (row, col) 그리드 좌표
        thumbnail:      WSI 썸네일 이미지 (optional, 배경용)
        patch_size:     그리드 셀 크기 (픽셀)
        alpha:          히트맵 투명도
        colormap:       matplotlib colormap 이름
        save_path:      저장 경로 (None이면 plt.show())

    TODO: 실제 WSI 해상도에 맞게 좌표 스케일링
    TODO: 인터랙티브 뷰어 연동 (e.g., QuPath, ASAP)
    """
    max_row = coords[:, 0].max() + 1
    max_col = coords[:, 1].max() + 1
    grid = np.zeros((max_row, max_col))

    for score, (r, c) in zip(heatmap_scores, coords):
        grid[r, c] = score

    fig, ax = plt.subplots(figsize=(max_col / 4, max_row / 4))

    if thumbnail is not None:
        ax.imshow(thumbnail)

    cmap = cm.get_cmap(colormap)
    hm = ax.imshow(
        grid,
        cmap=cmap,
        alpha=alpha,
        vmin=0,
        vmax=1,
        extent=[0, max_col * patch_size, max_row * patch_size, 0],
    )
    plt.colorbar(hm, ax=ax, label="Micro-metastasis Score")
    ax.set_title("Classification Heatmap")
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
    """
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    title = f"{slide_id}  GT={'N1+' if label else 'N0'}  score={score:.3f}"
    save_path = str(Path(out_dir) / f"{slide_id}.png")
    render_heatmap(heatmap, coords, save_path=save_path)
    # 제목 추가를 위해 재렌더링
    fig, ax = plt.subplots(figsize=(8, 6))
    max_row = coords[:, 0].max() + 1
    max_col = coords[:, 1].max() + 1
    grid = np.zeros((max_row, max_col))
    for s, (r, c) in zip(heatmap, coords):
        grid[r, c] = s
    import matplotlib.cm as cm
    hm = ax.imshow(grid, cmap=cm.get_cmap("RdYlGn_r"), vmin=0, vmax=1)
    plt.colorbar(hm, ax=ax, label="Metastasis Score")
    ax.set_title(title)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
