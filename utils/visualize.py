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
        score_grid[r, c] = float(score)
        gt_grid[r, c]    = float(label)

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
    from pathlib import Path
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
