"""
WSI 패치 공용 유틸리티 — 패치 파일명 좌표 파싱, 정렬된 패치 목록, 표준 patch transform.

data/dataset.py(WSISurvivalDataset)와 data/extract_features.py, data/fit_clusters.py 등
패치 단위로 동작하는 모듈들이 공통으로 재사용한다.
"""
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 레퍼런스(Leeyoungsup/pancreatic_cancer_pathology) M4_Train.ipynb::get_train_cached_patch_
# transform()과 동일한 학습용 타일 augmentation — 우리는 지금까지 feature를 한 번만 추출해
# 캐싱해서(features.pt) 재사용해왔기 때문에 이 augmentation이 구조적으로 아예 없었다
# (findings_backlog.md). 매 epoch 실시간으로 적용하면(backbone을 매번 다시 태움) 로컬/HPC
# 할당량 기준 모두 비현실적으로 느려서, seed 하나로 결정되는 augmentation을 슬라이드당 딱
# 1벌만 미리 적용해 feature를 뽑아두는 절충안을 쓴다(utils/extract_features_augmented.py,
# train.py --tile-augment) — 매 epoch 다른 view는 아니지만, 최소한 "증강 없음"보다는 나은
# 신호를 준다. val/test/external은 기본 PATCH_TRANSFORM(증강 없음)을 그대로 쓴다.
PATCH_TRANSFORM_AUGMENTED = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# UNI(ViT-L/16, models/uni_encoder.py)용 transform. 패치는 1024x1024 @ 1.0MPP로 저장돼
# 있는데, UNI 학습 해상도(224, ~0.5MPP)로 그대로 리사이즈하면 실효 해상도가 4.57MPP까지
# 뭉개져 UNI 학습 분포와 크게 어긋난다(실측상 ResNet50/Lunit SwAV보다 성능이 낮았음).
# 512로 리사이즈하면 실효 2.0MPP로 그나마 덜 뭉개진다 — Leeyoungsup/pancreatic_cancer_pathology
# 레퍼런스도 같은 이유로 224 대신 512를 썼다. UNIEncoder가 dynamic_img_size=True라 224가
# 아닌 해상도도 positional embedding을 보간해 그대로 받는다.
UNI_PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# [진짜 real-time augmentation, 2026-07-22] PATCH_TRANSFORM_AUGMENTED(위)를 원본 1024x1024
# 타일에 매 epoch 그대로 적용하면 디스크에서 매 epoch 다시 디코딩하느라 1 epoch도 못 끝낼 만큼
# 느렸다(findings_backlog.md). 레퍼런스(tmp_m1_cell1.py::preload_resized_tile_images +
# get_train_cached_patch_transform)를 그대로 이식 — 디코딩+리사이즈는 학습 시작 시 딱 한 번만
# 하고(RAM에 PIL Image로 캐싱, build_tile_cache), 매 epoch은 이미 작아진 이미지 위에서
# flip/ColorJitter/GaussianBlur/정규화 같은 값싼 연산만 다시 적용한다.
#
# [2026-07-23, 두 차례 시행착오]
# (1) 원본(1024) 그대로 RAM에 전부 캐싱: 타일당 3MB * 28,987개 ≈ 87GB로 시스템 RAM(32GB)을
#     한참 넘어 65% 지점에서 스와핑으로 시스템 전체가 응답 불능(VS Code까지 크래시)에 빠졌다.
#     RAM 전체 캐싱은 512 리사이즈가 사실상 강제 조건이다(28,987 * 0.75MB ≈ 21.7GB, 그래도 빠듯).
# (2) train만 512로 캐싱하고 eval(val/test/external)은 원본 1024 그대로 `PATCH_TRANSFORM`을
#     쓰면 train/eval 유효 배율(magnification)이 달라져 val_c_index가 체계적으로 낮게 나옴을
#     확인(구 원샷 augment 기록 대비 동일 epoch에서 -0.10~0.11). 레퍼런스도 train/eval 모두
#     512로 통일해서 성능을 냈으므로, eval도 512로 맞춘다(PATCH_TRANSFORM_512, train.py) —
#     다만 eval(val/train_eval 매 epoch, test/external 1회)은 RAM 캐싱 없이 그때그때
#     디코딩+리사이즈한다(train 캐시만으로 이미 RAM이 빠듯해 val까지 얹으면 다시 위험해짐).
TILE_CACHE_SIZE = 512  # 레퍼런스와 동일 — 1.0MPP 1024 타일 -> 512 입력, 실효 2.0MPP

PATCH_TRANSFORM_AUGMENTED_CACHED = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# eval(val/test/external, train_c_index 리포팅)용 — augmentation 없이 train과 동일한 512
# 해상도로 맞춘다. tile_cache 없이(RAM 캐싱 안 함) 그때그때 원본을 열어 리사이즈한다.
PATCH_TRANSFORM_512 = transforms.Compose([
    transforms.Resize((TILE_CACHE_SIZE, TILE_CACHE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# 디스크 타일 캐시 루트 — 512로 리사이즈한 JPEG을 슬라이드별로 미러링해 저장. 한 번 어떤
# seed(TCGA train split)가 이 타일을 거치면 디스크에 남아, 다른 seed/실행이 같은 슬라이드를
# 쓸 때 원본 디코딩+리사이즈를 건너뛰고 이미 작아진 JPEG만 읽는다(수 배 빠름).
# 키는 "슬라이드_id/파일명"이라 tcga/cptac 어느 쪽에서 와도, split이 달라져도 충돌 없이 재사용된다.
TILE_DISK_CACHE_DIR = Path("data/tile_cache_512")


def _disk_cache_path(patch_path: Path, cache_dir: Path) -> Path:
    return cache_dir / patch_path.parent.name / (patch_path.stem + ".jpg")


def _decode_with_disk_cache(p: Path, cached_path: Path | None, size: int | None) -> Image.Image:
    """디스크 JPEG 캐시가 있으면 읽고, 없거나(혹은 깨져 있으면) 원본을 디코딩해 캐시에 쓴다.

    2026-08-04: K-fold를 SLURM job array로 병렬 제출하면 여러 fold가 train/val pool을 상당 부분
    공유해(fold마다 test 20%만 바뀜) 같은 타일을 동시에 이 캐시에 쓰려는 경합이 실제로
    발생했다(UnidentifiedImageError — 한 프로세스가 쓰는 도중인 파일을 다른 프로세스가 읽음).
    두 가지로 막는다:
    (1) 쓰기를 원자적으로 한다 — 같은 파일에 여러 프로세스가 동시에 write해도 반쪽짜리 파일이
        보이는 대신 "아직 없음" 또는 "완성본"만 보이게, 임시 파일에 먼저 쓰고 os.replace로
        교체한다(POSIX/Windows 둘 다 os.replace는 원자적).
    (2) 그래도 이미 깨진 파일이 디스크에 남아있는 경우(이번 사고로 실제 발생, race 픽스 이전에
        쓰인 파일)를 자동 복구한다 — 캐시 읽기가 실패하면 예외를 죽이는 대신 원본을 다시
        디코딩해 캐시를 덮어쓴다. 재시도 없이 그냥 죽던 이전 동작 대비, 병렬이든 직렬이든
        어느 쪽으로 재실행해도 스스로 회복된다.
    """
    if cached_path is not None and cached_path.exists():
        try:
            with Image.open(cached_path) as img:
                img.load()  # exists()만으로는 못 잡는 반쪽짜리/깨진 파일을 여기서 확실히 걸러낸다
                return img.convert("RGB")
        except (OSError, ValueError):
            pass  # 깨진 캐시 파일 — 아래에서 원본부터 다시 디코딩해 복구한다

    with Image.open(p) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), resample=Image.BILINEAR)

    if cached_path is not None:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cached_path.with_name(f".{cached_path.name}.tmp{os.getpid()}")
        img.save(tmp_path, format="JPEG", quality=92)
        os.replace(tmp_path, cached_path)  # 원자적 교체 — 동시에 읽는 다른 프로세스가 반쪽을 못 봄

    return img


def build_tile_cache(
    patch_paths: list[Path],
    size: int | None = TILE_CACHE_SIZE,
    disk_cache_dir: Path | None = TILE_DISK_CACHE_DIR,
    workers: int = 1,
) -> dict[Path, Image.Image]:
    """patch_paths를 전부 디코딩해 RAM에 PIL Image로 캐싱한다(학습 시작 시 1회만 호출).

    train split 전체(모든 환자의 모든 슬라이드 패치)를 한 번에 넘기면 학습 내내 재사용 가능 —
    디코딩 비용을 매 epoch에서 학습 전체 기간 중 딱 1회로 줄인다(레퍼런스와 동일한 절충).
    val/test/external은 이 함수로 캐싱하지 않는다 — train 캐시(~22GB)만으로 이미 RAM이
    빠듯해 얹으면 위험하다. 대신 PATCH_TRANSFORM_512로 그때그때 디코딩+리사이즈한다.

    disk_cache_dir가 주어지면(기본값 TILE_DISK_CACHE_DIR) 디코딩 결과를 JPEG으로 디스크에도
    남긴다 — 이미 캐싱된 타일은 원본 PNG 대신 더 가벼운 JPEG만 읽어 로드한다. 다음 실행(다른
    seed 등)이 같은 슬라이드를 다시 쓸 때 프리로드 단계를 단축한다.

    2026-08-04: workers(기본 1=기존 동작, 순차) — 이 프리로드는 models/vit_m1.py::_patch_tokens의
    ThreadPoolExecutor(--tile-decode-workers)와 별개의 함수라, 그걸 고쳐도 이 단계는 여전히
    한 개 스레드로 수만 개 타일을 하나씩 디코딩했다. Epoch 1 로그가 뜨기 전 딱 한 번 도는
    단계인데, disk_cache_dir가 아직 비어있는 첫 실행(다른 머신/클러스터로 옮긴 직후 등)에는
    파일 I/O 지연이 그대로 누적돼 여기서만 수십 분~1시간 넘게 걸릴 수 있다(train.py
    --tile-decode-workers 값을 그대로 재사용).
    """
    cache: dict[Path, Image.Image] = {}
    disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None

    def _load_one(p: Path) -> tuple[Path, Image.Image]:
        cached_path = _disk_cache_path(p, disk_cache_dir) if disk_cache_dir is not None else None
        return p, _decode_with_disk_cache(p, cached_path, size)

    # 2026-07-30: mininterval=30 — 터미널에선 tqdm이 한 줄을 덮어써서(\r) 문제없지만, Kaggle/
    # SLURM처럼 로그를 파일·API로 그대로 받아적는 비-TTY 환경에서는 \r을 못 알아먹고 매
    # 갱신(기본 0.1초 간격)이 새 줄로 쌓여 수천~수만 줄이 된다(실측: Kaggle 로그 뷰어가
    # 그래서 렉 걸림). 30초 간격이면 터미널에서도 여전히 쓸만한 진행 상황을 보여주면서
    # 로그 폭주는 막는다.
    if workers <= 1:
        for p in tqdm(patch_paths, desc="Preloading tiles", unit="tile", mininterval=30):
            _, img = _load_one(p)
            cache[p] = img
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for p, img in tqdm(executor.map(_load_one, patch_paths), total=len(patch_paths),
                                desc="Preloading tiles", unit="tile", mininterval=30):
                cache[p] = img
    return cache


class TileLRUCache:
    """patch_path -> PIL Image(RGB) LRU 캐시, 항목 개수 상한(maxsize) 있음.

    2026-07-29: build_tile_cache()(위)는 무제한 전량 RAM 캐싱이라, train_multi.py(모델 4개
    동시 학습 — CNN backbone은 공유하지만 각 모델의 proj/ViT/pooling/optimizer state가 추가로
    얹힘)에서 이 캐시(~22GB 추정, 위 build_tile_cache 문서 참조)만으로도 이미 빠듯했던 이 머신의
    RAM(32GB) 한계를 넘겨 스와핑으로 6시간 넘게 멈춘 사고 이후 추가했다. build_tile_cache 자체는
    (지금까지 잘 동작해온) train.py --tile-augment 경로를 건드리지 않기 위해 그대로 남겨두고,
    이건 train_multi.py 전용 대안이다.

    캐시가 꽉 찬 상태에서 새 타일을 넣으면 가장 오래전에 접근된 항목부터 제거한다(OrderedDict
    move_to_end + popitem(last=False)). 디스크 캐시(TILE_DISK_CACHE_DIR) 정책은
    build_tile_cache와 동일 — RAM에서 밀려나도 디스크 JPEG은 남아있어 다음 접근이 원본
    디코딩보다는 빠르다.
    """

    def __init__(
        self, maxsize: int, size: int | None = TILE_CACHE_SIZE,
        disk_cache_dir: Path | None = TILE_DISK_CACHE_DIR,
    ):
        self.maxsize = maxsize
        self.size = size
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None
        self._cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def _load(self, p: Path) -> Image.Image:
        cached_path = _disk_cache_path(p, self.disk_cache_dir) if self.disk_cache_dir is not None else None
        return _decode_with_disk_cache(p, cached_path, self.size)

    def get(self, p: Path) -> Image.Image:
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        img = self._load(p)
        self._cache[p] = img
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return img

    def preload(self, patch_paths: list[Path]) -> None:
        """maxsize까지만 채운다 — 전량 시도하지 않는다(무제한 캐싱이 원인이었던 사고 재발 방지)."""
        # mininterval=30: build_tile_cache와 동일한 이유(비-TTY 로그 폭주 방지, 2026-07-30).
        for p in tqdm(patch_paths[: self.maxsize], desc="Preloading tiles (LRU-bounded)",
                      unit="tile", mininterval=30):
            self.get(p)


FEATURES_FILENAME      = "features.pt"       # data/extract_features.py 산출물 파일명(ResNet50/Lunit SwAV)
FEATURES_UNI_FILENAME  = "features_uni.pt"   # UNI 산출물 — 기존 features.pt와 별도 저장(롤백 가능)
FEATURES_NORM_FILENAME = "features_norm.pt"  # Macenko stain-normalized + ResNet50 산출물 (utils/extract_features_stain_norm.py)
FEATURES_AUG_FILENAME  = "features_aug.pt"   # 타일 augmentation(seed 고정, 1회성) + ResNet50 산출물
                                              # (utils/extract_features_augmented.py, train.py --tile-augment)

_COORD_RE = re.compile(r"r(\d+)_c(\d+)")


def _parse_coord(name: str) -> tuple[int, int]:
    m = _COORD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def list_patch_paths(node_dir: Path) -> list[Path]:
    """슬라이드 디렉터리의 패치 파일을 정렬된 순서로 나열.

    data/extract_features.py가 features.pt를 만들 때도 이 순서를 그대로 써야
    캐싱된 feature 행(row)과 패치(coords)가 어긋나지 않는다.
    """
    return sorted(list(node_dir.glob("*.png")) + list(node_dir.glob("*.jpg")))
