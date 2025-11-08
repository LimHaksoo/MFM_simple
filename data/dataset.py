# data/dataset.py
# -*- coding: utf-8 -*-
"""
Unified anomaly detection dataset for:
- MVTec AD
- VisA
- MVTec LOCO AD
- GoodsAD

Assumptions:
- 'good'/'ok'/'normal' 디렉토리는 정상(0), 나머지는 이상(1)
- 마스크는 다음 중 하나로 추론:
  - (MVTec 계열) test/<defect>/<file> -> ground_truth/<defect>/<file>_mask
  - sibling 'mask' or 'masks' 폴더 내 동일 파일명
만약 마스크를 찾지 못하면 None 반환 (또는 0 텐서로 대체)
"""

# ==== AUTO-DOWNLOAD: add below to data/dataset.py =================================
import io
import sys
import tarfile
import zipfile
import shutil
import urllib.request
from urllib.parse import urlparse
import subprocess
import argparse

from tqdm import tqdm

import os
import glob
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
NORMAL_KEYWORDS = ("good", "ok", "normal")

def _is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMG_EXTS

def _default_clip_preprocess(image_size: int = 224) -> transforms.Compose:
    # CLIP 기본 전처리 근사 (open_clip 미사용 시)
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])

def _guess_label_from_path(path: str) -> int:
    low = path.lower()
    return 0 if any(k in low.split(os.sep) for k in NORMAL_KEYWORDS) else 1

def _guess_mask_path_mvtec(img_path: str) -> Optional[str]:
    # .../<cat>/test/<defect>/<file>.png -> .../<cat>/ground_truth/<defect>/<file>_mask.png
    parts = img_path.replace("\\", "/").split("/")
    if "test" not in parts:
        return None
    try:
        idx = parts.index("test")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    defect = parts[idx + 1]
    fname = os.path.splitext(parts[-1])[0]
    # build ground truth path
    parts[idx] = "ground_truth"
    parts[idx + 1] = defect
    gt_dir = "/".join(parts[:-1])
    candidates = [
        os.path.join(gt_dir, f"{fname}_mask.png"),
        os.path.join(gt_dir, f"{fname}_mask.jpg"),
        os.path.join(gt_dir, f"{fname}.png"),
        os.path.join(gt_dir, f"{fname}.jpg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def _guess_mask_path_generic(img_path: str) -> Optional[str]:
    # 형제 디렉토리 'mask'/'masks' 내 동일 파일명 탐색
    d, fn = os.path.split(img_path)
    parent = os.path.dirname(d)
    fname, _ = os.path.splitext(fn)
    for mask_dir_name in ["mask", "masks", "ground_truth", "gt", "annotations"]:
        mdir = os.path.join(parent, mask_dir_name)
        if not os.path.isdir(mdir):
            continue
        cands = glob.glob(os.path.join(mdir, fname + ".*"))
        cands = [c for c in cands if _is_image(c)]
        if cands:
            return cands[0]
    return None

@dataclass
class Sample:
    img_path: str
    mask_path: Optional[str]
    label: int  # 0 normal, 1 anomaly
    dataset: str
    category: Optional[str]
    split: str  # 'train' | 'test' | 'all'

class AnomalyDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        image_size: int = 224,
        return_mask: bool = False,
        use_clip_preprocess: bool = True,
        normal_only_train: bool = True,
    ):
        """
        Args:
            root: dataset root (각 데이터셋 루트)
            dataset_name: ['mvtec', 'visa', 'mvtec_loco', 'goodsad']
            split: 'train' | 'test' | 'all'
            image_size: CLIP 입력 크기 (224 / 336 등)
            return_mask: True면 (C,H,W) 마스크 텐서를 함께 반환 (없으면 0 텐서)
            use_clip_preprocess: True면 CLIP 정규화 사용
            normal_only_train: train split에서 정상만 사용할지 여부(AD 전형)
        """
        self.root = root
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.image_size = image_size
        self.return_mask = return_mask
        self.normal_only_train = normal_only_train

        self.transform = _default_clip_preprocess(image_size) if use_clip_preprocess \
            else transforms.Compose([transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.ToTensor()])

        self.samples: List[Sample] = self._scan()

    def _scan(self) -> List[Sample]:
        root = self.root
        ds = self.dataset_name
        split = self.split
        samples: List[Sample] = []

        def add_sample(img_path: str, dataset: str, category: Optional[str], split_: str):
            label = _guess_label_from_path(img_path)
            # train에서 정상만 사용할지
            if split_ == "train" and self.normal_only_train and label != 0:
                return
            # 마스크 추론
            mask_path = None
            if ds in ("mvtec", "mvtec_loco"):
                mask_path = _guess_mask_path_mvtec(img_path) or _guess_mask_path_generic(img_path)
            else:
                mask_path = _guess_mask_path_generic(img_path)

            samples.append(Sample(
                img_path=img_path,
                mask_path=mask_path,
                label=label,
                dataset=dataset,
                category=category,
                split=split_,
            ))

        # 공통: 카테고리 폴더 = 1-depth 하위 디렉토리
        if split in ("train", "test"):
            split_dirs = [split]
        else:
            split_dirs = ["train", "test"]

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        # 1-depth 카테고리(혹은 제품) 디렉토리 자동 인식
        categories = [d for d in sorted(os.listdir(root))
                      if os.path.isdir(os.path.join(root, d))]

        # for cat in categories:
        #     cat_dir = os.path.join(root, cat)
        #     # split 디렉토리 없을 수 있으니, 재귀 탐색
        #     for sp in split_dirs:
        #         sp_dir = os.path.join(cat_dir, sp)
        #         search_dirs = [sp_dir] if os.path.isdir(sp_dir) else [cat_dir]  # 없으면 카테고리 루트에서 재귀
        #         for sdir in search_dirs:
        #             for ext in IMG_EXTS:
        #                 for p in glob.glob(os.path.join(sdir, "**", f"*{ext}"), recursive=True):
        #                     if "/ground_truth/" in p.replace("\\", "/"):
        #                         continue  # gt 폴더는 제외
        #                     if _is_image(p):
        #                         print(p)
        #                         add_sample(p, ds, cat, sp)
        """
        JISU ADDED
        데이터 명세서: https://github.com/amazon-science/spot-diff
        """
        for cat in categories:
            cat_dir = os.path.join(root, cat)
            for sp in split_dirs:
                sp_dir = os.path.join(cat_dir, sp)

                # ----- VisA 전용 경로 설정 -----
                if ds == "visa":
                    base_dirs = []
                    images_dir = os.path.join(cat_dir, "Data", "Images")
                    if os.path.isdir(images_dir):
                        if sp == "train":
                            # train: Normal만
                            base_dirs.append(os.path.join(images_dir, "Normal"))
                        elif sp == "test":
                            # test: Normal + Anomaly
                            base_dirs += [
                                os.path.join(images_dir, "Normal"),
                                os.path.join(images_dir, "Anomaly"),
                            ]
                        else:  # 'all'
                            base_dirs.append(images_dir)
                    else:
                        # 구조가 다를 때 폴백
                        base_dirs = [cat_dir]
                else:
                    # 기존 로직 유지
                    base_dirs = [sp_dir] if os.path.isdir(sp_dir) else [cat_dir]

                # ----- 실제 파일 스캔 -----
                for bdir in base_dirs:
                    if not os.path.isdir(bdir):
                        continue
                    # 확장자 대소문자 이슈 회피: 전체 스캔 후 _is_image로 필터
                    for p in glob.glob(os.path.join(bdir, "**", "*"), recursive=True):
                        if not os.path.isfile(p):
                            continue
                        if not _is_image(p):
                            continue
                        low = p.replace("\\", "/").lower()
                        # 마스크/어노테이션 폴더는 무조건 제외
                        if any(seg in low for seg in ["/ground_truth/", "/mask/", "/masks/", "/annotations/"]):
                            continue
                        add_sample(p, ds, cat, sp)

        if not samples:
            # 루트 하위 전체 재귀 (데이터 구조가 다를 때)
            for ext in IMG_EXTS:
                for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
                    if "/ground_truth/" in p.replace("\\", "/"):
                        continue
                    add_sample(p, ds, None, split if split in ("train", "test") else "all")

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def _load_mask(self, path: Optional[str], size: Tuple[int, int]) -> torch.Tensor:
        if not self.return_mask:
            return torch.tensor(0)  # placeholder
        if path is None or not os.path.exists(path):
            return torch.zeros((1, size[1], size[0]), dtype=torch.float32)
        m = Image.open(path).convert("L").resize(size, Image.NEAREST)
        t = transforms.ToTensor()(m)  # (1,H,W), [0,1]
        # 일부 마스크는 그레이스케일값일 수 있음 -> 이진화 근사
        t = (t > 0.5).float()
        return t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img = self._load_image(s.img_path)
        w, h = img.size
        img_t = self.transform(img)
        mask_t = self._load_mask(s.mask_path, (self.image_size, self.image_size)) if self.return_mask else torch.tensor(0)
        return {
            "image": img_t,                  # (3,H,W)
            "label": torch.tensor(s.label),  # 0 or 1
            "mask": mask_t,                  # (1,H,W) or scalar 0
            "meta": {
                "img_path": s.img_path,
                "mask_path": s.mask_path,
                "dataset": s.dataset,
                "category": s.category,
                "split": s.split,
                "orig_size": (h, w),
            }
        }

def make_dataloader(
    root: str,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    ds = AnomalyDataset(
        root=root,
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        return_mask=return_mask,
    )
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=_collate_with_meta,)   # JISU  ADDED

def make_concat_dataloader(
    roots: Dict[str, str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
) -> DataLoader:
    """
    Args:
        roots: {"mvtec": "/path/to/mvtec", "visa": "...", "mvtec_loco": "...", "goodsad": "..."}
    """
    datasets = []
    for name, path in roots.items():
        datasets.append(AnomalyDataset(
            root=path, dataset_name=name, split=split,
            image_size=image_size, return_mask=return_mask))
    concat = ConcatDataset(datasets)
    return DataLoader(concat, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=True,
                      collate_fn=_collate_with_meta,)   # JISU  ADDED

# =========================
# AUTO-DOWNLOAD INTEGRATION
# =========================

# --- 공인된/널리 쓰이는 배포 경로 (2025-10-29 확인) ---
# 참고: MVTec의 MyDrive 주소는 교체/갱신될 수 있음. 불가 시 페이지 참조 후 갱신 요망.
AUTO_DATASET_SOURCES = {
    # MVTec AD
    "mvtec": {
        "type": "direct",
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "needs_env_consent": "AGREE_MVTEC_LICENSE"
    },
    # MVTec LOCO AD
    "mvtec_loco": {
        "type": "direct",
        "url": "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz",
        "needs_env_consent": "AGREE_MVTEC_LICENSE"
    },
    # VisA — Amazon Research S3
    "visa": {
        "type": "direct",
        "url": "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        "needs_env_consent": None
    },
    # GoodsAD — Kaggle 미러
    "goodsad": {
        "type": "kaggle",
        "slug": "dtcrxs/goodsad",
        "needs_env_consent": None
    },
}

class _TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _human_size(n):
    for u in ["B","KB","MB","GB","TB","PB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}EB"

def _download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with _TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=f"Downloading {os.path.basename(out_path)}") as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    print(f"[download] saved: {out_path} ({_human_size(os.path.getsize(out_path))})")

def _extract_any(archive_path: str, dest_dir: str):
    print(f"[extract] {archive_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    low = archive_path.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return
    # tar(.gz/.bz2/.xz/...) 계열
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
        return
    except tarfile.ReadError as e:
        # 일부 플랫폼에 lzma 바인딩이 없을 수 있음 → 시스템 tar로 폴백
        print(f"[warn] Python tarfile failed ({e}). Trying system 'tar'...")
        cmd = ["tar", "-xf", archive_path, "-C", dest_dir]
        subprocess.check_call(cmd)

def _maybe_flatten_topdir(dst_dir: str):
    entries = [e for e in os.listdir(dst_dir) if not e.startswith(".")]
    if len(entries) == 1:
        only = os.path.join(dst_dir, entries[0])
        if os.path.isdir(only):
            tmp = dst_dir + "_tmp"
            os.makedirs(tmp, exist_ok=True)
            for it in os.listdir(only):
                shutil.move(os.path.join(only, it), os.path.join(tmp, it))
            shutil.rmtree(dst_dir)
            shutil.move(tmp, dst_dir)

def _has_enough_images(path: str, min_count: int = 50) -> bool:
    if not os.path.isdir(path):
        return False
    c = 0
    for ext in IMG_EXTS:
        c += len(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        if c >= min_count:
            return True
    return c >= min_count

def _download_mvtec_like(name: str, dest_root: str):
    src = AUTO_DATASET_SOURCES[name]
    env_flag = src.get("needs_env_consent")
    if env_flag:
        if os.environ.get(env_flag, "").lower() not in ("1", "true", "yes", "y"):
            raise RuntimeError(
                f"[{name}] 라이선스 동의가 필요합니다. 환경변수 {env_flag}=1 을 설정한 뒤 다시 실행하세요.\n"
                f"공식 다운로드 페이지: https://www.mvtec.com/company/research/datasets/{'mvtec-ad' if name=='mvtec' else 'mvtec-loco'}/downloads"
            )
    url = src["url"]
    archive_path = os.path.join(os.path.dirname(os.path.abspath(dest_root)), f"{name}.tar.xz")
    _download_file(url, archive_path)
    _extract_any(archive_path, dest_root)
    _maybe_flatten_topdir(dest_root)
    try: os.remove(archive_path)
    except Exception: pass

def _download_visa(dest_root: str):
    url = AUTO_DATASET_SOURCES["visa"]["url"]
    archive_path = os.path.join(os.path.dirname(os.path.abspath(dest_root)), "VisA_20220922.tar")
    _download_file(url, archive_path)
    _extract_any(archive_path, dest_root)
    _maybe_flatten_topdir(dest_root)
    try: os.remove(archive_path)
    except Exception: pass

def _download_goodsad_via_kaggle(dest_root: str):
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        raise RuntimeError(
            "[goodsad] Kaggle API가 필요합니다. `pip install kaggle` 후 "
            "~/.kaggle/kaggle.json 또는 KAGGLE_USERNAME/KAGGLE_KEY 환경변수를 설정하세요."
        )
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError(
            "[goodsad] Kaggle 인증 실패. ~/.kaggle/kaggle.json 혹은 환경변수를 점검하세요."
        ) from e
    slug = AUTO_DATASET_SOURCES["goodsad"]["slug"]
    dl_dir = os.path.dirname(os.path.abspath(dest_root))
    print(f"[goodsad] downloading Kaggle dataset: {slug}")
    api.dataset_download_files(slug, path=dl_dir, unzip=True, quiet=False)
    # unzip=True이면 이미 풀려있음 → 후보 폴더를 dest_root로 합치기
    cand = [p for p in glob.glob(os.path.join(dl_dir, "*")) if os.path.isdir(p) and "goods" in os.path.basename(p).lower()]
    if cand:
        src_dir = cand[0]
        os.makedirs(dest_root, exist_ok=True)
        for it in os.listdir(src_dir):
            shutil.move(os.path.join(src_dir, it), os.path.join(dest_root, it))
        shutil.rmtree(src_dir)
    _maybe_flatten_topdir(dest_root)

def ensure_dataset_ready(dataset_name: str, dest_root: str) -> str:
    """
    지정한 dataset_name(root 경로)에 이미지가 없으면 AUTO_DATASET_SOURCES로 다운로드합니다.
    Returns: dest_root
    """
    ds = dataset_name.lower()
    if _has_enough_images(dest_root):
        return dest_root

    print(f"[auto] '{ds}' not found at {dest_root}. Start downloading...")
    if ds == "visa":
        _download_visa(dest_root)
    elif ds in ("mvtec", "mvtec_loco"):
        _download_mvtec_like(ds, dest_root)
    elif ds == "goodsad":
        _download_goodsad_via_kaggle(dest_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not _has_enough_images(dest_root, min_count=10):
        raise RuntimeError(f"[auto] '{ds}' 다운로드/해제 후에도 이미지가 충분히 보이지 않습니다: {dest_root}")
    print(f"[auto] '{ds}' is ready at: {dest_root}")
    return dest_root

# --- 원본 함수 래핑: 로컬에 없으면 자동으로 받기 ---
_ORIG_make_dataloader = make_dataloader
def make_dataloader(
    root: str,
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
    shuffle: Optional[bool] = None,
):
    # root가 비어있거나 이미지가 없으면 자동 다운로드
    try:
        ensure_dataset_ready(dataset_name, root)
    except Exception as e:
        print(f"[warn] auto-download failed for {dataset_name}: {e}\n"
              f"→ 수동으로 내려받아 {root}에 두고 다시 시도하세요.")
    return _ORIG_make_dataloader(
        root=root, dataset_name=dataset_name, split=split, batch_size=batch_size,
        num_workers=num_workers, image_size=image_size, return_mask=return_mask, shuffle=shuffle
    )

_ORIG_make_concat_dataloader = make_concat_dataloader
def make_concat_dataloader(
    roots: Dict[str, str],
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    return_mask: bool = False,
):
    # 제공된 루트들 각각 확인/자동 다운로드
    ready = {}
    for name, path in roots.items():
        try:
            ready[name] = ensure_dataset_ready(name, path)
        except Exception as e:
            print(f"[warn] auto-download failed for {name}: {e}")
            ready[name] = path  # 실패 시 원래 경로로 시도(이미 존재할 수도 있음)
    return _ORIG_make_concat_dataloader(
        roots=ready, split=split, batch_size=batch_size, num_workers=num_workers,
        image_size=image_size, return_mask=return_mask
    )

# --- CLI: python data/dataset.py auto --name mvtec --dest /data/MVTecAD ---
def _build_auto_parser(subparsers):
    sp = subparsers.add_parser("auto", help="Ensure dataset exists; download if missing")
    sp.add_argument("--name", required=True, help="mvtec | visa | mvtec_loco | goodsad")
    sp.add_argument("--dest", required=True, help="Destination root (unzipped)")
    sp.set_defaults(func=_cli_auto)

def _cli_auto(args):
    ensure_dataset_ready(args.name, args.dest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset utilities")
    subs = parser.add_subparsers()
    _build_auto_parser(subs)
    if len(sys.argv) == 1:
        parser.print_help(); sys.exit(0)
    args = parser.parse_args(); args.func(args)
# ==== END AUTO-DOWNLOAD ===============================================

# JISU  ADDED
def _collate_with_meta(batch):
    """
    batch: List[{"image": Tensor(3,H,W), "label": Tensor(1) or int,
                 "mask": Tensor(...) or scalar, "meta": dict(...)}]
    Returns:
      - image: (B,3,H,W)
      - label: (B,)
      - mask:  (B,1,H,W) or (B,) placeholder
      - meta:  List[dict]  (그대로 유지; 스택하지 않음)
    """
    # images
    images = torch.stack([b["image"] for b in batch], dim=0)

    # labels (보장: long)
    labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)

    # masks: return_mask=False면 scalar 0이 올 수 있으니 형태 방어
    m0 = batch[0]["mask"]
    if isinstance(m0, torch.Tensor) and m0.dim() == 3:
        masks = torch.stack([b["mask"] for b in batch], dim=0)  # (B,1,H,W)
    else:
        masks = torch.tensor([0] * len(batch), dtype=torch.long)  # placeholder (B,)

    # meta는 스택하지 않고 리스트로 유지 (문자열/None 허용)
    metas = [b["meta"] for b in batch]

    return {"image": images, "label": labels, "mask": masks, "meta": metas}