"""
Keypoint crop dataset.

Loads pre-cropped object images and their single-keypoint labels, applies
conservative augmentation that keeps coordinate transforms correct, and
returns a normalised image tensor together with the Gaussian ground-truth
heatmap and the raw normalised keypoint coordinate.

Expected directory layout
-------------------------
<crops_dir>/
    images/   <- JPEG crops   (*.jpg or *.png)
    labels/   <- text files   (<stem>.txt)

Label file format (one line):
    <x_norm> <y_norm>
where x_norm, y_norm are in [0, 1] relative to the crop dimensions.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from heatmap_utils import generate_heatmap


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class KeypointCropDataset(Dataset):
    """
    Dataset of cropped objects with a single keypoint label each.

    Parameters
    ----------
    crops_dir  : Path to split directory (e.g. ``crops/train``).
    input_size : (height, width) to resize crops before the model.
    heatmap_size : (height, width) of the output Gaussian heatmap.
    sigma      : Gaussian sigma in heatmap pixels.
    augment    : Whether to apply conservative data augmentation.
    seed       : Random seed for augmentation.
    """

    def __init__(
        self,
        crops_dir,
        input_size=(224, 224),
        heatmap_size=(56, 56),
        sigma: float = 2.5,
        augment: bool = False,
        seed: int = 42,
        class_idx: int = None,
    ):
        """
        Parameters
        ----------
        class_idx : If given, only load crops whose filename ends with
                    ``_<class_idx>.jpg`` (e.g. ``_0.jpg`` for class 0).
                    None loads all crops regardless of class.
        """
        self.crops_dir   = Path(crops_dir)
        self.input_h, self.input_w = input_size
        self.hm_h,    self.hm_w   = heatmap_size
        self.sigma   = sigma
        self.augment = augment
        self.rng     = random.Random(seed)
        self.np_rng  = np.random.default_rng(seed)

        img_dir = self.crops_dir / "images"
        all_paths = sorted(img_dir.glob("*.jpg"))
        if not all_paths:
            all_paths = sorted(img_dir.glob("*.png"))

        # Filter to a single class if requested.
        # Crop filenames follow the pattern: <image_stem>_<cls_idx>.jpg
        if class_idx is not None:
            suffix = f"_{class_idx}.jpg"
            self.img_paths = [p for p in all_paths if p.name.endswith(suffix)]
        else:
            self.img_paths = all_paths

        lbl_dir = self.crops_dir / "labels"
        self.labels: list = []
        for p in self.img_paths:
            lbl = lbl_dir / p.with_suffix(".txt").name
            if lbl.exists():
                parts = lbl.read_text().strip().split()
                self.labels.append((float(parts[0]), float(parts[1])))
            else:
                self.labels.append((0.5, 0.5))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img = cv2.imread(str(self.img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x_norm, y_norm = self.labels[idx]

        if self.augment:
            img, x_norm, y_norm = self._augment(img, x_norm, y_norm)

        img = cv2.resize(img, (self.input_w, self.input_h))

        heatmap = generate_heatmap(x_norm, y_norm, self.hm_h, self.hm_w, self.sigma)

        # Normalise image
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img_tensor  = torch.from_numpy(img.transpose(2, 0, 1))         # (3, H, W)
        hm_tensor   = torch.from_numpy(heatmap).unsqueeze(0)           # (1, H, W)

        return {
            "image":    img_tensor,
            "heatmap":  hm_tensor,
            "keypoint": torch.tensor([x_norm, y_norm], dtype=torch.float32),
        }

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, img: np.ndarray, x_norm: float, y_norm: float):
        """
        Apply conservative augmentations.  Every geometric transform is
        propagated to the keypoint coordinate.
        """
        H, W = img.shape[:2]

        # --- Horizontal flip ---
        if self.rng.random() < 0.5:
            img    = cv2.flip(img, 1)
            x_norm = 1.0 - x_norm

        # --- Small rotation ±12° ---
        if self.rng.random() < 0.5:
            angle = self.rng.uniform(-12.0, 12.0)
            cx, cy = W / 2.0, H / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
            pt  = np.array([x_norm * W, y_norm * H, 1.0])
            new = M @ pt
            x_norm = float(np.clip(new[0] / W, 0.0, 1.0))
            y_norm = float(np.clip(new[1] / H, 0.0, 1.0))

        # --- Random scale (crop back to original size) ---
        if self.rng.random() < 0.5:
            scale  = self.rng.uniform(0.85, 1.15)
            new_w  = int(W * scale)
            new_h  = int(H * scale)
            scaled = cv2.resize(img, (new_w, new_h))

            if scale > 1.0:
                # random crop back to (H, W)
                x_off = self.rng.randint(0, new_w - W)
                y_off = self.rng.randint(0, new_h - H)
                img    = scaled[y_off:y_off + H, x_off:x_off + W]
                x_norm = float(np.clip((x_norm * new_w - x_off) / W, 0.0, 1.0))
                y_norm = float(np.clip((y_norm * new_h - y_off) / H, 0.0, 1.0))
            else:
                # pad back to (H, W)
                pad_x = (W - new_w) // 2
                pad_y = (H - new_h) // 2
                img = cv2.copyMakeBorder(
                    scaled,
                    pad_y, H - new_h - pad_y,
                    pad_x, W - new_w - pad_x,
                    cv2.BORDER_REFLECT_101,
                )
                x_norm = float(np.clip((x_norm * new_w + pad_x) / W, 0.0, 1.0))
                y_norm = float(np.clip((y_norm * new_h + pad_y) / H, 0.0, 1.0))

        # --- Brightness / contrast jitter (colour only) ---
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.8, 1.2)    # contrast
            beta  = self.rng.uniform(-20.0, 20.0) # brightness
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # --- Small translation (≤5 % of image size) ---
        if self.rng.random() < 0.3:
            max_tx = max(1, int(W * 0.05))
            max_ty = max(1, int(H * 0.05))
            tx = self.rng.randint(-max_tx, max_tx)
            ty = self.rng.randint(-max_ty, max_ty)
            M   = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)
            x_norm = float(np.clip(x_norm + tx / W, 0.0, 1.0))
            y_norm = float(np.clip(y_norm + ty / H, 0.0, 1.0))

        return img, x_norm, y_norm
