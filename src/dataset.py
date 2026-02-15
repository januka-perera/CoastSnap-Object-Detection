"""Dataset class with augmentation for heatmap regression."""

import json
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap_utils import generate_heatmaps_batch


class LandmarkDataset(Dataset):
    """Dataset for coastal landmark localisation.

    Loads images and landmark annotations, applies augmentations,
    and generates ground-truth Gaussian heatmaps on the fly.
    """

    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        num_landmarks: int,
        input_size: tuple[int, int] = (512, 384),
        output_size: tuple[int, int] = (128, 96),
        sigma: float = 2.0,
        augment: bool = False,
        aug_config: Optional[dict] = None,
    ):
        """
        Args:
            annotations_file: Path to annotations.json.
            images_dir: Path to directory containing images.
            num_landmarks: Total number of landmarks (N).
            input_size: (width, height) to resize images to.
            output_size: (width, height) of output heatmaps.
            sigma: Gaussian sigma for heatmap generation.
            augment: Whether to apply data augmentation.
            aug_config: Augmentation hyperparameters dict.
        """
        self.images_dir = Path(images_dir)
        self.num_landmarks = num_landmarks
        self.input_w, self.input_h = input_size
        self.output_w, self.output_h = output_size
        self.sigma = sigma
        self.augment = augment

        with open(annotations_file, "r") as f:
            data = json.load(f)
        self.samples = data["images"]

        self.transform = self._build_transforms(aug_config or {})

        # ImageNet normalisation
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _build_transforms(self, cfg: dict) -> A.Compose:
        """Build albumentations transform pipeline."""
        transforms = []

        if self.augment:
            # Geometric augmentations
            transforms.extend([
                A.ShiftScaleRotate(
                    shift_limit=cfg.get("translate_limit", 0.15),
                    scale_limit=cfg.get("scale_limit", 0.2),
                    rotate_limit=cfg.get("rotation_limit", 20),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.8,
                ),
            ])
            if cfg.get("horizontal_flip", False):
                transforms.append(A.HorizontalFlip(p=0.5))

            # Photometric augmentations
            transforms.extend([
                A.ColorJitter(
                    brightness=cfg.get("brightness_limit", 0.3),
                    contrast=cfg.get("contrast_limit", 0.3),
                    saturation=cfg.get("saturation_limit", 0.3),
                    hue=cfg.get("hue_shift_limit", 10) / 360.0,
                    p=0.8,
                ),
                A.GaussianBlur(
                    blur_limit=(3, cfg.get("blur_limit", 5)),
                    p=0.3,
                ),
                A.GaussNoise(p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])

        # Always resize to input size
        transforms.append(
            A.Resize(height=self.input_h, width=self.input_w)
        )

        return A.Compose(
            transforms,
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        img_path = self.images_dir / sample["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Build keypoints and visibility
        keypoints = []
        visibility = np.zeros(self.num_landmarks, dtype=bool)
        landmark_ids = []

        for lm in sample.get("landmarks", []):
            lid = lm["id"]
            landmark_ids.append(lid)
            visibility[lid] = True
            keypoints.append((float(lm["u"]), float(lm["v"])))

        # Apply augmentations
        transformed = self.transform(image=image, keypoints=keypoints)
        image = transformed["image"]
        aug_keypoints = transformed["keypoints"]

        # Update visibility for keypoints that went out of bounds
        norm_keypoints = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        for i, lid in enumerate(landmark_ids):
            if i < len(aug_keypoints):
                kp = aug_keypoints[i]
                u, v = kp[0], kp[1]
                if 0 <= u < self.input_w and 0 <= v < self.input_h:
                    norm_keypoints[lid, 0] = u / (self.input_w - 1)
                    norm_keypoints[lid, 1] = v / (self.input_h - 1)
                else:
                    visibility[lid] = False

        # Normalise image
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        keypoints_tensor = torch.from_numpy(norm_keypoints)
        visibility_tensor = torch.from_numpy(visibility)

        # Generate heatmaps
        heatmaps = generate_heatmaps_batch(
            keypoints_tensor.unsqueeze(0),
            visibility_tensor.unsqueeze(0),
            self.output_w,
            self.output_h,
            self.sigma,
        ).squeeze(0)  # (N, H, W)

        return {
            "image": image,
            "heatmaps": heatmaps,
            "keypoints": keypoints_tensor,
            "visibility": visibility_tensor,
            "filename": sample["filename"],
            "orig_width": orig_w,
            "orig_height": orig_h,
        }
