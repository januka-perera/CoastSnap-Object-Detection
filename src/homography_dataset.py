"""Synthetic and real-pair datasets for deep homography training.

SyntheticHomographyDataset generates unlimited (ref, warped_ref, H_gt)
training pairs on-the-fly from a single reference image using random
homography perturbations and photometric augmentations.

RealPairDataset provides all non-reference images paired with the
reference for optional self-supervised fine-tuning.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .homography_utils import (
    four_point_to_homography,
    random_homography_perturbation,
    warp_image,
)

logger = logging.getLogger(__name__)


def _photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random photometric augmentation to a grayscale image.

    Applies brightness, contrast, gamma, and Gaussian noise to simulate
    lighting variation between reference and target images.

    Args:
        image: (H, W) float32 grayscale image in [0, 1].
        rng: NumPy random generator.

    Returns:
        Augmented image clipped to [0, 1].
    """
    img = image.copy()

    # Brightness shift
    img = img + rng.uniform(-0.2, 0.2)

    # Contrast scale
    img = img * rng.uniform(0.7, 1.3)

    # Gamma correction
    gamma = rng.uniform(0.7, 1.5)
    img = np.clip(img, 0, 1)
    img = np.power(img, gamma)

    # Gaussian noise
    noise_std = rng.uniform(0, 0.03)
    img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)

    return np.clip(img, 0, 1).astype(np.float32)


class SyntheticHomographyDataset(Dataset):
    """Generates synthetic homography training pairs on-the-fly.

    From a single reference image, creates unlimited training data by:
    1. Randomly perturbing corner positions to get H_gt
    2. Warping the reference by H_inv to create a synthetic "target"
    3. Applying photometric augmentation to the target only
    4. Optionally applying a mask to focus loss on static regions

    Returns:
        input_pair: (2, H, W) float32 — stacked [ref, target] grayscale
        four_point: (8,) float32 — GT corner displacements
        mask: (H, W) float32 — static region mask (1=static, 0=dynamic)
    """

    def __init__(
        self,
        reference_image_path: str,
        working_size: tuple[int, int] = (512, 384),
        max_displacement: float = 32.0,
        samples_per_epoch: int = 10000,
        mask_path: str = None,
        augment: bool = True,
        seed: int = None,
    ):
        """
        Args:
            reference_image_path: Path to the reference snap image.
            working_size: (width, height) to resize reference to.
            max_displacement: Max corner displacement in pixels at working res.
            samples_per_epoch: Virtual dataset length.
            mask_path: Path to binary mask (white=static). If None, all pixels used.
            augment: Whether to apply photometric augmentation to target.
            seed: Random seed for reproducibility.
        """
        self.working_w, self.working_h = working_size
        self.max_displacement = max_displacement
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        # Load and cache reference at working resolution, grayscale
        ref_bgr = cv2.imread(str(reference_image_path))
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.reference = cv2.resize(ref_gray, (self.working_w, self.working_h)).astype(np.float32) / 255.0

        # Load mask if provided
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.mask = cv2.resize(mask, (self.working_w, self.working_h)).astype(np.float32) / 255.0
        else:
            self.mask = np.ones((self.working_h, self.working_w), dtype=np.float32)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict:
        # Generate random homography perturbation
        four_point, H = random_homography_perturbation(
            (self.working_w, self.working_h),
            self.max_displacement,
            self.rng,
        )

        # Warp reference by H_inv to create synthetic target
        # If H maps src corners to dst corners, then the "target" image
        # is the reference viewed from a different perspective.
        # We want: target warped by H -> reference
        # So target = warp(reference, H_inv)
        H_inv = np.linalg.inv(H)
        target = warp_image(self.reference, H_inv, (self.working_w, self.working_h))

        # Photometric augmentation on target only
        if self.augment:
            target = _photometric_augment(target, self.rng)

        # Stack as 2-channel input: [ref, target]
        input_pair = np.stack([self.reference, target], axis=0)  # (2, H, W)

        return {
            "input": torch.from_numpy(input_pair),
            "four_point": torch.from_numpy(four_point.astype(np.float32)),
            "mask": torch.from_numpy(self.mask),
        }


class RealPairDataset(Dataset):
    """All non-reference images paired with the reference.

    For optional self-supervised fine-tuning (Stage 2) where no GT
    homography is available. Loss is computed as masked photometric
    difference between warped target and reference.
    """

    def __init__(
        self,
        images_dir: str,
        reference_image_path: str,
        working_size: tuple[int, int] = (512, 384),
        mask_path: str = None,
    ):
        """
        Args:
            images_dir: Directory containing all snap images.
            reference_image_path: Path to the reference image (excluded from pairs).
            working_size: (width, height) to resize images to.
            mask_path: Path to binary mask for static regions.
        """
        self.working_w, self.working_h = working_size
        self.images_dir = Path(images_dir)

        ref_name = Path(reference_image_path).name
        self.image_paths = sorted([
            p for p in self.images_dir.glob("*.jpg")
            if p.name != ref_name
        ])

        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir} (excluding {ref_name})")

        # Load reference
        ref_bgr = cv2.imread(str(reference_image_path))
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.reference = cv2.resize(ref_gray, (self.working_w, self.working_h)).astype(np.float32) / 255.0

        # Load mask
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.mask = cv2.resize(mask, (self.working_w, self.working_h)).astype(np.float32) / 255.0
        else:
            self.mask = np.ones((self.working_h, self.working_w), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]

        # Load target image
        target_bgr = cv2.imread(str(img_path))
        if target_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        target = cv2.resize(target_gray, (self.working_w, self.working_h)).astype(np.float32) / 255.0

        # Stack as 2-channel input
        input_pair = np.stack([self.reference, target], axis=0)

        return {
            "input": torch.from_numpy(input_pair),
            "mask": torch.from_numpy(self.mask),
            "filename": img_path.name,
        }
