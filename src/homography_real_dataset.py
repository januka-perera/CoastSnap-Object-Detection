"""Real-pair dataset for homography training with GCP supervision.

Each sample is a real (reference, target) image pair with ground-truth
homography computed from annotated GCP correspondences. The GT homography
maps target coordinates to reference coordinates.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from .homography_utils import homography_to_four_point

logger = logging.getLogger(__name__)


def load_gcps_from_mat(mat_path: str, image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load GCP pixel coordinates from a .mat file.

    Scales from annotation resolution (NU x NV) to actual image resolution.

    Returns:
        coords: (N, 2) array of (u, v) pixel coordinates at image resolution.
        image_size: (width, height) of the image.
    """
    data = scipy.io.loadmat(str(mat_path))
    uv = data["metadata"]["gcps"][0, 0]["UVpicked"][0, 0]  # (N, 2)
    lcp = data["metadata"]["geom"][0, 0]["lcp"][0, 0]
    nu = lcp["NU"][0, 0].item()
    nv = lcp["NV"][0, 0].item()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img.shape[:2]

    coords = np.zeros_like(uv, dtype=np.float64)
    coords[:, 0] = uv[:, 0] * (img_w / nu)
    coords[:, 1] = uv[:, 1] * (img_h / nv)
    return coords, (img_w, img_h)


def _photometric_augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random photometric augmentation to a grayscale image."""
    img = image.copy()
    img = img + rng.uniform(-0.2, 0.2)
    img = img * rng.uniform(0.7, 1.3)
    gamma = rng.uniform(0.7, 1.5)
    img = np.clip(img, 0, 1)
    img = np.power(img, gamma)
    noise_std = rng.uniform(0, 0.03)
    img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


class RealGCPHomographyDataset(Dataset):
    """Real image pairs with GT homography from GCP annotations.

    For each image with a .mat file:
    1. Load reference and target GCPs at image resolution
    2. Scale both to working resolution
    3. Compute H_gt (target -> reference) from GCP correspondences
    4. Convert H_gt to 4-point corner displacements
    5. Validate homography (reject degenerate/extreme cases)

    Returns dict with:
        input: (2, H, W) stacked [ref, target] grayscale in [0, 1]
        four_point: (8,) GT corner displacements at working resolution
        target_gcps: (N, 2) target GCPs at working resolution
        ref_gcps: (N, 2) reference GCPs at working resolution
        mask: (H, W) static mask
        filename: str
    """

    def __init__(
        self,
        gcp_dir: str,
        images_dir: str,
        reference_image_path: str,
        reference_mat_path: str,
        working_size: tuple[int, int] = (512, 384),
        mask_path: str = None,
        sample_indices: list = None,
        augment: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            gcp_dir: Directory containing .mat GCP files.
            images_dir: Directory containing snap images.
            reference_image_path: Path to the reference image.
            reference_mat_path: Path to the reference .mat file.
            working_size: (width, height) to resize images to.
            mask_path: Path to binary mask (white=static).
            sample_indices: If provided, select only these indices from the
                sorted list of valid samples (for train/val/test splitting).
            augment: Whether to apply photometric augmentation to target.
            seed: Random seed for augmentation reproducibility.
        """
        self.working_w, self.working_h = working_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.images_dir = Path(images_dir)
        self.gcp_dir = Path(gcp_dir)

        # Load reference image
        ref_bgr = cv2.imread(str(reference_image_path))
        if ref_bgr is None:
            raise FileNotFoundError(f"Reference not found: {reference_image_path}")
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.ref_full_h, self.ref_full_w = ref_bgr.shape[:2]
        self.reference = cv2.resize(
            ref_gray, (self.working_w, self.working_h),
        ).astype(np.float32) / 255.0

        # Reference GCPs at working resolution
        ref_gcps_full, _ = load_gcps_from_mat(reference_mat_path, reference_image_path)
        self.ref_gcps = ref_gcps_full.copy()
        self.ref_gcps[:, 0] *= self.working_w / self.ref_full_w
        self.ref_gcps[:, 1] *= self.working_h / self.ref_full_h
        self.num_gcps = self.ref_gcps.shape[0]

        # Load mask
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.mask = cv2.resize(
                mask, (self.working_w, self.working_h),
            ).astype(np.float32) / 255.0
        else:
            self.mask = np.ones((self.working_h, self.working_w), dtype=np.float32)

        # Discover and validate all samples
        ref_name = Path(reference_image_path).name
        self.samples = self._discover_samples(ref_name)

        # Apply index filtering for train/val/test split
        if sample_indices is not None:
            self.samples = [self.samples[i] for i in sample_indices
                            if i < len(self.samples)]

        logger.info(f"RealGCPHomographyDataset: {len(self.samples)} samples")

    def _discover_samples(self, ref_name: str) -> list[dict]:
        """Find all valid image/.mat pairs and precompute GT homographies."""
        samples = []

        mat_files = sorted(self.gcp_dir.glob("*.mat"))
        for mat_path in mat_files:
            img_name = mat_path.name.replace(".plan.", ".snap.").replace(".mat", ".jpg")
            img_path = self.images_dir / img_name

            if not img_path.exists() or img_name == ref_name:
                continue

            try:
                target_gcps_full, (img_w, img_h) = load_gcps_from_mat(
                    str(mat_path), str(img_path),
                )
                if target_gcps_full.shape[0] != self.num_gcps:
                    logger.warning(f"Skipping {img_name}: GCP count mismatch")
                    continue

                # Scale target GCPs to working resolution
                target_gcps = target_gcps_full.copy()
                target_gcps[:, 0] *= self.working_w / img_w
                target_gcps[:, 1] *= self.working_h / img_h

                # Compute GT homography at working resolution (target -> reference)
                H_gt, _ = cv2.findHomography(
                    target_gcps, self.ref_gcps, method=0,  # all points, no RANSAC
                )
                if H_gt is None:
                    logger.warning(f"Skipping {img_name}: findHomography failed")
                    continue

                # Validate: reject degenerate homographies
                det = np.linalg.det(H_gt)
                if abs(det) < 1e-6:
                    logger.warning(f"Skipping {img_name}: degenerate H (det={det:.2e})")
                    continue

                cond = np.linalg.cond(H_gt)
                if cond > 1e6:
                    logger.warning(f"Skipping {img_name}: ill-conditioned H (cond={cond:.1e})")
                    continue

                # Convert to 4-point displacement
                four_point = homography_to_four_point(H_gt, (self.working_w, self.working_h))

                # Reject extreme displacements (> half the image width)
                max_disp = np.max(np.abs(four_point))
                if max_disp > self.working_w * 0.5:
                    logger.warning(f"Skipping {img_name}: extreme displacement ({max_disp:.1f}px)")
                    continue

                samples.append({
                    "img_path": str(img_path),
                    "img_name": img_name,
                    "target_gcps": target_gcps.astype(np.float32),
                    "four_point": four_point.astype(np.float32),
                })

            except Exception as e:
                logger.warning(f"Skipping {mat_path.name}: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load target image at working resolution
        target_bgr = cv2.imread(sample["img_path"])
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        target = cv2.resize(
            target_gray, (self.working_w, self.working_h),
        ).astype(np.float32) / 255.0

        if self.augment:
            target = _photometric_augment(target, self.rng)

        input_pair = np.stack([self.reference, target], axis=0)  # (2, H, W)

        return {
            "input": torch.from_numpy(input_pair),
            "four_point": torch.from_numpy(sample["four_point"]),
            "target_gcps": torch.from_numpy(sample["target_gcps"]),
            "ref_gcps": torch.from_numpy(self.ref_gcps.astype(np.float32)),
            "mask": torch.from_numpy(self.mask),
            "filename": sample["img_name"],
        }


def split_indices(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15,
                  seed: int = 42) -> tuple[list, list, list]:
    """Split indices into train/val/test sets.

    Args:
        n: Total number of samples.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    test_idx = sorted(indices[n_train + n_val:])

    return train_idx, val_idx, test_idx
