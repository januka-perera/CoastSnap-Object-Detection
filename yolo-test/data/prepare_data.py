"""
Prepare data for the YOLO + keypoint pipeline.

Supported annotation formats
------------------------------
1. CVAT XML  (CVAT 1.1 export — <box xtl ytl xbr ybr>)          ← primary
2. COCO JSON  (top-level "annotations" + "categories")
3. Landmark JSON  (project-native: top-level "images" + "landmarks")

For CVAT XML and COCO formats the bounding boxes come directly from the
annotations.  For the keypoint crop model the centre of each bounding box
is used as the keypoint label (override with --keypoints-file if you have
separate keypoint annotations).

For the landmark format, bounding boxes are AUTO-GENERATED as square
patches centred on each landmark; the landmark coordinate is the keypoint.

Output
------
yolo_data/
    dataset.yaml
    images/{train,val,test}/
    labels/{train,val,test}/   <- YOLO format: <class cx cy w h> per line

crops/
    {train,val,test}/
        images/  <- JPEG crops
        labels/  <- "<x_norm> <y_norm>" (keypoint relative to crop)

Usage
-----
    # CVAT XML (auto-detected)
    python prepare_data.py

    # Explicit format
    python prepare_data.py --format cvat_xml

    # Single class (all labels → class 0 "object")
    python prepare_data.py --single-class

    # Custom patch size for landmark format
    python prepare_data.py --format landmark --patch-size 256
"""

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(annotations_path: Path) -> str:
    suffix = annotations_path.suffix.lower()
    if suffix == ".xml":
        return "cvat_xml"
    if suffix == ".json":
        with open(annotations_path) as f:
            data = json.load(f)
        if "annotations" in data and "categories" in data:
            return "coco"
        if "images" in data:
            return "landmark"
    return "unknown"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int = 42):
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def make_yolo_dirs(base: Path) -> dict:
    dirs = {}
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (img_dir, lbl_dir)
    return dirs


def make_crop_dirs(base: Path) -> dict:
    dirs = {}
    for split in ("train", "val", "test"):
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        dirs[split] = base / split
    return dirs


def write_yolo_yaml(yolo_dir: Path, class_names: list):
    content = {
        "path":  str(yolo_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(yolo_dir / "dataset.yaml", "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"  Wrote: {yolo_dir / 'dataset.yaml'}")


def save_crop(
    img_bgr: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    kp_u: float, kp_v: float,
    crop_path: Path,
    label_path: Path,
):
    """Save one crop and its normalised keypoint label."""
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return
    bw = x2 - x1
    bh = y2 - y1
    kp_x = float(np.clip((kp_u - x1) / bw, 0.0, 1.0))
    kp_y = float(np.clip((kp_v - y1) / bh, 0.0, 1.0))
    cv2.imwrite(str(crop_path), crop)
    label_path.write_text(f"{kp_x:.6f} {kp_y:.6f}\n")


# ---------------------------------------------------------------------------
# CVAT XML format
# ---------------------------------------------------------------------------

def parse_cvat_xml(xml_path: Path) -> tuple:
    """
    Parse a CVAT 1.1 XML export.

    Returns
    -------
    images     : list of dicts with keys: name, width, height, boxes
                 boxes: list of dicts with keys: label, x1, y1, x2, y2
    all_labels : sorted list of unique label names found in <box> elements
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    images     = []
    all_labels = set()

    for img_el in root.findall("image"):
        name   = img_el.get("name")
        width  = int(img_el.get("width"))
        height = int(img_el.get("height"))
        boxes  = []

        for box_el in img_el.findall("box"):
            label = box_el.get("label")
            x1    = float(box_el.get("xtl"))
            y1    = float(box_el.get("ytl"))
            x2    = float(box_el.get("xbr"))
            y2    = float(box_el.get("ybr"))
            boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            all_labels.add(label)

        if boxes:                        # skip images with no annotations
            images.append({"name": name, "width": width, "height": height, "boxes": boxes})

    return images, sorted(all_labels)


def prepare_from_cvat_xml(
    xml_path: Path,
    images_dir: Path,
    yolo_dir: Path,
    crops_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    single_class: bool = False,
    class_names: Optional[list] = None,
):
    """
    Convert CVAT XML annotations to YOLO format + crops.

    Keypoint for each crop = centre of its bounding box
    (since CVAT XML has no explicit keypoint annotations).
    """
    images, xml_labels = parse_cvat_xml(xml_path)
    print(f"  Parsed {len(images)} annotated images, labels: {xml_labels}")

    # Build label → class-index mapping
    if single_class:
        label_to_cls = {lbl: 0 for lbl in xml_labels}
        class_names  = ["object"]
    else:
        if class_names is None:
            class_names = xml_labels               # use label names from XML
        label_to_cls = {name: idx for idx, name in enumerate(class_names)}

    tr, va, te = split_indices(len(images), train_ratio, val_ratio, seed)
    split_map  = {
        **{i: "train" for i in tr},
        **{i: "val"   for i in va},
        **{i: "test"  for i in te},
    }

    yolo_dirs = make_yolo_dirs(yolo_dir)
    crop_dirs = make_crop_dirs(crops_dir)

    missing = 0
    total_boxes = 0

    for i, meta in enumerate(images):
        fname = meta["name"]
        src   = images_dir / fname
        if not src.exists():
            print(f"  [WARN] image not found: {src}")
            missing += 1
            continue

        split = split_map[i]
        img_out_dir, lbl_out_dir = yolo_dirs[split]
        W, H = meta["width"], meta["height"]

        # Copy image to YOLO directory
        dst = img_out_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        img_bgr    = None        # lazy-load for crop generation
        yolo_lines = []

        for box in meta["boxes"]:
            label = box["label"]
            if label not in label_to_cls:
                continue                           # skip unknown labels

            cls_idx = label_to_cls[label]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            # Clamp to image bounds
            x1 = max(0.0, x1);  y1 = max(0.0, y1)
            x2 = min(W, x2);    y2 = min(H, y2)
            bw = x2 - x1;       bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            # YOLO label (normalised cx, cy, w, h)
            cx = (x1 + bw / 2) / W
            cy = (y1 + bh / 2) / H
            wn = bw / W
            hn = bh / H
            yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            total_boxes += 1

            # Crop for keypoint model — keypoint = bbox centre
            if img_bgr is None:
                img_bgr = cv2.imread(str(src))

            if img_bgr is not None:
                ix1, iy1 = int(x1), int(y1)
                ix2, iy2 = int(x2), int(y2)
                kp_u = x1 + bw / 2       # centre x in image space
                kp_v = y1 + bh / 2       # centre y in image space
                stem  = Path(fname).stem
                cname = f"{stem}_{cls_idx}.jpg"
                save_crop(
                    img_bgr, ix1, iy1, ix2, iy2, kp_u, kp_v,
                    crop_dirs[split] / "images" / cname,
                    crop_dirs[split] / "labels" / cname.replace(".jpg", ".txt"),
                )

        stem = Path(fname).stem
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

    write_yolo_yaml(yolo_dir, class_names)

    n = len(images) - missing
    splits = {"train": tr, "val": va, "test": te}
    print(f"\nCVAT XML preparation complete.")
    print(f"  Images   : {n} annotated  ({missing} missing)")
    print(f"  Boxes    : {total_boxes}")
    print(f"  Classes  : {class_names}")
    for name, idx_list in splits.items():
        print(f"  {name:5s}  : {len(idx_list)} images")


# ---------------------------------------------------------------------------
# COCO JSON format
# ---------------------------------------------------------------------------

def prepare_from_coco(
    annotations: dict,
    images_dir: Path,
    yolo_dir: Path,
    crops_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    single_class: bool = False,
):
    cats    = {c["id"]: c["name"] for c in annotations["categories"]}
    id2img  = {img["id"]: img for img in annotations["images"]}
    cat_ids = sorted(cats.keys())

    if single_class:
        label_to_cls = {cid: 0 for cid in cat_ids}
        class_names  = ["object"]
    else:
        label_to_cls = {cid: idx for idx, cid in enumerate(cat_ids)}
        class_names  = [cats[k] for k in cat_ids]

    img_anns: dict = {}
    for ann in annotations["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    img_ids  = list(img_anns.keys())
    tr, va, te = split_indices(len(img_ids), train_ratio, val_ratio, seed)
    split_map  = {
        **{img_ids[i]: "train" for i in tr},
        **{img_ids[i]: "val"   for i in va},
        **{img_ids[i]: "test"  for i in te},
    }

    yolo_dirs = make_yolo_dirs(yolo_dir)
    crop_dirs = make_crop_dirs(crops_dir)

    for img_id, anns in img_anns.items():
        meta  = id2img[img_id]
        fname = meta["file_name"]
        src   = images_dir / fname
        if not src.exists():
            print(f"  [WARN] missing: {src}")
            continue

        split = split_map[img_id]
        img_out_dir, lbl_out_dir = yolo_dirs[split]
        W, H = meta["width"], meta["height"]

        shutil.copy2(src, img_out_dir / fname)

        img_bgr    = None
        yolo_lines = []

        for ann in anns:
            cls_idx = label_to_cls[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            wn = w / W
            hn = h / H
            yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

            # Keypoint: explicit if present, otherwise bbox centre
            kps = ann.get("keypoints", [])
            kp_u, kp_v = None, None
            for k in range(0, len(kps), 3):
                if kps[k + 2] > 0:
                    kp_u, kp_v = kps[k], kps[k + 1]
                    break
            if kp_u is None:
                kp_u = x + w / 2
                kp_v = y + h / 2

            if img_bgr is None:
                img_bgr = cv2.imread(str(src))
            if img_bgr is not None:
                ix1, iy1 = max(0, int(x)),         max(0, int(y))
                ix2, iy2 = min(W, int(x + w)),     min(H, int(y + h))
                stem  = Path(fname).stem
                cname = f"{stem}_{ann['id']}.jpg"
                save_crop(
                    img_bgr, ix1, iy1, ix2, iy2, kp_u, kp_v,
                    crop_dirs[split] / "images" / cname,
                    crop_dirs[split] / "labels" / cname.replace(".jpg", ".txt"),
                )

        stem = Path(fname).stem
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

    write_yolo_yaml(yolo_dir, class_names)
    print(f"COCO preparation complete. Classes: {class_names}")


# ---------------------------------------------------------------------------
# Landmark JSON format
# ---------------------------------------------------------------------------

def prepare_from_landmarks(
    annotations: dict,
    images_dir: Path,
    yolo_dir: Path,
    crops_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    patch_size: int = 128,
    class_names: Optional[list] = None,
    single_class: bool = False,
):
    images  = annotations["images"]
    all_ids = sorted({lm["id"] for img in images for lm in img.get("landmarks", [])})

    if single_class:
        class_names  = ["object"]
        id_to_cls    = {lm_id: 0 for lm_id in all_ids}
    else:
        if class_names is None:
            class_names = [f"landmark_{i}" for i in all_ids]
        id_to_cls = {lm_id: idx for idx, lm_id in enumerate(all_ids)}

    tr, va, te = split_indices(len(images), train_ratio, val_ratio, seed)
    split_map  = {
        **{i: "train" for i in tr},
        **{i: "val"   for i in va},
        **{i: "test"  for i in te},
    }

    yolo_dirs = make_yolo_dirs(yolo_dir)
    crop_dirs = make_crop_dirs(crops_dir)
    half      = patch_size // 2

    for i, meta in enumerate(images):
        fname = meta["filename"]
        src   = images_dir / fname
        if not src.exists():
            print(f"  [WARN] missing: {src}")
            continue

        split = split_map[i]
        img_out_dir, lbl_out_dir = yolo_dirs[split]
        W, H = meta["width"], meta["height"]

        dst = img_out_dir / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        img_bgr    = None
        yolo_lines = []

        for lm in meta.get("landmarks", []):
            u, v    = lm["u"], lm["v"]
            cls_idx = id_to_cls[lm["id"]]

            x1 = max(0, int(u) - half)
            y1 = max(0, int(v) - half)
            x2 = min(W, x1 + patch_size)
            y2 = min(H, y1 + patch_size)
            bw = x2 - x1;  bh = y2 - y1

            cx = (x1 + bw / 2) / W
            cy = (y1 + bh / 2) / H
            yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")

            if img_bgr is None:
                img_bgr = cv2.imread(str(src))
            if img_bgr is not None:
                stem  = Path(fname).stem
                cname = f"{stem}_lm{lm['id']}.jpg"
                save_crop(
                    img_bgr, x1, y1, x2, y2, u, v,
                    crop_dirs[split] / "images" / cname,
                    crop_dirs[split] / "labels" / cname.replace(".jpg", ".txt"),
                )

        stem = Path(fname).stem
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

    write_yolo_yaml(yolo_dir, class_names)
    print(f"Landmark preparation complete. Classes: {class_names}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for YOLO + keypoint pipeline"
    )
    parser.add_argument("--config",       default="../configs/config.yaml")
    parser.add_argument("--format",       default="auto",
                        choices=["auto", "cvat_xml", "coco", "landmark"])
    parser.add_argument("--single-class", action="store_true",
                        help="Map all labels to class 0 ('object').")
    parser.add_argument("--patch-size",   type=int, default=128,
                        help="Bbox patch size for landmark format (pixels).")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_cfg = cfg["data"]

    images_dir       = Path(data_cfg["images_dir"])
    annotations_file = Path(data_cfg["annotations_file"])
    yolo_dir         = Path(data_cfg["yolo_data_dir"])
    crops_dir        = Path(data_cfg["crops_dir"])
    train_ratio      = data_cfg["train_ratio"]
    val_ratio        = data_cfg["val_ratio"]
    seed             = data_cfg.get("seed", 42)
    class_names      = data_cfg.get("classes", None)

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(annotations_file)
        if fmt == "unknown":
            print(f"ERROR: cannot detect format of {annotations_file}. "
                  "Use --format explicitly.")
            sys.exit(1)

    print(f"Format   : {fmt}")
    print(f"Input    : {annotations_file}")
    print(f"Images   : {images_dir}")
    print(f"YOLO out : {yolo_dir}")
    print(f"Crops out: {crops_dir}")
    print(f"Single class: {args.single_class}")

    if fmt == "cvat_xml":
        prepare_from_cvat_xml(
            annotations_file, images_dir, yolo_dir, crops_dir,
            train_ratio, val_ratio, seed,
            single_class=args.single_class,
            class_names=None if args.single_class else class_names,
        )

    elif fmt == "coco":
        with open(annotations_file) as f:
            annotations = json.load(f)
        prepare_from_coco(
            annotations, images_dir, yolo_dir, crops_dir,
            train_ratio, val_ratio, seed,
            single_class=args.single_class,
        )

    elif fmt == "landmark":
        with open(annotations_file) as f:
            annotations = json.load(f)
        prepare_from_landmarks(
            annotations, images_dir, yolo_dir, crops_dir,
            train_ratio, val_ratio, seed,
            patch_size=args.patch_size,
            class_names=class_names,
            single_class=args.single_class,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
