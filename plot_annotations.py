"""Plot annotations on 20 evenly-spaced images as a sanity check."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ANNOTATIONS = Path("data/annotations.json")
IMAGES_DIR = Path("data/images")
OUTPUT = Path("annotation_check.png")

COLORS = [
    (255, 0, 0),     # 0: red
    (0, 255, 0),     # 1: green
    (0, 0, 255),     # 2: blue
    (255, 255, 0),   # 3: yellow
    (255, 0, 255),   # 4: magenta
    (0, 255, 255),   # 5: cyan
    (255, 128, 0),   # 6: orange
    (128, 0, 255),   # 7: purple
    (0, 128, 255),   # 8: light blue
    (255, 128, 128), # 9: pink
]

with open(ANNOTATIONS) as f:
    data = json.load(f)

images = data["images"]
n = len(images)
print(f"Total annotated images: {n}")

# Pick 20 evenly spaced
indices = np.linspace(0, n - 1, min(20, n), dtype=int)
selected = [images[i] for i in indices]

fig, axes = plt.subplots(4, 5, figsize=(25, 20))
axes = axes.flatten()

for ax_idx, sample in enumerate(selected):
    img_path = IMAGES_DIR / sample["filename"]
    img = cv2.imread(str(img_path))
    if img is None:
        axes[ax_idx].set_title(f"MISSING: {sample['filename'][:30]}...", fontsize=7)
        axes[ax_idx].axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[ax_idx].imshow(img)

    for lm in sample["landmarks"]:
        u, v = lm["u"], lm["v"]
        lid = lm["id"]
        color = np.array(COLORS[lid % len(COLORS)]) / 255.0
        axes[ax_idx].plot(u, v, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=1)
        axes[ax_idx].annotate(
            str(lid), (u, v), color="white", fontsize=7, fontweight="bold",
            xytext=(4, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6),
        )

    short_name = sample["filename"][:25] + "..."
    n_lm = len(sample["landmarks"])
    axes[ax_idx].set_title(f"{short_name} ({n_lm} pts)", fontsize=7)
    axes[ax_idx].axis("off")

plt.suptitle("Annotation Sanity Check (20 images)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"Saved to {OUTPUT}")
plt.show()
