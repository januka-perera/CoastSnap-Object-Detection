"""Plot UV coordinates on the image as a sanity check."""
import scipy.io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load mat file
mat_path = Path("1577820840.Wed.Jan.01_05_34_00.AEST.2020.manly.plan.CherylWhite.mat")
data = scipy.io.loadmat(str(mat_path))
uv = data["metadata"]["gcps"][0, 0]["UVpicked"][0, 0]
lcp = data["metadata"]["geom"][0, 0]["lcp"][0, 0]
NU = lcp["NU"][0, 0].item()
NV = lcp["NV"][0, 0].item()

print("UVpicked shape:", uv.shape)
print("UVpicked values:")
print(uv)
print("Annotation resolution: %d x %d" % (NU, NV))

# Load image
img_path = Path("1577820840.Wed.Jan.01_05_34_00.AEST.2020.manly.snap.CherylWhite.jpg")
image = cv2.imread(str(img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_h, img_w = image.shape[:2]

# Scale UV from annotation resolution to actual image resolution
scale_u = img_w / NU
scale_v = img_h / NV

# Plot
fig, ax = plt.subplots(1, 1)
ax.imshow(image)

# uv is (N, 2) where columns are [u, v]
us = uv[:, 0] * scale_u
vs = uv[:, 1] * scale_v

for i in range(len(us)):
    ax.plot(us[i], vs[i], "ro", markersize=8)
    ax.annotate(str(i), (us[i], vs[i]), color="yellow", fontsize=10,
                fontweight="bold", xytext=(5, 5), textcoords="offset points")

ax.set_title("UV Landmarks on Image")
ax.axis("off")
plt.tight_layout()
plt.savefig("sanity_check.png", dpi=150)
plt.show()
print("Saved sanity_check.png")
