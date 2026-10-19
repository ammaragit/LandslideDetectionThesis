import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
IMG_DIR = ROOT_DIR / "Landslide4Sense" / "TrainData" / "img"
MASK_DIR = ROOT_DIR / "Landslide4Sense" / "TrainData" / "mask"
OUT_PATH = ROOT_DIR / "figures" / "dataset" / "figure1_l4s_sample_patch.png"
OUT_PATH_RAW_VS_IDX = ROOT_DIR / "figures" / "dataset" / "figure4_raw_vs_indices.png"

IDX_B2, IDX_B3, IDX_B4 = 1, 2, 3
IDX_B8, IDX_B11, IDX_B12 = 7, 10, 11
IDX_DEM = 13
EPS = 1e-8


def read_h5_array(path, candidate_keys):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        for key in candidate_keys + keys:
            if key in f:
                arr = f[key][:]
                break
    return arr


def load_image(path):
    img = read_h5_array(path, ["img", "image", "data"])
    if img.ndim == 3 and img.shape[0] == 14:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32)


def load_mask(path):
    mask = read_h5_array(path, ["mask", "label", "gt"])
    return mask.astype(np.int64).squeeze()


def find_landslide_patch(img_files, mask_dir, min_positive_pixels=200):
    for img_path in img_files:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = Path(mask_dir) / img_id.replace("image", "mask")
        mask_path = mask_path.with_suffix(".h5")
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        if mask.sum() >= min_positive_pixels:
            return img_path, mask_path, img_id
    raise RuntimeError(f"No patch found with at least {min_positive_pixels} landslide pixels.")


def normalize_for_display(band, low_pct=2, high_pct=98):
    vmin = np.percentile(band, low_pct)
    vmax = np.percentile(band, high_pct)
    if vmax <= vmin:
        vmax = vmin + EPS
    band = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    return band


img_files = sorted(Path(IMG_DIR).glob("*.h5"))
img_path, _, img_id = find_landslide_patch(img_files, MASK_DIR)
print(f"Selected patch {img_id}")

img = load_image(img_path)
mask_path = Path(MASK_DIR) / img_id.replace("image", "mask")
mask = load_mask(mask_path.with_suffix(".h5"))

ndvi = (img[:, :, IDX_B8] - img[:, :, IDX_B4]) / (img[:, :, IDX_B8] + img[:, :, IDX_B4] + EPS)
ndwi = (img[:, :, IDX_B8] - img[:, :, IDX_B11]) / (img[:, :, IDX_B8] + img[:, :, IDX_B11] + EPS)
nbr = (img[:, :, IDX_B8] - img[:, :, IDX_B12]) / (img[:, :, IDX_B8] + img[:, :, IDX_B12] + EPS)

fig, axes = plt.subplots(1, 6, figsize=(18, 3))
axes[0].imshow(normalize_for_display(img[:, :, [IDX_B4, IDX_B3, IDX_B2]].astype(np.float32)))
axes[0].set_title("RGB")
axes[1].imshow(ndvi, cmap="RdYlGn")
axes[1].set_title("NDVI")
axes[2].imshow(ndwi, cmap="Blues")
axes[2].set_title("NDWI")
axes[3].imshow(nbr, cmap="RdYlGn")
axes[3].set_title("NBR")
axes[4].imshow(img[:, :, IDX_DEM], cmap="terrain")
axes[4].set_title("DEM")
axes[5].imshow(mask, cmap="gray")
axes[5].set_title("Ground Truth")
for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
