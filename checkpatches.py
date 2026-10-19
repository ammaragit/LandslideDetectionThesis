import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
NPZ_DIR = ROOT_DIR / "data" / "preprocessed"
OUT_PATH = ROOT_DIR / "figures" / "dataset" / "figure1b_sen12_sample_patch.png"


def find_landslide_patch(npz_files, min_positive_pixels=200):
    for path in npz_files:
        data = np.load(path)
        mask = data["mask"]
        if mask.sum() >= min_positive_pixels:
            patch_id = os.path.splitext(os.path.basename(path))[0]
            return path, patch_id, int(mask.sum())
    raise RuntimeError(f"No patch found with at least {min_positive_pixels} landslide pixels.")


def to_hwc(arr):
    return np.transpose(arr, (1, 2, 0))


def normalize_channel(ch):
    lo, hi = ch.min(), ch.max()
    if hi - lo < 1e-8:
        return np.zeros_like(ch)
    return (ch - lo) / (hi - lo)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    npz_files = sorted(glob.glob(str(NPZ_DIR / "*.npz")))
    print(f"Found {len(npz_files)} patches in {NPZ_DIR}")
    if len(npz_files) == 0:
        print("No .npz files found -- check NPZ_DIR.")
        return

    path, patch_id, n_positive = find_landslide_patch(npz_files)
    print(f"Selected patch: {patch_id}  ({n_positive} landslide pixels)")

    data = np.load(path)
    sar = data["sar"]
    opt = data["opt"]
    dem = data["dem"]
    mask = data["mask"]

    sar_composite = to_hwc(sar)
    sar_composite = np.stack([
        normalize_channel(sar_composite[:, :, 0]),
        normalize_channel(sar_composite[:, :, 1]),
        normalize_channel(sar_composite[:, :, 2]),
    ], axis=-1)

    opt_rgb = np.clip(to_hwc(opt) * 4, 0, 1)
    dem_single = dem[0]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].imshow(sar_composite)
    axes[0].set_title("RO1 Temporal SAR Descriptor", fontsize=11)
    axes[1].imshow(opt_rgb)
    axes[1].set_title("Optical (RGB)", fontsize=12)
    im2 = axes[2].imshow(dem_single, cmap="terrain")
    axes[2].set_title("DEM (elevation)", fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[3].set_title("Ground Truth", fontsize=12)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Sen12Landslides sample patch ({patch_id}) — {n_positive} landslide-positive pixels", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
