from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
"""
Landslide4Sense Preprocessor — v2 (Paper 2)
Reads .h5 files and saves .npz files matching the Sen12Landslides format
so all existing models (CM1, CM2, M1, M2, M3) run without changes.

MODALITY_MODE controls what goes into the SAR/change slot:

  "raw"          — Original: B8(NIR), B11(SWIR1), B12(SWIR2) raw bands
                   This caused M1/M2/M3 non-convergence (cross-modal
                   redundancy with the optical slot, same Sentinel-2 sensor)

  "indices"      — Proposed fix: NDVI, NDWI, NBR spectral indices
                   Informationally orthogonal to raw RGB even though derived
                   from the same sensor. Expected to restore evidential
                   fusion stability.

  "dem_features" — Alternative: Slope, Aspect, Curvature from DEM (B14)
                   Geometrically independent from both optical and spectral
                   index channels. Pure morphological information.

Usage:
    # Run all three modes in sequence for Paper 2 experiments
    python preprocess_l4s_v2.py --mode raw
    python preprocess_l4s_v2.py --mode indices
    python preprocess_l4s_v2.py --mode dem_features
"""
import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.ndimage import sobel

# ── Paths ─────────────────────────────────────────────────────────────────────
L4S_ROOT = rstr(ROOT_DIR / "Landslide4Sense")

# ── Band indices (0-based, 14-band Sentinel-2 + ALOS image) ──────────────────
# B1-B12 = Sentinel-2 (0-indexed: 0–11)
# B13    = ALOS PALSAR slope (index 12)
# B14    = ALOS PALSAR DEM  (index 13)
BANDS_OPT = [3,  2,  1]   # B4(Red), B3(Green), B2(Blue) → optical slot
BANDS_DEM = [12, 13, 13]  # Slope, DEM, DEM (repeated)   → DEM slot

# Indices needed for spectral index computation (0-based)
IDX_B2  = 1    # Blue
IDX_B3  = 2    # Green
IDX_B4  = 3    # Red
IDX_B8  = 7    # NIR
IDX_B11 = 10   # SWIR1
IDX_B12 = 11   # SWIR2


# ── I/O helpers ───────────────────────────────────────────────────────────────
def read_h5_image(path):
    """Read 14-band image from h5 file. Returns (H, W, 14) float32."""
    with h5py.File(path, 'r') as f:
        for key in ['img', 'image', 'data', list(f.keys())[0]]:
            if key in f:
                img = f[key][:]
                break
    if img.ndim == 3 and img.shape[0] == 14:
        img = img.transpose(1, 2, 0)   # (14,H,W) → (H,W,14)
    return img.astype(np.float32)


def read_h5_mask(path):
    """Read binary mask from h5 file. Returns (H, W) int64."""
    with h5py.File(path, 'r') as f:
        for key in ['mask', 'label', 'gt', list(f.keys())[0]]:
            if key in f:
                mask = f[key][:]
                break
    return mask.astype(np.int64).squeeze()


def normalize_band(arr, vmin, vmax):
    """Clip and min-max normalize to [0, 1]."""
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, vmin, vmax)
    return (arr - vmin) / (vmax - vmin + 1e-8)


def normalize_01(arr):
    """Min-max normalize a single channel to [0, 1]."""
    arr = np.nan_to_num(arr, nan=0.0)
    vmin, vmax = arr.min(), arr.max()
    return (arr - vmin) / (vmax - vmin + 1e-8)


# ── SAR slot factories ────────────────────────────────────────────────────────
def make_sar_raw(img):
    """
    Mode: raw
    B8(NIR), B11(SWIR1), B12(SWIR2) — original approach.
    Same sensor as optical slot → caused evidential fusion collapse.
    Kept here as the controlled baseline.
    """
    channels = []
    for b in [IDX_B8, IDX_B11, IDX_B12]:
        channels.append(normalize_band(img[:, :, b], 0, 10000))
    return np.stack(channels, axis=0).astype(np.float32)  # (3,H,W)


def make_sar_indices(img):
    """
    Mode: indices (proposed)
    NDVI, NDWI, NBR — three spectral indices computed from Sentinel-2.

    Even though derived from the same sensor as the optical slot,
    spectral indices encode fundamentally different quantities:
      NDVI: vegetation density — low on landslide scars, high in intact forest
      NDWI: surface water / soil moisture — elevated near failure zones
      NBR:  disturbed bare soil — strongest signal for landslide scars

    These are informationally orthogonal to raw RGB, which is what the
    Dempster-Shafer evidential fusion mechanism actually requires — not
    different sensors per se, but different information content.
    """
    eps = 1e-8

    # Scale from [0,1] back to reflectance-like range for index computation
    NIR  = img[:, :, IDX_B8]  * 10000
    Red  = img[:, :, IDX_B4]  * 10000
    Green= img[:, :, IDX_B3]  * 10000
    SWIR = img[:, :, IDX_B11] * 10000

    # Normalized Difference Vegetation Index
    # Range [-1, 1]: high = intact vegetation, low = bare soil / landslide scar
    NDVI = (NIR - Red)   / (NIR + Red   + eps)

    # Normalized Difference Water Index (McFeeters 1996)
    # Range [-1, 1]: high = surface water / saturated soil
    NDWI = (Green - NIR) / (Green + NIR + eps)

    # Normalized Burn Ratio
    # Range [-1, 1]: strongly positive on bare, disturbed land
    NBR  = (NIR - SWIR)  / (NIR + SWIR  + eps)

    # Stack and normalize each index to [0, 1]
    stack = np.stack([NDVI, NDWI, NBR], axis=0).astype(np.float32)
    for i in range(3):
        stack[i] = normalize_01(stack[i])

    return stack  # (3,H,W)


def make_sar_dem_features(img):
    """
    Mode: dem_features (alternative)
    Slope, Aspect, Curvature derived from the DEM channel (B14).

    These are purely morphological — no spectral information whatsoever —
    making this slot genuinely independent of both optical and spectral
    index channels. Landslide susceptibility is strongly correlated with
    slope angle and slope aspect (south-facing slopes in monsoon zones),
    and slope curvature governs drainage patterns and run-out zones.
    """
    dem = img[:, :, 13]   # B14 = DEM elevation, already in [0,1] after read

    # Gradient components for slope and aspect
    dz_dx = sobel(dem.astype(np.float64), axis=1)
    dz_dy = sobel(dem.astype(np.float64), axis=0)

    # Slope magnitude (proxy for gradient steepness)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)

    # Aspect (direction of slope: N/S/E/W)
    aspect = np.arctan2(dz_dy, dz_dx)

    # Plan curvature (sum of second derivatives — positive = convex ridge,
    # negative = concave hollow, strongly associated with landslide run-out)
    d2z_dx2 = sobel(dz_dx, axis=1)
    d2z_dy2 = sobel(dz_dy, axis=0)
    curvature = d2z_dx2 + d2z_dy2

    stack = np.stack(
        [slope, aspect, curvature], axis=0).astype(np.float32)
    for i in range(3):
        stack[i] = normalize_01(stack[i])

    return stack  # (3,H,W)


# ── Main processing ───────────────────────────────────────────────────────────
def process_split(split_name, img_dir, out_dir, mode, mask_dir=None):
    """Process one split and save .npz files."""
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.h5")))
    print(f"\n{split_name}: {len(img_files)} images  [mode={mode}]")

    saved = 0
    skipped = 0

    for img_path in tqdm(img_files, desc=split_name):
        img_id   = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"l4s_{split_name}_{img_id}.npz")

        if os.path.exists(out_path):
            saved += 1
            continue

        try:
            img = read_h5_image(img_path)   # (128,128,14) in [0,1] range

            # ── SAR slot (mode-dependent) ──────────────────────────────────
            if mode == "raw":
                sar = make_sar_raw(img)
            elif mode == "indices":
                sar = make_sar_indices(img)
            elif mode == "dem_features":
                sar = make_sar_dem_features(img)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # ── Optical slot: RGB ──────────────────────────────────────────
            opt_ch = []
            for b in BANDS_OPT:
                opt_ch.append(normalize_band(img[:, :, b], 0, 10000))
            opt = np.stack(opt_ch, axis=0).astype(np.float32)

            # ── DEM slot: Slope + DEM ──────────────────────────────────────
            slope_ch = normalize_band(img[:, :, 12], 0, 90)
            dem_ch   = normalize_band(img[:, :, 13], 0, 8800)
            dem      = np.stack([slope_ch, dem_ch, dem_ch],
                                axis=0).astype(np.float32)

            # ── Mask ───────────────────────────────────────────────────────
            if mask_dir is not None:
                mask_path = os.path.join(
                    mask_dir,
                    img_id.replace("image", "mask") + ".h5")
                if os.path.exists(mask_path):
                    mask = read_h5_mask(mask_path)
                else:
                    mask = np.zeros((128, 128), dtype=np.int64)
            else:
                mask = np.zeros((128, 128), dtype=np.int64)

            np.savez_compressed(
                out_path, sar=sar, opt=opt, dem=dem, mask=mask)
            saved += 1

        except Exception as e:
            print(f"  Error {img_id}: {e}")
            skipped += 1

    print(f"  Saved: {saved} | Skipped: {skipped}")
    return saved


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="indices",
        choices=["raw", "indices", "dem_features"],
        help=(
            "raw:         B8/B11/B12 raw bands (baseline — caused non-convergence)\n"
            "indices:     NDVI/NDWI/NBR spectral indices (proposed fix)\n"
            "dem_features: Slope/Aspect/Curvature from DEM (alternative)"
        )
    )
    args = parser.parse_args()
    mode = args.mode

    # Each mode gets its own preprocessed directory
    out_dir = os.path.join(
        L4S_ROOT, f"preprocessed_{mode}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print(f"Landslide4Sense Preprocessor — mode: {mode}")
    print(f"SAR slot description:")
    desc = {
        "raw":         "B8(NIR), B11(SWIR1), B12(SWIR2) — original, caused collapse",
        "indices":     "NDVI, NDWI, NBR — spectral indices (proposed fix)",
        "dem_features":"Slope, Aspect, Curvature from DEM — morphological",
    }
    print(f"  {desc[mode]}")
    print(f"Output: {out_dir}")
    print("=" * 65)

    total = 0

    total += process_split(
        split_name="train",
        img_dir=os.path.join(L4S_ROOT, "TrainData", "img"),
        out_dir=out_dir,
        mode=mode,
        mask_dir=os.path.join(L4S_ROOT, "TrainData", "mask"),
    )
    total += process_split(
        split_name="val",
        img_dir=os.path.join(L4S_ROOT, "ValidData", "img"),
        out_dir=out_dir,
        mode=mode,
        mask_dir=None,
    )
    total += process_split(
        split_name="test",
        img_dir=os.path.join(L4S_ROOT, "TestData", "img"),
        out_dir=out_dir,
        mode=mode,
        mask_dir=None,
    )

    print(f"\n{'='*65}")
    print(f"Total .npz files saved: {total}")
    print(f"Output directory: {out_dir}")
    print(f"\nNext steps:")
    print(f"  1. Update l4s_config.yml → data.root_dir to use "
          f"preprocessed_{mode}/")
    print(f"  2. python l4s_splits.py")
    print(f"  3. python train.py --config configs/l4s_config.yml --model m1")


if __name__ == "__main__":
    main()
