import os
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
L4S_ROOT = ROOT_DIR / "Landslide4Sense"
OUT_DIR = L4S_ROOT / "preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

BANDS_SAR = [7, 10, 11]
BANDS_OPT = [3, 2, 1]


def read_h5_image(path):
    with h5py.File(path, "r") as f:
        for key in ["img", "image", "data", list(f.keys())[0]]:
            if key in f:
                img = f[key][:]
                break
    if img.ndim == 3 and img.shape[0] == 14:
        img = img.transpose(1, 2, 0)
    return img.astype(np.float32)


def read_h5_mask(path):
    with h5py.File(path, "r") as f:
        for key in ["mask", "label", "gt", list(f.keys())[0]]:
            if key in f:
                mask = f[key][:]
                break
    return mask.astype(np.int64).squeeze()


def normalize_band(arr, vmin, vmax):
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, vmin, vmax)
    return (arr - vmin) / (vmax - vmin + 1e-8)


def process_split(split_name, img_dir, mask_dir=None):
    img_files = sorted(Path(img_dir).glob("*.h5"))
    print(f"\n{split_name}: {len(img_files)} images")
    saved = 0
    skipped = 0

    for img_path in tqdm(img_files, desc=split_name):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        out_path = OUT_DIR / f"l4s_{split_name}_{img_id}.npz"

        if out_path.exists():
            saved += 1
            continue

        try:
            img = read_h5_image(img_path)
            sar_ch = [normalize_band(img[:, :, b], 0, 10000) for b in BANDS_SAR]
            sar = np.stack(sar_ch, axis=0).astype(np.float32)

            opt_ch = [normalize_band(img[:, :, b], 0, 10000) for b in BANDS_OPT]
            opt = np.stack(opt_ch, axis=0).astype(np.float32)

            slope = normalize_band(img[:, :, 12], 0, 90)
            dem = normalize_band(img[:, :, 13], 0, 8800)
            dem_ch = np.stack([slope, dem, dem], axis=0).astype(np.float32)

            if mask_dir is not None:
                mask_path = Path(mask_dir) / img_id.replace("image", "mask")
                mask_path = mask_path.with_suffix(".h5")
                if mask_path.exists():
                    mask = read_h5_mask(mask_path)
                else:
                    mask = np.zeros((128, 128), dtype=np.int64)
            else:
                mask = np.zeros((128, 128), dtype=np.int64)

            np.savez_compressed(out_path, sar=sar, opt=opt, dem=dem_ch, mask=mask)
            saved += 1

        except Exception as e:
            print(f"  Error {img_id}: {e}")
            skipped += 1

    print(f"  Saved: {saved} | Skipped: {skipped}")
    return saved


def main():
    print("=" * 60)
    print("Landslide4Sense Preprocessor")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    total = 0
    total += process_split(split_name="train", img_dir=L4S_ROOT / "TrainData" / "img", mask_dir=L4S_ROOT / "TrainData" / "mask")
    total += process_split(split_name="val", img_dir=L4S_ROOT / "ValidData" / "img", mask_dir=None)
    total += process_split(split_name="test", img_dir=L4S_ROOT / "TestData" / "img", mask_dir=None)

    print(f"\n{'=' * 60}")
    print(f"Total .npz files saved: {total}")
    print(f"Output directory: {OUT_DIR}")
    print("\nNext step: python train.py --config configs/l4s_config.yml --model m1")


if __name__ == "__main__":
    main()
