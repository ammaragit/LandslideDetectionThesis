import json
import os
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
BASE = ROOT_DIR / "data"
OUT_DIR = BASE / "preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

all_s1_dirs = sorted(BASE.glob("s1asc_part*"))
s1_files = []
for d in all_s1_dirs:
    s1_files += sorted(d.glob("*.nc"))

S2_DIRS = sorted(BASE.glob("s2_part*"))

print(f"S1 folders  : {len(all_s1_dirs)}")
print(f"S1 patches  : {len(s1_files)}")
print(f"S2 folders  : {len(S2_DIRS)}")

print("Building S2 lookup table...")
s2_lookup = {}
for s2_dir in S2_DIRS:
    for f in s2_dir.glob("*.nc"):
        fname = os.path.basename(f)
        s2_lookup[fname] = str(f)
print(f"S2 total patches in lookup: {len(s2_lookup)}")


def find_s2(pid):
    s2_pid = pid.replace("s1asc", "s2") + ".nc"
    return s2_lookup.get(s2_pid, None)


def get_pre_post(ds):
    try:
        pp = ds.attrs.get("pre_post_dates", None)
        if pp and pp != "":
            if isinstance(pp, str):
                pp = json.loads(pp.replace("'", '"'))
            return int(pp["pre"]), int(pp["post"])
    except Exception:
        pass
    n = ds.dims["time"]
    return max(0, n - 2), n - 1


skipped = 0
existing = 0
saved = 0

print(f"\nProcessing {len(s1_files)} S1 patches...")

for s1_path in tqdm(s1_files):
    pid = s1_path.stem
    s2_path = find_s2(pid)

    if s2_path is None:
        skipped += 1
        continue

    out_path = OUT_DIR / f"{pid}.npz"
    if out_path.exists():
        existing += 1
        continue

    try:
        s1 = xr.open_dataset(s1_path)
        s2 = xr.open_dataset(s2_path)

        pre_idx, post_idx = get_pre_post(s1)

        vv_pre = s1["VV"].values[pre_idx].astype(np.float32)
        vh_pre = s1["VH"].values[pre_idx].astype(np.float32)
        vv_post = s1["VV"].values[post_idx].astype(np.float32)
        vh_post = s1["VH"].values[post_idx].astype(np.float32)

        vv_pre = np.nan_to_num(vv_pre, nan=0.0)
        vh_pre = np.nan_to_num(vh_pre, nan=0.0)
        vv_post = np.nan_to_num(vv_post, nan=0.0)
        vh_post = np.nan_to_num(vh_post, nan=0.0)

        vv_pre = np.clip(vv_pre, -50.0, 10.0)
        vh_pre = np.clip(vh_pre, -50.0, 10.0)
        vv_post = np.clip(vv_post, -50.0, 10.0)
        vh_post = np.clip(vh_post, -50.0, 10.0)

        vv_change = vv_post - vv_pre
        vh_change = vh_post - vh_pre

        xpol_pre = vh_pre - vv_pre
        xpol_post = vh_post - vv_post
        xpol_change = xpol_post - xpol_pre

        sar = np.stack([vv_change, vh_change, xpol_change], axis=0)
        sar = np.clip(sar, -30.0, 30.0)
        sar = (sar + 30.0) / 60.0

        opt = np.stack([
            s2["B04"].values[post_idx],
            s2["B03"].values[post_idx],
            s2["B02"].values[post_idx],
        ], axis=0).astype(np.float32)
        opt = np.nan_to_num(opt, nan=0.0)
        opt = np.clip(opt, 0.0, 10000.0) / 10000.0

        dem = s2["DEM"].values[post_idx].astype(np.float32)
        dem = np.nan_to_num(dem, nan=0.0)
        dem = np.clip(dem, 0.0, 8800.0) / 8800.0
        dem = np.stack([dem, dem, dem], axis=0)

        mask = s2["MASK"].values[post_idx].astype(np.int64)
        mask = np.nan_to_num(mask, nan=0).astype(np.int64)

        s1.close()
        s2.close()

        np.savez_compressed(out_path, sar=sar, opt=opt, dem=dem, mask=mask)
        saved += 1

    except Exception as e:
        print(f"Error {pid}: {e}")
        skipped += 1

print(f"\n{'='*50}")
print(f"Saved    : {saved} new patches")
print(f"Existing : {existing} already done")
print(f"Skipped  : {skipped} no S2 match")
print(f"Total npz: {saved + existing}")
print(f"Output   : {OUT_DIR}")
