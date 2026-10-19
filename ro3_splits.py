from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
"""
RO3 — Leave-One-Region-Out Split Generator
Creates train/val/test split files for each held-out region.
Held-out regions: chimanimani, hiroshima, indonesia
Train regions: all remaining regions

Run once before training:
    python ro3_splits.py
"""
import os
import glob
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
NPZ_DIR      = rstr(ROOT_DIR / "data" / "preprocessed")
OUTPUT_DIR   = rstr(ROOT_DIR / "ro3_splits")
HELD_OUT     = ["chimanimani", "hiroshima", "indonesia"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Group patches by region ───────────────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
all_ids   = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

region_map = defaultdict(list)
for pid in all_ids:
    region = pid.split("_s1")[0]   # e.g. chimanimani_s1asc_001 → chimanimani
    region_map[region].append(pid)

print(f"Total patches: {len(all_ids)}")
print(f"Regions found:")
for r, ids in sorted(region_map.items()):
    ls_note = ""
    print(f"  {r:<20}: {len(ids):>4} patches")

# ── Create splits for each held-out region ────────────────────────────────────
for held_region in HELD_OUT:
    if held_region not in region_map:
        print(f"\nWARNING: region '{held_region}' not found in dataset — skipping")
        continue

    # Test = held-out region patches only
    test_ids  = region_map[held_region]

    # Train + val = all other regions combined
    other_ids = []
    for r, ids in region_map.items():
        if r != held_region:
            other_ids.extend(ids)

    # 90/10 train/val split within other regions
    import numpy as np
    rng       = np.random.default_rng(3407)
    perm      = rng.permutation(len(other_ids))
    val_count = max(int(len(other_ids) * 0.10), 1)
    val_ids   = [other_ids[i] for i in perm[:val_count]]
    train_ids = [other_ids[i] for i in perm[val_count:]]

    # Save split files
    split_dir = os.path.join(OUTPUT_DIR, f"holdout_{held_region}")
    os.makedirs(split_dir, exist_ok=True)

    for fname, ids in [("train.txt", train_ids),
                       ("val.txt",   val_ids),
                       ("test.txt",  test_ids)]:
        with open(os.path.join(split_dir, fname), "w") as f:
            f.write("\n".join(ids))

    print(f"\nHeld-out: {held_region}")
    print(f"  Train : {len(train_ids)} patches "
          f"({len(set(p.split('_s1')[0] for p in train_ids))} regions)")
    print(f"  Val   : {len(val_ids)} patches")
    print(f"  Test  : {len(test_ids)} patches (held-out region)")
    print(f"  Saved to: {split_dir}")

print(f"\nAll RO3 splits saved to: {OUTPUT_DIR}")
print("Next step: python train_ro3.py --held_out chimanimani")
