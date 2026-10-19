import os
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
L4S_ROOT = ROOT_DIR / "Landslide4Sense"
NPZ_DIR = L4S_ROOT / "preprocessed"

all_files = sorted(NPZ_DIR.glob("l4s_train_*.npz"))
all_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

print(f"Total labeled patches: {len(all_ids)}")

ls_ids = []
bg_ids = []
for pid in all_ids:
    d = np.load(NPZ_DIR / f"{pid}.npz")
    if d["mask"].sum() > 0:
        ls_ids.append(pid)
    else:
        bg_ids.append(pid)

print(f"With landslide:    {len(ls_ids)} ({100 * len(ls_ids) / len(all_ids):.1f}%)")
print(f"Without landslide: {len(bg_ids)}")

rng = np.random.default_rng(3407)
perm = rng.permutation(len(all_ids))
n = len(all_ids)
n_test = int(n * 0.20)
n_val = int(n * 0.10)
n_train = n - n_test - n_val

train_ids = [all_ids[i] for i in perm[:n_train]]
val_ids = [all_ids[i] for i in perm[n_train:n_train + n_val]]
test_ids = [all_ids[i] for i in perm[n_train + n_val:]]

print(f"\nSplits: train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)}")

for fname, ids in [("l4s_split_train.txt", train_ids), ("l4s_split_val.txt", val_ids), ("l4s_split_test.txt", test_ids)]:
    path = L4S_ROOT / fname
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(ids))
    print(f"Saved: {path}")

all_list_path = L4S_ROOT / "l4s_train_list.txt"
with open(all_list_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_ids))
print(f"Saved: {all_list_path}")
print("\nDone.")
