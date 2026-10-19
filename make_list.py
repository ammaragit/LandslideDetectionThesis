import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
s1_dir = ROOT_DIR / "data" / "s1asc_part01"

all_files = sorted(s1_dir.glob("*.nc"))
patch_ids = [os.path.splitext(os.path.basename(f))[0] for f in all_files]

output_path = ROOT_DIR / "data" / "train_list.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for pid in patch_ids:
        f.write(pid + "\n")

print(f"Done. Written {len(patch_ids)} patch IDs to {output_path}")
