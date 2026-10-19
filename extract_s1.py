import tarfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
download_dir = ROOT_DIR / "data" / "downloads" / "data_harmonized" / "s1asc"
out_base = ROOT_DIR / "data"

tar_files = sorted(download_dir.glob("*.tar.gz"))
print(f"Found {len(tar_files)} tar files")

for tar_path in tar_files:
    part_name = tar_path.name.replace(".tar.gz", "")
    out_folder = out_base / part_name

    if out_folder.exists() and len(list(out_folder.glob("*.nc"))) > 0:
        print(f"Skipping {part_name} — already extracted")
        continue

    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {part_name}...")

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(out_folder)

    count = len(list(out_folder.glob("*.nc")))
    print(f"  Done — {count} files")

print("All extraction complete.")
