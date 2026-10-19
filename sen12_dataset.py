import os
import glob
import json
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_splits(s1_dir, test_size=0.15, val_size=0.15, random_state=42):
    """
    Automatically scans the s1asc folder and returns
    train / val / test filename lists. No manual input needed.
    """
    all_files = glob.glob(os.path.join(s1_dir, '*.nc'))
    all_files = [os.path.basename(f) for f in sorted(all_files)]

    print(f"Total .nc files found: {len(all_files)}")

    train_files, temp_files = train_test_split(
        all_files, test_size=(test_size + val_size), random_state=random_state
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=random_state
    )

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


class Sen12LandslideDataset(Dataset):
    def __init__(self, s1_dir, s2_dir, split_list, normalize=True):
        """
        Parameters
        ----------
        s1_dir      : path to data_harmonized/s1asc/
        s2_dir      : path to data_harmonized/s2/
        split_list  : list of s1 .nc filenames (from get_splits)
        normalize   : whether to normalize inputs to [0, 1]
        """
        self.s1_dir    = s1_dir
        self.s2_dir    = s2_dir
        self.normalize = normalize

        # Keep only files that have a matching S2 file
        self.files = []
        skipped = 0
        for fname in split_list:
            s2_name = fname.replace('s1asc', 's2')
            s2_path = os.path.join(s2_dir, s2_name)
            if os.path.exists(s2_path):
                self.files.append(fname)
            else:
                skipped += 1

        if skipped > 0:
            print(f"Warning: {skipped} files skipped (no matching S2 file found)")
        print(f"Dataset ready with {len(self.files)} valid patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s1_name = self.files[idx]
        s2_name = s1_name.replace('s1asc', 's2')

        s1_path = os.path.join(self.s1_dir, s1_name)
        s2_path = os.path.join(self.s2_dir, s2_name)

        # ── Open NetCDF files ────────────────────────────────────────────────
        s1 = xr.open_dataset(s1_path)
        s2 = xr.open_dataset(s2_path)

        # ── Get pre/post event time indices from metadata ────────────────────
        pre_post = s1.attrs.get('pre_post_dates', None)

        if pre_post is not None:
            # pre_post_dates is stored as a dict: {'pre': 7, 'post': 8}
            if isinstance(pre_post, str):
                pre_post = json.loads(pre_post.replace("'", '"'))
            pre_idx  = int(pre_post['pre'])
            post_idx = int(pre_post['post'])
        else:
            # Fallback: use second-to-last and last time steps
            n_times  = s1.dims['time']
            pre_idx  = n_times - 2
            post_idx = n_times - 1

        # ── SAR: 4 channels (VV_pre, VH_pre, VV_post, VH_post) ─────────────
        vv_pre  = s1['VV'].values[pre_idx].astype(np.float32)   # (128, 128)
        vh_pre  = s1['VH'].values[pre_idx].astype(np.float32)
        vv_post = s1['VV'].values[post_idx].astype(np.float32)
        vh_post = s1['VH'].values[post_idx].astype(np.float32)

        sar = np.stack([vv_pre, vh_pre, vv_post, vh_post], axis=0)  # (4, 128, 128)

        # Replace NaN with 0
        sar = np.nan_to_num(sar, nan=0.0)

        # ── Optical: 4 channels (Red, Green, Blue, NIR) at post event ────────
        opt = np.stack([
            s2['B04'].values[post_idx],   # Red
            s2['B03'].values[post_idx],   # Green
            s2['B02'].values[post_idx],   # Blue
            s2['B08'].values[post_idx],   # NIR
        ], axis=0).astype(np.float32)     # (4, 128, 128)

        opt = np.nan_to_num(opt, nan=0.0)

        # ── DEM: 1 channel ───────────────────────────────────────────────────
        dem = s2['DEM'].values[post_idx].astype(np.float32)  # (128, 128)
        dem = np.nan_to_num(dem, nan=0.0)
        dem = dem[np.newaxis]                                 # (1, 128, 128)

        # ── Label / Mask ─────────────────────────────────────────────────────
        mask = s2['MASK'].values[post_idx].astype(np.int64)  # (128, 128) binary
        mask = np.nan_to_num(mask, nan=0).astype(np.int64)

        # ── Normalize to [0, 1] ──────────────────────────────────────────────
        if self.normalize:
            # SAR: harmonized dB range is [-50, 10]
            sar = np.clip(sar, -50.0, 10.0)
            sar = (sar - (-50.0)) / (10.0 - (-50.0))

            # Optical: Sentinel-2 reflectance range [0, 10000]
            opt = np.clip(opt, 0.0, 10000.0)
            opt = opt / 10000.0

            # DEM: elevation range [0, 8800] m
            dem = np.clip(dem, 0.0, 8800.0)
            dem = dem / 8800.0

        # ── Convert to tensors ───────────────────────────────────────────────
        sar  = torch.tensor(sar,  dtype=torch.float32)
        opt  = torch.tensor(opt,  dtype=torch.float32)
        dem  = torch.tensor(dem,  dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        # Close datasets to free memory
        s1.close()
        s2.close()

        return {
            'sar':     sar,    # (4, 128, 128)
            'optical': opt,    # (4, 128, 128)
            'dem':     dem,    # (1, 128, 128)
            'label':   mask    # (128, 128)
        }