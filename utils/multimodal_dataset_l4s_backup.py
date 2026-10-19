"""
Landslide4Sense Dataset
Loads preprocessed .npz files for Landslide4Sense.
Identical interface to MultiModalRSDataset so train.py needs no changes.

The dataset uses official train/val/test splits from the competition.
Note: ValidData and TestData have no ground truth labels in the
competition format, so only TrainData patches are used for training
and evaluation. We create an 80/20 internal train/test split from
TrainData (3,799 labeled patches).
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiModalRSDataset(Dataset):
    """
    Drop-in replacement for the Sen12Landslides MultiModalRSDataset.
    Loads Landslide4Sense preprocessed .npz files.
    Same __getitem__ interface: returns (images, label, meta).
    """

    def __init__(
        self,
        root_dir,
        file_list,
        modalities,
        transform=None,
        stage="train",
        target_size=None,
        require_label=True,
    ):
        self.root_dir    = root_dir
        self.modalities  = modalities
        self.transform   = transform
        self.stage       = stage
        self.target_size = target_size
        self.npz_dir     = os.path.join(root_dir, "preprocessed")

        # Load IDs from file list
        self.datalist = self._load_ids(file_list)

        # Keep only existing files
        self._verify()

        # Oversample landslide patches during training
        if stage == "train":
            self._oversample()

        print(f"[{stage}] Dataset ready: {len(self.datalist)} valid patches")

    def _load_ids(self, file_list):
        if isinstance(file_list, str):
            with open(file_list) as f:
                return [l.strip() for l in f if l.strip()]
        return [str(x).strip() for x in file_list if str(x).strip()]

    def _verify(self):
        valid   = []
        skipped = 0
        for pid in self.datalist:
            if os.path.exists(os.path.join(self.npz_dir, pid + ".npz")):
                valid.append(pid)
            else:
                skipped += 1
        if skipped:
            print(f"Warning: Skipped {skipped} patches (no .npz file)")
        self.datalist = valid

    def _oversample(self):
        """2x oversample patches with landslide pixels."""
        ls_ids = []
        bg_ids = []
        for pid in self.datalist:
            d = np.load(os.path.join(self.npz_dir, pid + ".npz"))
            if d["mask"].sum() > 50:
                ls_ids.append(pid)
            else:
                bg_ids.append(pid)
        self.datalist = bg_ids + ls_ids * 2
        import random
        random.seed(42)
        random.shuffle(self.datalist)
        print(f"  Landslide: {len(ls_ids)} × 2 | Background: {len(bg_ids)} "
              f"| Total: {len(self.datalist)}")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        pid  = self.datalist[idx]
        data = np.load(os.path.join(self.npz_dir, pid + ".npz"))

        sar  = torch.tensor(data["sar"],  dtype=torch.float32)  # (3,128,128)
        opt  = torch.tensor(data["opt"],  dtype=torch.float32)  # (3,128,128)
        dem  = torch.tensor(data["dem"],  dtype=torch.float32)  # (3,128,128)
        mask = torch.tensor(data["mask"], dtype=torch.long)      # (128,128)

        images = []
        for mod in self.modalities:
            if mod == "insar_vel": images.append(sar)
            elif mod == "rgb":     images.append(opt)
            elif mod == "dem":     images.append(dem)

        if self.target_size is not None:
            h_t, w_t = self.target_size
            images = [F.interpolate(img.unsqueeze(0), size=(h_t, w_t),
                                    mode="bilinear", align_corners=False
                                    ).squeeze(0) for img in images]
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                 size=(h_t, w_t), mode="nearest"
                                 ).squeeze(0).squeeze(0).long()

        if self.transform:
            sample = {m: img for m, img in zip(self.modalities, images)}
            sample["label"] = mask
            sample = self.transform(sample)
            images = [sample[m] for m in self.modalities]
            mask   = sample.get("label", mask)

        meta = {"file_id": pid}
        return images, mask, meta

    def set_stage(self, stage):
        self.stage = stage
