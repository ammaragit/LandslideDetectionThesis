import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MultiModalRSDataset(Dataset):

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

        self.npz_dir = os.path.join(root_dir, "data", "preprocessed")

        # Load IDs
        self.datalist = self._load_ids(file_list)

        # Keep only IDs with existing .npz files
        self._verify_dataset()
        if stage == 'train':
            landslide_ids  = []
            background_ids = []
            for pid in self.datalist:
                d = np.load(os.path.join(self.npz_dir, pid + '.npz'))
                if d['mask'].sum() > 50:
                    landslide_ids.append(pid)
                else:
                    background_ids.append(pid)
            # Oversample landslide patches 2x
            self.datalist = background_ids + landslide_ids * 2
            import random
            random.seed(42)
            random.shuffle(self.datalist)
            print(f"  Landslide patches: {len(landslide_ids)} × 2 = {len(landslide_ids)*2}")
            print(f"  Background patches: {len(background_ids)}")
            print(f"  Total after oversampling: {len(self.datalist)}")
        print(f"[{stage}] Dataset ready: {len(self.datalist)} valid patches")

    def _load_ids(self, file_list):
        if isinstance(file_list, str):
            with open(file_list, 'r') as f:
                return [l.strip() for l in f if l.strip()]
        return [str(x).strip() for x in file_list if str(x).strip()]

    def _verify_dataset(self):
        valid   = []
        skipped = 0
        for pid in self.datalist:
            if os.path.exists(os.path.join(self.npz_dir, pid + '.npz')):
                valid.append(pid)
            else:
                skipped += 1
        if skipped:
            print(f"Warning: Skipped {skipped} patches (no .npz file)")
        self.datalist = valid

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        pid  = self.datalist[idx]
        data = np.load(os.path.join(self.npz_dir, pid + '.npz'))

        sar  = torch.tensor(data['sar'],  dtype=torch.float32)  # (3,128,128)
        opt  = torch.tensor(data['opt'],  dtype=torch.float32)  # (3,128,128)
        dem  = torch.tensor(data['dem'],  dtype=torch.float32)  # (3,128,128)
        mask = torch.tensor(data['mask'], dtype=torch.long)      # (128,128)

        images = []
        for mod in self.modalities:
            if mod == 'insar_vel': images.append(sar)
            elif mod == 'rgb':     images.append(opt)
            elif mod == 'dem':     images.append(dem)

        if self.transform:
            sample = {m: img for m, img in zip(self.modalities, images)}
            sample['label'] = mask
            sample = self.transform(sample)
            images = [sample[m] for m in self.modalities]
            mask   = sample.get('label', mask)

        meta = {'file_id': pid}
        return images, mask, meta

    def set_stage(self, stage):
        self.stage = stage