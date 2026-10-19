
# save as check_predictions.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.multimodal_dataset import MultiModalRSDataset
import yaml, glob, os

with open('configs/exp_haefnet.yml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from models.builder import build_haefnet
model = build_haefnet(config)
ckpt = torch.load('checkpoints/model_best.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

# Load one batch manually
s1_dir = 'data/s1asc_part01'
files  = glob.glob(os.path.join(s1_dir, '*.nc'))[:8]
ids    = [os.path.splitext(os.path.basename(f))[0] for f in files]

from utils.augmentations import get_traditional_val_augmentation
norm_config = config['model']['modality_norms']
modalities  = [k[4:] for k, v in config['modalities'].items() if v]

ds = MultiModalRSDataset(
    root_dir   = config['data']['root_dir'],
    file_list  = ids,
    modalities = modalities,
    transform  = get_traditional_val_augmentation(norm_config),
    stage      = 'val'
)

images, labels, _ = ds[0]
images = [img.unsqueeze(0) for img in images]

with torch.no_grad():
    outputs, _ = model(images)
    pred = torch.argmax(outputs[0], dim=1)

print(f"Label  — unique values : {labels.unique().tolist()}")
print(f"Label  — landslide px  : {(labels == 1).sum().item()}")
print(f"Pred   — unique values : {pred.unique().tolist()}")
print(f"Pred   — landslide px  : {(pred == 1).sum().item()}")