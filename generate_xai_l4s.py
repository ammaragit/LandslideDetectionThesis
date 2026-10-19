from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
"""
XAI — Modality Confidence Heatmaps for Landslide4Sense (Spectral Indices)
Loads the L4S M3 checkpoint and generates per-modality confidence maps
showing which input source (NDVI/NDWI/NBR indices, optical RGB, DEM)
the model relies on at each spatial location.

Key differences from generate_xai.py (Sen12Landslides):
  - Config:    configs/l4s_config_indices.yml
  - Checkpoint: checkpoints/l4s_indices/m3/model_best.pth
  - Split file: Landslide4Sense/l4s_split_test.txt
  - npz_dir:   Landslide4Sense/preprocessed_indices/ (no subdirectory)
  - SAR slot:  NDVI / NDWI / NBR (not SAR amplitude change)

Usage:
    python generate_xai_l4s.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from argparse import ArgumentParser
from torch.utils.data import Dataset

from utils.augmentations import get_traditional_val_augmentation
from models.builder import build_model

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIGURES_DIR  = ROOT_DIR / "figures" / "xai_l4s"
CKPT_PATH    = ROOT_DIR / "checkpoints" / "l4s_indices" / "m3" / "model_best.pth"
SPLIT_FILE   = ROOT_DIR / "Landslide4Sense" / "l4s_split_test.txt"
NPZ_DIR      = ROOT_DIR / "Landslide4Sense" / "preprocessed_indices"
N_SAMPLES    = 6
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Inline dataset — bypasses multimodal_dataset.py path confusion ─────────────
class L4SDataset(Dataset):
    """Minimal dataset that loads directly from preprocessed_indices/ npz files."""

    def __init__(self, npz_dir, file_ids, transform=None):
        self.npz_dir  = npz_dir
        self.transform = transform
        # Keep only patches whose npz file actually exists
        self.ids = [
            pid for pid in file_ids
            if os.path.exists(os.path.join(npz_dir, pid + ".npz"))
        ]
        print(f"[L4S XAI] Dataset: {len(self.ids)} valid patches")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid  = self.ids[idx]
        data = np.load(os.path.join(self.npz_dir, pid + ".npz"))
        sar  = torch.tensor(data["sar"],  dtype=torch.float32)  # (3,128,128) NDVI/NDWI/NBR
        opt  = torch.tensor(data["opt"],  dtype=torch.float32)  # (3,128,128) RGB
        dem  = torch.tensor(data["dem"],  dtype=torch.float32)  # (3,128,128) slope+DEM
        mask = torch.tensor(data["mask"], dtype=torch.long)
        images = [sar, opt, dem]
        meta   = {"file_id": pid}
        return images, mask, meta


def load_config(config_path="configs/l4s_config_indices.yml", model_key="m3"):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    global_params   = config.get('global_training', {})
    model_section   = config.get(model_key, {})
    merged_training = {**global_params, **model_section.get('training', {})}
    if 'loss' not in model_section.get('training', {}):
        merged_training['loss'] = global_params.get('loss', {})
    return {
        'data':       config.get('data', {}),
        'modalities': config.get('modalities', {}),
        'model':      model_section.get('model', {}),
        'training':   merged_training,
    }


def get_confidence_maps(model, images):
    """Forward pass through M3, return prediction and conf_maps (3,H,W)."""
    model.eval()
    with torch.no_grad():
        imgs_gpu = [img.unsqueeze(0).to(DEVICE) for img in images]
        outputs, aux = model(imgs_gpu)
        out  = outputs[0]
        out  = F.interpolate(out, size=(128, 128),
                             mode='bilinear', align_corners=False)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        conf_maps = aux.get("conf_maps", None)
        if conf_maps is not None:
            conf_maps = F.interpolate(conf_maps, size=(128, 128),
                                      mode='bilinear', align_corners=False)
            conf_maps = conf_maps.squeeze(0).cpu().numpy()
        else:
            print("WARNING: conf_maps not in aux — using uniform weights")
            conf_maps = np.ones((3, 128, 128)) / 3.0
    return pred, conf_maps


def plot_xai_grid(dataset, model, n_samples, save_path):
    """Grid: for each patch show NDVI/Optical/DEM, GT, Prediction, 3 conf maps."""
    collected = []
    for idx in range(len(dataset)):
        images, label, meta = dataset[idx]
        if label.numpy().sum() > 100:
            collected.append((images, label.numpy(), meta))
        if len(collected) >= n_samples:
            break

    if not collected:
        print("No patches with landslide pixels found.")
        return

    n   = len(collected)
    fig = plt.figure(figsize=(22, 3.5 * n))
    gs  = gridspec.GridSpec(n, 8, figure=fig, hspace=0.4, wspace=0.3)

    # L4S-specific labels — SAR slot is spectral indices, not SAR amplitude
    modal_names  = ["Spectral\nIndices\n(NDVI ch)", "Optical\n(RGB)", "DEM"]
    modal_cmaps  = ["RdYlGn",   "viridis",    "terrain"]
    conf_titles  = ["Index\nConfidence", "Optical\nConfidence", "DEM\nConfidence"]
    conf_colors  = ["Blues",    "Greens",     "Oranges"]

    for row, (images, label_np, meta) in enumerate(collected):
        pred, conf_maps = get_confidence_maps(model, images)

        # Columns 0-2: input modalities
        for col, (img_t, name, cmap) in enumerate(
                zip(images, modal_names, modal_cmaps)):
            ax = fig.add_subplot(gs[row, col])
            img = img_t.numpy()
            ax.imshow(img[0], cmap=cmap, vmin=0, vmax=1)
            title = name if row > 0 else name
            ax.set_title(title, fontsize=7.5, fontfamily='Times New Roman',
                         fontweight='bold' if row == 0 else 'normal')
            ax.axis('off')

        # Column 3: ground truth
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(label_np, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("Ground\nTruth", fontsize=7.5,
                     fontfamily='Times New Roman')
        ax.axis('off')

        # Column 4: prediction
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(pred, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("M3\nPrediction", fontsize=7.5,
                     fontfamily='Times New Roman')
        ax.axis('off')

        # Columns 5-7: confidence heatmaps
        for col_offset, (cname, ccmap) in enumerate(
                zip(conf_titles, conf_colors)):
            ax = fig.add_subplot(gs[row, 5 + col_offset])
            im = ax.imshow(conf_maps[col_offset],
                           cmap=ccmap, vmin=0, vmax=1)
            ax.set_title(cname, fontsize=7.5,
                         fontfamily='Times New Roman')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row label — L4S patch IDs have format l4s_train_image_X
        pid    = meta.get('file_id', f'patch_{row}')
        label_ = pid.replace("l4s_train_", "").replace("image_", "img")
        fig.text(0.005, 1 - (row + 0.5) / n,
                 label_, va='center', ha='left',
                 fontsize=7, fontfamily='Times New Roman', rotation=90)

    plt.suptitle(
        "RO2 — Modality Confidence Heatmaps (XAI) — Landslide4Sense\n"
        "SAR slot = spectral indices (NDVI/NDWI/NBR)  |  "
        "Brighter = model relies more on this modality",
        fontsize=11, fontfamily='Times New Roman', y=1.01)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"XAI heatmaps saved: {save_path}")


def plot_avg_confidence(avg_conf, save_path):
    """Bar chart of average per-modality confidence across test patches."""
    modal_names = ["Spectral Indices\n(NDVI/NDWI/NBR)", "Optical (RGB)", "DEM"]
    colors = ["#1F78B4", "#2CA02C", "#D62728"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(modal_names, avg_conf * 100, color=colors,
                  alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, avg_conf):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{val*100:.1f}%", ha='center', va='bottom',
                fontsize=11, fontfamily='Times New Roman')

    ax.set_ylabel("Average Confidence Score (%)",
                  fontsize=11, fontfamily='Times New Roman')
    ax.set_title(
        "RO2 — Average Modality Confidence (L4S Test Set)\n"
        "SAR slot occupied by spectral indices (NDVI/NDWI/NBR)",
        fontsize=11, fontfamily='Times New Roman')
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Avg confidence chart saved: {save_path}")


def main():
    print("=" * 60)
    print("XAI — Landslide4Sense Spectral Indices")
    print("=" * 60)

    # Load config and build model
    config = load_config("configs/l4s_config_indices.yml", model_key="m3")
    print("Building M3 model...")
    model  = build_model(config).to(DEVICE)

    # Load checkpoint
    if not os.path.exists(CKPT_PATH):
        print(f"ERROR: {CKPT_PATH} not found.")
        return
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded M3 from epoch {ckpt.get('epoch', '?')}")

    # Load split
    with open(SPLIT_FILE) as f:
        test_ids = [l.strip() for l in f if l.strip()]
    print(f"Test split: {len(test_ids)} patches")

    # Build dataset
    dataset = L4SDataset(NPZ_DIR, test_ids)

    # Generate heatmap grid
    grid_path = os.path.join(FIGURES_DIR, "l4s_modality_confidence_heatmaps.png")
    print(f"\nGenerating {N_SAMPLES} XAI heatmap patches...")
    plot_xai_grid(dataset, model, N_SAMPLES, grid_path)

    # Average confidence across up to 100 test patches
    print("\nComputing average confidence across test set...")
    avg_conf  = np.zeros(3)
    n_counted = 0
    for idx in range(min(100, len(dataset))):
        images, _, _ = dataset[idx]
        _, conf_maps = get_confidence_maps(model, images)
        avg_conf    += conf_maps.mean(axis=(1, 2))
        n_counted   += 1

    avg_conf /= n_counted
    modal_names = ["Spectral Indices (NDVI/NDWI/NBR)", "Optical (RGB)", "DEM"]
    print("\nAverage modality confidence scores:")
    for name, score in zip(modal_names, avg_conf):
        bar = "█" * int(score * 40)
        print(f"  {name:<38}: {score*100:.2f}%  {bar}")

    bar_path = os.path.join(FIGURES_DIR, "l4s_avg_modality_confidence.png")
    plot_avg_confidence(avg_conf, bar_path)

    print(f"\nAll XAI outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
