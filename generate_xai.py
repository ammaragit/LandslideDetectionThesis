"""
XAI — Modality Confidence Heatmap Generator
Loads M3 (proposed model) and generates per-modality confidence maps
for a set of test patches, showing which input source (SAR, optical, DEM)
the model relies on at each spatial location.

Usage:
    python generate_xai.py
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

from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations import get_traditional_val_augmentation
from models.builder import build_model

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIGURES_DIR = "figures/xai"
N_SAMPLES   = 8    # number of test patches to visualize
os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/exp_haefnet.yml")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    return parser.parse_args()


def load_config(config_path, model_key="m3"):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    global_params  = config.get('global_training', {})
    model_section  = config.get(model_key, {})
    merged_training = {**global_params, **model_section.get('training', {})}
    if 'loss' not in model_section.get('training', {}):
        merged_training['loss'] = global_params.get('loss', {})
    return {
        'data':       config['data'],
        'modalities': config['modalities'],
        'model':      model_section.get('model', {}),
        'training':   merged_training,
    }


def load_test_patches(config, n_samples):
    """Load n_samples patches from test split."""
    root_dir    = config["data"]["root_dir"]
    test_path   = os.path.join(root_dir, "split_test.txt")
    modalities  = [k[4:] for k, v in config["modalities"].items()
                   if v and k.startswith("use_")]
    norm_config = config["model"].get("modality_norms", {})

    with open(test_path) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    # Select patches with landslide pixels for better visualization
    np.random.seed(42)
    selected = np.random.choice(test_ids,
                                size=min(n_samples * 3, len(test_ids)),
                                replace=False)

    dataset = MultiModalRSDataset(
        root_dir    = root_dir,
        file_list   = list(selected),
        modalities  = modalities,
        transform   = get_traditional_val_augmentation(norm_config),
        stage       = "test",
    )
    return dataset, modalities


def get_confidence_maps(model, images):
    """
    Run M3 forward pass and extract modality confidence maps.
    Returns conf_maps (3, H, W) normalized per-modality confidence.
    """
    model.eval()
    with torch.no_grad():
        imgs_gpu = [img.unsqueeze(0).to(DEVICE) for img in images]
        outputs, aux = model(imgs_gpu)

        # Get prediction
        out  = outputs[0]
        out  = F.interpolate(out, size=(128, 128),
                             mode='bilinear', align_corners=False)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Get confidence maps from RO2
        conf_maps = aux.get("conf_maps", None)
        if conf_maps is not None:
            conf_maps = F.interpolate(conf_maps, size=(128, 128),
                                      mode='bilinear',
                                      align_corners=False)
            conf_maps = conf_maps.squeeze(0).cpu().numpy()  # (3, 128, 128)
        else:
            print("WARNING: conf_maps not found in aux. "
                  "Is this M3 with RO2 enabled?")
            conf_maps = np.ones((3, 128, 128)) / 3.0

    return pred, conf_maps


def plot_xai_grid(dataset, model, modalities, n_samples, save_path):
    """
    For each sample: show SAR, Optical, DEM, Label, Prediction,
    and 3 modality confidence heatmaps.
    """
    collected = []
    for idx in range(len(dataset)):
        images, label, meta = dataset[idx]
        label_np = label.numpy()
        if label_np.sum() > 100:      # only patches with landslides
            collected.append((images, label_np, meta))
        if len(collected) >= n_samples:
            break

    if len(collected) == 0:
        print("No patches with landslide pixels found in test set.")
        return

    n = len(collected)
    fig = plt.figure(figsize=(20, 3.5 * n))
    gs  = gridspec.GridSpec(n, 8, figure=fig,
                            hspace=0.4, wspace=0.3)

    modal_names  = ["SAR\n(change)", "Optical\n(RGB)", "DEM"]
    conf_colors  = ["Blues", "Greens", "Oranges"]
    modal_cmaps  = ["gray", "viridis", "terrain"]

    for row, (images, label_np, meta) in enumerate(collected):
        pred, conf_maps = get_confidence_maps(model, images)

        # ── Column 0-2: Input modalities ──────────────────────────────────
        for col, (img_tensor, name, cmap) in enumerate(
                zip(images, modal_names, modal_cmaps)):
            ax = fig.add_subplot(gs[row, col])
            img = img_tensor.numpy()
            # Show first channel for visualization
            ax.imshow(img[0], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(name, fontsize=8, fontfamily='Times New Roman')
            ax.axis('off')
            if row == 0:
                ax.set_title(name, fontsize=9, fontweight='bold',
                             fontfamily='Times New Roman')

        # ── Column 3: Ground truth label ──────────────────────────────────
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(label_np, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("Ground\nTruth", fontsize=8,
                     fontfamily='Times New Roman')
        ax.axis('off')

        # ── Column 4: Prediction ──────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(pred, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("M3\nPrediction", fontsize=8,
                     fontfamily='Times New Roman')
        ax.axis('off')

        # ── Columns 5-7: Modality confidence heatmaps (XAI) ──────────────
        conf_titles = ["SAR\nConfidence", "Optical\nConfidence",
                       "DEM\nConfidence"]
        for col_offset, (cname, ccmap) in enumerate(
                zip(conf_titles, conf_colors)):
            ax = fig.add_subplot(gs[row, 5 + col_offset])
            im = ax.imshow(conf_maps[col_offset],
                           cmap=ccmap, vmin=0, vmax=1)
            ax.set_title(cname, fontsize=8, fontfamily='Times New Roman')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row label
        pid = meta.get('file_id', f'patch_{row}')
        region = pid.split('_s1')[0]
        fig.text(0.01, 1 - (row + 0.5) / n,
                 region, va='center', ha='left',
                 fontsize=8, fontfamily='Times New Roman',
                 rotation=90)

    plt.suptitle(
        "RO2 — Modality Confidence Heatmaps (XAI)\n"
        "Brighter = model relies more on this modality at this location",
        fontsize=12, fontfamily='Times New Roman', y=1.01)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"XAI heatmaps saved to {save_path}")


def main():
    args   = parse_args()
    config = load_config(args.config, model_key="m3")

    print("Loading M3 model...")
    model = build_model(config).to(DEVICE)

    # Load best M3 checkpoint
    ckpt_path = "checkpoints/m3/model_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found. Train M3 first.")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded M3 checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load test patches
    print(f"Loading test patches...")
    dataset, modalities = load_test_patches(config, args.n_samples)
    print(f"Dataset: {len(dataset)} patches available")

    # Generate XAI heatmaps
    save_path = os.path.join(FIGURES_DIR, "modality_confidence_heatmaps.png")
    print(f"Generating {args.n_samples} XAI heatmap figures...")
    plot_xai_grid(dataset, model, modalities,
                  args.n_samples, save_path)

    # Also generate average confidence per modality across all test patches
    print("\nComputing average modality confidence across test set...")
    avg_conf  = np.zeros(3)
    n_patches = 0
    for idx in range(min(100, len(dataset))):
        images, label, _ = dataset[idx]
        _, conf_maps = get_confidence_maps(model, images)
        avg_conf  += conf_maps.mean(axis=(1, 2))
        n_patches += 1

    avg_conf /= n_patches
    modal_names = ["SAR (change)", "Optical (RGB)", "DEM"]

    print("\nAverage modality confidence scores (test set):")
    for name, score in zip(modal_names, avg_conf):
        bar = "█" * int(score * 40)
        print(f"  {name:<20}: {score:.4f}  {bar}")

    # Save bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#1F78B4", "#2CA02C", "#D62728"]
    bars   = ax.bar(modal_names, avg_conf * 100, color=colors,
                    alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, avg_conf):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{val*100:.1f}%", ha='center', va='bottom',
                fontsize=10, fontfamily='Times New Roman')
    ax.set_ylabel("Average Confidence Score (%)",
                  fontsize=11, fontfamily='Times New Roman')
    ax.set_title("RO2 — Average Modality Confidence (Test Set)\n"
                 "Higher = model relies more on this modality overall",
                 fontsize=11, fontfamily='Times New Roman')
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "avg_modality_confidence.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nAverage confidence chart saved to {bar_path}")
    print(f"\nAll XAI outputs saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
