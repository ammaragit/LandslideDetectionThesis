
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import ConfusionMatrixDisplay

# ── Global constants ──────────────────────────────────────────────────────────
MODEL_KEYS   = ["cm1", "cm2"]
MODEL_LABELS = {
    "cm1": "CM1 — Concat U-Net",
    "cm2": "CM2 — Attention U-Net",
    "m1":  "M1 — HAEFNet [1]",
    "m2":  "M2 — HAEFNet + RO1",
    "m3":  "M3 — Proposed (RO1+RO2)",
}
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# HELPERS

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/exp_haefnet.yml")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_ckpt_dir(config, model_key):
    """
    Uses the --model key directly as folder name (cm1, cm2, m1, m2, m3)
    since train.py saves checkpoints to checkpoints/<model_key>/
    """
    base = config.get(model_key, {}).get(
        'training', {}).get('ckpt_dir', 'checkpoints/')
    # train.py uses the --model argument as folder name directly
    return os.path.join(base, model_key)


def load_results(ckpt_dir):
    """Load test_results.npy and history.npy from checkpoint dir."""
    results_path = os.path.join(ckpt_dir, "test_results.npy")
    history_path = os.path.join(ckpt_dir, "history.npy")

    results = None
    history = None

    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
    else:
        print(f"  WARNING: {results_path} not found — skipping")

    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True).item()

    return results, history


# 1. COMPARISON TABLE

def print_comparison_table(all_results):
    print("\n" + "="*90)
    print(f"{'Model':<35} {'IoU':>7} {'F1':>7} {'Precision':>10} {'Recall':>8} {'ECE':>7}")
    print("="*90)
    for key in MODEL_KEYS:
        label = MODEL_LABELS[key]
        r = all_results.get(key)
        if r is None:
            print(f"{label:<35} {'N/A':>7} {'N/A':>7} {'N/A':>10} {'N/A':>8} {'N/A':>7}")
            continue
        m = r["metrics"]
        print(f"{label:<35} "
              f"{m['IoU']:>6.2f}% "
              f"{m['F1']:>6.2f}% "
              f"{m['Precision']:>9.2f}% "
              f"{m['Recall']:>7.2f}% "
              f"{m['ECE']:>6.2f}%")
    print("="*90 + "\n")

    # Save as text file
    with open(os.path.join(FIGURES_DIR, "comparison_table.txt"), "w") as f:
        f.write(f"{'Model':<35} {'IoU':>7} {'F1':>7} {'Precision':>10} "
                f"{'Recall':>8} {'ECE':>7}\n")
        f.write("="*90 + "\n")
        for key in MODEL_KEYS:
            label = MODEL_LABELS[key]
            r = all_results.get(key)
            if r is None:
                f.write(f"{label:<35} {'N/A':>7}\n")
                continue
            m = r["metrics"]
            f.write(f"{label:<35} "
                    f"{m['IoU']:>6.2f}% "
                    f"{m['F1']:>6.2f}% "
                    f"{m['Precision']:>9.2f}% "
                    f"{m['Recall']:>7.2f}% "
                    f"{m['ECE']:>6.2f}%\n")
    print(f"Comparison table saved to {FIGURES_DIR}/comparison_table.txt")


# 2. METRICS BAR CHART

def plot_metrics_bar(all_results):
    metrics_to_plot = ["IoU", "F1", "Precision", "Recall"]
    n_models  = len(MODEL_KEYS)
    n_metrics = len(metrics_to_plot)
    x         = np.arange(n_models)
    width     = 0.2
    colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        values = []
        for key in MODEL_KEYS:
            r = all_results.get(key)
            values.append(r["metrics"][metric] if r else 0.0)
        bars = ax.bar(x + i * width, values, width, label=metric, color=color,
                      alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{val:.1f}", ha='center', va='bottom',
                        fontsize=7, fontfamily='Times New Roman')

    ax.set_xlabel("Model", fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel("Score (%)", fontsize=12, fontfamily='Times New Roman')
    ax.set_title("Performance Comparison Across All Models",
                 fontsize=13, fontfamily='Times New Roman')
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels([MODEL_LABELS[k] for k in MODEL_KEYS],
                       rotation=15, ha='right', fontsize=9,
                       fontfamily='Times New Roman')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics bar chart saved to {path}")


# 3. CONFUSION MATRICES

def plot_confusion_matrices(all_results):
    available = [(k, all_results[k]) for k in MODEL_KEYS
                 if all_results.get(k) and "conf_mat" in all_results[k]]
    if not available:
        print("No confusion matrices available.")
        return

    n  = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (key, r) in zip(axes, available):
        cm = np.array(r["conf_mat"])
        # Normalize
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-7)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm,
            display_labels=["Background", "Landslide"])
        disp.plot(ax=ax, colorbar=False, cmap='Blues',
                  values_format='.2f')
        ax.set_title(MODEL_LABELS[key], fontsize=10,
                     fontfamily='Times New Roman')
        ax.set_xlabel("Predicted", fontsize=9, fontfamily='Times New Roman')
        ax.set_ylabel("True", fontsize=9, fontfamily='Times New Roman')

    plt.suptitle("Normalized Confusion Matrices",
                 fontsize=13, fontfamily='Times New Roman', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {path}")


# 4. TRAINING LOSS CURVES

def plot_loss_curves(all_histories):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for (key, history), color in zip(all_histories.items(), colors):
        if history is None or "train_loss" not in history:
            continue
        train_loss = history["train_loss"]
        epochs     = list(range(1, len(train_loss) + 1))
        ax.plot(epochs, train_loss, label=MODEL_LABELS[key],
                color=color, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel("Training Loss", fontsize=12, fontfamily='Times New Roman')
    ax.set_title("Training Loss Curves — All Models",
                 fontsize=13, fontfamily='Times New Roman')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "loss_curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {path}")


def plot_iou_curves(all_histories, save_interval=5):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for (key, history), color in zip(all_histories.items(), colors):
        if history is None or "val_iou" not in history:
            continue
        val_iou = history["val_iou"]
        # val is recorded every save_interval epochs
        epochs  = [save_interval * (i + 1) for i in range(len(val_iou))]
        ax.plot(epochs, val_iou, label=MODEL_LABELS[key],
                color=color, linewidth=1.8, marker='o',
                markersize=4, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=12, fontfamily='Times New Roman')
    ax.set_ylabel("Validation IoU (%)", fontsize=12,
                  fontfamily='Times New Roman')
    ax.set_title("Validation IoU Curves — All Models",
                 fontsize=13, fontfamily='Times New Roman')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "iou_curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"IoU curves saved to {path}")


# 6. ECE COMPARISON BAR

def plot_ece_bar(all_results):
    keys   = [k for k in MODEL_KEYS if all_results.get(k)]
    labels = [MODEL_LABELS[k] for k in keys]
    ece    = [all_results[k]["metrics"]["ECE"] for k in keys]
    colors = ["#C44E52" if k == "m3" else "#4C72B0" for k in keys]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, ece, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ece):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}%", ha='center', va='bottom',
                fontsize=9, fontfamily='Times New Roman')

    ax.set_ylabel("ECE — Expected Calibration Error (%)",
                  fontsize=11, fontfamily='Times New Roman')
    ax.set_title("Prediction Reliability (lower ECE = better calibrated)",
                 fontsize=12, fontfamily='Times New Roman')
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9,
                       fontfamily='Times New Roman')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ece_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ECE comparison saved to {path}")


# 7. ABLATION TABLE

def print_ablation_table(all_results):
    print("\n" + "="*75)
    print("ABLATION STUDY")
    print("="*75)
    print(f"{'Comparison':<35} {'Delta IoU':>10} {'Delta F1':>10} {'Delta ECE':>10}")
    print("-"*75)

    pairs = [
        ("CM1 vs CM2", "cm1", "cm2", "Value of attention in naive fusion"),
        ("CM2 vs M1",  "cm2", "m1",  "Value of SCCA cross-modal attention"),
        ("M1  vs M2",  "m1",  "m2",  "Contribution of RO1 temporal encoding"),
        ("M2  vs M3",  "m2",  "m3",  "Contribution of RO2 uncertainty module"),
    ]

    with open(os.path.join(FIGURES_DIR, "ablation_table.txt"), "w") as f:
        f.write("ABLATION STUDY\n" + "="*75 + "\n")
        for label, k1, k2, desc in pairs:
            r1 = all_results.get(k1)
            r2 = all_results.get(k2)
            if r1 is None or r2 is None:
                line = f"{label:<35} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            else:
                d_iou = r2["metrics"]["IoU"] - r1["metrics"]["IoU"]
                d_f1  = r2["metrics"]["F1"]  - r1["metrics"]["F1"]
                d_ece = r2["metrics"]["ECE"]  - r1["metrics"]["ECE"]
                sign  = lambda v: f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"
                line  = (f"{label:<35} {sign(d_iou):>10} "
                         f"{sign(d_f1):>10} {sign(d_ece):>10}")
            print(f"{line}   # {desc}")
            f.write(f"{line}   # {desc}\n")

    print("="*75)
    print(f"Ablation table saved to {FIGURES_DIR}/ablation_table.txt\n")


# MAIN
def main():
    args   = parse_args()
    config = load_config(args.config)

    print("\n" + "="*60)
    print("EVALUATE ALL — loading results from checkpoints")
    print("="*60)

    all_results   = {}
    all_histories = {}

    for key in MODEL_KEYS:
        ckpt_dir = get_ckpt_dir(config, key)
        print(f"\nLoading {MODEL_LABELS[key]} from {ckpt_dir}...")
        results, history = load_results(ckpt_dir)
        all_results[key]   = results
        all_histories[key] = history
        if results:
            m = results["metrics"]
            print(f"  IoU={m['IoU']:.2f}%  F1={m['F1']:.2f}%  "
                  f"Prec={m['Precision']:.2f}%  Rec={m['Recall']:.2f}%  "
                  f"ECE={m['ECE']:.2f}%")

    print("\n" + "="*60)
    print("GENERATING FIGURES AND TABLES")
    print("="*60)

    save_interval = config.get("global_training", {}).get("save_interval", 5)

    print_comparison_table(all_results)
    print_ablation_table(all_results)
    plot_metrics_bar(all_results)
    plot_confusion_matrices(all_results)
    plot_loss_curves(all_histories)
    plot_iou_curves(all_histories, save_interval=save_interval)
    plot_ece_bar(all_results)

    print("\n" + "="*60)
    print(f"All outputs saved to: {os.path.abspath(FIGURES_DIR)}/")
    print("  metrics_comparison.png")
    print("  confusion_matrices.png")
    print("  loss_curves.png")
    print("  iou_curves.png")
    print("  ece_comparison.png")
    print("  comparison_table.txt")
    print("  ablation_table.txt")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()