"""
Evaluate all 5 models on Landslide4Sense spectral indices experiment.
Standalone script — no config parsing needed, paths are explicit.

Usage:
    python evaluate_l4s_indices.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Explicit checkpoint paths ──────────────────────────────────────────────────
CKPT_BASE = "checkpoints/l4s_indices"
MODELS = {
    "cm1": ("CM1 — Concat U-Net",          os.path.join(CKPT_BASE, "cm1")),
    "cm2": ("CM2 — Attention U-Net",        os.path.join(CKPT_BASE, "cm2")),
    "m1":  ("M1 — HAEFNet",                 os.path.join(CKPT_BASE, "m1")),
    "m2":  ("M2 — HAEFNet + RO1",           os.path.join(CKPT_BASE, "m2")),
    "m3":  ("M3 — Proposed (RO1+RO2)",      os.path.join(CKPT_BASE, "m3")),
}
FIGURES_DIR = "figures/l4s_indices"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Published L4S baselines (from Ghorbanzadeh et al. 2022)
PUBLISHED = {
    "U-Net":      48.91,   # raw bands baseline (this study)
    "ResU-Net":   54.0,    # published
    "Swin-UNet":  58.0,    # published (approx)
}


def load_results(ckpt_dir):
    """Load test_results.npy from checkpoint directory."""
    path = os.path.join(ckpt_dir, "test_results.npy")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    r = np.load(path, allow_pickle=True).item()
    return r


def load_history(ckpt_dir):
    """Load history.npy (loss/iou curves)."""
    path = os.path.join(ckpt_dir, "history.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def print_table(results):
    W = 100
    print("\n" + "=" * W)
    print("L4S SPECTRAL INDICES — COMPLETE RESULTS TABLE")
    print("=" * W)
    print(f"{'Model':<40} {'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'ECE':>7}")
    print("-" * W)

    for key, (label, _) in MODELS.items():
        r = results.get(key)
        if r is None:
            print(f"{label:<40} {'N/A':>7}")
            continue
        m = r if not isinstance(r, dict) else r
        print(
            f"{label:<40} "
            f"{m['iou']:>6.2f}% "
            f"{m['f1']:>6.2f}% "
            f"{m['precision']:>6.2f}% "
            f"{m['recall']:>6.2f}% "
            f"{m['ece']:>6.2f}%"
        )

    print("=" * W)

    # Ablation deltas
    keys = list(MODELS.keys())
    print("\nABLATION — Spectral Indices Run")
    print("-" * 60)
    pairs = [
        ("cm1", "cm2", "Attention over naive concatenation"),
        ("cm2", "m1",  "SCCA cross-modal attention"),
        ("m1",  "m2",  "RO1 (no temporal SAR — expected ~0)"),
        ("m2",  "m3",  "RO2 uncertainty module"),
    ]
    for k1, k2, label in pairs:
        r1, r2 = results.get(k1), results.get(k2)
        if r1 and r2:
            di  = r2['iou']       - r1['iou']
            df  = r2['f1']        - r1['f1']
            de  = r2['ece']       - r1['ece']
            sign_i = "+" if di >= 0 else ""
            sign_f = "+" if df >= 0 else ""
            sign_e = "+" if de >= 0 else ""
            print(f"  {k1.upper()} → {k2.upper()}  "
                  f"ΔIoU={sign_i}{di:.2f}%  "
                  f"ΔF1={sign_f}{df:.2f}%  "
                  f"ΔECE={sign_e}{de:.2f}%  "
                  f"# {label}")

    print("\nRAW BANDS vs SPECTRAL INDICES (key comparison)")
    print("-" * 60)
    raw_iou = {"cm1": 62.65, "cm2": 61.84, "m1": 48.91, "m2": 48.91, "m3": None}
    raw_f1  = {"cm1": 44.08, "cm2": 40.69, "m1":  0.00, "m2":  0.00, "m3": None}
    for key, (label, _) in MODELS.items():
        r = results.get(key)
        if r and raw_iou[key] is not None:
            di = r['iou'] - raw_iou[key]
            df = r['f1']  - raw_f1[key]
            conv = "converged" if raw_f1[key] == 0 else ""
            print(f"  {label:<35} "
                  f"ΔIoU=+{di:.2f}%  ΔF1=+{df:.2f}%  {conv}")


def plot_bar_comparison(results):
    """Bar chart comparing all 5 models across IoU, F1, ECE."""
    labels = [MODELS[k][0].replace("— ", "\n") for k in MODELS]
    iou_vals = [results[k]['iou']       if results.get(k) else 0 for k in MODELS]
    f1_vals  = [results[k]['f1']        if results.get(k) else 0 for k in MODELS]
    ece_vals = [results[k]['ece']       if results.get(k) else 0 for k in MODELS]

    x = np.arange(len(labels))
    w = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, ylabel, title in zip(
        axes,
        [iou_vals, f1_vals, ece_vals],
        ["IoU (%)", "F1-Score (%)", "ECE (%) — lower is better"],
        ["Intersection-over-Union", "F1-Score", "Expected Calibration Error"]
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}%", ha='center', va='bottom',
                        fontsize=8.5, fontfamily='Times New Roman')
        ax.set_ylabel(ylabel, fontsize=11, fontfamily='Times New Roman')
        ax.set_title(title, fontsize=12, fontfamily='Times New Roman')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, fontfamily='Times New Roman')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(
        "Landslide4Sense — Spectral Indices Experiment\n"
        "SAR slot: NDVI / NDWI / NBR",
        fontsize=13, fontfamily='Times New Roman')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {path}")


def plot_raw_vs_indices(results):
    """Side-by-side IoU comparison: raw bands vs spectral indices."""
    raw_iou = [62.65, 61.84, 48.91, 48.91, None]   # CM1,CM2,M1,M2,M3
    idx_iou = [results[k]['iou'] if results.get(k) else 0
               for k in MODELS]
    labels  = ["CM1", "CM2", "M1", "M2", "M3"]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, [v or 0 for v in raw_iou], w,
                label="Raw bands (B8/B11/B12)", color="#E07B54",
                alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, idx_iou, w,
                label="Spectral indices (NDVI/NDWI/NBR)", color="#4C72B0",
                alpha=0.85, edgecolor='white')

    # Annotate collapsed models
    for i, v in enumerate([62.65, 61.84, 48.91, 48.91, None]):
        if v is not None:
            note = " (collapsed)" if v < 50 else ""
            ax.text(x[i] - w/2, v + 0.3, f"{v:.1f}%{note}",
                    ha='center', va='bottom',
                    fontsize=8, fontfamily='Times New Roman',
                    color="#A32D2D" if v < 50 else "black")
    for i, v in enumerate(idx_iou):
        ax.text(x[i] + w/2, v + 0.3, f"{v:.1f}%",
                ha='center', va='bottom',
                fontsize=8, fontfamily='Times New Roman', color="#0C447C")

    # Published baseline reference line
    ax.axhline(58.0, color='gray', linestyle='--', linewidth=1,
               label="Best published L4S baseline (Swin-UNet ~58%)")

    ax.set_ylabel("IoU (%)", fontsize=11, fontfamily='Times New Roman')
    ax.set_title(
        "Raw Bands vs Spectral Indices — IoU Comparison\n"
        "Landslide4Sense Dataset",
        fontsize=12, fontfamily='Times New Roman')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontfamily='Times New Roman')
    ax.set_ylim(0, 90)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "raw_vs_indices.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {path}")


def main():
    print("\n" + "=" * 60)
    print("EVALUATE L4S — Spectral Indices Experiment")
    print("=" * 60)

    results = {}
    for key, (label, ckpt_dir) in MODELS.items():
        r = load_results(ckpt_dir)
        if r:
            # train.py saves: {"metrics": {"IoU": x, "F1": x, ...}}
            m = r.get("metrics", r)   # fallback to r itself if no metrics key
            results[key] = {
                'iou':       m.get('IoU',       m.get('iou',       0.0)),
                'f1':        m.get('F1',        m.get('f1',        0.0)),
                'precision': m.get('Precision', m.get('precision', 0.0)),
                'recall':    m.get('Recall',    m.get('recall',    0.0)),
                'ece':       m.get('ECE',       m.get('ece',       0.0)),
            }
            r = results[key]
            print(f"  {label}: "
                  f"IoU={r['iou']:.2f}%  "
                  f"F1={r['f1']:.2f}%  "
                  f"ECE={r['ece']:.2f}%")
        else:
            results[key] = None

    print_table(results)
    plot_bar_comparison(results)
    plot_raw_vs_indices(results)

    # Save text table
    txt = os.path.join(FIGURES_DIR, "l4s_indices_results.txt")
    with open(txt, "w") as f:
        f.write("L4S Spectral Indices Results\n")
        f.write("=" * 80 + "\n")
        for key, (label, _) in MODELS.items():
            r = results.get(key)
            if r:
                f.write(f"{label}: IoU={r['iou']:.2f}  F1={r['f1']:.2f}  "
                        f"Prec={r['precision']:.2f}  Rec={r['recall']:.2f}  "
                        f"ECE={r['ece']:.2f}\n")
    print(f"\nTable saved: {txt}")
    print(f"\nAll outputs in: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
