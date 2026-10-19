"""
RO3 Evaluation Script — Cross-Regional Generalization
Loads all RO3 checkpoint results and generates:
- Cross-regional comparison table (M1 vs M3)
- Bar chart per region
- Summary statistics

Usage:
    python evaluate_ro3.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
HELD_OUT_REGIONS = ["chimanimani", "hiroshima", "indonesia"]
MODELS           = ["m1", "m3"]
MODEL_LABELS = {
                    "m1": "M1 — HAEFNet [1]",
                    "m2": "M2 — HAEFNet + RO1",
                    "m3": "M3 — Proposed (RO1+RO2)"
                }
FIGURES_DIR      = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load results ──────────────────────────────────────────────────────────────
def load_ro3_result(model_key, held_out):
    path = os.path.join("checkpoints", "ro3",
                        f"{model_key}_holdout_{held_out}",
                        "test_results.npy")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    return np.load(path, allow_pickle=True).item()


# ── Print table ───────────────────────────────────────────────────────────────
def print_ro3_table(all_results):
    print("\n" + "="*85)
    print("RO3 — CROSS-REGIONAL GENERALIZATION RESULTS")
    print("="*85)
    print(f"{'Held-out Region':<20} {'Model':<30} "
          f"{'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'ECE':>7}")
    print("-"*85)

    rows = []
    for region in HELD_OUT_REGIONS:
        for model_key in MODELS:
            r = all_results.get((model_key, region))
            if r is None:
                print(f"{region:<20} {MODEL_LABELS[model_key]:<30} "
                      f"{'N/A':>7} {'N/A':>7} {'N/A':>7} "
                      f"{'N/A':>7} {'N/A':>7}")
                rows.append((region, model_key, None))
                continue
            m = r["metrics"]
            print(f"{region:<20} {MODEL_LABELS[model_key]:<30} "
                  f"{m['IoU']:>6.2f}% "
                  f"{m['F1']:>6.2f}% "
                  f"{m['Precision']:>6.2f}% "
                  f"{m['Recall']:>6.2f}% "
                  f"{m['ECE']:>6.2f}%")
            rows.append((region, model_key, m))
        print()

    print("="*85)

    # Print improvement row
    print("\nIMPROVEMENT — M3 vs M1 (Proposed vs Base Paper)")
    print("-"*50)
    improvements = []
    for region in HELD_OUT_REGIONS:
        r_m1 = all_results.get(("m1", region))
        r_m3 = all_results.get(("m3", region))
        if r_m1 and r_m3:
            d_iou = r_m3["metrics"]["IoU"] - r_m1["metrics"]["IoU"]
            d_ece = r_m3["metrics"]["ECE"] - r_m1["metrics"]["ECE"]
            sign  = "+" if d_iou >= 0 else ""
            print(f"  {region:<20}: Delta IoU = {sign}{d_iou:.2f}%  "
                  f"Delta ECE = {'+' if d_ece>=0 else ''}{d_ece:.2f}%")
            improvements.append(d_iou)

    if improvements:
        avg_imp = np.mean(improvements)
        print(f"\n  Average IoU improvement: "
              f"{'+' if avg_imp>=0 else ''}{avg_imp:.2f}%")

    # Save table to file
    with open(os.path.join(FIGURES_DIR, "ro3_table.txt"), "w") as f:
        f.write("RO3 — CROSS-REGIONAL GENERALIZATION RESULTS\n")
        f.write("="*85 + "\n")
        f.write(f"{'Held-out Region':<20} {'Model':<30} "
                f"{'IoU':>7} {'F1':>7} {'Prec':>7} "
                f"{'Rec':>7} {'ECE':>7}\n")
        f.write("-"*85 + "\n")
        for region in HELD_OUT_REGIONS:
            for model_key in MODELS:
                r = all_results.get((model_key, region))
                if r is None:
                    f.write(f"{region:<20} "
                            f"{MODEL_LABELS[model_key]:<30} N/A\n")
                    continue
                m = r["metrics"]
                f.write(f"{region:<20} {MODEL_LABELS[model_key]:<30} "
                        f"{m['IoU']:>6.2f}% {m['F1']:>6.2f}% "
                        f"{m['Precision']:>6.2f}% {m['Recall']:>6.2f}% "
                        f"{m['ECE']:>6.2f}%\n")
            f.write("\n")

    print(f"\nTable saved to {FIGURES_DIR}/ro3_table.txt")


# ── IoU bar chart per region ───────────────────────────────────────────────────
def plot_ro3_iou(all_results):
    regions    = HELD_OUT_REGIONS
    n_regions  = len(regions)
    x          = np.arange(n_regions)
    width      = 0.3
    colors     = {"m1": "#4C72B0", "m3": "#C44E52"}
    hatches    = {"m1": "", "m3": "//"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── IoU comparison ────────────────────────────────────────────────────────
    ax = axes[0]
    for i, model_key in enumerate(MODELS):
        values = []
        for region in regions:
            r = all_results.get((model_key, region))
            values.append(r["metrics"]["IoU"] if r else 0.0)
        bars = ax.bar(x + i * width, values, width,
                      label=MODEL_LABELS[model_key],
                      color=colors[model_key],
                      hatch=hatches[model_key],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{val:.1f}%", ha='center', va='bottom',
                        fontsize=9, fontfamily='Times New Roman')

    ax.set_xlabel("Held-out Region", fontsize=11,
                  fontfamily='Times New Roman')
    ax.set_ylabel("IoU (%)", fontsize=11, fontfamily='Times New Roman')
    ax.set_title("Cross-Regional IoU — M1 vs M3",
                 fontsize=12, fontfamily='Times New Roman')
    ax.set_xticks(x + width/2)
    region_labels = [r.capitalize() for r in regions]
    ax.set_xticklabels(region_labels, fontsize=10,
                       fontfamily='Times New Roman')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(axis='y', alpha=0.3)

    # ── ECE comparison ────────────────────────────────────────────────────────
    ax2 = axes[1]
    for i, model_key in enumerate(MODELS):
        values = []
        for region in regions:
            r = all_results.get((model_key, region))
            values.append(r["metrics"]["ECE"] if r else 0.0)
        bars = ax2.bar(x + i * width, values, width,
                       label=MODEL_LABELS[model_key],
                       color=colors[model_key],
                       hatch=hatches[model_key],
                       alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.02,
                         f"{val:.2f}%", ha='center', va='bottom',
                         fontsize=9, fontfamily='Times New Roman')

    ax2.set_xlabel("Held-out Region", fontsize=11,
                   fontfamily='Times New Roman')
    ax2.set_ylabel("ECE (%) — lower is better", fontsize=11,
                   fontfamily='Times New Roman')
    ax2.set_title("Cross-Regional Calibration — M1 vs M3",
                  fontsize=12, fontfamily='Times New Roman')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(region_labels, fontsize=10,
                        fontfamily='Times New Roman')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle("RO3 — Cross-Regional Generalization Evaluation",
                 fontsize=13, fontfamily='Times New Roman', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ro3_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RO3 figure saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("EVALUATE RO3 — Cross-Regional Generalization")
    print("="*60)

    all_results = {}
    for model_key in MODELS:
        for region in HELD_OUT_REGIONS:
            r = load_ro3_result(model_key, region)
            all_results[(model_key, region)] = r
            if r:
                m = r["metrics"]
                print(f"  {model_key.upper()} | {region:<15} "
                      f"IoU={m['IoU']:.2f}%  "
                      f"F1={m['F1']:.2f}%  "
                      f"ECE={m['ECE']:.2f}%")

    print_ro3_table(all_results)
    plot_ro3_iou(all_results)

    print(f"\nAll RO3 outputs saved to {FIGURES_DIR}/")
    print("  ro3_table.txt")
    print("  ro3_comparison.png")


if __name__ == "__main__":
    main()
