"""
Time Complexity Analysis
Measures parameters, FLOPs, and inference time per patch for all models.

Usage:
    python time_complexity.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.builder import build_model

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIGURES_DIR = "figures"
MODEL_KEYS  = ["cm1", "cm2", "m1", "m2", "m3"]
MODEL_LABELS = {
    "cm1": "CM1 — Concat U-Net",
    "cm2": "CM2 — Attention U-Net",
    "m1":  "M1 — HAEFNet [1]",
    "m2":  "M2 — HAEFNet + RO1",
    "m3":  "M3 — Proposed",
}
N_WARMUP    = 5     # warmup runs before timing
N_RUNS      = 50    # timing runs for average
BATCH_SIZE  = 1     # single patch inference
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_config(config_path, model_key):
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    global_params   = config.get('global_training', {})
    model_section   = config.get(model_key, {})
    merged_training = {**global_params, **model_section.get('training', {})}
    if 'loss' not in model_section.get('training', {}):
        merged_training['loss'] = global_params.get('loss', {})
    return {
        'data':       config['data'],
        'modalities': config['modalities'],
        'model':      model_section.get('model', {}),
        'training':   merged_training,
    }


def count_parameters(model):
    """Count total and trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    return total, trainable


def measure_inference_time(model, dummy_inputs, n_warmup=5, n_runs=50):
    """
    Measure average inference time per patch in milliseconds.
    Uses CUDA events for accurate GPU timing.
    """
    model.eval()

    # Warmup runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start_event.record()
                _ = model(dummy_inputs)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

        return np.mean(times), np.std(times)
    else:
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy_inputs)
                times.append((time.perf_counter() - t0) * 1000)
        return np.mean(times), np.std(times)


def main():
    config_path = "configs/exp_haefnet.yml"
    dummy_patch = [torch.randn(BATCH_SIZE, 3, 128, 128).to(DEVICE)
                   for _ in range(3)]

    results = {}
    print("\n" + "="*70)
    print("TIME COMPLEXITY ANALYSIS")
    print("="*70)
    print(f"{'Model':<35} {'Params (M)':>10} {'Inference (ms)':>16} {'Std':>8}")
    print("-"*70)

    for key in MODEL_KEYS:
        try:
            config = load_config(config_path, key)
            model  = build_model(config).to(DEVICE)

            total_params, trainable_params = count_parameters(model)
            params_M = total_params / 1e6

            avg_ms, std_ms = measure_inference_time(
                model, dummy_patch, N_WARMUP, N_RUNS)

            results[key] = {
                "params_M":   round(params_M, 2),
                "params_total": total_params,
                "inference_ms": round(avg_ms, 2),
                "inference_std": round(std_ms, 2),
            }

            print(f"{MODEL_LABELS[key]:<35} "
                  f"{params_M:>9.2f}M "
                  f"{avg_ms:>13.2f}ms "
                  f"±{std_ms:>6.2f}ms")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{MODEL_LABELS.get(key, key):<35} ERROR: {e}")
            results[key] = None

    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Patch size: 128×128 px | Batch size: {BATCH_SIZE}")
    print(f"Timing: {N_RUNS} runs after {N_WARMUP} warmup runs")

    # Save as text
    with open(os.path.join(FIGURES_DIR, "time_complexity.txt"), "w") as f:
        f.write("TIME COMPLEXITY ANALYSIS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Model':<35} {'Params (M)':>10} "
                f"{'Inference (ms)':>16} {'Std':>8}\n")
        f.write("-"*70 + "\n")
        for key in MODEL_KEYS:
            r = results.get(key)
            if r is None:
                f.write(f"{MODEL_LABELS[key]:<35} ERROR\n")
                continue
            f.write(f"{MODEL_LABELS[key]:<35} "
                    f"{r['params_M']:>9.2f}M "
                    f"{r['inference_ms']:>13.2f}ms "
                    f"±{r['inference_std']:>6.2f}ms\n")
        f.write(f"\nDevice: {DEVICE}\n")
        f.write(f"Patch size: 128x128 | Batch: {BATCH_SIZE} | "
                f"Runs: {N_RUNS}\n")
    print(f"\nSaved to {FIGURES_DIR}/time_complexity.txt")

    # ── Plot: Parameters vs Inference time ───────────────────────────────────
    valid_keys = [k for k in MODEL_KEYS if results.get(k)]
    params     = [results[k]["params_M"]    for k in valid_keys]
    times      = [results[k]["inference_ms"] for k in valid_keys]
    stds       = [results[k]["inference_std"] for k in valid_keys]
    labels     = [MODEL_LABELS[k]            for k in valid_keys]
    colors     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Parameters bar chart
    bars = ax1.bar(labels, params,
                   color=colors[:len(valid_keys)],
                   alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, params):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f"{val:.1f}M", ha='center', va='bottom',
                 fontsize=9, fontfamily='Times New Roman')
    ax1.set_ylabel("Parameters (Millions)",
                   fontsize=11, fontfamily='Times New Roman')
    ax1.set_title("Model Parameters",
                  fontsize=12, fontfamily='Times New Roman')
    ax1.set_xticklabels(labels, rotation=20, ha='right',
                        fontsize=8, fontfamily='Times New Roman')
    ax1.grid(axis='y', alpha=0.3)

    # Inference time bar chart
    bars2 = ax2.bar(labels, times,
                    color=colors[:len(valid_keys)],
                    alpha=0.85, edgecolor='white',
                    yerr=stds, capsize=4,
                    error_kw={'elinewidth': 1.5})
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(stds) * 0.1,
                 f"{val:.1f}ms", ha='center', va='bottom',
                 fontsize=9, fontfamily='Times New Roman')
    ax2.set_ylabel("Inference Time per Patch (ms)",
                   fontsize=11, fontfamily='Times New Roman')
    ax2.set_title("Inference Speed (lower is faster)",
                  fontsize=12, fontfamily='Times New Roman')
    ax2.set_xticklabels(labels, rotation=20, ha='right',
                        fontsize=8, fontfamily='Times New Roman')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle("Time Complexity Analysis — All Models",
                 fontsize=13, fontfamily='Times New Roman')
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "time_complexity.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
