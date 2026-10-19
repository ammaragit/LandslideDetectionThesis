

import os, sys, platform
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

# ── 1. SYSTEM INFO ────────────────────────────────────────────────────────────

def print_system_info():
    print("\n" + "=" * 60)
    print("  SYSTEM & SOFTWARE ENVIRONMENT")
    print("=" * 60)
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  Platform     : {platform.system()} {platform.release()}")
    try:
        import torch
        print(f"  PyTorch      : {torch.__version__}")
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU          : {gpu}  ({vram:.1f} GB VRAM)")
            print(f"  CUDA         : {torch.version.cuda}")
        else:
            print("  GPU          : NOT available — running on CPU")
    except ImportError:
        print("  PyTorch      : not installed")
    try:
        import torchvision
        print(f"  torchvision  : {torchvision.__version__}")
    except ImportError:
        pass
    try:
        import numpy as np
        print(f"  NumPy        : {np.__version__}")
    except ImportError:
        pass
    print("=" * 60 + "\n")


# ── 2. ARGS ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    type=str,
                   default="configs/exp_haefnet.yml")
    p.add_argument("--ckpt",      type=str,
                   default="checkpoints/m3/model_best.pth")
    p.add_argument("--dataset",   type=str, default="sen12",
                   choices=["sen12", "l4s"])
    p.add_argument("--patch_idx", type=int, default=None,
                   help="Force a specific test patch index.")
    p.add_argument("--out_dir",   type=str, default="demo_output")
    p.add_argument("--root_dir",  type=str, default=None,
                   help="Override config's data.root_dir for preprocessed .npz files.")
    p.add_argument("--split_dir", type=str, default=None,
                   help="Directory where split_test.txt lives, if different from root_dir.")
    return p.parse_args()


# ── 3. CONFIG ─────────────────────────────────────────────────────────────────

def load_config(config_path, model_key="m3"):
    print(f"  Config: {config_path}")
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
        '_raw':       config,
    }


# ── 4. SPLIT FILE ─────────────────────────────────────────────────────────────

def find_test_ids(config, split_dir=None):
    """
    Replicates train.py's prepare_splits logic to locate the test split file.
    Returns a list of patch IDs.
    split_dir: optional override for the directory containing split_test.txt
    """
    root_dir   = config['data']['root_dir']
    split_cfg  = config['data'].get('split', {})
    debug      = config['data'].get('max_samples') is not None
    suffix     = "_debug" if debug else ""

    # Search order: split_dir override → root_dir → project root (one level up)
    search_dirs = []
    if split_dir:
        search_dirs.append(split_dir)
    search_dirs.append(root_dir)
    search_dirs.append(os.path.dirname(root_dir))  # one level up

    test_path = None
    for d in search_dirs:
        candidate = os.path.join(d, f"split_test{suffix}.txt")
        if os.path.exists(candidate):
            test_path = candidate
            break

    if test_path is None:
        test_path = os.path.join(root_dir, f"split_test{suffix}.txt")  # fallback for error msg
    if os.path.exists(test_path):
        with open(test_path) as f:
            ids = [l.strip() for l in f if l.strip()]
        print(f"  Split file   : {test_path}  ({len(ids)} patches)")
        return ids

    # Fall back to train_list.txt and re-create splits on the fly
    train_list = os.path.join(root_dir,
                              config['data'].get('train_list', 'train_list.txt'))
    if not os.path.exists(train_list):
        print(f"  ERROR: Neither {test_path} nor {train_list} found.")
        print("  Run training first so split files are created.")
        sys.exit(1)

    with open(train_list) as f:
        all_ids = [l.strip() for l in f if l.strip()]

    val_ratio  = float(split_cfg.get('val_ratio',  0.10))
    test_ratio = float(split_cfg.get('test_ratio', 0.20))
    seed       = int(split_cfg.get('seed', 3407))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_ids))
    n_test = int(len(all_ids) * test_ratio)
    test_ids = [all_ids[i] for i in idx[:n_test]]
    print(f"  Split file not found — re-derived {len(test_ids)} test IDs "
          f"from {train_list} (seed={seed})")
    return test_ids


# ── 5. DATASET ────────────────────────────────────────────────────────────────

def load_dataset(config, test_ids, dataset_type):
    from utils.augmentations import get_traditional_val_augmentation
    norm_config = config['model'].get('modality_norms', {})
    modalities  = [k[4:] for k, v in config['modalities'].items()
                   if v and k.startswith('use_')]

    root_dir = config['data']['root_dir']

    if dataset_type == 'l4s':
        # multimodal_dataset.py appends "preprocessed/" to root_dir,
        # but L4S .npz files live directly in root_dir (preprocessed_indices/).
        # Fix: subclass to override npz_dir without touching existing code.
        from utils.multimodal_dataset import MultiModalRSDataset as _BaseDS
        class L4SDataset(_BaseDS):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Override the wrong subfolder with the actual location
                self.npz_dir = kwargs['root_dir']
                # Re-run verification with the correct path
                self.datalist = [
                    pid for pid in self.datalist
                    if os.path.exists(os.path.join(self.npz_dir, pid + '.npz'))
                ]
        dataset = L4SDataset(
            root_dir   = root_dir,
            file_list  = test_ids,
            modalities = modalities,
            transform  = get_traditional_val_augmentation(norm_config),
            stage      = 'test',
        )
    else:
        # Sen12: use multimodal_dataset_s12 which sets
        # npz_dir = root_dir/data/preprocessed  (root_dir = project root)
        from utils.multimodal_dataset_s12 import MultiModalRSDataset
        dataset = MultiModalRSDataset(
            root_dir   = root_dir,
            file_list  = test_ids,
            modalities = modalities,
            transform  = get_traditional_val_augmentation(norm_config),
            stage      = 'test',
        )

    print(f"  Valid patches: {len(dataset)}")
    if len(dataset) == 0:
        print("\n  ERROR: Dataset has 0 valid patches.")
        print("  Check that config data.root_dir points to the correct folder")
        print(f"  and that preprocessed .npz files exist inside it.")
        sys.exit(1)

    return dataset, modalities


# ── 6. MODEL ──────────────────────────────────────────────────────────────────

def load_model(config, ckpt_path, device):
    from models.builder import build_model
    print(f"  Building M3...")
    model = build_model(config).to(device)

    if not os.path.exists(ckpt_path):
        print(f"\n  ERROR: Checkpoint not found: {ckpt_path}")
        print("  Train M3 first with:  python train.py --model m3")
        sys.exit(1)

    print(f"  Checkpoint   : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    epoch    = ckpt.get('epoch', '?')
    best_iou = ckpt.get('best_iou', None)
    print(f"  Saved epoch  : {epoch}")
    if best_iou is not None:
        print(f"  Best val mIoU: {best_iou:.2f}%")
    model.eval()
    return model


# ── 7. INFERENCE ──────────────────────────────────────────────────────────────

def run_inference(model, images, device):
    model.eval()
    with torch.no_grad():
        imgs_gpu = [img.unsqueeze(0).to(device) for img in images]
        outputs, aux = model(imgs_gpu)

        out = F.interpolate(outputs[0], size=(128, 128),
                            mode='bilinear', align_corners=False)
        prob_map = torch.softmax(out, dim=1)[0, 1].cpu().numpy()
        pred     = (prob_map > 0.5).astype(np.uint8)

        conf_maps = aux.get('conf_maps', None)
        if conf_maps is not None:
            conf_maps = F.interpolate(conf_maps, size=(128, 128),
                                      mode='bilinear', align_corners=False)
            conf_maps = conf_maps.squeeze(0).cpu().numpy()
        else:
            print("  WARNING: conf_maps not in aux — using uniform weights.")
            conf_maps = np.ones((3, 128, 128)) / 3.0

    return pred, prob_map, conf_maps


# ── 8. METRICS ────────────────────────────────────────────────────────────────

def compute_metrics(pred, label):
    p = pred.flatten().astype(bool)
    g = label.flatten().astype(bool)
    tp = (p &  g).sum();  fp = (p & ~g).sum()
    fn = (~p & g).sum();  tn = (~p & ~g).sum()
    iou  = tp / (tp + fp + fn + 1e-7)
    f1   = 2*tp / (2*tp + fp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    rec  = tp / (tp + fn + 1e-7)
    return dict(IoU=iou*100, F1=f1*100, Precision=prec*100, Recall=rec*100)


# ── 9. FIGURE ─────────────────────────────────────────────────────────────────

def save_figure(images, label_np, pred, prob_map, conf_maps,
                metrics, patch_id, out_path, dataset_type):

    if dataset_type == 'l4s':
        in_titles   = ["Indices\n(Slot 1)", "Optical\n(RGB)", "DEM"]
        conf_titles = ["Index\nReliability", "Optical\nReliability",
                       "DEM\nReliability"]
    else:
        in_titles   = ["SAR\n(Change)", "Optical\n(RGB)", "DEM"]
        conf_titles = ["SAR\nReliability", "Optical\nReliability",
                       "DEM\nReliability"]

    in_cmaps   = ["gray", "viridis", "terrain"]
    conf_cmaps = ["Blues", "Greens", "Oranges"]

    fig = plt.figure(figsize=(22, 6))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    # Top row: inputs + ground truth
    for col, (img_t, title, cmap) in enumerate(
            zip(images, in_titles, in_cmaps)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_t.numpy()[0], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(label_np, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title("Ground Truth", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Bottom row: prediction + reliability maps
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(prob_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title("M3 Prediction\n(landslide prob.)",
                 fontsize=10, fontweight='bold')
    ax.contour(label_np, levels=[0.5], colors='white',
               linewidths=0.8, linestyles='--')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, (title, cmap) in enumerate(zip(conf_titles, conf_cmaps)):
        ax = fig.add_subplot(gs[1, col + 1])
        im = ax.imshow(conf_maps[col], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    m = metrics
    fig.suptitle(
        f"HAEFNet M3 — Patch: {patch_id}\n"
        f"IoU = {m['IoU']:.1f}%   F1 = {m['F1']:.1f}%   "
        f"Precision = {m['Precision']:.1f}%   Recall = {m['Recall']:.1f}%",
        fontsize=12, fontweight='bold', y=1.02)

    fig.text(0.5, -0.03,
             "Top: model inputs and ground-truth mask.  "
             "Bottom: predicted landslide probability "
             "(white dashed = GT boundary) and RO2 per-modality reliability weights.",
             ha='center', va='top', fontsize=8, color='#444444', style='italic')

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {os.path.abspath(out_path)}")


# ── 10. MAIN ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print_system_info()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Running on: {device}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # Config
    print("STEP 1 — Configuration")
    print("-" * 40)
    config = load_config(args.config, model_key='m3')

    # Allow root_dir override without touching the config file
    if args.root_dir is not None:
        config['data']['root_dir'] = args.root_dir
        print(f"  root_dir override: {args.root_dir}")

    # Resolve checkpoint path — auto-default for l4s if user didn't override
    if args.ckpt == "checkpoints/m3/model_best.pth" and args.dataset == 'l4s':
        args.ckpt = "checkpoints/l4s_indices/m3/model_best.pth"

    # Model
    print("\nSTEP 2 — Model")
    print("-" * 40)
    model = load_model(config, args.ckpt, device)

    # Dataset — use same split logic as train.py
    print("\nSTEP 3 — Dataset")
    print("-" * 40)
    test_ids = find_test_ids(config, split_dir=args.split_dir)
    dataset, modalities = load_dataset(config, test_ids, args.dataset)

    # Patch selection
    print("\nSTEP 4 — Patch selection")
    print("-" * 40)

    if args.patch_idx is not None:
        idx = args.patch_idx
        images, label, meta = dataset[idx]
        print(f"  Using patch index {idx} (forced via --patch_idx)")
    else:
        print("  Searching for patch with landslide pixels...")
        np.random.seed(42)
        probe_order = np.random.permutation(len(dataset))
        images = label = meta = None
        for idx in probe_order:
            imgs_i, lbl_i, meta_i = dataset[int(idx)]
            if lbl_i.numpy().sum() > 50:
                images, label, meta = imgs_i, lbl_i, meta_i
                ls_px = int(lbl_i.numpy().sum())
                print(f"  Found patch at index {idx}  "
                      f"(landslide pixels: {ls_px})")
                break
        if images is None:
            print("  No landslide patch found — using index 0")
            images, label, meta = dataset[0]

    label_np = label.numpy()
    patch_id = (meta.get('file_id', f'patch_{idx}')
                if isinstance(meta, dict) else f'patch_{idx}')
    print(f"  Patch ID: {patch_id}")
    print(f"  Landslide pixels: {int(label_np.sum())} / "
          f"{label_np.size} ({100*label_np.mean():.1f}%)")

    # Inference
    print("\nSTEP 5 — Inference")
    print("-" * 40)
    pred, prob_map, conf_maps = run_inference(model, images, device)

    # Metrics
    metrics = compute_metrics(pred, label_np)
    print(f"  IoU        : {metrics['IoU']:.2f}%")
    print(f"  F1         : {metrics['F1']:.2f}%")
    print(f"  Precision  : {metrics['Precision']:.2f}%")
    print(f"  Recall     : {metrics['Recall']:.2f}%")

    # Reliability weights
    modal_names = (["Indices (Slot 1)", "Optical (RGB)", "DEM"]
                   if args.dataset == 'l4s'
                   else ["SAR (change)", "Optical (RGB)", "DEM"])
    print("\n  Per-modality reliability weights (spatial mean):")
    for name, c in zip(modal_names, conf_maps.mean(axis=(1, 2))):
        bar = "█" * int(c * 40)
        print(f"    {name:<22}: {c:.4f}  {bar}")

    # Figure
    print("\nSTEP 6 — Saving figure")
    print("-" * 40)
    out_path = os.path.join(args.out_dir, "demo_result.png")
    save_figure(images, label_np, pred, prob_map, conf_maps,
                metrics, patch_id, out_path, args.dataset)

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print(f"  Open: {os.path.abspath(out_path)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()