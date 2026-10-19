from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
"""
RO3 Training Script — Leave-One-Region-Out Evaluation
Trains a model on all regions except the held-out region.
Tests on the held-out region only.

Usage:
    python train_ro3.py --model m1 --held_out chimanimani
    python train_ro3.py --model m3 --held_out chimanimani
    python train_ro3.py --model m1 --held_out hiroshima
    python train_ro3.py --model m3 --held_out hiroshima
    python train_ro3.py --model m1 --held_out indonesia
    python train_ro3.py --model m3 --held_out indonesia
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import yaml
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

from utils.multimodal_dataset import MultiModalRSDataset
from utils.augmentations import (get_traditional_train_augmentation,
                                  get_traditional_val_augmentation)
from utils.loss_factory import LossFactory
from models.builder import build_model
from utils.optimizer import PolyWarmupAdamW, CosineAnnealingWarmupAdamW
from utils.helpers import print_log
from utils.meter import AverageMeter, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES  = 2
IGNORE_LABEL = 255
RO3_DIR      = rstr(ROOT_DIR / "ro3_splits")
HELD_OUT_REGIONS = ["chimanimani", "hiroshima", "indonesia"]


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config",    type=str,
                        default="configs/exp_haefnet.yml")
    parser.add_argument("--model",     type=str, default="m1",
                        choices=["m1", "m3"],
                        help="Only M1 (baseline) and M3 (proposed) for RO3")
    parser.add_argument("--held_out",  type=str,
                        choices=HELD_OUT_REGIONS, required=True,
                        help="Region to hold out for testing")
    return parser.parse_args()


# ── Config ────────────────────────────────────────────────────────────────────
def load_config(config_path, model_key):
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


# ── Load split files ──────────────────────────────────────────────────────────
def load_ro3_splits(held_out):
    split_dir = os.path.join(RO3_DIR, f"holdout_{held_out}")
    def read(fname):
        with open(os.path.join(split_dir, fname)) as f:
            return [l.strip() for l in f if l.strip()]
    train_ids = read("train.txt")
    val_ids   = read("val.txt")
    test_ids  = read("test.txt")
    print_log(f"RO3 splits loaded for held_out={held_out}")
    print_log(f"  Train: {len(train_ids)} | Val: {len(val_ids)} | "
              f"Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# ── Dataloader ────────────────────────────────────────────────────────────────
def create_loader(config, file_ids, stage):
    modalities  = [k[4:] for k, v in config["modalities"].items()
                   if v and k.startswith("use_")]
    norm_config = config["model"].get("modality_norms", {})
    root_dir    = config["data"]["root_dir"]
    is_train    = (stage == "train")
    dataset     = MultiModalRSDataset(
        root_dir    = root_dir,
        file_list   = file_ids,
        modalities  = modalities,
        transform   = (get_traditional_train_augmentation(norm_config)
                       if is_train else
                       get_traditional_val_augmentation(norm_config)),
        stage       = stage,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset for stage={stage}")
    return DataLoader(
        dataset,
        batch_size  = config["training"]["batch_size"],
        shuffle     = is_train,
        num_workers = config["training"].get("num_workers", 0),
        pin_memory  = torch.cuda.is_available(),
        drop_last   = is_train,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(conf_mat, all_probs=None, all_labels=None):
    eps = 1e-7
    TP  = conf_mat[1, 1]; TN = conf_mat[0, 0]
    FP  = conf_mat[0, 1]; FN = conf_mat[1, 0]
    precision = TP / (TP + FP + eps) * 100
    recall    = TP / (TP + FN + eps) * 100
    f1        = (2 * precision * recall) / (precision + recall + eps)
    iou_ls    = TP / (TP + FP + FN + eps) * 100
    iou_bg    = TN / (TN + FN + FP + eps) * 100
    mean_iou  = (iou_ls + iou_bg) / 2

    ece = 0.0
    if all_probs is not None and all_labels is not None:
        bins  = np.linspace(0, 1, 16)
        total = len(all_probs) + eps
        for i in range(15):
            mask = (all_probs >= bins[i]) & (all_probs < bins[i+1])
            if mask.sum() > 0:
                ece += (np.abs(all_labels[mask].mean() -
                               all_probs[mask].mean()) *
                        mask.sum() / total)
        ece *= 100

    return {
        "IoU":       round(mean_iou,  2),
        "F1":        round(f1,        2),
        "Precision": round(precision, 2),
        "Recall":    round(recall,    2),
        "ECE":       round(ece,       2),
    }


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.1):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = 0.0
        self.counter     = 0
        self.should_stop = False

    def step(self, iou):
        if iou > self.best + self.min_delta:
            self.best    = iou
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch,
                lambda_head=0.0):
    model.train()
    total = 0.0
    with tqdm(total=len(loader), desc=f"Epoch {epoch}") as pbar:
        for images, labels, _ in loader:
            images = [img.to(DEVICE) for img in images]
            labels = labels.to(DEVICE)
            outputs, aux = model(images)
            modal_logits = product_logprob = theta = beta = None
            if isinstance(aux, dict):
                modal_logits    = aux.get("modal_logits")
                theta           = aux.get("theta")
                beta            = aux.get("beta")
                product_logprob = aux.get("product_logprob")
            loss = criterion(outputs[0], labels,
                             modal_logits=modal_logits, theta=theta,
                             beta=beta, product_logprob=product_logprob)
            if lambda_head > 0 and len(outputs) > 1:
                loss = loss + lambda_head * criterion(
                    outputs[1], labels,
                    modal_logits=None, theta=None,
                    beta=None, product_logprob=None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{total/len(loader):.4f}")
            pbar.update(1)
    return total / len(loader)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, lambda_head=0.0):
    model.eval()
    total    = 0.0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    probs_all, labels_all = [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = [img.to(DEVICE) for img in images]
            labels = labels.to(DEVICE)
            outputs, aux = model(images)
            modal_logits = product_logprob = theta = beta = None
            if isinstance(aux, dict):
                modal_logits    = aux.get("modal_logits")
                theta           = aux.get("theta")
                beta            = aux.get("beta")
                product_logprob = aux.get("product_logprob")
            loss = criterion(outputs[0], labels,
                             modal_logits=modal_logits, theta=theta,
                             beta=beta, product_logprob=product_logprob)
            total += loss.item()

            out   = outputs[0] if isinstance(outputs, list) else outputs
            out   = nn.functional.interpolate(
                out, size=labels.shape[1:],
                mode="bilinear", align_corners=False)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)

            lnp = labels.cpu().numpy().astype(np.int64)
            pnp = preds.cpu().numpy().astype(np.int64)
            pnp_prob = probs[:, 1].cpu().numpy()
            valid    = (lnp != IGNORE_LABEL)
            probs_all.append(pnp_prob[valid].flatten())
            labels_all.append(lnp[valid].flatten())
            try:
                conf_mat += confusion_matrix(
                    lnp.flatten(), pnp.flatten(),
                    NUM_CLASSES, ignore_label=IGNORE_LABEL)
            except Exception:
                pass

    probs_all  = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    metrics    = compute_metrics(conf_mat, probs_all, labels_all)
    return total / len(loader), metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args      = parse_args()
    config    = load_config(args.config, args.model)
    held_out  = args.held_out
    model_key = args.model

    # Reproducibility
    torch.manual_seed(3407)
    np.random.seed(3407)

    print_log("="*60)
    print_log(f"RO3 Training: model={model_key.upper()} "
              f"| held_out={held_out}")
    print_log("="*60)

    # Load RO3 splits
    train_ids, val_ids, test_ids = load_ro3_splits(held_out)

    # Dataloaders
    train_loader = create_loader(config, train_ids, "train")
    val_loader   = create_loader(config, val_ids,   "val")
    test_loader  = create_loader(config, test_ids,  "test")

    # Model
    model = build_model(config).to(DEVICE)

    # Optimizer
    param_groups = model.get_param_groups()
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])
    scheduler_type = config["training"].get(
        "scheduler", {}).get("type", "cos")

    opt_params = [
        {"params": param_groups[0], "lr": lr,      "weight_decay": wd},
        {"params": param_groups[1], "lr": lr,      "weight_decay": 0.0},
        {"params": param_groups[2], "lr": lr * 10, "weight_decay": wd},
    ]

    if scheduler_type.lower() == "cos":
        optimizer = CosineAnnealingWarmupAdamW(
            params            = opt_params,
            lr                = lr,
            weight_decay      = wd,
            betas             = [0.9, 0.999],
            T_max             = int(config["training"]["scheduler"]
                                   .get("T_max", 100000)),
            eta_min           = float(config["training"]["scheduler"]
                                     .get("eta_min", 1e-6)),
            warmup_iterations = int(config["training"]["scheduler"]
                                   .get("warmup_iterations", 500)),
            warmup_ratio      = 1e-6,
        )
    else:
        optimizer = PolyWarmupAdamW(
            params       = opt_params,
            lr           = lr, weight_decay = wd,
            betas        = [0.9, 0.999],
            warmup_iter  = 500, max_iter = 40000,
            warmup_ratio = 1e-6, power = 1.0,
        )

    criterion   = LossFactory.create_loss(config)
    lambda_head = float(config["training"].get(
        "loss_weights", {}).get("head", 0.0))

    # Checkpoint dir
    ckpt_dir = os.path.join(
        "checkpoints", "ro3",
        f"{model_key}_holdout_{held_out}")
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stop    = EarlyStopping(patience=8, min_delta=0.1)
    best_iou      = 0.0
    num_epochs    = config["training"].get("num_epochs", 50)
    save_interval = config["training"].get("save_interval", 5)

    # Training loop
    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion, epoch, lambda_head)

        if (epoch + 1) % save_interval == 0:
            _, val_metrics = evaluate(model, val_loader, criterion,
                                      lambda_head)
            elapsed = time.time() - t0
            print_log(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val IoU: {val_metrics['IoU']:.2f}% | "
                f"F1: {val_metrics['F1']:.2f}% | "
                f"Time: {elapsed/60:.1f}min"
            )

            if val_metrics["IoU"] > best_iou:
                best_iou = val_metrics["IoU"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "metrics": val_metrics},
                    os.path.join(ckpt_dir, "model_best.pth"))
                print_log(f"Saved best — IoU: {best_iou:.2f}%")

            if early_stop.step(val_metrics["IoU"]):
                print_log(f"Early stopping at epoch {epoch}")
                break

    # Final test on held-out region
    print_log(f"\n{'='*60}")
    print_log(f"TEST on held-out region: {held_out.upper()}")
    ckpt = torch.load(os.path.join(ckpt_dir, "model_best.pth"),
                      map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    _, test_metrics = evaluate(model, test_loader, criterion, lambda_head)

    print_log(f"{'='*60}")
    print_log(f"Model: {model_key.upper()} | Held-out: {held_out}")
    print_log(f"  IoU       : {test_metrics['IoU']:.2f}%")
    print_log(f"  F1        : {test_metrics['F1']:.2f}%")
    print_log(f"  Precision : {test_metrics['Precision']:.2f}%")
    print_log(f"  Recall    : {test_metrics['Recall']:.2f}%")
    print_log(f"  ECE       : {test_metrics['ECE']:.2f}%")
    print_log(f"{'='*60}")

    # Save results
    result = {
        "model":      model_key,
        "held_out":   held_out,
        "metrics":    test_metrics,
        "train_size": len(train_ids),
        "test_size":  len(test_ids),
    }
    np.save(os.path.join(ckpt_dir, "test_results.npy"), result)
    with open(os.path.join(ckpt_dir, "test_results.txt"), "w") as f:
        f.write(f"Model: {model_key} | Held-out: {held_out}\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.2f}%\n")
    print_log(f"Results saved to {ckpt_dir}/test_results.txt")


if __name__ == "__main__":
    main()
