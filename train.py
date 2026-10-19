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
from utils.augmentations import get_traditional_train_augmentation, get_traditional_val_augmentation
from utils.loss_factory import LossFactory
from models.builder import build_model
from utils.optimizer import PolyWarmupAdamW, CosineAnnealingWarmupAdamW
from utils.helpers import print_log, resolve_experiment_tag
from utils.meter import AverageMeter, confusion_matrix, getScores

import warnings
warnings.filterwarnings("ignore")

# ── Global constants ──────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2
IGNORE_LABEL = 255



def parse_args():
    parser = ArgumentParser(description="Landslide detection training pipeline")
    parser.add_argument("--config", type=str,
                        default="configs/exp_haefnet.yml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        choices=["cm1", "cm2", "m1", "m2", "m3"],
                        help="Model key to load (cm1/cm2/m1/m2/m3). "
                             "If None uses top-level training block.")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def load_config(config_path, model_key=None):
    """
    Load YAML config and merge global_training into the training block.
    If model_key is given, loads that model's section and overrides globals.
    """
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    global_params = config.get('global_training', {})

    if model_key and model_key in config:
        model_section = config[model_key]
        # Start from globals, override with model-specific values
        merged_training = {**global_params,
                           **model_section.get('training', {})}
        # Deep merge loss (model loss overrides global)
        if 'loss' not in model_section.get('training', {}):
            merged_training['loss'] = global_params.get('loss', {})
        return {
            'data':       config['data'],
            'modalities': config['modalities'],
            'model':      model_section.get('model', config.get('model', {})),
            'training':   merged_training,
        }

    # No model key → use top-level blocks, fill training from globals
    merged_training = {**global_params, **config.get('training', {})}
    config['training'] = merged_training
    return config


# ═════════════════════════════════════════════════════════════════════════════
# DATA SPLITTING  (70 / 10 / 20)
# ═════════════════════════════════════════════════════════════════════════════

def _load_id_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_splits(config):
    """
    Creates train / val / test splits and saves them to disk.
    Splits are reused across all models to guarantee identical test sets.
    Ratios: 70% train | 10% val | 20% test  (fixed seed for reproducibility)
    """
    root_dir        = config["data"]["root_dir"]
    train_list_path = os.path.join(root_dir, config["data"]["train_list"])
    split_cfg       = config["data"].get("split", {})

    val_ratio  = float(split_cfg.get("val_ratio",  0.10))
    test_ratio = float(split_cfg.get("test_ratio", 0.20))
    seed       = int(split_cfg.get("seed", 3407))

    # ── Check if debug mode (max_samples) ────────────────────────────────────
    max_samples = config["data"].get("max_samples", None)

    # ── Use different split files for debug vs full run ───────────────────────
    suffix     = f"_debug{max_samples}" if max_samples else ""
    train_path = os.path.join(root_dir, f"split_train{suffix}.txt")
    val_path   = os.path.join(root_dir, f"split_val{suffix}.txt")
    test_path  = os.path.join(root_dir, f"split_test{suffix}.txt")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train_ids = _load_id_list(train_path)
        val_ids   = _load_id_list(val_path)
        test_ids  = _load_id_list(test_path)
        print_log(f"Loaded existing splits → train={len(train_ids)}, "
                  f"val={len(val_ids)}, test={len(test_ids)}")
        return train_ids, val_ids, test_ids

    # ── Create fresh splits ───────────────────────────────────────────────────
    all_ids = _load_id_list(train_list_path)

    # ── Limit to max_samples for quick sanity check ───────────────────────────
    if max_samples is not None:
        rng_sub = np.random.default_rng(seed)
        all_ids = list(rng_sub.choice(
            all_ids,
            size=min(max_samples, len(all_ids)),
            replace=False))
        print_log(f"DEBUG MODE: limiting to {len(all_ids)} samples "
                  f"(set max_samples: null in config for full run)")
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(len(all_ids))

    test_count  = int(len(all_ids) * test_ratio)
    val_count   = int(len(all_ids) * val_ratio)
    train_count = len(all_ids) - val_count - test_count

    train_ids = [all_ids[i] for i in perm[:train_count]]
    val_ids   = [all_ids[i] for i in perm[train_count:train_count + val_count]]
    test_ids  = [all_ids[i] for i in perm[train_count + val_count:]]

    # Save splits to disk
    for path, ids in [(train_path, train_ids),
                      (val_path,   val_ids),
                      (test_path,  test_ids)]:
        with open(path, "w") as f:
            f.write("\n".join(ids))

    mode_str = f" [DEBUG {max_samples} samples]" if max_samples else ""
    print_log(f"Created splits{mode_str} u2192 train={len(train_ids)} ({1-val_ratio-test_ratio:.0%}), val={len(val_ids)} ({val_ratio:.0%}), test={len(test_ids)} ({test_ratio:.0%}), seed={seed}")
    return train_ids, val_ids, test_ids


# ═════════════════════════════════════════════════════════════════════════════
# DATALOADER
# ═════════════════════════════════════════════════════════════════════════════

def create_dataloader(config, file_ids, stage):
    """
    Creates a DataLoader for the given stage (train / val / test).
    All settings read from config — no hardcoded values.
    """
    modalities  = [k[4:] for k, v in config["modalities"].items()
                   if v and k.startswith("use_")]
    norm_config = config["model"].get("modality_norms", {})
    root_dir    = config["data"]["root_dir"]
    is_train    = (stage == "train")

    target_size = None
    if "target_size" in config["data"]:
        ts = config["data"]["target_size"]
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            target_size = (int(ts[0]), int(ts[1]))

    dataset = MultiModalRSDataset(
        root_dir    = root_dir,
        file_list   = file_ids,
        modalities  = modalities,
        transform   = (get_traditional_train_augmentation(norm_config)
                       if is_train
                       else get_traditional_val_augmentation(norm_config)),
        stage       = stage,
        target_size = target_size,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty for stage='{stage}'!")

    loader = DataLoader(
        dataset,
        batch_size  = config["training"]["batch_size"],
        shuffle     = is_train,
        num_workers = config["training"].get("num_workers", 0),
        pin_memory  = torch.cuda.is_available(),
        drop_last   = is_train,
    )
    return loader


class EarlyStopping:
    """
    Stops training if val IoU does not improve by min_delta for
    `patience` consecutive validation checks.
    """
    def __init__(self, patience=10, min_delta=0.1):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_iou    = 0.0
        self.counter     = 0
        self.should_stop = False

    def step(self, val_iou):
        if val_iou > self.best_iou + self.min_delta:
            self.best_iou = val_iou
            self.counter  = 0
        else:
            self.counter += 1
            print_log(f"EarlyStopping: no improvement "
                      f"{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                print_log("Early stopping triggered.")
        return self.should_stop


class Saver:
    def __init__(self, ckpt_dir, save_interval=5):
        self.directory     = ckpt_dir
        self.best_val      = 0.0
        self.save_interval = save_interval
        os.makedirs(self.directory, exist_ok=True)

    def save(self, val_iou, state_dict, epoch):
        # Always save latest
        torch.save(state_dict,
                   os.path.join(self.directory, "model_latest.pth"))
        # Save best
        if val_iou > self.best_val:
            self.best_val = val_iou
            torch.save(state_dict,
                       os.path.join(self.directory, "model_best.pth"))
            with open(os.path.join(self.directory, "best_score.txt"), "w") as f:
                f.write(f"Best IoU: {self.best_val:.4f}, Epoch: {epoch}")
            print_log(f"Saved new best model — IoU: {self.best_val:.4f}%")
        # Periodic checkpoint
        if (epoch + 1) % self.save_interval == 0:
            torch.save(state_dict,
                       os.path.join(self.directory,
                                    f"model_epoch_{epoch+1}.pth"))
            print_log(f"Saved periodic checkpoint at epoch {epoch+1}")


# METRICS  (IoU, F1, Precision, Recall, ECE)

def compute_metrics(conf_mat, all_probs=None, all_labels=None, n_bins=15):
    """
    Computes all evaluation metrics from confusion matrix.
    conf_mat : (2, 2) numpy array  [true × pred]
    all_probs  : flat numpy array of predicted probabilities for class 1
    all_labels : flat numpy array of true binary labels
    Returns dict with IoU, F1, Precision, Recall, ECE
    """
    eps = 1e-7
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]

    precision = TP / (TP + FP + eps) * 100.0
    recall    = TP / (TP + FN + eps) * 100.0
    f1        = (2 * precision * recall) / (precision + recall + eps)
    iou_ls    = TP / (TP + FP + FN + eps) * 100.0   # landslide class IoU
    iou_bg    = TN / (TN + FN + FP + eps) * 100.0   # background class IoU
    mean_iou  = (iou_ls + iou_bg) / 2.0

    # ECE — Expected Calibration Error
    ece = 0.0
    if all_probs is not None and all_labels is not None:
        bins  = np.linspace(0, 1, n_bins + 1)
        total = len(all_probs) + eps
        for i in range(n_bins):
            in_bin = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
            if in_bin.sum() > 0:
                acc_in_bin  = all_labels[in_bin].mean()
                conf_in_bin = all_probs[in_bin].mean()
                ece += np.abs(acc_in_bin - conf_in_bin) * in_bin.sum() / total
        ece *= 100.0  # express as percentage

    return {
        "IoU":       round(mean_iou,  4),
        "IoU_LS":    round(iou_ls,    4),
        "F1":        round(f1,        4),
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "ECE":       round(ece,       4),
    }


# TRAINING
def train_epoch(model, dataloader, optimizer, criterion,
                epoch, lambda_head=0.0):
    model.train()
    total_loss = 0.0
    batch_loss = AverageMeter()

    with tqdm(total=len(dataloader)) as pbar:
        for i, (images, labels, _) in enumerate(dataloader):
            images = [img.to(DEVICE) for img in images]
            labels = labels.to(DEVICE)

            outputs, aux_info = model(images)

            modal_logits = product_logprob = theta = beta = None
            if isinstance(aux_info, dict):
                modal_logits    = aux_info.get("modal_logits")
                theta           = aux_info.get("theta")
                beta            = aux_info.get("beta")
                product_logprob = aux_info.get("product_logprob")

            loss = criterion(outputs[0], labels,
                             modal_logits=modal_logits,
                             theta=theta, beta=beta,
                             product_logprob=product_logprob)

            if lambda_head > 0 and len(outputs) > 1:
                loss = loss + lambda_head * criterion(
                    outputs[1], labels,
                    modal_logits=None, theta=None,
                    beta=None, product_logprob=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.update(loss.item())
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss: {batch_loss.avg:.4f}")
            pbar.update(1)

    return total_loss / len(dataloader)



# VALIDATION / TEST

def evaluate(model, dataloader, criterion, lambda_head=0.0, stage="val"):
    """
    Runs evaluation on val or test set.
    Returns: avg_loss, metrics_dict, conf_mat
    """
    model.eval()
    total_loss = 0.0
    conf_mat   = np.zeros((NUM_CLASSES, NUM_CLASSES))
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(dataloader):
            images = [img.to(DEVICE) for img in images]
            labels = labels.to(DEVICE)

            outputs, aux_info = model(images)

            modal_logits = product_logprob = theta = beta = None
            if isinstance(aux_info, dict):
                modal_logits    = aux_info.get("modal_logits")
                theta           = aux_info.get("theta")
                beta            = aux_info.get("beta")
                product_logprob = aux_info.get("product_logprob")

            loss = criterion(outputs[0], labels,
                             modal_logits=modal_logits,
                             theta=theta, beta=beta,
                             product_logprob=product_logprob)

            if lambda_head > 0 and len(outputs) > 1:
                loss = loss + lambda_head * criterion(
                    outputs[1], labels,
                    modal_logits=None, theta=None,
                    beta=None, product_logprob=None)

            total_loss += loss.item()

            out = outputs[0] if isinstance(outputs, list) else outputs
            out = nn.functional.interpolate(
                out, size=labels.shape[1:],
                mode="bilinear", align_corners=False)

            probs       = torch.softmax(out, dim=1)
            preds       = torch.argmax(out, dim=1)
            labels_np   = labels.cpu().numpy().astype(np.int64)
            preds_np    = preds.cpu().numpy().astype(np.int64)
            probs_np    = probs[:, 1].cpu().numpy()   # prob of class 1

            valid_mask  = (labels_np != IGNORE_LABEL)
            all_probs.append(probs_np[valid_mask].flatten())
            all_labels.append(labels_np[valid_mask].flatten())

            try:
                batch_cm = confusion_matrix(
                    labels_np.flatten(), preds_np.flatten(),
                    NUM_CLASSES, ignore_label=IGNORE_LABEL)
                conf_mat += batch_cm
            except Exception as e:
                print(f"Error in batch {i}: {e}")

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics    = compute_metrics(conf_mat, all_probs, all_labels)

    return total_loss / len(dataloader), metrics, conf_mat


def main():
    args   = parse_args()
    config = load_config(args.config, model_key=args.model)

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = int(config["data"].get("split", {}).get("seed", 3407))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = args.model or config["training"].get("task_name", "model")
    print_log(f"{'='*60}")
    print_log(f"Training model: {model_name.upper()}")
    print_log(f"Device: {DEVICE}")
    print_log(f"{'='*60}")

    # ── Splits ────────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = prepare_splits(config)

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader = create_dataloader(config, train_ids, stage="train")
    val_loader   = create_dataloader(config, val_ids,   stage="val")
    test_loader  = create_dataloader(config, test_ids,  stage="test")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(DEVICE)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    param_groups   = model.get_param_groups()
    lr             = float(config["training"]["learning_rate"])
    wd             = float(config["training"]["weight_decay"])
    scheduler_type = config["training"].get("scheduler", {}).get("type", "cos")

    opt_params = [
        {"params": param_groups[0], "lr": lr,      "weight_decay": wd},
        {"params": param_groups[1], "lr": lr,      "weight_decay": 0.0},
        {"params": param_groups[2], "lr": lr * 10, "weight_decay": wd},
    ]

    if scheduler_type.lower() == "cos":
        optimizer = CosineAnnealingWarmupAdamW(
            params             = opt_params,
            lr                 = lr,
            weight_decay       = wd,
            betas              = [0.9, 0.999],
            T_max              = int(config["training"]["scheduler"].get("T_max", 100000)),
            eta_min            = float(config["training"]["scheduler"].get("eta_min", 1e-6)),
            warmup_iterations  = int(config["training"]["scheduler"].get("warmup_iterations", 500)),
            warmup_ratio       = float(config["training"]["scheduler"].get("warmup_ratio", 1e-6)),
        )
    else:
        optimizer = PolyWarmupAdamW(
            params       = opt_params,
            lr           = lr,
            weight_decay = wd,
            betas        = [0.9, 0.999],
            warmup_iter  = int(config["training"].get("optimizer_params", {}).get("warmup_iter", 500)),
            max_iter     = int(config["training"].get("optimizer_params", {}).get("max_iter", 40000)),
            warmup_ratio = float(config["training"].get("optimizer_params", {}).get("warmup_ratio", 1e-6)),
            power        = float(config["training"].get("optimizer_params", {}).get("power", 1.0)),
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion   = LossFactory.create_loss(config)
    lambda_head = float(config["training"].get("loss_weights", {}).get("head", 0.0))

    # ── Saver and early stopping ──────────────────────────────────────────────
    ckpt_dir = os.path.join(
        config["training"].get("ckpt_dir", "checkpoints"),
        model_name)

    saver = Saver(
        ckpt_dir      = ckpt_dir,
        save_interval = config["training"].get("save_interval", 5))

    es_cfg       = config["training"].get("early_stopping", {})
    early_stop   = EarlyStopping(
        patience  = int(es_cfg.get("patience",  10)),
        min_delta = float(es_cfg.get("min_delta", 0.1)))

    # ── Training curves storage ───────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_iou": [],
               "val_f1": [], "val_precision": [], "val_recall": [],
               "val_ece": []}

    num_epochs    = config["training"]["num_epochs"]
    save_interval = config["training"].get("save_interval", 5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print_log(f"Starting training — {num_epochs} epochs max, "
              f"early stopping patience={early_stop.patience}")

    for epoch in range(num_epochs):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer,
                                 criterion, epoch, lambda_head)
        history["train_loss"].append(train_loss)

        if (epoch + 1) % save_interval == 0:
            val_loss, val_metrics, _ = evaluate(
                model, val_loader, criterion, lambda_head, stage="val")

            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_metrics["IoU"])
            history["val_f1"].append(val_metrics["F1"])
            history["val_precision"].append(val_metrics["Precision"])
            history["val_recall"].append(val_metrics["Recall"])
            history["val_ece"].append(val_metrics["ECE"])

            elapsed = time.time() - t0
            print_log(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {val_metrics['IoU']:.2f}% | "
                f"F1: {val_metrics['F1']:.2f}% | "
                f"Prec: {val_metrics['Precision']:.2f}% | "
                f"Rec: {val_metrics['Recall']:.2f}% | "
                f"ECE: {val_metrics['ECE']:.2f}% | "
                f"Time: {elapsed/60:.1f}min"
            )

            saver.save(val_metrics["IoU"],
                       {"model": model.state_dict(), "epoch": epoch,
                        "metrics": val_metrics},
                       epoch=epoch)

            if early_stop.step(val_metrics["IoU"]):
                print_log(f"Early stopping at epoch {epoch}.")
                break

    # ── Save training history ─────────────────────────────────────────────────
    np.save(os.path.join(ckpt_dir, "history.npy"), history)
    print_log(f"Training history saved to {ckpt_dir}/history.npy")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print_log(f"\n{'='*60}")
    print_log("Loading best checkpoint for final TEST evaluation...")
    best_ckpt = os.path.join(ckpt_dir, "model_best.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print_log(f"Loaded best model from epoch {ckpt.get('epoch', '?')}")
    else:
        print_log("No best checkpoint found — using last epoch weights.")

    test_loss, test_metrics, test_conf_mat = evaluate(
        model, test_loader, criterion, lambda_head, stage="test")

    print_log(f"\n{'='*60}")
    print_log(f"FINAL TEST RESULTS — {model_name.upper()}")
    print_log(f"{'='*60}")
    print_log(f"  Test Loss  : {test_loss:.4f}")
    print_log(f"  IoU        : {test_metrics['IoU']:.2f}%  "
              f"(landslide class: {test_metrics['IoU_LS']:.2f}%)")
    print_log(f"  F1-Score   : {test_metrics['F1']:.2f}%")
    print_log(f"  Precision  : {test_metrics['Precision']:.2f}%")
    print_log(f"  Recall     : {test_metrics['Recall']:.2f}%")
    print_log(f"  ECE        : {test_metrics['ECE']:.2f}%")
    print_log(f"{'='*60}\n")

    # ── Save test results ─────────────────────────────────────────────────────
    results = {
        "model":       model_name,
        "test_loss":   test_loss,
        "metrics":     test_metrics,
        "conf_mat":    test_conf_mat.tolist(),
    }
    np.save(os.path.join(ckpt_dir, "test_results.npy"), results)

    # Also save as readable text
    with open(os.path.join(ckpt_dir, "test_results.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        for k, v in test_metrics.items():
            f.write(f"{k}: {v:.4f}%\n")
        f.write(f"\nConfusion Matrix (True x Pred):\n")
        f.write(f"  TN={int(test_conf_mat[0,0])}  FP={int(test_conf_mat[0,1])}\n")
        f.write(f"  FN={int(test_conf_mat[1,0])}  TP={int(test_conf_mat[1,1])}\n")

    print_log(f"Test results saved to {ckpt_dir}/test_results.txt")

    # ── Parameter count ───────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print_log(f"Model parameters: {n_params:.2f}M")


if __name__ == "__main__":
    main()
