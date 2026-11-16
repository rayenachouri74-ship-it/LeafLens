"""LeafLens training.

Usage:
  python3 train.py --data data/olive_github --epochs 12 --mode frozen
  python3 train.py --data data/olive_github --epochs 12 --mode finetuned \
      --init checkpoints/frozen_best.pt

- Builds (or reuses) the stratified split in results/split.csv.
- Trains MobileNetV2 + head, class-weighted loss, AdamW, cosine
  schedule, augmentation on train only.
- Early stopping on validation loss (patience 4) and keeps the best
  checkpoint by validation loss, not by the last epoch.
- Writes results/history_<mode>.json and checkpoints/<mode>_best.pt,
  plus a plain-text log that is the exact training record.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (CLASSES, LeafDataset, class_weights, discover_images,
                  load_split_csv, make_split, write_split_csv)
from model import build_model


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(x)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["frozen", "finetuned"],
                    default="frozen")
    ap.add_argument("--init", default=None,
                    help="Checkpoint to start from (finetuned mode).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    torch.set_num_threads(2)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = results_dir / "split.csv"

    if split_path.exists():
        splits = load_split_csv(split_path)
        print(f"Reusing existing split: {split_path}")
    else:
        items = discover_images(args.data)
        if not items:
            raise SystemExit(f"No images found under {args.data}")
        splits = make_split(items, seed=args.seed)
        write_split_csv(splits, split_path)
        print(f"Split written: {split_path}")
    for s in ("train", "val", "test"):
        from collections import Counter
        print(f"  {s}: {len(splits[s])} "
              f"{dict(Counter(c for _, c in splits[s]))}")

    weights, counts = class_weights(splits["train"])
    print(f"Class weights (train): {dict(zip(CLASSES, weights.tolist()))}")

    train_ds = LeafDataset(splits["train"], args.image_size, augment=True)
    val_ds = LeafDataset(splits["val"], args.image_size, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    lr = args.lr if args.mode == "frozen" else 1e-4
    model, used_pretrained = build_model(mode=args.mode,
                                         pretrained=True)
    if args.init:
        state = torch.load(args.init, map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        print(f"Initialised from {args.init}")
        # In finetuned mode, unfreeze everything after loading.
        if args.mode == "finetuned":
            for p in model.parameters():
                p.requires_grad = True
    model.to(device)
    print(f"Mode={args.mode} pretrained={used_pretrained} lr={lr}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    history = {"mode": args.mode, "pretrained": used_pretrained,
               "lr": lr, "batch_size": args.batch_size,
               "image_size": args.image_size, "seed": args.seed,
               "train_counts": dict(counts), "epochs": []}
    best_val_loss, best_path, patience = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion,
                                    optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": round(tr_loss, 4),
               "train_acc": round(tr_acc, 4),
               "val_loss": round(va_loss, 4),
               "val_acc": round(va_acc, 4),
               "elapsed_s": round(time.time() - t0, 1)}
        history["epochs"].append(rec)
        print(f"epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            patience = 0
            best_path = out_dir / f"{args.mode}_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best checkpoint: {best_path}")
        else:
            patience += 1
            print(f"  -> no improvement (patience {patience}/4)")
            if patience >= 4:
                print("Early stopping.")
                break
    history["best_val_loss"] = best_val_loss
    history["best_checkpoint"] = str(best_path)
    history["total_time_s"] = round(time.time() - t0, 1)
    hist_path = results_dir / f"history_{args.mode}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History: {hist_path}")
    print(f"Best checkpoint: {best_path} (val loss {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
