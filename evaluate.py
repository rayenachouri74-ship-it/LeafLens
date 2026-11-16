"""LeafLens evaluation on the held-out test split.

Usage:
  python3 evaluate.py --checkpoint checkpoints/finetuned_best.pt
                      --mode finetuned [--gradcam-out results/gradcam.png]

Writes results/evaluation_<mode>.json and results/confusion_<mode>.csv
and prints accuracy, per-class precision/recall/F1, the confusion
matrix, accuracy by confidence band, and the uncertain-rule outcome.
The test split is never used for training or checkpoint selection.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import CLASSES, LeafDataset, load_split_csv
from model import build_model, grad_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default="finetuned")
    ap.add_argument("--split", default="results/split.csv")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--uncertain-threshold", type=float, default=0.60)
    ap.add_argument("--gradcam-out", default=None)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(2)
    splits = load_split_csv(args.split)
    test_items = splits["test"]
    ds = LeafDataset(test_items, args.image_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0)
    model, _ = build_model(mode=args.mode, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_true, all_pred, all_conf = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            all_true.extend(y.tolist())
            all_pred.extend(pred.tolist())
            all_conf.extend(conf.tolist())
    elapsed = time.time() - t0
    n = len(all_true)
    acc = sum(t == p for t, p in zip(all_true, all_pred)) / n

    # Per-class metrics + confusion matrix.
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for t, p in zip(all_true, all_pred):
        cm[t][p] += 1
    per_class = {}
    for i, cls in enumerate(CLASSES):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(CLASSES))) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[cls] = {"precision": round(prec, 4),
                          "recall": round(rec, 4),
                          "f1": round(f1, 4),
                          "support": sum(cm[i])}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(CLASSES)

    # Confidence bands.
    bands = [("0.90-1.00", 0.90, 1.01), ("0.75-0.90", 0.75, 0.90),
             ("0.60-0.75", 0.60, 0.75), ("<0.60", 0.0, 0.60)]
    band_rows = []
    for name, lo, hi in bands:
        idx = [i for i, c in enumerate(all_conf) if lo <= c < hi]
        if idx:
            a = sum(all_true[i] == all_pred[i] for i in idx) / len(idx)
            band_rows.append({"band": name, "n": len(idx),
                              "accuracy": round(a, 4)})
        else:
            band_rows.append({"band": name, "n": 0, "accuracy": None})

    # Uncertain rule: top probability below threshold -> uncertain,
    # not a forced class. Report how many test cases fall in it and
    # the accuracy on the remaining (confident) cases.
    thr = args.uncertain_threshold
    uncertain = [i for i, c in enumerate(all_conf) if c < thr]
    confident = [i for i in range(n) if i not in set(uncertain)]
    conf_acc = (sum(all_true[i] == all_pred[i] for i in confident)
                / len(confident)) if confident else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint, "mode": args.mode, "n_test": n,
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class": per_class, "confusion_matrix": cm,
        "confusion_labels": CLASSES,
        "confidence_bands": band_rows,
        "uncertain_threshold": thr,
        "n_uncertain": len(uncertain),
        "uncertain_fraction": round(len(uncertain) / n, 4),
        "accuracy_on_confident_only": (round(conf_acc, 4)
                                       if conf_acc is not None else None),
        "mean_inference_ms_per_image": round(elapsed / n * 1000, 2),
    }
    with open(out_dir / f"evaluation_{args.mode}.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / f"confusion_{args.mode}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + CLASSES)
        for i, cls in enumerate(CLASSES):
            w.writerow([cls] + cm[i])

    print(f"Test n={n} accuracy={acc:.4f} macro-F1={macro_f1:.4f}")
    for cls, v in per_class.items():
        print(f"  {cls}: P={v['precision']:.4f} R={v['recall']:.4f} "
              f"F1={v['f1']:.4f} support={v['support']}")
    print("Confusion matrix (rows=true, cols=predicted):")
    print("  " + " ".join(f"{c:>18}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>18} " + " ".join(f"{v:18d}" for v in cm[i]))
    print("Confidence bands:")
    for b in band_rows:
        print(f"  {b['band']}: n={b['n']} accuracy={b['accuracy']}")
    print(f"Uncertain (top prob < {thr}): {len(uncertain)}/{n} "
          f"({len(uncertain)/n:.1%}); accuracy on confident only: "
          f"{conf_acc}")

    if args.gradcam_out:
        # First test image whose true label is a disease class, else
        # the first test image.
        pick = next((i for i, (_, c) in enumerate(test_items)
                     if c != "healthy"), 0)
        x, y = ds[pick]
        pred = all_pred[pick]
        cam = grad_cam(model, x, pred, device=device)
        if cam is not None:
            from PIL import Image
            import numpy as np
            orig = Image.open(test_items[pick][0]).convert("RGB")
            orig = orig.resize((x.shape[2], x.shape[1]))
            heat = (cam * 255).astype("uint8")
            heat_img = Image.fromarray(heat).convert("RGB")
            blended = Image.blend(orig, heat_img, alpha=0.4)
            blended.save(args.gradcam_out)
            print(f"Grad-CAM example saved: {args.gradcam_out} "
                  f"(true={CLASSES[y]}, predicted={CLASSES[pred]}, "
                  f"conf={all_conf[pick]:.3f})")
        else:
            print("Grad-CAM failed; no heatmap saved (see model.py).")


if __name__ == "__main__":
    main()
