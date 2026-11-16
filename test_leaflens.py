"""LeafLens tests. Run: python3 test_leaflens.py

Checks, all against real files (no mocked model):
  1. Data loader: split counts match the published dataset totals.
  2. Model: a forward pass returns the correct shape (batch, 3).
  3. predict.py works on a held-out test image and prints a triage.
  4. Uncertain handling: a very high threshold forces UNCERTAIN.
  5. Bad image path fails cleanly with exit code 2, not a traceback.
Requires a trained checkpoint at checkpoints/finetuned_best.pt
(or checkpoints/frozen_best.pt as fallback for tests 3-4).
"""
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

from data import CLASSES, LeafDataset, discover_images, load_split_csv
from model import build_model

ROOT = Path(__file__).parent


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}: {name} {detail}")
    if not cond:
        raise SystemExit(1)


def main():
    # 1. Data counts.
    items = discover_images(ROOT / "data" / "olive")
    counts = Counter(c for _, c in items)
    check("dataset totals", len(items) == 3400
          and counts["healthy"] == 1050
          and counts["aculus_olearius"] == 890
          and counts["olive_peacock_spot"] == 1460, str(dict(counts)))

    splits = load_split_csv(ROOT / "results" / "split.csv")
    test_items = splits["test"]
    check("test split held out and non-empty",
          len(test_items) > 400
          and not (set(p for p, _ in test_items)
                   & set(p for p, _ in splits["train"])),
          f"test n={len(test_items)}")

    # 2. Model shape.
    model, _ = build_model(mode="frozen", pretrained=False)
    model.eval()
    ds = LeafDataset(test_items[:4], image_size=128)
    x = torch.stack([ds[i][0] for i in range(4)])
    with torch.no_grad():
        out = model(x)
    check("model output shape", tuple(out.shape) == (4, len(CLASSES)),
          str(tuple(out.shape)))

    # 3-5. predict.py, using the best available checkpoint.
    ckpt = None
    for cand in ("checkpoints/finetuned_best.pt",
                 "checkpoints/frozen_best.pt"):
        if (ROOT / cand).is_file():
            ckpt = cand
            break
    if ckpt is None:
        print("SKIP: predict.py tests (no checkpoint yet -- run train.py)")
        return
    test_img = test_items[0][0]
    r = subprocess.run([sys.executable, "predict.py", test_img,
                        "--checkpoint", ckpt],
                       cwd=ROOT, capture_output=True, text=True)
    check("predict.py on held-out image",
          r.returncode == 0 and "Triage:" in r.stdout
          and "NOT a definitive diagnosis" in r.stdout,
          f"exit={r.returncode}")
    r2 = subprocess.run([sys.executable, "predict.py", test_img,
                         "--checkpoint", ckpt, "--threshold", "1.01"],
                        cwd=ROOT, capture_output=True, text=True)
    check("uncertain handling at threshold 1.01",
          r2.returncode == 0 and "UNCERTAIN" in r2.stdout)
    r3 = subprocess.run([sys.executable, "predict.py",
                         "does/not/exist.jpg", "--checkpoint", ckpt],
                        cwd=ROOT, capture_output=True, text=True)
    check("bad image path fails cleanly",
          r3.returncode == 2 and "ERROR" in r3.stderr
          and "Traceback" not in r3.stderr, f"exit={r3.returncode}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
