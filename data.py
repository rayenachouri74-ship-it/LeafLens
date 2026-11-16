"""LeafLens data handling.

Discovers image folders under a root directory, maps folder names to
canonical olive classes, and makes a stratified train/val/test split
with a fixed seed. The test split is written once and never used for
model selection except for the final evaluation in evaluate.py.
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Canonical classes, in a fixed order used everywhere (model head,
# confusion matrix, predict.py). Folder names from the Denizli set
# are normalised to these.
CLASSES = ["healthy", "olive_peacock_spot", "aculus_olearius"]

ALIASES = {
    "healthy": "healthy",
    "olive_peacock_spot": "olive_peacock_spot",
    "peacock_spot": "olive_peacock_spot",
    "olive_peack_spot": "olive_peacock_spot",
    "aculus_olearius": "aculus_olearius",
    "olive_bud_mite": "aculus_olearius",
    "aculus": "aculus_olearius",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root):
    """Return list of (path, canonical_class) for all images under root.

    Walks the tree, so both flat class folders and train/test nested
    layouts work. A file's class is its nearest ancestor folder whose
    name maps to a canonical class.
    """
    root = Path(root)
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        cls = None
        for parent in p.parents:
            if parent == root:
                break
            key = parent.name.strip().lower().replace(" ", "_")
            if key in ALIASES:
                cls = ALIASES[key]
                break
        if cls is not None:
            out.append((str(p), cls))
    return out


def make_split(items, seed=42, train_frac=0.70, val_frac=0.15):
    """Stratified split. Returns dict split -> list of (path, class)."""
    by_class = defaultdict(list)
    for path, cls in items:
        by_class[cls].append((path, cls))
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls in CLASSES:
        rows = sorted(by_class.get(cls, []))
        rng.shuffle(rows)
        n = len(rows)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:])
    return splits


def write_split_csv(splits, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for split in ("train", "val", "test"):
            for path, cls in sorted(splits[split]):
                w.writerow([path, cls, split])


def load_split_csv(csv_path):
    splits = {"train": [], "val": [], "test": []}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            splits[row["split"]].append((row["path"], row["label"]))
    return splits


def class_weights(train_items):
    """Inverse-frequency weights, normalised to mean 1. Used for the
    weighted CrossEntropyLoss so the minority class is not ignored."""
    counts = Counter(cls for _, cls in train_items)
    total = sum(counts.values())
    w = []
    for cls in CLASSES:
        c = counts.get(cls, 0)
        w.append(total / (len(CLASSES) * c) if c else 0.0)
    return torch.tensor(w, dtype=torch.float32), counts


def get_transforms(image_size=128, augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class LeafDataset(Dataset):
    def __init__(self, items, image_size=128, augment=False):
        self.items = list(items)
        self.tf = get_transforms(image_size, augment=augment)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        p = Path(path)
        if not p.exists():
            # split.csv stores paths relative to the project folder;
            # resolve them against this file's folder if needed.
            p = Path(__file__).parent / path
        img = Image.open(p).convert("RGB")
        return self.tf(img), CLASSES.index(label)
