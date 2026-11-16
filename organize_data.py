"""Organise the raw Denizli olive archive into canonical class folders.

Source: https://github.com/sinanuguz/CNN_olive_dataset (cloned into
data/olive_github; images shipped as 5 .rar files, extracted to
data/olive_extracted).

Archive layout and the mapping used here (documented, not hidden):
  Train1.rar -> Healthy/                     = 830 images
  Train2.rar -> aculus_olearius/             = 690 images
  test.rar   -> test/Healthy (220), test/aculus_olearius (200),
                test/olive_peacock_spot (260)
  train3.rar + train4.rar -> train3/ (423) + train4/ (777) = 1200 images

train3/train4 carry NO class folder in the archive. They are assigned
to olive_peacock_spot by elimination + exact arithmetic, and the
assignment was spot-checked visually (a train3 leaf shows the dark
peacock spots; a Healthy leaf does not):
  - they are the only class left after Healthy and Aculus are accounted for
  - 423 + 777 = 1200, exactly the published peacock training count
  - final totals then match the published dataset exactly:
    healthy 1050, aculus_olearius 890, olive_peacock_spot 1460, total 3400
This script FAILS unless those exact counts are reproduced, so a
different archive version cannot silently slip through.

Output: data/olive/<canonical_class>/ containing symlinks to the
extracted files (no 105 MB duplicate copy).
"""
import sys
from pathlib import Path

SRC = Path("data/olive_extracted")
DST = Path("data/olive")
MAPPING = [
    # (extracted path relative to data/olive_extracted, canonical class)
    ("Healthy", "healthy"),
    ("test/Healthy", "healthy"),
    ("aculus_olearius", "aculus_olearius"),
    ("test/aculus_olearius", "aculus_olearius"),
    ("test/olive_peacock_spot", "olive_peacock_spot"),
    ("train3", "olive_peacock_spot"),
    ("train4", "olive_peacock_spot"),
]
EXPECTED = {"healthy": 1050, "aculus_olearius": 890,
            "olive_peacock_spot": 1460}


def main():
    if not SRC.is_dir():
        sys.exit(f"Extracted archive not found: {SRC} "
                 "(extract data/olive_github/*.rar there first)")
    DST.mkdir(parents=True, exist_ok=True)
    # Clear old links first: train and test folders reuse filenames
    # (e.g. A163.jpg in both), so plain symlinks would overwrite each
    # other and silently lose ~100 healthy images.
    for old in DST.rglob("*"):
        if old.is_symlink():
            old.unlink()
    counts = {}
    for src_name, canonical in MAPPING:
        src_dir = SRC / src_name
        if not src_dir.is_dir():
            sys.exit(f"Missing source folder: {src_dir}")
        dst_dir = DST / canonical
        dst_dir.mkdir(parents=True, exist_ok=True)
        tag = src_name.replace("/", "_")
        for img in sorted(src_dir.glob("*.jpg")) + sorted(
                src_dir.glob("*.JPG")):
            # Unique link name per source folder, so colliding
            # filenames across train/test cannot overwrite each other.
            link = dst_dir / f"{tag}__{img.name}"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(img.resolve())
            counts[canonical] = counts.get(canonical, 0) + 1
    print(f"Counts: {counts} total={sum(counts.values())}")
    if counts != EXPECTED:
        sys.exit(f"COUNT MISMATCH: got {counts}, expected {EXPECTED}. "
                 "Do not train on a mislabelled dataset.")
    print("Counts match the published dataset exactly. OK.")


if __name__ == "__main__":
    main()
