"""
Leak-free train / val / test split for the rice leaf disease dataset.

Why this script exists
----------------------
The original notebook split a dataset that was reshuffled on every pass
(`image_dataset_from_directory(shuffle=True)` + `ds.shuffle(...)`), so
`take()` / `skip()` returned different images each epoch and test images
leaked into training.

The Kaggle dataset also appears to contain augmented copies of a smaller set
of original photos (flipped / rotated versions of the same leaf). If one copy
lands in train and another in test, the test score measures memorisation,
not generalisation.

This script:
  1. Computes a perceptual hash (pHash) for every image, under all 8
     flips/90-degree rotations, so mirrored or rotated copies are detected.
  2. Groups near-duplicate images (Hamming distance <= --threshold) with
     union-find, within each class.
  3. Splits *groups* (not images) into train / val / test, stratified by
     class, so all copies of a leaf stay in the same split.
  4. Copies the files into  <dst>/{train,val,test}/<class>/  and writes
     <dst>/split_report.csv listing every file, its group and its split.

Limitation: pHash catches flips, 90-degree rotations, resizing and mild
colour changes, but not heavy crops or arbitrary-angle rotations. Check a
few groups visually and tune --threshold if needed.

Usage
-----
    python scripts/split_dataset.py \
        --src "path/to/rice leaf diseases dataset" \
        --dst data/split
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dihedral_variants(img):
    """The 8 flip/rotation variants of an image."""
    variants = []
    for base in (img, ImageOps.mirror(img)):
        for angle in (0, 90, 180, 270):
            variants.append(base.rotate(angle, expand=True))
    return variants


def hash_bits(img, hash_size):
    """pHash of each dihedral variant as a (8, hash_size**2) bool array."""
    return np.stack(
        [imagehash.phash(v, hash_size=hash_size).hash.flatten() for v in dihedral_variants(img)]
    )


def group_near_duplicates(bits, threshold):
    """Union-find over images whose min dihedral Hamming distance <= threshold."""
    n = len(bits)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        ref = bits[:, 0, :].astype(np.int16)  # original orientation, (n, b)
        dist = np.full((n, n), np.iinfo(np.int16).max, dtype=np.int16)
        for k in range(bits.shape[1]):
            var = bits[:, k, :].astype(np.int16)
            # Hamming distance = count of differing bits
            d = ref @ (1 - var).T + (1 - ref) @ var.T
            np.minimum(dist, d.astype(np.int16), out=dist)
        ii, jj = np.where(np.triu(dist <= threshold, k=1))
        for i, j in zip(ii, jj):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def split_groups(groups, val_frac, test_frac, rng):
    """Assign whole groups to splits, aiming for the target image counts."""
    groups = groups[:]
    rng.shuffle(groups)
    total = sum(len(g) for g in groups)
    targets = {"test": test_frac * total, "val": val_frac * total}
    filled = {"test": 0, "val": 0}
    assignment = {}
    for g in groups:
        for split in ("test", "val"):
            if filled[split] + len(g) / 2 <= targets[split]:
                filled[split] += len(g)
                assignment[id(g)] = split
                break
        else:
            assignment[id(g)] = "train"
    return [(g, assignment[id(g)]) for g in groups]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Folder with one sub-folder per class")
    p.add_argument("--dst", default="data/split", help="Output folder")
    p.add_argument("--val", type=float, default=0.1, help="Validation fraction")
    p.add_argument("--test", type=float, default=0.1, help="Test fraction")
    p.add_argument("--threshold", type=int, default=8, help="Max Hamming distance (of 64 bits) to call two images duplicates")
    p.add_argument("--hash-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} is not empty - delete it or choose another --dst")

    rng = random.Random(args.seed)
    rows = []
    classes = sorted(d.name for d in src.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {src}")

    print(f"{'class':<20}{'images':>8}{'groups':>8}{'train':>8}{'val':>8}{'test':>8}")
    for cls in classes:
        files = sorted(f for f in (src / cls).iterdir() if f.suffix.lower() in IMG_EXTS)
        bits = []
        for f in files:
            with Image.open(f) as im:
                bits.append(hash_bits(im.convert("RGB"), args.hash_size))
        bits = np.stack(bits)
        groups = group_near_duplicates(bits, args.threshold)

        counts = defaultdict(int)
        for gid, (members, split) in enumerate(split_groups(groups, args.val, args.test, rng)):
            for i in members:
                out = dst / split / cls / files[i].name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(files[i], out)
                counts[split] += 1
                rows.append({"file": str(files[i].relative_to(src)), "class": cls,
                             "group": f"{cls}_{gid}", "group_size": len(members), "split": split})
        print(f"{cls:<20}{len(files):>8}{len(groups):>8}{counts['train']:>8}{counts['val']:>8}{counts['test']:>8}")

    with open(dst / "split_report.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote split to {dst} and report to {dst / 'split_report.csv'}")
    print("Many images per group means the dataset is mostly augmented copies of fewer originals.")


if __name__ == "__main__":
    main()
