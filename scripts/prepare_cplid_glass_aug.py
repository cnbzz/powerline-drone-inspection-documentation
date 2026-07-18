"""Stage 7.3-A2 — Build a CPLID-augmented glass-insulator training set.

Crops CPLID insulators using their VOC ground-truth bboxes (both Normal AND
Defective have per-image bbox XMLs), splits them deterministically into
train/val/test (stratified per class, no leakage), and assembles an augmented
data_root that mirrors the InsPLAD `organized/` layout but ONLY for the
glass-insulator equipment:

  organized_cplid_aug/
    train/glass-insulator/{good,missing-cap}/  = InsPLAD-train (copy) + CPLID-train crops
    val/glass-insulator/{good,missing-cap}/    = InsPLAD-val   (copy) + CPLID-val   crops
    test/glass-insulator/{good,missing-cap}/   = InsPLAD-test  (copy)  [UNCHANGED — comparable to baseline]

CPLID-test crops are NEVER added to training. A manifest records which CPLID
image went to which split so the A3 cross-dataset eval can flag leakage.

Class mapping (proven in A1c):
  Normal_Insulators    -> good
  Defective_Insulators -> missing-cap

Crop source: VOC bbox <object><bndbox> in
  Normal:    Defective? -> labels/<id>.xml
  Defective: labels/insulator/<id>.xml

Gotchas honored: PYTHONUTF8, PIL (Unicode paths), deterministic seed.
"""
from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
EQUIPMENT = "glass-insulator"
CPLID_DIR = _ROOT / "data" / "external" / "cplid"
INSPLAD_ROOT = _ROOT / "data" / "insplad_fault" / "organized"
AUG_ROOT = _ROOT / "data" / "insplad_fault" / "organized_cplid_aug"
MANIFEST = _ROOT / "outputs" / "cross_dataset_eval" / "cplid_split_manifest.json"

# Deterministic stratified split ratios
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remainder

# folder -> (class, label_xml_resolver)
SOURCES = {
    "Normal_Insulators": "good",
    "Defective_Insulators": "missing-cap",
}


def _voc_bbox(xml_path: Path) -> tuple[int, int, int, int] | None:
    """Return (xmin, ymin, xmax, ymax) of the first <object> bbox, or None."""
    try:
        root = ET.parse(xml_path).getroot()
        obj = root.find("object")
        if obj is None:
            return None
        b = obj.find("bndbox")
        if b is None:
            return None
        return (
            int(float(b.findtext("xmin"))),
            int(float(b.findtext("ymin"))),
            int(float(b.findtext("xmax"))),
            int(float(b.findtext("ymax"))),
        )
    except Exception:
        return None


def _label_xml(folder: str, stem: str) -> Path:
    if folder == "Normal_Insulators":
        return CPLID_DIR / folder / "labels" / f"{stem}.xml"
    # Defective: bbox lives under labels/insulator/<id>.xml
    return CPLID_DIR / folder / "labels" / "insulator" / f"{stem}.xml"


def _collect_cropped() -> dict[str, list[tuple[str, Path, tuple]]]:
    """Return {cls: [(stem, image_path, bbox), ...]} for every image with a valid bbox."""
    out: dict[str, list] = {"good": [], "missing-cap": []}
    missing_bbox: list[str] = []
    for folder, cls in SOURCES.items():
        img_dir = CPLID_DIR / folder / "images"
        for img_path in sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            bbox = _voc_bbox(_label_xml(folder, stem))
            if bbox is None:
                missing_bbox.append(f"{folder}/{stem}")
                continue
            out[cls].append((f"{folder}__{stem}", img_path, bbox))
    if missing_bbox:
        print(f"  ! {len(missing_bbox)} CPLID images had no valid bbox (skipped): "
              f"{missing_bbox[:5]}{'...' if len(missing_bbox) > 5 else ''}")
    return out


def _split(items: list, rng: np.random.Generator) -> dict[str, list]:
    """Deterministic stratified split of one class's items."""
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n = len(items)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    tr = [items[i] for i in idx[:n_train]]
    va = [items[i] for i in idx[n_train:n_train + n_val]]
    te = [items[i] for i in idx[n_train + n_val:]]
    return {"train": tr, "val": va, "test": te}


def _copytree_class(src: Path, dst: Path) -> int:
    """Copy all image files from InsPLAD class dir into dst. Returns count."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for ext in ("*.jpg", "*.png"):
        for p in src.glob(ext):
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


def _save_crop(image_path: Path, bbox: tuple, out_path: Path) -> bool:
    try:
        with open(image_path, "rb") as f:  # Unicode-safe
            img = Image.open(f).convert("RGB")
        x1, y1, x2, y2 = bbox
        # clamp to image bounds
        w, h = img.size
        x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
        crop = img.crop((x1, y1, x2, y2))
        crop.save(out_path, quality=95)
        return True
    except Exception as e:
        print(f"  ! crop failed {image_path.name}: {e}")
        return False


def main() -> None:
    print("=" * 72)
    print("CPLID-AUGMENTED glass-insulator dataset (Stage 7.3-A2)")
    print("=" * 72)
    rng = np.random.default_rng(SEED)

    # 1. Collect + crop CPLID
    collected = _collect_cropped()
    for cls, items in collected.items():
        print(f"  CPLID {cls:12s}: {len(items)} images with valid bbox")

    # 2. Deterministic stratified split per class
    splits = {cls: _split(items, rng) for cls, items in collected.items()}

    # 3. Fresh augmented root
    if AUG_ROOT.exists():
        shutil.rmtree(AUG_ROOT)

    manifest: dict = {"seed": SEED, "equipment": EQUIPMENT,
                      "class_mapping": SOURCES,
                      "ratios": {"train": TRAIN_FRAC, "val": VAL_FRAC,
                                 "test": round(1 - TRAIN_FRAC - VAL_FRAC, 4)},
                      "splits": {}, "counts": {}}

    # 4a. Copy InsPLAD glass into aug root (all 3 splits)
    print("-" * 72)
    counts: dict = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for cls in ("good", "missing-cap"):
            src = INSPLAD_ROOT / split / EQUIPMENT / cls
            dst = AUG_ROOT / split / EQUIPMENT / cls
            n_ins = _copytree_class(src, dst) if src.exists() else 0
            counts[split][cls] = {"insplad": n_ins, "cplid": 0}

    # 4b. Add CPLID crops to TRAIN and VAL only (test held out of training set)
    for cls in ("good", "missing-cap"):
        for split in ("train", "val", "test"):
            items = splits[cls][split]
            manifest["splits"].setdefault(split, {}).setdefault(cls, [])
            for stem, img_path, bbox in items:
                manifest["splits"][split][cls].append(stem)
                if split == "test":
                    continue  # NEVER add CPLID-test to the training data
                dst = AUG_ROOT / split / EQUIPMENT / cls / f"cplid_{stem}.jpg"
                if _save_crop(img_path, bbox, dst):
                    counts[split][cls]["cplid"] += 1

    manifest["counts"] = counts

    # 5. Report
    print("Augmented glass-insulator dataset composition:")
    for split in ("train", "val", "test"):
        for cls in ("good", "missing-cap"):
            c = counts[split][cls]
            print(f"  {split:5s} {cls:12s}: InsPLAD={c['insplad']:4d} + CPLID={c['cplid']:4d} "
                  f"= {c['insplad'] + c['cplid']:4d}")
    print("-" * 72)
    n_cplid_test = sum(len(splits[cls]['test']) for cls in ('good', 'missing-cap'))
    print(f"CPLID held-out TEST (NOT in training, for honest A3): {n_cplid_test} "
          f"(good={len(splits['good']['test'])}, missing-cap={len(splits['missing-cap']['test'])})")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {MANIFEST}")
    print(f"Aug root: {AUG_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
