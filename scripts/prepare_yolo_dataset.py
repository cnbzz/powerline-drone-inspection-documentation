"""Convert and merge InsPLAD-det, STN-PLAD and MPID into a unified 8-class YOLO dataset.

The three source datasets are READ-ONLY. Images are *copied* (never moved) into
``data/yolo/{train,val,test}/images`` and a matching YOLO ``.txt`` label is written
into ``data/yolo/{train,val,test}/labels``.

Unified 8-class schema::

    0 insulator_polymer  1 insulator_glass   2 insulator_porcelain  3 insulator_generic
    4 tower              5 spacer            6 damper               7 tower_id_plate

Usage::

    python scripts/prepare_yolo_dataset.py --dry-run      # count only, copy nothing
    python scripts/prepare_yolo_dataset.py --clean        # wipe data/yolo and rebuild
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Optional tqdm (not installed in this venv) -> graceful passthrough fallback. #
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    def tqdm(iterable: Iterable, **_kwargs):  # type: ignore
        return iterable


SEED = 42

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = PROJECT_ROOT / "data" / "external"
OUT_ROOT = PROJECT_ROOT / "data" / "yolo"
SANITY_DIR = PROJECT_ROOT / "outputs" / "sanity"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_FILE = LOG_DIR / "prepare_yolo.log"

INSPLAD_DIR = EXTERNAL / "insplad"
STN_DIR = EXTERNAL / "stn_plad"
MPID_DIR = EXTERNAL / "mpid"

SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------- #
# Unified 8-class schema                                                       #
# --------------------------------------------------------------------------- #
CLASS_NAMES: dict[int, str] = {
    0: "insulator_polymer",
    1: "insulator_glass",
    2: "insulator_porcelain",
    3: "insulator_generic",
    4: "tower",
    5: "spacer",
    6: "damper",
    7: "tower_id_plate",
}

# InsPLAD: map by lower-cased COCO category NAME. Names not present here are skipped.
INSPLAD_NAME_TO_ID: dict[str, int] = {
    "polymer insulator": 0,
    "polymer insulator lower shackle": 0,
    "polymer insulator upper shackle": 0,
    "polymer insulator tower shackle": 0,
    "glass insulator": 1,
    "glass insulator big shackle": 1,
    "glass insulator small shackle": 1,
    "glass insulator tower shackle": 1,
    "spacer": 5,
    "stockbridge damper": 6,
    "spiral damper": 6,
    "tower id plate": 7,
}
INSPLAD_SKIP: set[str] = {
    "yoke",
    "yoke suspension",
    "lightning rod shackle",
    "lightning rod suspension",
    "vari-grip",
    "sphere",
}

# STN-PLAD: map by COCO category ID.
STN_ID_TO_ID: dict[int, int] = {0: 4, 1: 3, 2: 5, 3: 6, 5: 7}

# MPID: each sub-dataset has class id 0 -> remap to the unified id.
MPID_SUBSETS: list[tuple[str, str, int]] = [
    # (folder name, output prefix, unified class id)
    ("composite_MPID", "mpid_comp_", 0),
    ("glass_MPID", "mpid_glass_", 1),
    ("porcelain_MPID", "mpid_porc_", 2),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------- #
# Stats container                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    images: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # class_counts[split][class_id] = n
    class_counts: dict[str, dict[int, int]] = field(
        default_factory=lambda: {s: defaultdict(int) for s in SPLITS}
    )
    dropped_annotations: int = 0
    missing_images: int = 0
    warnings: int = 0
    errors: int = 0
    # one produced sample per dataset for sanity rendering: dataset -> (img, lbl)
    samples: dict[str, tuple[Path, Path]] = field(default_factory=dict)

    def add_image(self, split: str) -> None:
        self.images[split] += 1

    def add_label(self, split: str, class_id: int) -> None:
        self.class_counts[split][class_id] += 1


STATS = Stats()
PREFIX_COUNTERS: dict[str, int] = defaultdict(int)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def next_name(prefix: str, ext: str = ".jpg") -> str:
    """Return a short, collision-free file stem like ``insplad_000123`` -> name+ext."""
    PREFIX_COUNTERS[prefix] += 1
    return f"{prefix}{PREFIX_COUNTERS[prefix]:06d}{ext}"


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int, ctx: str
) -> Optional[tuple[float, float, float, float]]:
    """Convert a COCO ``[x, y, w, h]`` (pixels) bbox to normalised YOLO ``cx cy w h``.

    Returns ``None`` (and logs a warning) when the box is degenerate or out of bounds.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        logger.warning(f"[{ctx}] dropping bbox with non-positive size: {bbox}")
        STATS.dropped_annotations += 1
        STATS.warnings += 1
        return None
    # 1px tolerance for boxes that touch / slightly exceed the border.
    if x < -1 or y < -1 or (x + w) > img_w + 1 or (y + h) > img_h + 1:
        logger.warning(
            f"[{ctx}] dropping out-of-bounds bbox {bbox} for image {img_w}x{img_h}"
        )
        STATS.dropped_annotations += 1
        STATS.warnings += 1
        return None

    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp tiny float overshoots into [0, 1].
    cx, cy, nw, nh = (min(max(v, 0.0), 1.0) for v in (cx, cy, nw, nh))
    if nw <= 0 or nh <= 0:
        STATS.dropped_annotations += 1
        STATS.warnings += 1
        return None
    return cx, cy, nw, nh


def write_label(path: Path, lines: list[str], dry_run: bool) -> None:
    if dry_run:
        return
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def copy_image(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    shutil.copy2(src, dst)


def record_sample(dataset: str, img: Path, lbl: Path) -> None:
    """Keep the first produced (image, label) pair per dataset for sanity rendering."""
    if dataset not in STATS.samples:
        STATS.samples[dataset] = (img, lbl)


# --------------------------------------------------------------------------- #
# InsPLAD-det (COCO)                                                           #
# --------------------------------------------------------------------------- #
def process_insplad(dry_run: bool) -> None:
    logger.info("=== InsPLAD-det ===")
    ann_train = INSPLAD_DIR / "annotations" / "instances_train.json"
    ann_val = INSPLAD_DIR / "annotations" / "instances_val.json"
    img_train = INSPLAD_DIR / "train"
    img_val = INSPLAD_DIR / "val"
    for p in (ann_train, ann_val, img_train, img_val):
        if not p.exists():
            raise FileNotFoundError(f"InsPLAD path missing: {p}")

    rng = random.Random(SEED)

    # --- train json -> 90% train / 10% test --------------------------------
    train_imgs, train_cat = _load_coco(ann_train)
    keys = list(train_imgs.keys())
    rng.shuffle(keys)
    n_test = round(len(keys) * 0.10)
    test_ids = set(keys[:n_test])
    split_of = {k: ("test" if k in test_ids else "train") for k in keys}
    _emit_coco(
        train_imgs, train_cat, img_train, split_of,
        prefix="insplad_", name_to_id=INSPLAD_NAME_TO_ID, skip=INSPLAD_SKIP,
        id_to_id=None, dataset="insplad", dry_run=dry_run,
    )

    # --- val json -> stays val ---------------------------------------------
    val_imgs, val_cat = _load_coco(ann_val)
    split_of = {k: "val" for k in val_imgs}
    _emit_coco(
        val_imgs, val_cat, img_val, split_of,
        prefix="insplad_", name_to_id=INSPLAD_NAME_TO_ID, skip=INSPLAD_SKIP,
        id_to_id=None, dataset="insplad", dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# STN-PLAD (COCO)                                                              #
# --------------------------------------------------------------------------- #
def process_stn(dry_run: bool) -> None:
    logger.info("=== STN-PLAD ===")
    ann = STN_DIR / "annotations.json"
    if not ann.exists():
        raise FileNotFoundError(f"STN-PLAD annotations missing: {ann}")

    imgs, cats = _load_coco(ann)
    rng = random.Random(SEED)
    keys = list(imgs.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_train = round(n * 0.80)
    n_val = round(n * 0.10)
    split_of: dict[str, str] = {}
    for i, key in enumerate(keys):
        if i < n_train:
            split_of[key] = "train"
        elif i < n_train + n_val:
            split_of[key] = "val"
        else:
            split_of[key] = "test"
    _emit_coco(
        imgs, cats, STN_DIR, split_of,
        prefix="stnplad_", name_to_id=None, skip=set(),
        id_to_id=STN_ID_TO_ID, dataset="stnplad", dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Shared COCO loading / emission                                               #
# --------------------------------------------------------------------------- #
def _load_coco(path: Path) -> tuple[dict[str, dict], dict[int, str]]:
    """Return (file_name -> image dict incl. merged 'annotations', category_id -> name).

    Images are keyed by ``file_name`` (not ``image_id``): InsPLAD-det contains a few
    physical images referenced by multiple ``image_id``s. Keying by file name merges
    their annotations into a single output image, which also prevents the same image
    from leaking across train/test splits.
    """
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"{path} is not valid COCO JSON (missing '{key}')")

    cats = {c["id"]: c["name"] for c in data["categories"]}

    # image_id -> file_name, and file_name -> merged image record
    id_to_name: dict[int, str] = {}
    by_name: dict[str, dict] = {}
    dup = 0
    for img in data["images"]:
        fname = img["file_name"]
        id_to_name[img["id"]] = fname
        if fname in by_name:
            dup += 1
        else:
            by_name[fname] = {**img, "annotations": []}
    if dup:
        logger.warning(f"{path.name}: {dup} duplicate file_name records merged")
        STATS.warnings += dup

    orphans = 0
    for ann in data["annotations"]:
        fname = id_to_name.get(ann["image_id"])
        if fname is None:
            orphans += 1
            continue
        by_name[fname]["annotations"].append(ann)
    if orphans:
        logger.warning(f"{path.name}: {orphans} annotations had no matching image_id (skipped)")
        STATS.warnings += orphans
    return by_name, cats


def _emit_coco(
    images: dict[str, dict],
    cats: dict[int, str],
    img_root: Path,
    split_of: dict[str, str],
    *,
    prefix: str,
    name_to_id: Optional[dict[str, int]],
    skip: set[str],
    id_to_id: Optional[dict[int, int]],
    dataset: str,
    dry_run: bool,
) -> None:
    for fname, img in tqdm(images.items(), desc=f"{dataset}", unit="img"):
        split = split_of[fname]
        src = img_root / img["file_name"]
        if not src.exists():
            logger.warning(f"[{dataset}] image file not found, skipping: {src}")
            STATS.missing_images += 1
            STATS.warnings += 1
            continue

        w, h = int(img["width"]), int(img["height"])
        lines: list[str] = []
        for ann in img["annotations"]:
            cat_id = ann["category_id"]
            if name_to_id is not None:
                name = cats.get(cat_id, "").strip().lower()
                if name in skip:
                    continue
                if name not in name_to_id:
                    logger.warning(f"[{dataset}] unmapped category '{name}' (skipped)")
                    STATS.dropped_annotations += 1
                    STATS.warnings += 1
                    continue
                unified = name_to_id[name]
            else:
                assert id_to_id is not None
                if cat_id not in id_to_id:
                    logger.warning(f"[{dataset}] unmapped category id {cat_id} (skipped)")
                    STATS.dropped_annotations += 1
                    STATS.warnings += 1
                    continue
                unified = id_to_id[cat_id]

            yolo = coco_bbox_to_yolo(ann["bbox"], w, h, ctx=f"{dataset}:{img['file_name']}")
            if yolo is None:
                continue
            cx, cy, nw, nh = yolo
            lines.append(f"{unified} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            STATS.add_label(split, unified)

        out_name = next_name(prefix)
        out_img = OUT_ROOT / split / "images" / out_name
        out_lbl = OUT_ROOT / split / "labels" / (Path(out_name).stem + ".txt")
        copy_image(src, out_img, dry_run)
        write_label(out_lbl, lines, dry_run)
        STATS.add_image(split)
        record_sample(dataset, out_img, out_lbl)


# --------------------------------------------------------------------------- #
# MPID (YOLO txt, class id always 0 -> remap)                                  #
# --------------------------------------------------------------------------- #
def process_mpid(dry_run: bool) -> None:
    logger.info("=== MPID ===")
    for folder, prefix, unified in MPID_SUBSETS:
        img_dir = MPID_DIR / folder / "train" / "images"
        lbl_dir = MPID_DIR / folder / "train" / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"MPID subset missing: {img_dir} / {lbl_dir}")

        images = sorted(p for p in img_dir.iterdir() if p.suffix in IMG_EXTS)
        rng = random.Random(SEED)
        rng.shuffle(images)
        n = len(images)
        n_train = round(n * 0.80)
        n_val = round(n * 0.10)

        for i, src in enumerate(tqdm(images, desc=folder, unit="img")):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            src_lbl = lbl_dir / (src.stem + ".txt")
            lines: list[str] = []
            if src_lbl.exists():
                for raw in src_lbl.read_text(encoding="utf-8").splitlines():
                    raw = raw.strip()
                    if not raw:
                        continue
                    parts = raw.split()
                    if len(parts) != 5:
                        logger.warning(f"[mpid:{folder}] malformed label line in {src_lbl.name}: '{raw}'")
                        STATS.dropped_annotations += 1
                        STATS.warnings += 1
                        continue
                    # remap class id (always 0 in MPID) -> unified; keep coords as-is
                    _, cx, cy, nw, nh = parts
                    lines.append(f"{unified} {cx} {cy} {nw} {nh}")
                    STATS.add_label(split, unified)
            else:
                logger.warning(f"[mpid:{folder}] no label for {src.name} (empty label written)")
                STATS.warnings += 1

            out_name = next_name(prefix, ext=src.suffix.lower())
            out_img = OUT_ROOT / split / "images" / out_name
            out_lbl = OUT_ROOT / split / "labels" / (Path(out_name).stem + ".txt")
            copy_image(src, out_img, dry_run)
            write_label(out_lbl, lines, dry_run)
            STATS.add_image(split)
            record_sample(f"mpid_{folder}", out_img, out_lbl)


# --------------------------------------------------------------------------- #
# data.yaml                                                                    #
# --------------------------------------------------------------------------- #
def write_data_yaml(dry_run: bool) -> Path:
    path = OUT_ROOT / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in CLASS_NAMES.items())
    content = (
        "# Auto-generated by scripts/prepare_yolo_dataset.py\n"
        "path: ../data/yolo\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 8\n"
        "names:\n"
        f"{names_block}\n"
    )
    if not dry_run:
        path.write_text(content, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Sanity check (draw bboxes)                                                   #
# --------------------------------------------------------------------------- #
def render_sanity(dry_run: bool) -> list[Path]:
    if dry_run:
        return []
    SANITY_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = random.Random(SEED)

    palette = [
        (255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 165, 0),
        (200, 0, 200), (0, 200, 200), (255, 0, 120), (120, 120, 0),
    ]
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    for dataset, (img_path, lbl_path) in STATS.samples.items():
        if not img_path.exists():
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"sanity: cannot open {img_path}: {exc}")
            continue
        draw = ImageDraw.Draw(im)
        W, H = im.size
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cid, cx, cy, nw, nh = line.split()
                cid = int(cid)
                cx, cy, nw, nh = float(cx), float(cy), float(nw), float(nh)
                x1 = (cx - nw / 2) * W
                y1 = (cy - nh / 2) * H
                x2 = (cx + nw / 2) * W
                y2 = (cy + nh / 2) * H
                color = palette[cid % len(palette)]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = CLASS_NAMES.get(cid, str(cid))
                draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, font=font)
        out = SANITY_DIR / f"sanity_{dataset}.jpg"
        im.convert("RGB").save(out, "JPEG", quality=90)
        saved.append(out)
        # use rng so future variation is reproducible if logic changes
        _ = rng.random()
    return saved


# --------------------------------------------------------------------------- #
# Cross-check + report                                                         #
# --------------------------------------------------------------------------- #
def verify_pairs() -> tuple[int, int]:
    """Confirm every image has a matching label (and vice versa). Returns (ok, mismatch)."""
    ok = mismatch = 0
    for split in SPLITS:
        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        img_stems = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.is_file()}
        only_img = img_stems - lbl_stems
        only_lbl = lbl_stems - img_stems
        ok += len(img_stems & lbl_stems)
        if only_img or only_lbl:
            mismatch += len(only_img) + len(only_lbl)
            logger.error(
                f"[{split}] pairing mismatch: {len(only_img)} images w/o label, "
                f"{len(only_lbl)} labels w/o image"
            )
            STATS.errors += 1
    return ok, mismatch


def print_report(yaml_path: Path, sanity: list[Path], dry_run: bool) -> None:
    total_train = STATS.images["train"]
    total_val = STATS.images["val"]
    total_test = STATS.images["test"]

    lines = []
    lines.append("=" * 60)
    lines.append("YOLO DATASET HAZIRLAMA RAPORU" + ("  [DRY-RUN]" if dry_run else ""))
    lines.append("=" * 60)
    lines.append(f"Toplam train görüntü: {total_train}")
    lines.append(f"Toplam val görüntü:   {total_val}")
    lines.append(f"Toplam test görüntü:  {total_test}")
    lines.append(f"Toplam görüntü:       {total_train + total_val + total_test}")
    lines.append("")
    lines.append("Sınıf bazında dağılım:")
    lines.append(f"{'id':<3} {'sınıf_adı':<20} {'train':>7} {'val':>6} {'test':>6} {'toplam':>7}")
    for cid in sorted(CLASS_NAMES):
        tr = STATS.class_counts["train"][cid]
        va = STATS.class_counts["val"][cid]
        te = STATS.class_counts["test"][cid]
        lines.append(
            f"{cid:<3} {CLASS_NAMES[cid]:<20} {tr:>7} {va:>6} {te:>6} {tr + va + te:>7}"
        )
    tot_lbl = sum(
        STATS.class_counts[s][c] for s in SPLITS for c in CLASS_NAMES
    )
    lines.append(f"{'':<3} {'TOPLAM':<20} "
                 f"{sum(STATS.class_counts['train'].values()):>7} "
                 f"{sum(STATS.class_counts['val'].values()):>6} "
                 f"{sum(STATS.class_counts['test'].values()):>6} {tot_lbl:>7}")
    lines.append("")
    lines.append(f"Atılan annotation: {STATS.dropped_annotations} "
                 f"(bozuk bbox / kategori eşleşmemesi)")
    lines.append(f"Bulunamayan görüntü: {STATS.missing_images}")
    lines.append(f"Hata: {STATS.errors} / Uyarı: {STATS.warnings}")
    lines.append("")
    lines.append(f"data.yaml: {yaml_path.relative_to(PROJECT_ROOT)} "
                 f"{'(dry-run, not written)' if dry_run else '✓'}")
    if sanity:
        lines.append("Sanity images:")
        for s in sanity:
            lines.append(f"  - {s.relative_to(PROJECT_ROOT)} ✓")
    elif dry_run:
        lines.append("Sanity images: (skipped in dry-run)")
    lines.append("=" * 60)
    report = "\n".join(lines)
    print("\n" + report)
    logger.info("\n" + report)


# --------------------------------------------------------------------------- #
# Scaffolding                                                                  #
# --------------------------------------------------------------------------- #
def prepare_dirs(clean: bool, dry_run: bool) -> None:
    if clean and OUT_ROOT.exists() and not dry_run:
        logger.info(f"--clean: removing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    if dry_run:
        return
    for split in SPLITS:
        (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    # Windows consoles default to cp1252 which cannot encode Turkish glyphs / ✓.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")
    logger.add(LOG_FILE, level="DEBUG", mode="w", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified 8-class YOLO dataset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only; do not copy images or write labels.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove data/yolo and rebuild from scratch.")
    args = parser.parse_args()

    setup_logging()
    random.seed(SEED)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"dry_run={args.dry_run}  clean={args.clean}  seed={SEED}")

    if not EXTERNAL.exists():
        logger.error(f"data/external not found: {EXTERNAL}")
        sys.exit(1)

    prepare_dirs(args.clean, args.dry_run)

    try:
        process_insplad(args.dry_run)
        process_stn(args.dry_run)
        process_mpid(args.dry_run)
    except (FileNotFoundError, ValueError) as exc:
        STATS.errors += 1
        logger.error(f"FATAL: {exc}")
        sys.exit(1)

    yaml_path = write_data_yaml(args.dry_run)

    if not args.dry_run:
        ok, mismatch = verify_pairs()
        logger.info(f"image/label pairing: {ok} matched, {mismatch} mismatched")

    sanity = render_sanity(args.dry_run)
    print_report(yaml_path, sanity, args.dry_run)
    logger.info(f"Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
