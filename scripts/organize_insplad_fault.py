"""InsPLAD-fault (supervised) → organized train/val/test (Aşama 6.1.1).

Gerçek dataset yapısı: data/insplad_fault/raw/defect_supervised/<ekipman>/{train,val}/<sınıf>/
Yazar zaten train/val'a bölmüş ama TEST yok; val gerçek dünya dağılımını yansıtıyor.

Karar (kullanıcı onayı, 6.1.1):
  - Yazar `val/`  → organized/test/   (resmi benchmark, gerçekçi dağılım korunur)
  - Yazar `train/`→ organized/train + organized/val  (sınıf-bazında 85/15, seed 42)
  - Sınıf adı birleşik şema: normal→good, corrosão→rust (her ekipman kendi setini korur)

Çalıştır:  PYTHONUTF8=1 uv run python scripts/organize_insplad_fault.py
"""
from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.85  # yazar train'inden train payı; kalan 0.15 → val
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Sınıf adı uyumlandırma (birleşik şema)
CLASS_MAP = {"normal": "good", "corrosão": "rust"}

RAW = Path("data/insplad_fault/raw/defect_supervised")
ORGANIZED = Path("data/insplad_fault/organized")


def norm_class(name: str) -> str:
    return CLASS_MAP.get(name, name)


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)


def copy_into(images: list[Path], dest_dir: Path, used: set[str]) -> int:
    """Kopyala; ad çakışması olursa kaynak sınıf adıyla öneklendir (merge güvenliği)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in images:
        name = img.name
        if name in used:  # farklı kaynak sınıftan aynı ad → benzersizleştir
            name = f"{img.parent.name}__{name}"
        used.add(name)
        shutil.copy2(img, dest_dir / name)
        n += 1
    return n


def main() -> None:
    random.seed(SEED)

    if ORGANIZED.exists():
        shutil.rmtree(ORGANIZED)

    # stats[equipment][split][class] = count
    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    equipments = sorted(d for d in RAW.iterdir() if d.is_dir())
    for eq in equipments:
        eq_name = eq.name

        # --- TEST = yazar val/ (sınıf adı harmonize) ---
        author_val = eq / "val"
        if author_val.is_dir():
            for cls_dir in sorted(c for c in author_val.iterdir() if c.is_dir()):
                tgt = norm_class(cls_dir.name)
                used: set[str] = set(
                    p.name for p in (ORGANIZED / "test" / eq_name / tgt).glob("*")
                ) if (ORGANIZED / "test" / eq_name / tgt).exists() else set()
                n = copy_into(list_images(cls_dir), ORGANIZED / "test" / eq_name / tgt, used)
                stats[eq_name]["test"][tgt] += n

        # --- TRAIN + VAL = yazar train/ , sınıf-bazında 85/15 ---
        author_train = eq / "train"
        if author_train.is_dir():
            # Önce harmonize sınıfa göre görüntüleri topla (merge olabilir: corrosão+rust)
            per_class: dict[str, list[Path]] = defaultdict(list)
            for cls_dir in sorted(c for c in author_train.iterdir() if c.is_dir()):
                per_class[norm_class(cls_dir.name)].extend(list_images(cls_dir))

            for tgt, imgs in per_class.items():
                imgs = sorted(imgs)
                random.shuffle(imgs)
                n_train = int(len(imgs) * TRAIN_RATIO)
                train_imgs, val_imgs = imgs[:n_train], imgs[n_train:]
                for split, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
                    used = set()
                    n = copy_into(split_imgs, ORGANIZED / split / eq_name / tgt, used)
                    stats[eq_name][split][tgt] += n

    # ---- Rapor ----
    print("=" * 74)
    print("ORGANIZED SPLIT İSTATİSTİKLERİ (train/val = yazar train 85/15, test = yazar val)")
    print("=" * 74)
    totals = defaultdict(int)
    for eq_name in sorted(stats):
        print(f"\n📦 {eq_name}")
        classes = sorted({c for sp in stats[eq_name].values() for c in sp})
        for cls in classes:
            tr = stats[eq_name]["train"].get(cls, 0)
            va = stats[eq_name]["val"].get(cls, 0)
            te = stats[eq_name]["test"].get(cls, 0)
            totals["train"] += tr
            totals["val"] += va
            totals["test"] += te
            print(f"   {cls:14s} train={tr:5d}  val={va:4d}  test={te:5d}")
    print("\n" + "=" * 74)
    print(f"TOPLAM  train={totals['train']}  val={totals['val']}  test={totals['test']}  "
          f"(genel {sum(totals.values())})")
    print("=" * 74)


if __name__ == "__main__":
    main()
