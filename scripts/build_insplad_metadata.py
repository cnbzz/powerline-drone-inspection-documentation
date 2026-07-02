"""organized/ klasöründen InsPLAD-fault metadata.json üret (Aşama 6.1.1).

Çalıştır:  PYTHONUTF8=1 uv run python scripts/build_insplad_metadata.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ORGANIZED = Path("data/insplad_fault/organized")
OUT = Path("data/insplad_fault/metadata.json")
IMG_EXTS = (".jpg", ".jpeg", ".png")
SPLITS = ["train", "val", "test"]


def count_images(d: Path) -> int:
    return sum(1 for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> None:
    equipment: dict[str, dict] = defaultdict(lambda: {"classes": [], "counts": {}})

    for split in SPLITS:
        sdir = ORGANIZED / split
        if not sdir.is_dir():
            continue
        for eq_dir in sorted(d for d in sdir.iterdir() if d.is_dir()):
            eq = eq_dir.name
            for cls_dir in sorted(c for c in eq_dir.iterdir() if c.is_dir()):
                cls = cls_dir.name
                if cls not in equipment[eq]["classes"]:
                    equipment[eq]["classes"].append(cls)
                equipment[eq]["counts"].setdefault(cls, {})
                equipment[eq]["counts"][cls][split] = count_images(cls_dir)

    # toplamlar
    totals = {s: 0 for s in SPLITS}
    for eq in equipment.values():
        for cls_counts in eq["counts"].values():
            for s in SPLITS:
                totals[s] += cls_counts.get(s, 0)

    metadata = {
        "dataset": "InsPLAD-fault (supervised_fault_classification)",
        "source": "https://github.com/andreluizbvs/InsPLAD",
        "drive_file": "supervised_fault_classification.zip (id=1Te3jRRM9gJXO8PwOoOoCPGY69Pq6YHUb)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": SPLITS,
        "split_policy": (
            "Yazar val/ -> test (resmi benchmark, gerçek dünya dağılımı korunur). "
            "Yazar train/ -> train+val, sınıf-bazında 85/15, seed=42."
        ),
        "class_harmonization": {"normal": "good", "corrosão": "rust"},
        "notes": (
            "Test seti aşırı dengesiz ve KASITLI gerçekçi dağılım (ör. yoke-suspension "
            "test/good=5742, rust=20). Metrikler accuracy yerine sınıf-bazında "
            "recall / PR-AUC ile yorumlanmalı; eğitimde class-weight/augmentation önerilir."
        ),
        "totals": totals,
        "equipment_classes": {k: equipment[k] for k in sorted(equipment)},
    }

    OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Metadata yazıldı: {OUT}")
    print(json.dumps({"totals": totals,
                      "equipment": {k: v["classes"] for k, v in metadata["equipment_classes"].items()}},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
