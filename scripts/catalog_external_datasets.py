"""Catalog external powerline/insulator datasets into data/external/catalog.json.

Read-only inventory: counts files by extension and sums sizes WITHOUT opening any
image. Covers every sub-directory of data/external/ plus the existing training set
data/insplad_fault/ (counted read-only). Never modifies dataset files.

RGB vs thermal split is a HEURISTIC: a token of {thermal, termal, ir, infrared}
in the sub-path (below the source root) or filename marks an image as thermal.
The source-root folder name is excluded so a root like "teias_ir_vl" does not
flag every file; per-file/sub-folder naming drives the split instead.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = _ROOT / "data" / "external"
INSPLAD_DIR = _ROOT / "data" / "insplad_fault"
CATALOG_PATH = EXTERNAL_DIR / "catalog.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
THERMAL_TOKENS = {"thermal", "termal", "ir", "infrared"}
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# Sources that were intentionally not downloaded in this step; surfaced in the
# catalog with an explicit skipped status + reason instead of being omitted.
SKIPPED_SOURCES = {
    "idid": "Kaggle kimlik bilgisi yok (~/.kaggle/kaggle.json) → atlandı, kullanıcı sağlamalı",
    "sfu_thermal": "SFU Thermal: indirme URL'si gerekli → atlandı (klasör boş oluşturuldu)",
}


def _is_thermal(rel_below_root: str) -> bool:
    """True if any path token (below source root) is a thermal marker."""
    tokens = set(_TOKEN_SPLIT.split(rel_below_root.lower()))
    return bool(tokens & THERMAL_TOKENS)


def _detect_annotation(root: Path, xml_count: int, txt_count: int) -> str:
    """Heuristic annotation-format detection (no file contents opened beyond names)."""
    has_yaml = any(root.rglob("data.yaml")) or any(root.rglob("*.yaml"))
    if has_yaml and txt_count > 0:
        return "yolo"
    if xml_count > 0 and xml_count >= txt_count:
        return "voc"
    if txt_count > 0:
        return "yolo"
    # Folder-based class layout: image-bearing leaf dirs that look like class names.
    class_like = False
    for sub in root.rglob("*"):
        if sub.is_dir():
            name = sub.name.lower()
            if name in ("good", "defect", "broken", "normal", "rust", "flashover",
                        "missing", "pollution"):
                class_like = True
                break
    return "folder_class" if class_like else "none"


def _scan_source(name: str, path: Path) -> Dict:
    """Scan one dataset directory; return its catalog entry (no images opened)."""
    image_count = rgb_count = thermal_count = 0
    total_bytes = 0
    xml_count = txt_count = 0
    root_str = str(path)

    for dirpath, _dirs, files in os.walk(path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            fp = os.path.join(dirpath, fn)
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
            if ext in IMAGE_EXTS:
                image_count += 1
                # Path below the source root (exclude root dir name from heuristic).
                rel = os.path.relpath(fp, root_str)
                if _is_thermal(rel):
                    thermal_count += 1
                else:
                    rgb_count += 1
            elif ext == ".xml":
                xml_count += 1
            elif ext == ".txt":
                # Ignore obvious non-annotation text files.
                if fn.lower() not in ("readme.txt", "requirements.txt", "license.txt"):
                    txt_count += 1

    annotation = _detect_annotation(path, xml_count, txt_count) if image_count else "none"
    status = "ok" if image_count > 0 else "empty"
    return {
        "name": name,
        "path": str(path.relative_to(_ROOT)).replace("\\", "/"),
        "image_count": image_count,
        "size_mb": round(total_bytes / (1024 * 1024), 1),
        "annotation": annotation,
        "rgb_count": rgb_count,
        "thermal_count": thermal_count,
        "status": status,
    }


def main() -> None:
    sources: List[Dict] = []
    seen = set()

    # 1) Every sub-directory currently under data/external/
    if EXTERNAL_DIR.exists():
        for sub in sorted(p for p in EXTERNAL_DIR.iterdir() if p.is_dir()):
            entry = _scan_source(sub.name, sub)
            # Apply explicit skipped status/reason where applicable.
            if sub.name in SKIPPED_SOURCES and entry["image_count"] == 0:
                entry["status"] = "skipped"
                entry["note"] = SKIPPED_SOURCES[sub.name]
            sources.append(entry)
            seen.add(sub.name)

    # 2) Ensure intentionally-skipped sources appear even if their dir is absent.
    for name, reason in SKIPPED_SOURCES.items():
        if name not in seen:
            sources.append({
                "name": name,
                "path": f"data/external/{name}",
                "image_count": 0, "size_mb": 0.0, "annotation": "none",
                "rgb_count": 0, "thermal_count": 0,
                "status": "skipped", "note": reason,
            })

    # 3) Existing training set (read-only) — include as an inventory source.
    #    Named "insplad_fault" to avoid colliding with the data/external/insplad copy.
    if INSPLAD_DIR.exists():
        insplad_entry = _scan_source("insplad_fault", INSPLAD_DIR)
        insplad_entry["note"] = "Mevcut eğitim verisi (read-only sayım, dokunulmadı)"
        sources.append(insplad_entry)

    totals = {
        "image_count": sum(s["image_count"] for s in sources),
        "rgb_count": sum(s["rgb_count"] for s in sources),
        "thermal_count": sum(s["thermal_count"] for s in sources),
        "size_mb": round(sum(s["size_mb"] for s in sources), 1),
        "sources_ok": sum(1 for s in sources if s["status"] == "ok"),
        "sources_failed": sum(1 for s in sources if s["status"] == "failed"),
        "sources_skipped": sum(1 for s in sources if s["status"] == "skipped"),
        "sources_empty": sum(1 for s in sources if s["status"] == "empty"),
    }

    catalog = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "heuristic_note": (
            "RGB/termal ayrımı heuristiktir: kaynak-kök adı hariç alt-yol/dosya "
            "adında {thermal, termal, ir, infrared} token'ı geçen görüntü termal "
            "sayılır. Görüntüler açılmadı; yalnızca dosya-düzeyi sayım."
        ),
        "sources": sources,
        "totals": totals,
    }

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"WROTE {CATALOG_PATH}")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
