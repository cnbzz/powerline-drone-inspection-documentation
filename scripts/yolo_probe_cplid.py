"""YOLO capability probe on CPLID (Stage 7.3-A1c, ADIM 0).

Before running any full-pipeline evaluation we must verify that the HHATS Stage 1
detector (YOLOv8m, 8-class) can actually DETECT the glass insulators in CPLID.
If it cannot, the full YOLO->crop->classify test would be meaningless.

This script only PROBES: it runs YOLO on a small deterministic sample of CPLID
images, reports the detection rate + mean confidence + which insulator classes
fire, and saves a few annotated + cropped samples for visual inspection. It does
NOT evaluate the classifier and does NOT modify anything.

Detection logic mirrors the real pipeline (scripts/run_inspection_pipeline.py
process_rgb_image): model.predict(conf=0.25), boxes -> xyxy, insulator classes 0-3.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw  # noqa: E402
from ultralytics import YOLO  # noqa: E402

SEED = 42
CONF = 0.25  # same as pipeline default
YOLO_CKPT = _ROOT / "models" / "trained" / "yolov8m_8class_v1" / "weights" / "best.pt"
CPLID_DIR = _ROOT / "data" / "external" / "cplid"
PROBE_DIR = _ROOT / "outputs" / "cross_dataset_eval" / "yolo_probe"

CLASS_NAMES = {
    0: "insulator_polymer", 1: "insulator_glass", 2: "insulator_porcelain",
    3: "insulator_generic", 4: "tower", 5: "spacer", 6: "damper", 7: "tower_id_plate",
}
INSULATOR_IDS = {0, 1, 2, 3}
N_NORMAL, N_DEFECTIVE, N_SAVE = 12, 8, 8


def _sample_images() -> list[tuple[Path, str]]:
    rng = random.Random(SEED)
    out: list[tuple[Path, str]] = []
    for folder, cls, n in [("Normal_Insulators", "good", N_NORMAL),
                           ("Defective_Insulators", "missing-cap", N_DEFECTIVE)]:
        imgs = sorted((CPLID_DIR / folder / "images").glob("*.jpg"))
        out += [(p, cls) for p in rng.sample(imgs, min(n, len(imgs)))]
    return out


def main() -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(YOLO_CKPT))
    samples = _sample_images()
    print("=" * 72)
    print("YOLO YETENEK KONTROLÜ — CPLID (Stage 7.3-A1c ADIM 0)")
    print(f"Model: {YOLO_CKPT.name}  conf={CONF}  örnek={len(samples)}")
    print("=" * 72)

    n_with_insulator = 0
    all_confs: list[float] = []
    class_hits: dict[str, int] = {}
    saved = 0

    for i, (path, cls) in enumerate(samples):
        res = model.predict(str(path), conf=CONF, verbose=False)[0]
        ins_boxes = []
        if res.boxes is not None:
            for box in res.boxes:
                cid = int(box.cls)
                conf = float(box.conf)
                if cid in INSULATOR_IDS:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    ins_boxes.append((cid, conf, x1, y1, x2, y2))
                    class_hits[CLASS_NAMES[cid]] = class_hits.get(CLASS_NAMES[cid], 0) + 1

        has = len(ins_boxes) > 0
        n_with_insulator += int(has)
        top = max(ins_boxes, key=lambda b: b[1]) if has else None
        if has:
            all_confs.append(top[1])
        print(f"  [{cls:11s}] {path.name:10s} insulator_det={len(ins_boxes)} "
              f"top={'%s %.2f' % (CLASS_NAMES[top[0]], top[1]) if top else '—'}")

        # Save a few annotated + cropped samples for visual inspection.
        if saved < N_SAVE and has:
            try:
                with open(path, "rb") as f:
                    pil = Image.open(f).convert("RGB")
                ann = pil.copy()
                d = ImageDraw.Draw(ann)
                for (cid, conf, x1, y1, x2, y2) in ins_boxes:
                    d.rectangle([x1, y1, x2, y2], outline=(0, 122, 255), width=4)
                    d.text((x1 + 4, y1 + 4), f"{CLASS_NAMES[cid]} {conf:.2f}",
                           fill=(255, 0, 0))
                ann.save(PROBE_DIR / f"{saved:02d}_{cls}_{path.stem}_annot.jpg")
                cid, conf, x1, y1, x2, y2 = top
                pil.crop((int(x1), int(y1), int(x2), int(y2))).save(
                    PROBE_DIR / f"{saved:02d}_{cls}_{path.stem}_crop.jpg")
                saved += 1
            except Exception:
                pass

    rate = n_with_insulator / len(samples)
    mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    print("-" * 72)
    print(f"İzolatör tespit edilen görüntü: {n_with_insulator}/{len(samples)} "
          f"(oran %{rate * 100:.0f})")
    print(f"Ortalama top-insulator güven: {mean_conf:.3f}")
    print(f"Tespit edilen insulator sınıf dağılımı: {class_hits}")
    print(f"Görsel örnekler: {PROBE_DIR}  ({saved} annot+crop çifti)")
    print("-" * 72)
    verdict = ("HAZIR (>%70): tam pipeline testine geçilebilir"
               if rate >= 0.70 else
               "ZAYIF (<%70): tam pipeline testi anlamsız olabilir → DUR/SOR")
    print(f"KARAR: %{rate * 100:.0f} tespit → {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
