"""Stage 7.3-A3 — Post-retraining cross-dataset eval (YOLO -> crop -> classify).

Re-runs the EXACT A1c pipeline (scripts/cross_dataset_pipeline_eval.py) on CPLID
with the RETRAINED glass-insulator model, to compare against the pre-training
baseline macro-F1 = 0.5185.

LEAKAGE-AWARE (critical for honesty): the A2 retraining added CPLID train+val
crops to the glass-insulator training set. Those images are now "seen". Reporting
macro-F1 on the full 848 would be optimistic. So this script reads the A2 split
manifest (outputs/cross_dataset_eval/cplid_split_manifest.json) and reports:

  1. FULL 848  — directly comparable to the 0.52 baseline, but FLAGGED as
     containing training images (optimistic / partial leakage).
  2. HELD-OUT CPLID-TEST only — images NEVER used in training → the HONEST
     generalization number.

Same detector, same conf=0.25, same crop logic, same seed. Numbers reported as-is.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.defect_classifier.inference import DefectInferenceEngine  # noqa: E402

SEED = 42
CONF = 0.25
EQUIPMENT = "glass-insulator"
CLASSES = ["good", "missing-cap"]
INSULATOR_IDS = {0, 1, 2, 3}
CLASS_NAMES = {
    0: "insulator_polymer", 1: "insulator_glass", 2: "insulator_porcelain",
    3: "insulator_generic", 4: "tower", 5: "spacer", 6: "damper", 7: "tower_id_plate",
}
FOLDER_TO_CLASS = {"Normal_Insulators": "good", "Defective_Insulators": "missing-cap"}

CPLID_DIR = _ROOT / "data" / "external" / "cplid"
OUT_DIR = _ROOT / "outputs" / "cross_dataset_eval"
YOLO_CKPT = _ROOT / "models" / "trained" / "yolov8m_8class_v1" / "weights" / "best.pt"
BASELINE_JSON = OUT_DIR / "pipeline_results.json"          # pre-training A1c (0.5185)
MANIFEST_JSON = OUT_DIR / "cplid_split_manifest.json"      # A2 split


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_split_lookup() -> dict[str, str]:
    """stem ('Folder__id') -> split ('train'/'val'/'test'). Empty if no manifest."""
    lookup: dict[str, str] = {}
    try:
        m = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        for split, by_cls in m["splits"].items():
            for _cls, stems in by_cls.items():
                for stem in stems:
                    lookup[stem] = split
    except Exception as e:
        print(f"  ! manifest not loaded ({e}) — held-out breakdown unavailable")
    return lookup


def _collect() -> list[tuple[Path, str, str]]:
    """(image_path, true_cls, stem) for all CPLID images."""
    out = []
    for folder, cls in FOLDER_TO_CLASS.items():
        for p in sorted((CPLID_DIR / folder / "images").glob("*.jpg")):
            out.append((p, cls, f"{folder}__{p.stem}"))
    return out


def _top_insulator_bbox(yolo, image_path: Path):
    res = yolo.predict(str(image_path), conf=CONF, verbose=False)[0]
    best = None
    if res.boxes is not None:
        for box in res.boxes:
            cid = int(box.cls)
            if cid in INSULATOR_IDS:
                conf = float(box.conf)
                if best is None or conf > best[1]:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best = (cid, conf, x1, y1, x2, y2)
    return best


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        return {"n": 0}
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0))
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    per_class = {c: {"precision": round(rep[c]["precision"], 4),
                     "recall": round(rep[c]["recall"], 4),
                     "f1": round(rep[c]["f1-score"], 4),
                     "support": int(rep[c]["support"])} for c in CLASSES}
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return {"n": len(y_true), "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def _plot_cm(cm, out_path: Path, title: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=120)
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (CPLID folder)")
    ax.set_title(title, fontsize=9)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    _set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("A3 — POST-RETRAIN ÇAPRAZ-VERİ (YOLO→KIRP→SINIFLANDIR) glass-insulator↔CPLID")
    print("=" * 72)
    print(f"Cihaz: {device}")

    split_lookup = _load_split_lookup()
    yolo = YOLO(str(YOLO_CKPT))
    engine = DefectInferenceEngine(device=device)
    if engine.load_model(EQUIPMENT) is None:
        print(f"HATA: {EQUIPMENT} modeli yüklenemedi."); sys.exit(1)

    samples = _collect()
    print(f"CPLID toplam: {len(samples)}")

    # overall + per-split accumulators
    y_true_all, y_pred_all = [], []
    subsets: dict[str, dict[str, list]] = {
        "train": {"t": [], "p": []}, "val": {"t": [], "p": []},
        "test": {"t": [], "p": []}, "unknown": {"t": [], "p": []}}
    n_undetected = 0
    yolo_label_hits: dict[str, int] = {}
    failed = 0

    for i, (path, true_cls, stem) in enumerate(samples):
        try:
            bbox = _top_insulator_bbox(yolo, path)
            if bbox is None:
                n_undetected += 1
                continue
            cid, conf, x1, y1, x2, y2 = bbox
            yolo_label_hits[CLASS_NAMES[cid]] = yolo_label_hits.get(CLASS_NAMES[cid], 0) + 1
            with open(path, "rb") as f:
                pil = Image.open(f).convert("RGB")
            crop = pil.crop((int(x1), int(y1), int(x2), int(y2)))
            res = engine.predict(crop, EQUIPMENT)
            if res is None:
                failed += 1
                continue
            pred = res["predicted_class"]
            y_true_all.append(true_cls); y_pred_all.append(pred)
            split = split_lookup.get(stem, "unknown")
            subsets[split]["t"].append(true_cls); subsets[split]["p"].append(pred)
        except Exception:
            failed += 1
            continue
        if (i + 1) % 150 == 0:
            print(f"  ... {i + 1}/{len(samples)} işlendi")

    detection_rate = (len(samples) - n_undetected) / len(samples)
    print(f"YOLO tespit oranı: %{detection_rate*100:.1f}  |  sınıflandırılan: {len(y_true_all)}  (hata: {failed})")

    overall = _metrics(y_true_all, y_pred_all)
    held_out = _metrics(subsets["test"]["t"], subsets["test"]["p"])
    seen = _metrics(subsets["train"]["t"] + subsets["val"]["t"],
                    subsets["train"]["p"] + subsets["val"]["p"])

    # baseline
    try:
        base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        base_macro = base.get("macro_f1")
    except Exception:
        base_macro = None

    print("-" * 72)
    print(f"[FULL 848]      acc={overall['accuracy']}  macro-F1={overall['macro_f1']}  "
          f"weighted-F1={overall['weighted_f1']}   (⚠️ eğitim görüntüleri DAHİL — iyimser)")
    if held_out.get("n"):
        print(f"[HELD-OUT TEST] n={held_out['n']}  acc={held_out['accuracy']}  "
              f"macro-F1={held_out['macro_f1']}   (✅ eğitimde GÖRÜLMEDİ — dürüst)")
    if seen.get("n"):
        print(f"[SEEN train+val] n={seen['n']}  macro-F1={seen['macro_f1']}   (referans)")
    print("-" * 72)
    print("KIYAS (macro-F1):")
    print(f"  Eğitim ÖNCESİ (baseline, full 848) : {base_macro}")
    print(f"  Eğitim SONRASI full 848 (leaky)    : {overall['macro_f1']}"
          + (f"   Δ={overall['macro_f1']-base_macro:+.4f}" if base_macro is not None else ""))
    if held_out.get("n"):
        print(f"  Eğitim SONRASI held-out (honest)   : {held_out['macro_f1']}"
              + (f"   Δ={held_out['macro_f1']-base_macro:+.4f}" if base_macro is not None else ""))

    _plot_cm(overall["confusion_matrix"], OUT_DIR / "confusion_matrix_after_full.png",
             f"AFTER RETRAIN — FULL 848 (leaky)\nmacro-F1={overall['macro_f1']}")
    if held_out.get("n"):
        _plot_cm(held_out["confusion_matrix"], OUT_DIR / "confusion_matrix_after_heldout.png",
                 f"AFTER RETRAIN — HELD-OUT CPLID-test\nmacro-F1={held_out['macro_f1']}")

    results = {
        "label": "A3 POST-RETRAIN — YOLO+CROP+CLASSIFY",
        "stage": "7.3-A3 cross-dataset after external-data retraining",
        "model": EQUIPMENT,
        "model_classes": CLASSES,
        "detector": "yolov8m_8class_v1",
        "eval_dataset": "CPLID (data/external/cplid)",
        "seed": SEED, "yolo_conf_threshold": CONF, "device": device,
        "n_images_total": len(samples),
        "yolo_detection_rate": round(detection_rate, 4),
        "n_undetected": n_undetected, "n_classified": len(y_true_all), "n_failed": failed,
        "yolo_label_distribution": yolo_label_hits,
        "class_mapping": FOLDER_TO_CLASS,
        "leakage_note": (
            "A2 retraining added CPLID train+val crops to glass-insulator training. "
            "FULL-848 metrics therefore include SEEN images (optimistic). HELD-OUT "
            "CPLID-test is the honest generalization number."),
        "full_848": overall,
        "held_out_test": held_out,
        "seen_train_val": seen,
        "baseline_macro_f1_pre_training_full848": base_macro,
        "comparison_macro_f1": {
            "pre_training_full848": base_macro,
            "post_training_full848_leaky": overall["macro_f1"],
            "post_training_heldout_honest": held_out.get("macro_f1"),
            "delta_full848": (round(overall["macro_f1"] - base_macro, 4)
                              if base_macro is not None else None),
            "delta_heldout": (round(held_out["macro_f1"] - base_macro, 4)
                              if (base_macro is not None and held_out.get("n")) else None),
        },
        "honesty_note": "Sayılar olduğu gibi raporlanmıştır; şişirme yok. Leaky vs honest ayrı verildi.",
    }
    (OUT_DIR / "pipeline_results_after_training.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"JSON: {OUT_DIR / 'pipeline_results_after_training.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
