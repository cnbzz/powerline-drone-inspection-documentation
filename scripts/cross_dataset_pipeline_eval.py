"""Full-pipeline cross-dataset baseline (Stage 7.3-A1c) — YOLO -> crop -> classify.

Replicates the REAL HHATS flow on CPLID: YOLOv8m detects the insulator, the
highest-confidence insulator bbox is cropped (pipeline detection logic, exact
bbox), and the crop is classified by the Stage 2 glass-insulator model. This fixes
the A1 shortcut (feeding full 1152x864 scenes straight into Stage 2), isolating the
input-format effect from the true domain shift. No retraining; nothing modified.

Detection mirrors scripts/run_inspection_pipeline.py::process_rgb_image
(model.predict(conf=0.25), boxes -> xyxy). Insulator classes = 0..3.

HONEST FRAMING: pipeline F1 is expected to be HIGHER than the A1 shortcut (0.28)
but LOWER than the model's own test F1 (0.98). Numbers are reported as-is.

NOTE (from the ADIM-0 probe): YOLO localizes CPLID insulators at 100% / ~0.90 conf
but LABELS them 'insulator_porcelain' (not glass). Localization is what cropping
needs, so we crop the detected bbox and feed it to the glass-insulator classifier
(the classifier under test). The class-label mismatch is reported separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)
from ultralytics import YOLO  # noqa: E402

from src.defect_classifier.inference import DefectInferenceEngine  # noqa: E402

SEED = 42
CONF = 0.25  # pipeline default
EQUIPMENT = "glass-insulator"
CLASSES = ["good", "missing-cap"]
INSULATOR_IDS = {0, 1, 2, 3}
CLASS_NAMES = {
    0: "insulator_polymer", 1: "insulator_glass", 2: "insulator_porcelain",
    3: "insulator_generic", 4: "tower", 5: "spacer", 6: "damper", 7: "tower_id_plate",
}
FOLDER_TO_CLASS = {"Normal_Insulators": "good", "Defective_Insulators": "missing-cap"}

_ROOT_P = _ROOT
CPLID_DIR = _ROOT / "data" / "external" / "cplid"
OUT_DIR = _ROOT / "outputs" / "cross_dataset_eval"
YOLO_CKPT = _ROOT / "models" / "trained" / "yolov8m_8class_v1" / "weights" / "best.pt"
GLASS_CKPT = _ROOT / "models" / "trained" / "defect_classifiers" / EQUIPMENT / "best.pt"
A1_SHORTCUT_JSON = OUT_DIR / "baseline_results.json"  # A1 (full-scene shortcut)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for folder, cls in FOLDER_TO_CLASS.items():
        for p in sorted((CPLID_DIR / folder / "images").glob("*.jpg")):
            out.append((p, cls))
    return out


def _self_test_f1() -> float | None:
    try:
        ck = torch.load(GLASS_CKPT, map_location="cpu", weights_only=False)
        return float(ck.get("val_f1")) if ck.get("val_f1") is not None else None
    except Exception:
        return None


def _a1_shortcut_f1() -> float | None:
    try:
        return float(json.loads(A1_SHORTCUT_JSON.read_text(encoding="utf-8"))["macro_f1"])
    except Exception:
        return None


def _plot_confusion(cm: np.ndarray, out_path: Path, acc: float, macro_f1: float) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=120)
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (CPLID folder)")
    ax.set_title(f"FULL PIPELINE (YOLO+CROP+CLASSIFY)\nglass-insulator on CPLID\n"
                 f"acc={acc:.3f}  macro-F1={macro_f1:.3f}", fontsize=10)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _top_insulator_bbox(yolo, image_path: Path):
    """Return (cid, conf, x1, y1, x2, y2) of the highest-conf insulator, or None."""
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


def main() -> None:
    _set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("TAM PIPELINE ÇAPRAZ-VERİ (YOLO→KIRP→SINIFLANDIR) — glass-insulator ↔ CPLID")
    print("=" * 72)
    print(f"Cihaz: {device}")

    yolo = YOLO(str(YOLO_CKPT))
    engine = DefectInferenceEngine(device=device)
    if engine.load_model(EQUIPMENT) is None:
        print(f"HATA: {EQUIPMENT} modeli yüklenemedi."); sys.exit(1)

    samples = _collect()
    print(f"CPLID toplam: {len(samples)} "
          f"(good={sum(1 for _, c in samples if c == 'good')}, "
          f"missing-cap={sum(1 for _, c in samples if c == 'missing-cap')})")

    y_true: list[str] = []
    y_pred: list[str] = []
    n_undetected = 0
    yolo_label_hits: dict[str, int] = {}
    failed = 0

    for i, (path, true_cls) in enumerate(samples):
        try:
            bbox = _top_insulator_bbox(yolo, path)
            if bbox is None:
                n_undetected += 1          # pipeline reality: no detection → no crop
                continue
            cid, conf, x1, y1, x2, y2 = bbox
            yolo_label_hits[CLASS_NAMES[cid]] = yolo_label_hits.get(CLASS_NAMES[cid], 0) + 1

            with open(path, "rb") as f:    # PIL, Unicode-safe
                pil = Image.open(f).convert("RGB")
            crop = pil.crop((int(x1), int(y1), int(x2), int(y2)))  # exact bbox
            res = engine.predict(crop, EQUIPMENT)  # forward-only, no_grad inside
            if res is None:
                failed += 1
                continue
            y_true.append(true_cls)
            y_pred.append(res["predicted_class"])
        except Exception:
            failed += 1
            continue
        if (i + 1) % 150 == 0:
            print(f"  ... {i + 1}/{len(samples)} işlendi")

    n_detected = len(y_true)
    detection_rate = (len(samples) - n_undetected) / len(samples)
    print(f"YOLO tespit oranı: %{detection_rate * 100:.1f} "
          f"(tespit={len(samples) - n_undetected}, tespit-yok={n_undetected})")
    print(f"Sınıflandırılan (kırpılıp Stage2'ye giren): {n_detected}  (hata: {failed})")

    if n_detected == 0:
        print("Hiç örnek sınıflandırılamadı — çıkılıyor.")
        sys.exit(1)

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    per_class = {
        c: {"precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"])}
        for c in CLASSES
    }

    self_f1 = _self_test_f1()
    shortcut_f1 = _a1_shortcut_f1()

    print("-" * 72)
    print(f"Accuracy (tam pipeline, tespit edilenlerde): {acc:.4f}")
    for c in CLASSES:
        pc = per_class[c]
        print(f"  {c:12s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")
    print(f"Macro-F1: {macro_f1:.4f}  |  Weighted-F1: {weighted_f1:.4f}")
    worst = min(CLASSES, key=lambda c: per_class[c]["f1"])
    print(f"En zayıf sınıf: {worst} (F1={per_class[worst]['f1']:.3f})")
    print(f"YOLO tespit sınıf dağılımı (kırpılanlar): {yolo_label_hits}")

    # 3-way comparison
    print("-" * 72)
    print("3'LÜ KIYAS (macro-F1):")
    print(f"  Kendi-test           : {self_f1:.4f}" if self_f1 else "  Kendi-test: —")
    print(f"  Eski kısayol (A1)    : {shortcut_f1:.4f}" if shortcut_f1 else "  Eski kısayol: —")
    print(f"  Gerçek pipeline (A1c): {macro_f1:.4f}")
    if shortcut_f1 is not None:
        print(f"  → Format etkisi (kısayol→pipeline): {macro_f1 - shortcut_f1:+.4f}")
    if self_f1 is not None:
        print(f"  → Kalan domain shift (pipeline→kendi-test): {self_f1 - macro_f1:+.4f}")

    cm_path = OUT_DIR / "confusion_matrix_pipeline.png"
    _plot_confusion(cm, cm_path, acc, macro_f1)

    results = {
        "label": "FULL PIPELINE — YOLO+CROP+CLASSIFY",
        "stage": "7.3-A1c cross-dataset via real pipeline (pre-retraining baseline)",
        "model": EQUIPMENT,
        "model_classes": CLASSES,
        "detector": "yolov8m_8class_v1",
        "eval_dataset": "CPLID (data/external/cplid)",
        "seed": SEED,
        "yolo_conf_threshold": CONF,
        "device": device,
        "n_images_total": len(samples),
        "yolo_detection_rate": round(detection_rate, 4),
        "n_detected": len(samples) - n_undetected,
        "n_undetected": n_undetected,
        "n_classified": n_detected,
        "n_failed": failed,
        "crop_logic": "en yüksek güvenli insulator (class 0-3) bbox'ı, TAM bbox (padding yok), PIL crop",
        "yolo_label_distribution": yolo_label_hits,
        "yolo_label_note": (
            "YOLO CPLID cam izolatörlerini LOKALİZE ediyor (yüksek güven) ama "
            "sınıfı çoğunlukla 'insulator_porcelain' veriyor. Kırpma için "
            "lokalizasyon yeterli; test edilen sınıflandırıcı glass-insulator. "
            "Tam-otomatik pipeline'da YOLO sınıf-yönlendirmesi (porcelain→Stage2 "
            "yok) bunları glass-Stage2'ye ULAŞTIRMAZDI — ayrı bulgu."
        ),
        "class_mapping": FOLDER_TO_CLASS,
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "worst_class": worst,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm.tolist(),
                             "note": "satır=gerçek (klasör), sütun=tahmin; sadece tespit edilenler"},
        "comparison_macro_f1": {
            "self_test": round(self_f1, 4) if self_f1 is not None else None,
            "a1_shortcut_full_scene": round(shortcut_f1, 4) if shortcut_f1 is not None else None,
            "a1c_full_pipeline": round(macro_f1, 4),
            "format_effect_shortcut_to_pipeline": (round(macro_f1 - shortcut_f1, 4)
                                                   if shortcut_f1 is not None else None),
            "residual_domain_shift_pipeline_to_self": (round(self_f1 - macro_f1, 4)
                                                       if self_f1 is not None else None),
        },
        "domain_shift_notes": [
            "A1 kısayolu tam sahneyi (1152x864) doğrudan Stage2'ye verdi → girdi/ölçek uyumsuzluğu.",
            "A1c gerçek akışı YOLO bbox'ını kırpıp Stage2'ye verir → format etkisi izole edilir.",
            "Kalan pipeline→kendi-test farkı GERÇEK domain shift'tir (sentetik defect, farklı kaynak).",
            "Çapraz-veri düşüşü BEKLENEN bulgu; başarısızlık değil.",
        ],
        "academic_framing": (
            "EĞİTİM ÖNCESİ baz çizgi, gerçek pipeline ile. Sayılar dürüstçe raporlanmıştır; "
            "A1 (kısayol) ve A1c (pipeline) yan yana korunur."
        ),
    }
    (OUT_DIR / "pipeline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 72)
    print(f"JSON: {OUT_DIR / 'pipeline_results.json'}")
    print(f"PNG : {cm_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
