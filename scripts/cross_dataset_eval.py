"""Cross-dataset generalization baseline (Stage 7.3 A1) — PRE-TRAINING.

Runs the EXISTING HHATS glass-insulator classifier (Stage 2, binary
{good, missing-cap}) on the external CPLID dataset (Chinese Power Line Insulator
Dataset) WITHOUT any retraining. Forward-only inference; nothing is modified.

WHY: before folding external data into training, we need the pre-training
baseline so a later re-training run can be compared against it. Cross-dataset
accuracy being LOWER than the model's own test set is EXPECTED (domain shift) —
that is a finding, not a failure. Numbers are reported honestly, not inflated.

CLASS MAPPING (image-level, folder = ground truth):
    CPLID Normal_Insulators    -> good
    CPLID Defective_Insulators -> missing-cap   (CPLID defect = missing/broken cap)
The glass-insulator model's classes are exactly {good, missing-cap}, so the
mapping is direct (not heuristic).

DOMAIN-SHIFT CAVEATS (documented in the JSON, honest baseline):
    * HHATS model was trained on TIGHT insulator CROPS (InsPLAD-fault); CPLID
      images are FULL aerial scenes (~1152x864) fed as-is -> input/scale mismatch.
    * CPLID defective images are SYNTHETIC (U-Net composited).
    * Different capture source (State Grid China vs InsPLAD Brazil).
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
matplotlib.use("Agg")  # Streamlit/headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.defect_classifier.inference import DefectInferenceEngine  # noqa: E402

SEED = 42
EQUIPMENT = "glass-insulator"
CPLID_DIR = _ROOT / "data" / "external" / "cplid"
STN_DIR = _ROOT / "data" / "external" / "stn_plad"
OUT_DIR = _ROOT / "outputs" / "cross_dataset_eval"
MODEL_CKPT = _ROOT / "models" / "trained" / "defect_classifiers" / EQUIPMENT / "best.pt"

CLASSES = ["good", "missing-cap"]
# CPLID folder -> ground-truth HHATS class.
FOLDER_TO_CLASS = {
    "Normal_Insulators": "good",
    "Defective_Insulators": "missing-cap",
}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collect_cplid_samples() -> list[tuple[Path, str]]:
    """Return (image_path, true_class) for every CPLID image."""
    samples: list[tuple[Path, str]] = []
    for folder, cls in FOLDER_TO_CLASS.items():
        img_dir = CPLID_DIR / folder / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.jpg")):
            samples.append((p, cls))
    return samples


def _self_test_f1() -> float | None:
    """Read the model's own validation F1 from its checkpoint (no recompute)."""
    try:
        ck = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)
        return float(ck.get("val_f1")) if ck.get("val_f1") is not None else None
    except Exception:
        return None


def _plot_confusion(cm: np.ndarray, out_path: Path, acc: float, macro_f1: float) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=120)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (CPLID folder)")
    ax.set_title(f"PRE-TRAINING BASELINE\nglass-insulator on CPLID\n"
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


def _stn_qualitative(engine: DefectInferenceEngine, n: int = 40) -> dict:
    """STN-PLAD has NO glass-insulator defect labels and is a detection dataset of
    transmission structures → NOT directly comparable. We only report the raw
    predicted-class distribution on a small sample as a qualitative observation."""
    imgs = sorted(STN_DIR.rglob("*.jpg"))[:n]
    dist: dict[str, int] = {c: 0 for c in CLASSES}
    used = 0
    for p in imgs:
        try:
            with open(p, "rb") as f:
                pil = Image.open(f).convert("RGB")
            res = engine.predict(pil, EQUIPMENT)
            if res:
                dist[res["predicted_class"]] = dist.get(res["predicted_class"], 0) + 1
                used += 1
        except Exception:
            continue
    return {
        "n_sampled": used,
        "predicted_distribution": dist,
        "note": ("STN-PLAD'de cam-izolatör defect etiketi YOK ve içerik iletim "
                 "yapılarıdır → doğrudan karşılaştırılamaz. Yalnızca modelin tahmin "
                 "dağılımı NİTEL gözlem olarak verildi (doğruluk hesaplanamaz)."),
    }


def main() -> None:
    _set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("ÇAPRAZ-VERİ BAZ ÇİZGİ (EĞİTİM ÖNCESİ) — glass-insulator ↔ CPLID")
    print("=" * 72)
    print(f"Cihaz: {device}")

    engine = DefectInferenceEngine(device=device)
    if engine.load_model(EQUIPMENT) is None:
        print(f"HATA: {EQUIPMENT} modeli yüklenemedi ({MODEL_CKPT}).")
        sys.exit(1)

    samples = _collect_cplid_samples()
    print(f"CPLID toplam görüntü: {len(samples)} "
          f"(good={sum(1 for _, c in samples if c == 'good')}, "
          f"missing-cap={sum(1 for _, c in samples if c == 'missing-cap')})")

    y_true: list[str] = []
    y_pred: list[str] = []
    failed = 0
    for i, (path, true_cls) in enumerate(samples):
        try:
            with open(path, "rb") as f:            # PIL (Unicode-safe), not cv2
                pil = Image.open(f).convert("RGB")
            res = engine.predict(pil, EQUIPMENT)   # forward-only, no_grad inside
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

    n_eval = len(y_true)
    print(f"Değerlendirilen: {n_eval}  (başarısız/atlanan: {failed})")

    # --- Metrics (honest) ---
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    report = classification_report(y_true, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASSES,
                              average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASSES,
                                 average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    self_f1 = _self_test_f1()

    per_class = {
        c: {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
        }
        for c in CLASSES
    }

    # --- Console: honest comparison ---
    print("-" * 72)
    print(f"Accuracy (çapraz-veri): {acc:.4f}")
    for c in CLASSES:
        pc = per_class[c]
        print(f"  {c:12s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} (n={pc['support']})")
    print(f"Macro-F1 (çapraz-veri): {macro_f1:.4f}  |  Weighted-F1: {weighted_f1:.4f}")
    if self_f1 is not None:
        drop = self_f1 - macro_f1
        print(f"Kendi-test F1: {self_f1:.4f}  →  Çapraz-veri macro-F1: {macro_f1:.4f}"
              f"  (DÜŞÜŞ: {drop:.4f} / %{drop / self_f1 * 100:.1f})")
    worst = min(CLASSES, key=lambda c: per_class[c]["f1"])
    print(f"En çok düşen/zayıf sınıf: {worst} (F1={per_class[worst]['f1']:.3f})")

    # --- STN-PLAD qualitative ---
    stn = _stn_qualitative(engine)
    print(f"STN-PLAD nitel: n={stn['n_sampled']} tahmin dağılımı={stn['predicted_distribution']}")

    # --- Confusion matrix PNG ---
    cm_path = OUT_DIR / "confusion_matrix_baseline.png"
    _plot_confusion(cm, cm_path, acc, macro_f1)

    # --- JSON ---
    results = {
        "label": "PRE-TRAINING BASELINE",
        "stage": "7.3-A1 cross-dataset generalization (pre-retraining baseline)",
        "model": EQUIPMENT,
        "model_classes": CLASSES,
        "eval_dataset": "CPLID (data/external/cplid)",
        "seed": SEED,
        "device": device,
        "n_images_evaluated": n_eval,
        "n_failed": failed,
        "class_mapping": FOLDER_TO_CLASS,
        "mapping_confidence": (
            "DOĞRUDAN eşleme (heuristik değil): model sınıfları {good, missing-cap} "
            "CPLID klasörleriyle (Normal/Defective) birebir. Etiket görüntü düzeyinde "
            "(klasör = doğru etiket)."
        ),
        "accuracy": round(acc, 4),
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "self_test_f1": round(self_f1, 4) if self_f1 is not None else None,
        "cross_vs_self_f1_drop": (round(self_f1 - macro_f1, 4)
                                  if self_f1 is not None else None),
        "worst_class": worst,
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": cm.tolist(),
            "note": "satır=gerçek (CPLID klasörü), sütun=tahmin",
        },
        "domain_shift_notes": [
            "HHATS modeli InsPLAD-fault KIRPILMIŞ izolatörlerle eğitildi; CPLID "
            "TAM aerial sahne (~1152x864) olduğu gibi verildi → girdi/ölçek uyumsuzluğu.",
            "CPLID defektif görüntüleri SENTETİK (U-Net kompozit).",
            "Farklı çekim kaynağı (State Grid China UAV vs InsPLAD Brezilya).",
            "Çapraz-veri F1'in kendi-test F1'den düşük olması BEKLENEN domain-shift "
            "bulgusudur; başarısızlık değildir.",
        ],
        "stn_plad_qualitative": stn,
        "academic_framing": (
            "Bu bir EĞİTİM ÖNCESİ baz çizgidir. Amaç, harici veriyle yeniden eğitim "
            "sonrası genelleme artışını kıyaslamak. Sayılar dürüstçe raporlanmıştır."
        ),
    }
    json_path = OUT_DIR / "baseline_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print("-" * 72)
    print(f"JSON: {json_path}")
    print(f"PNG : {cm_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
