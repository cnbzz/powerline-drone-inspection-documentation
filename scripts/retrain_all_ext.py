"""Stage 7.3-A2 — Retrain all 5 defect classifiers with external-data mixing.

- glass-insulator: trained on the CPLID-augmented data_root
  (data/insplad_fault/organized_cplid_aug) — InsPLAD + cropped CPLID.
- other 4 equipment: retrained on the ORIGINAL InsPLAD data_root (no proven
  external mapping → baseline preserved), so the whole fleet is refreshed under
  the same deterministic seed.

SAFETY: models are overwritten in place BUT a verified backup already exists
(models/trained/defect_classifiers_backup_20260630). If any single equipment
crashes (OOM/error), the others continue and the failed one keeps its backup.

Logs go to a SEPARATE run dir so the original all_results.json is not clobbered:
  outputs/training_logs/defect_classifier/retrain_ext_20260705/

Hyperparameters are UNCHANGED from the original run (epochs=50, batch=16,
lr=1e-3, patience=10, image_size=384) — only the dataset widened.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from src.defect_classifier.train import train_one_equipment  # noqa: E402

ORIG_ROOT = "data/insplad_fault/organized"
AUG_ROOT = "data/insplad_fault/organized_cplid_aug"
RUN_TAG = "retrain_ext_20260705"
LOG_DIR = f"outputs/training_logs/defect_classifier/{RUN_TAG}"
BASELINE_JSON = _ROOT / "outputs/training_logs/defect_classifier/all_results.json"

# (equipment, data_root, external_note)
PLAN = [
    ("glass-insulator", AUG_ROOT, "InsPLAD + CPLID (VOC-bbox crop, train/val only)"),
    ("lightning-rod-suspension", ORIG_ROOT, "InsPLAD only (no proven external mapping)"),
    ("polymer-insulator-upper-shackle", ORIG_ROOT, "InsPLAD only (no proven external mapping)"),
    ("vari-grip", ORIG_ROOT, "InsPLAD only (no proven external mapping)"),
    ("yoke-suspension", ORIG_ROOT, "InsPLAD only (no proven external mapping)"),
]


def _load_baseline() -> dict:
    try:
        return json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--only", type=str, default=None,
                    help="comma-sep equipment subset (default: all 5)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("HATA: CUDA yok — eğitim çok yavaş. Durduruluyor.")
        return

    baseline = _load_baseline()
    only = set(args.only.split(",")) if args.only else None
    plan = [p for p in PLAN if (only is None or p[0] in only)]

    print("=" * 72)
    print(f"Stage 7.3-A2 RETRAIN — device={device}  run_tag={RUN_TAG}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB "
          f"({torch.cuda.get_device_name(0)})")
    print("=" * 72)

    all_results: dict = {}
    total_start = datetime.now()

    for i, (equipment, data_root, note) in enumerate(plan, 1):
        print(f"\n{'='*72}\n[{i}/{len(plan)}] {equipment}  ←  {note}\n{'='*72}")
        try:
            res = train_one_equipment(
                equipment=equipment,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                image_size=args.image_size,
                device=device,
                data_root=data_root,
                log_dir=LOG_DIR,
                seed=42,
            )
            res["external_note"] = note
            res["data_root"] = data_root
            all_results[equipment] = res
        except Exception as e:  # OOM / any error → keep backup, continue
            print(f"❌ FAILED {equipment}: {e}")
            traceback.print_exc()
            all_results[equipment] = {"error": str(e),
                                      "traceback": traceback.format_exc(),
                                      "external_note": note,
                                      "note": "backup preserved; not overwritten if crash was pre-save"}
            torch.cuda.empty_cache()

    total_min = (datetime.now() - total_start).total_seconds() / 60

    # ---- Comparison table: old vs new ----
    print(f"\n{'='*72}\nKIYAS TABLOSU (eski vs yeni) — toplam {total_min:.1f} dk\n{'='*72}")
    header = f"{'Equipment':<34}{'old_val':>9}{'new_val':>9}{'old_test':>10}{'new_test':>10}  flag"
    print(header)
    print("-" * len(header))
    comparison = {}
    for equipment, _, _ in plan:
        res = all_results.get(equipment, {})
        base = baseline.get(equipment, {})
        old_val = base.get("best_val_f1")
        old_test = base.get("test_f1")
        if "error" in res:
            print(f"{equipment:<34}{'':>9}{'':>9}{'':>10}{'':>10}  ❌ FAILED (backup preserved)")
            comparison[equipment] = {"status": "failed", "old_val_f1": old_val,
                                     "old_test_f1": old_test, "error": res["error"]}
            continue
        new_val = res.get("best_val_f1")
        new_test = res.get("test_f1")
        # Regression flag on the DIRECTLY comparable metric (test on unchanged InsPLAD test)
        dropped = (old_test is not None and new_test is not None and new_test < old_test - 1e-6)
        flag = "⚠️ DÜŞTÜ" if dropped else "✅"
        print(f"{equipment:<34}"
              f"{(old_val or 0):>9.4f}{(new_val or 0):>9.4f}"
              f"{(old_test or 0):>10.4f}{(new_test or 0):>10.4f}  {flag}")
        comparison[equipment] = {
            "status": "ok",
            "old_val_f1": old_val, "new_val_f1": new_val,
            "old_test_f1": old_test, "new_test_f1": new_test,
            "test_delta": (round(new_test - old_test, 4)
                           if (old_test is not None and new_test is not None) else None),
            "regressed_on_test": bool(dropped),
            "external_note": res.get("external_note"),
            "val_note": ("glass val set is InsPLAD+CPLID (not directly comparable to old InsPLAD-only val)"
                         if equipment == "glass-insulator" else "same InsPLAD val set"),
        }

    out = {
        "run_tag": RUN_TAG,
        "timestamp": datetime.now().isoformat(),
        "total_minutes": round(total_min, 2),
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size,
                        "lr": args.lr, "image_size": args.image_size,
                        "patience": 10, "seed": 42},
        "comparison": comparison,
    }
    out_path = Path(LOG_DIR) / "comparison_old_vs_new.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # also dump the raw new results for this run (separate from baseline all_results.json)
    (Path(LOG_DIR) / "all_results_ext.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    regressed = [e for e, c in comparison.items() if c.get("regressed_on_test")]
    failed = [e for e, c in comparison.items() if c.get("status") == "failed"]
    print("-" * len(header))
    print(f"Düşen (test F1): {regressed or 'yok'}")
    print(f"Başarısız: {failed or 'yok'}")
    print(f"💾 {out_path}")


if __name__ == "__main__":
    main()
