"""Evaluate the 3 hotspot-detection methods on the 200 synthetic thermal images.

Computes, per method (absolute / statistical / hybrid):
* image-level detection (hotspot present yes/no): TP/FP/TN/FN -> P/R/F1/accuracy
* localization quality: mean bbox IoU and fraction of GT hotspots matched at IoU>0.5
* risk-level classification confusion (over IoU>0.5 matches)
* per-scenario breakdown (with explicit focus on the false-positive traps 07/08)

Outputs go to ``outputs/hotspot_evaluation/`` (CSV tables, confusion-matrix PNGs,
false_positives/negatives JSON, a human-readable summary, and 6 visual checks).

Usage::

    uv run python scripts/evaluate_hotspot_detector.py --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Make ``src`` importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thermal.hotspot_detector import DetectionResult, bbox_iou, detect_hotspots  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

SEED = 42
THERMAL_DIR = PROJECT_ROOT / "data" / "synthetic" / "thermal"
GT_FILE = THERMAL_DIR / "ground_truth_all.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "hotspot_evaluation"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

METHODS = ["absolute", "statistical", "hybrid"]
RISK_LEVELS = ["low", "medium", "high", "critical"]
IOU_MATCH = 0.5


# --------------------------------------------------------------------------- #
# Matching                                                                     #
# --------------------------------------------------------------------------- #
def greedy_match(pred_boxes: list[tuple], gt_boxes: list[tuple]) -> list[tuple[int, int, float]]:
    """Greedily match predictions to GT by descending IoU (1:1). Returns (pi, gi, iou)."""
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            iou = bbox_iou(pb, gb)
            if iou > 0:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, iou))
    return matches


# --------------------------------------------------------------------------- #
# Core evaluation                                                              #
# --------------------------------------------------------------------------- #
def npy_path(gt: dict) -> Path:
    return THERMAL_DIR / gt["scenario"] / f"{gt['image_id']}.npy"


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def evaluate() -> dict:
    gts = json.loads(GT_FILE.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(gts)} ground-truth records")

    # Per-method accumulators.
    img_conf = {m: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for m in METHODS}
    loc_ious = {m: [] for m in METHODS}          # best IoU per GT hotspot (0 if unmatched)
    risk_pairs = {m: [] for m in METHODS}        # (gt_risk, pred_risk) for IoU>0.5 matches
    scen_conf = {m: defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0}) for m in METHODS}
    false_pos = {m: [] for m in METHODS}
    false_neg = {m: [] for m in METHODS}

    for gt in gts:
        thermal = np.load(npy_path(gt))
        gt_boxes = [tuple(h["bbox"]) for h in gt["hotspots"]]
        gt_has = bool(gt["has_hotspot"])
        scenario = gt["scenario"]

        for m in METHODS:
            res: DetectionResult = detect_hotspots(thermal, method=m)
            pred_boxes = [hs.bbox for hs in res.hotspots]
            pred_has = len(pred_boxes) > 0

            # ---- image-level confusion ----
            if gt_has and pred_has:
                cell = "TP"
            elif (not gt_has) and pred_has:
                cell = "FP"
            elif (not gt_has) and (not pred_has):
                cell = "TN"
            else:
                cell = "FN"
            img_conf[m][cell] += 1
            scen_conf[m][scenario][cell] += 1

            if cell == "FP":
                false_pos[m].append({
                    "image_id": gt["image_id"], "scenario": scenario,
                    "n_pred": len(pred_boxes),
                    "max_pred_peak_c": max((hs.peak_temp_c for hs in res.hotspots), default=None),
                    "ambient_c": gt["ambient_temp_c"],
                    "reason": "predicted hotspot where ground truth has none",
                })
            elif cell == "FN":
                false_neg[m].append({
                    "image_id": gt["image_id"], "scenario": scenario,
                    "n_gt": len(gt_boxes),
                    "gt_peak_c": max((h["peak_temp_c"] for h in gt["hotspots"]), default=None),
                    "ambient_c": gt["ambient_temp_c"],
                    "reason": "missed a real hotspot",
                })

            # ---- localization + risk (only meaningful when GT has hotspots) ----
            if gt_boxes:
                matches = greedy_match(pred_boxes, gt_boxes)
                matched_g = {gi: iou for _, gi, iou in matches}
                for gi in range(len(gt_boxes)):
                    loc_ious[m].append(matched_g.get(gi, 0.0))
                for pi, gi, iou in matches:
                    if iou >= IOU_MATCH:
                        risk_pairs[m].append((gt["hotspots"][gi]["risk_level"],
                                              res.hotspots[pi].risk_level))

    return {
        "n_total": len(gts),
        "n_pos": sum(1 for g in gts if g["has_hotspot"]),
        "n_neg": sum(1 for g in gts if not g["has_hotspot"]),
        "img_conf": img_conf,
        "loc_ious": loc_ious,
        "risk_pairs": risk_pairs,
        "scen_conf": scen_conf,
        "false_pos": false_pos,
        "false_neg": false_neg,
    }


def prf(conf: dict) -> dict:
    tp, fp, tn, fn = conf["TP"], conf["FP"], conf["TN"], conf["FN"]
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r)
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}


# --------------------------------------------------------------------------- #
# Output writers                                                               #
# --------------------------------------------------------------------------- #
def write_confusion_png(conf: dict, method: str, path: Path) -> None:
    mat = np.array([[conf["TP"], conf["FN"]], [conf["FP"], conf["TN"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred +", "Pred -"])
    ax.set_yticklabels(["GT +", "GT -"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > mat.max() / 2 else "black", fontsize=16)
    ax.set_title(f"Image-level confusion — {method}")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def write_visual_checks(rng: random.Random) -> list[Path]:
    gts = json.loads(GT_FILE.read_text(encoding="utf-8"))
    pos = [g for g in gts if g["has_hotspot"]]
    neg = [g for g in gts if not g["has_hotspot"]]
    picks = rng.sample(pos, 3) + rng.sample(neg, 3)
    vc_dir = OUT_DIR / "visual_check"
    vc_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for gt in picks:
        thermal = np.load(npy_path(gt))
        res = detect_hotspots(thermal, method="hybrid")
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
        for ax in (axl, axr):
            ax.imshow(thermal, cmap="jet", vmin=-10, vmax=150)
        axl.set_title(f"GT: {gt['image_id']} ({gt['scenario']})")
        for h in gt["hotspots"]:
            x, y, w, h_ = h["bbox"]
            axl.add_patch(patches.Rectangle((x, y), w, h_, lw=2, edgecolor="white",
                                            facecolor="none", linestyle="--"))
        axr.set_title(f"Hybrid pred: {len(res.hotspots)} hotspot(s)")
        for hs in res.hotspots:
            x, y, w, h_ = hs.bbox
            axr.add_patch(patches.Rectangle((x, y), w, h_, lw=2, edgecolor="red",
                                            facecolor="none"))
            axr.text(x, y - 5, f"{hs.peak_temp_c:.0f}°C/{hs.risk_level}",
                     color="red", fontsize=9)
        plt.tight_layout()
        out = vc_dir / f"check_{gt['image_id']}_{gt['scenario'][:2]}.png"
        plt.savefig(out, dpi=90)
        plt.close(fig)
        saved.append(out)
    return saved


def build_report(results: dict) -> str:
    n_total, n_pos, n_neg = results["n_total"], results["n_pos"], results["n_neg"]
    metrics = {m: prf(results["img_conf"][m]) for m in METHODS}

    L = []
    L.append("=" * 60)
    L.append("HOTSPOT DETECTION EVALUATION")
    L.append("=" * 60)
    L.append(f"Toplam görüntü: {n_total}")
    L.append(f"Hotspot içeren: {n_pos}")
    L.append(f"Hotspot içermeyen: {n_neg}")
    L.append("")
    L.append("--- IMAGE-LEVEL DETECTION (hotspot var/yok) ---")
    L.append("")
    L.append(f"{'':<11}| {'Absolute':^10}| {'Statistical':^12}| {'Hybrid':^8}")
    L.append("-" * 11 + "+" + "-" * 11 + "+" + "-" * 13 + "+" + "-" * 9)
    for label, key in [("Precision", "precision"), ("Recall", "recall"),
                       ("F1 score", "f1"), ("Accuracy", "accuracy")]:
        L.append(f"{label:<11}|{metrics['absolute'][key]:^11.3f}|"
                 f"{metrics['statistical'][key]:^13.3f}|{metrics['hybrid'][key]:^9.3f}")
    L.append("")
    L.append("Raw confusion (TP/FP/TN/FN):")
    for m in METHODS:
        c = results["img_conf"][m]
        L.append(f"  {m:<12}: TP={c['TP']} FP={c['FP']} TN={c['TN']} FN={c['FN']}")

    L.append("")
    L.append("--- LOCALIZATION (bbox IoU, GT hotspots üzerinden) ---")
    L.append("")
    L.append(f"{'':<11}| {'Absolute':^10}| {'Statistical':^12}| {'Hybrid':^8}")
    L.append("-" * 11 + "+" + "-" * 11 + "+" + "-" * 13 + "+" + "-" * 9)
    mean_iou = {m: (float(np.mean(results["loc_ious"][m])) if results["loc_ious"][m] else 0.0)
                for m in METHODS}
    iou05 = {m: (float(np.mean([v >= IOU_MATCH for v in results["loc_ious"][m]]))
                 if results["loc_ious"][m] else 0.0) for m in METHODS}
    L.append(f"{'Mean IoU':<11}|{mean_iou['absolute']:^11.3f}|"
             f"{mean_iou['statistical']:^13.3f}|{mean_iou['hybrid']:^9.3f}")
    L.append(f"{'IoU > 0.5':<11}|{iou05['absolute']:^11.3f}|"
             f"{iou05['statistical']:^13.3f}|{iou05['hybrid']:^9.3f}")

    L.append("")
    L.append("--- RISK LEVEL CLASSIFICATION (IoU>0.5 eşleşmeler) ---")
    for m in METHODS:
        L.append(f"\n  [{m}] (satır=GT, sütun=Pred)")
        idx = {r: i for i, r in enumerate(RISK_LEVELS)}
        mat = np.zeros((4, 4), dtype=int)
        for gtr, pr in results["risk_pairs"][m]:
            mat[idx[gtr]][idx[pr]] += 1
        header = "    " + "".join(f"{r[:4]:>7}" for r in RISK_LEVELS)
        L.append(header)
        for i, r in enumerate(RISK_LEVELS):
            L.append(f"  {r[:4]:<4}" + "".join(f"{mat[i][j]:>7}" for j in range(4)))

    L.append("")
    L.append("--- SCENARIO-WISE PERFORMANCE (Hybrid method) ---")
    L.append("")
    L.append(f"{'Scenario':<30}| {'Prec':>5} | {'Recall':>6} | {'F1':>5} | {'FP':>3} {'FN':>3}")
    L.append("-" * 30 + "+" + "-" * 7 + "+" + "-" * 8 + "+" + "-" * 7 + "+" + "-" * 8)
    for scenario in sorted(results["scen_conf"]["hybrid"]):
        c = results["scen_conf"]["hybrid"][scenario]
        mm = prf(c)
        note = "  ← false-positive trap" if scenario in (
            "07_AmbientNoise_NoFault", "08_SunWarming_FalsePositive") else ""
        # recall undefined for all-negative scenarios -> show '-'
        pos_in = c["TP"] + c["FN"]
        rec = f"{mm['recall']:.3f}" if pos_in else "  -  "
        prec = f"{mm['precision']:.3f}" if (c["TP"] + c["FP"]) else "  -  "
        f1s = f"{mm['f1']:.3f}" if pos_in and (c["TP"] + c["FP"]) else "  -  "
        L.append(f"{scenario:<30}| {prec:>5} | {rec:>6} | {f1s:>5} | "
                 f"{c['FP']:>3} {c['FN']:>3}{note}")

    best = max(METHODS, key=lambda m: metrics[m]["f1"])
    L.append("")
    L.append(f"--- BEST METHOD: {best} ({metrics[best]['f1']:.3f} F1) ---")
    L.append("=" * 60)
    return "\n".join(L)


def write_csvs(results: dict) -> None:
    metrics = {m: prf(results["img_conf"][m]) for m in METHODS}
    mean_iou = {m: (float(np.mean(results["loc_ious"][m])) if results["loc_ious"][m] else 0.0)
                for m in METHODS}
    iou05 = {m: (float(np.mean([v >= IOU_MATCH for v in results["loc_ious"][m]]))
                 if results["loc_ious"][m] else 0.0) for m in METHODS}
    with (OUT_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + METHODS)
        for label, key in [("precision", "precision"), ("recall", "recall"),
                           ("f1", "f1"), ("accuracy", "accuracy")]:
            w.writerow([label] + [f"{metrics[m][key]:.4f}" for m in METHODS])
        w.writerow(["mean_iou"] + [f"{mean_iou[m]:.4f}" for m in METHODS])
        w.writerow(["iou_gt_0.5"] + [f"{iou05[m]:.4f}" for m in METHODS])
        for cell in ["TP", "FP", "TN", "FN"]:
            w.writerow([cell] + [results["img_conf"][m][cell] for m in METHODS])

    with (OUT_DIR / "per_scenario.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "method", "TP", "FP", "TN", "FN",
                    "precision", "recall", "f1"])
        for m in METHODS:
            for scenario in sorted(results["scen_conf"][m]):
                c = results["scen_conf"][m][scenario]
                mm = prf(c)
                w.writerow([scenario, m, c["TP"], c["FP"], c["TN"], c["FN"],
                            f"{mm['precision']:.4f}", f"{mm['recall']:.4f}", f"{mm['f1']:.4f}"])


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")
    logger.add(LOG_DIR / "evaluate_hotspot.log", level="DEBUG", mode="w", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hotspot detectors on synthetic data.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    setup_logging()
    random.seed(SEED)
    np.random.seed(SEED)

    if not GT_FILE.exists():
        logger.error(f"Ground truth not found: {GT_FILE}. Run generate_synthetic_thermal.py first.")
        sys.exit(1)

    if args.clean and OUT_DIR.exists():
        logger.info(f"--clean: removing {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating 3 methods on all images ...")
    results = evaluate()

    for m in METHODS:
        write_confusion_png(results["img_conf"][m], m, OUT_DIR / f"confusion_matrix_{m}.png")
    write_csvs(results)
    (OUT_DIR / "false_positives.json").write_text(
        json.dumps(results["false_pos"], indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "false_negatives.json").write_text(
        json.dumps(results["false_neg"], indent=2, ensure_ascii=False), encoding="utf-8")

    report = build_report(results)
    (OUT_DIR / "summary_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info("\n" + report)

    saved = write_visual_checks(random.Random(SEED))
    logger.info(f"Visual checks: {[p.name for p in saved]}")
    logger.success(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
