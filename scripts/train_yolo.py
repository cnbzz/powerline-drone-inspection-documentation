"""Train YOLOv8m on the unified 8-class powerline dataset.

Defaults are read from ``config/model_config.yaml`` (``yolo.training_run``) and can be
overridden from the CLI. Features:

* Dataset pre-flight validation (``--validate-only`` runs *only* this, no training).
* Automatic CUDA-OOM batch fallback: 16 -> 8 -> 4.
* TensorBoard logging (Ultralytics built-in) + optional W&B (if ``WANDB_API_KEY`` set).
* Resume from the last checkpoint with ``--resume``.

Examples::

    uv run python scripts/train_yolo.py --validate-only   # 5s dataset check, no training
    uv run python scripts/train_yolo.py                   # full 50-epoch run
    uv run python scripts/train_yolo.py --resume          # continue from last.pt
    uv run python scripts/train_yolo.py --epochs 10 --batch 8   # CLI overrides
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

# --- Force UTF-8 console (cp1252 chokes on Turkish glyphs / ✓) --------------- #
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
TRAIN_LOG = LOG_DIR / "train_yolo.log"
ERROR_LOG = LOG_DIR / "train_error.log"

# OOM fallback ladder.
BATCH_FALLBACKS = [8, 4]

EXPECTED_NC = 8
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def load_config() -> dict[str, Any]:
    """Read ``yolo.training_run`` defaults from config/model_config.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    try:
        run_cfg = cfg["yolo"]["training_run"]
    except (KeyError, TypeError) as exc:
        raise KeyError("config/model_config.yaml missing 'yolo.training_run' block") from exc
    return dict(run_cfg)


def build_arg_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train YOLOv8m on the 8-class powerline dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=defaults["model"])
    p.add_argument("--data", default=defaults["data"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--imgsz", type=int, default=defaults["imgsz"])
    p.add_argument("--batch", type=int, default=defaults["batch"])
    p.add_argument("--device", default=str(defaults["device"]))
    p.add_argument("--workers", type=int, default=defaults["workers"])
    p.add_argument("--project", default=defaults["project"])
    p.add_argument("--name", default=defaults["name"])
    p.add_argument("--patience", type=int, default=defaults["patience"])
    p.add_argument("--save-period", type=int, default=defaults["save_period"])
    p.add_argument("--optimizer", default=defaults["optimizer"])
    p.add_argument("--seed", type=int, default=defaults["seed"])
    p.add_argument("--cache", action="store_true", default=defaults["cache"])
    p.add_argument("--no-amp", dest="amp", action="store_false", default=defaults["amp"])
    p.add_argument("--validate-only", action="store_true",
                   help="Run dataset validation only, then exit (no training).")
    p.add_argument("--resume", action="store_true",
                   help="Resume training from the last checkpoint (weights/last.pt).")
    return p


# --------------------------------------------------------------------------- #
# Dataset validation                                                           #
# --------------------------------------------------------------------------- #
def validate_dataset(data_yaml: Path) -> dict[str, int]:
    """Validate data.yaml and split folders. Returns per-split image counts."""
    logger.info(f"Validating dataset: {data_yaml}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    with data_yaml.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    nc = data.get("nc")
    if nc != EXPECTED_NC:
        raise ValueError(f"data.yaml nc={nc}, expected {EXPECTED_NC}")
    names = data.get("names")
    if not names or len(names) != EXPECTED_NC:
        raise ValueError(f"data.yaml 'names' must list {EXPECTED_NC} classes, got {names}")

    # Resolve dataset root the same way Ultralytics does (relative to data.yaml dir).
    base = data_yaml.parent
    root = (base / data["path"]).resolve() if data.get("path") else base

    counts: dict[str, int] = {}
    img_exts = {".jpg", ".jpeg", ".png"}
    for split in SPLITS:
        rel = data.get(split)
        if not rel:
            raise ValueError(f"data.yaml missing '{split}' entry")
        img_dir = (root / rel).resolve()
        if not img_dir.is_dir():
            # Fall back: the 'path' field may not resolve; try relative to data.yaml dir.
            alt = (base / rel).resolve()
            img_dir = alt if alt.is_dir() else img_dir
        if not img_dir.is_dir():
            raise FileNotFoundError(f"{split} image dir not found: {img_dir}")
        n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in img_exts)
        if n == 0:
            raise ValueError(f"{split} image dir is empty: {img_dir}")
        counts[split] = n

    logger.success(
        f"Dataset OK | nc={nc} | "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )
    return counts


# --------------------------------------------------------------------------- #
# W&B                                                                          #
# --------------------------------------------------------------------------- #
def configure_loggers() -> bool:
    """Force-enable TensorBoard; enable W&B only if WANDB_API_KEY is present.

    Ultralytics' settings.json defaults ``tensorboard`` to False in recent versions,
    which silently disables event-file logging. We flip it on here so TensorBoard
    always works. Returns True if W&B was enabled.
    """
    try:
        from ultralytics import settings as ul_settings  # type: ignore
    except Exception:  # pragma: no cover
        ul_settings = None

    if ul_settings is not None and not ul_settings.get("tensorboard"):
        try:
            ul_settings.update({"tensorboard": True})
            logger.info("Enabled TensorBoard logging in Ultralytics settings.")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not enable TensorBoard setting: {exc}")

    if os.environ.get("WANDB_API_KEY"):
        if ul_settings is not None:
            try:
                ul_settings.update({"wandb": True})
            except Exception:
                pass
        logger.info("W&B enabled (WANDB_API_KEY detected).")
        return True

    if ul_settings is not None:
        try:
            ul_settings.update({"wandb": False})
        except Exception:
            pass
    logger.info("W&B disabled, set WANDB_API_KEY to enable.")
    return False


# --------------------------------------------------------------------------- #
# Summary banner                                                               #
# --------------------------------------------------------------------------- #
def print_banner(args: argparse.Namespace, counts: dict[str, int], device_name: str) -> None:
    banner = "\n".join([
        "=" * 60,
        "YOLO TRAINING SESSION",
        "=" * 60,
        f"Model: {args.model} (pretrained)",
        f"Dataset: {args.data} ({EXPECTED_NC} classes)",
        f"Train images: {counts['train']}",
        f"Val images:   {counts['val']}",
        f"Test images:  {counts['test']}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch}",
        f"Image size: {args.imgsz}",
        f"Device: {device_name}",
        "Estimated time: 3-5 hours",
        f"Output: {args.project}/{args.name}/",
        "=" * 60,
    ])
    print("\n" + banner + "\n")
    logger.info("\n" + banner)


def device_label(device: str) -> str:
    try:
        import torch
        if device not in ("cpu", "") and torch.cuda.is_available():
            return torch.cuda.get_device_name(int(device.split(",")[0]))
    except Exception:
        pass
    return device or "cpu"


# --------------------------------------------------------------------------- #
# Training with OOM fallback                                                    #
# --------------------------------------------------------------------------- #
def is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg or "cublas" in msg


def run_training(args: argparse.Namespace):
    """Train, automatically reducing batch on CUDA OOM (16 -> 8 -> 4)."""
    from ultralytics import YOLO

    batch_ladder = [args.batch] + [b for b in BATCH_FALLBACKS if b < args.batch]
    last_exc: Optional[Exception] = None

    for attempt, batch in enumerate(batch_ladder):
        try:
            if attempt > 0:
                logger.warning(f"Retrying with reduced batch size: {batch}")
            else:
                logger.info(f"Starting training with batch size: {batch}")

            if args.resume:
                # When resuming, Ultralytics needs the checkpoint as the model arg.
                ckpt = Path(args.project) / args.name / "weights" / "last.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(f"--resume given but checkpoint missing: {ckpt}")
                logger.info(f"Resuming from {ckpt}")
                model = YOLO(str(ckpt))
                results = model.train(resume=True)
                return model, results

            model = YOLO(args.model)
            results = model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch,
                device=args.device,
                workers=args.workers,
                project=args.project,
                name=args.name,
                patience=args.patience,
                save=True,
                save_period=args.save_period,
                cache=args.cache,
                optimizer=args.optimizer,
                verbose=True,
                seed=args.seed,
                deterministic=True,
                pretrained=True,
                amp=args.amp,
                exist_ok=True,
            )
            return model, results

        except Exception as exc:  # noqa: BLE001 - non-OOM re-raised below
            last_exc = exc
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            if is_oom(exc) and not args.resume and attempt < len(batch_ladder) - 1:
                logger.error(f"CUDA OOM at batch={batch}. Falling back.")
                continue
            raise

    # Exhausted the ladder.
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# Post-training report                                                          #
# --------------------------------------------------------------------------- #
def report_results(model, args: argparse.Namespace) -> None:
    out_dir = Path(args.project) / args.name
    best = out_dir / "weights" / "best.pt"

    lines = ["", "=" * 60, "TRAINING COMPLETE", "=" * 60]
    try:
        metrics = model.metrics  # validation metrics of the best/last epoch
        box = getattr(metrics, "box", None)
        if box is not None:
            lines.append("Best model metrics (val):")
            lines.append(f"  mAP50-95 : {box.map:.4f}")
            lines.append(f"  mAP50    : {box.map50:.4f}")
            lines.append(f"  Precision: {box.mp:.4f}")
            lines.append(f"  Recall   : {box.mr:.4f}")

            lines.append("")
            lines.append("Per-class mAP50-95:")
            names = model.names if hasattr(model, "names") else {}
            try:
                for cls_id in box.ap_class_index:
                    name = names.get(int(cls_id), str(cls_id))
                    lines.append(f"  {int(cls_id)} {name:<20} {box.maps[int(cls_id)]:.4f}")
            except Exception as exc:  # pragma: no cover
                lines.append(f"  (per-class unavailable: {exc})")
    except Exception as exc:  # pragma: no cover
        lines.append(f"(metrics unavailable: {exc})")

    lines.append("")
    lines.append(f"Best weights: {best}")
    lines.append("")
    lines.append("TensorBoard görmek için yeni terminal aç ve şu komutu çalıştır:")
    lines.append(f"  uv run tensorboard --logdir {out_dir}")
    lines.append("")
    lines.append("Crash olursa eğitime devam etmek için:")
    lines.append("  uv run python scripts/train_yolo.py --resume")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)
    logger.info("\n" + report)


# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")
    logger.add(TRAIN_LOG, level="DEBUG", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}")


def log_error(stage: str, exc: Exception) -> None:
    """Write a clear error report + full stack trace to outputs/logs/train_error.log."""
    etype = type(exc).__name__
    suggestion = "Check the stack trace below."
    if is_oom(exc):
        etype = "CUDA OutOfMemory"
        suggestion = "Reduce --batch (e.g. --batch 4) or --imgsz, or close other GPU apps."
    elif isinstance(exc, FileNotFoundError):
        suggestion = "Verify the missing path exists (data.yaml, checkpoint, image dirs)."
    elif isinstance(exc, (ValueError, KeyError)):
        suggestion = "Fix config/model_config.yaml or data/yolo/data.yaml contents."

    tb = traceback.format_exc()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.write_text(
        f"ERROR TYPE : {etype}\nSTAGE      : {stage}\nSUGGESTION : {suggestion}\n\n{tb}",
        encoding="utf-8",
    )
    logger.error(f"[{stage}] {etype}: {exc}")
    logger.error(f"Suggested fix: {suggestion}")
    logger.error(f"Full stack trace saved to {ERROR_LOG}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    setup_logging()
    stage = "init"
    try:
        defaults = load_config()
        args = build_arg_parser(defaults).parse_args()

        # Ultralytics nests a *relative* project under its runs_dir (e.g.
        # runs/detect/<project>). Force an absolute path so outputs land exactly
        # at models/trained/<name>/ as required.
        if not Path(args.project).is_absolute():
            args.project = str((PROJECT_ROOT / args.project).resolve())

        stage = "dataset-validation"
        data_yaml = (PROJECT_ROOT / args.data).resolve() if not Path(args.data).is_absolute() \
            else Path(args.data)
        counts = validate_dataset(data_yaml)

        if args.validate_only:
            logger.success("--validate-only: dataset checks passed. Not training. Exiting.")
            print("\nDataset validation OK ✓  (train={}, val={}, test={})".format(
                counts["train"], counts["val"], counts["test"]))
            return

        stage = "logger-setup"
        configure_loggers()

        print_banner(args, counts, device_label(args.device))

        stage = "training"
        model, _ = run_training(args)

        stage = "reporting"
        report_results(model, args)
        logger.success("Done.")

    except Exception as exc:  # noqa: BLE001 - top-level handler reports & exits
        log_error(stage, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
