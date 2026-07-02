---
name: yolo-training
description: How YOLOv8m is trained (scripts/train_yolo.py) + Ultralytics path/VRAM gotchas
metadata:
  type: project
---

`scripts/train_yolo.py` trains YOLOv8m on the 8-class dataset. Defaults live in `config/model_config.yaml` under `yolo.training_run` (50 epochs, imgsz 640, batch 16, device 0, seed 42, amp). CLI overrides any field. Flags: `--validate-only` (5s dataset check, no training), `--resume` (from `models/trained/yolov8m_8class_v1/weights/last.pt`). Output → `models/trained/yolov8m_8class_v1/`. Logs → `outputs/logs/train_yolo.log`, errors → `train_error.log`. Has auto CUDA-OOM batch fallback 16→8→4 and optional W&B (only if `WANDB_API_KEY` set).

**Gotcha 1 (Ultralytics relative project):** Ultralytics nests a *relative* `project` under its `runs_dir` setting (`runs`), so `project=models/trained` became `runs/detect/models/trained/...`. Fix: the script resolves `project` to an ABSOLUTE path before calling `model.train()`.

**Gotcha 1b (TensorBoard off by default):** Ultralytics `settings.json` ships `tensorboard: false` in 8.4.x, so NO `events.out.tfevents.*` is written and TensorBoard shows "No dashboards are active". `train_yolo.py` `configure_loggers()` now force-sets `settings.update({"tensorboard": True})` at startup. Flipping it mid-run does NOT add TB to the live trainer (callbacks bind at init) — must restart. Verify an event file appears under `models/trained/<name>/` after launch.

**Gotcha 2 (VRAM):** RTX 4060 Laptop has only 8 GB (8.6 GB reported, ~8188 MiB usable). yolov8m @ batch16/imgsz640/AMP uses ~6.4 GB — fits with headroom, OOM fallback not triggered. ~5 min/epoch → ~4 h for 50 epochs.

**Final run (v1, 50 epochs, 4.69h):** best val mAP50=0.798, mAP50-95=0.664, P=0.738, R=0.867. Per-class mAP50-95 strong: tower_id_plate 0.88, porcelain 0.88, polymer 0.87, damper 0.79, glass 0.72. Weak (low data): tower 0.49, insulator_generic 0.39, spacer 0.30 — these map to the smallest classes (tower 253, generic 312, spacer 347 labels). Improve via more data / class weighting for those 3 — see [[future-improvements]]. **Test set (1310 unseen):** mAP50=0.829, mAP50-95=0.684 — slightly ABOVE val, so no overfitting, good generalization. Test eval plots in `models/trained/yolov8m_8class_v1/test_eval/`; sample inference in `outputs/predictions/test_samples/`. See [[yolo-dataset-prep]] and [[venv-python-environment]].
