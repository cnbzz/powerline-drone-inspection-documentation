---
name: venv-python-environment
description: Project .venv is Python 3.11, has no pip, missing tqdm/pycocotools
metadata:
  type: project
---

The project `.venv` (invoke as `./.venv/Scripts/python.exe` from repo root) is Python 3.11 (uv-managed). It has `loguru`, `Pillow`, `numpy`, `cv2`, `pyyaml` but **NO `pip`** (`No module named pip`), and is missing `tqdm` and `pycocotools`.

**How to apply:** Don't assume `pip install` works in the venv. Parse COCO JSON manually instead of pycocotools; make `tqdm` an optional import with a passthrough fallback. On the Windows console, force UTF-8 (`sys.stdout.reconfigure(encoding="utf-8")` or `PYTHONIOENCODING=utf-8`) or Turkish chars / ✓ raise `UnicodeEncodeError` under cp1252. The bare `python` command isn't on PATH; system launcher is `py` (3.14). See [[yolo-dataset-prep]].
