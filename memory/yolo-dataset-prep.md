---
name: yolo-dataset-prep
description: How the unified 8-class YOLO dataset is built from 3 source datasets
metadata:
  type: project
---

`scripts/prepare_yolo_dataset.py` merges 3 read-only sources under `data/external/` into `data/yolo/{train,val,test}/{images,labels}` + `data.yaml` (8 classes). Images are COPIED, never moved. Flags: `--dry-run`, `--clean`. Seed 42. Log → `outputs/logs/prepare_yolo.log`, sanity bbox renders → `outputs/sanity/`.

Sources & splits:
- **InsPLAD-det** (COCO, `insplad/annotations/instances_{train,val}.json`, images in `insplad/{train,val}/`): map by category NAME to ids 0/1/5/6/7, skip yoke/lightning rod/vari-grip/sphere; train json → 90% train + 10% test, val json → val. See [[insplad-duplicate-filenames]].
- **STN-PLAD** (single `stn_plad/annotations.json`, 133 imgs, file_name points into `20181127-A1/` & `20181129-R/`): map by category ID (0→4,1→3,2→5,3→6,5→7), 80/10/10.
- **MPID** (YOLO txt, class always 0): `composite_MPID`→0, `glass_MPID`→1, `porcelain_MPID`→2; each 80/10/10.

Result: 11262 train / 3141 val / 1310 test = 15713 images. Output names are short prefixed sequentials (e.g. `insplad_000123.jpg`) to dodge Windows MAX_PATH. See [[venv-python-environment]].
