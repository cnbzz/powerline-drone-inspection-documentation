---
name: insplad-duplicate-filenames
description: InsPLAD-det train COCO JSON has 46 duplicate file_name records (data leakage risk)
metadata:
  type: project
---

`data/external/insplad/annotations/instances_train.json` lists 7981 image records but only 7935 unique `file_name`s exist on disk — 46 physical images are referenced by two different `image_id`s (each carrying a subset of annotations).

**Why:** If you key images by `image_id` when splitting, the same physical image can land in different splits (e.g. train AND test) → data leakage, and you emit duplicate image copies.

**How to apply:** When parsing InsPLAD COCO, key images by `file_name` and merge annotations across records sharing a name. `scripts/prepare_yolo_dataset.py` does this in `_load_coco`. val JSON (2626 records) has no duplicates. See [[yolo-dataset-prep]].
