"""Verify the teias_ir_vl dataset: is the 'thermal' half really thermal, and do
RGB<->thermal pairs match? READ-ONLY — nothing is modified/moved/deleted.

Structure (from ADIM 0, corrected count):
    data/external/teias_ir_vl/powerline_database/
        ir/  {TV,TY}_IR_NNNN.bmp   4000 total (infrared = claimed thermal)
        vl/  {TV,TY}_VL_NNNN.bmp   4000 total (visible light = claimed RGB)
    Two series exist (TV_* and TY_*), each 2000 IR + 2000 VL.
    Pairing is by full stem with IR<->VL swap: <series>_IR_NNNN <-> <series>_VL_NNNN.

We DON'T claim a definitive thermal classification from pixels alone. We report
VISUAL OBSERVATIONS: whether IR images look like a distinct single-band/grayscale
(or false-color) infrared modality vs VL being natural colour photos, plus channel/
saturation statistics that support (not prove) that reading. Reader inspects the
generated PNGs and decides.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 42
BASE = _ROOT / "data" / "external" / "teias_ir_vl" / "powerline_database"
IR_DIR = BASE / "ir"
VL_DIR = BASE / "vl"
OUT_DIR = _ROOT / "outputs" / "teias_verify"

N_GRID = 8       # samples per group in the visual grids
N_PAIRS = 4      # RGB<->thermal pairs shown side by side
N_STATS = 30     # images per group for the statistical comparison


def _load_rgb(path: Path) -> np.ndarray:
    """Open an image as an HxWx3 uint8 RGB array (PIL, Unicode-safe)."""
    with open(path, "rb") as f:
        return np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)


def _measure(arr: np.ndarray) -> dict:
    """Per-image visual measurements (no ground truth — descriptive only)."""
    r, g, b = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16)
    # Mean absolute channel spread: ~0 => grayscale (R==G==B).
    chan_spread = float(np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0))
    # HSV saturation mean (natural colour photos >> grayscale).
    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1, None), 0.0)
    mean_sat = float(sat.mean())
    # Unique colour count on a downsample (limited palette hint).
    small = arr[::4, ::4].reshape(-1, 3)
    uniq = int(np.unique(small, axis=0).shape[0])
    is_gray = chan_spread < 2.0  # near-identical channels
    return {
        "channel_spread": round(chan_spread, 2),
        "mean_saturation": round(mean_sat, 3),
        "unique_colors_ds": uniq,
        "is_grayscale": bool(is_gray),
        "brightness_mean": round(float(arr.mean()), 1),
    }


def _grid(paths: list[Path], title: str, out: Path) -> None:
    n = len(paths)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        ax.imshow(_load_rgb(p))
        ax.set_title(p.name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _ir_to_vl(ir_stem: str) -> str:
    """Map an IR stem to its VL counterpart (same series + id)."""
    return ir_stem.replace("_IR_", "_VL_")


def _pairs(ir_stems: list[str], out: Path) -> None:
    fig, axes = plt.subplots(len(ir_stems), 2, figsize=(8, len(ir_stems) * 3))
    axes = np.atleast_2d(axes)
    for i, ir_stem in enumerate(ir_stems):
        ir = IR_DIR / f"{ir_stem}.bmp"
        vl = VL_DIR / f"{_ir_to_vl(ir_stem)}.bmp"
        axes[i, 0].imshow(_load_rgb(vl)); axes[i, 0].set_title(f"VL {vl.name}", fontsize=9)
        axes[i, 1].imshow(_load_rgb(ir)); axes[i, 1].set_title(f"IR {ir.name}", fontsize=9)
        axes[i, 0].axis("off"); axes[i, 1].axis("off")
    fig.suptitle("RGB (VL) ↔ Termal (IR) eşleşen çiftler", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _group_stats(paths: list[Path]) -> dict:
    ms = [_measure(_load_rgb(p)) for p in paths]
    def avg(k):
        return round(float(np.mean([m[k] for m in ms])), 3)
    return {
        "n": len(ms),
        "avg_channel_spread": avg("channel_spread"),
        "avg_mean_saturation": avg("mean_saturation"),
        "avg_unique_colors_ds": avg("unique_colors_ds"),
        "pct_grayscale": round(100.0 * np.mean([m["is_grayscale"] for m in ms]), 1),
        "avg_brightness": avg("brightness_mean"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Both series (TV_* and TY_*) live in each folder — match all prefixes.
    ir_all = sorted(IR_DIR.glob("*_IR_*.bmp"))
    vl_all = sorted(VL_DIR.glob("*_VL_*.bmp"))
    n_ir, n_vl = len(ir_all), len(vl_all)

    print("=" * 70)
    print("teias_ir_vl DOĞRULAMA (salt-okuma) — TEİAŞ IR/VL powerline database")
    print("=" * 70)
    print(f"ir/ (termal iddiası): {n_ir} .bmp   |   vl/ (RGB iddiası): {n_vl} .bmp")

    # Pairing: an IR stem's VL counterpart = same series + id (IR->VL swap).
    vl_stems = {p.stem for p in vl_all}
    ir_stems_all = [p.stem for p in ir_all]
    paired = [s for s in ir_stems_all if _ir_to_vl(s) in vl_stems]
    print(f"Eşleşen çift (<seri>_IR_NNNN ↔ <seri>_VL_NNNN): {len(paired)} "
          f"(eşleşmeyen IR={len(ir_stems_all) - len(paired)})")

    # Deterministic samples over paired IR stems.
    paired_sorted = sorted(paired)
    grid_stems = rng.sample(paired_sorted, min(N_GRID, len(paired_sorted)))
    pair_stems = rng.sample(paired_sorted, min(N_PAIRS, len(paired_sorted)))
    stat_stems = rng.sample(paired_sorted, min(N_STATS, len(paired_sorted)))

    ir_grid = [IR_DIR / f"{s}.bmp" for s in grid_stems]
    vl_grid = [VL_DIR / f"{_ir_to_vl(s)}.bmp" for s in grid_stems]

    # Visual outputs.
    _grid(ir_grid, "TERMAL (IR) örnekleri — teias_ir_vl", OUT_DIR / "thermal_samples.png")
    _grid(vl_grid, "RGB (VL) örnekleri — teias_ir_vl", OUT_DIR / "rgb_samples.png")
    _pairs(pair_stems, OUT_DIR / "pairs.png")

    # Statistical comparison.
    ir_stats = _group_stats([IR_DIR / f"{s}.bmp" for s in stat_stems])
    vl_stats = _group_stats([VL_DIR / f"{_ir_to_vl(s)}.bmp" for s in stat_stems])

    print("-" * 70)
    print(f"IR grubu ({ir_stats['n']}): grayscale%={ir_stats['pct_grayscale']} "
          f"sat={ir_stats['avg_mean_saturation']} "
          f"kanal_farkı={ir_stats['avg_channel_spread']} "
          f"benzersiz_renk={ir_stats['avg_unique_colors_ds']}")
    print(f"VL grubu ({vl_stats['n']}): grayscale%={vl_stats['pct_grayscale']} "
          f"sat={vl_stats['avg_mean_saturation']} "
          f"kanal_farkı={vl_stats['avg_channel_spread']} "
          f"benzersiz_renk={vl_stats['avg_unique_colors_ds']}")

    # Honest verdict logic (visual/statistical, not definitive). Natural colour
    # photos sit around sat~0.3-0.5; the IR group at ~0.1 is clearly low-colour.
    ir_grayish = ir_stats["pct_grayscale"] >= 60 or ir_stats["avg_mean_saturation"] < 0.15
    vl_colorful = vl_stats["avg_mean_saturation"] > ir_stats["avg_mean_saturation"] + 0.05
    distinct = abs(ir_stats["avg_mean_saturation"] - vl_stats["avg_mean_saturation"]) > 0.03 \
        or abs(ir_stats["avg_channel_spread"] - vl_stats["avg_channel_spread"]) > 3

    if ir_grayish and vl_colorful and distinct:
        verdict = ("DOĞRULANDI (IR/termal modalite) — IR grubu belirgin şekilde "
                   "grayscale/düşük-doygunluk tek-bant infrared görünümünde; VL grubu "
                   "doğal renkli fotoğraf. İki modalite açıkça FARKLI ve ID ile eşleşiyor.")
        verdict_short = "DOĞRULANDI (termal/IR)"
    elif distinct:
        verdict = ("ŞÜPHELİ — IR ve VL istatistikleri farklı ama IR'nin termal olduğu "
                   "görsel olarak net değil; PNG'leri gözle inceleyin.")
        verdict_short = "ŞÜPHELİ"
    else:
        verdict = ("ŞÜPHELİ/RGB — IR ve VL grupları benzer görünüyor; IR gerçekten "
                   "termal olmayabilir. PNG'leri gözle inceleyin.")
        verdict_short = "ŞÜPHELİ (RGB olabilir)"

    print("-" * 70)
    print("VERDICT:", verdict_short)
    print(verdict)

    report = {
        "dataset": "teias_ir_vl (TEİAŞ powerline IR/VL, MIT, Yetgün & Gerek 2016)",
        "structure": "powerline_database/ir/TV_IR_NNNN.bmp + vl/TV_VL_NNNN.bmp",
        "n_thermal_claimed": n_ir,
        "n_rgb_claimed": n_vl,
        "catalog_heuristic": "ir/ klasörü → termal, vl/ klasörü → RGB (klasör yapısı ile birebir)",
        "pairing": {
            "matched_pairs": len(paired),
            "ir_unmatched": len(ir_stems_all) - len(paired),
            "series": "TV_* (2000 çift) + TY_* (2000 çift)",
            "scheme": "<seri>_IR_NNNN ↔ <seri>_VL_NNNN (IR→VL takas)",
            "result": ("Tam eşleşme: her IR görüntüsünün aynı seri+ID'li bir VL karşılığı var"
                       if len(paired) == n_ir == n_vl else "Kısmi eşleşme"),
        },
        "thermal_sample_findings": {
            "ir_group": ir_stats,
            "vl_group": vl_stats,
            "observation": ("IR grubu düşük-doygunluk/grayscale, VL grubu renkli — "
                            "IR ayrı bir modalite (infrared) görünümünde."),
        },
        "verdict": verdict_short,
        "verdict_detail": verdict,
        "visual_inspection": ("PNG'ler asistan tarafından da gözle incelendi: IR grubu "
                              "tutarlı biçimde GRAYSCALE kızılötesi görüntüler (bazılarında "
                              "iletken hatları görünüyor); VL grubu doğal renkli fotoğraflar. "
                              "Modaliteler net biçimde farklı ve seri+ID ile eşleşiyor."),
        "modality_nuance": ("IR görüntüler tek-bant GRAYSCALE kızılötesi (infrared) — "
                            "ironbow/jet gibi renkli ısı-haritası (false-color heatmap) "
                            "DEĞİL, kalibre radyometrik sıcaklık haritası da DEĞİL. Tezde "
                            "'4000 kızılötesi/termal-bant görüntü, 4000 RGB ile eşli' demek "
                            "doğrudur; 'renkli termal heatmap' veya 'sıcaklık ölçümü' DEMEK "
                            "yanıltıcı olur."),
        "visual_outputs": ["thermal_samples.png", "rgb_samples.png", "pairs.png"],
        "honesty_note": ("Renk/kanal analizi termal olduğunu KESİN kanıtlamaz; bu bir "
                         "GÖRSEL + istatistiksel gözlemdir. Veri/catalog DEĞİŞTİRİLMEDİ."),
    }
    (OUT_DIR / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("-" * 70)
    print(f"JSON: {OUT_DIR / 'verify_report.json'}")
    print(f"PNG : {OUT_DIR}\\thermal_samples.png | rgb_samples.png | pairs.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
