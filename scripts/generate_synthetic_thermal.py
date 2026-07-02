"""Generate synthetic *radiometric* thermal images for MVP-3 (hotspot detection).

Each sample is a ``float32`` array of real per-pixel temperatures in °C (an abstraction
of DJI R-JPEG radiometric data). Downstream algorithms read the ``.npy``; the ``.png``
(jet colormap) is a human-facing visualization, and the ``.json`` holds ground truth.

10 scenarios x 20 samples = 200 images covering normal cases, escalating hotspots, and
deliberate false-positive traps (sun warming, ambient noise, wind cooling).

Usage::

    uv run python scripts/generate_synthetic_thermal.py --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# UTF-8 console (cp1252 chokes on °C / Turkish glyphs)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

WIDTH = 640
HEIGHT = 480
SEED = 42
SAMPLES_PER_SCENARIO = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "thermal"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
LOG_FILE = LOG_DIR / "generate_synthetic_thermal.log"

# Single seeded RNG used everywhere for reproducibility.
rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Ground-truth dataclasses                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Hotspot:
    bbox: list[int]              # x, y, w, h
    center: list[int]            # cx, cy
    peak_temp_c: float
    mean_temp_c: float
    area_pixels: int
    risk_level: str              # mild/medium/critical/high


@dataclass
class EquipmentRegion:
    type: str                    # insulator / tower / conductor
    bbox: list[int]
    mean_temp_c: float


@dataclass
class GroundTruth:
    image_id: str
    scenario: str
    ambient_temp_c: float
    image_size: list[int]
    has_hotspot: bool
    hotspots: list[dict] = field(default_factory=list)
    equipment_regions: list[dict] = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------- #
# Field-building primitives                                                    #
# --------------------------------------------------------------------------- #
def create_ambient(base_temp: float, noise_std: float = 1.5) -> np.ndarray:
    """Ambient temperature field with Gaussian per-pixel noise."""
    return (base_temp + rng.normal(0, noise_std, (HEIGHT, WIDTH))).astype(np.float32)


def add_blob(thermal: np.ndarray, x: int, y: int, w: int, h: int,
             temp: float, sigma: float = 5.0) -> None:
    """Add a hotspot as additive 2D-Gaussian (peak ~= temp above ambient) at bbox center."""
    yy, xx = np.mgrid[0:thermal.shape[0], 0:thermal.shape[1]]
    cx, cy = x + w / 2.0, y + h / 2.0
    gauss = temp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    thermal += gauss.astype(np.float32)


def add_rect_region(thermal: np.ndarray, x: int, y: int, w: int, h: int, temp: float) -> None:
    """Stamp a rectangular equipment region (uses max so warmer wins)."""
    x2, y2 = min(x + w, WIDTH), min(y + h, HEIGHT)
    x, y = max(x, 0), max(y, 0)
    patch = thermal[y:y2, x:x2]
    jitter = rng.normal(0, 0.8, patch.shape).astype(np.float32)
    thermal[y:y2, x:x2] = np.maximum(patch, temp + jitter)


def measure_hotspot(thermal: np.ndarray, x: int, y: int, w: int, h: int,
                    risk: str) -> Hotspot:
    """Measure peak/mean temp and area within a bbox to build accurate ground truth."""
    x2, y2 = min(x + w, WIDTH), min(y + h, HEIGHT)
    region = thermal[max(y, 0):y2, max(x, 0):x2]
    peak = float(region.max())
    mean = float(region.mean())
    return Hotspot(
        bbox=[int(x), int(y), int(w), int(h)],
        center=[int(x + w / 2), int(y + h / 2)],
        peak_temp_c=round(peak, 2),
        mean_temp_c=round(mean, 2),
        area_pixels=int(w * h),
        risk_level=risk,
    )


def rand_insulators(thermal: np.ndarray, n: int,
                    temp_lo: float = 25.0, temp_hi: float = 35.0) -> list[EquipmentRegion]:
    """Place ``n`` insulator rectangles at random positions; return their regions."""
    regions: list[EquipmentRegion] = []
    for _ in range(n):
        w, h = int(rng.integers(60, 100)), int(rng.integers(24, 38))
        x, y = int(rng.integers(20, WIDTH - w - 20)), int(rng.integers(40, HEIGHT - h - 40))
        temp = float(rng.uniform(temp_lo, temp_hi))
        add_rect_region(thermal, x, y, w, h, temp)
        regions.append(EquipmentRegion("insulator", [x, y, w, h],
                                       round(float(thermal[y:y + h, x:x + w].mean()), 2)))
    return regions


# --------------------------------------------------------------------------- #
# Visualization                                                                #
# --------------------------------------------------------------------------- #
def visualize(thermal: np.ndarray, save_path: Path, hotspots: list[dict] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(thermal, cmap="jet", vmin=-10, vmax=150)
    plt.colorbar(im, ax=ax, label="Temperature (°C)")
    for hs in hotspots or []:
        bx, by, bw, bh = hs["bbox"]
        ax.add_patch(patches.Rectangle((bx, by), bw, bh, linewidth=2,
                                       edgecolor="white", facecolor="none", linestyle="--"))
        ax.text(bx, by - 5, f"{hs['peak_temp_c']:.1f}°C", color="white", fontsize=10)
    ax.set_title(save_path.stem)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario generators — each returns (thermal_array, GroundTruth)              #
# --------------------------------------------------------------------------- #
def sc01_normal_powerline(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 1.5)
    regions = rand_insulators(t, int(rng.integers(3, 6)))
    return t, GroundTruth(image_id, "01_Normal_PowerLine", round(amb, 2), [WIDTH, HEIGHT],
                          False, [], [asdict(r) for r in regions],
                          "Normal power line, insulators at ambient-ish temps, no hotspot")


def sc02_normal_tower(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(15, 25))
    t = create_ambient(amb, 1.5)
    # X-shaped tower from two diagonal bands
    temp = float(rng.uniform(28, 38))
    for i in range(-6, 7):
        for d in range(0, HEIGHT):
            x1 = int(WIDTH / 2 - HEIGHT / 2 + d) + i
            x2 = int(WIDTH / 2 + HEIGHT / 2 - d) + i
            if 0 <= x1 < WIDTH:
                t[d, x1] = max(t[d, x1], temp)
            if 0 <= x2 < WIDTH:
                t[d, x2] = max(t[d, x2], temp)
    region = EquipmentRegion("tower", [WIDTH // 2 - HEIGHT // 2, 0, HEIGHT, HEIGHT], round(temp, 2))
    return t, GroundTruth(image_id, "02_Normal_Tower", round(amb, 2), [WIDTH, HEIGHT],
                          False, [], [asdict(region)], "Normal tower (X structure), no hotspot")


def _insulators_plus_hotspot(image_id: str, scenario: str, temp_lo: float, temp_hi: float,
                             box: int, sigma: float, risk: str, notes: str,
                             n_lo: int = 2, n_hi: int = 4) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 1.5)
    regions = rand_insulators(t, int(rng.integers(n_lo, n_hi)))
    # Anchor the hotspot on an existing insulator (endpoint/middle).
    anchor = regions[int(rng.integers(0, len(regions)))]
    ax_, ay_, aw_, ah_ = anchor.bbox
    hx = int(np.clip(ax_ + rng.integers(-10, aw_), 0, WIDTH - box))
    hy = int(np.clip(ay_ + rng.integers(-5, ah_), 0, HEIGHT - box))
    peak = float(rng.uniform(temp_lo, temp_hi))
    add_blob(t, hx, hy, box, box, peak, sigma)
    hs = measure_hotspot(t, hx, hy, box, box, risk)
    return t, GroundTruth(image_id, scenario, round(amb, 2), [WIDTH, HEIGHT],
                          True, [asdict(hs)], [asdict(r) for r in regions], notes)


def sc03_mild(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    return _insulators_plus_hotspot(image_id, "03_MildHotspot_Connection", 45, 55, 15, 5.0,
                                    "medium", "Mild hotspot at insulator endpoint (loose connection)")


def sc04_medium(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    return _insulators_plus_hotspot(image_id, "04_MediumHotspot_Insulator", 65, 80, 25, 8.0,
                                    "high", "Medium hotspot mid-insulator")


def sc05_critical(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    return _insulators_plus_hotspot(image_id, "05_CriticalHotspot_Joint", 100, 130, 35, 12.0,
                                    "critical", "Critical hotspot at joint, Gaussian falloff")


def sc06_multi(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 1.5)
    regions = rand_insulators(t, int(rng.integers(3, 6)))
    hotspots: list[Hotspot] = []
    for _ in range(int(rng.integers(2, 4))):
        box = int(rng.integers(18, 32))
        peak = float(rng.uniform(50, 90))
        hx = int(rng.integers(20, WIDTH - box - 20))
        hy = int(rng.integers(40, HEIGHT - box - 40))
        add_blob(t, hx, hy, box, box, peak, sigma=box / 3.0)
        risk = "critical" if peak >= 80 else ("high" if peak >= 65 else "medium")
        hotspots.append(measure_hotspot(t, hx, hy, box, box, risk))
    return t, GroundTruth(image_id, "06_MultiHotspot", round(amb, 2), [WIDTH, HEIGHT],
                          True, [asdict(h) for h in hotspots], [asdict(r) for r in regions],
                          f"{len(hotspots)} hotspots at mixed temperatures")


def sc07_ambient_noise(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 5.0)  # high noise
    regions = rand_insulators(t, int(rng.integers(2, 4)))
    return t, GroundTruth(image_id, "07_AmbientNoise_NoFault", round(amb, 2), [WIDTH, HEIGHT],
                          False, [], [asdict(r) for r in regions],
                          "High Gaussian noise (sigma=5C), NO hotspot - algorithm must not confuse")


def sc08_sun_warming(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 1.5)
    regions = rand_insulators(t, int(rng.integers(2, 4)))
    # Smooth vertical gradient: top warmer (35-45C) fading to ambient at bottom.
    top_temp = float(rng.uniform(35, 45))
    grad = np.linspace(top_temp - amb, 0, HEIGHT).astype(np.float32)[:, None]
    t += grad
    return t, GroundTruth(image_id, "08_SunWarming_FalsePositive", round(amb, 2), [WIDTH, HEIGHT],
                          False, [], [asdict(r) for r in regions],
                          "Broad sun-warming gradient (top warm). NO real hotspot - false-positive trap")


def sc09_wind_cooling(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(5, 10))  # cold day
    t = create_ambient(amb, 1.5)
    regions = rand_insulators(t, int(rng.integers(2, 4)), temp_lo=12, temp_hi=20)
    return t, GroundTruth(image_id, "09_WindCooling", round(amb, 2), [WIDTH, HEIGHT],
                          False, [], [asdict(r) for r in regions],
                          "Cold ambient (5-10C), everything cool, no hotspot")


def sc10_load_heating(image_id: str) -> tuple[np.ndarray, GroundTruth]:
    amb = float(rng.uniform(18, 22))
    t = create_ambient(amb, 1.5)
    # Three vertical conductor phases: left/middle/right.
    phase_temps = [35.0, 40.0, 55.0]
    xs = [int(WIDTH * 0.25), int(WIDTH * 0.5), int(WIDTH * 0.75)]
    regions: list[EquipmentRegion] = []
    for xc, temp in zip(xs, phase_temps):
        x = xc - 8
        add_rect_region(t, x, 30, 16, HEIGHT - 60, temp)
        regions.append(EquipmentRegion("conductor", [x, 30, 16, HEIGHT - 60], round(temp, 2)))
    # Phase 3 (55C) is the anomaly relative to the other phases -> flag as hotspot.
    x3 = xs[2] - 8
    hs = measure_hotspot(t, x3, 30, 16, HEIGHT - 60, "high")
    return t, GroundTruth(image_id, "10_LoadHeating_Phase", round(amb, 2), [WIDTH, HEIGHT],
                          True, [asdict(hs)], [asdict(r) for r in regions],
                          "Phase imbalance: phase3=55C vs phase1=35C/phase2=40C - anomalous, flagged")


SCENARIOS: list[tuple[str, Callable[[str], tuple[np.ndarray, GroundTruth]]]] = [
    ("01_Normal_PowerLine", sc01_normal_powerline),
    ("02_Normal_Tower", sc02_normal_tower),
    ("03_MildHotspot_Connection", sc03_mild),
    ("04_MediumHotspot_Insulator", sc04_medium),
    ("05_CriticalHotspot_Joint", sc05_critical),
    ("06_MultiHotspot", sc06_multi),
    ("07_AmbientNoise_NoFault", sc07_ambient_noise),
    ("08_SunWarming_FalsePositive", sc08_sun_warming),
    ("09_WindCooling", sc09_wind_cooling),
    ("10_LoadHeating_Phase", sc10_load_heating),
]


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")
    logger.add(LOG_FILE, level="DEBUG", mode="w", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} {level: <8} {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic radiometric thermal data.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove data/synthetic/thermal and rebuild from scratch.")
    args = parser.parse_args()

    setup_logging()
    # Reset RNG so output is identical every run.
    global rng
    rng = np.random.default_rng(SEED)

    if args.clean and OUT_ROOT.exists():
        logger.info(f"--clean: removing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_gt: list[dict] = []
    scenario_counts: dict[str, int] = {}
    n_hot = n_nohot = 0
    g_min, g_max, g_sum, g_px = np.inf, -np.inf, 0.0, 0

    for scenario, fn in SCENARIOS:
        sc_dir = OUT_ROOT / scenario
        sc_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {scenario} ...")
        for i in range(1, SAMPLES_PER_SCENARIO + 1):
            image_id = f"synth_{scenario[:2]}{i:02d}"
            thermal, gt = fn(image_id)
            thermal = thermal.astype(np.float32)

            np.save(sc_dir / f"{image_id}.npy", thermal)
            visualize(thermal, sc_dir / f"{image_id}.png", gt.hotspots)
            (sc_dir / f"{image_id}.json").write_text(
                json.dumps(asdict(gt), indent=2, ensure_ascii=False), encoding="utf-8")

            all_gt.append(asdict(gt))
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            if gt.has_hotspot:
                n_hot += 1
            else:
                n_nohot += 1
            g_min = min(g_min, float(thermal.min()))
            g_max = max(g_max, float(thermal.max()))
            g_sum += float(thermal.sum())
            g_px += thermal.size

    (OUT_ROOT / "ground_truth_all.json").write_text(
        json.dumps(all_gt, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- report ----
    total = len(all_gt)
    report = ["", "=" * 60, "SYNTHETIC THERMAL GENERATION REPORT", "=" * 60,
              f"Toplam görüntü: {total}", "", "Senaryo bazında dağılım:"]
    for scenario, _ in SCENARIOS:
        report.append(f"  {scenario:<32} {scenario_counts.get(scenario, 0):>3}")
    report += ["",
               f"Hotspot içeren görüntü:   {n_hot}",
               f"Hotspot içermeyen görüntü: {n_nohot}",
               "",
               "Sıcaklık dağılımı (tüm pikseller, °C):",
               f"  min : {g_min:.2f}",
               f"  max : {g_max:.2f}",
               f"  mean: {g_sum / g_px:.2f}",
               "",
               f"Çıktı: {OUT_ROOT}",
               f"Ground truth özeti: {OUT_ROOT / 'ground_truth_all.json'}",
               "=" * 60]
    text = "\n".join(report)
    print(text)
    logger.info("\n" + text)


if __name__ == "__main__":
    main()
