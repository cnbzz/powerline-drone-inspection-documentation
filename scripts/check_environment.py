"""Verify that the project environment is set up correctly.

Checks:
  * Python version
  * Installed versions of all declared dependencies
  * CUDA availability (via torch)
  * Importability of every ``src.*`` sub-package

Run with::

    uv run python scripts/check_environment.py
    # or
    python scripts/check_environment.py

Exit code:
  * ``0`` if every check passes
  * ``1`` if any check fails
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import platform
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so non-ASCII paths (e.g. Turkish user names)
# do not blow up on Windows cp1252 consoles.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# (display_name, distribution_name_for_metadata)
REQUIRED_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyyaml", "pyyaml"),
    ("python-dotenv", "python-dotenv"),
    ("loguru", "loguru"),
    ("opencv-python", "opencv-python"),
    ("pillow", "pillow"),
    ("ultralytics", "ultralytics"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("exifread", "exifread"),
    ("piexif", "piexif"),
    ("folium", "folium"),
    ("geopy", "geopy"),
    ("sqlalchemy", "sqlalchemy"),
    ("streamlit", "streamlit"),
    ("plotly", "plotly"),
    ("reportlab", "reportlab"),
    ("openpyxl", "openpyxl"),
    ("jinja2", "jinja2"),
]

SRC_MODULES: list[str] = [
    "src",
    "src.ingestion",
    "src.ingestion.exif_reader",
    "src.ingestion.thermal_parser",
    "src.ingestion.rgb_thermal_pairing",
    "src.ingestion.video_frame_extractor",
    "src.detection",
    "src.detection.yolo_trainer",
    "src.detection.yolo_inference",
    "src.detection.classes",
    "src.thermal",
    "src.thermal.hotspot_detector",
    "src.thermal.temperature_analysis",
    "src.thermal.thermal_rgb_align",
    "src.fault",
    "src.fault.fault_record",
    "src.fault.risk_scorer",
    "src.fault.database",
    "src.reporting",
    "src.reporting.pdf_generator",
    "src.reporting.excel_generator",
    "src.dashboard",
    "src.utils",
    "src.utils.logger",
    "src.utils.geo_utils",
    "src.utils.visualization",
]


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_python() -> bool:
    """Print and validate the running Python interpreter."""
    _section("Python")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")
    print(f"  platform   : {platform.platform()}")
    ok = sys.version_info[:2] == (3, 11)
    if not ok:
        print(f"  WARNING    : project targets Python 3.11, got {sys.version_info.major}.{sys.version_info.minor}")
    return ok


def check_packages() -> bool:
    """Print version of each required dependency, return True iff all found."""
    _section("Required packages")
    width = max(len(n) for n, _ in REQUIRED_PACKAGES) + 2
    all_ok = True
    for display, dist in REQUIRED_PACKAGES:
        try:
            version = md.version(dist)
            print(f"  {display:<{width}} {version}")
        except md.PackageNotFoundError:
            print(f"  {display:<{width}} MISSING")
            all_ok = False
    return all_ok


def check_cuda() -> bool:
    """Report CUDA availability via torch (non-fatal: CPU-only is allowed)."""
    _section("CUDA / device")
    try:
        import torch
    except ImportError:
        print("  torch not installed - skipping")
        return False
    cuda_ok = torch.cuda.is_available()
    print(f"  torch       : {torch.__version__}")
    print(f"  CUDA build  : {getattr(torch.version, 'cuda', None)}")
    print(f"  CUDA avail. : {cuda_ok}")
    if cuda_ok:
        print(f"  device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("  (CPU-only - dev workflow OK, training will be slow)")
    return True


def check_src_imports() -> bool:
    """Import every src sub-module to surface syntax errors / bad imports."""
    _section("src.* import check")
    all_ok = True
    width = max(len(m) for m in SRC_MODULES) + 2
    for mod in SRC_MODULES:
        try:
            importlib.import_module(mod)
            print(f"  {mod:<{width}} OK")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:<{width}} FAIL: {type(exc).__name__}: {exc}")
            all_ok = False
    return all_ok


def main() -> int:
    print("Powerline Drone Inspection - environment check")
    py = check_python()
    pkgs = check_packages()
    check_cuda()  # informational only
    imports = check_src_imports()

    _section("Summary")
    print(f"  python 3.11   : {'OK' if py else 'WARN'}")
    print(f"  packages      : {'OK' if pkgs else 'FAIL'}")
    print(f"  src imports   : {'OK' if imports else 'FAIL'}")

    return 0 if (pkgs and imports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
