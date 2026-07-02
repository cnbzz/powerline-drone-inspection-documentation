"""Loguru-based logger.

Provides a single :func:`get_logger` factory that:
  * configures loguru on first call (idempotent),
  * reads level/file/rotation from ``config/config.yaml`` when present,
  * honours ``LOG_LEVEL`` / ``LOG_FILE`` env-vars (.env aware) as overrides,
  * writes to both console (stderr) and a rotating file sink.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

_CONFIGURED: bool = False
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH: Path = _PROJECT_ROOT / "config" / "config.yaml"
_DEFAULT_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
    "{name}:{function}:{line} | {message}"
)


def _load_logging_config(config_path: Path) -> dict[str, Any]:
    """Read the ``logging`` block from ``config.yaml`` if available."""
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = data.get("logging", {})
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def _resolve_log_file(raw_path: str | None) -> Path | None:
    """Resolve a log file path string to an absolute Path, creating parents."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _configure(config_path: Path = _DEFAULT_CONFIG_PATH) -> None:
    """One-time loguru configuration.

    Precedence (highest -> lowest):
      1. Environment variables (``LOG_LEVEL``, ``LOG_FILE``)
      2. ``config/config.yaml`` -> ``logging`` block
      3. Built-in defaults
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    cfg = _load_logging_config(config_path)

    level: str = (os.getenv("LOG_LEVEL") or cfg.get("level") or "INFO").upper()
    console: bool = bool(cfg.get("console", True))
    file_path: Path | None = _resolve_log_file(
        os.getenv("LOG_FILE") or cfg.get("file") or "outputs/logs/app.log"
    )
    rotation: str = cfg.get("rotation", "10 MB")
    retention: str = cfg.get("retention", "14 days")
    fmt: str = cfg.get("format", _DEFAULT_FORMAT)

    logger.remove()
    if console:
        logger.add(sys.stderr, level=level, format=fmt, enqueue=False, backtrace=True, diagnose=False)
    if file_path is not None:
        logger.add(
            str(file_path),
            level=level,
            format=fmt,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )

    _CONFIGURED = True


def get_logger(name: str | None = None) -> "logger.__class__":
    """Return a configured loguru logger.

    Args:
        name: Optional binding for ``name``; used in the format string.
              Defaults to ``__name__`` of the caller's module.

    Returns:
        The loguru logger instance (bound when ``name`` is given).
    """
    _configure()
    return logger.bind(name=name) if name else logger
