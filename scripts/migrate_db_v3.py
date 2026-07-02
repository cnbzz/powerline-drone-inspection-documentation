"""Migration v3: fault_records'a active kolonu (soft delete).

Idempotent — ALTER yalnızca kolon yoksa uygulanır (inspector ile kontrol).

ÇALIŞTIRMA::

    uv run python scripts/migrate_db_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from loguru import logger
from sqlalchemy import create_engine, inspect, text

DB_PATH = _ROOT / "outputs" / "db" / "faults.db"
FAULTS_TABLE = "fault_records"  # gerçek tablo adı (şablondaki "faults" değil)


def migrate() -> None:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns(FAULTS_TABLE)]

    with engine.connect() as conn:
        if "active" not in columns:
            conn.execute(text(
                f"ALTER TABLE {FAULTS_TABLE} ADD COLUMN active BOOLEAN DEFAULT 1 NOT NULL"
            ))
            conn.commit()
            logger.info(f"✓ {FAULTS_TABLE}.active kolonu eklendi")
        else:
            logger.info(f"- {FAULTS_TABLE}.active zaten var, atlanıyor")

    logger.success("Migration v3 tamamlandı.")


if __name__ == "__main__":
    migrate()
