"""Migration v2: personnel + fault_notes tabloları + fault_records'a yeni kolonlar.

Idempotent — birden çok kez çalıştırılabilir. ALTER TABLE yalnızca kolon yoksa
uygulanır (inspector ile kontrol). Seed personnel sadece tablo boşsa eklenir.

NOT: Bu kod tabanında arıza tablosunun adı ``fault_records`` (şablondaki ``faults``
değil) — ALTER/ inspect bu isme göre yapılır.

ÇALIŞTIRMA::

    uv run python scripts/migrate_db_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from loguru import logger
from sqlalchemy import create_engine, inspect, text

from src.fault.database import Base, FaultDatabase, FaultNoteDB, PersonnelDB

DB_PATH = _ROOT / "outputs" / "db" / "faults.db"
FAULTS_TABLE = "fault_records"  # gerçek tablo adı (şablon "faults" yanlış)

SEED_PERSONNEL = [
    ("Ömer Kürşat Canbaz", "Sistem Yöneticisi", "BEDAŞ Merkez"),
    ("Ali Yılmaz", "Saha Mühendisi", "Kocaeli Ekibi"),
    ("Mehmet Kaya", "Bölge Sorumlusu", "Sakarya Ekibi"),
    ("Ayşe Demir", "Teknisyen", "Bursa Ekibi"),
    ("Fatma Şahin", "Saha Mühendisi", "İstanbul Avrupa"),
    ("Hasan Çelik", "Bakım Şefi", "BEDAŞ Merkez"),
]


def migrate() -> None:
    db_url = f"sqlite:///{DB_PATH}"
    engine = create_engine(db_url)
    inspector = inspect(engine)

    # 1. Yeni tabloları oluştur (idempotent — varsa atlar)
    Base.metadata.create_all(engine, tables=[
        PersonnelDB.__table__,
        FaultNoteDB.__table__,
    ])
    logger.info("✓ personnel + fault_notes tabloları hazır")

    # 2. fault_records tablosuna yeni kolonlar ekle (idempotent)
    columns = [c["name"] for c in inspector.get_columns(FAULTS_TABLE)]
    with engine.connect() as conn:
        if "assigned_to_id" not in columns:
            conn.execute(text(f"ALTER TABLE {FAULTS_TABLE} ADD COLUMN assigned_to_id INTEGER"))
            conn.commit()
            logger.info(f"✓ {FAULTS_TABLE}.assigned_to_id kolonu eklendi")
        else:
            logger.info(f"- {FAULTS_TABLE}.assigned_to_id zaten var, atlanıyor")

        if "priority" not in columns:
            conn.execute(text(
                f"ALTER TABLE {FAULTS_TABLE} ADD COLUMN priority VARCHAR(20) DEFAULT 'normal'"
            ))
            conn.commit()
            logger.info(f"✓ {FAULTS_TABLE}.priority kolonu eklendi")
        else:
            logger.info(f"- {FAULTS_TABLE}.priority zaten var, atlanıyor")

    # 3. Seed personnel (sadece tablo boşsa)
    db = FaultDatabase(str(DB_PATH))
    existing = db.list_personnel(active_only=False)
    if not existing:
        for name, role, team in SEED_PERSONNEL:
            pid = db.add_personnel(name, role, team)
            logger.info(f"✓ Personel eklendi: {name} (ID={pid})")
    else:
        logger.info(f"- Personnel tablosunda {len(existing)} kayıt var, seed atlanıyor")

    logger.success("Migration tamamlandı.")


if __name__ == "__main__":
    migrate()
