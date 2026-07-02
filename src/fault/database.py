"""Fault persistence layer (SQLAlchemy 2.x, SQLite).

SQLite-backed storage for :class:`~src.fault.fault_record.FaultRecord`. Nested
structures (gps, bbox, thermal) are flattened into individual columns; the
converters reconstruct the dataclasses on read.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.fault.fault_record import (
    BoundingBox,
    FaultRecord,
    FaultStatus,
    FaultType,
    GpsLocation,
    RiskLevel,
    ThermalData,
)
from src.fault.personnel import FaultNote, NoteType, Personnel

Base = declarative_base()


class FaultRecordORM(Base):
    """SQLAlchemy ORM mapping for :class:`FaultRecord`."""

    __tablename__ = "fault_records"

    fault_id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # GPS (flattened)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude_m = Column(Float, nullable=True)

    # Paths
    rgb_image_path = Column(String, nullable=True)
    thermal_image_path = Column(String, nullable=True)

    # Detection
    fault_type = Column(SQLEnum(FaultType), nullable=False, default=FaultType.UNKNOWN)
    equipment_class = Column(String, nullable=True)
    equipment_class_id = Column(Integer, nullable=True)

    # BBox (flattened)
    bbox_x = Column(Integer, nullable=True)
    bbox_y = Column(Integer, nullable=True)
    bbox_w = Column(Integer, nullable=True)
    bbox_h = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)

    detection_confidence = Column(Float, nullable=True)

    # Thermal (flattened, all nullable)
    thermal_peak_temp_c = Column(Float, nullable=True)
    thermal_mean_temp_c = Column(Float, nullable=True)
    thermal_ambient_temp_c = Column(Float, nullable=True)
    thermal_area_pixels = Column(Integer, nullable=True)
    thermal_temp_delta_c = Column(Float, nullable=True)

    # Risk
    risk_level = Column(SQLEnum(RiskLevel), nullable=False, default=RiskLevel.LOW, index=True)
    risk_score = Column(Float, nullable=False, default=0.0)
    risk_factors = Column(JSON, nullable=True)

    # Lifecycle
    status = Column(SQLEnum(FaultStatus), nullable=False, default=FaultStatus.NEW, index=True)
    notes = Column(Text, nullable=True)
    reviewer = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    # Metadata
    inspection_flight_id = Column(String, nullable=True, index=True)

    # Atama (MVP-5.3 — personnel + priority; migrate_db_v2.py ile mevcut DB'ye eklenir)
    assigned_to_id = Column(Integer, ForeignKey("personnel.id"), nullable=True)
    priority = Column(String(20), default="normal")

    # Soft delete (MVP-5.3 — migrate_db_v3.py ile mevcut DB'ye eklenir)
    active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PersonnelDB(Base):
    """SQLAlchemy ORM mapping for :class:`~src.fault.personnel.Personnel`."""

    __tablename__ = "personnel"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    role = Column(String(100))
    team = Column(String(100))
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class FaultNoteDB(Base):
    """SQLAlchemy ORM mapping for :class:`~src.fault.personnel.FaultNote`."""

    __tablename__ = "fault_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fault_id = Column(String(36), ForeignKey("fault_records.fault_id"), nullable=False, index=True)
    note_type = Column(String(20), default="technical")
    content = Column(Text, nullable=False)
    author = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)


class FaultDatabase:
    """High-level API for fault record persistence."""

    def __init__(self, db_path: str | Path = "outputs/db/faults.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._SessionFactory = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized: {self.db_path}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._SessionFactory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ----------------------------------------------------------------- CRUD #
    def create(self, record: FaultRecord) -> str:
        """Insert a new fault record. Returns its fault_id."""
        with self.session() as s:
            s.add(self._record_to_orm(record))
        return record.fault_id

    def get(self, fault_id: str) -> Optional[FaultRecord]:
        with self.session() as s:
            orm = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            return self._orm_to_record(orm) if orm else None

    def list_all(
        self,
        status: Optional[FaultStatus] = None,
        risk_level: Optional[RiskLevel] = None,
        flight_id: Optional[str] = None,
        limit: int = 1000,
        include_inactive: bool = False,
    ) -> list[FaultRecord]:
        with self.session() as s:
            q = s.query(FaultRecordORM)
            if not include_inactive:
                q = q.filter(FaultRecordORM.active.is_(True))
            if status:
                q = q.filter_by(status=status)
            if risk_level:
                q = q.filter_by(risk_level=risk_level)
            if flight_id:
                q = q.filter_by(inspection_flight_id=flight_id)
            q = q.order_by(FaultRecordORM.timestamp.desc()).limit(limit)
            return [self._orm_to_record(o) for o in q.all()]

    def update_status(
        self,
        fault_id: str,
        status: FaultStatus,
        reviewer: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        with self.session() as s:
            orm = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not orm:
                return False
            orm.status = status
            if reviewer:
                orm.reviewer = reviewer
                orm.reviewed_at = datetime.now()
            if notes:
                orm.notes = notes
        return True

    def delete(self, fault_id: str) -> bool:
        with self.session() as s:
            orm = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not orm:
                return False
            s.delete(orm)
        return True

    def count(self) -> int:
        with self.session() as s:
            return s.query(FaultRecordORM).count()

    def stats(self) -> dict:
        with self.session() as s:
            total = s.query(FaultRecordORM).count()
            by_risk = {
                level.value: s.query(FaultRecordORM).filter_by(risk_level=level).count()
                for level in RiskLevel
            }
            by_status = {
                st.value: s.query(FaultRecordORM).filter_by(status=st).count()
                for st in FaultStatus
            }
        return {"total": total, "by_risk_level": by_risk, "by_status": by_status}

    # ------------------------------------------------------------ personnel #
    def add_personnel(self, name: str, role: str, team: str) -> int:
        """Yeni personel ekle, ID döner."""
        with self.session() as s:
            p = PersonnelDB(name=name, role=role, team=team, active=True)
            s.add(p)
            s.flush()  # autoincrement id'yi commit'ten önce ata
            return p.id

    def list_personnel(self, active_only: bool = True) -> list[Personnel]:
        """Personel listesi (isme göre sıralı)."""
        with self.session() as s:
            q = s.query(PersonnelDB)
            if active_only:
                q = q.filter(PersonnelDB.active.is_(True))
            rows = q.order_by(PersonnelDB.name).all()
            return [
                Personnel(
                    id=r.id, name=r.name, role=r.role, team=r.team,
                    active=r.active,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                )
                for r in rows
            ]

    def update_personnel(self, personnel_id: int, **kwargs) -> bool:
        """Personel güncelle (name/role/team/active)."""
        with self.session() as s:
            p = s.query(PersonnelDB).filter_by(id=personnel_id).first()
            if not p:
                return False
            for k, v in kwargs.items():
                if hasattr(p, k):
                    setattr(p, k, v)
            return True

    def deactivate_personnel(self, personnel_id: int) -> bool:
        """Personeli soft-delete (active=False)."""
        return self.update_personnel(personnel_id, active=False)

    # ------------------------------------------------------------ assignment #
    def assign_fault(
        self,
        fault_id: str,
        personnel_id: Optional[int],
        priority: str = "normal",
    ) -> bool:
        """Arızayı personele ata (personnel_id=None → atamayı kaldır)."""
        with self.session() as s:
            f = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not f:
                return False
            f.assigned_to_id = personnel_id
            f.priority = priority
            return True

    def get_fault_assignment(self, fault_id: str) -> Optional[dict]:
        """Bir arızanın atama bilgisi (assigned personnel + priority)."""
        with self.session() as s:
            f = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not f:
                return None
            result = {
                "priority": f.priority or "normal",
                "assigned_to_id": f.assigned_to_id,
                "assigned_to_name": None,
                "assigned_to_role": None,
                "assigned_to_team": None,
            }
            if f.assigned_to_id:
                p = s.query(PersonnelDB).filter_by(id=f.assigned_to_id).first()
                if p:
                    result["assigned_to_name"] = p.name
                    result["assigned_to_role"] = p.role
                    result["assigned_to_team"] = p.team
            return result

    # ----------------------------------------------------------------- notes #
    def add_note(self, fault_id: str, note_type: str, content: str, author: str) -> int:
        """Arızaya not ekle, ID döner."""
        with self.session() as s:
            n = FaultNoteDB(
                fault_id=fault_id, note_type=note_type,
                content=content, author=author,
            )
            s.add(n)
            s.flush()
            return n.id

    def list_notes(self, fault_id: str) -> list[FaultNote]:
        """Bir arızanın tüm notları (yeniden eskiye)."""
        with self.session() as s:
            rows = (
                s.query(FaultNoteDB)
                .filter_by(fault_id=fault_id)
                .order_by(FaultNoteDB.created_at.desc())
                .all()
            )
            return [
                FaultNote(
                    id=r.id, fault_id=r.fault_id, note_type=NoteType(r.note_type),
                    content=r.content, author=r.author,
                    created_at=r.created_at.isoformat() if r.created_at else None,
                )
                for r in rows
            ]

    def delete_note(self, note_id: int) -> bool:
        """Not sil."""
        with self.session() as s:
            n = s.query(FaultNoteDB).filter_by(id=note_id).first()
            if not n:
                return False
            s.delete(n)
            return True

    def deactivate_fault(self, fault_id: str) -> bool:
        """Arıza kaydını pasife al (soft delete — veri korunur)."""
        with self.session() as s:
            f = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not f:
                return False
            f.active = False
            return True

    def reactivate_fault(self, fault_id: str) -> bool:
        """Pasif arızayı tekrar aktif yap."""
        with self.session() as s:
            f = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not f:
                return False
            f.active = True
            return True

    def delete_fault(self, fault_id: str) -> bool:
        """Arıza kaydını ve bağlı notları KALICI sil (hard delete)."""
        with self.session() as s:
            # Önce bağlı notları sil (FK temizliği)
            s.query(FaultNoteDB).filter_by(fault_id=fault_id).delete()
            f = s.query(FaultRecordORM).filter_by(fault_id=fault_id).first()
            if not f:
                return False
            s.delete(f)
            return True

    # ----------------------------------------------------------- converters #
    @staticmethod
    def _record_to_orm(r: FaultRecord) -> FaultRecordORM:
        gps = r.gps
        bb = r.bbox
        th = r.thermal
        return FaultRecordORM(
            fault_id=r.fault_id,
            timestamp=r.timestamp,
            latitude=gps.latitude if gps else None,
            longitude=gps.longitude if gps else None,
            altitude_m=gps.altitude_m if gps else None,
            rgb_image_path=r.rgb_image_path,
            thermal_image_path=r.thermal_image_path,
            fault_type=r.fault_type,
            equipment_class=r.equipment_class,
            equipment_class_id=r.equipment_class_id,
            bbox_x=bb.x if bb else None,
            bbox_y=bb.y if bb else None,
            bbox_w=bb.width if bb else None,
            bbox_h=bb.height if bb else None,
            image_width=bb.image_width if bb else None,
            image_height=bb.image_height if bb else None,
            detection_confidence=r.detection_confidence,
            thermal_peak_temp_c=th.peak_temp_c if th else None,
            thermal_mean_temp_c=th.mean_temp_c if th else None,
            thermal_ambient_temp_c=th.ambient_temp_c if th else None,
            thermal_area_pixels=th.area_pixels if th else None,
            thermal_temp_delta_c=th.temperature_delta_c if th else None,
            risk_level=r.risk_level,
            risk_score=r.risk_score,
            risk_factors=r.risk_factors or None,
            status=r.status,
            notes=r.notes,
            reviewer=r.reviewer,
            reviewed_at=r.reviewed_at,
            inspection_flight_id=r.inspection_flight_id,
        )

    @staticmethod
    def _orm_to_record(o: FaultRecordORM) -> FaultRecord:
        gps = None
        if o.latitude is not None and o.longitude is not None:
            gps = GpsLocation(o.latitude, o.longitude, o.altitude_m)

        bbox = None
        if o.bbox_x is not None:
            bbox = BoundingBox(o.bbox_x, o.bbox_y, o.bbox_w, o.bbox_h,
                               o.image_width, o.image_height)

        thermal = None
        if o.thermal_peak_temp_c is not None:
            thermal = ThermalData(
                peak_temp_c=o.thermal_peak_temp_c,
                mean_temp_c=o.thermal_mean_temp_c,
                ambient_temp_c=o.thermal_ambient_temp_c,
                area_pixels=o.thermal_area_pixels,
                temperature_delta_c=o.thermal_temp_delta_c,
            )

        return FaultRecord(
            fault_id=o.fault_id,
            timestamp=o.timestamp,
            gps=gps,
            rgb_image_path=o.rgb_image_path,
            thermal_image_path=o.thermal_image_path,
            fault_type=o.fault_type if isinstance(o.fault_type, FaultType) else FaultType(o.fault_type),
            equipment_class=o.equipment_class,
            equipment_class_id=o.equipment_class_id,
            bbox=bbox,
            detection_confidence=o.detection_confidence,
            thermal=thermal,
            risk_level=o.risk_level if isinstance(o.risk_level, RiskLevel) else RiskLevel(o.risk_level),
            risk_score=o.risk_score,
            risk_factors=o.risk_factors or {},
            status=o.status if isinstance(o.status, FaultStatus) else FaultStatus(o.status),
            notes=o.notes,
            reviewer=o.reviewer,
            reviewed_at=o.reviewed_at,
            inspection_flight_id=o.inspection_flight_id,
        )
