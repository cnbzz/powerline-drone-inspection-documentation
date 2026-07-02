"""Personnel management models (MVP-5.3 ek).

Saha personeli + arızaya bağlı inceleme notları için domain modelleri. Persistence
katmanı :mod:`src.fault.database` içinde (PersonnelDB / FaultNoteDB ORM tabloları).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Priority(str, Enum):
    """Arıza atama önceliği."""
    LOW = "low"
    NORMAL = "normal"
    URGENT = "urgent"


class NoteType(str, Enum):
    """İnceleme notu kategorisi."""
    TECHNICAL = "technical"
    FIELD = "field"
    MANAGEMENT = "management"


@dataclass
class Personnel:
    """Personel kaydı."""
    id: Optional[int] = None
    name: str = ""
    role: str = ""   # Saha Mühendisi, Bölge Sorumlusu, Teknisyen, ...
    team: str = ""   # Kocaeli Ekibi, Sakarya Ekibi, ...
    active: bool = True
    created_at: Optional[str] = None


@dataclass
class FaultNote:
    """Bir arızaya bağlı inceleme notu."""
    id: Optional[int] = None
    fault_id: str = ""
    note_type: NoteType = NoteType.TECHNICAL
    content: str = ""
    author: str = ""
    created_at: Optional[str] = None
