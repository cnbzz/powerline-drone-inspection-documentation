---
name: fault-database
description: MVP-4 fault veri modeli, SQLite DB ve risk scoring formülü + kritik kısıt
metadata:
  type: project
---

**Veri modeli** (`src/fault/fault_record.py`): `FaultRecord` dataclass. Enum'lar: `RiskLevel`(low/medium/high/critical), `FaultStatus`(new/reviewed/resolved/false_positive), `FaultType`(thermal_hotspot/visual_defect/equipment_detected/combined/vegetation_encroachment/unknown). İç içe: `BoundingBox`, `ThermalData`(peak/mean/ambient/area/delta), `GpsLocation`. `to_dict()`/`from_dict()` round-trip çalışıyor (datetime→ISO, Enum→value).

**DB** (`src/fault/database.py`): SQLAlchemy 2.0 + SQLite, `FaultDatabase(db_path)`. Ana ORM `FaultRecordORM`, **tablo adı `fault_records`** (DİKKAT: "faults" DEĞİL — migration/ALTER bu isme göre). Nested alanlar düz kolonlara açılır (gps→lat/lon/alt, bbox→x/y/w/h, thermal→5 kolon). API: `create/get/list_all(status,risk_level,flight_id,limit)/update_status/delete/count/stats`. Indexler: timestamp, risk_level, status, inspection_flight_id. Varsayılan yol `outputs/db/faults.db`. Session pattern: `with self.session() as s:` context manager (commit on exit); autoincrement id gerekiyorsa `s.flush()`.

**Personel + Notlar + Atama (MVP-5.3 ek, 2026-06-02):** Domain modelleri `src/fault/personnel.py` (`Personnel`, `FaultNote` dataclass; `Priority` low/normal/urgent, `NoteType` technical/field/management enum). ORM `database.py`'de: `PersonnelDB` (tablo `personnel`: id/name/role/team/active/created_at), `FaultNoteDB` (tablo `fault_notes`: id/fault_id FK→fault_records.fault_id/note_type/content/author/created_at). `FaultRecordORM`'a 2 yeni kolon: `assigned_to_id` (FK→personnel.id), `priority` (default "normal"). Yeni metotlar: `add_personnel→id / list_personnel(active_only) / update_personnel(**kwargs) / deactivate_personnel(soft-delete) / assign_fault(fid,pid,priority) / get_fault_assignment(fid)→dict / add_note→id / list_notes(fid) / delete_note / delete_fault (bağlı notları da siler)`.

**Soft delete (MVP-5.3, 2026-06-02):** `FaultRecordORM`'a `active` kolonu (Boolean, default True). `list_all(..., include_inactive=False)` default sadece `active=True` döner. Metotlar: `deactivate_fault(fid)` (active=False), `reactivate_fault(fid)` (active=True). `delete_fault` artık KALICI hard-delete (dashboard'da kullanılmıyor; UI "Pasife Al" = soft delete). **NOT (kapsam sınırı):** `stats()` / `count()` hâlâ TÜM kayıtları sayar (inactive dahil) — sidebar "Toplam" ile aktif liste arasında küçük tutarsızlık olabilir; spec sadece list_all güncellemesini istedi.

**Migration'lar (hepsi IDEMPOTENT — ALTER sadece kolon yoksa, inspector ile kontrol):**
- `scripts/migrate_db_v2.py` — personnel + fault_notes tabloları + `assigned_to_id`/`priority` kolonları + 6 personel seed (Ömer Kürşat Canbaz + Ali/Mehmet/Ayşe/Fatma/Hasan; seed sadece tablo boşsa).
- `scripts/migrate_db_v3.py` — `fault_records.active` kolonu (soft delete).
Çalıştır: `uv run python scripts/migrate_db_vN.py`. Mevcut DB'ye uygulandı, 102 arıza korundu; tekrar çalıştırılabilir.

**Risk formülü** (`src/fault/risk_scorer.py`):
`Risk = 0.30·FaultTypeSeverity + 0.25·Thermal + 0.20·Distance + 0.15·Confidence + 0.10·History`
- FaultTypeSeverity: combined=90, hotspot=70, visual=60, vegetation=50, unknown=30, equipment=5
- Thermal: delta bantlı skor + peak override (>100°C→≥70, >130°C→≥90)
- Bantlar: <25 low, <50 medium, <75 high, ≥75 critical
- `apply_risk_to_record()` skoru/level/factors'ı kaydeder.

**Dinamik normalizasyon (DÜZELTİLDİ):** Her `compute_*_score` artık veri yoksa `None` döner (distance/history hep `None` placeholder; thermal/confidence veri yoksa `None`). `score_fault()` sadece aktif bileşenleri alır ve ağırlıklarını oranları koruyarak 1.0'a yeniden normalize eder. Böylece **0-100 aralığı erişilebilir**. `RiskBreakdown.to_dict()` her bileşen için `{raw, weight_used, contribution, active}` + `active_components` listesi verir.
- Doğrulama: combined+140°C+conf0.91 → fault_type 38.6 + thermal 35.6 + confidence 19.5 = **93.7 CRITICAL** (distance/history hariç). Test: 10 kayıttan 0 low / 1 medium / 5 high / **4 critical**.
- **Yan etki (dürüst not):** Az bileşen aktifken kalan ağırlıklar büyür → saf-termal/görsel kayıtlar yukarı kayar; normal ekipman (severity 5) yüksek confidence ile artık LOW yerine MEDIUM (~34) olabilir (confidence payı 2 bileşende 1/3'e çıkıyor). Semantik olarak tartışmalı; ileride confidence'ı sadece "fault" kayıtlarına uygulamak düşünülebilir.

Test scripti: `scripts/test_mvp4_db_risk.py`.

Bağlı: [[hotspot-detector]] (thermal girdisi), [[mvp-roadmap]] MVP-4→MVP-5, [[future-improvements]].
