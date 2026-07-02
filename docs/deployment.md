# Deployment

> Placeholder — production deployment senaryosu MVP-5 sonunda netleşecek.

## Hedef Çalışma Modu (planlanan)

- **Lokal saha makinesi**: Drone uçuş sonrası kartı boşaltıp pipeline'ı çalıştırma.
- **Streamlit dashboard**: Saha ekipleri için lokal ağ üzerinden erişim.
- **Opsiyonel bulut senkronizasyonu**: Rapor + arıza kaydı arşivi için.

## Gereksinimler

- Python 3.11, uv ile yönetilen virtualenv
- CUDA destekli GPU (RTX 3060+ önerilir)
- ~50 GB disk (ham görüntüler + model checkpoint'leri)

## Çalıştırma

```powershell
# Inspection pipeline (MVP-5 sonrası)
uv run python scripts/run_pipeline.py --input data/raw --output outputs/

# Dashboard
uv run streamlit run src/dashboard/app.py
```
