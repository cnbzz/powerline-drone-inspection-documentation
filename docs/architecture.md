# Mimari

> Placeholder — detaylı mimari diyagramları ve veri akışı sonraki fazlarda eklenecek.

## Yüksek Seviye Akış

```
[Drone capture] --> [ingestion] --> [detection (RGB)] --+
                                                        |
                                  +--> [thermal analiz] +--> [fault]
                                                              |
                                                              v
                                                       [reporting + dashboard]
```

## Modül Sorumlulukları

- **ingestion**: Ham görüntü/video → EXIF, GPS, termal sıcaklık matrisi, RGB-termal eşleme.
- **detection**: RGB üzerinde nesne tespiti (YOLOv8/v11).
- **thermal**: Hotspot tespiti, sıcaklık istatistikleri, RGB ile hizalama.
- **fault**: Bulguların `FaultRecord`'a dönüştürülmesi, risk skoru, DB'ye yazım.
- **reporting**: PDF + Excel rapor üretimi, Folium harita çıktısı.
- **dashboard**: Streamlit tabanlı operatör paneli.
