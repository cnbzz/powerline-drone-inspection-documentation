"""Predictive maintenance (Stage 7.4) — LSTM-based degradation forecasting.

ISOLATED module: it does NOT read or modify the fault DB, the existing models, or
the UI. It produces forward-looking projections ("this insulator will likely reach
the critical risk band in ~X months") from a per-equipment risk/thermal history.

ACADEMIC HONESTY / VERİ DÜRÜSTLÜĞÜ
----------------------------------
Saha boylamsal (zaman serisi) verisi mevcut DEĞİLDİR (drone tek uçuş yapar). Bu
modülün eğitim verisi bu nedenle SENTETİKTİR: fiziksel bir bozulma modelinden
deterministik olarak üretilir. Amaç, gerçek boylamsal veri elde edildiğinde
doğrudan kullanılabilecek, akademik olarak savunulabilir bir mimari kurmaktır.
Gerçek veriyle doğrulama GELECEK ÇALIŞMADIR; bu çıktılar gerçek saha ölçümü
olarak SUNULMAMALIDIR.
"""

SYNTHETIC_DISCLAIMER = (
    "Veri SENTETİKTİR; saha boylamsal verisi mevcut olmadığından fiziksel "
    "bozulma modelinden üretilmiştir. Gerçek veriyle doğrulama gelecek çalışmadır."
)
