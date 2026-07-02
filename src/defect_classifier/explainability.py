"""Stage 2 model explainability (XAI yorum katmanı, Aşama 7.1B).

EfficientNetV2-S sınıflandırıcısının kararını DENETLENEBİLİR hale getirir:
karar güveni, karar marjı (top1-top2), model performans bağlamı (val/test F1),
domain belirsizlik uyarıları ve genel güvenilirlik özeti.

SAF mantık — Streamlit'e ve torch'a bağımlı DEĞİL (bağımsız test edilebilir).
Girdi: ``inference.predict()`` çıktısı (dict) + ekipman adı.
Çıktı: ``ModelExplanation`` dataclass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# === Karar güveni eşikleri (top-1 olasılık) ===
CONF_HIGH = 0.85   # >= 0.85 → YÜKSEK
CONF_MEDIUM = 0.60  # 0.60-0.85 → ORTA, < 0.60 → DÜŞÜK

# === Karar marjı eşikleri (top1 - top2) ===
MARGIN_CLEAR = 0.30      # >= 0.30 → net
MARGIN_MODERATE = 0.10   # 0.10-0.30 → orta, < 0.10 → sınır

# === Model performans bağlamı (eğitim test F1, all_results.json'dan; fallback sabit) ===
_MODEL_F1_FALLBACK = {
    'glass-insulator': 0.9153,
    'lightning-rod-suspension': 0.9961,
    'polymer-insulator-upper-shackle': 0.9844,
    'vari-grip': 0.9756,
    'yoke-suspension': 0.9964,
}
_RESULTS_JSON = Path('outputs/training_logs/defect_classifier/all_results.json')

# UI Türkçe sınıf etiketleri (inference.CLASS_LABELS_TR ile aynı; torch import etmemek
# için burada yerel kopya — küçük ve nadir değişir).
CLASS_LABELS_TR = {
    'good': 'Normal',
    'rust': 'Korozyon (Pas)',
    'missing-cap': 'Kırık/Eksik Kap',
    'bird-nest': 'Kuş Yuvası',
}


def load_model_f1() -> dict:
    """Model test F1'lerini training_logs/all_results.json'dan oku, yoksa fallback."""
    f1 = dict(_MODEL_F1_FALLBACK)
    try:
        if _RESULTS_JSON.exists():
            data = json.loads(_RESULTS_JSON.read_text(encoding='utf-8'))
            for eq, res in data.items():
                if isinstance(res, dict) and isinstance(res.get('test_f1'), (int, float)):
                    f1[eq] = float(res['test_f1'])
    except Exception:  # noqa: BLE001 — bozuk JSON olsa bile fallback yeterli
        pass
    return f1


@dataclass
class ModelExplanation:
    """Stage 2 modelinin kararının denetlenebilir yorumu."""
    equipment: str
    predicted_class: str
    predicted_class_tr: str
    confidence_pct: float                 # top-1 olasılık * 100
    confidence_tier: str                  # 'high' | 'medium' | 'low'
    top_predictions: list                 # [{'class_tr', 'prob_pct'}] sıralı (desc)
    margin: float                         # top1 - top2 (0-1)
    margin_pct: float                     # margin * 100
    margin_tier: str                      # 'clear' | 'moderate' | 'borderline'
    model_f1: Optional[float]             # ekipman test F1 (0-1) veya None
    caveats: list = field(default_factory=list)   # Türkçe domain/belirsizlik uyarıları
    overall_verdict: str = ""             # 1 cümle genel güvenilirlik özeti


def _confidence_tier(conf: float) -> str:
    if conf >= CONF_HIGH:
        return 'high'
    if conf >= CONF_MEDIUM:
        return 'medium'
    return 'low'


def _margin_tier(margin: float) -> str:
    if margin >= MARGIN_CLEAR:
        return 'clear'
    if margin >= MARGIN_MODERATE:
        return 'moderate'
    return 'borderline'


def build_model_explanation(prediction: dict, equipment: str,
                            model_f1_table: Optional[dict] = None) -> ModelExplanation:
    """``inference.predict()`` çıktısından ModelExplanation üret (SAF fonksiyon).

    Args:
        prediction: {'predicted_class','predicted_class_tr','confidence',
                     'all_probabilities','is_defective'}
        equipment: defect classifier ekipman adı (ör. 'glass-insulator')
        model_f1_table: opsiyonel F1 lookup (varsayılan: load_model_f1())
    """
    f1_table = model_f1_table if model_f1_table is not None else load_model_f1()

    all_probs = prediction.get('all_probabilities', {}) or {}
    # Olasılıkları azalan sırala
    ranked = sorted(all_probs.items(), key=lambda kv: -kv[1])
    top1_prob = ranked[0][1] if ranked else float(prediction.get('confidence', 0.0) or 0.0)
    top2_prob = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = float(prediction.get('confidence', top1_prob) or top1_prob)
    conf_tier = _confidence_tier(confidence)

    margin = max(0.0, top1_prob - top2_prob)
    m_tier = _margin_tier(margin)

    top_predictions = [
        {'class_tr': CLASS_LABELS_TR.get(cls, cls), 'prob_pct': prob * 100.0}
        for cls, prob in ranked[:3]
    ]

    model_f1 = f1_table.get(equipment)
    predicted_class = prediction.get('predicted_class', '')

    # === Belirsizlik / domain uyarıları ===
    caveats: list = []
    if equipment == 'polymer-insulator-upper-shackle' and predicted_class == 'rust':
        caveats.append(
            "Polimer izolatörlerde 'pas/korozyon' sınıfı, plastik gövdenin değil "
            "metal askı/bağlantı elemanlarının korozyonunu (askıda korozyon) işaret eder."
        )
    if conf_tier == 'low' or m_tier == 'borderline':
        caveats.append(
            "Bu tahmin saha personeli tarafından manuel olarak doğrulanmalıdır."
        )

    # === Genel güvenilirlik özeti (1 cümle) ===
    f1_txt = f"F1={model_f1:.2f}" if model_f1 is not None else "F1 bilinmiyor"
    if conf_tier == 'low' or m_tier == 'borderline':
        verdict = (f"DÜŞÜK güvenilirlik (güven %{confidence*100:.0f}, marj %{margin*100:.0f}) "
                   f"→ bu teşhis manuel doğrulama gerektirir.")
    elif conf_tier == 'high' and m_tier == 'clear':
        verdict = (f"YÜKSEK güven + net marj + yüksek model başarısı ({f1_txt}) "
                   f"→ bu teşhise güvenilebilir.")
    else:
        verdict = (f"ORTA düzey güvenilirlik ({f1_txt}) → teşhis büyük olasılıkla doğru, "
                   f"ancak kritik kararlarda doğrulama önerilir.")

    return ModelExplanation(
        equipment=equipment,
        predicted_class=predicted_class,
        predicted_class_tr=prediction.get('predicted_class_tr', predicted_class),
        confidence_pct=confidence * 100.0,
        confidence_tier=conf_tier,
        top_predictions=top_predictions,
        margin=margin,
        margin_pct=margin * 100.0,
        margin_tier=m_tier,
        model_f1=model_f1,
        caveats=caveats,
        overall_verdict=verdict,
    )
