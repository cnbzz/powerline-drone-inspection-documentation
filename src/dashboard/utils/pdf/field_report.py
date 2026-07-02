"""Saha müdahale raporu PDF üretici — resmi BEDAŞ kurumsal şablon.

Tek bir arıza için detaylı, kurumsal PDF rapor. Her ana bölüm ayrı sayfada başlar
(PageBreak). Her sayfada üstte BEDAŞ logo + kurumsal başlık, altta kurumsal/KVKK
footer (canvas page-furniture) çizilir.

Bölümler (her biri ayrı sayfa):
  1. Başlık + Arıza Özeti
  2. Görsel Analiz (RGB + Termal)
  3. Model Teşhisi (ekipman tespiti + hasar sınıflandırması + görsel kanıt)
  4. Termal Değerlendirme (sabit eşik 60/80°C + sebep-sonuç analizi)
  5. Konum Bilgileri
  6. Önerilen Aksiyon
  7. Onay ve İmza

Not: DejaVu Sans emoji glyph içermez → bu şablonda hiçbir emoji kullanılmaz;
durum vurguları renkli metinle yapılır. Yapay zeka model adları (ham mimari
isimleri) rapora yazılmaz; nötr ("yapay zeka modeli" / "sınıflandırma modeli")
ifade kullanılır.
"""
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# === BEDAŞ Kurumsal + Apple iOS Renk Paleti ===
COLOR_BEDAS_BLUE = colors.HexColor('#003A8C')    # BEDAŞ kurumsal mavi
COLOR_BEDAS_GREEN = colors.HexColor('#009A3A')   # BEDAŞ kurumsal yeşil
COLOR_PRIMARY = colors.HexColor('#007aff')       # iOS Blue (bölüm başlıkları)
COLOR_DANGER = colors.HexColor('#ff3b30')        # iOS Red
COLOR_WARNING = colors.HexColor('#ffcc00')       # iOS Yellow
COLOR_SUCCESS = colors.HexColor('#34c759')       # iOS Green
COLOR_ORANGE = colors.HexColor('#ff9500')        # iOS Orange
COLOR_TEXT = colors.HexColor('#1c1c1e')          # Apple text
COLOR_SECONDARY = colors.HexColor('#8e8e93')     # Apple secondary
COLOR_BG_LIGHT = colors.HexColor('#f2f2f7')      # Apple bg
COLOR_BORDER = colors.HexColor('#e5e5ea')        # Apple border

LOGO_PATH = Path('assets/logo/bedas_logo.png')
LOGO_NATIVE_W, LOGO_NATIVE_H = 322, 156          # logo en-boy oranı (~2.06:1)


def _hex(c: colors.Color) -> str:
    """ReportLab Color → '#rrggbb' (markup için; .hexval() '0x..' döndürür)."""
    return '#%02x%02x%02x' % (int(c.red * 255), int(c.green * 255), int(c.blue * 255))


# === Font Yükleyici ===
def _register_fonts() -> str:
    """Türkçe destekli fontları kaydet, font ailesi adını döndür."""
    fonts_dir = Path('assets/fonts')

    # Sıralı tercih: DejaVu → Segoe UI → Default
    candidates = [
        ('DejaVu', 'DejaVuSans.ttf', 'DejaVuSans-Bold.ttf'),
        ('SegoeUI', 'SegoeUI.ttf', 'SegoeUI-Bold.ttf'),
    ]

    for family, regular_file, bold_file in candidates:
        reg_path = fonts_dir / regular_file
        bold_path = fonts_dir / bold_file
        if reg_path.exists() and bold_path.exists():
            try:
                # Aynı isim tekrar kaydedilse de sorun yok (idempotent)
                pdfmetrics.registerFont(TTFont(family, str(reg_path)))
                pdfmetrics.registerFont(TTFont(f'{family}-Bold', str(bold_path)))
                return family
            except Exception:
                continue

    # Fallback: Helvetica (Türkçe destek sınırlı ama çalışır)
    return 'Helvetica'


def _bold(font_family: str) -> str:
    return f'{font_family}-Bold' if font_family != 'Helvetica' else 'Helvetica-Bold'


# === Stil Tanımları ===
def _build_styles(font_family: str) -> Dict[str, ParagraphStyle]:
    """PDF stil sayfası."""
    bold_font = _bold(font_family)
    return {
        'title': ParagraphStyle(
            'Title', fontName=bold_font, fontSize=20, leading=24,
            textColor=COLOR_BEDAS_BLUE, alignment=TA_LEFT, spaceAfter=4,
        ),
        'subtitle': ParagraphStyle(
            'Subtitle', fontName=font_family, fontSize=11, leading=14,
            textColor=COLOR_SECONDARY, alignment=TA_LEFT, spaceAfter=12,
        ),
        'h2': ParagraphStyle(
            'H2', fontName=bold_font, fontSize=13, leading=16,
            textColor=COLOR_BEDAS_BLUE, alignment=TA_LEFT,
            spaceBefore=10, spaceAfter=6,
        ),
        'body': ParagraphStyle(
            'Body', fontName=font_family, fontSize=10, leading=14,
            textColor=COLOR_TEXT, alignment=TA_LEFT, spaceAfter=4,
        ),
        'small': ParagraphStyle(
            'Small', fontName=font_family, fontSize=8, leading=10,
            textColor=COLOR_SECONDARY, alignment=TA_LEFT,
        ),
        'note': ParagraphStyle(
            'Note', fontName=font_family, fontSize=9.5, leading=13,
            textColor=COLOR_TEXT, alignment=TA_LEFT, spaceAfter=4,
            leftIndent=8, borderPadding=2,
        ),
        'footer': ParagraphStyle(
            'Footer', fontName=font_family, fontSize=8, leading=10,
            textColor=COLOR_SECONDARY, alignment=TA_CENTER,
        ),
    }


# === Yardımcı Fonksiyonlar ===
def _risk_color(risk_level: str) -> colors.Color:
    return {
        'critical': COLOR_DANGER, 'high': COLOR_ORANGE,
        'medium': COLOR_WARNING, 'low': COLOR_SUCCESS,
    }.get(risk_level, COLOR_SECONDARY)


def _risk_label_tr(risk_level: str) -> str:
    return {
        'critical': 'KRİTİK', 'high': 'YÜKSEK',
        'medium': 'ORTA', 'low': 'DÜŞÜK',
    }.get(risk_level, str(risk_level).upper())


def _thermal_status_label(status: str) -> str:
    """Termal durum metni — emoji YOK (renk vurgusu tablo stilinde verilir)."""
    return {
        'critical': 'KRİTİK', 'warning': 'DİKKAT', 'normal': 'NORMAL',
    }.get(status, '—')


def _thermal_status_color(status: str) -> colors.Color:
    return {
        'critical': COLOR_DANGER, 'warning': COLOR_WARNING, 'normal': COLOR_SUCCESS,
    }.get(status, COLOR_SECONDARY)


def _get_thermal_recommendation(status: str) -> str:
    if status == 'critical':
        return ("ACİL: Saha ekibi 24-48 saat içinde müdahale etmelidir. "
                "Aşırı ısınma kalıcı hasara yol açabilir. Elektriksel kısa "
                "devre potansiyeli yüksektir.")
    elif status == 'warning':
        return ("DİKKAT: Planlı bakım programına eklenmelidir. "
                "Önümüzdeki 2 hafta içinde detaylı kontrol gereklidir.")
    return "NORMAL: Ekipman güvenli sıcaklık aralığında çalışmaktadır."


def _thermal_cause_effect(status: str) -> Tuple[str, str]:
    """Termal duruma göre (olası neden, önerilen aksiyon) sebep-sonuç metni.

    Mevcut eşik mantığı (60/80°C) üzerinden çalışır; sadece sıcaklık değil,
    teknik yorum sunar."""
    if status == 'critical':
        cause = ("Kritik seviye ısınma. Olası nedenler: gevşek veya oksitlenmiş "
                 "bağlantı (yüksek temas direnci), aşırı yüklenme ya da metal "
                 "bağlantı elemanlarında korozyon.")
        action = ("Hat enerjisi kesilerek bağlantı elemanları kontrol edilmeli; "
                  "gevşek klemensler sıkılmalı, korozyonlu/aşınmış parçalar "
                  "değiştirilmelidir.")
    elif status == 'warning':
        cause = ("Sınır üstü ısınma. Olası nedenler: kısmi gevşeklik veya zamanla "
                 "artan temas direnci.")
        action = ("Planlı bakımda termal ölçüm tekrarlanmalı; bağlantı sıkılığı ve "
                  "temas direnci ölçülmelidir.")
    else:
        cause = "Anormal ısınma tespit edilmedi; değerler güvenli aralıkta."
        action = "Rutin periyodik termal kontrol yeterlidir."
    return cause, action


def _section_title(text: str, styles: Dict) -> list:
    return [
        Paragraph(text, styles['h2']),
        HRFlowable(width="100%", thickness=0.5, color=COLOR_BORDER, spaceAfter=6),
    ]


# === Sayfa Mobilyası (her sayfada üst başlık + alt footer) ===
def _make_on_page(font_family: str, logo_reader: Optional[ImageReader],
                  generated_dt: datetime):
    """``(canvas, doc)`` imzalı bir sayfa-çizim callback'i üret (closure).

    onFirstPage/onLaterPages ReportLab tarafından sadece (canvas, doc) ile
    çağrılır — bu signature KORUNUR."""
    bold_font = _bold(font_family)

    def _draw_header(canvas, doc) -> None:
        page_w, page_h = A4
        left = doc.leftMargin
        right = page_w - doc.rightMargin

        # Logo (sol) — oran korunur (~110pt genişlik). Yoksa metinle kurulur.
        logo_w = 110
        logo_h = logo_w * LOGO_NATIVE_H / LOGO_NATIVE_W
        logo_top = page_h - 0.9 * cm
        text_x = left
        if logo_reader is not None:
            try:
                canvas.drawImage(
                    logo_reader, left, logo_top - logo_h,
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask='auto', anchor='nw',
                )
                text_x = left + logo_w + 14
            except Exception:
                text_x = left

        # Kurumsal başlık metni (logonun sağında, dikeyde ortalı)
        center_y = logo_top - logo_h / 2
        canvas.setFillColor(COLOR_BEDAS_BLUE)
        canvas.setFont(bold_font, 11)
        canvas.drawString(text_x, center_y + 3, 'BOĞAZİÇİ ELEKTRİK DAĞITIM A.Ş.')
        canvas.setFillColor(COLOR_SECONDARY)
        canvas.setFont(font_family, 8.5)
        canvas.drawString(text_x, center_y - 9,
                          'Havai Hat AI Arıza Tespit Sistemi (HHATS)')

        # Mavi ayraç şerit + ince yeşil aksan
        line_y = page_h - 3.0 * cm
        canvas.setStrokeColor(COLOR_BEDAS_BLUE)
        canvas.setLineWidth(1.5)
        canvas.line(left, line_y, right, line_y)
        canvas.setStrokeColor(COLOR_BEDAS_GREEN)
        canvas.setLineWidth(0.8)
        canvas.line(left, line_y - 2.2, right, line_y - 2.2)

    def _draw_footer(canvas, doc) -> None:
        page_w = A4[0]
        left = doc.leftMargin
        right = page_w - doc.rightMargin
        cx = page_w / 2

        # Ayraç çizgi
        canvas.setStrokeColor(COLOR_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(left, 2.0 * cm, right, 2.0 * cm)

        canvas.setFillColor(COLOR_SECONDARY)
        canvas.setFont(font_family, 7.5)
        canvas.drawCentredString(
            cx, 1.62 * cm,
            'Bu rapor BEDAŞ HHATS sistemi tarafından otomatik üretilmiştir.')
        canvas.drawCentredString(
            cx, 1.30 * cm,
            'Bu belge kişisel veri içermez; teknik denetim amaçlıdır.')
        canvas.setFont(font_family, 7)
        canvas.drawCentredString(
            cx, 0.95 * cm,
            f'BEDAŞ Genel Müdürlüğü, İstanbul   |   '
            f'Rapor Tarihi: {generated_dt.strftime("%d.%m.%Y %H:%M")}   |   '
            f'Sayfa {doc.page}')

    def on_page(canvas, doc) -> None:
        canvas.saveState()
        _draw_header(canvas, doc)
        _draw_footer(canvas, doc)
        canvas.restoreState()

    return on_page


# === Bölüm Builders ===
def _build_title_block(styles: Dict, font_family: str, fault_id: str) -> list:
    """Sayfa 1 başlık bloğu: rapor adı + rapor no + tarih."""
    bold_font = _bold(font_family)
    title_data = [[
        Paragraph(
            '<para alignment="left">'
            f'<font name="{bold_font}" size="18" color="{_hex(COLOR_BEDAS_BLUE)}">'
            'SAHA MÜDAHALE RAPORU</font>'
            '</para>',
            styles['body'],
        ),
        Paragraph(
            '<para alignment="right">'
            f'<font name="{font_family}" size="9" color="{_hex(COLOR_SECONDARY)}">'
            f'Rapor No: HHATS-{datetime.now().strftime("%Y%m%d")}-{fault_id[:6].upper()}</font><br/>'
            f'<font name="{font_family}" size="9" color="{_hex(COLOR_SECONDARY)}">'
            f'Tarih: {datetime.now().strftime("%d.%m.%Y %H:%M")}</font>'
            '</para>',
            styles['body'],
        ),
    ]]
    title_table = Table(title_data, colWidths=[10 * cm, 7 * cm])
    title_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
        ('LINEBELOW', (0, 0), (-1, 0), 1, COLOR_BEDAS_BLUE),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ]))
    return [title_table, Spacer(1, 6 * mm)]


def _build_summary_card(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Arıza özet kartı."""
    elements = _section_title('ARIZA ÖZETİ', styles)
    risk_level = fault.get('risk_level', 'medium')
    risk_color = _risk_color(risk_level)
    risk_label = _risk_label_tr(risk_level)
    bold_font = _bold(font_family)

    timestamp_str = (pd.to_datetime(fault['timestamp']).strftime('%d.%m.%Y %H:%M')
                     if pd.notna(fault.get('timestamp')) else '—')
    risk_score = fault.get('risk_score', 0) or 0

    summary_data = [
        ['Arıza ID:', str(fault.get('fault_id', '—'))],
        ['Tespit Tarihi:', timestamp_str],
        ['Uçuş:', str(fault.get('flight_label', '—'))],
        ['Arıza Tipi:', str(fault.get('fault_type_label', '—'))],
        ['Ekipman:', str(fault.get('equipment_label', '—') or '—')],
        ['Risk Seviyesi:', f'{risk_label} ({risk_score:.1f}/100)'],
        ['Durum:', str(fault.get('status_tr', '—'))],
    ]
    summary_table = Table(summary_data, colWidths=[4 * cm, 13 * cm])
    summary_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), bold_font),
        ('FONTNAME', (1, 0), (1, -1), font_family),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLOR_TEXT),
        ('TEXTCOLOR', (1, 5), (1, 5), risk_color),
        ('FONTNAME', (1, 5), (1, 5), bold_font),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LINEBEFORE', (0, 0), (0, -1), 3, risk_color),
        ('LEFTPADDING', (0, 0), (0, -1), 10),
    ]))
    elements.append(summary_table)
    return elements


def _build_images_section(fault: pd.Series, styles: Dict) -> list:
    """RGB + Termal görüntü yan yana."""
    elements = _section_title('GÖRSEL ANALİZ', styles)

    fault_id_short = fault.get('fault_id_short', '')
    image_dir = Path(f'outputs/fault_files/fault_{fault_id_short}')
    rgb_path = image_dir / '01_rgb_original.jpg'
    thermal_path = image_dir / '04_thermal_annotated.png'  # gerçek dosya adı

    if not fault_id_short or (not rgb_path.exists() and not thermal_path.exists()):
        elements.append(Paragraph(
            'Görüntü dosyaları henüz mevcut değil (drone verisi bekleniyor).', styles['body']))
        return elements

    img_w, img_h = 8.5 * cm, 6 * cm
    cells_label, cells_img = [], []
    if rgb_path.exists():
        try:
            cells_label.append(Paragraph('<b>RGB Görüntü</b>', styles['body']))
            cells_img.append(Image(str(rgb_path), width=img_w, height=img_h, kind='proportional'))
        except Exception:
            cells_label.append(Paragraph('RGB yüklenemedi', styles['body']))
            cells_img.append(Paragraph('—', styles['body']))
    if thermal_path.exists():
        try:
            cells_label.append(Paragraph('<b>Termal Görüntü</b>', styles['body']))
            cells_img.append(Image(str(thermal_path), width=img_w, height=img_h, kind='proportional'))
        except Exception:
            cells_label.append(Paragraph('Termal yüklenemedi', styles['body']))
            cells_img.append(Paragraph('—', styles['body']))

    if len(cells_label) == 2:
        img_table = Table([cells_label, cells_img], colWidths=[9 * cm, 9 * cm])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ]))
        elements.append(img_table)
    elif cells_label:
        elements.append(cells_label[0])
        elements.append(cells_img[0])

    return elements


def _is_polymer_insulator(equipment_class: str) -> bool:
    return bool(equipment_class) and 'polymer' in str(equipment_class).lower()


def _build_model_diagnosis_section(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Model Teşhisi: ekipman tespiti + hasar sınıflandırması + görsel kanıt.

    Ham model mimari adları (YOLO/EfficientNet vb.) RAPORA YAZILMAZ — nötr
    ifade kullanılır."""
    elements = _section_title('MODEL TEŞHİSİ', styles)
    bold_font = _bold(font_family)

    equipment_class = fault.get('equipment_class')
    equipment_label = fault.get('equipment_label', '—') or '—'
    det_conf = fault.get('detection_confidence', 0) or 0

    # Ekipman tespiti — nötr ifade
    if not equipment_class or str(equipment_class) in ('—', 'None'):
        elements.append(Paragraph(
            'Bu arıza için ekipman sınıflandırması yapılmadı '
            '(termal kaynaklı tespit). Model teşhisi verisi yok.', styles['body']))
        return elements

    elements.append(Paragraph(
        f'<b>Ekipman Tespiti (yapay zeka modeli):</b> {equipment_label} '
        f'(güven: %{det_conf * 100:.1f})',
        styles['body'],
    ))

    try:
        from src.defect_classifier.inference import (
            CLASS_LABELS_TR,
            YOLO_TO_DEFECT_EQUIPMENT,
            predict_defect,
        )

        defect_equipment = YOLO_TO_DEFECT_EQUIPMENT.get(equipment_class)
        fault_id_short = fault.get('fault_id_short', '')
        image_path = Path(f'outputs/fault_files/fault_{fault_id_short}/01_rgb_original.jpg')

        if defect_equipment and image_path.exists():
            try:
                result = predict_defect(image_path, equipment_class)
                if result:
                    pred_color = COLOR_DANGER if result['is_defective'] else COLOR_SUCCESS
                    elements.append(Paragraph(
                        f'<b>Hasar Sınıflandırması (sınıflandırma modeli):</b> '
                        f'{result["predicted_class_tr"]} '
                        f'(güven: %{result["confidence"] * 100:.1f})',
                        styles['body'],
                    ))
                    elements.append(Spacer(1, 2 * mm))

                    prob_data = [['Sınıf', 'Olasılık']]
                    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
                        prob_data.append([CLASS_LABELS_TR.get(cls, cls), f'%{prob * 100:.1f}'])
                    prob_table = Table(prob_data, colWidths=[5 * cm, 3 * cm])
                    prob_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (-1, 0), bold_font),
                        ('FONTNAME', (0, 1), (-1, -1), font_family),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BACKGROUND', (0, 0), (-1, 0), COLOR_BG_LIGHT),
                        ('TEXTCOLOR', (1, 1), (1, 1), pred_color),
                        ('FONTNAME', (1, 1), (1, 1), bold_font),
                        ('GRID', (0, 0), (-1, -1), 0.3, COLOR_BORDER),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 3),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ]))
                    elements.append(prob_table)
                    elements.append(Spacer(1, 4 * mm))

                    # Polimer izolatör + korozyon (rust) → bilimsel yorum
                    if _is_polymer_insulator(equipment_class) and \
                            result.get('predicted_class') == 'rust':
                        elements.append(Paragraph(
                            f'<b><font color="{_hex(COLOR_BEDAS_BLUE)}">'
                            f'Metal Askıda Korozyon:</font></b> Polimer izolatörlerde '
                            'tespit edilen korozyon, plastik (kompozit) gövdede değil; '
                            'metal askı/bağlantı elemanlarında (askıda korozyon) oluşur. '
                            'Değerlendirme ve müdahale metal bağlantı parçalarına '
                            'yönelik yapılmalıdır.',
                            styles['note'],
                        ))
                        elements.append(Spacer(1, 4 * mm))

                    # Görsel kanıt (Grad-CAM) — PIL ile kaydet (Unicode path, cv2 değil)
                    if result.get('gradcam_overlay') is not None:
                        try:
                            from PIL import Image as PILImage

                            overlay = result['gradcam_overlay']  # BGR (uint8, HxWx3)
                            rgb = overlay[:, :, ::-1]             # BGR → RGB
                            gradcam_path = image_path.parent / 'gradcam_report.jpg'
                            PILImage.fromarray(rgb).save(str(gradcam_path), quality=90)
                            if gradcam_path.exists():
                                elements.append(Paragraph(
                                    '<b>Görsel Kanıt</b> (yapay zeka modelinin '
                                    'incelediği bölgeler):',
                                    styles['body']))
                                elements.append(Spacer(1, 2 * mm))
                                elements.append(Image(str(gradcam_path), width=8 * cm,
                                                      height=8 * cm, kind='proportional'))
                        except Exception as e:  # noqa: BLE001
                            elements.append(Paragraph(
                                f'<i>Görsel kanıt üretilemedi: {str(e)[:60]}</i>',
                                styles['small']))
            except Exception as e:  # noqa: BLE001
                elements.append(Paragraph(
                    f'<i>Hasar sınıflandırması yapılamadı: {str(e)[:60]}</i>',
                    styles['small']))
        else:
            elements.append(Paragraph(
                '<i>Bu ekipman tipi için hasar sınıflandırması mevcut değil '
                'veya görüntü bulunamadı.</i>', styles['small']))
    except ImportError:
        elements.append(Paragraph(
            '<i>Hasar sınıflandırma modülü yüklenemedi.</i>', styles['small']))

    return elements


def _build_thermal_section(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Termal değerlendirme bölümü — sebep-sonuç analiziyle."""
    elements = _section_title('TERMAL DEĞERLENDİRME', styles)

    peak = fault.get('peak_temp')
    delta_raw = fault.get('delta_temp')
    delta = float(delta_raw) if pd.notna(delta_raw) else 0.0
    thermal_status = fault.get('thermal_status')

    if peak is None or pd.isna(peak):
        elements.append(Paragraph('Termal verisi mevcut değil.', styles['body']))
        return elements

    status = thermal_status or 'normal'
    status_color = _thermal_status_color(status)
    status_label = _thermal_status_label(status)
    bold_font = _bold(font_family)

    thermal_data = [
        ['Peak Sıcaklık:', f'{float(peak):.1f}°C', status_label],
        ['Δ Fark:', f'{delta:.1f}°C', ''],
        ['Standart:', 'IEEE C57.106 / IEC 62446-3', ''],
        ['Eşikler:', 'Normal <60°C | Dikkat 60-80°C | Kritik ≥80°C', ''],
    ]
    thermal_table = Table(thermal_data, colWidths=[4 * cm, 9 * cm, 4 * cm])
    thermal_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), bold_font),
        ('FONTNAME', (1, 0), (-1, -1), font_family),
        ('FONTNAME', (2, 0), (2, 0), bold_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTSIZE', (2, 0), (2, 0), 11),
        ('TEXTCOLOR', (2, 0), (2, 0), status_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLOR_TEXT),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LINEBEFORE', (0, 0), (0, -1), 3, status_color),
        ('LEFTPADDING', (0, 0), (0, -1), 10),
    ]))
    elements.append(thermal_table)
    elements.append(Spacer(1, 4 * mm))

    recommendation = _get_thermal_recommendation(status)
    elements.append(Paragraph(
        f'<b><font color="{_hex(status_color)}">Termal Değerlendirme Sonucu:</font></b> '
        f'{recommendation}',
        styles['body'],
    ))
    elements.append(Spacer(1, 3 * mm))

    # Sebep-sonuç analizi: sıcaklık → olası neden → önerilen aksiyon
    cause, action = _thermal_cause_effect(status)
    cause_action = [
        [Paragraph('<b>Olası Neden:</b>', styles['body']),
         Paragraph(cause, styles['body'])],
        [Paragraph('<b>Önerilen Aksiyon:</b>', styles['body']),
         Paragraph(action, styles['body'])],
    ]
    ca_table = Table(cause_action, colWidths=[3.8 * cm, 13.2 * cm])
    ca_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('LINEBEFORE', (0, 0), (0, -1), 3, status_color),
        ('LEFTPADDING', (0, 0), (0, -1), 10),
    ]))
    elements.append(ca_table)
    return elements


def _build_location_section(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Konum bilgileri bölümü."""
    elements = _section_title('KONUM BİLGİLERİ', styles)
    lat = fault.get('latitude')
    lon = fault.get('longitude')

    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        elements.append(Paragraph('GPS koordinatları mevcut değil.', styles['body']))
        return elements

    bold_font = _bold(font_family)
    location_data = [
        ['Enlem (Latitude):', f'{lat:.6f}°N'],
        ['Boylam (Longitude):', f'{lon:.6f}°E'],
        ['Google Maps:', f'https://www.google.com/maps?q={lat},{lon}'],
    ]
    loc_table = Table(location_data, colWidths=[5 * cm, 12 * cm])
    loc_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), bold_font),
        ('FONTNAME', (1, 0), (1, -1), font_family),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLOR_TEXT),
        ('TEXTCOLOR', (1, 2), (1, 2), COLOR_PRIMARY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(loc_table)
    return elements


def _build_action_section(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Önerilen aksiyon bölümü."""
    elements = _section_title('ÖNERİLEN AKSİYON', styles)
    risk_level = fault.get('risk_level', 'medium')
    risk_color = _risk_color(risk_level)

    actions = {
        'critical': (
            'ACİL MÜDAHALE GEREKLİ',
            ['• 24-48 saat içinde saha ekibi gönderilmelidir.',
             '• Ekipman çevresinde güvenlik tedbirleri alınmalıdır.',
             '• Termal anomali + görsel hasar birlikte değerlendirildiğinde '
             'elektriksel arıza riski yüksektir.',
             '• Hat enerjisi kesilerek detaylı inceleme yapılmalıdır.'],
        ),
        'high': (
            'YÜKSEK ÖNCELİKLİ MÜDAHALE',
            ['• 1 hafta içinde saha ekibi planlanmalıdır.',
             '• Ekipman durumu yakından takip edilmelidir.',
             '• Termal kontrol tekrarlanmalıdır.',
             '• Yedek parça hazırlığı yapılmalıdır.'],
        ),
        'medium': (
            'PLANLI BAKIM',
            ['• Mevcut bakım programına eklenmelidir.',
             '• 2-4 hafta içinde detaylı kontrol gereklidir.',
             '• Trend takibi yapılmalıdır.'],
        ),
        'low': (
            'İZLEME',
            ['• Rutin bakım kapsamında değerlendirilmelidir.',
             '• Bir sonraki periyodik kontrolde tekrar incelenmelidir.'],
        ),
    }
    title, lines = actions.get(risk_level, actions['medium'])
    elements.append(Paragraph(
        f'<b><font color="{_hex(risk_color)}" size="12">{title}</font></b>', styles['body']))
    elements.append(Spacer(1, 2 * mm))
    for line in lines:
        elements.append(Paragraph(line, styles['body']))
    return elements


def _build_signature_section(fault: pd.Series, styles: Dict, font_family: str) -> list:
    """Saha ekibi imza alanı."""
    elements = _section_title('ONAY VE İMZA', styles)
    bold_font = _bold(font_family)
    assigned = fault.get('assigned_to_name', '—') or '—'

    signature_data = [
        ['Atanan Personel:', str(assigned), '', ''],
        ['', '', '', ''],
        ['Hazırlayan:', 'Ömer Kürşat Canbaz', 'Onaylayan:', '___________________'],
        ['', '', '', ''],
        ['Tarih:', datetime.now().strftime('%d.%m.%Y'), 'Tarih:', '___________________'],
        ['', '', '', ''],
        ['İmza:', '___________________', 'İmza:', '___________________'],
    ]
    sig_table = Table(signature_data, colWidths=[3.5 * cm, 5 * cm, 3.5 * cm, 5 * cm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), bold_font),
        ('FONTNAME', (2, 0), (2, -1), bold_font),
        ('FONTNAME', (1, 0), (1, -1), font_family),
        ('FONTNAME', (3, 0), (3, -1), font_family),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), COLOR_TEXT),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(sig_table)
    return elements


# === Ana Fonksiyon ===
def generate_field_report(fault: pd.Series, output_path: Optional[Path] = None) -> bytes:
    """Tek bir arıza için saha müdahale PDF raporu üret.

    Args:
        fault: pd.Series — arıza kaydı (Faults sayfasından gelen)
        output_path: Opsiyonel — kaydedilecek dosya yolu

    Returns:
        PDF bytes (Streamlit download için)
    """
    font_family = _register_fonts()
    styles = _build_styles(font_family)
    generated_dt = datetime.now()

    # Logo'yu bir kez yükle (yoksa None → header metinle kurulur, çökme yok)
    logo_reader: Optional[ImageReader] = None
    try:
        if LOGO_PATH.exists():
            logo_reader = ImageReader(str(LOGO_PATH))
    except Exception:
        logo_reader = None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=3.5 * cm, bottomMargin=2.5 * cm,
        title=f"HHATS Saha Raporu - {fault.get('fault_id_short', 'Unknown')}",
        author='BEDAS HHATS',
    )

    fault_id = str(fault.get('fault_id_short', 'unknown'))

    # Her ANA bölüm yeni sayfada (PageBreak ile ayrılır)
    elements: list = []
    elements.extend(_build_title_block(styles, font_family, fault_id))
    elements.extend(_build_summary_card(fault, styles, font_family))
    elements.append(PageBreak())
    elements.extend(_build_images_section(fault, styles))
    elements.append(PageBreak())
    elements.extend(_build_model_diagnosis_section(fault, styles, font_family))
    elements.append(PageBreak())
    elements.extend(_build_thermal_section(fault, styles, font_family))
    elements.append(PageBreak())
    elements.extend(_build_location_section(fault, styles, font_family))
    elements.append(PageBreak())
    elements.extend(_build_action_section(fault, styles, font_family))
    elements.append(PageBreak())
    elements.extend(_build_signature_section(fault, styles, font_family))

    on_page = _make_on_page(font_family, logo_reader, generated_dt)
    doc.build(elements, onFirstPage=on_page, onLaterPages=on_page)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    return pdf_bytes
