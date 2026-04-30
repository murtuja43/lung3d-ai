import io
import base64
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import (
    HexColor, white, black
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, Image as RLImage,
    HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image as PILImage


# ─────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────
DARK_BG     = HexColor('#0a0e1a')
CARD_BG     = HexColor('#111827')
ACCENT      = HexColor('#00d4ff')
ACCENT_GREEN= HexColor('#00ff88')
ACCENT_RED  = HexColor('#ff4757')
ACCENT_YELLOW=HexColor('#ffa502')
TEXT_LIGHT  = HexColor('#e8f4fd')
TEXT_GRAY   = HexColor('#8899aa')
BORDER      = HexColor('#1e2d45')
WHITE       = white
BLACK       = black


# ─────────────────────────────────────────
# Build PDF report
# ─────────────────────────────────────────
def generate_pdf_report(
    patient_data,
    prediction_result,
    original_b64=None,
    heatmap_b64=None,
    output_path=None
):
    """
    Generate a professional PDF clinical report.

    Args:
        patient_data      : dict with patient parameters
        prediction_result : dict from TBPredictor.predict()
        original_b64      : base64 original X-ray
        heatmap_b64       : base64 heatmap overlay
        output_path       : where to save PDF

    Returns:
        bytes of the PDF (for Flask to send)
    """
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize    = A4,
        leftMargin  = 1.5 * cm,
        rightMargin = 1.5 * cm,
        topMargin   = 1.5 * cm,
        bottomMargin= 1.5 * cm,
    )

    styles   = getSampleStyleSheet()
    elements = []

    # ── Header ──
    elements += build_header(styles)
    elements.append(Spacer(1, 0.4*cm))
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=ACCENT, spaceAfter=0.3*cm
    ))

    # ── Report metadata ──
    elements += build_metadata(styles)
    elements.append(Spacer(1, 0.3*cm))

    # ── Patient info ──
    elements += build_patient_section(patient_data, styles)
    elements.append(Spacer(1, 0.3*cm))

    # ── Prediction result ──
    elements += build_prediction_section(prediction_result, styles)
    elements.append(Spacer(1, 0.3*cm))

    # ── X-ray images ──
    if original_b64 and heatmap_b64:
        elements += build_image_section(
            original_b64, heatmap_b64, styles
        )
        elements.append(Spacer(1, 0.3*cm))

    # ── Analysis breakdown ──
    elements += build_analysis_section(prediction_result, styles)
    elements.append(Spacer(1, 0.3*cm))

    # ── Clinical reasoning ──
    elements += build_reasoning_section(prediction_result, styles)
    elements.append(Spacer(1, 0.3*cm))

    # ── Disclaimer ──
    elements += build_disclaimer(styles)

    # Build PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Optionally save to file
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    return pdf_bytes


# ─────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────
def build_header(styles):
    title_style = ParagraphStyle(
        'Title',
        parent    = styles['Normal'],
        fontSize  = 22,
        textColor = ACCENT,
        alignment = TA_CENTER,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 4,
    )
    sub_style = ParagraphStyle(
        'Sub',
        parent    = styles['Normal'],
        fontSize  = 10,
        textColor = TEXT_GRAY,
        alignment = TA_CENTER,
        fontName  = 'Helvetica',
    )
    return [
        Paragraph("🫁 Lung3D AI", title_style),
        Paragraph(
            "AI-Powered Lung CT Scan Analyzer & TB Detection System",
            sub_style
        ),
    ]


def build_metadata(styles):
    now = datetime.now().strftime("%B %d, %Y at %H:%M")
    meta_style = ParagraphStyle(
        'Meta',
        parent    = styles['Normal'],
        fontSize  = 9,
        textColor = TEXT_GRAY,
        alignment = TA_RIGHT,
    )
    return [Paragraph(f"Report Generated: {now}", meta_style)]


def build_patient_section(patient_data, styles):
    section_style = ParagraphStyle(
        'Section',
        parent    = styles['Normal'],
        fontSize  = 13,
        textColor = ACCENT,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 8,
    )
    elements = [Paragraph("👤 Patient Information", section_style)]

    sex = str(patient_data.get('sex', 'N/A')).title()
    bmi = patient_data.get('bmi', 'N/A')
    try:
        bmi_val = float(bmi)
        if bmi_val < 18.5:
            bmi_label = "Underweight"
        elif bmi_val < 25:
            bmi_label = "Normal"
        elif bmi_val < 30:
            bmi_label = "Overweight"
        else:
            bmi_label = "Obese"
        bmi_str = f"{bmi_val:.1f} ({bmi_label})"
    except:
        bmi_str = str(bmi)

    data = [
        ["Field", "Value", "Field", "Value"],
        ["Age",
         f"{patient_data.get('age', 'N/A')} years",
         "Sex", sex],
        ["BMI",
         bmi_str,
         "Cough Duration",
         f"{patient_data.get('cough_weeks', 0)} weeks"],
        ["Fever",
         "Yes" if patient_data.get('fever') else "No",
         "Night Sweats",
         "Yes" if patient_data.get('night_sweats') else "No"],
        ["Weight Loss",
         "Yes" if patient_data.get('weight_loss') else "No",
         "Fatigue",
         "Yes" if patient_data.get('fatigue') else "No"],
        ["Chest Pain",
         "Yes" if patient_data.get('chest_pain') else "No",
         "TB Contact",
         "Yes" if patient_data.get('tb_contact') else "No"],
        ["Previous TB",
         "Yes" if patient_data.get('prev_tb') else "No",
         "", ""],
    ]

    table = Table(data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BORDER),
        ('TEXTCOLOR',  (0,0), (-1,0), ACCENT),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,1), (-1,-1), HexColor('#0d1520')),
        ('TEXTCOLOR',  (0,1), (-1,-1), TEXT_LIGHT),
        ('TEXTCOLOR',  (0,1), (0,-1), TEXT_GRAY),
        ('TEXTCOLOR',  (2,1), (2,-1), TEXT_GRAY),
        ('GRID',       (0,0), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [HexColor('#0d1520'), HexColor('#111827')]),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    elements.append(table)
    return elements


def build_prediction_section(result, styles):
    section_style = ParagraphStyle(
        'Section',
        parent    = styles['Normal'],
        fontSize  = 13,
        textColor = ACCENT,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 8,
    )
    elements = [Paragraph("🧠 AI Prediction Result", section_style)]

    is_tb      = result['prediction'] == 'TB Detected'
    pred_color = ACCENT_RED if is_tb else ACCENT_GREEN
    conf_pct   = int(result['confidence'] * 100)

    pred_style = ParagraphStyle(
        'Pred',
        parent    = styles['Normal'],
        fontSize  = 18,
        textColor = pred_color,
        fontName  = 'Helvetica-Bold',
        alignment = TA_CENTER,
        spaceAfter= 6,
    )

    icon = "⚠️" if is_tb else "✅"
    elements.append(
        Paragraph(f"{icon} {result['prediction']}", pred_style)
    )

    conf_style = ParagraphStyle(
        'Conf',
        parent    = styles['Normal'],
        fontSize  = 11,
        textColor = TEXT_GRAY,
        alignment = TA_CENTER,
    )
    elements.append(
        Paragraph(f"Confidence Score: {conf_pct}%", conf_style)
    )

    return elements


def build_image_section(original_b64, heatmap_b64, styles):
    section_style = ParagraphStyle(
        'Section',
        parent    = styles['Normal'],
        fontSize  = 13,
        textColor = ACCENT,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 8,
    )
    elements = [Paragraph("🖼️ X-Ray Analysis", section_style)]

    def b64_to_rl_image(b64_str, width, height):
        img_bytes = base64.b64decode(b64_str)
        img_buffer = io.BytesIO(img_bytes)
        return RLImage(img_buffer, width=width, height=height)

    try:
        orig_img    = b64_to_rl_image(original_b64, 7*cm, 7*cm)
        heatmap_img = b64_to_rl_image(heatmap_b64,  7*cm, 7*cm)

        label_style = ParagraphStyle(
            'ImgLabel',
            parent    = styles['Normal'],
            fontSize  = 9,
            textColor = TEXT_GRAY,
            alignment = TA_CENTER,
        )

        img_table = Table(
            [[orig_img, heatmap_img],
             [Paragraph("Original X-Ray", label_style),
              Paragraph("Grad-CAM Heatmap", label_style)]],
            colWidths=[8.5*cm, 8.5*cm]
        )
        img_table.setStyle(TableStyle([
            ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',  (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BACKGROUND', (0,0), (-1,-1), HexColor('#0d1520')),
            ('BOX',     (0,0), (-1,-1), 0.5, BORDER),
        ]))
        elements.append(img_table)
    except Exception as e:
        elements.append(
            Paragraph(f"Images unavailable: {e}", styles['Normal'])
        )

    return elements


def build_analysis_section(result, styles):
    section_style = ParagraphStyle(
        'Section',
        parent    = styles['Normal'],
        fontSize  = 13,
        textColor = ACCENT,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 8,
    )
    elements = [Paragraph("📊 Analysis Breakdown", section_style)]

    cnn_pct      = round(result.get('cnn_probability', 0) * 100, 1)
    clinical_pct = round(result.get('clinical_score',  0) * 100, 1)
    fused_pct    = round(result.get('confidence',      0) * 100, 1)

    data = [
        ["Analysis Component", "Score", "Weight"],
        ["CNN Image Analysis",
         f"{cnn_pct}%", "65%"],
        ["Clinical Risk Score",
         f"{clinical_pct}%", "35%"],
        ["Final Fused Score",
         f"{fused_pct}%", "Combined"],
    ]

    table = Table(data, colWidths=[8*cm, 4*cm, 5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BORDER),
        ('TEXTCOLOR',  (0,0), (-1,0), ACCENT),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,1), (-1,-1), HexColor('#0d1520')),
        ('TEXTCOLOR',  (0,1), (-1,-1), TEXT_LIGHT),
        ('GRID',       (0,0), (-1,-1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [HexColor('#0d1520'), HexColor('#111827')]),
        ('PADDING',    (0,0), (-1,-1), 6),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(table)
    return elements


def build_reasoning_section(result, styles):
    section_style = ParagraphStyle(
        'Section',
        parent    = styles['Normal'],
        fontSize  = 13,
        textColor = ACCENT,
        fontName  = 'Helvetica-Bold',
        spaceAfter= 8,
    )
    elements = [Paragraph("📋 Clinical Reasoning", section_style)]

    reasons = result.get('clinical_reasons', [])
    item_style = ParagraphStyle(
        'Item',
        parent    = styles['Normal'],
        fontSize  = 9,
        textColor = TEXT_LIGHT,
        spaceAfter= 4,
        leftIndent= 10,
    )

    if reasons:
        for r in reasons:
            elements.append(Paragraph(f"✅ {r}", item_style))
    else:
        elements.append(
            Paragraph(
                "✅ No major clinical risk factors identified.",
                item_style
            )
        )

    return elements


def build_disclaimer(styles):
    disc_style = ParagraphStyle(
        'Disc',
        parent    = styles['Normal'],
        fontSize  = 8,
        textColor = TEXT_GRAY,
        alignment = TA_CENTER,
        spaceAfter= 4,
    )
    elements = [
        HRFlowable(
            width="100%", thickness=0.5,
            color=BORDER, spaceBefore=0.2*cm
        ),
        Paragraph(
            "⚠️ DISCLAIMER: This report is generated by an AI "
            "demonstration system and is NOT intended for clinical "
            "use. Always consult a licensed medical professional "
            "for diagnosis and treatment.",
            disc_style
        ),
        Paragraph(
            "Lung3D AI — For educational and demonstration "
            "purposes only.",
            disc_style
        ),
    ]
    return elements


# ─────────────────────────────────────────
# Test the report generator
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing PDF Report Generator...\n")

    sample_patient = {
        'age':          35,
        'sex':          'male',
        'bmi':          17.5,
        'cough_weeks':  4,
        'fever':        True,
        'night_sweats': True,
        'weight_loss':  True,
        'fatigue':      True,
        'chest_pain':   False,
        'tb_contact':   True,
        'prev_tb':      False,
    }

    sample_result = {
        'prediction':       'TB Detected',
        'confidence':       0.78,
        'cnn_probability':  0.82,
        'clinical_score':   0.70,
        'clinical_reasons': [
            'Known TB contact',
            'Chronic cough (4 weeks)',
            '4 TB symptoms present',
            'Low BMI (underweight)',
        ],
        'explanation': 'Sample explanation text',
    }

    pdf_bytes = generate_pdf_report(
        patient_data      = sample_patient,
        prediction_result = sample_result,
        output_path       = 'models/sample_report.pdf'
    )

    print(f"✅ PDF generated!")
    print(f"   Size     : {len(pdf_bytes):,} bytes")
    print(f"   Saved to : models/sample_report.pdf")
    print(f"\n✅ PDF Report Generator is ready!")