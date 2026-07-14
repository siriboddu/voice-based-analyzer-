import os
from datetime import datetime
from reportlab.pdfgen import canvas
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)
from src.audio_visualizer import AudioVisualizer

class PDFReport:

    def generate(
        self,
        transcript,
        semantic_score,
        feedback,
        confidence,
        strengths,
        improvements,
        features,
        overall_score,
        grade,
        recommendation,
        audio_path  # <-- Corrected parameter name matching app.py exactly
    ):
        os.makedirs("reports", exist_ok=True)
        pdf_path = os.path.join("reports", "Voice_Analysis_Report.pdf")

        document = SimpleDocTemplate(pdf_path)
        styles = getSampleStyleSheet()
        elements = []

        # Generate the waveform image safely using our updated pydub visualizer
        visualizer = AudioVisualizer()
        waveform_img_path = visualizer.save_waveform(audio_path)

        # --------------------------------
        # Title
        # --------------------------------
        elements.append(
            Paragraph(
                "<font size=22><b>VOICE-BASED CONCEPT UNDERSTANDING ANALYSER</b></font>",
                styles["Title"]
            )
        )

        elements.append(
            Paragraph(
                "<b>AI Evaluation Report</b>",
                styles["Heading2"]
            )
        )

        elements.append(
            Paragraph(
                f"Generated on : {datetime.now().strftime('%d-%m-%Y %H:%M')}",
                styles["Normal"]
            )
        )

        elements.append(Spacer(1, 20))

        # --------------------------------
        # Section 1: Transcript
        # --------------------------------
        elements.append(Paragraph("<b>1. Audio Transcript Summary</b>", styles["Heading3"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph(escape(transcript), styles["Normal"]))
        elements.append(Spacer(1, 15))

        # --------------------------------
        # Section 2: Audio Features & Waveform
        # --------------------------------
        elements.append(Paragraph("<b>2. Voice & Speaking Metrics Analysis</b>", styles["Heading3"]))
        elements.append(Spacer(1, 10))

        # Insert generated waveform image safely
        if os.path.exists(waveform_img_path):
            elements.append(Image(waveform_img_path, width=450, height=135))
            elements.append(Spacer(1, 10))

        # Metrics Table Mapping
        data = [
            ["Metric Parameter", "Measured Value"],
            ["Speech Duration", f"{features['duration']} seconds"],
            ["Total Words Spoken", f"{features['word_count']} words"],
            ["Speaking Pace (WPM)", f"{features['wpm']} WPM"],
            ["Voice Energy Level", f"{features['energy']}"],
            ["Silence/Pause Ratio", f"{features['pause_ratio']}%"],
            ["Filler Word Count", f"{features['filler_count']} instances"]
        ]

        t = Table(data, colWidths=[200, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # --------------------------------
        # Section 3: Semantic Understanding
        # --------------------------------
        elements.append(Paragraph("<b>3. Conceptual Semantic Evaluation</b>", styles["Heading3"]))
        elements.append(Spacer(1, 5))
        
        semantic_data = [
            ["Evaluation Factor", "AI Assessment"],
            ["Semantic Alignment Score", f"{semantic_score}/100"],
            ["Conceptual Match Status", feedback],
            ["AI Grading Confidence", confidence]
        ]
        
        st_table = Table(semantic_data, colWidths=[200, 200])
        st_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.cadetblue),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(st_table)
        elements.append(Spacer(1, 20))

        # --------------------------------
        # Section 4: Highlights & Feedback
        # --------------------------------
        elements.append(Paragraph("<b>4. Pedagogical Performance Highlights</b>", styles["Heading3"]))
        elements.append(Spacer(1, 5))
        elements.append(Paragraph("<b>Key Observed Strengths:</b>", styles["Normal"]))
        for s in strengths:
            elements.append(Paragraph(f"• {escape(s)}", styles["Normal"]))
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("<b>Recommended Technical Improvements:</b>", styles["Normal"]))
        for imp in improvements:
            elements.append(Paragraph(f"• {escape(imp)}", styles["Normal"]))

        elements.append(PageBreak())

        # --------------------------------
        # Final Summary Certification Page
        # --------------------------------
        elements.append(Paragraph("<b>5. Performance Certificate Summary</b>", styles["Heading2"]))
        elements.append(Spacer(1, 15))

        score_data = [
            ["Aggregated Performance Metric", "Value / Scale Result"],
            ["Total Final Combined Score", f"{overall_score} / 100 Marks"],
            ["Assigned Conceptual Grade", f"Grade [ {grade} ]"],
            ["Actionable Recommendation", recommendation]
        ]

        score_table = Table(score_data, colWidths=[220, 200])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (1,0), colors.darkgreen),
            ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightgreen),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(score_table)
        elements.append(Spacer(1, 40))

        # Footer Signoff blocks
        elements.append(Paragraph("<b>Voice-Based Concept Understanding Analyser</b>", styles["Heading3"]))
        elements.append(Paragraph("AI-Powered Student Concept Evaluation System", styles["Italic"]))
        elements.append(Paragraph("Generated Automatically using Artificial Intelligence", styles["Italic"]))
        elements.append(Paragraph("© 2026 All Rights Reserved", styles["Italic"]))
        
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("<b>Thank you for using the Voice-Based Concept Understanding Analyser.</b>", styles["Heading3"]))
        elements.append(Paragraph("This report was automatically generated by the AI evaluation engine.", styles["Italic"]))

        document.build(
            elements,
            onFirstPage=PDFReport.add_page_number,
            onLaterPages=PDFReport.add_page_number
        )

        return pdf_path

    @staticmethod
    def add_page_number(canvas, doc):
        page = canvas.getPageNumber()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(560, 20, f"Page {page}")