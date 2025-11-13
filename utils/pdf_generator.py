from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

class PDFGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_fonts()
    
    def setup_fonts(self):
        """Setup fonts for different languages"""
        try:
            # You might need to add font files for different languages
            # For now, we'll use default fonts
            pass
        except Exception as e:
            print(f"Font setup error: {str(e)}")
    
    def create_pdf(self, text, output_path, title="Extracted Document"):
        """Create PDF from extracted text"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Content
            content_style = ParagraphStyle(
                'CustomContent',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                leading=14
            )
            
            # Split text into paragraphs
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), content_style))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            return True
            
        except Exception as e:
            print(f"PDF generation error: {str(e)}")
            return False