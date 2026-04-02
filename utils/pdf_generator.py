import os
import urllib.request
from typing import Optional

# ── Font registry ─────────────────────────────────────────────────────────────
# Priority order: best Unicode coverage first.
# NirmalaUI covers all Indian scripts (Hindi, Bengali, Tamil, etc.)
# and is pre-installed on every Windows 10/11 machine.
FONT_CANDIDATES = [
    # Windows 10 / 11
    (r"C:\Windows\Fonts\NirmalaUI.ttf",    "NirmalaUI"),
    (r"C:\Windows\Fonts\Nirmala.ttf",      "Nirmala"),
    (r"C:\Windows\Fonts\arialuni.ttf",     "ArialUnicode"),   # MS Office
    (r"C:\Windows\Fonts\arial.ttf",        "Arial"),
    (r"C:\Windows\Fonts\calibri.ttf",      "Calibri"),
    (r"C:\Windows\Fonts\seguisym.ttf",     "SegoeSymbol"),
    # Linux
    ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",      "NotoSans"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           "DejaVuSans"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "Liberation"),
    # macOS
    ("/Library/Fonts/Arial Unicode.ttf",      "ArialUnicode"),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "ArialUnicode"),
]

# Remote fallback fonts (downloaded once on first use)
REMOTE_FONTS = [
    # NotoSans covers Latin + many scripts
    (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        "NotoSans-Regular.ttf",
    ),
    # NotoSansDevanagari specifically for Hindi
    (
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
        "NotoSansDevanagari-Regular.ttf",
    ),
]


class PDFGenerator:
    """
    Unicode-aware PDF generator.
    Automatically finds or downloads a font that supports all scripts:
    Hindi, Arabic, Chinese, Japanese, Korean, Cyrillic, Latin, etc.
    Uses ReportLab as primary backend, fpdf2 as fallback.
    """

    def __init__(self):
        self._fonts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts"
        )
        os.makedirs(self._fonts_dir, exist_ok=True)

        self._font_path = self._find_or_download_font()
        self._font_name = "UniFont"
        self._backend = self._setup_backend()

    # ── Font discovery ─────────────────────────────────────────────────────────

    def _find_or_download_font(self) -> Optional[str]:
        # 1. Already downloaded into project fonts/
        for _, filename in REMOTE_FONTS:
            local = os.path.join(self._fonts_dir, filename)
            if os.path.exists(local):
                print(f"[PDF] Using cached font: {local}")
                return local

        # 2. System fonts
        for path, name in FONT_CANDIDATES:
            if os.path.exists(path):
                print(f"[PDF] Using system font: {name} ({path})")
                return path

        # 3. Download from remote
        for url, filename in REMOTE_FONTS:
            dest = os.path.join(self._fonts_dir, filename)
            try:
                print(f"[PDF] Downloading font: {filename}...")
                urllib.request.urlretrieve(url, dest)
                if os.path.getsize(dest) > 10_000:
                    print(f"[PDF] Font downloaded: {dest}")
                    return dest
                else:
                    os.remove(dest)  # Incomplete download
            except Exception as e:
                print(f"[PDF] Download failed ({filename}): {e}")

        print("[PDF] WARNING: No Unicode font found — non-Latin text will show as boxes.")
        return None

    # ── Backend setup ──────────────────────────────────────────────────────────

    def _setup_backend(self) -> Optional[str]:
        # Try ReportLab
        try:
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            if self._font_path:
                try:
                    pdfmetrics.registerFont(TTFont(self._font_name, self._font_path))
                    print(f"[PDF] ReportLab ready. Font: {self._font_name}")
                except Exception as e:
                    print(f"[PDF] Font registration failed: {e} — falling back to Helvetica")
                    self._font_name = "Helvetica"
            else:
                self._font_name = "Helvetica"

            return "reportlab"
        except ImportError:
            pass

        # Try fpdf2
        try:
            import fpdf  # noqa
            print("[PDF] fpdf2 backend ready")
            return "fpdf2"
        except ImportError:
            pass

        print("[PDF] ERROR: Install reportlab: pip install reportlab")
        return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def create_pdf(self, text_content: str, output_path: str,
                   title: str = "Document") -> bool:
        if not text_content or not text_content.strip():
            return False
        if self._backend == "reportlab":
            return self._create_reportlab(text_content, output_path, title)
        if self._backend == "fpdf2":
            return self._create_fpdf2(text_content, output_path, title)
        return False

    def is_available(self) -> bool:
        return self._backend is not None

    # ── ReportLab backend ──────────────────────────────────────────────────────

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape characters that break ReportLab Paragraph XML parser."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _create_reportlab(self, text_content: str, output_path: str,
                          title: str) -> bool:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=55, leftMargin=55,
                topMargin=55, bottomMargin=55,
                title=title,
                author="Smart Document Extractor",
            )

            title_style = ParagraphStyle(
                "DocTitle",
                fontName=self._font_name,
                fontSize=15,
                leading=22,
                spaceAfter=14,
                alignment=TA_LEFT,
                textColor="#111827",
            )

            body_style = ParagraphStyle(
                "DocBody",
                fontName=self._font_name,
                fontSize=11,
                leading=18,
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                # wordWrap='CJK' helps with non-Latin line breaking
                wordWrap="CJK",
            )

            story = [
                Paragraph(self._xml_escape(title), title_style),
                Spacer(1, 0.12 * inch),
            ]

            for para in text_content.split("\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(self._xml_escape(para), body_style))
                else:
                    story.append(Spacer(1, 0.08 * inch))

            doc.build(story)
            print(f"[PDF] Created: {output_path}")
            return True

        except Exception as e:
            print(f"[PDF] ReportLab error: {e}")
            # Try plain-text fallback
            return self._create_plain_text_fallback(text_content, output_path, title)

    # ── fpdf2 backend ──────────────────────────────────────────────────────────

    def _create_fpdf2(self, text_content: str, output_path: str,
                      title: str) -> bool:
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            font_registered = False
            if self._font_path:
                try:
                    pdf.add_font("UniFont", style="", fname=self._font_path)
                    font_registered = True
                except Exception as e:
                    print(f"[PDF] fpdf2 font error: {e}")

            def set_font(size):
                if font_registered:
                    pdf.set_font("UniFont", size=size)
                else:
                    pdf.set_font("Helvetica", size=size)

            set_font(15)
            pdf.multi_cell(0, 10, txt=title, align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

            set_font(11)
            for para in text_content.split("\n"):
                para = para.strip()
                if para:
                    pdf.multi_cell(0, 8, txt=para, align="J",
                                   new_x="LMARGIN", new_y="NEXT")
                    pdf.ln(1)
                else:
                    pdf.ln(4)

            pdf.output(output_path)
            print(f"[PDF] Created (fpdf2): {output_path}")
            return True

        except Exception as e:
            print(f"[PDF] fpdf2 error: {e}")
            return self._create_plain_text_fallback(text_content, output_path, title)

    # ── Plain text fallback ────────────────────────────────────────────────────

    @staticmethod
    def _create_plain_text_fallback(text_content: str, output_path: str,
                                    title: str) -> bool:
        """
        Last-resort: write content as UTF-8 text file instead of PDF.
        At least the user won't lose their data.
        """
        try:
            txt_path = output_path.replace(".pdf", "_fallback.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(text_content)
            print(f"[PDF] Saved as plain text fallback: {txt_path}")
            return False
        except Exception:
            return False
