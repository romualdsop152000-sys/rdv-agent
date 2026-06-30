"""
Export d'une fiche de briefing en PDF.
Retourne les bytes du PDF pour Streamlit st.download_button.
"""
import re
import io
from fpdf import FPDF

BLUE_DARK  = (15, 52, 96)
BLUE_MID   = (30, 100, 180)
BLUE_LIGHT = (219, 234, 254)
WHITE      = (255, 255, 255)
BLACK      = (20, 20, 20)
GREY       = (100, 100, 100)
GREY_BG    = (245, 247, 250)

SECTION_COLORS = {
    "Entreprise":  (30, 100, 180),
    "Actualit":    (22, 101, 52),
    "Profil":      (109, 40, 217),
    "Angles":      (234, 88, 12),
    "Questions":   (15, 118, 110),
    "Points":      (185, 28, 28),
}

FONTS = "C:/Windows/Fonts/"


def _section_color(title: str) -> tuple:
    for key, color in SECTION_COLORS.items():
        if key in title:
            return color
    return BLUE_DARK


def _strip_emoji(text: str) -> str:
    return re.sub(
        r"[\U0001F300-\U0001FFFF\U00002600-\U000027FF\U00002B00-\U00002BFF]+",
        "", text,
    ).strip()


def _parse_sections(briefing: str) -> list[dict]:
    """
    Parse le markdown de la fiche en sections structurées.
    Retourne une liste de {'level': 1|2, 'title': str, 'lines': [str]}
    """
    sections = []
    current = None
    for raw_line in briefing.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# ") and not line.startswith("## "):
            if current:
                sections.append(current)
            current = {"level": 1, "title": _strip_emoji(line[2:]), "lines": []}
        elif line.startswith("## "):
            if current:
                sections.append(current)
            current = {"level": 2, "title": _strip_emoji(line[3:]), "lines": []}
        else:
            if current is not None:
                current["lines"].append(line)
    if current:
        sections.append(current)
    return sections


class BriefingPDF(FPDF):

    def setup_fonts(self):
        try:
            self.add_font("A", "",   FONTS + "arial.ttf")
            self.add_font("A", "B",  FONTS + "arialbd.ttf")
            self.add_font("A", "I",  FONTS + "ariali.ttf")
            self._font_family = "A"
        except Exception:
            self._font_family = "Helvetica"

    def f(self, style="", size=10):
        self.set_font(self._font_family, style, size)

    def header(self):
        self.set_fill_color(*BLUE_DARK)
        self.rect(0, 0, 210, 12, "F")
        self.f("B", 8)
        self.set_text_color(*WHITE)
        self.set_xy(10, 2)
        self.cell(130, 8, self._header_title)
        self.set_xy(150, 2)
        self.cell(50, 8, "Fiche de Briefing RDV", align="R")
        self.set_text_color(*BLACK)
        self.ln(14)

    def footer(self):
        self.set_y(-12)
        self.f("I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Page {self.page_no()}  |  Agent Préparation RDV", align="C")

    def render_cover(self, title: str, contact: str, company: str, role: str):
        self.set_fill_color(*BLUE_DARK)
        self.rect(0, 0, 210, 80, "F")
        self.f("B", 22)
        self.set_text_color(*WHITE)
        self.set_xy(0, 22)
        self.cell(210, 12, "Fiche de Briefing", align="C")
        self.f("B", 16)
        self.set_xy(0, 38)
        self.cell(210, 10, f"{contact} @ {company}", align="C")
        if role:
            self.f("I", 12)
            self.set_text_color(180, 210, 255)
            self.set_xy(0, 52)
            self.cell(210, 8, role, align="C")
        self.set_draw_color(*WHITE)
        self.set_line_width(0.3)
        self.line(40, 66, 170, 66)
        self.f("", 9)
        self.set_text_color(200, 220, 255)
        self.set_xy(0, 70)
        self.cell(210, 6, "Agent Préparation RDV Commercial — LangGraph + GPT-4o-mini", align="C")

    def render_section(self, section: dict):
        title = section["title"]
        lines = section["lines"]
        level = section["level"]

        if level == 1:
            return

        color = _section_color(title)
        self.set_fill_color(*color)
        self.set_text_color(*WHITE)
        self.f("B", 11)
        self.cell(0, 9, f"  {title}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(*BLACK)
        self.ln(2)

        i = 0
        while i < len(lines):
            line = lines[i]

            # Ligne vide
            if not line.strip():
                self.ln(1)
                i += 1
                continue

            # Bullet point markdown (- texte ou * texte ou 1. texte)
            bullet_match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
            if bullet_match:
                indent_str = bullet_match.group(1)
                content = bullet_match.group(3)
                indent = 8 + len(indent_str) * 2

                # Détecter **bold** dans le contenu
                bold_match = re.match(r"\*\*(.+?)\*\*[:\s]*(.*)", content)
                if bold_match:
                    bold_part = bold_match.group(1)
                    rest = bold_match.group(2).strip()
                    self.set_x(self.get_x() + indent)
                    self.f("B", 9); self.set_text_color(*BLUE_DARK)
                    prefix_w = self.get_string_width(f"• {bold_part}")
                    self.cell(prefix_w + 2, 5.5, f"• {bold_part}")
                    if rest:
                        self.f("", 9); self.set_text_color(*BLACK)
                        self.multi_cell(0, 5.5, f" : {rest}")
                    else:
                        self.ln()
                else:
                    self.set_x(self.get_x() + indent)
                    self.f("", 9); self.set_text_color(*BLACK)
                    self.multi_cell(0, 5.5, f"• {content}")
                i += 1
                continue

            # Texte normal (potentiellement avec **bold**)
            clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            self.f("", 9); self.set_text_color(*BLACK)
            self.multi_cell(0, 5.5, clean)
            i += 1

        self.ln(3)


def briefing_to_pdf_bytes(
    briefing: str,
    contact: str = "",
    company: str = "",
    role: str = "",
) -> bytes:
    """
    Convertit une fiche de briefing markdown en PDF.
    Retourne les bytes du PDF.
    """
    pdf = BriefingPDF()
    pdf._header_title = f"{contact} @ {company}" if contact else "Fiche de Briefing"
    pdf.setup_fonts()
    pdf.set_margins(15, 18, 15)
    pdf.set_auto_page_break(auto=True, margin=16)

    # Page de couverture (sans header/footer)
    pdf.add_page()
    pdf.render_cover(briefing[:50], contact, company, role)

    # Sections de la fiche
    pdf.add_page()
    sections = _parse_sections(briefing)
    for section in sections:
        pdf.render_section(section)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
