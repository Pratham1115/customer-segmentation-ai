from fpdf import FPDF
import io

class BusinessReport(FPDF):
    def header(self):
        # Using built-in helvetica prevents cross-platform font errors
        self.set_font('helvetica', 'B', 15)
        # Title
        self.cell(0, 10, 'AI-Powered Customer Segmentation Report', border=0, ln=1, align='C')
        # Line break
        self.ln(5)
        self.set_draw_color(128, 128, 128)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(insights_text):
    pdf = BusinessReport()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    
    # Clean text of unsupported unicode characters
    clean_text = insights_text.replace('’', "'").replace('—', "-").replace('•', "-")
    lines = clean_text.split('\n')
    
    for line in lines:
        if line.startswith('### '):
            pdf.ln(4)
            pdf.set_font("helvetica", 'B', 12)
            pdf.cell(0, 8, line.replace('### ', ''), ln=1)
            pdf.set_font("helvetica", size=11)
        elif line.startswith('## '):
            pdf.ln(6)
            pdf.set_font("helvetica", 'B', 14)
            pdf.cell(0, 8, line.replace('## ', ''), ln=1)
            pdf.set_font("helvetica", size=11)
        elif line.startswith('# '):
            pdf.ln(6)
            pdf.set_font("helvetica", 'B', 16)
            pdf.cell(0, 10, line.replace('# ', ''), ln=1)
            pdf.set_font("helvetica", size=11)
        else:
            # Multi_cell handles text wrapping automatically
            pdf.multi_cell(0, 6, line)
            
    # Output PDF as bytes so Streamlit can download it directly from memory
    return pdf.output(dest='S')