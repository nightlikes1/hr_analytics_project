from fpdf import FPDF
import tempfile

def generate_pdf_report(emp_data, risk_prob, cost, recs):
    """Analiz sonuçlarını içeren bir PDF dosyası üretir."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Başlık
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(200, 10, "STRATEJIK IK ANALIZ RAPORU", ln=True, align="C")
    pdf.ln(10)
    
    # Çalışan Bilgisi
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(200, 10, f"Calisan ID: {emp_data['EmployeeNumber'].iloc[0]}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(200, 10, f"Departman: {emp_data['Department'].iloc[0]} | Rol: {emp_data['JobRole'].iloc[0]}", ln=True)
    
    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Metrikler Tablosu
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(100, 10, "Metrik", border=1)
    pdf.cell(80, 10, "Deger", border=1, ln=True)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(100, 10, "Ayrilma Riski", border=1)
    pdf.cell(80, 10, f"%{risk_prob*100:.1f}", border=1, ln=True)
    pdf.cell(100, 10, "Tahmini Finansal Kayip", border=1)
    pdf.cell(80, 10, f"${cost:,.0f}", border=1, ln=True)
    
    pdf.ln(10)
    
    # Aksiyon Önerileri
    if recs:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(200, 10, "Eylem Plani ve Oneriler", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for r in recs:
            pdf.multi_cell(190, 8, f"- {r['konu']}: {r['oneri']}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name
