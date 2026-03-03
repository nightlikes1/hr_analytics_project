def calculate_costs(prob, income):
    """
    İK Ayrılma maliyeti hesaplama: İşe alım + Eğitim + Kayıp Verimlilik.
    Genellikle çalışanın 6 aylık maaşına tekabül eder.
    """
    if prob < 0.2:
        return 0.0
    
    base_cost = income * 6.0
    risk_adjusted_cost = base_cost * prob
    return float(risk_adjusted_cost)

def calculate_roi(saving, investment):
    """
    Yatırım Getirisi (ROI) hesabı.
    """
    if investment <= 0:
        return 0.0
    
    roi = ((saving - investment) / investment) * 100.0
    return float(roi)
