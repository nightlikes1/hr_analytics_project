import pytest
from src.utils.hr_math import calculate_costs, calculate_roi

def test_calculate_costs_low_prob():
    # İşten ayrılma riski %20'nin altındaysa maliyet 0 olmalı
    assert calculate_costs(0.1, 5000) == 0.0

def test_calculate_costs_high_prob():
    # Risk %50, maaş 1000 ise: (1000 * 6) * 0.5 = 3000
    assert calculate_costs(0.5, 1000) == 3000.0

def test_calculate_roi_positive():
    # Tasarruf 5000, Yatırım 1000 ise ROI: ((5000-1000)/1000)*100 = %400
    assert calculate_roi(5000, 1000) == 400.0

def test_calculate_roi_zero_investment():
    # Yatırım 0 ise ROI 0 döner (hata vermemeli)
    assert calculate_roi(5000, 0) == 0.0
