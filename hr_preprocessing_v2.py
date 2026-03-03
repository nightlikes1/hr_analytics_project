"""
IBM HR Analytics - Gelişmiş Ön İşleme (Feature Engineering Dahil)
==============================================================
Bu script, model başarısını artırmak için yeni özellikler üretir (Feature Engineering).
"""

import pandas as pd
import numpy as np
import os

def run_advanced_preprocessing():
    CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(CSV_PATH):
        print(f"HATA: {CSV_PATH} bulunamadı.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # -------------------------------------------------
    # A. ÖZELLİK MÜHENDİSLİĞİ (Feature Engineering)
    # -------------------------------------------------
    print("[1] Yeni özellikler üretiliyor...")
    
    # 1. Yaş başına kazanç (Tecrübe ve verimlilik dengesi)
    df['Income_Per_Year'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    
    # 2. Şirketteki yıl oranı (Sadakat ölçüsü)
    df['Tenure_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # 3. Eğitim Seviyesi x Maaş (Eğitimin karşılığını alıyor mu?)
    df['Education_Income_Ratio'] = df['MonthlyIncome'] / (df['Education'] + 1)
    
    # 4. Uzaklık x Maaş (Uzak mesafe için maaş tatmin edici düzeyde mi?)
    df['Distance_Per_Income'] = df['DistanceFromHome'] / (df['MonthlyIncome'] + 1)
    
    # 5. Terhis hızı (Yılların terfi başına düşen sayısı)
    df['Years_Per_Promotion'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)

    # -------------------------------------------------
    # B. Klasik Ön İşleme Adımları
    # -------------------------------------------------
    # Varyansı sıfır olan sütunları düşür
    zero_var_cols = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=zero_var_cols, inplace=True)
    
    # Attrition donusumu
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Kategorik sütunları encodelayalım
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Islenmis veriyi kaydet
    OUTPUT_PATH = "data/hr_attrition_preprocessed_v2.csv"
    df_encoded.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Yeni veri seti kaydedildi: {OUTPUT_PATH} ({df_encoded.shape[1]} sütun)")
    
    return OUTPUT_PATH

if __name__ == "__main__":
    run_advanced_preprocessing()
