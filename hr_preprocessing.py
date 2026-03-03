"""
IBM HR Analytics - Veri On Isleme (Preprocessing)
===================================================
Bu script, veri setini makine ogrenmesi modelleri icin hazirlar.

Adimlar:
  1. Veriyi yukleme
  2. Varyansi sifir olan (tum satirlarda ayni deger) sutunlari dusurme
  3. 'Attrition' sutununu sayisal hale getirme (Yes=1, No=0)
  4. Diger kategorik degiskenleri One-Hot Encoding ile donusturme
  5. Son veri setinin boyutunu (shape) yazdirma
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

# -------------------------------------------------
# 1. Veriyi Yukleme
# -------------------------------------------------
CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

print("=" * 70)
print("  IBM HR Analytics - Veri On Isleme (Preprocessing)")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Veri seti yuklendi.")
print(f"    Baslangic boyutu: {df.shape[0]} satir x {df.shape[1]} sutun")

# -------------------------------------------------
# 2. Varyansi Sifir Olan Sutunlari Dusurme
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Varyansi Sifir Olan Sutunlar")
print("=" * 70)

# Sayisal sutunlarda varyans = 0 olanlari bul
zero_var_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        zero_var_cols.append(col)

print(f"\n[2] Tum satirlarda ayni degere sahip (varyans=0) sutunlar:")
for col in zero_var_cols:
    unique_val = df[col].unique()[0]
    print(f"    - {col:<20} (sabit deger: {unique_val})")

# Bu sutunlari dusur
df.drop(columns=zero_var_cols, inplace=True)
print(f"\n    => {len(zero_var_cols)} sutun dusurildi.")
print(f"    Yeni boyut: {df.shape[0]} satir x {df.shape[1]} sutun")

# -------------------------------------------------
# 3. 'Attrition' Sutununu Sayisal Hale Getirme
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Attrition (Isten Ayrilma) Donusumu")
print("=" * 70)

print(f"\n[3] 'Attrition' sutunu donusturuluyor (Yes=1, No=0)...")
print(f"    Donusum oncesi dagilim:")
print(f"    {df['Attrition'].value_counts().to_dict()}")

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"    Donusum sonrasi dagilim:")
print(f"    {df['Attrition'].value_counts().to_dict()}")
print(f"    Veri tipi: {df['Attrition'].dtype}")

# -------------------------------------------------
# 4. Kategorik Degiskenleri One-Hot Encoding
# -------------------------------------------------
print("\n" + "=" * 70)
print("  One-Hot Encoding (Kategorik Degiskenler)")
print("=" * 70)

# Kalan kategorik sutunlari bul
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n[4] One-Hot Encoding uygulanacak kategorik sutunlar ({len(cat_cols)} adet):")
for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"    - {col:<25} ({n_unique} benzersiz deger)")

# One-Hot Encoding uygula (drop_first=True ile multicollinearity onlenir)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

print(f"\n    => One-Hot Encoding tamamlandi.")

# -------------------------------------------------
# 5. Son Durumu Kontrol Et
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Son Durum Ozeti")
print("=" * 70)

print(f"\n[5] Veri setinin son hali:")
print(f"    Baslangic boyutu : 1470 satir x 35 sutun")
print(f"    Son boyut        : {df_encoded.shape[0]} satir x {df_encoded.shape[1]} sutun")
print(f"    Dusurulen sutunlar: {zero_var_cols}")
print(f"    Eklenen dummy sutun sayisi: {df_encoded.shape[1] - df.shape[1] + len(cat_cols)}")

# Veri tiplerinin kontrolu
print(f"\n    Veri Tipi Dagilimi:")
for dtype_name, count in df_encoded.dtypes.value_counts().items():
    print(f"      {str(dtype_name):<10} : {count} sutun")

# Null kontrolu
total_nulls = df_encoded.isnull().sum().sum()
print(f"\n    Toplam null deger: {total_nulls}")

# Ilk 5 satir (ilk 10 sutun)
print(f"\n    Ilk 5 satir (ilk 10 sutun):")
print(df_encoded.iloc[:5, :10].to_string())

# Tum sutun isimleri
print(f"\n    Tum sutun isimleri ({df_encoded.shape[1]} adet):")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"      {i:>3}. {col}")

# -------------------------------------------------
# 6. Islenmis veriyi kaydet
# -------------------------------------------------
OUTPUT_PATH = "data/hr_attrition_preprocessed.csv"
df_encoded.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Islenmis veri '{OUTPUT_PATH}' dosyasina kaydedildi.")

print("\n" + "=" * 70)
print("  [OK] Veri on isleme tamamlandi!")
print("=" * 70)
