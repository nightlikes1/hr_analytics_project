"""
IBM HR Analytics - Kesifsel Veri Analizi (EDA)
===============================================
Bu script, IBM HR Analytics Employee Attrition & Performance
veri setini yukler ve temel kesifsel veri analizi adimlarini gerceklestirir.

Adimlar:
  1. Veriyi yukleme
  2. Ilk 5 satiri gosterme
  3. Veri tiplerini listeleme
  4. Eksik veri (null values) kontrolu
  5. Temel istatistiksel ozet (mean, std, min, max vb.)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# -------------------------------------------------
# 1. Veriyi Yükleme
# -------------------------------------------------
CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

print("=" * 70)
print("  IBM HR Analytics - Keşifsel Veri Analizi (EDA)")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"\n[OK] Veri seti basariyla yuklendi!")
print(f"   Satir sayisi : {df.shape[0]}")
print(f"   Sutun sayisi : {df.shape[1]}")

# -------------------------------------------------
# 2. İlk 5 Satır
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Ilk 5 Satir")
print("=" * 70)
print(df.head().to_string())

# -------------------------------------------------
# 3. Veri Tipleri (dtypes)
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Veri Tipleri")
print("=" * 70)
print(df.dtypes.to_string())

# Özet: kaç sütun hangi tipte?
print("\nVeri Tipi Dagilimi:")
for dtype_name, count in df.dtypes.value_counts().items():
    print(f"   {str(dtype_name):<10} : {count} sutun")

# -------------------------------------------------
# 4. Eksik Veri (Null) Kontrolü
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Eksik Veri (Null Values) Kontrolu")
print("=" * 70)

missing = df.isnull().sum()
total_missing = missing.sum()

if total_missing == 0:
    print("\n[OK] Veri setinde HIC eksik deger (null) bulunmamaktadir!")
else:
    print(f"\n[!] Toplam {total_missing} eksik deger bulundu:")
    # Sadece eksik olan sutunlari goster
    missing_cols = missing[missing > 0]
    for col, cnt in missing_cols.items():
        pct = (cnt / len(df)) * 100
        print(f"   {col:<35} : {cnt:>5} eksik  ({pct:.1f}%)")

# Detayli null tablosu (tum sutunlar)
print("\nTum Sutunlardaki Null Sayilari:")
null_df = pd.DataFrame({
    "Sutun":       df.columns,
    "Null Sayisi": df.isnull().sum().values,
    "Null (%)":    (df.isnull().sum().values / len(df) * 100).round(2)
})
print(null_df.to_string(index=False))

# -------------------------------------------------
# 5. Temel İstatistiksel Özet
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Sayisal Sutunlar Icin Istatistiksel Ozet")
print("=" * 70)
print(df.describe().T.to_string())

# Kategorik sütunlar için özet
cat_cols = df.select_dtypes(include=["object"]).columns
if len(cat_cols) > 0:
    print("\n" + "=" * 70)
    print("  Kategorik Sutunlar Icin Ozet")
    print("=" * 70)
    print(df[cat_cols].describe().T.to_string())

print("\n" + "=" * 70)
print("  [OK] EDA tamamlandi!")
print("=" * 70)
