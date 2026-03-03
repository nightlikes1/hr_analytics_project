"""
IBM HR Analytics - Random Forest Classifier Modeli
====================================================
Bu script, on islenmis veri seti uzerinde:
  1. Veriyi %80 egitim / %20 test olarak ayirir
  2. Random Forest Classifier modeli kurar ve egitir
  3. Test verisi uzerinde Accuracy, Precision, Recall, F1-Score raporlar
  4. En onemli 5 ozelligin (feature importance) bar chart'ini cizer
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Islenmis Veriyi Yukleme
# -------------------------------------------------
CSV_PATH = "data/hr_attrition_preprocessed.csv"

print("=" * 70)
print("  IBM HR Analytics - Random Forest Classifier")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Islenmis veri seti yuklendi.")
print(f"    Boyut: {df.shape[0]} satir x {df.shape[1]} sutun")

# -------------------------------------------------
# 2. Ozellik (X) ve Hedef (y) Degiskenlerini Ayirma
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Ozellik ve Hedef Degisken Ayirma")
print("=" * 70)

TARGET = 'Attrition'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\n[2] Hedef degisken: '{TARGET}'")
print(f"    Ozellik sayisi (X): {X.shape[1]}")
print(f"    Ornek sayisi  (y) : {y.shape[0]}")
print(f"    Hedef dagilimi:")
print(f"      0 (Hayir) : {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
print(f"      1 (Evet)  : {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

# -------------------------------------------------
# 3. Egitim / Test Bolunmesi (%80 / %20)
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Egitim / Test Bolunmesi")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n[3] Veri %80 egitim / %20 test olarak ayrildi (stratify=y).")
print(f"    Egitim seti : {X_train.shape[0]} ornek")
print(f"    Test seti   : {X_test.shape[0]} ornek")
print(f"    Egitim hedef dagilimi: 0={(y_train == 0).sum()}, 1={(y_train == 1).sum()}")
print(f"    Test hedef dagilimi  : 0={(y_test == 0).sum()}, 1={(y_test == 1).sum()}")

# -------------------------------------------------
# 4. Random Forest Classifier Modeli Kurma ve Egitme
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Random Forest Classifier - Egitim")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print(f"\n[4] Model parametreleri:")
print(f"    n_estimators   : {rf_model.n_estimators}")
print(f"    max_depth      : {rf_model.max_depth}")
print(f"    min_samples_split: {rf_model.min_samples_split}")
print(f"    min_samples_leaf : {rf_model.min_samples_leaf}")
print(f"    class_weight   : {rf_model.class_weight}")
print(f"    random_state   : {rf_model.random_state}")

print("\n    Model egitiliyor...")
rf_model.fit(X_train, y_train)
print("    [OK] Egitim tamamlandi!")

# -------------------------------------------------
# 5. Test Verisi Uzerinde Tahmin ve Degerlendirme
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Model Degerlendirme (Test Verisi)")
print("=" * 70)

y_pred = rf_model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n[5] Test Seti Sonuclari ({X_test.shape[0]} ornek):")
print(f"    {'─' * 40}")
print(f"    │ {'Metrik':<20} │ {'Skor':>12} │")
print(f"    {'─' * 40}")
print(f"    │ {'Accuracy (Dogruluk)':<20} │ {accuracy:>11.4f} │")
print(f"    │ {'Precision':<20} │ {precision:>11.4f} │")
print(f"    │ {'Recall':<20} │ {recall:>11.4f} │")
print(f"    │ {'F1-Score':<20} │ {f1:>11.4f} │")
print(f"    {'─' * 40}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n    Confusion Matrix:")
print(f"                     Tahmin: 0    Tahmin: 1")
print(f"    Gercek: 0        {cm[0, 0]:<12} {cm[0, 1]}")
print(f"    Gercek: 1        {cm[1, 0]:<12} {cm[1, 1]}")

# Detayli Classification Report
print(f"\n    Detayli Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Hayir (0)', 'Evet (1)']))

# -------------------------------------------------
# 6. Feature Importance - En Onemli 5 Ozellik
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Feature Importance (Ozellik Onemliligi)")
print("=" * 70)

importances = rf_model.feature_importances_
feature_names = X.columns

# DataFrame'e donustur ve sirala
feat_imp_df = pd.DataFrame({
    'Ozellik': feature_names,
    'Onem': importances
}).sort_values(by='Onem', ascending=False)

# En onemli 5 ozellik
top5 = feat_imp_df.head(5)

print(f"\n[6] En Onemli 5 Ozellik:")
print(f"    {'─' * 50}")
print(f"    │ {'Sira':<4} │ {'Ozellik':<30} │ {'Onem':>8} │")
print(f"    {'─' * 50}")
for i, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"    │ {i:<4} │ {row['Ozellik']:<30} │ {row['Onem']:>8.4f} │")
print(f"    {'─' * 50}")

# -------------------------------------------------
# 7. Bar Chart Cizimi
# -------------------------------------------------
print("\n" + "=" * 70)
print("  Bar Chart Olusturuluyor...")
print("=" * 70)

# Renk paleti - koyu lacivertten acik maviye gecis
colors = ['#0D47A1', '#1565C0', '#1E88E5', '#42A5F5', '#90CAF9']

fig, ax = plt.subplots(figsize=(10, 6))

# Ters sirayla (en dusukten en yukseye) ciz ki en yuksek ustte olsun
top5_reversed = top5.iloc[::-1]

bars = ax.barh(
    top5_reversed['Ozellik'],
    top5_reversed['Onem'],
    color=colors[::-1],
    edgecolor='white',
    linewidth=1.2,
    height=0.6
)

# Bar uzerine deger yazma
for bar_item in bars:
    width = bar_item.get_width()
    ax.text(
        width + 0.002,
        bar_item.get_y() + bar_item.get_height() / 2,
        f'{width:.4f}',
        va='center',
        ha='left',
        fontsize=11,
        fontweight='bold',
        color='#333333'
    )

ax.set_xlabel('Önem Skoru (Feature Importance)', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('')
ax.set_title('Random Forest - En Önemli 5 Özellik', fontsize=16, fontweight='bold', pad=15)

# Eksen ayarlari
ax.set_xlim(0, top5['Onem'].max() * 1.20)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=10)

# Grid ve cerceve
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

OUTPUT_PATH = "output/feature_importance_top5.png"
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"\n    [OK] Bar chart '{OUTPUT_PATH}' olarak kaydedildi.")

# -------------------------------------------------
# Ozet
# -------------------------------------------------
print("\n" + "=" * 70)
print("  OZET")
print("=" * 70)
print(f"\n    Hedef Degisken     : {TARGET}")
print(f"    Model              : Random Forest Classifier")
print(f"    Egitim / Test      : %80 / %20 (stratify)")
print(f"    Accuracy           : {accuracy:.4f}")
print(f"    Precision          : {precision:.4f}")
print(f"    Recall             : {recall:.4f}")
print(f"    F1-Score           : {f1:.4f}")
print(f"    En Onemli Ozellik  : {top5.iloc[0]['Ozellik']}")
print(f"    Cikti Grafik       : {OUTPUT_PATH}")

print("\n" + "=" * 70)
print("  [OK] Model egitimi ve degerlendirme tamamlandi!")
print("=" * 70)
