"""
IBM HR Analytics - Veri Gorsellestirme
=======================================
4 farkli grafik ile isten ayrilma analizini gorsellestirir.

Grafikler:
  1. Attrition (Isten Ayrilma) Genel Orani - Pie Chart
  2. Aylik Gelir vs Isten Ayrilma - Boxplot
  3. Fazla Mesai vs Isten Ayrilma - Countplot
  4. Yas Dagilimi vs Isten Ayrilma - Histogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # GUI olmadan kaydet

# -------------------------------------------------
# Stil Ayarlari
# -------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.facecolor'] = '#f8f9fa'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['axes.edgecolor'] = '#dee2e6'
plt.rcParams['grid.color'] = '#e9ecef'

# Renk paleti
COLORS = {
    'stayed': '#2ecc71',    # Yesil - Kalanlar
    'left': '#e74c3c',      # Kirmizi - Ayrilanlar
    'accent': '#3498db',    # Mavi
    'dark': '#2c3e50',      # Koyu
}
palette_attrition = {0: COLORS['stayed'], 1: COLORS['left']}
palette_yn = {'No': COLORS['stayed'], 'Yes': COLORS['left']}

# -------------------------------------------------
# Veri Yukleme
# -------------------------------------------------
CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(CSV_PATH)

# Attrition'i sayisal yap (analiz icin)
df['Attrition_Num'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print("Veri yuklendi. Grafikler olusturuluyor...\n")

# -------------------------------------------------
# 4 Subplot Olustur
# -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('IBM HR Analytics - Isten Ayrilma (Attrition) Analizi',
             fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)
fig.patch.set_facecolor('#f8f9fa')

# =====================================================
# GRAFIK 1: Isten Ayrilma Genel Orani (Pie Chart)
# =====================================================
ax1 = axes[0, 0]
attrition_counts = df['Attrition'].value_counts()
labels = ['Kalan (%{:.1f})'.format(attrition_counts['No'] / len(df) * 100),
          'Ayrilan (%{:.1f})'.format(attrition_counts['Yes'] / len(df) * 100)]
colors_pie = [COLORS['stayed'], COLORS['left']]
explode = (0, 0.08)

wedges, texts, autotexts = ax1.pie(
    attrition_counts.values,
    labels=labels,
    colors=colors_pie,
    explode=explode,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    shadow=True
)
for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

ax1.set_title('Isten Ayrilma Genel Dagilimi',
              fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)

# Toplam bilgisi
total = len(df)
left = attrition_counts['Yes']
ax1.text(0, -1.35, f'Toplam: {total} calisan | Ayrilan: {left} kisi',
         ha='center', fontsize=10, color='#666666', style='italic')

# =====================================================
# GRAFIK 2: Aylik Gelir vs Isten Ayrilma (Boxplot)
# =====================================================
ax2 = axes[0, 1]

sns.boxplot(
    data=df, x='Attrition', y='MonthlyIncome',
    palette=palette_yn, ax=ax2,
    width=0.5, linewidth=1.5,
    flierprops={'marker': 'o', 'markerfacecolor': '#adb5bd', 'markersize': 4, 'alpha': 0.5}
)

ax2.set_title('Aylik Gelir ve Isten Ayrilma Iliskisi',
              fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)
ax2.set_xlabel('Isten Ayrilma Durumu', fontsize=12, fontweight='bold')
ax2.set_ylabel('Aylik Gelir ($)', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Kalan', 'Ayrilan'], fontsize=11)

# Medyan degerlerini goster
medians = df.groupby('Attrition')['MonthlyIncome'].median()
for i, attrition in enumerate(['No', 'Yes']):
    ax2.text(i, medians[attrition] + 300,
             f'Medyan: ${medians[attrition]:,.0f}',
             ha='center', fontsize=10, fontweight='bold', color=COLORS['dark'])

# Ortalama degerleri de goster
means = df.groupby('Attrition')['MonthlyIncome'].mean()
for i, attrition in enumerate(['No', 'Yes']):
    ax2.text(i, medians[attrition] - 800,
             f'Ort: ${means[attrition]:,.0f}',
             ha='center', fontsize=9, color='#555555', style='italic')

# =====================================================
# GRAFIK 3: Fazla Mesai vs Isten Ayrilma (Countplot)
# =====================================================
ax3 = axes[1, 0]

# Gruplanmis veri olustur
overtime_attrition = df.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')

sns.countplot(
    data=df, x='OverTime', hue='Attrition',
    palette=palette_yn, ax=ax3,
    edgecolor='white', linewidth=1.5
)

ax3.set_title('Fazla Mesai ve Isten Ayrilma Iliskisi',
              fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)
ax3.set_xlabel('Fazla Mesai Durumu', fontsize=12, fontweight='bold')
ax3.set_ylabel('Calisan Sayisi', fontsize=12, fontweight='bold')
ax3.set_xticklabels(['Hayir', 'Evet'], fontsize=11)
ax3.legend(title='Ayrilma', labels=['Kalan', 'Ayrilan'],
           fontsize=10, title_fontsize=11, loc='upper right')

# Yuzde oranlarini barlarin uzerine yaz
overtime_rates = df.groupby('OverTime')['Attrition_Num'].mean() * 100
for i, ot_val in enumerate(['No', 'Yes']):
    total_ot = len(df[df['OverTime'] == ot_val])
    left_ot = len(df[(df['OverTime'] == ot_val) & (df['Attrition'] == 'Yes')])
    rate = left_ot / total_ot * 100
    ax3.text(i, total_ot * 0.85 + 20,
             f'Ayrilma: %{rate:.1f}',
             ha='center', fontsize=10, fontweight='bold',
             color=COLORS['left'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['left'], alpha=0.9))

# =====================================================
# GRAFIK 4: Yas Dagilimi vs Isten Ayrilma (Histogram)
# =====================================================
ax4 = axes[1, 1]

# Kalan ve ayrilan gruplari ayir
stayed = df[df['Attrition'] == 'No']['Age']
left_group = df[df['Attrition'] == 'Yes']['Age']

# Histogram ciz
ax4.hist(stayed, bins=20, alpha=0.6, color=COLORS['stayed'],
         label=f'Kalan (n={len(stayed)})', edgecolor='white', linewidth=1)
ax4.hist(left_group, bins=20, alpha=0.7, color=COLORS['left'],
         label=f'Ayrilan (n={len(left_group)})', edgecolor='white', linewidth=1)

ax4.set_title('Yas Dagilimi ve Isten Ayrilma',
              fontsize=14, fontweight='bold', color=COLORS['dark'], pad=15)
ax4.set_xlabel('Yas', fontsize=12, fontweight='bold')
ax4.set_ylabel('Calisan Sayisi', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10, loc='upper right')

# Ortalama yas cizgileri
mean_stayed = stayed.mean()
mean_left = left_group.mean()
ax4.axvline(mean_stayed, color=COLORS['stayed'], linestyle='--', linewidth=2, alpha=0.8)
ax4.axvline(mean_left, color=COLORS['left'], linestyle='--', linewidth=2, alpha=0.8)

ax4.text(mean_stayed + 0.5, ax4.get_ylim()[1] * 0.9,
         f'Ort: {mean_stayed:.1f}', fontsize=9, color=COLORS['stayed'], fontweight='bold')
ax4.text(mean_left + 0.5, ax4.get_ylim()[1] * 0.82,
         f'Ort: {mean_left:.1f}', fontsize=9, color=COLORS['left'], fontweight='bold')

# -------------------------------------------------
# Duzenleme ve Kaydetme
# -------------------------------------------------
plt.tight_layout(rect=[0, 0.02, 1, 0.95])

OUTPUT_PATH = "output/attrition_analysis.png"
import os
os.makedirs("output", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"[OK] Grafik '{OUTPUT_PATH}' olarak kaydedildi.\n")

# -------------------------------------------------
# Yorum ve Analiz
# -------------------------------------------------
print("=" * 70)
print("  GRAFIK YORUMLARI: Sirketten En Cok Kimler Gidiyor?")
print("=" * 70)

print(f"""
  1. GENEL ORAN:
     Toplam {total} calisanin %{left/total*100:.1f}'i ({left} kisi) isten
     ayrilmis. Bu oran nispeten dusuk olsa da, kayip maliyetleri
     dusunuldugunde onemli bir rakamdir.

  2. AYLIK GELIR ETKISI:
     Ayrilan calisanlarin medyan geliri: ${medians['Yes']:,.0f}
     Kalan calisanlarin medyan geliri:   ${medians['No']:,.0f}
     => Dusuk gelirli calisanlar isten ayrilmaya daha yatkin.
        Gelir farki yaklasik ${medians['No'] - medians['Yes']:,.0f} dolar.

  3. FAZLA MESAI ETKISI:
     Fazla mesai YAPMAYANLARDA ayrilma orani: %{overtime_rates['No']:.1f}
     Fazla mesai YAPANLARDA ayrilma orani:    %{overtime_rates['Yes']:.1f}
     => Fazla mesai yapanlar {overtime_rates['Yes']/overtime_rates['No']:.1f}x daha fazla
        ayrilma egiliminde! Work-life balance kritik bir faktor.

  4. YAS ETKISI:
     Ayrilanlarin ortalama yasi: {mean_left:.1f}
     Kalanlarin ortalama yasi:   {mean_stayed:.1f}
     => Genc calisanlar (25-35 yas arasi) daha yuksek ayrilma oranina
        sahip. Kariyer basindaki calisanlar daha mobil.

  SONUC:
  Sirketten en cok ayrilanlar:
    - Dusuk gelirli calisanlar
    - Fazla mesai yapan calisanlar
    - Genc (25-35 yas) calisanlar
  
  ONERILER:
    - Dusuk gelir grubuna yonelik ucret iyilestirmesi
    - Fazla mesai politikasinin gozden gecirilmesi
    - Genc calisanlara kariyer gelisim programlari sunulmasi
""")
