# 📊 IBM HR Analytics – Çalışan Ayrılma Analizi

> **IBM HR Analytics Employee Attrition & Performance** veri seti üzerinde uçtan uca veri bilimi projesi: Keşifsel Veri Analizi (EDA), Görselleştirme, Veri Ön İşleme ve Makine Öğrenmesi ile çalışan ayrılma tahmini.

---

## 📁 Proje Yapısı

```
hr_analytics upgraded/
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Orijinal veri seti (1470 x 35)
│   └── hr_attrition_preprocessed.csv            # Ön işlenmiş veri seti (1470 x 46)
│
├── output/
│   ├── attrition_analysis.png                   # 4'lü görselleştirme paneli
│   └── feature_importance_top5.png              # En önemli 5 özellik bar chart
│
├── hr_eda.py                # Adım 1 – Keşifsel Veri Analizi (EDA)
├── hr_visualization.py      # Adım 2 – Veri Görselleştirme (4 grafik)
├── hr_preprocessing.py      # Adım 3 – Veri Ön İşleme
├── hr_model_rf.py           # Adım 4 – Random Forest Modeli
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Proje dokümantasyonu
```

---

## 📋 Veri Seti Hakkında

| Bilgi | Değer |
|-------|-------|
| **Kaynak** | IBM HR Analytics Employee Attrition & Performance |
| **Satır Sayısı** | 1.470 çalışan |
| **Sütun Sayısı** | 35 özellik |
| **Hedef Değişken** | `Attrition` (Yes / No) |
| **Eksik Veri** | Yok (0 null) |
| **Sınıf Dağılımı** | %83.9 Kalan (No) – %16.1 Ayrılan (Yes) |

### Önemli Özellikler

| Özellik | Açıklama |
|---------|----------|
| `Age` | Çalışan yaşı |
| `MonthlyIncome` | Aylık gelir ($) |
| `OverTime` | Fazla mesai durumu (Yes/No) |
| `TotalWorkingYears` | Toplam çalışma yılı |
| `DailyRate` | Günlük ücret |
| `JobSatisfaction` | İş memnuniyeti (1-4) |
| `YearsAtCompany` | Şirketteki çalışma yılı |
| `DistanceFromHome` | Eve uzaklık (km) |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Kütüphanelerin Yüklenmesi

```bash
pip install -r requirements.txt
```

### Scriptlerin Sırasıyla Çalıştırılması

Proje 4 adımdan oluşur. Scriptler sırasıyla çalıştırılmalıdır:

```bash
# Adım 1: Keşifsel Veri Analizi
python hr_eda.py

# Adım 2: Veri Görselleştirme
python hr_visualization.py

# Adım 3: Veri Ön İşleme
python hr_preprocessing.py

# Adım 4: Random Forest Modeli
python hr_model_rf.py
```

> **Not:** `hr_model_rf.py`, ön işlenmiş veri setini (`data/hr_attrition_preprocessed.csv`) kullanır. Bu dosyanın oluşturulması için önce `hr_preprocessing.py` scriptinin çalıştırılması gerekir.

---

## 📑 Proje Adımları

### Adım 1 – Keşifsel Veri Analizi (`hr_eda.py`)

Veri setini tanımaya yönelik temel istatistiksel analiz adımlarını içerir:

- ✅ Veri setinin yüklenmesi ve boyutunun kontrolü
- ✅ İlk 5 satırın incelenmesi
- ✅ Veri tiplerinin listelenmesi (sayısal vs kategorik)
- ✅ Eksik veri (null) kontrolü
- ✅ Sayısal sütunlar için temel istatistikler (mean, std, min, max)
- ✅ Kategorik sütunlar için frekans analizi

### Adım 2 – Veri Görselleştirme (`hr_visualization.py`)

4 farklı grafik ile işten ayrılma analizini görselleştirir:

| # | Grafik | Tür | Bulgu |
|---|--------|-----|-------|
| 1 | İşten Ayrılma Genel Oranı | Pie Chart | %16.1 ayrılma oranı |
| 2 | Aylık Gelir vs Ayrılma | Boxplot | Düşük gelirli çalışanlar daha yatkın |
| 3 | Fazla Mesai vs Ayrılma | Countplot | Fazla mesaicilerde ~3x daha yüksek oran |
| 4 | Yaş Dağılımı vs Ayrılma | Histogram | Genç çalışanlarda (25-35) yüksek oran |

**Çıktı:** `output/attrition_analysis.png`

### Adım 3 – Veri Ön İşleme (`hr_preprocessing.py`)

Makine öğrenmesi modeli için veriyi hazırlar:

- ✅ Varyansı sıfır olan sütunların çıkarılması (`EmployeeCount`, `Over18`, `StandardHours`)
- ✅ `Attrition` hedef değişkeninin sayısal dönüşümü (Yes=1, No=0)
- ✅ Kategorik değişkenlere One-Hot Encoding uygulanması (`drop_first=True`)
- ✅ İşlenmiş verinin CSV olarak kaydedilmesi

**Çıktı:** `data/hr_attrition_preprocessed.csv` (1470 satır × 46 sütun)

### Adım 4 – Random Forest Modeli (`hr_model_rf.py`)

Çalışan ayrılma tahmini için makine öğrenmesi modeli:

- ✅ Veri %80 eğitim / %20 test olarak bölünme (stratified)
- ✅ Random Forest Classifier modeli kurulması ve eğitilmesi
- ✅ Test verisi üzerinde performans metriklerinin raporlanması
- ✅ En önemli 5 özelliğin (Feature Importance) bar chart olarak çizilmesi

**Çıktı:** `output/feature_importance_top5.png`

---

## 📈 Model Sonuçları

### Model Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Algoritma | Random Forest Classifier |
| Ağaç Sayısı (`n_estimators`) | 200 |
| Maksimum Derinlik (`max_depth`) | 15 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `class_weight` | balanced |
| Eğitim / Test Oranı | %80 / %20 |
| `random_state` | 42 |

### Performans Metrikleri (Test Verisi – 294 Örnek)

| Metrik | Skor |
|--------|:----:|
| **Accuracy (Doğruluk)** | 0.8265 |
| **Precision** | 0.3750 |
| **Recall** | 0.1277 |
| **F1-Score** | 0.1905 |

### Confusion Matrix

|  | Tahmin: 0 (Kalan) | Tahmin: 1 (Ayrılan) |
|--|:------------------:|:--------------------:|
| **Gerçek: 0 (Kalan)** | 237 | 10 |
| **Gerçek: 1 (Ayrılan)** | 41 | 6 |

### En Önemli 5 Özellik (Feature Importance)

| Sıra | Özellik | Önem Skoru | Anlamı |
|:----:|---------|:----------:|--------|
| 1 | `MonthlyIncome` | 0.0665 | Aylık gelir |
| 2 | `Age` | 0.0635 | Yaş |
| 3 | `TotalWorkingYears` | 0.0510 | Toplam çalışma yılı |
| 4 | `DailyRate` | 0.0487 | Günlük ücret |
| 5 | `OverTime_Yes` | 0.0473 | Fazla mesai yapma durumu |

---

## 💡 Temel Bulgular ve İK Önerileri

### Bulgular

1. **Düşük Maaş:** Ayrılan çalışanların medyan geliri, kalanlardan belirgin şekilde düşüktür.
2. **Fazla Mesai:** Fazla mesai yapan çalışanlarda ayrılma oranı ~3 kat daha fazladır.
3. **Genç Yaş:** 25-35 yaş arası çalışanlar en yüksek ayrılma oranına sahiptir.

### Stratejik Öneriler

| # | Strateji | Hedef | Beklenen Etki |
|---|----------|-------|---------------|
| 1 | **Rekabetçi Ücret İyileştirmesi** – Piyasa araştırmasına dayalı kademeli maaş artışı ve performans primi | Düşük Maaş | Maaş kaynaklı ayrılmalarda %25-35 azalma |
| 2 | **Fazla Mesai Yönetimi** – İzleme dashboard'u, esnek çalışma modeli, TOIL (izin karşılığı) uygulaması | Fazla Mesai | Fazla mesai saatlerinde %30-40 azalma |
| 3 | **Genç Çalışan Bağlılık Programı** – Mentor atanması, kariyer gelişim planı, eğitim bütçesi, departman rotasyonu | Genç Yaş | Genç çalışan ayrılmasında %20-30 azalma |

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|-----------|----------------|
| **Python 3** | Ana programlama dili |
| **Pandas** | Veri manipülasyonu ve analiz |
| **NumPy** | Sayısal hesaplamalar |
| **Matplotlib** | Grafik ve görselleştirme |
| **Seaborn** | İstatistiksel görselleştirme |
| **Scikit-learn** | Makine öğrenmesi (Random Forest, train/test split, metrikler) |

---

## 📄 Lisans

Bu proje eğitim ve analiz amaçlıdır. Veri seti IBM tarafından Kaggle üzerinden paylaşılmıştır.

---

<p align="center">
  <i>Hazırlayan: Hasan Yiğit Doğanay | Mart 2026</i>
</p>
