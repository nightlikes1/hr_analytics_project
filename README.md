# 📊 IBM HR Analytics – İleri Seviye Çalışan Ayrılma ve Performans Analizi

> **IBM HR Analytics Employee Attrition & Performance** veri seti üzerinde uçtan uca modern bir veri bilimi ve İK analitiği projesi.
> Proje, makine öğrenmesi (XGBoost, Random Forest) tabanlı tahmin modeli, yerel veritabanı entegrasyonu ve kapsamlı, modüler bir Streamlit dashboard'u sunmaktadır.

---

## 📁 Proje Yapısı ve Modüler Mimari

Proje, maintainability (bakım kolaylığı) ve ölçeklenebilirlik prensipleriyle `src` modülü altında yeniden yapılandırılmıştır:

```
hr_analytics_advanced/
│
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Orijinal veri seti
│   ├── hr_attrition_preprocessed.csv           # Temel ön-işleme çıktısı
│   └── hr_attrition_preprocessed_v2.csv        # Gelişmiş özellik mühendisliği (v2) çıktısı
│
├── hr_analytics.db               # SQLite Yerel Veritabanı
│
├── src/                          # Çekirdek Uygulama Modülleri
│   ├── database/                 # Veritabanı yönetim işlevleri
│   │   └── engine.py             # DB bağlantısı ve CRUD operasyonları
│   ├── models/                   # Makine öğrenmesi servisleri
│   │   └── predictor.py          # Model yükleme ve tahminleme servisi
│   ├── reports/                  # Raporlama ve çıktı servisleri
│   │   └── pdf_generator.py      # PDF simülasyon raporu oluşturucu
│   └── utils/                    # Yardımcı modüller ve matematiksel hesaplamalar
│       └── hr_math.py            # Finansal kayıp ve ROI hesaplamaları
│
├── tests/                        # Birim (Unit) Testleri
│   └── test_hr_math.py           # İK finansal işlemlerinin testleri
│
├── Çalıştırılabilir Scriptler    # Bağımsız Çalışan Ana Scriptler
│   ├── hr_eda.py                 # Keşifsel Veri Analizi
│   ├── hr_visualization.py       # Statik Görselleştirme Raporu
│   ├── hr_preprocessing.py       # Veri Ön İşleme (v1)
│   ├── hr_preprocessing_v2.py    # Gelişmiş Özellik Mühendisliği (v2)
│   ├── hr_model_rf.py            # Random Forest Baseline Model Eğitimi
│   ├── hr_model_advanced.py      # XGBoost + SMOTE Model Eğitimi
│   ├── hr_model_advanced_v2.py   # Versiyon 2 Pipeline Model Eğitimi
│   ├── hr_database_setup.py      # Veritabanı (SQLite) Kurulum ve Enjeksiyonu
│   └── hr_dashboard.py           # Kapsamlı Streamlit Web Arayüzü
│
├── output/                       # Oluşturulan modeller (.joblib) ve grafik çıktıları
├── requirements.txt              # Proje bağımlılıkları
└── README.md                     # Proje dokümantasyonu (Bu dosya)
```

---

## 📋 Veri Seti Hakkında

| Bilgi | Değer |
|-------|-------|
| **Kaynak** | IBM HR Analytics Employee Attrition & Performance |
| **Satır Sayısı** | 1.470 çalışan |
| **Sütun Sayısı** | 35 özellik (Özellik mühendisliği ile artırıldı) |
| **Hedef Değişken** | `Attrition` (Yes / No) |
| **Sınıf Dağılımı** | %83.9 Kalan (No) – %16.1 Ayrılan (Yes) |

---

## 🚀 Yeni Streamlit Dashboard Özellikleri

Model sonuçlarını gerçek hayata entegre etmeyi sağlayan `hr_dashboard.py` aşağıdaki modülleri içerir:

1. **🏠 Karşılama & Uyarılar:** Tüm personeli tarayarak kritik/yüksek kaçış riski olanları anında uyaran genel bakış ekranı.
2. **📁 Veri Portalı:** Toplu yeni CSV verilerini yükleyerek anında risk sonuçlarını indirme özelliği.
3. **🏥 Departman Analizi:** Isı haritaları (Heatmap), Ağaç haritaları (Treemap) ve hiyerarşik raporlarla departman ve rol bazlı risk tespiti.
4. **📊 9-Box Yetenek Matrisi:** Çalışan potansiyeli ve risk skorlarını geleneksel 9-Box metoduyla ızgaralar halinde sunan yetenek yönetimi analiz ekranı.
5. **🔮 Tahmin & What-If:** Çalışanın maaşı, rolü veya mesai durumu gibi şartları değiştirildiğinde, ayrılma riskinin nasıl değişeceğini anlık gösteren interaktif simülatör (Gauge grafikleri ile).
6. **💰 Müdahale & ROI Analizi:** Bir çalışana verilecek ek maaş, mentorluk gibi "Müdahale Paketlerinin" yatırım getirisini (ROI) ve şirkete sağlayacağı tasarrufu Waterfall (şelale) grafikleri ile hesaplama.
7. **👯 Çalışan Kıyaslama:** İki farklı çalışanı aynı anda yan yana getiren, **8-boyutlu radar grafikleri** ve doğrudan farklılık göstergeleri içeren derinlemesine detay ekranı.
8. **🤖 Strateji Uzmanı:** Maaş, Memnuniyet, İş-Yaşam Dengesi, Mesai gibi alt boyutların geneline dair istatistikleri doğrudan model bazlı yorumlayan özet analiz sayfası.
9. **🔍 Model Şeffaflığı (SHAP):** Yapay zeka tahminlerinin hangi değişkenler tarafından ("Neden bu çalışan %80 riskli?") ne kadar etkilendiğini anlatan SHAP açıklanabilirlik raporları.

---

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimler

Sisteminizde `Python 3.8+` yüklü olmalıdır.

```bash
# Gerekli bağımlılıkların kurulumu:
pip install -r requirements.txt
```

### 2. Pipeline'ın (Adımların) Çalıştırılması

Ön işleme, model eğitimi ve veritabanı adımlarını çalıştırmak için aşağıdaki sırayı izleyebilirsiniz:

```bash
# Adım 1: Veri Ön İşleme (V2 özellik mühendisliği dahil)
python hr_preprocessing_v2.py

# Adım 2: Model Eğitimleri (XGBoost veya Random Forest)
python hr_model_advanced_v2.py

# Adım 3: SQLite Veritabanını Doldurma (Çalışan Envanteri)
python hr_database_setup.py
```

### 3. Dashboard'u Başlatma

Ana uygulamayı başlatmak için:
```bash
streamlit run hr_dashboard.py
```

### 4. Testleri Çalıştırma

Kod bütünlüğünü (finansal metrik hesaplamaları vb.) doğrulamak için TDD kapsamında yazılmış testleri çalıştırabilirsiniz:
```bash
pytest
```

---

## 📈 Model Sonuçları

Gelişmiş modellerde sentetik veri dengelenmesi (SMOTE) ve özellik mühendisliği (v2 Pipeline) ile özellikle ayrılan sınıfı tespit etmedeki **Recall** skorları dramatik oranda artırılmıştır. Modeller hyperparameter tuning için **Optuna** kullanılarak optimize edilmiştir.

### Temel Özellik Önem Skorları (SHAP & Feature Importance)

Modelin bir çalışanın ayrılmasında rol oynayan en kritik bulduğu etkenler:
1. **Aylık Gelir (MonthlyIncome)**
2. **Fazla Mesai (OverTime)**
3. **Toplam Çalışma Yılları (TotalWorkingYears)**
4. **Yaş (Age)**
5. **Günlük Ücret (DailyRate)**

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

| Teknoloji | Kullanım Alanı |
|-----------|----------------|
| **Python** | Çekirdek dil |
| **Streamlit** | Web arayüzü ve Dashboard inşası |
| **Scikit-learn & XGBoost** | Makine Öğrenmesi (Pipeline, Random Forest, Gradient Boosting) |
| **Imbalanced-learn (SMOTE)** | Dengesiz veri setleri sentezleme |
| **Optuna** | Hyperparameter Tuning |
| **Pandas & NumPy** | Veri yapılandırması ve işlemleri |
| **Plotly & Seaborn** | Etkileşimli (interaktif) ve istatistiksel grafikler |
| **SHAP** | Explainable AI (Model Şeffaflığı) |
| **SQLite (SQLAlchemy)** | Yerel veritabanı entegrasyonu |
| **Pytest** | Birim Testleri |
| **FPDF2** | Dinamik Rapor (PDF) Üretimi |

---

<p align="center">
  <i>Hazırlayan: Hasan Yiğit Doğanay | Güncelleme: 2026</i>
</p>
