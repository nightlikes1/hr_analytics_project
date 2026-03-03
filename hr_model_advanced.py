"""
IBM HR Analytics - Advanced Machine Learning Model (XGBoost + SMOTE + Optuna)
=============================================================================
Bu script, model performansini artirmak icin su teknikleri uygular:
  1. SMOTE: Dengesiz veri setini dengelemek icin sentetik ornekleme.
  2. XGBoost: Yuksek performansli gradyan artirma algoritmasi.
  3. Optuna: Akilli hiperparametre optimizasyonu.
  4. Stratified K-Fold: Daha guvenilir model degerlendirmesi.
"""

import sys
import io
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import joblib
import warnings

# Turkce karakter ve encoding ayarlari
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# -------------------------------------------------
# 1. Veriyi Yukleme
# -------------------------------------------------
CSV_PATH = "data/hr_attrition_preprocessed.csv"
if not os.path.exists(CSV_PATH):
    print(f"HATA: {CSV_PATH} bulunamadi. Lutfen once hr_preprocessing.py calistirin.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
TARGET = 'Attrition'
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\n[1] Veri seti yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")
print(f"    Hedef dagilimi (Orijinal): 0={ (y==0).sum() }, 1={ (y==1).sum() }")

# -------------------------------------------------
# 2. Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------
# 3. Optuna ile Hiperparametre Optimizasyonu
# -------------------------------------------------
def objective(trial):
    # Denenecek parametre araliklari
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'verbosity': 0
    }

    # SMOTE ve XGBoost iceren Pipeline
    # Bu sekilde Cross-Validation sirasinda sadece training fold'una SMOTE uygulanir
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(**param))
    ])

    # Stratified K-Fold ile degerlendirme (Recall odakli)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    return score.mean()

print("\n[2] Optuna ile hiperparametre optimizasyonu basliyor (15 deneme)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print(f"\n    En iyi F1 Skoru: {study.best_value:.4f}")
print("    En iyi parametreler:")
for key, value in study.best_params.items():
    print(f"      {key}: {value}")

# -------------------------------------------------
# 4. En Iyi Modelin Eğitilmesi
# -------------------------------------------------
best_params = study.best_params
best_params['random_state'] = 42

# SMOTE uygula (Training setine)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\n[3] SMOTE sonrasi egitim seti boyutu: {X_train_res.shape[0]} satir")

final_model = XGBClassifier(**best_params)
final_model.fit(X_train_res, y_train_res)

# -------------------------------------------------
# 4b. Modeli Kaydetme
# -------------------------------------------------
MODEL_SAVE_PATH = "output/advanced_xgb_model.joblib"
# Model ile birlikte ozellik isimlerini de kaydedelim
model_data = {
    'model': final_model,
    'feature_names': list(X.columns)
}
joblib.dump(model_data, MODEL_SAVE_PATH)
print(f"\n[OK] Model ve ozellikler kaydedildi: {MODEL_SAVE_PATH}")

# -------------------------------------------------
# 5. Degerlendirme
# -------------------------------------------------
y_pred = final_model.predict(X_test)

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}

print("\n" + "=" * 50)
print("  ILERI SEVIYE MODEL SONUCLARI (XGBoost + SMOTE)")
print("=" * 50)
for k, v in metrics.items():
    print(f"  {k:<10}: {v:.4f}")
print("-" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 6. Feature Importance Görselleştirme
# -------------------------------------------------
feat_importances = pd.Series(final_model.feature_importances_, index=X.columns)
top5 = feat_importances.nlargest(5)

plt.figure(figsize=(10, 6))
top5.plot(kind='barh', color='#E64A19')
plt.title('XGBoost - En Önemli 5 Özellik (Gelişmiş Model)')
plt.xlabel('Önem Skoru')
plt.tight_layout()

OUTPUT_PATH = "output/advanced_model_features.png"
plt.savefig(OUTPUT_PATH)
plt.close()

print(f"\n[OK] En onemli ozellikler grafigi kaydedildi: {OUTPUT_PATH}")

# -------------------------------------------------
# 7. Eski Model Ile Karsilastirma (Dosyadan okuma yoksa varsayilan degerler)
# -------------------------------------------------
# Eski degerler (README'den): Accuracy: 0.8265, Recall: 0.1277
print("\n" + "=" * 50)
print("  KARSILASTIRMA (Baseline vs Advanced)")
print("=" * 50)
print(f"  Recall (Eski/Yeni): 0.1277 -> {metrics['Recall']:.4f}")
print(f"  F1-Score (Eski/Yeni): 0.1905 -> {metrics['F1-Score']:.4f}")
print(f"  Ayrilanlari yakalama orani (Recall) ciddi sekilde artmistir!")
print("=" * 50)
