"""
IBM HR Analytics - Model Eğitimi V2 (Advanced Features & SMOTE-Tomek)
===================================================================
Yeni özellik mühendisliği adımları ile model başarısını maksimize eder.
"""

import sys
import io
import pandas as pd
import numpy as np
import joblib

# Encoding Fix for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, f1_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna

# -------------------------------------------------
# 1. Yeni Veri Setini Yükleme
# -------------------------------------------------
CSV_PATH = "data/hr_attrition_preprocessed_v2.csv"
df = pd.read_csv(CSV_PATH)
TARGET = 'Attrition'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# -------------------------------------------------
# 2. Optuna ile En İyileştirme
# -------------------------------------------------
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'random_state': 42,
        'verbosity': 0
    }

    # SMOTETomek: Hem veri dengeler hem gürültüyü temizler
    pipeline = ImbPipeline([
        ('smote', SMOTETomek(random_state=42)),
        ('classifier', XGBClassifier(**param))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Recall ve F1-Score dengesine bakiyoruz
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    return score.mean()

print("\nModel V2 Egitimi ve Optimizasyonu Basladi (Yeni Ozelliklerle)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# -------------------------------------------------
# 3. Final Model Kaydı
# -------------------------------------------------
best_params = study.best_params
best_params['random_state'] = 42

final_pipeline = ImbPipeline([
    ('smote', SMOTETomek(random_state=42)),
    ('classifier', XGBClassifier(**best_params))
])

final_pipeline.fit(X_train, y_train)

# Kayıt
MODEL_SAVE_PATH = "output/advanced_xgb_model_v2.joblib"
# Pipeline'i direkt kaydediyoruz (preprocessing dashboard icinde manuel yapilacak)
model_data = {
    'model': final_pipeline['classifier'],
    'feature_names': list(X.columns)
}
joblib.dump(model_data, MODEL_SAVE_PATH)

# Test Sonuclari
y_pred = final_pipeline.predict(X_test)
print("\n" + "=" * 50)
print("  MODEL V2 PERFORMANS SONUÇLARI")
print("=" * 50)
print(f"  Recall (Ayrılanları Yakalama): {recall_score(y_test, y_pred):.4f}")
print(f"  F1-Score (Dengeli Skor)     : {f1_score(y_test, y_pred):.4f}")
print("-" * 50)
print(classification_report(y_test, y_pred))
