import joblib
import os
import pandas as pd
import numpy as np

def load_model_resources():
    """Model dosyasını ve özellik listesini yükler."""
    model_path = "output/advanced_xgb_model_v2.joblib"
    if not os.path.exists(model_path):
        model_path = "output/advanced_xgb_model.joblib"
        
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['feature_names'], model_path

def apply_feature_engineering(df):
    """Veri setine yeni özellikler ekler."""
    df = df.copy()
    df['Income_Per_Year'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['Tenure_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['Education_Income_Ratio'] = df['MonthlyIncome'] / (df['Education'] + 1)
    df['Distance_Per_Income'] = df['DistanceFromHome'] / (df['MonthlyIncome'] + 1)
    df['Years_Per_Promotion'] = df['YearsAtCompany'] / (df['YearsSinceLastPromotion'] + 1)
    return df

def preprocess_input(raw_df, expected_features):
    """Girdi verisini modelin beklediği formata sokar."""
    df_feat = apply_feature_engineering(raw_df)
    df_feat = df_feat.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'Attrition'], errors='ignore')
    cat_cols = df_feat.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)
    
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    return df_encoded[expected_features]
