import pandas as pd
from sqlalchemy import create_engine
import os

# -------------------------------------------------
# Veritabani Baglantisi ve Veri Yukleme
# -------------------------------------------------
DB_PATH = "hr_analytics.db"
CSV_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

def setup_database():
    if not os.path.exists(CSV_PATH):
        print(f"HATA: {CSV_PATH} bulunamadi.")
        return

    print(f"[1] Orijinal veri CSV'den okunuyor...")
    df = pd.read_csv(CSV_PATH)

    print(f"[2] SQLite veritabani olusturuluyor: {DB_PATH}")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    # Veriyi veritabanina 'employees' tablosu olarak yaz
    df.to_sql('employees', engine, if_exists='replace', index=False)
    
    print(f"[OK] {len(df)} kayit 'employees' tablosuna aktarildi.")

if __name__ == "__main__":
    setup_database()
