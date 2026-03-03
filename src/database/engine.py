from sqlalchemy import create_engine, text
import pandas as pd

def get_db_engine(db_path="hr_analytics.db"):
    """SQLite veritabanı motorunu döner."""
    return create_engine(f"sqlite:///{db_path}")

def load_employees_from_db(engine):
    """Tüm çalışan kayıtlarını veritabanından yükler."""
    return pd.read_sql("SELECT * FROM employees", engine)

def load_single_employee(engine, emp_id):
    """Belirli bir çalışanı veritabanından getirir."""
    query = text("SELECT * FROM employees WHERE EmployeeNumber = :emp_id")
    return pd.read_sql(query, engine, params={"emp_id": int(emp_id)})
