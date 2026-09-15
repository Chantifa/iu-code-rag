# src/data_processing.py
import pandas as pd
from src.database import DatabaseManager, TrainingData, IdealFunctions, TestData
import os
from sqlalchemy import text

class CSVLoader:
    def __init__(self, file_path):
        self.file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_path)

    def load_data(self, session):
        df = pd.read_csv(self.file_path)
        print(f"Loading {self.file_path} with {len(df)} rows")
        if 'y1' in df.columns and 'y50' not in df.columns:  # Training data
            original_rows = len(df)
            df = df.drop_duplicates(subset=['x'], keep='first')
            print(f"Training data: original rows {original_rows}, after dedup {len(df)}")
            for _, row in df.iterrows():
                session.add(TrainingData(x=row['x'], y1=row['y1'], y2=row['y2'], y3=row['y3'], y4=row['y4']))
        elif 'y50' in df.columns:  # Ideal functions
            original_rows = len(df)
            df = df.drop_duplicates(subset=['x'], keep='first')
            print(f"Ideal functions: original rows {original_rows}, after dedup {len(df)}")
            for _, row in df.iterrows():
                session.add(IdealFunctions(**{col: row[col] for col in df.columns}))
        else:  # Test data
            original_rows = len(df)
            df = df.drop_duplicates(subset=['x'], keep='first')
            print(f"Test data: original rows {original_rows}, after dedup {len(df)}")
            for _, row in df.iterrows():
                session.add(TestData(x=row['x'], y=row['y']))

def load_data(db_manager):
    session = db_manager.get_session()
    try:
        with db_manager.engine.connect() as connection:
            connection.execute(text("DROP TABLE IF EXISTS training_data"))
            connection.execute(text("DROP TABLE IF EXISTS ideal_functions"))
            connection.execute(text("DROP TABLE IF EXISTS test_data"))
            connection.commit()
        db_manager.create_tables()

        train_loader = CSVLoader('data/train.csv')
        ideal_loader = CSVLoader('data/ideal.csv')
        test_loader = CSVLoader('data/test.csv')
        train_loader.load_data(session)
        ideal_loader.load_data(session)
        test_loader.load_data(session)
        session.commit()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise
    except Exception as e:
        print(f"Database error: {e}")
        session.rollback()
        raise
    finally:
        session.close()