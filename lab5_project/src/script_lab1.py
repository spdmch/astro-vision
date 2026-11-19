import pandas as pd
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine

def generate_random_timestamp():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31, 23, 59, 59)
    delta = end_date - start_date
    random_days = random.randrange(delta.days)
    random_seconds = random.randrange(24 * 60 * 60)
    return start_date + timedelta(days=random_days, seconds=random_seconds)

df = pd.read_csv('star_classification_dataset.csv')

df['timestamp'] = [generate_random_timestamp() for _ in range(len(df))]

db_connection_url = "postgresql://postgres.tczicntjyunlcnqmskdx:dUXHTdG%24bw8h%26n%2B@aws-1-eu-west-1.pooler.supabase.com:6543/postgres"
engine = create_engine(db_connection_url)

df.to_sql('stars_data', engine, if_exists='replace', index=False)

print("Data successfully uploaded to the database.")