import pandas as pd
import numpy as np

def generate_report(file_path):
    df = pd.read_csv(file_path)
    
    print("--- EDA and Data Quality Report ---")
    
    print("\n1. Data Shape")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    
    print("\n2. Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(missing_values[missing_values > 0])
        
    print("\n3. Anomalies (Outliers using IQR)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            print(f"Column '{col}': {len(outliers)} outliers")
            
    print("\n4. Categorical Distributions")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(7))

if __name__ == '__main__':
    generate_report('star_classification_dataset.csv')