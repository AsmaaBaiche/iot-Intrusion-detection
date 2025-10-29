import pandas as pd

df = pd.read_csv('data/IoT Network Intrusion Dataset.csv') 

print(f" Lignes : {len(df)}")
print(f" Colonnes : {len(df.columns)}")
print(f"\n Colonnes disponibles :")
print(df.columns.tolist())
