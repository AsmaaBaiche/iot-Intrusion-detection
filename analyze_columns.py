import pandas as pd
import numpy as np

df = pd.read_csv('IoT Network Intrusion Dataset.csv')

print("=" * 60)
print(" ANALYSE DES COLONNES INUTILES")
print("=" * 60)


cols_to_drop = []


print("\n1️ IDENTIFIANTS UNIQUES")
print("-" * 60)
for col in df.columns:
    unique_ratio = df[col].nunique() / len(df)
    if unique_ratio > 0.95: 
        print(f"⚠️ {col}: {df[col].nunique():,} valeurs uniques ({unique_ratio*100:.1f}%)")
        if col not in ['Label', 'Cat', 'Sub_Cat']:  
            cols_to_drop.append(col)


print("\n2️ COLONNES TEXTUELLES (IPs, Timestamps)")
print("-" * 60)
text_cols = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp']
for col in text_cols:
    if col in df.columns:
        print(f"📝 {col}: Type = {df[col].dtype}")
        cols_to_drop.append(col)


print("\n3️ COLONNES AVEC UNE SEULE VALEUR")
print("-" * 60)
for col in df.columns:
    if df[col].nunique() == 1:
        print(f" {col}: Valeur unique = {df[col].unique()[0]}")
        cols_to_drop.append(col)

if df.select_dtypes(include=[np.number]).apply(lambda x: x.nunique() == 1).sum() == 0:
    print(" Aucune colonne avec une seule valeur")

print("\n4️ COLONNES AVEC >50% DE NaN")
print("-" * 60)
missing_pct = (df.isnull().sum() / len(df)) * 100
high_missing = missing_pct[missing_pct > 50]
if len(high_missing) > 0:
    for col, pct in high_missing.items():
        print(f" {col}: {pct:.1f}% manquantes")
        cols_to_drop.append(col)
else:
    print(" Aucune colonne avec >50% NaN")


print("\n5️ COLONNES REDONDANTES (Corrélation >0.95)")
print("-" * 60)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr().abs()


high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.95:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))

if high_corr_pairs:
    for col1, col2, corr in high_corr_pairs[:10]:  # Top 10
        print(f"🔄 {col1} ↔ {col2}: Corrélation = {corr:.3f}")
    print(f"\n Conseil: Supprime une des deux colonnes dans chaque paire")
else:
    print(" Aucune corrélation excessive détectée")

# RÉSUMÉ
print("\n" + "=" * 60)
print(" RÉSUMÉ DES COLONNES À SUPPRIMER")
print("=" * 60)
cols_to_drop = list(set(cols_to_drop))  
print(f"\nTotal: {len(cols_to_drop)} colonnes à supprimer\n")
for col in sorted(cols_to_drop):
    print(f"   - {col}")

print(f"\n Colonnes restantes: {len(df.columns) - len(cols_to_drop)} / {len(df.columns)}")

with open('columns_to_drop.txt', 'w') as f:
    f.write('\n'.join(cols_to_drop))
print("\n Liste sauvegardée dans: columns_to_drop.txt")