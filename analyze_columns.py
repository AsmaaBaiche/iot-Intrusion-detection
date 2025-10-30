import pandas as pd
import numpy as np

df = pd.read_csv('data/IoT Network Intrusion Dataset.csv')

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
        if col not in ['Label', 'Cat', 'Sub_Cat']:   #car cest LES COLONNES À PRÉDIRE (les cibles) !
            cols_to_drop.append(col)


print("\n2️ COLONNES TEXTUELLES (IPs, Timestamps)")
print("-" * 60)
text_cols = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp']
for col in text_cols:
    if col in df.columns: #on verifi que ya vraimenet cette colonne
        print(f" {col}: Type = {df[col].dtype}")
        cols_to_drop.append(col)


print("\n3️ COLONNES AVEC UNE SEULE VALEUR")
print("-" * 60)
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:
        print(f" {col}: Valeur unique = {df[col].unique()[0]}")
        constant_cols.append(col)
        cols_to_drop.append(col)

if len(constant_cols) == 0:
    print("✅ Aucune colonne avec une seule valeur")

print("\n4️ COLONNES AVEC >50% DE NaN")
print("-" * 60)
missing_pct = (df.isnull().sum() / len(df)) * 100
high_missing = missing_pct[missing_pct > 50] #verification 
if len(high_missing) > 0:
    for col, pct in high_missing.items():
        print(f" {col}: {pct:.1f}% manquantes")
        cols_to_drop.append(col)
else:
    print(" Aucune colonne avec >50% NaN")


print("\n5️ COLONNES REDONDANTES (Corrélation >0.95)")
print("-" * 60)

# Sélectionner uniquement les colonnes numériques
numeric_df = df.select_dtypes(include=[np.number])

# Exclure les colonnes cibles de l'analyse de corrélation
target_cols = ['Label', 'Cat', 'Sub_Cat']
numeric_df = numeric_df.drop(columns=[col for col in target_cols if col in numeric_df.columns])

# Calculer la matrice de corrélation absolue
corr_matrix = numeric_df.corr().abs()

# Détecter les paires fortement corrélées
high_corr_pairs = []
cols_to_drop_corr = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > 0.95:
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            high_corr_pairs.append((col1, col2, corr_value))
            
            # Stratégie : Supprimer col1 (première colonne), garder col2
            cols_to_drop_corr.add(col1)
if high_corr_pairs:
    # Trier par corrélation décroissante
    high_corr_pairs_sorted = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)
    
    print(f"⚠️  {len(high_corr_pairs)} paires fortement corrélées détectées\n")
    
    # Afficher les 20 premières paires
    display_limit = min(20, len(high_corr_pairs_sorted))
    for col1, col2, corr in high_corr_pairs_sorted[:display_limit]:
        print(f"   🔄 {col1:30s} ↔ {col2:30s} : {corr:.4f}")
    
    if len(high_corr_pairs) > display_limit:
        print(f"\n   ... et {len(high_corr_pairs) - display_limit} autres paires")
    
    print(f"\n💡 Stratégie : Garder la 2ème colonne de chaque paire (plus stable)")
    print(f"➡️  {len(cols_to_drop_corr)} colonnes redondantes à supprimer")
    
    # Ajouter à la liste globale (éviter les doublons)
    for col in cols_to_drop_corr:
        if col not in cols_to_drop:
            cols_to_drop.append(col)
else:
    print("✅ Aucune corrélation excessive détectée")

    
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