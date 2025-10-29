import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import glob

print("=" * 60)
print("🧹 PREPROCESSING DES DONNÉES")
print("=" * 60)

print("\n Chargement des données...")
csv_file = glob.glob('*.csv')[0]
df = pd.read_csv(csv_file)
print(f" {len(df):,} lignes chargées")

print("\n Suppression des colonnes inutiles...")

cols_to_drop = [
    'Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp',
    
    'Fwd_PSH_Flags', 'Fwd_URG_Flags', 'Fwd_Byts/b_Avg', 
    'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg', 'Bwd_Byts/b_Avg',
    'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg', 'Init_Fwd_Win_Byts',
    'Fwd_Seg_Size_Min',
    
    'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min',  
    'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min',  
    'Flow_Duration',      
    'Fwd_IAT_Tot',        
    'Bwd_IAT_Tot',        
    'Bwd_IAT_Mean',       
    'Pkt_Len_Max',      
    
    'Src_Port', 'Dst_Port'
]

cols_to_drop = [col for col in cols_to_drop if col in df.columns]
df = df.drop(columns=cols_to_drop)
print(f" {len(cols_to_drop)} colonnes supprimées")
print(f" Colonnes restantes : {len(df.columns)}")

print("\n Gestion des valeurs manquantes...")
missing_before = df.isnull().sum().sum()
print(f"   Avant : {missing_before:,} NaN")

if missing_before > 0:
    df = df.dropna()
    print(f" Lignes avec NaN supprimées")
else:
    print(" Aucune valeur manquante")

print(f"   Après : {df.isnull().sum().sum()} NaN")

print("\n  Gestion des valeurs infinies...")
inf_count_before = 0
for col in df.select_dtypes(include=[np.number]).columns:
    inf_count_before += df[col].isin([np.inf, -np.inf]).sum()

print(f"   Avant : {inf_count_before:,} valeurs infinies")

if inf_count_before > 0:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f" Valeurs infinies remplacées et lignes supprimées")
else:
    print(" Aucune valeur infinie")

print(f"   Lignes restantes : {len(df):,}")

print("\n Analyse de la colonne cible...")
print(f"\n Distribution de 'Label' :")
print(df['Label'].value_counts())
print(f"\n Distribution de 'Cat' :")
print(df['Cat'].value_counts())

target_col = 'Cat'
print(f"\n Colonne cible choisie : '{target_col}'")

if 'Sub_Cat' in df.columns:
    df = df.drop(columns=['Sub_Cat'])
if target_col == 'Cat' and 'Label' in df.columns:
    df = df.drop(columns=['Label'])

print("\n Encodage de la colonne cible...")
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

print(f" Classes encodées :")
for i, class_name in enumerate(le.classes_):
    count = (df[target_col] == i).sum()
    pct = (count / len(df)) * 100
    print(f"   {i} = {class_name:20s} ({count:6,} - {pct:5.2f}%)")

import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("\n LabelEncoder sauvegardé : label_encoder.pkl")

print("\n  Séparation features / target...")
X = df.drop(columns=[target_col])
y = df[target_col]

print(f" X shape : {X.shape}")
print(f" y shape : {y.shape}")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\n Encodage de {len(categorical_cols)} colonnes catégorielles...")
    for col in categorical_cols:
        le_temp = LabelEncoder()
        X[col] = le_temp.fit_transform(X[col].astype(str))
    print(" Colonnes catégorielles encodées")

print("\n  Division Train/Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f" Train : {len(X_train):,} échantillons ({len(X_train)/len(X)*100:.1f}%)")
print(f" Test  : {len(X_test):,} échantillons ({len(X_test)/len(X)*100:.1f}%)")
print(f" Features : {X_train.shape[1]}")

print("\n Normalisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Données normalisées (StandardScaler)")

print("\n Sauvegarde des données préparées...")
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(X.columns.tolist()))

print(" Fichiers sauvegardés :")
print("   - X_train.npy")
print("   - X_test.npy")
print("   - y_train.npy")
print("   - y_test.npy")
print("   - scaler.pkl")
print("   - label_encoder.pkl")
print("   - feature_names.txt")

print("\n" + "=" * 60)
print(" PREPROCESSING TERMINÉ !")
print("=" * 60)
print(f"\n Résumé :")
print(f"   - Données originales : {625783:,} lignes × 86 colonnes")
print(f"   - Après nettoyage    : {len(X):,} lignes × {X.shape[1]} features")
print(f"   - Train set          : {len(X_train):,} échantillons")
print(f"   - Test set           : {len(X_test):,} échantillons")
print(f"   - Classes            : {len(le.classes_)}")
print(f"\n Prêt pour l'entraînement des modèles !")