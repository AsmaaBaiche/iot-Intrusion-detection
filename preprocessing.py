from pathlib import Path 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

print("=" * 80)
print(" PREPROCESSING DES DONNÉES")
print("=" * 80)


# 1️⃣ CHARGEMENT

print("\n1️⃣ Chargement des données...")

csv_path = Path('data/IoT Network Intrusion Dataset.csv')
df = pd.read_csv(csv_path)
original_rows, original_cols = df.shape

print(f"✅ {original_rows:,} lignes × {original_cols} colonnes chargées")


# 2️⃣ SUPPRESSION DES COLONNES (depuis analyze_columns.py)

print("\n2️⃣ Suppression des colonnes inutiles...")

cols_drop_file = Path('columns_to_drop.txt')

if cols_drop_file.exists():
    print(f"    Utilisation de {cols_drop_file.name}...")
    with open(cols_drop_file, 'r', encoding='utf-8') as f:
        cols_to_drop = [line.strip() for line in f if line.strip()]
    
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    print(f" {len(cols_to_drop)} colonnes supprimées")
    print(f"   Colonnes restantes : {len(df.columns)}")
else:
    print(f"  {cols_drop_file.name} introuvable")
    print(f"   → Lance d'abord analyze_columns.py")
    exit(1)


# 3️⃣ NETTOYAGE DES NaN ET INF

print("\n Nettoyage des NaN et valeurs infinies...")

rows_before = len(df)

# NaN
missing_before = df.isnull().sum().sum()
if missing_before > 0:
    df = df.dropna()
    print(f"   NaN : {missing_before:,} détectés → {rows_before - len(df):,} lignes supprimées")

rows_before = len(df)

# Inf
inf_before = sum(df[col].isin([np.inf, -np.inf]).sum() 
                 for col in df.select_dtypes(include=[np.number]).columns)

if inf_before > 0:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"   Inf : {inf_before:,} détectés → {rows_before - len(df):,} lignes supprimées")

print(f" Lignes restantes : {len(df):,}")

# 4️⃣ COLONNE CIBLE

print("\n Préparation de la colonne cible...")

target_col = 'Cat'

if 'Sub_Cat' in df.columns:
    df = df.drop(columns=['Sub_Cat'])
if 'Label' in df.columns:
    df = df.drop(columns=['Label'])

print(f" Colonne cible : '{target_col}'")
print(f"   Classes : {df[target_col].nunique()}")

# Encodage
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

print(f"\n   Distribution des classes :")
for i, class_name in enumerate(le.classes_[:10]):  # Top 10
    count = (df[target_col] == i).sum()
    pct = count / len(df) * 100
    print(f"      {i:2d} → {class_name:25s} : {count:7,} ({pct:5.2f}%)")

if len(le.classes_) > 10:
    print(f"      ... et {len(le.classes_) - 10} autres classes")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# 5️⃣ SÉPARATION X / y

print("\n5️⃣ Séparation features / target...")

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✅ X : {X.shape[0]:,} lignes × {X.shape[1]} features")
print(f"✅ y : {y.shape[0]:,} échantillons")

# Encodage des colonnes catégorielles restantes
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"   Encodage de {len(categorical_cols)} colonnes catégorielles...")
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# 6️⃣ DIVISION TRAIN/TEST
print("\n Division Train/Test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train : {len(X_train):,} échantillons")
print(f"✅ Test  : {len(X_test):,} échantillons")

# ============================================================================
# 7️⃣ NORMALISATION
# ============================================================================
print("\n Normalisation (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Données normalisées (mean=0, std=1)")

# ============================================================================
# 8️⃣ SAUVEGARDE
# ============================================================================
print("\n Sauvegarde des données préparées...")

output_dir = Path('processed_data')
output_dir.mkdir(exist_ok=True)

np.save(output_dir / 'X_train.npy', X_train_scaled)
np.save(output_dir / 'X_test.npy', X_test_scaled)
np.save(output_dir / 'y_train.npy', y_train.values)
np.save(output_dir / 'y_test.npy', y_test.values)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(X.columns.tolist()))

print(f"✅ Fichiers sauvegardés dans {output_dir}/")

# ============================================================================
# ✅ RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print(" PREPROCESSING TERMINÉ")
print("=" * 80)

print(f"\n Résumé :")
print(f"   • Dataset original   : {original_rows:,} lignes × {original_cols} colonnes")
print(f"   • Après nettoyage    : {len(X):,} lignes × {X.shape[1]} features")
print(f"   • Réduction          : {(1 - len(X)/original_rows)*100:.1f}% lignes, {(1 - X.shape[1]/original_cols)*100:.1f}% colonnes")
print(f"   • Train              : {len(X_train):,} échantillons")
print(f"   • Test               : {len(X_test):,} échantillons")
print(f"   • Classes            : {len(le.classes_)}")