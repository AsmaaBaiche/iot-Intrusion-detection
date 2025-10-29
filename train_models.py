import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle

print("=" * 70)
print(" ENTRAÎNEMENT DES MODÈLES DE DÉTECTION D'INTRUSION IoT")
print("=" * 70)

# 1. CHARGEMENT DES DONNÉES
print("\n Chargement des données...")
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Charge le LabelEncoder pour les noms de classes
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

class_names = le.classes_

print(f" Train : {X_train.shape[0]:,} échantillons × {X_train.shape[1]} features")
print(f" Test  : {X_test.shape[0]:,} échantillons")
print(f" Classes : {len(class_names)} ({', '.join(class_names)})")

# 2. ENTRAÎNEMENT RANDOM FOREST
print("\n" + "=" * 70)
print(" RANDOM FOREST CLASSIFIER")
print("=" * 70)

print("\n  Configuration du modèle...")
print("   - n_estimators: 100 arbres")
print("   - max_depth: 20")
print("   - class_weight: balanced (gère le déséquilibre)")
print("   - n_jobs: -1 (utilise tous les CPU)")

start_time = time.time()

# Modèle avec gestion du déséquilibre des classes
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\n Entraînement en cours...")
rf_model.fit(X_train, y_train)

train_time = time.time() - start_time
print(f"\n  Temps d'entraînement : {train_time:.2f}s ({train_time/60:.2f} min)")

# 3. PRÉDICTIONS
print("\n Prédictions sur le test set...")
y_pred_rf = rf_model.predict(X_test)

# 4. ÉVALUATION GLOBALE
print("\n" + "=" * 70)
print(" RÉSULTATS GLOBAUX")
print("=" * 70)

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)

print(f"\n✅ Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ F1-Score  : {f1:.4f}")

# 5. RAPPORT DÉTAILLÉ PAR CLASSE
print("\n" + "=" * 70)
print(" RAPPORT DE CLASSIFICATION PAR CLASSE")
print("=" * 70)
print(classification_report(y_test, y_pred_rf, target_names=class_names, zero_division=0))

# 6. MATRICE DE CONFUSION
print("\n Génération de la matrice de confusion...")
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Nombre de prédictions'})
plt.title('Matrice de Confusion - Random Forest\nDétection d\'Intrusion IoT', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Vraie Classe', fontsize=12)
plt.xlabel('Classe Prédite', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print(" Sauvegardée : confusion_matrix_rf.png")

# 7. MATRICE DE CONFUSION NORMALISÉE (%)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', 
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Pourcentage (%)'},
            vmin=0, vmax=100)
plt.title('Matrice de Confusion Normalisée (%)\nRandom Forest', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Vraie Classe', fontsize=12)
plt.xlabel('Classe Prédite', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(" Sauvegardée : confusion_matrix_normalized.png")

# 8. IMPORTANCE DES FEATURES
print("\n Analyse de l'importance des features...")

# Charge les noms de features
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

feature_importance = rf_model.feature_importances_
indices = np.argsort(feature_importance)[-15:]  # Top 15

plt.figure(figsize=(10, 8))
plt.barh(range(15), feature_importance[indices], color='steelblue')
plt.yticks(range(15), [feature_names[i] for i in indices])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Features les Plus Importantes\nRandom Forest', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardée : feature_importance.png")

# Affiche le top 10
print("\n Top 10 Features :")
top10_indices = np.argsort(feature_importance)[-10:][::-1]
for i, idx in enumerate(top10_indices, 1):
    print(f"   {i:2d}. {feature_names[idx]:30s} : {feature_importance[idx]:.4f}")

# 9. ANALYSE PAR CLASSE
print("\n" + "=" * 70)
print(" PERFORMANCE PAR CLASSE")
print("=" * 70)

for i, class_name in enumerate(class_names):
    mask = y_test == i
    correct = np.sum((y_test[mask] == y_pred_rf[mask]))
    total = np.sum(mask)
    accuracy_class = correct / total if total > 0 else 0
    
    print(f"\n📌 {class_name:20s}")
    print(f"   Total test : {total:6,} échantillons")
    print(f"   Correct    : {correct:6,} ({accuracy_class*100:5.2f}%)")
    print(f"   Erreurs    : {total-correct:6,} ({(1-accuracy_class)*100:5.2f}%)")

# 10. SAUVEGARDE DU MODÈLE
print("\n Sauvegarde du modèle...")
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Modèle sauvegardé : rf_model.pkl")

# 11. RÉSUMÉ FINAL
print("\n" + "=" * 70)
print("🎉 ENTRAÎNEMENT TERMINÉ !")
print("=" * 70)

print(f"""
📊 RÉSUMÉ DES PERFORMANCES :
   
   ✅ Accuracy globale    : {accuracy*100:.2f}%
   ✅ F1-Score moyen      : {f1:.4f}
   ✅ Temps d'entraînement: {train_time:.2f}s
   ✅ Nombre de features  : {X_train.shape[1]}
   ✅ Nombre de classes   : {len(class_names)}

📁 FICHIERS GÉNÉRÉS :
   - confusion_matrix_rf.png
   - confusion_matrix_normalized.png
   - feature_importance.png
   - rf_model.pkl

💡 PROCHAINES ÉTAPES :
   1. Analyse les graphiques générés
   2. Note l'accuracy pour ton CV (probablement 85-95%)
   3. Crée un README.md avec les résultats
   4. Push sur GitHub
""")

print("=" * 70)
print("🚀 Projet terminé avec succès !")
print("=" * 70)