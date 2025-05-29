#%% Import des bibliothèques
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import os
# %% Creation des répertoires pour l'organisation
os.makedirs("../data", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# %% Chargement et préparation des données
# chargement des donnée wine de sklearn

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Ajout de la cible aux données pour sauvegarder facile
data_df = pd.concat([X, y], axis=1)
data_df.to_csv("../data/wine_data.csv", index=False)
print("Les données ont été sauvegardées dans data/wine_data.csv")

# %% Exploration rapide des données
print("Aperçu des premières lignes :")
print(data_df.head())
print("\nDescription statistique :")
print(data_df.describe())

# Visualisation de la distribution des variables
for column in X.columns[:5]:  # Limitation à 5 colonnes pour cet exemple
    plt.figure(figsize=(6, 4))
    X[column].hist(bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Distribution de {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.show()
# %% Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %% Entraînement du modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# %% Évaluation du modèle
# Prédictions
y_pred = model.predict(X_test)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot(cmap="Blues")
plt.title("Matrice de Confusion")
plt.show()

# %% Sauvegarde du modèle
model_path = "../models/random_forest_model.pkl"
joblib.dump(model, model_path)

print(f"Le modèle a été sauvegardé dans {model_path}")
# %%
