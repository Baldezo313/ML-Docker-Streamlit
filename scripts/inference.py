#%% Importation des bibliothèques
import pandas as pd 
import numpy as np 
import joblib 
import os 



# %% Chargement du modèle

model_path = "../models/random_forest_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier {model_path} n'exist pas. Assurez-vous d'avoir entrainé et sauvegardé le modèle")

model = joblib.load(model_path)
print("Modèle chargé avec succès")

# %% Inference - Single Prediction 
def single_prediction(input_features):
    """ 
    Réaliser une prediction unique avec le modèle. 
    
    Arguments:
        - input_features: list ou np.array des caractéristiques (doit correspondre à l'ordre du modèle)
        
    Retour:
        - Class Prédite
    """
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    return prediction

# Exemple de prédiction unique
sample_input = [13.0, 2.0, 2.3, 15.0, 100.0, 2.8, 3.0, 0.3, 1.7, 6.0, 1.0, 3.0, 1000.0]  # Exemple de caractéristiques
predicted_class = single_prediction(sample_input)
print(f"Prédiction pour l'exemple donné : Classe {predicted_class}")


# %% Inference - Batch Prediction

def batch_prediction(input_file, output_file):
    """
    Réalise une prédiction par lot en utilisant un fichier CSV comme entrée.

    Arguments :
    - input_file : str, chemin du fichier CSV contenant les données d'entrée
    - output_file : str, chemin où sauvegarder les résultats

    Retour :
    - Aucun (les résultats sont sauvegardés dans output_file)
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Le fichier d'entrée {input_file} n'existe pas.")
    
    # Chargement des données
    input_data = pd.read_csv(input_file)
    print("Données d'entrée chargées avec succès.")
    
    # Prédictions 
    predictions = model.predict(input_data)
    
    # Ajout des prédictions aux données
    input_data["prediction"] = predictions
    
    # Sauvegarde des résultats
    input_data.to_csv(output_file, index=False)
    print(f"Les résultats ont été sauvegardés dans {output_file}.")
    

# Exemple de prédiction par lot
batch_input_path = "../data/batch_input.csv"  # Chemin vers un fichier CSV avec des caractéristiques
batch_output_path = "../data/batch_predictions.csv"  # Chemin pour sauvegarder les résultats

# Création d'un exemple de fichier batch_input.csv pour démonstration
example_batch_data = pd.DataFrame([
    [13.0, 2.0, 2.3, 15.0, 100.0, 2.8, 3.0, 0.3, 1.7, 6.0, 1.0, 3.0, 1000.0],
    [12.5, 1.5, 2.1, 14.0, 90.0, 2.6, 2.9, 0.25, 1.6, 5.8, 1.1, 2.8, 950.0]
], columns=[
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", 
    "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
])

example_batch_data.to_csv(batch_input_path, index=False)

# Exécution de la prédiction par lot
batch_prediction(batch_input_path, batch_output_path)
# %%
