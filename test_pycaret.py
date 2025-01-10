from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.time_series import TSForecastingExperiment
from pycaret.anomaly import AnomalyExperiment

# Charger les données
data = get_data('iris')

# Vérification si une cible est présente
is_target = input("Y a-t-il une cible? (yes ou no): ").strip().lower()
if is_target == 'yes':
    model_choosed = input("Choisissez un type de modèle (Classification ou Regression ou Times_Series): ").strip()
    
    # Choix du type d'expérience
    if model_choosed == "Classification":
        exp = ClassificationExperiment()
    elif model_choosed == "Regression":
        exp = RegressionExperiment()
    elif model_choosed == "Times_Series":
        exp = TSForecastingExperiment()
    else:
        raise ValueError("Type de modèle invalide. Choisissez entre 'Classification', 'Regression', ou 'Times_Series'")
    
    # Validation de la colonne cible
    print("Colonnes disponibles:", list(data.columns))
    data_target = input('Choisissez une colonne comme cible parmi les colonnes ci-dessus: ').strip()
    if data_target not in data.columns:
        raise ValueError(f"{data_target} n'est pas une colonne valide.")
    
    # Configurer l'expérience
    exp.setup(data=data, target=data_target, system_log=False, memory=False, verbose=False)
    
else:
    model_choosed = input("Choisissez un type de modèle (Clustering ou Anomalie): ").strip()
    
    # Choix du type d'expérience
    if model_choosed == "Clustering":
        exp = ClusteringExperiment()
    elif model_choosed == "Anomalie":
        exp = AnomalyExperiment()
    else:
        raise ValueError("Type de modèle invalide. Choisissez entre 'Clustering' ou 'Anomalie'")
    
    # Configurer l'expérience
    exp.setup(data=data, system_log=False, memory=False, verbose=False)

# Comparaison des modèles
try:
    best_model = exp.compare_models(verbose=False)
    print(f"Meilleur modèle trouvé: {type(best_model).__name__}")
except Exception as e:
    print(f"Erreur lors de la comparaison des modèles: {e}")

# Affiner les modèles
try:
    tuned_model = exp.tune_model(best_model)
    print("Modèle ajusté:", tuned_model)
except Exception as e:
    print(f"Erreur lors de l'ajustement des hyperparamètres: {e}")

# Finalisation du modèle
try:
    final_model = exp.finalize_model(tuned_model)
    print("Modèle finalisé:", final_model)
except Exception as e:
    print(f"Erreur lors de la finalisation du modèle: {e}")

# Évaluation et prédiction
if is_target == 'yes' and model_choosed != "Times_Series":
    try:
        exp.evaluate_model(final_model)
        predictions = exp.predict_model(final_model, data=data)
        print("Prédictions:", predictions.head())
    except Exception as e:
        print(f"Erreur lors de l'évaluation ou des prédictions: {e}")
else:
    print("Évaluation ou prédictions non applicables pour le type de modèle choisi.")
"""
# Sauvegarder
save_model(tuned_model, 'best_model')

# Charger
loaded_model = load_model('best_model')"""