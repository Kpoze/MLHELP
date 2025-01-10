from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Workspace, Datasets, MLModel
import pandas as pd
import json
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import chardet
from django.urls import reverse
import numpy as np
from pycaret.datasets import get_data
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment
from pycaret.clustering import ClusteringExperiment
from pycaret.time_series import TSForecastingExperiment
from pycaret.anomaly import AnomalyExperiment
from django.contrib.sessions.models import Session
import shap
from django.conf import settings
import os
import logging
# Fonction pour configurer l'expérience en fonction du type de modèle
def setup_experiment(request, df, model_choosed, is_target, data_target=None):
    if is_target == 'yes':  # Si le modèle a une colonne cible
        if model_choosed == "Classification":
            exp = ClassificationExperiment()
        elif model_choosed == "Regression":
            exp = RegressionExperiment()
        elif model_choosed == "Times_Series":
            exp = TSForecastingExperiment()
        else:
            raise ValueError("Type de modèle invalide.")
        # Validation de la colonne cible
        if data_target not in df.columns:
            raise ValueError("La colonne cible sélectionnée n'est pas valide.")
        exp.setup(data=df, target=data_target, system_log=False, memory=False, verbose=False, session_id=42)
    
    elif is_target == 'no':  # Si le modèle n'a pas de colonne cible
        if model_choosed == "Clustering":
            exp = ClusteringExperiment()
        elif model_choosed == "Anomalie":
            exp = AnomalyExperiment()
        else:
            raise ValueError("Type de modèle invalide.")
        exp.setup(data=df, system_log=False, memory=False, verbose=False, session_id=42)

    request.session['is_target']       = is_target
    request.session['model_choosed']   = model_choosed
    request.session['data_target']     = data_target if is_target == 'yes' else None
    return exp

# Fonction pour entraîner et sélectionner le meilleur modèle
def train_and_select_best_model(exp):

    # compare models
    top3 = exp.compare_models(n_select = 3) 

    # tune models
    tuned_top3 = [exp.tune_model(i) for i in top3]

    # ensemble models
    #bagged_top5 = [exp.ensemble_model(i) for i in tuned_top5]

    # blend models
    #blender = exp.blend_models(estimator_list = top5) 

    # stack models
    #stacker = exp.stack_models(estimator_list = top5) 

    # automl 
    best = exp.automl()

    """best_tuned = exp.tuned_model(best)
    print(best_tuned)"""

    return best

@login_required
def get_model(request, dataset_id, workspace_id):
    # Chargement des données et initialisation
    try:
        workspace = get_object_or_404(Workspace, id=workspace_id)
        dataset   = get_object_or_404(Datasets, id=dataset_id)
        
        # Détecter l'encodage et charger le fichier
        with open(dataset.file.path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            encoding = result['encoding']
        df = pd.read_csv(dataset.file.path, encoding=encoding)

    except Exception as e:
        return JsonResponse({"error": f"Erreur de chargement des données : {str(e)}"}, status=500)

    # Initialisation des variables
    best_model = None
    model_name = None
    model_param = None

    if request.method == "POST":
        try:
            # Récupérer les paramètres utilisateur
            is_target        = request.POST.get("is_target")
            model_choosed    = request.POST.get("model_choosed")
            data_target      = request.POST.get("data_target") if is_target == 'yes' else None
            model_name_input = request.POST.get("model_name") or "Robert"

            # Configurer l'expérience
            exp = setup_experiment(request, df, model_choosed, is_target, data_target)

            # Entraîner et sélectionner le meilleur modèle
            best_model = train_and_select_best_model(exp)

            # Obtenir le nom et les paramètres du modèle
            model_name  = type(best_model).__name__
            model_param = best_model.get_params()
            print(exp.get_config())
            print(model_param)
            print(type(exp.get_config()))
            print(type(model_param))
            json_model_param = json.dumps(model_param)
            print(json_model_param)

            # Sauvegarder le modèle et l'expérience
            exp_filepath    = save_experiment_to_media(exp, model_name_input)
            model_file_path = save_model_to_media(exp, best_model, model_name_input)
            ml_model        = save_model_and_experiment(request,
                workspace, dataset, 
                model_name_input, json_model_param, model_file_path, 
                exp, exp_filepath
            )

            print(f"Modèle sauvegardé avec succès (ID : {ml_model.id})")
            return redirect(reverse('mlapp:detail_model', kwargs={'workspace_id': workspace.id, 'dataset_id': dataset.id}))
        

        except Exception as e:
            return JsonResponse({"error": f"Erreur lors du traitement : {str(e)}"}, status=500)

    # Retourner les résultats à la vue
    return render(request, "mlapp/get_model.html", {
        "workspace" : workspace,
        "dataset"   : dataset,
        "columns": list(df.columns),
        "best_model": best_model,
        "model_name": model_name,
        "model_param": model_param,
    })



# Fonction pour supprimer temporairement le logger
def disable_logger(exp):
    exp.logger = logging.getLogger("pycaret")
    exp.logger.setLevel(logging.CRITICAL)

def save_experiment_to_media(exp, model_name):
    media_path = os.path.join(settings.MEDIA_ROOT, 'config')
    os.makedirs(media_path, exist_ok=True)  # Crée le dossier s'il n'existe pas
    experiment_name = f"{model_name}_experiment"
    experiment_path = os.path.join(media_path, experiment_name)
    saved_experiment_path = f"{experiment_path}.pkl"
    disable_logger(exp)
    exp.save_experiment(saved_experiment_path)
    print(experiment_path)
    print(saved_experiment_path)
    return saved_experiment_path

def save_model_to_media(exp,model, model_name):
    media_path = os.path.join(settings.MEDIA_ROOT, 'models')
    os.makedirs(media_path, exist_ok=True)  # Crée le dossier s'il n'existe pas
    model_path = os.path.join(media_path, f"{model_name}")
    saved_model_path = f"{model_path}.pkl"
    exp.save_model(model, model_path)
    print(model_path)
    print(saved_model_path)
    return saved_model_path

def save_model_and_experiment(request,workspace, dataset, model_name,model_param, model_file_path,exp,exp_filepath):
    # Enregistrer dans la base de données
    ml_model = MLModel.objects.create(
        workspace   = workspace,
        dataset     = dataset,
        model_name  = model_name,
        model_file  = model_file_path,
        model_param = model_param,
        exp_file    = exp_filepath,   
    )
    return ml_model


@login_required
def detail_model(request, workspace_id,dataset_id):
    # Get the Workspace instance based on the workspace_id from the URL
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset   = get_object_or_404(Datasets, id=dataset_id)
    
    # Filter datasets based on the retrieved workspace
    models = MLModel.objects.filter(workspace=workspace, dataset = dataset)
    
    # Render the template and pass the datasets to the context
    return render(request, 'mlapp/detail_model.html', {'dataset': dataset,'workspace': workspace, 'models' : models})

@login_required
def delete_model(request, workspace_id, dataset_id, model_id):
    # Get the Workspace instance based on the workspace_id from the URL
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)
    model = get_object_or_404(MLModel, id=model_id)

    print(model.id)
    print(os.path.exists(model.model_file.path))
    print(model.model_file)
    print(model.model_file.path)
    print(model.exp_file.path)
    print(os.path.exists(model.exp_file.path))

    if request.method == 'POST':
        # Supprimer les fichiers du modèle et de l'expérience
        model.delete()
        """if model.model_file and model.model_file.path and os.path.exists(model.model_file.path):
            #os.remove(model.model_file.path)  # Supprime le fichier modèle
            #os.remove(model.exp_file.path)  # Supprime le fichier expérience
            print("ok")
        if model.exp_file and model.exp_file.path and os.path.exists(model.exp_file.path):
            print("ok")"""
        
        return redirect(reverse('mlapp:detail_model', kwargs={'workspace_id': workspace.id, 'dataset_id': dataset.id}))
    return render(request, 'mlapp/delete_model.html', {'dataset': dataset,'workspace': workspace, 'model' : model})



def evaluate(exp, best_model):
    tuned_model = exp.tune_model(best_model)
    evaluation_results = exp.evaluate_model(tuned_model)
    exp.interpret_model(tuned_model)
    return tuned_model, evaluation_results