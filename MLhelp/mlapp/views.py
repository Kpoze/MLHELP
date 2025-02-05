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
import pickle 
from .forms import *
from django.forms import formset_factory
from django.http import JsonResponse
from django.template.loader import render_to_string
###################SKLEARN##################
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import base64
###############FUNCTION#####################

def get_encoding(df):
    with open(df, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Analyser les premiers 10k octets
        encoding = result['encoding']

    return encoding


# Fonction pour supprimer temporairement le logger
def disable_logger(exp):
    exp.logger = logging.getLogger("pycaret")
    exp.logger.setLevel(logging.CRITICAL)


def save_model_and_experiment(request,workspace, dataset,model_type_choosed,model_name,model_param, model_file_path,exp,exp_filepath):
    # Enregistrer dans la base de données
    ml_model = MLModel.objects.create(
        workspace   = workspace,
        dataset     = dataset,
        model_type  = model_type_choosed,
        model_name  = model_name,
        model_file  = model_file_path,
        model_param = model_param,
        exp_file    = exp_filepath,   
    )
    return ml_model

def get_data_post(request):
    
    data = {
        'target'              : request.POST.get("target"),
        'model_type_choosed'  : request.POST.get("model_type_choosed"),
        'data_target'         : request.POST.get("data_target") if data.target == 'yes' else None,
        'model_name_input'    : request.POST.get("model_name") or "Robert",
    }
    return data 
###############FUNCTION PRE-MODEL#####################

# Fonction pour configurer l'expérience en fonction du type de modèle
def setup_experiment(request, df, model_type_choosed, target, data_target=None):
    if target == 'yes':  # Si le modèle a une colonne cible
        if model_type_choosed == "Classification":
            exp = ClassificationExperiment()
        elif model_type_choosed == "Regression":
            exp = RegressionExperiment()
        elif model_type_choosed == "Times_Series":
            exp = TSForecastingExperiment()
        else:
            raise ValueError("Type de modèle invalide.")
        # Validation de la colonne cible
        if data_target not in df.columns:
            raise ValueError("La colonne cible sélectionnée n'est pas valide.")
        exp.setup(data=df, target=data_target, system_log=False, memory=False, verbose=False, session_id=42)
    
    elif target == 'no':  # Si le modèle n'a pas de colonne cible
        if model_type_choosed == "Clustering":
            exp = ClusteringExperiment()
        elif model_type_choosed == "Anomalie":
            exp = AnomalyExperiment()
        else:
            raise ValueError("Type de modèle invalide.")
        exp.setup(data=df, system_log=False, memory=False, verbose=False, session_id=42)

    request.session['target']         = target
    request.session['model_type_choosed']   = model_type_choosed
    request.session['data_target']       = data_target if target == 'yes' else None
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
    #saved_model_path = f"{model_path}.pkl"
    exp.save_model(model, model_path)
    print(model_path)
    #print(saved_model_path)
    return model_path


def load_model_type_pyc_librairies(model_type_choosed):
    if model_type_choosed == "Classification":
        from pycaret.classification import load_model, load_experiment
    elif model_type_choosed == "Regression":
        from pycaret.regression import load_model, load_experiment
    elif model_type_choosed == "Times_Series":
        from pycaret.time_series import load_model, load_experiment
    elif model_type_choosed == "Clustering":
        from pycaret.clustering import load_model, load_experiment
    elif model_type_choosed == "Anomalie":
        from pycaret.anomaly import load_model, load_experiment
    else:
        raise ValueError("Type de modèle invalide.")
    
    return load_model, load_experiment


###############FUNCTION CUSTOM-MODEL#####################
def split_data(df,target,test_size,seed):
    from sklearn.model_selection import train_test_split
    """ 
    Sépare les données en train/test et retourne un dictionnaire sérialisable en JSON.
    """
    if target is not None:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        data = {
            'X': X.to_json(orient='records'),
            'y': y.to_json(),
            'X_train': X_train.to_json(orient='records'),
            'X_test': X_test.to_json(orient='records'),
            'y_train': y_train.to_json(),
            'y_test': y_test.to_json(),
        }
    else:
        X = df
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=seed)

        data = {
            'X': X.to_json(orient='records'),
            'X_train': X_train.to_json(orient='records'),
            'X_test': X_test.to_json(orient='records'),
        }
    
    return data

def classify_columns(df):
    column_types = {'categorical': [], 'numeric': [], 'text': [], 'datetime': []}

    for column in df.columns:
        # Détection des colonnes datetime
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            column_types['datetime'].append(column)
        # Détection des colonnes numériques
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_types['numeric'].append(column)
        # Détection des colonnes textuelles et catégoriques
        elif pd.api.types.is_object_dtype(df[column]):
            unique_ratio = df[column].nunique() / len(df[column])
            if unique_ratio > 0.5:  # Texte brut
                column_types['text'].append(column)
            else:  # Catégorie
                column_types['categorical'].append(column)

    return column_types


def create_preprocessing_pipeline(cleaned_data, column_types):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cleaned_data['imputer']) if cleaned_data['imputer'] != 'passthrough' else 'passthrough' if cleaned_data['imputer'] == 'No imputer' else 'passthrought'),
        ('scaler', StandardScaler() if cleaned_data['scaler'] == 'StandardScaler' else
                   MinMaxScaler() if cleaned_data['scaler'] == 'MinMaxScaler' else
                   RobustScaler() if cleaned_data['scaler'] == 'RobustScaler' else 
                   'passthrough' if cleaned_data['scaler'] == 'No scaler' else 'passthrough')
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=cleaned_data['imputer']) if cleaned_data['imputer'] != 'passthrough' else 'passthrough'),
        ('encoder', OneHotEncoder() if cleaned_data['encoder'] == 'OneHotEncoder' else
                    LabelEncoder() if cleaned_data['encoder'] == 'LabelEncoder' else 'passthrough')
    ])

    text_transformer = Pipeline(steps=[
        ('vectorizer', CountVectorizer() if cleaned_data['vectorizer'] == 'CountVectorizer' else
                       TfidfVectorizer() if cleaned_data['vectorizer'] == 'TfidfVectorizer' else 
                       'passthrough' if cleaned_data == 'No vectorizer' else 'passthrough')
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, column_types['numeric']),
            ('cat', categorical_transformer, column_types['categorical']),
            ('text', text_transformer, column_types['text'])
        ])

    return preprocessor


def get_model_form(model_type, post_data=None):
    """Retourne le bon formulaire de sélection de modèle en fonction du type."""
    if model_type == 'classification':
        return ClassificationModelForm(post_data)
    elif model_type == 'regression':
        return RegressionModelForm(post_data)
    elif model_type == 'clustering':
        return ClusteringModelForm(post_data)
    return None  # Si aucun type n’est sélectionné

def get_hyperparameter_form(model_type, model_name, post_data=None):
    """Retourne le formulaire d'hyperparamètres selon le modèle choisi."""
    if model_type == 'classification':
        if model_name == 'random_forest':
            return RandomForestHyperparametersForm(post_data)
        elif model_name == 'svm':
            return SVMHyperparametersForm(post_data)
        
        elif model_name == 'logistic_regression':
            return LogisticRegressionHyperparametersForm(post_data)
        
    elif model_type == 'regression':
        if model_name == 'linear_regression':
            return LinearRegressionHyperparametersForm(post_data)
        elif model_name == 'ridge_regression':
            return RidgeRegressionHyperparametersForm(post_data)
        elif model_name == 'decision_tree':
            return DecisionTreeHyperparametersForm(post_data)

    elif model_type == 'clustering':
        if model_name == 'kmeans':
            return KMeansHyperparametersForm(post_data)
        elif model_name == 'dbscan':
            return DBSCANHyperparametersForm(post_data)
        elif model_name == 'hierarchical':
            return HierarchicalClusteringHyperparametersForm(post_data)

    return None  # Si aucun modèle n'est sélectionné
######ROBERT#########################
@login_required
def choose_model_way(request, workspace_id, dataset_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)
    return render(request, 'mlapp/choose_model_way.html', {'workspace':workspace,'dataset':dataset})


 #############################PRE-MODEL##################
@login_required
def get_pre_model(request, dataset_id, workspace_id):
    # Chargement des données et initialisation
    try:
        workspace = get_object_or_404(Workspace, id=workspace_id)
        dataset = get_object_or_404(Datasets, id=dataset_id)

        encoding = get_encoding(dataset.file.path)
        df = pd.read_csv(dataset.file.path, encoding=encoding)

    except Exception as e:
        return JsonResponse({"error": f"Erreur de chargement des données : {str(e)}"}, status=500)
    # Initialisation des variables
    model_data = {
        'best_model'  : None,
        'model_name'  : None,
        'model_param' : None,
        'model_name'  :None,
    }
    if request.method == "POST":
        try:
            # Récupérer les paramètres utilisateur
            data = get_data_post(request)
            model_type_choosed = data.model_type_choosed  # Récupère la valeur du formulaire
            request.session['model_type_choosed'] = model_type_choosed  # Stocke dans la session
            print(f"Type de modèle : {data.model_type_choosed}")
            print(f"Colonne cible : {data.data_target}")
            print(f"Nom du modèle : {data.model_name_input}")
            exp = setup_experiment(request, df, data.model_type_choosed, data.target, data.data_target)

                # Entraîner et sélectionner le meilleur modèle
            model_data.best_model = train_and_select_best_model(exp)

                # Obtenir le nom et les paramètres du modèle
            model_data.model_name       = type(model_data.best_model).__name__
            model_data.model_param      = model_data.best_model.get_params()
            json_model_param = json.dumps(model_data.model_param )

                # Sauvegarder le modèle et l'expérience
            exp_filepath = save_experiment_to_media(exp, data.model_name_input)
            model_file_path = save_model_to_media(exp, model_data.best_model, data.model_name_input)
            ml_model = save_model_and_experiment(
                    request, workspace, dataset, data.model_type_choosed, data.model_name_input,
                    json_model_param, model_file_path, exp, exp_filepath
                )

            print(f"Modèle PyCaret sauvegardé avec succès (ID : {ml_model.id})")
            return redirect(reverse('mlapp:detail_model', kwargs={'workspace_id': workspace.id, 'dataset_id': dataset.id}))
        except Exception as e:
            return JsonResponse({"error": f"Erreur lors du traitement : {str(e)}"}, status=500)

    # Retourner les résultats à la vue
    return render(request, "mlapp/get_model.html", {
        "workspace"         : workspace,
        "dataset"           : dataset,
        "columns"           : list(df.columns),
        "best_model"        : model_data['best_model'],
        "model_type_choosed": request.session.get('model_type_choosed', None),
        "model_param"       : model_data['model_param'],
    })

###################CUSTOM MODEL######################

def check_target_step(request, dataset_id, workspace_id):
    try:
        workspace = get_object_or_404(Workspace, id=workspace_id)
        dataset = get_object_or_404(Datasets, id=dataset_id)
        encoding = get_encoding(dataset.file.path)
        df = pd.read_csv(dataset.file.path, encoding=encoding)
    except Exception as e:
        return JsonResponse({"error": f"Erreur de chargement des données : {str(e)}"}, status=500)

    if request.method == "POST":
        # Récupère la valeur de 'is_target' et 'data_target' depuis la requête POST
        is_target = request.POST.get('is_target')
        data_target = request.POST.get('data_target') if is_target == 'yes' else None

        # Crée une instance du formulaire avec les données POST
        form = ModelTypeForm(is_target=data_target, data=request.POST)

        if form.is_valid():
            # Récupère le type de modèle choisi
            model_type = form.cleaned_data['model_type']

            # Sauvegarde la cible et le type de modèle dans la session
            request.session['data_target'] = data_target
            request.session['model_type'] = model_type

            # Redirection vers l'étape suivante de prétraitement
            return redirect('mlapp:get_custom_model_step_1', workspace_id=workspace.id, dataset_id=dataset.id)
        else:
            # Si le formulaire n'est pas valide, retourne les erreurs
            return JsonResponse({"error": form.errors}, status=400)

    else:
        # Requête GET, afficher la page de sélection de la cible et du type de modèle
        form = ModelTypeForm(is_target=None)  # Par défaut, pas de cible
        return render(request, 'mlapp/get_custom_model_step_0.html', {
            'workspace': workspace,
            'dataset': dataset,
            'columns': list(df.columns),  # Passer les colonnes du dataset à la vue
            'form': form,  # Passer le formulaire à la vue
        })

@login_required
def preprocessing_step(request, workspace_id, dataset_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset  = get_object_or_404(Datasets, id=dataset_id)
    encoding = get_encoding(dataset.file.path)
    df       = pd.read_csv(dataset.file.path, encoding=encoding)
    column_types = classify_columns(df)  # Récupère les colonnes classées par type ('numeric', 'categorical', etc.)
    target = request.session.get('data_target')
    print(target)

    if request.method == 'POST':
        # Vérifie si la requête contient des données JSON (drag-and-drop)
        if request.headers.get('Content-Type') == 'application/json':
            try:
                column_data = json.loads(request.body)
                column_name = column_data.get('column_name')
                new_type    = column_data.get('new_type')

                # Validation des données envoyées
                if column_name not in df.columns:
                    return JsonResponse({'success': False, 'error': 'Column not found in dataset.'}, status=400)

                if new_type not in column_types:
                    return JsonResponse({'success': False, 'error': 'Invalid column type.'}, status=400)

                # Déplacement de la colonne entre les types
                for key in column_types.keys():
                    if column_name in column_types[key]:
                        column_types[key].remove(column_name)
                        break
                column_types[new_type].append(column_name)
                print()

                # Retourner la liste mise à jour des colonnes
                return JsonResponse({'success': True, 'updated_column_types': column_types})

            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)

        # Sinon, traite la soumission du formulaire
        else:
            form = PreprocessingForm(request.POST)
            if form.is_valid():
                data   = split_data(df,target,0.2,42)
                preprocessor            = create_preprocessing_pipeline(form.cleaned_data,column_types)
                preprocessor_serialized = base64.b64encode(pickle.dumps(preprocessor)).decode('utf-8')
                request.session['data']         = data
                request.session['preprocessor'] = preprocessor_serialized
                # Sérialisation avec pickle et encodage Base64
                """print(data)
                print(preprocessor)
                print(pickle.loads(base64.b64decode(preprocessor_serialized)))
                print(form.cleaned_data)  # Logique pour traiter les données du formulaire"""
                return redirect('mlapp:get_custom_model_step_2', workspace_id, dataset_id)
            

    return render(request, 'mlapp/get_custom_model_step_1.html', {
        'workspace': workspace,
        'dataset': dataset,
        'column_types': column_types,  # Données des colonnes classées
        'form': PreprocessingForm(),  # Un formulaire global
    })


@login_required
def model_step(request, workspace_id, dataset_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)
    encoding = get_encoding(dataset.file.path)

    # Récupération des données
    data = request.session.get('data')
    preprocessor = pickle.loads(base64.b64decode(request.session.get('preprocessor')))
    target = request.session.get('data_target')
    selected_model_type = None
    selected_model = None
    selected_hyperparameter = None

    # Crée le formulaire pour le type de modèle en passant 'target' pour déterminer les choix
    model_type_form = ModelTypeForm(target=target)

    model_form = None
    hyperparameter_form = None

    # Si la requête est un POST classique, on récupère les données envoyées
    if request.method == 'POST':
        model_type = request.POST.get('model_type')
        model_form = get_model_form(model_type, request.POST) if model_type else None
        if model_type and model_form.is_valid():
            # Si un modèle est sélectionné, récupérer le formulaire des hyperparamètres
            selected_model = model_form.cleaned_data.get('model')
            hyperparameter_form = get_hyperparameter_form(model_type, selected_model, request.POST)

            if model_form.is_valid() and hyperparameter_form.is_valid():
                # Enregistrer les choix dans la session
                request.session['selected_model_type'] = model_type
                request.session['selected_model'] = selected_model
                request.session['selected_hyperparameter'] = hyperparameter_form.cleaned_data
                # Optionnel : redirection vers une autre étape
                # return redirect('mlapp:next_step')

    return render(request, 'mlapp/get_custom_model_step_2.html', {
        'workspace': workspace,
        'dataset': dataset,
        'model_type_form': model_type_form,
        'model_form': model_form,
        'hyperparameter_form': hyperparameter_form,
    })

"""def get_custom_model(request, dataset_id, workspace_id):
    # Chargement des données et initialisation
    try:
        workspace = get_object_or_404(Workspace, id=workspace_id)
        dataset   = get_object_or_404(Datasets, id=dataset_id)

        encoding = get_encoding(dataset.file.path)
        df       = pd.read_csv(dataset.file.path, encoding=encoding)

    except Exception as e:
        return JsonResponse({"error": f"Erreur de chargement des données : {str(e)}"}, status=500)
    # Initialisation des variables
    model_data = {
        'model'       : None,
        'model_param' : None,
    }
    if request.method == "POST":
            # Récupérer les paramètres utilisateur
            data = get_data_post(request)
            print(f"Type de modèle : {data.model_type_choosed}")
            print(f"Colonne cible : {data.data_target}")
            print(f"Nom du modèle : {data.model_name_input}")
            column_types      = classify_columns(df)  # Récupère les types de colonnes
            #df_split_data     = split_data(df,data.data_target,0.2,42)
            Preprocessing_form = PreprocessingForm()
            # Extraire les données JSON envoyées
                    # Créer le formulaire du modèle spécifique en fonction du type de modèle
            if data.data_target == 'Yes':
                if data.model_type_choosed == 'classification':
                    model_form = ClassificationModelForm(request.POST)
                    if model_data.model == 'random_forest':
                        hyperparameters_form = RandomForestHyperparametersForm(request.POST)
                    elif model_data.model == 'svm':
                        hyperparameters_form = SVMHyperparametersForm(request.POST)
                    else:
                        hyperparameters_form = None
                elif data.model_type_choosed == 'regression':
                    model_form = RegressionModelForm(request.POST)
                else:
                    model_form = None
            else : 
                if data.model_type_choosed == 'clustering':
                    model_form = ClusteringModelForm(request.POST)
                    if model_data.model == 'kmeans':
                        hyperparameters_form = KMeansHyperparametersForm(request.POST)
                    else:
                        hyperparameters_form = None
                else:
                    model_form = None
            
                    # Valider les formulaires
            if   model_form.is_valid() and  hyperparameters_form.is_valid():
                    print("Modèle spécifique:", model_form.cleaned_data)
                    print("Hyperparamètres:", hyperparameters_form.cleaned_data)
                
            return render(request, 'mlapp/get_custom_model.html', {
        'workspace'           : workspace,
        'dataset'             : dataset,
        'column_types'        : column_types,
        'Preprocessing_form'  : Preprocessing_form,
        'model_form'          : model_form,  
        'hyperparameters_form': hyperparameters_form, 
    })"""
            
            

# Vue pour gérer les changements de colonnes (drag-and-drop)
def check_classify_columns(request, workspace_id, dataset_id):
    if request.method == 'POST' and request.headers.get('Content-Type') == 'application/json':
        try:
            workspace = get_object_or_404(Workspace, id=workspace_id)
            dataset = get_object_or_404(Datasets, id=dataset_id)
            df = pd.read_csv(dataset.file.path)
            column_types = classify_columns(df)

            # Extraire les données JSON envoyées
            data = json.loads(request.body)
            column_name = data.get('column_name')
            new_type = data.get('new_type')

            if column_name not in df.columns:
                return JsonResponse({'success': False, 'error': 'Column not found in dataset.'}, status=400)

            if new_type not in column_types:
                return JsonResponse({'success': False, 'error': 'Invalid column type.'}, status=400)

            # Déplacer la colonne vers la nouvelle catégorie
            for key in column_types.keys():
                if column_name in column_types[key]:
                    column_types[key].remove(column_name)
                    break
            column_types[new_type].append(column_name)

            return JsonResponse({'success': True, 'updated_column_types': column_types})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    return JsonResponse({'success': False, 'error': 'Invalid request method.'}, status=405)

# Vue pour traiter le formulaire global
def custom_preprocess_form(request):
    if request.method == 'POST':
        PreprocessingForm_form = PreprocessingForm(request.POST)
        if PreprocessingForm_form.is_valid():
            cleaned_data = PreprocessingForm_form.cleaned_data
            print(cleaned_data)  # Logique de traitement des données ici
            return JsonResponse({'success': True, 'message': 'Form data processed successfully.'})
        else:
            return JsonResponse({'success': False, 'error': 'Invalid form data.'}, status=400)
    return JsonResponse({'success': False, 'error': 'Invalid request method.'}, status=405)



@login_required
def check_classify_columns(request, workspace_id, dataset_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)
    encoding = get_encoding(dataset.file.path)
    df = pd.read_csv(dataset.file.path, encoding=encoding)
    column_types = classify_columns(df)  # Récupère les colonnes classées par type ('numeric', 'categorical', etc.)

    if request.method == 'POST':
        # Vérifie si la requête contient des données JSON (drag-and-drop)
        if request.headers.get('Content-Type') == 'application/json':
            try:
                data        = json.loads(request.body)
                column_name = data.get('column_name')
                new_type    = data.get('new_type')

                # Validation des données envoyées
                if column_name not in df.columns:
                    return JsonResponse({'success': False, 'error': 'Column not found in dataset.'}, status=400)

                if new_type not in column_types:
                    return JsonResponse({'success': False, 'error': 'Invalid column type.'}, status=400)

                # Déplacement de la colonne entre les types
                for key in column_types.keys():
                    if column_name in column_types[key]:
                        column_types[key].remove(column_name)
                        break
                column_types[new_type].append(column_name)

                # Retourner la liste mise à jour des colonnes
                return JsonResponse({'success': True, 'updated_column_types': column_types})

            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)

        # Sinon, traite la soumission du formulaire
        else:
            form = PreprocessingForm(request.POST)
            if form.is_valid():
                cleaned_data = form.cleaned_data
                print(cleaned_data)  # Logique pour traiter les données du formulaire
                return JsonResponse({'success': True, 'message': 'Formulaire traité avec succès.'})
            else:
                return JsonResponse({'success': False, 'error': 'Formulaire invalide.'}, status=400)

    # Si méthode GET, renvoie le template avec les données initiales
    return render(request, 'mlapp/custom_model_preprocessing_step.html', {
        'workspace': workspace,
        'dataset': dataset,
        'column_types': column_types,  # Données des colonnes classées
        'form': PreprocessingForm(),  # Un formulaire global
    })
###########################################################################

@login_required
def detail_model(request, workspace_id,dataset_id):
    # Get the Workspace instance based on the workspace_id from the URL
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset   = get_object_or_404(Datasets, id=dataset_id)
    encoding = get_encoding(dataset.file.path)
    df       = pd.read_csv(dataset.file.path, encoding=encoding)
    column_types = classify_columns(df)
    print(column_types)
    
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



@login_required
def model_manip(request, workspace_id, dataset_id, model_id):
    # Chargement des objets principaux
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)
    model = get_object_or_404(MLModel, id=model_id)
    
    try:
        # Chargement des fonctions nécessaires
        #load_model, load_experiment = load_model_type_librairies(model.model_type)

        # Détection de l'encodage et chargement des données
        encoding = get_encoding(dataset.file.path)
        df       = pd.read_csv(dataset.file.path, encoding=encoding)
        """
        # Chargement de l'expérience et du modèle
        exp = load_experiment(model.exp_file, data=df)
        themodel = load_model(model.model_file)

        # Debugging ou logs (à remplacer par un logger en production)
        print(f"Configuration de l'expérience : {exp.get_config()}")
        print(f"Modèle chargé : {themodel}")"""

        # Rendu de la vue
        return render(request, 'mlapp/model_manip.html', {
            'dataset': dataset,
            'workspace': workspace,
            'model': model,
        })

    except Exception as e:
        return JsonResponse({"error": f"Erreur lors du traitement : {str(e)}"}, status=500)



