from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Workspace, Datasets, MLModel,CustomModelProgression
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
import uuid
###############FUNCTION#####################


def generate_unique_suffix():
    return str(uuid.uuid4())[:8] 



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
    target = request.POST.get("is_target")
    model_type_choosed = request.POST.get("model_type_choosed")
    data_target = request.POST.get("data_target") if target == 'yes' else None
    model_name_input = request.POST.get("model_name") or "Robert"
    unique_suffix = generate_unique_suffix()
    data = {
                'target': target,
                'model_type_choosed': model_type_choosed,
                'data_target': data_target,
                'model_name_input': model_name_input,
                'model_name_input_file' : f"{model_name_input}_{unique_suffix}"
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


def get_model_form(model_type, data_post=None):
    """Retourne le bon formulaire de sélection de modèle en fonction du type."""
    if model_type == 'classification':
        return ClassificationModelForm(data_post)  # ✅ Passer `data`
    elif model_type == 'regression':
        return RegressionModelForm(data_post)
    elif model_type == 'clustering':
        return ClusteringModelForm(data_post)
    return None  # Si aucun type n’est sélectionné

def get_hyperparameter_form(model_type, model_name, data_post=None):
    """Retourne le formulaire d'hyperparamètres selon le modèle choisi."""
    if model_type == 'classification':
        if model_name == 'random_forest_classifier':
            return RandomForestHyperparametersForm(data_post) 
        elif model_name == 'svm':
            return SVMHyperparametersForm(data_post)
        elif model_name == 'logistic_regression':
            return LogisticRegressionHyperparametersForm(data_post)

    elif model_type == 'regression':
        if model_name == 'linear_regression':
            return LinearRegressionHyperparametersForm(data_post)
        elif model_name == 'ridge_regression':
            return RidgeRegressionHyperparametersForm(data_post)
        elif model_name == 'decision_tree':
            return DecisionTreeHyperparametersForm(data_post)

    elif model_type == 'clustering':
        if model_name == 'kmeans':
            return KMeansHyperparametersForm(data_post)
        elif model_name == 'dbscan':
            return DBSCANHyperparametersForm(data_post)
        elif model_name == 'hierarchical':
            return HierarchicalClusteringHyperparametersForm(data_post)

    return None

def train_custom_model(model,data,preprocessor,hyperparameter):
    preprocess = pickle.loads(base64.b64decode(preprocessor))
    if model == 'random_forest_classifier':
        from sklearn.ensemble import RandomForestClassifier
        the_model = RandomForestClassifier(n_estimators=hyperparameter['n_estimators'],max_depth=hyperparameter['max_depth'])
    elif model == 'svm':
        from sklearn import svm
        the_model = svm.SVC(C=hyperparameter['C'],kernel=hyperparameter['kernel'])
    elif model == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        the_model = LogisticRegression(C=hyperparameter['C'],max_iter=hyperparameter['max_iter'],solver=hyperparameter['solver'],penalty=hyperparameter['penalty'],tol=hyperparameter['tol'])
    elif model == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        the_model = LinearRegression(fit_intercept=hyperparameter['fit_intercept'],normalize=hyperparameter['normalize'])
    
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
    }

    if request.method == "POST":
        #try:
            # Récupérer les paramètres utilisateur
            data = get_data_post(request)
            print(data)  
            request.session['model_type_choosed'] = data['model_type_choosed']
            print(f"Type de modèle : {data['model_type_choosed']}")
            print(f"Colonne cible : {data['data_target']}")
            print(f"Nom du modèle : {data['model_name_input']}")
            exp = setup_experiment(request, df, data['model_type_choosed'], data['target'], data['data_target'])

                # Entraîner et sélectionner le meilleur modèle
            model_data['best_model'] = train_and_select_best_model(exp)

                # Obtenir le nom et les paramètres du modèle
            model_data['model_name']       = type( model_data['best_model']).__name__
            model_data['model_param']      =  model_data['best_model'].get_params()
            json_model_param = json.dumps(model_data['model_param'])

                # Sauvegarder le modèle et l'expérience
            exp_filepath = save_experiment_to_media(exp, data['model_name_input'])
            model_file_path = save_model_to_media(exp, model_data['best_model'], data['model_name_input_file'])
            ml_model = save_model_and_experiment(
                    request, workspace, dataset, data['model_type_choosed'], data['model_name_input'],
                    json_model_param, model_file_path, exp, exp_filepath
                )

            print(f"Modèle PyCaret sauvegardé avec succès (ID : {ml_model.id})")
            return redirect(reverse('mlapp:detail_model', kwargs={'workspace_id': workspace.id, 'dataset_id': dataset.id}))
            """except Exception as e:
            return JsonResponse({"error": f"Erreur lors du traitement : {str(e)}"}, status=500)"""

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
@login_required
def get_custom_model(request, workspace_id, dataset_id):
    workspace    = get_object_or_404(Workspace, id=workspace_id)
    dataset      = get_object_or_404(Datasets, id=dataset_id)
    encoding     = get_encoding(dataset.file.path)
    df           = pd.read_csv(dataset.file.path, encoding=encoding)
    column_types = classify_columns(df)


    # Initialiser current_step si ce n'est pas déjà fait
    if 'current_step' not in request.session:
        request.session['current_step'] = 1

    current_step = request.session['current_step']


    if request.GET.get('back') == '1':
        if current_step > 1:  
            request.session['current_step'] -= 1  
            return redirect('mlapp:get_custom_model', workspace_id=workspace.id, dataset_id=dataset.id)
        
    form = None
    if current_step == 2:
            column_types = classify_columns(df)
            if 'column_types' not in request.session:
                column_types  = classify_columns(df)
            else:
                column_types  = request.session.get('column_types') # Récupère les colonnes classées par type ('numeric', 'categorical', etc.)
            is_target         = request.session.get('is_target')
            print(is_target)
            data_target       = request.session.get('data_target')
            print(data_target)
            form              = PreprocessingForm()
 
    elif current_step == 3:
            is_target         = request.session.get('is_target')
            print(is_target)
            data_target       = request.session.get('data_target')
            print(data_target)
            form              = ModelTypeForm(is_target)

    elif current_step == 4:
            model_type   = request.session.get('model_type')
            print(model_type)
            form         = get_model_form(model_type)

    elif current_step == 5:
            model      = request.session.get('model')
            print(model)
            model_type = request.session.get('model_type')
            print(model_type)
            form       = get_hyperparameter_form(model_type, model)
    
    elif current_step == 6 : 
        is_target       = request.session.get('is_target')
        data_target     = request.session.get('data_target')
        data            = request.session.get('data')
        preprocessor    = request.session.get('preprocessor')
        model_type      = request.session.get('model_type')
        model           = request.session.get('model')
        hyperparameter  = request.session.get('hyperparameter')



        # Gestion des soumissions
    if request.method == 'POST':
            if current_step == 1:
                    is_target                      = request.POST.get('is_target')
                    data_target                    = request.POST.get('data_target') if request.POST.get('is_target') == 'yes' else None
                    request.session['data_target'] = data_target
                    request.session['is_target']   = is_target

            elif current_step == 2:
                    if request.headers.get('Content-Type') == 'application/json':
                        try:
                            column_data = json.loads(request.body)
                            column_name = column_data.get('column_name')
                            new_type = column_data.get('new_type')
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

                            # Sauvegarder l'état global dans la session
                            print(column_types)
                            request.session['column_types'] = column_types

                            # Retourner la liste mise à jour des colonnes
                            return JsonResponse({'success': True, 'updated_column_types': column_types})

                        except Exception as e:
                            return JsonResponse({'success': False, 'error': str(e)}, status=400)
                    # Sinon, traite la soumission du formulaire
                    else:
                        form = PreprocessingForm(request.POST)
                        if form.is_valid():
                            print(column_types)
                            print(form.cleaned_data)
                            data                            = split_data(df,data_target,0.2,42)
                            preprocessor                    = create_preprocessing_pipeline(form.cleaned_data,column_types)
                            preprocessor_serialized         = base64.b64encode(pickle.dumps(preprocessor)).decode('utf-8')
                            request.session['data']         = data
                            request.session['preprocessor'] = preprocessor_serialized
                            #print(data)
                            #print(preprocessor_serialized)
                            #progression.save()
                            # Sérialisation avec pickle et encodage Base64
                            """print(data)
                            print(preprocessor)
                            print(pickle.loads(base64.b64decode(preprocessor_serialized)))
                            print(form.cleaned_data)  # Logique pour traiter les données du formulaire"""

            elif current_step == 3:
                    #print( request.POST.get('model_type'))
                    form = ModelTypeForm(is_target,request.POST) 
                    if form.is_valid():
                        model_type = form.cleaned_data['model_type']
                        request.session['model_type'] = model_type
                        print(request.session['model_type'])
                        print(model_type)
                        #progression.save()

            elif current_step == 4:
                    form = get_model_form(model_type,request.POST)
                    if form.is_valid():
                        #progression.model_name = model_name
                        model = form.cleaned_data['model']
                        request.session['model'] = model
                        print(model)
                        #progression.save()

            elif current_step == 5:
                    form = get_hyperparameter_form(model_type, model, request.POST)
                    if form.is_valid():
                        hyperparameter = form.cleaned_data
                        request.session['hyperparameter'] = hyperparameter
                        print(hyperparameter)

            elif current_step == 6:
                        print(is_target)
                        print(data_target)
                        print(data)
                        print(preprocessor)
                        print(model_type)
                        print(model)
                        print(hyperparameter)

                        return JsonResponse({'status': 'success', 'message': 'Modèle configuré avec succès!'})
            # Passer à l’étape suivante
            request.session['current_step'] += 1

            return redirect('mlapp:get_custom_model', workspace_id=workspace.id, dataset_id=dataset.id)

    return render(request, 'mlapp/get_custom_model.html', {
            'workspace'    : workspace,
            'dataset'      : dataset,
            'form'         : form,
            'current_step' : current_step,
            'columns'      : list(df.columns),
            'column_types' : column_types,
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



