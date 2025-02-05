from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import DatasetForm, FilterColumnForm
from .models import Workspace, Datasets
import pandas as pd
import json
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import DatasetSerializer
import chardet
from django.urls import reverse
import numpy as np


@login_required
def create_dataset(request, workspace_id):
    # Récupérer le workspace correspondant à l'ID fourni dans l'URL
    workspace = get_object_or_404(Workspace, id=workspace_id)

    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)  # Assurez-vous de gérer les fichiers aussi
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.workspace = workspace  # Assigner le workspace au dataset
            dataset.save()
            return redirect('dashboard:detail_workspace', workspace_id=workspace.id)  # Redirection après la création
    else:
        form = DatasetForm()

    return render(request, 'datasets/create_dataset.html', {'form': form, 'workspace': workspace})



def get_encoding(df):
    with open(df, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Analyser les premiers 10k octets
        encoding = result['encoding']

    return encoding



def filter_dataframe(df, form_data):
    """
    Applique un filtrage dynamique sur un DataFrame à partir des données du formulaire.

    :param df: Le DataFrame à filtrer.
    :param form_data: Les données du formulaire contenant les critères de filtrage.
    :return: Le DataFrame filtré.
    """
    # Récupération des données du formulaire
    columns = form_data.get('columns')
    column_input = form_data.get('column_input')
    relation_input = form_data.get('relation_input')
    value_input = form_data.get('value_input')
    show_nulls_only = form_data.get('show_nulls_only')

    # Filtrage des colonnes sélectionnées
    if columns:
        selected_columns = [col.strip() for col in columns.split(',') if col.strip() in df.columns]
        if selected_columns:
            df = df[selected_columns]

    # Application du filtrage dynamique
    if column_input and relation_input and value_input is not None:
        column_choosed_type = df[column_input].dtype
        
        if show_nulls_only:
            df = df[df[column_input].isnull()]
        else:
            # Conversion de la valeur à filtrer en fonction du type de la colonne
            if column_choosed_type == 'object':
                converted_value = value_input
            elif pd.api.types.is_bool_dtype(column_choosed_type):
                # Gestion spécifique des colonnes booléennes
                if value_input.lower() in ["true", "1"]:
                    converted_value = True
                elif value_input.lower() in ["false", "0"]:
                    converted_value = False
                else:
                    raise ValueError("Valeur non valide pour un booléen. Utilisez True, False, 1 ou 0.")
            else:
                converted_value = np.array([value_input]).astype(column_choosed_type)[0]

            # Dictionnaire d'opérateurs
            operators = {
                '==': lambda x, y: x == y,
                '<': lambda x, y: x < y,
                '>': lambda x, y: x > y,
                '<=': lambda x, y: x <= y,
                '>=': lambda x, y: x >= y,
                '!=': lambda x, y: x != y,
            }

            # Appliquer le filtre avec l'opérateur et la valeur spécifiés
            if column_input in df.columns and relation_input in operators:
                try:
                    df = df[operators[relation_input](df[column_input], converted_value)]
                except Exception as e:
                    raise ValueError(f"Erreur lors de l'application du filtre : {str(e)}")

    return df


@login_required
def detail_dataset(request, dataset_id, workspace_id):
    # Obtenir le dataset
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset = get_object_or_404(Datasets, id=dataset_id)

    # Détection de l'encodage
    encoding = get_encoding(dataset.file.path)

    
    # Lire le fichier CSV en DataFrame
    df = pd.read_csv(dataset.file.path, encoding=encoding)
    print(df.info())
    # Créer le formulaire avec les données POST (s'il y en a) pour capturer les filtres
    form = FilterColumnForm(request.POST or None)
    # Si le formulaire est valide, appliquer le filtrage
    if form.is_valid():
        df = filter_dataframe(df, form.cleaned_data)

    # Récupérer les colonnes et les lignes sous forme de listes
    columns = df.columns.tolist()  # Liste des noms de colonnes
    rows = df.values.tolist()  # Liste des lignes (sans index)
    column_types = df.dtypes.astype(str).tolist()  # Liste des types de colonnes
    rows_with_types = []

    # Créer une liste des données avec le nom de la colonne et le type pour chaque cellule
    for row in df.itertuples(index=False):
        rows_with_types.append([(cell, col, str(dtype)) for cell, col, dtype in zip(row, df.columns, df.dtypes)])

    # Passer le DataFrame au template pour affichage
    return render(request, 'datasets/detail_dataset.html', {
        'dataset': dataset,
        'columns': columns,
        'column_types': column_types,
        'columns_with_types': list(zip(df.columns, df.dtypes.astype(str))),
        'rows_with_types': rows_with_types,  # Lignes avec leurs types de colonne
        'rows': rows,
        'workspace': workspace,
        'form': form,
        #'hidden_columns': hidden_columns,  # Passer la variable hidden_columns au template
    })


@login_required
def save_dataset(request, workspace_id, dataset_id):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            # Récupérer le workspace et le dataset
            workspace = get_object_or_404(Workspace, id=workspace_id)
            dataset = get_object_or_404(Datasets, id=dataset_id)

            # Récupérer les colonnes masquées
            hidden_columns = data.get("hidden_columns")
            print(type(hidden_columns))
            print(hidden_columns)

            # Détection de l'encodage
            encoding = get_encoding(dataset.file.path)

            
            # Lire le fichier CSV en DataFrame
            df = pd.read_csv(dataset.file.path, encoding=encoding)
            # Supprimer les colonnes masquées si elles existent
            if hidden_columns:
                df = df.drop(columns=hidden_columns)

            # Sauvegarder le DataFrame dans le fichier CSV
            df.to_csv(dataset.file.path, index=False, encoding=encoding)

            # Mettre à jour le modèle Dataset si nécessaire
            #dataset.hidden_columns = ",".join(hidden_columns)  # Si ce champ existe dans le modèle
            dataset.save()

            return JsonResponse({"status": "success", "message": "Dataset sauvegardé avec succès."})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"status": "error", "message": "Méthode non autorisée."})


#UPDATE
@login_required
def update_dataset(request, dataset_id, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset   = get_object_or_404(Datasets, id=dataset_id)
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES, instance=dataset)
        if form.is_valid():
            form.save()
            return redirect('datasets:dataset_details', dataset_id=dataset.id)
    else:
        form = DatasetForm(instance=dataset)
    return render(request, 'datasets/update_dataset.html', {'form': form, 'workspace': workspace})
    
#DELETE
@login_required
def delete_dataset(request, dataset_id, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id)
    dataset   = get_object_or_404(Datasets, id=dataset_id)
    if request.method == 'POST':
        dataset.delete()
        return redirect(reverse('dashboard:detail_workspace', kwargs={'workspace_id': workspace.id}))
    return render(request, 'datasets/delete_dataset.html', {'dataset': dataset,'workspace': workspace })


@login_required
def update_cell(request, workspace_id, dataset_id):
    if request.method == "POST":
        workspace = get_object_or_404(Workspace, id=workspace_id)
        dataset = get_object_or_404(Datasets, id=dataset_id)
        try:
            # Obtenir les données de la requête
            row_input    = int(request.POST.get("row"))
            column_input = int(request.POST.get("column"))
            value_input  = request.POST.get("value")
            # Détection de l'encodage
            encoding = get_encoding(dataset.file.path)

            
            # Lire le fichier CSV en DataFrame
            df = pd.read_csv(dataset.file.path, encoding=encoding)
            column_type = df[df.columns[column_input]].dtype  # Type de la colonne

            # Conversion en fonction du type de la colonne
            if column_type == 'object':
                converted_value = value_input

            elif pd.api.types.is_bool_dtype(column_type):
                # Gestion spécifique des colonnes booléennes
                if value_input.lower() in ["true", "1"]:
                    converted_value = True
                elif value_input.lower() in ["false", "0"]:
                    converted_value = False
                else:
                    return JsonResponse({"status": "error", "message": "Valeur non valide pour un booléen. Utilisez True, False, 1 ou 0."})

            else:
                converted_value = np.array([value_input]).astype(column_type)[0]

            # Vérification et mise à jour de la cellule
            if column_type == 'object' or np.issubdtype(type(converted_value), column_type):
                df.iat[row_input, column_input] = converted_value  # Mise à jour
                df.to_csv(dataset.file.path, index=False, encoding=encoding)  # Sauvegarder le DataFrame
                return JsonResponse({"status": "success", "message": "Cellule mise à jour avec succès"})
            else:
                return JsonResponse({"status": "error", "message": "Le type de la valeur ne correspond pas à celui de la colonne"})
            
        except ValueError as e:
            return JsonResponse({"status": "error", "message": f"Erreur de conversion de la valeur : {e}"})
    else:
        return JsonResponse({"status": "error", "message": "Requête non valide"})
    

def delete_rows(request, workspace_id, dataset_id):
    if request.method == 'POST':
        try:
            # Extraire les données JSON envoyées par JavaScript
            data = json.loads(request.body)
            rows_to_delete = list(map(int, data.get('rows', [])))

            # Charger le dataset
            workspace = get_object_or_404(Workspace, id=workspace_id)
            dataset = get_object_or_404(Datasets, id=dataset_id)
            # Détection de l'encodage
            encoding = get_encoding(dataset.file.path)

            
            # Lire le fichier CSV en DataFrame
            df = pd.read_csv(dataset.file.path, encoding=encoding)

            # Supprimer les lignes sélectionnées
            df.drop(rows_to_delete, axis=0, inplace=True)

            # Sauvegarder le fichier CSV mis à jour
            df.to_csv(dataset.file.path, index=False, encoding=encoding)

            return JsonResponse({"status": "success", "message": "Lignes supprimées avec succès."})
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})
    else:
        return JsonResponse({"status": "error", "message": "Méthode non autorisée."})
    


        