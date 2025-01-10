from django import forms
from .models import Datasets

class DatasetForm(forms.ModelForm):
    class Meta:
        model  = Datasets
        fields = ['name', 'file'] 

class FilterColumnForm(forms.Form):
    columns = forms.CharField(required=False, help_text="Entrez les colonnes, séparées par des virgules.")
    column_input = forms.CharField(required=False, help_text="Nom de la colonne à filtrer.")
    relation_input = forms.ChoiceField(choices=[
        ('==', '=='),
        ('<', '<'),
        ('>', '>'),
        ('<=', '<='),
        ('>=', '>='),
        ('!=', '!=')
    ], required=False, help_text="Opérateur de comparaison.")
    value_input = forms.CharField(required=False, help_text="Valeur cible pour le filtre.")
    show_nulls_only = forms.BooleanField(required=False, label="Afficher uniquement les valeurs nulles")  # Nouveau champ