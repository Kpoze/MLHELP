from django import forms
from .models import Datasets

class DatasetForm(forms.ModelForm):
    class Meta:
        model  = Datasets
        fields = ['name', 'file'] 