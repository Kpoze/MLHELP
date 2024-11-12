from rest_framework import serializers
from .models import Datasets

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Datasets
        fields = '__all__'  # Liste des champs à exposer dans l'API