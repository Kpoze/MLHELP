from django.contrib import admin
from .models import Datasets  # Assure-toi d'importer ton modèle

# Enregistre le modèle Dataset pour l'administration
admin.site.register(Datasets)