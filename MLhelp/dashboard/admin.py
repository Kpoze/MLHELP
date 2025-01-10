from django.contrib import admin
from .models import Workspace  # Assure-toi d'importer ton modèle

# Enregistre le modèle Dataset pour l'administration
admin.site.register(Workspace)