from django.contrib import admin
from .models import User  # Importe ton modèle CustomUser

# Enregistre le modèle CustomUser dans l'administration
admin.site.register(User)
