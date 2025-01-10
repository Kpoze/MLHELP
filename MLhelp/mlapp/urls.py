# dataset/urls.py
from django.urls import path
from . import views

app_name = 'mlapp'  # Ceci définit le namespace

urlpatterns = [
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-model/', views.get_model, name='get_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/detail/', views.detail_model, name='detail_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
]