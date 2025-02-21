# dataset/urls.py
from django.urls import path
from . import views

app_name = 'mlapp'  # Ceci définit le namespace

urlpatterns = [
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/choose-model/', views.choose_model_way, name='choose_model_way'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-get-pre-model/', views.get_pre_model, name='get_pre_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/', views.get_custom_model, name='get_custom_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/detail/', views.detail_model, name='detail_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/<int:model_id>/detail/', views.model_manip, name='model_manip'),
]