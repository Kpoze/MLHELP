# dataset/urls.py
from django.urls import path
from . import views

app_name = 'datasets'  # Ceci définit le namespace

urlpatterns = [
    path('workspace/<int:workspace_id>/datasets/create/', views.create_dataset, name='create_dataset'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/', views.detail_dataset, name='detail_dataset'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/update-cell/', views.update_cell, name='update_cell'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/delete-rows/', views.delete_rows, name='delete_rows'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/save/', views.save_dataset, name='save_dataset'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/update/', views.update_dataset, name='update_dataset'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/delete/', views.delete_dataset, name='delete_dataset'),

]