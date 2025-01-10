from django.urls import path, include
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.home, name='home'),
    path('workspace',views.home, name='home'),
    path('workspace/create/', views.create_workspace, name='create_workspace'),
    path('workspace/<int:workspace_id>/', views.detail_workspace, name='detail_workspace'),
    path('workspace/<int:workspace_idid>/update/', views.update_workspace, name='update_workspace'),
    path('workspace/<int:workspace_idid>/delete/', views.delete_workspace, name='delete_workspace'),
    path('workspace/<int:workspace_id>/datasets/', include('datasets.urls', namespace='datasets')),
]