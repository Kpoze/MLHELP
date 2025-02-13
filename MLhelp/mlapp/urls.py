# dataset/urls.py
from django.urls import path
from . import views

app_name = 'mlapp'  # Ceci définit le namespace

urlpatterns = [
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/choose-model/', views.choose_model_way, name='choose_model_way'),
    #path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/', views.check_classify_columns, name='custom_model_preprocessing_step'),
    #path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/', views.custom_model, name='get_custom_model'),
    #path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/step-0', views.check_target_step, name='get_custom_model_step_0'),
    #path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/step-1', views.preprocessing_step, name='get_custom_model_step_1'),
    #path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/step-2', views.model_step, name='get_custom_model_step_2'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/update-columns', views.check_classify_columns, name='update_columns'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/custom-preprocess-form', views.custom_preprocess_form, name='custom_preprocess_form'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-get-pre-model/', views.get_pre_model, name='get_pre_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/get-custom-model/', views.get_custom_model, name='get_custom_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/detail/', views.detail_model, name='detail_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('workspace/<int:workspace_id>/datasets/<int:dataset_id>/model/<int:model_id>/detail/', views.model_manip, name='model_manip'),
]