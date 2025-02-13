from django.db import models
from django.conf import settings
from dashboard.models import Workspace
from datasets.models import Datasets
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import timedelta
# Create your models here.
class MLModel(models.Model):
    workspace      = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    dataset        = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    model_type     = models.CharField(max_length=255)
    model_name     = models.CharField(max_length=255)
    model_param    = models.JSONField()  
    model_file     = models.FileField(upload_to='models/')
    exp_file       = models.FileField(upload_to='config/')
    created_at     = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} - {self.workspace} - {self.dataframe}"
    

class CustomModelProgression(models.Model):
    workspace           = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    dataset             = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    current_step        = models.IntegerField(default=1)
    is_target           = models.CharField(max_length=10, null=True, blank=True)
    data_target         = models.CharField(max_length=255, null=True, blank=True)
    columns_type        = models.CharField(max_length=50, null=True, blank=True)
    preprocessing_steps = models.JSONField(null=True, blank=True)
    model_type          = models.CharField(max_length=50, null=True, blank=True)
    model_name          = models.CharField(max_length=50, null=True, blank=True)
    hyperparameters     = models.JSONField(null=True, blank=True)
    last_updated        = models.DateTimeField(auto_now=True)
    
    def has_expired(self):
        return self.last_updated < timezone.now() - timedelta(days=7)