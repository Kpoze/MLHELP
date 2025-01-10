from django.db import models
from django.conf import settings
from dashboard.models import Workspace
from datasets.models import Datasets
from django.core.exceptions import ValidationError

# Create your models here.
class MLModel(models.Model):
    workspace      = models.ForeignKey(Workspace, on_delete=models.CASCADE)
    dataset        = models.ForeignKey(Datasets, on_delete=models.CASCADE)
    model_name     = models.CharField(max_length=255)
    model_param    = models.JSONField()  
    model_file     = models.FileField(upload_to='models/')
    exp_file       = models.FileField(upload_to='config/')
    created_at     = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} - {self.workspace} - {self.dataframe}"