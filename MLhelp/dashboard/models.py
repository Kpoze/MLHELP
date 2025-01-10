from django.db import models
from django.conf import settings
# Create your models here.

@staticmethod
def is_workspace_table_empty():
    return not Workspace.objects.exists()


class Workspace(models.Model):
    owner        = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name         = models.CharField(max_length=255)
    description  = models.TextField(blank=True, null=True)
    created_at   = models.DateTimeField(auto_now_add=True)