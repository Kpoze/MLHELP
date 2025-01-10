from django.db import models
from django.conf import settings
from dashboard.models import Workspace
from django.core.exceptions import ValidationError

def validate_file_size(file):
    max_size_kb = 100000  # 100MB
    if file.size > max_size_kb * 1024:
        raise ValidationError(f"File size exceeds {max_size_kb} KB.")

def validate_file_extension(file):
    valid_extensions = ['.csv', '.xlsx']
    if not any([file.name.endswith(ext) for ext in valid_extensions]):
        raise ValidationError("Unsupported file extension. Only .csv and .xlsx files are allowed.")


    def __str__(self):
        return self.name


class Datasets(models.Model):
    STATUS_TERMINATED  = 'terminated'
    STATUS_IN_PROGRESS = 'in progress'

    STATUS_CHOICES = [
        (STATUS_TERMINATED, 'Terminated'),
        (STATUS_IN_PROGRESS, 'In progress'),
    ]
    
    workspace = models.ForeignKey(Workspace, on_delete=models.CASCADE, null=False, blank=False)
    name      = models.CharField(max_length=255)
    file      = models.FileField(upload_to='datasets/', validators=[validate_file_size, validate_file_extension])
    status    = models.CharField(max_length=15, choices=STATUS_CHOICES, default=STATUS_IN_PROGRESS)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not Workspace.objects.exists():
            raise ValidationError("You must create a Workspace before adding a Dataset.")
        super().save(*args, **kwargs)