from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    #profile_photo = models.ImageField()
     date_of_birth = models.DateField(blank=True, null=True)

    
