# Generated by Django 5.1.1 on 2025-01-17 21:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mlapp', '0005_rename_dataframe_mlmodel_dataset'),
    ]

    operations = [
        migrations.AddField(
            model_name='mlmodel',
            name='model_type',
            field=models.CharField(default=1, max_length=255),
            preserve_default=False,
        ),
    ]
