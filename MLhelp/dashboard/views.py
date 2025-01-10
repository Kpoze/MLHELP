from django.shortcuts import render
from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth.decorators import login_required  
from .models import Workspace
from datasets.models import Datasets
from .forms import WorkspaceForm
import pandas as pd
import json
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

#WORKPLACE
@login_required
def create_workspace(request):
    if request.method == 'POST':
        form = WorkspaceForm(request.POST)
        if form.is_valid():
            workspace       = form.save(commit=False)  # Ne pas encore sauvegarder dans la base de données
            workspace.owner = request.user           # Associer l'utilisateur connecté comme propriétaire
            workspace.save()                         # Maintenant, sauvegarder dans la base de données
            return redirect('dashboard:detail_workspace', workspace_id=workspace.id)                # Rediriger après le succès
        
    else:
        form = WorkspaceForm()
    return render(request, 'dashboard/create_workspace.html', {'form': form})

@login_required
def home(request):
    print("ok")
    workspaces = Workspace.objects.filter(owner=request.user)
    return render(request, 'dashboard/home.html',
                    {'workspaces' : workspaces}
                    )

@login_required
def detail_workspace(request, workspace_id):
    # Get the Workspace instance based on the workspace_id from the URL
    workspace = get_object_or_404(Workspace, id=workspace_id)
    
    # Filter datasets based on the retrieved workspace
    datasets = Datasets.objects.filter(workspace=workspace)
    
    # Render the template and pass the datasets to the context
    return render(request, 'dashboard/detail_workspace.html', {'datasets': datasets,'workspace': workspace})

@login_required
def update_workspace(request,workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    if request.method == 'POST':
        form = WorkspaceForm(request.POST, instance=workspace)
        if form.is_valid():
            form.save()
            return redirect('dashboard:detail_workspace', workspace_id=workspace_id)
    else:
        form = WorkspaceForm(instance=workspace)
    return render(request, 'dashboard/update_workspace.html', {'form': form})


@login_required
def delete_workspace(request, workspace_id):
    workspace = get_object_or_404(Workspace, id=workspace_id, owner=request.user)
    if request.method == 'POST':
        workspace.delete()
        return redirect('dashboard:home')
    return render(request, 'dashboard/delete_workspace.html', {'workspace': workspace})
