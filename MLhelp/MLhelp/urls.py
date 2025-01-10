"""
URL configuration for MLhelp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
import authentication.views 
from django.contrib.auth.views import LoginView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    #AUTH
    path('admin/', admin.site.urls),
    path('signup/', authentication.views.signup_page, name='signup'),
    path('login/', LoginView.as_view(
            template_name='authentication/login.html',
            redirect_authenticated_user=True),
        name='login'),
    path('logout/', authentication.views.logout_user, name='logout'),


    #Dashboard
    path('', include('dashboard.urls', namespace='dashboard')),


    #dataset
    path('', include('datasets.urls', namespace='datasets')),

    #graphics
    #path('', include('graphics.urls', namespace='graphics')),
    #mlapp
    path('', include('mlapp.urls', namespace='mlapp')),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
