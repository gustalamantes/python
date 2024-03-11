from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('acceso/', include('rest_framework.urls')),
    path('accounts/profile/', include('acceso.urls')),
]
