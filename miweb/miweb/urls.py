from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path('admin/', admin.site.urls),
    #path('hello/', include("hello.urls")),
    #path('newyear/', include("newyear.urls")),
    #path('panel/', include("panel.urls")),
    path('datos/', include("datos.urls")),
    path('api-auth/', include('rest_framework.urls'))
]
