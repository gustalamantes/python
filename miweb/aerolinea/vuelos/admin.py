from django.contrib import admin

from .models import Aeropuerto, Vuelo, Pasajero

# Register your models here.

class VueloAdmin(admin.ModelAdmin):
    list_display = ("__str__", "duration")

#class PasajeroAdmin(admin.ModelAdmin):
#    filter_horizontal = ("vuelos",)
    

admin.site.register(Aeropuerto)
admin.site.register(Pasajero)
admin.site.register(Vuelo, VueloAdmin)
#admin.site.register(Pasajero, PasajeroAdmin)
