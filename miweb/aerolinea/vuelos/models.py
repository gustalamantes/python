from django.db import models

# Create your models here.
class Aeropuerto(models.Model):
    code = models.CharField(max_length=3)
    city = models.CharField(max_length=64)

    def __str__(self):
        return f"{self.city} ({self.code})"
    
    
class Vuelo(models.Model):
    origin = models.ForeignKey(Aeropuerto, on_delete=models.CASCADE, related_name="salidas")
    destination = models.ForeignKey(Aeropuerto, on_delete=models.CASCADE, related_name="llegadas")
    duration = models.IntegerField()
    
    def __str__(self):
        return f"{self.id} : {self.origin} to {self.destination}"

     
class Pasajero(models.Model):
    first = models.CharField(max_length=64)
    last = models.CharField(max_length=64)
    vuelos = models.ManyToManyField(Vuelo, blank=True, related_name="pasajeros")

    def __str__(self):
        return f"{self.first} {self.last}"