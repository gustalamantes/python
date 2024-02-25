from django.shortcuts import render
from django.http import HttpResponseBadRequest, HttpResponseRedirect, Http404
from django.urls import reverse
from .models import Vuelo, Aeropuerto, Pasajero
#from mysql.connector import connect, Error

# Create your views here.
def index(request):
    return render(request, "vuelos/index.html", {
        "vuelos": Vuelo.objects.all()
    })


def vuelo(request, vuelo_id):
    try:
        vuelo = Vuelo.objects.get(id=vuelo_id)
    except Vuelo.DoesNotExist:
        raise Http404("Vuelo inexistente.")
    return render(request, "vuelos/vuelo.html", {
        "vuelo": vuelo,
        "pasajeros": vuelo.pasajeros.all(),
        "non_pasajero": Pasajero.objects.exclude(vuelos=vuelo).all()
    })


def book(request, vuelo_id):
    if request.method == "POST":
        try:
            pasajero = Pasajero.objects.get(pk=int(request.POST["pasajero"]))
            vuelo = Vuelo.objects.get(pk=vuelo_id)
        except KeyError:
            return HttpResponseBadRequest("Error: no selecciono vuelo")
        except Vuelo.DoesNotExist:
            return HttpResponseBadRequest("Error: el vuelo no existe")
        except Pasajero.DoesNotExist:
            return HttpResponseBadRequest("Error: el pasajero no existe")
        pasajero.vuelos.add(vuelo)
        return HttpResponseRedirect(reverse("vuelo", args=(vuelo_id,)))