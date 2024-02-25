from django.shortcuts import render
import datetime, zoneinfo

#import os
#cwd = os.path.realpath('.')

from .grafica import *


#grafica()
cad = ''
#graficas libreria numpy, pandas, matplotlib
#cad = gra_disPuntos(cad)
#cad = gra_diaLineas(cad)
#cad = gra_diaAreas(cad)
#cad = gra_diaBarrasV(cad)
#cad = gra_diaBarrasH(cad)
#cad = gra_Histograma(cad)
#cad = gra_Sectores(cad)
#cad = gra_CajaBigotes(cad)
#cad = gra_Violin(cad)
#cad = gra_Contorno(cad)
#cad = gra_MapaColor(cad)
#cad = gra_MapaColorHist(cad)
#cad = gra_Lineas2(cad)
#cad = gra_Lineas3(cad)
#cad = gra_Lineas4(cad)
#cad = gra_Lineas5(cad)
#cad = gra_Lineas6(cad)
#cad = gra_Lineas7(cad)
#cad = gra_Lineas8(cad)
#cad = gra_Multiples(cad)
#cad = gra_PandasLineas(cad)
#graficas TOP y seaborn
#cad = gra_Scatter(cad)
#cad = gra_PlotEncircling(cad)
#cad = gra_PlotRegression(cad)
#cad = gra_PlotRegressionCol(cad)
    #Revisar colores y medida de jitter
#cad = gra_Jittering(cad) 
#cad = gra_CountPlot(cad)
    #Termina revisar
cad = gra_HistogramaM(cad)
cad = gra_BoxplotM(cad)
cad = gra_Correlograma(cad)
cad = gra_Pairwise(cad)



zona = zoneinfo.ZoneInfo("America/Mexico_City")
ahora = datetime.datetime.now(zona).strftime ("%d-%m-%Y %H:%M:%S")

# Create your views here.
#"panel" : "<img src=\"{% static 'panel/lineas.png' %}\" />",
def indexp(request):
    now = datetime.datetime.now()
    return render(request, "panel/indexp.html", {
        "panel" : cad,
        "fecha": "Fecha de impresion: " + ahora
    })