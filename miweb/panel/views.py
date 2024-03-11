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
#cad = gra_Jittering(cad) #No hace lo que debe
#cad = gra_CountPlot(cad) #No hace lo que debe
#cad = gra_HistogramaM(cad)
#cad = gra_BoxplotM(cad)
#cad = gra_Correlograma(cad)
#cad = gra_Pairwise(cad)
#cad = gra_DiverginB(cad)
#cad = gra_DiverginT(cad)
#cad = gra_DiverginP(cad)
#cad = gra_DiverginL(cad)
#cad = gra_Areachart(cad)
#cad = gra_OrderedB(cad)
#cad = gra_Lollipop(cad)
#cad = gra_DotPlot(cad)
#cad = gra_Slopechar(cad)
#cad = gra_DumbbellPlot(cad)
#cad = gra_HistogramaCont(cad)
#cad = gra_HistogramaCateg(cad)
#cad = gra_DensityP(cad)
#cad = gra_DensityCH(cad) #No funciona
#cad = gra_Joyplot(cad)
#cad = gra_DotplotD(cad)
#cad = gra_Boxplot(cad)
#cad = gra_DotyBoxplot(cad)
#cad = gra_Violinplot(cad)
#cad = gra_Piramid(cad) #No esta completo
#cad = gra_PlotCategorical(cad) #No funciona
#cad = gra_Pie(cad)
#cad = gra_PieE(cad)
#cad = gra_Treemap(cad)
#cad = gra_Barchart(cad)
#cad = gra_Timeplot(cad)
#cad = gra_TimeSeries(cad)
#cad = gra_Autocorrelation(cad)
#cad = gra_CrossCorrelation(cad)
#cad = gra_TimeDecomposition(cad)
#cad = gra_MultipleTimeSeries(cad)
#cad = gra_DifferentScales(cad)
#cad = gra_TimeError(cad)
#cad = gra_StacketArea(cad)
#cad = gra_UnstacketArea(cad)
#cad = gra_Calendar(cad) #Falla libreria calmap
#cad = gra_SeasonalPlot(cad)
#cad = gra_Dendrogram(cad)
#cad = gra_Cluster(cad)
#cad = gra_AndreusCurve(cad)
cad = gra_ParallelCoordinates(cad)



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