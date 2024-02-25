import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import warnings; 
import ssl

#Ignorar warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd

def gra_disPuntos(c):
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3, 4], [1, 2, 0, 0.5])
    plt.savefig("./panel/static/panel/graf_disPuntos.png")
    cad = c + '<p>Dispersion de Puntos</p>\n' + '<p><img src="/static/panel/graf_disPuntos.png" /></p>\n' 
    return cad

def gra_diaLineas(c):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 2, 0, 0.5])
    plt.savefig("./panel/static/panel/graf_diaLineas.png")
    cad = c + '<p>Diagrama de Lineas</p>\n' + '<p><img src="/static/panel/graf_diaLineas.png" /></p>\n' 
    return cad

def gra_diaAreas(c):
    fig, ax = plt.subplots()
    ax.fill_between([1, 2, 3, 4], [1, 2, 0, 0.5])
    plt.savefig("./panel/static/panel/graf_diaAreas.png")
    cad = c + '<p>Diagrama de Areas</p>\n' + '<p><img src="/static/panel/graf_diaAreas.png" /></p>\n' 
    return cad

def gra_diaBarrasV(c):
    fig, ax = plt.subplots()
    ax.bar([1, 2, 3], [3, 2, 1])
    plt.savefig("./panel/static/panel/graf_diaBarrasV.png")
    cad = c + '<p>Diagrama de Areas</p>\n' + '<p><img src="/static/panel/graf_diaBarrasV.png" /></p>\n' 
    return cad

def gra_diaBarrasH(c):
    fig, ax = plt.subplots()
    ax.barh([1, 2, 3], [3, 2, 1])
    plt.savefig("./panel/static/panel/graf_diaBarrasH.png")
    cad = c + '<p>Diagrama de Areas</p>\n' + '<p><img src="/static/panel/graf_diaBarrasH.png" /></p>\n' 
    return cad

def gra_Histograma(c):
    fig, ax = plt.subplots()
    x = np.random.normal(5, 1.5, size=1000)
    ax.hist(x, np.arange(0, 11))
    plt.savefig("./panel/static/panel/graf_Histograma.png")
    cad = c + '<p>Histograma</p>\n' + '<p><img src="/static/panel/graf_Histograma.png" /></p>\n' 
    return cad

def gra_Sectores(c):
    fig, ax = plt.subplots()
    ax.pie([5, 4, 3, 2, 1])
    plt.savefig("./panel/static/panel/graf_Sectores.png")
    cad = c + '<p>Diagrama Sectores</p>\n' + '<p><img src="/static/panel/graf_Sectores.png" /></p>\n' 
    return cad

def gra_CajaBigotes(c):
    fig, ax = plt.subplots()
    ax.boxplot([1, 2, 1, 2, 3, 4, 3, 3, 5, 7])
    plt.savefig("./panel/static/panel/graf_CajaBigotes.png")
    cad = c + '<p>Diagrama Caja y Bigotes</p>\n' + '<p><img src="/static/panel/graf_CajaBigotes.png" /></p>\n' 
    return cad

def gra_Violin(c):
    fig, ax = plt.subplots()
    ax.violinplot([1, 2, 1, 2, 3, 4, 3, 3, 5, 7])
    plt.savefig("./panel/static/panel/graf_Violin.png")
    cad = c + '<p>Diagrama de Violin</p>\n' + '<p><img src="/static/panel/graf_Violin.png" /></p>\n' 
    return cad

def gra_Contorno(c):
    fig, ax = plt.subplots()
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.0, 3.0, 100)
    x, y = np.meshgrid(x, y)
    z = np.sqrt(x**2 + 2*y**2)
    ax.contourf(x, y, z)
    plt.savefig("./panel/static/panel/graf_Contorno.png")
    cad = c + '<p>Diagrama de Contorno</p>\n' + '<p><img src="/static/panel/graf_Contorno.png" /></p>\n' 
    return cad

def gra_MapaColor(c):
    fig, ax = plt.subplots()
    x = np.random.random((16, 16))
    ax.imshow(x)
    plt.savefig("./panel/static/panel/graf_MapaColor.png")
    cad = c + '<p>Diagrama Mapa de Color</p>\n' + '<p><img src="/static/panel/graf_MapaColor.png" /></p>\n' 
    return cad

def gra_MapaColorHist(c):
    fig, ax = plt.subplots()
    x, y = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, 0.4], [0.4, 0.5]], size=1000).T
    ax.hist2d(x, y)
    plt.savefig("./panel/static/panel/graf_MapaColorHist.png")
    cad = c + '<p>Diagrama Mapa de Color Histograma</p>\n' + '<p><img src="/static/panel/graf_MapaColorHist.png" /></p>\n' 
    return cad


def gra_Lineas2(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'], color = 'tab:purple')
    ax.plot(dias, temperaturas['Barcelona'], color = 'tab:green')
    plt.savefig("./panel/static/panel/graf_Lineas2.png")
    cad = c + '<p>Diagrama Lineas Color</p>\n' + '<p><img src="/static/panel/graf_Lineas2.png" /></p>\n' 
    return cad

def gra_Lineas3(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'], marker = '^')
    ax.plot(dias, temperaturas['Barcelona'], marker = 'o')
    plt.savefig("./panel/static/panel/graf_Lineas3.png")
    cad = c + '<p>Diagrama Lineas Marcadores</p>\n' + '<p><img src="/static/panel/graf_Lineas3.png" /></p>\n' 
    return cad

def gra_Lineas4(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'], linestyle = 'dashed')
    ax.plot(dias, temperaturas['Barcelona'], linestyle = 'dotted')
    plt.savefig("./panel/static/panel/graf_Lineas4.png")
    cad = c + '<p>Diagrama Lineas Discontinuas</p>\n' + '<p><img src="/static/panel/graf_Lineas4.png" /></p>\n' 
    return cad

def gra_Lineas5(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'])
    ax.plot(dias, temperaturas['Barcelona'])
    ax.set_title('Evolución de la temperatura diaria', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    plt.savefig("./panel/static/panel/graf_Lineas5.png")
    cad = c + '<p>Diagrama Lineas Titulo</p>\n' + '<p><img src="/static/panel/graf_Lineas5.png" /></p>\n' 
    return cad

#Otras opciones de ejes en https://aprendeconalf.es/docencia/python/manual/matplotlib/

def gra_Lineas6(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'])
    ax.plot(dias, temperaturas['Barcelona'])
    ax.set_xlabel("Días", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    ax.set_ylabel("Temperatura ºC")
    ax.set_ylim([20,35])
    ax.set_yticks(range(20, 35))
    plt.savefig("./panel/static/panel/graf_Lineas6.png")
    cad = c + '<p>Diagrama Lineas Ejes</p>\n' + '<p><img src="/static/panel/graf_Lineas6.png" /></p>\n' 
    return cad

def gra_Lineas7(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'], label = 'Madrid')
    ax.plot(dias, temperaturas['Barcelona'], label = 'Barcelona')
    ax.legend(loc = 'upper right')
    plt.savefig("./panel/static/panel/graf_Lineas7.png")
    cad = c + '<p>Diagrama Lineas Leyenda</p>\n' + '<p><img src="/static/panel/graf_Lineas7.png" /></p>\n' 
    return cad

def gra_Lineas8(c):
    fig, ax = plt.subplots()
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax.plot(dias, temperaturas['Madrid'])
    ax.plot(dias, temperaturas['Barcelona'])
    ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed')
    plt.savefig("./panel/static/panel/graf_Lineas8.png")
    cad = c + '<p>Diagrama Lineas Rejilla</p>\n' + '<p><img src="/static/panel/graf_Lineas8.png" /></p>\n' 
    return cad

def gra_Multiples(c):
    fig, ax = plt.subplots(2, 2, sharey = True)
    dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D']
    temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]}
    ax[0, 0].plot(dias, temperaturas['Madrid'])
    ax[0, 1].plot(dias, temperaturas['Barcelona'], color = 'tab:orange')
    ax[1, 0].bar(dias, temperaturas['Madrid'])
    ax[1, 1].bar(dias, temperaturas['Barcelona'], color = 'tab:orange')
    plt.savefig("./panel/static/panel/graf_Multiples.png")
    cad = c + '<p>Diagrama Multiple</p>\n' + '<p><img src="/static/panel/graf_Multiples.png" /></p>\n' 
    return cad

def gra_PandasLineas(c):
    df = pd.DataFrame({'Días':['L', 'M', 'X', 'J', 'V', 'S', 'D'], 
                   'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], 
                   'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]})
    fig, ax = plt.subplots()
    df.plot(x = 'Días', y = 'Madrid', ax = ax)
    df.plot(x = 'Días', y = 'Barcelona', ax = ax)
    plt.savefig("./panel/static/panel/graf_PandasLineas.png")
    cad = c + '<p>Diagrama Pandas y Lineas</p>\n' + '<p><img src="/static/panel/graf_PandasLineas.png" /></p>\n' 
    return cad

#Terminan graficas matplotlib
#Inicia TOP y seaborn
def gra_Scatter(c):
    # Import dataset 
    midwest = pd.read_csv("./panel/static/panel/midwest_filter.csv")
    # Prepare Data 
    # Create as many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]
    # Draw Plot for Each Category
    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')
    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal', 
                data=midwest.loc[midwest.category==category, :], s=20, c=colors[i], label=str(category))
    # Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000), xlabel='Area', ylabel='Population')
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Scatterplot of Midwest Area vs Population", fontsize=22)
    plt.legend(fontsize=12)   
    plt.savefig("./panel/static/panel/graf_Scatter.png")
    cad = c + '<p>Diagrama Pandas y Lineas</p>\n' + '<p><img src="/static/panel/graf_Scatter.png" /></p>\n' 
    return cad

#Se usa en PlotEncircling
def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

def gra_PlotEncircling(c):
    sns.set_style("white")
    # Step 1: Prepare Data
    midwest = pd.read_csv("./panel/static/panel/midwest_filter.csv")
    # As many colors as there are unique midwest['category']
    categories = np.unique(midwest['category'])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))] 
    # Step 2: Draw Scatterplot with unique color for each category
    fig = plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')    
    for i, category in enumerate(categories):
        plt.scatter('area', 'poptotal', data=midwest.loc[midwest.category==category, :], s='dot_size', c=colors[i], label=str(category), edgecolors='black', linewidths=.5)

    # Step 3: Encircling
    # https://stackoverflow.com/questions/44575681/how-do-i-encircle-different-data-sets-in-scatter-plot
    # Select data to be encircled
    midwest_encircle_data = midwest.loc[midwest.state=='IN', :]                         
    # Draw polygon surrounding vertices    
    encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal, ec="k", fc="gold", alpha=0.1)
    encircle(midwest_encircle_data.area, midwest_encircle_data.poptotal, ec="firebrick", fc="none", linewidth=1.5)
    # Step 4: Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),xlabel='Area', ylabel='Population')
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Bubble Plot with Encircling", fontsize=22)
    plt.legend(fontsize=12)     
    plt.savefig("./panel/static/panel/graf_PlotEncircling.png")
    cad = c + '<p>Diagrama Plot Encircling</p>\n' + '<p><img src="/static/panel/graf_PlotEncircling.png" /></p>\n' 
    return cad

def gra_PlotRegression(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df_select = df.loc[df.cyl.isin([4,8]), :]
    # Plot
    sns.set_style("white")
    gridobj = sns.lmplot(x="displ", y="hwy", hue="cyl", data=df_select, 
                     height=7, aspect=1.6, robust=True, palette='tab10', 
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
    # Decorations
    gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
    plt.title("Scatterplot with line of best fit grouped by number of cylinders", fontsize=20)
    plt.savefig("./panel/static/panel/graf_PlotRegression.png")
    cad = c + '<p>Diagrama Plot con Regresion Lineal</p>\n' + '<p><img src="/static/panel/graf_PlotRegression.png" /></p>\n' 
    return cad

def gra_PlotRegressionCol(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df_select = df.loc[df.cyl.isin([4,8]), :]
    # Plot
    sns.set_style("white")
    gridobj = sns.lmplot(x="displ", y="hwy", 
                     data=df_select, 
                     height=7, 
                     robust=True, 
                     palette='Set1', 
                     col="cyl",
                     scatter_kws=dict(s=60, linewidths=.7, edgecolors='black'))
    # Decorations
    gridobj.set(xlim=(0.5, 7.5), ylim=(0, 50))
    plt.savefig("./panel/static/panel/graf_PlotRegressionCol.png")
    cad = c + '<p>Diagrama Plot con Regresion en Columna</p>\n' + '<p><img src="/static/panel/graf_PlotRegressionCol.png" /></p>\n' 
    return cad

def gra_Jittering(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Draw Stripplot
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
    sns.stripplot(data=df, x=df.cty, y=df.hwy, jitter=0.8, size=8, ax=ax, linewidth=.5)
    # Decorations
    plt.title('Use jittered plots to avoid overlapping of points', fontsize=22)
    plt.savefig("./panel/static/panel/graf_Jittering.png")
    cad = c + '<p>Diagrama Jittering</p>\n' + '<p><img src="/static/panel/graf_Jittering.png" /></p>\n' 
    return cad

def gra_CountPlot(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df_counts = df.groupby(['hwy', 'cty']).size().reset_index(name='counts')
    # Draw Stripplot
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
    #sns.stripplot(data=df_counts, x=df_counts.cty, y=df_counts.hwy, size=df_counts.counts*2, ax=ax)
    sns.stripplot(data=df_counts, x=df_counts.cty, y=df_counts.hwy, ax=ax)
    # Decorations
    plt.title('Counts Plot - Size of circle is bigger as more points overlap', fontsize=22)
    plt.savefig("./panel/static/panel/graf_CountPlot.png")
    cad = c + '<p>Diagrama Plot Ver cambio diametro</p>\n' + '<p><img src="/static/panel/graf_CountPlot.png" /></p>\n' 
    return cad

def gra_HistogramaM(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Create Fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    # Scatterplot on main ax
    ax_main.scatter('displ', 'hwy', s=df.cty*4, c=df.manufacturer.astype('category').cat.codes, alpha=.9, data=df, cmap="tab10", edgecolors='gray', linewidths=.5)
    # histogram on the right
    ax_bottom.hist(df.displ, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
    ax_bottom.invert_yaxis()
    # histogram in the bottom
    ax_right.hist(df.hwy, 40, histtype='stepfilled', orientation='horizontal', color='deeppink')
    # Decorations
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)
    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.savefig("./panel/static/panel/graf_HistogramaM.png")
    cad = c + '<p>Histograma Marginal</p>\n' + '<p><img src="/static/panel/graf_HistogramaM.png" /></p>\n' 
    return cad

def gra_BoxplotM(c):
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Create Fig and gridspec
    fig = plt.figure(figsize=(16, 10), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
    # Scatterplot on main ax
    ax_main.scatter('displ', 'hwy', s=df.cty*5, c=df.manufacturer.astype('category').cat.codes, alpha=.9, data=df, cmap="Set1", edgecolors='black', linewidths=.5)
    # Add a graph in each part
    sns.boxplot(df.hwy, ax=ax_right, orient="v")
    sns.boxplot(df.displ, ax=ax_bottom, orient="h")
    # Decorations ------------------
    # Remove x axis name for the boxplot
    ax_bottom.set(xlabel='')
    ax_right.set(ylabel='')
    # Main Title, Xlabel and YLabel
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    # Set font size of different components
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)
    plt.savefig("./panel/static/panel/graf_BoxplotM.png")
    cad = c + '<p>Botplot Marginal</p>\n' + '<p><img src="/static/panel/graf_BoxplotM.png" /></p>\n' 
    return cad

def gra_Correlograma(c):
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    # Plot
    plt.figure(figsize=(12,10), dpi= 80)
    #df_heatmap = df.pivot("STATION", "TIME", "HOURLY_TOTAL_TRAFFIC")
    dtype_df = df.dtypes
    float_cols = dtype_df.iloc[(dtype_df=='float64').values].index
    df_f = df[float_cols].corr()
    sns.heatmap(data=df_f, xticklabels=df_f.corr().columns, yticklabels=df_f.corr().columns, cmap='RdYlGn', center=0, annot=True)
    # Decorations
    plt.title('Correlogram of mtcars', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("./panel/static/panel/graf_Correlograma.png")
    cad = c + '<p>Correlograma</p>\n' + '<p><img src="/static/panel/graf_Correlograma.png" /></p>\n' 
    return cad

def gra_Pairwise(c):
    ssl._create_default_https_context = ssl._create_unverified_context
    df = sns.load_dataset('iris')
    # Plot
    plt.figure(figsize=(10,8), dpi= 80)
    sns.pairplot(df, kind="scatter", hue="species", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.show()
    plt.savefig("./panel/static/panel/graf_Pairwise.png")
    cad = c + '<p>Correlograma</p>\n' + '<p><img src="/static/panel/graf_Pairwise.png" /></p>\n' 
    return cad
