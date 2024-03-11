import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import warnings;
from scipy.spatial import ConvexHull
import ssl

#Ignorar warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import pandas as pd
import seaborn as sns

#Para extraccion de datos con certificado ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    #Limpiar
    ax_main.set_xticks(ax_main.get_xticks())
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
    df = sns.load_dataset('iris')
    # Plot
    plt.figure(figsize=(10,8), dpi= 80)
    sns.pairplot(df, kind="scatter", hue="species", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
    plt.savefig("./panel/static/panel/graf_Pairwise.png")
    cad = c + '<p>Pairwise Plot</p>\n' + '<p><img src="/static/panel/graf_Pairwise.png" /></p>\n' 
    return cad

def gra_DiverginB(c):
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    x = df.loc[:, ['mpg']]
    df['mpg_z'] = (x - x.mean())/x.std()
    df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]
    df.sort_values('mpg_z', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    plt.figure(figsize=(14,10), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors, alpha=0.4, linewidth=5)
    # Decorations
    plt.gca().set(ylabel='$Model$', xlabel='$Mileage$')
    plt.yticks(df.index, df.cars, fontsize=12)
    plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig("./panel/static/panel/graf_DiverginB.png")
    cad = c + '<p>Diverging Bars</p>\n' + '<p><img src="/static/panel/graf_DiverginB.png" /></p>\n' 
    return cad

def gra_DiverginT(c):
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    x = df.loc[:, ['mpg']]
    df['mpg_z'] = (x - x.mean())/x.std()
    df['colors'] = ['red' if x < 0 else 'green' for x in df['mpg_z']]
    df.sort_values('mpg_z', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    plt.figure(figsize=(14,14), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z)
    for x, y, tex in zip(df.mpg_z, df.index, df.mpg_z):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':14})
    # Decorations    
    plt.yticks(df.index, df.cars, fontsize=12)
    plt.title('Diverging Text Bars of Car Mileage', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.savefig("./panel/static/panel/graf_DiverginT.png")
    cad = c + '<p>Diverging Text</p>\n' + '<p><img src="/static/panel/graf_DiverginT.png" /></p>\n' 
    return cad

def gra_DiverginP(c):
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    x = df.loc[:, ['mpg']]
    df['mpg_z'] = (x - x.mean())/x.std()
    df['colors'] = ['red' if x < 0 else 'darkgreen' for x in df['mpg_z']]
    df.sort_values('mpg_z', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    plt.figure(figsize=(14,16), dpi= 80)
    plt.scatter(df.mpg_z, df.index, s=450, alpha=.6, color=df.colors)
    for x, y, tex in zip(df.mpg_z, df.index, df.mpg_z):
        t = plt.text(x, y, round(tex, 1), horizontalalignment='center', 
                 verticalalignment='center', fontdict={'color':'white'})
    # Decorations
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.yticks(df.index, df.cars)
    plt.title('Diverging Dotplot of Car Mileage', fontdict={'size':20})
    plt.xlabel('$Mileage$')
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.savefig("./panel/static/panel/graf_DiverginP.png")
    cad = c + '<p>Diverging Dot Plot</p>\n' + '<p><img src="/static/panel/graf_DiverginP.png" /></p>\n' 
    return cad

def gra_DiverginL(c):
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    x = df.loc[:, ['mpg']]
    df['mpg_z'] = (x - x.mean())/x.std()
    df['colors'] = 'black'
    # color fiat differently
    df.loc[df.cars == 'Fiat X1-9', 'colors'] = 'darkorange'
    df.sort_values('mpg_z', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    import matplotlib.patches as patches
    plt.figure(figsize=(14,16), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors, alpha=0.4, linewidth=1)
    plt.scatter(df.mpg_z, df.index, color=df.colors, s=[600 if x == 'Fiat X1-9' else 300 for x in df.cars], alpha=0.6)
    plt.yticks(df.index, df.cars)
    plt.xticks(fontsize=12)
    # Annotate
    plt.annotate('Mercedes Models', xy=(0.0, 11.0), xytext=(1.0, 11), xycoords='data', 
            fontsize=15, ha='center', va='center',
            bbox=dict(boxstyle='square', fc='firebrick'),
            arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1.5', lw=2.0, color='steelblue'), color='white')
    # Add Patches
    p1 = patches.Rectangle((-2.0, -1), width=.3, height=3, alpha=.2, facecolor='red')
    p2 = patches.Rectangle((1.5, 27), width=.8, height=5, alpha=.2, facecolor='green')
    plt.gca().add_patch(p1)
    plt.gca().add_patch(p2)
    # Decorate
    plt.title('Diverging Bars of Car Mileage', fontdict={'size':20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig("./panel/static/panel/graf_DiverginL.png")
    cad = c + '<p>Diverging Lollipop Chart</p>\n' + '<p><img src="/static/panel/graf_DiverginL.png" /></p>\n' 
    return cad

def gra_Areachart(c):
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv", parse_dates=['date']).head(100)
    x = np.arange(df.shape[0])
    y_returns = (df.psavert.diff().fillna(0)/df.psavert.shift(1)).fillna(0) * 100
    # Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] >= 0, facecolor='green', interpolate=True, alpha=0.7)
    plt.fill_between(x[1:], y_returns[1:], 0, where=y_returns[1:] <= 0, facecolor='red', interpolate=True, alpha=0.7)
    # Annotate
    plt.annotate('Peak \n1975', xy=(94.0, 21.0), xytext=(88.0, 28),
             bbox=dict(boxstyle='square', fc='firebrick'),
             arrowprops=dict(facecolor='steelblue', shrink=0.05), fontsize=15, color='white')
    # Decorations
    xtickvals = [str(m)[:3].upper()+"-"+str(y) for y,m in zip(df.date.dt.year, df.date.dt.month_name())]
    plt.gca().set_xticks(x[::6])
    plt.gca().set_xticklabels(xtickvals[::6], rotation=90, fontdict={'horizontalalignment': 'center', 'verticalalignment': 'center_baseline'})
    plt.ylim(-35,35)
    plt.xlim(1,100)
    plt.title("Month Economics Return %", fontsize=22)
    plt.ylabel('Monthly returns %')
    plt.grid(alpha=0.5)
    plt.savefig("./panel/static/panel/graf_Areachart.png")
    cad = c + '<p>Area Chart</p>\n' + '<p><img src="/static/panel/graf_Areachart.png" /></p>\n' 
    return cad

def gra_OrderedB(c):
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
    df.sort_values('cty', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
    ax.vlines(x=df.index, ymin=0, ymax=df.cty, color='firebrick', alpha=0.7, linewidth=20)
    # Annotate Text
    for i, cty in enumerate(df.cty):
        ax.text(i, cty+0.5, round(cty, 1), horizontalalignment='center')
    # Title, Label, Ticks and Ylim
    ax.set_title('Bar Chart for Highway Mileage', fontdict={'size':22})
    ax.set(ylabel='Miles Per Gallon', ylim=(0, 30))
    plt.xticks(df.index, df.manufacturer.str.upper(), rotation=60, horizontalalignment='right', fontsize=12)
    # Add patches to color the X axis labels
    p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
    p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
    fig.add_artist(p1)
    fig.add_artist(p2)
    plt.savefig("./panel/static/panel/graf_OrderedB.png")
    cad = c + '<p>Area Chart</p>\n' + '<p><img src="/static/panel/graf_OrderedB.png" /></p>\n' 
    return cad

def gra_Lollipop(c):
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
    df.sort_values('cty', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    ax.vlines(x=df.index, ymin=0, ymax=df.cty, color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=df.index, y=df.cty, s=75, color='firebrick', alpha=0.7)
    # Title, Label, Ticks and Ylim
    ax.set_title('Lollipop Chart for Highway Mileage', fontdict={'size':22})
    ax.set_ylabel('Miles Per Gallon')
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.manufacturer.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
    ax.set_ylim(0, 30)
    # Annotate
    for row in df.itertuples():
        ax.text(row.Index, row.cty+.5, s=round(row.cty, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)
    plt.savefig("./panel/static/panel/graf_Lollipop.png")
    cad = c + '<p>Lollipop Chart</p>\n' + '<p><img src="/static/panel/graf_Lollipop.png" /></p>\n' 
    return cad

def gra_DotPlot(c):
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
    df.sort_values('cty', inplace=True)
    df.reset_index(inplace=True)
    # Draw plot
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    ax.hlines(y=df.index, xmin=11, xmax=26, color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
    ax.scatter(y=df.index, x=df.cty, s=75, color='firebrick', alpha=0.7)
    # Title, Label, Ticks and Ylim
    ax.set_title('Dot Plot for Highway Mileage', fontdict={'size':22})
    ax.set_xlabel('Miles Per Gallon')
    ax.set_yticks(df.index)
    ax.set_yticklabels(df.manufacturer.str.title(), fontdict={'horizontalalignment': 'right'})
    ax.set_xlim(10, 27)
    plt.savefig("./panel/static/panel/graf_DotPlot.png")
    cad = c + '<p>Dot Plot</p>\n' + '<p><img src="/static/panel/graf_DotPlot.png" /></p>\n' 
    return cad

def gra_Slopechar(c):
    import matplotlib.lines as mlines
    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/gdppercap.csv")
    left_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1952'])]
    right_label = [str(c) + ', '+ str(round(y)) for c, y in zip(df.continent, df['1957'])]
    klass = ['red' if (y1-y2) < 0 else 'green' for y1, y2 in zip(df['1952'], df['1957'])]
    # draw line
    # https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
    def newline(p1, p2, color='black'):
        ax = plt.gca()
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='red' if p1[1]-p2[1] > 0 else 'green', marker='o', markersize=6)
        ax.add_line(l)
        return l
    fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)
    # Vertical Lines
    ax.vlines(x=1, ymin=500, ymax=13000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    ax.vlines(x=3, ymin=500, ymax=13000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    # Points
    ax.scatter(y=df['1952'], x=np.repeat(1, df.shape[0]), s=10, color='black', alpha=0.7)
    ax.scatter(y=df['1957'], x=np.repeat(3, df.shape[0]), s=10, color='black', alpha=0.7)
    # Line Segmentsand Annotation
    for p1, p2, c in zip(df['1952'], df['1957'], df['continent']):
        newline([1,p1], [3,p2])
        ax.text(1-0.05, p1, c + ', ' + str(round(p1)), horizontalalignment='right', verticalalignment='center', fontdict={'size':14})
        ax.text(3+0.05, p2, c + ', ' + str(round(p2)), horizontalalignment='left', verticalalignment='center', fontdict={'size':14})
    # 'Before' and 'After' Annotations
    ax.text(1-0.05, 13000, 'BEFORE', horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
    ax.text(3+0.05, 13000, 'AFTER', horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})
    # Decoration
    ax.set_title("Slopechart: Comparing GDP Per Capita between 1952 vs 1957", fontdict={'size':22})
    ax.set(xlim=(0,4), ylim=(0,14000), ylabel='Mean GDP Per Capita')
    ax.set_xticks([1,3])
    ax.set_xticklabels(["1952", "1957"])
    plt.yticks(np.arange(500, 13000, 2000), fontsize=12)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.0)
    plt.savefig("./panel/static/panel/graf_DotPlot.png")
    cad = c + '<p>Slope Chart</p>\n' + '<p><img src="/static/panel/graf_DotPlot.png" /></p>\n' 
    return cad

def gra_DumbbellPlot(c):
    import matplotlib.lines as mlines
    # Import Data
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/health.csv")
    df.sort_values('pct_2014', inplace=True)
    df.reset_index(inplace=True)
    # Func to draw line segment
    def newline(p1, p2, color='black'):
        ax = plt.gca()
        l = mlines.Line2D([p1[0],p2[0]], [p1[1],p2[1]], color='skyblue')
        ax.add_line(l)
        return l
    # Figure and Axes
    fig, ax = plt.subplots(1,1,figsize=(14,14), facecolor='#f7f7f7', dpi= 80)
    # Vertical Lines
    ax.vlines(x=.05, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.10, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.15, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    ax.vlines(x=.20, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')
    # Points
    ax.scatter(y=df['index'], x=df['pct_2013'], s=50, color='#0e668b', alpha=0.7)
    ax.scatter(y=df['index'], x=df['pct_2014'], s=50, color='#a3c4dc', alpha=0.7)
    # Line Segments
    for i, p1, p2 in zip(df['index'], df['pct_2013'], df['pct_2014']):
        newline([p1, i], [p2, i])
    # Decoration
    ax.set_facecolor('#f7f7f7')
    ax.set_title("Dumbell Chart: Pct Change - 2013 vs 2014", fontdict={'size':22})
    ax.set(xlim=(0,.25), ylim=(-1, 27), ylabel='Mean GDP Per Capita')
    ax.set_xticks([.05, .1, .15, .20])
    ax.set_xticklabels(['5%', '15%', '20%', '25%'])
    ax.set_xticklabels(['5%', '15%', '20%', '25%'])  
    plt.savefig("./panel/static/panel/graf_DumbbellPlot.png")
    cad = c + '<p>Dombbell Plot</p>\n' + '<p><img src="/static/panel/graf_DumbbellPlot.png" /></p>\n' 
    return cad

def gra_HistogramaCont(c):
    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    # Prepare data
    x_var = 'displ'
    groupby_var = 'class'
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]
    # Draw
    plt.figure(figsize=(16,9), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])
    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 25)
    plt.xticks(ticks=bins[::3], labels=[round(b,1) for b in bins[::3]])
    plt.savefig("./panel/static/panel/graf_HistogramaCont.png")
    cad = c + '<p>Histogram for Continuous Variable</p>\n' + '<p><img src="/static/panel/graf_HistogramaCont.png" /></p>\n' 
    return cad

def gra_HistogramaCateg(c):
    # Import Data
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
    x_var = 'manufacturer'
    groupby_var = 'class'
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]
    plt.figure(figsize=(16,9), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False, color=colors[:len(vals)])
    plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 40)
    #plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')
    plt.savefig("./panel/static/panel/graf_HistogramaCateg.png")
    cad = c + '<p>Histogram for Categorical Variable</p>\n' + '<p><img src="/static/panel/graf_HistogramaCateg.png" /></p>\n' 
    return cad

def gra_DensityP(c):
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    #Revisar warning
    sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
    sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)
    # Decoration
    plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=22)
    plt.legend()
    plt.savefig("./panel/static/panel/graf_DensityP.png")
    cad = c + '<p>Density Plot</p>\n' + '<p><img src="/static/panel/graf_DensityP.png" /></p>\n' 
    return cad

def gra_DensityCH(c):
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Draw Plot
    #plt.figure(figsize=(13,10), dpi= 80)
    #sns.histplot(data = df.loc[df["class"] == "compact"], x = "cty", color="dodgerblue", label="Compac", kde_kws={'linewidth':3})
    #sns.histplot(data = df.loc[df["class"] == "suv"], x = "cty", color="orange", label="Suv", kde_kws={'linewidth':3})
    sns.histplot(data = df.loc[df["class"] == "minivan"], x = "cty", color="green", label="Minivan", kde_kws={'linewidth':3})
    
    #plt.ylim(0, 0.35)
    # Decoration
    #plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=22)
    #plt.legend()
    plt.savefig("./panel/static/panel/graf_DensityP.png")
    cad = c + '<p>Density Curves Histograma</p>\n' + '<p><img src="/static/panel/graf_DensityP.png" /></p>\n' 
    return cad


def gra_Joyplot(c):
    import joypy
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
   # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    fig, axes = joypy.joyplot(df, column=['hwy', 'cty'], by="class", ylim='own', figsize=(14,10))
    # Decoration
    plt.title('Joy Plot of City and Highway Mileage by Class', fontsize=22)
    plt.savefig("./panel/static/panel/graf_Joyplot.png")
    cad = c + '<p>Joy plot</p>\n' + '<p><img src="/static/panel/graf_Joyplot.png" /></p>\n' 
    return cad

def gra_DotplotD(c):
    #import matplotlib.patches as mpatches
    # Prepare Data
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    cyl_colors = {4:'tab:red', 5:'tab:green', 6:'tab:blue', 8:'tab:orange'}
    df_raw['cyl_color'] = df_raw.cyl.map(cyl_colors)
    # Mean and Median city mileage by make
    df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
    df.sort_values('cty', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df_median = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.median())
    # Draw horizontal lines
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    ax.hlines(y=df.index, xmin=0, xmax=40, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')
    # Draw the Dots
    for i, make in enumerate(df.manufacturer):
        df_make = df.loc[df.manufacturer==make, :]
        ax.scatter(y=np.repeat(i, df_make.shape[0]), x='cty', data=df_make, s=75, edgecolors='gray', c='w', alpha=0.5)
        ax.scatter(y=i, x='cty', data=df_median.loc[df_median.index==make, :], s=75, c='firebrick')
    # Annotate    
    ax.text(33, 13, "$red ; dots ; are ; the : median$", fontdict={'size':12}, color='firebrick')
    # Decorations
    red_patch = plt.plot([],[], marker="o", ms=10, ls="", mec=None, color='firebrick', label="Median")
    plt.legend(handles=red_patch)
    ax.set_title('Distribution of City Mileage by Make', fontdict={'size':22})
    ax.set_xlabel('Miles Per Gallon (City)', alpha=0.7)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df.manufacturer.str.title(), fontdict={'horizontalalignment': 'right'}, alpha=0.7)
    ax.set_xlim(1, 40)
    plt.xticks(alpha=0.7)
    plt.gca().spines["top"].set_visible(False)    
    plt.gca().spines["bottom"].set_visible(False)    
    plt.gca().spines["right"].set_visible(False)    
    plt.gca().spines["left"].set_visible(False)   
    plt.grid(axis='both', alpha=.4, linewidth=.1)
    plt.savefig("./panel/static/panel/graf_DotplotD.png")
    cad = c + '<p>Joy plot</p>\n' + '<p><img src="/static/panel/graf_DotplotD.png" /></p>\n' 
    return cad

def gra_Boxplot(c):
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    #Ver classes
    #df['class'].unique()
    # Draw Plot
    plt.figure(figsize=(13,10), dpi= 80)
    my_pal = {"compact" : "green", "midsize" : "orange", "suv" : "m", "2seater" : "red", "minivan" : "blue", "pickup" : "black","subcompact" : "gold"}
    sns.boxplot(x='class', y='hwy', data=df, notch=False, palette=my_pal)
    # Add N Obs inside boxplot (optional)
    def add_n_obs(df,group_col,y):
        medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}
        xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
        n_obs = df.groupby(group_col)[y].size().values
        for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
            plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')

    add_n_obs(df,group_col='class',y='hwy')    
    # Decoration
    plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
    plt.ylim(10, 40)
    plt.savefig("./panel/static/panel/graf_Boxplot.png")
    cad = c + '<p>Box plot</p>\n' + '<p><img src="/static/panel/graf_Boxplot.png" /></p>\n' 
    return cad

def gra_DotyBoxplot(c):
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Draw Plot
    plt.figure(figsize=(13,10), dpi= 80)
    sns.boxplot(x='class', y='hwy', data=df, hue='cyl')
    sns.stripplot(x='class', y='hwy', data=df, color='black', size=3, jitter=1)
    for i in range(len(df['class'].unique())-1):
        plt.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)
    # Decoration
    plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
    plt.legend(title='Cylinders')
    plt.savefig("./panel/static/panel/graf_DotBoxplot.png")
    cad = c + '<p>Dot y Box plot</p>\n' + '<p><img src="/static/panel/graf_DotBoxplot.png" /></p>\n' 
    return cad

def gra_Violinplot(c):
    # Import Data
    df = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Draw Plot
    plt.figure(figsize=(13,10), dpi= 80)
    sns.violinplot(x='class', y='hwy', data=df, scale='width', inner='quartile')
    # Decoration
    plt.title('Violin Plot of Highway Mileage by Vehicle Class', fontsize=22)
    plt.savefig("./panel/static/panel/graf_DotBoxplot.png")
    cad = c + '<p>Violin plot</p>\n' + '<p><img src="/static/panel/graf_DotBoxplot.png" /></p>\n' 
    return cad

def gra_Piramid(c):
    df = pd.read_csv("./panel/static/panel/email_campaign_funnel.csv")
    # Draw Plot
    plt.figure(figsize=(13,10), dpi= 80)
    group_col = 'Gender'
    order_of_bars = df.Stage.unique()[::-1]
    colors = [plt.cm.Spectral(i/float(len(df[group_col].unique())-1)) for i in range(len(df[group_col].unique()))]
    sns.barplot(x="Users", y="Stage", data=df, order=order_of_bars, label="Stage")
    #for c, group in zip(colors, df[group_col].unique()):
    #    sns.barplot(x="Users", y="Stage", data=df, order=order_of_bars, color=c, label=group)
        #sns.barplot(x='Users', y='Stage', data=df.loc[df[group_col]==group, :], order=order_of_bars, color=c, label=group)
    # Decorations    
    plt.xlabel("$Users$")
    plt.ylabel("Stage of Purchase")
    plt.yticks(fontsize=12)
    plt.title("Population Pyramid of the Marketing Funnel", fontsize=22)
    plt.legend()
    plt.savefig("./panel/static/panel/graf_Piramid.png")
    cad = c + '<p>Piramid population</p>\n' + '<p><img src="/static/panel/graf_Piramid.png" /></p>\n' 
    return cad

def gra_PlotCategorical(c):
    # Load Dataset
    titanic = sns.load_dataset("titanic")
    dat = data=titanic[titanic.deck.notnull()]
    # Plot
    g = sns.catplot("alive", col="deck", col_wrap=4,data=dat,kind="count", height=3.5, aspect=.8, palette='tab20')
    fig.suptitle('sf')
    plt.savefig("./panel/static/panel/graf_PlotCategorical.png")
    cad = c + '<p>Piramid population</p>\n' + '<p><img src="/static/panel/graf_PlotCategorical.png" /></p>\n' 
    return cad

def gra_Pie(c):
    # Import
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Prepare Data
    df = df_raw.groupby(['class']).count()
    values = df['cty']
    explode = (0,0,0,0,0,0,0.2)
    df.plot(kind='pie', y='cty', explode=explode, shadow=True, legend=False, autopct=lambda p:f'{p:.2f}%\n {p*sum(values)/100 :.0f} pzas')
    plt.title("Pie Chart of Vehicle Class - Bad")
    plt.ylabel("")
    plt.savefig("./panel/static/panel/graf_Pie.png")
    cad = c + '<p>Pie chart</p>\n' + '<p><img src="/static/panel/graf_Pie.png" /></p>\n' 
    return cad

def gra_PieE(c):
    # Import
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Prepare Data
    df = df_raw.groupby('class').size().reset_index(name='counts')
    # Draw Plot
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)
    data = df['counts']
    categories = df['class']
    explode = [0,0,0,0,0,0.1,0]
    
    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)
    wedges, texts, autotexts = ax.pie(data, 
                                  autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"), 
                                  colors=plt.cm.Dark2.colors,
                                 startangle=140,
                                 explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="Vehicle Class", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Class of Vehicles: Pie Chart")
    plt.savefig("./panel/static/panel/graf_PieE.png")
    cad = c + '<p>Pie E chart</p>\n' + '<p><img src="/static/panel/graf_PieE.png" /></p>\n' 
    return cad

def gra_Treemap(c):
    import squarify 
    # Import Data
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Prepare Data
    df = df_raw.groupby('class').size().reset_index(name='counts')
    labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
    sizes = df['counts'].values.tolist()
    colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]
    # Draw Plot
    plt.figure(figsize=(12,8), dpi= 80)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)
    # Decorate
    plt.title('Treemap of Vechile Class')
    plt.axis('off')
    plt.savefig("./panel/static/panel/graf_Treemap.png")
    cad = c + '<p>Treemap chart</p>\n' + '<p><img src="/static/panel/graf_Treemap.png" /></p>\n' 
    return cad

def gra_Barchart(c):
    import random
    # Import Data
    df_raw = pd.read_csv("./panel/static/panel/mpg_ggplot2.csv")
    # Prepare Data
    df = df_raw.groupby('manufacturer').size().reset_index(name='counts')
    n = df['manufacturer'].unique().__len__()+1
    all_colors = list(plt.cm.colors.cnames.keys())
    random.seed(100)
    cl = random.choices(all_colors, k=n)
    # Plot Bars
    plt.figure(figsize=(16,10), dpi= 80)
    plt.bar(df['manufacturer'], df['counts'], color=cl, width=.5)
    for i, val in enumerate(df['counts'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})
    # Decoration
    plt.gca().set_xticklabels(df['manufacturer'], rotation=60, horizontalalignment= 'right')
    plt.title("Number of Vehicles by Manaufacturers", fontsize=22)
    plt.ylabel('# Vehicles')
    plt.ylim(0, 45)
    plt.savefig("./panel/static/panel/graf_Barchart.png")
    cad = c + '<p>Bar chart</p>\n' + '<p><img src="/static/panel/graf_Barchart.png" /></p>\n' 
    return cad

def gra_Timeplot(c):
    # Import Data
    df = pd.read_csv('./panel/static/panel/AirPassengers.csv')
    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot('date', 'value', data=df, color='tab:red')
    # Decoration
    plt.ylim(50, 750)
    xtick_location = df.index.tolist()[::12]
    xtick_labels = [x[-4:] for x in df.date.tolist()[::12]]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.grid(axis='both', alpha=.3)
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.3)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.3)   
    plt.savefig("./panel/static/panel/graf_Timeplot.png")
    cad = c + '<p>Time plot</p>\n' + '<p><img src="/static/panel/graf_Timeplot.png" /></p>\n' 
    return cad

def gra_TimeSeries(c):
    # Import Data
    df = pd.read_csv('./panel/static/panel/AirPassengers.csv')
    # Get the Peaks and Troughs
    data = df['value'].values
    doublediff = np.diff(np.sign(np.diff(data)))
    peak_locations = np.where(doublediff == -2)[0] + 1
    doublediff2 = np.diff(np.sign(np.diff(-1*data)))
    trough_locations = np.where(doublediff2 == -2)[0] + 1
    # Draw Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.plot('date', 'value', data=df, color='tab:blue', label='Air Traffic')
    plt.scatter(df.date[peak_locations], df.value[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
    plt.scatter(df.date[trough_locations], df.value[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')
    # Annotate
    for t, p in zip(trough_locations[1::5], peak_locations[::3]):
        plt.text(df.date[p], df.value[p]+15, df.date[p], horizontalalignment='center', color='darkgreen')
        plt.text(df.date[t], df.value[t]-35, df.date[t], horizontalalignment='center', color='darkred')

    # Decoration
    plt.ylim(50,750)
    xtick_location = df.index.tolist()[::6]
    xtick_labels = df.date.tolist()[::6]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
    plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=.3)  
    plt.savefig("./panel/static/panel/graf_TimeSeries.png")
    cad = c + '<p>Time series</p>\n' + '<p><img src="/static/panel/graf_TimeSeries.png" /></p>\n' 
    return cad

def gra_Autocorrelation(c):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # Import Data
    df = pd.read_csv('./panel/static/panel/AirPassengers.csv')
    # Draw Plot
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
    plot_acf(df.value.tolist(), ax=ax1, lags=50)
    plot_pacf(df.value.tolist(), ax=ax2, lags=20)
    # Decorate
    # lighten the borders
    ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)
    ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)
    ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)
    ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)
    # font size of tick labels
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    plt.savefig("./panel/static/panel/graf_Autocorrelation.png")
    cad = c + '<p>Autocorrelation</p>\n' + '<p><img src="/static/panel/graf_Autocorrelation.png" /></p>\n' 
    return cad

def gra_CrossCorrelation(c):
    import statsmodels.tsa.stattools as stattools
    # Import Data
    df = pd.read_csv('./panel/static/panel/mortality.csv')
    x = df['mdeaths']
    y = df['fdeaths']
    # Compute Cross Correlations
    ccs = stattools.ccf(x, y)[:100]
    nlags = len(ccs)
    # Compute the Significance level
    # ref: https://stats.stackexchange.com/questions/3115/cross-correlation-significance-in-r/3128#3128
    conf_level = 2 / np.sqrt(nlags)
    # Draw Plot
    plt.figure(figsize=(12,7), dpi= 80)
    plt.hlines(0, xmin=0, xmax=100, color='gray')  # 0 axis
    plt.hlines(conf_level, xmin=0, xmax=100, color='gray')
    plt.hlines(-conf_level, xmin=0, xmax=100, color='gray')
    plt.bar(x=np.arange(len(ccs)), height=ccs, width=.3)
    # Decoration
    plt.title('$Cross ; Correlation ; Plot: ; mdeaths ; vs ; fdeaths$', fontsize=22)
    plt.xlim(0,len(ccs))
    plt.savefig("./panel/static/panel/graf_CrossCorrelation.png")
    cad = c + '<p>Cross Correlation</p>\n' + '<p><img src="/static/panel/graf_CrossCorrelation.png" /></p>\n' 
    return cad

def gra_TimeDecomposition(c):
    from statsmodels.tsa.seasonal import seasonal_decompose
    from dateutil.parser import parse
    # Import Data
    df = pd.read_csv('./panel/static/panel/AirPassengers.csv')
    dates = pd.DatetimeIndex([parse(d).strftime('%Y-%m-01') for d in df['date']])
    df.set_index(dates, inplace=True)
    # Decompose 
    result = seasonal_decompose(df['value'], model='multiplicative')
    # Plot
    plt.rcParams.update({'figure.figsize': (10,10)})
    result.plot().suptitle('Time Series Decomposition of Air Passengers')
    plt.savefig("./panel/static/panel/graf_TimeDecomposition.png")
    cad = c + '<p>Time Decomposition</p>\n' + '<p><img src="/static/panel/graf_TimeDecomposition.png" /></p>\n' 
    return cad

def gra_MultipleTimeSeries(c):
    df = pd.read_csv('./panel/static/panel/mortality.csv')
    # Define the upper limit, lower limit, interval of Y axis and colors
    y_LL = 100
    y_UL = int(df.iloc[:, 1:].max().max()*1.1)
    y_interval = 400
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']    
    # Draw Plot and Annotate
    fig, ax = plt.subplots(1,1,figsize=(16, 9), dpi= 80)    
    columns = df.columns[1:]  
    for i, column in enumerate(columns):    
        plt.plot(df.date.values, df[column].values, lw=1.5, color=mycolors[i])    
        plt.text(df.shape[0]+1, df[column].values[-1], column, fontsize=14, color=mycolors[i])
    # Draw Tick lines  
    for y in range(y_LL, y_UL, y_interval):    
        plt.hlines(y, xmin=0, xmax=71, colors='black', alpha=0.3, linestyles="--", lw=0.5)
    # Decorations    
    plt.tick_params(axis="both", which="both", bottom=False, top=False,    
                labelbottom=True, left=False, right=False, labelleft=True)        
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.title('Number of Deaths from Lung Diseases in the UK (1974-1979)', fontsize=22)
    plt.yticks(range(y_LL, y_UL, y_interval), [str(y) for y in range(y_LL, y_UL, y_interval)], fontsize=12)    
    plt.xticks(range(0, df.shape[0], 12), df.date.values[::12], horizontalalignment='left', fontsize=12)    
    plt.ylim(y_LL, y_UL)    
    plt.xlim(-2, 80) 
    plt.savefig("./panel/static/panel/graf_MultipleTimeSeries.png")
    cad = c + '<p>Multiple Time Series</p>\n' + '<p><img src="/static/panel/graf_MultipleTimeSeries.png" /></p>\n' 
    return cad

def gra_DifferentScales(c):
    df = pd.read_csv("./panel/static/panel/economics.csv")
    x = df['date']
    y1 = df['psavert']
    y2 = df['unemploy']
    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1,1,figsize=(16,9), dpi= 80)
    ax1.plot(x, y1, color='tab:red')
    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x, y2, color='tab:blue')
    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red' )
    ax1.grid(alpha=.4)
    # ax2 (right Y axis)
    ax2.set_ylabel("# Unemployed (1000's)", color='tab:blue', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_xticks(np.arange(0, len(x), 60))
    ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})
    ax2.set_title("Personal Savings Rate vs Unemployed: Plotting in Secondary Y Axis", fontsize=22)
    fig.tight_layout()
    plt.savefig("./panel/static/panel/graf_DifferentScales.png")
    cad = c + '<p>Different Scales</p>\n' + '<p><img src="/static/panel/graf_DifferentScales.png" /></p>\n' 
    return cad


def gra_TimeError(c):
    # Import
    from scipy.stats import sem
    # Import Data
    df = pd.read_csv("./panel/static/panel/user_orders_hourofday.csv")
    df_mean = df.groupby('order_hour_of_day').quantity.mean()
    df_se = df.groupby('order_hour_of_day').quantity.apply(sem).mul(1.96)
    # Plot
    plt.figure(figsize=(16,10), dpi= 80)
    plt.ylabel("# Orders", fontsize=16)  
    x = df_mean.index
    plt.plot(x, df_mean, color="white", lw=2) 
    plt.fill_between(x, df_mean - df_se, df_mean + df_se, color="#3F5D7D")  
    # Decorations
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)
    plt.xticks(x[::2], [str(d) for d in x[::2]] , fontsize=12)
    plt.title("User Orders by Hour of Day (95% confidence)", fontsize=22)
    plt.xlabel("Hour of Day")
    s, e = plt.gca().get_xlim()
    plt.xlim(s, e)
    # Draw Horizontal Tick lines  
    for y in range(8, 20, 2):    
        plt.hlines(y, xmin=s, xmax=e, colors='black', alpha=0.5, linestyles="--", lw=0.5)

    plt.savefig("./panel/static/panel/graf_TimeError.png")
    cad = c + '<p>Time Erros</p>\n' + '<p><img src="/static/panel/graf_TimeError.png" /></p>\n' 
    return cad

def gra_StacketArea(c):
    df = pd.read_csv('./panel/static/panel/nightvisitors.csv')
    # Decide Colors 
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']      
    # Draw Plot and Annotate
    fig, ax = plt.subplots(1,1,figsize=(16, 9), dpi= 80)
    columns = df.columns[1:]
    labs = columns.values.tolist()
    # Prepare data
    x  = df['yearmon'].values.tolist()
    y0 = df[columns[0]].values.tolist()
    y1 = df[columns[1]].values.tolist()
    y2 = df[columns[2]].values.tolist()
    y3 = df[columns[3]].values.tolist()
    y4 = df[columns[4]].values.tolist()
    y5 = df[columns[5]].values.tolist()
    y6 = df[columns[6]].values.tolist()
    y7 = df[columns[7]].values.tolist()
    y = np.vstack([y0, y2, y4, y6, y7, y5, y1, y3])
    # Plot for each column
    labs = columns.values.tolist()
    ax = plt.gca()
    ax.stackplot(x, y, labels=labs, colors=mycolors, alpha=0.8)
    # Decorations
    ax.set_title('Night Visitors in Australian Regions', fontsize=18)
    ax.set(ylim=[0, 100000])
    ax.legend(fontsize=10, ncol=4)
    plt.xticks(x[::5], fontsize=10, horizontalalignment='center')
    plt.yticks(np.arange(10000, 100000, 20000), fontsize=10)
    plt.xlim(x[0], x[-1])
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.savefig("./panel/static/panel/graf_StacketArea.png")
    cad = c + '<p>Stacket Area</p>\n' + '<p><img src="/static/panel/graf_StacketArea.png" /></p>\n' 
    return cad

def gra_UnstacketArea(c):
    df = pd.read_csv('./panel/static/panel/economics.csv')
    # Prepare Data
    x = df['date'].values.tolist()
    y1 = df['psavert'].values.tolist()
    y2 = df['uempmed'].values.tolist()
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']      
    columns = ['psavert', 'uempmed']
    # Draw Plot 
    fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi= 80)
    ax.fill_between(x, y1=y1, y2=0, label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
    ax.fill_between(x, y1=y2, y2=0, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)
    # Decorations
    ax.set_title('Personal Savings Rate vs Median Duration of Unemployment', fontsize=18)
    ax.set(ylim=[0, 30])
    ax.legend(loc='best', fontsize=12)
    plt.xticks(x[::50], fontsize=10, horizontalalignment='center')
    plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
    plt.xlim(-10, x[-1])
    # Draw Tick lines  
    for y in np.arange(2.5, 30.0, 2.5):    
        plt.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.savefig("./panel/static/panel/graf_UnstacketArea.png")
    cad = c + '<p>Unstacket Area</p>\n' + '<p><img src="/static/panel/graf_UnstacketArea.png" /></p>\n' 
    return cad

def gra_Calendar(c):
    import matplotlib as mpl
    import calmap
    # Import Data
    df = pd.read_csv("./panel/static/panel/yahoo.csv", parse_dates=['date'])
    df.set_index('date', inplace=True)
    # Plot
    plt.figure(figsize=(16,10), dpi= 80)
    calmap.calendarplot(df['2014']['VIX.Close'], fig_kws={'figsize': (16,10)}, yearlabel_kws={'color':'black', 'fontsize':14}, subplot_kws={'title':'Yahoo Stock Prices'})

    plt.savefig("./panel/static/panel/graf_Calendar.png")
    cad = c + '<p>Calendar</p>\n' + '<p><img src="/static/panel/graf_Calendar.png" /></p>\n' 
    return cad

def gra_SeasonalPlot(c):
    #from ast import parse
    # Import Data
    df = pd.read_csv('./panel/static/panel/AirPassengers.csv', parse_dates=True)
    # Prepare data
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    #df['year'] = [parse(d).year for d in df.date]
    #df['month'] = [parse(d).strftime('%b') for d in df.date]
    years = df['year'].unique()
    # Draw Plot
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive', 'deeppink', 'steelblue', 'firebrick', 'mediumseagreen']      
    plt.figure(figsize=(16,10), dpi= 80)
    for i, y in enumerate(years):
        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])
    # Decoration
    plt.ylim(50,750)
    plt.xlim(-0.3, 11)
    plt.ylabel('$Air Traffic$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Monthly Seasonal Plot: Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.grid(axis='y', alpha=.3)
    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.5)   
    # plt.legend(loc='upper right', ncol=2, fontsize=12)
    plt.savefig("./panel/static/panel/graf_SeasonalPlot.png")
    cad = c + '<p>Seasonal Plot</p>\n' + '<p><img src="/static/panel/graf_SeasonalPlot.png" /></p>\n' 
    return cad

def gra_Dendrogram(c):
    import scipy.cluster.hierarchy as shc
    # Import Data
    df = pd.read_csv('./panel/static/panel/USArrests.csv')
    # Plot
    plt.figure(figsize=(16, 10), dpi= 80)  
    plt.title("USArrests Dendograms", fontsize=22)  
    dend = shc.dendrogram(shc.linkage(df[['Murder', 'Assault', 'UrbanPop', 'Rape']], method='ward'), labels=df.State.values, color_threshold=100)  
    plt.xticks(fontsize=12)
    plt.savefig("./panel/static/panel/graf_Dendrogram.png")
    cad = c + '<p>Dendogram Plot</p>\n' + '<p><img src="/static/panel/graf_Dendrogram.png" /></p>\n' 
    return cad

def gra_Cluster(c):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial import ConvexHull
    # Import Data
    df = pd.read_csv('./panel/static/panel/USArrests.csv')
    # Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')  
    cluster.fit_predict(df[['Murder', 'Assault', 'UrbanPop', 'Rape']])  
    # Plot
    plt.figure(figsize=(14, 10), dpi= 80)  
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c=cluster.labels_, cmap='tab10')  
    # Encircle
    def encircle(x,y, ax=None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[x,y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices,:], **kw)
        ax.add_patch(poly)

    # Draw polygon surrounding vertices    
    encircle(df.loc[cluster.labels_ == 0, 'Murder'], df.loc[cluster.labels_ == 0, 'Assault'], ec="k", fc="gold", alpha=0.2, linewidth=0)
    encircle(df.loc[cluster.labels_ == 1, 'Murder'], df.loc[cluster.labels_ == 1, 'Assault'], ec="k", fc="tab:blue", alpha=0.2, linewidth=0)
    encircle(df.loc[cluster.labels_ == 2, 'Murder'], df.loc[cluster.labels_ == 2, 'Assault'], ec="k", fc="tab:red", alpha=0.2, linewidth=0)
    encircle(df.loc[cluster.labels_ == 3, 'Murder'], df.loc[cluster.labels_ == 3, 'Assault'], ec="k", fc="tab:green", alpha=0.2, linewidth=0)
    encircle(df.loc[cluster.labels_ == 4, 'Murder'], df.loc[cluster.labels_ == 4, 'Assault'], ec="k", fc="tab:orange", alpha=0.2, linewidth=0)
    # Decorations
    plt.xlabel('Murder'); plt.xticks(fontsize=12)
    plt.ylabel('Assault'); plt.yticks(fontsize=12)
    plt.title('Agglomerative Clustering of USArrests (5 Groups)', fontsize=22)

    plt.savefig("./panel/static/panel/graf_Cluster.png")
    cad = c + '<p>Cluster Plot</p>\n' + '<p><img src="/static/panel/graf_Cluster.png" /></p>\n' 
    return cad

def gra_AndreusCurve(c):
    from pandas.plotting import andrews_curves
    # Import
    df = pd.read_csv("./panel/static/panel/mtcars.csv")
    df.drop(['cars', 'carname'], axis=1, inplace=True)
    # Plot
    plt.figure(figsize=(12,9), dpi= 80)
    andrews_curves(df, 'cyl', colormap='Set1')
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.title('Andrews Curves of mtcars', fontsize=22)
    plt.xlim(-3,3)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("./panel/static/panel/graf_AndreusCurve.png")
    cad = c + '<p>Andreus Curve</p>\n' + '<p><img src="/static/panel/graf_AndreusCurve.png" /></p>\n' 
    return cad

def gra_ParallelCoordinates(c):
    from pandas.plotting import parallel_coordinates
    # Import Data
    df_final = pd.read_csv("./panel/static/panel/diamonds_filter.csv")
    # Plot
    plt.figure(figsize=(12,9), dpi= 80)
    parallel_coordinates(df_final, 'cut', colormap='Dark2')
    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.title('Parallel Coordinated of Diamonds', fontsize=22)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("./panel/static/panel/graf_ParallelCoordinates.png")
    cad = c + '<p>ParallelCoordinates</p>\n' + '<p><img src="/static/panel/graf_ParallelCoordinates.png" /></p>\n' 
    return cad