import matplotlib.pyplot as plt
#import numpy as np
import seaborn as sns
from pydataset import data
import warnings;
#Ignorar warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import pandas as pd

# parametros esteticos de seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

# importando dataset
titanic = data("titanic")
print(titanic)
# ver primeros 10 registros
#print(titanic.head(10))

print("Pasajeros por clase")
perclase = pd.Series(titanic["class"]).value_counts()
print(perclase)

print("Frecuencia pasajeros por clase")
frepercla = 100 * perclase / len(titanic)
print(frepercla)

print("Sexo de pasajeros")
persex = pd.Series(titanic["sex"]).value_counts()
print(persex)

print("Frecuencia de sexo de pasajeros")
fresex = 100 * persex / len(titanic)
print(fresex)

print("Sexos por clase")
sexclase = titanic.groupby(['class','sex']).size().reset_index(name='counts')
print(sexclase)

print("Frecuencia de sexo por clase")
fresexclase = sexclase.assign(Frecuencia=(100 * sexclase["counts"] / len(titanic)))
print(fresexclase)

print("Trabajar para los supervivientes")

titanicS = titanic.copy()
titanicS.drop(titanicS[titanicS['survived'] == "no"].index, inplace=True)
print(titanicS)

print("Supervivientes por clase")
supcla = titanicS.groupby(['class']).size().reset_index(name='counts')
print(supcla)

print("Frecuencia de supervivientes por clase")
fresupclase = supcla.assign(Frecuencia=(100 * supcla["counts"] / len(titanicS)))
print(fresupclase)

print("Supervivientes por sexo")
supsex = titanicS.groupby(['sex']).size().reset_index(name='counts')
print(supsex)

print("Frecuencia de supervivientes por sexo")
fresupsex = supsex.assign(Frecuencia=(100 * supsex["counts"] / len(titanicS)))
print(fresupsex)

print("Supervivientes por clase y sexo")
supsexcla = titanicS.groupby(['class','sex']).size().reset_index(name='counts')
print(supsexcla)

print("Frecuencia de supervivientes por clase y sexo")
fresupsexcla = supsexcla.assign(Frecuencia=(100 * supsexcla["counts"] / len(titanicS)))
print(fresupsexcla)

# Gráfico de barras de pasajeros del Titanic
#plot = titanic["class"].value_counts().plot(kind="bar",title="Pasajeros del Titanic")
#plt.show()

# gráfico de barras de frecuencias relativas.
#plot = (100 * titanic["class"].value_counts() / len(titanic["class"])).plot(kind="bar", title="Pasajeros del Titanic %")
#plt.show()

print("grafica de pastel con porcentajes y pasajeros")

val = titanic['class'].value_counts()
valS = titanicS['class'].value_counts() 
#print(val)
plot = val.plot(kind="pie", autopct=lambda p:f"{p:.2f}%\n {p*sum(val)/100 :.0f} pasajeros", figsize=(6, 6),title="Pasajeros del Titanic")
plt.show()
plot = valS.plot(kind="pie", autopct=lambda p:f"{p:.2f}%\n {p*sum(valS)/100 :.0f} pasajeros", figsize=(6, 6),title="Sobrevivientes del Titanic")
plt.show()

