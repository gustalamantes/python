#import matplotlib.pyplot as plt
#figure, axes = plt.subplots(2) # para dos subgraficos

# Ahora se puede configurar un gráfico usando
# funciones disponibles para estos objetos.

# Simple Histograma
#_ = plt.hist(train_df['target'], bins=5, edgecolors='white')

import matplotlib.pyplot as plt
# Crear la figura y los ejes
fig, ax = plt.subplots()
# Dibujar puntos
ax.scatter(x = [1, 2, 3], y = [3, 2, 1])
# Guardar el gráfico en formato png
plt.savefig('diagrama-dispersion.png')
# Mostrar el gráfico
plt.show()

#print("Hola mundo!!")