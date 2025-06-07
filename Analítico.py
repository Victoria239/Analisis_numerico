import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
k = 0.0005
x0 = 50
t0 = 4
tf = 6
h = 0.01

# Definir función analítica
def solucion_analitica(t, k, x0, t0):
    A = (1000 - x0) / x0
    return 1000 / (1 + A * np.exp(-1000 * k * (t - t0)))

# Crear valores de t
t_vals = np.arange(t0, tf + h, h)
x_vals_analitica = solucion_analitica(t_vals, k, x0, t0)

# Crear tabla
tabla_analitica = pd.DataFrame({
    'Día (t)': t_vals,
    'Infectados (x)': x_vals_analitica
})

# Mostrar primeras filas
print("Tabla de resultados analíticos (primeros 20 pasos):")
print(tabla_analitica.head(20))

# Mostrar valor final
print(f"\n>>> Aproximación analítica de infectados en t = {tf} días: x({tf}) ≈ {x_vals_analitica[-1]:.2f}")

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals_analitica, label='Solución analítica', color='green')
plt.title('Propagación del virus (Solución Analítica)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Estudiantes infectados')
plt.grid(True)
plt.legend()
plt.show()

