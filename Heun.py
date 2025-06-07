import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
k = 0.0005
x0 = 50
t0 = 4
tf = 6
h = 0.01

# Ecuación diferencial
def f(t, x):
    return k * x * (1000 - x)

# Método de Heun con registro de iteraciones
def heun(f, x0, t0, tf, h):
    n = int((tf - t0) / h)
    t_values = [t0]
    x_values = [x0]

    x = x0
    t = t0
    for i in range(n):
        x_pred = x + h * f(t, x)                          # Predicción (Euler)
        x = x + (h / 2) * (f(t, x) + f(t + h, x_pred))    # Corrección
        t = t + h
        t_values.append(t)
        x_values.append(x)
        print(f"Iteración {i + 1}: t = {t:.2f}, x = {x:.4f}")

    return np.array(t_values), np.array(x_values)

# Ejecutar el método de Heun
t_vals, x_vals = heun(f, x0, t0, tf, h)

# Mostrar el resultado final
print(f"\n>>> Aproximación de infectados en t = {tf} días (Heun): x({tf}) ≈ {x_vals[-1]:.2f}")

# Crear tabla en pandas para visualización
tabla_resultados = pd.DataFrame({
    'Día (t)': t_vals,
    'Infectados (x)': x_vals
})

# Mostrar primeras filas
print("\nTabla de resultados (primeros 20 pasos):")
print(tabla_resultados.head(20))

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='Método de Heun (h = 0.01)', color='orange')
plt.title('Propagación del virus (Heun)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Estudiantes infectados')
plt.grid(True)
plt.legend()
plt.show()
