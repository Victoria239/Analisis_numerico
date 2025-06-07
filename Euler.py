import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
k = 0.0005  # constante (puedes ajustarla si conoces otra condición)
x0 = 50  # condición inicial: x(4) = 50
t0 = 4  # tiempo inicial
tf = 6  # tiempo final
h = 0.01  # paso pequeño para mayor precisión


# Ecuación diferencial
def f(t, x):
    return k * x * (1000 - x)


# Método de Euler con registro de iteraciones
def euler(f, x0, t0, tf, h):
    n = int((tf - t0) / h)
    t_values = [t0]
    x_values = [x0]

    x = x0
    t = t0
    for i in range(n):
        x = x + h * f(t, x)
        t = t + h
        t_values.append(t)
        x_values.append(x)
        print(f"Iteración {i + 1}: t = {t:.2f}, x = {x:.4f}")

    return np.array(t_values), np.array(x_values)


# Ejecutar el método de Euler
t_vals, x_vals = euler(f, x0, t0, tf, h)

# Mostrar el resultado final
print(f"\n>>> Aproximación de infectados en t = {tf} días: x({tf}) ≈ {x_vals[-1]:.2f}")

# Crear tabla en pandas para visualización
tabla_resultados = pd.DataFrame({
    'Día (t)': t_vals,
    'Infectados (x)': x_vals
})

# Mostrar primeras filas de la tabla
print("\nTabla de resultados (primeros 20 pasos):")
print(tabla_resultados.head(20))

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='Método de Euler (h = 0.01)', color='blue')
plt.title('Propagación del virus (Euler)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Estudiantes infectados')
plt.grid(True)
plt.legend()
plt.show()

