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

# Método de Euler
def euler(f, x0, t0, tf, h):
    n = int((tf - t0) / h)
    t_values = [t0]
    x_values = [x0]
    x, t = x0, t0
    for i in range(n):
        x += h * f(t, x)
        t += h
        t_values.append(t)
        x_values.append(x)
    return np.array(t_values), np.array(x_values)

# Método de Heun
def heun(f, x0, t0, tf, h):
    n = int((tf - t0) / h)
    t_values = [t0]
    x_values = [x0]
    x, t = x0, t0
    for i in range(n):
        x_pred = x + h * f(t, x)
        x += (h / 2) * (f(t, x) + f(t + h, x_pred))
        t += h
        t_values.append(t)
        x_values.append(x)
    return np.array(t_values), np.array(x_values)

# Solución analítica
def analitica(t):
    C = (1000 - x0) / x0 * np.exp(-k * 1000 * (t - t0))
    return 1000 / (1 + C)

# Ejecutar métodos
t_vals = np.arange(t0, tf + h, h)
t_euler, x_euler = euler(f, x0, t0, tf, h)
t_heun, x_heun = heun(f, x0, t0, tf, h)
x_exacta = analitica(t_vals)

# Mostrar resultado final
print(f"\n>>> Resultado en t = {tf} días:")
print(f"Euler:     x ≈ {x_euler[-1]:.2f}")
print(f"Heun:      x ≈ {x_heun[-1]:.2f}")
print(f"Analítica: x ≈ {x_exacta[-1]:.2f}")

# Tabla (primeros 20 pasos)
tabla = pd.DataFrame({
    't (días)': t_vals[:20],
    'Euler': x_euler[:20],
    'Heun': x_heun[:20],
    'Analítica': x_exacta[:20]
})
print("\nTabla de resultados (primeros 20 pasos):")
print(tabla)

# Gráfica comparativa
plt.figure(figsize=(12, 6))
plt.plot(t_vals, x_euler, label='Euler', color='blue', linestyle='--', linewidth=2, marker='o', markersize=3)
plt.plot(t_vals, x_heun, label='Heun', color='red', linestyle='-.', linewidth=2, marker='s', markersize=3)
plt.plot(t_vals, x_exacta, label='Solución Analítica', color='green', linestyle='-', linewidth=3)

plt.title('Comparación de métodos: Propagación del virus', fontsize=14)
plt.xlabel('Tiempo (días)', fontsize=12)
plt.ylabel('Estudiantes infectados', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
