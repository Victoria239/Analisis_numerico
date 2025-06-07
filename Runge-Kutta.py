import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros
k = 0.0005  # constante de propagación
x0 = 50  # condición inicial x(4) = 50
t0 = 4  # tiempo inicial
tf = 6  # tiempo final
h = 0.01  # tamaño del paso


# Ecuación diferencial
def f(t, x):
    return k * x * (1000 - x)


# Método de Runge-Kutta de cuarto orden (RK4)
def runge_kutta(f, x0, t0, tf, h):
    n = int((tf - t0) / h)
    t_values = [t0]
    x_values = [x0]

    x = x0
    t = t0
    for i in range(n):
        k1 = h * f(t, x)
        k2 = h * f(t + h / 2, x + k1 / 2)
        k3 = h * f(t + h / 2, x + k2 / 2)
        k4 = h * f(t + h, x + k3)

        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h

        t_values.append(t)
        x_values.append(x)
        print(f"Iteración {i + 1}: t = {t:.2f}, x = {x:.4f}")

    return np.array(t_values), np.array(x_values)


# Ejecutar Runge-Kutta
t_vals, x_vals = runge_kutta(f, x0, t0, tf, h)

# Mostrar el resultado final
print(f"\n>>> Aproximación de infectados en t = {tf} días: x({tf}) ≈ {x_vals[-1]:.2f}")

# Crear tabla con pandas
tabla_resultados = pd.DataFrame({
    'Día (t)': t_vals,
    'Infectados (x)': x_vals
})

# Mostrar primeras filas
print("\nTabla de resultados (primeros 20 pasos):")
print(tabla_resultados.head(20))

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(t_vals, x_vals, label='Runge-Kutta 4 (h = 0.01)', color='green')
plt.title('Propagación del virus (Runge-Kutta 4)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Estudiantes infectados')
plt.grid(True)
plt.legend()
plt.show()

