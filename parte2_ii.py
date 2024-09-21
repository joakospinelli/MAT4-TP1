import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Función de costo (error cuadrático medio)
def compute_cost(X, y, theta):
    m = len(y)  # Número de ejemplos
    predictions = X.dot(theta)  # Predicciones con los coeficientes actuales
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Función para registrar los valores en cada iteración
def add_row(data: dict, iteration, prev, new, fnew, error):
    data['iteration'].append(iteration)
    data['prev'].append(prev.copy())  # Copiamos para evitar sobrescribir
    data['new'].append(new.copy())  # Guardamos la nueva versión de theta
    data['fnew'].append(fnew)
    data['error'].append(error)

# Descenso por gradiente para regresión lineal multivariable
def gradient_descent(X, y, theta, alpha, tol, max_iters):
    m = len(y)
    iteration = 1
    data = {'iteration': [], 'prev': [], 'new': [], 'fnew': [], 'error': []}

    # Calcular el costo inicial
    f_prev = compute_cost(X, y, theta)
    
    # Solo comienza el algoritmo si no estamos en el mínimo
    gradients = (1 / m) * X.T.dot(X.dot(theta) - y)  # Gradiente inicial
    if not np.all(gradients == 0):
        theta_new = theta - alpha * gradients
        f_new = compute_cost(X, y, theta_new)
        error = abs(f_new - f_prev)
        add_row(data, iteration, theta, theta_new, f_new, error)

        # Ejecutar el descenso por gradiente hasta que el error sea menor que la tolerancia
        while error > tol and iteration < max_iters:
            iteration += 1
            theta = theta_new  # Actualizamos theta
            f_prev = f_new  # Actualizamos el valor de la función de costo
            gradients = (1 / m) * X.T.dot(X.dot(theta) - y)  # Recalculamos el gradiente
            theta_new = theta - alpha * gradients  # Actualizamos theta
            f_new = compute_cost(X, y, theta_new)  # Recalculamos el costo
            error = abs(f_new - f_prev)
            add_row(data, iteration, theta, theta_new, f_new, error)

    return theta_new, data

# Inicio del algoritmo de descenso por gradiente
inicio = time.process_time()

dataset = pd.read_csv("players_21.csv")
dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

x = dataset[['wage_eur','overall','potential','international_reputation']].values
y = dataset['value_eur'].values.reshape(-1, 1)

# Agrego una columna de unos para el término independiente
x = np.column_stack((np.ones(x.shape[0]), x))

# Inicializar los coeficientes en 0
theta = np.zeros((x.shape[1], 1))

# Parámetros del descenso por gradiente
alpha = 0.01  # Tasa de aprendizaje
tol = 1e-9  # Tolerancia
max_iters = 1000  # Máximo número de iteraciones

# Ejecutar el descenso por gradiente
theta_final, data = gradient_descent(x, y, theta, alpha, tol, max_iters)

# Crear el DataFrame para mostrar los resultados
df = pd.DataFrame(data)

# Mostrar los coeficientes finales y la tabla de iteraciones
print(f"Coeficientes finales (theta):\n{theta_final}")
print(df.to_string(index=False))

fin = time.process_time()  # Finalizo el conteo del tiempo de ejecución
print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")
