import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# función de costo (error cuadrático medio)
def compute_cost(X, y, beta):
    m = len(y)  
    predictions = X.dot(beta)  # predicciones con los coeficientes actuales
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# función para registrar los valores en cada iteración
def add_row(data: dict, iteration, prev, new, fnew, error):
    data['iteration'].append(iteration)
    data['prev'].append(prev.copy()) 
    data['new'].append(new.copy()) 
    data['fnew'].append(fnew)
    data['error'].append(error)


def gradient_descent(X, y, beta, alpha, tol, max_iters):
    m = len(y)
    iteration = 1
    data = {'iteration': [], 'prev': [], 'new': [], 'fnew': [], 'error': []}

    # calcular costo inicial
    f_prev = compute_cost(X, y, beta)
    
    # solo arranco el algoritmo si no estamos en el mínimo
    gradients = (1 / m) * X.T.dot(X.dot(beta) - y)  # gradiente inicial
    if not np.all(gradients == 0):
        beta_new = beta - alpha * gradients
        f_new = compute_cost(X, y, beta_new)
        error = abs(f_new - f_prev)
        add_row(data, iteration, beta, beta_new, f_new, error)

        # ejecuto el descenso por gradiente hasta que el error sea menor que la tolerancia
        while error > tol and iteration < max_iters:
            iteration += 1
            beta = beta_new  
            f_prev = f_new  
            gradients = (1 / m) * X.T.dot(X.dot(beta) - y)  
            beta_new = beta - alpha * gradients  
            f_new = compute_cost(X, y, beta_new)  
            error = abs(f_new - f_prev)
            add_row(data, iteration, beta, beta_new, f_new, error)

    return beta_new, data

# inicio del programa
inicio = time.process_time()

dataset = pd.read_csv("players_21.csv")
dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

x = dataset[['wage_eur','overall','potential','international_reputation']].values
y = dataset['value_eur'].values.reshape(-1, 1)

x = np.column_stack((np.ones(x.shape[0]), x))

# inicializar los coeficientes en 0
beta = np.zeros((x.shape[1], 1))

# parámetros del descenso por gradiente
alpha = 0.01  # tasa de aprendizaje
tol = 1e-9  # tolerancia
max_iters = 1000  # máximo número de iteraciones

# ejecutar el descenso por gradiente
beta_final, data = gradient_descent(x, y, beta, alpha, tol, max_iters)

# df de los resultados
df = pd.DataFrame(data)

# coeficientes finales y tabla de iteraciones
print(f"Coeficientes finales (beta):\n{beta_final}")
print(df.to_string(index=False))

fin = time.process_time()  
print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")
