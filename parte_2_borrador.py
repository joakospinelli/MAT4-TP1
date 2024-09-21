import pandas as pd
import matplotlib.pyplot as plt
import funciones_reg_simple as func
import numpy as np
import scipy.stats as stats

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])


x = dataset[['wage_eur','overall','potential','international_reputation']].values
y = dataset['value_eur']

# Los resultados son (filas, columnas) (ojalá)
print(f"dimension de X: {x.shape}")
print(f"dimension de Y: {y.shape}")

# Agrego una columna de unos para el término independiente
x = np.column_stack((np.ones(x.shape[0]), x))
print(f"dimension de X: {x.shape}")


# Resuelvo productos punto e inversas.
""" 
x -> matriz de las caracteristicas q elegimos
x^t -> es la transpuesta de X
x^t x -> es el producto de X por la transpuesta, hace una matriz cuadrada que representa el sistema de ecuaciones
(x^t x)^1 -> inversa de la matriz
x^t.y -> producto de la transpuesta de X por el vector de salidaa y
"""
xtx = np.dot(x.T, x) 
print(xtx.shape)
xtx_inv = np.linalg.inv(xtx)
xty = np.dot(x.T, y)
beta = np.dot(xtx_inv, xty)

print("Coeficientes:")
print(f"B0 (Ordenada al origen): {beta[0]:.4f}")
print(f"B1 (Sueldo en euros): {beta[1]:.4f}")
print(f"B2 (Puntuación general): {beta[2]:.4f}")
print(f"B3 (Potencial): {beta[3]:.4f}")
print(f"B4 (Reputación internacional): {beta[4]:.4f}")

y_pred = np.dot(x, beta)

# me falta coeficiente de determinación y la correlación
ssr = np.sum((y - y_pred) ** 2)
syy = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ssr / syy)
r = (r2 ** 0.5)

print("r2 - coeficiente de determinación", r2)
print("r - coeficiente de correlación", r)
# -----------------------------
print("\nPrimeras filas del DataFrame:")
print(dataset.head(10)) #muestro las primeras 10 lineas

""" ESTO NO ENTENDI Q HACE CREO Q GRAFICA :p"""
plt.scatter(y, y_pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red')

plt.title('Valor de mercado real vs Valor estimado')
plt.xlabel('Valor de mercado')
plt.ylabel('Valor de mercado predicho por el modelo')
plt.legend()
plt.grid(True)
plt.show()
