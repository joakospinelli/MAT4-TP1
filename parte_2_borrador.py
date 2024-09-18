import pandas as pd
import matplotlib.pyplot as plt
import funciones_reg_simple as func
import numpy as np
import scipy.stats as stats

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

"""
AGREGAR BIEN COMO ESTOY COPIANDO LO DEL EJERCICIO 15 NO ME FIJO TODO
"""

x = dataset['wage_eur','overall','age'] # <--- ACTUALIZAR 
y = dataset['value_eur']
""" 
n = len(x)
plt.scatter(x, y)
plt.xlabel("Sueldo del jugador")
plt.ylabel("Valor de mercado del jugador")
"""

# Los resultados son (filas, columnas) (ojalá)
print(f"dimension de X: {x.shape}")
print(f"dimension de Y: {y.shape}")

# Agrego una columna de unos para el término independiente
x = np.column_stack((np.ones(x.shape[0]), x))
print(f"dimension de X: {x.shape}")


# Resuelvo productos punto e inversas.

xtx = np.dot(x.t, x)
print(xtx.shape)
xtx_inv = np.linalg.inv(xtx)
xty = np.dot(x.t, y)
beta = np.dot(xtx_inv, xty)

y_pred = np.dot(x, beta)

print("Coeficientes:")
print(f"B0 (Ordenada al origen): {beta[0]:.4f}")
print(f"B1 (Algoritmos): {beta[1]:.4f}")
print(f"B2 (Base de Datos): {beta[2]:.4f}")
print(f"B3 (Programación): {beta[3]:.4f}")


print("\nPrimeras filas del DataFrame:")
print(dataset.head(10)) #muestro las primeras 10 lineas

""" ESTO NO ENTENDI Q HACE CREO Q GRAFICA :p"""
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', label='Predicción')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Línea ideal')

plt.title('Notas reales de PHP vs Predicción del modelo')
plt.xlabel('Notas reales de PHP')
plt.ylabel('Notas de PHP que predice el modelo')
plt.legend()
plt.grid(True)
plt.show()

"""
STATS MODEL NO SE USA PERO CAPAZ LO USAMOS PARA VER SI NOS DA BIEN NUESTRO CODIGO?
"""
import statsmodels.api as sm

x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
summary = model.summary()

rsquared = model.rsquared

summary, rsquared


"""
descenso de gradiente supongo que lo haremos manual?
"""