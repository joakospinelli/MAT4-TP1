import pandas as pd
import matplotlib.pyplot as plt
import funciones_reg_simple as func
import numpy as np
import scipy.stats as stats

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

x = dataset['wage_eur']
y = dataset['value_eur']
n = len(x)
plt.scatter(x, y)
plt.xlabel("Sueldo del jugador")
plt.ylabel("Valor de mercado del jugador")

# Cálculo de estimadores
b1 = func.get_s_xy(x, y) / func.get_s_xx(x)
b0 = func.get_mean(y) - (b1 * func.get_mean(x))

print(f"La recta de regresión tiene los parámetros beta_0={b0} y beta_1={b1}")

# Gráfico de recta de ajuste
x_line = [i for i in range(0, 600000)]
y_line = [b0 + b1*x for x in x_line]

plt.plot(x_line, y_line, label="Recta de ajuste", c="red")

# Cálculo de varianza
variance = func.get_sce(x,y) / len(y) - 2

#i) prueba de significancia
# Predicciones y residuos
Y_pred = b0 + b1 * x
residuals = y - Y_pred
s_residuals = np.sum(residuals**2)

# Error estándar de los residuos
s_e = np.sqrt(s_residuals / (n - 2))

# Error estándar de la pendiente
se_b1 = s_e / np.sqrt(func.get_s_xx(x))

# Estadístico t para la pendiente
t_stat = b1 / se_b1
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_value}")

#ii) Inferencias sobre los parámetros de la recta (intervalo de confianza del 95%)

t_value = stats.t.ppf(0.975, df=n-2)  # Valor t para 95% de confianza
CI_lower = b1 - t_value * se_b1
CI_upper = b1 + t_value * se_b1

print(f"Intervalo de confianza del 95% para la pendiente: [{CI_lower}, {CI_upper}]")

# iii) Proporción de veces que el valor de mercado supera la incertidumbre de predicción
# Intervalo de predicción para cada valor que se predijo
CI_lower_pred = Y_pred - t_value * s_e
CI_upper_pred = Y_pred + t_value * s_e


# Proporción de veces que el valor de mercado real cae fuera del intervalo de predicción
exceeds = np.sum((y < CI_lower_pred) | (y > CI_upper_pred))
proportion_exceeds = exceeds / n
print(f"Proporción de veces que el valor de mercado supera la incertidumbre: {proportion_exceeds}")


# Proporción de veces que el valor de mercado supera la incertidumbre de la respuesta media
mean_Y_pred = np.mean(Y_pred)
CI_lower_mean = mean_Y_pred - t_value * s_e
CI_upper_mean = mean_Y_pred + t_value * s_e

exceeds_mean = np.sum((y < CI_lower_mean) | (y > CI_upper_mean))
proportion_exceeds_mean = exceeds_mean / n
print(f"Proporción de veces que el valor de mercado supera la incertidumbre de la respuesta media: {proportion_exceeds_mean}")



plt.show()