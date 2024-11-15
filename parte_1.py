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

# Cálculo del intervalo de confianza para beta_0

t_value_ic = stats.t.ppf(0.975, n-2)

ic_b0_aux = (variance * ( 1/n + (func.get_mean(x)**2 / func.get_s_xx(x)) ) )**0.5

t_statistic_b0 = b0 / ic_b0_aux

ic_b0_lower = b0 - t_value_ic * ic_b0_aux
ic_b0_upper = b0 + t_value_ic * ic_b0_aux

print(f"Intervalo de confianza del 95% para Beta_0: [{ic_b0_lower} ; {ic_b0_upper}]")

# Cálculo de intervalo de confianza para beta_1

ic_b1_aux = (variance / func.get_s_xx(x))**0.5
t_statistic_b1 = b1 / ic_b1_aux

ic_b1_lower = b1 - t_value_ic * ic_b1_aux
ic_b1_upper = b1 + t_value_ic * ic_b1_aux

print(f"Intervalo de confianza del 95% para Beta_1: [{ic_b1_lower} ; {ic_b1_upper}]")

# Intervalo de predicción Y* con x* = mean(x)
x_mean = func.get_mean(x)
x_star = x_mean

y_pred = b0 + b1 * x_mean

residuals = y - y_pred
s_residuals = np.sum(residuals**2)

s_e = np.sqrt(s_residuals / (n - 2))

ic_pred_aux = (variance * (1 + (1/n) + ((x_star - x_mean)**2 / func.get_s_xx(x)) ) )**0.5

ic_pred_lower = y_pred - t_value_ic * ic_pred_aux
ic_pred_upper = y_pred + t_value_ic * ic_pred_aux

print(f"Intervalo de predicción del 95% para Y*: [{ic_pred_lower} ; {ic_pred_upper}]")

# Intervalo de confianza para la respuesta media

ic_mean_aux = (variance * ((1/n) + ((x_star - x_mean)**2 / func.get_s_xx(x)) ) )**0.5

ic_mean_lower = (b0+b1*x_star) - t_value_ic * ic_mean_aux
ic_mean_upper = (b0+b1*x_star) + t_value_ic * ic_mean_aux

print(f"Intervalo de confianza para la media: [{ic_mean_lower} ; {ic_mean_upper}]")

# Proporción de veces que el valor de mercado real cae fuera del intervalo de predicción
exceeds = np.sum((y < ic_pred_lower) | (y > ic_pred_upper))
proportion_exceeds = exceeds / n
print(f"Proporción de veces que el valor de mercado supera la incertidumbre: {proportion_exceeds}")

# Proporción de veces que el valor de mercado supera la incertidumbre de la respuesta media
mean_Y_pred = np.mean(y_pred)
CI_lower_mean = mean_Y_pred - t_value_ic * ic_pred_aux
CI_upper_mean = mean_Y_pred + t_value_ic * ic_pred_aux

exceeds_mean = np.sum((y < CI_lower_mean) | (y > CI_upper_mean))
proportion_exceeds_mean = exceeds_mean / n
print(f"Proporción de veces que el valor de mercado supera la incertidumbre de la respuesta media: {proportion_exceeds_mean}")

plt.show()