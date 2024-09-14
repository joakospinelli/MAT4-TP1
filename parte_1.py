import pandas as pd
import matplotlib.pyplot as plt
import funciones_reg_simple as func

dataset = pd.read_csv("players_21.csv")

dataset = dataset.dropna(axis=1, how="any").select_dtypes(include=["number"])

x = dataset['wage_eur']
y = dataset['value_eur']

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

plt.show()